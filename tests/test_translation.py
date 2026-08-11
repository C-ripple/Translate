"""
CUDA to RIPPLE Test Suite

Tests for both source-level and IR-level translation paths.
"""

import pytest
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.semantic_model import (
    TranslationContext, HexagonConfig, RIPPLEBlockShape,
    RIPPLEProcessingElement, CUDADim3
)
from core.translation_rules import (
    TranslationRuleEngine, ThreadIdxRule, BlockDimRule,
    SharedMemoryRule, AtomicAddRule, infer_block_shape
)
from frontends.source.cuda_frontend import (
    CUDALexer, TokenType, CUDAToRIPPLETransformer, translate_cuda_source
)
from frontends.ir.ir_frontend import (
    LLVMIRParser, IRAnalyzer, CUDAIRToRIPPLETranslator
)


# =============================================================================
# Lexer Tests
# =============================================================================

class TestCUDALexer:
    """Tests for the CUDA lexer."""
    
    def test_tokenize_keywords(self):
        source = "__global__ __device__ __shared__"
        lexer = CUDALexer(source)
        tokens = lexer.tokenize()
        
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.GLOBAL in types
        assert TokenType.DEVICE in types
        assert TokenType.SHARED in types
    
    def test_tokenize_kernel_launch(self):
        source = "kernel<<<grid, block>>>(args)"
        lexer = CUDALexer(source)
        tokens = lexer.tokenize()
        
        types = [t.type for t in tokens]
        assert TokenType.TRIPLE_CHEVRON_OPEN in types
        assert TokenType.TRIPLE_CHEVRON_CLOSE in types
    
    def test_tokenize_thread_idx(self):
        source = "int idx = threadIdx.x + blockIdx.x * blockDim.x;"
        lexer = CUDALexer(source)
        tokens = lexer.tokenize()
        
        identifiers = [t.value for t in tokens if t.type == TokenType.IDENTIFIER]
        assert "threadIdx" in identifiers
        assert "blockIdx" in identifiers
        assert "blockDim" in identifiers
    
    def test_tokenize_comments(self):
        source = "// line comment\nint x; /* block comment */"
        lexer = CUDALexer(source)
        tokens = lexer.tokenize()
        
        comment_tokens = [t for t in tokens if t.type == TokenType.COMMENT]
        assert len(comment_tokens) == 2
    
    def test_tokenize_strings(self):
        source = '"hello world" \'c\''
        lexer = CUDALexer(source)
        tokens = lexer.tokenize()
        
        assert any(t.type == TokenType.STRING for t in tokens)
        assert any(t.type == TokenType.CHAR for t in tokens)
    
    def test_tokenize_numbers(self):
        source = "42 3.14f 0xFF 1e-10"
        lexer = CUDALexer(source)
        tokens = lexer.tokenize()
        
        numbers = [t for t in tokens if t.type == TokenType.NUMBER]
        assert len(numbers) == 4


# =============================================================================
# Translation Rule Tests
# =============================================================================

class TestTranslationRules:
    """Tests for individual translation rules."""
    
    def test_thread_idx_rule(self):
        rule = ThreadIdxRule()
        ctx = TranslationContext()
        
        assert rule.matches("threadIdx.x")
        assert rule.matches("int idx = threadIdx.y;")
        assert not rule.matches("int idx = 0;")
        
        result = rule.apply("threadIdx.x + threadIdx.y", ctx)
        assert "ripple_id(ripple_block, 0)" in result
        assert "ripple_id(ripple_block, 1)" in result
    
    def test_block_dim_rule(self):
        rule = BlockDimRule()
        ctx = TranslationContext()
        
        result = rule.apply("blockDim.x * blockDim.y", ctx)
        assert "ripple_get_size(ripple_block, 0)" in result
        assert "ripple_get_size(ripple_block, 1)" in result
    
    def test_shared_memory_rule(self):
        rule = SharedMemoryRule()
        ctx = TranslationContext(target_platform="hexagon")
        
        source = "__shared__ float sdata[256]"
        result = rule.apply(source, ctx)
        
        assert "__attribute__((aligned(128)))" in result
        assert "sdata[256]" in result
    
    def test_atomic_add_rule(self):
        rule = AtomicAddRule()
        ctx = TranslationContext()
        
        source = "atomicAdd(&sum, val)"
        result = rule.apply(source, ctx)
        
        assert "ripple_atomic_add" in result


class TestTranslationRuleEngine:
    """Tests for the rule engine."""
    
    def test_apply_all_rules(self):
        engine = TranslationRuleEngine()
        ctx = TranslationContext()
        
        source = "int idx = threadIdx.x + blockIdx.x * blockDim.x;"
        result = engine.apply_all(source, ctx)
        
        assert "threadIdx" not in result
        assert "ripple_id" in result
        assert "ripple_get_size" in result
    
    def test_rule_priority(self):
        engine = TranslationRuleEngine()
        
        # Rules should be sorted by priority
        for i in range(len(engine.rules) - 1):
            assert engine.rules[i].priority >= engine.rules[i + 1].priority


# =============================================================================
# Block Shape Inference Tests
# =============================================================================

class TestBlockShapeInference:
    """Tests for block shape inference."""
    
    def test_infer_1d_block(self):
        source = "int idx = threadIdx.x;"
        shape = infer_block_shape(source)
        
        assert shape.dimensionality() == 1
        assert shape.pe_type == RIPPLEProcessingElement.HVX_PE
    
    def test_infer_2d_block(self):
        source = "int row = threadIdx.y; int col = threadIdx.x;"
        shape = infer_block_shape(source)
        
        assert shape.dimensionality() == 2
    
    def test_infer_from_float(self):
        source = "float *data; threadIdx.x;"
        ctx = TranslationContext()
        shape = infer_block_shape(source, ctx=ctx)
        
        # 128 bytes / 4 bytes per float = 32 lanes
        assert shape.dimensions[0] == 32


# =============================================================================
# Source-Level Transformer Tests
# =============================================================================

class TestCUDAToRIPPLETransformer:
    """Tests for the complete source-level transformer."""
    
    def test_simple_kernel(self):
        source = """
__global__ void add(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""
        result = translate_cuda_source(source)
        
        assert "ripple.h" in result
        assert "__global__" not in result
        assert "ripple_id" in result
        assert "HVX_VECTOR_SIZE" in result
    
    def test_shared_memory_kernel(self):
        source = """
__global__ void reduce(float *input, float *output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = input[tid];
    __syncthreads();
}
"""
        result = translate_cuda_source(source)

        assert not re.search(SharedMemoryRule.PATTERN, result)
        assert "vtcm" in result.lower() or "aligned" in result
    
    def test_device_function(self):
        source = """
__device__ float square(float x) {
    return x * x;
}
"""
        result = translate_cuda_source(source)
        
        assert "__device__" not in result
        assert "inline" in result
    
    def test_warnings_generated(self):
        ctx = TranslationContext()
        transformer = CUDAToRIPPLETransformer(ctx)
        
        source = "__syncthreads();"
        transformer.transform(source)
        
        # Should generate warning about syncthreads
        assert len(ctx.warnings) > 0


# =============================================================================
# IR-Level Translator Tests
# =============================================================================

class TestLLVMIRParser:
    """Tests for LLVM IR parsing."""
    
    def test_parse_module_header(self):
        ir = '''
source_filename = "test.cu"
target datalayout = "e-m:e-p:64:64"
target triple = "nvptx64-nvidia-cuda"
'''
        parser = LLVMIRParser(ir)
        module = parser.parse()
        
        assert module.source_filename == "test.cu"
        assert "nvptx" in module.target_triple
    
    def test_parse_function(self):
        ir = '''
define void @kernel(ptr %a, ptr %b) {
entry:
  ret void
}
'''
        parser = LLVMIRParser(ir)
        module = parser.parse()
        
        assert len(module.functions) == 1
        assert module.functions[0].name == "kernel"


class TestIRAnalyzer:
    """Tests for IR analysis."""
    
    def test_detect_thread_idx(self):
        ir = '''
define void @kernel() {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  ret void
}
'''
        parser = LLVMIRParser(ir)
        module = parser.parse()
        analyzer = IRAnalyzer(module)
        result = analyzer.analyze()
        
        assert len(result.thread_idx_uses) > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_vector_add_translation(self):
        """Test complete translation of vector addition kernel."""
        source = """
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}
"""
        result = translate_cuda_source(source, target="hexagon")
        
        # Verify key transformations
        assert "#include <ripple.h>" in result
        assert "cuda_runtime.h" not in result
        assert "ripple_id" in result
        assert "block_idx_x" in result
    
    def test_reduction_translation(self):
        """Test translation of parallel reduction kernel."""
        source = """
__global__ void reduce(float *g_idata, float *g_odata, unsigned int n) {
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
"""
        result = translate_cuda_source(source, target="hexagon")
        
        # Verify shared memory translation
        assert not re.search(SharedMemoryRule.PATTERN, result)
        assert "sdata[256]" in result or "sdata" in result


# =============================================================================
# Hexagon-Specific Tests
# =============================================================================

class TestHexagonConfig:
    """Tests for Hexagon-specific configuration."""
    
    def test_vector_lanes_calculation(self):
        config = HexagonConfig(hvx_width=128)
        
        assert config.get_vector_lanes("float") == 32
        assert config.get_vector_lanes("int32_t") == 32
        assert config.get_vector_lanes("int16_t") == 64
        assert config.get_vector_lanes("int8_t") == 128
    
    def test_hvx_width_options(self):
        config_128 = HexagonConfig(hvx_width=128)
        config_64 = HexagonConfig(hvx_width=64)
        
        assert config_128.get_vector_lanes("float") == 32
        assert config_64.get_vector_lanes("float") == 16


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
