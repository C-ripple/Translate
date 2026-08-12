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
    RIPPLEProcessingElement, CUDADim3, TranslationError
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
        assert "ripple_get_block_size(ripple_block, 0)" in result
        assert "ripple_get_block_size(ripple_block, 1)" in result
    
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
        assert "ripple_get_block_size" in result
    
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

    def test_warp_reduction_emits_correct_reduceadd_arity(self):
        """WarpReductionRule must emit the 2-arg ripple_reduceadd(dims, val)
        form per api.md, not the 1-arg form (GitHub issue #10)."""
        source = """
__global__ void reduce(float *val) {
    float sum = *val;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    *val = sum;
}
"""
        result = translate_cuda_source(source)
        assert "ripple_reduceadd(0b1, sum)" in result
        assert "ripple_reduceadd(sum)" not in result  # the old, wrong 1-arg form


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
# Warp Shuffle Hoisting Tests (GitHub issue #8)
# =============================================================================

def test_shuffle_xor_hoists_named_function_not_lambda():
    # The lane_mask argument (1) is passed as a literal, not through a
    # local variable — this is deliberate. The hoisted permutation
    # function is file-scope, so it can only safely capture a
    # compile-time-constant-shaped expression; a variable reference here
    # is the separate untranslated-with-warning case covered by
    # test_shuffle_with_variable_argument_is_left_untranslated_with_warning
    # below (GitHub issue #8 follow-up).
    source = """
__global__ void kernel(int *a) {
    int val = a[0];
    int result = __shfl_xor_sync(0xffffffff, val, 1);
    a[0] = result;
}
"""
    result = translate_cuda_source(source)
    assert "[]" not in result  # no C++ lambda capture syntax anywhere
    assert "auto " not in result  # no C++ auto either
    # The call site should reference a plain named function, not inline it.
    call_match = re.search(r'ripple_shuffle\(\s*val\s*,\s*(\w+)\s*\)', result)
    assert call_match, f"expected a plain ripple_shuffle(val, <name>) call, got:\n{result}"
    fn_name = call_match.group(1)
    # That name must be defined as a real, named, file-scope function
    # BEFORE the kernel that calls it (i.e. before the kernel's own
    # opening brace in the output), not nested inside it.
    fn_def_pos = result.find(f"size_t {fn_name}(size_t")
    kernel_start_pos = result.find("kernel_ripple(")
    assert fn_def_pos != -1, f"expected a definition of {fn_name}, got:\n{result}"
    assert fn_def_pos < kernel_start_pos, (
        f"{fn_name} must be defined before the kernel that uses it, "
        f"not nested inside it (def at {fn_def_pos}, kernel at {kernel_start_pos})"
    )


def test_multiple_shuffle_calls_get_unique_function_names():
    source = """
__global__ void kernel(int *a, int *b) {
    int v1 = a[0];
    int v2 = b[0];
    int r1 = __shfl_xor_sync(0xffffffff, v1, 1);
    int r2 = __shfl_xor_sync(0xffffffff, v2, 2);
    a[0] = r1;
    b[0] = r2;
}
"""
    result = translate_cuda_source(source)
    names = re.findall(r'ripple_shuffle\(\s*\w+\s*,\s*(\w+)\s*\)', result)
    assert len(names) == 2, f"expected 2 ripple_shuffle calls, found: {names}"
    assert names[0] != names[1], f"shuffle helper function names collided: {names}"


def test_shuffle_with_variable_argument_is_left_untranslated_with_warning():
    # The hoisted permutation function is file-scope, so it cannot
    # reference a kernel-local variable like `n` — the old behavior
    # spliced `n` into the function body verbatim, producing C that
    # fails to compile ("use of undeclared identifier"). The correct
    # behavior is to leave the CUDA call untranslated and warn, rather
    # than emit confusingly-broken output.
    source = """
__global__ void kernel(int *a, int n) {
    int val = a[0];
    int result = __shfl_xor_sync(0xffffffff, val, n);
    a[0] = result;
}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    result = transformer.transform(source)
    assert "__shfl_xor_sync" in result  # left untranslated, not silently broken
    assert "ripple_shuffle(" not in result  # and definitely not ALSO translated
    assert any(
        "compile-time constant" in w or "not a compile-time constant" in w
        for w in ctx.warnings
    ), f"expected a compile-time-constant warning, got: {ctx.warnings}"


def test_shuffle_with_literal_argument_still_hoists_correctly():
    # Regression guard: the fix for the variable case must not break the
    # already-fixed literal case.
    source = """
__global__ void kernel(int *a) {
    int val = a[0];
    int result = __shfl_xor_sync(0xffffffff, val, 1);
    a[0] = result;
}
"""
    result = translate_cuda_source(source)
    assert "__shfl_xor_sync" not in result
    assert "ripple_shuffle(" in result


def test_shuffle_with_parenthesized_constant_argument_left_untranslated_with_warning():
    # Regression guard for a paren-balance bug: the outer capture regex
    # (`[^,\)]+`) truncates at the FIRST `)`, so a perfectly ordinary
    # parenthesized constant like `(1)` arrives at
    # _is_compile_time_constant_expr as `(1` — a dangling unmatched open
    # paren. A shape-only check (digits/operators/parens, no balance
    # requirement) would accept that as "constant" and splice the
    # dangling paren into the hoisted function body, producing malformed
    # C ("return k ^ ((1);") with zero warnings. The fix requires
    # balanced parens, so this must degrade to the same safe
    # untranslated+warned path as a real variable reference.
    source = """
__global__ void kernel(int *a) {
    int val = a[0];
    int result = __shfl_xor_sync(0xffffffff, val, (1));
    a[0] = result;
}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    result = transformer.transform(source)
    assert "__shfl_xor_sync" in result  # left untranslated, not silently broken
    assert "ripple_shuffle(" not in result
    assert "((1)" not in result  # the specific malformed-paren splice must not appear
    assert any(
        "compile-time constant" in w or "not a compile-time constant" in w
        for w in ctx.warnings
    ), f"expected a compile-time-constant warning, got: {ctx.warnings}"


@pytest.mark.parametrize("intrinsic", [
    "__shfl_xor_sync(0xffffffff, val, {arg})",
    "__shfl_up_sync(0xffffffff, val, {arg})",
    "__shfl_sync(0xffffffff, val, {arg})",
    "__shfl_down_sync(0xffffffff, val, {arg})",
])
def test_shuffle_variable_argument_left_untranslated_across_all_rules(intrinsic):
    # All 4 shuffle rules share the same (hand-copied, not shared-function)
    # gating pattern. Prove each one actually gates, not just ShuffleXorRule
    # — a future edit to one rule could silently diverge from the others
    # with nothing to catch it otherwise.
    call = intrinsic.format(arg="n")
    source = f"""
__global__ void kernel(int *a, int n) {{
    int val = a[0];
    int result = {call};
    a[0] = result;
}}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    result = transformer.transform(source)
    assert call in result  # left untranslated
    assert "ripple_shuffle(" not in result
    assert any("compile-time constant" in w for w in ctx.warnings), (
        f"expected a compile-time-constant warning, got: {ctx.warnings}"
    )


@pytest.mark.parametrize("intrinsic", [
    "__shfl_xor_sync(0xffffffff, val, 1)",
    "__shfl_up_sync(0xffffffff, val, 1)",
    "__shfl_sync(0xffffffff, val, 1)",
    "__shfl_down_sync(0xffffffff, val, 1)",
])
def test_shuffle_literal_argument_hoists_across_all_rules(intrinsic):
    source = f"""
__global__ void kernel(int *a) {{
    int val = a[0];
    int result = {intrinsic};
    a[0] = result;
}}
"""
    result = translate_cuda_source(source)
    assert intrinsic.split("(")[0] not in result  # the CUDA intrinsic name is gone
    assert "ripple_shuffle(" in result


def test_unroll_doubling_loop_resolves_xor_shuffle():
    # The exact butterfly-reduction example demonstrated in conversation:
    # i doubles from 1 to 16 (5 iterations), feeding __shfl_xor_sync's
    # lane_mask argument — previously left untranslated with a warning,
    # since 'i' is a kernel-local loop variable, not a compile-time
    # constant, from the shuffle rule's point of view in isolation.
    source = """
__global__ void butterflyReduce(float *data) {
    float val = data[threadIdx.x];
    for (int i = 1; i < 32; i *= 2) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    assert "__shfl_xor_sync" not in result
    assert result.count("ripple_shuffle(") == 5  # i = 1, 2, 4, 8, 16


def test_unroll_halving_loop_resolves_down_shuffle():
    # Same mechanism, opposite direction (halving instead of doubling),
    # and a different shuffle intrinsic — proves the rule is shape-based
    # (any counting loop with literal bounds), not hardcoded to one
    # specific loop direction or intrinsic. Deliberately NOT the
    # warpSize/2 + __shfl_down_sync + accumulate shape WarpReductionRule
    # already owns (that rule requires the symbolic literal "warpSize",
    # not a plain digit, so it can't fire here) — this uses a literal
    # loop bound and a non-accumulating shuffle_up to stay clear of it.
    source = """
__global__ void haloExchange(float *data) {
    float val = data[threadIdx.x];
    for (int offset = 16; offset > 0; offset /= 2) {
        val = __shfl_up_sync(0xffffffff, val, offset);
    }
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    assert "__shfl_up_sync" not in result
    assert result.count("ripple_shuffle(") == 5  # offset = 16, 8, 4, 2, 1


def test_unroll_produces_distinct_literal_values_per_iteration():
    # Not just "5 calls exist" — each one must actually reference the
    # right literal value, not the same one 5 times or a mangled one.
    source = """
__global__ void butterflyReduce(float *data) {
    float val = data[threadIdx.x];
    for (int i = 1; i < 8; i *= 2) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    import re
    hoisted_bodies = re.findall(r'return k \^ \((\d+)\);', result)
    assert sorted(int(v) for v in hoisted_bodies) == [1, 2, 4]


def test_unroll_does_not_fire_on_unrelated_countable_loop():
    # A loop with literal bounds but no shuffle call in its body should
    # be left completely alone — unrolling is not a general-purpose
    # optimization this translator applies proactively, only a targeted
    # fix for the one thing that would otherwise be untranslatable.
    source = """
__global__ void sumLoop(int *data) {
    int total = 0;
    for (int i = 0; i < 4; i += 1) {
        total += data[i];
    }
    data[0] = total;
}
"""
    result = translate_cuda_source(source)
    assert "for (int i = 0; i < 4; i += 1)" in result


def test_unroll_does_not_fire_when_warp_reduction_rule_already_owns_the_pattern():
    # Regression guard: the classic warpSize/2 + __shfl_down_sync +
    # accumulate shape must still go through WarpReductionRule's single
    # ripple_reduceadd(0b1, ...) call, not get unrolled into 5 separate
    # ripple_shuffle calls — unrolling only fires on loops it doesn't
    # already recognize (literal, not symbolic, bounds).
    source = """
__global__ void reduce(float *val) {
    float sum = *val;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    *val = sum;
}
"""
    result = translate_cuda_source(source)
    assert "ripple_reduceadd(0b1, sum)" in result
    assert "ripple_shuffle(" not in result


def test_unroll_braceless_body_resolves_xor_shuffle():
    # Gap 1: an ordinary braceless single-statement loop body is a
    # completely idiomatic way to write this same butterfly-reduction
    # loop — the unroll rule must not require literal {} to fire, or
    # this common shape is left untranslated (and, post-hard-fail,
    # crashes) for no reason other than a formatting choice.
    source = """
__global__ void butterflyReduce(float *data) {
    float val = data[threadIdx.x];
    for (int i = 1; i < 32; i *= 2)
        val += __shfl_xor_sync(0xffffffff, val, i);
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    assert "__shfl_xor_sync" not in result
    assert result.count("ripple_shuffle(") == 5  # i = 1, 2, 4, 8, 16
    hoisted_bodies = re.findall(r'return k \^ \((\d+)\);', result)
    assert sorted(int(v) for v in hoisted_bodies) == [1, 2, 4, 8, 16]


def test_unroll_warpsize_bound_resolves_xor_shuffle():
    # Gap 2: warpSize is a real, well-known CUDA built-in that is
    # always 32 in practice on any CUDA-capable GPU. A loop bounded by
    # `warpSize` (rather than the digit 32) is completely idiomatic —
    # it shouldn't be left untranslated just because the bound isn't
    # spelled as a literal digit, and this shape is neither
    # WarpReductionRule's exact warpSize/2-init shape nor a plain
    # digit-bounded loop.
    source = """
__global__ void butterflyReduce(float *data) {
    float val = data[threadIdx.x];
    for (int i = 1; i < warpSize; i *= 2) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    assert "__shfl_xor_sync" not in result
    assert result.count("ripple_shuffle(") == 5  # i = 1, 2, 4, 8, 16 (warpSize=32)
    hoisted_bodies = re.findall(r'return k \^ \((\d+)\);', result)
    assert sorted(int(v) for v in hoisted_bodies) == [1, 2, 4, 8, 16]


def test_unroll_warpsize_init_resolves_halving_sequence():
    # Gap 2 continued: warpSize is accepted in the INIT position too,
    # not just BOUND — a halving-from-warpSize loop is just as
    # idiomatic as a doubling-toward-warpSize one, and both directions
    # need to resolve the same symbolic-but-known-constant.
    source = """
__global__ void haloExchange(float *data) {
    float val = data[threadIdx.x];
    for (int i = warpSize; i > 0; i /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    assert "__shfl_xor_sync" not in result
    assert result.count("ripple_shuffle(") == 6  # i = 32, 16, 8, 4, 2, 1
    hoisted_bodies = re.findall(r'return k \^ \((\d+)\);', result)
    assert sorted(int(v) for v in hoisted_bodies) == [1, 2, 4, 8, 16, 32]


@pytest.mark.parametrize("statement", [
    pytest.param(
        "if (flag[0] < 16) val[0] += __shfl_xor_sync(0xffffffff, val[0], i); else val[0] -= 1.0f;",
        id="if_spaced",
    ),
    pytest.param(
        "if(flag[0] < 16) val[0] += __shfl_xor_sync(0xffffffff, val[0], i); else val[0] -= 1.0f;",
        id="if_unspaced",
    ),
    pytest.param(
        "while (flag[0] < 16) val[0] += __shfl_xor_sync(0xffffffff, val[0], i);",
        id="while",
    ),
    pytest.param(
        "do val[0] += __shfl_xor_sync(0xffffffff, val[0], i); while (flag[0] > 0);",
        id="do",
    ),
    pytest.param(
        "for (int j = 0; j < 1; j++) val[0] += __shfl_xor_sync(0xffffffff, val[0], i);",
        id="for",
    ),
    pytest.param(
        "switch (flag[0]) val[0] += __shfl_xor_sync(0xffffffff, val[0], i);",
        id="switch",
    ),
])
def test_unroll_braceless_control_flow_body_left_untouched(statement):
    # Critical regression guard: PATTERN_BRACELESS's body capture
    # (`[^;{}]+;`) truncates at the FIRST semicolon. A braceless
    # if/else has an internal semicolon (ending the if-branch) before
    # the statement is actually finished, so naively unrolling this
    # shape corrupts it — confirmed by reproducing the exact reported
    # bug: only the LAST unrolled copy kept its else-branch, and the
    # rest silently lost theirs. That's not a syntax error (it still
    # compiles), it's silently wrong output — worse than leaving it
    # untranslated, since Task 3's hard-fail can't catch what still
    # compiles clean. The fix bails out of any braceless body starting
    # with a control-flow keyword, leaving the loop completely
    # unmodified — same degrade path as any other "can't safely handle
    # this shape" case in this rule.
    #
    # Parametrized across all 5 keywords CONTROL_FLOW_PREFIX guards
    # against (if/while/do/for/switch), plus both the spaced (`if (x)`)
    # and unspaced (`if(x)`) spellings for `if` specifically — a naive
    # whitespace-tokenized "first word is a keyword" check catches the
    # spaced form but misses `if(...)` entirely (its first
    # whitespace-delimited token is `if(flag[0]`, not `if`), which
    # would let the exact same bug back in for equally idiomatic C.
    source = f"""
__global__ void kernel(float *val, int *flag) {{
    for (int i = 1; i < 8; i *= 2)
        {statement}
}}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    result = transformer.transform(source)

    # The loop and its body must survive completely intact — no
    # unrolling, no lost branches, no duplication.
    assert statement in result
    assert "ripple_shuffle(" not in result
    assert result.count("for (int i = 1; i < 8; i *= 2)") == 1  # loop kept intact, not unrolled


def test_unroll_braceless_body_starting_with_keyword_like_identifier_still_unrolls():
    # Regression guard for CONTROL_FLOW_PREFIX's negative case: an
    # identifier that merely starts with a control-flow keyword's
    # letters (e.g. "forward_val", not the "for" keyword) must NOT
    # trigger the bail-out. The `\b` word-boundary requirement after
    # the keyword group means there's no boundary between "for" and
    # the following "w" in "forward_val", so the regex correctly
    # doesn't match here — this body has no control flow at all and
    # should unroll exactly like any other braceless shuffle body.
    source = """
__global__ void kernel(float *data) {
    float forward_val = data[threadIdx.x];
    for (int i = 1; i < 8; i *= 2)
        forward_val += __shfl_xor_sync(0xffffffff, forward_val, i);
    data[threadIdx.x] = forward_val;
}
"""
    result = translate_cuda_source(source)
    assert "__shfl_xor_sync" not in result
    assert result.count("ripple_shuffle(") == 3  # i = 1, 2, 4


def test_unroll_braced_body_with_string_literal_left_untouched():
    # Critical regression guard, more severe than the braceless
    # control-flow bug above: PATTERN_BRACED's body capture
    # (`\{([^{}]*)\}`) is also unaware of string/char literals, so a
    # `}` INSIDE a string (e.g. `printf("}")`) is misread as the
    # loop's real closing brace. Reproducing the exact reported bug
    # without this fix: the capture truncated mid-string, and
    # everything textually after it in the file — including the
    # loop's REAL closing brace and the trailing `*val_ptr = val;`
    # assignment — ended up spliced OUTSIDE the function body
    # entirely. That's structural corruption of the whole rest of the
    # file, and it was invisible to every safety mechanism: ctx.errors
    # stayed empty, and only the benign "Unrolled loop..." success
    # warning fired — transform() returned normally, so even Task 3's
    # eventual hard-fail wouldn't have caught it. The fix bails out on
    # any quote character anywhere in the captured body, for both
    # braced and braceless bodies.
    source = """
__global__ void kernel(float *val_ptr) {
    float val = *val_ptr;
    for (int i = 1; i < 8; i *= 2) {
        val += __shfl_xor_sync(0xffffffff, val, i);
        printf("}");
    }
    *val_ptr = val;
}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    result = transformer.transform(source)

    # The loop, including the string literal containing a brace, must
    # survive completely intact — no unrolling attempted at all.
    assert 'printf("}");' in result
    assert "ripple_shuffle(" not in result
    assert ctx.errors == []

    # Structural integrity: the trailing assignment must still be
    # INSIDE the function body, immediately followed by the function's
    # own closing brace — not floating outside it (the corruption the
    # original bug produced).
    assert "*val_ptr = val;\n}" in result
    # And the loop body's own embedded '}' (inside the string) must
    # appear BEFORE that trailing assignment in the output — proving
    # the whole loop, string literal included, was left in its
    # original position rather than truncated and reordered.
    assert result.index('printf("}");') < result.index("*val_ptr = val;")


def test_unroll_braceless_body_with_string_literal_left_untouched():
    # Companion to the braced case above: the quote-character bail-out
    # is meant to cover both PATTERN_BRACED and PATTERN_BRACELESS, but
    # only the braced shape had a committed regression test (found in
    # review — the braceless path was only verified manually, not
    # guarded by CI). A braceless body containing a quote character
    # (here, a semicolon inside a string, which would otherwise hit the
    # SAME truncation class of bug PATTERN_BRACELESS had before its own
    # control-flow fix) must also bail out untouched.
    source = """
__global__ void kernel(float *val_ptr) {
    float val = *val_ptr;
    for (int i = 1; i < 8; i *= 2)
        val += __shfl_xor_sync(0xffffffff, val, i) + strlen(";");
    *val_ptr = val;
}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    result = transformer.transform(source)

    assert 'strlen(";")' in result
    assert "ripple_shuffle(" not in result
    assert ctx.errors == []
    assert "*val_ptr = val;\n}" in result


def test_unroll_braceless_body_with_comment_left_untouched():
    # Companion to the two braced comment tests above, mirroring
    # test_unroll_braceless_body_with_string_literal_left_untouched's
    # structure: the comment bail-out is meant to cover both
    # PATTERN_BRACED and PATTERN_BRACELESS, but only the braced shape
    # had committed comment regression tests (found in review). The
    # comment is placed INSIDE the captured single statement, before
    # its terminating semicolon — the same position the string-literal
    # companion test uses for its literal — so this actually exercises
    # the UNSAFE_BODY_TOKENS bail-out itself, rather than merely
    # confirming PATTERN_BRACELESS fails to match (which a
    # differently-positioned comment, e.g. one after the terminating
    # semicolon, would only prove trivially: that text was never part
    # of the captured body to begin with, so it wouldn't need this
    # check at all — see the round-6 commit message for the verified
    # distinction between the two positions).
    source = """
__global__ void kernel(float *val_ptr) {
    float val = *val_ptr;
    for (int i = 1; i < 8; i *= 2)
        val += __shfl_xor_sync(0xffffffff, val, i) /* note */;
    *val_ptr = val;
}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    result = transformer.transform(source)

    assert "/* note */" in result
    assert "ripple_shuffle(" not in result
    assert ctx.errors == []
    assert "*val_ptr = val;\n}" in result


def test_unroll_braced_body_with_block_comment_left_untouched():
    # Critical regression guard, same danger class as the string-literal
    # case above but via a comment instead: PATTERN_BRACED's body
    # capture (`\{([^{}]*)\}`) is also unaware of comments, so a `}`
    # INSIDE a `/* ... */` block comment is misread as the loop's real
    # closing brace. This is arguably MORE likely in real code than the
    # printf case — an ordinary trailing note with a stray bracket is
    # mundane, where a printf debug call inside a hot shuffle loop is
    # contrived. Reproducing the exact reported bug without this fix:
    # the capture truncated mid-comment, and everything textually after
    # it in the file — including the loop's REAL closing brace and the
    # trailing `*val_ptr = val;` assignment — ended up spliced OUTSIDE
    # the function body, with ctx.errors staying empty and only the
    # benign "Unrolled loop..." success warning firing.
    source = """
__global__ void kernel(float *val_ptr) {
    float val = *val_ptr;
    for (int i = 1; i < 8; i *= 2) {
        val += __shfl_xor_sync(0xffffffff, val, i); /* note [3] } */
    }
    *val_ptr = val;
}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    result = transformer.transform(source)

    assert "/* note [3] } */" in result
    assert "ripple_shuffle(" not in result
    assert ctx.errors == []
    # Structural integrity: trailing assignment still inside the
    # function, immediately followed by the function's own closing
    # brace — not floating outside it.
    assert "*val_ptr = val;\n}" in result
    assert result.index("/* note [3] } */") < result.index("*val_ptr = val;")


def test_unroll_braced_body_with_line_comment_left_untouched():
    # Companion to the block-comment case above, for `//` line
    # comments — the coordinator's exact reported repro. Same
    # corruption mechanism: a `}` inside a `//` comment is misread as
    # the loop's real closing brace, silently splicing the function's
    # real structure apart with zero errors raised.
    source = """
__global__ void kernel(float *val_ptr) {
    float val = *val_ptr;
    for (int i = 1; i < 8; i *= 2) {
        val += __shfl_xor_sync(0xffffffff, val, i); // note [3] }
    }
    *val_ptr = val;
}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    result = transformer.transform(source)

    assert "// note [3] }" in result
    assert "ripple_shuffle(" not in result
    assert ctx.errors == []
    assert "*val_ptr = val;\n}" in result
    assert result.index("// note [3] }") < result.index("*val_ptr = val;")


def test_transform_raises_when_ctx_has_errors():
    ctx = TranslationContext()
    ctx.add_error("synthetic test error")
    transformer = CUDAToRIPPLETransformer(ctx)
    with pytest.raises(TranslationError) as exc_info:
        transformer.transform("__global__ void kernel() {}")
    assert "synthetic test error" in str(exc_info.value)


def test_transform_raises_with_all_errors_when_ctx_has_multiple():
    ctx = TranslationContext()
    ctx.add_error("first synthetic test error")
    ctx.add_error("second synthetic test error")
    transformer = CUDAToRIPPLETransformer(ctx)
    with pytest.raises(TranslationError) as exc_info:
        transformer.transform("__global__ void kernel() {}")
    assert exc_info.value.errors == [
        "first synthetic test error",
        "second synthetic test error",
    ]
    assert "first synthetic test error" in str(exc_info.value)
    assert "second synthetic test error" in str(exc_info.value)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
