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
    SharedMemoryRule, AtomicAddRule, infer_block_shape,
    UnrollConstantShuffleLoopRule
)
from frontends.source.cuda_frontend import (
    CUDALexer, TokenType, CUDAToRIPPLETransformer, translate_cuda_source
)
from frontends.ir.ir_frontend import (
    LLVMIRParser, IRAnalyzer, CUDAIRToRIPPLETranslator
)
from tests.compile_verify import requires_clang, verify_ripple_syntax


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
        """__shared__ arrays translate to vtcm_malloc, with a matching
        vtcm_free() inserted before the enclosing block's closing brace —
        the real Ripple VTCM mechanism, not an attribute (which doesn't
        exist in the Ripple API)."""
        rule = SharedMemoryRule()
        ctx = TranslationContext(target_platform="hexagon")

        source = "void k() {\n    __shared__ float sdata[256];\n    sdata[0] = 1.0f;\n}"
        result = rule.apply(source, ctx)

        assert "float *sdata = vtcm_malloc(sizeof(float) * (256), /*align_as=*/128);" in result
        assert "vtcm_free(sdata);" in result
        assert result.index("vtcm_malloc") < result.index("vtcm_free")
        # the free() call must land before the function's closing brace
        assert result.index("vtcm_free") < result.rindex("}")

    def test_shared_memory_rule_multidim_hard_fails(self):
        """Multi-dimensional __shared__ arrays have no safe vtcm_malloc
        translation — a flat pointer can't be redeclared with a trailing
        [dim], and every indexing site would need flat-arithmetic
        rewriting this translator can't do. Must hard-fail, not emit
        invalid C."""
        rule = SharedMemoryRule()
        ctx = TranslationContext(target_platform="hexagon")

        source = "void k() {\n    __shared__ float tile[18][18];\n}"
        result = rule.apply(source, ctx)

        assert ctx.has_errors()
        assert "tile" in ctx.errors[0]

    def test_shared_memory_rule_early_return_hard_fails(self):
        """A 'return' between the __shared__ declaration and the enclosing
        block's end would leak the VTCM allocation on that path — VTCM is
        a small, real hardware scratchpad, not virtual memory. Must
        hard-fail rather than silently leak."""
        rule = SharedMemoryRule()
        ctx = TranslationContext(target_platform="hexagon")

        source = (
            "void k(int n) {\n"
            "    __shared__ float sdata[256];\n"
            "    if (n < 0) return;\n"
            "    sdata[0] = 1.0f;\n"
            "}"
        )
        result = rule.apply(source, ctx)

        assert ctx.has_errors()
        assert "sdata" in ctx.errors[0]

    def test_atomic_add_rule(self):
        """atomicAdd has no Ripple equivalent outside a recognized
        reduction idiom (handled separately by WarpReductionRule etc.) —
        AtomicAddRule itself must hard-fail, not emit a fictional call."""
        rule = AtomicAddRule()
        ctx = TranslationContext()

        source = "atomicAdd(&sum, val)"
        result = rule.apply(source, ctx)

        assert ctx.has_errors()
        assert "atomicAdd" in ctx.errors[0]
        assert result == source  # left untranslated, not rewritten


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
    # is the separate hard-fail case covered by
    # test_shuffle_with_variable_argument_raises below (GitHub issue #8
    # follow-up).
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


def test_shuffle_with_variable_argument_raises():
    # A shuffle argument that's a kernel-local variable, and NOT the
    # induction variable of an unrollable loop (it's a plain local
    # assigned a runtime-looking value here, not looping at all) — this
    # is exactly the case UnrollConstantShuffleLoopRule can't help with,
    # so it must now fail loudly rather than silently leave broken
    # output. The hoisted permutation function is file-scope, so it
    # cannot reference a kernel-local variable like `n` — the old
    # behavior spliced `n` into the function body verbatim, producing C
    # that fails to compile ("use of undeclared identifier").
    source = """
__global__ void kernel(int *a, int n) {
    int val = a[0];
    int result = __shfl_xor_sync(0xffffffff, val, n);
    a[0] = result;
}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    with pytest.raises(TranslationError) as exc_info:
        transformer.transform(source)
    assert "not a compile-time constant" in str(exc_info.value)
    assert ctx.has_errors()


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


def test_shuffle_with_parenthesized_constant_argument_raises():
    # Regression guard for a paren-balance bug: the outer capture regex
    # (`[^,\)]+`) truncates at the FIRST `)`, so a perfectly ordinary
    # parenthesized constant like `(1)` arrives at
    # _is_compile_time_constant_expr as `(1` — a dangling unmatched open
    # paren. A shape-only check (digits/operators/parens, no balance
    # requirement) would accept that as "constant" and splice the
    # dangling paren into the hoisted function body, producing malformed
    # C ("return k ^ ((1);") with zero warnings. The fix requires
    # balanced parens, so this must degrade to the same hard-fail path
    # as a real variable reference — the paren-truncation case still
    # must not silently emit malformed C, now raising instead of
    # warning-and-continuing.
    source = """
__global__ void kernel(int *a) {
    int val = a[0];
    int result = __shfl_xor_sync(0xffffffff, val, (1));
    a[0] = result;
}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    with pytest.raises(TranslationError):
        transformer.transform(source)


@pytest.mark.parametrize("intrinsic", [
    "__shfl_xor_sync(0xffffffff, val, {arg})",
    "__shfl_up_sync(0xffffffff, val, {arg})",
    "__shfl_sync(0xffffffff, val, {arg})",
    "__shfl_down_sync(0xffffffff, val, {arg})",
])
def test_shuffle_variable_argument_raises_across_all_rules(intrinsic):
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
    with pytest.raises(TranslationError) as exc_info:
        transformer.transform(source)
    assert "not a compile-time constant" in str(exc_info.value)


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
    # i doubles from 1 to 16 (5 iterations), feeding __shfl_xor_sync's
    # lane_mask argument — previously left untranslated with a warning,
    # since 'i' is a kernel-local loop variable, not a compile-time
    # constant, from the shuffle rule's point of view in isolation.
    # Mask is deliberately non-full (0x0000ffff, not 0xffffffff): the
    # full-mask, bound-32 variant of this exact shape is now owned by
    # ButterflyAllReduceRule (see test_butterfly_allreduce_* below),
    # which fires first and replaces it with a single ripple_reduceadd
    # call instead of unrolling. Using a non-full mask here keeps this
    # test actually exercising UnrollConstantShuffleLoopRule's digit-32
    # bound handling rather than colliding with that higher-priority rule.
    source = """
__global__ void butterflyReduce(float *data) {
    float val = data[threadIdx.x];
    for (int i = 1; i < 32; i *= 2) {
        val += __shfl_xor_sync(0x0000ffff, val, i);
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
    # Mask is deliberately non-full (0xffff0000, not 0xffffffff): the
    # full-mask, warpSize-bound variant of this exact shape is now owned
    # by ButterflyAllReduceRule (see test_butterfly_allreduce_* below),
    # which fires first and replaces it with a single ripple_reduceadd
    # call instead of unrolling. Using a non-full mask here keeps this
    # test actually exercising UnrollConstantShuffleLoopRule's warpSize
    # bound handling rather than colliding with that higher-priority rule.
    source = """
__global__ void butterflyReduce(float *data) {
    float val = data[threadIdx.x];
    for (int i = 1; i < warpSize; i *= 2) {
        val += __shfl_xor_sync(0xffff0000, val, i);
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
    # compiles), it's silently wrong output. The fix bails out of any
    # braceless body starting with a control-flow keyword, leaving the
    # loop completely unmodified — same degrade path as any other
    # "can't safely handle this shape" case in this rule. Since 'i'
    # then survives untouched into ShuffleXorRule, and 'i' is not a
    # compile-time constant, Task 3's hard-fail is exactly what proves
    # the bail-out happened correctly here: a raise naming 'i' means
    # the loop var reached the shuffle rule unmodified, not spliced in
    # as a (corrupted) literal. A corrupted unroll would have consumed
    # 'i' and NOT raised — so the raise is the regression guard now.
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
    with pytest.raises(TranslationError) as exc_info:
        transformer.transform(source)

    # The raise must come from the shuffle rule rejecting the still-intact
    # loop variable 'i' — proof the loop was left completely unmodified,
    # not corrupted by a partial unroll.
    assert "not a compile-time constant" in str(exc_info.value)
    assert "'i'" in str(exc_info.value)


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
    # warning fired — transform() returned normally. The fix bails out
    # on any quote character anywhere in the captured body, for both
    # braced and braceless bodies. The structural checks below run the
    # unroll rule in isolation to prove the bail-out (not a corrupted
    # unroll) — a corrupted unroll would consume 'i', so as a second,
    # pipeline-level check, the full transform() must now raise
    # TranslationError (Task 3's hard-fail) naming the still-intact 'i',
    # since a bailed-out loop leaves a genuinely untranslatable
    # variable-argument shuffle behind.
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
    result = UnrollConstantShuffleLoopRule().apply(source, ctx)

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

    # Pipeline level: the untouched 'i' now reaches ShuffleXorRule,
    # which can't resolve it — the full transform must hard-fail.
    transformer = CUDAToRIPPLETransformer(TranslationContext())
    with pytest.raises(TranslationError) as exc_info:
        transformer.transform(source)
    assert "not a compile-time constant" in str(exc_info.value)
    assert "'i'" in str(exc_info.value)


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
    result = UnrollConstantShuffleLoopRule().apply(source, ctx)

    assert 'strlen(";")' in result
    assert "ripple_shuffle(" not in result
    assert ctx.errors == []
    assert "*val_ptr = val;\n}" in result

    # Pipeline level: the untouched 'i' now reaches ShuffleXorRule,
    # which can't resolve it — the full transform must hard-fail (Task
    # 3), proving this was a genuine bail-out rather than a corrupted
    # unroll that silently consumed 'i'.
    transformer = CUDAToRIPPLETransformer(TranslationContext())
    with pytest.raises(TranslationError) as exc_info:
        transformer.transform(source)
    assert "not a compile-time constant" in str(exc_info.value)
    assert "'i'" in str(exc_info.value)


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
    result = UnrollConstantShuffleLoopRule().apply(source, ctx)

    assert "/* note */" in result
    assert "ripple_shuffle(" not in result
    assert ctx.errors == []
    assert "*val_ptr = val;\n}" in result

    # Pipeline level: the untouched 'i' now reaches ShuffleXorRule,
    # which can't resolve it — the full transform must hard-fail (Task
    # 3), proving this was a genuine bail-out rather than a corrupted
    # unroll that silently consumed 'i'.
    transformer = CUDAToRIPPLETransformer(TranslationContext())
    with pytest.raises(TranslationError) as exc_info:
        transformer.transform(source)
    assert "not a compile-time constant" in str(exc_info.value)
    assert "'i'" in str(exc_info.value)


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
    result = UnrollConstantShuffleLoopRule().apply(source, ctx)

    assert "/* note [3] } */" in result
    assert "ripple_shuffle(" not in result
    assert ctx.errors == []
    # Structural integrity: trailing assignment still inside the
    # function, immediately followed by the function's own closing
    # brace — not floating outside it.
    assert "*val_ptr = val;\n}" in result
    assert result.index("/* note [3] } */") < result.index("*val_ptr = val;")

    # Pipeline level: the untouched 'i' now reaches ShuffleXorRule,
    # which can't resolve it — the full transform must hard-fail (Task
    # 3), proving this was a genuine bail-out rather than a corrupted
    # unroll that silently consumed 'i'.
    transformer = CUDAToRIPPLETransformer(TranslationContext())
    with pytest.raises(TranslationError) as exc_info:
        transformer.transform(source)
    assert "not a compile-time constant" in str(exc_info.value)
    assert "'i'" in str(exc_info.value)


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
    result = UnrollConstantShuffleLoopRule().apply(source, ctx)

    assert "// note [3] }" in result
    assert "ripple_shuffle(" not in result
    assert ctx.errors == []
    assert "*val_ptr = val;\n}" in result
    assert result.index("// note [3] }") < result.index("*val_ptr = val;")

    # Pipeline level: the untouched 'i' now reaches ShuffleXorRule,
    # which can't resolve it — the full transform must hard-fail (Task
    # 3), proving this was a genuine bail-out rather than a corrupted
    # unroll that silently consumed 'i'.
    transformer = CUDAToRIPPLETransformer(TranslationContext())
    with pytest.raises(TranslationError) as exc_info:
        transformer.transform(source)
    assert "not a compile-time constant" in str(exc_info.value)
    assert "'i'" in str(exc_info.value)


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


def test_butterfly_allreduce_recognized_as_single_reduceadd():
    # The exact motivating example: a full-warp (32-lane) XOR-butterfly
    # all-reduce with a full mask should now translate to ONE
    # ripple_reduceadd call instead of being unrolled into 5 separate
    # ripple_shuffle calls — better output for a pattern RIPPLE has a
    # native intrinsic for, matching WarpReductionRule's existing
    # idiom for the shfl_down-halving variant of the same idea.
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
    assert "ripple_reduceadd(0b1, val)" in result
    assert "ripple_shuffle(" not in result
    assert "__shfl_xor_sync" not in result


def test_butterfly_allreduce_fires_on_real_fixture():
    # The real fixture already shipped in tests/examples/ — proves this
    # isn't only working on a synthetic example.
    source = (Path(__file__).parent / "examples" / "butterfly_reduction.cu").read_text()
    result = translate_cuda_source(source)
    assert "ripple_reduceadd(0b1, val)" in result
    assert "ripple_shuffle(" not in result


def test_butterfly_allreduce_accepts_symbolic_warpsize_bound():
    # Real CUDA code commonly writes the bound symbolically rather than
    # hardcoding 32 — must be recognized the same way.
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
    assert "ripple_reduceadd(0b1, val)" in result
    assert "ripple_shuffle(" not in result


def test_butterfly_allreduce_does_not_fire_on_partial_width_loop():
    # CRITICAL correctness guard: a loop bounded to fewer than the full
    # warp (here, 8 lanes) is a segmented/sub-group reduction, NOT a
    # whole-block all-reduce. ripple_reduceadd(0b1, ...) always reduces
    # across the entire dimension, so firing here would silently change
    # the kernel's meaning (every lane would get the sum of all 32
    # lanes instead of just its own group of 8). Must fall through to
    # UnrollConstantShuffleLoopRule instead, which unrolls it correctly
    # (3 iterations: i = 1, 2, 4) without attaching any
    # reduction-specific meaning to the loop.
    source = """
__global__ void segmentedReduce(float *data) {
    float val = data[threadIdx.x];
    for (int i = 1; i < 8; i *= 2) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    assert "ripple_reduceadd" not in result
    assert result.count("ripple_shuffle(") == 3  # i = 1, 2, 4


def test_butterfly_allreduce_does_not_fire_on_non_full_mask():
    # CRITICAL correctness guard: a non-full mask means not every lane
    # participates in the shuffle, so this is not provably a whole-warp
    # all-reduce even though the loop shape otherwise matches exactly.
    # Must fall through to UnrollConstantShuffleLoopRule rather than
    # assume full participation.
    source = """
__global__ void partialMaskReduce(float *data) {
    float val = data[threadIdx.x];
    for (int i = 1; i < 32; i *= 2) {
        val += __shfl_xor_sync(0x0000ffff, val, i);
    }
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    assert "ripple_reduceadd" not in result
    assert result.count("ripple_shuffle(") == 5  # i = 1, 2, 4, 8, 16


def test_butterfly_allreduce_does_not_collide_with_warp_reduction_rule():
    # Regression guard: the classic warpSize/2 + shfl_down + accumulate
    # shape must still go through WarpReductionRule unchanged — the two
    # rules match mutually exclusive loop shapes (init=1-doubling+xor
    # vs init=warpSize/2-halving+down), so there should be no overlap.
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


def test_predicated_unroll_halving_loop_with_runtime_lower_bound():
    # The motivating shape: offset's value set (16, 8, 4, 2, 1) is
    # 100% determined by the literal init/bound/step — min_offset only
    # decides how many of those 5 already-known iterations actually
    # run, not which values occur.
    source = """
__global__ void haloExchange(float *data, int min_offset) {
    float val = data[threadIdx.x];
    for (int offset = 16; offset > 0 && offset > min_offset; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    assert "__shfl_down_sync" not in result
    assert result.count("ripple_shuffle(") == 5
    assert "if (16 > min_offset)" in result
    assert "if (8 > min_offset)" in result
    assert "if (4 > min_offset)" in result
    assert "if (2 > min_offset)" in result
    assert "if (1 > min_offset)" in result


def test_predicated_unroll_doubling_loop_with_runtime_upper_bound():
    # Mirror shape, opposite direction: '<' literal clause paired with
    # a '<' runtime clause — proves direction-matching isn't hardcoded
    # to the halving/'>' case.
    source = """
__global__ void butterflyPartial(float *data, int max_i) {
    float val = data[threadIdx.x];
    for (int i = 1; i < 32 && i < max_i; i *= 2) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    assert "__shfl_xor_sync" not in result
    assert result.count("ripple_shuffle(") == 5
    assert "if (1 < max_i)" in result
    assert "if (16 < max_i)" in result


@requires_clang
def test_predicated_unroll_output_passes_syntax_check():
    source = """
__global__ void haloExchange(float *data, int min_offset) {
    float val = data[threadIdx.x];
    for (int offset = 16; offset > 0 && offset > min_offset; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    success, output = verify_ripple_syntax(result)
    assert success, output


def test_predicated_unroll_declines_on_direction_mismatch_and_hard_fails():
    # CRITICAL correctness guard: a '>' literal clause paired with a
    # '<' runtime clause breaks the "once false, stays false"
    # guarantee independent per-iteration guards rely on. Must decline
    # entirely — leaving 'offset' unsubstituted — so the existing
    # hard-fail machinery (not a corrupted guess) is what catches this.
    source = """
__global__ void mismatched(float *data, int min_offset) {
    float val = data[threadIdx.x];
    for (int offset = 16; offset > 0 && offset < min_offset; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    data[threadIdx.x] = val;
}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    with pytest.raises(TranslationError) as exc_info:
        transformer.transform(source)
    assert "not a compile-time constant" in str(exc_info.value)
    assert "'offset'" in str(exc_info.value)


def test_predicated_unroll_declines_when_body_touches_runtime_id_and_hard_fails():
    # CRITICAL correctness guard: if the body reassigns or reads the
    # runtime bound, the "fixed threshold for the loop's whole
    # lifetime" assumption the guards depend on no longer holds. Must
    # decline rather than guess it's still safe.
    source = """
__global__ void bodyTouchesBound(float *data, int min_offset) {
    float val = data[threadIdx.x];
    for (int offset = 16; offset > 0 && offset > min_offset; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset) + min_offset;
    }
    data[threadIdx.x] = val;
}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    with pytest.raises(TranslationError) as exc_info:
        transformer.transform(source)
    assert "not a compile-time constant" in str(exc_info.value)


def test_predicated_unroll_reuses_unsafe_body_token_guard():
    # Proves the delegation to UnrollConstantShuffleLoopRule.UNSAFE_BODY_TOKENS
    # is actually wired up, not just referenced — a comment containing a
    # stray brace must still bail this rule out too, for the same
    # invisible-corruption reason the original 6 hardening rounds fixed.
    source = """
__global__ void hasComment(float *data, int min_offset) {
    float val = data[threadIdx.x];
    for (int offset = 16; offset > 0 && offset > min_offset; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset); // note [3] }
    }
    data[threadIdx.x] = val;
}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    with pytest.raises(TranslationError) as exc_info:
        transformer.transform(source)
    assert "not a compile-time constant" in str(exc_info.value)


@requires_clang
def test_predicated_unroll_braceless_body_supported():
    source = """
__global__ void haloExchange(float *data, int min_offset) {
    float val = data[threadIdx.x];
    for (int offset = 16; offset > 0 && offset > min_offset; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    assert result.count("ripple_shuffle(") == 5
    success, output = verify_ripple_syntax(result)
    assert success, output


def test_predicated_unroll_braceless_declines_on_control_flow_body():
    # Proves the delegation to UnrollConstantShuffleLoopRule.CONTROL_FLOW_PREFIX
    # is actually wired up for the braceless path.
    source = """
__global__ void controlFlowBody(float *data, int min_offset) {
    float val = data[threadIdx.x];
    for (int offset = 16; offset > 0 && offset > min_offset; offset /= 2)
        if (offset > 4) val += __shfl_down_sync(0xffffffff, val, offset);
    data[threadIdx.x] = val;
}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    with pytest.raises(TranslationError):
        transformer.transform(source)


def test_ternary_dispatch_resolves_declaration_form():
    # The MVP shape: a fresh-declared variable assigned directly from a
    # shuffle call whose dynamic argument is a two-way ternary with
    # compile-time-constant branches. Both literal-argument calls must
    # end up hoisted into distinct ripple_shuffle calls — this is the
    # non-loop sibling of UnrollConstantShuffleLoopRule's loop-shaped
    # resolution (GitHub issue #11).
    source = """
__global__ void kernel(float *data, int lane) {
    float val = data[lane];
    float neighbor = __shfl_xor_sync(0xffffffff, val, lane < 16 ? 1 : 2);
    data[lane] = neighbor;
}
"""
    result = translate_cuda_source(source)
    assert "__shfl_xor_sync" not in result
    # NOTE: deliberately not "?" not in result — asserting the original
    # ternary is gone from the kernel body is a more direct check than a
    # blanket "no '?' anywhere in the file" assertion regardless. (This
    # comment previously cited a since-removed ripple_atomic_cas fallback
    # macro that also contained a '?' as the reason — that macro no
    # longer exists, but the more-direct assertion below is still the
    # better check on its own merits.)
    assert "? 1 : 2" not in result
    assert result.count("ripple_shuffle(") == 2
    assert "if (lane < 16)" in result
    hoisted_bodies = re.findall(r'return k \^ \((\d+)\);', result)
    assert sorted(int(v) for v in hoisted_bodies) == [1, 2]


def test_ternary_dispatch_resolves_compound_assign_form():
    # Same mechanism, the accumulate-into-existing-variable shape
    # (VAR += INTRINSIC(...)) rather than a fresh declaration — proves
    # the rule handles both statement shapes it claims to, not just one.
    source = """
__global__ void kernel(float *data, int lane, int use_far) {
    float sum = data[lane];
    sum += __shfl_down_sync(0xffffffff, sum, use_far ? 8 : 1);
    data[lane] = sum;
}
"""
    result = translate_cuda_source(source)
    assert "__shfl_down_sync" not in result
    assert result.count("ripple_shuffle(") == 2
    assert "if (use_far)" in result
    assert "sum += ripple_shuffle(" in result


def test_ternary_dispatch_resolves_plain_reassignment_form():
    # Third statement shape PATTERN_ASSIGN claims to support: a plain
    # '=' reassignment to an already-declared variable, with no type
    # prefix and no compound-assign operator — distinct from both the
    # declaration form (fresh TYPE VAR = ...) and the compound-assign
    # form (VAR += ...) already covered above. Proves PATTERN_DECL's
    # "requires two identifiers before '='" gate correctly does NOT
    # swallow this shape, and PATTERN_ASSIGN's bare '=' alternative
    # correctly does.
    source = """
__global__ void kernel(float *data, int lane, int use_far) {
    float val;
    val = __shfl_xor_sync(0xffffffff, val, use_far ? 4 : 2);
    data[lane] = val;
}
"""
    result = translate_cuda_source(source)
    assert "__shfl_xor_sync" not in result
    assert result.count("ripple_shuffle(") == 2
    assert "if (use_far)" in result
    assert "val = ripple_shuffle(" in result
    hoisted_bodies = re.findall(r'return k \^ \((\d+)\);', result)
    assert sorted(int(v) for v in hoisted_bodies) == [2, 4]


@pytest.mark.parametrize("intrinsic,arg_name", [
    ("__shfl_xor_sync", "lane_mask"),
    ("__shfl_up_sync", "delta"),
    ("__shfl_down_sync", "delta"),
    ("__shfl_sync", "src_lane"),
])
def test_ternary_dispatch_resolves_across_all_four_intrinsics(intrinsic, arg_name):
    # Proves the rule's shared _INTRINSIC alternation actually covers
    # all 4 shuffle intrinsics, not just the one used in the other tests.
    source = f"""
__global__ void kernel(float *data, int lane, int cond) {{
    float val = data[lane];
    float result = {intrinsic}(0xffffffff, val, cond ? 3 : 5);
    data[lane] = result;
}}
"""
    result = translate_cuda_source(source)
    assert intrinsic not in result
    assert result.count("ripple_shuffle(") == 2


def test_ternary_dispatch_declines_when_a_branch_is_not_constant():
    # CRITICAL correctness guard: if either ternary branch references a
    # kernel-local variable rather than a literal, the value set is no
    # longer provably static (it could be anything 'n' takes at
    # runtime) — this rule must decline entirely (leave the ternary
    # untouched) rather than guess, same "decline rather than guess"
    # convention as every other rule in this file's shuffle-handling
    # family. The untouched ternary then correctly reaches
    # ShuffleXorRule, which hard-fails on it exactly as it does today.
    source = """
__global__ void kernel(float *data, int lane, int n) {
    float val = data[lane];
    float neighbor = __shfl_xor_sync(0xffffffff, val, lane < 16 ? n : 2);
    data[lane] = neighbor;
}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    with pytest.raises(TranslationError) as exc_info:
        transformer.transform(source)
    assert "not a compile-time constant" in str(exc_info.value)


def test_ternary_dispatch_declines_on_nested_ternary():
    # A 3-way (or more) chained ternary is out of scope for this rule —
    # LIT_B ('cond2 ? 5 : 7') contains '?'/':' and fails
    # _is_compile_time_constant_expr, so this rule declines and the
    # untouched (outer) ternary reaches ShuffleXorRule, which hard-fails.
    # This is a deliberate bail-out, not a translation attempt — see the
    # rule's docstring.
    source = """
__global__ void kernel(float *data, int lane, int cond1, int cond2) {
    float val = data[lane];
    float neighbor = __shfl_xor_sync(0xffffffff, val, cond1 ? 3 : cond2 ? 5 : 7);
    data[lane] = neighbor;
}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    with pytest.raises(TranslationError):
        transformer.transform(source)


def test_ternary_dispatch_passes_through_width_argument():
    source = """
__global__ void kernel(float *data, int lane, int wide) {
    float val = data[lane];
    float r = __shfl_sync(0xffffffff, val, wide ? 0 : 1, 16);
    data[lane] = r;
}
"""
    result = translate_cuda_source(source)
    assert result.count("ripple_shuffle(") == 2
    hoisted_bodies = re.findall(r'return \((\d+)\);', result)
    assert sorted(int(v) for v in hoisted_bodies) == [0, 1]


@requires_clang
def test_ternary_dispatch_output_passes_syntax_check():
    source = """
__global__ void kernel(float *data, int lane) {
    float val = data[lane];
    float neighbor = __shfl_xor_sync(0xffffffff, val, lane < 16 ? 1 : 2);
    data[lane] = neighbor;
}
"""
    result = translate_cuda_source(source)
    success, output = verify_ripple_syntax(result)
    assert success, output


def test_predicated_unroll_does_not_collide_with_plain_unroll_rule():
    # A plain (non-compound-condition) loop must still go through
    # UnrollConstantShuffleLoopRule exactly as before — the new rule's
    # pattern requires '&&' and must not accidentally also match this.
    source = """
__global__ void plainLoop(float *data) {
    float val = data[threadIdx.x];
    for (int i = 1; i < 8; i *= 2) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    # NOTE: deliberately not "if (" not in result — asserting on which
    # rule's warning fired is a more direct, less brittle check than a
    # blanket "no 'if (' anywhere in the file" assertion regardless.
    # (This comment previously cited since-removed ripple_atomic_max/min
    # fallback macros that also contained "if (" as the reason — those
    # macros no longer exist, but the more-direct assertion below is
    # still the better check on its own merits.)
    assert "Unrolled loop over 'i'" in result  # plain rule fired
    assert "Predicated-unrolled" not in result  # new rule did not fire
    assert result.count("ripple_shuffle(") == 3  # i = 1, 2, 4

def test_warp_minmax_reduction_translates_fmaxf_to_ripple_reducemax():
    source = """
__global__ void warpMax(float *data) {
    float local_max = data[threadIdx.x];
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, local_max, offset);
        local_max = fmaxf(local_max, other);
    }
    data[threadIdx.x] = local_max;
}
"""
    result = translate_cuda_source(source)
    assert "__shfl_down_sync" not in result
    assert "ripple_reducemax(0b1, local_max)" in result


def test_warp_minmax_reduction_tolerates_optional_width_argument():
    # WarpReductionRule's '+=' sibling tolerates __shfl_down_sync's
    # optional 4th (width) argument — this rule should too, for the
    # same reason: real CUDA code can pass it, and a reader shouldn't
    # have to guess why the min/max variant is narrower than the '+='
    # one it's explicitly documented as a sibling of.
    source = """
__global__ void warpMax(float *data) {
    float local_max = data[threadIdx.x];
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, local_max, offset, 32);
        local_max = fmaxf(local_max, other);
    }
    data[threadIdx.x] = local_max;
}
"""
    result = translate_cuda_source(source)
    assert "__shfl_down_sync" not in result
    assert "ripple_reducemax(0b1, local_max)" in result


def test_warp_minmax_reduction_translates_fminf_to_ripple_reducemin():
    source = """
__global__ void warpMin(float *data) {
    float local_min = data[threadIdx.x];
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, local_min, offset);
        local_min = fminf(local_min, other);
    }
    data[threadIdx.x] = local_min;
}
"""
    result = translate_cuda_source(source)
    assert "__shfl_down_sync" not in result
    assert "ripple_reducemin(0b1, local_min)" in result


def test_warp_minmax_reduction_does_not_collide_with_plain_warp_reduction_rule():
    # Regression guard: the classic '+=' accumulate shape must still go
    # through WarpReductionRule unchanged — the two rules match
    # mutually exclusive body shapes (single '+=' statement vs. a
    # two-statement declare-then-fmaxf/fminf), so there's no overlap.
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
    assert "ripple_reducemax" not in result
    assert "ripple_reducemin" not in result


def test_warp_minmax_reduction_fires_on_real_fixture_snippet():
    # The exact loop from tests/examples/cuda_kernels.cu's softmax
    # kernel — the specific loop that orphaned that fixture (it's the
    # only unresolved shuffle in the whole 10-kernel file).
    source = """
__global__ void softmax(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_max = (idx < n) ? input[idx] : -1.0f;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, local_max, offset);
        local_max = fmaxf(local_max, other);
    }
    output[idx] = local_max;
}
"""
    result = translate_cuda_source(source)
    assert "ripple_reducemax(0b1, local_max)" in result


class TestBoilerplateGeneration:
    """Tests for the RIPPLE boilerplate CUDAToRIPPLETransformer emits around
    translated kernels."""

    def test_hvx_pe_self_defined(self):
        """HVX_PE must be defined by the generated output itself, not assumed
        to come from <ripple.h> — no example in the Ripple docs relies on the
        header for its PE constant."""
        cuda_code = "__global__ void k(float *a) { a[0] = 1.0f; }"
        result = translate_cuda_source(cuda_code)
        assert "#define HVX_PE 0" in result

    def test_compile_flag_documented_in_output(self):
        """Generated output must tell the reader how to actually compile
        it — -fenable-ripple is required and easy to miss."""
        cuda_code = "__global__ void k(float *a) { a[0] = 1.0f; }"
        result = translate_cuda_source(cuda_code)
        assert "-fenable-ripple" in result

    def test_math_h_included_for_translated_math_intrinsics(self):
        """MathFunctionRule translates CUDA math intrinsics (__expf,
        __sqrtf_rn, etc.) to plain libm names like expf/sqrtf, which are
        declared in <math.h> — without it, clang rejects the call with
        'call to undeclared library function' under -std=c99 and later
        (confirmed directly: this exact kernel failed to compile before
        <math.h> was added to the boilerplate)."""
        assert "#include <math.h>" in translate_cuda_source(
            "__global__ void k(float *a) { a[0] = 1.0f; }"
        )

    @requires_clang
    def test_translated_math_intrinsic_output_passes_syntax_check(self):
        source = """
__global__ void k(float *a, float *b) {
    int idx = threadIdx.x;
    b[idx] = __expf(a[idx]);
}
"""
        result = translate_cuda_source(source)
        success, output = verify_ripple_syntax(result)
        assert success, output


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
