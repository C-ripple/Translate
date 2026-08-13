"""
Translation + real-syntax-check tests against the sample kernels in
tests/examples/ — these are kernels shaped like real CUDA code
(nested control flow, atomics, warp shuffles), not the minimal
one-liners used to unit-test individual translation rules.

Structural assertions (does it look translated) run always.
Syntax-check assertions (does it actually parse as valid C, against
the real RIPPLE API surface) run whenever clang is on PATH — see
tests/compile_verify.py.

warp_reduction.cu previously failed its syntax check here — GitHub
issue #10 tracked that WarpReductionRule emitted a ripple_reduceadd
call with the wrong arity (1 arg, real API takes 2). That's fixed
(WarpReductionRule now emits ripple_reduceadd(0b1, val)), so this file
now passes its syntax check cleanly like the others. Note this file's
loop shape gets fully replaced by WarpReductionRule (priority 85)
before ShuffleDownRule (priority 70) ever sees the __shfl_down_sync
call, so it doesn't exercise the separate warp-shuffle lambda bug
tracked as issue #8.

warp_shuffle_xor.cu covers issue #8: a direct __shfl_xor_sync call
(butterfly-exchange shape, not the halving loop WarpReductionRule
matches) that reaches ShuffleXorRule. All four shuffle rules used to
emit either a C++ lambda or a nested function definition — neither
valid C — as an inline ripple_shuffle() argument; they now hoist a
uniquely-named, file-scope helper function instead (see
TranslationContext.hoisted_declarations in core/semantic_model.py).

cuda_kernels.cu is a larger, 10-kernel fixture that was previously
orphaned entirely (wired into no test) because it failed to translate
at all — a softmax kernel's max-reduction shuffle loop (fmaxf, not
'+=') didn't match any rule's shape. WarpMinMaxReductionRule closes
that gap, so this file now translates successfully and is covered
here structurally. It's deliberately NOT in SYNTAX_CHECK_PARAMS yet:
getting it through a real clang syntax check requires fixing three
separate, unrelated gaps this file is the first fixture to exercise —
scalar (non-array) __shared__ declarations aren't translated at all
(only array-form ones are), a bare warpSize token used outside a
recognized loop pattern passes through untranslated, and a few CUDA
math intrinsics (expf, INFINITY, __float_as_int) have no RIPPLE
mapping. None of that is shuffle-related; closing it is separate,
larger, unscoped work.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from frontends.source.cuda_frontend import translate_cuda_source
from tests.compile_verify import requires_clang, verify_ripple_syntax

EXAMPLES_DIR = Path(__file__).parent / "examples"

KERNEL_FILES = [
    "ast_flat.cu",
    "ast_if_no_braces.cu",
    "atomics_cas_exch.cu",
    "bitwise_intrinsics.cu",
    "global_thread_index.cu",
    "warp_reduction.cu",
    "warp_shuffle_xor.cu",
    "butterfly_reduction.cu",
    "cuda_kernels.cu",
]

SYNTAX_CHECK_PARAMS = [
    "ast_flat.cu",
    "ast_if_no_braces.cu",
    "atomics_cas_exch.cu",
    "bitwise_intrinsics.cu",
    "global_thread_index.cu",
    "warp_reduction.cu",
    "warp_shuffle_xor.cu",
    "butterfly_reduction.cu",
]


@pytest.mark.parametrize("filename", KERNEL_FILES)
def test_translates_without_error(filename):
    source = (EXAMPLES_DIR / filename).read_text()
    result = translate_cuda_source(source)
    assert result, f"{filename} produced empty output"
    assert "__global__" not in result, f"{filename}: untranslated __global__ remains"


@requires_clang
@pytest.mark.parametrize("filename", SYNTAX_CHECK_PARAMS)
def test_translated_output_is_valid_syntax(filename):
    source = (EXAMPLES_DIR / filename).read_text()
    translated = translate_cuda_source(source)
    success, output = verify_ripple_syntax(translated)
    assert success, (
        f"{filename}: translated output failed syntax check:\n{output}\n\n"
        f"--- translated source ---\n{translated}"
    )
