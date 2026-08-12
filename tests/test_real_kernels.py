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
tracked as issue #8 — that bug currently has no coverage in this file.
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
]

SYNTAX_CHECK_PARAMS = [
    "ast_flat.cu",
    "ast_if_no_braces.cu",
    "atomics_cas_exch.cu",
    "bitwise_intrinsics.cu",
    "global_thread_index.cu",
    "warp_reduction.cu",
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
