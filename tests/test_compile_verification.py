"""Smoke test for the lightweight syntax-verification helper itself."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.compile_verify import requires_clang, verify_ripple_syntax
from frontends.source.cuda_frontend import translate_cuda_source


@requires_clang
def test_translated_vector_add_is_valid_syntax():
    source = """
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""
    translated = translate_cuda_source(source)
    success, output = verify_ripple_syntax(translated)
    assert success, f"Translated output failed syntax check:\n{output}\n\n--- translated source ---\n{translated}"


@requires_clang
def test_deliberately_broken_c_fails_syntax_check():
    # Negative control: confirms the helper actually catches a real failure
    # rather than silently reporting success no matter what it's given.
    success, output = verify_ripple_syntax("this is not valid C at all {{{")
    assert not success
    assert output  # some diagnostic text should be present
