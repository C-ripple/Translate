"""
Tests for the AST-based parser in frontends/source/cuda_frontend.py
(CUDALexer + AIRBuilder), independent of the regex-based translation path.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from frontends.source.cuda_frontend import CUDALexer, AIRBuilder
from core.semantic_model import TranslationContext


def _parse(source: str):
    """Run the full lex + AST-build pipeline and return the resulting translation unit."""
    lexer = CUDALexer(source)
    tokens = lexer.tokenize()
    builder = AIRBuilder(tokens, TranslationContext())
    return builder.build_translation_unit()


# Brace-nesting depth 0: no braces beyond the function body itself.
KERNEL_FLAT = """
__global__ void flat(float *a) {
    int i = threadIdx.x;
    a[i] = 0;
}
"""

# Brace-nesting depth 1: a single-level nested block (an if-body with braces).
KERNEL_ONE_NESTED = """
__global__ void oneNested(float *a, int n) {
    int i = threadIdx.x;
    if (i < n) {
        a[i] = 0;
    }
}
"""

# Brace-nesting depth 2: matches the exact shape that hung before the fix —
# a for-loop body containing an if-body (test_reduction.cu in the repo).
KERNEL_TWO_NESTED = """
__global__ void reduce(float *val) {
    float sum = *val;
    for (int offset = 16; offset > 0; offset /= 2) {
        if (offset > 0) {
            sum += offset;
        }
    }
    *val = sum;
}
"""


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "source,expected_function_count",
    [
        (KERNEL_FLAT, 1),
        (KERNEL_ONE_NESTED, 1),
        (KERNEL_TWO_NESTED, 1),
    ],
)
def test_parses_without_hanging(source, expected_function_count):
    unit = _parse(source)
    assert len(unit.functions) == expected_function_count
    assert unit.functions[0].is_kernel
