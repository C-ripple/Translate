"""
Tests for the AST-based parser in frontends/source/cuda_frontend.py
(CUDALexer + AIRBuilder), independent of the regex-based translation path.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from frontends.source.cuda_frontend import CUDALexer, AIRBuilder, CUDAToRIPPLETransformer, TokenType
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
# a for-loop body containing an if-body (tests/examples/warp_reduction.cu in the repo).
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


def test_transform_warns_when_ast_finds_no_kernels():
    # No __global__ function in this source at all.
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    transformer.transform("int add(int a, int b) { return a + b; }")
    assert any("did not recognize any __global__ kernel" in w for w in ctx.warnings)


def test_transform_does_not_warn_on_valid_kernel():
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    transformer.transform(KERNEL_TWO_NESTED)
    assert not any("AST pre-pass" in w for w in ctx.warnings)


def test_transform_recognizes_launch_bounds_kernel():
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    transformer.transform(
        "__global__ __launch_bounds__(256) void kernel(float* a, int n) { "
        "int i = threadIdx.x; a[i] = 0; }"
    )
    assert not any("did not recognize any __global__" in w for w in ctx.warnings)


@pytest.mark.timeout(5)
def test_unclosed_paren_does_not_hang():
    # Regression test for a second EOF-handling gap found while fixing
    # Task 3 — same bug class as _parse_block's original hang, just in
    # the parameter-list loop.
    _parse("__global__ void broken(float *a")


@pytest.mark.timeout(5)
def test_non_consumable_token_in_param_position_does_not_hang():
    # Regression test for a no-progress gap in the parameter-list loop,
    # found by review: a bare NUMBER token in parameter position is not
    # EOF, so the earlier EOF-only guard didn't catch it. Neither
    # _parse_type, the name check, nor the array-bracket check advances
    # past a token they can't interpret, so the loop spun forever
    # re-parsing the same token before the no-progress guard was added.
    _parse("__global__ void broken(1)")


@pytest.mark.timeout(5)
def test_ampersand_ref_param_does_not_hang():
    # Regression test for the same no-progress gap as above, triggered by
    # a legal CUDA reference-parameter character ('&') that the parameter
    # parser doesn't recognize — a realistic input, not adversarial
    # garbage.
    _parse("__global__ void broken(float &a) { a = 0; }")


@pytest.mark.timeout(5)
def test_top_level_static_declaration_does_not_hang():
    # Regression test for a no-progress gap in build_translation_unit's
    # outer loop, found by review: _is_function_start() treats STATIC and
    # INLINE as function starters, but _parse_type() never consumes those
    # token types, so _parse_function() bailed via return None without
    # advancing at all — and the outer loop only advances in its "unknown
    # token" branch, which a matched function-start token never reaches.
    # An ordinary top-level "static int x;" declaration (e.g. preceding a
    # kernel in a real .cu file) hung forever before the fix.
    _parse("static int counter;")


@pytest.mark.timeout(5)
def test_static_declaration_before_kernel_still_finds_kernel():
    unit = _parse(
        "static int counter;\n\n"
        "__global__ void kernel(float *a, int n) {\n"
        "    int i = threadIdx.x;\n"
        "    if (i < n) a[i] = 0;\n"
        "}\n"
    )
    assert len(unit.functions) == 1
    assert unit.functions[0].name == "kernel"
    assert unit.functions[0].is_kernel


def test_transform_warns_when_ast_parse_fails():
    # GitHub issue #7 (CUDALexer.read_number() TypeError on trailing bare digit)
    # has been fixed. This test now uses monkeypatching to verify that
    # transform() correctly catches and warns on AST parse exceptions.
    # This exercises the except clause in transform(), giving coverage for that
    # branch (unlike the SyntaxError in AIRBuilder.expect(), which remains
    # structurally unreachable — see the commit message for issue #7).
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)

    with patch('frontends.source.cuda_frontend.AIRBuilder.build_translation_unit') as mock_build:
        mock_build.side_effect = RuntimeError("Simulated parse failure")
        transformer.transform("__global__ void dummy() {}")

    assert any("AST pre-pass failed" in w for w in ctx.warnings)


def test_lexer_handles_trailing_bare_digit():
    # CUDALexer.read_number() did `self.peek() in 'xXoObB'` to detect a
    # 0x/0o/0b prefix; peek() returns None at end-of-source, and
    # `None in 'xXoObB'` raises TypeError. A source ending in a bare
    # digit (nothing after it at all) triggered this. GitHub issue #7.
    lexer = CUDALexer("0")
    tokens = lexer.tokenize()
    number_tokens = [t for t in tokens if t.type == TokenType.NUMBER]
    assert len(number_tokens) == 1
    assert number_tokens[0].value == "0"
