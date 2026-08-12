"""
Lightweight syntax-verification helper: runs translated RIPPLE C through a
standard, locally-installed clang in syntax-only mode (-fsyntax-only).

This catches gross syntax breakage in the translator's output (unbalanced
constructs, invalid tokens, C++-only syntax accidentally emitted into C
output) using only a stub ripple.h (tests/stub_headers/ripple.h) declaring
the real upstream RIPPLE API by name. It does NOT validate real Hexagon/HVX
semantics or codegen — that requires the full toolchain vendored in
docker/ (see docker/README.md), which is deliberately not wired in here.

Requires only a standard clang on PATH (already present on most dev
machines) — no Docker, no long build, no network access.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest

STUB_HEADER_DIR = Path(__file__).parent / "stub_headers"


def clang_available() -> bool:
    """Check whether a clang binary is available on PATH."""
    try:
        result = subprocess.run(["clang", "--version"], capture_output=True, timeout=10)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return result.returncode == 0


def verify_ripple_syntax(ripple_c_source: str) -> tuple[bool, str]:
    """
    Check translated RIPPLE C source for valid C syntax using a generic
    clang and a stub ripple.h (-fsyntax-only, -xc — no codegen, no linking).

    This does NOT validate RIPPLE semantics or real Hexagon/HVX codegen —
    only that the output is syntactically valid C referencing the real
    RIPPLE API by name. See docker/README.md for the heavier check.

    Returns (success, diagnostic_output). diagnostic_output is clang's
    combined stdout+stderr — empty on success, the compiler's error text
    on failure.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "kernel.c"
        src_path.write_text(ripple_c_source)

        result = subprocess.run(
            [
                "clang",
                "-fsyntax-only",
                "-xc",
                f"-I{STUB_HEADER_DIR}",
                str(src_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode == 0, output


requires_clang = pytest.mark.skipif(
    not clang_available(),
    reason="clang not found on PATH — see tests/compile_verify.py",
)
