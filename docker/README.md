# Hexagon toolchain (compile verification — deferred, not currently wired in)

**Status:** vendored and ready, but not built or used by the active test suite. The
project's actual compile-verification check (`tests/compile_verify.py`) uses a
lightweight, no-build `clang -fsyntax-only` check instead — see that file's
docstring. This full toolchain is heavy (30-90+ min build) and buys real
Hexagon/HVX semantic and codegen validation, which is more than this project has
needed so far; every translator bug found while this was under evaluation was
caught by grammar/logic reasoning and targeted tests, not by compiling anything.
Kept here for whenever RIPPLE-semantic correctness, not syntax, becomes the
actual bottleneck.

`hexagon-toolchain.Dockerfile` is a vendored copy of the official Hexagon/RIPPLE
toolchain build from `qualcomm/learn-ripple` (see the provenance comment at the
top of the file for the exact commit). It builds a real `clang` targeting
`hexagon-unknown-unknown-elf` with HVX and RIPPLE support, cloned and built
from `qualcomm/ripple` (the RIPPLE-enabled LLVM fork) — not a mock.

## Build (once)

```bash
./scripts/build-hexagon-toolchain.sh
```

This is a **heavy** build: full LLVM/clang from source, plus the Hexagon SDK
download and the ELD linker. Expect it to take well over 30 minutes and use
several GB of disk on first run. It only needs to be run once — the resulting
image (`cuda2ripple-hexagon-toolchain:latest`) is cached locally by Docker
and reused by every test run after that.

## What it would be for (if wired in)

A future, heavier `tests/compile_verify.py` could use this image to actually
compile translated RIPPLE C output through a real Hexagon-targeting compiler,
validating real semantics and HVX codegen rather than just C syntax. Nothing
in the test suite does this today — see the Status note above.

## Re-vendoring

If upstream's Dockerfile changes in a way that matters (e.g. the toolchain
install path or clang binary name moves), re-copy it wholesale from a fresh
`learn-ripple` checkout and update the provenance comment's commit hash —
don't hand-patch the vendored copy, since drift makes future re-vendoring
a manual diff nightmare instead of a straight copy.
