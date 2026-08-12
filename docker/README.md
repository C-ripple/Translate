# Hexagon toolchain (compile verification)

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

## What it's for

`tests/compile_verify.py` uses this image to actually compile translated
RIPPLE C output through a real Hexagon-targeting compiler, rather than
trusting string-based assertions about what the translator's own output
looks like. Tests using it (see `tests/test_compile_verification.py` and
the real-kernel tests) skip automatically — not fail — when this image
hasn't been built locally, so the normal `pytest tests/` run never depends
on Docker being installed or this image existing.

## Re-vendoring

If upstream's Dockerfile changes in a way that matters (e.g. the toolchain
install path or clang binary name moves), re-copy it wholesale from a fresh
`learn-ripple` checkout and update the provenance comment's commit hash —
don't hand-patch the vendored copy, since drift makes future re-vendoring
a manual diff nightmare instead of a straight copy.
