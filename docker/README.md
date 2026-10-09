# Hexagon toolchain and native verification

The regular test suite uses host C compilation and reference SIMD execution.
Native checks are separate and run through `scripts/verify_native_launch.py`.
They use a real Ripple clang, SDK linker and Hexagon simulator, and can detect
target failures that host tests cannot.

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

## Native launch verification

The verifier currently supports the cached toolchain layout used by the
neighboring rust-to-ripple compiler project: `artifacts/ripple-build/bin/clang`,
`artifacts/hexagon-sdk`, `artifacts/simulator-image.json` and
`scripts/run_hexagon_linker.py` / `run_hexagon_simulator.py`. Docker must be
running with the SDK simulator image cached. The vendored Dockerfile above is
a separate build route and is not automatically discovered by this script.

From the cuda2ripple repository:

```bash
venv/bin/python scripts/verify_native_launch.py \
  --toolchain-root ../rust-to-ripple/compiler \
  --thread-header output/verification/upstream-runtime/ripple/thread.h
```

The optional thread header must be the official
`clang/lib/Headers/ripple_include/ripple/thread.h` from Qualcomm's Ripple source.
The local verified header is pinned to commit
`3468adcb686511de236e0f991a8bc0af28730a0a`; its hash/provenance are retained.
Omit that option to run only the standalone kernel probes.

The verifier retains generated sources, compile/simulator logs, commands and
hashes in `output/verification/native-launch-verification.json`. It exits
nonzero when any reference or guard check fails and invalidates old success
before running. `--compiler` selects an isolated alternate clang. `--case`, `--optimization`
and `--float-mode` isolate compiler
issues without changing generated kernels or weakening the reference.

The default v68 mode now uses `-mno-hvx-qfloat -mhvx-ieee-fp`.
This resolves the masked float tail and numerical failures found with other
modes, including on the installed compiler. Native probes check integer/float
tails, 2D shape, worker assignment and IEEE edge cases using an independent
scalar reference. No tolerance or guard checks were relaxed.

`--float-mode default` reproduces QFloat numerical differences;
`--float-mode disabled` reproduces inactive-lane writes. The handwritten control
is available through `--case float_tail_baseline`. A modern compiler's
`strict-ieee` QFloat lowering applies only to v79+, so the verifier does not
select it for v68 merely because the flag is accepted.

Auditing the complete pinned SDK ZIP directory (19,039 entries) found no
`ripple_thd` headers/libraries. This also checks files excluded from the local
SDK extraction. The public compiler-rt Ripple sources provide vector helpers,
not vendor threading integration.

The target suite includes both labeled sequential SPMD reference probes and
actual SDK standalone hardware-thread probes. The latter use thread_create,
thread_join and hardware thread IDs, test 1/2/3/4/6 workers over a 7x3x2 grid,
and require nonzero simulator instruction counts for every secondary hardware
thread exercised. Automatic vectorization is disabled for normal driver loops;
explicit Ripple SIMD code is preserved.

The executable driver lives in tests/native/standalone_workers.c. Configure
global 128-byte HVX mode before spawning workers, then acquire/release HVX
contexts per worker. SDK 19.0.07 releases its allocator lock before finishing
the old owner's hardware-permission reset. The driver adds a handoff lock around
allocation/release, uses nonblocking acquisition, and runs kernels outside that
lock. Six-worker float/integer guard and full-output failures are retained as
regressions. Disassembly/provenance of the exact SDK helper are retained in the
local verification artifacts.

The verifier stops entire compiler process groups on timeout, records failures,
and continues independent cases. Its --compile-timeout default is 300 seconds
for the installed unoptimized compiler; missing tools or failed checks still
exit nonzero.

Reports separately identify native_thread_concurrency_verified (standalone)
and vendor_runtime_concurrency_verified (QuRT/QHPI). Real hardware-thread proof
does not certify the unavailable QuRT/QHPI Ripple libraries.

## Re-vendoring

If upstream's Dockerfile changes in a way that matters (e.g. the toolchain
install path or clang binary name moves), re-copy it wholesale from a fresh
`learn-ripple` checkout and update the provenance comment's commit hash —
don't hand-patch the vendored copy, since drift makes future re-vendoring
a manual diff nightmare instead of a straight copy.
