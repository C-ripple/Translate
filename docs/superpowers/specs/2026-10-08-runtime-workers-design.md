# Runtime worker scheduling and native verification

Each logical CUDA grid block remains an indivisible task with its exact static
SIMD shape. A new optional worker launcher accepts ripple_thd_block_t from an
already running QuRT/QHPI environment. It queries dimension zero for current
worker ID and count, and assigns flattened grid blocks cyclically across workers.
All active workers invoke the launcher; the surrounding runtime owns fork/join
or completion barriers. The launcher does not initialize, destroy, or resize
someone else's runtime.

Expose threaded=True through Python and HTTP and --threaded through the CLI.
The existing sequential launcher remains available. A threaded translation
includes <ripple/thread.h> and an additional int <kernel>_ripple_launch_workers.
Return explicit statuses for invalid worker IDs/counts and grid-volume overflow.
Use checked uint64_t products and overflow-safe loop advancement. Empty grids are
successful no-ops. SIMD shapes are unaffected by worker count.

We deliberately use the stable ID/size SPMD queries, not loop annotations or
init APIs whose public compiler headers differ from the documentation.
A one-dimensional worker block is the required caller contract; launchers
distribute all grid axes rather than treating worker count as grid.x.
The opaque API exposes no rank query. Query only dimension zero; unsupported
higher-dimension queries are tested with a runtime that traps on them.

Verification must distinguish:
1. Generated C with a scalar SIMD harness and real pthreads checks concurrent
   workers, once-only ownership, empty grids, uneven workloads, and oversubscription.
2. The available Ripple compiler + actual Hexagon simulator checks unmodified
   generated SIMD code against independent references and guard regions.
3. Actual SDK standalone hardware threads exercise native SIMD concurrently;
   hardware-ID queries and independent simulator instruction counts establish
   real execution. Configure global HVX mode before fork and acquire contexts
   per worker. See 2026-10-09-native-hardware-workers.md.
4. Native QuRT/QHPI concurrency requires its runtime headers and library. If those
   are absent, record that as unavailable rather than calling a stub native proof.
Reports retain compiler identity, commands, source hashes, logs, and scope.

Upstream spot check: qualcomm/ripple commit
3468adcb686511de236e0f991a8bc0af28730a0a, clang/lib/Headers/ripple_include/ripple/thread.h.
The compiler supplies opaque ripple_thd_block_t and ID/size declarations and
delegates runtime integration with include_next; it is not a QuRT runtime library.

Native findings: use -mno-hvx-qfloat -mhvx-ieee-fp together on v68.
The installed compiler passes guarded float tails and IEEE edge cases in this
mode. Default QFloat differs from exact IEEE values, and QFloat-disabled
lowering without IEEE support writes inactive lanes, also in handwritten Ripple.
Modern strict-ieee QFloat lowering applies only to v79+. Retain diagnostic reports.
The full pinned SDK ZIP directory contains no Ripple vendor thread runtime.
