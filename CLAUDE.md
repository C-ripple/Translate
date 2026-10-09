# cuda2ripple / C-Ripple — Project Notes

Standing, cross-cutting facts about this project that don't belong to any one task.
Task-specific design history lives in `docs/superpowers/specs/` and
`docs/superpowers/plans/`; specific known gaps are tracked as GitHub issues on
`C-ripple/Translate`. This file is for facts that should inform *any* future work here,
not a specific feature.

## Primary source of truth for Ripple API claims

`temp_ripple_docs/` is a local, gitignored checkout of `qualcomm/learn-ripple` — the
actual upstream docs. It exists only on disk, not in git, so it's absent from any git
worktree (a worktree only gets tracked files).

**Never trust a secondhand summary or citation of Ripple's API — including from a
subagent report — without spot-checking it directly against this directory first.** A
research subagent's report was trusted once without re-verifying one specific citation
(`temp_ripple_docs/src/ripple-spec/multi-threading.md` and an entire `ripple_thd_*`
multicore/thread API family). At the time (checkout `0dc48ff`, through 2026-07-10),
neither the file nor the API family existed anywhere in the actual docs. It shipped in
merged, pushed error messages before being caught and corrected. The lesson stands even
though the underlying fact has since changed (see below): a well-formatted, heavily-cited
report is not evidence of accuracy — grep the actual files, every time, even when a claim
matches something you remember rejecting before.

**Update, 2026-08-17: `multi-threading.md` and the `ripple_thd_*` API are now real.**
Qualcomm published the doc upstream that morning; pulling `temp_ripple_docs/` to
`ef517bf` confirms `src/ripple-spec/multi-threading.md` now exists (769 lines, added in
that commit) and `src/SUMMARY.md` now links it. Benoit Meister (Ripple's creator)
confirmed directly that Hexagon-side multi-threading via `ripple_thd_*` is landing in
"the top-of-tree version" next week, with QuRT/QHPI runtime support "in an upcoming
release" after that — i.e. **documented but not yet in the 21.0-alpha3 release this
translator currently targets**. Confirmed independently: `release-notes.md` at the same
commit is unchanged and still says 21.0-alpha3 has no native multicore construct and
suggests OpenMP for thread-level parallelism — the release notes haven't caught up to
the spec doc yet, consistent with "top-of-tree, not released." See
`ripple_thd_*` API details below.

## Ripple 21.0-alpha3 — real capabilities and limits (independently verified)

- **No atomics API at all.** Confirmed directly: `atomic` appears zero times anywhere in
  the docs, including `api.md`'s full function listing and `niy.md`'s
  not-yet-implemented list. There is no barrier/partial-sum "alternative pattern"
  documented either — kernels needing cross-lane/cross-block atomicity have no
  automatic translation path today.
- **Only one SIMD PE type is supported this release** — `release-notes.md`: "Ripple
  only supports a machine with one type of SIMD processing elements... the PE id
  argument is unused." There is no native SIMD multicore/multi-block construct.
  `vs-cuda.md`'s `blockIdx.x → ripple_id(multicore_block, 0)` mapping is illustrative
  for a hypothetical future machine, not a usable API today — "multicore" appears
  nowhere else in the entire docs corpus. (This is a distinct axis from the
  `ripple_thd_*` *threading* API below — SIMD multicore still doesn't exist; CPU/DSP
  multi-threading is what's newly documented.)
- Requires `clang -fenable-ripple` to activate at all — omitting it produces
  undefined-symbol errors for every `ripple_*` call despite otherwise-valid code
  (`troubleshooting/src/generic-ts.md`).
- `vtcm_malloc(size, align_as)` / `vtcm_free(ptr)` — two-argument signature. No formal
  prototype exists upstream; this is inferred from the one usage example in the HVX
  optimization guide's `SpVV` example.
- `ripple_shuffle(value, size_t(*fn)(size_t,size_t))` — function-pointer form. There is
  no `(mask, val, delta)`-style shuffle API.
- `ripple_set_block_shape`'s `pe_id` argument is currently unused/ignored by the
  compiler. Every doc example self-defines its own PE constant (`#define VECTOR_PE 0`)
  rather than relying on `<ripple.h>` to provide one.
- Math functions (`sqrtf`, `expf`, etc.) need `<math.h>` (or `<ripple_math.h>` for
  vectorized/f16 variants) — not automatic.

## `ripple_thd_*` — Hexagon multi-threading API (documented 2026-08-17, not yet released)

Source: `temp_ripple_docs/src/ripple-spec/multi-threading.md` at `ef517bf`. Distinct
prefix (`ripple_thd_`) from SIMD's `ripple_`. Key surface:
- `ripple_thd_init(pe_id, underlying, n_blocks, flags, max_dims)` /
  `ripple_thd_exit(underlying)` — bind a runtime object (QuRT's `qthd_runtime_t*` or
  QHPI's `QHPI_RuntimeHandle*`) to Ripple.
- `ripple_thd_set_block_shape(underlying, block_id, n_dims, size_t...shape)` →
  `ripple_thd_block_t` — `RIPPLE_THD_DYNAMIC` in a dimension means "use all available
  threads."
- `ripple_thd_id(block, dim)`, `ripple_thd_get_block_size(block, dim)`,
  `ripple_thd_barrier(block, dims_bitset)`, `ripple_thd_is_main(block, dims_bitset)`.
- `ripple_thd_parallel(block, chunk_size, flags, dims...)` (static) /
  `ripple_thd_parallel_dyn(...)` (dynamic, dimension 0 only) — loop-annotation sugar
  over SPMD, analogous to `ripple_parallel` for SIMD. At most one dimension may be
  dynamically scheduled, and it must be dimension 0.
- QuRT adds `ripple_thd_call(block, func, args)` (fork-join) + `qthd_runtime_init/exit`.
  QHPI has no equivalent entry-point call — the QHPI environment invokes the kernel
  directly with a runtime handle.
- Max 3 thread dimensions; max blocks is runtime-dependent (QuRT: 2, QHPI: 1).
- Thread and SIMD annotations can combine on the same loop; thread annotation must
  precede the SIMD one.
- Still true: **no atomics API anywhere** — confirmed zero "atomic" hits in the full
  corpus post-pull, including inside `multi-threading.md` itself, which explicitly
  reduces without atomics via `ripple_thd_barrier` + per-thread scratch buffers (see its
  `sum_along_dim1` example) rather than an atomic add.

Benoit's own sketch of a per-CUDA-block HTP-side loop (to avoid host↔HTP round-trip
latency) nests `ripple_thd_set_block_shape` once, then `blockIdx_z`/`blockIdx_y` loops,
then `ripple_thd_parallel(...)` immediately before the `blockIdx_x` loop that holds the
translated per-block body. Note his pasted snippet reuses `blockDim_x` as the bound for
all three loops (`blockIdx_z`, `blockIdx_y`, `blockIdx_x`) — almost certainly a
copy-paste typo for `blockDim_z`/`blockDim_y`/`blockDim_x` respectively; verify with him
before treating it as the literal shape to emit. His guidance: since the QuRT/QHPI
runtime isn't released yet, C-Ripple should skip emitting the `ripple_thd_parallel` call
itself for now (leaving that loop sequential) rather than wait entirely.

## This translator's architecture (source-level path, `frontends/source/` + `core/`)

`GlobalKernelRule` emits a per-block function taking a caller-supplied
`ripple_block_t` and a translator-owned `ripple_launch_context_t`, followed by
the original CUDA arguments. A generated `<kernel>_ripple_launch` iterates all
grid dimensions sequentially on the target. `block_shape` is optional API input:
positive static extents, or a mapping by kernel name. Explicit shapes are
constructed in the launcher; omitted shapes produce a diagnostic and require
the caller to pass a block constructed with the original CUDA launch shape.
Shapes are never inferred from element types, HVX width, or index usage.
See `docs/superpowers/specs/2026-10-08-preserve-block-shape-design.md`.

Passing SIMD block objects requires caller and callee in the same compilation
unit (upstream `ripple-spec/calling.md`). Optional `threaded=True` / `--threaded`
also emits a worker launcher accepting an already running 1D
`ripple_thd_block_t`. It queries dimension-zero worker IDs/sizes and
distributes flattened grid blocks cyclically without changing SIMD shapes.
The caller must create a 1D block; do not query higher dimensions without rank
information (the opaque API has no rank query). The runtime owns initialization
and completion. CUDA host launch syntax is
rejected rather than translated.
See `docs/superpowers/specs/2026-10-08-runtime-workers-design.md`.

## Deferred / known gaps (intentionally out of scope so far, not forgotten)

- **Vendor runtime integration.** Worker launchers now ship and are verified with
  real host pthreads, target simulator reference queries and actual SDK standalone
  hardware threads. The latter bind queries to hardware IDs and prove secondary
  thread execution using simulator statistics. Configure global HVX mode before
  fork and acquire/release contexts per worker. SDK 19.0.07 release publishes
  the context free before clearing old SSR permission; protect allocation and
  release with an outer lock and use NO_WAIT acquisition to avoid deadlock.
  Kernels execute outside that lock. Six-worker failures are retained.
  QuRT/QHPI creation,
  initialization and completion remain caller-owned and unverified because the
  available SDK lacks runtime headers/libraries. Do not call reference queries
  native concurrency proof.
- **Floating-point compile mode.** On v68 use both `-mno-hvx-qfloat` and
  `-mhvx-ieee-fp`. The installed compiler passes guarded tails and IEEE edge probes
  in this mode. Default QFloat differs from exact IEEE values; disabling QFloat
  alone fails guarded float tails even in handwritten Ripple. Preserve diagnostic
  failures. Newer QFloat strict-ieee lowering runs only on v79+, so accepting
  the flag is insufficient for v68. Do not loosen references to claim success.
- **VS Code extension** (`interfaces/vscode/`) duplicates the source-level translation
  logic independently in TypeScript, rather than calling into the Python translator —
  tracked as GitHub issue #9, deliberately deferred as "a separate, bigger decision."
  It currently hard-fails on the same fictional-API cases the Python translator does
  (atomics, multi-dim `__shared__`), but doesn't get the real flattening/atomics-idiom
  logic — duplicating that a second time in TypeScript wasn't judged worth it yet.
- **LLVM-IR translation path** (`frontends/ir/`) has 5 separate open GitHub issues and
  is largely unmigrated to the current Ripple API — the source-level path is what
  Benoit's team actually uses (via the Flutter app / `server.py`), so IR-path work has
  been consistently out of scope.

## Process notes

- Task-specific design decisions belong in `docs/superpowers/specs/<date>-<topic>-design.md`
  and their paired `docs/superpowers/plans/<date>-<topic>.md` — this file is only for
  facts that outlive a single task.
- `README.md` and `docs/README.md` are kept byte-identical (confirmed via `diff`) —
  when editing one, copy it over the other rather than editing both independently.
