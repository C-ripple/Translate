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

`GlobalKernelRule` always adds `block_idx_x/y/z`, `grid_dim_x/y/z`, `block_dim_x/y/z` as
real, declared parameters to every generated function — they are not undefined. Each
generated function represents the work for a **single CUDA grid block**; multi-block
grid iteration is left to an external, hand-written C driver loop the translator does
not generate or see. This is a deliberate consequence of Ripple having no native
multi-block construct (see above), not an oversight — but it's also not documented
anywhere a user would see it today.

## Deferred / known gaps (intentionally out of scope so far, not forgotten)

- **~~Host-side~~ HTP-side grid-loop auto-generation — no longer deferred, Benoit gave
  the signal on 2026-08-17.** Originally scoped out as "would let the translator emit
  the outer per-block driver loop itself... no signal yet from Benoit's team that it's
  actually blocking them." Benoit has now explicitly requested this, and redirected it
  from a host-side loop to an HTP-side loop using the new `ripple_thd_*` API (to avoid
  host↔HTP round-trip latency) — see the `ripple_thd_*` section above. This is a bigger
  architectural change than the original host-loop idea would have been: it changes the
  output contract (translator emits the block-iteration loop + thread-block setup
  itself, using an API not yet in the release the translator otherwise targets) and its
  correct shape depends on a currently-unreleased runtime. Needs a design spec
  (`docs/superpowers/specs/`) before implementation, not a direct patch — open questions
  include the loop-bound typo noted above, which of `ripple_thd_parallel`'s deferred
  parts to stub vs. skip until QuRT/QHPI ship, and how `underlying`/`rt` (the runtime
  object) gets threaded into a translator that currently emits self-contained kernels.
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
