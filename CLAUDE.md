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
multicore/thread API family), and it was fabricated — neither the file nor the API
family exists anywhere in the actual docs (confirmed: zero hits for "atomic" or
"thd"/"thread" across the whole corpus). It shipped in merged, pushed error messages
before being caught and corrected. The lesson: a well-formatted, heavily-cited report is
not evidence of accuracy — grep the actual files.

## Ripple 21.0-alpha3 — real capabilities and limits (independently verified)

- **No atomics API at all.** Confirmed directly: `atomic` appears zero times anywhere in
  the docs, including `api.md`'s full function listing and `niy.md`'s
  not-yet-implemented list. There is no barrier/partial-sum "alternative pattern"
  documented either — kernels needing cross-lane/cross-block atomicity have no
  automatic translation path today.
- **Only one SIMD PE type is supported this release** — `release-notes.md`: "Ripple
  only supports a machine with one type of SIMD processing elements... the PE id
  argument is unused." There is no native multicore/multi-block construct.
  `vs-cuda.md`'s `blockIdx.x → ripple_id(multicore_block, 0)` mapping is illustrative
  for a hypothetical future machine, not a usable API today — "multicore" appears
  nowhere else in the entire docs corpus.
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

## This translator's architecture (source-level path, `frontends/source/` + `core/`)

`GlobalKernelRule` always adds `block_idx_x/y/z`, `grid_dim_x/y/z`, `block_dim_x/y/z` as
real, declared parameters to every generated function — they are not undefined. Each
generated function represents the work for a **single CUDA grid block**; multi-block
grid iteration is left to an external, hand-written C driver loop the translator does
not generate or see. This is a deliberate consequence of Ripple having no native
multi-block construct (see above), not an oversight — but it's also not documented
anywhere a user would see it today.

## Deferred / known gaps (intentionally out of scope so far, not forgotten)

- **Host-side grid-loop auto-generation** — would let the translator emit the outer
  per-block driver loop itself. A real architectural commitment (changes the output
  contract from "self-contained program" to "expects a generated host wrapper"), scoped
  out because there's no signal yet from Benoit's team that it's actually blocking them.
  Revisit if that changes.
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
