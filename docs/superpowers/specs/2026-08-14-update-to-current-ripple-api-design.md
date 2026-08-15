# Update C-Ripple's source translator to the current Ripple API (21.0-alpha3)

## Context

Benoit (Qualcomm) emailed asking us to update C-Ripple to the current Ripple API — his
team couldn't fully evaluate the tool because the API had moved since it was built. He
pointed at `qualcomm/learn-ripple` (User Manual, HVX Optimization Guide, Troubleshooting
Guide) as the reference.

Benoit's team evaluates through the Flutter app / `server.py`, which only exercises the
**source-level translation path** (`frontends/source/cuda_frontend.py` +
`core/translation_rules.py` + `core/semantic_model.py`) — not the LLVM-IR path
(`frontends/ir/`).

Investigation before this spec was written turned up two things that changed the shape
of the work:

1. A real, recent migration effort already closed GitHub issues #7–10 (2026-08-12): thread/
   block indexing, warp reductions, and constant-shift shuffle translation are already
   correct against the real API. Shuffle translation already cleanly hard-fails (with a
   documented workaround) for the one case it can't handle — runtime-variable shift amounts,
   tracked as open issue #11.
2. What's still broken is narrower and more specific than "everything is stale": five
   atomic-translation rules, shared-memory/VTCM translation, and one invented math macro
   emit calls to Ripple symbols that were never real, plus the required `-fenable-ripple`
   compile flag is undocumented everywhere.

The primary source for every API fact below is the vendored docs checkout at
`temp_ripple_docs/` in this repo (a local copy of `qualcomm/learn-ripple`), read directly
— not summarized secondhand.

## Scope

Source-level path only. The LLVM-IR path (`frontends/ir/`) has 5 separately-tracked open
issues (#2–6: no atomics/shuffle/VTCM translation, missing block-index intrinsics, no
tests, two divergent web servers) and is explicitly out of scope for this update.

Verification stays at the level already established elsewhere in this codebase: doc-verified
rewrites checked via `clang -fsyntax-only` against an updated stub `ripple.h`
(`tests/stub_headers/ripple.h`), not a real Hexagon toolchain compile. The vendored
`docker/hexagon-toolchain.Dockerfile` build (30–90+ min) stays deferred, per the existing
documented decision in `docker/README.md` and confirmed for this round.

## Changes

### 1. Atomics — hard-fail instead of fake translation

Real Ripple has no atomics API. `temp_ripple_docs/src/ripple-spec/multi-threading.md`
shows a barrier + per-lane partial-sum pattern as the documented alternative
("we don't want to implement the summation using atomics").

`AtomicMaxRule`, `AtomicMinRule`, `AtomicCASRule`, `AtomicExchRule` currently emit
`ripple_atomic_max/min/cas/exch(...)` unconditionally — none of these exist in the real
API. Change all four to call `ctx.add_error()` instead, in the same style
`ShuffleDownRule` already uses for its unsupported case (clear message, points at the
documented workaround).

`AtomicAddRule` gets the same hard-fail treatment. This is safe: `AtomicAddRule` only
ever fires on `atomicAdd(...)` text that's still present after the rule engine's
higher-priority pass — meaning `WarpReductionRule`/`ButterflyAllReduceRule` did *not*
recognize it as a full-block reduction idiom and rewrite it to real `ripple_reduceadd`
first. Any `atomicAdd` that reaches `AtomicAddRule` is therefore genuinely untranslatable
today, not a regression.

Also remove the `ripple_atomic_add/max/min/cas/exch` macro `#define`s from
`_add_ripple_boilerplate()`'s generated header — they reference nonexistent Ripple
concepts and should never appear in output now that nothing calls them.

### 2. VTCM / shared memory — real `vtcm_malloc`/`vtcm_free`

`SharedMemoryRule` currently emits `__attribute__((section(".vtcm")))`, which does not
exist anywhere in the Ripple API. Confirmed directly from the `SpVV` example in
`temp_ripple_docs/opt/hexagon/src/hvx-opt.md`:

```c
float * gathered_V = vtcm_malloc(sizeof(float) * nS, /*align_as=*/128);
...
vtcm_free(gathered_V);
```

Rewrite `SharedMemoryRule` to:
- Replace `__shared__ TYPE name[SIZE];` with `TYPE *name = vtcm_malloc(sizeof(TYPE) * (SIZE), /*align_as=*/128);`
- Insert a matching `vtcm_free(name);` immediately before the enclosing kernel function's
  closing brace, found via a raw-text brace counter. The codebase already has this exact
  brace-counting pattern in two places (`frontends/source/cuda_frontend.py`'s AST block
  parser, `frontends/ir/ir_frontend.py`'s instruction-block reader) — follow the same
  approach, adapted for regex-stage text rather than tokens.

`DynamicSharedMemoryRule` (`extern __shared__`, size supplied at kernel launch) is left
as-is: it already warns and passes through without claiming a fake translation, and
Ripple has no launch-time-size equivalent to translate it to.

**Multi-dimensional arrays are a hard-fail, not silently broken.** The old
attribute-based rule tolerated `__shared__ float tile[18][18]` by accident — it only
ever rewrote the prefix up to the first `[dim]`, leaving the second `[18]` as trailing
array-declarator syntax, which is valid C for a real array. A `vtcm_malloc()`-returned
pointer can't be redeclared with a trailing `[dim]` the same way, and every
`tile[ty][tx]`-style indexing expression elsewhere in the kernel would need rewriting to
flat pointer arithmetic — an AST-level transformation, not something safe to do with
regex substitution. `SharedMemoryRule` must detect a second `[...]` immediately after
the matched declaration and `ctx.add_error()` instead of translating it, rather than
emit array-declarator syntax after an initializer (a compile error). This is a real,
if narrow, capability loss versus today for kernels using 2D+ shared tiles — flagging
it here rather than discovering it silently in the implementation. Found while writing
the implementation plan: `tests/test_complex_kernels.py::test_2d_convolution` exercises
exactly this case (`__shared__ float tile[18][18]`) but only asserts on `ripple_id`
output, not shared-memory output, so it currently passes without really exercising 2D
correctness — that test's kernel needs simplifying to drop the shared-memory tile (its
actual point is 2D thread-index translation), and a new dedicated test should assert the
multi-dimensional case hard-fails with a clear diagnostic.

### 3. `__sad` — rename, don't remove

No real Ripple equivalent exists (confirmed: zero hits for `sad` anywhere in
`temp_ripple_docs/`). The current `ripple_sad` macro (`(__builtin_abs(x-y)+z)`) is valid,
portable C that doesn't depend on Ripple at all — the only problem is the misleading
`ripple_`-prefixed name, which implies a real API symbol. Rename to `cripple_sad` and note in the boilerplate comment that it's a translator-provided
helper, not Ripple API.

### 4. `HVX_PE` — self-define instead of assuming `<ripple.h>` provides it

`RIPPLE_SETUP_BLOCK()` uses `HVX_PE` as the `pe_id` argument to
`ripple_set_block_shape()`, but nothing in the generated output ever defines it — it's
assumed to come from `<ripple.h>`. Checked the entire vendored docs corpus: every single
example self-defines its own PE constant (`#define VECTOR_PE 0`) rather than relying on
the header, and the release notes confirm the `pe_id` argument is unused/ignored (Ripple
currently supports only one SIMD PE type). Change `_add_ripple_boilerplate()` to emit
`#define HVX_PE 0` itself, matching documented convention.

### 5. Document the required compile flag

`-fenable-ripple` is required (`temp_ripple_docs`'s troubleshooting guide, "Missing
ripple_* symbols" section) and is currently undocumented everywhere. Add it in exactly
four places: `README.md`, the CLI's top-level `--help` epilog
(`interfaces/cli/cuda2ripple.py` — shown both on explicit `--help` and on bare
`cuda2ripple` with no subcommand; per-subparser help is not touched, since the top-level
epilog is already the discoverable, low-duplication spot), the generated output file's header comment (natural
to add alongside the item-4 boilerplate change, since it's already being touched), and
the Flask web UI's `HTML_TEMPLATE` status bar (`interfaces/web/server.py`) — not the
Flutter app, which has no help/instructions surface today and is out of scope to add
one to. Note there are two Flask servers in this repo (root `server.py`, minimal API
only; `interfaces/web/server.py`, has the HTML UI) — this item touches only the one with
user-facing copy.

### 6. Tests

Existing tests that currently assert *successful* translation of what's becoming a
hard-fail need to be rewritten to assert `TranslationError`/`ctx.add_error`, not just
supplemented with new ones:
- `tests/test_complex_kernels.py`: `test_atomic_cas`, `test_atomic_exch`, and
  `test_atomic_min_max` (this one already exercises both `atomicMin` and `atomicMax` in
  one kernel — kept combined, just converted from a success assertion to
  `pytest.raises(TranslationError)`; between the three rewritten tests, all of
  Add/Max/Min/CAS/Exch get direct hard-fail coverage, so no further new atomics fixtures
  are needed). Also `test_sad_computation` (mixed — asserts both `ripple_sad` and
  `ripple_atomic_add` in one kernel; the `atomicAdd` there is a bare per-thread call, not
  the recognized block-reduction idiom, so it becomes a hard-fail case while the `__sad`
  part becomes a `cripple_sad` rename — split into two tests).
- `tests/test_translation.py`: `test_atomic_add_rule` (asserts `ripple_atomic_add` from a
  direct `AtomicAddRule().apply()` call) and `test_shared_memory_rule` (asserts the old
  `__attribute__((aligned(128)))` + array-declaration output from a direct
  `SharedMemoryRule().apply()` call — needs rewriting to assert the `vtcm_malloc` form).
- VTCM rewrite fixtures, including a multi-statement kernel body to exercise the
  brace-matching `vtcm_free()` insertion.
- Update `tests/stub_headers/ripple.h`: add `vtcm_malloc`/`vtcm_free`/`HVX_PE`
  declarations. Nothing to remove there for atomics — it never declared the fake atomic
  API, since those were boilerplate-local macros, not calls the stub needed to resolve.
- Existing shuffle/reduction/thread-indexing tests are untouched — already correct.

## Out of scope

- The LLVM-IR translation path and its 5 open issues.
- Building/wiring the real Hexagon toolchain Docker image.
- Open issue #11 (runtime-variable shuffle shift amounts) — already cleanly diagnosed,
  not a regression, not part of this update.
- Any HVX-specific performance optimization beyond keeping existing idioms correct.
