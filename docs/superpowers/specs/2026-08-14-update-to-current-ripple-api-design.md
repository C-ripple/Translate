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
ripple_* symbols" section) and is currently undocumented in README.md, the CLI, the web
UI, and the generated file's header comment. Add it to all four.

### 6. Tests

- New fixtures for each of the 5 atomics hard-fail paths.
- New fixtures for the VTCM rewrite, including a multi-statement kernel body to exercise
  the brace-matching `vtcm_free()` insertion.
- Update/rename fixtures referencing `ripple_sad` → `cripple_sad`.
- Update `tests/stub_headers/ripple.h`: add `vtcm_malloc`/`vtcm_free`/`HVX_PE` declarations,
  remove nothing atomics-related (it never declared the fake atomic API, since those were
  boilerplate-local macros, not calls the stub needed to resolve).
- Existing shuffle/reduction/thread-indexing tests are untouched — already correct.

## Out of scope

- The LLVM-IR translation path and its 5 open issues.
- Building/wiring the real Hexagon toolchain Docker image.
- Open issue #11 (runtime-variable shuffle shift amounts) — already cleanly diagnosed,
  not a regression, not part of this update.
- Any HVX-specific performance optimization beyond keeping existing idioms correct.
