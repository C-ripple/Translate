# Update C-Ripple's Source Translator to Current Ripple API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Actual Outcome addendum (post-implementation correction):** every reference in this
> plan to `temp_ripple_docs/src/ripple-spec/multi-threading.md` and the "barrier +
> per-lane partial-sum pattern in the Ripple multi-threading guide" is fabricated — no
> such file or API family exists anywhere in the vendored docs (verified: zero hits for
> "atomic" or "thd"/"thread" in the entire `temp_ripple_docs/` corpus). See the design
> spec's own addendum for the full correction. The code blocks below still show the
> fabricated phrasing as originally written and executed; the actual shipped error
> messages were corrected in a follow-up commit to drop the false "see this guide"
> pointer and instead say plainly that no documented Ripple alternative exists. Treat
> every `f"... multi-threading guide."` string in this document's code blocks as
> superseded by that correction, not as current shipped text.
>
> **Second correction (same audit pass):** Task 2's "trailing atomicAdd" reasoning
> (around line 423 and its Step 6a rationale) wrongly claims `block_idx_x` "is never
> declared or assigned anywhere in the generated output." It is: `GlobalKernelRule`
> always adds it as a real `int` function parameter. The hard-fail decision is still
> correct — Ripple has no native multi-block construct in this release
> (`release-notes.md`: only one SIMD PE type is supported), so the generated per-block
> function is meant to be invoked once per grid block by an external, hand-written C
> driver loop this translator never sees or generates, and there's no way to know from
> inside the translator whether that external loop makes a trailing write race-free —
> but the "undefined identifier" framing was factually wrong. See the design spec's
> addendum for the corrected reasoning.

**Goal:** Make C-Ripple's source-level translator (`frontends/source/cuda_frontend.py` +
`core/translation_rules.py`) emit only real Ripple 21.0-alpha3 API calls, so Benoit's
team can compile and evaluate translated output instead of hitting undefined-symbol
errors on invented functions.

**Architecture:** No new files, no new abstractions. Every change is either (a) a rule's
`apply()` method switching from emitting a fictional call to calling `ctx.add_error()`
with a diagnostic, matching the style `ShuffleDownRule` already established, or (b) a
rule's `apply()` rewritten to emit a real, doc-verified API call. One rule
(`SharedMemoryRule`) gains a small brace-counting helper to place a `vtcm_free()` call
correctly — modeled on the identical brace-counting pattern already used in
`frontends/source/cuda_frontend.py`'s AST block parser and `frontends/ir/ir_frontend.py`.

**Tech Stack:** Python 3.14, pytest, regex-based source-to-source translation. No new
dependencies.

**Spec:** `docs/superpowers/specs/2026-08-14-update-to-current-ripple-api-design.md`

---

## Task 1: Self-define `HVX_PE` in generated output

**Files:**
- Modify: `frontends/source/cuda_frontend.py:587-593`
- Test: `tests/test_translation.py` (new `TestBoilerplateGeneration` class)

`RIPPLE_SETUP_BLOCK()` uses `HVX_PE` as the `pe_id` argument to
`ripple_set_block_shape()`, but nothing in the generated output defines it — it silently
assumes `<ripple.h>` provides it. It doesn't: every example in the vendored Ripple docs
self-defines its own PE constant (`#define VECTOR_PE 0`), and the release notes confirm
the value is unused (Ripple currently supports only one SIMD PE type).

- [x] **Step 1: Write the failing test**

Add to `tests/test_translation.py`, after the `TestTranslationRuleEngine` class (end of
file is fine — check the current end of file and append):

```python
class TestBoilerplateGeneration:
    """Tests for the RIPPLE boilerplate CUDAToRIPPLETransformer emits around
    translated kernels."""

    def test_hvx_pe_self_defined(self):
        """HVX_PE must be defined by the generated output itself, not assumed
        to come from <ripple.h> — no example in the Ripple docs relies on the
        header for its PE constant."""
        cuda_code = "__global__ void k(float *a) { a[0] = 1.0f; }"
        result = translate_cuda_source(cuda_code)
        assert "#define HVX_PE 0" in result
```

- [x] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_translation.py::TestBoilerplateGeneration::test_hvx_pe_self_defined -v`
Expected: FAIL — `assert "#define HVX_PE 0" in result` is False (nothing defines `HVX_PE` today).

- [x] **Step 3: Add the `#define`**

In `frontends/source/cuda_frontend.py`, replace lines 587-593:

```python
        header += f"""
/* Block shape configuration */
#define HVX_VECTOR_SIZE {self.hexagon_config.hvx_width}
#define RIPPLE_BLOCK_DIM_X {block_shape.dimensions[0]}
#define RIPPLE_BLOCK_DIM_Y {block_shape.dimensions[1] if len(block_shape.dimensions) > 1 else 1}
#define RIPPLE_BLOCK_DIM_Z {block_shape.dimensions[2] if len(block_shape.dimensions) > 2 else 1}

"""
```

with:

```python
        header += f"""
/* Block shape configuration */
#define HVX_VECTOR_SIZE {self.hexagon_config.hvx_width}

/* Ripple's pe_id argument is unused on Hexagon today (release notes: only
 * one SIMD PE type is supported), but <ripple.h> does not define a
 * constant for it — every example in the Ripple docs self-defines its own
 * PE constant instead of relying on the header. Do the same. */
#define HVX_PE 0
#define RIPPLE_BLOCK_DIM_X {block_shape.dimensions[0]}
#define RIPPLE_BLOCK_DIM_Y {block_shape.dimensions[1] if len(block_shape.dimensions) > 1 else 1}
#define RIPPLE_BLOCK_DIM_Z {block_shape.dimensions[2] if len(block_shape.dimensions) > 2 else 1}

"""
```

- [x] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_translation.py::TestBoilerplateGeneration::test_hvx_pe_self_defined -v`
Expected: PASS

- [x] **Step 5: Run the full test suite to check for regressions**

Run: `venv/bin/python -m pytest -v`
Expected: All previously-passing tests still pass (this change only adds a line; nothing
depended on its absence).

- [x] **Step 6: Commit**

```bash
git add frontends/source/cuda_frontend.py tests/test_translation.py
git commit -m "Self-define HVX_PE in generated output instead of assuming <ripple.h> provides it

Every example in the Ripple docs self-defines its own PE constant
(#define VECTOR_PE 0); nothing in ripple.h supplies HVX_PE, and the
release notes confirm the value is unused on Hexagon's single-PE-type
target. Generated output was silently relying on an undefined symbol."
```

---

## Task 2: Hard-fail atomics instead of emitting fictional calls

**Files:**
- Modify: `core/translation_rules.py:1428-1544` (all 5 atomic rule classes)
- Modify: `frontends/source/cuda_frontend.py:601-614` (remove fake atomic macros)
- Modify: `tests/test_translation.py:130-137` (`test_atomic_add_rule`)
- Modify: `tests/test_complex_kernels.py:1-14` (imports), `:204-237` (3 tests)

Real Ripple has no atomics API at all — `temp_ripple_docs/src/ripple-spec/multi-threading.md`
shows a barrier + per-lane partial-sum pattern as the documented alternative. All five
`Atomic*Rule` classes currently emit `ripple_atomic_*(...)` calls unconditionally; none
of these exist in the real API.

`AtomicAddRule` is included even though the *reduction-idiom* case (shared-memory
reduce-then-`atomicAdd`-once) already translates correctly via `WarpReductionRule`
(priority 85) / `WarpMinMaxReductionRule` (84) / `ButterflyAllReduceRule` (86) —
`AtomicAddRule` (priority 60) only ever fires on `atomicAdd(...)` text still present
after those higher-priority rules ran, meaning it's text those rules didn't recognize as
a reduction idiom. Hard-failing it is safe.

- [x] **Step 1: Write the failing tests**

Replace `tests/test_translation.py:130-137` (`test_atomic_add_rule`):

```python
    def test_atomic_add_rule(self):
        """atomicAdd has no Ripple equivalent outside a recognized
        reduction idiom (handled separately by WarpReductionRule etc.) —
        AtomicAddRule itself must hard-fail, not emit a fictional call."""
        rule = AtomicAddRule()
        ctx = TranslationContext()

        source = "atomicAdd(&sum, val)"
        result = rule.apply(source, ctx)

        assert ctx.has_errors()
        assert "atomicAdd" in ctx.errors[0]
        assert result == source  # left untranslated, not rewritten
```

In `tests/test_complex_kernels.py`, add `TranslationError` to the imports (line 14):

```python
from core.semantic_model import TranslationContext, TranslationError
```

Replace `tests/test_complex_kernels.py:204-237` (`test_atomic_cas`, `test_atomic_exch`,
`test_atomic_min_max`):

```python
    def test_atomic_cas(self):
        """Ripple has no atomics API — atomicCAS must hard-fail, not
        translate to a fictional ripple_atomic_cas() call."""
        cuda_code = """
        __global__ void cas_kernel(int *lock) {
            int old = atomicCAS(lock, 0, 1);
        }
        """

        with pytest.raises(TranslationError):
            translate_cuda_source(cuda_code)

    def test_atomic_exch(self):
        """Ripple has no atomics API — atomicExch must hard-fail."""
        cuda_code = """
        __global__ void exch_kernel(int *data, int new_val) {
            int old = atomicExch(data, new_val);
        }
        """

        with pytest.raises(TranslationError):
            translate_cuda_source(cuda_code)

    def test_atomic_min_max(self):
        """Ripple has no atomics API — atomicMin/atomicMax must hard-fail."""
        cuda_code = """
        __global__ void minmax_kernel(int *data, int val) {
            atomicMin(data, val);
            atomicMax(data + 1, val);
        }
        """

        with pytest.raises(TranslationError):
            translate_cuda_source(cuda_code)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_translation.py::TestTranslationRules::test_atomic_add_rule tests/test_complex_kernels.py::TestAtomicOperations -v`
Expected: FAIL — `test_atomic_add_rule` fails on `assert ctx.has_errors()` (no error
recorded today); the other three fail because `translate_cuda_source` returns normally
instead of raising.

- [x] **Step 3: Rewrite the five atomic rules**

In `core/translation_rules.py`, replace lines 1428-1544 (all five `Atomic*Rule` classes,
from `class AtomicAddRule(TranslationRule):` through the end of `AtomicExchRule`) with:

```python
class AtomicAddRule(TranslationRule):
    """Diagnoses atomicAdd calls with no Ripple equivalent.

    This only ever fires on atomicAdd(...) text still present after
    WarpReductionRule/WarpMinMaxReductionRule/ButterflyAllReduceRule have
    already run (priority 84-86, all higher than this rule's 60) — those
    rules already translate the recognized single-output block-reduction
    idiom to real ripple_reduceadd(). Anything reaching this rule is a
    genuinely untranslatable atomicAdd, not a regression.
    """

    PATTERN = r'atomicAdd\s*\(\s*([^,]+),\s*([^)]+)\)'

    def __init__(self):
        super().__init__(
            name="atomic_add",
            description="Diagnose unsupported atomicAdd (no Ripple atomics API)",
            cuda_pattern=self.PATTERN,
            priority=60
        )

    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            target = match.group(1).strip()
            value = match.group(2).strip()
            ctx.add_error(
                f"atomicAdd({target}, {value}) cannot be translated — Ripple "
                f"has no atomics API. If this is a single-output block "
                f"reduction (every lane computes a value, then they're "
                f"combined into one result), restructure it as a "
                f"shared-memory or full-warp reduction and it will be "
                f"recognized automatically. Otherwise, see the barrier + "
                f"per-lane partial-sum pattern in the Ripple "
                f"multi-threading guide."
            )
            return match.group(0)

        return re.sub(self.PATTERN, replace, cuda_code)


class AtomicMaxRule(TranslationRule):
    """Diagnoses atomicMax calls with no Ripple equivalent."""

    PATTERN = r'atomicMax\s*\(\s*([^,]+),\s*([^)]+)\)'

    def __init__(self):
        super().__init__(
            name="atomic_max",
            description="Diagnose unsupported atomicMax (no Ripple atomics API)",
            cuda_pattern=self.PATTERN,
            priority=60
        )

    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            target = match.group(1).strip()
            value = match.group(2).strip()
            ctx.add_error(
                f"atomicMax({target}, {value}) cannot be translated — Ripple "
                f"has no atomics API. If this is a full-block max reduction, "
                f"restructure it to use ripple_reducemax directly. "
                f"Otherwise, see the barrier + per-lane partial-result "
                f"pattern in the Ripple multi-threading guide."
            )
            return match.group(0)

        return re.sub(self.PATTERN, replace, cuda_code)


class AtomicMinRule(TranslationRule):
    """Diagnoses atomicMin calls with no Ripple equivalent."""

    PATTERN = r'atomicMin\s*\(\s*([^,]+),\s*([^)]+)\)'

    def __init__(self):
        super().__init__(
            name="atomic_min",
            description="Diagnose unsupported atomicMin (no Ripple atomics API)",
            cuda_pattern=self.PATTERN,
            priority=60
        )

    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            target = match.group(1).strip()
            value = match.group(2).strip()
            ctx.add_error(
                f"atomicMin({target}, {value}) cannot be translated — Ripple "
                f"has no atomics API. If this is a full-block min reduction, "
                f"restructure it to use ripple_reducemin directly. "
                f"Otherwise, see the barrier + per-lane partial-result "
                f"pattern in the Ripple multi-threading guide."
            )
            return match.group(0)

        return re.sub(self.PATTERN, replace, cuda_code)


class AtomicCASRule(TranslationRule):
    """Diagnoses atomicCAS calls with no Ripple equivalent."""

    PATTERN = r'atomicCAS\s*\(\s*([^,]+),\s*([^,]+),\s*([^)]+)\)'

    def __init__(self):
        super().__init__(
            name="atomic_cas",
            description="Diagnose unsupported atomicCAS (no Ripple atomics API)",
            cuda_pattern=self.PATTERN,
            priority=60
        )

    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            target = match.group(1).strip()
            compare = match.group(2).strip()
            value = match.group(3).strip()
            ctx.add_error(
                f"atomicCAS({target}, {compare}, {value}) cannot be "
                f"translated — Ripple has no atomics API, and "
                f"compare-and-swap has no reduction-idiom equivalent. See "
                f"the barrier + per-lane partial-result pattern in the "
                f"Ripple multi-threading guide."
            )
            return match.group(0)

        return re.sub(self.PATTERN, replace, cuda_code)


class AtomicExchRule(TranslationRule):
    """Diagnoses atomicExch calls with no Ripple equivalent."""

    PATTERN = r'atomicExch\s*\(\s*([^,]+),\s*([^)]+)\)'

    def __init__(self):
        super().__init__(
            name="atomic_exch",
            description="Diagnose unsupported atomicExch (no Ripple atomics API)",
            cuda_pattern=self.PATTERN,
            priority=60
        )

    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            target = match.group(1).strip()
            value = match.group(2).strip()
            ctx.add_error(
                f"atomicExch({target}, {value}) cannot be translated — "
                f"Ripple has no atomics API, and exchange has no "
                f"reduction-idiom equivalent. See the barrier + per-lane "
                f"partial-result pattern in the Ripple multi-threading "
                f"guide."
            )
            return match.group(0)

        return re.sub(self.PATTERN, replace, cuda_code)
```

- [x] **Step 4: Remove the fake atomic macros from generated output**

In `frontends/source/cuda_frontend.py`, delete lines 601-614 (the entire
`/* Atomic operation wrappers for Hexagon */` block, from `#ifdef __HEXAGON__` through
the matching `#endif`) — nothing calls these macros anymore, and they reference
functions (`ripple_atomic_add` etc.) that were never real Ripple API.

- [x] **Step 5: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_translation.py::TestTranslationRules::test_atomic_add_rule tests/test_complex_kernels.py::TestAtomicOperations -v`
Expected: PASS

- [x] **Step 6: Run the full test suite — expect 5 failures, diagnose each**

Run: `venv/bin/python -m pytest -v`

Expected failures (found during implementation — the "reduction idiom already handles
the common case" reasoning above is narrower than it sounds):
1. `tests/test_complex_kernels.py::TestReductionKernels::test_warp_reduction_optimization`
2. `tests/test_complex_kernels.py::TestImageProcessing::test_sad_computation` (unrelated —
   fixed in Task 3, ignore for now)
3. `tests/test_real_kernels.py::test_translates_without_error[atomics_cas_exch.cu]`
4. `tests/test_real_kernels.py::test_translates_without_error[cuda_kernels.cu]`
5. `tests/test_real_kernels.py::test_translated_output_is_valid_syntax[atomics_cas_exch.cu]`

`WarpReductionRule` only rewrites the shuffle loop itself
(`val = ripple_reduceadd(0b1, val);`) — it does not consume a trailing
`if (tid == 0) { atomicAdd(output, val); }` statement, which is the standard way a
warp's fully-reduced value gets written into a shared/global accumulator. That call now
correctly hard-fails: real Ripple has no atomics, and there's no way to know whether a
plain (non-atomic) write would be safe instead, because this translator doesn't actually
model CUDA's multi-block grid execution yet — `restructure_kernel()` (the function that
would turn `blockIdx`-driven grids into an explicit loop) is dead code, never called
anywhere (verify: `grep -rn "restructure_kernel" --include="*.py" .` finds only its own
`def`), and `BlockIdxRule` today just substitutes `blockIdx.x` for the bare identifier
`block_idx_x`, which is never declared or assigned anywhere in the generated output — a
pre-existing, unrelated bug confirming multi-block execution isn't modeled. A kernel that
really does launch many blocks needs a genuine cross-block accumulation primitive; one
that launches a single block doesn't need atomicity at all — the translator has no way
to tell which case it's looking at, so hard-failing is the only honest option.

- [x] **Step 6a: Fix `test_warp_reduction_optimization`**

In `tests/test_complex_kernels.py`, add `CUDAToRIPPLETransformer` to the existing import
(alongside `translate_cuda_source`):

```python
from frontends.source.cuda_frontend import translate_cuda_source, CUDAToRIPPLETransformer
```

Replace the `test_warp_reduction_optimization` method (in `TestReductionKernels`):

```python
    def test_warp_reduction_optimization(self):
        """Warp reduction loops are optimized to ripple_reduceadd, but a
        subsequent single-thread atomicAdd (writing the already-reduced
        value into global memory) correctly hard-fails — Ripple has no
        atomics API, and this translator doesn't yet model CUDA's
        multi-block grid execution, so there's no safe way to know whether
        a plain (non-atomic) write would be correct here."""
        cuda_code = """
        __global__ void reduce_sum(float *input, float *output, int N) {
            int tid = threadIdx.x;
            int idx = blockIdx.x * blockDim.x + tid;

            float val = (idx < N) ? input[idx] : 0.0f;

            // Warp reduction
            for (int offset = warpSize/2; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }

            if (tid == 0) {
                atomicAdd(output, val);
            }
        }
        """

        ctx = TranslationContext()
        transformer = CUDAToRIPPLETransformer(ctx)
        with pytest.raises(TranslationError):
            transformer.transform(cuda_code)

        # The shuffle loop itself is still correctly recognized — only the
        # trailing atomicAdd (no Ripple equivalent) is the hard-fail.
        assert any("Warp Reduction" in w for w in ctx.warnings)
        assert any("atomicAdd" in e for e in ctx.errors)
```

Run: `venv/bin/python -m pytest tests/test_complex_kernels.py::TestReductionKernels::test_warp_reduction_optimization -v`
Expected: PASS

- [x] **Step 6b: Fix the `atomics_cas_exch.cu` fixture tests**

In `tests/test_real_kernels.py`, add `TranslationError` to the imports:

```python
from core.semantic_model import TranslationError
```

Remove `"atomics_cas_exch.cu"` from both the `KERNEL_FILES` and `SYNTAX_CHECK_PARAMS`
lists (it stays as a file in `tests/examples/`, just no longer run through the two
generic "translates successfully" parametrized tests). Then add a new dedicated test
after `test_translated_output_is_valid_syntax`:

```python
def test_atomics_cas_exch_hard_fails():
    """atomicCAS/atomicExch have no Ripple equivalent — this fixture's sole
    purpose (both calls, no other content) now exercises the hard-fail
    path rather than translation."""
    source = (EXAMPLES_DIR / "atomics_cas_exch.cu").read_text()
    with pytest.raises(TranslationError):
        translate_cuda_source(source)
```

Run: `venv/bin/python -m pytest tests/test_real_kernels.py -v -k "atomics_cas_exch or not cu"`
Expected: no test collected for `atomics_cas_exch.cu` under the two generic tests; the
new `test_atomics_cas_exch_hard_fails` passes.

- [x] **Step 6c: Fix the `cuda_kernels.cu` fixture test and its stale docstring**

`cuda_kernels.cu` is a 10-kernel fixture with 5 unrelated atomic call sites (lines 116,
129, 199, 254, 268 — none are the recognized single-block reduction idiom). It's already
excluded from `SYNTAX_CHECK_PARAMS` (a large module docstring at the top of
`tests/test_real_kernels.py` explains several pre-existing, unrelated reasons why — do
not touch that exclusion or that docstring's other content). It IS in `KERNEL_FILES`,
whose `test_translates_without_error` asserted successful translation — that assertion
is now wrong for this file specifically.

Remove `"cuda_kernels.cu"` from `KERNEL_FILES`. Add a dedicated replacement test after
`test_translates_without_error`:

```python
def test_cuda_kernels_file_hard_fails_on_atomics():
    """cuda_kernels.cu (see module docstring above) mixes many patterns in
    one file, including 5 atomicAdd/atomicMax call sites that aren't the
    recognized single-block reduction idiom. The whole file now correctly
    hard-fails rather than silently mistranslating them.
    CUDAToRIPPLETransformer.transform() accumulates every rule's errors
    before raising (it doesn't stop at the first), so a real translation
    bug in one of this file's other, unrelated kernels — the shuffle loops
    and math intrinsics this file also exercises — would still show up as
    an additional, distinguishable error here rather than being masked."""
    source = (EXAMPLES_DIR / "cuda_kernels.cu").read_text()
    with pytest.raises(TranslationError) as exc_info:
        translate_cuda_source(source)

    assert len(exc_info.value.errors) >= 5
```

Then update the module docstring's claim about this file (currently says it "now
translates successfully and is covered here structurally" — find that exact sentence
near the top of the file and correct it to reflect the new hard-fail behavior, keeping
the rest of the surrounding paragraph about `WarpMinMaxReductionRule` intact, since that
part is still true and unrelated to this change).

Run: `venv/bin/python -m pytest tests/test_real_kernels.py::test_cuda_kernels_file_hard_fails_on_atomics -v`
Expected: PASS

- [x] **Step 6d: Run the full test suite again to confirm only the expected failure remains**

Run: `venv/bin/python -m pytest -v`
Expected: All pass except `test_sad_computation` in `tests/test_complex_kernels.py` (it
combines `__sad` with a bare `atomicAdd` in one kernel; fixed in Task 3, not this task).
If any other test fails, that's a real regression — investigate before continuing.

- [x] **Step 7: Commit**

```bash
git add core/translation_rules.py frontends/source/cuda_frontend.py \
        tests/test_translation.py tests/test_complex_kernels.py tests/test_real_kernels.py
git commit -m "Hard-fail atomics instead of translating to fictional ripple_atomic_* calls

Ripple has no atomics API — confirmed via direct read of
temp_ripple_docs/src/ripple-spec/multi-threading.md, which shows a
barrier + per-lane partial-sum pattern as the documented alternative.
All five Atomic*Rule classes now call ctx.add_error() with guidance,
matching the style ShuffleDownRule already uses for its unsupported
case. AtomicAddRule stays safe to hard-fail: it only ever fires after
the higher-priority reduction-idiom rules (WarpReductionRule etc.,
priority 84-86 vs this rule's 60) didn't already translate the call.

Also removes the fake ripple_atomic_* macro #defines from generated
output's boilerplate header, since nothing calls them anymore.

This turned out broader than originally scoped: WarpReductionRule only
rewrites the shuffle loop, not a trailing single-thread atomicAdd that
writes the reduced value out — that call now hard-fails too, correctly,
since this translator doesn't model CUDA's multi-block grid execution
(restructure_kernel() is dead code; blockIdx.x becomes an undefined
dangling identifier today). Updated test_warp_reduction_optimization
and two tests/test_real_kernels.py fixtures (atomics_cas_exch.cu,
cuda_kernels.cu) that encoded the old fictional-atomics behavior."
```

*(Expected test failure after this commit, fixed in the next task: `test_sad_computation`.)*

---

## Task 3: Rename `ripple_sad` → `cripple_sad`

**Files:**
- Modify: `core/translation_rules.py:1667` (`MathFunctionRule.MATH_MAP`)
- Modify: `frontends/source/cuda_frontend.py:617` (boilerplate macro)
- Modify: `tests/test_complex_kernels.py:333-349` (`test_sad_computation` → split in two)

No real Ripple equivalent for CUDA's `__sad` exists anywhere in the docs (confirmed: zero
hits repo-wide). The current `ripple_sad` macro (`__builtin_abs(x-y)+z`) is valid,
portable C that doesn't depend on Ripple at all — the problem is only the misleading
`ripple_`-prefixed name, which implies a real API symbol. Renaming, not removing.

- [x] **Step 1: Write the failing tests**

Replace `tests/test_complex_kernels.py:333-349` (`test_sad_computation`) with two tests:

```python
    def test_sad_rename(self):
        """__sad has no real Ripple equivalent — it translates to the
        translator's own cripple_sad helper, not a ripple_-prefixed name
        that would imply a real API symbol."""
        cuda_code = """
        __global__ void compute_sad_only(unsigned char *img1, unsigned char *img2,
                                         int *out, int N) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < N) {
                out[idx] = __sad(img1[idx], img2[idx], 0);
            }
        }
        """

        result = translate_cuda_source(cuda_code)
        assert "cripple_sad(" in result

    def test_sad_with_bare_atomic_add_fails(self):
        """A bare per-thread atomicAdd (not a recognized block-reduction
        idiom) has no Ripple equivalent, independent of the __sad rename."""
        cuda_code = """
        __global__ void compute_sad(unsigned char *img1, unsigned char *img2,
                                    int *sad, int N) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx < N) {
                int local_sad = __sad(img1[idx], img2[idx], 0);
                atomicAdd(sad, local_sad);
            }
        }
        """

        with pytest.raises(TranslationError):
            translate_cuda_source(cuda_code)
```

- [x] **Step 2: Run tests to verify the rename test fails**

Run: `venv/bin/python -m pytest tests/test_complex_kernels.py::TestImageProcessing::test_sad_rename tests/test_complex_kernels.py::TestImageProcessing::test_sad_with_bare_atomic_add_fails -v`
Expected: `test_sad_rename` FAILs (`cripple_sad(` not in result — it's still `ripple_sad`).
`test_sad_with_bare_atomic_add_fails` should already PASS (Task 2 made this hard-fail).

- [x] **Step 3: Rename the macro**

In `core/translation_rules.py:1667`, in `MathFunctionRule.MATH_MAP`, change:

```python
        '__sad': 'ripple_sad',  # We'll define a macro for this
```

to:

```python
        '__sad': 'cripple_sad',  # Translator-provided helper — no real Ripple SAD primitive exists
```

In `frontends/source/cuda_frontend.py:617`, change:

```python
/* Math intrinsics */
#define ripple_sad(x, y, z) (__builtin_abs((x) - (y)) + (z))
```

to:

```python
/* Math intrinsics (translator-provided, not part of the Ripple API — no
 * real Ripple SAD primitive exists) */
#define cripple_sad(x, y, z) (__builtin_abs((x) - (y)) + (z))
```

- [x] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_complex_kernels.py::TestImageProcessing -v`
Expected: PASS (all 5 tests in this class, including the two from Task 2 and the two from this task).

- [x] **Step 5: Run the full test suite to check for regressions**

Run: `venv/bin/python -m pytest -v`
Expected: All pass except `test_2d_convolution`, which fails starting in Task 4 (not yet
reached) — if it already fails here, something is wrong; stop and investigate before
continuing.

- [x] **Step 6: Commit**

```bash
git add core/translation_rules.py frontends/source/cuda_frontend.py \
        tests/test_complex_kernels.py
git commit -m "Rename ripple_sad to cripple_sad — no real Ripple SAD primitive exists

Confirmed via repo-wide search of the vendored Ripple docs: zero hits
for 'sad' anywhere. The macro itself (__builtin_abs(x-y)+z) is valid,
portable C independent of Ripple — only the misleading ripple_-prefixed
name is fixed here, which implied a real API symbol that doesn't exist.

Splits test_sad_computation into test_sad_rename (the __sad rename in
isolation) and test_sad_with_bare_atomic_add_fails (the unrelated
atomics hard-fail from the prior commit) — the original test asserted
on both concerns in one kernel."
```

---

## Task 4: VTCM — real `vtcm_malloc`/`vtcm_free` for shared memory

**Files:**
- Modify: `core/translation_rules.py:163-199` (`SharedMemoryRule`)
- Modify: `tests/test_translation.py:120-128` (`test_shared_memory_rule`)
- Modify: `tests/test_complex_kernels.py:267-302` (`test_2d_convolution`)
- Modify: `tests/stub_headers/ripple.h`
- Test (new): `tests/test_complex_kernels.py` (brace-matching + multi-dim hard-fail)

`SharedMemoryRule` currently emits `__attribute__((section(".vtcm")))`, which does not
exist anywhere in the Ripple API. The real mechanism, confirmed directly from the `SpVV`
example in `temp_ripple_docs/opt/hexagon/src/hvx-opt.md`:

```c
float * gathered_V = vtcm_malloc(sizeof(float) * nS, /*align_as=*/128);
...
vtcm_free(gathered_V);
```

This task also discovered (while writing this plan) that the old rule silently tolerated
multi-dimensional arrays like `__shared__ float tile[18][18]` by accident — it only ever
rewrote the prefix up to the first `[dim]`, leaving the second `[18]` as valid trailing
array-declarator syntax. A `vtcm_malloc()`-returned pointer can't be redeclared with a
trailing `[dim]` the same way, and every `tile[y][x]`-style indexing site elsewhere in
the kernel would need rewriting to flat pointer arithmetic — an AST-level transformation
regex substitution can't do safely. Multi-dimensional `__shared__` arrays must hard-fail
instead.

- [x] **Step 1: Write the failing tests**

Replace `tests/test_translation.py:120-128` (`test_shared_memory_rule`):

```python
    def test_shared_memory_rule(self):
        """__shared__ arrays translate to vtcm_malloc, with a matching
        vtcm_free() inserted before the enclosing block's closing brace —
        the real Ripple VTCM mechanism, not an attribute (which doesn't
        exist in the Ripple API)."""
        rule = SharedMemoryRule()
        ctx = TranslationContext(target_platform="hexagon")

        source = "void k() {\n    __shared__ float sdata[256];\n    sdata[0] = 1.0f;\n}"
        result = rule.apply(source, ctx)

        assert "float *sdata = vtcm_malloc(sizeof(float) * (256), /*align_as=*/128);" in result
        assert "vtcm_free(sdata);" in result
        assert result.index("vtcm_malloc") < result.index("vtcm_free")
        # the free() call must land before the function's closing brace
        assert result.index("vtcm_free") < result.rindex("}")

    def test_shared_memory_rule_multidim_hard_fails(self):
        """Multi-dimensional __shared__ arrays have no safe vtcm_malloc
        translation — a flat pointer can't be redeclared with a trailing
        [dim], and every indexing site would need flat-arithmetic
        rewriting this translator can't do. Must hard-fail, not emit
        invalid C."""
        rule = SharedMemoryRule()
        ctx = TranslationContext(target_platform="hexagon")

        source = "void k() {\n    __shared__ float tile[18][18];\n}"
        result = rule.apply(source, ctx)

        assert ctx.has_errors()
        assert "tile" in ctx.errors[0]
```

Replace `tests/test_complex_kernels.py:267-302` (`test_2d_convolution` — drop the shared
tile, which isn't what this test is actually verifying; keep the 2D `ripple_id`
assertions that are its real point):

```python
    def test_2d_convolution(self):
        """Test 2D convolution kernel (direct global-memory access — see
        test_shared_memory_rule_multidim_hard_fails in test_translation.py
        for the separate, dedicated 2D-shared-memory coverage)."""
        cuda_code = """
        #define KERNEL_SIZE 3

        __global__ void conv2d(float *input, float *kernel, float *output,
                               int width, int height) {
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int x = blockIdx.x * blockDim.x + tx;
            int y = blockIdx.y * blockDim.y + ty;

            if (x < width && y < height) {
                float sum = 0.0f;
                for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                    for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                        int sx = x + kx - KERNEL_SIZE / 2;
                        int sy = y + ky - KERNEL_SIZE / 2;
                        if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
                            sum += input[sy * width + sx] *
                                   kernel[ky * KERNEL_SIZE + kx];
                        }
                    }
                }
                output[y * width + x] = sum;
            }
        }
        """

        result = translate_cuda_source(cuda_code)
        assert "ripple_id(ripple_block, 0)" in result
        assert "ripple_id(ripple_block, 1)" in result
```

Add a new test to `TestConvolutionKernels` in `tests/test_complex_kernels.py` (after the
rewritten `test_2d_convolution`) exercising the brace-matching `vtcm_free()` insertion
through nested control flow, via the full pipeline:

```python
    def test_shared_memory_vtcm_free_placement(self):
        """vtcm_free() must be inserted at the kernel's closing brace, after
        the nested for/if statements that use the shared buffer — not
        immediately after the malloc."""
        cuda_code = """
        __global__ void reduce_sum(float *input, float *output, int n) {
            __shared__ float sdata[256];
            int tid = threadIdx.x;
            sdata[tid] = input[tid];
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
            }
            if (tid == 0) output[0] = sdata[0];
        }
        """

        result = translate_cuda_source(cuda_code)
        assert "vtcm_malloc(sizeof(float) * (256)" in result
        assert "vtcm_free(sdata);" in result
        assert result.index("for (") < result.index("vtcm_free(sdata)")
```

- [x] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_translation.py::TestTranslationRules::test_shared_memory_rule tests/test_translation.py::TestTranslationRules::test_shared_memory_rule_multidim_hard_fails tests/test_complex_kernels.py::TestConvolutionKernels -v`
Expected: FAIL — `test_shared_memory_rule` (old attribute output, no vtcm_malloc);
`test_shared_memory_rule_multidim_hard_fails` (`SharedMemoryRule` doesn't exist as an
attribute name yet — no error recorded); `test_shared_memory_vtcm_free_placement` (no
such function/assertions met yet). `test_2d_convolution` should already PASS as rewritten
(it no longer touches shared memory) — if it fails, the kernel rewrite has a bug; fix
before continuing.

- [x] **Step 3: Rewrite `SharedMemoryRule`**

Replace `core/translation_rules.py:163-199` (the entire `SharedMemoryRule` class) with:

```python
class SharedMemoryRule(TranslationRule):
    """Translates __shared__ array declarations to VTCM malloc/free pairs.

    Ripple's VTCM (Vector Tightly Coupled Memory) is accessed through
    vtcm_malloc()/vtcm_free() runtime calls (see the SpVV example in
    Ripple's HVX optimization guide) — there is no attribute-based way to
    place a variable in VTCM. The free() call is inserted immediately
    before the closing brace of the block enclosing the declaration
    (normally the kernel body itself, since __shared__ arrays are
    declared at kernel scope and live for the kernel's duration).

    Multi-dimensional arrays (`__shared__ float tile[18][18]`) are a
    hard-fail: a vtcm_malloc()-returned pointer can't be redeclared with
    a trailing [dim] the way a real array can, and every tile[y][x]-style
    indexing site elsewhere in the kernel would need rewriting to flat
    pointer arithmetic — an AST-level transformation this regex-based
    rule can't do safely.
    """

    # Array-form declarations only (requires trailing [...]) — scalar
    # `__shared__ float x;` is not matched or translated by this rule.
    # tests/test_translation.py reuses this pattern to detect leftover
    # untranslated declarations, so it inherits the same array-only scope.
    PATTERN = r'__shared__\s+(\w+)\s+(\w+)\s*\[([^\]]*)\]'
    EXTRA_DIM_PATTERN = re.compile(r'\s*\[[^\]]*\]')

    def __init__(self):
        super().__init__(
            name="shared_memory",
            description="Translate __shared__ array to vtcm_malloc/vtcm_free",
            cuda_pattern=self.PATTERN,
            priority=90
        )

    @staticmethod
    def _find_enclosing_brace_end(text: str, start: int) -> int:
        """Find the index of the '}' that closes the block containing `start`."""
        depth = 0
        i = start
        while i < len(text):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                if depth == 0:
                    return i
                depth -= 1
            i += 1
        return -1

    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        matches = list(re.finditer(self.PATTERN, cuda_code))
        result = cuda_code

        # Process right-to-left so earlier matches' offsets stay valid
        # while later matches are rewritten (both the declaration itself
        # and the vtcm_free() insertion grow the string).
        for match in reversed(matches):
            elem_type = match.group(1)
            var_name = match.group(2)
            size_expr = match.group(3)

            ctx.shared_mem_mappings[var_name] = elem_type

            if ctx.target_platform != "hexagon":
                # No VTCM claim to fix outside Hexagon — unchanged behavior.
                decl = f"__attribute__((aligned(128))) {elem_type} {var_name}[{size_expr}]"
                result = result[:match.start()] + decl + result[match.end():]
                continue

            extra_dim = self.EXTRA_DIM_PATTERN.match(result, match.end())
            if extra_dim:
                ctx.add_error(
                    f"SharedMemoryRule: multi-dimensional __shared__ array "
                    f"'{var_name}' (declared as {elem_type} {var_name}"
                    f"[{size_expr}]{extra_dim.group(0)}) is not supported. "
                    f"Ripple's vtcm_malloc() returns a flat buffer, so "
                    f"every indexing expression using '{var_name}' would "
                    f"need rewriting to flat/manual indexing, which this "
                    f"translator cannot do automatically. Flatten "
                    f"'{var_name}' to a 1D array with manual index "
                    f"arithmetic and retranslate."
                )
                continue

            free_pos = self._find_enclosing_brace_end(result, match.end())
            if free_pos == -1:
                ctx.add_error(
                    f"SharedMemoryRule: could not find the closing brace of "
                    f"the block containing '__shared__ {elem_type} "
                    f"{var_name}[{size_expr}]' — cannot place its matching "
                    f"vtcm_free() call. Check for unbalanced braces in the "
                    f"kernel."
                )
                continue

            free_call = f"\n    vtcm_free({var_name});"
            result = result[:free_pos] + free_call + result[free_pos:]

            malloc_decl = (
                f"// CUDA __shared__ -> Ripple VTCM\n"
                f"    {elem_type} *{var_name} = vtcm_malloc("
                f"sizeof({elem_type}) * ({size_expr}), /*align_as=*/128)"
            )
            result = result[:match.start()] + malloc_decl + result[match.end():]

        return result
```

- [x] **Step 4: Update the syntax-check stub header**

In `tests/stub_headers/ripple.h`, add before the closing `#endif` (currently line 46):

```c
/* opt/hexagon/src/hvx-opt.md's SpVV example uses vtcm_malloc()/vtcm_free()
 * but never gives them a formal declared signature — only usage. This
 * signature is inferred directly from that usage (size + alignment for
 * malloc; a single pointer for free), not from an upstream prototype. */
void *vtcm_malloc(size_t size, size_t align_as);
void vtcm_free(void *ptr);
```

- [x] **Step 5: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_translation.py::TestTranslationRules::test_shared_memory_rule tests/test_translation.py::TestTranslationRules::test_shared_memory_rule_multidim_hard_fails tests/test_complex_kernels.py::TestConvolutionKernels -v`
Expected: PASS

- [x] **Step 6: Run the full test suite to check for regressions**

Run: `venv/bin/python -m pytest -v`
Expected: All pass.

- [x] **Step 7: Commit**

```bash
git add core/translation_rules.py tests/test_translation.py \
        tests/test_complex_kernels.py tests/stub_headers/ripple.h
git commit -m "Translate __shared__ to real vtcm_malloc/vtcm_free, not a fake attribute

__attribute__((section(\".vtcm\"))) doesn't exist in the Ripple API.
Confirmed the real mechanism directly from the SpVV example in
temp_ripple_docs/opt/hexagon/src/hvx-opt.md: vtcm_malloc()/vtcm_free()
runtime calls. The free() call is placed via a brace-counting scan for
the enclosing block's end, mirroring the identical pattern already used
in the AST block parser and the IR frontend.

Multi-dimensional __shared__ arrays now hard-fail with a clear
diagnostic instead of silently emitting invalid C (a pointer can't be
redeclared with a trailing [dim] the way a real array can) — the old
attribute-based rule tolerated 2D+ arrays only by accident, since a
plain array declarator supports trailing dimensions the same way a
malloc'd pointer's declaration doesn't.

test_2d_convolution simplified to drop its incidental shared-memory
tile (not what the test verifies — 2D thread-index translation is);
dedicated 2D-shared-memory coverage now lives in
test_shared_memory_rule_multidim_hard_fails."
```

- [x] **Step 8 (optional but recommended — matches existing repo convention):**
Open a tracking issue for the multi-dimensional VTCM gap, the same way issue #11 tracks
the unsupported runtime-variable shuffle case:

```bash
gh issue create --repo C-ripple/Translate \
  --title "Multi-dimensional __shared__ arrays can't translate to VTCM" \
  --label enhancement \
  --body "SharedMemoryRule hard-fails on __shared__ arrays with more than one dimension (e.g. \`__shared__ float tile[18][18]\`). Ripple's vtcm_malloc() returns a flat pointer, which can't be redeclared with a trailing [dim] the way a real array can, and every tile[y][x]-style indexing site in the kernel would need rewriting to flat pointer arithmetic — an AST-level transformation the current regex-based rule engine can't do safely. See docs/superpowers/specs/2026-08-14-update-to-current-ripple-api-design.md for context."
```

---

## Task 4b: Hard-fail VTCM allocations that leak on an early return

**Files:**
- Modify: `core/translation_rules.py` (`SharedMemoryRule`)
- Modify: `tests/test_translation.py` (new test)

Task 4's code-quality review found a real, currently-latent gap:
`SharedMemoryRule` places `vtcm_free()` only immediately before the *enclosing block's*
closing brace. A guard-clause early `return` between the `__shared__` declaration and
that brace — an extremely common CUDA idiom (`if (idx >= n) return;`) — skips the
`vtcm_free()` call entirely on that path. VTCM is a small, real hardware scratchpad, not
virtual memory the OS reclaims on function exit, so this is a genuine resource leak, not
a style nit. Confirmed during review: no currently-passing fixture triggers it (nothing
in the test suite has a `return` between a surviving `__shared__` declaration and its
enclosing brace), so this is a latent gap in kernels the test suite doesn't yet cover,
not a regression — but it's real, and it fails silently, which is exactly the failure
mode this same rule already avoids for the multi-dimensional case one branch away. Fix it
the same way: hard-fail with a clear diagnostic, matching this rule's own established
convention, rather than leave it to bite a real kernel later.

Also fold in the review's second finding: `_find_enclosing_brace_end`'s
literal/comment-blind-spot caveat (a `}` inside a string or comment in the scanned range
would misplace `vtcm_free()`) currently lives only in the Task 4 commit message, not in
the code. Move it into the method's own docstring, where a future reader will actually
see it. Do NOT attempt to make the scanner comment/string-aware in this task — that's a
bigger, separately-scoped change (this rule's scan region is often the whole kernel body,
where real kernels routinely contain comments and string-free but brace-containing
constructs; a blanket bail-out on any comment in range would hard-fail legitimate,
currently-passing kernels, which is a worse regression than the documented limitation).

- [x] **Step 1: Write the failing test**

Add to the `TestTranslationRules` class in `tests/test_translation.py` (near the other
`test_shared_memory_rule*` tests):

```python
    def test_shared_memory_rule_early_return_hard_fails(self):
        """A 'return' between the __shared__ declaration and the enclosing
        block's end would leak the VTCM allocation on that path — VTCM is
        a small, real hardware scratchpad, not virtual memory. Must
        hard-fail rather than silently leak."""
        rule = SharedMemoryRule()
        ctx = TranslationContext(target_platform="hexagon")

        source = (
            "void k(int n) {\n"
            "    __shared__ float sdata[256];\n"
            "    if (n < 0) return;\n"
            "    sdata[0] = 1.0f;\n"
            "}"
        )
        result = rule.apply(source, ctx)

        assert ctx.has_errors()
        assert "sdata" in ctx.errors[0]
```

- [x] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_translation.py::TestTranslationRules::test_shared_memory_rule_early_return_hard_fails -v`
Expected: FAIL — no error is recorded today; the rule currently emits `vtcm_free(sdata);`
right before the final `}`, leaking on the `return` path instead of hard-failing.

- [x] **Step 3: Add the return-path check**

In `core/translation_rules.py`, in `SharedMemoryRule`, add a class-level pattern next to
`EXTRA_DIM_PATTERN`:

```python
    EXTRA_DIM_PATTERN = re.compile(r'\s*\[[^\]]*\]')
    RETURN_PATTERN = re.compile(r'\breturn\b')
```

In `apply()`, right after the existing `free_pos == -1` check (and its `continue`) and
before building `free_call`/`malloc_decl`, add:

```python
            if self.RETURN_PATTERN.search(result, match.end(), free_pos):
                ctx.add_error(
                    f"SharedMemoryRule: '{var_name}' is followed by a "
                    f"'return' before the end of its enclosing block. "
                    f"Placing vtcm_free('{var_name}') only at the block's "
                    f"closing brace would leak the VTCM allocation on "
                    f"that early-return path — VTCM is a small, real "
                    f"hardware scratchpad, not virtual memory. "
                    f"Restructure the kernel so '{var_name}' is freed on "
                    f"every exit path (e.g. move any early-exit checks "
                    f"before the __shared__ declaration, not after it), "
                    f"and retranslate."
                )
                continue
```

Then update `_find_enclosing_brace_end`'s docstring to note the literal/comment caveat.
Find:

```python
    @staticmethod
    def _find_enclosing_brace_end(text: str, start: int) -> int:
        """Find the index of the '}' that closes the block containing `start`."""
```

Replace with:

```python
    @staticmethod
    def _find_enclosing_brace_end(text: str, start: int) -> int:
        """Find the index of the '}' that closes the block containing `start`.

        Counts every '{'/'}' character with no awareness of string/char
        literals or comments — a brace inside either would misplace the
        result. No kernel in this codebase's test suite currently
        triggers this (verified: none have a comment or string literal
        containing a brace between a surviving __shared__ declaration and
        its enclosing '}'), but it's a real limitation, not a
        theoretical one — the same bug class UnrollConstantShuffleLoopRule
        needed several hardening rounds to close for its own, narrower
        scan region. A blanket bail-out on any comment/string in range
        isn't the right fix here: unlike that rule's small loop-body
        scan, this method's scan region is often an entire kernel body,
        where real kernels routinely contain comments — bailing out on
        their mere presence would hard-fail legitimate kernels. Making
        this scanner literal/comment-aware is a real fix but a separately
        scoped one.
        """
```

- [x] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_translation.py::TestTranslationRules::test_shared_memory_rule_early_return_hard_fails -v`
Expected: PASS

- [x] **Step 5: Run the full test suite to check for regressions**

Run: `venv/bin/python -m pytest -v`
Expected: All tests pass, 0 failures. (145 before this task; +1 new test.) If any
existing test now fails, check whether it has a `return` anywhere between a `__shared__`
declaration and the end of its enclosing block — that would mean this task's new check
is (correctly) catching a real pre-existing case a fixture happens to contain; if so,
that fixture needs the same kind of fix `test_tiled_matmul` got in Task 4 (restructure to
avoid the leak pattern, or accept the hard-fail as correct and adjust the test's
expectation) rather than weakening the new check.

- [x] **Step 6: Commit**

```bash
git add core/translation_rules.py tests/test_translation.py
git commit -m "Hard-fail VTCM allocations that would leak on an early return

Task 4's code-quality review found SharedMemoryRule's vtcm_free()
placement only covers the enclosing block's closing brace — a guard-
clause early return between the __shared__ declaration and that brace
(a common CUDA idiom) skips the free entirely, leaking VTCM, a small
real hardware scratchpad, not virtual memory. No current fixture hits
this, but it's a real gap, and it fails silently. Hard-failing it
matches this rule's own existing convention for the adjacent
multi-dimensional-array case.

Also moves the brace-counter's literal/comment-blind-spot caveat from
the Task 4 commit message into _find_enclosing_brace_end's own
docstring, where a future reader will actually see it, per the same
review."
```

---

## Task 5: Document the required `-fenable-ripple` compile flag

**Files:**
- Modify: `README.md`
- Modify: `interfaces/cli/cuda2ripple.py:387-395`
- Modify: `frontends/source/cuda_frontend.py:565-575` (boilerplate header comment)
- Modify: `interfaces/web/server.py` (status bar `<span>` text)
- Test: `tests/test_translation.py` (add to `TestBoilerplateGeneration`)

`-fenable-ripple` is required to compile any translated output (confirmed in the
troubleshooting guide's "Missing ripple_* symbols" section: *"Ripple is not activated
through command-line options... Use clang with the `-fenable-ripple` flag"*) and is
currently undocumented everywhere in this repo. Adding it in four places: README, CLI
help text, the generated file's own header comment, and the Flask server that has a
user-facing HTML page (`interfaces/web/server.py` — not the root `server.py`, which is a
minimal JSON API with no copy to update, and not the Flutter app, which has no
help/instructions surface today).

- [x] **Step 1: Write the failing test**

Add to the `TestBoilerplateGeneration` class in `tests/test_translation.py` (created in
Task 1):

```python
    def test_compile_flag_documented_in_output(self):
        """Generated output must tell the reader how to actually compile
        it — -fenable-ripple is required and easy to miss."""
        cuda_code = "__global__ void k(float *a) { a[0] = 1.0f; }"
        result = translate_cuda_source(cuda_code)
        assert "-fenable-ripple" in result
```

- [x] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_translation.py::TestBoilerplateGeneration::test_compile_flag_documented_in_output -v`
Expected: FAIL

- [x] **Step 3: Add the note to the generated header comment**

In `frontends/source/cuda_frontend.py`, replace lines 565-575:

```python
        header = f"""/*
 * Auto-generated RIPPLE code from CUDA source
 * Target: Hexagon HVX ({self.hexagon_config.hvx_mode})
 * Vector width: {self.hexagon_config.hvx_width} bytes
 * 
 * Translation warnings:
"""
```

with:

```python
        header = f"""/*
 * Auto-generated RIPPLE code from CUDA source
 * Target: Hexagon HVX ({self.hexagon_config.hvx_mode})
 * Vector width: {self.hexagon_config.hvx_width} bytes
 *
 * Compile with: clang -fenable-ripple ... — Ripple support is not
 * enabled by default; omitting this flag produces undefined-symbol
 * errors for every ripple_* call in this file even though the code
 * below is valid RIPPLE C.
 *
 * Translation warnings:
"""
```

- [x] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_translation.py::TestBoilerplateGeneration::test_compile_flag_documented_in_output -v`
Expected: PASS

- [x] **Step 5: Add the note to README.md**

In `README.md`, insert the following new blockquote immediately after the `### Command
Line` code block (after the closing ` ``` ` that follows `cuda2ripple interactive`, before
the `### Web Interface` heading):

```markdown
> **Compiling the output:** Ripple support isn't enabled by default in a
> Ripple-capable clang build — compile translated output with
> `clang -fenable-ripple ...` (see the Ripple Troubleshooting Guide's
> "Missing ripple\_\* symbols" section). Without this flag, translated
> code fails to compile with undefined-symbol errors even though it's
> valid RIPPLE C.
```

- [x] **Step 6: Add the note to the CLI epilog**

In `interfaces/cli/cuda2ripple.py`, replace lines 387-395:

```python
        epilog="""
Examples:
  cuda2ripple source kernel.cu -o kernel.ripple.c
  cuda2ripple ir kernel.ll -o kernel.ripple.ll
  cuda2ripple analyze kernel.cu --json
  cuda2ripple batch *.cu -o output/
  cuda2ripple interactive
        """
```

with:

```python
        epilog="""
Examples:
  cuda2ripple source kernel.cu -o kernel.ripple.c
  cuda2ripple ir kernel.ll -o kernel.ripple.ll
  cuda2ripple analyze kernel.cu --json
  cuda2ripple batch *.cu -o output/
  cuda2ripple interactive

Note: compile translated output with `clang -fenable-ripple ...` —
Ripple support is not enabled by default.
        """
```

- [x] **Step 7: Add the note to the web UI status bar**

In `interfaces/web/server.py`, find the status bar `<span>` (in `HTML_TEMPLATE`):

```html
        <div class="status-item">
            <span>Target: Hexagon HVX v68 | Vector: 128 bytes | RIPPLE v0.1</span>
        </div>
```

Replace with:

```html
        <div class="status-item">
            <span>Target: Hexagon HVX v68 | Vector: 128 bytes | RIPPLE v0.1 | Compile with: clang -fenable-ripple ...</span>
        </div>
```

- [x] **Step 8: Run the full test suite to check for regressions**

Run: `venv/bin/python -m pytest -v`
Expected: All pass.

- [x] **Step 9: Commit**

```bash
git add README.md interfaces/cli/cuda2ripple.py frontends/source/cuda_frontend.py \
        interfaces/web/server.py tests/test_translation.py
git commit -m "Document the required -fenable-ripple compile flag

Confirmed required via the Ripple troubleshooting guide's 'Missing
ripple_* symbols' section — without it, translated output fails to
compile with undefined-symbol errors despite being valid RIPPLE C, and
nothing in this repo mentioned it anywhere. Added to the generated
output's own header comment, README, CLI --help epilog, and the Flask
web UI's status bar (the one with user-facing copy — not the minimal
root server.py API, and not the Flutter app, which has no
help/instructions surface today)."
```

---

## Task 6: Documentation cleanup and final end-to-end verification

**Files:**
- Modify: `README.md`, `docs/README.md` (byte-identical copy — confirmed via `diff`;
  keep them in sync, don't investigate why both exist, that's unrelated pre-existing
  repo structure)
- Modify: `docs/ARCHITECTURE.md`
- Modify: `tests/test_translation.py` (two stale comments)

Code-quality review of Task 2 surfaced real documentation staleness that no other task's
file list covers: `README.md`/`docs/README.md`/`docs/ARCHITECTURE.md` all document
`atomicAdd(ptr, val)` translating to `ripple_atomic_add(ptr, val)` (Task 2 made this
false — it now hard-fails), and `README.md`'s "Parallel Reduction" worked example ends
with exactly that pattern. The same tables also document `__shared__ T arr[N]`
translating to `__attribute__((section(".vtcm"))) T arr[N]` (Task 4 makes this false too
— it's `vtcm_malloc`/`vtcm_free` now). Fixing this here, after Tasks 2-5 have all landed,
avoids editing these files twice.

- [x] **Step 1: Fix the translation-mapping tables and worked example**

In `README.md`, replace these two table rows (in the "Translation Mappings" section):

```markdown
| `__shared__ T arr[N]` | `__attribute__((section(".vtcm"))) T arr[N]` |
| `atomicAdd(ptr, val)` | `ripple_atomic_add(ptr, val)` |
```

with:

```markdown
| `__shared__ T arr[N]` | `T *arr = vtcm_malloc(sizeof(T) * N, /*align_as=*/128); ... vtcm_free(arr);` |
| `atomicAdd(ptr, val)` | *(no equivalent — Ripple has no atomics API; see the barrier + per-lane partial-sum pattern in the Ripple multi-threading guide)* |
```

Then replace the "Parallel Reduction" worked example (the `### Parallel Reduction`
section, both the CUDA Input and RIPPLE Output code blocks) with:

```markdown
### Parallel Reduction

**CUDA Input:**
```cuda
__global__ void reduceSum(float *input, float *output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = input[tid];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[0] = sdata[0];
}
```

**RIPPLE Output:**
```c
#include <ripple.h>

void reduceSum_ripple(...) {
    float *sdata = vtcm_malloc(sizeof(float) * (256), /*align_as=*/128);

    ripple_block_t ripple_block = ripple_set_block_shape(HVX_PE, 256);
    int tid = ripple_id(ripple_block, 0);
    sdata[tid] = input[tid];
    // __syncthreads: implicit in SIMD model

    for (int s = ripple_get_block_size(ripple_block, 0) / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
    }
    if (tid == 0) output[0] = sdata[0];
    vtcm_free(sdata);
}
```

Note this example writes the block's single reduced value directly rather than using
`atomicAdd` — Ripple has no atomics API, so accumulating across *multiple* blocks into
one shared output needs the barrier + per-lane partial-sum pattern from the Ripple
multi-threading guide instead; see the Translation Mappings table above.
```

Copy the fully-edited `README.md` over `docs/README.md` (`cp README.md docs/README.md`)
to preserve their current byte-identical state.

In `docs/ARCHITECTURE.md`, replace these two table rows:

```markdown
| `__shared__` | VTCM-allocated array | Hexagon tightly-coupled memory |
| `atomicAdd()` | `ripple_atomic_add()` | Or HVX scatter-accumulate |
```

with:

```markdown
| `__shared__` | `vtcm_malloc()`/`vtcm_free()` pair | Hexagon tightly-coupled memory |
| `atomicAdd()` | *(no equivalent)* | Ripple has no atomics API — see the barrier + per-lane partial-sum pattern in the Ripple multi-threading guide |
```

- [x] **Step 2: Fix two stale test comments**

In `tests/test_translation.py`, around line 1370, replace this comment (keep the
assertions below it unchanged — only the comment text is stale):

```python
    # NOTE: deliberately not "?" not in result — like the "if (" note on
    # test_predicated_unroll_does_not_collide_with_plain_unroll_rule
    # above, the RIPPLE boilerplate unconditionally emits a
    # ripple_atomic_cas fallback macro containing a '?' ternary, so
    # that check would fail for any translation output regardless of
    # this rule's behavior. Assert directly on the original ternary
    # being gone from the kernel body instead.
```

with:

```python
    # NOTE: deliberately not "?" not in result — asserting the original
    # ternary is gone from the kernel body is a more direct check than a
    # blanket "no '?' anywhere in the file" assertion regardless. (This
    # comment previously cited a since-removed ripple_atomic_cas fallback
    # macro that also contained a '?' as the reason — that macro no
    # longer exists, but the more-direct assertion below is still the
    # better check on its own merits.)
```

Around line 1531, replace this comment (same rule — keep the assertions unchanged):

```python
    # NOTE: deliberately not "if (" not in result — the RIPPLE
    # boilerplate header unconditionally emits ripple_atomic_max/min
    # macros containing "if (", so that check would fail for any
    # translation output regardless of this rule's behavior. Assert
    # directly on which rule's warning fired instead.
```

with:

```python
    # NOTE: deliberately not "if (" not in result — asserting on which
    # rule's warning fired is a more direct, less brittle check than a
    # blanket "no 'if (' anywhere in the file" assertion regardless.
    # (This comment previously cited since-removed ripple_atomic_max/min
    # fallback macros that also contained "if (" as the reason — those
    # macros no longer exist, but the more-direct assertion below is
    # still the better check on its own merits.)
```

- [x] **Step 3: Run the full test suite to confirm no regressions**

Run: `venv/bin/python -m pytest -v`
Expected: All tests pass, 0 failures, 0 errors (same count as after Task 5 — this step
only touched docs and comments, no behavior).

- [x] **Step 4: Commit the documentation cleanup**

```bash
git add README.md docs/README.md docs/ARCHITECTURE.md tests/test_translation.py
git commit -m "Fix documentation left stale by the atomics and VTCM rewrites

README.md, docs/README.md, and docs/ARCHITECTURE.md all documented
atomicAdd translating to ripple_atomic_add and __shared__ translating
to a VTCM attribute — both now false after Tasks 2 and 4. Found during
Task 2's code-quality review. Updated the translation-mapping tables
and README's worked 'Parallel Reduction' example to match current
behavior, and fixed two test comments in test_translation.py that
cited since-removed fallback macros as their rationale."
```

- [x] **Step 5: Run the full test suite**

Run: `venv/bin/python -m pytest -v`
Expected: All tests pass, 0 failures, 0 errors.

- [x] **Step 6: Verify a real, previously-broken kernel now hard-fails correctly**

```bash
venv/bin/python server.py > /tmp/cripple_verify_server.log 2>&1 &
sleep 2
curl -s http://127.0.0.1:5001/translate -X POST -H "Content-Type: application/json" \
  -d '{"source": "__global__ void reduceSum(float *input, float *output, int n) {\n    __shared__ float sdata[256];\n    int tid = threadIdx.x;\n    sdata[tid] = input[tid];\n    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) sdata[tid] += sdata[tid + s];\n    }\n    if (tid == 0) atomicAdd(output, sdata[0]);\n}"}' \
  | python3 -m json.tool
```

Expected: a JSON `error` response (not `translated`) — the `atomicAdd(output, sdata[0])`
here is a bare per-thread-scope call the reduction-idiom rules don't recognize (there's
no shared-memory reduce-to-`sdata[0]`-then-single-atomicAdd pattern matcher for this
exact shape), so it must hard-fail per Task 2, not silently emit `ripple_atomic_add`.

*(If this instead returns a real `ripple_reduceadd`-based translation, that means
`WarpReductionRule`/`ButterflyAllReduceRule` already recognizes this exact shape as a
reduction idiom — check the response for `ripple_reduceadd`; if present, this is a
correct pass, not a bug. Either a hard-fail or a `ripple_reduceadd` translation is
correct here; a bare `ripple_atomic_add` in the output is not.)*

- [x] **Step 7: Verify a real kernel using VTCM now compiles**

```bash
curl -s http://127.0.0.1:5001/translate -X POST -H "Content-Type: application/json" \
  -d '{"source": "__global__ void reduceSum(float *input, float *output, int n) {\n    __shared__ float sdata[256];\n    int tid = threadIdx.x;\n    sdata[tid] = input[tid];\n    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) sdata[tid] += sdata[tid + s];\n    }\n}"}' \
  | python3 -m json.tool
```

Expected: a JSON `translated` response containing `vtcm_malloc`, `vtcm_free`, and
`#define HVX_PE 0`, and NOT containing `__attribute__((section(".vtcm")))`.

- [x] **Step 8: Verify the translated output actually syntax-checks**

Save the `translated` field from Step 3 to a file and run it through the same check
`tests/compile_verify.py` uses:

```bash
venv/bin/python3 -c "
import json, subprocess, sys
resp = json.load(open('/tmp/cripple_vtcm_output.json'))
open('/tmp/cripple_vtcm_output.c', 'w').write(resp['translated'])
result = subprocess.run(
    ['clang', '-fsyntax-only', '-xc', '-Itests/stub_headers', '/tmp/cripple_vtcm_output.c'],
    capture_output=True, text=True
)
print(result.returncode, result.stdout, result.stderr)
"
```

(First save Step 3's response to `/tmp/cripple_vtcm_output.json` — pipe `curl` there
instead of through `python3 -m json.tool`.)

Expected: exit code `0`, no output (clean syntax check against the updated stub header).

- [x] **Step 9: Stop the server**

```bash
pkill -f "venv/bin/python server.py"
```

- [x] **Step 10: Final commit (only if Steps 5-8 needed any fixes)**

Step 4 already committed the documentation cleanup. If the verification steps (5-8)
passed cleanly with no further fixes needed, no additional commit is needed here. If
verification surfaced a bug, fix it, re-run the full suite, and commit with a message
describing what this step caught.
