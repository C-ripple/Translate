# Warp Min/Max Reduction Rule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the specific translation failure that orphaned `tests/examples/cuda_kernels.cu` — a real, 10-kernel CUDA fixture that exists in the repo but was never wired into any test because it fails to translate. The failure is a single unresolved shuffle: the softmax kernel's max-reduction loop (`for (int offset = warpSize / 2; offset > 0; offset /= 2) { float other = __shfl_down_sync(0xffffffff, local_max, offset); local_max = fmaxf(local_max, other); }`) doesn't match any existing rule's shape — `WarpReductionRule` requires a single `accum += __shfl_down_sync(...)` statement, and this is a two-statement max-reduction, not a `+=` accumulate. RIPPLE has native `ripple_reducemax`/`ripple_reducemin` intrinsics for exactly this idiom (confirmed in `temp_ripple_docs/src/ripple-spec/api.md`'s reduction-functions list). This plan adds a sibling rule recognizing that shape and wires the fixture into the "translates without error" test coverage.

**Explicitly out of scope for this plan:** getting `cuda_kernels.cu` to additionally pass a real clang syntax check. That requires fixing three separate, pre-existing gaps unrelated to shuffles — scalar (non-array) `__shared__` declarations aren't translated at all (only array-form ones are, a documented limitation predating this plan), a bare `warpSize` token used outside a recognized loop pattern passes through untranslated, and a few CUDA math intrinsics (`expf`, `INFINITY`, `__float_as_int`) have no RIPPLE mapping. This fixture is the *first* thing in the whole test suite to exercise any of those three gaps (confirmed: no other fixture uses `__shared__` at all). Fixing them is a materially larger, unrelated scope — this plan only closes the shuffle-specific failure and documents the remaining gaps clearly rather than silently omitting them.

**Architecture:** A new sibling rule, `WarpMinMaxReductionRule`, registered alongside `WarpReductionRule` — same tier (recognize a known reduction idiom, replace with the matching native intrinsic), same `warpSize/2`-halving loop shape, but matching the two-statement `fmaxf`/`fminf` pattern instead of the single-statement `+=` pattern. Mutually exclusive with `WarpReductionRule` by construction (different body shape), so no collision risk.

**Tech Stack:** Python 3.13, existing translator codebase, pytest.

---

## Before you start

Grounded in things verified directly in this session, not assumed:

- `ripple_reducemax`/`ripple_reducemin` are real, documented RIPPLE intrinsics — confirmed in `temp_ripple_docs/src/ripple-spec/api.md`'s "other reduction functions" list (alongside `ripple_reduceand`, `ripple_reduceor`, `ripple_reducexor`), same family as `ripple_reduceadd` which `WarpReductionRule` already uses.
- The new rule's pattern (a `\w+\s+\w+\s*=\s*__shfl_down_sync(...)` declaration statement followed by `accum = fmaxf(accum, temp)` or `fminf`) was tested directly against `WarpReductionRule`'s exact pattern and confirmed mutually exclusive both ways: the `+=`-shaped loop doesn't match the new pattern, and the `fmaxf`-shaped loop doesn't match `WarpReductionRule`'s pattern.
- The fix was prototyped and run against the *actual* `cuda_kernels.cu` file before writing this plan: with the new rule registered, `CUDAToRIPPLETransformer.transform()` on the full file completes without raising `TranslationError` (previously it raised, citing exactly this loop's `offset` argument). The warnings list confirms the right rule fired: `"Optimized warp fmaxf reduction for 'local_max' to 'ripple_reducemax'"`, alongside the two `WarpReductionRule` firings that already worked (`val`, `local_sum`).
- Running the same output through `verify_ripple_syntax()` fails, but for reasons independent of this rule — confirmed by reading the clang errors directly: `__attribute__((section(".vtcm")))` on local (non-file-scope) arrays, bare `warpSize`/`INFINITY`/`expf`/`__float_as_int` with no translation, and scalar `__shared__` declarations passed through untouched. None of these errors reference `ripple_reducemax`, `ripple_reducemin`, or anything this plan's rule touches.

---

## Task 1: Implement `WarpMinMaxReductionRule` and wire the fixture into "translates without error" coverage

**Files:**
- Modify: `core/translation_rules.py`
- Modify: `tests/test_translation.py`
- Modify: `tests/test_real_kernels.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_translation.py`:

```python
def test_warp_minmax_reduction_translates_fmaxf_to_ripple_reducemax():
    source = """
__global__ void warpMax(float *data) {
    float local_max = data[threadIdx.x];
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, local_max, offset);
        local_max = fmaxf(local_max, other);
    }
    data[threadIdx.x] = local_max;
}
"""
    result = translate_cuda_source(source)
    assert "__shfl_down_sync" not in result
    assert "ripple_reducemax(0b1, local_max)" in result


def test_warp_minmax_reduction_translates_fminf_to_ripple_reducemin():
    source = """
__global__ void warpMin(float *data) {
    float local_min = data[threadIdx.x];
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, local_min, offset);
        local_min = fminf(local_min, other);
    }
    data[threadIdx.x] = local_min;
}
"""
    result = translate_cuda_source(source)
    assert "__shfl_down_sync" not in result
    assert "ripple_reducemin(0b1, local_min)" in result


def test_warp_minmax_reduction_does_not_collide_with_plain_warp_reduction_rule():
    # Regression guard: the classic '+=' accumulate shape must still go
    # through WarpReductionRule unchanged — the two rules match
    # mutually exclusive body shapes (single '+=' statement vs. a
    # two-statement declare-then-fmaxf/fminf), so there's no overlap.
    source = """
__global__ void reduce(float *val) {
    float sum = *val;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    *val = sum;
}
"""
    result = translate_cuda_source(source)
    assert "ripple_reduceadd(0b1, sum)" in result
    assert "ripple_reducemax" not in result
    assert "ripple_reducemin" not in result


def test_warp_minmax_reduction_fires_on_real_fixture_snippet():
    # The exact loop from tests/examples/cuda_kernels.cu's softmax
    # kernel — the specific loop that orphaned that fixture (it's the
    # only unresolved shuffle in the whole 10-kernel file).
    source = """
__global__ void softmax(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_max = (idx < n) ? input[idx] : -1.0f;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, local_max, offset);
        local_max = fmaxf(local_max, other);
    }
    output[idx] = local_max;
}
"""
    result = translate_cuda_source(source)
    assert "ripple_reducemax(0b1, local_max)" in result
```

- [ ] **Step 2: Run to confirm they fail**

Run: `source venv/bin/activate && python -m pytest tests/test_translation.py -k "warp_minmax" -v`
Expected: `test_warp_minmax_reduction_translates_fmaxf_to_ripple_reducemax`, `test_warp_minmax_reduction_translates_fminf_to_ripple_reducemin`, and `test_warp_minmax_reduction_fires_on_real_fixture_snippet` FAIL (the rule doesn't exist yet, so `__shfl_down_sync` remains untranslated and `translate_cuda_source` raises `TranslationError` instead of returning). `test_warp_minmax_reduction_does_not_collide_with_plain_warp_reduction_rule` should already PASS (nothing to wrongly fire on yet).

- [ ] **Step 3: Implement the rule**

In `core/translation_rules.py`, add this class immediately after `class WarpReductionRule` (i.e., right before `class ButterflyAllReduceRule` — grouping it there matches this file's "related rules sit near each other" convention, since it's the min/max sibling of `WarpReductionRule`'s add-reduction):

```python
class WarpMinMaxReductionRule(TranslationRule):
    """
    Detects and optimizes the two-statement warp min/max reduction
    idiom — the fmaxf/fminf sibling of WarpReductionRule's single-
    statement '+=' accumulate pattern:

    Pattern:
      for (int offset = warpSize/2; offset > 0; offset /= 2) {
          TYPE other = __shfl_down_sync(..., val, offset);
          val = fmaxf(val, other);   // or fminf
      }

    Replacement:
      val = ripple_reducemax(0b1, val);   // or ripple_reducemin

    ripple_reducemax/ripple_reducemin are real RIPPLE intrinsics (see
    api.md's reduction-functions list, alongside ripple_reduceadd,
    ripple_reduceand, ripple_reduceor, ripple_reducexor) — same family
    WarpReductionRule already uses for ripple_reduceadd, just for the
    two-statement min/max shape instead of the single-statement '+='
    shape. Mutually exclusive with WarpReductionRule by construction:
    that rule's PATTERN requires the loop body be exactly one '+='
    statement, which can't match this rule's two-statement
    declare-then-fmaxf/fminf shape, and vice versa.
    """

    # Group 1: Loop variable (e.g. "offset")
    # Group 2: Temporary variable holding the shuffled-in value (e.g. "other")
    # Group 3: Accumulator variable (e.g. "local_max")
    # Group 4: "fmaxf" or "fminf"
    PATTERN = (
        r'for\s*\(\s*int\s+(\w+)\s*=\s*warpSize\s*/\s*2\s*;\s*\1\s*>\s*0\s*;\s*\1\s*/=\s*2\s*\)\s*\{\s*'
        r'\w+\s+(\w+)\s*=\s*__shfl_down_sync\s*\(\s*[^,]+,\s*(\w+)\s*,\s*\1\s*\)\s*;\s*'
        r'\3\s*=\s*(fmaxf|fminf)\s*\(\s*\3\s*,\s*\2\s*\)\s*;\s*\}'
    )

    FUNC_TO_INTRINSIC = {'fmaxf': 'ripple_reducemax', 'fminf': 'ripple_reducemin'}

    def __init__(self):
        super().__init__(
            name="warp_minmax_reduction",
            description="Optimize warp min/max reduction loop to ripple_reducemax/ripple_reducemin",
            cuda_pattern=self.PATTERN,
            priority=84  # Same tier as WarpReductionRule (85) — a
                         # distinct value to avoid relying on
                         # stable-sort tie-breaking, though it doesn't
                         # matter for correctness since the two
                         # patterns are mutually exclusive (single '+='
                         # statement vs. two-statement declare-then-
                         # fmaxf/fminf). Above the shuffle rules (70)
                         # so this rule gets first look at the loop.
        )

    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            accum_var = match.group(3)
            func = match.group(4)
            intrinsic = self.FUNC_TO_INTRINSIC[func]

            ctx.add_warning(
                f"Optimized warp {func} reduction for '{accum_var}' to '{intrinsic}'"
            )

            # Same dims-bitfield reasoning as WarpReductionRule: this
            # rule only matches the classic single-dimension
            # warpSize-halving reduction, i.e. dimension 0 — hence
            # 0b1, not an arbitrary placeholder.
            return f"""/* CUDA Warp {func} Reduction Loop -> RIPPLE Intrinsic */
    {accum_var} = {intrinsic}(0b1, {accum_var});"""

        return re.sub(self.PATTERN, replace, cuda_code, flags=re.DOTALL)
```

- [ ] **Step 4: Register the rule**

In `core/translation_rules.py`, in `TranslationRuleEngine._register_default_rules()`, in the `default_rules` list, add `WarpMinMaxReductionRule(),` immediately after `WarpReductionRule(),`:

```python
            # Shuffles
            WarpReductionRule(),
            WarpMinMaxReductionRule(),
            ButterflyAllReduceRule(),
            UnrollConstantShuffleLoopRule(),
            PredicatedShuffleUnrollRule(),
            ShuffleDownRule(),
            ShuffleUpRule(),
            ShuffleXorRule(),
            ShuffleSyncRule(),
```

- [ ] **Step 5: Run all the tests from Step 1**

Run: `source venv/bin/activate && python -m pytest tests/test_translation.py -k "warp_minmax" -v`
Expected: all 4 PASS.

- [ ] **Step 6: Wire the real fixture into "translates without error" coverage — NOT syntax-check coverage**

In `tests/test_real_kernels.py`, add `"cuda_kernels.cu"` to `KERNEL_FILES` only (NOT `SYNTAX_CHECK_PARAMS` — this fixture doesn't pass a real clang syntax check yet, for reasons unrelated to this task; see the module docstring addition below):

```python
KERNEL_FILES = [
    "ast_flat.cu",
    "ast_if_no_braces.cu",
    "atomics_cas_exch.cu",
    "bitwise_intrinsics.cu",
    "global_thread_index.cu",
    "warp_reduction.cu",
    "warp_shuffle_xor.cu",
    "butterfly_reduction.cu",
    "cuda_kernels.cu",
]
```

Also add a paragraph to the module docstring at the top of `tests/test_real_kernels.py` (after the existing `warp_shuffle_xor.cu covers issue #8...` paragraph), explaining the exclusion clearly so a future reader doesn't mistake it for an oversight:

```python
cuda_kernels.cu is a larger, 10-kernel fixture that was previously
orphaned entirely (wired into no test) because it failed to translate
at all — a softmax kernel's max-reduction shuffle loop (fmaxf, not
'+=') didn't match any rule's shape. WarpMinMaxReductionRule closes
that gap, so this file now translates successfully and is covered
here structurally. It's deliberately NOT in SYNTAX_CHECK_PARAMS yet:
getting it through a real clang syntax check requires fixing three
separate, unrelated gaps this file is the first fixture to exercise —
scalar (non-array) __shared__ declarations aren't translated at all
(only array-form ones are), a bare warpSize token used outside a
recognized loop pattern passes through untranslated, and a few CUDA
math intrinsics (expf, INFINITY, __float_as_int) have no RIPPLE
mapping. None of that is shuffle-related; closing it is separate,
larger, unscoped work.
```

- [ ] **Step 7: Run the real-kernel tests**

Run: `source venv/bin/activate && python -m pytest tests/test_real_kernels.py -v`
Expected: `test_translates_without_error[cuda_kernels.cu]` PASSES. `test_translated_output_is_valid_syntax[cuda_kernels.cu]` should NOT exist as a test case (since `cuda_kernels.cu` isn't in `SYNTAX_CHECK_PARAMS`) — confirm the parametrize list doesn't include it. Total: 17 passed (16 prior + 1 new structural test; no new syntax-check test since it's deliberately excluded).

- [ ] **Step 8: Run the full suite**

Run: `source venv/bin/activate && python -m pytest tests/ -v`
Expected: 130 passed, 0 failed (125 prior + 4 new `warp_minmax` tests + 1 new `cuda_kernels.cu` structural test).

- [ ] **Step 9: Manually reproduce the fix end-to-end via the CLI, against the real fixture**

Run:
```bash
python -m interfaces.cli.cuda2ripple source tests/examples/cuda_kernels.cu -o /tmp/cuda_kernels.ripple.c
echo "exit code: $?"
grep -n "ripple_reducemax(0b1\|ripple_reduceadd(0b1" /tmp/cuda_kernels.ripple.c
```
Expected: exit code 0 (translation succeeds — previously this exited 1 with a `TranslationError` about `offset`). The grep, anchored on `(0b1` to match only the actual emitted call lines (not the header comment block, which separately echoes each intrinsic's name inside its own translation-warnings summary and would otherwise inflate a naive count), should show exactly 3 lines: `val = ripple_reduceadd(0b1, val);`, `local_sum = ripple_reduceadd(0b1, local_sum);` (both from the existing `WarpReductionRule`), and `local_max = ripple_reducemax(0b1, local_max);` (from this task's new rule).

- [ ] **Step 10: Commit**

```bash
git add core/translation_rules.py tests/test_translation.py tests/test_real_kernels.py
git commit -m "Recognize the warp min/max reduction idiom as ripple_reducemax/ripple_reducemin

Closes the specific translation failure that orphaned
tests/examples/cuda_kernels.cu — a real 10-kernel fixture that existed
in the repo but was wired into no test because it failed to translate.
The failure was a single unresolved shuffle: the softmax kernel's
max-reduction loop uses the classic warpSize/2-halving shape
WarpReductionRule already recognizes, but with a two-statement
fmaxf-based body instead of a single '+=' accumulate statement, so it
didn't match that rule (or anything else) and fell through to a hard
fail.

WarpMinMaxReductionRule is the fmaxf/fminf sibling of
WarpReductionRule: same loop shape, same tier, replacing the
two-statement declare-then-fmaxf/fminf body with a single
ripple_reducemax/ripple_reducemin call — both real RIPPLE intrinsics,
same family as ripple_reduceadd. Mutually exclusive with
WarpReductionRule by construction (different body shape).

Wired cuda_kernels.cu into the 'translates without error' test
coverage now that it actually does. Deliberately NOT added to the
syntax-check coverage list: this fixture is the first thing in the
whole suite to use scalar __shared__ declarations, a bare warpSize
token outside a recognized loop pattern, or several CUDA math
intrinsics with no RIPPLE mapping (expf, INFINITY, __float_as_int) —
three separate, unrelated gaps this task doesn't attempt to close.
Documented clearly in the test file's module docstring so the
exclusion reads as a deliberate, scoped decision rather than an
oversight.

Verified end-to-end via the CLI: the real fixture now translates
successfully (exit 0), where it previously hard-failed."
```

---

## Self-Review

**Spec coverage:** "fix the orphaned cuda_kernels.cu fixture too" → scoped explicitly (per direct conversation confirmation) to the shuffle-specific failure only, not the separate syntax-check gaps. The new rule directly targets the one loop that caused the original `TranslationError`, confirmed by prototyping against the real file before writing this plan. ✓

**Placeholder scan:** no TBD/TODO/"add appropriate handling" phrasing; every code step is complete, runnable code (prototyped and verified against the real fixture file before being written into this plan); every "Run:" step has a concrete command and expected count.

**Type/name consistency:** `WarpMinMaxReductionRule` is defined once (Task 1, Step 3) and registered once, by that exact name, in `_register_default_rules()` (Step 4). `FUNC_TO_INTRINSIC` maps exactly the two function names the pattern's own alternation captures (`fmaxf|fminf`) — no drift possible between what the regex can capture and what the dict handles.

**Expected test counts, traced cumulatively:** baseline before this plan = 125 passed (confirmed live in the worktree before writing this plan). This task adds 4 tests in `test_translation.py` + 1 in `test_real_kernels.py` → 130.

**Dependency ordering:** none — self-contained task, mutually exclusive by pattern shape with every existing rule (verified directly, not just argued, per "Before you start").

---

## Actual Outcome (added post-merge-review — this plan document went stale during implementation, corrected here rather than left silently wrong)

This branch shipped as 3 commits, not the single commit this plan describes:

1. `56c2313` — the implementation exactly as planned above.
2. `e47798a` — a follow-up, found by spec-compliance review, correcting the syntax-check-exclusion docstring in `tests/test_real_kernels.py`. This plan's own line 7 says "three separate, pre-existing gaps" — that count was wrong even at planning time: line 22 above ("Before you start") already lists a 4th, the `__attribute__((section(".vtcm")))` local-array error, but it never made it into the Step 6 docstring text that shipped. The corrected docstring lists five categories total, including a 5th this task itself introduces (`ripple_reducemax` isn't declared in `tests/stub_headers/ripple.h` — irrelevant today since the fixture isn't in `SYNTAX_CHECK_PARAMS`, but would bite the moment someone tries to move it there without also updating the stub header).
3. `2174359` — a follow-up, found by code-quality review, adding tolerance for `__shfl_down_sync`'s optional 4th (width) argument — `WarpReductionRule`'s pattern already tolerates it, and this rule's docstring explicitly calls itself that rule's sibling, so the asymmetry was unexplained inconsistency, not a deliberate narrowing. Not anticipated by this plan at all; added a regression test alongside the fix.

Final test count: **131 passed**, not the 130 predicted above (125 baseline + 4 `warp_minmax` tests + 1 `cuda_kernels.cu` structural test + 1 width-argument regression test from commit 3).
