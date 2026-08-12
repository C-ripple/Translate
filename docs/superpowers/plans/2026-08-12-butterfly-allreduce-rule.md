# Butterfly All-Reduce Rule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recognize the classic full-warp XOR-butterfly all-reduce loop (`for (int i = 1; i < 32; i *= 2) { val += __shfl_xor_sync(mask, val, i); }`) as a semantic pattern and translate it directly to `val = ripple_reduceadd(0b1, val);` — one native intrinsic call — instead of falling through to `UnrollConstantShuffleLoopRule`, which correctly but sub-optimally unrolls it into 5 separate hoisted permutation functions plus 5 `ripple_shuffle` calls.

**Architecture:** A single new `TranslationRule`, `ButterflyAllReduceRule`, registered at a priority above `UnrollConstantShuffleLoopRule` (82) so it gets first look at the loop, alongside `WarpReductionRule` (85) which already does the equivalent optimization for the `shfl_down`-halving-reduction variant of this same idea. Unlike `WarpReductionRule`, this new rule also validates the shuffle's mask argument is the literal full-warp mask (`0xffffffff`) before firing — a `shfl_xor` loop with a non-full or variable mask is not provably a whole-block all-reduce, and mistranslating it to `ripple_reduceadd(0b1, ...)` would silently change which lanes participate, which is a correctness bug, not a missed optimization. When the mask check fails (or the pattern doesn't match at all — e.g. a partial-width loop bounded to fewer than the full warp, which is a segmented/sub-group reduction and NOT expressible as a single whole-block `ripple_reduceadd`), the rule declines silently and the loop falls through unchanged to the existing `UnrollConstantShuffleLoopRule` → shuffle-rules → hard-fail pipeline, which is still correct for those cases, just not optimal.

**Tech Stack:** Python 3.13, existing translator codebase, pytest.

---

## Before you start

Grounded in things verified directly, not assumed:

- `ripple_reduceadd(int dims, TYPE to_reduce)` reduces along the dimension(s) set in the `dims` bitfield — confirmed by reading `temp_ripple_docs/src/ripple-spec/api.md:45` and its usage examples. RIPPLE's block-shaped values auto-promote (broadcast) back up on reassignment to a per-lane variable — confirmed by the spec's own text at `api.md:389-390` ("automatic broadcast promotes lower-dimensional blocks... to higher-dimensional ones") — so no explicit `ripple_broadcast` call is needed after a reduce that gets reassigned to the original accumulator variable.
- `WarpReductionRule` (`core/translation_rules.py:779-819`) already does exactly this optimization for the `shfl_down`-halving variant, emitting `{accum_var} = ripple_reduceadd(0b1, {accum_var});` with no explicit broadcast — confirmed by reading its `apply()` method directly. This new rule mirrors that established, working idiom rather than inventing a new one.
- Existing rule priorities (confirmed via `grep -n "priority=" core/translation_rules.py`): `WarpReductionRule` = 85, `UnrollConstantShuffleLoopRule` = 82, the 4 shuffle rules (`ShuffleDownRule`/`ShuffleXorRule`/`ShuffleUpRule`/`ShuffleSyncRule`) = 70. `TranslationRuleEngine.register_rule()` sorts `self.rules` by `priority` descending on every insert, so a new rule's priority value alone determines its place in execution order regardless of list position.
- `tests/examples/butterfly_reduction.cu` (the real fixture already in the suite) is the exact full-warp shape this rule targets: `for (int i = 1; i < 32; i *= 2) { val += __shfl_xor_sync(0xffffffff, val, i); }`. Its existing coverage in `tests/test_real_kernels.py` (`KERNEL_FILES`/`SYNTAX_CHECK_PARAMS`) only asserts generic properties (non-empty output, no leftover `__global__`, valid syntax) — confirmed by reading `tests/test_real_kernels.py:66-83` — so it will keep passing regardless of which rule ends up translating this fixture, and doesn't need modification. This plan adds a new, more specific test proving this exact fixture is now handled by the new rule rather than by unrolling.
- A loop bounded to fewer lanes than the full warp (e.g. `i < 8`) is a segmented/sub-group reduction, not a whole-block all-reduce — `ripple_reduceadd(0b1, ...)` always reduces across the *entire* dimension, so firing on a partial-width loop would silently produce a wrong answer (every lane would get the sum of all 32 lanes, not just its group of 8). This rule's pattern only accepts a literal bound of `32` or the symbolic token `warpSize` (mirroring `WarpReductionRule`'s existing assumption that `warpSize` always means "the whole warp, dimension 0") — any other literal bound doesn't match this pattern at all and falls through to `UnrollConstantShuffleLoopRule`, which handles it correctly (if less optimally) since it doesn't attach any reduction-specific meaning to the loop.

---

## Task 1: Implement and test `ButterflyAllReduceRule`

**Files:**
- Modify: `core/translation_rules.py`
- Modify: `tests/test_translation.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_translation.py`:

```python
def test_butterfly_allreduce_recognized_as_single_reduceadd():
    # The exact motivating example: a full-warp (32-lane) XOR-butterfly
    # all-reduce with a full mask should now translate to ONE
    # ripple_reduceadd call instead of being unrolled into 5 separate
    # ripple_shuffle calls — better output for a pattern RIPPLE has a
    # native intrinsic for, matching WarpReductionRule's existing
    # idiom for the shfl_down-halving variant of the same idea.
    source = """
__global__ void butterflyReduce(float *data) {
    float val = data[threadIdx.x];
    for (int i = 1; i < 32; i *= 2) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    assert "ripple_reduceadd(0b1, val)" in result
    assert "ripple_shuffle(" not in result
    assert "__shfl_xor_sync" not in result


def test_butterfly_allreduce_fires_on_real_fixture():
    # The real fixture already shipped in tests/examples/ — proves this
    # isn't only working on a synthetic example.
    source = (Path(__file__).parent / "examples" / "butterfly_reduction.cu").read_text()
    result = translate_cuda_source(source)
    assert "ripple_reduceadd(0b1, val)" in result
    assert "ripple_shuffle(" not in result


def test_butterfly_allreduce_accepts_symbolic_warpsize_bound():
    # Real CUDA code commonly writes the bound symbolically rather than
    # hardcoding 32 — must be recognized the same way.
    source = """
__global__ void butterflyReduce(float *data) {
    float val = data[threadIdx.x];
    for (int i = 1; i < warpSize; i *= 2) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    assert "ripple_reduceadd(0b1, val)" in result
    assert "ripple_shuffle(" not in result


def test_butterfly_allreduce_does_not_fire_on_partial_width_loop():
    # CRITICAL correctness guard: a loop bounded to fewer than the full
    # warp (here, 8 lanes) is a segmented/sub-group reduction, NOT a
    # whole-block all-reduce. ripple_reduceadd(0b1, ...) always reduces
    # across the entire dimension, so firing here would silently change
    # the kernel's meaning (every lane would get the sum of all 32
    # lanes instead of just its own group of 8). Must fall through to
    # UnrollConstantShuffleLoopRule instead, which unrolls it correctly
    # (3 iterations: i = 1, 2, 4) without attaching any
    # reduction-specific meaning to the loop.
    source = """
__global__ void segmentedReduce(float *data) {
    float val = data[threadIdx.x];
    for (int i = 1; i < 8; i *= 2) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    assert "ripple_reduceadd" not in result
    assert result.count("ripple_shuffle(") == 3  # i = 1, 2, 4


def test_butterfly_allreduce_does_not_fire_on_non_full_mask():
    # CRITICAL correctness guard: a non-full mask means not every lane
    # participates in the shuffle, so this is not provably a whole-warp
    # all-reduce even though the loop shape otherwise matches exactly.
    # Must fall through to UnrollConstantShuffleLoopRule rather than
    # assume full participation.
    source = """
__global__ void partialMaskReduce(float *data) {
    float val = data[threadIdx.x];
    for (int i = 1; i < 32; i *= 2) {
        val += __shfl_xor_sync(0x0000ffff, val, i);
    }
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    assert "ripple_reduceadd" not in result
    assert result.count("ripple_shuffle(") == 5  # i = 1, 2, 4, 8, 16


def test_butterfly_allreduce_does_not_collide_with_warp_reduction_rule():
    # Regression guard: the classic warpSize/2 + shfl_down + accumulate
    # shape must still go through WarpReductionRule unchanged — the two
    # rules match mutually exclusive loop shapes (init=1-doubling+xor
    # vs init=warpSize/2-halving+down), so there should be no overlap.
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
    assert "ripple_shuffle(" not in result
```

You'll need `Path` imported in the test file if it isn't already — check the existing imports at the top of `tests/test_translation.py` first; if `from pathlib import Path` is already there (it's commonly needed for fixture-loading tests), don't add a duplicate import line.

- [ ] **Step 2: Run to confirm they fail**

Run: `source venv/bin/activate && python -m pytest tests/test_translation.py -k "butterfly_allreduce" -v`
Expected: `test_butterfly_allreduce_recognized_as_single_reduceadd`, `test_butterfly_allreduce_fires_on_real_fixture`, and `test_butterfly_allreduce_accepts_symbolic_warpsize_bound` FAIL (`ripple_reduceadd` absent, rule doesn't exist yet — output still has 5 `ripple_shuffle` calls from `UnrollConstantShuffleLoopRule`). `test_butterfly_allreduce_does_not_fire_on_partial_width_loop`, `test_butterfly_allreduce_does_not_fire_on_non_full_mask`, and `test_butterfly_allreduce_does_not_collide_with_warp_reduction_rule` PASS already (nothing to wrongly fire on yet, so trivially true) — that's fine, they're regression guards for *after* the rule exists, not proof it doesn't exist yet.

- [ ] **Step 3: Implement the rule**

In `core/translation_rules.py`, add this class immediately after `class WarpReductionRule` (i.e., right before `class UnrollConstantShuffleLoopRule` — grouping it there matches this file's "related rules sit near each other" convention, since it's the `shfl_xor` sibling of `WarpReductionRule`'s `shfl_down` optimization):

```python
class ButterflyAllReduceRule(TranslationRule):
    """
    Recognizes the classic full-warp XOR-butterfly all-reduce loop and
    replaces it with a single ripple_reduceadd call, the same
    optimization WarpReductionRule already applies to the shfl_down-
    halving-reduction variant of this same idea (see that class's
    docstring). Both patterns compute a whole-warp reduction; this one
    additionally leaves every lane holding the total (an all-reduce),
    which is exactly what ripple_reduceadd's documented auto-broadcast-
    on-reassignment semantics already produce for free (api.md: "automatic
    broadcast promotes lower-dimensional blocks... to higher-dimensional
    ones") — no explicit ripple_broadcast call is needed.

    Matches only:
      for (int VAR = 1; VAR < BOUND; VAR *= 2) { ACCUM += __shfl_xor_sync(MASK, ACCUM, VAR); }
    where BOUND is the literal `32` or the symbolic token `warpSize`
    (mirroring WarpReductionRule's existing assumption that `warpSize`
    always means "the whole warp, dimension 0"), and MASK must be
    exactly the literal full mask `0xffffffff` (case-insensitive).

    Both constraints are deliberate correctness guards, not style
    preferences:
      - A BOUND smaller than the full warp (e.g. `i < 8`) is a
        segmented/sub-group reduction. ripple_reduceadd(0b1, ...)
        always reduces across the ENTIRE dimension, so firing here
        would silently change the kernel's meaning — every lane would
        get the sum of all 32 lanes instead of just its own group.
      - A non-full MASK means not every lane participates in the
        shuffle, so the loop shape alone doesn't prove this is a
        whole-warp all-reduce.
    Either mismatch is a silent, semantics-changing bug if ignored, not
    a merely-suboptimal one — so on either mismatch this rule declines
    entirely (leaves the loop untouched) rather than firing anyway.
    The loop then falls through to UnrollConstantShuffleLoopRule, which
    is still correct for these cases (it attaches no reduction-specific
    meaning to the loop, just literal substitution), just not optimal.
    """

    PATTERN = (
        r'for\s*\(\s*int\s+(\w+)\s*=\s*1\s*;\s*'
        r'\1\s*<\s*(32|warpSize)\s*;\s*'
        r'\1\s*\*=\s*2\s*\)\s*\{\s*'
        r'(\w+)\s*\+=\s*__shfl_xor_sync\s*\(\s*([^,]+?)\s*,\s*\3\s*,\s*\1\s*\)\s*;\s*\}'
    )

    FULL_MASK = "0xffffffff"

    def __init__(self):
        super().__init__(
            name="butterfly_allreduce",
            description="Recognize full-warp XOR-butterfly all-reduce loops and replace with ripple_reduceadd",
            cuda_pattern=self.PATTERN,
            priority=86  # Distinct from WarpReductionRule's 85 to avoid
                         # relying on stable-sort tie-breaking; doesn't
                         # matter for correctness since the two patterns
                         # are mutually exclusive (init=1-doubling+xor
                         # vs init=warpSize/2-halving+down). Above
                         # UnrollConstantShuffleLoopRule (82) so this
                         # rule gets first look at the loop.
        )

    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            accum_var = match.group(3)
            mask = match.group(4).strip()

            if mask.lower() != self.FULL_MASK:
                # Not a full-mask reduction — can't safely assume every
                # lane participates, so this isn't provably a whole-warp
                # all-reduce. Leave the loop for UnrollConstantShuffleLoopRule.
                return match.group(0)

            ctx.add_warning(
                f"Recognized full-warp XOR-butterfly all-reduce loop over "
                f"'{accum_var}' and replaced it with a single "
                f"ripple_reduceadd call, matching WarpReductionRule's "
                f"established idiom for the shfl_down variant of this "
                f"same pattern."
            )

            return f"""/* CUDA XOR-Butterfly All-Reduce -> RIPPLE Intrinsic */
    {accum_var} = ripple_reduceadd(0b1, {accum_var});"""

        return re.sub(self.PATTERN, replace, cuda_code)
```

- [ ] **Step 4: Register the rule**

In `core/translation_rules.py`, in `TranslationRuleEngine._register_default_rules()`, in the `default_rules` list, add `ButterflyAllReduceRule(),` immediately after `WarpReductionRule(),` and before `UnrollConstantShuffleLoopRule(),` (list position doesn't affect execution order — `register_rule()` re-sorts by `priority` on every insert — but grouping it there matches the existing convention):

```python
            # Shuffles
            WarpReductionRule(),
            ButterflyAllReduceRule(),
            UnrollConstantShuffleLoopRule(),
            ShuffleDownRule(),
            ShuffleUpRule(),
            ShuffleXorRule(),
            ShuffleSyncRule(),
```

- [ ] **Step 5: Run all the tests from Step 1**

Run: `source venv/bin/activate && python -m pytest tests/test_translation.py -k "butterfly_allreduce" -v`
Expected: all 6 PASS.

- [ ] **Step 6: Run the full suite**

Run: `source venv/bin/activate && python -m pytest tests/ -v`
Expected: all previous tests still pass, plus these 6 new ones — 116 passed, 0 failed (110 prior + 6 new). If the count differs, investigate before proceeding — in particular, re-check that no *other* existing test asserted the old "butterfly_reduction.cu unrolls into 5 ripple_shuffle calls" behavior as part of some other test's incidental assertion (search for `ripple_shuffle` and `butterfly` together across the test file) — this plan's own research found none, but confirm it yourself rather than trusting that research.

- [ ] **Step 7: Manually reproduce the exact motivating example end-to-end via the CLI**

Run:
```bash
cat > /tmp/butterfly_allreduce.cu <<'EOF'
__global__ void butterflyReduce(float *data) {
    float val = data[threadIdx.x];
    for (int i = 1; i < 32; i *= 2) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    data[threadIdx.x] = val;
}
EOF
python -m interfaces.cli.cuda2ripple source /tmp/butterfly_allreduce.cu -o /tmp/butterfly_allreduce.ripple.c
cat /tmp/butterfly_allreduce.ripple.c
```
Expected: exit code 0, no `[Error]`, an informational "Recognized full-warp XOR-butterfly all-reduce..." warning is fine and expected, and the output contains `val = ripple_reduceadd(0b1, val);` with NO hoisted `__ripple_shfl_xor_N` functions and NO `ripple_shuffle(` calls at all. Then run it through the syntax checker to confirm it's real, compilable C:
```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from tests.compile_verify import verify_ripple_syntax
print(verify_ripple_syntax(open('/tmp/butterfly_allreduce.ripple.c').read()))
"
```
Expected: `(True, '')`.

Then confirm the partial-width guard live, via the CLI, with a segmented (8-lane) version of the same loop:
```bash
cat > /tmp/segmented_reduce.cu <<'EOF'
__global__ void segmentedReduce(float *data) {
    float val = data[threadIdx.x];
    for (int i = 1; i < 8; i *= 2) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    data[threadIdx.x] = val;
}
EOF
python -m interfaces.cli.cuda2ripple source /tmp/segmented_reduce.cu -o /tmp/segmented_reduce.ripple.c
cat /tmp/segmented_reduce.ripple.c
```
Expected: exit code 0 (still translates successfully, just via unrolling, not this new rule), output contains 3 `ripple_shuffle(` calls and hoisted `__ripple_shfl_xor_N` functions with literal values 1, 2, 4 — and does NOT contain `ripple_reduceadd`, confirming the correctness guard held live, not just under pytest.

- [ ] **Step 8: Commit**

```bash
git add core/translation_rules.py tests/test_translation.py
git commit -m "Recognize full-warp XOR-butterfly all-reduce loops as ripple_reduceadd

UnrollConstantShuffleLoopRule correctly translates the classic
butterfly all-reduce idiom (for (int i = 1; i < 32; i *= 2) { val +=
__shfl_xor_sync(mask, val, i); }) by unrolling it into 5 separate
hoisted permutation functions and ripple_shuffle calls. That's
correct but not optimal: RIPPLE has a native reduceadd intrinsic for
exactly this shape, and WarpReductionRule already recognizes the
shfl_down-halving variant of the same idea and replaces it with a
single ripple_reduceadd call. This adds the shfl_xor-doubling sibling
of that same optimization, at a priority above
UnrollConstantShuffleLoopRule so it gets first look at the loop.

Two correctness guards, both required because getting either wrong
would silently change the kernel's meaning rather than just miss an
optimization: the loop bound must be the literal 32 or symbolic
warpSize (a smaller bound is a segmented/sub-group reduction, not a
whole-block all-reduce — ripple_reduceadd(0b1, ...) always reduces
across the entire dimension), and the shuffle's mask must be exactly
the full mask 0xffffffff (a partial mask means not every lane
participates, so the loop shape alone doesn't prove whole-warp
participation). Either mismatch makes the rule decline silently and
fall through to the existing, still-correct unroll-or-hard-fail
pipeline.

Verified end-to-end via the CLI: the exact butterfly-reduction example
now translates to a single ripple_reduceadd call with zero hoisted
shuffle functions, passes a real clang syntax check, and a
segmented (partial-width) variant of the same loop correctly falls
through to unrolling instead of firing this rule."
```

---

## Self-Review

**Spec coverage:** the reduce+broadcast abstraction-shift idea discussed in conversation → this plan's single task, implementing it as `ripple_reduceadd` alone (no explicit `ripple_broadcast` call needed, per RIPPLE's documented auto-broadcast-on-reassignment semantics, matching `WarpReductionRule`'s existing working idiom). The two correctness guards (full-width bound, full mask) discussed as the key risk are both implemented as explicit checks with dedicated regression tests, not left as unstated assumptions. ✓

**Placeholder scan:** no TBD/TODO/"add appropriate handling" phrasing; every code step is complete, runnable code; every "Run:" step has a concrete command and expected count.

**Type/name consistency:** `ButterflyAllReduceRule` is defined once (Task 1, Step 3) and registered once, by that exact name, in `_register_default_rules()` (Step 4). No other file references this class, so there's nothing else to keep in sync.

**Expected test counts, traced cumulatively:** baseline before this plan = 110 passed (confirmed live in the worktree before writing this plan, not assumed from the prior plan's stale numbers). This task adds 6 tests → 116.

**Dependency ordering:** none — this is a single, self-contained task that only adds a new rule and its tests; it doesn't modify any existing rule's behavior for cases it doesn't match (verified via the "does not fire" / "does not collide" regression tests in Step 1).
