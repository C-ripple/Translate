# Shuffle Loop Unrolling + Hard-Fail on Unresolvable Arguments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Handle the common case of GitHub issue #11 (variable-argument warp shuffles) by auto-unrolling small, compile-time-bounded counting loops that feed a shuffle intrinsic — turning the classic CUDA butterfly-reduction idiom into something the already-working literal-argument shuffle path can translate cleanly. Then, for whatever's left that genuinely can't be resolved (a truly runtime-determined shift amount), stop silently succeeding with broken output embedded in a warning comment — raise a real exception instead, so a caller (CLI, web server, library user) can't miss it.

**Architecture:** Two pieces, built in order because the second only makes sense once the first exists. Piece 1 is a new `TranslationRule` (`UnrollConstantShuffleLoopRule`) that runs before the shuffle rules and expands eligible loops into literal-substituted copies of their body. Piece 2 activates `TranslationContext`'s already-defined-but-completely-unused `errors`/`has_errors()` fields — currently every warning-emitting rule in this codebase calls `add_warning()`, nothing calls `add_error()` — by making the 4 shuffle rules call `add_error()` instead of `add_warning()` when a shuffle argument still can't be resolved after unrolling has had its chance, and making `CUDAToRIPPLETransformer.transform()` raise a new `TranslationError` if any error was recorded. Every existing call site (CLI, web server) already has a generic `except Exception` handler that reports failure correctly — verified by reading them, not assumed — so this requires zero changes to those interfaces to get correct fail-loud behavior.

**Tech Stack:** Python 3.13, existing translator codebase, pytest.

---

## Before you start

Both pieces are grounded in things verified directly in this session, not assumed:

- `TranslationRuleEngine.register_rule()` sorts `self.rules` by `priority` (descending) on every insert (`core/translation_rules.py`) — so a new rule's execution order relative to existing rules is controlled entirely by its `priority` value, not where it's added to the registration list. Confirmed by reading `register_rule()` directly, not inferred from the `priority=` kwargs alone (which, on their own, don't prove anything about execution order — only reading the sort call does).
- `WarpReductionRule` (priority 85) already owns the `warpSize/2`-halving + `__shfl_down_sync` + `+=`-accumulate pattern, converting it to a single `ripple_reduceadd` call — better than unrolling would produce for that specific shape. The new rule in this plan requires its loop's init/bound to be plain integer literals (`\d+`), which `warpSize/2` is not (it's a symbolic expression), so there is no collision — confirmed by reading `WarpReductionRule`'s actual regex, not assumed from its description.
- `TranslationContext.errors`/`add_error()`/`has_errors()` exist in `core/semantic_model.py` today but are called from nowhere in the codebase — confirmed via `grep -rn "add_error\|has_errors" --include="*.py" .` returning only the definition itself.
- The CLI's `cmd_source` (`interfaces/cli/cuda2ripple.py`) and the web server's `/translate` route (`server.py`) both already wrap their call to the transformer in `try: ... except Exception as e: ...` and report failure appropriately (non-zero exit / HTTP 500) — confirmed by reading both files directly. A raised `TranslationError` needs no changes to either to produce correct fail-loud behavior; this plan verifies that live in Task 4 rather than trusting the reasoning alone.

---

## Priority 1: Auto-unroll compile-time-bounded shuffle loops

### File Structure

- Modify: `core/translation_rules.py` — new `UnrollConstantShuffleLoopRule` class, registered in `TranslationRuleEngine._register_default_rules()`
- Modify: `tests/test_translation.py` — new tests for the unrolling behavior

### Task 1: Implement and test `UnrollConstantShuffleLoopRule`

**Files:**
- Modify: `core/translation_rules.py`
- Modify: `tests/test_translation.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_translation.py`:

```python
def test_unroll_doubling_loop_resolves_xor_shuffle():
    # The exact butterfly-reduction example demonstrated in conversation:
    # i doubles from 1 to 16 (5 iterations), feeding __shfl_xor_sync's
    # lane_mask argument — previously left untranslated with a warning,
    # since 'i' is a kernel-local loop variable, not a compile-time
    # constant, from the shuffle rule's point of view in isolation.
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
    assert "__shfl_xor_sync" not in result
    assert result.count("ripple_shuffle(") == 5  # i = 1, 2, 4, 8, 16


def test_unroll_halving_loop_resolves_down_shuffle():
    # Same mechanism, opposite direction (halving instead of doubling),
    # and a different shuffle intrinsic — proves the rule is shape-based
    # (any counting loop with literal bounds), not hardcoded to one
    # specific loop direction or intrinsic. Deliberately NOT the
    # warpSize/2 + __shfl_down_sync + accumulate shape WarpReductionRule
    # already owns (that rule requires the symbolic literal "warpSize",
    # not a plain digit, so it can't fire here) — this uses a literal
    # loop bound and a non-accumulating shuffle_up to stay clear of it.
    source = """
__global__ void haloExchange(float *data) {
    float val = data[threadIdx.x];
    for (int offset = 16; offset > 0; offset /= 2) {
        val = __shfl_up_sync(0xffffffff, val, offset);
    }
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    assert "__shfl_up_sync" not in result
    assert result.count("ripple_shuffle(") == 5  # offset = 16, 8, 4, 2, 1


def test_unroll_produces_distinct_literal_values_per_iteration():
    # Not just "5 calls exist" — each one must actually reference the
    # right literal value, not the same one 5 times or a mangled one.
    source = """
__global__ void butterflyReduce(float *data) {
    float val = data[threadIdx.x];
    for (int i = 1; i < 8; i *= 2) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    import re
    hoisted_bodies = re.findall(r'return k \^ \((\d+)\);', result)
    assert sorted(int(v) for v in hoisted_bodies) == [1, 2, 4]


def test_unroll_does_not_fire_on_unrelated_countable_loop():
    # A loop with literal bounds but no shuffle call in its body should
    # be left completely alone — unrolling is not a general-purpose
    # optimization this translator applies proactively, only a targeted
    # fix for the one thing that would otherwise be untranslatable.
    source = """
__global__ void sumLoop(int *data) {
    int total = 0;
    for (int i = 0; i < 4; i += 1) {
        total += data[i];
    }
    data[0] = total;
}
"""
    result = translate_cuda_source(source)
    assert "for (int i = 0; i < 4; i += 1)" in result or "for(int i = 0; i < 4; i += 1)" in result.replace(" ", "").replace("for(", "for (")


def test_unroll_does_not_fire_when_warp_reduction_rule_already_owns_the_pattern():
    # Regression guard: the classic warpSize/2 + __shfl_down_sync +
    # accumulate shape must still go through WarpReductionRule's single
    # ripple_reduceadd(0b1, ...) call, not get unrolled into 5 separate
    # ripple_shuffle calls — unrolling only fires on loops it doesn't
    # already recognize (literal, not symbolic, bounds).
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

- [ ] **Step 2: Run to confirm they fail**

Run: `source venv/bin/activate && python -m pytest tests/test_translation.py -k "test_unroll" -v`
Expected: `test_unroll_doubling_loop_resolves_xor_shuffle`, `test_unroll_halving_loop_resolves_down_shuffle`, `test_unroll_produces_distinct_literal_values_per_iteration` FAIL (no unrolling exists yet, `__shfl_xor_sync`/`__shfl_up_sync` still present in output). `test_unroll_does_not_fire_on_unrelated_countable_loop` and `test_unroll_does_not_fire_when_warp_reduction_rule_already_owns_the_pattern` PASS already (nothing to unroll yet, so trivially true) — that's fine, they're regression guards for *after* the rule exists, not proof it doesn't exist yet.

- [ ] **Step 3: Implement the rule**

In `core/translation_rules.py`, add this class immediately before `class ShuffleDownRule` (i.e., right after `_is_compile_time_constant_expr`, before the first shuffle rule — it needs to run before them, and grouping it there matches the file's existing "related rules sit near each other" convention):

```python
class UnrollConstantShuffleLoopRule(TranslationRule):
    """
    Unrolls a small, compile-time-bounded counting loop whose body
    contains a warp-shuffle call using the loop's induction variable —
    substituting each literal value the variable takes, so the
    already-correct literal-argument shuffle rules can hoist a real
    permutation function for each, instead of the whole loop being left
    as an untranslatable variable-argument shuffle (see
    _is_compile_time_constant_expr's docstring, GitHub issue #11).

    Deliberately narrow, matching only:
      for (int VAR = INIT; VAR OP BOUND; VAR OP= STEP) { BODY }
    where INIT/BOUND/STEP are plain integer literals (not symbolic
    constants like `warpSize` — WarpReductionRule already owns that
    specific shape via a single ripple_reduceadd call, which is better
    than unrolling for it, and this rule's literal-only pattern can't
    match a symbolic initializer at all, so there's no collision) and
    BODY contains no nested braces (no if/for/while inside — multiple
    simple statements are fine, control flow is not, since substituting
    a loop variable's value into arbitrary nested logic safely is a much
    larger problem than this rule is scoped to solve).

    Only fires when BODY actually references the loop variable inside a
    shuffle call — an unrelated countable loop is left as a loop, since
    this is a targeted unblocking mechanism, not a general "unroll every
    eligible loop" optimization pass.
    """

    PATTERN = (
        r'for\s*\(\s*int\s+(\w+)\s*=\s*(\d+)\s*;\s*'
        r'\1\s*(<=|>=|<|>)\s*(\d+)\s*;\s*'
        r'\1\s*(\*=|/=|\+=|-=)\s*(\d+)\s*\)\s*'
        r'\{([^{}]*)\}'
    )

    MAX_ITERATIONS = 64

    def __init__(self):
        super().__init__(
            name="unroll_constant_shuffle_loop",
            description="Unroll small compile-time-bounded loops feeding a shuffle intrinsic",
            cuda_pattern=self.PATTERN,
            priority=82  # below WarpReductionRule (85), above shuffle rules (70)
        )

    @staticmethod
    def _compute_unroll_values(init, cond_op, bound, step_op, step, max_iterations):
        """
        Returns the list of literal values the loop variable takes, or
        None if the loop can't be safely/finitely unrolled (step doesn't
        make progress toward the bound, or would take more than
        max_iterations steps — either way, leave the original loop text
        untouched rather than guess).
        """
        current = init
        values = []

        def condition_holds(v):
            if cond_op == '<':
                return v < bound
            if cond_op == '<=':
                return v <= bound
            if cond_op == '>':
                return v > bound
            if cond_op == '>=':
                return v >= bound
            return False

        def apply_step(v):
            if step_op == '*=':
                return v * step
            if step_op == '/=':
                return v // step if step != 0 else None
            if step_op == '+=':
                return v + step
            if step_op == '-=':
                return v - step
            return None

        while condition_holds(current):
            values.append(current)
            if len(values) > max_iterations:
                return None
            next_val = apply_step(current)
            if next_val is None or next_val == current:
                return None
            current = next_val

        return values or None

    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            var_name = match.group(1)
            init = int(match.group(2))
            cond_op = match.group(3)
            bound = int(match.group(4))
            step_op = match.group(5)
            step = int(match.group(6))
            body = match.group(7)

            if not re.search(r'__shfl(?:_(?:down|up|xor))?_sync\s*\(', body):
                return match.group(0)
            if not re.search(rf'\b{re.escape(var_name)}\b', body):
                return match.group(0)

            values = self._compute_unroll_values(
                init, cond_op, bound, step_op, step, self.MAX_ITERATIONS
            )
            if values is None:
                # Not safely/finitely unrollable — leave the loop as-is.
                # The shuffle rule underneath will still see the intact
                # variable reference and correctly refuse to translate it.
                return match.group(0)

            unrolled_statements = []
            for v in values:
                substituted = re.sub(rf'\b{re.escape(var_name)}\b', str(v), body)
                unrolled_statements.append(substituted.strip())

            ctx.add_warning(
                f"Unrolled loop over '{var_name}' into {len(values)} "
                f"iterations ({', '.join(str(v) for v in values)}) to "
                f"resolve a compile-time-constant shuffle argument"
            )

            joined = "\n    ".join(unrolled_statements)
            values_str = ', '.join(str(v) for v in values)
            return f"/* Unrolled: {var_name} = {values_str} */\n    {joined}"

        return re.sub(self.PATTERN, replace, cuda_code, flags=re.DOTALL)
```

- [ ] **Step 4: Register the rule**

In `core/translation_rules.py`, in `TranslationRuleEngine._register_default_rules()`, in the `default_rules` list, add `UnrollConstantShuffleLoopRule(),` immediately after `WarpReductionRule(),` (list position doesn't affect execution order — `register_rule()` re-sorts by `priority` on every insert — but grouping it there matches the existing "related rules near each other" convention in this list):

```python
            # Shuffles
            WarpReductionRule(),
            UnrollConstantShuffleLoopRule(),
            ShuffleDownRule(),
            ShuffleUpRule(),
            ShuffleXorRule(),
            ShuffleSyncRule(),
```

- [ ] **Step 5: Run all the tests from Step 1**

Run: `source venv/bin/activate && python -m pytest tests/test_translation.py -k "test_unroll" -v`
Expected: all 5 PASS.

- [ ] **Step 6: Run the full suite**

Run: `source venv/bin/activate && python -m pytest tests/ -v`
Expected: all previous tests still pass, plus these 5 new ones — 91 passed, 0 failed, 0 xfailed (86 prior + 5 new).

- [ ] **Step 7: Manually reproduce the exact conversation example end-to-end via the CLI**

Run:
```bash
cat > /tmp/butterfly_reduce.cu <<'EOF'
__global__ void butterflyReduce(float *data) {
    float val = data[threadIdx.x];
    for (int i = 1; i < 32; i *= 2) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    data[threadIdx.x] = val;
}
EOF
python -m interfaces.cli.cuda2ripple source /tmp/butterfly_reduce.cu -o /tmp/butterfly_reduce.ripple.c
cat /tmp/butterfly_reduce.ripple.c
```
Expected: no `[Warnings]` about a non-constant shuffle argument (an informational "Unrolled loop..." warning is fine and expected); the output file has 5 hoisted `__ripple_shfl_xor_N` functions with literal values `1, 2, 4, 8, 16` in their bodies, and 5 corresponding `ripple_shuffle(val, __ripple_shfl_xor_N)` calls where the original loop used to be. Then run it through the syntax check to confirm it's real, compilable C:
```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from tests.compile_verify import verify_ripple_syntax
print(verify_ripple_syntax(open('/tmp/butterfly_reduce.ripple.c').read()))
"
```
Expected: `(True, '')`.

- [ ] **Step 8: Commit**

```bash
git add core/translation_rules.py tests/test_translation.py
git commit -m "Auto-unroll compile-time-bounded loops feeding a shuffle intrinsic

Handles the common case of GitHub issue #11 (variable-argument warp
shuffles): a loop like 'for (int i = 1; i < 32; i *= 2)' feeding
__shfl_xor_sync's lane_mask has a small, fully-knowable set of values
(1, 2, 4, 8, 16) even though 'i' itself is a kernel-local variable —
so unroll the loop into 5 literal-argument copies before the shuffle
rules ever see it, letting the already-correct literal-argument path
hoist a real permutation function for each.

Deliberately narrow: only literal integer bounds/step (not symbolic
constants like warpSize — WarpReductionRule already owns that shape
via a single ripple_reduceadd call), no nested braces in the loop
body, and only fires when the body actually uses the loop variable in
a shuffle call. Anything outside that shape is left untouched, same
as before this change.

Verified end-to-end via the CLI: the exact butterfly-reduction example
translates clean with no shuffle warnings and passes a real clang
syntax check."
```

---

## Priority 2: Hard-fail on genuinely unresolvable shuffle arguments

### File Structure

- Modify: `core/semantic_model.py` — new `TranslationError` exception
- Modify: `frontends/source/cuda_frontend.py` — `transform()` raises when `ctx.has_errors()`
- Modify: `core/translation_rules.py` — the 4 shuffle rules call `ctx.add_error()` instead of `ctx.add_warning()` on the non-constant-argument path
- Modify: `tests/test_translation.py` — update the existing tests that currently expect a warning-and-continue for this exact case, since that's no longer the behavior

### Task 2: Add `TranslationError` and wire it into `transform()`

**Files:**
- Modify: `core/semantic_model.py`
- Modify: `frontends/source/cuda_frontend.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_translation.py`:

```python
def test_transform_raises_when_ctx_has_errors():
    ctx = TranslationContext()
    ctx.add_error("synthetic test error")
    transformer = CUDAToRIPPLETransformer(ctx)
    with pytest.raises(TranslationError) as exc_info:
        transformer.transform("__global__ void kernel() {}")
    assert "synthetic test error" in str(exc_info.value)
```

You'll need `TranslationError` imported — add it to wherever `TranslationContext` is already imported from `core.semantic_model` at the top of the file.

- [ ] **Step 2: Run to confirm it fails**

Run: `source venv/bin/activate && python -m pytest tests/test_translation.py::test_transform_raises_when_ctx_has_errors -v`
Expected: FAIL — either an `ImportError` (no `TranslationError` yet) or the transform completes without raising.

- [ ] **Step 3: Add the exception class**

In `core/semantic_model.py`, add this near `TranslationContext` (immediately before or after the class — check the file's existing convention for where related small classes sit relative to `TranslationContext` and match it):

```python
class TranslationError(Exception):
    """
    Raised by CUDAToRIPPLETransformer.transform() when translation
    cannot produce valid output — i.e. ctx.has_errors() is True after
    all rules have run. Carries the full list of errors (ctx.errors),
    not just the first one, since multiple independent constructs in
    the same file can each fail for their own reason.

    This is a deliberate design choice: a translator that silently
    writes an output file containing untranslated, non-compiling
    fragments (with the only signal being a warning a caller can miss)
    is worse than one that fails loudly. Every existing entry point —
    the CLI, the web server — already wraps its call to the transformer
    in a generic except Exception handler and reports failure
    correctly, so raising here requires no changes to those callers.
    """

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        message = "Translation failed:\n" + "\n".join(f"  - {e}" for e in self.errors)
        super().__init__(message)
```

- [ ] **Step 4: Wire it into `transform()`**

In `frontends/source/cuda_frontend.py`, change:

```python
        # Phase 4: Post-process and format
        result = self._postprocess(result)
        
        return result
```

to:

```python
        # Phase 4: Post-process and format
        result = self._postprocess(result)
        
        if self.ctx.has_errors():
            raise TranslationError(self.ctx.errors)
        
        return result
```

You'll need `TranslationError` imported into this file from `core.semantic_model` — add it to the existing import line, don't add a new one.

- [ ] **Step 5: Run the test from Step 1**

Run: `source venv/bin/activate && python -m pytest tests/test_translation.py::test_transform_raises_when_ctx_has_errors -v`
Expected: PASSED.

- [ ] **Step 6: Run the full suite**

Run: `source venv/bin/activate && python -m pytest tests/ -v`
Expected: 92 passed, 0 failed, 0 xfailed (91 from Task 1 + this 1 new test). Nothing else should change yet — no existing rule calls `add_error()` yet, so no existing translation should start raising at this point.

- [ ] **Step 7: Commit**

```bash
git add core/semantic_model.py frontends/source/cuda_frontend.py tests/test_translation.py
git commit -m "Add TranslationError, raised by transform() when ctx.has_errors()

TranslationContext.errors/add_error()/has_errors() have existed since
this file's original version but were never called from anywhere in
the codebase — every rule that couldn't fully translate something
called add_warning() instead, meaning the CLI and web server always
reported success (exit 0 / HTTP 200) even when the output file
contained untranslated, non-compiling fragments.

This wires has_errors() up to something real: transform() now raises
TranslationError, carrying the full list of errors, if any rule
recorded one. No existing rule calls add_error() yet — this task only
adds the mechanism; the next task connects it to the shuffle rules'
unresolvable-argument case specifically.

Verified both the CLI (interfaces/cli/cuda2ripple.py) and the web
server (server.py) already wrap the transform() call in a generic
except Exception handler and report failure correctly — no changes
needed to either to get correct fail-loud behavior from this."
```

### Task 3: Convert the shuffle rules' unresolvable-argument path to a hard error

**Files:**
- Modify: `core/translation_rules.py` — `ShuffleDownRule`, `ShuffleXorRule`, `ShuffleUpRule`, `ShuffleSyncRule`
- Modify: `tests/test_translation.py` — update the 3 existing tests that currently expect warn-and-continue for this exact case

**Why now, not earlier:** doing this before Priority 1 existed would have made the common butterfly/reduction idiom a hard failure too, which is a worse regression than today's silent-warning behavior. With unrolling in place, only the genuinely unresolvable cases (a real runtime value, or a loop shape too complex to unroll) reach this path.

- [ ] **Step 1: Update the existing tests that this change makes obsolete**

These three tests in `tests/test_translation.py` currently assert `ctx.warnings` contains a message and the original CUDA call remains in the output — that's the *old* behavior this task removes. Find each one (search for `left __shfl` or `not a compile-time constant` or the test names below) and replace them:

Find `test_shuffle_with_variable_argument_is_left_untranslated_with_warning` and replace it with:

```python
def test_shuffle_with_variable_argument_raises():
    # A shuffle argument that's a kernel-local variable, and NOT the
    # induction variable of an unrollable loop (it's a plain local
    # assigned a runtime-looking value here, not looping at all) — this
    # is exactly the case UnrollConstantShuffleLoopRule can't help with,
    # so it must now fail loudly rather than silently leave broken
    # output.
    source = """
__global__ void kernel(int *a, int n) {
    int val = a[0];
    int result = __shfl_xor_sync(0xffffffff, val, n);
    a[0] = result;
}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    with pytest.raises(TranslationError) as exc_info:
        transformer.transform(source)
    assert "not a compile-time constant" in str(exc_info.value)
    assert ctx.has_errors()
```

Find `test_shuffle_variable_argument_left_untranslated_across_all_rules` (the parametrized one) and replace it with:

```python
@pytest.mark.parametrize("intrinsic", [
    "__shfl_xor_sync(0xffffffff, val, {arg})",
    "__shfl_up_sync(0xffffffff, val, {arg})",
    "__shfl_sync(0xffffffff, val, {arg})",
    "__shfl_down_sync(0xffffffff, val, {arg})",
])
def test_shuffle_variable_argument_raises_across_all_rules(intrinsic):
    call = intrinsic.format(arg="n")
    source = f"""
__global__ void kernel(int *a, int n) {{
    int val = a[0];
    int result = {call};
    a[0] = result;
}}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    with pytest.raises(TranslationError) as exc_info:
        transformer.transform(source)
    assert "not a compile-time constant" in str(exc_info.value)
```

Find `test_shuffle_with_parenthesized_constant_argument_left_untranslated_with_warning` and replace it with:

```python
def test_shuffle_with_parenthesized_constant_argument_raises():
    # The paren-truncation case (see _is_compile_time_constant_expr's
    # docstring) — still must not silently emit malformed C. Now it
    # raises instead of warning-and-continuing.
    source = """
__global__ void kernel(int *a) {
    int val = a[0];
    int result = __shfl_xor_sync(0xffffffff, val, (1));
    a[0] = result;
}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    with pytest.raises(TranslationError):
        transformer.transform(source)
```

Leave `test_shuffle_with_literal_argument_still_hoists_correctly` and `test_shuffle_literal_argument_hoists_across_all_rules` exactly as they are — those cover the still-working, unaffected case.

- [ ] **Step 2: Run to confirm the updated tests fail against the current (unchanged) rule code**

Run: `source venv/bin/activate && python -m pytest tests/test_translation.py -k "raises" -v`
Expected: the 3 updated tests FAIL (rules still call `add_warning`, so `transform()` doesn't raise yet).

- [ ] **Step 3: Update the 4 shuffle rules**

In `core/translation_rules.py`, in each of `ShuffleDownRule`, `ShuffleXorRule`, `ShuffleUpRule`, `ShuffleSyncRule`, change the non-constant-argument branch from `ctx.add_warning(...)` to `ctx.add_error(...)`. The message text itself doesn't need to change — only which list it goes into. For example, in `ShuffleDownRule`:

```python
            if not _is_compile_time_constant_expr(delta):
                ctx.add_warning(
                    f"ShuffleDownRule: left __shfl_down_sync(...) untranslated — "
                    f"delta '{delta}' is not a compile-time constant. "
                    f"ripple_shuffle's permutation function is file-scope and "
                    f"cannot reference kernel-local variables like '{delta}'."
                )
                return match.group(0)
```

becomes:

```python
            if not _is_compile_time_constant_expr(delta):
                ctx.add_error(
                    f"ShuffleDownRule: cannot translate __shfl_down_sync(...) — "
                    f"delta '{delta}' is not a compile-time constant. "
                    f"ripple_shuffle's permutation function is file-scope and "
                    f"cannot reference kernel-local variables like '{delta}'. "
                    f"If '{delta}' is a small, compile-time-bounded loop variable, "
                    f"see GitHub issue #11 for what's supported; otherwise this "
                    f"shuffle cannot currently be translated."
                )
                return match.group(0)
```

Apply the equivalent change to `ShuffleXorRule` (its `lane_mask` variable), `ShuffleUpRule` (its `delta` variable), and `ShuffleSyncRule` (its `src_lane` variable) — same structure, `add_warning` → `add_error`, "left ... untranslated" → "cannot translate", plus the same trailing sentence pointing at issue #11. Keep `return match.group(0)` unchanged in all 4 — the returned text is discarded once `transform()` raises, but leaving the original call intact is still the right thing to do if anything ever inspects partial state.

- [ ] **Step 4: Run the tests from Step 1 again**

Run: `source venv/bin/activate && python -m pytest tests/test_translation.py -k "raises" -v`
Expected: all pass now.

- [ ] **Step 5: Run the full suite**

Run: `source venv/bin/activate && python -m pytest tests/ -v`
Expected: 92 passed, 0 failed, 0 xfailed (same count as after Task 2 — this task replaced 3 old tests with 3 new ones covering the same call sites, no net change in count; the "across_all_rules" parametrized test still contributes 4 individual cases the same as before).

- [ ] **Step 6: Commit**

```bash
git add core/translation_rules.py tests/test_translation.py
git commit -m "Shuffle rules raise TranslationError instead of warn-and-continue

The 4 shuffle rules' non-constant-argument path (see
_is_compile_time_constant_expr) now calls ctx.add_error() instead of
ctx.add_warning(). Combined with the prior task wiring
transform() to raise TranslationError when ctx.has_errors(), a
shuffle argument that's genuinely unresolvable (not a literal, and
not the induction variable of a loop UnrollConstantShuffleLoopRule
can unroll) now fails translation outright instead of writing a file
containing an untranslated, non-compiling __shfl_*_sync call with
only a warning comment as the signal.

Updated the 3 existing tests that asserted the old warn-and-continue
behavior for this exact case to assert pytest.raises(TranslationError)
instead — they were testing a behavior this task deliberately changes."
```

### Task 4: Real-kernel fixture + full end-to-end verification

**Files:**
- Create: `tests/examples/butterfly_reduction.cu`
- Modify: `tests/test_real_kernels.py`

- [ ] **Step 1: Add the fixture**

Create `tests/examples/butterfly_reduction.cu`:

```c
__global__ void butterflyReduce(float *data) {
    float val = data[threadIdx.x];
    for (int i = 1; i < 32; i *= 2) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    data[threadIdx.x] = val;
}
```

This is the exact example demonstrated in conversation — a real-world butterfly all-reduce that was, before this plan, a variable-argument shuffle producing an untranslated CUDA call plus a warning easy to miss. It now has to translate completely clean.

- [ ] **Step 2: Wire it into `tests/test_real_kernels.py`**

Add `"butterfly_reduction.cu"` to both `KERNEL_FILES` and `SYNTAX_CHECK_PARAMS` (plain string entries, matching the other 7 — no special-casing needed, since this file is now expected to translate and compile cleanly like everything else in the list).

- [ ] **Step 3: Run the real-kernel tests**

Run: `source venv/bin/activate && python -m pytest tests/test_real_kernels.py -v`
Expected: 16 passed, 0 xfailed (8 structural + 8 syntax-check).

- [ ] **Step 4: Run the full suite**

Run: `source venv/bin/activate && python -m pytest tests/ -v`
Expected: 94 passed, 0 failed, 0 xfailed (92 from Task 3 + 2 new: 1 structural + 1 syntax-check for the new fixture).

- [ ] **Step 5: Live end-to-end verification of the hard-fail path via the actual CLI**

This is the other half of the story — Task 1's Step 7 already proved the unroll-and-succeed path live through the CLI. Now prove the fail-loud path live too, with a genuinely unresolvable case:

```bash
cat > /tmp/unresolvable_shuffle.cu <<'EOF'
__global__ void kernel(int *a, int shift) {
    int val = a[0];
    int result = __shfl_xor_sync(0xffffffff, val, shift);
    a[0] = result;
}
EOF
python -m interfaces.cli.cuda2ripple source /tmp/unresolvable_shuffle.cu -o /tmp/unresolvable_shuffle.ripple.c
echo "exit code: $?"
ls -la /tmp/unresolvable_shuffle.ripple.c 2>&1
```
Expected: a non-zero exit code, an `[Error]` message printed mentioning `shift` and "not a compile-time constant", and — check this specifically — confirm whether `/tmp/unresolvable_shuffle.ripple.c` was written or not. If the CLI writes the file *before* checking for success (look at `cmd_source`'s actual code — the write happens, then warnings are printed, per the code read during this plan's research), a raised exception happening inside `transformer.transform()` would prevent the file write entirely, since the exception propagates before the `with open(args.output, 'w')` block ever runs — confirm this is actually true by checking the file doesn't exist, don't just assume the ordering protects you.

- [ ] **Step 6: Commit**

```bash
git add tests/examples/butterfly_reduction.cu tests/test_real_kernels.py
git commit -m "Add real-kernel coverage for the unroll-and-translate path

butterfly_reduction.cu is the exact example that motivated this whole
change — a standard CUDA butterfly all-reduce that used to translate
'successfully' while silently leaving its core operation as
untranslated CUDA syntax. It now has to translate and compile clean
like every other fixture in this suite, with no special-casing.

Also live-verified via the CLI (not just pytest) that a genuinely
unresolvable shuffle argument now fails loudly — non-zero exit,
clear error message, no half-written output file — closing the loop
on the 'silent warning' problem this plan set out to fix."
```

---

## Self-Review

**Spec coverage:**
- "Implement the compiler logic to auto-unroll fixed loops" → Priority 1, Task 1. ✓
- "Change the tool's behavior to hard-crash with an explicit error instead of printing a silent warning" → Priority 2, Tasks 2-3 (the mechanism, then wiring it to the specific case), verified live via the CLI in Task 4. ✓
- The user's exact demonstrated example (butterfly reduction with `__shfl_xor_sync` in a doubling loop) is the literal content of the new fixture in Task 4 and the first test in Task 1 — not a different, similar-looking example.

**Placeholder scan:** No TBD/TODO/"add appropriate handling" phrasing; every code step is complete, runnable code; every "Run:" step has a concrete command and expected count.

**Type/name consistency:** `UnrollConstantShuffleLoopRule` is defined once (Task 1) and registered once, by that exact name, in `_register_default_rules()`. `TranslationError` is defined once (Task 2, `core/semantic_model.py`) and imported/raised identically in `frontends/source/cuda_frontend.py` and referenced identically in every test that expects it (Tasks 2-3). `ctx.add_error()`/`ctx.has_errors()` are pre-existing, unchanged APIs — Task 2 doesn't redefine them, only starts calling them for real.

**Expected test counts, traced cumulatively:** baseline before this plan = 86 passed, 0 xfailed. Task 1 adds 5 tests → 91. Task 2 adds 1 test → 92. Task 3 replaces 3 old tests with 3 new ones covering the same call sites (net zero change in count, but different assertions) → still 92. Task 4 adds 2 (1 structural + 1 syntax-check for the new fixture) → 94. Each task's own "Expected" lines match this running total.

**Dependency ordering, explicit:** Priority 1 must land before Priority 2 — Task 3 specifically depends on Priority 1 already existing, or converting the shuffle rules to hard-fail would break the common butterfly-reduction case that Priority 1 exists to fix. This is stated in Task 3's own "Why now, not earlier" note, not just implied by section ordering.
