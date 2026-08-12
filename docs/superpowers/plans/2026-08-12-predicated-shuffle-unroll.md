# Predicated Shuffle Unroll Rule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Handle a shuffle loop whose induction variable's value *set* is fully compile-time-known (literal init/bound/step, exactly like `UnrollConstantShuffleLoopRule` already requires) but whose loop condition also carries a second, runtime-dependent early-exit clause (`VAR OP BOUND && VAR OP2 RUNTIME_ID`) — e.g. `for (int offset = 16; offset > 0 && offset > min_offset; offset /= 2)`. Today this falls through both existing rules untouched (neither's pattern matches a compound `&&` condition) and hard-fails, even though the shuffle argument's possible values are just as statically known as the plain case — only *how many* of them actually run is runtime-dependent, not *which* values occur.

**Architecture:** A new sibling rule, `PredicatedShuffleUnrollRule`, registered alongside `UnrollConstantShuffleLoopRule`. It reuses that class's already-reviewed, already-tested static machinery directly (`_resolve_literal`, `_compute_unroll_values`, `UNSAFE_BODY_TOKENS`, `CONTROL_FLOW_PREFIX`) rather than duplicating it, so none of the invisible-corruption bail-outs that class went through 6 hardening rounds to get right have to be re-derived or re-proven here. It unrolls to the *same* literal value set the plain rule would compute from the literal clause alone, but wraps each substituted iteration in an `if (LITERAL_VALUE OP2 RUNTIME_ID)` guard reproducing the original loop's early-exit condition — each guard is independently correct because the induction variable moves monotonically under a literal step, so once a guard evaluates false for a given literal value, every later (smaller-magnitude, in the walk direction) value's guard is false too, matching the original loop's actual behavior of stopping and never resuming.

**Why this is sound, and where the line is drawn:** two guards make this the *only* provably-safe subset of "runtime-bounded shuffle loop," not a general mechanism:
- **Direction consistency** — the runtime clause's comparison operator must be in the same family (`<`/`<=` vs `>`/`>=`) as the literal clause's. This is what guarantees "once false, stays false" — without it, the runtime clause could flip back to true for a later literal value, and independent per-iteration guards would silently produce the wrong set of executed iterations.
- **Runtime identifier untouched in body** — if the runtime-bound variable is reassigned or otherwise referenced inside the loop body, the "fixed threshold for the loop's whole lifetime" assumption breaks, and guarding against a stale/moving value would be unsound. The rule declines rather than trying to prove the body doesn't mutate it.

This deliberately does NOT attempt the broader claim that "runtime-bounded shuffle loops are usually finite" — e.g. `for (int offset = n; offset > 0; offset >>= 1)` where `n` is an unconstrained kernel parameter has no provable finite domain at all (`n` could be arbitrarily large), and guessing one would be exactly the silent-wrongness failure class the rest of this shuffle-handling work exists to eliminate. This rule only fires when the *entire* value domain is already provably static from the literal clause alone — the runtime clause can only shorten the sequence, never change what's in it.

**Tech Stack:** Python 3.13, existing translator codebase, pytest.

---

## Before you start

Grounded in things verified directly in this session, not assumed:

- `UnrollConstantShuffleLoopRule` (`core/translation_rules.py:332`) already exposes `_resolve_literal` (staticmethod), `_compute_unroll_values` (staticmethod), `UNSAFE_BODY_TOKENS` (class attribute tuple), and `CONTROL_FLOW_PREFIX` (class attribute compiled regex) — confirmed by reading the class directly. This plan's rule calls/reads these via `UnrollConstantShuffleLoopRule.<name>`, not by copying their logic, so any future fix to those (e.g. a 7th hardening round) automatically applies here too.
- The new rule's compound-condition pattern (`VAR OP BOUND && VAR OP2 RUNTIME_ID`) cannot match `UnrollConstantShuffleLoopRule`'s existing pattern (which requires the bound token to be followed immediately by `;`, with no `&&`), and vice versa — confirmed by testing both regexes against the same compound-condition source directly: the existing rule's pattern produces no match on it. The two rules are mutually exclusive by construction; no priority ordering is needed to prevent collision, only to ensure this new rule runs before the shuffle rules (priority 70) would otherwise hard-fail on the untouched loop.
- The full transformation was prototyped end-to-end through the *real* `CUDAToRIPPLETransformer` (not just the isolated rule) before writing this plan: the happy-path example produces 5 correctly-guarded literal `ripple_shuffle` calls and passes a real `clang -fsyntax-only` check (`verify_ripple_syntax` returned `(True, '')`); a direction-mismatch input and a body-touches-runtime-id input were both confirmed to fall through the rule untouched and correctly raise `TranslationError` via the existing hard-fail machinery (no new failure path needed — the existing one already does the right thing once this rule correctly declines).
- `ShuffleDownRule` (and the other 3 shuffle rules) apply their regex via `re.sub` over the whole code string with no awareness of surrounding brace/`if` nesting — confirmed the wrapped-in-`if` literal shuffle calls this rule emits are still found and hoisted correctly by the existing, unmodified `ShuffleDownRule`.

---

## Task 1: Implement and test `PredicatedShuffleUnrollRule`

**Files:**
- Modify: `core/translation_rules.py`
- Modify: `tests/test_translation.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_translation.py`:

```python
def test_predicated_unroll_halving_loop_with_runtime_lower_bound():
    # The motivating shape: offset's value set (16, 8, 4, 2, 1) is
    # 100% determined by the literal init/bound/step — min_offset only
    # decides how many of those 5 already-known iterations actually
    # run, not which values occur.
    source = """
__global__ void haloExchange(float *data, int min_offset) {
    float val = data[threadIdx.x];
    for (int offset = 16; offset > 0 && offset > min_offset; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    assert "__shfl_down_sync" not in result
    assert result.count("ripple_shuffle(") == 5
    assert "if (16 > min_offset)" in result
    assert "if (8 > min_offset)" in result
    assert "if (4 > min_offset)" in result
    assert "if (2 > min_offset)" in result
    assert "if (1 > min_offset)" in result


def test_predicated_unroll_doubling_loop_with_runtime_upper_bound():
    # Mirror shape, opposite direction: '<' literal clause paired with
    # a '<' runtime clause — proves direction-matching isn't hardcoded
    # to the halving/'>' case.
    source = """
__global__ void butterflyPartial(float *data, int max_i) {
    float val = data[threadIdx.x];
    for (int i = 1; i < 32 && i < max_i; i *= 2) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    assert "__shfl_xor_sync" not in result
    assert result.count("ripple_shuffle(") == 5
    assert "if (1 < max_i)" in result
    assert "if (16 < max_i)" in result


def test_predicated_unroll_output_passes_syntax_check():
    source = """
__global__ void haloExchange(float *data, int min_offset) {
    float val = data[threadIdx.x];
    for (int offset = 16; offset > 0 && offset > min_offset; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    success, output = verify_ripple_syntax(result)
    assert success, output


def test_predicated_unroll_declines_on_direction_mismatch_and_hard_fails():
    # CRITICAL correctness guard: a '>' literal clause paired with a
    # '<' runtime clause breaks the "once false, stays false"
    # guarantee independent per-iteration guards rely on. Must decline
    # entirely — leaving 'offset' unsubstituted — so the existing
    # hard-fail machinery (not a corrupted guess) is what catches this.
    source = """
__global__ void mismatched(float *data, int min_offset) {
    float val = data[threadIdx.x];
    for (int offset = 16; offset > 0 && offset < min_offset; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    data[threadIdx.x] = val;
}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    with pytest.raises(TranslationError) as exc_info:
        transformer.transform(source)
    assert "not a compile-time constant" in str(exc_info.value)
    assert "'offset'" in str(exc_info.value)


def test_predicated_unroll_declines_when_body_touches_runtime_id_and_hard_fails():
    # CRITICAL correctness guard: if the body reassigns or reads the
    # runtime bound, the "fixed threshold for the loop's whole
    # lifetime" assumption the guards depend on no longer holds. Must
    # decline rather than guess it's still safe.
    source = """
__global__ void bodyTouchesBound(float *data, int min_offset) {
    float val = data[threadIdx.x];
    for (int offset = 16; offset > 0 && offset > min_offset; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset) + min_offset;
    }
    data[threadIdx.x] = val;
}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    with pytest.raises(TranslationError) as exc_info:
        transformer.transform(source)
    assert "not a compile-time constant" in str(exc_info.value)


def test_predicated_unroll_reuses_unsafe_body_token_guard():
    # Proves the delegation to UnrollConstantShuffleLoopRule.UNSAFE_BODY_TOKENS
    # is actually wired up, not just referenced — a comment containing a
    # stray brace must still bail this rule out too, for the same
    # invisible-corruption reason the original 6 hardening rounds fixed.
    source = """
__global__ void hasComment(float *data, int min_offset) {
    float val = data[threadIdx.x];
    for (int offset = 16; offset > 0 && offset > min_offset; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset); // note [3] }
    }
    data[threadIdx.x] = val;
}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    with pytest.raises(TranslationError) as exc_info:
        transformer.transform(source)
    assert "not a compile-time constant" in str(exc_info.value)


def test_predicated_unroll_braceless_body_supported():
    source = """
__global__ void haloExchange(float *data, int min_offset) {
    float val = data[threadIdx.x];
    for (int offset = 16; offset > 0 && offset > min_offset; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    assert result.count("ripple_shuffle(") == 5
    success, output = verify_ripple_syntax(result)
    assert success, output


def test_predicated_unroll_braceless_declines_on_control_flow_body():
    # Proves the delegation to UnrollConstantShuffleLoopRule.CONTROL_FLOW_PREFIX
    # is actually wired up for the braceless path.
    source = """
__global__ void controlFlowBody(float *data, int min_offset) {
    float val = data[threadIdx.x];
    for (int offset = 16; offset > 0 && offset > min_offset; offset /= 2)
        if (offset > 4) val += __shfl_down_sync(0xffffffff, val, offset);
    data[threadIdx.x] = val;
}
"""
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    with pytest.raises(TranslationError):
        transformer.transform(source)


def test_predicated_unroll_does_not_collide_with_plain_unroll_rule():
    # A plain (non-compound-condition) loop must still go through
    # UnrollConstantShuffleLoopRule exactly as before — the new rule's
    # pattern requires '&&' and must not accidentally also match this.
    source = """
__global__ void plainLoop(float *data) {
    float val = data[threadIdx.x];
    for (int i = 1; i < 8; i *= 2) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    data[threadIdx.x] = val;
}
"""
    result = translate_cuda_source(source)
    assert "Unrolled loop over 'i'" in result  # plain rule fired
    assert "Predicated-unrolled" not in result  # new rule did not fire
    assert result.count("ripple_shuffle(") == 3  # i = 1, 2, 4
```

You'll need `verify_ripple_syntax` imported if it isn't already — check the existing imports at the top of `tests/test_translation.py` first (other tests in this file already use it; if there's no existing import line for it, add `from tests.compile_verify import verify_ripple_syntax`).

- [ ] **Step 2: Run to confirm they fail**

Run: `source venv/bin/activate && python -m pytest tests/test_translation.py -k "predicated_unroll" -v`
Expected: `test_predicated_unroll_halving_loop_with_runtime_lower_bound`, `test_predicated_unroll_doubling_loop_with_runtime_upper_bound`, `test_predicated_unroll_output_passes_syntax_check`, and `test_predicated_unroll_braceless_body_supported` FAIL (the rule doesn't exist yet, so the loop is left untouched and `__shfl_*_sync` is still present, or `translate_cuda_source` raises `TranslationError` unexpectedly since the shuffle rules already hard-fail on the untouched compound-condition loop). The 5 decline/guard tests (`direction_mismatch`, `body_touches_runtime_id`, `unsafe_body_token`, `control_flow_body`, `does_not_collide`) should already PASS — they assert today's actual (correct, hard-fail) behavior, which doesn't change once the rule exists for the *cases it declines*.

**Correction (found during implementation, applied here for anyone re-running this plan):** `test_predicated_unroll_does_not_collide_with_plain_unroll_rule`'s assertions above already reflect a fix to an assertion that was broken as originally drafted — an earlier draft asserted `"if (" not in result`, which can never pass for ANY `translate_cuda_source()` output, since the RIPPLE boilerplate header unconditionally emits `if (` inside its `ripple_atomic_max`/`ripple_atomic_min` macros. Confirmed this fails even against a completely unrelated kernel with no shuffle loop at all, before any of this task's code exists. The test code above already asserts on which rule's warning message actually fired instead, which is the real intent and was already passing before this task's rule existed.

- [ ] **Step 3: Implement the rule**

In `core/translation_rules.py`, add this class immediately after `class UnrollConstantShuffleLoopRule` (i.e., right before `class ShuffleDownRule` — grouping it there matches this file's "related rules sit near each other" convention, since it's a sibling of the class it reuses):

```python
class PredicatedShuffleUnrollRule(TranslationRule):
    """
    Handles a shuffle loop whose induction variable's value SET is
    fully compile-time-known (literal init/bound/step, exactly like
    UnrollConstantShuffleLoopRule requires) but whose loop condition
    also carries a second, runtime-dependent early-exit clause:

      for (int VAR = INIT; VAR OP BOUND && VAR OP2 RUNTIME_ID; VAR OP= STEP) { BODY }

    e.g. `for (int offset = 16; offset > 0 && offset > min_offset;
    offset /= 2)`. RUNTIME_ID doesn't change WHICH values the loop
    variable takes — that's still 100% determined by the literal
    clause alone, exactly as it is for UnrollConstantShuffleLoopRule —
    it only decides how many of those already-known iterations
    actually run. That's a provably static value domain, not a guess,
    which is what makes unrolling safe here.

    Deliberately reuses UnrollConstantShuffleLoopRule's static
    machinery directly (_resolve_literal, _compute_unroll_values,
    UNSAFE_BODY_TOKENS, CONTROL_FLOW_PREFIX) rather than duplicating
    it — that class went through 6 review-driven hardening rounds to
    get its invisible-corruption bail-outs right, and duplicating that
    logic here would mean re-deriving (and re-risking) all of it.

    Two additional guards, both required for soundness and both
    reasons to decline (leave the loop untouched) rather than fire:
      - DIRECTION CONSISTENCY: OP and OP2 must be in the same family
        (both '<'/'<=' or both '>'/'>='). The induction variable moves
        monotonically under a literal step; this is what guarantees
        each emitted guard's "once false, stays false" behavior
        matches the original loop's actual early-exit semantics. A
        mismatched direction (e.g. OP='>' with OP2='<') would let the
        runtime clause flip back to true for a later literal value,
        silently changing which iterations run.
      - RUNTIME_ID UNTOUCHED IN BODY: if the runtime bound is
        reassigned or even just referenced inside the loop body, the
        "fixed threshold for the loop's whole lifetime" assumption the
        guards depend on breaks. Declines rather than trying to prove
        the body doesn't mutate it.

    This is deliberately NOT a general "runtime-bounded loops are
    finite" mechanism. `for (int offset = n; offset > 0; offset >>=
    1)` where `n` is an unconstrained kernel parameter has NO provable
    finite domain (n could be arbitrarily large) and is correctly left
    for the shuffle rules to hard-fail on — guessing a domain for that
    shape would be exactly the silent-wrongness failure class the rest
    of this shuffle-handling work exists to eliminate. This rule only
    fires when the ENTIRE domain is already provable from the literal
    clause alone.

    Emits one `if (LITERAL_VALUE OP2 RUNTIME_ID) { substituted_body }`
    per unrolled value, reproducing the original loop's per-iteration
    guard. Each literal-argument shuffle call inside is then hoisted
    normally by the existing shuffle rules — the `if` wrapper doesn't
    interfere, since those rules match __shfl_*_sync(...) anywhere in
    the text regardless of surrounding brace/if nesting.

    Matches only the same braced/braceless BODY shapes
    UnrollConstantShuffleLoopRule matches, with the same known
    limitations (no nested braces, no braceless control-flow prefix,
    no string/char literal or comment anywhere in the body) — see that
    class's docstring for the full rationale, since these are the same
    guards, reused rather than restated.
    """

    _LITERAL_OR_WARPSIZE = r'(\d+|warpSize)'

    PATTERN_BRACED = (
        r'for\s*\(\s*int\s+(\w+)\s*=\s*' + _LITERAL_OR_WARPSIZE + r'\s*;\s*'
        r'\1\s*(<=|>=|<|>)\s*' + _LITERAL_OR_WARPSIZE + r'\s*&&\s*'
        r'\1\s*(<=|>=|<|>)\s*(\w+)\s*;\s*'
        r'\1\s*(\*=|/=|\+=|-=)\s*(\d+)\s*\)\s*'
        r'\{([^{}]*)\}'
    )
    PATTERN_BRACELESS = (
        r'for\s*\(\s*int\s+(\w+)\s*=\s*' + _LITERAL_OR_WARPSIZE + r'\s*;\s*'
        r'\1\s*(<=|>=|<|>)\s*' + _LITERAL_OR_WARPSIZE + r'\s*&&\s*'
        r'\1\s*(<=|>=|<|>)\s*(\w+)\s*;\s*'
        r'\1\s*(\*=|/=|\+=|-=)\s*(\d+)\s*\)\s*'
        r'(?!\{)([^;{}]+;)'
    )

    # Kept for stylistic consistency with every other rule class in this
    # file (each defines a `PATTERN` attribute) — not actually read
    # anywhere for this class specifically, since matches()/apply()
    # below explicitly use PATTERN_BRACED/PATTERN_BRACELESS instead.
    PATTERN = PATTERN_BRACED

    def __init__(self):
        super().__init__(
            name="predicated_shuffle_unroll",
            description="Unroll a compile-time-bounded shuffle loop with a runtime early-exit clause, guarding each iteration",
            cuda_pattern=self.PATTERN_BRACED,
            priority=81  # Between UnrollConstantShuffleLoopRule (82) and
                         # the shuffle rules (70) — the two patterns are
                         # mutually exclusive (compound '&&' condition
                         # vs single condition) so exact ordering
                         # relative to 82 doesn't affect correctness,
                         # but must be > 70 so this rule gets a chance
                         # before the shuffle rules would otherwise
                         # hard-fail on the untouched loop.
        )

    def matches(self, cuda_code: str) -> bool:
        return bool(
            re.search(self.PATTERN_BRACED, cuda_code)
            or re.search(self.PATTERN_BRACELESS, cuda_code)
        )

    def _replace(self, match, ctx: TranslationContext, is_braceless: bool = False) -> str:
        var_name = match.group(1)
        init = UnrollConstantShuffleLoopRule._resolve_literal(match.group(2))
        cond_op = match.group(3)
        bound = UnrollConstantShuffleLoopRule._resolve_literal(match.group(4))
        cond_op2 = match.group(5)
        runtime_id = match.group(6)
        step_op = match.group(7)
        step = int(match.group(8))
        body = match.group(9)

        if cond_op[0] != cond_op2[0]:
            # Direction mismatch — see class docstring. Independent
            # per-iteration guards would not correctly reproduce the
            # original loop's early-exit behavior.
            return match.group(0)

        if any(tok in body for tok in UnrollConstantShuffleLoopRule.UNSAFE_BODY_TOKENS):
            # Same invisible-corruption risk UnrollConstantShuffleLoopRule
            # already hardened against — reused directly, not re-derived.
            return match.group(0)

        if is_braceless and UnrollConstantShuffleLoopRule.CONTROL_FLOW_PREFIX.match(body):
            return match.group(0)

        if not re.search(r'__shfl(?:_(?:down|up|xor))?_sync\s*\(', body):
            return match.group(0)
        if not re.search(rf'\b{re.escape(var_name)}\b', body):
            return match.group(0)
        if re.search(rf'\b{re.escape(runtime_id)}\b', body):
            # The runtime bound must be untouched by the body — see
            # class docstring's second guard.
            return match.group(0)

        values = UnrollConstantShuffleLoopRule._compute_unroll_values(
            init, cond_op, bound, step_op, step, UnrollConstantShuffleLoopRule.MAX_ITERATIONS
        )
        if values is None:
            return match.group(0)

        guarded_statements = []
        for v in values:
            substituted = re.sub(rf'\b{re.escape(var_name)}\b', str(v), body).strip()
            guarded_statements.append(f"if ({v} {cond_op2} {runtime_id}) {{ {substituted} }}")

        ctx.add_warning(
            f"Predicated-unrolled loop over '{var_name}' into {len(values)} "
            f"runtime-guarded iterations ({', '.join(str(v) for v in values)}), "
            f"each gated on 'VALUE {cond_op2} {runtime_id}', to resolve a "
            f"compile-time-constant shuffle argument while preserving the "
            f"loop's runtime early-exit condition"
        )

        joined = "\n    ".join(guarded_statements)
        values_str = ', '.join(str(v) for v in values)
        return (
            f"/* Predicated unroll: {var_name} = {values_str}, "
            f"each guarded by (VALUE {cond_op2} {runtime_id}) */\n    {joined}"
        )

    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        result = re.sub(
            self.PATTERN_BRACED,
            lambda m: self._replace(m, ctx, is_braceless=False),
            cuda_code,
            flags=re.DOTALL
        )
        result = re.sub(
            self.PATTERN_BRACELESS,
            lambda m: self._replace(m, ctx, is_braceless=True),
            result,
            flags=re.DOTALL
        )
        return result
```

- [ ] **Step 4: Register the rule**

In `core/translation_rules.py`, in `TranslationRuleEngine._register_default_rules()`, in the `default_rules` list, add `PredicatedShuffleUnrollRule(),` immediately after `UnrollConstantShuffleLoopRule(),`:

```python
            # Shuffles
            WarpReductionRule(),
            ButterflyAllReduceRule(),
            UnrollConstantShuffleLoopRule(),
            PredicatedShuffleUnrollRule(),
            ShuffleDownRule(),
            ShuffleUpRule(),
            ShuffleXorRule(),
            ShuffleSyncRule(),
```

- [ ] **Step 5: Run all the tests from Step 1**

Run: `source venv/bin/activate && python -m pytest tests/test_translation.py -k "predicated_unroll" -v`
Expected: all 9 PASS.

- [ ] **Step 6: Run the full suite**

Run: `source venv/bin/activate && python -m pytest tests/ -v`
Expected: all previous tests still pass, plus these 9 new ones — 125 passed, 0 failed (116 prior + 9 new). If the count differs, investigate before proceeding.

- [ ] **Step 7: Manually reproduce the motivating example end-to-end via the CLI**

Run:
```bash
cat > /tmp/predicated_unroll.cu <<'EOF'
__global__ void haloExchange(float *data, int min_offset) {
    float val = data[threadIdx.x];
    for (int offset = 16; offset > 0 && offset > min_offset; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    data[threadIdx.x] = val;
}
EOF
python -m interfaces.cli.cuda2ripple source /tmp/predicated_unroll.cu -o /tmp/predicated_unroll.ripple.c
cat /tmp/predicated_unroll.ripple.c
```
Expected: exit code 0, no `[Error]`, an informational "Predicated-unrolled loop..." warning is fine and expected, and the output contains 5 `if (VALUE > min_offset) { ... ripple_shuffle(...) ... }` guards with literal values 16, 8, 4, 2, 1. Then run it through the syntax checker:
```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from tests.compile_verify import verify_ripple_syntax
print(verify_ripple_syntax(open('/tmp/predicated_unroll.ripple.c').read()))
"
```
Expected: `(True, '')`.

Then confirm the direction-mismatch guard live, via the CLI:
```bash
cat > /tmp/mismatched.cu <<'EOF'
__global__ void mismatched(float *data, int min_offset) {
    float val = data[threadIdx.x];
    for (int offset = 16; offset > 0 && offset < min_offset; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    data[threadIdx.x] = val;
}
EOF
python -m interfaces.cli.cuda2ripple source /tmp/mismatched.cu -o /tmp/mismatched.ripple.c
echo "exit code: $?"
ls -la /tmp/mismatched.ripple.c 2>&1
```
Expected: non-zero exit code, an `[Error]` mentioning `offset` and "not a compile-time constant", and confirm `/tmp/mismatched.ripple.c` was NOT written (same propagation behavior already verified for the plain hard-fail case in the prior branch's Task 4).

- [ ] **Step 8: Commit**

```bash
git add core/translation_rules.py tests/test_translation.py
git commit -m "Handle shuffle loops with a runtime early-exit clause via predicated unroll

A shuffle loop like 'for (int offset = 16; offset > 0 && offset >
min_offset; offset /= 2)' has a value set (16, 8, 4, 2, 1) that's just
as compile-time-known as the plain case UnrollConstantShuffleLoopRule
already handles — min_offset only decides how many of those 5
already-known iterations run, not which values occur. Today this falls
through both existing rules untouched (neither's pattern matches a
compound '&&' condition) and hard-fails.

PredicatedShuffleUnrollRule unrolls to the same literal value set the
plain rule would compute, wrapping each substituted iteration in an
'if (VALUE OP2 RUNTIME_ID)' guard reproducing the original early-exit
condition. Each guard is independently correct because the induction
variable moves monotonically under a literal step: once one guard is
false, every later one is too, matching the original loop's actual
stop-and-never-resume behavior.

Two guards keep this from becoming an unsound 'runtime bounds are
usually small' guess: the runtime clause's comparison must be in the
same direction as the literal clause's (guarantees monotonic
'once-false-stays-false'), and the runtime bound must not be touched
by the loop body (guarantees it's a fixed threshold for the loop's
whole lifetime). Anything that doesn't meet both guards is left
untouched and correctly falls through to the existing hard-fail —
deliberately NOT extended to genuinely unbounded runtime loops (e.g. a
loop bounded by an unconstrained kernel parameter with no literal
clause at all), which have no provable finite domain and should keep
failing loudly rather than have one guessed for them.

Reuses UnrollConstantShuffleLoopRule's already-hardened static
machinery directly (_resolve_literal, _compute_unroll_values,
UNSAFE_BODY_TOKENS, CONTROL_FLOW_PREFIX) rather than duplicating it.

Verified end-to-end via the CLI: the motivating example produces 5
correctly-guarded literal shuffle calls and passes a real clang syntax
check; a direction-mismatch variant correctly falls through to the
existing hard-fail with no output file written."
```

---

## Self-Review

**Spec coverage:** the narrow, provably-sound subset of "3a" from the design conversation — a fully literal-determined value domain with a runtime early-exit clause — is implemented exactly as scoped, with both required safety guards (direction consistency, runtime-id-untouched) as explicit, tested checks, not implicit assumptions. The broader, unsound "runtime bounds are usually finite" claim is explicitly NOT implemented, and the docstring states why. ✓

**Placeholder scan:** no TBD/TODO/"add appropriate handling" phrasing; every code step is complete, runnable code (prototyped and verified end-to-end through the real pipeline before being written into this plan); every "Run:" step has a concrete command and expected count.

**Type/name consistency:** `PredicatedShuffleUnrollRule` is defined once (Task 1, Step 3) and registered once, by that exact name, in `_register_default_rules()` (Step 4). It references `UnrollConstantShuffleLoopRule`'s existing static methods and class attributes by their exact current names (`_resolve_literal`, `_compute_unroll_values`, `UNSAFE_BODY_TOKENS`, `CONTROL_FLOW_PREFIX`, `MAX_ITERATIONS`) — confirmed these exist with these exact names by reading the class directly in "Before you start."

**Expected test counts, traced cumulatively:** baseline before this plan = 116 passed (confirmed live in the worktree before writing this plan). This task adds 9 tests → 125.

**Dependency ordering:** none — this is a single, self-contained task. It doesn't modify any existing rule and is mutually exclusive by pattern shape with every existing rule, verified directly (not just argued) by testing the existing rule's pattern against the new rule's target shape and confirming no match.
