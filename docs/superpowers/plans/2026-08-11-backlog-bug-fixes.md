# Backlog Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the four real translator bugs found and deliberately deferred during the prior hardening pass (GitHub issues #7, #8, #9, #10 on `C-ripple/Translate`), each with real regression coverage — not just a patch, but proof the check that would have caught it now exists.

**Architecture:** Four independent fixes, ordered cheapest/lowest-risk first. #7 and #9 are one-line corrections. #10 is a small, semantically-grounded fix (the correct `dims` bitfield value, derived from the real RIPPLE API spec and the pattern's own semantics) that also un-blocks an existing `strict=True` xfail in the test suite — fixing it is only complete when that marker comes off too. #8 is the substantial one: all four warp-shuffle rules currently emit invalid C because C has no closures, so representing "a different permutation function per call site" requires actually hoisting a uniquely-named function to file scope, which the current rule-engine architecture doesn't yet support (each rule does pure in-place text substitution) — this task adds that capability.

**Tech Stack:** Python 3.13, existing translator codebase, pytest + pytest-timeout, clang for syntax verification (`tests/compile_verify.py`, already built).

---

## Before you start

All four fixes are grounded in the real upstream RIPPLE API spec (`temp_ripple_docs/src/ripple-spec/api.md`, a local gitignored reference checkout), not guesswork:

- `ripple_shuffle`'s real signature and idiom (a named, file-scope C function passed by pointer — never a lambda, never nested) is documented at `api.md:443-527`, with a full worked example (`transpose_8x8`/`transpose_tile`/`permute`). Task 4 below implements exactly that pattern.
- `ripple_reduceadd`'s real signature (`TYPE ripple_reduceadd(int dims, TYPE to_reduce)`, a dimension bitfield first) is at `api.md:45`. Task 3 derives the correct bitfield value from what `WarpReductionRule`'s own pattern actually represents (a single-dimension warp-shuffle reduction), not an arbitrary choice.

---

## Task 1: Fix #7 — `CUDALexer.read_number()` crashes on a trailing bare digit

**Files:**
- Modify: `frontends/source/cuda_frontend.py:281`
- Modify: `tests/test_ast_parsing.py` (add a regression test)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ast_parsing.py` (append — the file already has tests from the prior hardening branch, don't overwrite):

```python
def test_lexer_handles_trailing_bare_digit():
    # CUDALexer.read_number() did `self.peek() in 'xXoObB'` to detect a
    # 0x/0o/0b prefix; peek() returns None at end-of-source, and
    # `None in 'xXoObB'` raises TypeError. A source ending in a bare
    # digit (nothing after it at all) triggered this. GitHub issue #7.
    lexer = CUDALexer("0")
    tokens = lexer.tokenize()
    number_tokens = [t for t in tokens if t.type == TokenType.NUMBER]
    assert len(number_tokens) == 1
    assert number_tokens[0].value == "0"
```

You'll need `TokenType` imported — check the top of the file; if it's not already imported from `frontends.source.cuda_frontend`, add it to the existing import line rather than a new one.

- [ ] **Step 2: Run it to confirm it fails**

Run: `source venv/bin/activate && python -m pytest tests/test_ast_parsing.py::test_lexer_handles_trailing_bare_digit -v`
Expected: FAIL with `TypeError: 'in <string>' requires string as left operand, not NoneType`

- [ ] **Step 3: Fix it**

In `frontends/source/cuda_frontend.py`, in `CUDALexer.read_number()`, change:

```python
        # Handle hex, octal, binary
        if self.current_char() == '0' and self.peek() in 'xXoObB':
            value += self.advance()
            value += self.advance()
```

to:

```python
        # Handle hex, octal, binary
        if self.current_char() == '0' and self.peek() is not None and self.peek() in 'xXoObB':
            value += self.advance()
            value += self.advance()
```

- [ ] **Step 4: Run it to confirm it passes**

Run: `source venv/bin/activate && python -m pytest tests/test_ast_parsing.py::test_lexer_handles_trailing_bare_digit -v`
Expected: PASSED

- [ ] **Step 5: Run the full suite**

Run: `source venv/bin/activate && python -m pytest tests/ -v`
Expected: 69 passed, 1 xfailed (68 prior + this 1 new test; the `warp_reduction.cu` xfail from issue #10 is untouched by this task)

- [ ] **Step 6: Commit**

```bash
git add frontends/source/cuda_frontend.py tests/test_ast_parsing.py
git commit -m "Fix CUDALexer.read_number() crash on trailing bare digit

read_number() did self.peek() in 'xXoObB' to detect a 0x/0o/0b
prefix. peek() returns None at end-of-source, and None in 'xXoObB'
raises TypeError. Source ending in a bare digit (e.g. literally just
\"0\") triggered this — GitHub issue #7. Already failed safe (caught
by the AST pre-pass's exception handler, surfaced as a warning, not
a crash) but the lexer itself should degrade gracefully, not throw."
```

---

## Task 2: Fix #9 — VS Code extension emits the same wrong `ripple_get_size` name

**Files:**
- Modify: `interfaces/vscode/src/extension.ts`

Decision (already made): minimal parity fix. Patch the string to match the Python core's already-fixed name, keep the duplicated-logic architecture as-is — the bigger architectural question (should the extension shell out to Python instead of reimplementing rules in TypeScript) is a separate, larger decision not part of this round.

This project has no JS/TS test infrastructure (`interfaces/vscode/` has no test runner configured), so verification here is a manual read-through and a grep confirming no other file has the same wrong name — not an automated test.

- [ ] **Step 1: Confirm current occurrences**

Run: `grep -n "ripple_get_size" interfaces/vscode/src/extension.ts`
Expected: 3 matches (blockDim.x/y/z)

- [ ] **Step 2: Fix it**

In `interfaces/vscode/src/extension.ts`, change:

```typescript
        // Block dimensions
        output = output.replace(/blockDim\.x/g, 'ripple_get_size(ripple_block, 0)');
        output = output.replace(/blockDim\.y/g, 'ripple_get_size(ripple_block, 1)');
        output = output.replace(/blockDim\.z/g, 'ripple_get_size(ripple_block, 2)');
```

to:

```typescript
        // Block dimensions
        output = output.replace(/blockDim\.x/g, 'ripple_get_block_size(ripple_block, 0)');
        output = output.replace(/blockDim\.y/g, 'ripple_get_block_size(ripple_block, 1)');
        output = output.replace(/blockDim\.z/g, 'ripple_get_block_size(ripple_block, 2)');
```

- [ ] **Step 3: Verify no occurrences remain, and confirm no other file has the same bug**

Run: `grep -rn "ripple_get_size" --include="*.ts" --include="*.py" --include="*.md" . | grep -v temp_ripple_docs | grep -v __pycache__`
Expected: empty output

- [ ] **Step 4: Commit**

```bash
git add interfaces/vscode/src/extension.ts
git commit -m "Fix VS Code extension emitting wrong RIPPLE API function name

Same ripple_get_size -> ripple_get_block_size bug already fixed in
the Python core (core/translation_rules.py, frontends/ir/ir_frontend.py)
— GitHub issue #9. The extension has its own independent, duplicated
translation implementation (not a wrapper around the Python core),
so it needed this fixed separately. Minimal parity fix only; the
architectural question of whether the extension should shell out to
the Python translator instead of reimplementing rules in TypeScript
is a separate, larger decision, deliberately not part of this fix."
```

---

## Task 3: Fix #10 — `ripple_reduceadd` called with wrong arity

**Files:**
- Modify: `core/translation_rules.py` — `WarpReductionRule.apply()`
- Modify: `tests/test_real_kernels.py` — remove the now-resolved `xfail` for `warp_reduction.cu`

**The `dims` bitfield value:** the real API (`temp_ripple_docs/src/ripple-spec/api.md:45`) defines `TYPE ripple_reduceadd(int dims, TYPE to_reduce)` — a bitfield where bit `i` set means "reduce along dimension `i`". `WarpReductionRule`'s pattern (`core/translation_rules.py:392-408`) matches exactly one shape: a CUDA `warpSize/2`-halving loop doing a shuffle-down reduction — the classic single-dimension warp reduction, which in this translator's block model corresponds to dimension 0 (the dimension `threadIdx.x`/`RIPPLE_BLOCK_DIM_X` maps to — see `ThreadIdxRule`/`BlockDimRule`, both hardcode component `x` to dimension `0`). So the correct, semantically-grounded value is `0b1` (bit 0 set, reduce along dimension 0 only) — not an arbitrary placeholder.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_translation.py` (find `TestIntegration` or a similar existing class covering warp-reduction — search for `warp_reduction` or `WarpReductionRule` in the file first to see if there's already a relevant test class to extend; if not, add a new test function near the other integration-style tests):

```python
def test_warp_reduction_emits_correct_reduceadd_arity():
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
    assert "ripple_reduceadd(sum)" not in result  # the old, wrong 1-arg form
```

Check the top of `tests/test_translation.py` for how `translate_cuda_source` is imported (it should already be imported given other tests use it) — reuse the existing import, don't add a duplicate.

- [ ] **Step 2: Run it to confirm it fails**

Run: `source venv/bin/activate && python -m pytest tests/test_translation.py::test_warp_reduction_emits_correct_reduceadd_arity -v`
Expected: FAIL — `assert "ripple_reduceadd(0b1, sum)" in result` fails, actual output has `ripple_reduceadd(sum)`

- [ ] **Step 3: Fix the rule**

In `core/translation_rules.py`, in `WarpReductionRule.apply()`, change:

```python
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            loop_var = match.group(1)
            accum_var = match.group(2)
            
            ctx.add_warning(f"Optimized Warp Reduction loop for '{accum_var}' to 'ripple_reduceadd'")
            
            return f"""/* CUDA Warp Reduction Loop -> RIPPLE Intrinsic */
    {accum_var} = ripple_reduceadd({accum_var});"""
        
        # Use DOTALL to match across newlines if user formatted code vertically
        return re.sub(self.PATTERN, replace, cuda_code, flags=re.DOTALL)
```

to:

```python
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            loop_var = match.group(1)
            accum_var = match.group(2)
            
            ctx.add_warning(f"Optimized Warp Reduction loop for '{accum_var}' to 'ripple_reduceadd'")
            
            # api.md: TYPE ripple_reduceadd(int dims, TYPE to_reduce) — dims
            # is a bitfield, bit i set = reduce along dimension i. This rule
            # only matches the classic single-dimension warpSize-halving
            # reduction (threadIdx.x-shaped), i.e. dimension 0 — hence 0b1,
            # not an arbitrary placeholder. A multi-dimensional variant of
            # this pattern, if one is ever added, would need a different
            # bitfield derived from which dimension(s) it actually reduces.
            return f"""/* CUDA Warp Reduction Loop -> RIPPLE Intrinsic */
    {accum_var} = ripple_reduceadd(0b1, {accum_var});"""
        
        # Use DOTALL to match across newlines if user formatted code vertically
        return re.sub(self.PATTERN, replace, cuda_code, flags=re.DOTALL)
```

- [ ] **Step 4: Run it to confirm it passes**

Run: `source venv/bin/activate && python -m pytest tests/test_translation.py::test_warp_reduction_emits_correct_reduceadd_arity -v`
Expected: PASSED

- [ ] **Step 5: Remove the now-resolved xfail in `tests/test_real_kernels.py`**

`warp_reduction.cu`'s syntax-check failure was exactly this bug (see the xfail's own reason text, which already documents this). With the rule fixed, this file should now pass its syntax check cleanly — and because the existing marker is `strict=True`, leaving it in place would make the now-passing test report as a hard FAILURE (unexpected pass), not a clean pass. This is the marker doing its job; removing it is required, not optional.

In `tests/test_real_kernels.py`, change:

```python
SYNTAX_CHECK_PARAMS = [
    "ast_flat.cu",
    "ast_if_no_braces.cu",
    "atomics_cas_exch.cu",
    "bitwise_intrinsics.cu",
    "global_thread_index.cu",
    pytest.param(
        "warp_reduction.cu",
        marks=pytest.mark.xfail(
            reason=(
                "ripple_reduceadd arity mismatch, GitHub issue #10 — not "
                "issue #8: WarpReductionRule (priority 85) fully replaces "
                "this file's loop before ShuffleDownRule (priority 70) "
                "ever sees the __shfl_down_sync call, so it never "
                "exercises the shuffle-lambda bug tracked as issue #8"
            ),
            strict=True,
        ),
    ),
]
```

to:

```python
SYNTAX_CHECK_PARAMS = [
    "ast_flat.cu",
    "ast_if_no_braces.cu",
    "atomics_cas_exch.cu",
    "bitwise_intrinsics.cu",
    "global_thread_index.cu",
    "warp_reduction.cu",
]
```

Also update the module docstring — remove the paragraph describing the xfail (it starts with "warp_reduction.cu is expected to FAIL its syntax check right now") since it's no longer accurate; replace it with a short note that this file previously failed here due to issue #10, now fixed.

- [ ] **Step 6: Run the real-kernel tests**

Run: `source venv/bin/activate && python -m pytest tests/test_real_kernels.py -v`
Expected: 12 passed, 0 xfailed (6 structural + 6 syntax, `warp_reduction.cu` now passes cleanly instead of xfailing)

- [ ] **Step 7: Run the full suite**

Run: `source venv/bin/activate && python -m pytest tests/ -v`
Expected: 71 passed, 0 xfailed (69 from Task 1 + this task's 1 new unit test + 1 more passing that used to xfail = 71; zero xfails now, since the only one in the suite is resolved)

- [ ] **Step 8: Commit**

```bash
git add core/translation_rules.py tests/test_translation.py tests/test_real_kernels.py
git commit -m "Fix ripple_reduceadd arity mismatch (GitHub issue #10)

WarpReductionRule emitted ripple_reduceadd(accum_var) — 1 argument.
The real API (api.md:45) is TYPE ripple_reduceadd(int dims, TYPE
to_reduce) — a dimension bitfield first. This rule only ever matches
a single-dimension (threadIdx.x-shaped) warp reduction, so the
correct bitfield is 0b1 (dimension 0), not an arbitrary value.

Removes the strict=True xfail marker on warp_reduction.cu in
tests/test_real_kernels.py — that file's syntax-check failure was
exactly this bug (already correctly attributed in the marker's own
reason text from the prior review round), and now passes cleanly.
Leaving the marker in place would fail the suite outright, since
strict xfail treats an unexpected pass as a hard failure by design."
```

---

## Task 4: Fix #8 — All four warp-shuffle rules emit invalid C

**Files:**
- Modify: `core/semantic_model.py` — add a hoisted-declarations list to `TranslationContext`
- Modify: `core/translation_rules.py` — all four shuffle rules (`ShuffleDownRule`, `ShuffleXorRule`, `ShuffleUpRule`, `ShuffleSyncRule`)
- Modify: `frontends/source/cuda_frontend.py` — `_add_ripple_boilerplate()` splices hoisted declarations into the file preamble
- Create: `tests/examples/warp_shuffle_xor.cu` — new fixture giving issue #8 real regression coverage (the existing `warp_reduction.cu` fixture never reaches the shuffle rules at all — `WarpReductionRule` pre-empts it, which is why issue #10 was the one actually being tested there)
- Modify: `tests/test_real_kernels.py` — wire the new fixture into both test lists

**The real fix, grounded in the actual RIPPLE API idiom** (`temp_ripple_docs/src/ripple-spec/api.md:443-527`): C has no closures. The documented, real way to express "a different permutation function per `ripple_shuffle` call site" in C is a plain, named, **file-scope** function with signature `size_t fn(size_t k, size_t block_size)`, passed to `ripple_shuffle` by its bare name (a function pointer). Not a lambda (C++ only). Not a function nested inside another function (not valid standard C, and not what the spec's own C example does — its `transpose_8x8` helper is defined at file scope, before the function that calls `ripple_shuffle`).

This requires the translator to accumulate "things to define before the kernel" rather than doing pure text-in-place substitution, which none of the current rules need — they all substitute inline. Adding this capability is the actual scope of this task.

### Step 1: Add the hoisting mechanism to `TranslationContext`

In `core/semantic_model.py`, in the `TranslationContext` dataclass, add a new field. Find:

```python
    # Diagnostics
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
```

and add immediately after it:

```python
    # Code to hoist to file scope, before any kernel definition. Used by
    # rules that need a named helper function rather than an inline
    # expression — e.g. warp-shuffle permutation functions, which C has
    # no closure syntax for (see ShuffleDownRule/ShuffleXorRule/etc. in
    # core/translation_rules.py). Each entry is a complete, self-contained
    # C declaration/definition string. Consumed by
    # CUDAToRIPPLETransformer._add_ripple_boilerplate(), which places
    # these after the standard header/macros and before any translated
    # kernel body, so "must be defined before use" holds regardless of
    # which kernel in the file references which hoisted function.
    hoisted_declarations: list[str] = field(default_factory=list)
```

### Step 2: Write the failing tests

Add to `tests/test_translation.py`:

```python
def test_shuffle_xor_hoists_named_function_not_lambda():
    source = """
__global__ void kernel(int *a) {
    int val = a[0];
    int lane_mask = 1;
    int result = __shfl_xor_sync(0xffffffff, val, lane_mask);
    a[0] = result;
}
"""
    result = translate_cuda_source(source)
    assert "[]" not in result  # no C++ lambda capture syntax anywhere
    assert "auto " not in result  # no C++ auto either
    # The call site should reference a plain named function, not inline it.
    import re
    call_match = re.search(r'ripple_shuffle\(\s*val\s*,\s*(\w+)\s*\)', result)
    assert call_match, f"expected a plain ripple_shuffle(val, <name>) call, got:\n{result}"
    fn_name = call_match.group(1)
    # That name must be defined as a real, named, file-scope function
    # BEFORE the kernel that calls it (i.e. before the kernel's own
    # opening brace in the output), not nested inside it.
    fn_def_pos = result.find(f"size_t {fn_name}(size_t")
    kernel_start_pos = result.find("kernel_ripple(")
    assert fn_def_pos != -1, f"expected a definition of {fn_name}, got:\n{result}"
    assert fn_def_pos < kernel_start_pos, (
        f"{fn_name} must be defined before the kernel that uses it, "
        f"not nested inside it (def at {fn_def_pos}, kernel at {kernel_start_pos})"
    )


def test_multiple_shuffle_calls_get_unique_function_names():
    source = """
__global__ void kernel(int *a, int *b) {
    int v1 = a[0];
    int v2 = b[0];
    int r1 = __shfl_xor_sync(0xffffffff, v1, 1);
    int r2 = __shfl_xor_sync(0xffffffff, v2, 2);
    a[0] = r1;
    b[0] = r2;
}
"""
    result = translate_cuda_source(source)
    import re
    names = re.findall(r'ripple_shuffle\(\s*\w+\s*,\s*(\w+)\s*\)', result)
    assert len(names) == 2, f"expected 2 ripple_shuffle calls, found: {names}"
    assert names[0] != names[1], f"shuffle helper function names collided: {names}"
```

- [ ] **Step 2a: Run to confirm both fail**

Run: `source venv/bin/activate && python -m pytest tests/test_translation.py::test_shuffle_xor_hoists_named_function_not_lambda tests/test_translation.py::test_multiple_shuffle_calls_get_unique_function_names -v`
Expected: both FAIL (current output has `[](size_t k, size_t n) { ... }`, no named function)

### Step 3: Fix `ShuffleXorRule`

In `core/translation_rules.py`, change:

```python
class ShuffleXorRule(TranslationRule):
    """Translates __shfl_xor_sync to ripple_shuffle."""
    
    PATTERN = r'__shfl_xor_sync\s*\(\s*([^,]+),\s*([^,]+),\s*([^,\)]+)(?:,\s*([^)]+))?\)'
    
    def __init__(self):
        super().__init__(
            name="shuffle_xor",
            description="Translate __shfl_xor_sync to ripple_shuffle",
            cuda_pattern=self.PATTERN,
            priority=70
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            mask = match.group(1).strip()
            value = match.group(2).strip()
            lane_mask = match.group(3).strip()
            
            return f"ripple_shuffle({value}, [](size_t k, size_t n) {{ return k ^ {lane_mask}; }})"
        
        return re.sub(self.PATTERN, replace, cuda_code)
```

to:

```python
class ShuffleXorRule(TranslationRule):
    """Translates __shfl_xor_sync to ripple_shuffle."""
    
    PATTERN = r'__shfl_xor_sync\s*\(\s*([^,]+),\s*([^,]+),\s*([^,\)]+)(?:,\s*([^)]+))?\)'
    
    def __init__(self):
        super().__init__(
            name="shuffle_xor",
            description="Translate __shfl_xor_sync to ripple_shuffle",
            cuda_pattern=self.PATTERN,
            priority=70
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            mask = match.group(1).strip()
            value = match.group(2).strip()
            lane_mask = match.group(3).strip()
            
            # C has no lambdas/closures — ripple_shuffle takes a plain
            # named function pointer (api.md:443-527: "In C, this is
            # expressed using a 'shuffle function'... which exclusively
            # takes k and the block size"). Hoist a uniquely-named,
            # file-scope function instead of inlining a C++-only lambda.
            fn_name = f"__ripple_shfl_xor_{len(ctx.hoisted_declarations)}"
            ctx.hoisted_declarations.append(f"""
static inline size_t {fn_name}(size_t k, size_t block_size) {{
    return k ^ ({lane_mask});
}}""")
            
            return f"ripple_shuffle({value}, {fn_name})"
        
        return re.sub(self.PATTERN, replace, cuda_code)
```

### Step 4: Fix `ShuffleUpRule`

Change:

```python
class ShuffleUpRule(TranslationRule):
    """Translates __shfl_up_sync to ripple_shuffle."""
    
    PATTERN = r'__shfl_up_sync\s*\(\s*([^,]+),\s*([^,]+),\s*([^,\)]+)(?:,\s*([^)]+))?\)'
    
    def __init__(self):
        super().__init__(
            name="shuffle_up",
            description="Translate __shfl_up_sync to ripple_shuffle",
            cuda_pattern=self.PATTERN,
            priority=70
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            mask = match.group(1).strip()
            value = match.group(2).strip()
            delta = match.group(3).strip()
            
            return f"ripple_shuffle({value}, [](size_t k, size_t n) {{ return (k >= {delta}) ? k - {delta} : k; }})"
        
        return re.sub(self.PATTERN, replace, cuda_code)
```

to:

```python
class ShuffleUpRule(TranslationRule):
    """Translates __shfl_up_sync to ripple_shuffle."""
    
    PATTERN = r'__shfl_up_sync\s*\(\s*([^,]+),\s*([^,]+),\s*([^,\)]+)(?:,\s*([^)]+))?\)'
    
    def __init__(self):
        super().__init__(
            name="shuffle_up",
            description="Translate __shfl_up_sync to ripple_shuffle",
            cuda_pattern=self.PATTERN,
            priority=70
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            mask = match.group(1).strip()
            value = match.group(2).strip()
            delta = match.group(3).strip()
            
            fn_name = f"__ripple_shfl_up_{len(ctx.hoisted_declarations)}"
            ctx.hoisted_declarations.append(f"""
static inline size_t {fn_name}(size_t k, size_t block_size) {{
    return (k >= ({delta})) ? k - ({delta}) : k;
}}""")
            
            return f"ripple_shuffle({value}, {fn_name})"
        
        return re.sub(self.PATTERN, replace, cuda_code)
```

### Step 5: Fix `ShuffleSyncRule`

Change:

```python
class ShuffleSyncRule(TranslationRule):
    """Translates __shfl_sync (direct lane access)."""
    
    PATTERN = r'__shfl_sync\s*\(\s*([^,]+),\s*([^,]+),\s*([^,\)]+)(?:,\s*([^)]+))?\)'
    
    def __init__(self):
        super().__init__(
            name="shuffle_sync",
            description="Translate __shfl_sync to ripple_shuffle",
            cuda_pattern=self.PATTERN,
            priority=70
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            mask = match.group(1).strip()
            value = match.group(2).strip()
            src_lane = match.group(3).strip()
            
            return f"ripple_shuffle({value}, [](size_t k, size_t n) {{ return {src_lane}; }})"
        
        return re.sub(self.PATTERN, replace, cuda_code)
```

to:

```python
class ShuffleSyncRule(TranslationRule):
    """Translates __shfl_sync (direct lane access)."""
    
    PATTERN = r'__shfl_sync\s*\(\s*([^,]+),\s*([^,]+),\s*([^,\)]+)(?:,\s*([^)]+))?\)'
    
    def __init__(self):
        super().__init__(
            name="shuffle_sync",
            description="Translate __shfl_sync to ripple_shuffle",
            cuda_pattern=self.PATTERN,
            priority=70
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            mask = match.group(1).strip()
            value = match.group(2).strip()
            src_lane = match.group(3).strip()
            
            fn_name = f"__ripple_shfl_sync_{len(ctx.hoisted_declarations)}"
            ctx.hoisted_declarations.append(f"""
static inline size_t {fn_name}(size_t k, size_t block_size) {{
    return ({src_lane});
}}""")
            
            return f"ripple_shuffle({value}, {fn_name})"
        
        return re.sub(self.PATTERN, replace, cuda_code)
```

### Step 6: Fix `ShuffleDownRule`

This one already generates a named function — the bug is that it *inlines* the definition at the call site (nested inside the kernel) instead of hoisting it. Change:

```python
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            mask = match.group(1).strip()
            value = match.group(2).strip()
            delta = match.group(3).strip()
            width = match.group(4).strip() if match.group(4) else "32"
            
            # Generate shuffle function for down-shuffle
            shuffle_fn = f"""
// Shuffle down by {delta}
static inline size_t shfl_down_fn(size_t k, size_t n) {{
    size_t src = k + {delta};
    return (src < n) ? src : k;  // Clamp to valid range
}}"""
            
            ctx.add_warning(f"Shuffle down: ensure {delta} is compile-time constant for best codegen")
            
            return f"ripple_shuffle({value}, shfl_down_fn) /* mask={mask}, width={width} */"
        
        return re.sub(self.PATTERN, replace, cuda_code)
```

to:

```python
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            mask = match.group(1).strip()
            value = match.group(2).strip()
            delta = match.group(3).strip()
            width = match.group(4).strip() if match.group(4) else "32"
            
            # Hoist to file scope instead of inlining — a function
            # definition inside another function's body is not valid
            # standard C, and the old inline placement did exactly that
            # since this substitution always fires inside a kernel's
            # statement list.
            fn_name = f"__ripple_shfl_down_{len(ctx.hoisted_declarations)}"
            ctx.hoisted_declarations.append(f"""
// Shuffle down by {delta}
static inline size_t {fn_name}(size_t k, size_t block_size) {{
    size_t src = k + ({delta});
    return (src < block_size) ? src : k;  // Clamp to valid range
}}""")
            
            ctx.add_warning(f"Shuffle down: ensure {delta} is compile-time constant for best codegen")
            
            return f"ripple_shuffle({value}, {fn_name}) /* mask={mask}, width={width} */"
        
        return re.sub(self.PATTERN, replace, cuda_code)
```

### Step 7: Splice hoisted declarations into the file preamble

In `frontends/source/cuda_frontend.py`, in `_add_ripple_boilerplate()`, find the end of the method:

```python
        # Math intrinsics
        header += """
#define ripple_sad(x, y, z) (__builtin_abs((x) - (y)) + (z))

"""
        
        return header + source
```

and change it to:

```python
        # Math intrinsics
        header += """
#define ripple_sad(x, y, z) (__builtin_abs((x) - (y)) + (z))

"""
        
        # Hoisted declarations (e.g. warp-shuffle permutation functions —
        # C has no closures, so these are named, file-scope functions
        # generated by rules like ShuffleXorRule). Placed after the
        # standard header and before any translated kernel body in
        # `source`, so "must be defined before use" holds for every
        # kernel in the file regardless of which one references which
        # hoisted function.
        if self.ctx.hoisted_declarations:
            header += "/* Shuffle permutation functions (hoisted to file scope) */\n"
            header += "\n".join(self.ctx.hoisted_declarations)
            header += "\n\n"
        
        return header + source
```

Check the exact surrounding indentation/context first (read the method) — the plan's earlier phase captured this method's structure, but confirm `self.ctx` is the correct reference (it's used elsewhere in this same method, e.g. for warnings) before applying.

- [ ] **Step 7a: Run the two failing tests from Step 2 again**

Run: `source venv/bin/activate && python -m pytest tests/test_translation.py::test_shuffle_xor_hoists_named_function_not_lambda tests/test_translation.py::test_multiple_shuffle_calls_get_unique_function_names -v`
Expected: both PASS

### Step 8: Add a real fixture kernel for issue #8, since none of the existing sample kernels exercise the raw shuffle rules

Create `tests/examples/warp_shuffle_xor.cu`:

```c
__global__ void butterflyXor(int *data) {
    int val = data[threadIdx.x];
    int partner_val = __shfl_xor_sync(0xffffffff, val, 1);
    data[threadIdx.x] = val + partner_val;
}
```

This is a classic butterfly-exchange pattern using `__shfl_xor_sync` directly (not wrapped in the `warpSize/2`-halving loop shape `WarpReductionRule` matches), so it actually reaches `ShuffleXorRule` — unlike `warp_reduction.cu`, which issue #10's investigation found never reaches any shuffle rule at all.

### Step 9: Wire the new fixture into `tests/test_real_kernels.py`

Add `"warp_shuffle_xor.cu"` to both `KERNEL_FILES` and `SYNTAX_CHECK_PARAMS` (plain string, no xfail — it should pass cleanly once Steps 3-7 land).

- [ ] **Step 9a: Run the real-kernel tests**

Run: `source venv/bin/activate && python -m pytest tests/test_real_kernels.py -v`
Expected: 14 passed, 0 xfailed (7 structural + 7 syntax)

### Step 10: Run the full suite

Run: `source venv/bin/activate && python -m pytest tests/ -v`
Expected: 74 passed, 0 xfailed (71 from Task 3 + 2 new unit tests from Step 2 + 1 more passing kernel from the new fixture = 74)

### Step 11: Commit

```bash
git add core/semantic_model.py core/translation_rules.py frontends/source/cuda_frontend.py tests/test_translation.py tests/examples/warp_shuffle_xor.cu tests/test_real_kernels.py
git commit -m "Fix all 4 warp-shuffle rules emitting invalid C (GitHub issue #8)

ShuffleXorRule/ShuffleUpRule/ShuffleSyncRule emitted C++ lambda
expressions as inline ripple_shuffle() arguments — not valid C at
all. ShuffleDownRule avoided the lambda but spliced a full function
*definition* into the substitution text at the call site, which
always lands inside a kernel's statement list — also not valid C
(nested function definitions aren't standard).

The real RIPPLE API idiom (api.md:443-527) is a plain, named,
file-scope function passed by pointer. Added TranslationContext.
hoisted_declarations (core/semantic_model.py) so rules can register
code to place before any kernel body, rather than only substituting
in place; _add_ripple_boilerplate() now splices these into the file
preamble after the standard header. Each shuffle call site gets a
uniquely-named helper function (a running counter over
hoisted_declarations — verified collision-free for multiple shuffle
calls in one file).

Added tests/examples/warp_shuffle_xor.cu — real regression coverage
for this bug, since the existing warp_reduction.cu fixture never
actually reaches the shuffle rules (WarpReductionRule pre-empts it,
which is what issue #10 was really about, per the prior review
round's investigation)."
```

---

## Self-Review

**Spec coverage:**
- Issue #7 (lexer TypeError) → Task 1. ✓
- Issue #8 (invalid-C shuffle output) → Task 4, all 4 rules fixed with a real hoisting mechanism, plus new regression coverage since none existed. ✓
- Issue #9 (VS Code extension duplicate bug) → Task 2, minimal parity fix per explicit scope decision. ✓
- Issue #10 (`ripple_reduceadd` arity) → Task 3, including removing the now-resolved `strict=True` xfail this bug was blocking. ✓

**Placeholder scan:** No TBD/TODO/"add appropriate handling" phrasing; every code step is complete, runnable code (all four shuffle rule replacements are full method bodies, not fragments); every "Run:" step has a concrete command and expected pass count.

**Type/name consistency:** `hoisted_declarations: list[str]` is defined once in `TranslationContext` (Task 4, Step 1) and read/written identically everywhere else (`ctx.hoisted_declarations.append(...)` in all 4 shuffle rules, `self.ctx.hoisted_declarations` in `_add_ripple_boilerplate`). The naming scheme `f"__ripple_shfl_<kind>_{len(ctx.hoisted_declarations)}"` is consistent across all 4 rules and relies on `len()` being read *before* that rule's own `append()` — confirmed correct since each rule's `replace()` closure computes `fn_name` before calling `.append()`. Task 3's `0b1` bitfield and Task 4's hoisting mechanism are independent (different rules, different files touched) but both land in `core/translation_rules.py` — Task 3 should be committed and merged before Task 4 starts, per the task ordering above, so Task 4's diff doesn't need to account for Task 3's in-flight change to the same file.

**Expected test counts, traced cumulatively:** baseline before this plan = 68 passed, 1 xfailed. Task 1 adds 1 test → 69 passed, 1 xfailed. Task 3 adds 1 test and resolves the 1 xfail → 71 passed, 0 xfailed. Task 4 adds 2 unit tests + 1 new fixture kernel's 2 tests (1 structural + 1 syntax-check) → 75 passed, 0 xfailed (this arithmetic was off by one in the original plan text, which said 74 — 71 + 2 + 2 = 75, not 74; see the "Actual outcome" section below for what Task 4 grew into beyond this baseline).

---

## Actual outcome (post-execution addendum)

Tasks 1-3 executed exactly as planned above, with no material deviation. Task 4 did not — code review found real, verified defects in each round it was believed complete, and each was fixed rather than waved through. This addendum exists because the plan text above still reads as if Task 4 were the single, one-shot fix originally scoped, which is no longer an accurate description of what shipped. Per this project's own standard (a stale doc gets fixed at the source, not left standing next to the thing that corrected it), rather than rewrite the plan's Task 4 section itself (which remains an accurate record of the *starting* design — the hoisting mechanism it describes is exactly what shipped), this section records what was added on top of it:

**Round 2 (beyond the original Task 4 text):**
- Variable-argument shuffles (e.g. a loop-local `offset` passed as the delta) were found to produce broken C — the hoisted file-scope function referenced an out-of-scope variable. Fixed by adding `_is_compile_time_constant_expr()`, gating all 4 rules: a non-constant argument now leaves the original CUDA call untranslated plus a `ctx.add_warning(...)`, rather than emitting invalid C. This is a deliberate, permanent scope boundary, not a temporary gap — `ripple_shuffle`'s function-pointer signature is fixed at exactly `(k, block_size)` in the real API, so there is no clean way to thread an arbitrary runtime value through it. Tracked as [GitHub issue #11](https://github.com/C-ripple/Translate/issues/11).
- The CLI's interactive REPL (`interfaces/cli/cuda2ripple.py`, `cmd_interactive()`) was found to reuse one `TranslationContext` across every `file <path>` command, leaking one file's hoisted shuffle helper functions into a later, unrelated file's printed output — a new, observable bug the hoisting mechanism introduced (nothing existed to leak before). Fixed by constructing a fresh context per `file` command.
- While fixing the above, found that Round 1's own new unit test (`test_shuffle_xor_hoists_named_function_not_lambda`) was accidentally using a variable argument instead of a literal, meaning it had been unknowingly exercising the exact broken path being fixed. Corrected the test fixture to use a literal, matching what the test's name actually claims to prove.

**Round 3 (found by code review of round 2, not by the original bug report):**
- `_is_compile_time_constant_expr()`'s shape-only regex accepted a parenthesized constant like `(1)` — truncated by the (pre-existing, unchanged) outer capture regex to a dangling `(1` — as "constant," silently splicing invalid C (`return k ^ ((1);`) with **zero warnings**. This was the exact failure mode Round 2 existed to eliminate, reintroduced through the mechanism meant to prevent it. Fixed with a paren-count check; verified red→green by deliberately reverting the fix and confirming the new regression test failed with the exact reported malformed output.
- Test coverage for the constant-check gate had been concentrated entirely on `ShuffleXorRule` — the other 3 rules shared the identical (hand-copied, not centralized) gating logic with zero dedicated tests. Added a parametrized sweep across all 4 shuffle intrinsics for both the constant and variable-argument cases.

**Final state:** 4/4 backlog issues (#7, #8, #9, #10) genuinely fixed, 3 additional real bugs found and fixed during this branch's own review process (variable-argument shuffle breakage, CLI REPL state leak, paren-truncation false positive), 1 new issue filed for a disclosed, architecturally-deferred scope boundary (#11). Final test count: **86 passed, 0 failed, 0 xfailed** (75 from Tasks 1-4-as-originally-planned + 11 more from Task 4's rounds 2-3: 1 paren-balance regression + 4 variable-argument parametrized + 4 literal-argument parametrized + 2 from round 2's variable/literal pair before the parametrized sweep replaced the need for separate hand-written ones — see `tests/test_translation.py` for the exact final set rather than re-deriving the count by hand here, since by this point in the branch's history hand-tracing the arithmetic is more error-prone than just running `pytest`).
