# Update C-Ripple's Source Translator to Current Ripple API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

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

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_translation.py::TestBoilerplateGeneration::test_hvx_pe_self_defined -v`
Expected: FAIL — `assert "#define HVX_PE 0" in result` is False (nothing defines `HVX_PE` today).

- [ ] **Step 3: Add the `#define`**

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

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_translation.py::TestBoilerplateGeneration::test_hvx_pe_self_defined -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `venv/bin/python -m pytest -v`
Expected: All previously-passing tests still pass (this change only adds a line; nothing
depended on its absence).

- [ ] **Step 6: Commit**

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

- [ ] **Step 1: Write the failing tests**

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

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_translation.py::TestTranslationRules::test_atomic_add_rule tests/test_complex_kernels.py::TestAtomicOperations -v`
Expected: FAIL — `test_atomic_add_rule` fails on `assert ctx.has_errors()` (no error
recorded today); the other three fail because `translate_cuda_source` returns normally
instead of raising.

- [ ] **Step 3: Rewrite the five atomic rules**

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

- [ ] **Step 4: Remove the fake atomic macros from generated output**

In `frontends/source/cuda_frontend.py`, delete lines 601-614 (the entire
`/* Atomic operation wrappers for Hexagon */` block, from `#ifdef __HEXAGON__` through
the matching `#endif`) — nothing calls these macros anymore, and they reference
functions (`ripple_atomic_add` etc.) that were never real Ripple API.

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_translation.py::TestTranslationRules::test_atomic_add_rule tests/test_complex_kernels.py::TestAtomicOperations -v`
Expected: PASS

- [ ] **Step 6: Run the full test suite to check for regressions**

Run: `venv/bin/python -m pytest -v`
Expected: All pass. (`test_sad_computation` in `test_complex_kernels.py` will now fail —
it combines `__sad` with a bare `atomicAdd`; this is expected and fixed in Task 3.)

- [ ] **Step 7: Commit**

```bash
git add core/translation_rules.py frontends/source/cuda_frontend.py \
        tests/test_translation.py tests/test_complex_kernels.py
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
output's boilerplate header, since nothing calls them anymore."
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

- [ ] **Step 1: Write the failing tests**

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

- [ ] **Step 2: Run tests to verify the rename test fails**

Run: `venv/bin/python -m pytest tests/test_complex_kernels.py::TestAtomicOperations::test_sad_rename tests/test_complex_kernels.py::TestAtomicOperations::test_sad_with_bare_atomic_add_fails -v`
Expected: `test_sad_rename` FAILs (`cripple_sad(` not in result — it's still `ripple_sad`).
`test_sad_with_bare_atomic_add_fails` should already PASS (Task 2 made this hard-fail).

- [ ] **Step 3: Rename the macro**

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

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_complex_kernels.py::TestAtomicOperations -v`
Expected: PASS (all 5 tests in this class, including the two from Task 2 and the two from this task).

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `venv/bin/python -m pytest -v`
Expected: All pass except `test_2d_convolution`, which fails starting in Task 4 (not yet
reached) — if it already fails here, something is wrong; stop and investigate before
continuing.

- [ ] **Step 6: Commit**

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

- [ ] **Step 1: Write the failing tests**

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

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_translation.py::TestTranslationRules::test_shared_memory_rule tests/test_translation.py::TestTranslationRules::test_shared_memory_rule_multidim_hard_fails tests/test_complex_kernels.py::TestConvolutionKernels -v`
Expected: FAIL — `test_shared_memory_rule` (old attribute output, no vtcm_malloc);
`test_shared_memory_rule_multidim_hard_fails` (`SharedMemoryRule` doesn't exist as an
attribute name yet — no error recorded); `test_shared_memory_vtcm_free_placement` (no
such function/assertions met yet). `test_2d_convolution` should already PASS as rewritten
(it no longer touches shared memory) — if it fails, the kernel rewrite has a bug; fix
before continuing.

- [ ] **Step 3: Rewrite `SharedMemoryRule`**

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

- [ ] **Step 4: Update the syntax-check stub header**

In `tests/stub_headers/ripple.h`, add before the closing `#endif` (currently line 46):

```c
/* opt/hexagon/src/hvx-opt.md's SpVV example uses vtcm_malloc()/vtcm_free()
 * but never gives them a formal declared signature — only usage. This
 * signature is inferred directly from that usage (size + alignment for
 * malloc; a single pointer for free), not from an upstream prototype. */
void *vtcm_malloc(size_t size, size_t align_as);
void vtcm_free(void *ptr);
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_translation.py::TestTranslationRules::test_shared_memory_rule tests/test_translation.py::TestTranslationRules::test_shared_memory_rule_multidim_hard_fails tests/test_complex_kernels.py::TestConvolutionKernels -v`
Expected: PASS

- [ ] **Step 6: Run the full test suite to check for regressions**

Run: `venv/bin/python -m pytest -v`
Expected: All pass.

- [ ] **Step 7: Commit**

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

- [ ] **Step 8 (optional but recommended — matches existing repo convention):**
Open a tracking issue for the multi-dimensional VTCM gap, the same way issue #11 tracks
the unsupported runtime-variable shuffle case:

```bash
gh issue create --repo C-ripple/Translate \
  --title "Multi-dimensional __shared__ arrays can't translate to VTCM" \
  --label enhancement \
  --body "SharedMemoryRule hard-fails on __shared__ arrays with more than one dimension (e.g. \`__shared__ float tile[18][18]\`). Ripple's vtcm_malloc() returns a flat pointer, which can't be redeclared with a trailing [dim] the way a real array can, and every tile[y][x]-style indexing site in the kernel would need rewriting to flat pointer arithmetic — an AST-level transformation the current regex-based rule engine can't do safely. See docs/superpowers/specs/2026-08-14-update-to-current-ripple-api-design.md for context."
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

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_translation.py::TestBoilerplateGeneration::test_compile_flag_documented_in_output -v`
Expected: FAIL

- [ ] **Step 3: Add the note to the generated header comment**

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

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_translation.py::TestBoilerplateGeneration::test_compile_flag_documented_in_output -v`
Expected: PASS

- [ ] **Step 5: Add the note to README.md**

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

- [ ] **Step 6: Add the note to the CLI epilog**

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

- [ ] **Step 7: Add the note to the web UI status bar**

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

- [ ] **Step 8: Run the full test suite to check for regressions**

Run: `venv/bin/python -m pytest -v`
Expected: All pass.

- [ ] **Step 9: Commit**

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

## Task 6: Final end-to-end verification

**Files:** None modified — verification only.

- [ ] **Step 1: Run the full test suite**

Run: `venv/bin/python -m pytest -v`
Expected: All tests pass, 0 failures, 0 errors.

- [ ] **Step 2: Verify a real, previously-broken kernel now hard-fails correctly**

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

- [ ] **Step 3: Verify a real kernel using VTCM now compiles**

```bash
curl -s http://127.0.0.1:5001/translate -X POST -H "Content-Type: application/json" \
  -d '{"source": "__global__ void reduceSum(float *input, float *output, int n) {\n    __shared__ float sdata[256];\n    int tid = threadIdx.x;\n    sdata[tid] = input[tid];\n    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n        if (tid < s) sdata[tid] += sdata[tid + s];\n    }\n}"}' \
  | python3 -m json.tool
```

Expected: a JSON `translated` response containing `vtcm_malloc`, `vtcm_free`, and
`#define HVX_PE 0`, and NOT containing `__attribute__((section(".vtcm")))`.

- [ ] **Step 4: Verify the translated output actually syntax-checks**

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

- [ ] **Step 5: Stop the server**

```bash
pkill -f "venv/bin/python server.py"
```

- [ ] **Step 6: Final commit (only if Steps 1-4 needed any fixes)**

If everything passed cleanly, no commit is needed here — Tasks 1-5 already committed
everything. If verification surfaced a bug, fix it, re-run the full suite, and commit
with a message describing what Task 6 caught.
