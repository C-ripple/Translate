# N-dimensional `__shared__` Array Flattening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `SharedMemoryRule`'s hard-fail on multi-dimensional `__shared__` arrays
with a real flattening rewrite to `vtcm_malloc`/`vtcm_free`, including rewriting every
`tile[y][x]`-style usage elsewhere in the kernel to flat row-major indexing — while still
hard-failing any usage the rewrite can't be confident about (passed by pointer, used in
`sizeof()`, wrong bracket count, nested-bracket indices).

**Architecture:** One method (`SharedMemoryRule.apply()` in `core/translation_rules.py`)
gets extended, reusing its existing brace-counting free-placement and early-return
leak-check unchanged. No new files, no new classes.

**Tech Stack:** Python 3.14, pytest, regex-based source-to-source translation. No new
dependencies.

**Spec:** `docs/superpowers/specs/2026-08-15-nd-shared-memory-flattening-design.md`

---

## Task 1: N-dimensional flattening

**Files:**
- Modify: `core/translation_rules.py` (`SharedMemoryRule`)
- Modify: `tests/test_translation.py` (repurpose one test, add several)
- Modify: `tests/test_complex_kernels.py` (`test_tiled_matmul` — revert to natural 2D
  CUDA syntax as a real-kernel integration test)

### Step 1: Write the failing tests

In `tests/test_translation.py`, replace `test_shared_memory_rule_multidim_hard_fails`
(its premise — that ANY multi-dimensional array hard-fails — is no longer true; a
well-formed one now flattens successfully) with these tests, added to
`TestTranslationRules` in the same location:

```python
    def test_shared_memory_rule_2d_flattens(self):
        """A 2D __shared__ array flattens to a single vtcm_malloc, and a
        real usage elsewhere in the kernel is rewritten to flat row-major
        indexing — exactly what the original 2D indexing already meant
        under the hood."""
        rule = SharedMemoryRule()
        ctx = TranslationContext(target_platform="hexagon")

        source = "void k() {\n    __shared__ float tile[16][32];\n    tile[3][5] = 1.0f;\n}"
        result = rule.apply(source, ctx)

        assert not ctx.has_errors(), ctx.errors
        assert "float *tile = vtcm_malloc(sizeof(float) * ((16) * (32)), /*align_as=*/128);" in result
        assert "tile[(3) * (32) + (5)] = 1.0f;" in result
        assert "vtcm_free(tile);" in result
        assert result.index("vtcm_malloc") < result.index("tile[(3)")
        assert result.index("tile[(3)") < result.index("vtcm_free")

    def test_shared_memory_rule_3d_flattens(self):
        """Flattening generalizes past 2D — this is genuinely N-dimensional,
        not a 2D special case."""
        rule = SharedMemoryRule()
        ctx = TranslationContext(target_platform="hexagon")

        source = "void k() {\n    __shared__ float vol[4][5][6];\n    vol[1][2][3] = 1.0f;\n}"
        result = rule.apply(source, ctx)

        assert not ctx.has_errors(), ctx.errors
        assert "vtcm_malloc(sizeof(float) * ((4) * (5) * (6))" in result
        assert "vol[(1) * ((5) * (6)) + (2) * (6) + (3)] = 1.0f;" in result
        assert "vtcm_free(vol);" in result

    def test_shared_memory_rule_nd_hard_fails_on_sizeof_usage(self):
        """A usage that doesn't match the declared dimensionality — here,
        sizeof(tile) — must hard-fail rather than guess. Left unrewritten,
        sizeof(tile) after flattening would silently return a pointer
        size instead of the original array size."""
        rule = SharedMemoryRule()
        ctx = TranslationContext(target_platform="hexagon")

        source = "void k() {\n    __shared__ float tile[16][32];\n    int s = sizeof(tile);\n}"
        result = rule.apply(source, ctx)

        assert ctx.has_errors()
        assert "tile" in ctx.errors[0]

    def test_shared_memory_rule_nd_hard_fails_on_bracket_count_mismatch(self):
        """A usage indexed with fewer brackets than declared (passing a
        row by reference, effectively) must hard-fail — this translator
        can't confirm the caller's expectations about that reference."""
        rule = SharedMemoryRule()
        ctx = TranslationContext(target_platform="hexagon")

        source = "void k() {\n    __shared__ float tile[16][32];\n    float *row = tile[3];\n}"
        result = rule.apply(source, ctx)

        assert ctx.has_errors()
        assert "tile" in ctx.errors[0]

    def test_shared_memory_rule_nd_multiple_declarations(self):
        """Two 2D __shared__ arrays in one kernel — exercises the
        right-to-left multi-match processing order with N-D flattening,
        not just the 1D case it was originally built for."""
        rule = SharedMemoryRule()
        ctx = TranslationContext(target_platform="hexagon")

        source = (
            "void k() {\n"
            "    __shared__ float tile_a[4][4];\n"
            "    __shared__ float tile_b[4][4];\n"
            "    tile_a[1][2] = tile_b[2][1];\n"
            "}"
        )
        result = rule.apply(source, ctx)

        assert not ctx.has_errors(), ctx.errors
        assert "vtcm_malloc(sizeof(float) * ((4) * (4))" in result
        assert "tile_a[(1) * (4) + (2)] = tile_b[(2) * (4) + (1)];" in result
        assert "vtcm_free(tile_a);" in result
        assert "vtcm_free(tile_b);" in result

    def test_shared_memory_rule_nonhexagon_2d_unchanged(self):
        """Non-Hexagon targets keep the pre-VTCM attribute-based behavior
        for multi-dimensional arrays too — must not double-bracket into
        invalid syntax like tile[[16][32]] now that the capture group
        includes the brackets themselves."""
        rule = SharedMemoryRule()
        ctx = TranslationContext(target_platform="x86")

        source = "void k() {\n    __shared__ float tile[16][32];\n}"
        result = rule.apply(source, ctx)

        assert not ctx.has_errors(), ctx.errors
        assert "__attribute__((aligned(128))) float tile[16][32]" in result
        assert "[[" not in result
```

Also update `tests/test_complex_kernels.py`'s `test_tiled_matmul` — it currently declares
its two tiles in CUDA source already manually pre-flattened to 1D
(`tile_A[TILE_SIZE * TILE_SIZE]`, indexed as `tile_A[ty * TILE_SIZE + tx]`), because
2D was a hard-fail when this test was written. Revert it to natural CUDA 2D syntax as a
real-kernel integration test of this feature — replace the whole method with:

```python
    def test_tiled_matmul(self):
        """Test tiled matrix multiplication with shared memory, using
        CUDA's natural 2D tile_A[TILE_SIZE][TILE_SIZE] syntax — this was
        hand-flattened to 1D when SharedMemoryRule only supported one
        dimension; now that it flattens 2D arrays automatically, this is
        a real-kernel integration test of that path, including two
        declarations in one kernel."""
        cuda_code = """
        #define TILE_SIZE 16

        __global__ void matmul_tiled(float *A, float *B, float *C, int N) {
            __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
            __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int row = blockIdx.y * TILE_SIZE + ty;
            int col = blockIdx.x * TILE_SIZE + tx;

            float sum = 0.0f;

            for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
                if (row < N && t * TILE_SIZE + tx < N)
                    tile_A[ty][tx] = A[row * N + t * TILE_SIZE + tx];
                else
                    tile_A[ty][tx] = 0.0f;

                if (col < N && t * TILE_SIZE + ty < N)
                    tile_B[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
                else
                    tile_B[ty][tx] = 0.0f;

                __syncthreads();

                for (int k = 0; k < TILE_SIZE; k++) {
                    sum += tile_A[ty][k] * tile_B[k][tx];
                }

                __syncthreads();
            }

            if (row < N && col < N) {
                C[row * N + col] = sum;
            }
        }
        """

        result = translate_cuda_source(cuda_code)

        # Verify shared memory translation to real VTCM malloc/free
        assert "vtcm_malloc(sizeof(float) * ((TILE_SIZE) * (TILE_SIZE))" in result
        assert "vtcm_free(tile_A);" in result
        assert "vtcm_free(tile_B);" in result
        # 2D usages rewritten to flat row-major indexing
        assert "tile_A[(ty) * (TILE_SIZE) + (tx)]" in result
        assert "tile_B[(ty) * (TILE_SIZE) + (tx)]" in result
        assert "tile_A[(ty) * (TILE_SIZE) + (k)]" in result
        assert "tile_B[(k) * (TILE_SIZE) + (tx)]" in result
        assert "ripple_id(ripple_block" in result
        # __syncthreads should be converted to comment (implicit in SIMD)
        assert "/* __syncthreads" in result or "__syncthreads" not in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_translation.py::TestTranslationRules::test_shared_memory_rule_2d_flattens tests/test_translation.py::TestTranslationRules::test_shared_memory_rule_3d_flattens tests/test_translation.py::TestTranslationRules::test_shared_memory_rule_nd_hard_fails_on_sizeof_usage tests/test_translation.py::TestTranslationRules::test_shared_memory_rule_nd_hard_fails_on_bracket_count_mismatch tests/test_translation.py::TestTranslationRules::test_shared_memory_rule_nd_multiple_declarations tests/test_translation.py::TestTranslationRules::test_shared_memory_rule_nonhexagon_2d_unchanged tests/test_complex_kernels.py::TestMatrixOperations::test_tiled_matmul -v`

Expected: FAIL — the flatten/hard-fail-on-bad-usage tests fail because current code
unconditionally hard-fails on ANY multi-dim array (`EXTRA_DIM_PATTERN`); the non-hexagon
test fails because current 1-bracket capture already produces correct (not yet broken)
output — it should currently PASS actually, since the bug it guards against doesn't
exist until Step 3's PATTERN change; note this and move on, it'll still be green after
Step 3 too. `test_tiled_matmul` fails because natural 2D syntax currently hard-fails.

- [ ] **Step 3: Rewrite `SharedMemoryRule`**

Replace the entire `SharedMemoryRule` class in `core/translation_rules.py` with:

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

    Multi-dimensional arrays (`__shared__ float tile[16][32]`) are
    flattened to a single vtcm_malloc() allocation, with every
    `tile[y][x]`-style usage elsewhere in the kernel rewritten to flat
    row-major indexing (`tile[(y) * (32) + (x)]`) — exactly what the
    original 2D indexing already meant under the hood, so the rewrite is
    semantics-preserving, not an approximation. A usage that doesn't
    cleanly match the declared dimensionality (passed bare to a
    function, used in sizeof(), indexed with the wrong number of
    brackets, or containing nested-bracket index expressions) is a
    hard-fail rather than a guess: e.g. sizeof(tile) after flattening
    would silently return a pointer size instead of the original array
    size if left unrewritten, so declining to guess there is the only
    correct choice, not merely the cautious one.

    1D arrays never needed this usage-rewriting in the first place:
    var[i] on a pointer and var[i] on a real array compile to identical
    code in C, so the syntax was already correct by accident once the
    declaration became a pointer. That equivalence breaks down at 2+
    dimensions, which is why this logic is scoped to len(dims) > 1.
    """

    # Array-form declarations only (requires trailing [...]) — scalar
    # `__shared__ float x;` is not matched or translated by this rule.
    # tests/test_translation.py reuses this pattern to detect leftover
    # untranslated declarations, so it inherits the same array-only scope.
    PATTERN = r'__shared__\s+(\w+)\s+(\w+)\s*((?:\[[^\]]*\])+)'
    RETURN_PATTERN = re.compile(r'\breturn\b')

    def __init__(self):
        super().__init__(
            name="shared_memory",
            description="Translate __shared__ array to vtcm_malloc/vtcm_free",
            cuda_pattern=self.PATTERN,
            priority=90
        )

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
        # while later matches are rewritten (the declaration itself, the
        # vtcm_free() insertion, and now N-D usage rewrites all grow the
        # string).
        for match in reversed(matches):
            elem_type = match.group(1)
            var_name = match.group(2)
            dims_group = match.group(3)
            dims = re.findall(r'\[([^\]]*)\]', dims_group)

            ctx.shared_mem_mappings[var_name] = elem_type

            if ctx.target_platform != "hexagon":
                # No VTCM claim to fix outside Hexagon — unchanged behavior.
                # dims_group already carries its own brackets (e.g.
                # "[16][32]"), so it's spliced directly, not re-wrapped —
                # re-wrapping would double-bracket into tile[[16][32]].
                decl = f"__attribute__((aligned(128))) {elem_type} {var_name}{dims_group}"
                result = result[:match.start()] + decl + result[match.end():]
                continue

            free_pos = self._find_enclosing_brace_end(result, match.end())
            if free_pos == -1:
                ctx.add_error(
                    f"SharedMemoryRule: could not find the closing brace of "
                    f"the block containing '__shared__ {elem_type} "
                    f"{var_name}{dims_group}' — cannot place its matching "
                    f"vtcm_free() call. Check for unbalanced braces in the "
                    f"kernel."
                )
                continue

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

            if len(dims) > 1:
                usage_region = result[match.end():free_pos]
                usage_pattern = re.compile(
                    re.escape(var_name)
                    + ''.join(r'\s*\[([^\[\]]*)\]' for _ in dims)
                    + r'(?!\s*\[)'
                )
                name_pattern = re.compile(r'\b' + re.escape(var_name) + r'\b')

                usages = []
                bad_usage = None
                for name_match in name_pattern.finditer(usage_region):
                    usage_match = usage_pattern.match(usage_region, name_match.start())
                    if usage_match is None:
                        bad_usage = usage_region[
                            name_match.start():name_match.start() + 40
                        ]
                        break
                    usages.append(usage_match)

                if bad_usage is not None:
                    ctx.add_error(
                        f"SharedMemoryRule: '{var_name}' (declared as "
                        f"{elem_type} {var_name}{dims_group}, "
                        f"{len(dims)} dimensions) is used in a way that "
                        f"doesn't match its declared dimensionality — "
                        f"near '{bad_usage}...'. This could be a bare "
                        f"reference (passed to a function, used in "
                        f"sizeof()), a different number of brackets than "
                        f"declared, or an index expression containing "
                        f"nested brackets. This translator can't confirm "
                        f"the flattened rewrite would be correct there — "
                        f"rewrite '{var_name}' to a 1D array with manual "
                        f"index arithmetic and retranslate."
                    )
                    continue

                # Right-to-left within the usage region too, for the same
                # offset-safety reason as the outer declaration loop.
                for usage_match in reversed(usages):
                    indices = usage_match.groups()
                    terms = []
                    for i, idx_expr in enumerate(indices):
                        stride_dims = dims[i + 1:]
                        if stride_dims:
                            stride = " * ".join(f"({d})" for d in stride_dims)
                            if len(stride_dims) > 1:
                                stride = f"({stride})"
                            terms.append(f"({idx_expr}) * {stride}")
                        else:
                            terms.append(f"({idx_expr})")
                    flat_index = " + ".join(terms)
                    replacement = f"{var_name}[{flat_index}]"
                    usage_region = (
                        usage_region[:usage_match.start()]
                        + replacement
                        + usage_region[usage_match.end():]
                    )

                result = result[:match.end()] + usage_region + result[free_pos:]
                # free_pos shifted by however much the usage rewrites
                # changed the region's length.
                free_pos = match.end() + len(usage_region)

            free_call = f"\n    vtcm_free({var_name});"
            result = result[:free_pos] + free_call + result[free_pos:]

            if len(dims) > 1:
                total_size_expr = " * ".join(f"({d})" for d in dims)
            else:
                total_size_expr = dims[0]
            malloc_decl = (
                f"// CUDA __shared__ -> Ripple VTCM\n"
                f"    {elem_type} *{var_name} = vtcm_malloc("
                f"sizeof({elem_type}) * ({total_size_expr}), /*align_as=*/128)"
            )
            result = result[:match.start()] + malloc_decl + result[match.end():]

        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_translation.py::TestTranslationRules::test_shared_memory_rule_2d_flattens tests/test_translation.py::TestTranslationRules::test_shared_memory_rule_3d_flattens tests/test_translation.py::TestTranslationRules::test_shared_memory_rule_nd_hard_fails_on_sizeof_usage tests/test_translation.py::TestTranslationRules::test_shared_memory_rule_nd_hard_fails_on_bracket_count_mismatch tests/test_translation.py::TestTranslationRules::test_shared_memory_rule_nd_multiple_declarations tests/test_translation.py::TestTranslationRules::test_shared_memory_rule_nonhexagon_2d_unchanged tests/test_complex_kernels.py::TestMatrixOperations::test_tiled_matmul -v`

Expected: PASS. The nested-parens shape in Step 3's code only wraps a stride/size
product in an extra outer paren when it has *more than one* term — a single-term
stride/size stays unwrapped (`stride = "(6)"`, not `"((6))"`; `total_size_expr = "256"`
for 1D, substituted into the single pair of parens the malloc f-string already provides).
This is what keeps the existing 1D tests (`test_shared_memory_rule`,
`test_shared_memory_vtcm_free_placement`) passing unchanged — a uniform "always add one
more wrap" formula would double-parenthesize those and break them. If a test fails on
the exact expected string, re-derive it by hand from this two-case rule rather than
"simplifying" the implementation to match a different string.

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `venv/bin/python -m pytest -v`
Expected: All tests pass, 0 failures. (149 before this task; net change: removed 1 test
[`test_shared_memory_rule_multidim_hard_fails`], added 6 → 154 total. `test_tiled_matmul`
still passes, now exercising real 2D syntax instead of hand-flattened 1D.)

- [ ] **Step 6: Add a real-syntax-check test**

Add to `TestBoilerplateGeneration` in `tests/test_translation.py` (reuse the existing
`@requires_clang`/`verify_ripple_syntax` pattern already used elsewhere in this file):

```python
    @requires_clang
    def test_2d_shared_memory_output_passes_syntax_check(self):
        cuda_code = """
__global__ void k(float *out) {
    __shared__ float tile[4][4];
    int y = threadIdx.y;
    int x = threadIdx.x;
    tile[y][x] = 1.0f;
    out[y * 4 + x] = tile[y][x];
}
"""
        result = translate_cuda_source(cuda_code)
        success, output = verify_ripple_syntax(result)
        assert success, output
```

Run: `venv/bin/python -m pytest tests/test_translation.py::TestBoilerplateGeneration::test_2d_shared_memory_output_passes_syntax_check -v`
Expected: PASS — confirms the flattened output actually compiles against the stub
header, not just that it contains the right substrings.

- [ ] **Step 7: Run the full test suite one more time**

Run: `venv/bin/python -m pytest -v`
Expected: All tests pass, 0 failures.

- [ ] **Step 8: Close the tracking issue**

```bash
gh issue close 12 --repo C-ripple/Translate --comment "Fixed: SharedMemoryRule now flattens N-dimensional __shared__ arrays to a single vtcm_malloc() allocation, rewriting every usage elsewhere in the kernel to flat row-major indexing. Usages that don't cleanly match the declared dimensionality (passed by pointer, used in sizeof(), wrong bracket count, nested-bracket indices) still hard-fail with a diagnostic rather than risk incorrect output. See docs/superpowers/specs/2026-08-15-nd-shared-memory-flattening-design.md."
```

If this fails (e.g. `gh` not authenticated), note it in your report and move on — don't
block on it.

- [ ] **Step 9: Commit**

```bash
git add core/translation_rules.py tests/test_translation.py tests/test_complex_kernels.py
git commit -m "Flatten N-dimensional __shared__ arrays to real VTCM instead of hard-failing

SharedMemoryRule previously hard-failed on any __shared__ array with
more than one dimension (tracked as GitHub issue #12, now closed).
This adds the index-rewriting infrastructure that was missing at the
time: every tile[y][x]-style usage elsewhere in the kernel is now
rewritten to flat row-major indexing (tile[(y)*(32)+(x)]) — exactly
what the original 2D indexing already meant under the hood, so the
rewrite is semantics-preserving, not an approximation.

A usage that doesn't cleanly match the declared dimensionality (passed
bare to a function, used in sizeof(), wrong bracket count, or an index
expression containing nested brackets) still hard-fails rather than
guessing — sizeof(tile) after flattening would silently return a
pointer size instead of the array size if left unrewritten, so
declining to guess there is the only correct choice.

Reuses the existing brace-counting free-placement and early-return
leak-check unchanged. Also fixes the non-hexagon branch, which needed
a matching update for the new capture-group shape (group(3) now
includes the brackets themselves) to avoid double-bracketing into
invalid syntax.

test_tiled_matmul reverted from its hand-flattened 1D workaround back
to natural CUDA 2D tile syntax, now serving as a real-kernel
integration test of this path."
```

---

## Self-Review Notes (for whoever executes this plan)

**Spec coverage:** all 6 numbered design points from the spec are covered — capture-group
change + non-hexagon fix (Step 3), allocation-size formula (Step 3), usage verification
before mutation (Step 3), flattening rewrite (Step 3), reuse of existing infrastructure
(Step 3, unchanged), closing the tracking issue (Step 8).

**Placeholder scan:** no TBD/TODO. Every test has concrete expected strings, derived by
hand-tracing the algorithm in the spec review, not left for the implementer to guess.

**Type/signature consistency:** `SharedMemoryRule.PATTERN`, `RETURN_PATTERN`,
`_find_enclosing_brace_end` — all reused with identical names/signatures from the
existing class; `EXTRA_DIM_PATTERN` is removed (no longer needed — its job is now done
by the usage-verification logic, which is strictly more capable: it not only detects
"more than one dimension" but confirms every usage is safe to rewrite before proceeding).
