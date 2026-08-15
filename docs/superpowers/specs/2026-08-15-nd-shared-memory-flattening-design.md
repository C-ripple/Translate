# N-dimensional `__shared__` arrays: flatten to real VTCM instead of hard-failing

## Context

The Ripple-API-migration work (2026-08-14) made `SharedMemoryRule` translate 1D
`__shared__` arrays to real `vtcm_malloc()`/`vtcm_free()` calls, but hard-fails on any
array with more than one dimension (`__shared__ float tile[16][32]`) — a
`vtcm_malloc()`-returned pointer can't be redeclared with a trailing `[dim]` the way a
real array can, and every `tile[y][x]`-style indexing site elsewhere in the kernel needs
rewriting to flat pointer arithmetic. That was explicitly scoped out at the time
(tracked as GitHub issue #12) since it required index-rewriting infrastructure this
codebase didn't have yet.

This spec adds that infrastructure: real N-dimensional flattening, replacing the
hard-fail for the common case, while still hard-failing anything the rewrite can't be
confident about.

Prompted by a suggestion to auto-generate a host-side grid-loop wrapper alongside this —
holding that out of scope for this spec. It's a much bigger architectural commitment (it
changes what the translator's output contract even is — a self-contained program vs. one
that expects a generated host wrapper) and nothing from Benoit's team has indicated
multi-block orchestration is actually blocking them yet. Revisit once there's real
signal it's needed.

## Scope

`SharedMemoryRule` in `core/translation_rules.py` only. No changes to
`DynamicSharedMemoryRule` (`extern __shared__`, unrelated — launch-time-sized, no
equivalent to translate to, unchanged since the original migration). No host-loop
generation. No IR-path changes.

## Design

### 1. Capture all declared dimensions, not just detect a second one

Current pattern captures exactly one bracket group:
```python
PATTERN = r'__shared__\s+(\w+)\s+(\w+)\s*\[([^\]]*)\]'
```

New pattern captures the full bracket chain as one group, parsed separately:
```python
PATTERN = r'__shared__\s+(\w+)\s+(\w+)\s*((?:\[[^\]]*\])+)'
```
`group(3)` is then split into individual dimension expressions via
`re.findall(r'\[([^\]]*)\]', group(3))` — e.g. `"[TILE_DIM][TILE_DIM + 1]"` →
`["TILE_DIM", "TILE_DIM + 1"]`. Using `[^\]]*` (not a bare-identifier pattern) is
required: real kernels commonly pad a dimension by an expression like `TILE_DIM + 1` to
avoid shared-memory bank conflicts (confirmed: this exact shape already exists in
`tests/examples/cuda_kernels.cu`'s `transpose` kernel).

If `len(dims) == 1`: existing 1D behavior, unchanged (see the note in section 5 on why
1D never needed usage-rewriting in the first place).

**Found on review — the non-hexagon branch also needs a matching update.**
`group(3)`'s meaning changes under the new pattern: today it's the bare inner content of
the single bracket (`"256"`); under the new pattern it's the *entire* bracket chain,
brackets included (`"[16][32]"`, or `"[256]"` for the 1D case — still no behavior change
there since `f"{var_name}[{size_expr}]"` and `f"{var_name}{size_expr}"` produce identical
text when `size_expr` already carries its own brackets). The non-hexagon branch
(`ctx.target_platform != "hexagon"`, unchanged VTCM-attribute-free behavior, out of scope
for this spec otherwise) currently does:
```python
decl = f"__attribute__((aligned(128))) {elem_type} {var_name}[{size_expr}]"
```
This must become:
```python
decl = f"__attribute__((aligned(128))) {elem_type} {var_name}{dims_group}"
```
(using the new group(3) directly, without re-wrapping in `[...]`) — otherwise a
multi-dimensional array on a non-Hexagon target would double-bracket into
`var_name[[16][32]]`, invalid syntax. This branch's own behavior (tolerate multi-dim
arrays via the attribute form, unchanged since before the original VTCM migration) is
not otherwise touched by this spec — this is purely keeping it correct under the new
capture-group shape.

### 2. N-dimensional case: allocation size

Each dimension gets its own parentheses before multiplying:
```python
total_size_expr = " * ".join(f"({d})" for d in dims)
```
Not `f"({dims[0]} * {dims[1]})"` — if a dimension is itself an expression
(`TILE_DIM + 1`), multiplying without per-term parens changes the arithmetic
(`TILE_DIM + 1 * TILE_DIM` ≠ `(TILE_DIM + 1) * TILE_DIM`, since `*` binds tighter than
`+` in C).

This formula subsumes the existing 1D case exactly: for `dims = ["256"]`,
`" * ".join(f"({d})" for d in dims)` produces `"(256)"`, byte-identical to what the
current code already emits (`sizeof({elem_type}) * ({size_expr})`). No separate 1D/N-D
branch is needed for the size computation — one formula covers both.

### 3. Before rewriting anything: verify every other usage is safe to flatten

Using the same `[match.end(), free_pos)` region already computed for the early-return
check (Task 4b), find every occurrence of the variable name as a whole word:
```python
re.finditer(r'\b' + re.escape(var_name) + r'\b', region)
```
For each occurrence, check whether it's immediately followed by exactly `len(dims)`
bracket groups and no more (a trailing `(?!\s*\[)` after the Nth group rules out an
over-indexed usage):
```python
usage_pattern = re.compile(
    re.escape(var_name) + r''.join(r'\s*\[([^\[\]]*)\]' for _ in dims) + r'(?!\s*\[)'
)
```
If **every** occurrence matches this shape, all usages are confirmed safe to rewrite. If
**any** occurrence doesn't (the array passed bare to a function, used in `sizeof(...)`,
indexed with a different number of brackets than declared), hard-fail via
`ctx.add_error()` *before mutating any text* — naming the variable and the unmatched
usage — and skip this declaration entirely (leave it untranslated, matching the existing
hard-fail-before-mutation discipline this rule already follows for the early-return and
unbalanced-brace cases).

This check is not just caution: `sizeof(tile)` after flattening `tile` to a pointer
would silently return a pointer size instead of the original array size if left
unrewritten — hard-failing here is the only *correct* choice, not merely the safe one.

### 4. Rewrite confirmed usages to flat row-major indexing

For confirmed usages, in right-to-left order within the region (so earlier usages'
offsets stay valid as later ones are rewritten — the same right-to-left discipline
`SharedMemoryRule.apply()` already uses across multiple `__shared__` declarations),
replace `var[i0][i1]...[iN-1]` with `var[flat]` where:
```
flat = i0*(d1*d2*...*dN-1) + i1*(d2*...*dN-1) + ... + iN-1
```
This is standard C row-major flattening — exactly what `tile[y][x]` already meant under
the hood for a real 2D array declared `tile[DIM_Y][DIM_X]`, so the rewrite is
semantics-preserving, not an approximation.

### 5. Reuse existing infrastructure unchanged

- The enclosing-block brace-counting scan (`_find_enclosing_brace_end`) and `vtcm_free()`
  placement: unchanged, applies identically to N-D declarations.
- The early-return leak check (`RETURN_PATTERN`, Task 4b): unchanged, runs before the
  N-D-specific usage scan.
- The outer right-to-left processing across multiple `__shared__` declarations in one
  kernel: unchanged.

**Why the usage-scan (section 3) is new logic only for N-D, not a gap in the existing 1D
path:** for a 1D array, `var[i]` on a pointer and `var[i]` on a real array compile to
identical code (`arr[i]` is `*(arr+i)` in C regardless of which declaration `arr` came
from) — so 1D usages never needed rewriting after `SharedMemoryRule` turned the
declaration into a pointer; the syntax was already correct by accident. That equivalence
breaks down at 2+ dimensions: `tile[y][x]` on a real 2D array means "get row `y`, then
index `x` into it," which has no meaning on a flat `float*` — hence needing the rewrite
this spec adds, and hence why it's scoped to `len(dims) > 1` only.

### 6. Close out the tracking issue

GitHub issue #12 ("Multi-dimensional `__shared__` arrays can't translate to VTCM") gets
closed as part of this work, referencing the commit that fixes it.

## Out of scope

- Host-side grid-loop generation (see Context above).
- `DynamicSharedMemoryRule` / `extern __shared__`.
- Arrays indexed by an expression containing nested brackets (e.g. `tile[arr[i]][j]`) —
  the index-capture pattern (`[^\[\]]*`) doesn't handle nested brackets; this is a known,
  narrow limitation shared with the existing 1D size-expression capture, not something
  this spec introduces. Falls under the "uncertain usage" hard-fail path if it doesn't
  cleanly match — not silently wrong, just not automatically rewritten.
- The VS Code extension's independent `LocalTranslator` (already hard-fails on any
  `__shared__` array today, 1D or N-D, and points users at the CLI/web UI — that's
  unchanged; duplicating this flattening algorithm a second time in TypeScript isn't
  worth it for the reasons already noted in that file's own comment).
