# Translator Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the AST parser's infinite-loop bug and wire it into the source-level translator as a structural safety net, stand up real compile verification against the actual Hexagon toolchain, and rebuild the test corpus around real kernels validated by that compile step — replacing self-referential string-assertion tests with tests that prove the translator's output is actually correct.

**Architecture:** Three independent, sequentially-dependent-in-value (but not code-dependent) hardening efforts. Priority 1 fixes and (partially) activates the already-half-built AST parser in `frontends/source/cuda_frontend.py`. Priority 2 adds an opt-in, Docker-based ground-truth compiler check that the fast everyday test suite never depends on. Priority 3 spends that compile-verification capability on the real (currently untracked-by-tests) sample kernels already sitting in the repo. Priorities 2 and 3 have a real dependency (3 needs 2's helper); Priority 1 is independent of both.

**Tech Stack:** Python 3.13 (existing stack), pytest + pytest-timeout, Docker (for the vendored Hexagon toolchain image built from upstream `qualcomm/learn-ripple`'s `containers/Dockerfile`).

---

## Before you start

All three priorities were scoped from empirical investigation done in the prior session, not guesswork:

- The AST parser's hang was reproduced and root-caused (see Priority 1) — it is not a hypothetical bug.
- The Dockerfile's stage structure and the toolchain's install path (`/opt/hexagon`) were read from the actual vendored file (see Priority 2) — not assumed.
- One assumption is *not* independently verified and is called out explicitly where it matters: that the toolchain's installed `clang` resolves `<ripple.h>` from its own builtin resource directory without us supplying a header. This is well-supported by the docs (`temp_ripple_docs/src/ripple-spec/niy.md`: "memory IDs are defined in the machine model and ripple.h" — i.e. it ships with the Ripple-aware compiler, the same way `<omp.h>`-style compiler intrinsics ship with compilers that support them) but the plan's first live compile in Priority 2 is the actual test of this assumption, and includes a diagnosis step for if it's wrong.

---

## Priority 1: Fix the AST parser's infinite loop, wire it as a structural safety net

### File Structure

- Modify: `frontends/source/cuda_frontend.py` — `_parse_block()` (the bug) and `CUDAToRIPPLETransformer.transform()` (the wiring)
- Create: `tests/test_ast_parsing.py` — hang-regression tests and structural-validation tests
- Modify: `requirements.txt`, `pyproject.toml` — add `pytest-timeout` (needed to make a hang *fail fast* instead of hanging the test run itself)

### Task 1: Add pytest-timeout and write a failing hang-regression test

**Files:**
- Modify: `requirements.txt`
- Modify: `pyproject.toml`
- Create: `tests/test_ast_parsing.py`

- [ ] **Step 1: Add the dependency**

In `requirements.txt`, add this line under the `# Development dependencies` section (after `pytest-cov>=4.0.0`):

```
pytest-timeout>=2.3.0
```

In `pyproject.toml`, add the same to the `dev` optional-dependencies list:

```toml
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-timeout>=2.3.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
]
```

- [ ] **Step 2: Install it**

Run: `source venv/bin/activate && pip install pytest-timeout>=2.3.0`
Expected: `Successfully installed pytest-timeout-...`

- [ ] **Step 3: Write the failing test**

Create `tests/test_ast_parsing.py`:

```python
"""
Tests for the AST-based parser in frontends/source/cuda_frontend.py
(CUDALexer + AIRBuilder), independent of the regex-based translation path.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from frontends.source.cuda_frontend import CUDALexer, AIRBuilder
from core.semantic_model import TranslationContext


def _parse(source: str):
    """Run the full lex + AST-build pipeline and return the resulting translation unit."""
    lexer = CUDALexer(source)
    tokens = lexer.tokenize()
    builder = AIRBuilder(tokens, TranslationContext())
    return builder.build_translation_unit()


# Brace-nesting depth 0: no braces beyond the function body itself.
KERNEL_FLAT = """
__global__ void flat(float *a) {
    int i = threadIdx.x;
    a[i] = 0;
}
"""

# Brace-nesting depth 1: a single-level nested block (an if-body with braces).
KERNEL_ONE_NESTED = """
__global__ void oneNested(float *a, int n) {
    int i = threadIdx.x;
    if (i < n) {
        a[i] = 0;
    }
}
"""

# Brace-nesting depth 2: matches the exact shape that hung before the fix —
# a for-loop body containing an if-body (test_reduction.cu in the repo).
KERNEL_TWO_NESTED = """
__global__ void reduce(float *val) {
    float sum = *val;
    for (int offset = 16; offset > 0; offset /= 2) {
        if (offset > 0) {
            sum += offset;
        }
    }
    *val = sum;
}
"""


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "source,expected_function_count",
    [
        (KERNEL_FLAT, 1),
        (KERNEL_ONE_NESTED, 1),
        (KERNEL_TWO_NESTED, 1),
    ],
)
def test_parses_without_hanging(source, expected_function_count):
    unit = _parse(source)
    assert len(unit.functions) == expected_function_count
    assert unit.functions[0].is_kernel
```

- [ ] **Step 4: Run it to confirm the hang reproduces**

Run: `source venv/bin/activate && python -m pytest tests/test_ast_parsing.py -v --timeout=5`
Expected: `test_parses_without_hanging[KERNEL_FLAT-1]` PASSES; `[KERNEL_ONE_NESTED-1]` and `[KERNEL_TWO_NESTED-1]` both FAIL with `Failed: Timeout >5.0s`.

(Correction from initial plan authoring: the root cause is "any nested brace hangs the loop," not "hangs past some nesting depth" — the original root-cause investigation only empirically tested 0-nested and 2-nested samples, never exactly 1-nested, and this line originally predicted `KERNEL_ONE_NESTED` would pass. It doesn't. Confirmed by actually running this step during Task 1's implementation — the fix in Task 2 is unaffected since it addresses the mechanism, not a specific depth.)

### Task 2: Fix the root cause in `_parse_block`

**Files:**
- Modify: `frontends/source/cuda_frontend.py:834-866`

**Root cause:** `_parse_block()`'s outer `while brace_count > 0:` loop increments `brace_count` on `TokenType.LBRACE` and decrements on `TokenType.RBRACE`, but never calls `self.advance()` in either branch. When a nested `{` or `}` is encountered (anything beyond the function's own opening brace), the token position never moves past it, so the loop re-observes the same `LBRACE` token forever, incrementing `brace_count` without bound. (The `RBRACE` branch doesn't hang — repeatedly decrementing the same frozen token eventually crosses zero and the outer condition exits — but it's the same missing-`advance()` bug and produces incorrect double-counting; both must be fixed together.)

- [ ] **Step 1: Apply the fix**

Replace the full method:

```python
    def _parse_block(self) -> list[AIRNode]:
        """Parse a block of statements."""
        body = []
        
        self.expect(TokenType.LBRACE)
        brace_count = 1
        
        while brace_count > 0:
            if self.current().type == TokenType.LBRACE:
                brace_count += 1
            elif self.current().type == TokenType.RBRACE:
                brace_count -= 1
            elif self.current().type == TokenType.EOF:
                break
            
            if brace_count > 0:
                # Collect statement as expression
                stmt_tokens = []
                while self.current().type not in (TokenType.SEMICOLON, TokenType.LBRACE, 
                                                   TokenType.RBRACE, TokenType.EOF):
                    stmt_tokens.append(self.advance())
                
                if stmt_tokens:
                    expr = ' '.join(t.value for t in stmt_tokens)
                    body.append(AIRExpression(expr=expr))
                
                if self.current().type == TokenType.SEMICOLON:
                    self.advance()
        
        if self.current().type == TokenType.RBRACE:
            self.advance()
        
        return body
```

with:

```python
    def _parse_block(self) -> list[AIRNode]:
        """Parse a block of statements."""
        body = []
        
        self.expect(TokenType.LBRACE)
        brace_count = 1
        
        while brace_count > 0:
            if self.current().type == TokenType.LBRACE:
                brace_count += 1
                self.advance()
                continue
            elif self.current().type == TokenType.RBRACE:
                brace_count -= 1
                self.advance()
                continue
            elif self.current().type == TokenType.EOF:
                break
            
            # Collect statement as expression
            stmt_tokens = []
            while self.current().type not in (TokenType.SEMICOLON, TokenType.LBRACE, 
                                               TokenType.RBRACE, TokenType.EOF):
                stmt_tokens.append(self.advance())
            
            if stmt_tokens:
                expr = ' '.join(t.value for t in stmt_tokens)
                body.append(AIRExpression(expr=expr))
            
            if self.current().type == TokenType.SEMICOLON:
                self.advance()
        
        return body
```

(The trailing `if self.current().type == TokenType.RBRACE: self.advance()` is removed — it's now dead code, since the loop itself consumes the final closing brace via the `RBRACE` branch above before exiting.)

- [ ] **Step 2: Run the new tests to verify the fix**

Run: `source venv/bin/activate && python -m pytest tests/test_ast_parsing.py -v --timeout=5`
Expected: all 3 `test_parses_without_hanging` cases PASS, including `KERNEL_TWO_NESTED`

- [ ] **Step 3: Run the full suite to confirm no regressions**

Run: `source venv/bin/activate && python -m pytest tests/ -v`
Expected: all tests pass (43 existing + 3 new = 46 passed)

- [ ] **Step 4: Commit**

```bash
git add frontends/source/cuda_frontend.py tests/test_ast_parsing.py requirements.txt pyproject.toml
git commit -m "Fix AST parser infinite loop on nested braces

_parse_block()'s brace-tracking loop incremented/decremented
brace_count on LBRACE/RBRACE without ever advancing the token
position, so any nested block (a for/if body — i.e. almost any
real kernel) re-observed the same LBRACE token forever. Verified
by reproducing the hang against every existing sample .cu file:
every file with more than one brace pair hung, every flat one
didn't. Advance the token in both branches; the now-dead trailing
RBRACE check is removed since the loop consumes it directly.

Added pytest-timeout so a future regression here fails fast
instead of hanging the test run."
```

### Task 3: Wire the (now-safe) AST parse into the translator as a structural validation pre-pass

This does **not** make the AST drive code generation — that's a substantially larger effort (real expression/statement tree walking, not the flat per-statement text blobs `_parse_block` currently produces) and is out of scope here. What this task restores is the integration point that was scaffolded in the original code but never finished (the `transform()` method had this exact AST call, commented out, with a trailing `# Future: Generate code from AST 'translation_unit'` that was never acted on even when the code was live). The value: the AST now runs on every real translation, catches structurally malformed input, and — since `AIRFunction` correctly enumerates every kernel by parsing signatures rather than regex-scanning raw text — gives an early, structurally-grounded signal about what's actually in the file, surfaced as a warning when something looks off.

**Files:**
- Modify: `frontends/source/cuda_frontend.py` — `CUDAToRIPPLETransformer.transform()`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ast_parsing.py`:

```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from frontends.source.cuda_frontend import translate_cuda_source
from core.semantic_model import TranslationContext
from frontends.source.cuda_frontend import CUDAToRIPPLETransformer


def test_transform_warns_when_ast_finds_no_kernels():
    # No __global__ function in this source at all.
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    transformer.transform("int add(int a, int b) { return a + b; }")
    assert any("no __global__ kernel" in w for w in ctx.warnings)


def test_transform_does_not_warn_on_valid_kernel():
    ctx = TranslationContext()
    transformer = CUDAToRIPPLETransformer(ctx)
    transformer.transform(KERNEL_TWO_NESTED)
    assert not any("AST pre-pass" in w for w in ctx.warnings)
```

- [ ] **Step 2: Run to verify it fails**

Run: `source venv/bin/activate && python -m pytest tests/test_ast_parsing.py::test_transform_warns_when_ast_finds_no_kernels -v`
Expected: FAIL — `ctx.warnings` is empty (the AST pre-pass isn't wired in yet)

- [ ] **Step 3: Wire it in**

In `frontends/source/cuda_frontend.py`, in `CUDAToRIPPLETransformer.transform()`, replace this block:

```python
        # Lexical Analysis (temporarily disabled - causes hang on complex kernels)
        # lexer = CUDALexer(cuda_source)
        # tokens = lexer.tokenize()
        
        # AST Parsing (Intermediate Representation)
        # builder = AIRBuilder(tokens, self.ctx)
        # try:
        #     translation_unit = builder.build_translation_unit()
        #     # detected_kernels = [f for f in translation_unit.functions if f.is_kernel]
        #     # if detected_kernels:
        #     #     # Future: Generate code from AST 'translation_unit'
        #     #     pass
        # except Exception as e:
        #     self.ctx.add_warning(f"AST Parsing failed, falling back to Regex: {e}")
            
        # -- Legacy Regex Transformation (Current Working Path) --
```

with:

```python
        # AST pre-pass: structural validation and kernel detection.
        # This does not drive code generation — it exists to catch malformed
        # input and enumerate kernels structurally (by parsing signatures,
        # not regex-scanning raw text) before the regex pass runs, and to
        # surface a warning instead of silently mistranslating when the
        # parser can't make sense of the source.
        try:
            lexer = CUDALexer(cuda_source)
            tokens = lexer.tokenize()
            builder = AIRBuilder(tokens, self.ctx)
            translation_unit = builder.build_translation_unit()
            detected_kernels = [f.name for f in translation_unit.functions if f.is_kernel]
            if not detected_kernels:
                self.ctx.add_warning(
                    "AST pre-pass found no __global__ kernel functions in the source; "
                    "translation will proceed via regex rules but the result may be incomplete."
                )
        except Exception as e:
            self.ctx.add_warning(f"AST pre-pass failed, proceeding with regex-only translation: {e}")

        # -- Regex Transformation (current translation path; AST does not drive codegen yet) --
```

- [ ] **Step 4: Run both new tests to verify they pass**

Run: `source venv/bin/activate && python -m pytest tests/test_ast_parsing.py -v --timeout=5`
Expected: all tests PASS (5 total in this file)

- [ ] **Step 5: Run the full suite**

Run: `source venv/bin/activate && python -m pytest tests/ -v`
Expected: all pass (48 total)

- [ ] **Step 6: Commit**

```bash
git add frontends/source/cuda_frontend.py tests/test_ast_parsing.py
git commit -m "Wire AST parsing into transform() as a structural validation pass

Restores the integration point that existed in the original code,
commented out, and was never finished even when live — it parsed
the AST and immediately discarded the result. Now every translation
runs the (now hang-safe) AST parser and surfaces a ctx.warnings
entry when it can't find a kernel or can't parse the input at all,
instead of silently proceeding with whatever the regex rules
happen to match.

Does not change regex-path output for well-formed input — this is
diagnostic only. AST-driven code generation remains future work;
_parse_block() produces flat per-statement text, not a real
expression tree, so there isn't yet a structure to generate from."
```

**Note:** this warning is already surfaced by the CLI (`interfaces/cli/cuda2ripple.py:118-120`) and the standalone web UI (`interfaces/web/server.py:524`), but not by the root `server.py` the Flutter app talks to — that's tracked separately as backlog issue #5 (the two-servers question), not part of this plan.

---

## Priority 2: Compile verification (revised: lightweight syntax check, not the full Hexagon toolchain)

**Revision note (post Task 4):** the original version of this priority built a full Hexagon-targeting RIPPLE toolchain from source via Docker (30-90+ minutes, heavy). Reconsidered mid-execution: the translator's actual job stops at emitting RIPPLE C source — it doesn't compile or run Hexagon binaries itself — and every real bug found in this session so far was caught by grammar/logic reasoning and targeted tests, not by compiling anything. At this project's current maturity (alpha, narrow rule coverage, two more real bugs found below), a full semantic/HVX-codegen compile check is disproportionate to what's actually been needed. Replaced with a lightweight syntax-only check using the system's already-installed `clang` (no Docker, no build) — enough to catch gross syntax breakage (which, it turns out, exists — see Task 5). The vendored Dockerfile from Task 4 is kept as-is (already committed, already reviewed) as ready-made infrastructure for whenever RIPPLE-semantic correctness, not syntax, becomes the actual bottleneck — tracked as a deferred backlog item, not deleted.

**Two real, verified bugs found while scoping this revision** (both from cross-referencing the translator's own output against the real upstream RIPPLE API spec and plain C syntax rules, not from running any compiler yet):
1. `blockDim.x/y/z` translates to a call to `ripple_get_size`, which doesn't exist anywhere in the real RIPPLE API (the real function is `ripple_get_block_size`) — fixed in Task 5 below, small and immediate.
2. All four warp-shuffle translation rules (`__shfl_xor_sync`, `__shfl_up_sync`, `__shfl_sync`, `__shfl_down_sync`) emit output that isn't valid C at all (C++ lambda expressions, or a nested function definition inside another function's body) — filed as [GitHub issue #8](https://github.com/C-ripple/Translate/issues/8), deliberately deferred (needs real design work: how to represent a per-call-site closure in valid C), not fixed as part of this plan. The lightweight syntax check built in Task 6 will correctly fail on this — handled explicitly via `xfail` in Task 8, not silently skipped or masked.

### File Structure

- Create: `docker/hexagon-toolchain.Dockerfile` — vendored copy of upstream `learn-ripple`'s `containers/Dockerfile` (done, Task 4 — kept for deferred future use, not wired into the active checks below)
- Create: `docker/README.md` — provenance, build instructions, what it's for (done, Task 4)
- Create: `scripts/build-hexagon-toolchain.sh` — one-time build helper (done, Task 4)
- Modify: `core/translation_rules.py` — fix the `ripple_get_size` → `ripple_get_block_size` bug (Task 5)
- Modify: `tests/test_translation.py` — update the 3 assertions that currently assert the wrong name (Task 5)
- Modify: `README.md`, `docs/README.md`, `docs/ARCHITECTURE.md` — same wrong name is documented as correct in 3 places (Task 5)
- Modify: `frontends/ir/ir_frontend.py` — same wrong name used in the IR path's `NVPTX_TO_RIPPLE` map (Task 5)
- Create: `tests/stub_headers/ripple.h` — minimal, real-API-accurate stub header so a generic clang can resolve `#include <ripple.h>` without the full toolchain (Task 6)
- Create: `tests/compile_verify.py` — the reusable syntax-verification helper (Task 6)
- Create: `tests/test_compile_verification.py` — smoke test for the helper itself (Task 6)

### Task 4: Vendor the Hexagon toolchain Dockerfile

**Files:**
- Create: `docker/hexagon-toolchain.Dockerfile`
- Create: `docker/README.md`

- [ ] **Step 1: Copy the real Dockerfile from the local learn-ripple checkout**

```bash
mkdir -p docker
cp temp_ripple_docs/containers/Dockerfile docker/hexagon-toolchain.Dockerfile
```

- [ ] **Step 2: Add a provenance header**

At the very top of `docker/hexagon-toolchain.Dockerfile`, before the first line (`# =====...` / `# Stage 1: Builder...`), insert:

```dockerfile
# Vendored from https://github.com/qualcomm/learn-ripple containers/Dockerfile
# at commit 0dc48ff (2026-07-11). Used here only through the `toolchain`
# build stage (`docker build --target toolchain`) — the later `test-suite`
# stage (clones+builds the separate ripple-test-suite repo, builds mdbook
# docs) is irrelevant to compile verification and is skipped automatically
# by targeting `toolchain`, since Docker only builds a stage's dependencies.
# Do not hand-edit below this line — replace wholesale on re-vendor so this
# stays a faithful, diffable copy of upstream.
#
```

- [ ] **Step 3: Document it**

Create `docker/README.md`:

```markdown
# Hexagon toolchain (compile verification)

`hexagon-toolchain.Dockerfile` is a vendored copy of the official Hexagon/RIPPLE
toolchain build from `qualcomm/learn-ripple` (see the provenance comment at the
top of the file for the exact commit). It builds a real `clang` targeting
`hexagon-unknown-unknown-elf` with HVX and RIPPLE support, cloned and built
from `qualcomm/ripple` (the RIPPLE-enabled LLVM fork) — not a mock.

## Build (once)

```bash
./scripts/build-hexagon-toolchain.sh
```

This is a **heavy** build: full LLVM/clang from source, plus the Hexagon SDK
download and the ELD linker. Expect it to take well over 30 minutes and use
several GB of disk on first run. It only needs to be run once — the resulting
image (`cuda2ripple-hexagon-toolchain:latest`) is cached locally by Docker
and reused by every test run after that.

## What it's for

`tests/compile_verify.py` uses this image to actually compile translated
RIPPLE C output through a real Hexagon-targeting compiler, rather than
trusting string-based assertions about what the translator's own output
looks like. Tests using it (see `tests/test_compile_verification.py` and
the real-kernel tests) skip automatically — not fail — when this image
hasn't been built locally, so the normal `pytest tests/` run never depends
on Docker being installed or this image existing.

## Re-vendoring

If upstream's Dockerfile changes in a way that matters (e.g. the toolchain
install path or clang binary name moves), re-copy it wholesale from a fresh
`learn-ripple` checkout and update the provenance comment's commit hash —
don't hand-patch the vendored copy, since drift makes future re-vendoring
a manual diff nightmare instead of a straight copy.
```

- [ ] **Step 4: Create the build script**

Create `scripts/build-hexagon-toolchain.sh`:

```bash
#!/usr/bin/env bash
# Builds the Hexagon toolchain Docker image used for compile verification
# (see docker/README.md). Heavy build — full LLVM/clang from source, expect
# 30-90+ minutes and several GB of disk on first run. Run once; cached after.
set -euo pipefail
cd "$(dirname "$0")/.."

docker build \
  --target toolchain \
  -t cuda2ripple-hexagon-toolchain:latest \
  -f docker/hexagon-toolchain.Dockerfile \
  .

echo ""
echo "Built cuda2ripple-hexagon-toolchain:latest"
echo "Verify: docker run --rm cuda2ripple-hexagon-toolchain:latest /opt/hexagon/bin/clang --version"
```

Run: `chmod +x scripts/build-hexagon-toolchain.sh`

- [ ] **Step 5: Commit**

```bash
git add docker/ scripts/build-hexagon-toolchain.sh
git commit -m "Vendor Hexagon toolchain Dockerfile for compile verification

Copied verbatim from qualcomm/learn-ripple containers/Dockerfile
(commit 0dc48ff) rather than hand-authored, so it stays a faithful,
diffable copy of the real upstream build. Builds through the
toolchain stage only (--target toolchain) — the test-suite stage
clones and builds an unrelated separate repo and isn't needed here.

Not built automatically by anything yet — that's the next task."
```

### Task 5: Fix `ripple_get_size` → `ripple_get_block_size` (wrong RIPPLE API function name)

**Root cause:** `core/translation_rules.py`'s `BlockDimRule` translates CUDA's `blockDim.x/y/z` to a call named `ripple_get_size`. This function does not exist anywhere in the real RIPPLE API — confirmed via `grep -rn "ripple_get_size" temp_ripple_docs/` (zero hits) versus the real function documented in `temp_ripple_docs/src/ripple-spec/api.md`: `size_t ripple_get_block_size(ripple_block_t block_shape, int dim);`. `blockDim` is used in nearly every real CUDA kernel (grid-stride loops, tiling), so this breaks compilation broadly. The wrong name is also propagated into the IR-level translation path and documented as correct in three places.

**Files:**
- Modify: `core/translation_rules.py` — the actual rule (4 occurrences: a doc comment, a docstring, a `description=` string, and the generated code itself)
- Modify: `frontends/ir/ir_frontend.py` — `NVPTX_TO_RIPPLE` map (3 occurrences: `NTID_X`/`NTID_Y`/`NTID_Z`)
- Modify: `tests/test_translation.py` — 3 assertions currently asserting the wrong name
- Modify: `README.md`, `docs/README.md`, `docs/ARCHITECTURE.md` — the translation mapping table/examples document the wrong name as correct

- [ ] **Step 1: Confirm current occurrences**

Run: `grep -rn "ripple_get_size" --include="*.py" --include="*.md" . | grep -v temp_ripple_docs | grep -v __pycache__`
Expected: matches in exactly the 6 files listed above (this confirms the fix's scope before touching anything — if it finds a 7th file, investigate before proceeding, the fix needs to cover everywhere the wrong name appears).

- [ ] **Step 2: Fix the rule**

In `core/translation_rules.py`, replace every occurrence of `ripple_get_size` with `ripple_get_block_size`. This includes:
- Line ~10, a doc comment: `CUDA blockDim.{x,y,z}   ->  ripple_get_size(block, {0,1,2})` → `...  ->  ripple_get_block_size(block, {0,1,2})`
- Line ~90, docstring: `"""Translates blockDim.{x,y,z} to ripple_get_size()."""` → `"""Translates blockDim.{x,y,z} to ripple_get_block_size()."""`
- Line ~98, `description="Translate blockDim to ripple_get_size"` → `description="Translate blockDim to ripple_get_block_size"`
- Line ~107, the actual generated code: `return f"ripple_get_size(ripple_block, {dim})"` → `return f"ripple_get_block_size(ripple_block, {dim})"`

- [ ] **Step 3: Fix the IR path's map**

In `frontends/ir/ir_frontend.py`, in `NVPTX_TO_RIPPLE`, change:
```python
    NVPTXIntrinsic.NTID_X: ("ripple_get_size", 0),
    NVPTXIntrinsic.NTID_Y: ("ripple_get_size", 1),
    NVPTXIntrinsic.NTID_Z: ("ripple_get_size", 2),
```
to:
```python
    NVPTXIntrinsic.NTID_X: ("ripple_get_block_size", 0),
    NVPTXIntrinsic.NTID_Y: ("ripple_get_block_size", 1),
    NVPTXIntrinsic.NTID_Z: ("ripple_get_block_size", 2),
```

- [ ] **Step 4: Fix the 3 test assertions**

In `tests/test_translation.py`, update the 3 assertions found in Step 1 (search for `ripple_get_size`) to assert `ripple_get_block_size` instead — same assertion style, just the corrected name. Read each one in context first (don't blind-replace) to make sure the surrounding assertion logic still makes sense with the corrected name.

- [ ] **Step 5: Fix the documentation**

In `README.md`, `docs/README.md`, and `docs/ARCHITECTURE.md`, replace every `ripple_get_size` occurrence with `ripple_get_block_size` (translation mapping tables and code examples — same content in `README.md` and `docs/README.md`, likely duplicated files; fix both rather than assuming they're symlinked).

- [ ] **Step 6: Run the full suite**

Run: `source venv/bin/activate && python -m pytest tests/ -v`
Expected: all pass (still 55 — this is a rename, not new test coverage)

- [ ] **Step 7: Verify no occurrences remain**

Run: `grep -rn "ripple_get_size" --include="*.py" --include="*.md" . | grep -v temp_ripple_docs | grep -v __pycache__`
Expected: empty output

- [ ] **Step 8: Commit**

```bash
git add core/translation_rules.py frontends/ir/ir_frontend.py tests/test_translation.py README.md docs/README.md docs/ARCHITECTURE.md
git commit -m "Fix blockDim translation calling a function that doesn't exist

BlockDimRule translated blockDim.{x,y,z} to ripple_get_size(...), a
function name that appears nowhere in the real RIPPLE API (confirmed
against temp_ripple_docs/src/ripple-spec/api.md). The real function
is ripple_get_block_size. Since blockDim is used in nearly every real
CUDA kernel, this broke compilation broadly — found by cross-checking
generated output against the real upstream API spec rather than just
against this project's own internal consistency (its own docs and
tests had the same wrong name, so they didn't catch it).

Also fixes the same wrong name in the IR translation path's
NVPTX_TO_RIPPLE map, which had the identical bug independently."
```

### Task 6: Build a lightweight syntax-verification helper (no Docker)

Checks translated RIPPLE C output for valid C syntax using the system's already-installed `clang` in `-fsyntax-only` mode against a minimal stub `ripple.h` — not the full Hexagon-targeting toolchain vendored in Task 4 (kept, deferred — see the Priority 2 revision note above). This catches gross syntax breakage (unbalanced constructs, invalid tokens, C++-only syntax accidentally emitted into C output) without any build step.

**Files:**
- Create: `tests/stub_headers/ripple.h`
- Create: `tests/compile_verify.py`
- Create: `tests/test_compile_verification.py`

- [ ] **Step 1: Write the stub header**

Create `tests/stub_headers/ripple.h`:

```c
/*
 * Minimal stub of the real RIPPLE API (see the upstream spec at
 * temp_ripple_docs/src/ripple-spec/api.md) — just enough for a generic
 * clang to resolve #include <ripple.h> during a syntax-only check.
 *
 * Deliberately uses the REAL upstream function names, not whatever the
 * translator happens to emit. If the translator's output doesn't match
 * this stub, that IS a real bug the check should catch — that's exactly
 * how the ripple_get_size / ripple_get_block_size mismatch fixed in
 * this same round was found.
 *
 * NOT a full implementation and NOT suitable for compiling against the
 * real Hexagon toolchain — see docker/README.md for that heavier check.
 */
#ifndef RIPPLE_STUB_H
#define RIPPLE_STUB_H

#include <stddef.h>

typedef struct ripple_block_s *ripple_block_t;

#define HVX_PE 0

ripple_block_t ripple_set_block_shape(int pe_id, ...);
size_t ripple_id(ripple_block_t block_shape, int dim);
size_t ripple_get_block_size(ripple_block_t block_shape, int dim);

/* Shuffle/reduction: declared for completeness, though every current
 * call site of ripple_shuffle is known-broken (not valid C at all —
 * see GitHub issue #8), so declaring it correctly doesn't make those
 * kernels pass — they fail on the actual invalid syntax at the call
 * site, which is the correct, honest failure. */
typedef size_t (*ripple_shuffle_fn_t)(size_t, size_t);
size_t ripple_shuffle(size_t value, ripple_shuffle_fn_t fn);
size_t ripple_reduceadd(size_t value);

#endif
```

- [ ] **Step 2: Write the helper module**

Create `tests/compile_verify.py`:

```python
"""
Lightweight syntax-verification helper: runs translated RIPPLE C through a
standard, locally-installed clang in syntax-only mode (-fsyntax-only).

This catches gross syntax breakage in the translator's output (unbalanced
constructs, invalid tokens, C++-only syntax accidentally emitted into C
output) using only a stub ripple.h (tests/stub_headers/ripple.h) declaring
the real upstream RIPPLE API by name. It does NOT validate real Hexagon/HVX
semantics or codegen — that requires the full toolchain vendored in
docker/ (see docker/README.md), which is deliberately not wired in here.

Requires only a standard clang on PATH (already present on most dev
machines) — no Docker, no long build, no network access.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest

STUB_HEADER_DIR = Path(__file__).parent / "stub_headers"


def clang_available() -> bool:
    """Check whether a clang binary is available on PATH."""
    try:
        result = subprocess.run(["clang", "--version"], capture_output=True, timeout=10)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return result.returncode == 0


def verify_ripple_syntax(ripple_c_source: str) -> tuple[bool, str]:
    """
    Check translated RIPPLE C source for valid C syntax using a generic
    clang and a stub ripple.h (-fsyntax-only, -xc — no codegen, no linking).

    This does NOT validate RIPPLE semantics or real Hexagon/HVX codegen —
    only that the output is syntactically valid C referencing the real
    RIPPLE API by name. See docker/README.md for the heavier check.

    Returns (success, diagnostic_output). diagnostic_output is clang's
    combined stdout+stderr — empty on success, the compiler's error text
    on failure.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "kernel.c"
        src_path.write_text(ripple_c_source)

        result = subprocess.run(
            [
                "clang",
                "-fsyntax-only",
                "-xc",
                f"-I{STUB_HEADER_DIR}",
                str(src_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode == 0, output


requires_clang = pytest.mark.skipif(
    not clang_available(),
    reason="clang not found on PATH — see tests/compile_verify.py",
)
```

- [ ] **Step 3: Write the smoke test**

Create `tests/test_compile_verification.py`:

```python
"""Smoke test for the lightweight syntax-verification helper itself."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.compile_verify import requires_clang, verify_ripple_syntax
from frontends.source.cuda_frontend import translate_cuda_source


@requires_clang
def test_translated_vector_add_is_valid_syntax():
    source = """
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""
    translated = translate_cuda_source(source)
    success, output = verify_ripple_syntax(translated)
    assert success, f"Translated output failed syntax check:\n{output}\n\n--- translated source ---\n{translated}"


@requires_clang
def test_deliberately_broken_c_fails_syntax_check():
    # Negative control: confirms the helper actually catches a real failure
    # rather than silently reporting success no matter what it's given.
    success, output = verify_ripple_syntax("this is not valid C at all {{{")
    assert not success
    assert output  # some diagnostic text should be present
```

- [ ] **Step 4: Run it**

Run: `source venv/bin/activate && python -m pytest tests/test_compile_verification.py -v`
Expected (clang present, which it should be on essentially any dev machine): both tests PASS — notably the negative-control test, which proves the helper isn't vacuously reporting success. If `test_translated_vector_add_is_valid_syntax` fails, do not weaken it — that would mean either the stub header is wrong (fix the stub) or there's a real, previously-undiscovered syntax bug in the translator (file it, same as the two found this round).

- [ ] **Step 5: Run the full suite to confirm nothing else broke**

Run: `source venv/bin/activate && python -m pytest tests/ -v`
Expected: all previous tests still pass; the two new ones pass (or skip if clang genuinely isn't on PATH, which would be unusual)

- [ ] **Step 6: Commit**

```bash
git add tests/stub_headers/ripple.h tests/compile_verify.py tests/test_compile_verification.py
git commit -m "Add lightweight syntax-verification helper (no Docker required)

verify_ripple_syntax() runs translated output through the system's
clang in -fsyntax-only mode against a stub ripple.h that declares
the real upstream RIPPLE API by name (not whatever the translator
happens to emit) — deliberately, so a name mismatch is a real,
caught failure rather than something the stub quietly accommodates.

Replaces the original plan for this task, which built a full
Hexagon-targeting toolchain via Docker (see docker/, vendored and
kept but not wired in here — deferred to whenever RIPPLE-semantic
correctness, not syntax, is the actual bottleneck). Includes a
negative-control test to prove the check can actually fail."
```

---

## Priority 3: Rebuild the test corpus around real kernels, validated by compile

### File Structure

- Modify: relocate 6 existing root-level scratch `.cu` files into `tests/examples/`
- Delete: 3 stale ad hoc output snapshots (`test_output.json`, `test_bitwise_output.json`, `test_grid_output.json`) — superseded by real pytest assertions
- Create: `tests/test_real_kernels.py` — translation + compile-verification tests for each relocated kernel

### Task 7: Relocate the existing sample kernels into `tests/examples/`

These six `.cu` files already exist in the repo (added across several early commits) but aren't referenced by any test — they were used for manual/ad hoc runs during development and never wired in. `tests/examples/` already exists and already holds one file (`cuda_kernels.cu`), so this is following an existing convention, not inventing a new one.

**Files:**
- Move: `test_ast.cu` → `tests/examples/ast_flat.cu`
- Move: `test_ast_complex.cu` → `tests/examples/ast_if_no_braces.cu`
- Move: `test_atomics_new.cu` → `tests/examples/atomics_cas_exch.cu`
- Move: `test_bitwise.cu` → `tests/examples/bitwise_intrinsics.cu`
- Move: `test_grid.cu` → `tests/examples/grid_stride.cu`
- Move: `test_reduction.cu` → `tests/examples/warp_reduction.cu`
- Delete: `test_manual.cu` (content is a strict subset of `grid_stride.cu` — same single-statement kernel shape, nothing distinct to preserve)
- Delete: `test_output.json`, `test_bitwise_output.json`, `test_grid_output.json` (ad hoc snapshots from manual runs, no longer referenced by anything once the tests below exist)

- [ ] **Step 1: Move the files, renamed to describe what each one exercises**

```bash
git mv test_ast.cu tests/examples/ast_flat.cu
git mv test_ast_complex.cu tests/examples/ast_if_no_braces.cu
git mv test_atomics_new.cu tests/examples/atomics_cas_exch.cu
git mv test_bitwise.cu tests/examples/bitwise_intrinsics.cu
git mv test_grid.cu tests/examples/grid_stride.cu
git mv test_reduction.cu tests/examples/warp_reduction.cu
git rm test_manual.cu
git rm test_output.json test_bitwise_output.json test_grid_output.json
```

- [ ] **Step 2: Confirm nothing else in the repo referenced the old paths**

Run: `grep -rn "test_ast\.cu\|test_ast_complex\.cu\|test_atomics_new\.cu\|test_bitwise\.cu\|test_grid\.cu\|test_reduction\.cu\|test_manual\.cu\|test_output\.json\|test_bitwise_output\.json\|test_grid_output\.json" --include="*.py" --include="*.sh" --include="*.md" .`
Expected: no output (nothing referenced these paths — confirmed in the prior investigation session, re-checking here since it's cheap and this is the step that would catch it if wrong)

- [ ] **Step 3: Commit**

```bash
git commit -m "Relocate ad hoc sample kernels into tests/examples/

These six .cu files existed at the repo root since early commits but
were never referenced by any test — pure manual/ad hoc scratch files
from development, alongside three stale JSON output snapshots from
those manual runs. Moved into tests/examples/ (following the existing
convention set by cuda_kernels.cu) and renamed to describe what each
one exercises, ahead of wiring real tests against them. Confirmed via
grep that nothing else in the repo referenced the old paths."
```

### Task 8: Write real translation + compile-verification tests against them

**Files:**
- Create: `tests/test_real_kernels.py`

- [ ] **Step 1: Write the test file**

Create `tests/test_real_kernels.py`:

```python
"""
Translation + real-syntax-check tests against the sample kernels in
tests/examples/ — these are kernels shaped like real CUDA code
(nested control flow, atomics, warp shuffles), not the minimal
one-liners used to unit-test individual translation rules.

Structural assertions (does it look translated) run always.
Syntax-check assertions (does it actually parse as valid C, against
the real RIPPLE API surface) run whenever clang is on PATH — see
tests/compile_verify.py.

warp_reduction.cu is expected to FAIL its syntax check right now —
GitHub issue #8 tracks that all 4 warp-shuffle translation rules emit
invalid C (C++ lambdas or a nested function definition). Marked xfail,
not skipped and not excluded, so: (a) the failure stays visible and
traceable to the tracked issue rather than silently disappearing from
the suite, and (b) if issue #8 gets fixed without updating this test,
pytest reports an unexpected pass (XPASS) — a loud, hard-to-miss signal
that this xfail marker needs to come off, not a silent status quo.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from frontends.source.cuda_frontend import translate_cuda_source
from tests.compile_verify import requires_clang, verify_ripple_syntax

EXAMPLES_DIR = Path(__file__).parent / "examples"

KERNEL_FILES = [
    "ast_flat.cu",
    "ast_if_no_braces.cu",
    "atomics_cas_exch.cu",
    "bitwise_intrinsics.cu",
    "grid_stride.cu",
    "warp_reduction.cu",
]

# https://github.com/C-ripple/Translate/issues/8 — all 4 shuffle rules
# emit invalid C. warp_reduction.cu uses __shfl_down_sync.
KNOWN_INVALID_SYNTAX = {"warp_reduction.cu"}


@pytest.mark.parametrize("filename", KERNEL_FILES)
def test_translates_without_error(filename):
    source = (EXAMPLES_DIR / filename).read_text()
    result = translate_cuda_source(source)
    assert result, f"{filename} produced empty output"
    assert "__global__" not in result, f"{filename}: untranslated __global__ remains"


@requires_clang
@pytest.mark.parametrize("filename", KERNEL_FILES)
def test_translated_output_is_valid_syntax(filename):
    if filename in KNOWN_INVALID_SYNTAX:
        pytest.xfail(f"{filename}: known invalid-C output, see GitHub issue #8")
    source = (EXAMPLES_DIR / filename).read_text()
    translated = translate_cuda_source(source)
    success, output = verify_ripple_syntax(translated)
    assert success, (
        f"{filename}: translated output failed syntax check:\n{output}\n\n"
        f"--- translated source ---\n{translated}"
    )
```

- [ ] **Step 2: Run the structural tests (these always run)**

Run: `source venv/bin/activate && python -m pytest tests/test_real_kernels.py -v -k "not syntax"`
Expected: 6 passed. If any fail, that's a real, previously-hidden translation gap on kernel shapes nobody had tested before (e.g. `atomics_cas_exch.cu`'s two-atomic-call kernel, or `warp_reduction.cu`'s nested for+if) — fix the underlying rule in `core/translation_rules.py` before continuing, don't weaken the assertion.

- [ ] **Step 3: Run the syntax-check tests**

Run: `source venv/bin/activate && python -m pytest tests/test_real_kernels.py -v -k "syntax"`
Expected: 5 passed, 1 xfailed (`warp_reduction.cu`, per issue #8). Any *other* failure among the 5 expected-to-pass kernels is a real, previously-hidden translation-correctness bug — same instruction as Step 2, fix the rule (or if it's a stub-header gap, fix `tests/stub_headers/ripple.h`), don't weaken the test. If `warp_reduction.cu` unexpectedly passes (XPASS), issue #8 has apparently already been fixed elsewhere — remove it from `KNOWN_INVALID_SYNTAX` rather than leaving a stale xfail marker.
Expected (clang not on PATH): all syntax-check tests skipped

- [ ] **Step 4: Run the full suite**

Run: `source venv/bin/activate && python -m pytest tests/ -v`
Expected: everything from Priorities 1 and 2 still passes, plus these new tests pass/xfail/skip per Steps 2-3

- [ ] **Step 5: Commit**

```bash
git add tests/test_real_kernels.py
git commit -m "Add translation + syntax-check tests against real sample kernels

Structural tests (translates without error, no leftover __global__)
always run. Syntax-check tests run translated output through clang
in -fsyntax-only mode against the real RIPPLE API's stub header when
clang is available. This replaces 'trust the string assertions' with
'prove it's valid C referencing real RIPPLE function names' for the
kernel shapes most likely to expose gaps the narrow hand-picked unit
tests in test_translation.py and test_complex_kernels.py wouldn't
catch — nested control flow, two atomics in one kernel, warp shuffle
inside a loop.

warp_reduction.cu is marked xfail, not excluded — its known-invalid
shuffle output (GitHub issue #8) stays visible and traceable in the
suite, and flips to a loud XPASS if that issue is ever fixed without
updating this test."
```

---

## Self-Review

**Spec coverage:**
- "Fix the AST parser's infinite loop and wire it into the source-level translator as a structural safety net" → Priority 1, Tasks 1-3. ✓ (executed; grew to include 4 additional hang-guard fixes and a real-attribute-parsing fix found during review — see git history on `translator-hardening`)
- "Stand up compile verification" → revised mid-execution from a full Hexagon toolchain build to a lightweight clang syntax check, Priority 2, Tasks 4-6, after determining the heavy version was disproportionate to what the project has actually needed so far (see the Priority 2 revision note). Task 4 (vendoring) still done and kept for later. Two real translator bugs found while scoping this (`ripple_get_size`, Task 5; warp-shuffle invalid-C output, deferred to GitHub issue #8) were bugs in the codebase, not gaps in this plan's coverage. ✓
- "Rebuild the test corpus around real kernels, validated through the compile step" → Priority 3, Tasks 7-8, using the revised lightweight helper; `warp_reduction.cu` explicitly `xfail`s per issue #8 rather than being silently excluded. ✓

**Placeholder scan:** No TBD/TODO/"add appropriate handling" phrasing anywhere above; every code step is complete, runnable code; every "Run:" step has a concrete command and expected output.

**Type/name consistency:** `verify_ripple_syntax(ripple_c_source) -> tuple[bool, str]` is defined once in Task 6 Step 2 and called identically in Task 6 Step 3 and Task 8 Step 1. `clang_available()` / `requires_clang` likewise defined once, imported by name everywhere else. `KERNEL_TWO_NESTED` is defined in Priority 1 Task 1 and reused (not redefined) in Priority 1 Task 3 — both edits land in the same file (`tests/test_ast_parsing.py`), so this holds only if Task 3's edit is applied as an *addition* to the file created in Task 1, not a fresh overwrite; noted here explicitly since it's the one place a later task depends on an earlier task's file state rather than just the codebase's. `KNOWN_INVALID_SYNTAX` (Task 8) and GitHub issue #8 both name `warp_reduction.cu`/`__shfl_down_sync` consistently.
