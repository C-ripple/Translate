"""
CUDA to RIPPLE Translation Rules

This module defines the semantic transformation rules for converting CUDA
constructs to their RIPPLE equivalents. These rules are used by both the
source-level and IR-level frontends.

Key Mappings:
    CUDA threadIdx.{x,y,z}  ->  ripple_id(block, {0,1,2})
    CUDA blockDim.{x,y,z}   ->  ripple_get_block_size(block, {0,1,2})
    CUDA __shared__         ->  local array + VTCM hints (Hexagon)
    CUDA __syncthreads()    ->  implicit (SIMD model) or explicit barrier
    CUDA atomicAdd          ->  ripple_reduction or HVX scatter-accumulate
    CUDA warp shuffles      ->  ripple_shuffle with permutation function
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import re

from .semantic_model import (
    AIRNode, AIRFunction, AIRExpression, AIRThreadIndex, AIRLoop,
    AIRMemoryOp, AIRShuffleOp, AIRReductionOp, AIRSynchronization,
    AIRConditional, AIRTranslationUnit, AIRVariable, AIRType,
    CUDABuiltinAccess, CUDADim3, CUDAMemorySpace, CUDASyncScope,
    CUDAWarpShuffle, CUDAReduction, CUDAAtomicOp,
    RIPPLEBlockShape, RIPPLEIndex, RIPPLEBroadcast, RIPPLEReduction,
    RIPPLEShuffle, RIPPLEParallelLoop, RIPPLEProcessingElement,
    TranslationContext, HexagonConfig
)


# =============================================================================
# Pattern Matching Infrastructure
# =============================================================================

@dataclass
class TranslationRule:
    """A single translation rule with pattern and replacement."""
    name: str
    description: str
    cuda_pattern: str  # Regex or structured pattern
    priority: int = 0  # Higher = applied first
    
    def matches(self, cuda_code: str) -> bool:
        """Check if this rule matches the CUDA code."""
        return bool(re.search(self.cuda_pattern, cuda_code))
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        """Apply the rule and return transformed code."""
        raise NotImplementedError("Subclasses must implement apply()")


# =============================================================================
# Built-in Variable Translation Rules
# =============================================================================

class ThreadIdxRule(TranslationRule):
    """Translates threadIdx.{x,y,z} to ripple_id()."""
    
    PATTERN = r'threadIdx\.([xyz])'
    COMPONENT_MAP = {'x': 0, 'y': 1, 'z': 2}
    
    def __init__(self):
        super().__init__(
            name="thread_idx",
            description="Translate threadIdx to ripple_id",
            cuda_pattern=self.PATTERN,
            priority=100
        )
    
    def matches(self, cuda_code: str) -> bool:
        return bool(re.search(self.PATTERN, cuda_code))
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            component = match.group(1)
            dim = self.COMPONENT_MAP[component]
            ctx.thread_idx_mappings[f"threadIdx.{component}"] = RIPPLEIndex(
                block_var="ripple_block",
                dimension=dim
            )
            return f"ripple_id(ripple_block, {dim})"
        
        return re.sub(self.PATTERN, replace, cuda_code)


class BlockDimRule(TranslationRule):
    """Translates blockDim.{x,y,z} to ripple_get_block_size()."""
    
    PATTERN = r'blockDim\.([xyz])'
    COMPONENT_MAP = {'x': 0, 'y': 1, 'z': 2}
    
    def __init__(self):
        super().__init__(
            name="block_dim",
            description="Translate blockDim to ripple_get_block_size",
            cuda_pattern=self.PATTERN,
            priority=100
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            component = match.group(1)
            dim = self.COMPONENT_MAP[component]
            return f"ripple_get_block_size(ripple_block, {dim})"
        
        return re.sub(self.PATTERN, replace, cuda_code)


class BlockIdxRule(TranslationRule):
    """Translates blockIdx.{x,y,z} to loop iteration variables."""
    
    PATTERN = r'blockIdx\.([xyz])'
    
    def __init__(self):
        super().__init__(
            name="block_idx",
            description="Translate blockIdx to outer loop index",
            cuda_pattern=self.PATTERN,
            priority=100
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        # blockIdx becomes an outer loop variable in RIPPLE
        # This is handled at a higher level - the kernel restructuring
        def replace(match):
            component = match.group(1)
            ctx.add_warning(
                f"blockIdx.{component} requires kernel restructuring to outer loop"
            )
            return f"block_idx_{component}"
        
        return re.sub(self.PATTERN, replace, cuda_code)


class GridDimRule(TranslationRule):
    """Translates gridDim.{x,y,z} to loop bounds."""
    
    PATTERN = r'gridDim\.([xyz])'
    
    def __init__(self):
        super().__init__(
            name="grid_dim",
            description="Translate gridDim to loop bounds",
            cuda_pattern=self.PATTERN,
            priority=100
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            component = match.group(1)
            return f"grid_dim_{component}"
        
        return re.sub(self.PATTERN, replace, cuda_code)


# =============================================================================
# Memory Space Translation Rules
# =============================================================================

class SharedMemoryRule(TranslationRule):
    """Translates __shared__ declarations to local arrays with VTCM hints."""

    # Array-form declarations only (requires trailing [...]) — scalar
    # `__shared__ float x;` is not matched or translated by this rule.
    # tests/test_translation.py reuses this pattern to detect leftover
    # untranslated declarations, so it inherits the same array-only scope.
    PATTERN = r'__shared__\s+(\w+)\s+(\w+)\s*\[([^\]]*)\]'
    
    def __init__(self):
        super().__init__(
            name="shared_memory",
            description="Translate __shared__ to local array with VTCM",
            cuda_pattern=self.PATTERN,
            priority=90
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            elem_type = match.group(1)
            var_name = match.group(2)
            size_expr = match.group(3)
            
            ctx.shared_mem_mappings[var_name] = f"__attribute__((aligned(128))) {elem_type}"
            
            if ctx.target_platform == "hexagon":
                # Use VTCM for shared memory on Hexagon
                return (
                    f"// CUDA __shared__ -> Hexagon VTCM\n"
                    f"    __attribute__((section(\".vtcm\"))) "
                    f"__attribute__((aligned(128))) "
                    f"{elem_type} {var_name}[{size_expr}]"
                )
            else:
                return f"__attribute__((aligned(128))) {elem_type} {var_name}[{size_expr}]"
        
        return re.sub(self.PATTERN, replace, cuda_code)


class DynamicSharedMemoryRule(TranslationRule):
    """Translates extern __shared__ (dynamic shared memory)."""
    
    PATTERN = r'extern\s+__shared__\s+(\w+)\s+(\w+)\s*\[\s*\]'
    
    def __init__(self):
        super().__init__(
            name="dynamic_shared_memory",
            description="Translate extern __shared__ to dynamic allocation",
            cuda_pattern=self.PATTERN,
            priority=90
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            elem_type = match.group(1)
            var_name = match.group(2)
            
            ctx.add_warning(
                f"Dynamic shared memory '{var_name}' requires size parameter"
            )
            
            return (
                f"// CUDA extern __shared__ -> passed as parameter\n"
                f"    // {elem_type}* {var_name}  // Add to function parameters"
            )
        
        return re.sub(self.PATTERN, replace, cuda_code)


# =============================================================================
# Synchronization Translation Rules
# =============================================================================

class SyncThreadsRule(TranslationRule):
    """Translates __syncthreads() - often not needed in SIMD model."""
    
    PATTERN = r'__syncthreads\s*\(\s*\)'
    
    def __init__(self):
        super().__init__(
            name="sync_threads",
            description="Translate __syncthreads to RIPPLE equivalent",
            cuda_pattern=self.PATTERN,
            priority=80
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        ctx.add_warning(
            "__syncthreads() may not be needed in RIPPLE SIMD model - "
            "all lanes execute in lockstep"
        )
        
        # In pure SIMD, sync is implicit. For Hexagon with multiple HW threads:
        if ctx.target_platform == "hexagon":
            replacement = "/* __syncthreads: implicit in SIMD (HVX lanes are lockstep) */"
        else:
            replacement = "/* __syncthreads: implicit in SIMD model */"
        
        return re.sub(self.PATTERN, replacement, cuda_code)


class SyncWarpRule(TranslationRule):
    """Translates __syncwarp()."""
    
    PATTERN = r'__syncwarp\s*\([^)]*\)'
    
    def __init__(self):
        super().__init__(
            name="sync_warp",
            description="Translate __syncwarp to no-op (implicit in SIMD)",
            cuda_pattern=self.PATTERN,
            priority=80
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        return re.sub(self.PATTERN, "/* __syncwarp: implicit in SIMD model */", cuda_code)


# =============================================================================
# Warp Shuffle Translation Rules
# =============================================================================

def _is_compile_time_constant_expr(expr: str) -> bool:
    """
    Rough check for whether a captured shuffle argument (delta/lane_mask/
    src_lane) is a compile-time-constant-shaped expression (digits and
    arithmetic operators only) rather than a reference to a kernel-local
    variable. Hoisted shuffle helper functions are file-scope, so a
    variable reference in the body would be out of scope — see the
    non-constant case's handling in each shuffle rule below. Variable
    (non-constant) shuffle arguments are a known, deliberate gap tracked
    as GitHub issue #11 — ripple_shuffle's function-pointer signature is
    fixed at exactly (k, block_size), so there's no clean way to thread
    an arbitrary runtime value through it. UnrollConstantShuffleLoopRule
    resolves the common case (a small, compile-time-bounded loop
    variable) before any shuffle rule ever sees it; this function exists
    to detect what's left over — genuinely unresolvable arguments — and
    hand them off to the hard-fail path below, not to solve them.

    Deliberate convention note: on a non-constant argument, the shuffle
    rules below call ctx.add_error(...) and return the ORIGINAL CUDA
    call unmodified — ctx.add_error() makes transform() raise
    TranslationError once all rules have run, so no output file is ever
    written for a kernel containing an unresolvable shuffle. This is
    unlike DynamicSharedMemoryRule elsewhere in this file, which
    replaces its untranslatable construct with a commented-out
    placeholder and only warns. Both are valid strategies for their
    respective cases, not one being an oversight: a silently-degraded
    shuffle would compile into wrong (or non-compiling) code with only
    an easy-to-miss warning as the signal, which is worse than failing
    the whole translation loudly.

    Also rejects any argument containing a paren: the outer capture
    regex ([^,\\)]+) excludes ')' by construction, so a parenthesized
    argument like `(1)` always arrives here as `(1` (dangling open
    paren, zero close parens) — a shape-only check would accept that as
    "constant" and splice invalid C. The `count('(') == count(')')`
    check below isn't really balance-checking (a truly balanced `(1)`
    can never reach this function intact) — in practice it just rejects
    any '(' at all, which correctly routes the truncated-paren case to
    the same hard-fail path as a real variable reference, instead of
    emitting broken C silently.
    """
    stripped = expr.strip()
    if not stripped or re.fullmatch(r'[\d\s+\-*/()]+', stripped) is None:
        return False
    return stripped.count('(') == stripped.count(')')


class UnrollConstantShuffleLoopRule(TranslationRule):
    """
    Unrolls a small, compile-time-bounded counting loop whose body
    contains a warp-shuffle call using the loop's induction variable —
    substituting each literal value the variable takes, so the
    already-correct literal-argument shuffle rules can hoist a real
    permutation function for each, instead of the whole loop being left
    as an untranslatable variable-argument shuffle (see
    _is_compile_time_constant_expr's docstring, GitHub issue #11).

    Matches:
      for (int VAR = INIT; VAR OP BOUND; VAR OP= STEP) { BODY }
      for (int VAR = INIT; VAR OP BOUND; VAR OP= STEP) STATEMENT;
    INIT/BOUND accept a plain integer literal or the token `warpSize`
    (resolved to 32); STEP is digit-only (WarpReductionRule already
    owns the symbolic warpSize/2-init + accumulate shape, and this
    rule's INIT/BOUND grammar can't match that shape anyway). Only
    fires when BODY contains a shuffle call and references VAR
    somewhere in the body — an unrelated countable loop is left alone.

    Known limitations (each one is a deliberate bail-out to the
    original, untouched loop text, not a translation attempt):
      - A braced body may contain nested simple statements but not
        nested braces (no if/for/while with their own {}).
      - A braceless body starting with a control-flow keyword
        (if/while/do/for/switch, spelled `kw (` or `kw(`) is left
        untouched — its single trailing-semicolon capture can't safely
        represent multi-statement control flow (e.g. if/else).
      - A body containing a string/char literal (`"` or `'`) or a
        comment opener (`/*` or `//`) anywhere is left untouched, for
        both braced and braceless bodies — neither pattern is
        literal- or comment-aware, so a `}` or `;` inside a string,
        char literal, or comment could otherwise be misread as the
        loop's real delimiter (confirmed severe for the braced case:
        a `}` inside a comment can splice the function's real closing
        brace and everything after it outside the function body,
        invisibly — ctx.errors stays empty).
    """

    # Matched at the start of a braceless body to bail out on control
    # flow (see _replace()). `\b` after the keyword group correctly
    # matches BOTH `if (x)` and the no-space `if(x)` spelling (a word
    # boundary sits between "if" and "(" either way), while rejecting
    # identifiers that merely start with a keyword's letters, like
    # `do_something()` or `iffy_call()` (no boundary between "if"/"do"
    # and the following word character). A plain whitespace-split
    # first-token check would miss the no-space spelling entirely and
    # let the exact bug back in for idiomatic code like `if(cond) ...`.
    CONTROL_FLOW_PREFIX = re.compile(r'\s*(?:if|while|do|for|switch)\b')

    # Any of these appearing anywhere in a captured body means "don't
    # touch this loop" (see _replace()). Neither PATTERN_BRACED's
    # `}`-delimited capture nor PATTERN_BRACELESS's `;`-delimited
    # capture understands string/char literals or comments, so a `}`
    # or `;` inside one of these can be misread as the loop's real
    # delimiter. `"` and `'` cover string/char literals; `/*` and `//`
    # cover both comment styles (a `}` or `;` inside either kind is
    # just as invisible to the regex as one inside a string).
    UNSAFE_BODY_TOKENS = ('"', "'", '/*', '//')

    _LITERAL_OR_WARPSIZE = r'(\d+|warpSize)'

    PATTERN_BRACED = (
        r'for\s*\(\s*int\s+(\w+)\s*=\s*' + _LITERAL_OR_WARPSIZE + r'\s*;\s*'
        r'\1\s*(<=|>=|<|>)\s*' + _LITERAL_OR_WARPSIZE + r'\s*;\s*'
        r'\1\s*(\*=|/=|\+=|-=)\s*(\d+)\s*\)\s*'
        r'\{([^{}]*)\}'
    )
    PATTERN_BRACELESS = (
        r'for\s*\(\s*int\s+(\w+)\s*=\s*' + _LITERAL_OR_WARPSIZE + r'\s*;\s*'
        r'\1\s*(<=|>=|<|>)\s*' + _LITERAL_OR_WARPSIZE + r'\s*;\s*'
        r'\1\s*(\*=|/=|\+=|-=)\s*(\d+)\s*\)\s*'
        r'(?!\{)([^;{}]+;)'
    )

    # Kept for stylistic consistency with every other rule class in this
    # file (each defines a `PATTERN` attribute) — not actually read
    # anywhere for this class specifically, since matches()/apply()
    # below explicitly use PATTERN_BRACED/PATTERN_BRACELESS instead.
    PATTERN = PATTERN_BRACED

    MAX_ITERATIONS = 64

    def __init__(self):
        super().__init__(
            name="unroll_constant_shuffle_loop",
            description="Unroll small compile-time-bounded loops feeding a shuffle intrinsic",
            cuda_pattern=self.PATTERN_BRACED,
            priority=82  # below WarpReductionRule (85), above shuffle rules (70)
        )

    def matches(self, cuda_code: str) -> bool:
        return bool(
            re.search(self.PATTERN_BRACED, cuda_code)
            or re.search(self.PATTERN_BRACELESS, cuda_code)
        )

    @staticmethod
    def _resolve_literal(token: str) -> int:
        """Resolves a captured INIT/BOUND token ('warpSize' or a plain
        integer literal string) to its integer value."""
        return 32 if token == 'warpSize' else int(token)

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

    def _replace(self, match, ctx: TranslationContext, is_braceless: bool = False) -> str:
        var_name = match.group(1)
        init = self._resolve_literal(match.group(2))
        cond_op = match.group(3)
        bound = self._resolve_literal(match.group(4))
        step_op = match.group(5)
        step = int(match.group(6))
        body = match.group(7)

        if any(tok in body for tok in self.UNSAFE_BODY_TOKENS):
            # Neither PATTERN_BRACED's `}`-delimited capture nor
            # PATTERN_BRACELESS's `;`-delimited capture is aware of
            # string/char literals OR comments, so a `}` or `;` INSIDE
            # one of these (e.g. `printf("}")`, or a trailing
            # `// note [3] }` comment) can be misread as the loop's
            # real closing delimiter. For the braced case this is
            # severe: the capture truncates mid-literal/mid-comment,
            # and everything textually after it in the file —
            # including the loop's REAL closing brace and any code
            # after the loop — ends up spliced outside the function
            # body, with no warning or error raised (ctx.errors stays
            # empty; only the benign "Unrolled loop..." success
            # warning fires). Real tokenization to skip literal/comment
            # contents correctly is out of scope for what should be a
            # narrow, defensive check, so bail out bluntly instead: any
            # of '"', "'", '/*', '//' anywhere in the captured body,
            # for BOTH body shapes, checked before any other logic
            # runs. (Verified PATTERN_BRACELESS's own bounded,
            # first-semicolon capture can't reach the same severe
            # cross-statement corruption — its match span never
            # extends past its own terminator — but it's included here
            # too for the same reason the quote check already covers
            # both shapes: one uniform rule is simpler to reason about
            # than two shape-specific exceptions, and it costs nothing
            # for realistic input. A shuffle-loop body containing an
            # actual string/char literal or comment is already a
            # contrived shape — nobody puts printf debug statements or
            # footnote comments inside a hot warp-shuffle reduction
            # loop.)
            return match.group(0)

        if is_braceless and self.CONTROL_FLOW_PREFIX.match(body):
            # A braceless body starting with control flow can contain
            # its own semicolons (if/else, do/while) that this pattern's
            # single-semicolon truncation can't safely capture — leave
            # it untouched rather than risk silently dropping part of
            # the statement. The user would need to wrap it in braces
            # for this rule to handle it.
            return match.group(0)

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

    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        # Braced first — it's the common/already-tested case, and once
        # it consumes a loop's text there's nothing left for the
        # braceless pattern to accidentally match on that same loop.
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


class ShuffleDownRule(TranslationRule):
    """Translates __shfl_down_sync to ripple_shuffle."""
    
    PATTERN = r'__shfl_down_sync\s*\(\s*([^,]+),\s*([^,]+),\s*([^,\)]+)(?:,\s*([^)]+))?\)'
    
    def __init__(self):
        super().__init__(
            name="shuffle_down",
            description="Translate __shfl_down_sync to ripple_shuffle",
            cuda_pattern=self.PATTERN,
            priority=70
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            mask = match.group(1).strip()
            value = match.group(2).strip()
            delta = match.group(3).strip()
            width = match.group(4).strip() if match.group(4) else "32"

            # The hoisted function is file-scope, so it can only reference
            # what the call site captures textually. A kernel-local
            # variable (e.g. a loop counter) isn't in scope there — only a
            # compile-time-constant-shaped expression is safe to splice in.
            if not _is_compile_time_constant_expr(delta):
                ctx.add_error(
                    f"ShuffleDownRule: cannot translate __shfl_down_sync(...) — "
                    f"delta '{delta}' is not a compile-time constant. "
                    f"ripple_shuffle's permutation function is file-scope and "
                    f"cannot reference kernel-local variables like '{delta}'. "
                    f"If '{delta}' is a small, compile-time-bounded loop variable, "
                    f"see GitHub issue #11 for what's supported "
                    f"(UnrollConstantShuffleLoopRule); otherwise this shuffle "
                    f"cannot currently be translated."
                )
                return match.group(0)

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

            return f"ripple_shuffle({value}, {fn_name}) /* mask={mask}, width={width} */"

        return re.sub(self.PATTERN, replace, cuda_code)


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

            # The hoisted function is file-scope, so it can only reference
            # what the call site captures textually. A kernel-local
            # variable isn't in scope there — only a compile-time-constant-
            # shaped expression is safe to splice in.
            if not _is_compile_time_constant_expr(lane_mask):
                ctx.add_error(
                    f"ShuffleXorRule: cannot translate __shfl_xor_sync(...) — "
                    f"lane_mask '{lane_mask}' is not a compile-time constant. "
                    f"ripple_shuffle's permutation function is file-scope and "
                    f"cannot reference kernel-local variables like '{lane_mask}'. "
                    f"If '{lane_mask}' is a small, compile-time-bounded loop "
                    f"variable, see GitHub issue #11 for what's supported "
                    f"(UnrollConstantShuffleLoopRule); otherwise this shuffle "
                    f"cannot currently be translated."
                )
                return match.group(0)

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

            # The hoisted function is file-scope, so it can only reference
            # what the call site captures textually. A kernel-local
            # variable isn't in scope there — only a compile-time-constant-
            # shaped expression is safe to splice in.
            if not _is_compile_time_constant_expr(delta):
                ctx.add_error(
                    f"ShuffleUpRule: cannot translate __shfl_up_sync(...) — "
                    f"delta '{delta}' is not a compile-time constant. "
                    f"ripple_shuffle's permutation function is file-scope and "
                    f"cannot reference kernel-local variables like '{delta}'. "
                    f"If '{delta}' is a small, compile-time-bounded loop variable, "
                    f"see GitHub issue #11 for what's supported "
                    f"(UnrollConstantShuffleLoopRule); otherwise this shuffle "
                    f"cannot currently be translated."
                )
                return match.group(0)

            fn_name = f"__ripple_shfl_up_{len(ctx.hoisted_declarations)}"
            ctx.hoisted_declarations.append(f"""
static inline size_t {fn_name}(size_t k, size_t block_size) {{
    return (k >= ({delta})) ? k - ({delta}) : k;
}}""")

            return f"ripple_shuffle({value}, {fn_name})"

        return re.sub(self.PATTERN, replace, cuda_code)


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

            # The hoisted function is file-scope, so it can only reference
            # what the call site captures textually. A kernel-local
            # variable isn't in scope there — only a compile-time-constant-
            # shaped expression is safe to splice in.
            if not _is_compile_time_constant_expr(src_lane):
                ctx.add_error(
                    f"ShuffleSyncRule: cannot translate __shfl_sync(...) — "
                    f"src_lane '{src_lane}' is not a compile-time constant. "
                    f"ripple_shuffle's permutation function is file-scope and "
                    f"cannot reference kernel-local variables like '{src_lane}'. "
                    f"If '{src_lane}' is a small, compile-time-bounded loop "
                    f"variable, see GitHub issue #11 for what's supported "
                    f"(UnrollConstantShuffleLoopRule); otherwise this shuffle "
                    f"cannot currently be translated."
                )
                return match.group(0)

            fn_name = f"__ripple_shfl_sync_{len(ctx.hoisted_declarations)}"
            ctx.hoisted_declarations.append(f"""
static inline size_t {fn_name}(size_t k, size_t block_size) {{
    return ({src_lane});
}}""")

            return f"ripple_shuffle({value}, {fn_name})"

        return re.sub(self.PATTERN, replace, cuda_code)


class WarpReductionRule(TranslationRule):
    """
    Detects and optimizes standard Warp Reduction loops.
    
    Pattern:
      for (int offset = warpSize/2; offset > 0; offset /= 2)
          val += __shfl_down_sync(..., val, offset);
          
    Replacement:
      val = ripple_reduceadd(0b1, val);
    """
    
    # Regex handles variations in spacing and variable names
    # Group 1: Loop variable (e.g. "offset" or "i")
    # Group 2: Accumulator variable (e.g. "val" or "sum")
    PATTERN = r'for\s*\(\s*int\s+(\w+)\s*=\s*warpSize\s*/\s*2\s*;\s*\1\s*>\s*0\s*;\s*\1\s*/=\s*2\s*\)\s*\{\s*(\w+)\s*\+=\s*__shfl_down_sync\s*\([^,]+,\s*\2\s*,\s*\1\s*(?:,[^)]+)?\);\s*\}'
    
    def __init__(self):
        super().__init__(
            name="warp_reduction_optimization",
            description="Optimize warp reduction loop to ripple_reduceadd",
            cuda_pattern=self.PATTERN,
            priority=85  # Higher priority than generic shuffle rules
        )
    
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


# =============================================================================
# Atomic Operation Translation Rules
# =============================================================================

class AtomicAddRule(TranslationRule):
    """Translates atomicAdd to ripple_reduction or HVX scatter."""
    
    PATTERN = r'atomicAdd\s*\(\s*([^,]+),\s*([^)]+)\)'
    
    def __init__(self):
        super().__init__(
            name="atomic_add",
            description="Translate atomicAdd to RIPPLE reduction",
            cuda_pattern=self.PATTERN,
            priority=60
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            target = match.group(1).strip()
            value = match.group(2).strip()
            
            if ctx.target_platform == "hexagon":
                ctx.add_warning(
                    f"atomicAdd on {target}: consider HVX scatter-accumulate "
                    f"for better performance"
                )
                return f"ripple_atomic_add({target}, {value}) /* HVX: use Q6_vscatter_acc */"
            else:
                return f"ripple_atomic_add({target}, {value})"
        
        return re.sub(self.PATTERN, replace, cuda_code)


class AtomicMaxRule(TranslationRule):
    """Translates atomicMax."""
    
    PATTERN = r'atomicMax\s*\(\s*([^,]+),\s*([^)]+)\)'
    
    def __init__(self):
        super().__init__(
            name="atomic_max",
            description="Translate atomicMax to RIPPLE",
            cuda_pattern=self.PATTERN,
            priority=60
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            target = match.group(1).strip()
            value = match.group(2).strip()
            return f"ripple_atomic_max({target}, {value})"
        
        return re.sub(self.PATTERN, replace, cuda_code)


class AtomicMinRule(TranslationRule):
    """Translates atomicMin."""
    
    PATTERN = r'atomicMin\s*\(\s*([^,]+),\s*([^)]+)\)'
    
    def __init__(self):
        super().__init__(
            name="atomic_min",
            description="Translate atomicMin to RIPPLE",
            cuda_pattern=self.PATTERN,
            priority=60
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            target = match.group(1).strip()
            value = match.group(2).strip()
            return f"ripple_atomic_min({target}, {value})"
        
        return re.sub(self.PATTERN, replace, cuda_code)


class AtomicCASRule(TranslationRule):
    """Translates atomicCAS (compare-and-swap)."""
    
    PATTERN = r'atomicCAS\s*\(\s*([^,]+),\s*([^,]+),\s*([^)]+)\)'
    
    def __init__(self):
        super().__init__(
            name="atomic_cas",
            description="Translate atomicCAS to RIPPLE",
            cuda_pattern=self.PATTERN,
            priority=60
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            target = match.group(1).strip()
            compare = match.group(2).strip()
            value = match.group(3).strip()
            return f"ripple_atomic_cas({target}, {compare}, {value})"
        
        return re.sub(self.PATTERN, replace, cuda_code)


class AtomicExchRule(TranslationRule):
    """Translates atomicExch."""
    
    PATTERN = r'atomicExch\s*\(\s*([^,]+),\s*([^)]+)\)'
    
    def __init__(self):
        super().__init__(
            name="atomic_exch",
            description="Translate atomicExch to RIPPLE",
            cuda_pattern=self.PATTERN,
            priority=60
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            target = match.group(1).strip()
            value = match.group(2).strip()
            return f"ripple_atomic_exch({target}, {value})"
        
        return re.sub(self.PATTERN, replace, cuda_code)


# =============================================================================
# Kernel Declaration Translation Rules
# =============================================================================

class GlobalKernelRule(TranslationRule):
    """Translates __global__ kernel declarations."""
    
    PATTERN = r'__global__\s+void\s+(\w+)\s*\(([^)]*)\)'
    
    def __init__(self):
        super().__init__(
            name="global_kernel",
            description="Translate __global__ kernel to RIPPLE function",
            cuda_pattern=r'__global__\s+void\s+(\w+)\s*\(([^)]*)\)\s*\{',
            priority=200
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            kernel_name = match.group(1)
            params = match.group(2).strip()

            # Parse parameters to add grid/block dimensions. No trailing
            # comma when the original kernel takes zero parameters (e.g.
            # __global__ void foo()) — a bare comma before the closing
            # paren is invalid C, which is exactly what shipped here
            # until caught by tests/examples/ast_flat.cu's syntax check.
            param_suffix = f",\n    {params}" if params else ""
            return f"""void {kernel_name}_ripple(
    int block_idx_x, int block_idx_y, int block_idx_z,
    int grid_dim_x, int grid_dim_y, int grid_dim_z,
    int block_dim_x, int block_dim_y, int block_dim_z{param_suffix}) {{
    RIPPLE_SETUP_BLOCK();"""

        return re.sub(r'__global__\s+void\s+(\w+)\s*\(([^)]*)\)\s*\{', replace, cuda_code)


class DeviceFunctionRule(TranslationRule):
    """Translates __device__ function declarations."""
    
    PATTERN = r'__device__\s+((?:inline\s+)?[\w\s\*]+)\s+(\w+)\s*\(([^)]*)\)'
    
    def __init__(self):
        super().__init__(
            name="device_function",
            description="Translate __device__ function to inline function",
            cuda_pattern=self.PATTERN,
            priority=190
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            return_type = match.group(1).strip()
            func_name = match.group(2)
            params = match.group(3)
            
            # __device__ functions become inline functions in RIPPLE
            if "inline" not in return_type:
                return f"static inline {return_type} {func_name}({params})"
            else:
                return f"static {return_type} {func_name}({params})"
        
        return re.sub(self.PATTERN, replace, cuda_code)


class HostDeviceRule(TranslationRule):
    """Translates __host__ __device__ combined declarations."""
    
    PATTERN = r'__host__\s+__device__\s+([\w\s\*]+)\s+(\w+)\s*\(([^)]*)\)'
    
    def __init__(self):
        super().__init__(
            name="host_device_function",
            description="Translate __host__ __device__ to regular function",
            cuda_pattern=self.PATTERN,
            priority=190
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        def replace(match):
            return_type = match.group(1).strip()
            func_name = match.group(2)
            params = match.group(3)
            return f"static inline {return_type} {func_name}({params})"
        
        return re.sub(self.PATTERN, replace, cuda_code)


# =============================================================================
# Math Function Translation Rules
# =============================================================================

class MathFunctionRule(TranslationRule):
    """Translates CUDA math functions to standard equivalents."""
    
    # Map CUDA math functions to standard C/HVX equivalents
    MATH_MAP = {
        '__fadd_rn': '+',  # Round-to-nearest add
        '__fmul_rn': '*',  # Round-to-nearest mul
        '__fdiv_rn': '/',  # Round-to-nearest div
        '__fsqrt_rn': 'sqrtf',
        '__expf': 'expf',
        '__logf': 'logf',
        '__powf': 'powf',
        '__sinf': 'sinf',
        '__cosf': 'cosf',
        '__tanf': 'tanf',
        'rsqrtf': '(1.0f / sqrtf',  # Reciprocal sqrt
        '__saturatef': 'fminf(1.0f, fmaxf(0.0f,',
        'fmaf': 'fmaf',
        '__fmaf_rn': 'fmaf',
        # Integer Intrinsics
        '__popc': '__builtin_popcount',
        '__popcll': '__builtin_popcountll',
        '__clz': '__builtin_clz',
        '__clzll': '__builtin_clzll',
        '__ffs': '__builtin_ffs',
        '__ffsll': '__builtin_ffsll',
        '__brev': '__builtin_bitreverse32',
        '__brevll': '__builtin_bitreverse64',
        '__sad': 'ripple_sad',  # We'll define a macro for this
    }
    
    PATTERN = r'(' + '|'.join(re.escape(k) for k in MATH_MAP.keys()) + r')\s*\('
    
    def __init__(self):
        super().__init__(
            name="math_functions",
            description="Translate CUDA math functions to standard",
            cuda_pattern=self.PATTERN,
            priority=50
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        result = cuda_code
        for cuda_fn, std_fn in self.MATH_MAP.items():
            if std_fn in ['+', '*', '/']:
                # Binary operators need different handling
                continue
            result = result.replace(f"{cuda_fn}(", f"{std_fn}(")
        return result


# =============================================================================
# Cooperative Groups Translation Rules
# =============================================================================

class CooperativeGroupsRule(TranslationRule):
    """Translates cooperative_groups constructs."""
    
    PATTERN = r'cooperative_groups::(\w+)'
    
    def __init__(self):
        super().__init__(
            name="cooperative_groups",
            description="Translate cooperative_groups to RIPPLE",
            cuda_pattern=self.PATTERN,
            priority=40
        )
    
    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        ctx.add_warning(
            "Cooperative groups require manual translation - "
            "see generated comments for guidance"
        )
        
        # Handle common patterns
        result = cuda_code
        result = re.sub(
            r'cooperative_groups::this_thread_block\(\)',
            '/* cg::this_thread_block() -> use ripple_block */',
            result
        )
        result = re.sub(
            r'cooperative_groups::tiled_partition<(\d+)>\([^)]+\)',
            r'/* cg::tiled_partition<\1> -> use ripple_slice for sub-block operations */',
            result
        )
        return result


# =============================================================================
# Rule Engine
# =============================================================================

class TranslationRuleEngine:
    """
    Manages and applies translation rules.
    
    Rules are applied in priority order (highest first).
    """
    
    def __init__(self):
        self.rules: list[TranslationRule] = []
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register all default translation rules."""
        default_rules = [
            # Kernel declarations (highest priority)
            GlobalKernelRule(),
            DeviceFunctionRule(),
            HostDeviceRule(),
            
            # Built-in variables
            ThreadIdxRule(),
            BlockDimRule(),
            BlockIdxRule(),
            GridDimRule(),
            
            # Memory
            SharedMemoryRule(),
            DynamicSharedMemoryRule(),
            
            # Synchronization
            SyncThreadsRule(),
            SyncWarpRule(),
            
            # Shuffles
            WarpReductionRule(),
            UnrollConstantShuffleLoopRule(),
            ShuffleDownRule(),
            ShuffleUpRule(),
            ShuffleXorRule(),
            ShuffleSyncRule(),
            
            # Atomics
            AtomicAddRule(),
            AtomicMaxRule(),
            AtomicMinRule(),
            AtomicCASRule(),
            AtomicExchRule(),
            
            # Math functions
            MathFunctionRule(),
            
            # Cooperative groups
            CooperativeGroupsRule(),
        ]
        
        for rule in default_rules:
            self.register_rule(rule)
    
    def register_rule(self, rule: TranslationRule):
        """Register a translation rule."""
        self.rules.append(rule)
        # Keep sorted by priority (descending)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def apply_all(self, cuda_code: str, ctx: TranslationContext) -> str:
        """Apply all matching rules to the CUDA code."""
        result = cuda_code
        
        for rule in self.rules:
            if rule.matches(result):
                result = rule.apply(result, ctx)
        
        return result
    
    def get_matching_rules(self, cuda_code: str) -> list[TranslationRule]:
        """Get all rules that match the given code."""
        return [r for r in self.rules if r.matches(cuda_code)]


# =============================================================================
# Block Shape Inference
# =============================================================================

def infer_block_shape(
    cuda_code: str,
    launch_config: Optional[CUDADim3] = None,
    ctx: TranslationContext = None
) -> RIPPLEBlockShape:
    """
    Infer optimal RIPPLE block shape from CUDA code patterns.
    
    For Hexagon HVX:
    - 128 bytes = 32 x int32 or 64 x int16 or 128 x int8
    - Prefer 1D blocks for simple kernels
    - Use 2D blocks for matrix operations
    """
    if ctx is None:
        ctx = TranslationContext()
    
    # Analyze usage patterns
    uses_2d = bool(re.search(r'threadIdx\.y|blockIdx\.y', cuda_code))
    uses_3d = bool(re.search(r'threadIdx\.z|blockIdx\.z', cuda_code))
    
    # Detect element type from common patterns
    if re.search(r'float\s*\*|float\s+\w+\[', cuda_code):
        elem_type = "float"
        hexagon_lanes = 32  # 128 bytes / 4 bytes
    elif re.search(r'double\s*\*|double\s+\w+\[', cuda_code):
        elem_type = "double"
        hexagon_lanes = 16  # 128 bytes / 8 bytes
    elif re.search(r'int16_t|short', cuda_code):
        elem_type = "int16_t"
        hexagon_lanes = 64  # 128 bytes / 2 bytes
    elif re.search(r'int8_t|char', cuda_code):
        elem_type = "int8_t"
        hexagon_lanes = 128  # 128 bytes / 1 byte
    else:
        elem_type = "int32_t"
        hexagon_lanes = 32  # Default: 128 bytes / 4 bytes
    
    # Determine dimensionality
    if uses_3d:
        dims = [hexagon_lanes, 1, 1]  # Flatten to 1D for Hexagon
        ctx.add_warning("3D blocks flattened for Hexagon HVX")
    elif uses_2d:
        # For 2D, use square-ish dimensions
        side = int(hexagon_lanes ** 0.5)
        dims = [side, side]
    else:
        dims = [hexagon_lanes]
    
    return RIPPLEBlockShape(
        pe_type=RIPPLEProcessingElement.HVX_PE,
        dimensions=dims
    )


# =============================================================================
# Kernel Restructuring
# =============================================================================

def restructure_kernel(
    air_function: AIRFunction,
    ctx: TranslationContext
) -> AIRFunction:
    """
    Restructure a CUDA kernel for RIPPLE.
    
    CUDA: grid of blocks, each block has threads
    RIPPLE: single function with ripple_parallel loops for grid iteration
    
    Transformation:
        kernel<<<grid, block>>>(args)
        
        becomes:
        
        for (block_idx_x = 0..grid.x)
          for (block_idx_y = 0..grid.y)
            for (block_idx_z = 0..grid.z)
              ripple_block = set_block_shape(block.x, block.y, block.z)
              // kernel body with threadIdx -> ripple_id
    """
    # This is a complex restructuring - we'll handle it in the full transformer
    return air_function
