"""
CUDA to RIPPLE Translation Rules

This module defines the semantic transformation rules for converting CUDA
constructs to their RIPPLE equivalents. These rules are used by both the
source-level and IR-level frontends.

Key Mappings:
    CUDA threadIdx.{x,y,z}  ->  ripple_id(block, {0,1,2})
    CUDA blockDim.{x,y,z}   ->  ripple_get_block_size(block, {0,1,2})
    CUDA __shared__         ->  vtcm_malloc()/vtcm_free() (Hexagon)
    CUDA __syncthreads()    ->  implicit (SIMD model) or explicit barrier
    CUDA atomicAdd          ->  no equivalent — hard-fails (see the Ripple
                                 multi-threading guide's barrier + partial-sum
                                 pattern), except the recognized block-reduction
                                 idiom, which maps to ripple_reduceadd
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

            free_call = f"\n    vtcm_free({var_name});"
            result = result[:free_pos] + free_call + result[free_pos:]

            malloc_decl = (
                f"// CUDA __shared__ -> Ripple VTCM\n"
                f"    {elem_type} *{var_name} = vtcm_malloc("
                f"sizeof({elem_type}) * ({size_expr}), /*align_as=*/128)"
            )
            result = result[:match.start()] + malloc_decl + result[match.end():]

        return result


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

    ButterflyAllReduceRule (higher priority, checked first) claims the
    specific subset of this shape that's a full-warp XOR-butterfly
    all-reduce with a full mask — see its docstring. This rule still
    handles everything else: partial-width/segmented reductions,
    non-full masks, and every non-reduction shuffle loop this pattern
    matches.

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

    # Also called directly by PredicatedShuffleUnrollRule — check that
    # class too when changing this method's signature or behavior.
    @staticmethod
    def _resolve_literal(token: str) -> int:
        """Resolves a captured INIT/BOUND token ('warpSize' or a plain
        integer literal string) to its integer value."""
        return 32 if token == 'warpSize' else int(token)

    # Also called directly by PredicatedShuffleUnrollRule — check that
    # class too when changing this method's signature or behavior.
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

    Two additional guards, both reasons to decline (leave the loop
    untouched) rather than fire — but not equally load-bearing: one
    closes a proven correctness gap; the other is a conservative
    decline — it bails rather than trying to prove the runtime bound
    stays untouched.
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


class TernaryDispatchShuffleArgumentRule(TranslationRule):
    r"""
    Resolves a shuffle argument that is a two-way, compile-time-constant
    ternary (`COND ? LITERAL_A : LITERAL_B`) by splitting the single
    dynamic call into two statically-argument'd calls behind an if/else
    on COND — the non-loop sibling of UnrollConstantShuffleLoopRule
    (GitHub issue #11). Where that rule resolves a shuffle argument
    whose value set is fixed by a loop's literal bounds, this rule
    resolves one whose value set is fixed to exactly two literals by a
    ternary, with COND deciding at runtime which literal applies. Same
    principle as PredicatedShuffleUnrollRule's docstring: a provably
    static VALUE SET, runtime-decided WHICH member of it — never a
    genuinely unbounded runtime value, which is still correctly a hard
    fail (ripple_shuffle's permutation function is file-scope and
    cannot close over a runtime value at all — see
    _is_compile_time_constant_expr's docstring).

    Matches two statement shapes feeding any of the 4 shuffle
    intrinsics (__shfl_sync, __shfl_down_sync, __shfl_up_sync,
    __shfl_xor_sync):
      TYPE VAR = INTRINSIC(mask, value, COND ? LIT_A : LIT_B[, width]);
      VAR (=|+=|-=|*=|/=) INTRINSIC(mask, value, COND ? LIT_A : LIT_B[, width]);
    Emits, respectively:
      TYPE VAR;
      if (COND) { VAR = INTRINSIC(mask, value, LIT_A[, width]); }
      else { VAR = INTRINSIC(mask, value, LIT_B[, width]); }
    or (assignment form, no separate declaration):
      if (COND) { VAR OP INTRINSIC(mask, value, LIT_A[, width]); }
      else { VAR OP INTRINSIC(mask, value, LIT_B[, width]); }
    Runs at priority 75 — above the shuffle rules (70), so the two
    literal-argument INTRINSIC calls this rule emits are still present,
    unconsumed, when the shuffle rules get their turn in the same pass
    (TranslationRuleEngine.apply_all runs each registered rule exactly
    once, highest priority first) and hoist a real ripple_shuffle
    permutation function for each branch, same as any other
    literal-argument call.

    LIT_A and LIT_B must each independently pass
    _is_compile_time_constant_expr — reused directly rather than
    re-derived, same rationale as PredicatedShuffleUnrollRule's
    docstring: that check went through review-driven hardening to get
    its digit/operator/paren-balance handling right. COND is NOT
    constrained the same way — it's expected to be a genuine runtime
    expression (that's the entire point of this rule), and is spliced
    into the emitted if/else verbatim. The declaration-form pattern's
    TYPE capture is intentionally imprecise about where a multi-word
    type ends (identical ambiguity to DeviceFunctionRule's existing
    `[\w\s\*]+` type capture elsewhere in this file) — this
    translator is pattern-based throughout, not a real C parser.

    Unlike UnrollConstantShuffleLoopRule/PredicatedShuffleUnrollRule,
    this rule's capture is a single statement terminated by the first
    `;` after the call's closing `)` — it never spans a `{...}` block,
    so the invisible brace/comment-swallowing risk those two classes
    guard against (see UnrollConstantShuffleLoopRule's docstring)
    structurally cannot happen here; no UNSAFE_BODY_TOKENS-style guard
    is needed.

    Known limitation, a deliberate bail-out and not a translation
    attempt: a nested/chained ternary (`COND1 ? A : COND2 ? B : C`)
    does not pass LIT_B's compile-time-constant-only check (it contains
    `?`/`:`), so this rule declines entirely and the original text
    reaches the shuffle rules unchanged, which correctly hard-fail on
    it — same "decline rather than guess" convention as every other
    rule in this file's shuffle-handling family. A 2-way ternary is the
    common case (a data-dependent binary choice of neighbor/stride) and
    the one this rule targets.
    """

    _INTRINSIC = r'__shfl(?:_(?:down|up|xor))?_sync'

    # Group numbers noted per pattern below — the two patterns are
    # matched and substituted independently, each with its own group
    # numbering, not shared.

    # TYPE VAR = INTRINSIC(mask, value, COND ? LIT_A : LIT_B[, width]);
    #   1: TYPE   2: VAR   3: INTRINSIC name   4: mask   5: value
    #   6: COND   7: LIT_A   8: LIT_B   9: optional width
    PATTERN_DECL = (
        r'(\w[\w\s\*]*?)\s+(\w+)\s*=\s*'
        r'(' + _INTRINSIC + r')\s*\(\s*'
        r'([^,]+),\s*([^,]+),\s*'
        r'([^,\)\?]+?)\s*\?\s*([^,\):]+?)\s*:\s*([^,\)]+?)\s*'
        r'(?:,\s*([^)]+))?\)\s*;'
    )

    # VAR (=|+=|-=|*=|/=) INTRINSIC(mask, value, COND ? LIT_A : LIT_B[, width]);
    #   1: VAR   2: assign op   3: INTRINSIC name   4: mask   5: value
    #   6: COND   7: LIT_A   8: LIT_B   9: optional width
    PATTERN_ASSIGN = (
        r'(\w+)\s*(\+=|-=|\*=|/=|=)\s*'
        r'(' + _INTRINSIC + r')\s*\(\s*'
        r'([^,]+),\s*([^,]+),\s*'
        r'([^,\)\?]+?)\s*\?\s*([^,\):]+?)\s*:\s*([^,\)]+?)\s*'
        r'(?:,\s*([^)]+))?\)\s*;'
    )

    # Kept for stylistic consistency with every other rule class in this
    # file (each defines a `PATTERN` attribute) — not actually read
    # anywhere for this class specifically, since matches()/apply()
    # below explicitly use PATTERN_DECL/PATTERN_ASSIGN instead.
    PATTERN = PATTERN_ASSIGN

    def __init__(self):
        super().__init__(
            name="ternary_dispatch_shuffle_argument",
            description="Dispatch a two-way compile-time-constant ternary shuffle argument into an if/else calling two literal-argument shuffle intrinsics",
            cuda_pattern=self.PATTERN_ASSIGN,
            priority=75  # Below PredicatedShuffleUnrollRule (81) /
                         # UnrollConstantShuffleLoopRule (82) — those
                         # match a for-loop shape, structurally disjoint
                         # from this rule's statement shape, so ordering
                         # relative to them doesn't affect correctness.
                         # Must be above the shuffle rules (70) so the
                         # two literal-argument calls this rule emits
                         # are still unconsumed text when the shuffle
                         # rules get their turn in the same pass.
        )

    def matches(self, cuda_code: str) -> bool:
        return bool(
            re.search(self.PATTERN_DECL, cuda_code)
            or re.search(self.PATTERN_ASSIGN, cuda_code)
        )

    @staticmethod
    def _build_calls(intrinsic, mask, value, lit_a, lit_b, width):
        width_arg = f", {width.strip()}" if width else ""
        true_call = f"{intrinsic}({mask}, {value}, {lit_a}{width_arg})"
        false_call = f"{intrinsic}({mask}, {value}, {lit_b}{width_arg})"
        return true_call, false_call

    def _replace_decl(self, match, ctx: TranslationContext) -> str:
        type_ = match.group(1).strip()
        var = match.group(2)
        intrinsic = match.group(3)
        mask = match.group(4).strip()
        value = match.group(5).strip()
        cond = match.group(6).strip()
        lit_a = match.group(7).strip()
        lit_b = match.group(8).strip()
        width = match.group(9)

        if not (_is_compile_time_constant_expr(lit_a)
                and _is_compile_time_constant_expr(lit_b)):
            # Not a provably static 2-value domain — leave untouched so
            # the shuffle rules hard-fail on the intact ternary, same
            # as any other unresolvable argument.
            return match.group(0)

        true_call, false_call = self._build_calls(
            intrinsic, mask, value, lit_a, lit_b, width
        )
        ctx.add_warning(
            f"Dispatched a two-way ternary shuffle argument for '{var}' "
            f"into an if/else with two literal-argument shuffle calls"
        )
        return (
            f"{type_} {var};\n"
            f"    if ({cond}) {{ {var} = {true_call}; }} "
            f"else {{ {var} = {false_call}; }}"
        )

    def _replace_assign(self, match, ctx: TranslationContext) -> str:
        var = match.group(1)
        assign_op = match.group(2)
        intrinsic = match.group(3)
        mask = match.group(4).strip()
        value = match.group(5).strip()
        cond = match.group(6).strip()
        lit_a = match.group(7).strip()
        lit_b = match.group(8).strip()
        width = match.group(9)

        if not (_is_compile_time_constant_expr(lit_a)
                and _is_compile_time_constant_expr(lit_b)):
            return match.group(0)

        true_call, false_call = self._build_calls(
            intrinsic, mask, value, lit_a, lit_b, width
        )
        ctx.add_warning(
            f"Dispatched a two-way ternary shuffle argument for '{var}' "
            f"into an if/else with two literal-argument shuffle calls"
        )
        return (
            f"if ({cond}) {{ {var} {assign_op} {true_call}; }} "
            f"else {{ {var} {assign_op} {false_call}; }}"
        )

    def apply(self, cuda_code: str, ctx: TranslationContext) -> str:
        # Declaration form first — it fully consumes "TYPE VAR = ..."
        # including the TYPE text, so nothing is left for the
        # assignment-only pattern to accidentally re-match on the same
        # call once this pass is done.
        result = re.sub(
            self.PATTERN_DECL,
            lambda m: self._replace_decl(m, ctx),
            cuda_code
        )
        result = re.sub(
            self.PATTERN_ASSIGN,
            lambda m: self._replace_assign(m, ctx),
            result
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
                    f"(UnrollConstantShuffleLoopRule) — this call cannot currently "
                    f"be translated automatically. If '{delta}' is genuinely "
                    f"runtime-dynamic, ripple_shuffle cannot help at all: stage the "
                    f"exchange through __shared__ instead — write at a known index, "
                    f"read back at the runtime index. RIPPLE lowers that read into "
                    f"a real gather automatically, and __syncthreads() is "
                    f"unnecessary since SIMD lanes already run in lockstep."
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
                    f"(UnrollConstantShuffleLoopRule) — this call cannot currently "
                    f"be translated automatically. If '{lane_mask}' is genuinely "
                    f"runtime-dynamic, ripple_shuffle cannot help at all: stage the "
                    f"exchange through __shared__ instead — write at a known index, "
                    f"read back at the runtime index. RIPPLE lowers that read into "
                    f"a real gather automatically, and __syncthreads() is "
                    f"unnecessary since SIMD lanes already run in lockstep."
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
                    f"(UnrollConstantShuffleLoopRule) — this call cannot currently "
                    f"be translated automatically. If '{delta}' is genuinely "
                    f"runtime-dynamic, ripple_shuffle cannot help at all: stage the "
                    f"exchange through __shared__ instead — write at a known index, "
                    f"read back at the runtime index. RIPPLE lowers that read into "
                    f"a real gather automatically, and __syncthreads() is "
                    f"unnecessary since SIMD lanes already run in lockstep."
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
                    f"(UnrollConstantShuffleLoopRule) — this call cannot currently "
                    f"be translated automatically. If '{src_lane}' is genuinely "
                    f"runtime-dynamic, ripple_shuffle cannot help at all: stage the "
                    f"exchange through __shared__ instead — write at a known index, "
                    f"read back at the runtime index. RIPPLE lowers that read into "
                    f"a real gather automatically, and __syncthreads() is "
                    f"unnecessary since SIMD lanes already run in lockstep."
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
        r'\w+\s+(\w+)\s*=\s*__shfl_down_sync\s*\(\s*[^,]+,\s*(\w+)\s*,\s*\1\s*(?:,[^)]+)?\)\s*;\s*'
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

    The two guards are enforced at different layers: BOUND is baked
    into PATTERN itself (the `32|warpSize` alternation), so a smaller
    bound simply never matches — matches() already returns False. MASK
    can't be guard-railed the same way (it needs a semantic literal
    comparison, not a shape check), so PATTERN captures it loosely and
    apply() checks it explicitly against FULL_MASK before deciding
    whether to fire.
    """

    # Group 1: Loop variable (e.g. "i")
    # Group 2: Loop bound — literal "32" or symbolic "warpSize"
    # Group 3: Accumulator variable (e.g. "val")
    # Group 4: Shuffle mask expression — checked against FULL_MASK below,
    #          not constrained by the pattern itself (see class docstring)
    PATTERN = (
        r'for\s*\(\s*int\s+(\w+)\s*=\s*1\s*;\s*'
        r'\1\s*<\s*(32|warpSize)\s*;\s*'
        r'\1\s*\*=\s*2\s*\)\s*\{\s*'
        r'(\w+)\s*\+=\s*__shfl_xor_sync\s*\(\s*([^,]+)\s*,\s*\3\s*,\s*\1\s*\)\s*;\s*\}'
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


# =============================================================================
# Atomic Operation Translation Rules
# =============================================================================

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
        '__sad': 'cripple_sad',  # Translator-provided helper — no real Ripple SAD primitive exists
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
            WarpMinMaxReductionRule(),
            ButterflyAllReduceRule(),
            UnrollConstantShuffleLoopRule(),
            PredicatedShuffleUnrollRule(),
            TernaryDispatchShuffleArgumentRule(),
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
