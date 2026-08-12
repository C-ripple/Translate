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


class WarpReductionRule(TranslationRule):
    """
    Detects and optimizes standard Warp Reduction loops.
    
    Pattern:
      for (int offset = warpSize/2; offset > 0; offset /= 2)
          val += __shfl_down_sync(..., val, offset);
          
    Replacement:
      val = ripple_reduceadd(val);
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
            
            return f"""/* CUDA Warp Reduction Loop -> RIPPLE Intrinsic */
    {accum_var} = ripple_reduceadd({accum_var});"""
        
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
