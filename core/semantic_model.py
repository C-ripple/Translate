"""
CUDA to RIPPLE Semantic Model

This module defines the abstract intermediate representation (AIR) that bridges
CUDA semantics to RIPPLE semantics. Both the source-level and IR-level frontends
produce this representation, and the backend consumes it.

Architecture:
    CUDA Source  ──┐                    ┌──> RIPPLE C Code
                   ├──> Semantic AIR ───┤
    CUDA LLVM IR ──┘                    └──> RIPPLE LLVM IR
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Union
from abc import ABC, abstractmethod


# =============================================================================
# CUDA Semantic Concepts
# =============================================================================

class CUDAMemorySpace(Enum):
    """CUDA memory hierarchy."""
    GLOBAL = auto()      # __device__ / global memory
    SHARED = auto()      # __shared__ / on-chip scratchpad
    LOCAL = auto()       # thread-local (registers/spill)
    CONSTANT = auto()    # __constant__ / read-only cache
    TEXTURE = auto()     # texture memory (deprecated pattern)
    UNIFIED = auto()     # managed memory


class CUDASyncScope(Enum):
    """Synchronization scope levels."""
    WARP = auto()        # warp-level (32 threads)
    BLOCK = auto()       # block-level (__syncthreads)
    DEVICE = auto()      # device-level (cooperative groups)
    SYSTEM = auto()      # system-level (multi-GPU)


class CUDAAtomicOp(Enum):
    """Atomic operation types."""
    ADD = auto()
    SUB = auto()
    EXCH = auto()
    MIN = auto()
    MAX = auto()
    AND = auto()
    OR = auto()
    XOR = auto()
    CAS = auto()         # compare-and-swap


@dataclass
class CUDADim3:
    """Represents CUDA dim3 type for grid/block dimensions."""
    x: Union[int, str]  # Can be literal or expression
    y: Union[int, str] = 1
    z: Union[int, str] = 1

    def to_tuple(self) -> tuple:
        return (self.x, self.y, self.z)

    def dimensionality(self) -> int:
        """Returns effective dimensionality (1D, 2D, or 3D)."""
        if self.z != 1:
            return 3
        if self.y != 1:
            return 2
        return 1


@dataclass
class CUDABuiltinAccess:
    """Represents access to CUDA built-in variables."""
    builtin: str         # threadIdx, blockIdx, blockDim, gridDim, warpSize
    component: str       # x, y, z
    context: Optional[str] = None  # surrounding expression for context


@dataclass
class CUDASharedMemory:
    """Represents shared memory declaration."""
    name: str
    element_type: str
    size: Optional[Union[int, str]]  # None for dynamic shared memory
    is_dynamic: bool = False


@dataclass
class CUDAWarpShuffle:
    """Represents warp shuffle operations."""
    operation: str       # shfl_sync, shfl_up_sync, shfl_down_sync, shfl_xor_sync
    mask: str           # active thread mask
    value: str          # value to shuffle
    lane_or_delta: str  # source lane or delta
    width: int = 32     # logical warp width


@dataclass 
class CUDAReduction:
    """Represents a reduction operation pattern."""
    operation: CUDAAtomicOp
    target: str          # reduction target variable
    value: str           # value being reduced
    scope: CUDASyncScope


@dataclass
class CUDAKernelLaunch:
    """Represents a kernel launch configuration."""
    kernel_name: str
    grid_dim: CUDADim3
    block_dim: CUDADim3
    shared_mem_bytes: Union[int, str] = 0
    stream: Optional[str] = None
    arguments: list[str] = field(default_factory=list)


# =============================================================================
# RIPPLE Semantic Concepts (Target)
# =============================================================================

class RIPPLEProcessingElement(Enum):
    """RIPPLE processing element types."""
    VECTOR_PE = 0        # General vector unit
    SCALAR_PE = 1        # Scalar unit
    MATRIX_PE = 2        # Matrix unit (for SME-like targets)
    # Hexagon-specific
    HVX_PE = 10          # Hexagon Vector eXtensions


@dataclass
class RIPPLEBlockShape:
    """RIPPLE block shape definition."""
    pe_type: RIPPLEProcessingElement
    dimensions: list[Union[int, str]]  # Multi-dimensional shape

    def dimensionality(self) -> int:
        return len(self.dimensions)

    def to_call(self) -> str:
        """Generate ripple_set_block_shape call."""
        dims = ", ".join(str(d) for d in self.dimensions)
        return f"ripple_set_block_shape({self.pe_type.name}, {dims})"


@dataclass
class RIPPLEIndex:
    """RIPPLE lane index access."""
    block_var: str       # Block shape variable name
    dimension: int       # Which dimension to index


@dataclass
class RIPPLEBroadcast:
    """RIPPLE broadcast operation."""
    block_var: str
    dimension_mask: int  # Bitmask of dimensions to broadcast along
    value: str


@dataclass
class RIPPLEReduction:
    """RIPPLE reduction operation."""
    block_var: str
    dimension: int
    operation: str       # add, mul, min, max, and, or, xor
    value: str


@dataclass
class RIPPLEShuffle:
    """RIPPLE shuffle operation with shuffle function."""
    value: str
    shuffle_function: str  # Lambda or function defining the permutation


@dataclass
class RIPPLEParallelLoop:
    """RIPPLE parallel loop annotation."""
    block_var: str
    dimension: int
    loop_var: str
    start: str
    end: str
    is_full: bool = False  # True = ripple_parallel_full (no remainder)


# =============================================================================
# Abstract Intermediate Representation (AIR)
# =============================================================================

class AIRNode(ABC):
    """Base class for all AIR nodes."""
    
    @abstractmethod
    def accept(self, visitor: AIRVisitor) -> any:
        """Accept a visitor for traversal."""
        pass


@dataclass
class AIRType:
    """Type representation in AIR."""
    base_type: str           # int, float, double, etc.
    is_pointer: bool = False
    pointer_depth: int = 0
    is_const: bool = False
    address_space: Optional[CUDAMemorySpace] = None
    vector_width: Optional[int] = None  # For explicit vector types


@dataclass
class AIRVariable(AIRNode):
    """Variable in AIR."""
    name: str
    var_type: AIRType
    is_parameter: bool = False
    initial_value: Optional[str] = None
    
    def accept(self, visitor: AIRVisitor) -> any:
        return visitor.visit_variable(self)


@dataclass
class AIRExpression(AIRNode):
    """Expression in AIR."""
    expr: str
    expr_type: Optional[AIRType] = None
    cuda_origin: Optional[str] = None  # Original CUDA construct if relevant
    
    def accept(self, visitor: AIRVisitor) -> any:
        return visitor.visit_expression(self)


@dataclass
class AIRThreadIndex(AIRNode):
    """Thread index access (CUDA threadIdx -> RIPPLE ripple_id)."""
    cuda_builtin: CUDABuiltinAccess
    ripple_equivalent: Optional[RIPPLEIndex] = None
    
    def accept(self, visitor: AIRVisitor) -> any:
        return visitor.visit_thread_index(self)


@dataclass
class AIRMemoryOp(AIRNode):
    """Memory operation (load/store with address space awareness)."""
    operation: str           # load, store
    address: str
    value: Optional[str]     # For stores
    memory_space: CUDAMemorySpace
    is_atomic: bool = False
    atomic_op: Optional[CUDAAtomicOp] = None
    
    def accept(self, visitor: AIRVisitor) -> any:
        return visitor.visit_memory_op(self)


@dataclass
class AIRSynchronization(AIRNode):
    """Synchronization operation."""
    scope: CUDASyncScope
    cuda_call: str           # Original CUDA call
    
    def accept(self, visitor: AIRVisitor) -> any:
        return visitor.visit_synchronization(self)


@dataclass
class AIRShuffleOp(AIRNode):
    """Shuffle/permutation operation."""
    cuda_shuffle: Optional[CUDAWarpShuffle]
    ripple_shuffle: Optional[RIPPLEShuffle] = None
    
    def accept(self, visitor: AIRVisitor) -> any:
        return visitor.visit_shuffle_op(self)


@dataclass
class AIRReductionOp(AIRNode):
    """Reduction operation."""
    cuda_reduction: Optional[CUDAReduction]
    ripple_reduction: Optional[RIPPLEReduction] = None
    
    def accept(self, visitor: AIRVisitor) -> any:
        return visitor.visit_reduction_op(self)


@dataclass
class AIRLoop(AIRNode):
    """Loop construct."""
    loop_var: str
    init: str
    condition: str
    increment: str
    body: list[AIRNode]
    is_parallel: bool = False
    parallel_annotation: Optional[RIPPLEParallelLoop] = None
    
    def accept(self, visitor: AIRVisitor) -> any:
        return visitor.visit_loop(self)


@dataclass
class AIRConditional(AIRNode):
    """Conditional construct (if/else)."""
    condition: str
    then_body: list[AIRNode]
    else_body: Optional[list[AIRNode]] = None
    is_divergent: bool = False  # True if condition varies across threads
    
    def accept(self, visitor: AIRVisitor) -> any:
        return visitor.visit_conditional(self)


@dataclass
class AIRFunction(AIRNode):
    """Function/kernel definition."""
    name: str
    return_type: AIRType
    parameters: list[AIRVariable]
    body: list[AIRNode]
    
    # CUDA-specific
    is_kernel: bool = False
    is_device: bool = False
    cuda_attributes: list[str] = field(default_factory=list)
    
    # RIPPLE-specific
    block_shape: Optional[RIPPLEBlockShape] = None
    
    def accept(self, visitor: AIRVisitor) -> any:
        return visitor.visit_function(self)


@dataclass
class AIRTranslationUnit(AIRNode):
    """Top-level translation unit."""
    includes: list[str] = field(default_factory=list)
    typedefs: list[str] = field(default_factory=list)
    global_variables: list[AIRVariable] = field(default_factory=list)
    functions: list[AIRFunction] = field(default_factory=list)
    kernel_launches: list[CUDAKernelLaunch] = field(default_factory=list)
    
    # Metadata
    source_file: Optional[str] = None
    target_platform: str = "hexagon"
    
    def accept(self, visitor: AIRVisitor) -> any:
        return visitor.visit_translation_unit(self)


# =============================================================================
# Visitor Pattern for AIR Traversal
# =============================================================================

class AIRVisitor(ABC):
    """Abstract visitor for AIR traversal."""
    
    @abstractmethod
    def visit_variable(self, node: AIRVariable) -> any:
        pass
    
    @abstractmethod
    def visit_expression(self, node: AIRExpression) -> any:
        pass
    
    @abstractmethod
    def visit_thread_index(self, node: AIRThreadIndex) -> any:
        pass
    
    @abstractmethod
    def visit_memory_op(self, node: AIRMemoryOp) -> any:
        pass
    
    @abstractmethod
    def visit_synchronization(self, node: AIRSynchronization) -> any:
        pass
    
    @abstractmethod
    def visit_shuffle_op(self, node: AIRShuffleOp) -> any:
        pass
    
    @abstractmethod
    def visit_reduction_op(self, node: AIRReductionOp) -> any:
        pass
    
    @abstractmethod
    def visit_loop(self, node: AIRLoop) -> any:
        pass
    
    @abstractmethod
    def visit_conditional(self, node: AIRConditional) -> any:
        pass
    
    @abstractmethod
    def visit_function(self, node: AIRFunction) -> any:
        pass
    
    @abstractmethod
    def visit_translation_unit(self, node: AIRTranslationUnit) -> any:
        pass


# =============================================================================
# Translation Context
# =============================================================================

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


@dataclass
class TranslationContext:
    """Context maintained during translation."""
    
    # Target configuration
    target_platform: str = "hexagon"
    vector_width: int = 128  # Hexagon HVX default: 128 bytes = 1024 bits
    
    # Block shape inference
    inferred_block_shape: Optional[RIPPLEBlockShape] = None
    
    # Variable mappings
    thread_idx_mappings: dict[str, RIPPLEIndex] = field(default_factory=dict)
    shared_mem_mappings: dict[str, str] = field(default_factory=dict)
    
    # Diagnostics
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

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

    # Translation mode
    translation_mode: str = "source"  # "source" or "ir"
    preserve_comments: bool = True
    generate_debug_info: bool = False
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
    
    def add_error(self, msg: str):
        self.errors.append(msg)
    
    def has_errors(self) -> bool:
        return len(self.errors) > 0


# =============================================================================
# Hexagon-Specific Configuration
# =============================================================================

@dataclass
class HexagonConfig:
    """Hexagon-specific configuration for code generation."""
    
    # HVX configuration
    hvx_width: int = 128        # 128 bytes (1024 bits) or 64 bytes (512 bits)
    hvx_mode: str = "v68"       # v60, v62, v65, v66, v68, v69, v73
    
    # Memory configuration
    l2_cache_size: int = 1024   # KB
    vtcm_size: int = 256        # KB (Vector Tightly Coupled Memory)
    
    # Optimization hints
    prefer_hvx_scatter_gather: bool = True
    use_vtcm_for_shared: bool = True
    unroll_factor: int = 2
    
    def get_vector_lanes(self, element_type: str) -> int:
        """Get number of vector lanes for given element type."""
        type_sizes = {
            "int8_t": 1, "uint8_t": 1, "char": 1,
            "int16_t": 2, "uint16_t": 2, "short": 2,
            "int32_t": 4, "uint32_t": 4, "int": 4, "float": 4,
            "int64_t": 8, "uint64_t": 8, "double": 8,
        }
        elem_size = type_sizes.get(element_type, 4)
        return self.hvx_width // elem_size
