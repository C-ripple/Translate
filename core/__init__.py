"""
CUDA to RIPPLE Core Module

This module provides the core semantic model and translation rules
for converting CUDA to RIPPLE.
"""

from .semantic_model import (
    # CUDA concepts
    CUDAMemorySpace,
    CUDASyncScope,
    CUDAAtomicOp,
    CUDADim3,
    CUDABuiltinAccess,
    CUDASharedMemory,
    CUDAWarpShuffle,
    CUDAReduction,
    CUDAKernelLaunch,
    
    # RIPPLE concepts
    RIPPLEProcessingElement,
    RIPPLEBlockShape,
    RIPPLEIndex,
    RIPPLEBroadcast,
    RIPPLEReduction,
    RIPPLEShuffle,
    RIPPLEParallelLoop,
    
    # AIR (Abstract Intermediate Representation)
    AIRNode,
    AIRType,
    AIRVariable,
    AIRExpression,
    AIRThreadIndex,
    AIRMemoryOp,
    AIRSynchronization,
    AIRShuffleOp,
    AIRReductionOp,
    AIRLoop,
    AIRConditional,
    AIRFunction,
    AIRTranslationUnit,
    AIRVisitor,
    
    # Context
    TranslationContext,
    HexagonConfig,
)

from .translation_rules import (
    TranslationRule,
    TranslationRuleEngine,
    infer_block_shape,
    
    # Individual rules (for customization)
    ThreadIdxRule,
    BlockDimRule,
    BlockIdxRule,
    GridDimRule,
    SharedMemoryRule,
    SyncThreadsRule,
    ShuffleDownRule,
    ShuffleUpRule,
    ShuffleXorRule,
    AtomicAddRule,
    GlobalKernelRule,
    DeviceFunctionRule,
)

__all__ = [
    # CUDA
    'CUDAMemorySpace', 'CUDASyncScope', 'CUDAAtomicOp', 'CUDADim3',
    'CUDABuiltinAccess', 'CUDASharedMemory', 'CUDAWarpShuffle',
    'CUDAReduction', 'CUDAKernelLaunch',
    
    # RIPPLE
    'RIPPLEProcessingElement', 'RIPPLEBlockShape', 'RIPPLEIndex',
    'RIPPLEBroadcast', 'RIPPLEReduction', 'RIPPLEShuffle', 'RIPPLEParallelLoop',
    
    # AIR
    'AIRNode', 'AIRType', 'AIRVariable', 'AIRExpression', 'AIRThreadIndex',
    'AIRMemoryOp', 'AIRSynchronization', 'AIRShuffleOp', 'AIRReductionOp',
    'AIRLoop', 'AIRConditional', 'AIRFunction', 'AIRTranslationUnit', 'AIRVisitor',
    
    # Context
    'TranslationContext', 'HexagonConfig',
    
    # Rules
    'TranslationRule', 'TranslationRuleEngine', 'infer_block_shape',
]
