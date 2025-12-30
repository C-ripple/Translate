"""
CUDA to RIPPLE Frontends

This package provides two translation frontends:
- source: Source-level CUDA to RIPPLE C translation
- ir: LLVM IR level NVPTX to RIPPLE translation
"""

from .source.cuda_frontend import (
    CUDALexer,
    CUDAToRIPPLETransformer,
    AIRBuilder,
    translate_cuda_source,
    translate_cuda_file,
)

from .ir.ir_frontend import (
    LLVMIRParser,
    IRAnalyzer,
    IRTransformer,
    CUDAIRToRIPPLETranslator,
    translate_cuda_ir,
    analyze_cuda_ir,
)

__all__ = [
    # Source frontend
    'CUDALexer', 'CUDAToRIPPLETransformer', 'AIRBuilder',
    'translate_cuda_source', 'translate_cuda_file',
    
    # IR frontend
    'LLVMIRParser', 'IRAnalyzer', 'IRTransformer',
    'CUDAIRToRIPPLETranslator', 'translate_cuda_ir', 'analyze_cuda_ir',
]
