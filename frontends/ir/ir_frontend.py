"""
CUDA LLVM IR Frontend

This module provides IR-level translation from CUDA LLVM IR to RIPPLE.
It analyzes NVPTX/NVVM IR patterns and transforms them to patterns that
the RIPPLE pass can process.

Architecture:
    CUDA LLVM IR (NVPTX) -> IR Analyzer -> AIR Builder -> AIR
                                                           |
                                                           v
                                      RIPPLE LLVM IR <- IR Generator
                                            |
                                            v
                                      RIPPLE Pass -> Target SIMD Code

Use Case:
    When you have compiled CUDA (.ptx, .cubin, or .ll) and want to
    retarget to Hexagon or other SIMD architectures.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Iterator
from pathlib import Path
import re

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.semantic_model import (
    AIRNode, AIRFunction, AIRVariable, AIRType, AIRExpression,
    AIRThreadIndex, AIRLoop, AIRConditional, AIRMemoryOp,
    AIRTranslationUnit, AIRSynchronization, AIRShuffleOp, AIRReductionOp,
    CUDABuiltinAccess, CUDADim3, CUDAMemorySpace, CUDASyncScope,
    RIPPLEBlockShape, RIPPLEIndex, RIPPLEProcessingElement,
    TranslationContext, HexagonConfig
)


# =============================================================================
# LLVM IR Types
# =============================================================================

@dataclass
class LLVMType:
    """Represents an LLVM IR type."""
    name: str
    is_pointer: bool = False
    pointer_address_space: int = 0  # NVPTX: 0=generic, 1=global, 3=shared, 4=const
    element_type: Optional[LLVMType] = None
    vector_size: Optional[int] = None
    
    @classmethod
    def parse(cls, type_str: str) -> LLVMType:
        """Parse an LLVM type string."""
        type_str = type_str.strip()
        
        # Vector type: <N x type>
        vec_match = re.match(r'<(\d+)\s+x\s+(.+)>', type_str)
        if vec_match:
            return cls(
                name="vector",
                vector_size=int(vec_match.group(1)),
                element_type=cls.parse(vec_match.group(2))
            )
        
        # Pointer type: type* or ptr addrspace(N)
        if type_str.endswith('*'):
            return cls(
                name="ptr",
                is_pointer=True,
                element_type=cls.parse(type_str[:-1])
            )
        
        ptr_match = re.match(r'ptr(?:\s+addrspace\((\d+)\))?', type_str)
        if ptr_match:
            return cls(
                name="ptr",
                is_pointer=True,
                pointer_address_space=int(ptr_match.group(1) or 0)
            )
        
        # Basic types
        return cls(name=type_str)


@dataclass
class LLVMValue:
    """Represents an LLVM IR value."""
    name: str
    llvm_type: LLVMType
    is_constant: bool = False
    constant_value: Optional[str] = None


@dataclass
class LLVMInstruction:
    """Represents an LLVM IR instruction."""
    opcode: str
    result: Optional[LLVMValue] = None
    operands: list[LLVMValue] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    raw_text: str = ""


@dataclass
class LLVMBasicBlock:
    """Represents an LLVM basic block."""
    label: str
    instructions: list[LLVMInstruction] = field(default_factory=list)
    predecessors: list[str] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)


@dataclass
class LLVMFunction:
    """Represents an LLVM function."""
    name: str
    return_type: LLVMType
    parameters: list[LLVMValue] = field(default_factory=list)
    basic_blocks: list[LLVMBasicBlock] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    is_kernel: bool = False  # NVPTX: has nvptx.kernel metadata


@dataclass
class LLVMModule:
    """Represents an LLVM module."""
    source_filename: str = ""
    target_datalayout: str = ""
    target_triple: str = ""
    globals: list[LLVMValue] = field(default_factory=list)
    functions: list[LLVMFunction] = field(default_factory=list)
    named_metadata: dict[str, list[str]] = field(default_factory=dict)


# =============================================================================
# NVPTX Intrinsic Patterns
# =============================================================================

class NVPTXIntrinsic(Enum):
    """NVPTX intrinsics that need to be translated."""
    # Thread/block IDs
    TID_X = "llvm.nvvm.read.ptx.sreg.tid.x"
    TID_Y = "llvm.nvvm.read.ptx.sreg.tid.y"
    TID_Z = "llvm.nvvm.read.ptx.sreg.tid.z"
    NTID_X = "llvm.nvvm.read.ptx.sreg.ntid.x"
    NTID_Y = "llvm.nvvm.read.ptx.sreg.ntid.y"
    NTID_Z = "llvm.nvvm.read.ptx.sreg.ntid.z"
    CTAID_X = "llvm.nvvm.read.ptx.sreg.ctaid.x"
    CTAID_Y = "llvm.nvvm.read.ptx.sreg.ctaid.y"
    CTAID_Z = "llvm.nvvm.read.ptx.sreg.ctaid.z"
    NCTAID_X = "llvm.nvvm.read.ptx.sreg.nctaid.x"
    NCTAID_Y = "llvm.nvvm.read.ptx.sreg.nctaid.y"
    NCTAID_Z = "llvm.nvvm.read.ptx.sreg.nctaid.z"
    
    # Barriers
    BAR_SYNC = "llvm.nvvm.barrier0"
    BAR_WARP_SYNC = "llvm.nvvm.bar.warp.sync"
    
    # Shuffles
    SHFL_SYNC = "llvm.nvvm.shfl.sync"
    SHFL_DOWN_SYNC = "llvm.nvvm.shfl.sync.down"
    SHFL_UP_SYNC = "llvm.nvvm.shfl.sync.up"
    SHFL_XOR_SYNC = "llvm.nvvm.shfl.sync.bfly"
    
    # Atomics
    ATOMIC_ADD = "llvm.nvvm.atomic.add"
    ATOMIC_EXCH = "llvm.nvvm.atomic.exch"
    ATOMIC_CAS = "llvm.nvvm.atomic.cas"


# Map NVPTX intrinsics to RIPPLE equivalents
NVPTX_TO_RIPPLE = {
    NVPTXIntrinsic.TID_X: ("ripple_id", 0),
    NVPTXIntrinsic.TID_Y: ("ripple_id", 1),
    NVPTXIntrinsic.TID_Z: ("ripple_id", 2),
    NVPTXIntrinsic.NTID_X: ("ripple_get_size", 0),
    NVPTXIntrinsic.NTID_Y: ("ripple_get_size", 1),
    NVPTXIntrinsic.NTID_Z: ("ripple_get_size", 2),
}


# =============================================================================
# LLVM IR Parser
# =============================================================================

class LLVMIRParser:
    """
    Parser for LLVM IR text format.
    
    Handles NVPTX-flavored LLVM IR from compiled CUDA code.
    """
    
    def __init__(self, ir_text: str):
        self.ir_text = ir_text
        self.lines = ir_text.split('\n')
        self.pos = 0
        self.module = LLVMModule()
    
    def current_line(self) -> Optional[str]:
        if self.pos >= len(self.lines):
            return None
        return self.lines[self.pos]
    
    def advance(self) -> Optional[str]:
        line = self.current_line()
        self.pos += 1
        return line
    
    def parse(self) -> LLVMModule:
        """Parse the entire module."""
        while self.current_line() is not None:
            line = self.current_line().strip()
            
            # Skip empty lines and comments
            if not line or line.startswith(';'):
                self.advance()
                continue
            
            # Module-level declarations
            if line.startswith('source_filename'):
                self._parse_source_filename(line)
            elif line.startswith('target datalayout'):
                self._parse_datalayout(line)
            elif line.startswith('target triple'):
                self._parse_triple(line)
            elif line.startswith('@'):
                self._parse_global(line)
            elif line.startswith('define'):
                self._parse_function()
            elif line.startswith('declare'):
                self._parse_declaration(line)
            elif line.startswith('!'):
                self._parse_metadata(line)
            else:
                self.advance()
        
        return self.module
    
    def _parse_source_filename(self, line: str):
        match = re.search(r'source_filename\s*=\s*"([^"]*)"', line)
        if match:
            self.module.source_filename = match.group(1)
        self.advance()
    
    def _parse_datalayout(self, line: str):
        match = re.search(r'target datalayout\s*=\s*"([^"]*)"', line)
        if match:
            self.module.target_datalayout = match.group(1)
        self.advance()
    
    def _parse_triple(self, line: str):
        match = re.search(r'target triple\s*=\s*"([^"]*)"', line)
        if match:
            self.module.target_triple = match.group(1)
        self.advance()
    
    def _parse_global(self, line: str):
        # Parse global variable declaration
        match = re.match(r'@(\w+)\s*=\s*(.*)', line)
        if match:
            name = match.group(1)
            rest = match.group(2)
            self.module.globals.append(LLVMValue(
                name=name,
                llvm_type=LLVMType(name="unknown"),
                is_constant='constant' in rest
            ))
        self.advance()
    
    def _parse_function(self):
        """Parse a function definition."""
        line = self.current_line()
        
        # Parse function header
        # define [linkage] [attrs] rettype @name(params) [attrs] {
        header_match = re.match(
            r'define\s+(?:[\w]+\s+)*'  # linkage/attrs
            r'([\w\s\*<>]+?)\s*'       # return type
            r'@([\w\.]+)\s*'           # function name
            r'\(([^)]*)\)',            # parameters
            line
        )
        
        if not header_match:
            self.advance()
            return
        
        return_type_str = header_match.group(1).strip()
        func_name = header_match.group(2)
        params_str = header_match.group(3)
        
        func = LLVMFunction(
            name=func_name,
            return_type=LLVMType.parse(return_type_str)
        )
        
        # Check for kernel marker
        if 'ptx_kernel' in line or 'nvvm.annotations' in line:
            func.is_kernel = True
        
        # Parse parameters
        if params_str.strip():
            for param in params_str.split(','):
                param = param.strip()
                if param:
                    parts = param.rsplit(' ', 1)
                    if len(parts) == 2:
                        param_type, param_name = parts
                        func.parameters.append(LLVMValue(
                            name=param_name.strip('%'),
                            llvm_type=LLVMType.parse(param_type)
                        ))
        
        self.advance()
        
        # Parse basic blocks until we hit the closing brace
        current_bb = None
        brace_count = 1
        
        while self.current_line() is not None and brace_count > 0:
            line = self.current_line().strip()
            
            # Track braces
            brace_count += line.count('{') - line.count('}')
            
            if brace_count <= 0:
                self.advance()
                break
            
            # Basic block label
            if line.endswith(':') and not line.startswith('%'):
                label = line[:-1]
                current_bb = LLVMBasicBlock(label=label)
                func.basic_blocks.append(current_bb)
            elif line and current_bb is not None:
                # Parse instruction
                inst = self._parse_instruction(line)
                if inst:
                    current_bb.instructions.append(inst)
            
            self.advance()
        
        self.module.functions.append(func)
    
    def _parse_instruction(self, line: str) -> Optional[LLVMInstruction]:
        """Parse a single LLVM instruction."""
        line = line.strip()
        
        if not line or line.startswith(';'):
            return None
        
        # Assignment form: %result = opcode operands
        assign_match = re.match(r'%(\w+)\s*=\s*(\w+)\s*(.*)', line)
        if assign_match:
            result_name = assign_match.group(1)
            opcode = assign_match.group(2)
            result = LLVMValue(
                name=result_name,
                llvm_type=LLVMType(name="unknown")
            )
        else:
            # No result: opcode operands
            opcode_match = re.match(r'(\w+)\s*(.*)', line)
            if not opcode_match:
                return None
            opcode = opcode_match.group(1)
            result = None

        return LLVMInstruction(opcode=opcode, result=result, raw_text=line)
    
    def _parse_declaration(self, line: str):
        """Parse a function declaration."""
        self.advance()
    
    def _parse_metadata(self, line: str):
        """Parse named metadata."""
        match = re.match(r'!(\w+)\s*=\s*!{(.+)}', line)
        if match:
            name = match.group(1)
            values = match.group(2).split(',')
            self.module.named_metadata[name] = [v.strip() for v in values]
        self.advance()


# =============================================================================
# IR Analyzer - Detects CUDA patterns in LLVM IR
# =============================================================================

@dataclass
class IRAnalysisResult:
    """Results of analyzing LLVM IR for CUDA patterns."""
    kernels: list[LLVMFunction] = field(default_factory=list)
    device_functions: list[LLVMFunction] = field(default_factory=list)
    
    # Detected patterns
    thread_idx_uses: list[tuple[str, int]] = field(default_factory=list)  # (func, dim)
    block_dim_uses: list[tuple[str, int]] = field(default_factory=list)
    shared_mem_accesses: list[str] = field(default_factory=list)
    barrier_calls: list[str] = field(default_factory=list)
    shuffle_calls: list[str] = field(default_factory=list)
    atomic_calls: list[str] = field(default_factory=list)
    
    # Inferred properties
    max_block_dim: int = 1
    uses_shared_memory: bool = False
    uses_warp_shuffles: bool = False
    uses_atomics: bool = False


class IRAnalyzer:
    """
    Analyzes NVPTX LLVM IR to detect CUDA patterns.
    
    This is the first phase of IR-level translation.
    """
    
    def __init__(self, module: LLVMModule):
        self.module = module
        self.result = IRAnalysisResult()
    
    def analyze(self) -> IRAnalysisResult:
        """Perform full analysis of the module."""
        for func in self.module.functions:
            self._analyze_function(func)
        
        return self.result
    
    def _analyze_function(self, func: LLVMFunction):
        """Analyze a single function."""
        # Check if this is a kernel
        if func.is_kernel or self._detect_kernel_pattern(func):
            self.result.kernels.append(func)
            func.is_kernel = True
        else:
            self.result.device_functions.append(func)
        
        # Analyze instructions
        for bb in func.basic_blocks:
            for inst in bb.instructions:
                self._analyze_instruction(inst, func.name)
    
    def _detect_kernel_pattern(self, func: LLVMFunction) -> bool:
        """Detect if function is a kernel by its patterns."""
        # Kernels typically use threadIdx/blockIdx intrinsics
        for bb in func.basic_blocks:
            for inst in bb.instructions:
                if 'nvvm.read.ptx.sreg' in inst.raw_text:
                    return True
        return False
    
    def _analyze_instruction(self, inst: LLVMInstruction, func_name: str):
        """Analyze a single instruction for CUDA patterns."""
        raw = inst.raw_text
        
        # Thread index intrinsics
        for intrinsic in [NVPTXIntrinsic.TID_X, NVPTXIntrinsic.TID_Y, NVPTXIntrinsic.TID_Z]:
            if intrinsic.value in raw:
                dim = 0 if 'tid.x' in raw else (1 if 'tid.y' in raw else 2)
                self.result.thread_idx_uses.append((func_name, dim))
                self.result.max_block_dim = max(self.result.max_block_dim, dim + 1)
        
        # Block dimension intrinsics
        for intrinsic in [NVPTXIntrinsic.NTID_X, NVPTXIntrinsic.NTID_Y, NVPTXIntrinsic.NTID_Z]:
            if intrinsic.value in raw:
                dim = 0 if 'ntid.x' in raw else (1 if 'ntid.y' in raw else 2)
                self.result.block_dim_uses.append((func_name, dim))
        
        # Shared memory (address space 3 in NVPTX)
        if 'addrspace(3)' in raw:
            self.result.shared_mem_accesses.append(func_name)
            self.result.uses_shared_memory = True
        
        # Barriers
        if 'barrier0' in raw or 'bar.warp.sync' in raw:
            self.result.barrier_calls.append(func_name)
        
        # Shuffles
        if 'shfl' in raw:
            self.result.shuffle_calls.append(func_name)
            self.result.uses_warp_shuffles = True
        
        # Atomics
        if 'atomicrmw' in raw or 'cmpxchg' in raw or 'nvvm.atomic' in raw:
            self.result.atomic_calls.append(func_name)
            self.result.uses_atomics = True


# =============================================================================
# IR Transformer - Transforms NVPTX IR to RIPPLE-compatible IR
# =============================================================================

class IRTransformer:
    """
    Transforms NVPTX LLVM IR to RIPPLE-compatible LLVM IR.
    
    This is the second phase of IR-level translation.
    """
    
    def __init__(self, module: LLVMModule, analysis: IRAnalysisResult, ctx: TranslationContext):
        self.module = module
        self.analysis = analysis
        self.ctx = ctx
        self.hexagon_config = HexagonConfig()
    
    def transform(self) -> str:
        """Transform the module and return RIPPLE-compatible LLVM IR."""
        lines = []
        
        # Header
        lines.extend(self._generate_header())
        
        # Transformed functions
        for func in self.module.functions:
            lines.extend(self._transform_function(func))
        
        return '\n'.join(lines)
    
    def _generate_header(self) -> list[str]:
        """Generate module header for RIPPLE."""
        target = "hexagon-unknown-linux" if self.ctx.target_platform == "hexagon" else "x86_64-unknown-linux"
        
        return [
            f'; ModuleID = "cuda2ripple_output"',
            f'source_filename = "{self.module.source_filename}"',
            f'target datalayout = "e-m:e-p:32:32-i64:64-v1024:1024-n8:16:32:64-S128"',  # Hexagon
            f'target triple = "{target}"',
            '',
            '; Declare RIPPLE intrinsics',
            'declare i32 @llvm.ripple.set.block.shape(i32, i32, ...)',
            'declare i32 @llvm.ripple.id(ptr, i32)',
            'declare i32 @llvm.ripple.get.size(ptr, i32)',
            'declare void @llvm.ripple.parallel.loop(ptr, i32)',
            '',
        ]
    
    def _transform_function(self, func: LLVMFunction) -> list[str]:
        """Transform a single function."""
        lines = []
        
        if func.is_kernel:
            lines.extend(self._transform_kernel(func))
        else:
            lines.extend(self._transform_device_function(func))
        
        return lines
    
    def _transform_kernel(self, func: LLVMFunction) -> list[str]:
        """Transform a CUDA kernel to RIPPLE function."""
        lines = []
        
        # Function signature
        ret_type = func.return_type.name
        param_strs = []
        
        # Add grid dimension parameters
        param_strs.extend([
            'i32 %grid_dim_x', 'i32 %grid_dim_y', 'i32 %grid_dim_z',
            'i32 %block_dim_x', 'i32 %block_dim_y', 'i32 %block_dim_z'
        ])
        
        # Original parameters
        for param in func.parameters:
            param_strs.append(f'{param.llvm_type.name} %{param.name}')
        
        lines.append(f'define {ret_type} @{func.name}_ripple({", ".join(param_strs)}) {{')
        lines.append('entry:')
        
        # Initialize RIPPLE block
        lines.append('  ; Initialize RIPPLE block')
        lines.append('  %ripple_block = alloca i8, i32 64, align 8')
        lines.append('  call i32 @llvm.ripple.set.block.shape(i32 0, i32 %block_dim_x, i32 %block_dim_y, i32 %block_dim_z)')
        lines.append('')
        
        # Transform basic blocks
        for bb in func.basic_blocks:
            if bb.label != 'entry':
                lines.append(f'{bb.label}:')
            
            for inst in bb.instructions:
                transformed = self._transform_instruction(inst)
                if transformed:
                    lines.append(f'  {transformed}')
        
        lines.append('}')
        lines.append('')
        
        return lines
    
    def _transform_device_function(self, func: LLVMFunction) -> list[str]:
        """Transform a device function to inline function."""
        lines = []
        
        ret_type = func.return_type.name
        param_strs = [f'{p.llvm_type.name} %{p.name}' for p in func.parameters]
        
        lines.append(f'define internal {ret_type} @{func.name}({", ".join(param_strs)}) alwaysinline {{')
        
        for bb in func.basic_blocks:
            lines.append(f'{bb.label}:')
            for inst in bb.instructions:
                transformed = self._transform_instruction(inst)
                if transformed:
                    lines.append(f'  {transformed}')
        
        lines.append('}')
        lines.append('')
        
        return lines
    
    def _transform_instruction(self, inst: LLVMInstruction) -> Optional[str]:
        """Transform a single instruction."""
        raw = inst.raw_text
        
        # Transform thread index intrinsics
        for intrinsic, (ripple_call, dim) in NVPTX_TO_RIPPLE.items():
            if intrinsic.value in raw:
                result = inst.result.name if inst.result else "tmp"
                return f'%{result} = call i32 @llvm.{ripple_call}(ptr %ripple_block, i32 {dim})'
        
        # Transform barriers (often no-op in SIMD)
        if 'barrier0' in raw:
            return '; barrier - implicit in RIPPLE SIMD model'
        
        # Transform shared memory accesses
        if 'addrspace(3)' in raw:
            # Change address space from 3 (shared) to 0 (generic)
            return raw.replace('addrspace(3)', 'addrspace(0)')
        
        # Pass through unchanged
        return raw


# =============================================================================
# IR-Level Translator Entry Point
# =============================================================================

class CUDAIRToRIPPLETranslator:
    """
    Main entry point for IR-level CUDA to RIPPLE translation.
    
    Usage:
        translator = CUDAIRToRIPPLETranslator()
        ripple_ir = translator.translate(cuda_ir)
    """
    
    def __init__(self, ctx: Optional[TranslationContext] = None):
        self.ctx = ctx or TranslationContext(target_platform="hexagon")
    
    def translate(self, cuda_ir: str) -> str:
        """
        Translate CUDA LLVM IR to RIPPLE LLVM IR.
        
        Args:
            cuda_ir: NVPTX LLVM IR text
        
        Returns:
            RIPPLE-compatible LLVM IR text
        """
        # Phase 1: Parse
        parser = LLVMIRParser(cuda_ir)
        module = parser.parse()
        
        # Phase 2: Analyze
        analyzer = IRAnalyzer(module)
        analysis = analyzer.analyze()
        
        # Phase 3: Transform
        transformer = IRTransformer(module, analysis, self.ctx)
        ripple_ir = transformer.transform()
        
        return ripple_ir
    
    def translate_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """Translate a CUDA IR file."""
        with open(input_path, 'r') as f:
            cuda_ir = f.read()
        
        ripple_ir = self.translate(cuda_ir)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(ripple_ir)
        
        return ripple_ir
    
    def get_analysis(self, cuda_ir: str) -> IRAnalysisResult:
        """Get analysis results without full translation."""
        parser = LLVMIRParser(cuda_ir)
        module = parser.parse()
        analyzer = IRAnalyzer(module)
        return analyzer.analyze()


# =============================================================================
# Convenience Functions
# =============================================================================

def translate_cuda_ir(cuda_ir: str, target: str = "hexagon") -> str:
    """
    Translate CUDA LLVM IR to RIPPLE LLVM IR.
    
    Args:
        cuda_ir: NVPTX LLVM IR text
        target: Target platform
    
    Returns:
        RIPPLE-compatible LLVM IR text
    """
    ctx = TranslationContext(target_platform=target)
    translator = CUDAIRToRIPPLETranslator(ctx)
    return translator.translate(cuda_ir)


def analyze_cuda_ir(cuda_ir: str) -> IRAnalysisResult:
    """
    Analyze CUDA IR for patterns.
    
    Returns analysis results without transformation.
    """
    ctx = TranslationContext()
    translator = CUDAIRToRIPPLETranslator(ctx)
    return translator.get_analysis(cuda_ir)
