# CUDA to RIPPLE Translator - Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CUDA to RIPPLE Translator                           │
│                       Target: Hexagon HVX / SIMD                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║                         USER INTERFACES                               ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐ ║  │
│  ║  │     CLI      │  │  Web Server  │  │    VS Code Extension         │ ║  │
│  ║  │  cuda2ripple │  │  Flask App   │  │    (TypeScript)              │ ║  │
│  ║  │              │  │              │  │                              │ ║  │
│  ║  │ • source     │  │ • Live edit  │  │ • Context menu               │ ║  │
│  ║  │ • ir         │  │ • Preview    │  │ • Side-by-side               │ ║  │
│  ║  │ • analyze    │  │ • Export     │  │ • Status bar                 │ ║  │
│  ║  │ • batch      │  │              │  │                              │ ║  │
│  ║  │ • interactive│  │              │  │                              │ ║  │
│  ║  └──────────────┘  └──────────────┘  └──────────────────────────────┘ ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                    │                                        │
│                                    ▼                                        │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║                          FRONTENDS                                    ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║  ┌─────────────────────────────┐  ┌─────────────────────────────────┐ ║  │
│  ║  │    SOURCE FRONTEND          │  │      IR FRONTEND                │ ║  │
│  ║  │    (cuda_frontend.py)       │  │      (ir_frontend.py)           │ ║  │
│  ║  │                             │  │                                 │ ║  │
│  ║  │  CUDA Source (.cu)          │  │  LLVM IR (.ll)                  │ ║  │
│  ║  │         │                   │  │         │                       │ ║  │
│  ║  │         ▼                   │  │         ▼                       │ ║  │
│  ║  │  ┌───────────┐              │  │  ┌───────────┐                  │ ║  │
│  ║  │  │  Lexer    │              │  │  │ IR Parser │                  │ ║  │
│  ║  │  └─────┬─────┘              │  │  └─────┬─────┘                  │ ║  │
│  ║  │        ▼                    │  │        ▼                        │ ║  │
│  ║  │  ┌───────────┐              │  │  ┌───────────┐                  │ ║  │
│  ║  │  │Transformer│              │  │  │ Analyzer  │                  │ ║  │
│  ║  │  └─────┬─────┘              │  │  └─────┬─────┘                  │ ║  │
│  ║  │        │                    │  │        │                        │ ║  │
│  ║  └────────┼────────────────────┘  └────────┼────────────────────────┘ ║  │
│  ╚═══════════╪════════════════════════════════╪════════════════════════╝  │
│              │                                │                            │
│              └────────────────┬───────────────┘                            │
│                               │                                            │
│                               ▼                                            │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║                            CORE                                       ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║  ┌─────────────────────────────────────────────────────────────────┐  ║  │
│  ║  │                    SEMANTIC MODEL (AIR)                         │  ║  │
│  ║  │                    (semantic_model.py)                          │  ║  │
│  ║  │                                                                 │  ║  │
│  ║  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │  ║  │
│  ║  │  │ CUDA Types  │  │   AIR       │  │   RIPPLE Types          │  │  ║  │
│  ║  │  │             │=>│   Nodes     │=>│                         │  │  ║  │
│  ║  │  │• threadIdx  │  │             │  │• ripple_id()            │  │  ║  │
│  ║  │  │• blockDim   │  │• AIRFunction│  │• ripple_get_block_size()│  │  ║  │
│  ║  │  │• __shared__ │  │• AIRLoop    │  │• ripple_set_block()     │  │  ║  │
│  ║  │  │• atomicAdd  │  │• AIRMemOp   │  │• ripple_shuffle()       │  │  ║  │
│  ║  │  │• __shfl_*   │  │• AIRSync    │  │• ripple_parallel()      │  │  ║  │
│  ║  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │  ║  │
│  ║  └─────────────────────────────────────────────────────────────────┘  ║  │
│  ║                                                                       ║  │
│  ║  ┌─────────────────────────────────────────────────────────────────┐  ║  │
│  ║  │                   TRANSLATION RULES                             │  ║  │
│  ║  │                   (translation_rules.py)                        │  ║  │
│  ║  │                                                                 │  ║  │
│  ║  │  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │  ║  │
│  ║  │  │ ThreadIdxRule    │  │ SharedMemoryRule │  │ AtomicAddRule │  │  ║  │
│  ║  │  │ BlockDimRule     │  │ SyncThreadsRule  │  │ ShuffleRule   │  │  ║  │
│  ║  │  │ GlobalKernelRule │  │ DeviceFuncRule   │  │ MathFuncRule  │  │  ║  │
│  ║  │  └──────────────────┘  └──────────────────┘  └───────────────┘  │  ║  │
│  ║  │                                                                 │  ║  │
│  ║  │            TranslationRuleEngine (priority-ordered)             │  ║  │
│  ║  └─────────────────────────────────────────────────────────────────┘  ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                               │                                            │
│                               ▼                                            │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║                           OUTPUT                                      ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║  ┌─────────────────────────────┐  ┌─────────────────────────────────┐ ║  │
│  ║  │      RIPPLE C Code          │  │      RIPPLE LLVM IR             │ ║  │
│  ║  │      (.ripple.c)            │  │      (.ripple.ll)               │ ║  │
│  ║  │                             │  │                                 │ ║  │
│  ║  │  #include <ripple.h>        │  │  define void @kernel_ripple()   │ ║  │
│  ║  │  void kernel_ripple(...) {  │  │  {                              │ ║  │
│  ║  │    ripple_set_block_shape() │  │    call @llvm.ripple.set.block  │ ║  │
│  ║  │    int id = ripple_id(...)  │  │    %id = call @llvm.ripple.id   │ ║  │
│  ║  │    ...                      │  │    ...                          │ ║  │
│  ║  │  }                          │  │  }                              │ ║  │
│  ║  └─────────────────────────────┘  └─────────────────────────────────┘ ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                               │                                            │
│                               ▼                                            │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║                    RIPPLE COMPILER PASS                               ║  │
│  ║                    (External LLVM)                                    ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                       ║  │
│  ║       RIPPLE C/IR  →  Ripple.cpp Pass  →  Target SIMD Code            ║  │
│  ║                                                                       ║  │
│  ║  ┌─────────────────────────────────────────────────────────────────┐  ║  │
│  ║  │                     TARGET PLATFORMS                            │  ║  │
│  ║  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │  ║  │
│  ║  │  │ Hexagon HVX │  │  x86 AVX    │  │     ARM SVE/SME         │  │  ║  │
│  ║  │  │ 128B/64B    │  │  256/512b   │  │     Scalable            │  │  ║  │
│  ║  │  │ v60-v73     │  │             │  │                         │  │  ║  │
│  ║  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │  ║  │
│  ║  └─────────────────────────────────────────────────────────────────┘  ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Translation Flow

### Source-Level Path (Option A)
```
CUDA Source (.cu)
       │
       ▼
┌─────────────────┐
│   CUDALexer     │  Tokenizes CUDA-specific constructs
└────────┬────────┘
         ▼
┌─────────────────┐
│ TranslationRule │  Pattern matching with priority ordering
│    Engine       │  Applies ~20 rules for CUDA→RIPPLE
└────────┬────────┘
         ▼
┌─────────────────┐
│  Transformer    │  Adds headers, macros, Hexagon hints
└────────┬────────┘
         ▼
    RIPPLE C Code
```

### IR-Level Path (Option B)
```
CUDA LLVM IR (.ll)
       │
       ▼
┌─────────────────┐
│  LLVMIRParser   │  Parses NVPTX-flavored IR
└────────┬────────┘
         ▼
┌─────────────────┐
│   IRAnalyzer    │  Detects CUDA patterns:
│                 │  - Thread indices (llvm.nvvm.read.ptx.sreg.tid.*)
│                 │  - Shared memory (addrspace(3))
│                 │  - Barriers, shuffles, atomics
└────────┬────────┘
         ▼
┌─────────────────┐
│  IRTransformer  │  Replaces NVPTX intrinsics with RIPPLE
└────────┬────────┘
         ▼
   RIPPLE LLVM IR
```

## Key Translation Mappings

| CUDA | RIPPLE | Notes |
|------|--------|-------|
| `threadIdx.x` | `ripple_id(block, 0)` | Lane index in SIMD vector |
| `blockDim.x` | `ripple_get_block_size(block, 0)` | Vector width |
| `blockIdx.x` | Loop variable `block_idx_x` | Grid parallelism → loops |
| `__global__` | Regular function + `ripple_set_block_shape()` | |
| `__shared__` | `vtcm_malloc()`/`vtcm_free()` pair | Hexagon tightly-coupled memory |
| `__syncthreads()` | Implicit (comment) | SIMD lanes are lockstep |
| `atomicAdd()` | *(no equivalent)* | Ripple has no atomics API and no documented alternative for this pattern |
| `__shfl_down_sync()` | `ripple_shuffle(val, fn)` | With permutation function |

## Hexagon HVX Configuration

```python
HexagonConfig:
  hvx_width: 128          # bytes (1024 bits)
  hvx_mode: "v68"         # Instruction set version
  vtcm_size: 256          # KB
  use_vtcm_for_shared: True

# Vector lanes by type:
# int8_t:  128 lanes
# int16_t:  64 lanes  
# int32_t:  32 lanes
# float:    32 lanes
# int64_t:  16 lanes
```

## File Structure

```
cuda2ripple/
├── __init__.py              # Package entry point, high-level API
├── pyproject.toml           # Build configuration
├── setup.py                 # Legacy setup
├── core/
│   ├── semantic_model.py    # AIR definitions (800+ LOC)
│   └── translation_rules.py # Rule engine (800+ LOC)
├── frontends/
│   ├── source/
│   │   └── cuda_frontend.py # Lexer, transformer (700+ LOC)
│   └── ir/
│       └── ir_frontend.py   # IR parser, analyzer (500+ LOC)
├── interfaces/
│   ├── cli/
│   │   └── cuda2ripple.py   # CLI application (400+ LOC)
│   ├── web/
│   │   └── server.py        # Flask web app (300+ LOC)
│   └── vscode/
│       ├── package.json     # Extension manifest
│       └── src/extension.ts # Extension code
├── tests/
│   ├── test_translation.py  # Test suite
│   └── examples/
│       └── cuda_kernels.cu  # Example CUDA code
└── docs/
    └── README.md            # Documentation
```

## Usage Examples

### Python API
```python
from cuda2ripple import translate, analyze

# Translate CUDA to RIPPLE
ripple_code = translate(cuda_code, target="hexagon")

# Analyze without translation
info = analyze(cuda_code)
```

### Command Line
```bash
# Source translation
cuda2ripple source kernel.cu -o kernel.ripple.c

# IR translation
cuda2ripple ir kernel.ll -o kernel.ripple.ll

# Analysis
cuda2ripple analyze kernel.cu --json
```

### Web Interface
```bash
python -m cuda2ripple.interfaces.web.server
# Open http://localhost:5000
```
