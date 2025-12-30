# CUDA to RIPPLE Translator

A comprehensive toolchain for translating CUDA code to [RIPPLE](https://discourse.llvm.org/t/rfc-ripple-a-compiler-interpreted-api-to-support-spmd-and-loop-annotation-programming-for-simd-targets/88241) for Hexagon HVX and other SIMD targets.

## Overview

This translator enables porting existing CUDA codebases to non-GPU SIMD hardware, specifically targeting Qualcomm's Hexagon processor with HVX (Hexagon Vector eXtensions).

### Key Features

- **Dual Translation Paths**
  - **Source-level**: CUDA C → RIPPLE C (cleaner, preserves comments)
  - **IR-level**: CUDA LLVM IR → RIPPLE LLVM IR (for compiled code)

- **Multiple Interfaces**
  - Command-line interface (CLI)
  - Web-based editor with live preview
  - VS Code extension

- **Hexagon HVX Optimized**
  - 128-byte (1024-bit) vector operations
  - VTCM (Vector Tightly Coupled Memory) for shared memory
  - HVX-specific intrinsics mapping

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cuda2ripple.git
cd cuda2ripple

# Install dependencies
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

## Quick Start

### As a Python Library

```python
from cuda2ripple import translate

cuda_code = '''
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
'''

ripple_code = translate(cuda_code, target="hexagon")
print(ripple_code)
```

### Command Line

```bash
# Source-level translation
cuda2ripple source kernel.cu -o kernel.ripple.c

# IR-level translation
cuda2ripple ir kernel.ll -o kernel.ripple.ll

# Analyze code complexity
cuda2ripple analyze kernel.cu --json

# Batch translation
cuda2ripple batch *.cu -o output/

# Interactive mode
cuda2ripple interactive
```

### Web Interface

```bash
# Start the web server
python -m cuda2ripple.interfaces.web.server --port 5000

# Open http://localhost:5000 in your browser
```

### VS Code Extension

1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "CUDA to RIPPLE"
4. Install and reload
5. Open a `.cu` file and use Ctrl+Shift+R to translate

## Translation Mappings

| CUDA Construct | RIPPLE Equivalent |
|----------------|-------------------|
| `threadIdx.x` | `ripple_id(block, 0)` |
| `threadIdx.y` | `ripple_id(block, 1)` |
| `blockDim.x` | `ripple_get_size(block, 0)` |
| `blockIdx.x` | Outer loop variable `block_idx_x` |
| `__global__ void kernel()` | `void kernel_ripple(grid_dims, block_dims, ...)` |
| `__device__` | `static inline` |
| `__shared__ T arr[N]` | `__attribute__((section(".vtcm"))) T arr[N]` |
| `__syncthreads()` | Implicit (SIMD lanes are lockstep) |
| `atomicAdd(ptr, val)` | `ripple_atomic_add(ptr, val)` |
| `__shfl_down_sync(mask, val, delta)` | `ripple_shuffle(val, shuffle_fn)` |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CUDA to RIPPLE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │ CUDA Source  │      │   Semantic   │      │  RIPPLE C    │  │
│  │    (.cu)     │─────▶│     AIR      │─────▶│    Code      │  │
│  └──────────────┘      │              │      └──────────────┘  │
│                        │  (Abstract   │                        │
│  ┌──────────────┐      │ Intermediate │      ┌──────────────┐  │
│  │ CUDA LLVM IR │      │  Represent-  │      │ RIPPLE LLVM  │  │
│  │    (.ll)     │─────▶│   ation)     │─────▶│     IR       │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Interfaces: CLI | Web | VS Code                                │
└─────────────────────────────────────────────────────────────────┘
```

## Hexagon HVX Configuration

```python
from cuda2ripple import translate, HexagonConfig

# Custom Hexagon configuration
config = HexagonConfig(
    hvx_width=128,        # 128 bytes (1024 bits) vector width
    hvx_mode="v68",       # HVX instruction set version
    vtcm_size=256,        # VTCM size in KB
    use_vtcm_for_shared=True
)

# Translate with custom config
ripple_code = translate(cuda_code, target="hexagon", hvx_width=128)
```

### HVX Vector Lanes by Data Type

| Data Type | 128-byte HVX | 64-byte HVX |
|-----------|--------------|-------------|
| `int8_t`  | 128 lanes    | 64 lanes    |
| `int16_t` | 64 lanes     | 32 lanes    |
| `int32_t` | 32 lanes     | 16 lanes    |
| `float`   | 32 lanes     | 16 lanes    |
| `int64_t` | 16 lanes     | 8 lanes     |
| `double`  | 16 lanes     | 8 lanes     |

## Examples

### Vector Addition

**CUDA Input:**
```cuda
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

**RIPPLE Output:**
```c
#include <ripple.h>

void vectorAdd_ripple(
    int grid_dim_x, int grid_dim_y, int grid_dim_z,
    int block_dim_x, int block_dim_y, int block_dim_z,
    float *a, float *b, float *c, int n) 
{
    for (int block_idx_x = 0; block_idx_x < grid_dim_x; block_idx_x++) {
        ripple_block_t ripple_block = ripple_set_block_shape(HVX_PE, block_dim_x);
        
        int idx = ripple_id(ripple_block, 0) + block_idx_x * ripple_get_size(ripple_block, 0);
        if (idx < n) {
            c[idx] = a[idx] + b[idx];
        }
    }
}
```

### Parallel Reduction

**CUDA Input:**
```cuda
__global__ void reduceSum(float *input, float *output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = input[tid];
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(output, sdata[0]);
}
```

**RIPPLE Output:**
```c
#include <ripple.h>

void reduceSum_ripple(...) {
    __attribute__((section(".vtcm"))) __attribute__((aligned(128))) float sdata[256];
    
    ripple_block_t ripple_block = ripple_set_block_shape(HVX_PE, 256);
    int tid = ripple_id(ripple_block, 0);
    sdata[tid] = input[tid];
    // __syncthreads: implicit in SIMD model
    
    for (int s = ripple_get_size(ripple_block, 0) / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
    }
    if (tid == 0) ripple_atomic_add(output, sdata[0]);
}
```

## Project Structure

```
cuda2ripple/
├── __init__.py              # Main package entry point
├── core/
│   ├── semantic_model.py    # AIR definitions, CUDA/RIPPLE concepts
│   └── translation_rules.py # Pattern matching and transformation rules
├── frontends/
│   ├── source/
│   │   └── cuda_frontend.py # Source-level lexer/parser/transformer
│   └── ir/
│       └── ir_frontend.py   # IR-level parser/analyzer/transformer
├── interfaces/
│   ├── cli/
│   │   └── cuda2ripple.py   # Command-line interface
│   ├── web/
│   │   └── server.py        # Flask web server
│   └── vscode/
│       ├── package.json     # VS Code extension manifest
│       └── src/extension.ts # Extension source
├── tests/
│   ├── test_translation.py  # Test suite
│   └── examples/
│       └── cuda_kernels.cu  # Example CUDA kernels
└── docs/
    └── README.md            # This file
```

## Limitations and Caveats

1. **Shared Memory Semantics**: CUDA shared memory maps to Hexagon VTCM, but semantics differ. Manual review may be needed.

2. **Synchronization**: `__syncthreads()` is typically not needed in RIPPLE (SIMD lanes execute in lockstep), but complex patterns may need attention.

3. **Dynamic Parallelism**: Not directly supported. Kernels launching kernels need restructuring.

4. **Cooperative Groups**: Require manual translation.

5. **Texture/Surface Memory**: Not supported in current version.

6. **Grid-stride Loops**: Automatically restructured, but verify performance.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License - See LICENSE file for details.

## References

- [RIPPLE RFC](https://discourse.llvm.org/t/rfc-ripple-a-compiler-interpreted-api-to-support-spmd-and-loop-annotation-programming-for-simd-targets/88241)
- [RIPPLE GitHub](https://github.com/Syllo/llvm-project/tree/ripple)
- [Hexagon HVX Programmer's Guide](https://developer.qualcomm.com/software/hexagon-dsp-sdk)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
