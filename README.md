# CUDA to RIPPLE Translator

A comprehensive toolchain for translating CUDA code to [RIPPLE](https://discourse.llvm.org/t/rfc-ripple-a-compiler-interpreted-api-to-support-spmd-and-loop-annotation-programming-for-simd-targets/88241) for Hexagon HVX and other SIMD targets.

The independent Rust frontend now lives in [rust-to-ripple](https://github.com/aseesy/rust-to-ripple).
This repository contains the CUDA translator and its Ripple threading notes.

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

ripple_code = translate(cuda_code, target="hexagon", block_shape=(64, 1, 1))
print(ripple_code)
```

### Command Line

```bash
# Source-level translation
cuda2ripple source kernel.cu --block-shape 64,1,1 -o kernel.ripple.c

# IR-level translation
cuda2ripple ir kernel.ll -o kernel.ripple.ll

# Analyze code complexity
cuda2ripple analyze kernel.cu --json

# Batch translation
cuda2ripple batch *.cu -o output/

# Interactive mode
cuda2ripple interactive
```

> **Compiling the output:** Ripple support isn't enabled by default in a
> Ripple-capable clang build — compile translated output with
> `clang -fenable-ripple ...` (see the Ripple Troubleshooting Guide's
> "Missing ripple\_\* symbols" section). Without this flag, translated
> code fails to compile with undefined-symbol errors even though it's
> valid RIPPLE C.

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
| `blockDim.x` | `ripple_get_block_size(block, 0)` |
| `blockIdx.x` | Outer loop variable `block_idx_x` |
| `__global__ void kernel()` | Per-block `kernel_ripple(SIMD_block, context, ...)` plus `kernel_ripple_launch(...)` |
| `__device__` | `static inline` |
| `__shared__ T arr[N]` | `T *arr = vtcm_malloc(sizeof(T) * N, /*align_as=*/128); ... vtcm_free(arr);` (multi-dimensional arrays are flattened to one allocation, with every indexing site rewritten to flat row-major indexing) |
| `__syncthreads()` | Implicit (SIMD lanes are lockstep) |
| `atomicAdd(ptr, val)` | *(no equivalent — Ripple has no atomics API and no documented alternative for this pattern)* |
| `__shfl_down_sync(mask, val, delta)` | `ripple_shuffle(val, shuffle_fn)` |

## Preserving launch shape

The source translator preserves logical thread coordinates independently of HVX
vector width. Supply the original CUDA launch's static block shape:

```python
translate(cuda_code, block_shape=(64,))  # exactly (64, 1, 1)
translate(cuda_code, block_shape={"vecadd": (64,), "matmul": (16, 16)})
```

```bash
cuda2ripple source kernels.cu --block-shape vecadd=64 --block-shape matmul=16,16 -o kernels.c
```

Both HTTP translation endpoints accept a `block_shape` JSON array or a mapping
from kernel names to arrays. The Python web editor also has an optional CUDA
Block field. Shapes contain one to three positive integer extents; omitted axes
are one. Dynamic SIMD extents are rejected. Kernel-only source cannot establish
the original launch shape, even when indexing contains a literal such as 64.

Each generated `kernel_ripple` takes a `ripple_block_t` and a translator-provided
`ripple_launch_context_t` instead of nine scalar execution arguments.
`threadIdx.*` and `blockDim.*` query the supplied SIMD block.
The generated `kernel_ripple_launch` iterates every grid block in three nested
target-side loops; grid dimensions remain dynamic. For a known shape:

```c
vectorAdd_ripple_launch((ripple_dim3_t){grid_x, grid_y, grid_z}, a, b, c, n);
```

If shape information is omitted, the translator emits a warning and generates a
launcher requiring a caller-created block. It never guesses a hardware-sized shape:

```c
ripple_block_t BS = ripple_set_block_shape(0, 64, 1, 1);
vectorAdd_ripple_launch(BS, (ripple_dim3_t){grid_x, grid_y, grid_z}, a, b, c, n);
```

Keep the caller and generated functions in the same compilation unit when passing
a SIMD block. Existing callers using the old nine-argument function signature
must migrate to the generated launcher (or supply the new context to a per-block
call). Host-side CUDA `<<<...>>>` launches are rejected with instructions to use
the launcher; they are not translated automatically.

The default launcher executes grid loops sequentially. For runtime workers,
use `threaded=True` in Python/HTTP or `--threaded` in the CLI:

```python
output = translate(source, block_shape=(64, 1, 1), threaded=True)
```

This also generates `vectorAdd_ripple_launch_workers`, accepting a
`ripple_thd_block_t` before the grid and kernel arguments. Every active worker
in an already initialized one-dimensional runtime block of shape `(worker_count)` must call
it once with the same grid and arguments. For a known SIMD shape:

```c
int status = vectorAdd_ripple_launch_workers(
    worker_block, (ripple_dim3_t){grid_x, grid_y, grid_z}, a, b, c, n);
```

Workers receive disjoint grid blocks in round-robin order across all three grid
axes. The worker count can change at runtime; each CUDA block keeps its original
static SIMD shape. If the SIMD shape was omitted at translation, pass its
caller-created `ripple_block_t` as the first argument.

The surrounding QuRT/QHPI runtime owns initialization, worker creation and
completion/join. The translator includes `<ripple/thread.h>` and uses its
ID/size queries; it does not supply a vendor thread runtime. The launcher returns
`CUDA2RIPPLE_LAUNCH_OK`, `CUDA2RIPPLE_BAD_WORKERS` (null block, invalid ID or zero worker count),
or `CUDA2RIPPLE_BAD_GRID` (grid-volume overflow). Empty grids are successful no-ops.
The caller must create a one-dimensional worker block; the opaque API has no
rank query. The launcher queries only dimension zero and does not assume that
uncreated dimensions exist.

These contracts apply to the Python source frontend; the independent VS Code
TypeScript translator and LLVM-IR path have not been migrated.

### Verification

The test suite compiles generated launchers with a scalar SIMD reference and
runs real pthread workers. It checks exact results, once-only grid coverage,
uneven workloads, excess workers, guard regions and invalid inputs.

`scripts/verify_native_launch.py` additionally compiles unmodified generated
code with Ripple clang and runs it in the Hexagon simulator. It retains commands,
compiler/source/binary hashes and logs in `output/verification/`. See
[toolchain verification](docker/README.md) for prerequisites and commands.

For **v68 IEEE floating-point arithmetic**, compile with both flags:

```bash
-mno-hvx-qfloat -mhvx-ieee-fp
```

The native verifier selects these flags by default. Default QFloat differs from
the exact reference, and disabling QFloat alone caused inactive float lanes to
be written in the installed build. Enabling HVX IEEE arithmetic resolves both
failures in the probes. Modern QFloat `strict-ieee` lowering applies to v79+;
accepting that option does not establish IEEE behavior on v68
([upstream pass selection](https://github.com/qualcomm/ripple/blob/3468adcb686511de236e0f991a8bc0af28730a0a/llvm/lib/Target/Hexagon/HexagonTargetMachine.cpp#L429)).

Native probes cover float/integer tails, 2D shapes and worker assignment.
The suite also creates actual SDK standalone hardware workers (1/2/3/4/6 when
available) over a 7x3x2 grid. Worker IDs come from hardware registers, and
simulator instruction counts must show every secondary worker executing.
These tests use unmodified generated SIMD kernels and check additive output,
once-only work, tail guards, worker rendezvous and join.
A separate float probe compares with a non-vectorized scalar reference for
rounding, subnormals, signed zeros, infinities and NaNs. Reports retain the failed
diagnostic modes as well as the corrected-mode results. This validates these
probes on the recorded compiler/simulator, not every possible kernel.

The standalone concurrency driver configures global HVX vector mode before
spawning workers. Each worker acquires/releases its HVX context around the
kernel. The driver also protects HVX allocation/release with a handoff lock:
the installed SDK helper publishes a free context before resetting the old
worker's hardware permission. Nonblocking acquisition lets waiting workers
release the lock so owners can free their contexts. Kernel bodies run outside
this lock. Six-worker tests retain the cases that previously corrupted guards
or missed active integer output. See
[the executable integration example](tests/native/standalone_workers.c).

Native QuRT/QHPI integration remains unverified: auditing all 19,039 entries in
the official SDK 6.5.0.0 archive found no Ripple thread runtime headers or
libraries. The compiler's public runtime source supplies vector helpers rather
than these thread libraries. Native vendor concurrency needs a separate runtime
installation. Reports distinguish host pthreads, sequential reference SPMD
queries, actual SDK hardware threads, and QuRT/QHPI integration.

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

**RIPPLE Output (excerpt, translated with `block_shape=(64, 1, 1)`):**
```c
void vectorAdd_ripple(
    ripple_block_t ripple_block, ripple_launch_context_t ripple_ctx,
    float *a, float *b, float *c, int n)
{
    size_t block_idx_x = ripple_ctx.block_idx.x;
    // Other block-index and grid-dimension aliases omitted from this excerpt.
    int idx = ripple_id(ripple_block, 0)
            + block_idx_x * ripple_get_block_size(ripple_block, 0);
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void vectorAdd_ripple_launch(
    ripple_dim3_t ripple_grid, float *a, float *b, float *c, int n)
{
    ripple_block_t ripple_block = ripple_set_block_shape(HVX_PE, 64, 1, 1);
    ripple_launch_context_t ripple_ctx = {{0, 0, 0}, ripple_grid};
    for (ripple_ctx.block_idx.z = 0; ripple_ctx.block_idx.z < ripple_grid.z; ++ripple_ctx.block_idx.z) {
        for (ripple_ctx.block_idx.y = 0; ripple_ctx.block_idx.y < ripple_grid.y; ++ripple_ctx.block_idx.y) {
            for (ripple_ctx.block_idx.x = 0; ripple_ctx.block_idx.x < ripple_grid.x; ++ripple_ctx.block_idx.x) {
                vectorAdd_ripple(ripple_block, ripple_ctx, a, b, c, n);
            }
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
    if (tid == 0) output[0] = sdata[0];
}
```

**RIPPLE Output (per-block excerpt; supply a 256-thread shape at launch):**
```c
#include <ripple.h>

void reduceSum_ripple(ripple_block_t ripple_block, ripple_launch_context_t ripple_ctx,
                      float *input, float *output, int n) {
    float *sdata = vtcm_malloc(sizeof(float) * (256), /*align_as=*/128);

    int tid = ripple_id(ripple_block, 0);
    sdata[tid] = input[tid];
    // __syncthreads: implicit in SIMD model

    for (int s = ripple_get_block_size(ripple_block, 0) / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
    }
    if (tid == 0) output[0] = sdata[0];
    vtcm_free(sdata);
}
```

Note this example writes the block's single reduced value directly rather than using
`atomicAdd` — Ripple has no atomics API, and accumulating across *multiple* blocks into
one shared output has no documented Ripple equivalent; see the Translation Mappings
table above.

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
