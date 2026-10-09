# Preserve CUDA block shape and generate grid launchers

The source translator must preserve logical CUDA thread coordinates independently
of hardware vector width. Never derive a CUDA launch shape from element types,
index expressions, grid dimensionality, or shared-memory declarations.

## Output contract

Each translated kernel takes a caller-provided `ripple_block_t`, a
`ripple_launch_context_t` (block index and grid dimensions), and its original
arguments. The kernel queries the supplied SIMD object for threadIdx and blockDim.
A generated `<kernel>_ripple_launch` function iterates grid z/y/x on the target,
calling the translated per-block kernel once for each logical CUDA block.

The library, CLI, and HTTP APIs accept an optional block shape: one to three
positive integer extents, padded to three dimensions. A mapping from kernel names
to shapes supports multiple kernels in one translation unit. When provided, the
launcher constructs the exact static shape. When omitted, the launcher takes the
SIMD block as another argument and emits a diagnostic explaining that the caller
must construct it with the original launch shape. This permits existing
kernel-only input without guessing or dropping work. No template block indices
or runtime-shaped SIMD objects are generated.

Caller and callee must be compiled in the same compilation unit when passing a
SIMD block (upstream ripple-spec/calling.md). Grid dimensions remain dynamic.
The sequential loops are the first implementation; runtime thread scheduling is
deferred and no ripple_thd API is required.

The regex source frontend does not translate host-side <<<...>>> launches.
Reject them with an actionable diagnostic rather than leave untranslated launch
syntax in otherwise advertised generated code. Dynamic/nonpositive SIMD extents
and unknown kernel names in shape mappings also fail with diagnostics.

## Validation

Regression tests compile generated C with a scalar lane harness, execute all
SIMD coordinates, and check 64-wide vecadd coverage and multidimensional
coordinates across nontrivial grids. This validates logical coverage and grid
iteration, not native Ripple vectorization. Also validate omitted-shape warnings,
invalid shapes, independent kernel shapes, CLI and HTTP inputs, and existing
syntax checks. Native Hexagon execution remains an external validation step.

## Scope

Python source frontend and its library, CLI, and HTTP entry points. The separate
VS Code TypeScript translator and LLVM-IR path remain outside this change.
