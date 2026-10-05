# Rust front end: implementation instructions

Implement the companion [design](../specs/2026-10-04-rust-frontend-design.md).
Working assumption: Rust source kernels, compiler implemented in Rust. This remains
a provisional product choice. For CUDA input, develop the design's Clang track into
a CUDA-specific semantic/migration plan before beginning language implementation;
do not treat language support as just swapping adapters.
This plan does not authorize claiming support for either language beyond tested
capabilities. All commands/interfaces below describing the new compiler are proposed.

## 1. Establish the compatibility contract before scaffolding broadly

Read `CLAUDE.md`, the actual source/IR pipelines, and the upstream Ripple docs.
Record repository status and preserve existing changes. Capture baseline tests.
Create a toolchain manifest recording exact rustc nightly, rustc components,
Ripple compiler/LLVM commit, SDK/linker/runtime, target, sysroot, CPU, vector width,
compiler flags, and documentation revision. Also record the exact MIR query/dialect,
query ordering, optimization/pass configuration, marker encoding, and treatment of
local/dependency bodies. Verify selected artifacts are available.
Use a separate pinned build recipe; do not hand-edit the vendored upstream Dockerfile.

Specify source language, kernel ABI, panic policy, initial block shape, and supported
types. Use the design's per-logical-lane semantics and initially one-dimensional
shapes. Specify a generated C entry, parameter types/calling convention, and a
versioned launch descriptor, including target-address-space buffer ownership.
Do not choose a nightly by date alone: prove driver linkage and target `core`
compatibility. Stop dependent feature work on a failed feasibility gate and report
the concrete incompatibility rather than filling it with mocks.

Acceptance: a small checked Rust kernel's target layout is known; a hand-written
equivalent Ripple C kernel compiles and executes on an available simulator/device.
The probe includes a target C caller exercising the proposed ABI, descriptor, and
buffers; the generated version must pass the same call-boundary test in milestone 2.
If execution infrastructure is unavailable, record this as incomplete validation,
not a passed milestone. A scalar-only probe is insufficient to validate SPMD.

## 2. Prove compiler extraction with one executable slice

Create `compiler/Cargo.toml`, an exact `rust-toolchain.toml`, and a committed lockfile.
Initially implement the minimal driver, rustc adapter, IR, diagnostics, and C backend.
Keep crate boundaries from the design, but avoid empty frameworks or speculative
plugin systems. Document the install/build process and target sysroot construction.

Select a kernel explicitly through CLI configuration initially; annotation macros
can follow when their preservation and definition identity are tested. Import its
reachable typed MIR into owned IR, then emit C. Run rustc checking for the actual
kernel target before emission. The first slice should copy a buffer or add two
buffers using raw pointers, lengths, a fixed lane shape, and a tail predicate.

Enumerate and support that fixture's concrete `core` pointer/intrinsic dependencies
now, by verified MIR import or identity-based lowering. General user generics can
wait. Prove marker survival both directly and through helper calls with all supported
optimization settings, and reject normal compilation that would execute markers.

Implement minimum scalar/lane shape, active-predicate, memory-effect, and join
legality analysis for this slice. Preserve per-lane locals and reject scalar effects
under varying control unless their semantics are proven. For inactive-lane arithmetic
and pointer formation, implement the design's proof/sanitization/safe-lowering-or-reject
policy before emitting the first tail-predicated kernel. A C `if` alone is not proof.

Acceptance: ordinary Rust type and borrow errors are diagnosed; known kernel API
calls are recognized by identity; generated C compiles with actual Ripple Clang;
target results match a scalar reference for empty, partial, exact, and multiple-block
input lengths using an explicit test launch driver. Block iteration in that driver
must be visible and tested. A target C driver invokes the generated C entry through
its published ABI, including descriptor-version failure tests. Add divergent local
assignments/joins, masked memory, guarded division/shift and tail-pointer cases;
unsupported cases must diagnose rather than emit unsafe code. Test unavailable
dependency-body rejection. No fake intrinsic declarations or regex fallback.

## 3. Make the IR and semantic rejection dependable

Implement typed CFG operations, source spans, deterministic dumping, schema versions,
and verification. Specify the input/output invariants of every lowering pass.
Preserve MIR assertions or reject them explicitly. Reject unsupported calls,
unwinding, drops, and memory operations transitively through reachable helper bodies.

Add negative tests for invalid IR types and references, unresolved callees, malformed
control flow, and unsupported operations with exact error spans/codes. Add semantic
tests for wrapping arithmetic, casts, evaluation order, shift/division edge cases,
and target-width constants. Tests must exercise emitted code as well as IR snapshots.

Acceptance: every supported operation has executable semantics tests; every deferred
operation has a diagnostic test; source errors cannot yield a successful artifact.

## 4. Extend the supported subset in small vertical changes

Expand helper calls and structured loops beyond the minimal extraction fixtures,
then add fixed arrays, general user generic instances, and selected aggregates.
For each addition, implement resolution/import,
IR representation, verification, C lowering, diagnostics, and real-toolchain tests
together. Do not mark a feature complete after parsing it.

Define ABI sizes/alignment and field offsets using the target layout. Initially
reject unsupported representations rather than attempting unsafe layout guesses.
For dependency calls, either obtain and lower an allowed body or reject with a
clear message; do not silently emit an unresolved host Rust symbol.

Acceptance: each feature's support-matrix row names positive, negative, and runtime
tests. Golden C output is useful for review but is not a semantic oracle.

## 5. Extend SPMD analysis for collectives

Extend the mandatory milestone-2 shape/predication analysis with collective lane
participation, memory effects, and convergence/target legality checks. Multidimensional
support requires dimensional shapes and explicit broadcast semantics. Start with a documented, fixed-shape reduction or
shuffle only after its capabilities are probed. Preserve masks and participation
semantics; reject cases the target cannot express.

Test partial lanes, divergent branches, helper calls, non-native block sizes when
supported, and integer/floating-point distinctions. Establish floating-point
tolerances from the operation contract; do not loosen tolerances to hide failures.
Shared memory, atomics, threading, and grid-runtime generation require separate
designs and target evidence. Never copy unconditional barrier erasure.

Acceptance: both legal and illegal convergence cases have tests; actual target
execution validates supported collectives; unsupported cases fail predictably.

## 6. Package one compiler interface

Implement `check`, `compile`, `explain`, target profiles, and requested-stage output.
Define a versioned request/response envelope for Python/web/editor adapters with
source files, entries, profile, compiler version, diagnostics, and artifact paths.
Use argument arrays for compiler invocation, isolated temporary files, bounded
execution, and atomic output replacement. Do not splice user inputs into a shell.

Expose Rust mode explicitly in existing interfaces. Keep CUDA behavior until its
own adapter is ready. Check that no interface claims success after compiler errors,
and that downstream errors preserve original source locations where available.

Acceptance: CLI and adapter integration tests yield the same result and diagnostic
codes for a fixture; old CUDA API tests still pass; no duplicate Rust compiler logic
exists in Python or TypeScript.

## 7. Establish release gates

Fast CI: formatting, clippy, unit/IR verifier tests, diagnostic tests, deterministic
output checks, regression tests, and the existing Python suite. Register and install
the timeout plugin so timeout tests are actually enforced.

Toolchain CI: build generated output with the pinned real Ripple compiler and
headers, using its validated target flags and optimization level. Assert required
SPMD operations/codegen survive; generic Clang plus stub headers is a separate
syntax check, not a substitute.

Execution CI: compare target/simulator results with independently written reference
kernels; include tail handling, multiple blocks, memory canaries, arithmetic edge
cases, and supported collective behavior. Optional CUDA differential tests may
compare defined supported inputs, but old translator output is not ground truth.

Robustness: fuzz IR/protocol decoding and lowering boundaries, test malformed source,
compiler timeouts, missing SDKs, and unsupported target capabilities. Bound compile
time and memory on a versioned corpus. Benchmark against equivalent hand-written
Ripple kernels before setting performance budgets; do not invent speedup claims.

Acceptance: a clean machine can reproduce documented artifacts; all mandatory
target tests execute, with skips preventing release qualification; release notes
state exactly which Rust subset, target, runtime, and toolchain are supported.

## Instruction to an implementing engineer or coding agent

Build the Rust-to-Ripple pipeline described in the companion design. Begin with
milestones 1 and 2 and deliver a real end-to-end buffer kernel before expanding
the language. Reuse rustc language checking, isolate unstable internals, introduce
a verified typed IR, and emit Ripple C through the actual pinned Ripple compiler.
Preserve Rust semantics and the documented unsafe SPMD memory contract. Reject
unsupported semantics with source diagnostics. Keep existing CUDA entry points
working. For each milestone, report files changed, verified behavior, exact test
commands/results, and remaining limitations. Do not count stubs, parser-only
support, syntax checks, or unexecuted target tests as compiler completion.
