# Rust front end for Ripple: review and design

Status: proposed implementation brief, revised after independent senior review;
not an implemented compiler. Rust-input product scope remains provisional.
Reviewed 2026-10-04 at project commit `1cb3ee5`; local Ripple documentation at
`ef517bf`. The existing modification to `CLAUDE.md` was left untouched.

## Scope and recommendation

Working assumption: accept **Rust kernel source**, with the new compiler components
also written in Rust. If the intended input remains CUDA, use the alternative at
the end of this document. These are different language front ends, not interchangeable
parser choices.

Build a compiler pipeline with explicit semantic stages, verified intermediate
representation, source diagnostics, and a separate target backend. Reuse rustc for
Rust language semantics. Begin by generating Ripple C for Qualcomm's compiler;
do not start by copying the Python regex rules or inventing LLVM intrinsics.

“Like LLVM” should mean modular compiler architecture here. LLVM supplies reusable
compiler infrastructure; Clang is a language front end. For Rust, rustc already
provides parsing, macro expansion, resolution, type checking, and borrow checking.
The new work is the checked Rust-to-Ripple lowering and kernel programming model.
From rustc's perspective, parts of this project are a backend; from Ripple's
perspective, they provide Rust language input.

## Findings from the repository

| Finding | Evidence | Consequence |
| --- | --- | --- |
| AST does not drive source generation | `frontends/source/cuda_frontend.py:486–537` explicitly runs an AST pre-pass, then regex rules, including fallback after parser exceptions | Replace semantic translation with typed lowering; a Rust port of the same rules would retain the design limitation |
| AIR is not a complete typed executable IR | `core/semantic_model.py:206–259` stores expressions, addresses, and values as strings; `_parse_block` at `cuda_frontend.py:900` collects textual statements | Build a new versioned IR rather than treating existing AIR as a ready compiler boundary |
| LLVM path is not a shared verified backend | `frontends/ir/ir_frontend.py:509–638` emits textual IR, hardcodes data layout, declares its own Ripple intrinsic names, and rewrites address-space strings | Do not route Rust LLVM output through it; validate any future direct LLVM path against the actual Ripple fork |
| Current kernel ABI handles one CUDA grid block | `core/translation_rules.py:1789–1806` supplies block indices and grid/block dimensions as arguments | Separate kernel execution from launch/grid scheduling and document both |
| Synchronization conversion lacks a proof step | `core/translation_rules.py:439–451` replaces a barrier with a comment | A new compiler must verify supported synchronization semantics or reject the program |
| Editor duplicates compiler logic | `interfaces/vscode/src/extension.ts:54` onward | Ultimately use one compiler service/CLI rather than independent TypeScript transformations |
| Compile verification is syntax-only | `tests/compile_verify.py:1–13,34–61`, `docker/README.md` | Add real Ripple compilation and runtime comparisons before claiming target correctness |
| Toolchain build is not reproducible from its Dockerfile alone | `docker/hexagon-toolchain.Dockerfile:74–75` clones moving upstream heads | Pin compiler, linker, SDK, runtime, documentation, and image versions in a separate reproducible build definition |

Baseline: `venv/bin/python -m pytest -q -o addopts=''` reports **155 passed**.
Six `PytestUnknownMarkWarning` warnings show timeout markers are unrecognized in
this environment; timeout enforcement was therefore not demonstrated. No actual
Hexagon compilation or execution was performed for this review.

The README's architecture and generated examples should not override observed
code behavior. In particular, its vector-add example includes a grid loop that
the current `GlobalKernelRule` does not generate.

## Proposed pipeline

```text
Rust kernel crate + selected kernel entries
    -> pinned rustc: expand / resolve / type-check / borrow-check
    -> reachable concrete function instances and typed MIR
    -> Rust adapter: supported-subset checks and explicit semantic lowering
    -> Kernel IR: typed values, control flow, memory and lane operations
    -> verifier / uniformity analysis / target capability checks
    -> Ripple C backend + ABI header + kernel metadata
    -> pinned Ripple-enabled Clang (-fenable-ripple, validated target flags)
    -> object + explicit runtime/launch integration
```

Use `rustc_driver` callbacks, with a narrow adapter around compiler internals.
The Rust compiler guide recommends that driver where possible and explicitly
warns that internal APIs are unstable. Pin an exact tested nightly and its
`rustc-dev`/LLVM components; keep those dependencies out of the IR and backend
libraries. Do not parse `--emit=mir` text as a stable interchange format.
[Driver documentation](https://rustc-dev-guide.rust-lang.org/rustc-driver/intro.html),
[rustc_private requirements](https://doc.rust-lang.org/unstable-book/language-features/rustc-private.html).

The extraction milestone must prove that borrow checking runs for every selected
kernel and reachable body. A callback name alone is not evidence of this. Resolve
concrete callee instances and substitutions; MIR bodies can still contain generic
parameters. Follow rustc's instance/monomorphization model rather than assuming an
arbitrary MIR query yields fully monomorphized code.
[Monomorphization](https://rustc-dev-guide.rust-lang.org/backend/monomorph.html).

The compatibility artifact must name the exact MIR query/dialect, query order,
enabled passes, optimization settings, marker encoding, and local/dependency body
handling. rustc has distinct MIR stages, and later queries can consume earlier
bodies; an earlier MIR query is not a drop-in substitute for optimized MIR.
Demonstrate marker preservation directly and through helpers at every supported
optimization configuration. Ordinary compilation must reject kernel markers rather
than execute placeholder implementations. These are feasibility requirements,
not a claim that a particular callback/query combination has already been proven.
[MIR queries and passes](https://rustc-dev-guide.rust-lang.org/mir/passes.html).

The host compiler executable and kernel target are separate. Kernel type checking,
constant evaluation, `cfg`, layout, and pointer widths must use the selected target
configuration. Never evaluate a kernel as a 64-bit macOS program and later emit
32-bit Hexagon code without reconciling its semantics. Establish a compatible
target specification and target `core`/sysroot as an early feasibility gate.

## Initial language and execution contract

Support a named, documented kernel subset rather than promising arbitrary Rust.

- Start with one crate, explicitly selected non-generic kernel entries, fixed-width
  integers, `f32`, booleans, local variables, assignments, branches, bounded test
  loops, and statically resolved helper calls. Lower only verified reachable code.
- Use an explicit unsafe kernel ABI with raw buffer pointers, lengths, scalar
  parameters, and a launch descriptor. Document validity, alignment, lifetimes,
  non-overlap where required, and exclusive writes. A safe host wrapper is a later
  deliverable that must enforce its contract.
- Expose a generated C entry with a C-compatible calling convention and a versioned
  launch descriptor; it is not Rust's native calling convention. Specify parameter
  types, layout, ownership, and descriptor-version rejection. Prove the boundary
  by calling the generated entry from a target C driver. Buffer pointers are valid
  in the target address space; host-to-DSP buffer transport is a separate runtime
  concern, not a raw-pointer cast.
- Provide a small `no_std` kernel API for lane identity and block dimensions.
  Recognize compiler-known operations by resolved definition identity, not their
  spelling. Prove that marker calls survive until extraction; do not ship marker
  stubs that silently execute as ordinary functions on the device.
- Begin with one-dimensional, fixed, validated SIMD block shapes and one block per invocation.
  Metadata must identify the shape, supported target, ABI, and buffer requirements.
  Grid scheduling is a distinct runtime contract. The first demo can use an explicit
  test driver; it must not imply a production DSP launcher exists.
- Initially reject heap allocation, `std`, unwinding, async, dynamic dispatch,
  recursion, unsupported external calls, inline assembly, atomics, and shared-memory
  collectives. Add arrays, aggregates, generics, and reductions in individually
  tested milestones. Reject reachable unsupported operations even if parsing succeeds.
- Rust syntax that rustc can process does not automatically belong to the supported
  kernel subset. For example, an accepted macro may expand to an unsupported operation.

Define a kernel as one logical Rust invocation per lane, with lane-local locals
and control flow. Uniform inputs can share computation only when that preserves
observable behavior. This contract is not Ripple's implicit shape semantics:
Ripple can collapse a varying predicate for a scalar effect. The first executable
milestone must track scalar/lane-shaped values and active predicates, implement
lane-correct assignments and joins, and reject effects whose multiplicity or mask
cannot be preserved. Initially reject scalar memory effects under varying control
unless a single participating lane is proven. Multidimensional support is deferred
until the IR tracks dimensional shapes and broadcast rules explicitly.
See local `temp_ripple_docs/src/ripple-spec/spmd.md`, especially “How shape affects masking.”

Supporting a non-generic user kernel still requires resolving concrete instances
of generic `core` pointer operations. Enumerate the exact `core`/intrinsic closure
used by the first buffer fixture and implement verified body import or identity-based
intrinsic lowering for each operation. Defer general user generics, not this basic
instance handling; reject unavailable dependency bodies explicitly.

Rust checking covers ordinary Rust semantics, not the hidden SPMD execution model.
Borrow checking alone cannot prove that different lanes write different addresses.
Before offering safe shared-reference APIs, establish and test the additional
cross-lane memory rules. For the initial raw-pointer API, make that obligation
explicit at the unsafe boundary.

## Semantic requirements that must survive C emission

Rust arithmetic cannot be translated by printing superficially similar C.
Preserve MIR's checked/unchecked/wrapping distinctions, integer casts, shift rules,
division edge cases, bounds checks, and evaluation order. Signed C overflow must
not replace defined Rust behavior. Introduce temporaries or helper operations
where required. Preserve aliasing and alignment requirements; do not infer
`restrict` from arbitrary raw pointers. Rust's operator semantics are specified
in the [Rust Reference](https://doc.rust-lang.org/reference/expressions/operator-expr.html).

For the first milestone, reject any reachable panic/assert path the compiler
cannot prove unnecessary, including residual bounds and overflow assertions.
Do not equate `panic=abort` with permission to erase checks. A later target trap
implementation needs its own runtime contract and tests. Likewise, reject
unsupported drop glue, unwinding, or layout-sensitive constructs rather than
silently omitting them. Preserve floating-point behavior by default; no implicit
fast-math, reassociation, or contraction policy changes.

Inactive lanes require a separate legality check. The local Ripple specification
says it masks loads, stores, and reductions; do not assume a C branch prevents
all intermediate arithmetic or pointer formation on inactive lanes. For each
potentially invalid operation, prove inactive evaluation harmless, sanitize operands
before evaluation while preserving active-lane behavior, use a target-tested safe
lowering, or reject it. A conditional select of an already-invalid result is not
sanitization. Test guarded division, invalid shifts, tail pointer calculations,
and use of masked results after CFG joins with the pinned compiler/runtime.
Rust pointer `add` has requirements at pointer formation as well as dereference;
masking a later load alone does not discharge those requirements.
[Pointer arithmetic contract](https://doc.rust-lang.org/core/primitive.pointer.html#method.add).

## Kernel IR and crate boundaries

Suggested workspace, under a new `compiler/` directory:

| Crate | Responsibility |
| --- | --- |
| `ripple-diagnostics` | File/source identities, spans, diagnostic codes, labels, notes, JSON output |
| `ripple-ir` | Owned typed IR, serialization/versioning, printer, verifier; no rustc types |
| `ripple-rustc` | Pinned rustc integration, entry selection, reachable-instance collection, MIR import |
| `ripple-analysis` | Uniformity, memory effects, convergence constraints, target legality |
| `ripple-codegen-c` | Structured C emission, ABI headers, source mapping, capability checks |
| `ripple-driver` | CLI, configuration, compiler invocation, artifact publication |
| `ripple-kernel` | Restricted device API and documented unsafe contracts |

Keep the compiler-internals adapter behind a process boundary where practical so
ordinary consumers do not inherit rustc's unstable linkage requirements.

IR must represent typed constants and operations, locals/places or SSA values,
basic blocks and terminators, calls with resolved signatures, pointer/address-space
information, loads/stores, block/lane identity, scalar/lane shape, active predicates,
definedness at joins, and source spans. Choose a simple
typed CFG initially; full SSA optimization infrastructure is not required to ship
the first correct kernel. Preserve enough structure or reconstruct it during C
emission to satisfy Ripple's control-flow requirements; arbitrary goto emission
must be tested with the real compiler rather than assumed acceptable.

The verifier checks references, operand/result types, CFG termination, call
signatures, layout/ABI consistency, and legal memory/collective operations. If SSA
is used, also check dominance and block argument consistency. Run it at each pass
boundary in development and before backend emission in release builds.

Track uniform versus lane-varying values and active-lane/convergence constraints.
The minimum one-dimensional analysis is required for the first kernel, not deferred
until collectives. Treat unknown information conservatively. Never remove a barrier solely because
SIMD instructions execute in lockstep. Reductions and shuffles require a defined
participation set, shape, type, and ordering contract before lowering.

## Ripple backend and version policy

Local `temp_ripple_docs/src/ripple-spec/api.md` documents
`ripple_set_block_shape`, `ripple_id`, and a function-based `ripple_shuffle`.
Its `calling.md` describes vectorized function-call behavior. Use the actual pinned
headers and compiler probes to resolve signatures and compiler restrictions;
documentation pseudotypes are not an FFI ABI.

The local release notes describe **21.0-alpha3**, with `-O2` as the most-tested
optimization level. The same checkout contains threading documentation, while
project notes say its runtime was not released at the time of those notes. These
are historical local facts, not a claim about the latest October release. Before
implementation, select and probe the exact supported release from
[Qualcomm's Ripple repository](https://github.com/qualcomm/ripple) and
[official learning resources](https://github.com/qualcomm/learn-ripple).

Keep a machine-readable capability manifest tied to compiler commit, SDK/runtime,
target triple, CPU, HVX width, headers, flags, and documentation commit. Baseline
features must pass compile-and-run probes. Threading, atomics, and synchronization
must remain disabled unless that specific toolchain's implementation and semantics
are verified. Local threading prose mentions atomics, but that is not an atomics
API; a keyword search alone is not capability validation.

First emit `kernel.ripple.c`, an ABI header, metadata, and a source map. Publish
outputs only after successful requested-stage validation. Object mode must fail
if the compiler is missing, not report a C file as a compiled object.

Direct LLVM emission is a later backend, contingent on documenting the Ripple
fork's real intrinsics/metadata, pass order, convergence rules, LLVM version,
data layout, and runtime ABI. Ordinary Rust LLVM output is not automatically
Ripple-aware, and LLVM verification alone cannot prove Ripple semantics.

## Product quality and migration

Provide `check`, `compile`, and `explain` commands; explicit target profiles;
`--emit=ir,c,obj`; human and JSON diagnostics; and deterministic outputs. These
are proposed interfaces, not existing commands. Include original Rust spans in
errors and preserve the underlying C compiler diagnostic when mapping is incomplete.
Return nonzero status for unsupported semantics; never fall back to regex output.

Keep Python/CUDA behavior available during development. Introduce the Rust mode as
an explicit option and test CLI/web/editor adapters against the same compiler
protocol. Do not replace CUDA mode with Rust-source compilation: they serve different
input languages. Consolidate editor translation only when that compiler supports
the corresponding mode. Keep the paired README files identical when updating them.

Release requires a support matrix, reproducible toolchain setup, licensed dependency
inventory, diagnostic reference, examples, CI, fuzzing of import/IR boundaries,
and semantic execution tests. Compilation failures should produce actionable
diagnostics; malformed input should not panic the driver or leave successful-looking
partial artifacts.

## Alternative: a CUDA front end implemented in Rust

If CUDA is still the source language, this Rust-input implementation plan is not
the selected product plan. Write a CUDA-specific semantic and migration plan using
a Clang adapter, including CUDA memory spaces, execution and synchronization rules.
Shared compiler infrastructure is reusable; language-semantic work is not merely
an adapter substitution. Reuse Clang parsing, preprocessing, overload resolution, templates, and
types. Probe libclang's coverage of the required CUDA AST constructs first. If its
C API is insufficient, use a small pinned C++ LibTooling helper exporting a typed,
versioned protocol to the Rust pipeline. This entails a C++ dependency; a strict
all-Rust parser requirement would instead require a much larger C++/CUDA language
implementation and must be scoped explicitly.
[Clang interface guidance](https://clang.llvm.org/docs/Tooling.html).

The Rust-owned IR, diagnostics, legality passes, backend, tests, and driver remain
applicable. Retain CUDA source locations and compile commands; never use an AST
dump or token regex as a substitute for resolved semantic information. Port the
existing fixture corpus as behavioral cases, with reviewed expected semantics,
then migrate users incrementally. Preserve existing Python entry points through
an adapter and remove the TypeScript duplicate only after compatibility is proven.
