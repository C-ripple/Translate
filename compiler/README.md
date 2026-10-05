# Rust-to-Ripple compiler prototype

This implements the first Rust buffer-transfer slice from the reviewed plan.
Rust extraction, owned IR verification, restricted SPMD lowering, and Ripple C
generation are implemented. **All six real target cases pass on the Hexagon v68
simulator**, covering the handwritten reference and five Rust-generated variants.
This is a restricted compiler prototype,
not a production Rust compiler or launch runtime.

The real rustc probe checks the fixture for 32-bit Hexagon v68, explicitly runs
local borrow checking, reads typed MIR, resolves concrete direct-call instances,
and publishes an inventory plus typed owned IR. It does not parse textual MIR. The two buffer-copy
fixtures preserve a lane marker directly and through a helper; their concrete
`*const f32::add` and `*mut f32::add` dependency bodies are available in the tested
release configuration.

The inventory counts assertions and drops but does not traverse implicit panic/drop
paths or intrinsic bodies. The separate importer rejects unsupported types,
operations, assertions, and drops instead of silently approximating them.
A successful import alone is not a claim of safe SPMD lowering. There is no executable implementation of the
lane marker, and ordinary compilation is rejected.

`ripple-ir` has no rustc dependency. Its verifier checks target widths, assignment
types, call signatures, acyclic control flow, definite initialization, and
nonrecursive calls. Unsigned arithmetic wraps modulo 2^32; checked/unchecked MIR
operations are rejected, not collapsed into wrapping operations.

`ripple-codegen` symbolically inlines this IR with bounded path and expression
expansion. Entry zero must have signature `fn(*const f32, *mut f32, u32) -> ()`.
The fixtures declare it unsafe because buffer validity is a caller obligation.
Reads must be `src[lane]`, writes `dst[lane]`, and pointer offsets and accesses
must be dominated by `lane < n`. Initial support is f32 transfers, without float
arithmetic, loops, reductions, atomics, references, allocation, scalar memory
effects, general pointer arithmetic, or aliasing. No source-text patterns or
callee names identify pointer operations; concrete dependency MIR is imported.

The ABI launches one 32-lane block. Buffers cover `length` initialized f32 elements,
are 128-byte-aligned and disjoint, and remain valid with exclusive output writes.
Each call rebases pointers by `block_offset` and passes **remaining length**
(`length - block_offset`) as the Rust entry's `n`, with local lane IDs 0–31.
The caller schedules all blocks and handles host/DSP transport. Generated partial
blocks use a scalar loop over valid lanes; full blocks use Ripple with distinct
lane addresses. Empty launches do not use the buffers.

## Reproduce the Rust checks

Run from this directory. The exact toolchain is in `rust-toolchain.toml`; installing
it adds a toolchain without changing your global Rust default.

```sh
rustup toolchain install nightly-2026-01-20 --profile minimal \
  --component rustc-dev --component llvm-tools --component rust-src \
  --component rustfmt --component clippy
python3 scripts/verify_rust_frontend.py --online
cargo test --offline --locked --workspace
python3 -m unittest discover -s scripts -p 'test_*.py'
cargo fmt --all -- --check
cargo clippy --offline --locked --all-targets -- -D warnings
```

Use the verification script without `--online` after dependencies are cached.
It uses the committed Cargo lockfiles and builds target `core` from the pinned
nightly's sources. Tested host: `aarch64-apple-darwin`. Other hosts are unqualified.
The target kernel check does not invoke a native Hexagon linker or emit a device
executable. The separate target suite validates the generated C wrapper ABI through
a real target C caller; no native Rust ABI object is linked into that executable.

Twenty-seven compiler checks cover direct/helper markers, divergent local joins, selective masks, repeatable reports,
concrete pointer bodies, generated C, unguarded/scalar-store rejection, unsupported
types, division/shift/loop/drop rejection, wrapping arithmetic, borrow/type errors, nonexistent entries, unavailable dependency bodies,
normal-compilation rejection, and the unsupported debug runtime dependency.
Other tests cover verifier failures, bounded expansion, safe output publication,
cache integrity, container cleanup, and stale evidence. They do not mock DSP success.

After extraction, generate C separately with:

```sh
cargo run --offline --locked -p ripple-codegen -- \
  artifacts/direct-marker.json artifacts/direct-marker.c
```

Fresh logs and inventories are written under ignored `artifacts/`. Only
`artifacts/rust-verification.json` denotes a completed Rust/IR/C-generation test run.
Published success inventories are cleared before a new run. Compiler failure
also invalidates the active report. Diagnostic source paths can include the
local sysroot; cross-machine byte-identical inventory output is not promised.

## Exact extraction configuration

- rustc `d940e56841ddcc05671ead99290e35ff2e98369f`, nightly 2026-01-20.
- Target `hexagon-unknown-none-elf`, CPU `hexagonv68`, 32-bit pointers.
- Release profile, `panic=abort`, target `core` via `-Zbuild-std=core`, with
  `-Zbuild-std-features=compiler-builtins-mem`.
- The workspace wrapper sets `-Zmir-opt-level=0` only for the kernel fixture;
  dependency MIR uses rustc's release defaults. It recognizes the lane marker by
  resolved diagnostic-item identity from `ripple-kernel`.
- Query sequence: `after_analysis`, explicit local `mir_borrowck`, then
  `instance_mir` (optimized MIR for ordinary function items), with concrete
  substitutions and resolved direct callees. Indirect/unavailable calls fail.
- Probe applies only to the named test crate. It is intentionally not a generic
  Cargo wrapper, compiler plugin, or implementation of the planned public CLI.

The initial debug dependency build with a global MIR override failed with duplicate
`fma` symbols. A release build with that override succeeded, so the observation is
configuration-specific. Keeping the override local avoids changing dependency
compilation unnecessarily. In the scoped debug configuration, pointer checks
reach `core::panicking::panic_nounwind_fmt`, whose MIR body is unavailable here;
the probe rejects this instead of omitting the check. Release mode is currently
the only positive extraction configuration tested.

## Build the real target tools

The official Ripple release has source archives but no prebuilt release assets.
All tools remain in ignored local artifacts. Bootstrap pins the source archive,
SDK archive metadata, compatibility packages, and Docker base image; successful
builds record executable hashes and the immutable simulator image ID.

```sh
python3 scripts/build_ripple_host.py --bootstrap
python3 scripts/fetch_hexagon_runtime.py
python3 scripts/prepare_simulator.py --download
```

The native build can take substantial time. It uses two jobs and stops below
3 GiB free disk; rerun to resume. Host binaries are built with `-O0` to reduce
build cost; kernels still use `-O2`. Cached source bytes are verified against the
pinned archive plus an explicit two-line Darwin portability patch: a constant-vector
callback uses `size_t`, matching its declared interface. Provenance records the
patch and patched-file hash. No moving branch is used.

The SDK fetcher uses bounded HTTP ranges, verifies pinned ZIP metadata and entry
CRCs, repairs file modes, and records per-file SHA256 values. Qualcomm tools remain
local and are not redistributed. The Linux x86-64 simulator runs in Docker on
Apple Silicon with explicit `--mv68`, no network, and a read-only artifact mount. Its uniquely named
container is removed on completion or timeout. The installed simulator starts
and reports version 19.0.07. Linking uses the matching SDK QCLD 19.0.07 and G0
libraries through `run_hexagon_linker.py`. The SDK is read-only, and only the current
isolated target-test directory is writable by the linker container. Native LLD
was built but is not used: compatibility probes found unsupported default-library
GP relocations and, with G0, missing SDK startup symbols. No startup symbols were
stubbed or linker layout guessed to bypass those failures.

## Real Ripple/Hexagon gate

After the bootstrap completes, run `python3 scripts/verify_target_suite.py`.
It verifies local toolchain hashes, reruns Rust extraction, and runs the handwritten,
direct, helper, divergent-join, selective-mask, and arithmetic variants. Each keeps separate logs/reports;
`artifacts/target-suite.json` records overall success or the failed case.

`probes/copy.c`, `probes/launch.h`, and `probes/driver.c` are **hand-written**
feasibility fixtures. They propose a versioned C launch descriptor and exercise
an explicit one-block call boundary, zero length, tails, full/multiple blocks,
buffer guards, invalid descriptor versions/sizes, and invalid launch arguments.
The selective-mask variant leaves half of each output block untouched. Comparisons
are bitwise, including signed zero, NaN payloads, infinities, and subnormal values.
Generated full blocks use distinct lane addresses; partial blocks use a scalar
loop over valid lanes. All six cases passed
with exact float-bit comparisons over 13 lengths (0, 1, 2, 15, 16, 17, 31, 32, 33,
63, 64, 65, 96). Rejected and no-work calls also preserve both buffers bit-for-bit.

```sh
python3 scripts/verify_ripple_target.py \
  --clang "$PWD/artifacts/ripple-build/bin/clang" \
  --simulator "$PWD/scripts/run_hexagon_simulator.py" \
  --extra-args-json "[\"-B$PWD/artifacts/hexagon-sdk/target\",\"-fuse-ld=$PWD/scripts/run_hexagon_linker.py\"]"
```

Add SDK-specific linker/runtime arguments to that JSON array as required by the
chosen installation. They are passed directly as arguments, never through a shell.
The script uses real `<ripple.h>`, `-fenable-ripple`, v68/128-byte HVX flags, and the
vendored upstream simulator invocation convention. It requires both a zero exit
status and the runtime success sentinel. Missing tools and configuration/compiler/
runtime errors return nonzero. It records executable hashes and compiler output.
A pass without `--generated-report` covers only the handwritten ABI probe.
Add `--generated-report artifacts/direct-marker.json` to freshly lower and compile
the Rust-generated kernel, or use `helper-marker.json` for the helper variant.
The report records which source was tested, input/source/executable hashes, and
build provenance. Only the generated stage can qualify this Rust transfer slice.
Neither stage qualifies reductions, division, or general predication.

Current gate: **passed for the ABI-v1 f32 transfer subset on the simulator**.
The observed simulator model is `0x00008d68 (v68n_1024)`. The native Ripple compiler
uses the recorded Darwin portability patch; the target executable uses SDK QCLD,
G0 runtime libraries, and real `<ripple.h>`. Physical-device behavior, broader
language support, and performance have not been qualified.

## Next implementation step

The diagnostic/IR hardening milestone is complete for this restricted slice.
See [the support and diagnostic contract](SUPPORTED_SUBSET.md), including the
SDK integer-width correction and scalar-tail lowering. Next, design arbitrary
project entry selection/public CLI and expand semantic/shape analysis incrementally. The wrapper currently applies only
to the fixture crate. The existing Python and VSCode CUDA paths remain independent.

See the [reviewed implementation plan](../docs/superpowers/plans/2026-10-04-rust-frontend.md).
