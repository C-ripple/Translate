# Independent senior-level review

Reviewed 2026-10-04 by a separate AI reviewer acting as a senior compiler engineer.
This is an independent technical review, not human sign-off or target validation.

Documents: [design](2026-10-04-rust-frontend-design.md) and
[implementation plan](../plans/2026-10-04-rust-frontend.md).

Initial verdict: credible architecture, with changes needed before implementation.
The reviewer inspected relevant repository code and local Ripple specifications,
and consulted primary Rust documentation. The repository findings about regex
generation, the separate textual LLVM path, and syntax-only validation held up.

| Priority | Finding | Resolution |
| --- | --- | --- |
| P1 | Source lane semantics were underspecified, and SPMD legality came too late | Defined one logical invocation per lane, restricted initial kernels to one dimension, and moved minimum shape/predicate/join analysis into milestone 2 |
| P1 | A C branch does not establish safe inactive-lane evaluation | Added proof, operand sanitization, target-tested safe lowering, or rejection policy for arithmetic and pointer formation; required target probes |
| P2 | Marker preservation and MIR extraction were not concrete compatibility artifacts | Required exact query/dialect, query ordering, pass/optimization settings, marker encoding, dependency handling, and direct/helper preservation tests |
| P2 | The first raw-pointer kernel needs generic core operations before general generics are scheduled | Required the canonical fixture's concrete core/intrinsic dependency closure in milestone 2 |
| Refinement | CUDA support was described too much like an adapter substitution | Required a separate CUDA semantic and migration plan, while retaining shared infrastructure |
| Refinement | Layout checks alone do not establish calling-convention compatibility | Required a generated C entry, versioned launch descriptor, target address-space contract, and real C caller tests |

The first two findings identify missing correctness obligations, not demonstrated
Ripple compiler bugs. Supporting evidence includes local
`temp_ripple_docs/src/ripple-spec/spmd.md`: operation shapes depend on data,
lower-shape effects can use an OR-reduced predicate, and the documented masked
operations are loads, stores, and reductions. The
[Rust pointer contract](https://doc.rust-lang.org/core/primitive.pointer.html#method.add)
also constrains pointer formation. Extraction constraints follow the
[MIR query/pass documentation](https://rustc-dev-guide.rust-lang.org/mir/passes.html).

After revision, the same reviewer re-read both documents and found all four
prioritized findings and both refinements addressed, without a material new
contradiction. No further blocking document changes were identified.

Final disposition: **approved to begin feasibility milestones**. This is not an
approval to claim compiler support or production readiness. Exact toolchain
compatibility, MIR extraction/marker preservation, and actual target execution
remain required evidence. The Rust-input product choice remains provisional.

Only documentation changed during this review. Document links, code fences, and
whitespace were checked. The prior 155-test baseline was not rerun; no compiler
implementation or target compilation was performed during review.

## Implementation review: extraction, restricted lowering, and bootstrap

Reviewed 2026-10-04 by the same independent AI reviewer. This section records
source inspection and review of existing evidence, not human approval. The reviewer
did not start additional builds or rerun the implementation test suites.

Scope inspected: the pinned rustc inventory and owned-MIR importer, the typed IR
verifier, the restricted Ripple C emitter and publication CLI, ABI probes, and
compiler/SDK/simulator bootstrap scripts. The implementation supports a narrow
guarded f32 buffer-transfer subset; it is not a general Rust front end.

The following implementation findings were corrected and the corrections inspected:

| Finding | Inspected correction |
| --- | --- |
| Failed target configuration and Rust reruns could leave stale successful evidence | Verification scripts invalidate earlier reports; target configuration errors produce failed evidence |
| SDK refresh could retain a stale manifest or accept incorrect cached file modes/kinds | Manifest invalidation precedes fetching, cached permissions are restored, wrong symlink kinds are repaired, and completion is published by rename |
| Simulator timeout did not ensure daemon-managed container cleanup | Each run has a unique container name and forced cleanup in `finally`, with time reserved before the outer timeout |
| Existing source-directory presence was treated as evidence of a complete pinned extraction | Extraction uses a temporary directory and rename; source files, kinds, symlink targets, missing files, and extra files are compared with the pinned archive before configuration |
| Symbolic expression growth and effects inside a block bypassed expansion limits | Assignment/block work, expression size, stores at insertion, and emitted-source size now have explicit limits |
| Codegen could overwrite its own input or leave partial generated C | Canonical-path and Unix inode/device alias checks precede invalidation; output uses exclusive temporary creation, synchronization, and rename |
| The block wrapper's Rust argument mapping was implicit | `compiler/probes/launch.h` specifies rebased buffers, `n = length - block_offset`, local lane IDs, and caller obligations |

The earlier inventory limitation remains explicit: direct-call traversal is not
complete traversal of implicit panic/drop paths or intrinsic implementations.
The owned importer rejects unsupported operations instead of claiming their
semantics have been implemented. IR verification establishes type, CFG,
initialization, and direct-call properties; SPMD legality remains a separate gate.

Within the emitter's accepted subset, every read is from immutable `src[lane]`
and every store targets `dst[lane]`, with disjoint buffers and a proven lane bound.
These restrictions justify symbolic load substitution and path-effect ordering
for the supported programs. Inactive offsets are sanitized before emitted pointer
formation. Actual Ripple execution must still establish the target behavior.

The Darwin portability patch was checked against the local pinned upstream source:
`TensorShapeAny::foreachIndex` supplies `ArrayRef<size_t>`, and `getOffsetAt`
accepts that same coordinate type. Changing the broadcasting callback parameter
and its temporary coordinate vector to `size_t` matches both interfaces. The
patch does not change target integer types or the indexing algorithm. The build
script requires each original replacement site exactly once, compares cached
source against the archive with only this declared transformation, and records
the replacements and patched-file digest in build provenance. This is a pinned
upstream source build with a disclosed host portability patch, not an unmodified
upstream binary.

Existing `artifacts/rust-verification.json` records 17 extraction/code-generation
checks and explicitly leaves target validation false. The implementing agent also
reports 11 Rust tests, eight Python evidence/bootstrap tests, and clippy passing;
those suites were not independently rerun in this review.

Disposition: **no remaining blocking source-review finding in this restricted
implementation**. Native Ripple compilation was still in progress at this review;
generated-kernel target compilation and simulator execution remained pending.
Do not describe the end-to-end milestone or release gates as passed until those
checks execute successfully. This review does not establish broader Rust support,
production readiness, or performance claims.

## Target evidence and SDK linker review

Follow-up reviewed 2026-10-04 by the independent AI reviewer. The target suite
completed during this review. The reviewer inspected the scripts, all five case
reports and simulator logs, and the final successful `artifacts/target-suite.json`;
no additional build or simulator run was started by the reviewer.

The suite records successful handwritten, direct-marker, helper-marker,
divergent-join, and selective-marker cases. All five simulator logs contain the
runtime success sentinel and identify model `0x00008d68 (v68n_1024)`. The four Rust
cases are explicitly marked compiler-generated. Each case's recorded caller,
ABI-header, and linker-adapter SHA256 values match the files inspected in this
review. The driver compares exact float bits and buffer canaries over 13 lengths;
the selective oracle leaves the lower half of each block untouched. A final
review finding was corrected: rejected and zero-work launches now snapshot and
compare both buffers as well as checking return codes.

`run_hexagon_linker.py` was inspected as a narrow bootstrap adapter for native
Clang's linker invocation. It maps explicit artifact paths into Docker, confines
the output to an isolated target-probe directory, rejects response files, mounts
the SDK read-only, and makes only that probe directory writable. It uses the
recorded immutable image ID and a unique container name with timeout cleanup.
The regression test covers outside-artifact input rejection and cleanup after a
timeout. It is not presented as a general linker service for arbitrary arguments.

The tested combination is the pinned Ripple Clang source build with the disclosed
Darwin portability patch, SDK QCLD/linker dependencies and runtime 19.0.07, `-G0`,
and the SDK simulator. Native LLD was built but is not the linker used by these
passing cases. The scripts record this distinction and verify the compiler and
SDK binary/library pins before the suite. The compiler version banner embeds the
enclosing checkout's Git identification; the source-archive pin, verified source
tree, declared patch, and executable hashes are the relevant provenance evidence.

Disposition: **the first restricted Rust-to-Ripple transfer slice has real
simulator execution evidence, with no new blocking review finding**. This
supersedes the earlier pending-target status in this chronological review record.
It establishes these tested ABI-v1 buffer-transfer cases on the named toolchain
and simulator, not a completed professional front end, release qualification,
physical-device validation, general predication correctness, or performance.

## Diagnostic and semantic hardening review

On resumption, the independent senior-style AI reviewer assessed structured
source-bearing diagnostics, IR/input limits, strict deserialization, transitive
MIR-effect rejection, and the new wrapping-arithmetic target oracle. It requested
nested unknown-field coverage through the actual CLI and unwind rejection tests.
Both were added. M005 is tested with actual rustc Continue/Cleanup actions and
call-site context; the pinned panic-abort configuration does not provide an
end-to-end unwind test, and no unwind support is claimed.

The new real target test exposed SDK-dependent `UINT32_C` widening. The reviewer
confirmed different macro expansion with the actual SDK `-B` path. Generated
integer literals now use unsigned suffixes and arithmetic operations explicitly
produce u32. A subsequent sparse-tail failure remained at length 15. The emitter
now uses a scalar loop for partial blocks and distinct-address Ripple operations
for full blocks. The reviewer found this split correct under the existing
immutable-source, disjoint-buffer and lane-exclusive-write contract. The exact
downstream cause of the previous tail failure remains unproven; these results do
not qualify duplicate-address masked scatter.

Final review inspected the six-case passing target suite and arithmetic success
log. Recorded generated source, input report, codegen executable, caller and ABI
header hashes matched current files. The reviewer found no remaining
implementation blocker. Three stale README counts/tail descriptions were corrected.
Support and limitation details are recorded in compiler/SUPPORTED_SUBSET.md.

Final checks: six real simulator cases, 27 frontend checks, 18 Rust tests,
10 Python harness tests and 155 existing project tests passed. Formatting and
clippy with warnings denied passed. This completes the bounded diagnostic and
semantic hardening milestone, not the professional frontend roadmap. All changes
remain local and uncommitted.
