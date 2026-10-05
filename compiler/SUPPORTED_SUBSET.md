# Rust frontend support and diagnostic contract

This is a restricted compiler prototype, pinned to nightly-2026-01-20 and the
32-bit Hexagon target. Support requires both IR verification and SPMD legality.
A well-typed IR module alone is not permission to emit memory operations.

| Construct | Current contract / evidence |
| --- | --- |
| Scalars | Unit, bool, u32, 32-bit usize, f32 transfer values |
| Integer arithmetic | Wrapping Add/Sub/Mul, unsigned Lt/Eq, u32/usize casts; arithmetic simulator fixture uses an independent u64 modulo oracle |
| Float arithmetic | Rejected; float loads and stores preserve bits |
| Memory | Raw f32 pointers; only guarded src[lane] to dst[lane] transfers; disjoint, initialized, aligned buffers are caller obligations |
| Control flow | Acyclic boolean branches and joins; direct concrete helper calls; recursion and loops rejected |
| Rust safety obligations | Rust type/borrow checking first; reachable MIR assertions, drops and unwinding calls rejected, including helper bodies |
| Division / shifts | Rejected, with assertion/unsupported-operation diagnostics; no target semantics claimed |
| Unsupported types | Includes u64, references and aggregate values outside the restricted importer |
| ABI | Version 1 descriptor, explicit 32-lane block scheduling; see README and launch.h |
| Entry selection | Fixture crate only; public arbitrary-project driver remains deferred |

## Diagnostics

Compiler-owned failures emit one `RIPPLE_DIAGNOSTIC: ` JSON line followed by a
human-readable error. Schema version 1 has `code`, `message`, optional `source`
and optional `function`. Source is an opaque rustc span display (or IR-provided
label), not a machine-readable source range. Native rustc type/borrow errors keep
rustc's own diagnostics. Startup/publication errors may still use legacy probe
codes; these are outside the importer diagnostic contract.

| Codes | Meaning |
| --- | --- |
| RIPPLE-D000 / D001 | Pipeline or I/O failure / malformed JSON or IR representation |
| RIPPLE-I001–I006 | Invalid IR, control flow, types, initialization, calls, resource limits |
| RIPPLE-M001–M006 | Unsupported type, operation, assertion, drop, unwind, call |
| RIPPLE-S001–S004 | Missing bounds proof, unsupported address/access, expansion limit, entry contract |

Messages may evolve; consumers should match codes and schema version. Context is
attached when available; a malformed whole report need not have a source span.
Unknown IR fields fail deserialization. Reports are limited to 16 MiB; IR limits
include 256 functions, 4096 locals/blocks per function, 4096 assignments per block,
and 16384 operations per module. Lowering has additional expression, step, store
and output limits. Failed generation invalidates prior output while protecting
input/output aliases.

M005 is a defensive importer guard, unit-tested with rustc `Continue` and
`Cleanup` actions and call-site context. The pinned panic-abort configuration
rewrites supported calls to non-unwinding actions; no end-to-end unwind support
or unwind-configuration qualification is claimed.

## Target lowering

Integer constants use explicit `U` suffixes and arithmetic results are cast to
`uint32_t`. The SDK's `UINT32_C` macro can widen large decimal constants to signed
64-bit, so emitted arithmetic must not depend on that macro.

Full blocks use Ripple with distinct lane addresses. Partial blocks use a scalar
loop over valid lanes, with the identical generated predicates and store order.
The earlier duplicate inactive-address scheme failed the sparse arithmetic
regression at length 15 after integer-width correction; the exact downstream
backend cause is not established. Scalar tails avoid that address pattern. This
is not qualification of general masked gather/scatter semantics.

## Verification scope

The target suite tests handwritten, direct, helper, divergent-join, selective-mask
and arithmetic kernels over 13 buffer lengths. It checks exact float bits, guard
regions, inactive lanes and invalid/no-work launches. The arithmetic case covers
wrapping intermediates, target-width constants, casts, comparisons and branch
selection; it is not an exhaustive integer arithmetic proof.

Malformed-IR tests cover invalid references, types, definite initialization,
limits, strict deserialization and stale-output handling. A deterministic bounded
reference-mutation test is included; sustained fuzzing, physical-device testing,
performance qualification and broad Rust compatibility remain future work.
