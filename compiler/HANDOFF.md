# Checkpoint: diagnostic and semantic hardening

Resumed on the user's “continue” instruction and completed this bounded milestone
on 2026-10-04. Verification has finished; no builds or simulators remain running.

## Completed

- Pinned rustc adapter performs Rust checking, resolves concrete MIR instances,
  and imports a typed, owned IR with source locations.
- IR verification and bounded SPMD lowering emit Ripple C for the documented
  f32 buffer-transfer subset. Unsupported operations and unsafe memory patterns fail.
- The native Ripple compiler built from pinned source, with an explicitly recorded
  Darwin `size_t` portability patch. SDK QCLD/runtime/simulator 19.0.07 are installed
  locally; target flags include v68, HVX128, and G0.
- All six real simulator cases passed: handwritten reference, direct Rust copy,
  helper, divergent join, selective mask, and wrapping arithmetic. Thirteen lengths cover empty/tail/full/
  multiple blocks. Tests check exact float bits, canaries, untouched masked lanes,
  and no memory changes on rejected/no-work launches.
- Independent senior-style AI review found no remaining blocking findings in this
  restricted implementation. It reviewed source and target evidence; it is not a
  human review or a release certification.

## Hardening changes

- Versioned diagnostics distinguish malformed IR, types, control flow, calls,
  unsupported MIR effects and SPMD legality, with source context where available.
- Strict IR fields and bounded report/IR sizes; malformed-input and stale-output
  regressions, plus deterministic reference mutations.
- Arithmetic simulator regression caught SDK `UINT32_C` widening. Emitted
  literals now have explicit unsigned suffixes and arithmetic results stay u32.
- A further sparse-tail failure led to scalar partial blocks and distinct-address
  Ripple full blocks. The exact backend cause remains unproven; duplicate-address
  masked scatter is not qualified.
- Added loop and transitive drop rejection checks. Unwind rejection is unit-tested
  as a defensive guard because the supported target uses panic-abort.

## Final checks

| Check | Result |
| --- | --- |
| Real target suite | 6 cases passed |
| Rust extraction and C-generation checks | 27 passed |
| Rust workspace tests | 18 passed |
| Python bootstrap/evidence tests | 10 passed |
| Existing project regression suite | 155 passed |
| Formatting, clippy with warnings denied, diff whitespace | Passed |

A final parallel-test temporary-directory collision was corrected with a process
counter; the complete Rust test suite and clippy passed afterward. This changed
only test isolation, not generated kernels or target behavior.

## Resume

Read [README.md](README.md), [the implementation plan](../docs/superpowers/plans/2026-10-04-rust-frontend.md),
and [the review record](../docs/superpowers/specs/2026-10-04-rust-frontend-review.md).
The toolchain is already built; do not rebuild or redownload it unnecessarily.
From this directory, `python3 scripts/verify_target_suite.py` reproduces the full
target gate using Docker and the cached pinned tools. Evidence lives in ignored
`artifacts/target-suite.json` and per-case `target-*.json` / `target-*.run.log`.

The initial gate is complete, not the full professional frontend roadmap. The
adapter still targets the fixture crate; arbitrary project selection/public CLI,
broader arithmetic/control flow, collectives, editor integration, physical-device
qualification, fuzzing, and performance work remain. The IR/diagnostic hardening and [supported-subset matrix](SUPPORTED_SUBSET.md)
are now implemented. Next milestone: design arbitrary-project entry selection
and a public driver, retaining the pinned target and fail-closed language subset.

All work is local and uncommitted. The pre-existing `CLAUDE.md` modification was
preserved. SDK/compiler build artifacts are ignored and must not be committed.
