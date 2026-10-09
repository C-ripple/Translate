# Preserve block shape implementation

Design: ../specs/2026-10-08-preserve-block-shape-design.md

1. Add failing coverage and coordinate regression fixtures.
2. Replace heuristic shape selection with explicit static shape validation.
3. Pass SIMD blocks and compact execution context into translated kernels.
4. Emit per-kernel launch wrappers with dynamic x/y/z grid loops.
5. Expose optional shape configuration through library, CLI, and HTTP APIs.
6. Update the web editor, migration notes, examples, and standing project notes.
7. Run the full project suite, scalar coverage fixtures, C syntax checks, and
   web editor smoke checks. Review the diff before reporting completion.

No native Ripple runtime thread API is required. No native Hexagon execution is
claimed by the scalar fixtures.
