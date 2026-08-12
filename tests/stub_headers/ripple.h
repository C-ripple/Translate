/*
 * Minimal stub of the real RIPPLE API (see the upstream spec at
 * temp_ripple_docs/src/ripple-spec/api.md) — just enough for a generic
 * clang to resolve #include <ripple.h> during a syntax-only check.
 *
 * Deliberately uses the REAL upstream function names, not whatever the
 * translator happens to emit. If the translator's output doesn't match
 * this stub, that IS a real bug the check should catch — that's exactly
 * how the ripple_get_size / ripple_get_block_size mismatch fixed in
 * this same round was found.
 *
 * NOT a full implementation and NOT suitable for compiling against the
 * real Hexagon toolchain — see docker/README.md for that heavier check.
 */
#ifndef RIPPLE_STUB_H
#define RIPPLE_STUB_H

#include <stddef.h>

typedef struct ripple_block_s *ripple_block_t;

#define HVX_PE 0

ripple_block_t ripple_set_block_shape(int pe_id, ...);
size_t ripple_id(ripple_block_t block_shape, int dim);
size_t ripple_get_block_size(ripple_block_t block_shape, int dim);

/* Shuffle/reduction: declared for completeness, though every current
 * call site of ripple_shuffle is known-broken (not valid C at all —
 * see GitHub issue #8), so declaring it correctly doesn't make those
 * kernels pass — they fail on the actual invalid syntax at the call
 * site, which is the correct, honest failure. */
typedef size_t (*ripple_shuffle_fn_t)(size_t, size_t);
size_t ripple_shuffle(size_t value, ripple_shuffle_fn_t fn);

/* api.md line 45: TYPE ripple_reduceadd(int dims, TYPE to_reduce) — the
 * first argument is a dimension bitfield, not optional. This stub
 * intentionally matches upstream arity rather than the translator's
 * current 1-arg emission (core/translation_rules.py, warp reduction
 * optimization rule) — that emission mismatch is a real, undiscovered
 * bug of the same class as the shuffle issue above, not something to
 * paper over here. Filed as GitHub issue #10, not fixed here. */
size_t ripple_reduceadd(int dims, size_t to_reduce);

#endif
