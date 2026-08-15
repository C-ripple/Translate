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

/* Shuffle/reduction: GitHub issue #8 (all 4 shuffle rules emitting
 * invalid C — lambdas or nested function defs) is fixed as of this
 * checkout — shuffle calls with a compile-time-constant argument now
 * hoist a real, named, file-scope function matching this typedef and
 * compile clean. A variable (non-constant) shuffle argument can't be
 * threaded through this fixed 2-parameter function-pointer signature
 * at all, so those calls are deliberately left untranslated with a
 * warning rather than emitting broken C — tracked separately as
 * GitHub issue #11, not a defect in this stub or the fix. */
typedef size_t (*ripple_shuffle_fn_t)(size_t, size_t);
size_t ripple_shuffle(size_t value, ripple_shuffle_fn_t fn);

/* api.md line 45: TYPE ripple_reduceadd(int dims, TYPE to_reduce) — the
 * first argument is a dimension bitfield, not optional. GitHub issue
 * #10 (the translator emitting a 1-arg call, mismatched against this
 * stub's upstream-accurate arity) is fixed as of this checkout. */
size_t ripple_reduceadd(int dims, size_t to_reduce);

/* opt/hexagon/src/hvx-opt.md's SpVV example uses vtcm_malloc()/vtcm_free()
 * but never gives them a formal declared signature — only usage. This
 * signature is inferred directly from that usage (size + alignment for
 * malloc; a single pointer for free), not from an upstream prototype. */
void *vtcm_malloc(size_t size, size_t align_as);
void vtcm_free(void *ptr);

#endif
