/* Hand-written Ripple feasibility probe. NOT compiler-generated output. */
#include <stddef.h>
#include <ripple.h>
#include "launch.h"

_Static_assert(sizeof(void *) == 4, "expected Hexagon pointer width");
_Static_assert(sizeof(ripple_launch_v1) == 16, "unexpected descriptor size");
_Static_assert(offsetof(ripple_launch_v1, block_offset) == 12, "unexpected ABI layout");

static void copy_block(const float *src, float *dst, uint32_t remaining) {
    ripple_block_t block = ripple_set_block_shape(0, 32);
    size_t lane = ripple_id(block, 0);
    /* remaining is nonzero. Inactive lanes form pointers to element zero,
     * which is valid. Sanitization happens before pointer formation, not after
     * an invalid pointer has already been formed. Target behavior still needs
     * execution evidence; retaining a C if alone is not a proof of predication. */
    size_t safe_lane = lane < remaining ? lane : 0;
    if (lane < remaining) {
        dst[safe_lane] = src[safe_lane];
    }
}

int ripple_probe_copy_v1(const ripple_launch_v1 *launch, const float *src, float *dst) {
    if (!launch || launch->abi_version != 1 || launch->struct_bytes != sizeof(*launch))
        return RIPPLE_BAD_ABI;
    if (launch->block_offset > launch->length || launch->block_offset % 32 != 0)
        return RIPPLE_BAD_LAUNCH;
    uint32_t remaining = launch->length - launch->block_offset;
    if (remaining == 0)
        return RIPPLE_OK;
    if (!src || !dst)
        return RIPPLE_BAD_LAUNCH;
    copy_block(src + launch->block_offset, dst + launch->block_offset, remaining);
    return RIPPLE_OK;
}
