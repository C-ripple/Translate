#ifndef RIPPLE_PROBE_LAUNCH_H
#define RIPPLE_PROBE_LAUNCH_H
#include <stdint.h>

/* Prototype ABI v1, exercised by the target simulator suite. All pointers
 * designate target memory; this header does not implement host/DSP transport. */
typedef struct {
    uint32_t abi_version;
    uint32_t struct_bytes;
    uint32_t length;
    uint32_t block_offset;
} ripple_launch_v1;

enum { RIPPLE_OK = 0, RIPPLE_BAD_ABI = 1, RIPPLE_BAD_LAUNCH = 2 };

/* src/dst cover length floats and do not overlap. The caller owns their lifetime
 * and guarantees exclusive writes. Buffers are aligned to 128 bytes. One call
 * executes at most one 32-lane block. The Rust entry receives src/dst rebased
 * by block_offset, n = length - block_offset, and a local lane ID in [0,32).
 * n is the remaining length, not the original total length. Host/DSP transfer,
 * allocation validity, and scheduling all blocks are the caller's responsibility. */
int ripple_probe_copy_v1(const ripple_launch_v1 *, const float *src, float *dst);
#endif
