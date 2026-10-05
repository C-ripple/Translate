#include <stdio.h>
#include <string.h>
#include "launch.h"

enum { GUARD = 32, CAPACITY = 96, TOTAL = GUARD + CAPACITY + GUARD };

static uint32_t input_bits(unsigned i) {
    /* Include exceptional values once; distinct other elements expose wrong
     * lane offsets and source-block rebasing, even in one-element tails. */
    const uint32_t special[] = {0, 0x80000000u, 0x7fc12345u, 0x7f800000u,
                               0xff800000u, 1u, 0x3f800000u, 0xbf123456u};
    if (i >= GUARD && i < GUARD + sizeof(special) / sizeof(special[0]))
        return special[i - GUARD];
    return 0x3f000000u + i * 0x10203u;
}

int main(void) {
    _Alignas(128) float src[TOTAL], dst[TOTAL];
    const uint32_t sizes[] = {0, 1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 96};
    /* Transfers must preserve bits, including signed zero and NaN payloads. */
    const uint32_t untouched = 0xc640e400u;
    for (unsigned test = 0; test < sizeof(sizes) / sizeof(sizes[0]); ++test) {
        uint32_t n = sizes[test];
        for (unsigned i = 0; i < TOTAL; ++i) {
            uint32_t bits = input_bits(i);
            memcpy(&src[i], &bits, sizeof(bits));
            memcpy(&dst[i], &untouched, sizeof(untouched));
        }
        ripple_launch_v1 launch = {1, sizeof(launch), n, 0};
        /* This loop is the explicit test launcher, not generated grid support. */
        for (uint32_t offset = 0; offset < n || (n == 0 && offset == 0); offset += 32) {
            launch.block_offset = offset;
            if (ripple_probe_copy_v1(&launch, src + GUARD, dst + GUARD) != RIPPLE_OK)
                return 1;
        }
        for (unsigned i = 0; i < TOTAL; ++i) {
            int active = i >= GUARD && i < GUARD + n;
#ifdef RIPPLE_TEST_UPPER_HALF
            active = active && (i - GUARD) % 32 >= 16;
#endif
#ifdef RIPPLE_TEST_ARITHMETIC
            if (active) {
                uint32_t index = i - GUARD;
                uint32_t lane = index % 32;
                uint32_t remaining = n - (index / 32) * 32;
                /* Independent wide-integer reference, with explicit modulo. */
                uint64_t sum = ((uint64_t)lane + UINT32_MAX) % (UINT64_C(1) << 32);
                uint64_t product = (sum * UINT64_C(2147483649)) % (UINT64_C(1) << 32);
                uint64_t value = (product + (UINT64_C(1) << 32) - 3) % (UINT64_C(1) << 32);
                active = lane == 2 || value < remaining;
            }
#endif
            uint32_t original = input_bits(i);
            uint32_t expected = active ? original : untouched;
            uint32_t output_bits, source_bits;
            memcpy(&output_bits, &dst[i], sizeof(output_bits));
            memcpy(&source_bits, &src[i], sizeof(source_bits));
            if (output_bits != expected || source_bits != original) {
                printf("mismatch n=%u index=%u actual=%08x expected=%08x source=%08x\n",
                       n, i, output_bits, expected, source_bits);
                return 2;
            }
        }
    }
    uint32_t before_src[TOTAL], before_dst[TOTAL];
#define REQUIRE_NO_EFFECT(call, expected, code) do {                         \
    memcpy(before_src, src, sizeof(src));                                   \
    memcpy(before_dst, dst, sizeof(dst));                                   \
    int status = (call);                                                   \
    if (status != (expected) || memcmp(before_src, src, sizeof(src)) != 0 || \
        memcmp(before_dst, dst, sizeof(dst)) != 0) return (code);           \
} while (0)
    ripple_launch_v1 launch = {2, sizeof(launch), 1, 0};
    REQUIRE_NO_EFFECT(ripple_probe_copy_v1(&launch, src, dst), RIPPLE_BAD_ABI, 3);
    launch.abi_version = 1;
    launch.struct_bytes = 0;
    REQUIRE_NO_EFFECT(ripple_probe_copy_v1(&launch, src, dst), RIPPLE_BAD_ABI, 4);
    launch.struct_bytes = sizeof(launch);
    launch.block_offset = 32;
    REQUIRE_NO_EFFECT(ripple_probe_copy_v1(&launch, src, dst), RIPPLE_BAD_LAUNCH, 5);
    launch.block_offset = 0;
    REQUIRE_NO_EFFECT(ripple_probe_copy_v1(&launch, 0, dst), RIPPLE_BAD_LAUNCH, 6);
    launch.length = 0;
    REQUIRE_NO_EFFECT(ripple_probe_copy_v1(&launch, 0, 0), RIPPLE_OK, 7);
    REQUIRE_NO_EFFECT(ripple_probe_copy_v1(0, src, dst), RIPPLE_BAD_ABI, 8);
    launch.length = 1;
    REQUIRE_NO_EFFECT(ripple_probe_copy_v1(&launch, src, 0), RIPPLE_BAD_LAUNCH, 9);
    launch.length = 33;
    launch.block_offset = 1;
    REQUIRE_NO_EFFECT(ripple_probe_copy_v1(&launch, src, dst), RIPPLE_BAD_LAUNCH, 10);
    launch.length = 32;
    launch.block_offset = 32;
    REQUIRE_NO_EFFECT(ripple_probe_copy_v1(&launch, 0, 0), RIPPLE_OK, 11);
#undef REQUIRE_NO_EFFECT
    puts("RIPPLE_ABI_COPY_V1_PASS");
    return 0;
}
