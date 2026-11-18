/* Q4_K 16-row repack and its inverse, with a round-trip check.
 *
 *   block_q4_K    : d 2 | dmin 2 | scales 12 | qs 128                 = 144/row
 *   block_q4_Kx16 : d[16] 32 | dmin[16] 32 | scales 192 | qs 2048     = 2304
 *
 * 2304 == 16 * 144, so the transform is a pure permutation inside one window
 * and can run in place.
 *
 * Layout chosen here (our own - the kernel is ours, so only make/unmake have to
 * agree):
 *   d[r]    at   0 + 2r        dmin[r] at  32 + 2r        <- contiguous, so a
 *                                                          vector load gets all
 *                                                          sixteen at once
 *   scales  at  64: eight records of 24 bytes, one per sub-block, each holding
 *            sixteen 6-bit scales packed bitwise into 12 bytes followed by
 *            sixteen 6-bit mins in the next 12
 *   qs      at 256: 8-byte interleave across the sixteen rows, so 128
 *            consecutive bytes hold one 8-byte chunk of each row
 *
 * The 8-row version of this already round-tripped; this is the same shape with
 * sixteen rows, which is what lets one zmm hold sixteen output rows.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define QK_K        256
#define Q4K_SCALES   12u
#define Q4K_ROW     144u
#define Q4K_X16    2304u
#define ROWS         16

typedef struct {
    uint16_t d, dmin;
    uint8_t scales[Q4K_SCALES];
    uint8_t qs[QK_K / 2];
} row_q4k;

static void sc_min(const uint8_t *q, int j, uint8_t *s, uint8_t *m) {
    if (j < 4) { *s = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *s = (uint8_t)(((q[j - 4] & 192) >> 2) | (q[j + 4] & 15));
        *m = (uint8_t)(((q[j] & 192) >> 2) | ((q[j + 4] & 240) >> 4));
    }
}

/* sixteen 6-bit values, bitwise, into twelve bytes */
static void pack6(const uint8_t *v, uint8_t *out) {
    memset(out, 0, 12);
    for (int i = 0; i < ROWS; i++)
        for (int b = 0; b < 6; b++) {
            const int bit = i * 6 + b;
            if (v[i] & (1u << b)) out[bit >> 3] |= (uint8_t)(1u << (bit & 7));
        }
}

static void unpack6(const uint8_t *in, uint8_t *v) {
    for (int i = 0; i < ROWS; i++) {
        unsigned x = 0;
        for (int b = 0; b < 6; b++) {
            const int bit = i * 6 + b;
            if (in[bit >> 3] & (1u << (bit & 7))) x |= 1u << b;
        }
        v[i] = (uint8_t)x;
    }
}

static void repack16(const row_q4k *src, uint8_t *dst,
                     uint64_t row_stride, uint64_t blocks) {
    for (uint64_t b = 0; b < blocks; b++) {
        uint8_t *o = dst + b * Q4K_X16;
        const row_q4k *in[ROWS];
        for (int r = 0; r < ROWS; r++)
            in[r] = (const row_q4k *)((const uint8_t *)src + (uint64_t)r * row_stride)
                    + b;
        for (int r = 0; r < ROWS; r++) {
            memcpy(o + 0 + r * 2, &in[r]->d, 2);
            memcpy(o + 32 + r * 2, &in[r]->dmin, 2);
        }
        uint8_t sv[ROWS], mv[ROWS];
        for (int j = 0; j < 8; j++) {
            for (int r = 0; r < ROWS; r++) sc_min(in[r]->scales, j, &sv[r], &mv[r]);
            pack6(sv, o + 64 + j * 24);
            pack6(mv, o + 64 + j * 24 + 12);
        }
        /* 4-byte interleave, not 8: dpbusd lane k consumes bytes 4k..4k+3, so
         * four bytes per row puts row k exactly in lane k and all sixteen rows
         * fill the sixteen lanes. With eight bytes per row the lanes cover only
         * eight rows at two lanes each. */
        for (int i = 0; i < 512; i++) {
            const int r = i % ROWS, off = (i / ROWS) * 4;
            memcpy(o + 256 + (uint64_t)i * 4, in[r]->qs + off, 4);
        }
    }
}

static void unrepack16(const uint8_t *src, row_q4k *dst,
                       uint64_t row_stride, uint64_t blocks) {
    for (uint64_t b = 0; b < blocks; b++) {
        const uint8_t *o = src + b * Q4K_X16;
        row_q4k *out[ROWS];
        for (int r = 0; r < ROWS; r++)
            out[r] = (row_q4k *)((uint8_t *)dst + (uint64_t)r * row_stride) + b;
        for (int r = 0; r < ROWS; r++) {
            memcpy(&out[r]->d, o + 0 + r * 2, 2);
            memcpy(&out[r]->dmin, o + 32 + r * 2, 2);
        }
        for (int i = 0; i < 512; i++) {
            const int r = i % ROWS, off = (i / ROWS) * 4;
            memcpy(out[r]->qs + off, o + 256 + (uint64_t)i * 4, 4);
        }
        uint8_t sv[ROWS], mv[ROWS];
        for (int j = 0; j < 8; j++) {
            unpack6(o + 64 + j * 24, sv);
            unpack6(o + 64 + j * 24 + 12, mv);
            for (int r = 0; r < ROWS; r++) {
                uint8_t *q = out[r]->scales;
                if (j < 4) {
                    q[j]     = (uint8_t)((q[j] & ~63)     | (sv[r] & 63));
                    q[j + 4] = (uint8_t)((q[j + 4] & ~63) | (mv[r] & 63));
                } else {
                    q[j - 4] = (uint8_t)((q[j - 4] & ~192) | ((sv[r] & 48) << 2));
                    q[j]     = (uint8_t)((q[j] & ~192)     | ((mv[r] & 48) << 2));
                    q[j + 4] = (uint8_t)((sv[r] & 15) | ((mv[r] & 15) << 4));
                }
            }
        }
    }
}

int main(void) {
    for (uint64_t blocks = 1; blocks <= 4; blocks++) {
        const uint64_t row_stride = blocks * Q4K_ROW;
        const size_t n = (size_t)ROWS * row_stride;
        uint8_t *orig = malloc(n), *tmp = malloc(blocks * Q4K_X16),
                *back = malloc(n);
        for (size_t i = 0; i < n; i++) orig[i] = (uint8_t)((i * 197u + 31u) & 0xFF);
        repack16((const row_q4k *)orig, tmp, row_stride, blocks);
        int permuted = memcmp(tmp, orig, n) != 0;
        unrepack16(tmp, (row_q4k *)back, row_stride, blocks);
        const int ok = memcmp(back, orig, n) == 0;
        printf("blocks=%llu permuted=%d roundtrip=%s\n",
               (unsigned long long)blocks, permuted, ok ? "OK" : "MISMATCH");
        if (!ok) {
            for (size_t i = 0; i < n; i++)
                if (back[i] != orig[i]) {
                    printf("  first diff at byte %zu (row %zu): orig=%02x back=%02x\n",
                           i, i / row_stride, orig[i], back[i]);
                    break;
                }
            return 1;
        }
        free(orig); free(tmp); free(back);
    }
    printf("all round trips OK - 16-row repack is a correct permutation\n");
    return 0;
}
