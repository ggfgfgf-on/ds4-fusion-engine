/* 16 rows x 8 tokens, free layout, against a scalar reference.
 *
 * No llama compatibility, so the layout can be whatever the kernel wants:
 *
 *   struct block_q4_x16 {          per (group, block) of 16 rows
 *       float   d[16];      64     f32, so the kernel loads instead of
 *                                  converting and assembling with set_ps
 *       float   dmin[16];   64
 *       uint8_t sc[8][16]; 128     plain bytes: no six-bit unpacking anywhere
 *       uint8_t mn[8][16]; 128
 *       uint8_t qs[2048];         four-byte interleave over the sixteen rows,
 *                                  so dpbusd lane k is row k
 *   }                          2432, sixty-four byte aligned
 *
 * The record stays interleaved on purpose: a block is processed with its scales
 * and mins, and separating them into their own streams would cost an extra
 * cache line and a second index for data that is only 1.5 percent of the
 * volume.
 *
 * Eight tokens per pass because one weight load then feeds eight tokens; the
 * earlier eight-token attempts died on sixteen scalar min accumulators
 * exhausting the general-purpose registers, and those are gone - the min term
 * is computed outside this kernel.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#define QK_K        256
#define ROWS         16
#define NTOK          8
#define BLOCKS        4
#define ROW_BYTES   (BLOCKS * 144u)
#define X16_BYTES   2432u

typedef struct {
    float d[ROWS];
    float dmin[ROWS];
    uint8_t sc[8][ROWS];
    uint8_t mn[8][ROWS];
    uint8_t qs[2048];
} blk16;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[16]; } q8b;

static void scmn(const uint8_t *q, int j, uint8_t *s, uint8_t *m) {
    if (j < 4) { *s = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *s = (uint8_t)(((q[j - 4] & 192) >> 2) | (q[j + 4] & 15));
        *m = (uint8_t)(((q[j] & 192) >> 2) | ((q[j + 4] & 240) >> 4));
    }
}

/* ---- the kernel: 16 rows x up to 8 tokens, main term only ---- */
static void gemm16x8(const uint8_t *grp, uint64_t blocks, const q8b *y,
                     uint64_t ystride, uint64_t nt, float out[ROWS][NTOK]) {
    const __m512i m0F = _mm512_set1_epi8(0x0F);
    const uint64_t lim = nt < 8u ? nt : 8u;
    __m512 f[8];
    for (int t = 0; t < 8; t++) f[t] = _mm512_setzero_ps();

    for (uint64_t b = 0; b < blocks; b++) {
        const blk16 *xp = (const blk16 *)(grp + b * X16_BYTES);
        const __m512 dv = _mm512_loadu_ps(xp->d);
        for (int jp = 0; jp < 4; jp++) {
            const int jlo = 2 * jp;
            __m512i lo[8], hi[8];
            for (int t = 0; t < 8; t++) lo[t] = hi[t] = _mm512_setzero_si512();
            for (int c = 0; c < 8; c++) {
                const __m512i raw = _mm512_loadu_si512(
                        (const void *)(xp->qs + (jp * 8 + c) * 64));
                const __m512i ql = _mm512_and_si512(raw, m0F);
                const __m512i qh = _mm512_and_si512(_mm512_srli_epi16(raw, 4), m0F);
                for (uint64_t t = 0; t < lim; t++) {
                    const q8b *yt = &y[t * ystride + b];
                    const int32_t al = *(const int32_t *)(yt->qs + jlo * 32 + c * 4);
                    const int32_t ah = *(const int32_t *)(yt->qs + (jlo+1)*32 + c*4);
                    lo[t] = _mm512_dpbusd_epi32(lo[t], ql, _mm512_set1_epi32(al));
                    hi[t] = _mm512_dpbusd_epi32(hi[t], qh, _mm512_set1_epi32(ah));
                }
            }
            /* sc is stored as bytes: sixteen of them widen to sixteen float
             * lanes in two instructions. Loading them as float read garbage. */
            const __m512 bl = _mm512_mul_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                            _mm_loadu_si128((const __m128i *)xp->sc[jlo]))), dv);
            const __m512 bh = _mm512_mul_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                            _mm_loadu_si128((const __m128i *)xp->sc[jlo + 1]))), dv);
            for (uint64_t t = 0; t < lim; t++) {
                const float yd = y[t * ystride + b].d;
                const __m512 k = _mm512_set1_ps(yd);
                __m512 v = _mm512_mul_ps(_mm512_cvtepi32_ps(lo[t]),
                                         _mm512_mul_ps(bl, k));
                v = _mm512_fmadd_ps(_mm512_cvtepi32_ps(hi[t]),
                                    _mm512_mul_ps(bh, k), v);
                f[t] = _mm512_add_ps(f[t], v);
            }
        }
    }
    for (uint64_t t = 0; t < lim; t++) {
        float lane[ROWS];
        _mm512_storeu_ps(lane, f[t]);
        for (int r = 0; r < ROWS; r++) out[r][t] = lane[r];
    }
}

/* ---- scalar reference over the same layout ---- */
static void ref_main(const uint8_t *grp, uint64_t blocks, const q8b *y,
                     uint64_t ystride, uint64_t nt, float out[ROWS][NTOK]) {
    const uint64_t lim = nt < 8u ? nt : 8u;
    for (int r = 0; r < ROWS; r++)
        for (uint64_t t = 0; t < lim; t++) out[r][t] = 0.f;
    for (uint64_t b = 0; b < blocks; b++) {
        const blk16 *xp = (const blk16 *)(grp + b * X16_BYTES);
        for (uint64_t t = 0; t < lim; t++) {
            const q8b *yt = &y[t * ystride + b];
            for (int j = 0; j < 8; j++) {
                for (int r = 0; r < ROWS; r++) {
                    double acc = 0.0;
                    for (int p = 0; p < 32; p++) {
                        /* row r's four bytes sit at r*4 inside each 64-byte
                         * chunk, which is what puts row r in dpbusd lane r */
                        const uint8_t byte = xp->qs[((j / 2) * 8 + p / 4) * 64 +
                                                     r * 4 + (p % 4)];
                        const int v = (j & 1) ? (byte >> 4) : (byte & 0x0F);
                        acc += (double)v * yt->qs[j * 32 + p];
                    }
                    out[r][t] += (float)(acc * xp->sc[j][r] * xp->d[r] * yt->d);
                }
            }
        }
    }
}

/* build the layout from a row-major q4_K stand-in */
static void repack(const uint8_t *rows, uint8_t *dst) {
    for (uint64_t b = 0; b < BLOCKS; b++) {
        blk16 *o = (blk16 *)(dst + b * X16_BYTES);
        for (int r = 0; r < ROWS; r++) {
            const uint8_t *rp = rows + (uint64_t)r * ROW_BYTES + b * 144;
            o->d[r] = (float)(0.01 + 0.001 * r);
            o->dmin[r] = 0.005f;
            for (int j = 0; j < 8; j++) {
                uint8_t s, m;
                scmn(rp + 4, j, &s, &m);
                o->sc[j][r] = s;
                o->mn[j][r] = m;
            }
            /* four-byte interleave over the sixteen rows, written by row */
            for (int g = 0; g < 32; g++)
                memcpy(o->qs + g * 64 + r * 4, rp + 16 + g * 4, 4);
        }
    }
}

int main(void) {
    srand(777);
    const size_t n = (size_t)ROWS * ROW_BYTES;
    uint8_t *rows = malloc(n), *packed = malloc(BLOCKS * X16_BYTES);
    q8b y[NTOK * BLOCKS];
    for (size_t i = 0; i < n; i++) rows[i] = (uint8_t)((i * 131u + 29u) & 0xFF);
    for (uint64_t t = 0; t < NTOK; t++)
        for (uint64_t b = 0; b < BLOCKS; b++) {
            q8b *q = &y[t * BLOCKS + b];
            q->d = 0.002f + 0.0001f * (float)(t * BLOCKS + b);
            for (int i = 0; i < QK_K; i++) q->qs[i] = (int8_t)((rand() % 255) - 127);
            for (int g = 0; g < 16; g++) {
                int s = 0;
                for (int i = 0; i < 16; i++) s += q->qs[g * 16 + i];
                q->bsums[g] = (int16_t)s;
            }
        }
    repack(rows, packed);

    static float got[ROWS][NTOK], want[ROWS][NTOK];
    memset(got, 0, sizeof got);
    memset(want, 0, sizeof want);
    for (uint64_t nt = 1; nt <= NTOK; nt *= 2) {
        memset(got, 0, sizeof got);
        memset(want, 0, sizeof want);
        gemm16x8(packed, BLOCKS, y, BLOCKS, nt, got);
        ref_main(packed, BLOCKS, y, BLOCKS, nt, want);
        double worst = 0.0;
        for (int r = 0; r < ROWS; r++)
            for (uint64_t t = 0; t < nt; t++) {
                const double e = fabs((double)got[r][t] - (double)want[r][t]);
                const double rel = e / (fabs((double)want[r][t]) + 1e-6);
                if (rel > worst) worst = rel;
            }
        printf("tokens=%llu  worst relative error = %.3e  %s\n",
               (unsigned long long)nt, worst,
               worst < 1e-4 ? "OK" : "WRONG");
    }
    /* throughput: how many multiply-accumulates per second does this shape get */
    {
        static float sink[ROWS][NTOK];
        const int iters = 200000;
        const uint64_t t0 = __rdtsc();
        for (int i = 0; i < iters; i++) {
            /* mutate an input so the compiler cannot hoist the whole call out
             * of the loop - the first version measured 675 GMAC/s, which is
             * more multiply-accumulates per cycle than dpbusd can physically
             * issue, and that is what it was doing */
            y[0].qs[i & 255] = (int8_t)(i & 127);
            gemm16x8(packed, BLOCKS, y, BLOCKS, NTOK, sink);
        }
        const uint64_t t1 = __rdtsc();
        const double sec = (double)(t1 - t0) / 4.0e9;   /* ~4 GHz nominal */
        /* per call: 16 rows x 8 tokens x 256 values x 4 blocks */
        const double macs = (double)ROWS * NTOK * QK_K * BLOCKS * iters;
        printf("\n%.1f GMAC/s   (%.3f ns per row-token-block-pair, 1 thread)\n",
               macs / sec / 1e9, sec * 1e9 / (double)(iters * 64 * 4));
        printf("for reference the engine's 2x4 kernel measures 12.8 GMAC/s\n");
        printf("(sink %g)\n", (double)sink[0][0]);
    }
    free(rows); free(packed);
    return 0;
}
