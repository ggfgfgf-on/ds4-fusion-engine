/* 16-row Q4_K kernel, multi-block, against a scalar reference.
 *
 * The previous test used a single super-block and ystride == 1, so it never
 * exercised the block accumulator, the per-block d/dmin, or the y[t*ystride+b]
 * indexing - and passed while the engine, which has 24 blocks and
 * ystride == blocks, produced max|diff| 29. This mirrors the engine: several
 * blocks, four tokens, ystride == blocks.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#define QK_K        256
#define Q4K_ROW     144u
#define Q4K_X16    2304u
#define ROWS         16
#define BLOCKS        4
#define NTOK          4

typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; } row_q4k;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[16]; } block_q8_K;

static float f16(uint16_t h) {
    const uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF;
    if (!e) return ldexpf(m / 1024.0f, -14);
    return ldexpf(1.0f + m / 1024.0f, (int)e - 15);
}

static void sc_min(const uint8_t *q, int j, uint8_t *s, uint8_t *m) {
    if (j < 4) { *s = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *s = (uint8_t)(((q[j - 4] & 192) >> 2) | (q[j + 4] & 15));
        *m = (uint8_t)(((q[j] & 192) >> 2) | ((q[j + 4] & 240) >> 4));
    }
}

static void pack6(const uint8_t *v, uint8_t *out) {
    for (unsigned k = 0; k < 4; k++) {
        const unsigned a = v[4*k+0] & 63u, b = v[4*k+1] & 63u;
        const unsigned c = v[4*k+2] & 63u, d = v[4*k+3] & 63u;
        out[3*k+0] = (uint8_t)(a | (b << 6));
        out[3*k+1] = (uint8_t)((b >> 2) | (c << 4));
        out[3*k+2] = (uint8_t)((c >> 4) | (d << 2));
    }
}
static void unpack6(const uint8_t *in, uint8_t *v) {
    for (unsigned k = 0; k < 4; k++) {
        const unsigned b0 = in[3*k+0], b1 = in[3*k+1], b2 = in[3*k+2];
        v[4*k+0] = (uint8_t)(b0 & 63u);
        v[4*k+1] = (uint8_t)(((b0 >> 6) & 3u) | ((b1 & 15u) << 2));
        v[4*k+2] = (uint8_t)(((b1 >> 4) & 15u) | ((b2 & 3u) << 4));
        v[4*k+3] = (uint8_t)((b2 >> 2) & 63u);
    }
}

static void repack16(const uint8_t *src, uint8_t *dst, uint64_t row_bytes) {
    const uint64_t blocks = row_bytes / Q4K_ROW;
    for (uint64_t b = 0; b < blocks; b++) {
        uint8_t *o = dst + b * Q4K_X16;
        const uint8_t *in[ROWS];
        for (unsigned r = 0; r < ROWS; r++)
            in[r] = src + (uint64_t)r * row_bytes + b * Q4K_ROW;
        for (unsigned r = 0; r < ROWS; r++) {
            memcpy(o + r * 2, in[r], 2);
            memcpy(o + 32 + r * 2, in[r] + 2, 2);
        }
        uint8_t sv[ROWS], mv[ROWS];
        for (int j = 0; j < 8; j++) {
            for (unsigned r = 0; r < ROWS; r++) sc_min(in[r] + 4, j, &sv[r], &mv[r]);
            pack6(sv, o + 64 + j * 24);
            pack6(mv, o + 64 + j * 24 + 12);
        }
        for (int i = 0; i < 512; i++) {
            const unsigned r = (unsigned)i % ROWS;
            const int off = (i / (int)ROWS) * 4;
            memcpy(o + 256 + (uint64_t)i * 4, in[r] + 16 + off, 4);
        }
    }
}

/* the engine kernel, multi-block */
static void gemm16(const uint8_t *grp, const block_q8_K *y, uint64_t ystride,
                   uint64_t blocks, uint64_t nt, float out[ROWS][NTOK]) {
    const __m512i m0F = _mm512_set1_epi8(0x0F);
    const uint64_t lim = nt < 4u ? nt : 4u;
    __m512 f0 = _mm512_setzero_ps(), f1 = f0, f2 = f0, f3 = f0;

    for (uint64_t b = 0; b < blocks; b++) {
        const uint8_t *xp = grp + b * Q4K_X16;
        float dv[ROWS];
        for (unsigned r = 0; r < ROWS; r++)
            dv[r] = f16(*(const uint16_t *)(xp + r * 2));
        const __m512 dvec = _mm512_set_ps(
                dv[15],dv[14],dv[13],dv[12],dv[11],dv[10],dv[9],dv[8],
                dv[7],dv[6],dv[5],dv[4],dv[3],dv[2],dv[1],dv[0]);
        for (int j = 0; j < 8; j++) {
            const int sh = (j & 1) ? 4 : 0;
            __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
            for (int c = 0; c < 8; c++) {
                const __m512i raw = _mm512_loadu_si512(
                        (const void *)(xp + 256 + (((j / 2) * 8 + c) * 64)));
                const __m512i q = sh
                        ? _mm512_and_si512(_mm512_srli_epi16(raw, 4), m0F)
                        : _mm512_and_si512(raw, m0F);
                for (uint64_t t = 0; t < lim; t++) {
                    int32_t four;
                    memcpy(&four, y[t * ystride + b].qs + j * 32 + c * 4, 4);
                    const __m512i r = _mm512_dpbusd_epi32(
                            _mm512_setzero_si512(), q, _mm512_set1_epi32(four));
                    if (t == 0) a0 = _mm512_add_epi32(a0, r);
                    else if (t == 1) a1 = _mm512_add_epi32(a1, r);
                    else if (t == 2) a2 = _mm512_add_epi32(a2, r);
                    else a3 = _mm512_add_epi32(a3, r);
                }
            }
            uint8_t sv[ROWS];
            unpack6(xp + 64 + j * 24, sv);
            const __m512 scv = _mm512_set_ps(
                    (float)sv[15],(float)sv[14],(float)sv[13],(float)sv[12],
                    (float)sv[11],(float)sv[10],(float)sv[9],(float)sv[8],
                    (float)sv[7],(float)sv[6],(float)sv[5],(float)sv[4],
                    (float)sv[3],(float)sv[2],(float)sv[1],(float)sv[0]);
            const __m512 base = _mm512_mul_ps(scv, dvec);
            for (uint64_t t = 0; t < lim; t++) {
                const __m512i pv = (t == 0) ? a0 : (t == 1) ? a1 : (t == 2) ? a2 : a3;
                const __m512 k = _mm512_mul_ps(base,
                        _mm512_set1_ps(y[t * ystride + b].d));
                const __m512 v = _mm512_mul_ps(_mm512_cvtepi32_ps(pv), k);
                if (t == 0) f0 = _mm512_add_ps(f0, v);
                else if (t == 1) f1 = _mm512_add_ps(f1, v);
                else if (t == 2) f2 = _mm512_add_ps(f2, v);
                else f3 = _mm512_add_ps(f3, v);
            }
        }
    }
    float minv[ROWS][NTOK];
    memset(minv, 0, sizeof minv);
    for (uint64_t b = 0; b < blocks; b++) {
        const uint8_t *xp = grp + b * Q4K_X16;
        float mn[8][ROWS];
        for (int j = 0; j < 8; j++) {
            uint8_t mvb[ROWS];
            unpack6(xp + 64 + j * 24 + 12, mvb);
            for (unsigned r = 0; r < ROWS; r++) mn[j][r] = (float)mvb[r];
        }
        for (unsigned r = 0; r < ROWS; r++) {
            const float dm = f16(*(const uint16_t *)(xp + 32 + r * 2));
            for (uint64_t t = 0; t < lim; t++) {
                const block_q8_K *yt = y + t * ystride + b;
                float acc = 0.f;
                for (int j = 0; j < 8; j++)
                    acc += mn[j][r] *
                           (float)((int32_t)yt->bsums[2*j] + yt->bsums[2*j+1]);
                minv[r][t] -= yt->d * dm * acc;
            }
        }
    }
    for (uint64_t t = 0; t < lim; t++) {
        __m512 fv = (t == 0) ? f0 : (t == 1) ? f1 : (t == 2) ? f2 : f3;
        float lane[ROWS];
        _mm512_storeu_ps(lane, fv);
        for (unsigned r = 0; r < ROWS; r++) out[r][t] = lane[r] + minv[r][t];
    }
}

/* scalar reference: all blocks, all tokens */
static void ref_main(const uint8_t *rows, uint64_t row_bytes,
                     const block_q8_K *y, uint64_t ystride,
                     uint64_t blocks, uint64_t nt, float out[ROWS][NTOK]) {
    for (unsigned r = 0; r < ROWS; r++) {
        const row_q4k *x = (const row_q4k *)(rows + (uint64_t)r * row_bytes);
        for (uint64_t t = 0; t < nt; t++) {
            double acc = 0.0;
            for (uint64_t b = 0; b < blocks; b++) {
                const float d = f16(x[b].d), yd = y[t * ystride + b].d;
                double pos = 0.0, neg = 0.0;
                for (int j = 0; j < 8; j++) {
                    uint8_t s, m;
                    sc_min(x[b].scales, j, &s, &m);
                    for (int p = 0; p < 32; p++) {
                        const uint8_t byte = x[b].qs[(j >> 1) * 32 + p];
                        const int v = (j & 1) ? (byte >> 4) : (byte & 0x0F);
                        pos += (double)s * v * y[t * ystride + b].qs[j * 32 + p];
                        neg += (double)m * y[t * ystride + b].qs[j * 32 + p];
                    }
                }
                acc += d * yd * pos;
                acc -= f16(x[b].dmin) * yd * neg;
            }
            out[r][t] = (float)acc;
        }
    }
}

int main(void) {
    srand(90210);
    const uint64_t row_bytes = BLOCKS * Q4K_ROW;
    const size_t n = (size_t)ROWS * row_bytes;
    uint8_t *rows = malloc(n), *packed = malloc(BLOCKS * Q4K_X16);
    block_q8_K y[NTOK * BLOCKS];

    for (size_t i = 0; i < n; i++) rows[i] = (uint8_t)((i * 167u + 91u) & 0xFF);
    for (unsigned r = 0; r < ROWS; r++)
        for (uint64_t b = 0; b < BLOCKS; b++) {
            row_q4k *x = (row_q4k *)(rows + (uint64_t)r * row_bytes) + b;
            x->d = (uint16_t)(0x2000 + r * 0x40 + b * 0x10);
            x->dmin = (uint16_t)(0x1C00 + r * 0x30);
        }
    for (uint64_t t = 0; t < NTOK; t++)
        for (uint64_t b = 0; b < BLOCKS; b++) {
            block_q8_K *q = &y[t * BLOCKS + b];
            q->d = 0.003f + 0.0007f * (float)(t * BLOCKS + b);
            for (int i = 0; i < QK_K; i++) q->qs[i] = (int8_t)((rand() % 255) - 127);
            for (int g = 0; g < 16; g++) {
                int sum = 0;
                for (int i = 0; i < 16; i++) sum += q->qs[g * 16 + i];
                q->bsums[g] = (int16_t)sum;
            }
        }
    repack16(rows, packed, row_bytes);

    static float got[ROWS][NTOK], want[ROWS][NTOK];
    memset(got, 0, sizeof got);
    memset(want, 0, sizeof want);
    gemm16(packed, y, BLOCKS, BLOCKS, NTOK, got);
    ref_main(rows, row_bytes, y, BLOCKS, BLOCKS, NTOK, want);

    double worst = 0.0;
    int wr = -1, wt = -1;
    for (unsigned r = 0; r < ROWS; r++)
        for (uint64_t t = 0; t < NTOK; t++) {
            const double e = fabs((double)got[r][t] - (double)want[r][t]);
            const double rel = e / (fabs((double)want[r][t]) + 1e-6);
            if (rel > worst) { worst = rel; wr = (int)r; wt = (int)t; }
        }
    printf("worst relative error = %.3e (row %d token %d)\n", worst, wr, wt);
    printf("got  [0][0..3] = %.4f %.4f %.4f %.4f\n",
           got[0][0], got[0][1], got[0][2], got[0][3]);
    printf("want [0][0..3] = %.4f %.4f %.4f %.4f\n",
           want[0][0], want[0][1], want[0][2], want[0][3]);
    printf("got  [15][0..3] = %.4f %.4f %.4f %.4f\n",
           got[15][0], got[15][1], got[15][2], got[15][3]);
    printf("want [15][0..3] = %.4f %.4f %.4f %.4f\n",
           want[15][0], want[15][1], want[15][2], want[15][3]);
    printf("%s\n", worst < 1e-4 ? "MULTI-BLOCK KERNEL CORRECT"
                                : "MULTI-BLOCK KERNEL WRONG");
    free(rows); free(packed);
    return worst < 1e-4 ? 0 : 1;
}
