/* Packed Q6_K against vpdpbusd, with the unpacking done correctly.
 *
 * An earlier version of this file fed a register built from 16-bit lanes into
 * vpdpbusd, which reads bytes: the values were interleaved wrongly and the
 * speedup it reported was meaningless. The unpacking below follows
 * qwen35_q6k_at exactly, for one sixty-four value group at a time:
 *
 *   positions  0..31   (ql[ 0..31] & 0x0F) | (((qh[ 0..31] >> 0) & 3) << 4)
 *   positions 32..63   (ql[32..63] & 0x0F) | (((qh[ 0..31] >> 2) & 3) << 4)
 *
 * The two halves differ only in which two bits of qh they take, so the byte
 * vector holds qh duplicated across both halves and a masked blend picks the
 * shift for the upper half.
 *
 * The comparison is against the kernel as it stands: six-bit values widened to
 * int16 and multiplied with madd_epi16, which is what the 2x4 path does.
 */
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <windows.h>

static double now_s(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

#define BLOCKS 8                 /* eight blocks, 2048 values a row */
#define ITER   600000

static uint8_t g_ql[BLOCKS][128] __attribute__((aligned(64)));
static uint8_t g_qh[BLOCKS][64]  __attribute__((aligned(64)));
static int8_t  g_a[BLOCKS][256]  __attribute__((aligned(64)));
static int8_t  g_sc[BLOCKS][16]  __attribute__((aligned(64)));
static int32_t g_asum[BLOCKS][16] __attribute__((aligned(64)));

/* ---- current shape: unpack once to int16, four tokens --------------- */
static float run_madd(void)
{
    float total = 0.0f;
    for (int b = 0; b < BLOCKS; b++) {
        int32_t acc[4][16];
        for (int t = 0; t < 4; t++)
            for (int g = 0; g < 16; g++) acc[t][g] = 0;
        for (int g = 0; g < 16; g++) {
            /* unpack the sixteen values of this group once */
            __m256i wv;
            {
                uint8_t v[16];
                const int n128 = (g >= 8) ? 1 : 0;
                for (int i = 0; i < 16; i++) {
                    const int p = g * 16 + i;
                    const int r = p - n128 * 128;
                    const uint8_t *ql = g_ql[b] + (n128 ? 64 : 0);
                    const uint8_t *qh = g_qh[b] + (n128 ? 32 : 0);
                    int l;
                    if (r < 32)      { l = r;      v[i] = (uint8_t)((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)); }
                    else if (r < 64) { l = r - 32; v[i] = (uint8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)); }
                    else if (r < 96) { l = r - 64; v[i] = (uint8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)); }
                    else             { l = r - 96; v[i] = (uint8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)); }
                }
                wv = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)v));
            }
            for (int t = 0; t < 4; t++) {
                const __m256i av = _mm256_cvtepi8_epi16(
                        _mm_loadu_si128((const __m128i *)(g_a[b] + g * 16)));
                __m256i a = _mm256_madd_epi16(
                        _mm256_unpacklo_epi16(wv, _mm256_setzero_si256()), av);
                a = _mm256_add_epi32(a, _mm256_madd_epi16(
                        _mm256_unpackhi_epi16(wv, _mm256_setzero_si256()), av));
                int32_t lane[8];
                _mm256_storeu_si256((__m256i *)lane, a);
                for (int i = 0; i < 8; i++) acc[t][g] += lane[i];
            }
        }
        for (int t = 0; t < 4; t++)
            for (int g = 0; g < 16; g++)
                total += (float)acc[t][g] * (float)g_sc[b][g];
    }
    return total;
}

/* ---- proposed: unpack once to bytes, four tokens -------------------- */
static float run_dpbusd(void)
{
    float total = 0.0f;
    const __m512i m0F = _mm512_set1_epi8(0x0F);
    const __m512i m03 = _mm512_set1_epi8(0x03);
    const __mmask64 upper = 0xFFFFFFFF00000000ull;

    for (int b = 0; b < BLOCKS; b++) {
        int32_t acc[4][16];
        for (int t = 0; t < 4; t++)
            for (int g = 0; g < 16; g++) acc[t][g] = 0;
        for (int half = 0; half < 2; half++) {
            for (int q = 0; q < 4; q++) {
                const int base = q * 64;
                const uint8_t *qlp = g_ql[b] + (half ? 64 : 0);
                const uint8_t *qhp = g_qh[b] + (half ? 32 : 0);
                const int shift = ((base >> 5) & 1) * 2;
                const __m512i ql = _mm512_loadu_si512((const void *)qlp);
                const __m256i qh = _mm256_loadu_si256((const __m256i *)qhp);
                __m512i qd = _mm512_castsi256_si512(qh);
                qd = _mm512_inserti64x4(qd, qh, 1);
                const __m512i lo = _mm512_and_si512(ql, m0F);
                const __m512i hi_a = _mm512_and_si512(qd, m03);
                const __m512i hi_b = _mm512_and_si512(_mm512_srli_epi16(qd, 2), m03);
                const __m512i hi = (shift == 0)
                        ? _mm512_mask_blend_epi8(upper, hi_a, hi_b)
                        : _mm512_mask_blend_epi8(upper, hi_b, hi_a);
                const __m512i v = _mm512_or_si512(lo, _mm512_slli_epi16(hi, 4));
                for (int t = 0; t < 4; t++) {
                    const __m512i r = _mm512_dpbusd_epi32(
                            _mm512_setzero_si512(), v,
                            _mm512_loadu_si512((const void *)(g_a[b] + base)));
                    int32_t lane[16];
                    _mm512_storeu_si512(lane, r);
                    for (int g = 0; g < 4; g++) {
                        const int gi = q * 4 + g;
                        acc[t][gi] += lane[g * 4] + lane[g * 4 + 1]
                                    + lane[g * 4 + 2] + lane[g * 4 + 3];
                    }
                }
            }
        }
        for (int t = 0; t < 4; t++)
            for (int g = 0; g < 16; g++)
                total += (float)(acc[t][g] - 32 * g_asum[b][g]) * (float)g_sc[b][g];
    }
    return total;
}


/* Unpacking with the multiply removed: the gap to run_madd is the MAC share. */
static float run_unpack_only(void)
{
    float total = 0.0f;
    for (int b = 0; b < BLOCKS; b++) {
        for (int g = 0; g < 16; g++) {
            uint8_t v[16];
            const int n128 = (g >= 8) ? 1 : 0;
            for (int i = 0; i < 16; i++) {
                const int p = g * 16 + i;
                const int r = p - n128 * 128;
                const uint8_t *ql = g_ql[b] + (n128 ? 64 : 0);
                const uint8_t *qh = g_qh[b] + (n128 ? 32 : 0);
                int l;
                if (r < 32)      { l = r;      v[i] = (uint8_t)((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)); }
                else if (r < 64) { l = r - 32; v[i] = (uint8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)); }
                else if (r < 96) { l = r - 64; v[i] = (uint8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)); }
                else             { l = r - 96; v[i] = (uint8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)); }
            }
            for (int i = 0; i < 16; i++) total += (float)v[i] * (float)g_sc[b][g];
        }
    }
    return total;
}


/* Vectorised unpacking with the dpbusd removed, so the split inside the shape
 * the engine actually uses can be read directly. */
static float run_unpack_vec(void)
{
    float total = 0.0f;
    const __m512i m0F = _mm512_set1_epi8(0x0F);
    const __m512i m03 = _mm512_set1_epi8(0x03);
    const __mmask64 upper = 0xFFFFFFFF00000000ull;
    for (int b = 0; b < BLOCKS; b++) {
        for (int half = 0; half < 2; half++) {
            for (int q = 0; q < 4; q++) {
                const int base = q * 64;
                const uint8_t *qlp = g_ql[b] + (half ? 64 : 0);
                const uint8_t *qhp = g_qh[b] + (half ? 32 : 0);
                const int shift = ((base >> 5) & 1) * 2;
                const __m512i ql = _mm512_loadu_si512((const void *)qlp);
                const __m256i qh = _mm256_loadu_si256((const __m256i *)qhp);
                __m512i qd = _mm512_castsi256_si512(qh);
                qd = _mm512_inserti64x4(qd, qh, 1);
                const __m512i lo = _mm512_and_si512(ql, m0F);
                const __m512i ha = _mm512_and_si512(qd, m03);
                const __m512i hb = _mm512_and_si512(_mm512_srli_epi16(qd, 2), m03);
                const __m512i hi = (shift == 0)
                        ? _mm512_mask_blend_epi8(upper, ha, hb)
                        : _mm512_mask_blend_epi8(upper, hb, ha);
                const __m512i v = _mm512_or_si512(lo, _mm512_slli_epi16(hi, 4));
                int8_t lane[64];
                _mm512_storeu_si512(lane, v);
                for (int i = 0; i < 64; i++) total += (float)lane[i];
            }
        }
    }
    return total;
}

int main(void)
{
    for (int b = 0; b < BLOCKS; b++) {
        for (int i = 0; i < 128; i++) g_ql[b][i] = (uint8_t)(i * 5 + b);
        for (int i = 0; i < 64;  i++) g_qh[b][i] = (uint8_t)(i * 3 + b);
        for (int i = 0; i < 256; i++) g_a[b][i] = (int8_t)((i * 13) % 127 - 63);
        for (int i = 0; i < 16;  i++) g_sc[b][i] = (int8_t)((i % 31) + 1);
        for (int g = 0; g < 16; g++) {
            int32_t t = 0;
            for (int i = 0; i < 16; i++) t += g_a[b][g * 16 + i];
            g_asum[b][g] = t;
        }
    }

    volatile float sink = 0.0f;
    double t_m = 0.0, t_d = 0.0;
    for (int round = 0; round < 3; round++) {
        double t0 = now_s();
        for (int i = 0; i < ITER; i++) sink += run_madd();
        double t1 = now_s();
        t_m = t1 - t0;
        t0 = now_s();
        for (int i = 0; i < ITER; i++) sink += run_dpbusd();
        t1 = now_s();
        t_d = t1 - t0;
    }

    const double macs = (double)BLOCKS * 256.0 * 4.0 * ITER;
    printf("shape                     ns/pass   G MAC/s   vs current\n");
    printf("int16 widen + madd        %8.1f   %7.2f   1.00x\n",
           t_m / ITER * 1e9, macs / t_m / 1e9);
    printf("byte unpack + vpdpbusd    %8.1f   %7.2f   %.2fx\n",
           t_d / ITER * 1e9, macs / t_d / 1e9, t_m / t_d);
    double t4 = 0.0;
    for (int round = 0; round < 3; round++) {
        double t0 = now_s();
        for (int i = 0; i < ITER; i++) sink += run_unpack_only();
        double t1 = now_s();
        t4 = t1 - t0;
    }
    printf("unpack only (no MAC)      %8.1f                    %.0f%% of madd path\n",
           t4 / ITER * 1e9, 100.0 * t4 / t_m);
    double t5 = 0.0;
    for (int round = 0; round < 3; round++) {
        double t0 = now_s();
        for (int i = 0; i < ITER; i++) sink += run_unpack_vec();
        double t1 = now_s();
        t5 = t1 - t0;
    }
    printf("vector unpack only        %8.1f                    %.0f%% of dpbusd path\n",
           t5 / ITER * 1e9, 100.0 * t5 / t_d);
    printf("(sink %f)\n", (double)sink);
    return 0;
}
