/* Does shrinking the per-pass working set below L1 buy what split-K would cost?
 *
 * The wide Q4_K kernel runs at 0.56 dpbusd per cycle in the engine. With a
 * 48.6 KB row group against a 48 KB L1, every load there is served by L2 at
 * about 22.8 cycles. split-K would cut the working set to roughly 24 KB and
 * could in principle reach the 2.01 the instruction itself sustains - but it
 * costs a partial-sum buffer, an extra pass, a new kernel variant, and the one
 * measurement that already touches this idea (four tokens per pass) came back
 * three percent worse.
 *
 * So measure the curve first. This is the same kernel, same layout, same inner
 * loop, with one difference: the number of blocks per row group is a runtime
 * parameter, so the working set can be swept from 10 KB to 48.6 KB while the
 * weights stay hot by re-reading them. Whatever the cache hierarchy gives for a
 * 24 KB working set is the ceiling split-K could reach, before its own overhead.
 */
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define BLOCKREC 2432            /* 64 + 64 + 128 + 128 + 2048 */
typedef struct { int8_t qs[256]; float d; int16_t bsums[16]; } q8b;

static double now_ms(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e3 / (double)f.QuadPart;
}

/* The engine's lim == 8 path, with blocks runtime. */
static void kern(const uint8_t *grp, uint32_t blocks, const q8b *y)
{
    const __m512i m0F = _mm512_set1_epi8(0x0F);
    __m512 f[8];
    for (int t = 0; t < 8; t++) f[t] = _mm512_setzero_ps();

    for (uint32_t b = 0; b < blocks; b++) {
        const uint8_t *xp = grp + (uint64_t)b * BLOCKREC;
        const __m512 dv = _mm512_loadu_ps((const float *)xp);
        for (int jp = 0; jp < 4; jp++) {
            const int jlo = 2 * jp;
            __m512i lo[8], hi[8];
            for (int t = 0; t < 8; t++) {
                lo[t] = _mm512_setzero_si512();
                hi[t] = _mm512_setzero_si512();
            }
            for (int c = 0; c < 8; c++) {
                const __m512i raw = _mm512_loadu_si512(
                        (const void *)(xp + 384 + (uint64_t)(jp * 8 + c) * 64));
                const __m512i ql = _mm512_and_si512(raw, m0F);
                const __m512i qh = _mm512_and_si512(_mm512_srli_epi16(raw, 4), m0F);
                for (int t = 0; t < 8; t++) {
                    const int32_t al = *(const int32_t *)(y[t].qs + jlo * 32 + c * 4);
                    const int32_t ah = *(const int32_t *)(y[t].qs + (jlo + 1) * 32 + c * 4);
                    lo[t] = _mm512_dpbusd_epi32(lo[t], ql, _mm512_set1_epi32(al));
                    hi[t] = _mm512_dpbusd_epi32(hi[t], qh, _mm512_set1_epi32(ah));
                }
            }
            const __m512 bl2 = _mm512_mul_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                            _mm_loadu_si128((const __m128i *)(xp + 128 + jlo * 16)))), dv);
            const __m512 bh2 = _mm512_mul_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                            _mm_loadu_si128((const __m128i *)(xp + 128 + (jlo + 1) * 16)))), dv);
            for (int t = 0; t < 8; t++) {
                const __m512 yd = _mm512_set1_ps(y[t].d);
                __m512 v = _mm512_mul_ps(_mm512_cvtepi32_ps(lo[t]), _mm512_mul_ps(bl2, yd));
                v = _mm512_fmadd_ps(_mm512_cvtepi32_ps(hi[t]), _mm512_mul_ps(bh2, yd), v);
                f[t] = _mm512_add_ps(f[t], v);
            }
            f[0] = _mm512_add_ps(f[0], _mm512_cvtepi32_ps(
                    _mm512_add_epi32(lo[0], hi[0])));
        }
    }
    float acc = 0.0f, lane[16];
    for (int t = 0; t < 8; t++) {
        _mm512_storeu_ps(lane, f[t]);
        for (int r = 0; r < 16; r++) acc += lane[r];
    }
    if (acc == 1234.5678f) printf(" ");
}

int main(void)
{
    const uint32_t max_blocks = 20;
    uint8_t *buf = VirtualAlloc(NULL, (size_t)max_blocks * BLOCKREC,
                                MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    q8b *y = calloc(8, sizeof(q8b));
    if (!buf || !y) return 1;
    for (int t = 0; t < 8; t++) {
        for (int i = 0; i < 256; i++) y[t].qs[i] = (int8_t)(i * 7);
        y[t].d = 0.01f;
    }
    for (uint32_t i = 0; i < max_blocks * BLOCKREC; i++) buf[i] = (uint8_t)(i * 37u);

    printf("blocks   working set KB   dpbusd/cycle\n");
    const uint32_t ks[] = { 4, 6, 8, 10, 12, 14, 16, 20 };
    for (unsigned ki = 0; ki < sizeof(ks) / sizeof(ks[0]); ki++) {
        const uint32_t blocks = ks[ki];
        const uint64_t bytes = (uint64_t)blocks * BLOCKREC;

        /* enough repetitions that the weights stay hot in whatever level holds them */
        const long reps = 40000;
        double best = 1e18;
        for (int round = 0; round < 3; round++) {
            const double t0 = now_ms();
            for (long r = 0; r < reps; r++) kern(buf, blocks, y);
            const double dt = now_ms() - t0;
            if (dt < best) best = dt;
        }
        /* Four sub-blocks, eight chunks, eight tokens, two nibbles. The count
         * used to carry sixteen where eight belongs and reported every rate
         * twice what it is. */
        const double dpbusd = (double)reps * (double)blocks * 4.0 * 8.0 * 8.0 * 2.0;
        const double cyc = best * 1e-3 * 5.4e9;
        printf("%6u   %14.1f   %12.3f\n", blocks, bytes / 1024.0, dpbusd / cyc);
        fflush(stdout);
    }
    return 0;
}
