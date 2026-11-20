/* The same kernel, threaded, on a working set that fits L1.
 *
 * uProf's memory view settles the cache question: the engine's Q4_K kernel hits
 * L1 on ninety-eight percent of its loads and L2 on two, with DRAM under a tenth
 * of a percent. So it is not stalled on memory at all. Yet it runs at 0.56
 * dpbusd per cycle against the 2.78 the same code reaches in a single-threaded
 * microbenchmark over the same forty-eight kilobyte working set.
 *
 * The one difference left is concurrency. This runs that kernel on a working set
 * small enough to stay in L1, at increasing thread counts, so the per-core rate
 * can be compared against the single-threaded 2.78. If it collapses the same way,
 * the engine is already at the machine's limit and twenty-eight threads on
 * sixteen cores is simply what this costs.
 */
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define BLOCKS   20
#define BLOCKREC 2432
#define REC      (BLOCKS * BLOCKREC)
#define MAXT     64

typedef struct { int8_t qs[256]; float d; int16_t bsums[16]; } q8b;

static double now_ms(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e3 / (double)f.QuadPart;
}

static void kern(const uint8_t *grp, const q8b *y, uint32_t blocks)
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
            for (int t = 0; t < 8; t++) { lo[t] = _mm512_setzero_si512();
                                          hi[t] = _mm512_setzero_si512(); }
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

/* No barrier. Each thread runs its share the moment it is created and the
 * measurement is creation-to-join. The earlier version handshook through a spin
 * barrier and deadlocked above one thread; with tens of milliseconds of work per
 * thread the creation cost is noise, and there is nothing to get wrong. */
static long g_reps;
static const uint8_t *g_buf;
static q8b *g_y;

static DWORD WINAPI worker(LPVOID arg)
{
    (void)arg;
    const uint8_t *restrict buf = g_buf;
    const q8b *restrict y = g_y;
    const long reps = g_reps;
    for (long r = 0; r < reps; r++) kern(buf, y, BLOCKS);
    return 0;
}

int main(void)
{
    uint8_t *buf = VirtualAlloc(NULL, REC, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    q8b *y = calloc(8, sizeof(q8b));
    if (!buf || !y) return 1;
    for (int t = 0; t < 8; t++) { for (int i = 0; i < 256; i++) y[t].qs[i] = (int8_t)(i * 7);
                                  y[t].d = 0.01f; }
    for (uint32_t i = 0; i < REC; i++) buf[i] = (uint8_t)(i * 37u);
    g_buf = buf; g_y = y;

    printf("threads   dpbusd/cycle/core   vs 1 thread   total G dpbusd/s\n");
    const int ts[] = { 1, 2, 4, 8, 16, 24, 28, 32 };
    double base = 0.0;
    for (unsigned ti = 0; ti < sizeof(ts)/sizeof(ts[0]); ti++) {
        const int T = ts[ti];
        HANDLE th[MAXT];
        g_reps = (long)(200000.0 / ((double)T));
        if (g_reps < 200) g_reps = 200;
        double best = 1e18;
        for (int round = 0; round < 3; round++) {
            const double t0 = now_ms();
            for (int i = 0; i < T; i++)
                th[i] = CreateThread(NULL, 0, worker, (LPVOID)(intptr_t)i, 0, NULL);
            WaitForMultipleObjects((DWORD)T, th, TRUE, INFINITE);
            const double dt = now_ms() - t0;
            for (int i = 0; i < T; i++) CloseHandle(th[i]);
            if (dt < best) best = dt;
        }

        /* Four sub-blocks, eight chunks, eight tokens, two nibbles: five
         * hundred and twelve per block. The count used to carry two hundred
         * and fifty-six and reported half the true rate. */
        const double dpbusd = (double)T * (double)g_reps * BLOCKS * 512.0;
        const double cyc = best * 1e-3 * 5.4e9 * (double)T;
        const double rate = dpbusd / cyc;
        if (T == 1) base = rate;
        printf("%7d   %17.3f   %9.2fx   %14.1f\n",
               T, rate, base > 0 ? rate / base : 1.0, dpbusd / (best * 1e-3) / 1e9);
        fflush(stdout);
    }
    return 0;
}
