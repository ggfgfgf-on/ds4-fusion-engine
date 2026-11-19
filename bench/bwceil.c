/* What the memory system actually delivers, and what the kernel gets from it.
 *
 * The question this answers: the engine's prefill runs at 14.3 GB/s, and a
 * practical ceiling for dual-channel DDR5-5600 was assumed to be 55 to 63. Is
 * that assumption right, and is the engine near it?
 *
 * Two workloads, same persistent pool, same buffer:
 *
 *   stream   pure sequential read, nothing but loads and adds. This is the
 *            memory system's own number.
 *   kernel   the engine's Q4_K wide kernel, fed from the same buffer.
 *
 * Written fresh rather than patched: the previous attempts deadlocked twice on
 * event-signalling mistakes (auto-reset wakes one waiter, manual-reset lets a
 * finished worker re-enter the same generation). A generation counter and an
 * arrival counter have neither failure mode.
 */
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define BLOCKS   20
#define BLOCKREC (uint64_t)(64 + 64 + 128 + 128 + 2048)   /* 2432 */
#define REC      (BLOCKS * BLOCKREC)                       /* 48640 */
#define ROWS     16
#define MAXT     64

typedef struct { int8_t qs[256]; float d; int16_t bsums[16]; } q8b;

/* ---- shared state, all of it read-only once the pool is up --------------- */
static volatile LONG     g_gen;       /* bumped by main, seen by workers */
static volatile LONG     g_arrived;   /* workers that finished this gen */
static volatile LONG     g_quit;
static int               g_T;
static uint64_t          g_from, g_to;
static int               g_mode;      /* 0 stream, 1 kernel */
static const uint8_t    *g_base;
static q8b              *g_y;
static volatile LONG64   g_sink;

static double now_ms(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e3 / (double)f.QuadPart;
}

/* ---- workload 1: sequential read ---------------------------------------- */
static void stream_read(const uint8_t *p, uint64_t bytes)
{
    __m512i a = _mm512_setzero_si512(), b = a, c = a, d = a;
    for (uint64_t o = 0; o + 256 <= bytes; o += 256) {
        a = _mm512_add_epi64(a, _mm512_loadu_si512((const void *)(p + o)));
        b = _mm512_add_epi64(b, _mm512_loadu_si512((const void *)(p + o + 64)));
        c = _mm512_add_epi64(c, _mm512_loadu_si512((const void *)(p + o + 128)));
        d = _mm512_add_epi64(d, _mm512_loadu_si512((const void *)(p + o + 192)));
    }
    a = _mm512_add_epi64(_mm512_add_epi64(a, b), _mm512_add_epi64(c, d));
    int64_t t[8];
    _mm512_storeu_si512(t, a);
    InterlockedAdd64(&g_sink, t[0] + t[1]);
}

/* ---- workload 2: the engine's kernel ------------------------------------ */
static void kern(const uint8_t *grp, const q8b *y)
{
    const __m512i m0F = _mm512_set1_epi8(0x0F);
    __m512 f[8];
    for (int t = 0; t < 8; t++) f[t] = _mm512_setzero_ps();
    for (uint64_t b = 0; b < BLOCKS; b++) {
        const uint8_t *xp = grp + b * BLOCKREC;
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
                        (const void *)(xp + 384 + (jp * 8 + c) * 64));
                const __m512i ql = _mm512_and_si512(raw, m0F);
                const __m512i qh = _mm512_and_si512(
                        _mm512_srli_epi16(raw, 4), m0F);
                for (int t = 0; t < 8; t++) {
                    const int32_t al = *(const int32_t *)
                            (y[t].qs + jlo * 32 + c * 4);
                    const int32_t ah = *(const int32_t *)
                            (y[t].qs + (jlo + 1) * 32 + c * 4);
                    lo[t] = _mm512_dpbusd_epi32(lo[t], ql,
                                                _mm512_set1_epi32(al));
                    hi[t] = _mm512_dpbusd_epi32(hi[t], qh,
                                                _mm512_set1_epi32(ah));
                }
            }
            const __m512 bl2 = _mm512_mul_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                            _mm_loadu_si128((const __m128i *)
                                    (xp + 128 + jlo * 16)))), dv);
            const __m512 bh2 = _mm512_mul_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                            _mm_loadu_si128((const __m128i *)
                                    (xp + 128 + (jlo + 1) * 16)))), dv);
            for (int t = 0; t < 8; t++) {
                const __m512 yd = _mm512_set1_ps(y[t].d);
                __m512 v = _mm512_mul_ps(_mm512_cvtepi32_ps(lo[t]),
                                         _mm512_mul_ps(bl2, yd));
                v = _mm512_fmadd_ps(_mm512_cvtepi32_ps(hi[t]),
                                    _mm512_mul_ps(bh2, yd), v);
                f[t] = _mm512_add_ps(f[t], v);
            }
            f[0] = _mm512_add_ps(f[0], _mm512_cvtepi32_ps(
                    _mm512_add_epi32(lo[0], hi[0])));
        }
    }
    float acc = 0.0f, lane[16];
    for (int t = 0; t < 8; t++) {
        _mm512_storeu_ps(lane, f[t]);
        for (int r = 0; r < ROWS; r++) acc += lane[r];
    }
    InterlockedAdd64(&g_sink, (LONG64)acc);
}

/* ---- pool --------------------------------------------------------------- */
static DWORD WINAPI worker(LPVOID arg)
{
    const int id = (int)(intptr_t)arg;
    LONG seen = 0;
    for (;;) {
        while (g_gen == seen) {
            if (g_quit) return 0;
            _mm_pause();
        }
        seen = g_gen;
        const uint64_t lo = g_from + (g_to - g_from) * (uint64_t)id / (uint64_t)g_T;
        const uint64_t hi = g_from + (g_to - g_from) * (uint64_t)(id + 1) / (uint64_t)g_T;
        if (g_mode == 0)
            stream_read(g_base + lo * REC, (hi - lo) * REC);
        else
            for (uint64_t g = lo; g < hi; g++) kern(g_base + g * REC, g_y);
        InterlockedIncrement(&g_arrived);
    }
}

/* run one timed pass; returns milliseconds */
static double dispatch(uint64_t n_groups, int mode, int T)
{
    g_from = 0;
    g_to = n_groups;
    g_mode = mode;
    g_arrived = 0;
    const double t0 = now_ms();
    InterlockedIncrement(&g_gen);
    while (g_arrived < T) _mm_pause();
    return now_ms() - t0;
}

int main(int argc, char **argv)
{
    FILE *out = argc > 1 ? fopen(argv[1], "w") : stdout;
    if (!out) return 1;

    const uint64_t N = 8000;                 /* 8000 * 48.6 KB = 380 MB */
    uint8_t *buf = VirtualAlloc(NULL, N * REC, MEM_COMMIT | MEM_RESERVE,
                                PAGE_READWRITE);
    q8b *y = calloc(8, sizeof(q8b));
    if (!buf || !y) return 1;
    for (int t = 0; t < 8; t++) {
        for (int i = 0; i < 256; i++) y[t].qs[i] = (int8_t)(i * 7);
        y[t].d = 0.01f;
    }
    for (uint64_t i = 0; i < N * REC; i++) buf[i] = (uint8_t)(i * 37u);
    g_base = buf;
    g_y = y;

    fprintf(out, "mode    threads   best_ms   GB/s    dpbusd/cyc/core\n");
    fprintf(out, "--------------------------------------------------------\n");
    fflush(out);

    const int tcs[] = { 1, 4, 8, 16, 24, 28, 32 };
    for (int mode = 0; mode < 2; mode++) {
        for (int ti = 0; ti < (int)(sizeof(tcs) / sizeof(tcs[0])); ti++) {
            const int T = tcs[ti];
            HANDLE th[MAXT];
            g_T = T;
            g_quit = 0;
            for (int i = 0; i < T; i++)
                th[i] = CreateThread(NULL, 0, worker, (LPVOID)(intptr_t)i,
                                     0, NULL);

            double best = 1e18;
            for (int rep = 0; rep < 4; rep++) {
                const double ms = dispatch(N, mode, T);
                if (rep > 0 && ms < best) best = ms;   /* rep 0 warms */
            }

            g_quit = 1;
            InterlockedIncrement(&g_gen);
            WaitForMultipleObjects((DWORD)T, th, TRUE, INFINITE);
            for (int i = 0; i < T; i++) CloseHandle(th[i]);

            const double bytes = (double)N * (double)REC;
            const double gbs = bytes / best / 1e6;
            if (mode == 0) {
                fprintf(out, "stream  %7d  %8.2f  %6.2f  %14s\n",
                        T, best, gbs, "-");
            } else {
                const double dpbusd = (double)N * (double)BLOCKS *
                                      32.0 * 8.0;      /* 8 tokens, 4 c, 2 nib */
                const double cyc = best * 1e-3 * 4.5e9 * (double)T;
                fprintf(out, "kernel  %7d  %8.2f  %6.2f  %14.3f\n",
                        T, best, gbs, dpbusd / cyc);
            }
            fflush(out);
        }
    }
    if (out != stdout) fclose(out);
    fprintf(stderr, "sink=%lld\n", (long long)g_sink);
    return 0;
}
