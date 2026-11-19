/* Same kernel, now threaded.
 *
 * The single-threaded sweep put the kernel at 1.09 to 1.46 dpbusd per cycle with
 * the weights in L1 or L2, and 0.42 with them in DRAM. The engine measures 0.26
 * single-threaded and 0.13 per core at twenty-eight, three to four times worse
 * than this same code at every working-set size.
 *
 * Two explanations remain and this separates them. If adding threads here also
 * collapses the rate, the limit is the memory system shared across cores and the
 * engine is innocent. If the threaded rate stays near the single-threaded one,
 * the engine's dispatch is losing the difference and that is where to look.
 */
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define ROWS 16
#define BLOCKS 20
#define REC (BLOCKS * (uint64_t)(64 + 64 + 128 + 128 + 2048))

typedef struct { int8_t qs[256]; float d; int16_t bsums[16]; } q8b;

static double now(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e3 / (double)f.QuadPart;
}

static void kern(const uint8_t *grp, const q8b *y) {
    const __m512i m0F = _mm512_set1_epi8(0x0F);
    __m512 f[8];
    for (int t = 0; t < 8; t++) f[t] = _mm512_setzero_ps();
    for (uint64_t b = 0; b < BLOCKS; b++) {
        const uint8_t *xp = grp + b * (uint64_t)(64 + 64 + 128 + 128 + 2048);
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
    if (acc == 1234.5678f) printf(" ");
}

typedef struct {
    const uint8_t *buf;
    uint64_t g0, g1, blocks_total;   /* blocks_total = groups in the buffer */
    const q8b *y;
} job;

static HANDLE g_start;
static DWORD WINAPI worker(LPVOID p) {
    job *j = p;
    WaitForSingleObject(g_start, INFINITE);
    for (uint64_t g = j->g0; g < j->g1; g++)
        kern(j->buf + g * REC, j->y);
    return 0;
}

int main(int argc, char **argv) {
    FILE *out = argc > 1 ? fopen(argv[1], "w") : stdout;
    if (!out) return 1;
    fprintf(out, "threads  groups  weightMB   ms      dpbusd/cycle/core  totalGB/s\n");

    const uint64_t maxN = 20000;
    uint8_t *buf = VirtualAlloc(NULL, maxN * REC, MEM_COMMIT | MEM_RESERVE,
                                PAGE_READWRITE);
    q8b *y = malloc(8 * sizeof(q8b));
    if (!buf || !y) return 1;
    memset(y, 3, 8 * sizeof(q8b));
    for (int t = 0; t < 8; t++) y[t].d = 0.01f;
    for (uint64_t i = 0; i < maxN * REC; i++) buf[i] = (uint8_t)(i * 37u);

    g_start = CreateEvent(NULL, TRUE, FALSE, NULL);
    const int tcounts[] = { 1, 8, 28 };
    const uint64_t Ns[] = { 32, 256, 1024, 8192 };

    for (int ti = 0; ti < 3; ti++) {
        const int T = tcounts[ti];
        HANDLE th[32];
        job jobs[32];
        for (uint64_t ni = 0; ni < sizeof(Ns) / sizeof(Ns[0]); ni++) {
            const uint64_t N = Ns[ni];
            uint64_t reps = 16000000ull / (N * 4096ull);
            if (reps < 1) reps = 1;
            if (reps > 500) reps = 500;

            double t0 = now();
            for (uint64_t r = 0; r < reps; r++) {
                ResetEvent(g_start);
                for (int i = 0; i < T; i++) {
                    jobs[i].buf = buf;
                    jobs[i].y = y;
                    jobs[i].g0 = N * (uint64_t)i / (uint64_t)T;
                    jobs[i].g1 = N * (uint64_t)(i + 1) / (uint64_t)T;
                    th[i] = CreateThread(NULL, 0, worker, &jobs[i], 0, NULL);
                }
                SetEvent(g_start);
                WaitForMultipleObjects((DWORD)T, th, TRUE, INFINITE);
                for (int i = 0; i < T; i++) CloseHandle(th[i]);
            }
            double ms = now() - t0;

            const double bytes = (double)N * (double)REC;
            const double dpbusd = (double)N * (double)reps *
                                  (double)BLOCKS * 4.0 * 8.0 * 16.0 * 2.0;
            const double cyc = ms * 1e-3 * 4.5e9 * (double)T;
            fprintf(out, "%7d  %6llu  %8.2f  %8.2f  %18.3f  %8.2f\n",
                    T, (unsigned long long)N, bytes / 1048576.0, ms,
                    dpbusd / cyc, bytes * (double)reps / ms / 1e6);
            fflush(out);
        }
    }
    if (out != stdout) fclose(out);
    return 0;
}
