/* Working-set sweep for the Q4_K wide kernel.
 *
 * The kernel measures 0.23 dpbusd per cycle while llvm-mca models its inner loop
 * at one per cycle, and llvm-mca cannot say why because it treats L1 as having
 * no latency. Reading the hardware counters would settle it, but that needs a
 * driver this machine does not have - Get-Counter exposes no Hardware Events
 * set, wpr has no PMC profile, and no MSR driver is installed.
 *
 * Timing can still separate the two possibilities. The kernel draws on two
 * independent pools:
 *
 *   weights      N row groups of blocks * 2432 bytes
 *   activations  8 tokens of blocks * 340 bytes, fixed
 *
 * Sweeping N moves the weight pool through L1, L2, L3 and DRAM while the
 * activation pool stays where it is. If the kernel is already at its measured
 * rate with N small enough that the weights fit L1, the gap is not memory at
 * all - it is the dependency structure of the loop itself, and no amount of
 * cache tuning will help. If instead the rate holds at small N and falls as N
 * grows, the curve's knee names the level responsible.
 */
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define ROWS 16
#define BLOCKS 20
#define BYTES (BLOCKS * (uint64_t)(64 + 64 + 128 + 128 + 2048))

typedef struct { int8_t qs[256]; float d; int16_t bsums[16]; } q8b;

static double now(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e3 / (double)f.QuadPart;
}

/* One row group, identical to the engine's lim == 8 path. */
static void kern(const uint8_t *grp, const q8b *y) {
    const __m512i m0F = _mm512_set1_epi8(0x0F);
    __m512 f[8];
    for (int t = 0; t < 8; t++) f[t] = _mm512_setzero_ps();
    for (uint64_t b = 0; b < BLOCKS; b++) {
        const uint8_t *xp = grp + b * BYTES;
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
            /* keep the four sub-components live so nothing is folded away */
            f[0] = _mm512_add_ps(f[0], _mm512_cvtepi32_ps(
                    _mm512_add_epi32(lo[0], hi[0])));
        }
    }
    float acc = 0.0f;
    float lane[16];
    for (int t = 0; t < 8; t++) {
        _mm512_storeu_ps(lane, f[t]);
        for (int r = 0; r < ROWS; r++) acc += lane[r];
    }
    if (acc == 1234.5678f) printf(" ");
}

int main(int argc, char **argv) {
    FILE *out = argc > 1 ? fopen(argv[1], "w") : stdout;
    if (!out) return 1;
    fprintf(out, "groups  weightMB  ms        dpbusd/cycle/core  GB/s\n");

    const uint64_t maxN = 20000;          /* 20000 * 48.6 KB = 972 MB */
    uint8_t *buf = VirtualAlloc(NULL, maxN * BLOCKS * BYTES,
                                MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    q8b *y = malloc(8 * sizeof(q8b));
    if (!buf || !y) return 1;
    memset(y, 3, 8 * sizeof(q8b));
    for (int t = 0; t < 8; t++) y[t].d = 0.01f;
    for (uint64_t i = 0; i < maxN * BLOCKS * BYTES; i++)
        buf[i] = (uint8_t)(i * 37u);

    const uint64_t Ns[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                            1024, 2048, 4096, 8192, 16384 };
    for (size_t k = 0; k < sizeof(Ns) / sizeof(Ns[0]); k++) {
        const uint64_t N = Ns[k];
        const uint64_t bytes = N * BLOCKS * BYTES;
        /* enough repetitions for ~200 ms at the slow end */
        uint64_t reps = 20000000ull / (N * 4096ull);
        if (reps < 1) reps = 1;
        if (reps > 2000) reps = 2000;

        double t0 = now();
        for (uint64_t r = 0; r < reps; r++)
            for (uint64_t g = 0; g < N; g++)
                kern(buf + g * BLOCKS * BYTES, y);
        double t1 = now();

        double ms = t1 - t0;
        double dpbusd = (double)N * (double)reps * (double)BLOCKS * 4.0 * 8.0 * 16.0;
        /* cycles at a fixed 4.5 GHz reference; the ratio is what matters */
        double cyc = ms * 1e-3 * 4.5e9;
        fprintf(out, "%6llu  %8.2f  %8.2f  %18.3f  %7.2f\n",
                (unsigned long long)N, (double)bytes / 1048576.0, ms,
                dpbusd / cyc, (double)bytes * (double)reps / ms / 1e6);
        fflush(out);
    }
    if (out != stdout) fclose(out);
    return 0;
}
