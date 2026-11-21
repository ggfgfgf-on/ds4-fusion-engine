// What does this machine actually sustain for vpdpbusd, in MACs per second?
#include <immintrin.h>
#include <stdio.h>
#include <windows.h>

static double now_s(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

/* 8 independent accumulators, one dependent chain each: measures issue rate. */
#define NACC 8
static volatile long long sink;

int main(void) {
    __m512i a = _mm512_set1_epi8(3);
    __m512i b = _mm512_set1_epi8(5);
    __m512i acc[NACC];
    for (int i = 0; i < NACC; i++) acc[i] = _mm512_setzero_si512();

    const long iters = 20000000;
    double t0 = now_s();
    for (long n = 0; n < iters; n++) {
        /* 8 dpbusd per iteration, independent accumulators */
        acc[0] = _mm512_dpbusd_epi32(acc[0], a, b);
        acc[1] = _mm512_dpbusd_epi32(acc[1], a, b);
        acc[2] = _mm512_dpbusd_epi32(acc[2], a, b);
        acc[3] = _mm512_dpbusd_epi32(acc[3], a, b);
        acc[4] = _mm512_dpbusd_epi32(acc[4], a, b);
        acc[5] = _mm512_dpbusd_epi32(acc[5], a, b);
        acc[6] = _mm512_dpbusd_epi32(acc[6], a, b);
        acc[7] = _mm512_dpbusd_epi32(acc[7], a, b);
    }
    double t1 = now_s();

    __m512i s = _mm512_setzero_si512();
    for (int i = 0; i < NACC; i++) s = _mm512_add_epi32(s, acc[i]);
    {   /* keep the accumulators live without AVX512DQ reductions */
        __m256i lo = _mm512_castsi512_si256(s);
        __m256i hi = _mm512_extracti64x4_epi64(s, 1);
        __m256i t = _mm256_add_epi32(lo, hi);
        __m128i u = _mm_add_epi32(_mm256_castsi256_si128(t),
                                  _mm256_extracti128_si256(t, 1));
        u = _mm_add_epi32(u, _mm_shuffle_epi32(u, 0x4E));
        u = _mm_add_epi32(u, _mm_shuffle_epi32(u, 0xB1));
        sink = (long long)_mm_cvtsi128_si32(u);
    }

    const double total = (double)iters * NACC;
    /* each dpbusd is 4 int32 lanes x 16 products = 64 MACs */
    printf("  dpbusd issued    : %.2f G/s  (1 thread)\n", total / (t1 - t0) / 1e9);
    printf("  MACs             : %.1f G/s  (x64)\n", total * 64.0 / (t1 - t0) / 1e9);
    printf("  cycles per dpbusd: ~%.2f  (assuming 5.4 GHz)\n",
           (t1 - t0) * 5.4e9 / total);
    return 0;
}

