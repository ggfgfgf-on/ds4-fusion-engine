/* Instruction-level throughput: VPMADDUBSW (256) against VPDPBUSD (512 and 256).
 *
 * This is the one measurement never taken, and it decides whether the 16-row
 * Q4_K path can ever pay off. Everything else about that kernel has been
 * measured: halving the weight loads bought 8.6 percent, vectorising the
 * min term bought 56, and reducing instructions per output from 3.25 to 1.25
 * still leaves the kernel 3.7x behind a kernel that uses the two-instruction
 * maddubs/madd pair.
 *
 * Eight independent accumulators per loop, so what is measured is throughput
 * and not latency. Reported as cycles per instruction.
 */
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>

static inline uint64_t rdtsc(void) {
    _mm_lfence();
    return __rdtsc();
}

#define N 50000000ull

int main(void) {
    /* --- VPMADDUBSW ymm, ymm, ymm --- */
    {
        __m256i a0 = _mm256_set1_epi8(3), a1 = a0, a2 = a0, a3 = a0;
        __m256i a4 = a0, a5 = a0, a6 = a0, a7 = a0;
        const __m256i b = _mm256_set1_epi8(2);
        uint64_t best = UINT64_MAX;
        for (int rep = 0; rep < 3; rep++) {
            const uint64_t t0 = rdtsc();
            for (uint64_t i = 0; i < N; i++) {
                a0 = _mm256_add_epi16(a0, _mm256_maddubs_epi16(a0, b));
                a1 = _mm256_add_epi16(a1, _mm256_maddubs_epi16(a1, b));
                a2 = _mm256_add_epi16(a2, _mm256_maddubs_epi16(a2, b));
                a3 = _mm256_add_epi16(a3, _mm256_maddubs_epi16(a3, b));
                a4 = _mm256_add_epi16(a4, _mm256_maddubs_epi16(a4, b));
                a5 = _mm256_add_epi16(a5, _mm256_maddubs_epi16(a5, b));
                a6 = _mm256_add_epi16(a6, _mm256_maddubs_epi16(a6, b));
                a7 = _mm256_add_epi16(a7, _mm256_maddubs_epi16(a7, b));
            }
            const uint64_t dt = rdtsc() - t0;
            if (dt < best) best = dt;
        }
        const uint64_t sum = (uint64_t)_mm256_extract_epi16(a0, 0) +
                             (uint64_t)_mm256_extract_epi16(a7, 0);
        /* sixteen instructions issued per iteration (eight maddubs, eight add) */
        printf("VPMADDUBSW ymm : %.3f cycles/op   (%llu)\n",
               (double)best / (double)(N * 16ull), (unsigned long long)(sum & 1));
    }
    /* --- VPDPBUSD ymm, ymm, ymm --- */
    {
        __m256i a0 = _mm256_set1_epi8(3), a1 = a0, a2 = a0, a3 = a0;
        __m256i a4 = a0, a5 = a0, a6 = a0, a7 = a0;
        const __m256i b = _mm256_set1_epi8(2);
        uint64_t best = UINT64_MAX;
        for (int rep = 0; rep < 3; rep++) {
            const uint64_t t0 = rdtsc();
            for (uint64_t i = 0; i < N; i++) {
                a0 = _mm256_dpbusd_epi32(a0, a0, b);
                a1 = _mm256_dpbusd_epi32(a1, a1, b);
                a2 = _mm256_dpbusd_epi32(a2, a2, b);
                a3 = _mm256_dpbusd_epi32(a3, a3, b);
                a4 = _mm256_dpbusd_epi32(a4, a4, b);
                a5 = _mm256_dpbusd_epi32(a5, a5, b);
                a6 = _mm256_dpbusd_epi32(a6, a6, b);
                a7 = _mm256_dpbusd_epi32(a7, a7, b);
            }
            const uint64_t dt = rdtsc() - t0;
            if (dt < best) best = dt;
        }
        const uint64_t sum = (uint64_t)_mm256_extract_epi32(a0, 0) +
                             (uint64_t)_mm256_extract_epi32(a7, 0);
        printf("VPDPBUSD   ymm : %.3f cycles/op   (%llu)\n",
               (double)best / (double)(N * 8ull), (unsigned long long)(sum & 1));
    }
    /* --- VPDPBUSD zmm, zmm, zmm --- */
    {
        __m512i a0 = _mm512_set1_epi8(3), a1 = a0, a2 = a0, a3 = a0;
        __m512i a4 = a0, a5 = a0, a6 = a0, a7 = a0;
        const __m512i b = _mm512_set1_epi8(2);
        uint64_t best = UINT64_MAX;
        for (int rep = 0; rep < 3; rep++) {
            const uint64_t t0 = rdtsc();
            for (uint64_t i = 0; i < N; i++) {
                a0 = _mm512_dpbusd_epi32(a0, a0, b);
                a1 = _mm512_dpbusd_epi32(a1, a1, b);
                a2 = _mm512_dpbusd_epi32(a2, a2, b);
                a3 = _mm512_dpbusd_epi32(a3, a3, b);
                a4 = _mm512_dpbusd_epi32(a4, a4, b);
                a5 = _mm512_dpbusd_epi32(a5, a5, b);
                a6 = _mm512_dpbusd_epi32(a6, a6, b);
                a7 = _mm512_dpbusd_epi32(a7, a7, b);
            }
            const uint64_t dt = rdtsc() - t0;
            if (dt < best) best = dt;
        }
        const uint64_t sum = (uint64_t)_mm512_reduce_add_epi32(a0) +
                             (uint64_t)_mm512_reduce_add_epi32(a7);
        printf("VPDPBUSD   zmm : %.3f cycles/op   (%llu)\n",
               (double)best / (double)(N * 8ull), (unsigned long long)(sum & 1));
    }
    printf("\nA 512-bit dpbusd covers twice the bytes of a 256-bit one, so it is\n"
           "only a win if zmm cycles/op is less than twice ymm cycles/op.\n"
           "Compare VPDPBUSD zmm against 2 * VPDPBUSD ymm for the same bytes.\n");
    return 0;
}
