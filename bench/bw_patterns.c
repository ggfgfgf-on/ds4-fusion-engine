// Establish the real streaming ceiling: several access patterns, same buffer.
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

static double now_s(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

#define N ((size_t)512 * 1024 * 1024)

int main(void) {
    const size_t n = N;
    float *a = (float *)_aligned_malloc(n * sizeof(float), 64);
    if (!a) { printf("alloc failed\n"); return 1; }
    for (size_t i = 0; i < n; i++) a[i] = 1.0f;
    const size_t bytes = n * sizeof(float);
    const int reps = 6;
    double t0, t1, v = 0.0;
    __m512 acc;

    t0 = now_s();
    acc = _mm512_setzero_ps();
    for (int r = 0; r < reps; r++)
        for (size_t i = 0; i < n; i += 16)
            acc = _mm512_add_ps(acc, _mm512_loadu_ps(a + i));
    t1 = now_s();
    v += _mm512_reduce_add_ps(acc);
    printf("  %-24s %8.1f GB/s\n", "16-wide sum", (double)bytes/(t1-t0)*reps/1e9);

    t0 = now_s();
    acc = _mm512_setzero_ps();
    for (int r = 0; r < reps; r++)
        for (size_t i = 0; i < n; i += 16) {
            _mm_prefetch((const char *)(a + i + 512), _MM_HINT_T0);
            acc = _mm512_add_ps(acc, _mm512_loadu_ps(a + i));
        }
    t1 = now_s();
    v += _mm512_reduce_add_ps(acc);
    printf("  %-24s %8.1f GB/s\n", "16-wide + prefetch", (double)bytes/(t1-t0)*reps/1e9);

    t0 = now_s();
    acc = _mm512_setzero_ps();
    for (int r = 0; r < reps; r++)
        for (size_t i = 0; i < n; i += 16)
            acc = _mm512_add_ps(acc, _mm512_castsi512_ps(
                    _mm512_stream_load_si512((void *)(a + i))));
    t1 = now_s();
    v += _mm512_reduce_add_ps(acc);
    printf("  %-24s %8.1f GB/s\n", "16-wide NT load", (double)bytes/(t1-t0)*reps/1e9);

    {
        __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        const size_t q = n / 4;
        t0 = now_s();
        for (int r = 0; r < reps; r++)
            for (size_t i = 0; i < q; i += 16) {
                a0 = _mm512_add_ps(a0, _mm512_loadu_ps(a + i));
                a1 = _mm512_add_ps(a1, _mm512_loadu_ps(a + q + i));
                a2 = _mm512_add_ps(a2, _mm512_loadu_ps(a + 2*q + i));
                a3 = _mm512_add_ps(a3, _mm512_loadu_ps(a + 3*q + i));
            }
        t1 = now_s();
        v += _mm512_reduce_add_ps(a0)+_mm512_reduce_add_ps(a1)+
             _mm512_reduce_add_ps(a2)+_mm512_reduce_add_ps(a3);
        printf("  %-24s %8.1f GB/s\n", "4 independent streams", (double)bytes/(t1-t0)*reps/1e9);
    }
    printf("  (checksum %.1f)\n", v);
    _aligned_free(a);
    return 0;
}
