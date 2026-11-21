// Does more than one thread reach more of the memory system?
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#define N ((size_t)1024 * 1024 * 1024)   /* 1 GB shared */

static double now_s(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

typedef struct { float *base; size_t stride; size_t count; int reps;
                 volatile double sink; } work_t;

static DWORD WINAPI worker(LPVOID p) {
    work_t *w = (work_t *)p;
    __m512 acc = _mm512_setzero_ps();
    for (int r = 0; r < w->reps; r++) {
        const float *a = w->base;
        for (size_t i = 0; i < w->count; i += 16)
            acc = _mm512_add_ps(acc, _mm512_loadu_ps(a + i));
        w->base += w->stride;
    }
    w->sink = (double)_mm512_reduce_add_ps(acc);
    return 0;
}

int main(void) {
    const size_t n = N;
    float *a = (float *)_aligned_malloc(n * sizeof(float), 64);
    if (!a) { printf("alloc failed\n"); return 1; }
    for (size_t i = 0; i < n; i++) a[i] = 1.0f;
    const int nt_list[] = {1, 8, 16, 28, 32};
    const int reps = 3;
    printf("  threads |  GB/s\n");
    for (int t = 0; t < 5; t++) {
        const int nt = nt_list[t];
        /* each thread walks a contiguous slice, so the whole buffer is read once */
        const size_t per = n / (size_t)nt;
        work_t *w = (work_t *)calloc((size_t)nt, sizeof(*w));
        HANDLE *h = (HANDLE *)calloc((size_t)nt, sizeof(*h));
        double t0 = now_s();
        for (int i = 0; i < nt; i++) {
            w[i].base = a + (size_t)i * per;
            w[i].stride = 0;
            w[i].count = per;
            w[i].reps = reps;
            h[i] = CreateThread(NULL, 0, worker, &w[i], 0, NULL);
        }
        WaitForMultipleObjects((DWORD)nt, h, TRUE, INFINITE);
        double t1 = now_s();
        for (int i = 0; i < nt; i++) CloseHandle(h[i]);
        const double bytes = (double)(per * (size_t)nt) * sizeof(float) * (double)reps;
        printf("  %7d | %5.1f\n", nt, bytes / (t1 - t0) / 1e9);
        free(w); free(h);
    }
    _aligned_free(a);
    return 0;
}
