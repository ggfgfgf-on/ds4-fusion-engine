/* What this machine can actually read, measured rather than assumed.
 *
 * The prefill moves fifteen and a half gigabytes through the host on every chunk, and the
 * decode reads that much per verify round, so the read rate bounds both. This allocates a
 * buffer the way the engine does - two-megabyte pages when the privilege is available -
 * and reads it with every core, reporting the rate the memory system gives.
 *
 * Build: clang -O2 -mavx512f -o bwbench.exe bwbench.c
 * Run:   bwbench.exe [gigabytes] [threads]
 */
#include <windows.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static double now_sec(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

typedef struct {
    const uint8_t *base;
    size_t bytes;
    unsigned long long sum;
    int id, nthreads;
} job;

static DWORD WINAPI worker(LPVOID arg) {
    job *j = arg;
    const size_t chunk = j->bytes / (size_t)j->nthreads;
    const uint8_t *p = j->base + chunk * (size_t)j->id;
    const uint8_t *end = (j->id == j->nthreads - 1) ? j->base + j->bytes : p + chunk;
    unsigned long long acc = 0;
#if defined(__AVX512F__)
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
    while (p + 256 <= end) {
        a0 = _mm512_add_epi64(a0, _mm512_loadu_si512((const void *)p));
        a1 = _mm512_add_epi64(a1, _mm512_loadu_si512((const void *)(p + 64)));
        a2 = _mm512_add_epi64(a2, _mm512_loadu_si512((const void *)(p + 128)));
        a3 = _mm512_add_epi64(a3, _mm512_loadu_si512((const void *)(p + 192)));
        p += 256;
    }
    a0 = _mm512_add_epi64(_mm512_add_epi64(a0, a1), _mm512_add_epi64(a2, a3));
    acc += (unsigned long long)_mm512_reduce_add_epi64(a0);
#endif
    while (p + 8 <= end) {
        unsigned long long v;
        memcpy(&v, p, 8);
        acc += v;
        p += 8;
    }
    j->sum = acc;
    return 0;
}

int main(int argc, char **argv) {
    const double gib = (argc > 1) ? atof(argv[1]) : 4.0;
    const int nt = (argc > 2) ? atoi(argv[2]) : 28;
    const size_t bytes = (size_t)(gib * 1073741824.0);

    HANDLE tok = NULL;
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) {
        TOKEN_PRIVILEGES tp;
        LUID luid;
        if (LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &luid)) {
            tp.PrivilegeCount = 1;
            tp.Privileges[0].Luid = luid;
            tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
            AdjustTokenPrivileges(tok, FALSE, &tp, sizeof(tp), NULL, NULL);
        }
        CloseHandle(tok);
    }
    const SIZE_T lp = GetLargePageMinimum();
    uint8_t *buf = NULL;
    const char *kind = "ordinary";
    if (lp) {
        const size_t rounded = (bytes + lp - 1) & ~(size_t)(lp - 1);
        buf = VirtualAlloc(NULL, rounded, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                           PAGE_READWRITE);
        if (buf) kind = "large";
    }
    size_t got = bytes;
    if (!buf) {
        buf = VirtualAlloc(NULL, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!buf) { printf("alloc failed\n"); return 1; }
    }
    /* touch it so the reads see resident memory rather than a fresh mapping */
    memset(buf, 1, got);

    job jobs[64];
    HANDLE th[64];
    for (int i = 0; i < nt; i++) {
        jobs[i].base = buf;
        jobs[i].bytes = got;
        jobs[i].sum = 0;
        jobs[i].id = i;
        jobs[i].nthreads = nt;
        th[i] = CreateThread(NULL, 0, worker, &jobs[i], 0, NULL);
    }
    const double t0 = now_sec();
    for (int i = 0; i < nt; i++) {
        WaitForSingleObject(th[i], INFINITE);
        CloseHandle(th[i]);
    }
    const double t1 = now_sec();
    unsigned long long sum = 0;
    for (int i = 0; i < nt; i++) sum += jobs[i].sum;
    const double gb = (double)got / 1073741824.0;
    printf("%.2f GiB %s pages, %d threads: %.0f ms = %.1f GiB/s  (checksum %llu)\n", gb, kind,
           nt, (t1 - t0) * 1e3, gb / (t1 - t0), sum);
    return 0;
}
