/* Where does the slow upload come from?
 *
 * The engine registers its 15.66 GiB Windows large-page weight allocation with
 * cudaHostRegister and still measures 2.9 GB/s on a 17.7 MB block copy, against
 * 28.7 GB/s measured on a cudaHostAlloc buffer. This isolates the variables:
 *
 *   A  cudaHostAlloc (the library's own pinned pool)             - the baseline
 *   B  VirtualAlloc MEM_LARGE_PAGES, then cudaHostRegister        - what the engine does
 *   C  VirtualAlloc ordinary pages, then cudaHostRegister         - is it the large pages?
 *   D  VirtualAlloc MEM_LARGE_PAGES, no registration              - pageable equivalent
 *   E  B but read by a kernel through cudaHostGetDevicePointer    - zero copy
 *
 * and it uses a small block (17.7 MB, the size of one projection) as well as a large
 * one, because a single small copy is where transfer start-up shows up.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

typedef unsigned long long u64;

__global__ void touch_bytes(const unsigned char *p, size_t n, u64 *sink) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if ((p[i] & 0x80u) != 0u) atomicAdd(sink, (u64)1);
    }
}

static float rate_gbps(size_t bytes, float ms) {
    return ms > 0.0f ? (float)((double)bytes / ((double)ms * 1e6)) : 0.0f;
}

int main(void) {
    const size_t small_bytes = 17u * 1024u * 1024u;      /* one projection's weights */
    const size_t big_bytes = 256u * 1024u * 1024u;
    printf("device: %s\n", "RTX 5070 Laptop");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("canMapHostMemory=%d  largePageMin=%llu KiB\n", prop.canMapHostMemory,
           (double)(GetLargePageMinimum() / 1024u));

    HANDLE tok = NULL;
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) {
        TOKEN_PRIVILEGES tp;
        LUID luid;
        if (LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &luid)) {
            tp.PrivilegeCount = 1;
            tp.Privileges[0].Luid = luid;
            tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
            (void)AdjustTokenPrivileges(tok, FALSE, &tp, sizeof(tp), NULL, NULL);
        }
        CloseHandle(tok);
    }
    const size_t lp = (size_t)GetLargePageMinimum();

    void *hosts[5] = {NULL, NULL, NULL, NULL, NULL};
    const char *names[5] = {
        "A cudaHostAlloc (baseline)",
        "B large pages + HostRegister",
        "C normal pages + HostRegister",
        "D large pages, unregistered",
        "E B read by kernel (zero copy)",
    };
    bool mapped[5] = {false, true, true, false, true};

    if (cudaHostAlloc(&hosts[0], big_bytes, cudaHostAllocMapped) != cudaSuccess) hosts[0] = NULL;
    hosts[1] = VirtualAlloc(NULL, (big_bytes + lp - 1) & ~(lp - 1),
                            MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
    hosts[2] = VirtualAlloc(NULL, big_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    hosts[3] = VirtualAlloc(NULL, (big_bytes + lp - 1) & ~(lp - 1),
                            MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
    if (hosts[1] && cudaHostRegister(hosts[1], big_bytes, cudaHostRegisterPortable) != cudaSuccess) {
        printf("  register on B failed\n");
        hosts[1] = hosts[1]; /* keep it, just unregistered */
    }
    if (hosts[2] && cudaHostRegister(hosts[2], big_bytes, cudaHostRegisterPortable) != cudaSuccess) {
        printf("  register on C failed\n");
    }
    for (int i = 0; i < 4; i++)
        if (hosts[i]) memset(hosts[i], 0x5A, big_bytes);

    void *dev = NULL;
    if (cudaMalloc(&dev, big_bytes) != cudaSuccess) { puts("device alloc failed"); return 1; }
    u64 *sink = NULL;
    cudaMalloc(&sink, sizeof(u64));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    for (int i = 0; i < 4; i++) {
        if (!hosts[i]) { printf("  %-32s (unavailable)\n", names[i]); continue; }
        for (int rep = 0; rep < 2; rep++) cudaMemcpy(dev, hosts[i], big_bytes, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for (int rep = 0; rep < 3; rep++)
            cudaMemcpy(dev, hosts[i], big_bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        ms /= 3.0f;
        printf("  %-32s 256 MB  %.1f ms  %.1f GB/s\n", names[i], ms, rate_gbps(big_bytes, ms));

        /* the small block, which is what a projection actually moves */
        float ms_small = 0.0f;
        for (int rep = 0; rep < 2; rep++) cudaMemcpy(dev, hosts[i], small_bytes, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for (int rep = 0; rep < 5; rep++)
            cudaMemcpy(dev, hosts[i], small_bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms_small, e0, e1);
        ms_small /= 5.0f;
        printf("  %-32s 17 MB   %.2f ms  %.1f GB/s\n", "", ms_small, rate_gbps(small_bytes, ms_small));
    }

    /* E: zero copy, the kernel walks the host buffer through the mapped pointer */
    if (mapped[1] && hosts[1]) {
        void *vp = NULL;
        if (cudaHostGetDevicePointer(&vp, hosts[1], 0) == cudaSuccess && vp) {
            const unsigned threads = 256;
            const unsigned grid = (unsigned)((big_bytes + threads - 1) / threads);
            touch_bytes<<<grid, threads>>>((const unsigned char *)vp, big_bytes, sink);
            cudaDeviceSynchronize();
            cudaEventRecord(e0);
            for (int rep = 0; rep < 3; rep++)
                touch_bytes<<<grid, threads>>>((const unsigned char *)vp, big_bytes, sink);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            ms /= 3.0f;
            printf("  %-32s 256 MB  %.1f ms  %.1f GB/s\n", names[4], ms, rate_gbps(big_bytes, ms));
        } else {
            printf("  %-32s (no device pointer)\n", names[4]);
        }
    }
    return 0;
}
