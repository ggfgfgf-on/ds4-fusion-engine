// test_pinned2.cu — probe WDDM mapped-host-memory variants for the
// unaligned-read bug. Runs all register/alloc flag combos and reports
// aligned + unaligned results per variant.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

__global__ void read_mapped(const float *w, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = w[i] * 2.0f;
}
__global__ void read_mapped_offset(const float *base, int off_elems, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = base[i + off_elems] * 2.0f;
}

static int check(const float *got, const float *want, int n) {
    int bad = 0;
    for (int i = 0; i < n; i++)
        if (got[i] != want[i]) bad++;
    return bad;
}

static void run_variant(const char *name, float *host, float *d_out,
                        const float *devptr, int n, int off, int m) {
    // aligned over whole range (read-only smoke)
    cudaMemset(d_out, 0, n * sizeof(float));
    read_mapped<<<(n + 255) / 256, 256>>>((const float *)devptr, d_out, n);
    cudaError_t err = cudaGetLastError();
    cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("%-28s aligned: KERNEL ERR %s\n", name, cudaGetErrorString(err)); return; }
    float *back = (float *)malloc(n * sizeof(float));
    cudaMemcpy(back, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    float *wantA = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) wantA[i] = host[i] * 2.0f;
    int badA = check(back, wantA, n);
    free(back); free(wantA);

    float *want = (float *)malloc(m * sizeof(float));
    for (int i = 0; i < m; i++) want[i] = host[off + i] * 2.0f;
    back = (float *)malloc(m * sizeof(float));

    cudaMemset(d_out, 0, m * sizeof(float));
    read_mapped<<<1, 256>>>((const float *)devptr + off, d_out, m);
    err = cudaGetLastError();
    cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("%-28s ptr-off : KERNEL ERR %s\n", name, cudaGetErrorString(err)); free(want); free(back); return; }
    cudaMemcpy(back, d_out, m * sizeof(float), cudaMemcpyDeviceToHost);
    int badP = check(back, want, m);

    cudaMemset(d_out, 0, m * sizeof(float));
    read_mapped_offset<<<1, 256>>>((const float *)devptr, off, d_out, m);
    err = cudaGetLastError();
    cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("%-28s compoff : KERNEL ERR %s\n", name, cudaGetErrorString(err)); free(want); free(back); return; }
    cudaMemcpy(back, d_out, m * sizeof(float), cudaMemcpyDeviceToHost);
    int badC = check(back, want, m);

    printf("%-28s aligned=%s(%d) ptr=%s(%d) computed=%s(%d)\n",
           name,
           badA ? "FAIL" : "OK", badA,
           badP ? "FAIL" : "OK", badP,
           badC ? "FAIL" : "OK", badC);
    free(want); free(back);
}

int main() {
    const int n = 1024 * 256;            // 1M floats = 4MB
    const int off = 1234, m = 1024;      // off = 4936 bytes, not page aligned
    float *d_out = NULL;
    cudaMalloc(&d_out, n * sizeof(float));
    cudaError_t err = cudaGetLastError();

    {   // A: malloc + Mapped|ReadOnly (original repro)
        float *host = (float *)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) host[i] = (float)(i % 1000);
        err = cudaHostRegister(host, n * sizeof(float),
                               cudaHostRegisterMapped | cudaHostRegisterReadOnly);
        if (err != cudaSuccess) { printf("A register failed: %s\n", cudaGetErrorString(err)); }
        else {
            void *dev = NULL;
            cudaHostGetDevicePointer(&dev, host, 0);
            run_variant("A mapped|ro", host, d_out, (const float *)dev, n, off, m);
            cudaHostUnregister(host);
        }
        free(host);
    }
    {   // B: + IoMemory
        float *host = (float *)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) host[i] = (float)(i % 1000);
        err = cudaHostRegister(host, n * sizeof(float),
                               cudaHostRegisterMapped | cudaHostRegisterReadOnly |
                               cudaHostRegisterIoMemory);
        if (err != cudaSuccess) { printf("B register failed: %s\n", cudaGetErrorString(err)); }
        else {
            void *dev = NULL;
            cudaHostGetDevicePointer(&dev, host, 0);
            run_variant("B mapped|ro|iomem", host, d_out, (const float *)dev, n, off, m);
            cudaHostUnregister(host);
        }
        free(host);
    }
    {   // C: + Portable
        float *host = (float *)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) host[i] = (float)(i % 1000);
        err = cudaHostRegister(host, n * sizeof(float),
                               cudaHostRegisterMapped | cudaHostRegisterReadOnly |
                               cudaHostRegisterPortable);
        if (err != cudaSuccess) { printf("C register failed: %s\n", cudaGetErrorString(err)); }
        else {
            void *dev = NULL;
            cudaHostGetDevicePointer(&dev, host, 0);
            run_variant("C mapped|ro|portable", host, d_out, (const float *)dev, n, off, m);
            cudaHostUnregister(host);
        }
        free(host);
    }
    {   // D: cudaHostAlloc mapped (driver-allocated pinned pool)
        float *host = NULL;
        err = cudaHostAlloc((void **)&host, n * sizeof(float),
                            cudaHostAllocMapped);
        if (err != cudaSuccess) { printf("D hostalloc failed: %s\n", cudaGetErrorString(err)); }
        else {
            for (int i = 0; i < n; i++) host[i] = (float)(i % 1000);
            void *dev = NULL;
            cudaHostGetDevicePointer(&dev, host, 0);
            run_variant("D hostalloc|mapped", host, d_out, (const float *)dev, n, off, m);
            cudaFreeHost(host);
        }
    }
    {   // E: cudaHostAlloc mapped + portable + writable
        float *host = NULL;
        err = cudaHostAlloc((void **)&host, n * sizeof(float),
                            cudaHostAllocMapped | cudaHostAllocPortable);
        if (err != cudaSuccess) { printf("E hostalloc failed: %s\n", cudaGetErrorString(err)); }
        else {
            for (int i = 0; i < n; i++) host[i] = (float)(i % 1000);
            void *dev = NULL;
            cudaHostGetDevicePointer(&dev, host, 0);
            run_variant("E hostalloc|mapped|port", host, d_out, (const float *)dev, n, off, m);
            cudaFreeHost(host);
        }
    }
    cudaFree(d_out);
    return 0;
}
