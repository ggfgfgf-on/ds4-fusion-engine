// test_pinned20.cu — special read paths on WDDM-EVICTED pages:
// __ldg (texture/NC path), __ldcg (L2-only), volatile, and cp.async
// (async copy to shared memory — what mmq tile loaders use on sm_120).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

__global__ void read_plain(const float *w, float *out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        out[i] = w[i] * 2.0f;
}
__global__ void read_ldg(const float *w, float *out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        out[i] = __ldg(w + i) * 2.0f;
}
__global__ void read_ldcg(const float *w, float *out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        out[i] = __ldcg(w + i) * 2.0f;
}
__global__ void read_vol(const float *w, float *out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        out[i] = ((const volatile float *)w)[i] * 2.0f;
}
__global__ void read_cpasync(const float *w, float *out, int n) {
    __shared__ float smem[1024];
    for (int base = blockIdx.x * 1024; base < n; base += gridDim.x * 1024) {
        __pipeline_memcpy_async(&smem[0], &w[base], sizeof(float4) * 256);
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
        const int chunk_n = (n - base < 1024) ? (n - base) : 1024;
        for (int i = threadIdx.x; i < chunk_n; i += blockDim.x)
            out[base + i] = smem[i] * 2.0f;
        __syncthreads();
    }
}
__global__ void read_f4_ldg(const float4 *w, float4 *out, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 v = __ldg(w + i);
        out[i].x = v.x * 2.0f; out[i].y = v.y * 2.0f;
        out[i].z = v.z * 2.0f; out[i].w = v.w * 2.0f;
    }
}

static void fill_host(float *p, int n, unsigned seed) {
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = (float)((int)(s >> 8) % 100000) * 0.5f;
    }
}

int main() {
    float *filler[6];
    int nf = 0;
    for (int i = 0; i < 6; i++) {
        filler[i] = NULL;
        if (cudaMalloc((void **)&filler[i], 1024ull * 1024ull * 1024ull) != cudaSuccess) break;
        nf++;
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    printf("filler %d GB, free %.2f GiB\n", nf, (double)free_b / 1073741824.0);

    float *evict = NULL;
    cudaMalloc((void **)&evict, 1024ull * 1024ull * 1024ull);
    cudaMemGetInfo(&free_b, &total_b);
    printf("evict allocated, free %.2f GiB\n", (double)free_b / 1073741824.0);

    const int chunk = 4 * 1024 * 1024;
    float *host = (float *)malloc((size_t)chunk * sizeof(float));
    fill_host(host, chunk, 1234);
    // write data into the evicted region via DMA (known good)
    cudaMemcpy(evict, host, (size_t)chunk * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    float *out = NULL;
    cudaMalloc(&out, (size_t)chunk * sizeof(float));
    float *back = (float *)malloc((size_t)chunk * sizeof(float));
    float *want = (float *)malloc((size_t)chunk * sizeof(float));
    for (int i = 0; i < chunk; i++) want[i] = host[i] * 2.0f;

    struct { const char *name; void (*kern)(const float *, float *, int); } cases[] = {
        {"plain ld", read_plain}, {"__ldg", read_ldg}, {"__ldcg", read_ldcg},
        {"volatile", read_vol}, {"cp.async", read_cpasync},
    };
    for (int c = 0; c < 5; c++) {
        cudaMemset(out, 0, (size_t)chunk * sizeof(float));
        cases[c].kern<<<64, 256>>>((const float *)evict, out, chunk);
        cudaError_t e = cudaGetLastError();
        cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            printf("%-12s : ERR %s\n", cases[c].name, cudaGetErrorString(e));
            continue;
        }
        cudaMemcpy(back, out, (size_t)chunk * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < chunk; i++) if (back[i] != want[i]) bad++;
        printf("%-12s : %s (%d bad)\n", cases[c].name, bad ? "FAIL" : "OK  ", bad);
    }
    // float4 __ldg
    {
        const int n4 = chunk / 4;
        cudaMemset(out, 0, (size_t)chunk * sizeof(float));
        read_f4_ldg<<<64, 256>>>((const float4 *)evict, (float4 *)out, n4);
        cudaGetLastError();
        cudaDeviceSynchronize();
        cudaMemcpy(back, out, (size_t)chunk * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < chunk; i++) if (back[i] != want[i]) bad++;
        printf("float4 __ldg : %s (%d bad)\n", bad ? "FAIL" : "OK  ", bad);
    }

    free(want); free(back); free(host);
    cudaFree(out); cudaFree(evict);
    for (int i = 0; i < nf; i++) cudaFree(filler[i]);
    return 0;
}
