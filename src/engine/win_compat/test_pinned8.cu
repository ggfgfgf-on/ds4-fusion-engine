// test_pinned8.cu — can cache modifiers work around the WDDM mapped-host
// per-thread-2nd-access corruption? Same access pattern (1 block x 256
// threads, 4 elements per thread) through different load paths:
// plain, __ldg (ld.global.nc), __ldcg (ld.global.cg), volatile (ld.volatile),
// inline PTX ld.global.cv, and __ldcv equivalent.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

__global__ void read_plain(const float *w, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = w[i] * 2.0f;
}
__global__ void read_ldg(const float *w, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __ldg(w + i) * 2.0f;
}
__global__ void read_ldcg(const float *w, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __ldcg(w + i) * 2.0f;
}
__global__ void read_vol(const float *w, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ((volatile const float *)w)[i] * 2.0f;
}
__global__ void read_ptx_cv(const float *w, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v;
        asm volatile("ld.global.cv.f32 %0, [%1];" : "=f"(v) : "l"(w + i));
        out[i] = v * 2.0f;
    }
}
__global__ void read_ptx_cg(const float *w, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v;
        asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(w + i));
        out[i] = v * 2.0f;
    }
}
// per-thread loop (4 elements): the failing pattern
__global__ void read_loop_plain(const float *w, float *out, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k = 0; k < 4; k++) {
        int i = t + k * 256;
        if (i < n) out[i] = w[i] * 2.0f;
    }
}
__global__ void read_loop_ldcg(const float *w, float *out, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k = 0; k < 4; k++) {
        int i = t + k * 256;
        if (i < n) out[i] = __ldcg(w + i) * 2.0f;
    }
}
__global__ void read_loop_ldg(const float *w, float *out, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k = 0; k < 4; k++) {
        int i = t + k * 256;
        if (i < n) out[i] = __ldg(w + i) * 2.0f;
    }
}

static void fill_data(float *p, int n, unsigned seed) {
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = (float)((int)(s >> 8) % 100000) * 0.5f;
    }
}

int main() {
    const int n = 1024 * 256;
    float *host = (float *)malloc(n * sizeof(float));
    fill_data(host, n, 777);
    cudaError_t err = cudaHostRegister(host, n * sizeof(float),
                                       cudaHostRegisterMapped | cudaHostRegisterReadOnly);
    void *dev = NULL;
    if (err == cudaSuccess) cudaHostGetDevicePointer(&dev, host, 0);
    if (err != cudaSuccess || !dev) { printf("register failed\n"); return 1; }
    cudaDeviceSynchronize();

    float *d_out = NULL;
    cudaMalloc(&d_out, n * sizeof(float));
    float *back = (float *)malloc(n * sizeof(float));
    const int m = 1024;

    struct { const char *name; void (*kern)(const float *, float *, int); } cases[] = {
        {"plain", read_plain}, {"ldg", read_ldg}, {"ldcg", read_ldcg},
        {"volatile", read_vol}, {"ptx.cv", read_ptx_cv}, {"ptx.cg", read_ptx_cg},
    };
    for (int c = 0; c < 6; c++) {
        cudaMemset(d_out, 0, m * sizeof(float));
        cases[c].kern<<<1, 256>>>((const float *)dev, d_out, m);
        cudaError_t e = cudaGetLastError();
        cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("%-10s : ERR %s\n", cases[c].name, cudaGetErrorString(e)); continue; }
        cudaMemcpy(back, d_out, m * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < m; i++)
            if (back[i] != host[i] * 2.0f) bad++;
        printf("%-10s (4 elem/thread) : %s (%d/%d bad)\n",
               cases[c].name, bad ? "FAIL" : "OK  ", bad, m);
    }

    // loop forms
    struct { const char *name; void (*kern)(const float *, float *, int); } loops[] = {
        {"loop-plain", read_loop_plain}, {"loop-ldcg", read_loop_ldcg}, {"loop-ldg", read_loop_ldg},
    };
    for (int c = 0; c < 3; c++) {
        cudaMemset(d_out, 0, m * sizeof(float));
        loops[c].kern<<<1, 256>>>((const float *)dev, d_out, m);
        cudaError_t e = cudaGetLastError();
        cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("%-12s : ERR %s\n", loops[c].name, cudaGetErrorString(e)); continue; }
        cudaMemcpy(back, d_out, m * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < m; i++)
            if (back[i] != host[i] * 2.0f) bad++;
        printf("%-12s (loop 4/thread)  : %s (%d/%d bad)\n",
               loops[c].name, bad ? "FAIL" : "OK  ", bad, m);
    }

    free(back);
    cudaFree(d_out);
    cudaHostUnregister(host);
    free(host);
    return 0;
}
