// test_pinned10.cu — CORRECT re-validation of WDDM mapped host memory reads.
// All earlier probes had a checking bug: grids smaller than the element
// count left part of d_out at memset(0), which compared as "mismatch".
// Here every kernel writes EVERY element (grid covers n, or threads loop).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

__global__ void read_full(const float *w, float *out, int n) {  // grid covers n
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = w[i] * 2.0f;
}
__global__ void read_loop(const float *w, float *out, int n) {  // loop covers n
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        out[i] = w[i] * 2.0f;
}
__global__ void read_f4(const float4 *w, float4 *out, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 v = w[i];
        out[i].x = v.x * 2.0f; out[i].y = v.y * 2.0f;
        out[i].z = v.z * 2.0f; out[i].w = v.w * 2.0f;
    }
}
__global__ void read_f2(const float2 *w, float2 *out, int n2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n2) {
        float2 v = w[i];
        out[i].x = v.x * 2.0f; out[i].y = v.y * 2.0f;
    }
}
__global__ void read_offset(const float *base, int off, float *out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        out[i] = base[i + off] * 2.0f;
}

static void fill_data(float *p, int n, unsigned seed) {
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = (float)((int)(s >> 8) % 100000) * 0.5f;
    }
}

int main() {
    const int n = 1024 * 256;   // 1M floats
    float *host = (float *)malloc(n * sizeof(float));
    fill_data(host, n, 9001);
    cudaError_t err = cudaHostRegister(host, n * sizeof(float),
                                       cudaHostRegisterMapped | cudaHostRegisterReadOnly);
    void *dev = NULL;
    if (err == cudaSuccess) cudaHostGetDevicePointer(&dev, host, 0);
    if (err != cudaSuccess || !dev) { printf("register failed: %s\n", cudaGetErrorString(err)); return 1; }
    cudaDeviceSynchronize();

    float *d_out = NULL;
    cudaMalloc(&d_out, n * sizeof(float));
    float *back = (float *)malloc(n * sizeof(float));
    float *want = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) want[i] = host[i] * 2.0f;

    // 1) full-range, single block, loop (the pattern that "failed" before)
    cudaMemset(d_out, 0, n * sizeof(float));
    read_loop<<<1, 256>>>((const float *)dev, d_out, n);
    cudaGetLastError(); cudaDeviceSynchronize();
    cudaMemcpy(back, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < n; i++) if (back[i] != want[i]) bad++;
    printf("loop 1 block full range      : %s (%d bad)\n", bad ? "FAIL" : "OK  ", bad);

    // 2) 4KB slice with 4 blocks (all elements written)
    cudaMemset(d_out, 0, 1024 * sizeof(float));
    read_full<<<4, 256>>>((const float *)dev, d_out, 1024);
    cudaGetLastError(); cudaDeviceSynchronize();
    cudaMemcpy(back, d_out, 1024 * sizeof(float), cudaMemcpyDeviceToHost);
    bad = 0;
    for (int i = 0; i < 1024; i++) if (back[i] != want[i]) bad++;
    printf("4KB slice 4 blocks          : %s (%d bad)\n", bad ? "FAIL" : "OK  ", bad);

    // 3) unaligned offset (ptr form), full coverage
    cudaMemset(d_out, 0, 1024 * sizeof(float));
    read_offset<<<4, 256>>>((const float *)dev, 1234, d_out, 1024);
    cudaGetLastError(); cudaDeviceSynchronize();
    cudaMemcpy(back, d_out, 1024 * sizeof(float), cudaMemcpyDeviceToHost);
    bad = 0;
    for (int i = 0; i < 1024; i++) if (back[i] != host[1234 + i] * 2.0f) bad++;
    printf("unaligned ptr+1234 4 blocks : %s (%d bad)\n", bad ? "FAIL" : "OK  ", bad);

    // 4) float4 vectorized, full coverage
    const int n4 = n / 4;
    cudaMemset(d_out, 0, n * sizeof(float));
    read_f4<<<(n4 + 255) / 256, 256>>>((const float4 *)dev, (float4 *)d_out, n4);
    cudaGetLastError(); cudaDeviceSynchronize();
    cudaMemcpy(back, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    bad = 0;
    for (int i = 0; i < n; i++) if (back[i] != want[i]) bad++;
    printf("float4 full range           : %s (%d bad)\n", bad ? "FAIL" : "OK  ", bad);

    // 5) float2 vectorized, full coverage
    const int n2 = n / 2;
    cudaMemset(d_out, 0, n * sizeof(float));
    read_f2<<<(n2 + 255) / 256, 256>>>((const float2 *)dev, (float2 *)d_out, n2);
    cudaGetLastError(); cudaDeviceSynchronize();
    cudaMemcpy(back, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    bad = 0;
    for (int i = 0; i < n; i++) if (back[i] != want[i]) bad++;
    printf("float2 full range           : %s (%d bad)\n", bad ? "FAIL" : "OK  ", bad);

    free(want); free(back);
    cudaFree(d_out);
    cudaHostUnregister(host);
    free(host);
    return 0;
}
