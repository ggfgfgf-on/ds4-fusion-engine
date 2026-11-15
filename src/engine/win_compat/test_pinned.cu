// test_pinned.cu — verify cudaHostRegisterMapped zero-copy reads work on this
// Windows WDDM + CUDA 13.3 system, including unaligned sub-range access.
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

int main() {
    const int n = 1024 * 256;          // 1M floats = 4MB
    float *host = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) host[i] = (float)(i % 1000);
    cudaError_t err = cudaHostRegister(host, n * sizeof(float),
                                       cudaHostRegisterMapped | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        printf("register failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    void *dev = NULL;
    err = cudaHostGetDevicePointer(&dev, host, 0);
    if (err != cudaSuccess || !dev) {
        printf("get device ptr failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    float *d_out = NULL;
    cudaMalloc(&d_out, n * sizeof(float));
    // aligned access over the whole range
    read_mapped<<<(n + 255) / 256, 256>>>((const float *)dev, d_out, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("aligned kernel failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();
    float *back = (float *)malloc(n * sizeof(float));
    cudaMemcpy(back, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < n; i++)
        if (back[i] != host[i] * 2.0f) bad++;
    printf("aligned: %s (%d mismatches)\n", bad ? "FAIL" : "OK", bad);

    // unaligned via pointer value (off 1234 floats)
    const int off = 1234, m = 1024;
    cudaMemset(d_out, 0, m * sizeof(float));
    read_mapped<<<1, 256>>>((const float *)dev + off, d_out, m);
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("ptr-offset kernel failed: %s\n", cudaGetErrorString(err)); return 1; }
    cudaDeviceSynchronize();
    cudaMemcpy(back, d_out, m * sizeof(float), cudaMemcpyDeviceToHost);
    bad = 0;
    for (int i = 0; i < m; i++)
        if (back[i] != host[off + i] * 2.0f) bad++;
    printf("ptr-offset: %s (%d mismatches)\n", bad ? "FAIL" : "OK", bad);

    // unaligned via kernel-computed offset (base + off)
    cudaMemset(d_out, 0, m * sizeof(float));
    read_mapped_offset<<<1, 256>>>((const float *)dev, off, d_out, m);
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("computed-offset kernel failed: %s\n", cudaGetErrorString(err)); return 1; }
    cudaDeviceSynchronize();
    cudaMemcpy(back, d_out, m * sizeof(float), cudaMemcpyDeviceToHost);
    bad = 0;
    for (int i = 0; i < m; i++)
        if (back[i] != host[off + i] * 2.0f) bad++;
    printf("computed-offset: %s (%d mismatches)\n", bad ? "FAIL" : "OK", bad);

    cudaHostUnregister(host);
    free(host); free(back);
    cudaFree(d_out);
    return 0;
}
