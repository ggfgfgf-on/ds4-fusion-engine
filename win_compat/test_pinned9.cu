// test_pinned9.cu — load-width boundary on WDDM mapped host memory:
// scalar f32 loop (known OK), explicit float2, explicit float4, and
// compiler-vectorized contiguous (known FAIL). Also 64-bit via double2.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

__global__ void read_scalar_loop(const float *w, float *out, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k = 0; k < 4; k++) {
        int i = t + k * 256;
        if (i < n) out[i] = w[i] * 2.0f;
    }
}
__global__ void read_contig(const float *w, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = w[i] * 2.0f;
}
__global__ void read_float2(const float2 *w, float2 *out, int n2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n2) {
        float2 v = w[i];
        out[i].x = v.x * 2.0f;
        out[i].y = v.y * 2.0f;
    }
}
__global__ void read_float4(const float4 *w, float4 *out, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 v = w[i];
        out[i].x = v.x * 2.0f;
        out[i].y = v.y * 2.0f;
        out[i].z = v.z * 2.0f;
        out[i].w = v.w * 2.0f;
    }
}
__global__ void read_float4_ldg(const float4 *w, float4 *out, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 v = __ldg(w + i);
        out[i].x = v.x * 2.0f;
        out[i].y = v.y * 2.0f;
        out[i].z = v.z * 2.0f;
        out[i].w = v.w * 2.0f;
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
    const int n = 1024 * 256;   // 1M floats
    float *host = (float *)malloc(n * sizeof(float));
    fill_data(host, n, 31337);
    cudaError_t err = cudaHostRegister(host, n * sizeof(float),
                                       cudaHostRegisterMapped | cudaHostRegisterReadOnly);
    void *dev = NULL;
    if (err == cudaSuccess) cudaHostGetDevicePointer(&dev, host, 0);
    if (err != cudaSuccess || !dev) { printf("register failed\n"); return 1; }
    cudaDeviceSynchronize();

    float *d_out = NULL;
    cudaMalloc(&d_out, n * sizeof(float));
    float *back = (float *)malloc(4096 * sizeof(float));
    const int m = 1024;

    {
        cudaMemset(d_out, 0, m * sizeof(float));
        read_scalar_loop<<<1, 256>>>((const float *)dev, d_out, m);
        cudaGetLastError(); cudaDeviceSynchronize();
        cudaMemcpy(back, d_out, m * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < m; i++) if (back[i] != host[i] * 2.0f) bad++;
        printf("scalar loop (f32)      : %s (%d bad)\n", bad ? "FAIL" : "OK  ", bad);
    }
    {
        cudaMemset(d_out, 0, m * sizeof(float));
        read_contig<<<1, 256>>>((const float *)dev, d_out, m);
        cudaGetLastError(); cudaDeviceSynchronize();
        cudaMemcpy(back, d_out, m * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < m; i++) if (back[i] != host[i] * 2.0f) bad++;
        printf("contiguous (auto-vec)  : %s (%d bad)\n", bad ? "FAIL" : "OK  ", bad);
    }
    {
        const int m2 = m / 2;
        cudaMemset(d_out, 0, m * sizeof(float));
        read_float2<<<1, 256>>>((const float2 *)dev, (float2 *)d_out, m2);
        cudaGetLastError(); cudaDeviceSynchronize();
        cudaMemcpy(back, d_out, m * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < m; i++) if (back[i] != host[i] * 2.0f) bad++;
        printf("explicit float2 (64b)  : %s (%d bad)\n", bad ? "FAIL" : "OK  ", bad);
    }
    {
        const int m4 = m / 4;
        cudaMemset(d_out, 0, m * sizeof(float));
        read_float4<<<1, 256>>>((const float4 *)dev, (float4 *)d_out, m4);
        cudaGetLastError(); cudaDeviceSynchronize();
        cudaMemcpy(back, d_out, m * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < m; i++) if (back[i] != host[i] * 2.0f) bad++;
        printf("explicit float4 (128b) : %s (%d bad)\n", bad ? "FAIL" : "OK  ", bad);
    }
    {
        const int m4 = m / 4;
        cudaMemset(d_out, 0, m * sizeof(float));
        read_float4_ldg<<<1, 256>>>((const float4 *)dev, (float4 *)d_out, m4);
        cudaGetLastError(); cudaDeviceSynchronize();
        cudaMemcpy(back, d_out, m * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < m; i++) if (back[i] != host[i] * 2.0f) bad++;
        printf("float4 __ldg (128b)    : %s (%d bad)\n", bad ? "FAIL" : "OK  ", bad);
    }

    free(back);
    cudaFree(d_out);
    cudaHostUnregister(host);
    free(host);
    return 0;
}
