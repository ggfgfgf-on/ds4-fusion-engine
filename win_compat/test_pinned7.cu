// test_pinned7.cu — pin down the exact boundary: blocks 1..8, thread counts,
// and whether a single-block kernel reading 4KB fails while the same range
// read by >=2 blocks succeeds.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

__global__ void read_f(const float *w, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = w[i] * 2.0f;
}
__global__ void read_f_stride(const float *w, float *out, int n) {
    // one block, strided so a single warp covers the range sparsely
    int i = threadIdx.x;
    if (i < n) out[i] = w[i] * 2.0f;
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
    fill_data(host, n, 555);
    cudaError_t err = cudaHostRegister(host, n * sizeof(float),
                                       cudaHostRegisterMapped | cudaHostRegisterReadOnly);
    void *dev = NULL;
    if (err == cudaSuccess) cudaHostGetDevicePointer(&dev, host, 0);
    if (err != cudaSuccess || !dev) { printf("register failed\n"); return 1; }
    cudaDeviceSynchronize();

    float *d_out = NULL;
    cudaMalloc(&d_out, n * sizeof(float));
    float *back = (float *)malloc(16384 * sizeof(float));
    const int m = 1024;

    int blocks_list[8] = {1, 1, 1, 2, 3, 4, 8, 64};
    int threads_list[8] = {64, 128, 256, 256, 256, 256, 256, 256};
    for (int t = 0; t < 8; t++) {
        const int blocks = blocks_list[t], threads = threads_list[t];
        cudaMemset(d_out, 0, m * sizeof(float));
        read_f<<<blocks, threads>>>((const float *)dev, d_out, m);
        cudaError_t e = cudaGetLastError();
        cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("blocks=%d threads=%d : ERR %s\n", blocks, threads, cudaGetErrorString(e)); continue; }
        cudaMemcpy(back, d_out, m * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < m; i++)
            if (back[i] != host[i] * 2.0f) bad++;
        printf("blocks=%2d threads=%3d : %s (%d/%d bad)\n",
               blocks, threads, bad ? "FAIL" : "OK  ", bad, m);
    }

    // single block, strided sparse read
    {
        cudaMemset(d_out, 0, m * sizeof(float));
        read_f_stride<<<1, 256>>>((const float *)dev, d_out, m);
        cudaGetLastError();
        cudaDeviceSynchronize();
        cudaMemcpy(back, d_out, m * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < 256; i++)
            if (back[i] != host[i] * 2.0f) bad++;
        printf("1 block strided(256 elems): %s (%d bad)\n", bad ? "FAIL" : "OK", bad);
    }

    free(back);
    cudaFree(d_out);
    cudaHostUnregister(host);
    free(host);
    return 0;
}
