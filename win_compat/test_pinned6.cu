// test_pinned6.cu — scan read size vs block count on ONE registration.
// Earlier: 4MB full-range read (4096 blocks) OK; 4KB reads (1 block) FAIL.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

__global__ void read_f(const float *w, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
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
    const int n = 1024 * 256;   // 1M floats = 4MB
    float *host = (float *)malloc(n * sizeof(float));
    fill_data(host, n, 999);
    cudaError_t err = cudaHostRegister(host, n * sizeof(float),
                                       cudaHostRegisterMapped | cudaHostRegisterReadOnly);
    void *dev = NULL;
    if (err == cudaSuccess) cudaHostGetDevicePointer(&dev, host, 0);
    if (err != cudaSuccess || !dev) {
        printf("register failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    float *d_out = NULL;
    cudaMalloc(&d_out, n * sizeof(float));
    float *back = (float *)malloc(n * sizeof(float));

    const int sizes[6] = {256, 1024, 4096, 16384, 65536, 262144};
    const int grids[3] = {1, 64, 4096};
    for (int si = 0; si < 6; si++) {
        const int m = sizes[si];
        for (int gi = 0; gi < 3; gi++) {
            const int blocks = grids[gi];
            if (blocks * 256 < m) continue;
            cudaMemset(d_out, 0, m * sizeof(float));
            read_f<<<blocks, 256>>>((const float *)dev, d_out, m);
            cudaError_t e = cudaGetLastError();
            cudaDeviceSynchronize();
            if (e != cudaSuccess) {
                printf("size=%7d blocks=%4d : KERNEL ERR %s\n", m, blocks, cudaGetErrorString(e));
                continue;
            }
            cudaMemcpy(back, d_out, m * sizeof(float), cudaMemcpyDeviceToHost);
            int bad = 0;
            for (int i = 0; i < m; i++)
                if (back[i] != host[i] * 2.0f) bad++;
            printf("size=%7d blocks=%4d : %s (%d/%d bad)\n",
                   m, blocks, bad ? "FAIL" : "OK  ", bad, m);
        }
    }

    // also: read the SAME 4KB region twice back-to-back (cold vs warm L2)
    {
        cudaMemset(d_out, 0, 1024 * sizeof(float));
        read_f<<<1, 256>>>((const float *)dev, d_out, 1024);
        cudaGetLastError();
        cudaDeviceSynchronize();
        cudaMemcpy(back, d_out, 1024 * sizeof(float), cudaMemcpyDeviceToHost);
        int bad1 = 0;
        for (int i = 0; i < 1024; i++)
            if (back[i] != host[i] * 2.0f) bad1++;
        read_f<<<1, 256>>>((const float *)dev, d_out, 1024);
        cudaGetLastError();
        cudaDeviceSynchronize();
        cudaMemcpy(back, d_out, 1024 * sizeof(float), cudaMemcpyDeviceToHost);
        int bad2 = 0;
        for (int i = 0; i < 1024; i++)
            if (back[i] != host[i] * 2.0f) bad2++;
        printf("back-to-back 4KB: first=%s(%d) second=%s(%d)\n",
               bad1 ? "FAIL" : "OK", bad1, bad2 ? "FAIL" : "OK", bad2);
    }

    free(back);
    cudaFree(d_out);
    cudaHostUnregister(host);
    free(host);
    return 0;
}
