// test_pinned11.cu — can we register a LARGE (2GB) pinned block and what
// bandwidth does the GPU get reading it? (prep for a host-resident slot
// table: no VRAM cost, PCIe zero-copy reads)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

__global__ void read_loop_bw(const float *w, float *out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        out[i] = w[i] * 2.0f;
}
__global__ void read_f4_bw(const float4 *w, float4 *out, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 v = w[i];
        out[i].x = v.x + v.y + v.z + v.w;
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
    const size_t gb = 2;
    const size_t bytes = gb * 1024ull * 1024ull * 1024ull;
    const int n = (int)(bytes / sizeof(float));
    printf("registering %zu GiB ...\n", gb);
    float *host = NULL;
    cudaError_t err = cudaHostAlloc((void **)&host, bytes,
                                    cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("hostalloc 2GB failed: %s\n", cudaGetErrorString(err));
        // try malloc + register
        host = (float *)malloc(bytes);
        if (!host) { printf("malloc 2GB failed\n"); return 1; }
        err = cudaHostRegister(host, bytes,
                               cudaHostRegisterMapped | cudaHostRegisterReadOnly);
        if (err != cudaSuccess) {
            printf("register 2GB failed: %s\n", cudaGetErrorString(err));
            free(host);
            return 1;
        }
    }
    void *dev = NULL;
    cudaHostGetDevicePointer(&dev, host, 0);
    if (!dev) { printf("no device ptr\n"); return 1; }
    cudaDeviceSynchronize();

    // fill first and last 64MB
    const int chunk = 16 * 1024 * 1024;
    fill_data(host, chunk, 1000);
    fill_data(host + (n - chunk), chunk, 2000);

    float *d_out = NULL;
    cudaMalloc(&d_out, (size_t)chunk * sizeof(float));
    float *back = (float *)malloc((size_t)chunk * sizeof(float));

    // correctness at both ends
    cudaMemset(d_out, 0, (size_t)chunk * sizeof(float));
    read_loop_bw<<<512, 256>>>((const float *)dev, d_out, chunk);
    cudaGetLastError(); cudaDeviceSynchronize();
    cudaMemcpy(back, d_out, (size_t)chunk * sizeof(float), cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < chunk; i++)
        if (back[i] != host[i] * 2.0f) bad++;
    printf("read start 64MB (512 blocks loop): %s (%d bad)\n", bad ? "FAIL" : "OK  ", bad);

    cudaMemset(d_out, 0, (size_t)chunk * sizeof(float));
    read_loop_bw<<<512, 256>>>((const float *)dev + (n - chunk), d_out, chunk);
    cudaGetLastError(); cudaDeviceSynchronize();
    cudaMemcpy(back, d_out, (size_t)chunk * sizeof(float), cudaMemcpyDeviceToHost);
    bad = 0;
    for (int i = 0; i < chunk; i++)
        if (back[i] != host[(n - chunk) + i] * 2.0f) bad++;
    printf("read end 64MB   (512 blocks loop): %s (%d bad)\n", bad ? "FAIL" : "OK  ", bad);

    // bandwidth: read 512MB once (grid covers all)
    const int bw_n = 128 * 1024 * 1024;  // 128M floats = 512MB
    float *d_out_bw = NULL;
    cudaMalloc(&d_out_bw, (size_t)bw_n * sizeof(float));
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    // warm
    read_loop_bw<<<2048, 256>>>((const float *)dev, d_out_bw, bw_n);
    cudaDeviceSynchronize();
    cudaEventRecord(t0, 0);
    read_loop_bw<<<2048, 256>>>((const float *)dev, d_out_bw, bw_n);
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    const double gbs = ((double)bw_n * sizeof(float)) / (ms / 1000.0) / 1e9;
    printf("read bandwidth: %.2f ms for 512MB = %.2f GB/s\n", ms, gbs);

    // float4 read bandwidth
    const int n4 = bw_n / 4;
    cudaEventRecord(t0, 0);
    read_f4_bw<<<2048, 256>>>((const float4 *)dev, (float4 *)d_out_bw, n4);
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    printf("float4 bandwidth: %.2f ms for 512MB = %.2f GB/s\n",
           ms, ((double)bw_n * sizeof(float)) / (ms / 1000.0) / 1e9);

    free(back);
    cudaFree(d_out); cudaFree(d_out_bw);
    cudaFreeHost(host);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
