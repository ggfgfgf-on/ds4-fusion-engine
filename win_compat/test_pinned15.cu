// test_pinned15.cu — reproduce the REAL eviction-corruption pattern:
// VRAM at capacity, small-block async H2D writes into evicted pages, then
// SM kernel reads. Compare async memcpy vs sync memcpy vs device memset.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

__global__ void read_loop(const float *w, float *out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        out[i] = w[i] * 2.0f;
}

static void fill_host(float *p, int n, unsigned seed) {
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = (float)((int)(s >> 8) % 100000) * 0.5f;
    }
}

int main() {
    // 1) fill VRAM near capacity
    float *filler[6];
    int nf = 0;
    for (int i = 0; i < 6; i++) {
        filler[i] = NULL;
        if (cudaMalloc((void **)&filler[i], 1024ull * 1024ull * 1024ull) != cudaSuccess)
            break;
        nf++;
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    printf("VRAM filler: %d GB, free now: %.2f GiB\n", nf, (double)free_b / 1073741824.0);

    // 2) one more block that will be evicted/partial
    float *block = NULL;
    bool block_ok = cudaMalloc((void **)&block, 1024ull * 1024ull * 1024ull) == cudaSuccess;
    cudaMemGetInfo(&free_b, &total_b);
    printf("evict block alloc: %s, free now: %.2f GiB\n",
           block_ok ? "ok" : "FAILED", (double)free_b / 1073741824.0);
    if (!block_ok) { printf("cannot even allocate evict block\n"); return 1; }

    const int chunk = 1024 * 256;   // 1M floats
    float *host = (float *)malloc((size_t)chunk * sizeof(float));
    fill_host(host, chunk, 424242);
    float *want = (float *)malloc((size_t)chunk * sizeof(float));
    for (int i = 0; i < chunk; i++) want[i] = host[i] * 2.0f;
    float *out = NULL;
    cudaMalloc(&out, (size_t)chunk * sizeof(float));
    float *back = (float *)malloc((size_t)chunk * sizeof(float));

    // 3) async H2D write into evicted block, then SM read
    cudaStream_t st;
    cudaStreamCreate(&st);
    for (int trial = 0; trial < 3; trial++) {
        const size_t off = (size_t)trial * chunk * sizeof(float);
        cudaMemcpyAsync(block + trial * chunk, host, (size_t)chunk * sizeof(float),
                        cudaMemcpyHostToDevice, st);
        cudaStreamSynchronize(st);
        cudaMemset(out, 0, (size_t)chunk * sizeof(float));
        read_loop<<<512, 256>>>(block + trial * chunk, out, chunk);
        cudaGetLastError();
        cudaDeviceSynchronize();
        cudaMemcpy(back, out, (size_t)chunk * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < chunk; i++) if (back[i] != want[i]) bad++;
        printf("async H2D trial %d (offset %zuMB): %s (%d bad)\n",
               trial, off / 1048576, bad ? "FAIL" : "OK  ", bad);
    }
    // 4) sync memcpy write, SM read
    for (int trial = 0; trial < 2; trial++) {
        const size_t off = (size_t)(trial + 3) * chunk * sizeof(float);
        cudaMemcpy(block + (trial + 3) * chunk, host, (size_t)chunk * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemset(out, 0, (size_t)chunk * sizeof(float));
        read_loop<<<512, 256>>>(block + (trial + 3) * chunk, out, chunk);
        cudaGetLastError();
        cudaDeviceSynchronize();
        cudaMemcpy(back, out, (size_t)chunk * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < chunk; i++) if (back[i] != want[i]) bad++;
        printf("sync memcpy trial %d (offset %zuMB): %s (%d bad)\n",
               trial, off / 1048576, bad ? "FAIL" : "OK  ", bad);
    }
    // 5) device-side memset write, SM read (write via SM)
    {
        const size_t off = 5ull * (size_t)chunk;
        cudaMemset(block + 5 * chunk, 7, (size_t)chunk * sizeof(float));
        cudaMemset(out, 0, (size_t)chunk * sizeof(float));
        read_loop<<<512, 256>>>(block + 5 * chunk, out, chunk);
        cudaGetLastError();
        cudaDeviceSynchronize();
        cudaMemcpy(back, out, (size_t)chunk * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < chunk; i++) {
            if (back[i] != 14.0f) bad++;
        }
        printf("device memset + SM read: %s (%d bad)\n", bad ? "FAIL" : "OK  ", bad);
    }

    cudaStreamDestroy(st);
    cudaFree(out);
    free(back); free(host); free(want);
    if (block_ok) cudaFree(block);
    for (int i = 0; i < nf; i++) cudaFree(filler[i]);
    return 0;
}
