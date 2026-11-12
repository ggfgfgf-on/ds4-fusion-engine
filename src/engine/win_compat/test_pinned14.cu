// test_pinned14.cu — DIRECT test of path 3: cudaMalloc beyond physical VRAM
// (WDDM evicts pages to shared system memory). Write a known pattern into
// every block, read it back with a GPU KERNEL (SM reads, not DMA), verify.
// Also prints cudaMemGetInfo before/after.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

__global__ void read_loop(const float *w, float *out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        out[i] = w[i] * 2.0f;
}
__global__ void write_f(float *w, float v, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        w[i] = v;
}

int main() {
    size_t free0 = 0, total0 = 0;
    cudaMemGetInfo(&free0, &total0);
    printf("free before: %.2f GiB / %.2f GiB\n", (double)free0 / 1073741824.0,
           (double)total0 / 1073741824.0);

    // allocate 3 x 2.5GB = 7.5GB VRAM; physical is 8GB with other users,
    // so the tail will be evicted to shared memory
    const size_t block_bytes = 2560ull * 1024ull * 1024ull;  // 2.5GB
    const int nb = 3;
    float *blocks[3];
    bool alloc_ok[3];
    for (int i = 0; i < nb; i++) {
        blocks[i] = NULL;
        alloc_ok[i] = cudaMalloc((void **)&blocks[i], block_bytes) == cudaSuccess;
        if (!alloc_ok[i]) {
            printf("block %d alloc failed: %s\n", i, cudaGetErrorString(cudaGetLastError()));
        }
    }
    size_t free1 = 0, total1 = 0;
    cudaMemGetInfo(&free1, &total1);
    printf("free after alloc: %.2f GiB / %.2f GiB\n", (double)free1 / 1073741824.0,
           (double)total1 / 1073741824.0);

    const int n = (int)(block_bytes / sizeof(float));
    const int chunk = 4 * 1024 * 1024;   // verify first/last 16MB of each block
    float *out = NULL;
    cudaMalloc(&out, (size_t)chunk * sizeof(float));
    float *back = (float *)malloc((size_t)chunk * sizeof(float));

    for (int i = 0; i < nb; i++) {
        if (!alloc_ok[i]) continue;
        // write pattern: value = 1000 + i, via kernel (SM writes)
        write_f<<<2048, 256>>>(blocks[i], (float)(1000 + i), n);
        cudaError_t e = cudaGetLastError();
        cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            printf("block %d write failed: %s\n", i, cudaGetErrorString(e));
            continue;
        }
        // read back via kernel at start and end
        int bad = 0;
        for (int region = 0; region < 2; region++) {
            const int off = region ? n - chunk : 0;
            cudaMemset(out, 0, (size_t)chunk * sizeof(float));
            read_loop<<<512, 256>>>(blocks[i] + off, out, chunk);
            e = cudaGetLastError();
            cudaDeviceSynchronize();
            if (e != cudaSuccess) {
                printf("block %d read failed: %s\n", i, cudaGetErrorString(e));
                bad = -1;
                break;
            }
            cudaMemcpy(back, out, (size_t)chunk * sizeof(float), cudaMemcpyDeviceToHost);
            for (int k = 0; k < chunk; k++) {
                if (back[k] != (float)(1000 + i) * 2.0f) bad++;
            }
        }
        printf("block %d (2.5GB) kernel read: %s (%d bad of %d)\n",
               i, bad == 0 ? "OK  " : (bad < 0 ? "ERR " : "FAIL"),
               bad, chunk * 2);
    }

    for (int i = 0; i < nb; i++) if (blocks[i]) cudaFree(blocks[i]);
    cudaFree(out);
    free(back);
    return 0;
}
