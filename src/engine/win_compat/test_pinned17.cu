// test_pinned17.cu — WDDM-evicted pages: is a KERNEL write correct
// (cudaMemset fails there, DMA writes are fine, test_pinned15)?
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

__global__ void read_loop(const float *w, float *out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        out[i] = w[i] * 2.0f;
}
__global__ void write_kernel(float *w, float v, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        w[i] = v;
}

int main() {
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
    printf("filler %d GB, free %.2f GiB\n", nf, (double)free_b / 1073741824.0);

    float *block = NULL;
    if (cudaMalloc((void **)&block, 1024ull * 1024ull * 1024ull) != cudaSuccess) {
        printf("evict block alloc failed\n");
        return 1;
    }
    cudaMemGetInfo(&free_b, &total_b);
    printf("evict block allocated, free %.2f GiB\n", (double)free_b / 1073741824.0);

    const int chunk = 4 * 1024 * 1024;
    float *out = NULL;
    cudaMalloc(&out, (size_t)chunk * sizeof(float));
    float *back = (float *)malloc((size_t)chunk * sizeof(float));

    for (int trial = 0; trial < 3; trial++) {
        const int off = trial * chunk;
        write_kernel<<<2048, 256>>>(block + off, 9.25f, chunk);
        cudaError_t e = cudaGetLastError();
        cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("kwrite %d err: %s\n", trial, cudaGetErrorString(e)); continue; }
        cudaMemset(out, 0, (size_t)chunk * sizeof(float));
        read_loop<<<512, 256>>>(block + off, out, chunk);
        cudaGetLastError();
        cudaDeviceSynchronize();
        cudaMemcpy(back, out, (size_t)chunk * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < chunk; i++) if (back[i] != 18.5f) bad++;
        printf("kernel write evicted page trial %d: %s (%d bad)\n",
               trial, bad ? "FAIL" : "OK  ", bad);
    }

    free(back);
    cudaFree(out);
    cudaFree(block);
    for (int i = 0; i < nf; i++) cudaFree(filler[i]);
    return 0;
}
