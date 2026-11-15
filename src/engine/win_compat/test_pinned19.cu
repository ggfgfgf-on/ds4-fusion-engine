// test_pinned19.cu — eviction THRASH test: keep allocating/freeing to force
// pages in/out of shared memory while a verification block is being
// written/read by kernels. Static-eviction probes all pass; the real
// prefill churns allocations continuously.
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
    const int chunk = 4 * 1024 * 1024;
    float *out = NULL;
    cudaMalloc(&out, (size_t)chunk * sizeof(float));
    float *back = (float *)malloc((size_t)chunk * sizeof(float));

    // base fill
    float *filler[6];
    int nf = 0;
    for (int i = 0; i < 6; i++) {
        filler[i] = NULL;
        if (cudaMalloc((void **)&filler[i], 1024ull * 1024ull * 1024ull) != cudaSuccess) break;
        nf++;
    }
    // verification block (in the evicted zone)
    float *vblock = NULL;
    cudaMalloc((void **)&vblock, 512ull * 1024ull * 1024ull);

    int total_bad = 0, rounds = 0;
    for (int round = 0; round < 8; round++) {
        // churn: allocate/free 1GB blocks to force eviction/reclaim
        float *churn[4];
        for (int i = 0; i < 4; i++) {
            churn[i] = NULL;
            cudaMalloc((void **)&churn[i], 256ull * 1024ull * 1024ull);
        }
        for (int i = 0; i < 4; i++) if (churn[i]) cudaFree(churn[i]);

        // write + read verification block during churn pressure
        const int off = (round % 4) * (chunk / 4);
        write_kernel<<<512, 256>>>(vblock + off, (float)(round + 1) * 1.5f, chunk / 4);
        cudaGetLastError();
        cudaDeviceSynchronize();
        cudaMemset(out, 0, (size_t)chunk * sizeof(float));
        read_loop<<<256, 256>>>(vblock + off, out, chunk / 4);
        cudaGetLastError();
        cudaDeviceSynchronize();
        cudaMemcpy(back, out, (size_t)(chunk / 4) * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        const float want = (float)(round + 1) * 3.0f;
        for (int i = 0; i < chunk / 4; i++) if (back[i] != want) bad++;
        total_bad += bad;
        rounds++;
        printf("thrash round %d: %s (%d bad)\n", round, bad ? "FAIL" : "OK  ", bad);
    }
    printf("TOTAL: %d bad over %d rounds\n", total_bad, rounds);

    free(back);
    cudaFree(out);
    cudaFree(vblock);
    for (int i = 0; i < nf; i++) cudaFree(filler[i]);
    return 0;
}
