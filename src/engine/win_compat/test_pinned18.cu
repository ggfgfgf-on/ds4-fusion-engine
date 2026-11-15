// test_pinned18.cu — hunt the second eviction-corruption source:
// D2D memcpy (sync + async) and atomic writes to evicted pages.
// (Known so far: SM read OK, kernel write OK, H2D DMA OK, cudaMemset FAIL.)
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
__global__ void atomic_write(float *w, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        atomicExch((int *)&w[i], __float_as_int(4.25f));
}

static void fill_host(float *p, int n, unsigned seed) {
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = (float)((int)(s >> 8) % 100000) * 0.5f;
    }
}

int main() {
    float *filler[6];
    int nf = 0;
    for (int i = 0; i < 6; i++) {
        filler[i] = NULL;
        if (cudaMalloc((void **)&filler[i], 1024ull * 1024ull * 1024ull) != cudaSuccess) break;
        nf++;
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    printf("filler %d GB, free %.2f GiB\n", nf, (double)free_b / 1073741824.0);

    float *evict = NULL;
    if (cudaMalloc((void **)&evict, 1024ull * 1024ull * 1024ull) != cudaSuccess) {
        printf("evict alloc failed\n"); return 1;
    }
    float *src = NULL;
    if (cudaMalloc((void **)&src, 64ull * 1024ull * 1024ull) != cudaSuccess) {
        printf("src alloc failed\n"); return 1;
    }
    cudaMemGetInfo(&free_b, &total_b);
    printf("evict+src allocated, free %.2f GiB\n", (double)free_b / 1073741824.0);

    const int chunk = 4 * 1024 * 1024;
    float *host = (float *)malloc((size_t)chunk * sizeof(float));
    fill_host(host, chunk, 777);
    cudaMemcpy(src, host, (size_t)chunk * sizeof(float), cudaMemcpyHostToDevice);

    float *out = NULL;
    cudaMalloc(&out, (size_t)chunk * sizeof(float));
    float *back = (float *)malloc((size_t)chunk * sizeof(float));

    // A) sync D2D into evicted region
    for (int trial = 0; trial < 2; trial++) {
        const int off = trial * chunk;
        cudaMemcpy(evict + off, src, (size_t)chunk * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaError_t e = cudaGetLastError();
        cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("d2d sync %d err: %s\n", trial, cudaGetErrorString(e)); continue; }
        cudaMemset(out, 0, (size_t)chunk * sizeof(float));
        read_loop<<<512, 256>>>(evict + off, out, chunk);
        cudaGetLastError(); cudaDeviceSynchronize();
        cudaMemcpy(back, out, (size_t)chunk * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < chunk; i++) if (back[i] != host[i] * 2.0f) bad++;
        printf("D2D sync  trial %d: %s (%d bad)\n", trial, bad ? "FAIL" : "OK  ", bad);
    }
    // B) async D2D into evicted region
    cudaStream_t st;
    cudaStreamCreate(&st);
    for (int trial = 0; trial < 2; trial++) {
        const int off = (trial + 2) * chunk;
        cudaMemcpyAsync(evict + off, src, (size_t)chunk * sizeof(float),
                        cudaMemcpyDeviceToDevice, st);
        cudaStreamSynchronize(st);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { printf("d2d async %d err: %s\n", trial, cudaGetErrorString(e)); continue; }
        cudaMemset(out, 0, (size_t)chunk * sizeof(float));
        read_loop<<<512, 256>>>(evict + off, out, chunk);
        cudaGetLastError(); cudaDeviceSynchronize();
        cudaMemcpy(back, out, (size_t)chunk * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < chunk; i++) if (back[i] != host[i] * 2.0f) bad++;
        printf("D2D async trial %d: %s (%d bad)\n", trial, bad ? "FAIL" : "OK  ", bad);
    }
    // C) atomic write into evicted region
    for (int trial = 0; trial < 2; trial++) {
        const int off = (trial + 4) * chunk;
        atomic_write<<<512, 256>>>(evict + off, chunk);
        cudaError_t e = cudaGetLastError();
        cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("atomic %d err: %s\n", trial, cudaGetErrorString(e)); continue; }
        cudaMemset(out, 0, (size_t)chunk * sizeof(float));
        read_loop<<<512, 256>>>(evict + off, out, chunk);
        cudaGetLastError(); cudaDeviceSynchronize();
        cudaMemcpy(back, out, (size_t)chunk * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < chunk; i++) if (back[i] != 8.5f) bad++;
        printf("atomic    trial %d: %s (%d bad)\n", trial, bad ? "FAIL" : "OK  ", bad);
    }
    // D) kernel write (control, known OK)
    {
        const int off = 6 * chunk;
        write_kernel<<<2048, 256>>>(evict + off, 9.25f, chunk);
        cudaGetLastError(); cudaDeviceSynchronize();
        cudaMemset(out, 0, (size_t)chunk * sizeof(float));
        read_loop<<<512, 256>>>(evict + off, out, chunk);
        cudaGetLastError(); cudaDeviceSynchronize();
        cudaMemcpy(back, out, (size_t)chunk * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < chunk; i++) if (back[i] != 18.5f) bad++;
        printf("kernel    control: %s (%d bad)\n", bad ? "FAIL" : "OK  ", bad);
    }

    cudaStreamDestroy(st);
    free(back); free(host);
    cudaFree(out); cudaFree(src); cudaFree(evict);
    for (int i = 0; i < nf; i++) cudaFree(filler[i]);
    return 0;
}
