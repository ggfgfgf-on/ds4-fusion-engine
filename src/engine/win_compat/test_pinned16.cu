// test_pinned16.cu — can UVM (cudaMallocManaged) handle device-side writes
// to pages that live in SYSTEM memory (oversubscribed), where plain
// cudaMalloc + WDDM eviction corrupts (cudaMemset FAIL, test_pinned15)?
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
    const size_t gb = 3;
    const size_t bytes = gb * 1024ull * 1024ull * 1024ull;
    const int n = (int)(bytes / sizeof(float));
    const int chunk = 4 * 1024 * 1024;

    // occupy VRAM first so the managed block cannot fully fit physically
    float *filler[5];
    int nf = 0;
    for (int i = 0; i < 5; i++) {
        filler[i] = NULL;
        if (cudaMalloc((void **)&filler[i], 1024ull * 1024ull * 1024ull) != cudaSuccess)
            break;
        nf++;
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    printf("filler %d GB, free %.2f GiB\n", nf, (double)free_b / 1073741824.0);

    float *u = NULL;
    cudaError_t err = cudaMallocManaged(&u, bytes);
    if (err != cudaSuccess) {
        printf("managed alloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("managed 3GB allocated (pages on host, oversubscribed)\n");

    float *out = NULL;
    cudaMalloc(&out, (size_t)chunk * sizeof(float));
    float *back = (float *)malloc((size_t)chunk * sizeof(float));

    // A) device memset to host-resident UVM pages, then SM read
    for (int trial = 0; trial < 2; trial++) {
        const int off = trial * chunk;
        cudaMemset(u + off, 0x5A, (size_t)chunk * sizeof(float));  // device-side write
        cudaError_t e = cudaGetLastError();
        cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("memset trial %d err: %s\n", trial, cudaGetErrorString(e)); continue; }
        cudaMemset(out, 0, (size_t)chunk * sizeof(float));
        read_loop<<<512, 256>>>(u + off, out, chunk);
        e = cudaGetLastError();
        cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("read trial %d err: %s\n", trial, cudaGetErrorString(e)); continue; }
        cudaMemcpy(back, out, (size_t)chunk * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        const unsigned char pattern = 0x5A;
        for (int i = 0; i < chunk; i++) {
            const unsigned char *b = (const unsigned char *)&back[i];
            for (int j = 0; j < 4; j++) if (b[j] != pattern) { bad++; break; }
        }
        printf("UVM memset+SMread trial %d (offset %dMB): %s (%d bad)\n",
               trial, off * 4 / 1048576, bad ? "FAIL" : "OK  ", bad);
    }
    // B) kernel write to host-resident UVM pages, then SM read
    for (int trial = 0; trial < 2; trial++) {
        const int off = (trial + 2) * chunk;
        write_kernel<<<2048, 256>>>(u + off, 3.5f, chunk);
        cudaError_t e = cudaGetLastError();
        cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("kwrite trial %d err: %s\n", trial, cudaGetErrorString(e)); continue; }
        cudaMemset(out, 0, (size_t)chunk * sizeof(float));
        read_loop<<<512, 256>>>(u + off, out, chunk);
        cudaGetLastError();
        cudaDeviceSynchronize();
        cudaMemcpy(back, out, (size_t)chunk * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < chunk; i++) if (back[i] != 7.0f) bad++;
        printf("UVM kwrite+SMread trial %d: %s (%d bad)\n", trial, bad ? "FAIL" : "OK  ", bad);
    }
    // C) host write, then SM read (host->device migration on read)
    {
        const int off = 4 * chunk;
        for (int i = 0; i < chunk; i++) u[off + i] = (float)(i % 1000);  // host write
        cudaMemset(out, 0, (size_t)chunk * sizeof(float));
        read_loop<<<512, 256>>>(u + off, out, chunk);
        cudaGetLastError();
        cudaDeviceSynchronize();
        cudaMemcpy(back, out, (size_t)chunk * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < chunk; i++)
            if (back[i] != (float)(i % 1000) * 2.0f) bad++;
        printf("UVM hostwrite+SMread: %s (%d bad)\n", bad ? "FAIL" : "OK  ", bad);
    }

    free(back);
    cudaFree(out);
    cudaFree(u);
    for (int i = 0; i < nf; i++) cudaFree(filler[i]);
    return 0;
}
