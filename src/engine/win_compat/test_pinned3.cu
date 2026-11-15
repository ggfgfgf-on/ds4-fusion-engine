// test_pinned3.cu — two follow-ups on the WDDM mapped-host-memory bug:
//  1) UVM managed memory unaligned reads (different driver path);
//  2) page-aligned repack: copy unaligned source data into a page-aligned
//     pinned buffer with per-block 4KB alignment + padding, then read
//     aligned offsets on the GPU. If (2) passes, zero-copy can be revived
//     by repacking expert tables into aligned pinned staging.
//  3) characterize the unaligned misread pattern (what offset the GPU
//     actually reads).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

__global__ void read_f(const float *w, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = w[i] * 2.0f;
}
__global__ void read_f_off(const float *base, int off, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = base[i + off] * 2.0f;
}

static int count_bad(const float *got, const float *want, int n) {
    int bad = 0;
    for (int i = 0; i < n; i++) if (got[i] != want[i]) bad++;
    return bad;
}

int main() {
    const int n = 1024 * 256;      // 1M floats = 4MB
    const int off = 1234, m = 1024;
    cudaError_t err;
    float *d_out = NULL;
    cudaMalloc(&d_out, n * sizeof(float));

    // ---- 1) UVM managed ----
    {
        float *u = NULL;
        err = cudaMallocManaged(&u, n * sizeof(float));
        if (err != cudaSuccess) {
            printf("UVM alloc failed: %s\n", cudaGetErrorString(err));
        } else {
            for (int i = 0; i < n; i++) u[i] = (float)(i % 1000);
            // pages are on host; GPU reads fault them in
            cudaMemset(d_out, 0, m * sizeof(float));
            read_f_off<<<1, 256>>>(u, off, d_out, m);
            err = cudaGetLastError();
            cudaDeviceSynchronize();
            float *back = (float *)malloc(m * sizeof(float));
            cudaMemcpy(back, d_out, m * sizeof(float), cudaMemcpyDeviceToHost);
            float *want = (float *)malloc(m * sizeof(float));
            for (int i = 0; i < m; i++) want[i] = u[off + i] * 2.0f;
            int bad = err == cudaSuccess ? count_bad(back, want, m) : -1;
            printf("UVM unaligned : %s (%d)\n", bad == 0 ? "OK" : "FAIL", bad);
            free(back); free(want);
            cudaFree(u);
        }
    }

    // ---- 2) page-aligned repack ----
    {
        // source: unaligned region of a fake file buffer
        const int block_floats = 1024;              // 4KB per block
        const int nb = 64;                          // 64 blocks = 256KB
        float *src = (float *)malloc((size_t)nb * block_floats * sizeof(float) + 4096 + 64);
        // make data start 4KB+60 aligned but unaligned relative to pages
        uintptr_t base = (uintptr_t)src;
        uintptr_t aligned = (base + 4095) & ~(uintptr_t)4095;
        float *data = (float *)(aligned + 60);      // 60-byte unaligned start
        for (int i = 0; i < nb * block_floats; i++) data[i] = (float)(i % 977);

        // repack into page-aligned pinned buffer, one block per 4KB, zero pad
        float *pack = NULL;
        err = cudaHostAlloc((void **)&pack, (size_t)nb * block_floats * sizeof(float),
                            cudaHostAllocMapped);
        if (err != cudaSuccess) {
            printf("repack hostalloc failed: %s\n", cudaGetErrorString(err));
        } else {
            memset(pack, 0, (size_t)nb * block_floats * sizeof(float));
            memcpy(pack, data, (size_t)nb * block_floats * sizeof(float));
            void *dev = NULL;
            cudaHostGetDevicePointer(&dev, pack, 0);
            // read block 5 (aligned offset 5*1024 floats = 20KB)
            const int boff = 5 * block_floats;
            cudaMemset(d_out, 0, block_floats * sizeof(float));
            read_f<<<1, 256>>>((const float *)dev + boff, d_out, block_floats);
            err = cudaGetLastError();
            cudaDeviceSynchronize();
            float *back = (float *)malloc(block_floats * sizeof(float));
            cudaMemcpy(back, d_out, block_floats * sizeof(float), cudaMemcpyDeviceToHost);
            float *want = (float *)malloc(block_floats * sizeof(float));
            for (int i = 0; i < block_floats; i++) want[i] = data[boff + i] * 2.0f;
            int bad = err == cudaSuccess ? count_bad(back, want, block_floats) : -1;
            printf("repack aligned: %s (%d)\n", bad == 0 ? "OK" : "FAIL", bad);
            free(back); free(want);
            cudaFreeHost(pack);
        }
        free(src);
    }

    // ---- 3) characterize misread pattern on plain registered host mem ----
    {
        float *host = (float *)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) host[i] = (float)(i % 1000);
        err = cudaHostRegister(host, n * sizeof(float),
                               cudaHostRegisterMapped | cudaHostRegisterReadOnly);
        if (err != cudaSuccess) {
            printf("register failed: %s\n", cudaGetErrorString(err));
        } else {
            void *dev = NULL;
            cudaHostGetDevicePointer(&dev, host, 0);
            cudaMemset(d_out, 0, m * sizeof(float));
            read_f_off<<<1, 256>>>((const float *)dev, off, d_out, m);
            cudaGetLastError();
            cudaDeviceSynchronize();
            float *back = (float *)malloc(m * sizeof(float));
            cudaMemcpy(back, d_out, m * sizeof(float), cudaMemcpyDeviceToHost);
            // find actual source offset: back[i] == host[k]*2 for some k
            int first_ok = -1;
            for (int k = 0; k < n - m; k++) {
                bool all = true;
                for (int i = 0; i < 16; i++) {
                    if (back[i] != host[k + i] * 2.0f) { all = false; break; }
                }
                if (all) { first_ok = k; break; }
            }
            printf("misread pattern: requested off=%d (%d bytes), data matches host offset %d (%d bytes), delta=%d\n",
                   off, off * 4, first_ok, first_ok * 4, first_ok - off);
            // also check page-aligned multiples within registered range
            int badA = 0;
            cudaMemset(d_out, 0, m * sizeof(float));
            read_f<<<1, 256>>>((const float *)dev + 1024, d_out, m);  // 4KB aligned
            cudaGetLastError();
            cudaDeviceSynchronize();
            cudaMemcpy(back, d_out, m * sizeof(float), cudaMemcpyDeviceToHost);
            for (int i = 0; i < m; i++)
                if (back[i] != host[1024 + i] * 2.0f) badA++;
            printf("registered aligned(4KB): %s (%d)\n", badA ? "FAIL" : "OK", badA);
            free(back);
            cudaHostUnregister(host);
        }
        free(host);
    }

    cudaFree(d_out);
    return 0;
}
