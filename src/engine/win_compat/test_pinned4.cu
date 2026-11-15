// test_pinned4.cu — characterize the WDDM mapped-host-memory bug precisely:
//   A) is offset-0 reads from the registered base reliable (non-periodic data)?
//   B) are MULTIPLE independent registrations each reliable at offset 0?
//   C) is page N of one registration misread (page reordering hypothesis)?
//   D) cost of cudaHostRegister/unregister for many small blocks.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

__global__ void read_f(const float *w, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = w[i] * 2.0f;
}

static int count_bad(const float *got, const float *want, int n) {
    int bad = 0;
    for (int i = 0; i < n; i++) if (got[i] != want[i]) bad++;
    return bad;
}

// non-periodic pseudo-random data
static void fill_data(float *p, int n, unsigned seed) {
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = (float)((int)(s >> 8) % 100000) * 0.5f;
    }
}

int main() {
    const int n = 1024 * 256;   // 1MB floats = 4MB
    const int m = 1024;
    cudaError_t err;
    float *d_out = NULL;
    cudaMalloc(&d_out, n * sizeof(float));

    // A) single registration, offset-0 reads of 4KB slices at block starts
    {
        float *host = (float *)malloc(n * sizeof(float));
        fill_data(host, n, 12345);
        err = cudaHostRegister(host, n * sizeof(float),
                               cudaHostRegisterMapped | cudaHostRegisterReadOnly);
        void *dev = NULL;
        if (err == cudaSuccess) cudaHostGetDevicePointer(&dev, host, 0);
        if (err != cudaSuccess || !dev) {
            printf("A register failed: %s\n", cudaGetErrorString(err));
        } else {
            int total_bad = 0, blocks = 0;
            for (int blk = 0; blk < 8; blk++) {
                const int off = blk * m;   // 4KB-aligned offsets
                cudaMemset(d_out, 0, m * sizeof(float));
                read_f<<<1, 256>>>((const float *)dev + off, d_out, m);
                cudaGetLastError();
                cudaDeviceSynchronize();
                float *back = (float *)malloc(m * sizeof(float));
                cudaMemcpy(back, d_out, m * sizeof(float), cudaMemcpyDeviceToHost);
                for (int i = 0; i < m; i++)
                    if (back[i] != host[off + i] * 2.0f) total_bad++;
                blocks++;
                free(back);
            }
            printf("A offset-reads (8 x 4KB @ page-aligned offsets): %s (%d/%d bad)\n",
                   total_bad ? "FAIL" : "OK", total_bad, blocks * m);
            cudaHostUnregister(host);
        }
        free(host);
    }

    // B) many independent small registrations, each read at offset 0
    {
        const int nb = 8, per = 65536;   // 8 x 256KB
        float *hosts[8];
        void *devs[8];
        bool ok = true;
        for (int b = 0; b < nb; b++) {
            hosts[b] = (float *)malloc(per * sizeof(float));
            fill_data(hosts[b], per, 7000 + b);
            err = cudaHostRegister(hosts[b], per * sizeof(float),
                                   cudaHostRegisterMapped | cudaHostRegisterReadOnly);
            if (err != cudaSuccess) { ok = false; break; }
            cudaHostGetDevicePointer(&devs[b], hosts[b], 0);
        }
        if (!ok) {
            printf("B register failed: %s\n", cudaGetErrorString(err));
        } else {
            int total_bad = 0;
            for (int b = 0; b < nb; b++) {
                cudaMemset(d_out, 0, m * sizeof(float));
                read_f<<<1, 256>>>((const float *)devs[b], d_out, m);  // offset 0
                cudaGetLastError();
                cudaDeviceSynchronize();
                float *back = (float *)malloc(m * sizeof(float));
                cudaMemcpy(back, d_out, m * sizeof(float), cudaMemcpyDeviceToHost);
                for (int i = 0; i < m; i++)
                    if (back[i] != hosts[b][i] * 2.0f) total_bad++;
                free(back);
            }
            printf("B multi-register offset-0: %s (%d/%d bad)\n",
                   total_bad ? "FAIL" : "OK", total_bad, nb * m);
            for (int b = 0; b < nb; b++) {
                cudaHostUnregister(hosts[b]);
                free(hosts[b]);
            }
        }
    }

    // C) page-N reads inside ONE registration (page reordering probe)
    {
        float *host = (float *)malloc(n * sizeof(float));
        fill_data(host, n, 4242);
        err = cudaHostRegister(host, n * sizeof(float),
                               cudaHostRegisterMapped | cudaHostRegisterReadOnly);
        void *dev = NULL;
        if (err == cudaSuccess) cudaHostGetDevicePointer(&dev, host, 0);
        if (err != cudaSuccess || !dev) {
            printf("C register failed: %s\n", cudaGetErrorString(err));
        } else {
            float *back = (float *)malloc(m * sizeof(float));
            int page_bad[8] = {0};
            for (int p = 0; p < 8; p++) {
                const int off = p * m;   // each 4KB page start
                cudaMemset(d_out, 0, m * sizeof(float));
                read_f<<<1, 256>>>((const float *)dev + off, d_out, m);
                cudaGetLastError();
                cudaDeviceSynchronize();
                cudaMemcpy(back, d_out, m * sizeof(float), cudaMemcpyDeviceToHost);
                for (int i = 0; i < m; i++)
                    if (back[i] != host[off + i] * 2.0f) page_bad[p]++;
            }
            printf("C per-page (4KB) reads: ");
            for (int p = 0; p < 8; p++) printf("p%d=%s ", p, page_bad[p] ? "BAD" : "ok");
            printf("\n");
            free(back);
            cudaHostUnregister(host);
        }
        free(host);
    }

    // D) register/unregister cost for many small blocks
    {
        const int nb = 128, per = 262144;  // 128 x 1MB
        float **hosts = (float **)malloc(nb * sizeof(float *));
        for (int b = 0; b < nb; b++) hosts[b] = (float *)malloc(per * sizeof(float));
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0, 0);
        for (int b = 0; b < nb; b++) {
            err = cudaHostRegister(hosts[b], per * sizeof(float),
                                   cudaHostRegisterMapped | cudaHostRegisterReadOnly);
            if (err != cudaSuccess) break;
        }
        cudaEventRecord(t1, 0);
        cudaEventSynchronize(t1);
        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        printf("D register %d x 1MB: %.2f ms total, %.2f ms each\n",
               nb, ms, ms / nb);
        for (int b = 0; b < nb; b++) {
            cudaHostUnregister(hosts[b]);
            free(hosts[b]);
        }
        free(hosts);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    cudaFree(d_out);
    return 0;
}
