// test_pinned5.cu — is the WDDM mapped-host-memory misread a RACE between
// cudaHostRegister (GART mapping setup) and the first GPU access?
// Matrix: sync/not before first read, cold vs warm reads, read sizes.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#ifdef _WIN32
#include <windows.h>
#endif

__global__ void read_f(const float *w, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = w[i] * 2.0f;
}

static int count_bad(const float *got, const float *want, int n) {
    int bad = 0;
    for (int i = 0; i < n; i++) if (got[i] != want[i]) bad++;
    return bad;
}
static void fill_data(float *p, int n, unsigned seed) {
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = (float)((int)(s >> 8) % 100000) * 0.5f;
    }
}

static int do_read(const float *dev, float *d_out, float *host, int off, int m) {
    cudaMemset(d_out, 0, m * sizeof(float));
    read_f<<<1, 256>>>(dev + off, d_out, m);
    cudaError_t err = cudaGetLastError();
    cudaDeviceSynchronize();
    if (err != cudaSuccess) return -1;
    float *back = (float *)malloc(m * sizeof(float));
    cudaMemcpy(back, d_out, m * sizeof(float), cudaMemcpyDeviceToHost);
    int bad = count_bad(back, host + off, m);
    free(back);
    return bad;
}

static void run_case(const char *name, float *host, int n, int m,
                     int sync_first, int sleep_us, int off) {
    cudaError_t err = cudaHostRegister(host, n * sizeof(float),
                                       cudaHostRegisterMapped | cudaHostRegisterReadOnly);
    void *dev = NULL;
    if (err == cudaSuccess) cudaHostGetDevicePointer(&dev, host, 0);
    if (err != cudaSuccess || !dev) {
        printf("%-32s register failed: %s\n", name, cudaGetErrorString(err));
        return;
    }
    if (sync_first) cudaDeviceSynchronize();
#ifdef _WIN32
    if (sleep_us > 0) Sleep(sleep_us / 1000 > 0 ? sleep_us / 1000 : 1);
#endif
    float *d_out = NULL;
    cudaMalloc(&d_out, m * sizeof(float));
    // want buffer doubles host in place? no: compare against host*2 manually
    int bad1 = do_read((const float *)dev, d_out, host, off, m);
    int bad2 = do_read((const float *)dev, d_out, host, off, m);
    int bad3 = do_read((const float *)dev, d_out, host, 0, m);
    printf("%-32s first=%s(%d) second=%s(%d) offset0=%s(%d)\n",
           name,
           bad1 == 0 ? "OK" : "FAIL", bad1,
           bad2 == 0 ? "OK" : "FAIL", bad2,
           bad3 == 0 ? "OK" : "FAIL", bad3);
    cudaFree(d_out);
    cudaHostUnregister(host);
}

int main() {
    const int n = 1024 * 256;
    const int m = 1024;
    const int off = 4096;   // page aligned, non-zero

    {   // 1: no sync, immediate read
        float *h = (float *)malloc(n * sizeof(float));
        fill_data(h, n, 111);
        run_case("1 no-sync", h, n, m, 0, 0, off);
        free(h);
    }
    {   // 2: device sync after register
        float *h = (float *)malloc(n * sizeof(float));
        fill_data(h, n, 222);
        run_case("2 sync-after-register", h, n, m, 1, 0, off);
        free(h);
    }
    {   // 3: sleep 2ms after register
        float *h = (float *)malloc(n * sizeof(float));
        fill_data(h, n, 333);
        run_case("3 sleep-2ms", h, n, m, 0, 2000, off);
        free(h);
    }
    {   // 4: register on page-aligned _aligned_malloc base, no sync
        float *h = (float *)malloc(n * sizeof(float) + 4096);
        fill_data(h, n, 444);
        run_case("4 no-sync (unaligned base)", h, n, m, 0, 0, off);
        free(h);
    }
    {   // 5: cudaHostAlloc (driver pinned) + immediate read
        float *h = NULL;
        cudaError_t err = cudaHostAlloc((void **)&h, n * sizeof(float), cudaHostAllocMapped);
        fill_data(h, n, 555);
        void *dev = NULL;
        cudaHostGetDevicePointer(&dev, h, 0);
        float *d_out = NULL;
        cudaMalloc(&d_out, m * sizeof(float));
        int bad1 = do_read((const float *)dev, d_out, h, off, m);
        int bad2 = do_read((const float *)dev, d_out, h, off, m);
        printf("%-32s first=%s(%d) second=%s(%d)\n",
               "5 hostalloc no-sync", bad1 == 0 ? "OK" : "FAIL", bad1,
               bad2 == 0 ? "OK" : "FAIL", bad2);
        cudaFree(d_out);
        cudaFreeHost(h);
    }
    return 0;
}
