// test_cuda.cu - minimal end-to-end CUDA check on Windows:
// device enumeration + properties + a tiny kernel roundtrip.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void addk(float *a, float b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b;
}

int main() {
    int ndev = 0;
    cudaError_t e = cudaGetDeviceCount(&ndev);
    if (e != cudaSuccess || ndev < 1) {
        printf("no cuda device: %s\n", cudaGetErrorString(e));
        return 1;
    }
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("device: %s | sm_%d%d | vram=%zu MiB | cudart=%d.%d.%d\n",
           p.name, p.major, p.minor, p.totalGlobalMem >> 20,
           CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10,
           CUDART_VERSION % 10);

    const int n = 1024;
    float *d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemset(d, 0, n * sizeof(float));
    addk<<<2, 512>>>(d, 1.5f, n);
    e = cudaDeviceSynchronize();
    float h[1024];
    cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
    printf("sync=%s | h[0]=%f | h[1023]=%f\n",
           cudaGetErrorString(e), h[0], h[1023]);
    cudaFree(d);
    int ok = (e == cudaSuccess && h[0] == 1.5f && h[1023] == 1.5f);
    printf(ok ? "CUDA OK\n" : "CUDA FAILED\n");
    return ok ? 0 : 2;
}
