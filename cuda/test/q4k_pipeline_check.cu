/* Why does the same GEMM measure 0.69 ms in one harness and 41.7 ms in another?
 *
 * Both call cublasGemmEx with identical shapes and types (T=576, K=5120, M=6144,
 * F16 operands, FP32 accumulate, TF32 math mode). The two things that differ are the
 * data in the buffers and what ran immediately before. This isolates them:
 *
 *   A  weights dequantised from Q4_K, activations filled with real values
 *   B  weights forced to the sweep's byte pattern, same activations
 *   C  weights dequantised, activations all zero
 *   D  A, but with the dequantiser re-run before every timed product
 *
 * and it also counts non-finite and subnormal values in the dequantised output, so a
 * bad data path shows up as a number instead of a guess.
 */
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_BYTES 144u

__device__ __forceinline__ void q4k_scale_min(int j, const unsigned char *q,
                                              int *d, int *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__global__ void q4k_to_f16_kernel(__half *out, const unsigned char *in, unsigned long long nblocks) {
    const unsigned long long b = (unsigned long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (b >= nblocks) return;
    const unsigned char *blk = in + b * BLOCK_BYTES;
    const unsigned lane = threadIdx.x & 31u;
    const float d = __half2float(*(const __half *)blk);
    const float dmin = __half2float(*(const __half *)(blk + 2));
    const unsigned char *scales = blk + 4;
    const unsigned char *qs = blk + 16;
    __half2 packed[4];
    for (unsigned v = 0; v < 8u; v += 2u) {
        float f[2];
        for (unsigned u = 0; u < 2u; u++) {
            const unsigned idx = lane * 8u + v + u;
            const unsigned group = idx >> 6;
            const unsigned within = idx & 63u;
            const unsigned sub = within >> 5;
            int s6 = 0, m6 = 0;
            q4k_scale_min((int)(group * 2u + sub), scales, &s6, &m6);
            const unsigned char byte = qs[group * 32u + (within & 31u)];
            const unsigned q4 = sub ? (unsigned)(byte >> 4) : (unsigned)(byte & 0xF);
            f[u] = d * (float)s6 * (float)q4 - dmin * (float)m6;
        }
        packed[v >> 1] = __floats2half2_rn(f[0], f[1]);
    }
    *(uint4 *)(out + b * 256u + lane * 8u) = *(const uint4 *)packed;
}

__global__ void fill_real(__half *x, size_t n, float seed, float scale) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = __float2half_rn(scale * (float)((int)((i + (size_t)seed) % 97) - 48));
}

__global__ void scan_half(const __half *x, size_t n, unsigned long long *bad,
                          unsigned long long *subnormal) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float f = __half2float(x[i]);
    if (!isfinite(f)) atomicAdd(bad, 1ull);
    else if (fabsf(f) < 6.1e-5f && f != 0.0f) atomicAdd(subnormal, 1ull);
}

static float ms_of(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    return ms;
}

int main(void) {
    const int T = 576, K = 5120, M = 6144;
    const size_t wy = (size_t)K * M;                 /* elements per weight matrix */
    const size_t wbytes = wy * BLOCK_BYTES / 256u;   /* Q4_K bytes */
    const size_t ach = (size_t)T * K;

    unsigned char *host = (unsigned char *)malloc(wbytes);
    if (!host) return 1;
    /* plausible super-block scales: d ~ 5e-4, dmin ~ 3e-4 */
    for (size_t b = 0; b < wbytes / BLOCK_BYTES; b++) {
        unsigned char *blk = host + b * BLOCK_BYTES;
        const __half hd = __float2half(0.0005f), hm = __float2half(0.0003f);
        const unsigned short db = *(const unsigned short *)&hd;
        const unsigned short mb = *(const unsigned short *)&hm;
        blk[0] = (unsigned char)(db & 0xFF);
        blk[1] = (unsigned char)(db >> 8);
        blk[2] = (unsigned char)(mb & 0xFF);
        blk[3] = (unsigned char)(mb >> 8);
        for (int i = 0; i < 12; i++) blk[4 + i] = (unsigned char)((b * 7u + i * 31u) & 0x3F);
        for (int i = 0; i < 128; i++) blk[16 + i] = (unsigned char)((b * 13u + i * 17u) & 0xFF);
    }

    unsigned char *dq4 = NULL;
    __half *df16 = NULL, *dpat = NULL, *dact = NULL;
    float *dy = NULL;
    unsigned long long *dbad = NULL, *dsub = NULL;
    cudaMalloc(&dq4, wbytes);
    cudaMalloc(&df16, wy * 2);
    cudaMalloc(&dpat, wy * 2);
    cudaMalloc(&dact, ach * 2);
    cudaMalloc(&dy, (size_t)T * M * 4);
    cudaMalloc(&dbad, sizeof(unsigned long long));
    cudaMalloc(&dsub, sizeof(unsigned long long));
    cudaMemcpy(dq4, host, wbytes, cudaMemcpyHostToDevice);
    cudaMemset(dpat, 1, wy * 2);                     /* the sweep's byte pattern */
    fill_real<<<(unsigned)((ach + 255) / 256), 256>>>(dact, ach, 3.0f, 0.02f);

    cublasHandle_t bh = NULL;
    cublasCreate(&bh);
    cublasSetMathMode(bh, CUBLAS_TF32_TENSOR_OP_MATH);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const float one = 1.0f, zero = 0.0f;
    const unsigned long long nblocks = wy / 256u;
    const unsigned warps = 8u;
    const unsigned grid = (unsigned)((nblocks + warps - 1u) / warps);
    const double flops = 2.0 * (double)T * K * M;

    q4k_to_f16_kernel<<<grid, warps * 32u>>>(df16, dq4, nblocks);
    cudaDeviceSynchronize();
    cudaMemset(dbad, 0, sizeof(unsigned long long));
    cudaMemset(dsub, 0, sizeof(unsigned long long));
    scan_half<<<(unsigned)((wy + 255) / 256), 256>>>(df16, wy, dbad, dsub);
    cudaDeviceSynchronize();
    unsigned long long hbad = 0, hsub = 0;
    cudaMemcpy(&hbad, dbad, sizeof(hbad), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hsub, dsub, sizeof(hsub), cudaMemcpyDeviceToHost);
    printf("dequantised weights: %llu non-finite, %llu subnormal, of %zu\n",
           hbad, hsub, wy);

    /* Plain helper rather than a macro: the pre-actions contain kernel launches,
     * whose commas break macro argument parsing. */
    const char *labels[3] = {"A weights=dequant, act=real",
                             "B weights=byte pattern, act=real",
                             "C weights=dequant, act=zero"};
    const __half *wt[3] = {df16, dpat, df16};
    for (int c = 0; c < 3; c++) {
        if (c == 2) cudaMemset(dact, 0, ach * 2);
        for (int r = 0; r < 2; r++)
            cublasGemmEx(bh, CUBLAS_OP_T, CUBLAS_OP_N, M, T, K, &one, wt[c], CUDA_R_16F,
                         K, dact, CUDA_R_16F, K, &zero, dy, CUDA_R_32F, M,
                         CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for (int r = 0; r < 5; r++)
            cublasGemmEx(bh, CUBLAS_OP_T, CUBLAS_OP_N, M, T, K, &one, wt[c], CUDA_R_16F,
                         K, dact, CUDA_R_16F, K, &zero, dy, CUDA_R_32F, M,
                         CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        const float ms = ms_of(e0, e1) / 5.0f;
        printf("  %-40s %.2f ms  %.0f GFLOPS\n", labels[c], ms, flops / (ms * 1e6));
    }
    /* D: is the slowness caused by the dequantiser running just before the product? */
    {
        cudaMemset(dact, 0, ach * 2);
        fill_real<<<(unsigned)((ach + 255) / 256), 256>>>(dact, ach, 3.0f, 0.02f);
        for (int r = 0; r < 2; r++) {
            q4k_to_f16_kernel<<<grid, warps * 32u>>>(df16, dq4, nblocks);
            cublasGemmEx(bh, CUBLAS_OP_T, CUBLAS_OP_N, M, T, K, &one, df16, CUDA_R_16F,
                         K, dact, CUDA_R_16F, K, &zero, dy, CUDA_R_32F, M,
                         CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
        }
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for (int r = 0; r < 5; r++) {
            q4k_to_f16_kernel<<<grid, warps * 32u>>>(df16, dq4, nblocks);
            cublasGemmEx(bh, CUBLAS_OP_T, CUBLAS_OP_N, M, T, K, &one, df16, CUDA_R_16F,
                         K, dact, CUDA_R_16F, K, &zero, dy, CUDA_R_32F, M,
                         CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
        }
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        const float ms = ms_of(e0, e1) / 5.0f;
        printf("  %-40s %.2f ms  (gemm %.0f GFLOPS incl. dequant)\n",
               "D dequant + gemm interleaved", ms, flops / (ms * 1e6));
    }
    return 0;
}
