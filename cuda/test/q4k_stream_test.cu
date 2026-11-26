/* Timing of the "stream one layer's weights and multiply" pipeline.
 *
 * The dequantiser is validated separately (q4k_dequant_test.cu); this measures what
 * the GPU-side prefill would actually cost per projection: copy the Q4_K bytes into
 * VRAM, dequantise them to F16, then let cuBLAS run the product with 576 tokens.
 *
 * Contents do not matter for a timing run, so the blocks are synthesised at the
 * right size rather than read from the model file. K and M come from argv.
 */
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static float ms_between(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    return ms;
}

int main(int argc, char **argv) {
    const unsigned T = (argc > 1) ? (unsigned)atoi(argv[1]) : 576u;
    const unsigned K = (argc > 2) ? (unsigned)atoi(argv[2]) : 5120u;
    const unsigned M = (argc > 3) ? (unsigned)atoi(argv[3]) : 6144u;
    if (!T || !K || !M || K % 256u) { puts("bad shape"); return 1; }

    const size_t wbytes = (size_t)M * (K / 256u) * BLOCK_BYTES;
    unsigned char *host = (unsigned char *)malloc(wbytes);
    if (!host) { puts("host alloc failed"); return 1; }
    for (size_t i = 0; i < wbytes; i++) host[i] = (unsigned char)(i * 37u + 11u);

    unsigned char *dq4 = NULL;
    __half *df16 = NULL, *dact = NULL;
    float *dy = NULL;
    if (cudaMalloc(&dq4, wbytes) != cudaSuccess) { puts("malloc q4 failed"); return 1; }
    if (cudaMalloc(&df16, (size_t)M * K * 2) != cudaSuccess) { puts("malloc f16 failed"); return 1; }
    if (cudaMalloc(&dact, (size_t)T * K * 2) != cudaSuccess) { puts("malloc act failed"); return 1; }
    if (cudaMalloc(&dy, (size_t)T * M * 4) != cudaSuccess) { puts("malloc out failed"); return 1; }
    cudaMemset(dact, 0, (size_t)T * K * 2);

    cublasHandle_t bh = NULL;
    if (cublasCreate(&bh) != CUBLAS_STATUS_SUCCESS) { puts("cublas init failed"); return 1; }
    cublasSetMathMode(bh, CUBLAS_TF32_TENSOR_OP_MATH);
    cudaEvent_t e0, e1, e2, e3, e4;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    cudaEventCreate(&e3); cudaEventCreate(&e4);
    const float one = 1.0f, zero = 0.0f;
    const unsigned long long nblocks = (unsigned long long)M * (K / 256u);
    const unsigned warps = 8u;
    const unsigned grid = (unsigned)((nblocks + warps - 1u) / warps);

    const int reps = 3;
    float t_copy = 0, t_deq = 0, t_gemm = 0;
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(e0);
        cudaMemcpy(dq4, host, wbytes, cudaMemcpyHostToDevice);
        cudaEventRecord(e1);
        q4k_to_f16_kernel<<<grid, warps * 32u>>>(df16, dq4, nblocks);
        cudaEventRecord(e2);
        cublasGemmEx(bh, CUBLAS_OP_T, CUBLAS_OP_N, (int)M, (int)T, (int)K,
                     &one, df16, CUDA_R_16F, (int)K, dact, CUDA_R_16F, (int)K,
                     &zero, dy, CUDA_R_32F, (int)M,
                     CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
        cudaEventRecord(e3);
        cudaEventSynchronize(e3);
        t_copy += ms_between(e0, e1);
        t_deq  += ms_between(e1, e2);
        t_gemm += ms_between(e2, e3);
    }
    const double flops = 2.0 * (double)T * K * M;
    const double total = (t_copy + t_deq + t_gemm) / reps;
    printf("T=%u K=%u M=%u   weights %.1f MB (Q4_K) -> %.1f MB (F16)\n",
           T, K, M, (double)wbytes / 1e6, (double)M * K * 2 / 1e6);
    printf("  copy   %.2f ms  (%.1f GB/s)\n", t_copy / reps,
           (double)wbytes / (t_copy / reps * 1e6));
    printf("  dequant %.2f ms  (%.1f GB/s written)\n", t_deq / reps,
           (double)M * K * 2 / (t_deq / reps * 1e6));
    printf("  gemm   %.2f ms  (%.0f GFLOPS)\n", t_gemm / reps, flops / (t_gemm / reps * 1e6));
    printf("  total  %.2f ms  (%.0f GFLOPS end to end)\n", total, flops / (total * 1e6));
    return 0;
}
