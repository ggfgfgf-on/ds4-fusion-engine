/* Which cuBLAS compute type actually reaches the tensor cores for F16 inputs.
 *
 * Two earlier measurements of the same shape disagreed by 56x (48.9 TFLOPS against
 * 0.87), which is the signature of a call that failed fast and was timed anyway.
 * This sweeps every compute type and prints the status for each, so the number that
 * survives is one that actually ran.
 */
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    const int T = 576, K = 5120, M = 6144;
    cublasHandle_t bh = NULL;
    if (cublasCreate(&bh) != CUBLAS_STATUS_SUCCESS) { puts("no handle"); return 1; }
    cublasSetMathMode(bh, CUBLAS_TF32_TENSOR_OP_MATH);
    __half *A = NULL, *B = NULL;
    float *C = NULL, *Ch = NULL;
    cudaMalloc(&A, (size_t)T * K * 2);
    cudaMalloc(&B, (size_t)K * M * 2);
    cudaMalloc(&C, (size_t)T * M * 4);
    cudaMalloc(&Ch, (size_t)T * M * 2);
    cudaMemset(A, 1, (size_t)T * K * 2);
    cudaMemset(B, 1, (size_t)K * M * 2);
    const float one = 1.0f, zero = 0.0f;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    struct { const char *name; cublasComputeType_t t; cudaDataType_t cdt; void *c; } cases[] = {
        {"32F_FAST_TF32 / f32 out", CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F, C},
        {"32F_FAST_16F  / f32 out", CUBLAS_COMPUTE_32F_FAST_16F,  CUDA_R_32F, C},
        {"32F_FAST_16F  / f16 out", CUBLAS_COMPUTE_32F_FAST_16F,  CUDA_R_16F, Ch},
        {"16F           / f16 out", CUBLAS_COMPUTE_16F,            CUDA_R_16F, Ch},
        {"32F           / f32 out", CUBLAS_COMPUTE_32F,            CUDA_R_32F, C},
    };
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        cublasStatus_t st = cublasGemmEx(bh, CUBLAS_OP_T, CUBLAS_OP_N, M, T, K,
                                         &one, B, CUDA_R_16F, K, A, CUDA_R_16F, K,
                                         &zero, cases[i].c, cases[i].cdt, M,
                                         cases[i].t, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            printf("  %-26s REJECTED (status %d)\n", cases[i].name, (int)st);
            continue;
        }
        cudaDeviceSynchronize();
        for (int r = 0; r < 2; r++) {
            cublasGemmEx(bh, CUBLAS_OP_T, CUBLAS_OP_N, M, T, K,
                         &one, B, CUDA_R_16F, K, A, CUDA_R_16F, K,
                         &zero, cases[i].c, cases[i].cdt, M,
                         cases[i].t, CUBLAS_GEMM_DEFAULT);
        }
        cudaEventRecord(e0);
        for (int r = 0; r < 5; r++) {
            cublasGemmEx(bh, CUBLAS_OP_T, CUBLAS_OP_N, M, T, K,
                         &one, B, CUDA_R_16F, K, A, CUDA_R_16F, K,
                         &zero, cases[i].c, cases[i].cdt, M,
                         cases[i].t, CUBLAS_GEMM_DEFAULT);
        }
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        ms /= 5.0f;
        printf("  %-26s %.2f ms  %.0f GFLOPS\n", cases[i].name, ms,
               (double)(2.0 * T * K * M) / (ms * 1e6));
    }
    return 0;
}
