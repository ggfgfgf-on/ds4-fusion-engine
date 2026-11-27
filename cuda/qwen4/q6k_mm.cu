/* Q6_K weights as F16 on the device, and a product against them.
 *
 * The host spends twelve hundred milliseconds of every three-thousand-nine-hundred
 * millisecond prefill decoding these: four and four tenths gigabytes of six-bit weights,
 * three and a third teraflops of arithmetic, which is two and seven tenths teraflops - a
 * third of what the same silicon does on four-bit weights, because the six-bit layout
 * costs more per value to unpack and the wide path is already in use.
 *
 * The device has no such difficulty: unpacking is cheap next to a tensor-core product,
 * and the weights need no special residency because the arithmetic pays for the transfer
 * several times over. Four and four tenths gigabytes across the bus is a hundred and
 * fifty milliseconds at the measured rate; the same product at forty teraflops is ninety.
 *
 * Block layout is the standard one, the same fields the host's dequantiser reads:
 *   ql[128] at half*64 + (jj & 1 ? 32 : 0), high nibble when jj >= 2
 *   qh[64]  at half*32 + l, shifted right by jj * 2 and masked to two bits
 *   scales[16] at half*8 + l/16 + 2*jj
 * one super-block is two hundred and ten bytes covering two hundred and fifty-six values.
 */
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define Q6K_BYTES 210u

__global__ void q6k_to_f16(__half *out, const unsigned char *in,
                           unsigned long long nblocks) {
    const unsigned long long b =
        (unsigned long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (b >= nblocks) return;
    const unsigned char *blk = in + b * Q6K_BYTES;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned char *ql = blk;
    const unsigned char *qh = blk + 128u;
    const signed char *sc = (const signed char *)(blk + 192u);
    const float d = __half2float(*(const __half *)(blk + 208u));

    /* Eight consecutive values, which is lane*8 .. lane*8+7 within the super-block. */
    __half h[8];
    for (unsigned u = 0; u < 8u; u++) {
        const unsigned n = lane * 8u + u;
        const unsigned half = n >> 7;
        const unsigned within = n & 127u;
        const unsigned l = within & 31u;
        const unsigned jj = within >> 5;
        const unsigned ql_idx = half * 64u + (jj & 1u ? 32u : 0u) + l;
        const unsigned nib = (jj >= 2u) ? (ql[ql_idx] >> 4) : (ql[ql_idx] & 0xFu);
        const unsigned top = (qh[half * 32u + l] >> (jj * 2u)) & 3u;
        const int q = (int)(nib | (top << 4)) - 32;
        const int s = sc[half * 8u + (l >> 4) + 2u * jj];
        h[u] = __float2half_rn(d * (float)s * (float)q);
    }
    *(uint4 *)(out + b * 256u + lane * 8u) = *(const uint4 *)h;
}

#define Q6K_CACHE_SLOTS 256
typedef struct {
    uint64_t key;
    unsigned char *dev;
    size_t bytes;
} q6k_cache_entry;

static q6k_cache_entry g_q6cache[Q6K_CACHE_SLOTS];
static int g_q6cache_n = 0;
static size_t g_q6cache_bytes = 0;
static size_t g_q6cache_budget = 0;

extern "C" void ds4_gpu_q6k_cache_budget(uint64_t bytes) { g_q6cache_budget = bytes; }

static unsigned char *q6k_cached(uint64_t key, const void *host_q6, size_t bytes,
                                 long *hit) {
    *hit = 0;
    for (int i = 0; i < g_q6cache_n; i++)
        if (g_q6cache[i].key == key) { *hit = 1; return g_q6cache[i].dev; }
    if (!g_q6cache_budget || g_q6cache_bytes + bytes > g_q6cache_budget) return NULL;
    if (g_q6cache_n >= Q6K_CACHE_SLOTS) return NULL;
    unsigned char *dev = NULL;
    if (cudaMalloc((void **)&dev, bytes) != cudaSuccess) return NULL;
    if (cudaMemcpy(dev, host_q6, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(dev);
        return NULL;
    }
    g_q6cache[g_q6cache_n].key = key;
    g_q6cache[g_q6cache_n].dev = dev;
    g_q6cache[g_q6cache_n].bytes = bytes;
    g_q6cache_n++;
    g_q6cache_bytes += bytes;
    return dev;
}

extern "C" size_t ds4_gpu_q6k_cache_held(void) { return g_q6cache_bytes; }

typedef struct {
    cublasHandle_t cublas;
    unsigned char *q4src;    /* the packed weights, on the device */
    __half *w;
    __half *x;
    float *xf;
    float *y;
    size_t cap_q4src, cap_w, cap_x, cap_xf, cap_y;
    int ready;
} q6k_ctx;

static q6k_ctx g_q6;

static int q6grow(void **p, size_t *cap, size_t need) {
    if (*cap >= need) return 1;
    if (*p) cudaFree(*p);
    *p = NULL;
    *cap = 0;
    if (cudaMalloc(p, need) != cudaSuccess) return 0;
    *cap = need;
    return 1;
}

static __global__ void q6k_f32_to_f16(__half *dst, const float *src,
                                      unsigned long long n) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn(src[i]);
}

extern "C" int ds4_gpu_q6k_mm_keyed(uint64_t key, const void *host_q6, uint32_t K,
                                    uint32_t M, const float *host_act, uint32_t T,
                                    float *host_out);

extern "C" int ds4_gpu_q6k_mm(const void *host_q6, uint32_t K, uint32_t M,
                              const float *host_act, uint32_t T, float *host_out) {
    return ds4_gpu_q6k_mm_keyed(0, host_q6, K, M, host_act, T, host_out);
}

extern "C" int ds4_gpu_q6k_mm_keyed(uint64_t key, const void *host_q6, uint32_t K,
                                    uint32_t M, const float *host_act, uint32_t T,
                                    float *host_out) {
    if (!g_q6.ready) {
        if (cublasCreate(&g_q6.cublas) != CUBLAS_STATUS_SUCCESS) return 0;
        /* No cublasSetMathMode call here: the four-bit module does not make one, and
         * making one with CUBLAS_TF32_TENSOR_OP_MATH is what turned this shape into
         * CUBLAS_STATUS_NOT_SUPPORTED. The compute type on the product is what selects
         * the arithmetic. */
        g_q6.ready = 1;
    }
    if (!host_q6 || !K || !M || !T || (K % 256u)) return 0;
    const size_t xh = (size_t)T * K;
    const size_t qbytes = (size_t)M * (K / 256u) * Q6K_BYTES;
    {
        static int first = 0;
        if (first++ == 0) {
            size_t vf = 0, vt = 0;
            cudaMemGetInfo(&vf, &vt);
            fprintf(stderr, "q6k: device %.0f MB free of %.0f MB at first call\n",
                    (double)vf / 1048576.0, (double)vt / 1048576.0);
        }
    }
    {
        static int reported = 0;
        const int ok = q6grow((void **)&g_q6.q4src, &g_q6.cap_q4src, qbytes) &&
                       q6grow((void **)&g_q6.w, &g_q6.cap_w, (size_t)M * K * 2) &&
                       q6grow((void **)&g_q6.x, &g_q6.cap_x, xh * 2) &&
                       q6grow((void **)&g_q6.xf, &g_q6.cap_xf, xh * 4) &&
                       q6grow((void **)&g_q6.y, &g_q6.cap_y, (size_t)T * M * 4);
        if (!ok) {
            if (reported++ < 3) {
                size_t vf = 0, vt = 0;
                cudaMemGetInfo(&vf, &vt);
                fprintf(stderr, "q6k: allocation failed K=%u M=%u T=%u (%.0f MB free)\n",
                        K, M, T, (double)vf / 1048576.0);
            }
            return 0;
        }
    }

    static double t_upload = 0.0, t_dequant = 0.0, t_gemm = 0.0, t_down = 0.0;
    static long calls = 0;
    cudaEvent_t e0, e1, e2, e3, e4;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    cudaEventCreate(&e3); cudaEventCreate(&e4);
    cudaEventRecord(e0);
    const unsigned long long nb = (unsigned long long)M * (K / 256u);
    const unsigned warps = 8u;
    long hit = 0;
    unsigned char *cached = key ? q6k_cached(key, host_q6, qbytes, &hit) : NULL;
    if (cached) {
        /* already on the device: nothing to copy for the weights */
    } else {
        cudaMemcpy(g_q6.q4src, host_q6, qbytes, cudaMemcpyHostToDevice);
        cached = g_q6.q4src;
    }
    cudaMemcpy(g_q6.xf, host_act, xh * 4, cudaMemcpyHostToDevice);
    q6k_f32_to_f16<<<(unsigned)((xh + 255) / 256), 256>>>(g_q6.x, g_q6.xf, xh);
    q6k_to_f16<<<(unsigned)((nb + warps - 1u) / warps), warps * 32u>>>(
        g_q6.w, cached, nb);
    cudaEventRecord(e1);
    {
        const cudaError_t de = cudaGetLastError();
        if (de != cudaSuccess) {
            static int rep = 0;
            if (rep++ < 3)
                fprintf(stderr, "q6k: dequant launch failed: %s\n", cudaGetErrorString(de));
            return 0;
        }
    }
    const float one = 1.0f, zero = 0.0f;
    /* Rows in tiles. Ten thousand two hundred and forty rows comes back
     * CUBLAS_STATUS_NOT_SUPPORTED from this handle while five thousand one hundred and
     * twenty and twelve thousand two hundred and eighty-eight both succeed, so the shape
     * is cut rather than the question pursued - the tiling costs one extra call per tile
     * and nothing else. */
    cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
    const uint32_t tile = 5120u;
    for (uint32_t r0 = 0; r0 < M && st == CUBLAS_STATUS_SUCCESS; r0 += tile) {
        const uint32_t rows = (M - r0) < tile ? (M - r0) : tile;
        /* C is column-major M x T with leading dimension M, so a row tile starts at
         * offset r0 and keeps that leading dimension. */
        st = cublasGemmEx(g_q6.cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)rows, (int)T,
                          (int)K, &one, g_q6.w + (size_t)r0 * K, CUDA_R_16F, (int)K,
                          g_q6.x, CUDA_R_16F, (int)K, &zero, g_q6.y + r0,
                          CUDA_R_32F, (int)M, CUBLAS_COMPUTE_32F_FAST_TF32,
                          CUBLAS_GEMM_DEFAULT);
    }
    cudaEventRecord(e2);
    if (st != CUBLAS_STATUS_SUCCESS) {
        static int reported = 0;
        if (reported++ < 3)
            fprintf(stderr, "q6k: gemm failed K=%u M=%u T=%u status=%d err=%s\n", K, M, T,
                    (int)st, cudaGetErrorString(cudaGetLastError()));
        return 0;
    }
    cudaEventRecord(e3);
    const cudaError_t rb =
        cudaMemcpy(host_out, g_q6.y, (size_t)T * M * 4, cudaMemcpyDeviceToHost);
    cudaEventRecord(e4);
    cudaEventSynchronize(e4);
    {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1); t_upload += ms;
        cudaEventElapsedTime(&ms, e1, e2); t_dequant += ms;
        cudaEventElapsedTime(&ms, e2, e3); t_gemm += ms;
        cudaEventElapsedTime(&ms, e3, e4); t_down += ms;
        calls++;
        if ((calls % 50) == 0)
            fprintf(stderr,
                    "q6k: %ld calls | upload %.0f | dequant %.0f | gemm %.0f | down %.0f ms "
                    "(total %.0f ms)\n", calls, t_upload, t_dequant, t_gemm, t_down,
                    t_upload + t_dequant + t_gemm + t_down);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
    cudaEventDestroy(e3); cudaEventDestroy(e4);
    return rb == cudaSuccess;
}

#ifndef Q6K_NO_MAIN
#include <math.h>
/* Host reference over the same field mapping, and a comparison against the kernel. The
 * layout is the one place a mistake would be silent: a transposed nibble still produces
 * plausible-looking numbers that are simply the wrong ones. */
static void ref_dequant(const unsigned char *blk, float *y) {
    const unsigned char *ql = blk;
    const unsigned char *qh = blk + 128u;
    const signed char *sc = (const signed char *)(blk + 192u);
    const float d = *(const __half *)(blk + 208u);
    for (unsigned n = 0; n < 256u; n++) {
        const unsigned half = n >> 7, within = n & 127u;
        const unsigned l = within & 31u, jj = within >> 5;
        const unsigned qi = half * 64u + (jj & 1u ? 32u : 0u) + l;
        const unsigned nib = (jj >= 2u) ? (ql[qi] >> 4) : (ql[qi] & 0xFu);
        const unsigned top = (qh[half * 32u + l] >> (jj * 2u)) & 3u;
        const int q = (int)(nib | (top << 4)) - 32;
        y[n] = d * (float)sc[half * 8u + (l >> 4) + 2u * jj] * (float)q;
    }
}

int main(void) {
    const unsigned blocks = 64;
    unsigned char *h = (unsigned char *)malloc((size_t)blocks * Q6K_BYTES);
    for (unsigned b = 0; b < blocks; b++) {
        unsigned char *blk = h + (size_t)b * Q6K_BYTES;
        memset(blk, 0, Q6K_BYTES);
        for (unsigned i = 0; i < 128u; i++) blk[i] = (unsigned char)(rand() & 0xFF);
        for (unsigned i = 0; i < 64u; i++) blk[128u + i] = (unsigned char)(rand() & 0xFF);
        for (unsigned i = 0; i < 16u; i++) blk[192u + i] = (unsigned char)(1 + (rand() % 40));
        const float d = 0.001f + 0.0005f * (float)(b % 7);
        const __half hd = __float2half_rn(d);
        memcpy(blk + 208u, &hd, 2);
    }
    unsigned char *hdev = NULL;
    __half *dev = NULL;
    cudaMalloc((void **)&hdev, (size_t)blocks * Q6K_BYTES);
    cudaMemcpy(hdev, h, (size_t)blocks * Q6K_BYTES, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&dev, (size_t)blocks * 256u * 2);
    const unsigned warps = 8u;
    q6k_to_f16<<<(blocks + warps - 1u) / warps, warps * 32u>>>(dev, hdev,
                                                               (unsigned long long)blocks);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("kernel: %s\n", cudaGetErrorString(e)); return 1; }
    __half *out = (__half *)malloc((size_t)blocks * 256u * 2);
    cudaMemcpy(out, dev, (size_t)blocks * 256u * 2, cudaMemcpyDeviceToHost);
    double worst = 0.0, mean = 0.0;
    unsigned long long n = 0, covered = 0;
    for (unsigned b = 0; b < blocks; b++) {
        float ref[256];
        ref_dequant(h + (size_t)b * Q6K_BYTES, ref);
        for (unsigned i = 0; i < 256u; i++) {
            const double g = (double)__half2float(out[(size_t)b * 256u + i]);
            const double r = (double)ref[i];
            const double err = fabs(g - r);
            const double denom = fabs(r) > 1e-9 ? fabs(r) : 1.0;
            if (err / denom > worst) worst = err / denom;
            mean += err;
            n++;
            if (fabs(g) > 1e-9) covered++;
        }
    }
    printf("values %llu  nonzero %llu  worst rel %.3e  mean abs %.3e\n", n, covered,
           worst, mean / (double)n);
    printf("%s\n", worst < 1e-2 ? "PASS" : "FAIL");

    /* The shape the engine presents first, called through the same entry point the
     * engine uses, to tell a shape problem from an integration one. */
    {
        const uint32_t K = 5120u, M = 10240u, T = 305u;
        unsigned char *wq = (unsigned char *)malloc((size_t)M * (K / 256u) * Q6K_BYTES);
        float *act = (float *)malloc((size_t)T * K * 4);
        float *res = (float *)malloc((size_t)T * M * 4);
        for (size_t i = 0; i < (size_t)M * (K / 256u) * Q6K_BYTES; i++)
            wq[i] = (unsigned char)(rand() & 0xFF);
        for (size_t i = 0; i < (size_t)M * (K / 256u); i++) {
            unsigned char *blk = wq + i * Q6K_BYTES;
            for (unsigned j = 0; j < 16u; j++) blk[192u + j] = (unsigned char)(1 + (rand() % 40));
            const __half hd = __float2half_rn(0.001f);
            memcpy(blk + 208u, &hd, 2);
        }
        for (size_t i = 0; i < (size_t)T * K; i++) act[i] = (float)(rand() % 1000) / 1000.0f;
        const int ok = ds4_gpu_q6k_mm(wq, K, M, act, T, res);
        printf("shape test K=%u M=%u T=%u -> %s (res[0]=%.4f)\n", K, M, T,
               ok ? "OK" : "FAILED", ok ? (double)res[0] : 0.0);
        /* Sweep the compute type and the token count against a hand-made pair of
         * half-precision matrices, which is the exact call the module makes. */
        {
            __half *A = NULL, *B = NULL;
            float *C = NULL;
            cublasHandle_t h;
            cublasCreate(&h);
            cudaMalloc((void **)&A, (size_t)M * K * 2);
            cudaMalloc((void **)&B, (size_t)T * K * 2);
            cudaMalloc((void **)&C, (size_t)M * T * 4);
            const cublasComputeType_t tys[4] = {
                CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_32F_FAST_16F,
                CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_COMPUTE_16F};
            const char *nm[4] = {"32F", "32F_FAST_16F", "32F_FAST_TF32", "16F"};
            for (int ti = 0; ti < 4; ti++) {
                const float one = 1.0f, zero = 0.0f;
                const cublasStatus_t st = cublasGemmEx(
                    h, CUBLAS_OP_T, CUBLAS_OP_N, (int)M, (int)T, (int)K, &one, A,
                    CUDA_R_16F, (int)K, B, CUDA_R_16F, (int)K, &zero, C, CUDA_R_32F,
                    (int)M, tys[ti], CUBLAS_GEMM_DEFAULT);
                printf("  compute %-14s -> %d\n", nm[ti], (int)st);
            }
            cudaFree(A); cudaFree(B); cudaFree(C);
            cublasDestroy(h);
        }
        free(wq); free(act); free(res);
    }
    return 0;
}
#endif
