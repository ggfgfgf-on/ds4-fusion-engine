/* Streaming Q4_K projection: upload, dequantise to F16, multiply - one module.
 *
 * This is the piece a GPU prefill would call once per projection per layer. It stays
 * self-contained (no engine, no GGUF, no shared handle) so it can be checked and
 * timed directly:
 *
 *   - correctness: a shape small enough for a double-precision CPU reference
 *   - speed: the shape a real layer projection has at 576 prompt tokens
 *
 * Why it is worth doing at all: the Q4_K dot-product kernel issues about four TFLOPS
 * on this part, while dequantising to F16 and calling cuBLAS measured 36.6 TFLOPS
 * end to end for the same projection.
 */
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define BLOCK_BYTES 144u

/* ---- kernels ------------------------------------------------------------- */

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

__global__ void q4k_to_f16_kernel(__half *out, const unsigned char *in,
                                  unsigned long long nblocks) {
    const unsigned long long b =
        (unsigned long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
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

__global__ void f32_to_f16_kernel(__half *dst, const float *src, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn(src[i]);
}

/* ---- streaming projection ------------------------------------------------- */

typedef struct {
    cublasHandle_t cublas;
    unsigned char *dev_q4;
    __half *dev_w;
    __half *dev_x;
    float *dev_xf;      /* activations arrive as float and are converted on device */
    float *dev_y;
    size_t cap_q4, cap_w, cap_x, cap_xf, cap_y;
} q4k_stream;

static int grow(void **p, size_t *cap, size_t need) {
    if (*cap >= need) return 1;
    if (*p) cudaFree(*p);
    *p = NULL;
    *cap = 0;
    if (cudaMalloc(p, need) != cudaSuccess) return 0;
    *cap = need;
    return 1;
}

static int q4k_stream_open(q4k_stream *s) {
    memset(s, 0, sizeof(*s));
    if (cublasCreate(&s->cublas) != CUBLAS_STATUS_SUCCESS) return 0;
    /* default math mode: the F16 operands do not need TF32 emulation */
    return 1;
}

static void q4k_stream_close(q4k_stream *s) {
    if (s->cublas) cublasDestroy(s->cublas);
    if (s->dev_q4) cudaFree(s->dev_q4);
    if (s->dev_w) cudaFree(s->dev_w);
    if (s->dev_x) cudaFree(s->dev_x);
    if (s->dev_xf) cudaFree(s->dev_xf);
    if (s->dev_y) cudaFree(s->dev_y);
    memset(s, 0, sizeof(*s));
}

/* out[T][M] = act[T][K] * W[K][M], with W supplied as Q4_K bytes on the host.
 * The stages are separable so a caller can overlap the upload of the next layer
 * with the multiply of this one. */
static int q4k_stream_gemm(q4k_stream *s, const unsigned char *host_q4k,
                           unsigned K, unsigned M, const float *host_act, unsigned T,
                           float *host_out, double *out_stage_ms) {
    if (!K || !M || !T || (K % 256u)) return 0;
    const size_t q4_bytes = (size_t)M * (K / 256u) * BLOCK_BYTES;
    const size_t w_half = (size_t)M * K;
    const size_t x_half = (size_t)T * K;
    if (!grow((void **)&s->dev_q4, &s->cap_q4, q4_bytes)) return 0;
    if (!grow((void **)&s->dev_w, &s->cap_w, w_half * 2)) return 0;
    if (!grow((void **)&s->dev_x, &s->cap_x, x_half * 2)) return 0;
    if (!grow((void **)&s->dev_xf, &s->cap_xf, x_half * 4)) return 0;
    if (!grow((void **)&s->dev_y, &s->cap_y, (size_t)T * M * 4)) return 0;

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const float one = 1.0f, zero = 0.0f;
    float t_up = 0.0f, t_cvt = 0.0f, t_deq = 0.0f, t_gemm = 0.0f;
    if (getenv("Q4K_STREAM_DBG"))
        fprintf(stderr, "dbg K=%u M=%u T=%u q4=%zu wh=%zu xh=%zu | q4=%p w=%p x=%p y=%p\n",
                K, M, T, q4_bytes, w_half, x_half,
                (void *)s->dev_q4, (void *)s->dev_w, (void *)s->dev_x, (void *)s->dev_y);

    cudaEventRecord(e0);
    {
        const cudaError_t cp = cudaMemcpy(s->dev_q4, host_q4k, q4_bytes, cudaMemcpyHostToDevice);
        if (cp != cudaSuccess) {
            fprintf(stderr, "q4k_stream: upload of %zu bytes failed: %s\n", q4_bytes,
                    cudaGetErrorString(cp));
            return 0;
        }
        const cudaError_t kc = cudaPeekAtLastError();
        if (kc != cudaSuccess) {
            fprintf(stderr, "q4k_stream: cuda state before the convert kernel: %s\n",
                    cudaGetErrorString(kc));
            return 0;
        }
        const cudaError_t au = cudaMemcpy(s->dev_xf, host_act, x_half * 4,
                                          cudaMemcpyHostToDevice);
        if (au != cudaSuccess) {
            fprintf(stderr, "q4k_stream: activation upload failed: %s\n", cudaGetErrorString(au));
            return 0;
        }
        f32_to_f16_kernel<<<(unsigned)((x_half + 255) / 256), 256>>>(s->dev_x, s->dev_xf,
                                                                    x_half);
        const cudaError_t kk = cudaPeekAtLastError();
        if (kk != cudaSuccess) {
            fprintf(stderr, "q4k_stream: convert kernel launch failed: %s\n",
                    cudaGetErrorString(kk));
            return 0;
        }
        const cudaError_t ks = cudaDeviceSynchronize();
        if (ks != cudaSuccess) {
            fprintf(stderr, "q4k_stream: convert kernel failed: %s (n=%zu, grid=%u)\n",
                    cudaGetErrorString(ks), x_half, (unsigned)((x_half + 255) / 256));
            return 0;
        }
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    cudaEventElapsedTime(&t_up, e0, e1);
    if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "stage upload/convert faulted\n"); return 0; }

    const unsigned long long nblocks = (unsigned long long)M * (K / 256u);
    const unsigned warps = 8u;
    const unsigned grid = (unsigned)((nblocks + warps - 1u) / warps);
    cudaEventRecord(e0);
    q4k_to_f16_kernel<<<grid, warps * 32u>>>(s->dev_w, s->dev_q4, nblocks);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    cudaEventElapsedTime(&t_deq, e0, e1);
    if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "stage dequant faulted\n"); return 0; }

    cudaEventRecord(e0);
    const cublasStatus_t st =
        cublasGemmEx(s->cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)M, (int)T, (int)K, &one,
                     s->dev_w, CUDA_R_16F, (int)K, s->dev_x, CUDA_R_16F, (int)K, &zero,
                     s->dev_y, CUDA_R_32F, (int)M, CUBLAS_COMPUTE_32F_FAST_TF32,
                     CUBLAS_GEMM_DEFAULT);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    cudaEventElapsedTime(&t_gemm, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "q4k_stream: cublas status %d (%s) for M=%u N=%u K=%u\n",
                (int)st, cublasGetStatusString(st), M, T, K);
        const cudaError_t ce = cudaGetLastError();
        if (ce != cudaSuccess) fprintf(stderr, "q4k_stream: cuda says %s\n", cudaGetErrorString(ce));
        return 0;
    }
    if (host_out) cudaMemcpy(host_out, s->dev_y, (size_t)T * M * 4, cudaMemcpyDeviceToHost);
    if (out_stage_ms) {
        out_stage_ms[0] = t_up;
        out_stage_ms[1] = t_cvt;
        out_stage_ms[2] = t_deq;
        out_stage_ms[3] = t_gemm;
    }
    return 1;
}

/* ---- CPU reference ------------------------------------------------------- */

static float half_bits_to_float(unsigned short h) {
    const unsigned sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    float v;
    if (exp == 0) v = (float)man * (1.0f / 16777216.0f);
    else if (exp == 31) v = man ? NAN : INFINITY;
    else v = (1.0f + (float)man / 1024.0f) * exp2f((float)((int)exp - 15));
    return sign ? -v : v;
}

static void ref_scale_min(int j, const unsigned char *q, int *d, int *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (q[j + 4] & 0xF) | (((int)q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | (((int)q[j] >> 6) << 4);
    }
}

/* row r of the weight matrix, decoded; K/RW values */
static void ref_dequant_row(const unsigned char *q4, unsigned K, unsigned r, float *out) {
    const size_t blocks = K / 256u;
    const unsigned char *base = q4 + (size_t)r * blocks * BLOCK_BYTES;
    for (size_t b = 0; b < blocks; b++) {
        const unsigned char *blk = base + b * BLOCK_BYTES;
        const float d = half_bits_to_float(*(const unsigned short *)blk);
        const float dmin = half_bits_to_float(*(const unsigned short *)(blk + 2));
        const unsigned char *scales = blk + 4, *qs = blk + 16;
        float *y = out + b * 256u;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            int s = 0, m = 0;
            ref_scale_min(is + 0, scales, &s, &m);
            const float d1 = d * (float)s, m1 = dmin * (float)m;
            ref_scale_min(is + 1, scales, &s, &m);
            const float d2 = d * (float)s, m2 = dmin * (float)m;
            const unsigned char *q = qs + (j >> 1);
            for (int l = 0; l < 32; l++) *y++ = d1 * (float)(q[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) *y++ = d2 * (float)(q[l] >> 4) - m2;
            is += 2;
        }
    }
}

/* ---- driver -------------------------------------------------------------- */

static void make_q4k(unsigned char *dst, unsigned K, unsigned M, unsigned seed) {
    const size_t blocks = (size_t)M * (K / 256u);
    for (size_t b = 0; b < blocks; b++) {
        unsigned char *blk = dst + b * BLOCK_BYTES;
        const __half hd = __float2half(0.0005f + 0.0002f * (float)((b + seed) % 7) / 7.0f);
        const __half hm = __float2half(0.0003f + 0.0001f * (float)((b + seed) % 5) / 5.0f);
        const unsigned short db = *(const unsigned short *)&hd;
        const unsigned short mb = *(const unsigned short *)&hm;
        blk[0] = (unsigned char)(db & 0xFF);
        blk[1] = (unsigned char)(db >> 8);
        blk[2] = (unsigned char)(mb & 0xFF);
        blk[3] = (unsigned char)(mb >> 8);
        for (int i = 0; i < 12; i++)
            blk[4 + i] = (unsigned char)(((b * 7u + (unsigned)i * 31u + seed) & 0x3F));
        for (int i = 0; i < 128; i++)
            blk[16 + i] = (unsigned char)(((b * 13u + (unsigned)i * 17u + seed) & 0xFF));
    }
}

int main(void) {
    /* Correctness first, on a shape a double-precision CPU reference can do. */
    const unsigned cK = 512, cM = 64, cT = 8;
    const size_t c_q4 = (size_t)cM * (cK / 256u) * BLOCK_BYTES;
    unsigned char *cq4 = (unsigned char *)malloc(c_q4);
    float *cx = (float *)malloc((size_t)cT * cK * sizeof(float));
    float *cy = (float *)malloc((size_t)cT * cM * sizeof(float));
    if (!cq4 || !cx || !cy) { puts("alloc failed"); return 1; }
    make_q4k(cq4, cK, cM, 1u);
    for (size_t i = 0; i < (size_t)cT * cK; i++)
        cx[i] = 0.02f * (float)((int)(i % 61) - 30);

    q4k_stream s;
    if (!q4k_stream_open(&s)) { puts("stream open failed"); return 1; }
    const int small_ok = q4k_stream_gemm(&s, cq4, cK, cM, cx, cT, cy, NULL);
    if (!small_ok) puts("small-shape gemm failed; continuing to the real shape");

    double worst = 0.0, scale = 1e-30;
    if (small_ok) {
        float *row = (float *)malloc((size_t)cK * sizeof(float));
        double *ref = (double *)malloc((size_t)cT * cM * sizeof(double));
        if (!row || !ref) { puts("alloc failed"); return 1; }
        for (unsigned r = 0; r < cM; r++) {
            ref_dequant_row(cq4, cK, r, row);
            for (unsigned t = 0; t < cT; t++) {
                double acc = 0.0;
                for (unsigned k = 0; k < cK; k++) acc += (double)cx[(size_t)t * cK + k] * (double)row[k];
                ref[(size_t)t * cM + r] = acc;
                if (fabs(acc) > scale) scale = fabs(acc);
            }
        }
        for (unsigned t = 0; t < cT; t++)
            for (unsigned r = 0; r < cM; r++) {
                const double a = ref[(size_t)t * cM + r];
                const double b = cy[(size_t)t * cM + r];
                const double rel = fabs(a - b) / scale;
                if (rel > worst) worst = rel;
            }
        free(row);
        free(ref);
    }
    printf("correctness  K=%u M=%u T=%u   worst relative error %.3e (scale-normalised)\n",
           cK, cM, cT, worst);
    /* F16 storage of the weights and of the activations costs about 2*2^-11. */
    const int ok = worst < 5e-3;
    printf("%s\n", ok ? "PASS" : "FAIL");

    /* Then the shape a real projection has. */
    {
        const unsigned K = 5120, M = 6144, T = 576;
        const size_t q4 = (size_t)M * (K / 256u) * BLOCK_BYTES;
        unsigned char *w = (unsigned char *)malloc(q4);
        float *x = (float *)malloc((size_t)T * K * sizeof(float));
        float *y = (float *)malloc((size_t)T * M * sizeof(float));
        if (!w || !x || !y) { puts("alloc failed"); return 1; }
        make_q4k(w, K, M, 3u);
        for (size_t i = 0; i < (size_t)T * K; i++)
            x[i] = 0.02f * (float)((int)(i % 61) - 30);
        double best = 1e30, up = 0, deq = 0, gemm = 0;
        for (int rep = 0; rep < 3; rep++) {
            double st[4] = {0, 0, 0, 0};
            if (!q4k_stream_gemm(&s, w, K, M, x, T, y, st)) { puts("gemm failed"); return 1; }
            const double total = st[0] + st[2] + st[3];
            if (total < best) best = total;
            up = st[0]; deq = st[2]; gemm = st[3];
        }
        const double fl = 2.0 * (double)T * K * M;
        printf("timing       K=%u M=%u T=%u  weights %.1f MB Q4_K -> %.1f MB F16\n",
               K, M, T, (double)q4 / 1e6, (double)M * K * 2 / 1e6);
        printf("  upload+convert %.2f ms | dequant %.2f ms | gemm %.2f ms\n", up, deq, gemm);
        printf("  per projection %.2f ms  -> %.1f GFLOPS, %.0f GB/s of weight traffic\n",
               best, fl / (best * 1e6), (double)q4 / (best * 1e6));
        free(w); free(x); free(y);
    }
    q4k_stream_close(&s);
    free(cq4); free(cx); free(cy);
    return ok ? 0 : 1;
}
