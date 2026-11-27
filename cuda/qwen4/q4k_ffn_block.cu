/* The feed-forward block, end to end on the device.
 *
 * This is the unit worth offloading: it is about eighty-five percent of a layer's
 * arithmetic and it is self-contained - a normalised hidden state goes in, the block's
 * contribution comes out, and nothing in between needs the host. Per layer that is two
 * transfers of the hidden state, twelve megabytes together, against the ten megabytes of
 * weight upload the projection-at-a-time design was paying for every one of three
 * projections.
 *
 * The arithmetic matches the host block exactly:
 *
 *   gate = x @ W_gate ; up = x @ W_up
 *   a    = up * silu(gate)
 *   out  = a @ W_down
 *
 * with each weight dequantised from Q4_K to F16 on the device and multiplied by cuBLAS.
 */
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef Q4K_STREAM_BLOCK_ONLY
#define Q4K_STREAM_BLOCK_ONLY 0
#endif

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

__global__ void ffn_q4k_to_f16(__half *out, const unsigned char *in,
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

__global__ void ffn_f32_to_f16(__half *dst, const float *src, unsigned long long n) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn(src[i]);
}

/* a = up * silu(gate), written straight out as halves for the next product. */
__global__ void ffn_silu_mul(__half *a, const float *gate, const float *up,
                             unsigned long long n) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    const float s = g / (1.0f + expf(-g));
    a[i] = __float2half_rn(up[i] * s);
}

#define Q6K_BLK_BYTES 144u

typedef struct {
    cublasHandle_t cublas;
    unsigned char *q4src;    /* packed weights when they are streamed */
    size_t cap_q4src;
    __half *xh;          /* activations, halves */
    __half *wf;          /* one dequantised weight at a time */
    __half *af;          /* silu(up*gate) halves */
    float *g;            /* gate product */
    float *u;            /* up product */
    float *y;            /* block output */
    float *xf;           /* the host state, staged before conversion */
    float *pin_in;       /* pinned landing and launch buffers */
    float *pin_out;
    size_t cap_pin_in, cap_pin_out;
    size_t cap_xh, cap_wf, cap_af, cap_g, cap_u, cap_y, cap_xf;
    int ready;
} ffn_ctx;

static ffn_ctx g_ffn;

static int fpin(float **p, size_t *cap, size_t need) {
    if (*cap >= need) return 1;
    if (*p) cudaFreeHost(*p);
    *p = NULL;
    *cap = 0;
    if (cudaHostAlloc((void **)p, need, cudaHostAllocDefault) != cudaSuccess) return 0;
    *cap = need;
    return 1;
}

static int fgrow(void **p, size_t *cap, size_t need) {
    if (*cap >= need) return 1;
    if (*p) cudaFree(*p);
    *p = NULL;
    *cap = 0;
    if (cudaMalloc(p, need) != cudaSuccess) return 0;
    *cap = need;
    return 1;
}

static int ffn_open(void) {
    if (g_ffn.ready) return 1;
    if (cublasCreate(&g_ffn.cublas) != CUBLAS_STATUS_SUCCESS) return 0;
    cublasSetMathMode(g_ffn.cublas, CUBLAS_TF32_TENSOR_OP_MATH);
    g_ffn.ready = 1;
    return 1;
}

/* One Q4_K matrix times the activation block, into out (or A of the next stage). */
static double g_dq_ms, g_gemm_ms;

static int ffn_mm(__half *w, const void *q4, unsigned rows, unsigned cols,
                  const __half *a, unsigned n_tok, const __half *a_override,
                  float *out_f32, __half *out_f16) {
    const unsigned long long nb = (unsigned long long)rows * (cols / 256u);
    const unsigned warps = 8u;
    {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventRecord(e0);
        ffn_q4k_to_f16<<<(unsigned)((nb + warps - 1u) / warps), warps * 32u>>>(
            w, (const unsigned char *)q4, nb);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        g_dq_ms += ms;
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }
    if (cudaGetLastError() != cudaSuccess) return 0;
    const __half *src = a_override ? a_override : a;
    const float one = 1.0f, zero = 0.0f;
    if (out_f32) {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventRecord(e0);
        const cublasStatus_t st0 = cublasGemmEx(
            g_ffn.cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)rows, (int)n_tok, (int)cols, &one,
            w, CUDA_R_16F, (int)cols, src, CUDA_R_16F, (int)cols, &zero, out_f32,
            CUDA_R_32F, (int)rows, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        g_gemm_ms += ms;
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        return st0 == CUBLAS_STATUS_SUCCESS;
    }
    /* halves out for the next stage: accumulate into a float view is not possible, so
     * this intermediate is handled by the caller converting through floats. */
    (void)out_f16;
    return 0;
}

extern "C" void ds4_gpu_q4k_ffn_times(double *dequant_ms, double *gemm_ms) {
    *dequant_ms = g_dq_ms;
    *gemm_ms = g_gemm_ms;
}

/* weights_on_host: the three pointers are packed Q4_K in host memory and are copied in
 * here; otherwise they are device pointers that are already resident. */
/* One four-bit matrix, streamed: upload, unpack, multiply by cuBLAS, read back.
 *
 * The feed-forward block already does this for three matrices at a time; the attention
 * projections are the other large four-bit consumers and take the same route. The
 * transfer is a fifth of a millisecond per megabyte and the arithmetic in the product pays
 * for it several times over, which is why this needs no residency.
 */
extern "C" int ds4_gpu_q4k_mm_stream(const void *host_q4k, uint32_t K, uint32_t M,
                                     const float *host_act, uint32_t T, float *host_out) {
    if (!ffn_open()) return 0;
    if (!host_q4k || !host_act || !host_out || !K || !M || !T || (K % 256u)) return 0;
    const size_t xbytes = (size_t)T * K;
    const size_t qbytes = ((size_t)M * K / 256u) * Q6K_BLK_BYTES;
    if (!fgrow((void **)&g_ffn.xh, &g_ffn.cap_xh, xbytes * 2)) return 0;
    if (!fgrow((void **)&g_ffn.xf, &g_ffn.cap_xf, xbytes * 4)) return 0;
    if (!fgrow((void **)&g_ffn.y, &g_ffn.cap_y, (size_t)T * M * 4)) return 0;
    if (!fgrow((void **)&g_ffn.q4src, &g_ffn.cap_q4src, qbytes)) return 0;
    if (!fgrow((void **)&g_ffn.wf, &g_ffn.cap_wf, (size_t)M * K * 2)) return 0;
    cudaMemcpy(g_ffn.q4src, host_q4k, qbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(g_ffn.xf, host_act, xbytes * 4, cudaMemcpyHostToDevice);
    ffn_f32_to_f16<<<(unsigned)((xbytes + 255) / 256), 256>>>(g_ffn.xh, g_ffn.xf, xbytes);
    const unsigned long long nb = (unsigned long long)M * (K / 256u);
    const unsigned warps = 8u;
    ffn_q4k_to_f16<<<(unsigned)((nb + warps - 1u) / warps), warps * 32u>>>(
        g_ffn.wf, g_ffn.q4src, nb);
    const float one = 1.0f, zero = 0.0f;
    cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
    const uint32_t tile = 5120u;
    for (uint32_t r0 = 0; r0 < M && st == CUBLAS_STATUS_SUCCESS; r0 += tile) {
        const uint32_t rows = (M - r0) < tile ? (M - r0) : tile;
        st = cublasGemmEx(g_ffn.cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)rows, (int)T,
                          (int)K, &one, g_ffn.wf + (size_t)r0 * K, CUDA_R_16F, (int)K,
                          g_ffn.xh, CUDA_R_16F, (int)K, &zero, g_ffn.y + r0,
                          CUDA_R_32F, (int)M, CUBLAS_COMPUTE_32F_FAST_TF32,
                          CUBLAS_GEMM_DEFAULT);
    }
    if (st != CUBLAS_STATUS_SUCCESS) return 0;
    return cudaMemcpy(host_out, g_ffn.y, (size_t)T * M * 4, cudaMemcpyDeviceToHost) ==
           cudaSuccess;
}

extern "C" int ds4_gpu_q4k_ffn_block_ex(const void *gate_q4, const void *up_q4,
                                        const void *down_q4, const float *host_x,
                                        uint32_t n_tok, uint32_t K, uint32_t FF,
                                        float *host_out, int weights_on_host);

extern "C" int ds4_gpu_q4k_ffn_block(const void *gate_q4, const void *up_q4,
                                     const void *down_q4, const float *host_x,
                                     uint32_t n_tok, uint32_t K, uint32_t FF,
                                     float *host_out) {
    return ds4_gpu_q4k_ffn_block_ex(gate_q4, up_q4, down_q4, host_x, n_tok, K, FF,
                                    host_out, 0);
}

extern "C" int ds4_gpu_q4k_ffn_block_ex(const void *gate_q4, const void *up_q4,
                                        const void *down_q4, const float *host_x,
                                        uint32_t n_tok, uint32_t K, uint32_t FF,
                                        float *host_out, int weights_on_host) {
    if (!ffn_open()) return 0;
    if (!gate_q4 || !up_q4 || !down_q4 || !host_x || !host_out) return 0;
    if (!n_tok || !K || !FF || (K % 256u) || (FF % 256u)) return 0;
    const size_t n_x = (size_t)n_tok * K;
    const size_t n_f = (size_t)n_tok * FF;
    {
        static int reported = 0;
        const size_t want_x = n_x * 2, want_w = (size_t)(FF > K ? FF : K) * K * 2,
                     want_f = n_f * 4;
        int ok = fgrow((void **)&g_ffn.xh, &g_ffn.cap_xh, want_x) &&
                 fgrow((void **)&g_ffn.wf, &g_ffn.cap_wf, want_w) &&
                 fgrow((void **)&g_ffn.af, &g_ffn.cap_af, n_f * 2) &&
                 fgrow((void **)&g_ffn.g, &g_ffn.cap_g, want_f) &&
                 fgrow((void **)&g_ffn.u, &g_ffn.cap_u, want_f) &&
                 fgrow((void **)&g_ffn.y, &g_ffn.cap_y, n_x * 4);
        if (!ok && reported++ < 3) {
            size_t vf = 0, vt = 0;
            cudaMemGetInfo(&vf, &vt);
            fprintf(stderr,
                    "q4k_ffn: cannot allocate (T=%u K=%u FF=%u need %.0f+%.0f+%.0f MB, "
                    "device %.0f MB free)\n", n_tok, K, FF,
                    (double)want_x / 1048576.0, (double)want_w / 1048576.0,
                    (double)(n_f * 4 * 2 + n_f * 2) / 1048576.0, (double)vf / 1048576.0);
        }
        if (!ok) return 0;
    }

    if (!fgrow((void **)&g_ffn.xf, &g_ffn.cap_xf, n_x * 4)) return 0;
    if (!fpin(&g_ffn.pin_in, &g_ffn.cap_pin_in, n_x * 4)) return 0;
    if (!fpin(&g_ffn.pin_out, &g_ffn.cap_pin_out, n_x * 4)) return 0;
    memcpy(g_ffn.pin_in, host_x, n_x * 4);
    cudaMemcpy(g_ffn.xf, g_ffn.pin_in, n_x * 4, cudaMemcpyHostToDevice);
    ffn_f32_to_f16<<<(unsigned)((n_x + 255) / 256), 256>>>(g_ffn.xh, g_ffn.xf, n_x);
    {
        static int stage_reported = 0;
        const void *gw = gate_q4, *uw = up_q4, *dw = down_q4;
    if (weights_on_host) {
        /* one staging buffer reused for the three matrices, in the order they are used */
        const size_t bytes = ((size_t)FF * K / 256u) * Q6K_BLK_BYTES;
        if (!fgrow((void **)&g_ffn.q4src, &g_ffn.cap_q4src, bytes)) return 0;
    }
    if (weights_on_host) {
        const size_t gb = ((size_t)FF * K / 256u) * Q6K_BLK_BYTES;
        cudaMemcpy(g_ffn.q4src, gate_q4, gb, cudaMemcpyHostToDevice);
        gw = g_ffn.q4src;
    }
    if (!ffn_mm(g_ffn.wf, gw, FF, K, g_ffn.xh, n_tok, NULL, g_ffn.g, NULL)) {
            if (stage_reported++ < 3) fprintf(stderr, "q4k_ffn: gate product failed\n");
            return 0;
        }
        if (weights_on_host) {
        const size_t ub = ((size_t)FF * K / 256u) * Q6K_BLK_BYTES;
        cudaMemcpy(g_ffn.q4src, up_q4, ub, cudaMemcpyHostToDevice);
        uw = g_ffn.q4src;
    }
    if (!ffn_mm(g_ffn.wf, uw, FF, K, g_ffn.xh, n_tok, NULL, g_ffn.u, NULL)) {
            if (stage_reported++ < 3) fprintf(stderr, "q4k_ffn: up product failed\n");
            return 0;
        }
        ffn_silu_mul<<<(unsigned)((n_f + 255) / 256), 256>>>(g_ffn.af, g_ffn.g, g_ffn.u, n_f);
        if (weights_on_host) {
        const size_t db = ((size_t)K * FF / 256u) * Q6K_BLK_BYTES;
        cudaMemcpy(g_ffn.q4src, down_q4, db, cudaMemcpyHostToDevice);
        dw = g_ffn.q4src;
    }
    if (!ffn_mm(g_ffn.wf, dw, K, FF, NULL, n_tok, g_ffn.af, g_ffn.y, NULL)) {
            if (stage_reported++ < 3) fprintf(stderr, "q4k_ffn: down product failed\n");
            return 0;
        }
    }
    if (cudaMemcpy(g_ffn.pin_out, g_ffn.y, n_x * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
        return 0;
    memcpy(host_out, g_ffn.pin_out, n_x * 4);
    return 1;
}
