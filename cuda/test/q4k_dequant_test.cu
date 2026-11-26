/* Standalone check of the Q4_K -> F16 dequantiser.
 *
 * Deliberately free of the engine: no GGUF, no tensors, no cuBLAS handles. Random
 * blocks go through a straightforward CPU reference and through the GPU kernel, and
 * the two are compared element by element. This catches the class of mistake the
 * in-engine probe kept hiding behind (wrong sub-block order, wrong nibble, wrong
 * scale/min unpacking), and it builds in seconds rather than minutes.
 *
 * Block layout (llama.cpp): d and dmin are half-precision super-block scale and min,
 * scales[12] packs eight 6-bit scales and eight 6-bit mins, qs[128] holds 256
 * four-bit values with the low nibbles first.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define BLOCK_BYTES 144u

/* ---- CPU reference ------------------------------------------------------- */

static float half_bits_to_float(unsigned short h) {
    const unsigned sign = (h >> 15) & 1u;
    const unsigned exp = (h >> 10) & 0x1Fu;
    const unsigned man = h & 0x3FFu;
    float v;
    if (exp == 0) {
        v = (float)man * (1.0f / 16777216.0f);
    } else if (exp == 31) {
        v = man ? NAN : INFINITY;
    } else {
        v = (1.0f + (float)man / 1024.0f) * exp2f((float)((int)exp - 15));
    }
    return sign ? -v : v;
}

static void get_scale_min_k4(int j, const unsigned char *q, int *d, int *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | (((int)q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | (((int)q[j] >> 6) << 4);
    }
}

static void ref_dequant_block(const unsigned char *blk, float *out) {
    const float d = half_bits_to_float(*(const unsigned short *)blk);
    const float dmin = half_bits_to_float(*(const unsigned short *)(blk + 2));
    const unsigned char *scales = blk + 4;
    const unsigned char *q = blk + 16;
    float *y = out;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        int s = 0, m = 0;
        get_scale_min_k4(is + 0, scales, &s, &m);
        const float d1 = d * (float)s, m1 = dmin * (float)m;
        get_scale_min_k4(is + 1, scales, &s, &m);
        const float d2 = d * (float)s, m2 = dmin * (float)m;
        for (int l = 0; l < 32; l++) *y++ = d1 * (float)(q[l] & 0xF) - m1;
        for (int l = 0; l < 32; l++) *y++ = d2 * (float)(q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}

/* ---- GPU kernel (the one destined for the engine) ------------------------ */

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

/* ---- driver -------------------------------------------------------------- */

int main(int argc, char **argv) {
    const int nblocks = (argc > 1) ? atoi(argv[1]) : 4096;
    if (nblocks <= 0) { puts("bad block count"); return 1; }
    const size_t in_bytes = (size_t)nblocks * BLOCK_BYTES;
    unsigned char *host_in = (unsigned char *)malloc(in_bytes);
    float *ref = (float *)malloc((size_t)nblocks * 256u * sizeof(float));
    __half *gpu_half = NULL;
    float *gpu_f32 = (float *)malloc((size_t)nblocks * 256u * sizeof(float));
    if (!host_in || !ref || !gpu_f32) { puts("host alloc failed"); return 1; }

    /* Random blocks, but with plausible d/dmin so the comparison is meaningful. */
    srand(12345);
    for (int b = 0; b < nblocks; b++) {
        unsigned char *blk = host_in + (size_t)b * BLOCK_BYTES;
        const float d = 0.0005f + 0.0005f * (float)(rand() % 100) / 100.0f;
        const float dmin = 0.0002f + 0.0003f * (float)(rand() % 100) / 100.0f;
        __half hd = __float2half(d), hm = __float2half(dmin);
        memcpy(blk, &hd, 2);
        memcpy(blk + 2, &hm, 2);
        for (int i = 0; i < 12; i++) blk[4 + i] = (unsigned char)(rand() & 0xFF);
        for (int i = 0; i < 128; i++) blk[16 + i] = (unsigned char)(rand() & 0xFF);
        ref_dequant_block(blk, ref + (size_t)b * 256u);
    }

    unsigned char *dev_in = NULL;
    if (cudaMalloc(&dev_in, in_bytes) != cudaSuccess) { puts("cudaMalloc in failed"); return 1; }
    if (cudaMalloc(&gpu_half, (size_t)nblocks * 256u * 2) != cudaSuccess) { puts("cudaMalloc out failed"); return 1; }
    cudaMemcpy(dev_in, host_in, in_bytes, cudaMemcpyHostToDevice);
    const unsigned warps = 8u;
    const unsigned grid = (unsigned)(((unsigned long long)nblocks + warps - 1u) / warps);
    q4k_to_f16_kernel<<<grid, warps * 32u>>>(gpu_half, dev_in, (unsigned long long)nblocks);
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("kernel failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    /* widen on the host: half -> float, so a bad half write shows up as well */
    {
        __half *tmp = (__half *)malloc((size_t)nblocks * 256u * 2);
        cudaMemcpy(tmp, gpu_half, (size_t)nblocks * 256u * 2, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < (size_t)nblocks * 256u; i++) gpu_f32[i] = __half2float(tmp[i]);
        free(tmp);
    }

    double worst_abs = 0.0, worst_rel = 0.0;
    size_t first_bad = 0;
    for (size_t i = 0; i < (size_t)nblocks * 256u; i++) {
        const double a = ref[i], b = gpu_f32[i];
        const double diff = fabs(a - b);
        if (diff > worst_abs) worst_abs = diff;
        const double den = fabs(a) > 1e-6 ? fabs(a) : 1e-6;
        const double rel = diff / den;
        if (rel > worst_rel) { worst_rel = rel; if (first_bad == 0) first_bad = i; }
    }
    printf("blocks=%d values=%zu\n", nblocks, (size_t)nblocks * 256u);
    printf("worst absolute difference: %.3e\n", worst_abs);
    printf("worst relative difference: %.3e (first at value %zu)\n", worst_rel, first_bad);
    /* Half has 11 bits of mantissa, so a correct dequantiser lands near 5e-4 relative. */
    const int pass = worst_rel < 2e-3;
    printf("%s\n", pass ? "PASS" : "FAIL");
    if (!pass) {
        const size_t i = first_bad;
        printf("  ref=%.8f gpu=%.8f\n", ref[i], gpu_f32[i]);
        for (size_t k = (i / 256) * 256; k < (i / 256) * 256 + 8 && k < (size_t)nblocks * 256u; k++)
            printf("  [%zu] ref=%.8f gpu=%.8f\n", k, ref[k], gpu_f32[k]);
    }
    free(host_in);
    free(ref);
    free(gpu_f32);
    cudaFree(dev_in);
    cudaFree(gpu_half);
    return pass ? 0 : 1;
}
