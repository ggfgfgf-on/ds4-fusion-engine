/* What does the {1to16} broadcast operand on vpdpbusd actually cost?
 *
 * The wide Q4_K kernel runs at 0.56 dpbusd per cycle, while the same kernel with
 * its weights in L2 reaches about 2.0. Every memory explanation has been
 * eliminated by measurement: token group size one, which reads the weights eight
 * times over, is only twelve percent slower than eight, so the weight traffic is
 * not what limits it. That leaves the instruction itself.
 *
 * Each dpbusd in that kernel takes its second operand as a four byte memory
 * reference the hardware broadcasts to all sixteen lanes. Three forms are timed
 * here, all doing sixty-four multiplies per instruction:
 *
 *   reg    the activation is in a register, no memory reference at all
 *   bcast  a four byte memory reference, broadcast - what the kernel uses
 *   full   a sixty-four byte memory reference, no broadcast
 *
 * If bcast tracks reg, the broadcast is free and the kernel's cost is elsewhere.
 * If it tracks full, or falls below both, the load port is the limiter and the
 * only fix is fewer broadcasts per multiply, which means more rows per
 * accumulator than the sixteen the register file allows.
 */
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <windows.h>

static double now_s(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

#define ITER 4000000

static int32_t g_act[16] __attribute__((aligned(64)));
static int8_t  g_w[64]   __attribute__((aligned(64)));

static double g_reg, g_bcast, g_full;

int main(void)
{
    for (int i = 0; i < 16; i++) g_act[i] = 3;
    for (int i = 0; i < 64; i++) g_w[i] = (int8_t)(i & 0x3f);

    const __m512i w = _mm512_loadu_si512((const void *)g_w);
    const __m512i r = _mm512_set1_epi32(3);
    const int32_t *p = g_act;
    const __m512i z = _mm512_setzero_si512();

    volatile int64_t sink = 0;
    int64_t lane[8];

    for (int round = 0; round < 3; round++) {
        /* ---- reg: register operand ------------------------------------- */
        __m512i a0 = z, a1 = z, a2 = z, a3 = z, a4 = z, a5 = z, a6 = z, a7 = z;
        double t0 = now_s();
        for (long i = 0; i < ITER; i++) {
            a0 = _mm512_dpbusd_epi32(a0, w, r);
            a1 = _mm512_dpbusd_epi32(a1, w, r);
            a2 = _mm512_dpbusd_epi32(a2, w, r);
            a3 = _mm512_dpbusd_epi32(a3, w, r);
            a4 = _mm512_dpbusd_epi32(a4, w, r);
            a5 = _mm512_dpbusd_epi32(a5, w, r);
            a6 = _mm512_dpbusd_epi32(a6, w, r);
            a7 = _mm512_dpbusd_epi32(a7, w, r);
        }
        double t1 = now_s();
        g_reg = t1 - t0;
        _mm512_storeu_si512(lane, _mm512_add_epi64(_mm512_add_epi64(a0, a1),
                                                   _mm512_add_epi64(a2, a3)));
        sink += lane[0];

        /* ---- bcast: four byte reference, broadcast to sixteen lanes ----- */
        __m512i b0 = z, b1 = z, b2 = z, b3 = z, b4 = z, b5 = z, b6 = z, b7 = z;
        t0 = now_s();
        for (long i = 0; i < ITER; i++) {
            b0 = _mm512_dpbusd_epi32(b0, w, _mm512_set1_epi32(*(const int32_t *)p));
            b1 = _mm512_dpbusd_epi32(b1, w, _mm512_set1_epi32(*(const int32_t *)p));
            b2 = _mm512_dpbusd_epi32(b2, w, _mm512_set1_epi32(*(const int32_t *)p));
            b3 = _mm512_dpbusd_epi32(b3, w, _mm512_set1_epi32(*(const int32_t *)p));
            b4 = _mm512_dpbusd_epi32(b4, w, _mm512_set1_epi32(*(const int32_t *)p));
            b5 = _mm512_dpbusd_epi32(b5, w, _mm512_set1_epi32(*(const int32_t *)p));
            b6 = _mm512_dpbusd_epi32(b6, w, _mm512_set1_epi32(*(const int32_t *)p));
            b7 = _mm512_dpbusd_epi32(b7, w, _mm512_set1_epi32(*(const int32_t *)p));
        }
        t1 = now_s();
        g_bcast = t1 - t0;
        _mm512_storeu_si512(lane, _mm512_add_epi64(_mm512_add_epi64(b0, b1),
                                                   _mm512_add_epi64(b2, b3)));
        sink += lane[1];

        /* ---- full: sixty-four byte reference ---------------------------- */
        __m512i c0 = z, c1 = z, c2 = z, c3 = z, c4 = z, c5 = z, c6 = z, c7 = z;
        const __m512i m = _mm512_loadu_si512((const void *)g_w);
        t0 = now_s();
        for (long i = 0; i < ITER; i++) {
            c0 = _mm512_dpbusd_epi32(c0, w, m);
            c1 = _mm512_dpbusd_epi32(c1, w, m);
            c2 = _mm512_dpbusd_epi32(c2, w, m);
            c3 = _mm512_dpbusd_epi32(c3, w, m);
            c4 = _mm512_dpbusd_epi32(c4, w, m);
            c5 = _mm512_dpbusd_epi32(c5, w, m);
            c6 = _mm512_dpbusd_epi32(c6, w, m);
            c7 = _mm512_dpbusd_epi32(c7, w, m);
        }
        t1 = now_s();
        g_full = t1 - t0;
        _mm512_storeu_si512(lane, _mm512_add_epi64(_mm512_add_epi64(c0, c1),
                                                   _mm512_add_epi64(c2, c3)));
        sink += lane[2];
    }

    /* cycles at a fixed reference so the ratio is what is read, not the clock */
    const double GHZ = 5.4e9;
    const double ops = (double)ITER * 8.0;
    printf("form    ns/op    dpbusd/cycle   vs reg\n");
    printf("reg     %6.3f    %6.3f        1.00x\n",
           g_reg / ops * 1e9, ops / (g_reg * GHZ));
    printf("bcast   %6.3f    %6.3f        %.2fx\n",
           g_bcast / ops * 1e9, ops / (g_bcast * GHZ), g_bcast / g_reg);
    printf("full    %6.3f    %6.3f        %.2fx\n",
           g_full / ops * 1e9, ops / (g_full * GHZ), g_full / g_reg);
    printf("(sink %lld)\n", (long long)sink);
    return 0;
}
