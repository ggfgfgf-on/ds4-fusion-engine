#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <windows.h>
static double now(void){
    LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart*1e3/(double)f.QuadPart;
}
int main(void){
    __m512i a0=_mm512_set1_epi32(1),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
    __m512i b0=_mm512_set1_epi32(2),b1=b0,b2=b0,b3=b0,b4=b0,b5=b0,b6=b0,b7=b0;
    __m512i q=_mm512_set1_epi32(3), w=_mm512_set1_epi32(4);
    const long N=20000000;
    double t0=now();
    for(long i=0;i<N;i++){
        a0=_mm512_dpbusd_epi32(a0,q,w); a1=_mm512_dpbusd_epi32(a1,q,w);
        a2=_mm512_dpbusd_epi32(a2,q,w); a3=_mm512_dpbusd_epi32(a3,q,w);
        a4=_mm512_dpbusd_epi32(a4,q,w); a5=_mm512_dpbusd_epi32(a5,q,w);
        a6=_mm512_dpbusd_epi32(a6,q,w); a7=_mm512_dpbusd_epi32(a7,q,w);
        b0=_mm512_dpbusd_epi32(b0,q,w); b1=_mm512_dpbusd_epi32(b1,q,w);
        b2=_mm512_dpbusd_epi32(b2,q,w); b3=_mm512_dpbusd_epi32(b3,q,w);
        b4=_mm512_dpbusd_epi32(b4,q,w); b5=_mm512_dpbusd_epi32(b5,q,w);
        b6=_mm512_dpbusd_epi32(b6,q,w); b7=_mm512_dpbusd_epi32(b7,q,w);
    }
    double t1=now();
    long ops=N*16;
    __m512i s=_mm512_add_epi32(_mm512_add_epi32(a0,a1),_mm512_add_epi32(a2,a3));
    s=_mm512_add_epi32(s,_mm512_add_epi32(a4,a5));
    s=_mm512_add_epi32(s,_mm512_add_epi32(a6,a7));
    s=_mm512_add_epi32(s,_mm512_add_epi32(_mm512_add_epi32(b0,b1),_mm512_add_epi32(b2,b3)));
    s=_mm512_add_epi32(s,_mm512_add_epi32(_mm512_add_epi32(b4,b5),_mm512_add_epi32(b6,b7)));
    int32_t out[16]; _mm512_storeu_si512(out,s);
    double ms=t1-t0;
    printf("ops=%ld ms=%.1f => %.2f Gops/s\n", ops, ms, ops/ms/1e6);
    printf("per-core ops/cycle at 4.5GHz: %.2f\n", ops/ms/1e6/(4500.0));
    printf("check=%d\n", out[0]);
    return 0;
}
