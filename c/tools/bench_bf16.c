/* bf16 dot-product strategies on Apple silicon, measured.
 *
 * The engine's matmul_h converts bf16->f32 one element at a time (shift into a
 * uint32_t, memcpy into a float). docs/REFERENCE.md measured that path at
 * 23.9 GB/s against f32's 47.5 GB/s, i.e. the conversion, not memory, is the
 * limit. Candidates:
 *   scalar      what the engine does today
 *   neon_shl    widen 8 lanes at a time with vshll (no BF16 extension needed)
 *   bfdot       FEAT_BF16's BFDOT: bf16 pairs -> f32 accumulate, 8/instr
 * Build:
 *   clang -O3 -mcpu=native -Xclang -fopenmp -I$(brew --prefix libomp)/include \
 *     bench_bf16.c -o /tmp/bench_bf16 -lm -L$(brew --prefix libomp)/lib -lomp
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

static inline float bf16_scalar(uint16_t h){uint32_t u=(uint32_t)h<<16;float f;memcpy(&f,&u,4);return f;}

/* --- today's engine --- */
static float dot_scalar(const uint16_t *w,const float *x,int n){
    float a=0; for(int i=0;i<n;i++) a+=x[i]*bf16_scalar(w[i]); return a;
}

#ifdef __ARM_NEON
/* --- widen with shift-left-long: bf16 -> f32 is just <<16, so vshll_n_u16 by
 * 16 IS the conversion. Works on any NEON, no BF16 extension. --- */
static float dot_neon_shl(const uint16_t *w,const float *x,int n){
    float32x4_t a0=vdupq_n_f32(0), a1=vdupq_n_f32(0);
    int i=0;
    for(; i+8<=n; i+=8){
        uint16x8_t v=vld1q_u16(w+i);
        float32x4_t lo=vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v),16));
        float32x4_t hi=vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(v),16));
        a0=vfmaq_f32(a0,vld1q_f32(x+i),  lo);
        a1=vfmaq_f32(a1,vld1q_f32(x+i+4),hi);
    }
    float a=vaddvq_f32(vaddq_f32(a0,a1));
    for(; i<n; i++) a+=x[i]*bf16_scalar(w[i]);
    return a;
}
#endif

#if defined(__ARM_FEATURE_BF16)
/* --- BFDOT: 2-way bf16 dot per f32 lane. Needs BOTH operands bf16, so the
 * activations get converted once per row-block, not per weight. --- */
static float dot_bfdot(const bfloat16_t *w,const bfloat16_t *x,int n){
    float32x4_t a0=vdupq_n_f32(0), a1=vdupq_n_f32(0);
    int i=0;
    for(; i+16<=n; i+=16){
        a0=vbfdotq_f32(a0, vld1q_bf16(w+i),   vld1q_bf16(x+i));
        a1=vbfdotq_f32(a1, vld1q_bf16(w+i+8), vld1q_bf16(x+i+8));
    }
    float a=vaddvq_f32(vaddq_f32(a0,a1));
    for(; i<n; i++) a+=(float)w[i]*(float)x[i];
    return a;
}
#endif

int main(void){
    int I=3072, O=12288, iters=12;
    uint16_t *w=malloc((size_t)I*O*2);
    float *x=malloc(I*4), *y=malloc(O*4);
    for(int64_t i=0;i<(int64_t)I*O;i++) w[i]=(uint16_t)(0x3F00 + (i&0x3F));
    for(int i=0;i<I;i++) x[i]=0.5f+(float)(i&7)*0.01f;

    double t0,dt; double gb=(double)I*O*2*iters/1e9;

    t0=now();
    for(int it=0;it<iters;it++){
        #pragma omp parallel for schedule(static)
        for(int o=0;o<O;o++) y[o]=dot_scalar(w+(int64_t)o*I,x,I);
    }
    dt=now()-t0; printf("scalar    %7.2f ms  %6.1f GB/s   y[0]=%.4f\n", dt/iters*1e3, gb/dt, y[0]);

#ifdef __ARM_NEON
    t0=now();
    for(int it=0;it<iters;it++){
        #pragma omp parallel for schedule(static)
        for(int o=0;o<O;o++) y[o]=dot_neon_shl(w+(int64_t)o*I,x,I);
    }
    dt=now()-t0; printf("neon_shl  %7.2f ms  %6.1f GB/s   y[0]=%.4f\n", dt/iters*1e3, gb/dt, y[0]);
#endif

#if defined(__ARM_FEATURE_BF16)
    bfloat16_t *xb=malloc((size_t)I*2);
    for(int i=0;i<I;i++) xb[i]=(bfloat16_t)x[i];
    t0=now();
    for(int it=0;it<iters;it++){
        #pragma omp parallel for schedule(static)
        for(int o=0;o<O;o++) y[o]=dot_bfdot((const bfloat16_t*)(w+(int64_t)o*I),xb,I);
    }
    dt=now()-t0; printf("bfdot     %7.2f ms  %6.1f GB/s   y[0]=%.4f  (activations bf16: lossy)\n",
                        dt/iters*1e3, gb/dt, y[0]);
#else
    printf("bfdot     (not compiled: no __ARM_FEATURE_BF16 -- needs -mcpu=native)\n");
#endif
    return 0;
}
