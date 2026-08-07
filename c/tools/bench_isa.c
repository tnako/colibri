/* What is the real arithmetic ceiling on this machine, per kernel style?
 *
 * The roofline says prefill in 20s needs ~1.8 TFLOP/s and 50 tok/s needs
 * ~39 GB/s. Before redesigning anything, measure which instruction family can
 * actually get there:
 *   f32 FMA    - what the engine uses today (4 lanes/instr)
 *   UDOT       - FEAT_DotProd, 16 int8 MACs/instr into 4 int32 lanes
 *   SMMLA      - FEAT_I8MM, 8x8 int8 outer product, 64 MACs/instr
 * All three do the same logical work: dot products over K.
 *
 * Build: clang -O3 -mcpu=native -Xclang -fopenmp -I$(brew --prefix libomp)/include \
 *   bench_isa.c -o /tmp/bench_isa -lm -L$(brew --prefix libomp)/lib -lomp
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <arm_neon.h>
#include <omp.h>

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

/* ---- f32: today's kernel shape ---- */
static void gemv_f32(float *y,const float *x,const float *W,int I,int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const float *w=W+(int64_t)o*I;
        float32x4_t a0=vdupq_n_f32(0),a1=vdupq_n_f32(0);
        for(int i=0;i+8<=I;i+=8){
            a0=vfmaq_f32(a0,vld1q_f32(x+i),vld1q_f32(w+i));
            a1=vfmaq_f32(a1,vld1q_f32(x+i+4),vld1q_f32(w+i+4));
        }
        y[o]=vaddvq_f32(vaddq_f32(a0,a1));
    }
}

/* ---- UDOT: int8 codes x int8 activations ---- */
static void gemv_udot(int32_t *y,const uint8_t *x,const uint8_t *W,int I,int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=W+(int64_t)o*I;
        uint32x4_t a0=vdupq_n_u32(0),a1=vdupq_n_u32(0);
        for(int i=0;i+32<=I;i+=32){
            a0=vdotq_u32(a0,vld1q_u8(w+i),   vld1q_u8(x+i));
            a1=vdotq_u32(a1,vld1q_u8(w+i+16),vld1q_u8(x+i+16));
        }
        y[o]=(int32_t)vaddvq_u32(vaddq_u32(a0,a1));
    }
}

/* ---- SMMLA: 2x8x2 int8 matmul per instruction (I8MM) ----
 * Computes a 2x2 block of int32 from two 2x8 int8 operands. Used here in the
 * GEMM shape it is meant for: 2 output rows x 2 activation rows at a time. */
#if defined(__ARM_FEATURE_MATMUL_INT8)
static void gemm_smmla(int32_t *y,const int8_t *X,const int8_t *W,int I,int O,int S){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o+=2){
        for(int s=0;s<S;s+=2){
            int32x4_t acc=vdupq_n_s32(0);
            const int8_t *w0=W+(int64_t)o*I,*w1=W+(int64_t)(o+1)*I;
            const int8_t *x0=X+(int64_t)s*I,*x1=X+(int64_t)(s+1)*I;
            for(int i=0;i+16<=I;i+=16){
                /* rows interleaved: [w0[i..i+7], w1[i..i+7]] etc */
                int8x16_t wv=vcombine_s8(vld1_s8(w0+i),vld1_s8(w1+i));
                int8x16_t xv=vcombine_s8(vld1_s8(x0+i),vld1_s8(x1+i));
                acc=vmmlaq_s32(acc,wv,xv);
                wv=vcombine_s8(vld1_s8(w0+i+8),vld1_s8(w1+i+8));
                xv=vcombine_s8(vld1_s8(x0+i+8),vld1_s8(x1+i+8));
                acc=vmmlaq_s32(acc,wv,xv);
            }
            y[(int64_t)s*O+o]=vgetq_lane_s32(acc,0);
        }
    }
}
#endif

int main(void){
    int I=2048,O=4096,S=64;
    int iters=200;
    printf("threads=%d  I=%d O=%d\n",omp_get_max_threads(),I,O);

    float *xf=malloc(I*4),*Wf=malloc((size_t)I*O*4),*yf=malloc(O*4);
    for(int i=0;i<I;i++)xf[i]=0.01f*(i&15);
    for(int64_t i=0;i<(int64_t)I*O;i++)Wf[i]=0.001f*(i&31);
    double t0=now(); for(int k=0;k<iters;k++) gemv_f32(yf,xf,Wf,I,O);
    double dt=now()-t0;
    double macs=(double)I*O*iters;
    printf("f32  GEMV   %7.2f ms  %7.1f GFLOP/s\n",dt/iters*1e3,2*macs/dt/1e9);

    uint8_t *xu=malloc(I),*Wu=malloc((size_t)I*O); int32_t *yi=malloc(O*4);
    for(int i=0;i<I;i++)xu[i]=i&7;
    for(int64_t i=0;i<(int64_t)I*O;i++)Wu[i]=i&3;
    t0=now(); for(int k=0;k<iters;k++) gemv_udot(yi,xu,Wu,I,O);
    dt=now()-t0;
    printf("UDOT GEMV   %7.2f ms  %7.1f GOP/s   (%.1fx f32)\n",dt/iters*1e3,2*macs/dt/1e9,
           (2*macs/dt/1e9)/(2*macs/((now()-t0)+1e-9)/1e9)*0+0.0);

#if defined(__ARM_FEATURE_MATMUL_INT8)
    int8_t *Xs=malloc((size_t)S*I),*Ws=malloc((size_t)I*O);
    int32_t *ys=malloc((size_t)S*O*4);
    for(int64_t i=0;i<(int64_t)S*I;i++)Xs[i]=i&7;
    for(int64_t i=0;i<(int64_t)I*O;i++)Ws[i]=i&3;
    int it2=20;
    t0=now(); for(int k=0;k<it2;k++) gemm_smmla(ys,Xs,Ws,I,O,S);
    dt=now()-t0;
    double macs2=(double)I*O*S*it2/4.0;  /* only 1 of 4 block outputs stored */
    printf("SMMLA GEMM  %7.2f ms  %7.1f GOP/s  (S=%d)\n",dt/it2*1e3,2*macs2/dt/1e9,S);
#else
    printf("SMMLA: not compiled (no __ARM_FEATURE_MATMUL_INT8)\n");
#endif
    return 0;
}
