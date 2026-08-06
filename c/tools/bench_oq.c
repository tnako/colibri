/* Where does time actually go in the oQ kernel? Compare, on the same K/N:
 *   1. bf16 matvec (what the engine does today for resident weights)
 *   2. f32 matvec
 *   3. oQ packed matvec at each bit width
 * Synthetic weights: we are measuring throughput, not accuracy. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "/Users/anton.korshikov/GIT/colibri-laguna/c/oq.h"

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static float bf16f(uint16_t h){uint32_t u=(uint32_t)h<<16;float f;memcpy(&f,&u,4);return f;}

int main(void){
    int K=3072, N=12288, iters=12;
    float *x=malloc(K*4), *y=malloc(N*4);
    for(int i=0;i<K;i++) x[i]=(float)drand48()-0.5f;

    float *wf=malloc((size_t)K*N*4);
    uint16_t *wh=malloc((size_t)K*N*2);
    for(int64_t i=0;i<(int64_t)K*N;i++){ wf[i]=(float)drand48()-0.5f; wh[i]=(uint16_t)(i*2654435761u>>16); }

    double t0,dt;
    /* f32 */
    t0=now();
    for(int it=0;it<iters;it++){
        #pragma omp parallel for schedule(static)
        for(int n=0;n<N;n++){const float*w=wf+(int64_t)n*K;float a=0;for(int i=0;i<K;i++)a+=x[i]*w[i];y[n]=a;}
    }
    dt=now()-t0; printf("f32      %7.2f ms/matvec   %6.1f GB/s\n", dt/iters*1e3, (double)K*N*4*iters/dt/1e9);
    /* bf16 (the engine's matmul_h) */
    t0=now();
    for(int it=0;it<iters;it++){
        #pragma omp parallel for schedule(static)
        for(int n=0;n<N;n++){const uint16_t*w=wh+(int64_t)n*K;float a=0;for(int i=0;i<K;i++)a+=x[i]*bf16f(w[i]);y[n]=a;}
    }
    dt=now()-t0; printf("bf16     %7.2f ms/matvec   %6.1f GB/s\n", dt/iters*1e3, (double)K*N*2*iters/dt/1e9);

    int bl[]={2,3,4,6,8};
    for(int bi=0;bi<5;bi++){
        int bits=bl[bi], gs=64;
        OQTensor t={0}; t.rows=N;t.K=K;t.bits=bits;t.gs=gs;
        t.ngroups=K/gs; t.words=K*bits/32;
        t.code=malloc((size_t)N*t.words*4);
        t.scale=malloc((size_t)N*t.ngroups*4); t.bias=malloc((size_t)N*t.ngroups*4);
        for(int64_t i=0;i<(int64_t)N*t.words;i++) t.code[i]=(uint32_t)(i*2654435761u);
        for(int64_t i=0;i<(int64_t)N*t.ngroups;i++){t.scale[i]=0.01f;t.bias[i]=-0.1f;}
        t0=now();
        for(int it=0;it<iters;it++) oq_matvec(y,x,&t);
        dt=now()-t0;
        double bytes=(double)oq_bytes(&t);
        printf("oQ %d-bit %7.2f ms/matvec   %6.1f GB/s   weights %5.1f MB (%.2fx vs bf16)\n",
               bits, dt/iters*1e3, bytes*iters/dt/1e9, bytes/1e6, (double)K*N*2/bytes);
        oq_free(&t);
    }
    return 0;
}
