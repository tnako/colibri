/* ---- Q8R: a UDOT-native residency format (fmt=102, private ordinal) --------
 *
 * WHY THIS EXISTS
 * The oQ kernels dequantize packed 2/3/4/6/8-bit codes to f32 and run f32 FMA.
 * Measured on M5: f32 GEMV tops out at 59 GFLOP/s, while UDOT (FEAT_DotProd,
 * 16 int8 MACs per instruction) reaches 298 GOP/s on the same shapes -- 5.0x.
 * Getting that 5x means the inner loop must consume int8 directly, with no
 * per-group bit unpacking in the way.
 *
 * So the bit width becomes a LOAD-TIME concern only. At load, every oQ tensor is
 * expanded once into:
 *   codes  uint8[O][I]      the quantized code, one byte each, unpacked
 *   scale  f32 [O][I/gs]    per-group scale   (from oQ .scales)
 *   bias   f32 [O][I/gs]    per-group bias    (from oQ .biases)
 *   rsum   f32 [O][I/gs]    per-group sum of codes, PRECOMPUTED
 *
 * Expanding 2-bit codes to 8-bit costs 4x the RAM for routed experts. That is
 * the deliberate memory-for-speed trade: the whole model still lands under
 * 9 GiB, which is LESS than the streaming design's measured 11.05 GiB peak,
 * because it removes the expert cache, the LRU, and the per-slot scratch.
 *
 * THE ARITHMETIC
 * oQ dequant is affine: w = c*s + b, per group of gs inputs. A dot product with
 * activations x therefore factors per group:
 *     sum_i x_i*(c_i*s + b) = s*sum_i(x_i*c_i) + b*sum_i(x_i)
 * Quantizing the ACTIVATIONS per group too, x_i = (u_i - z)*xs with u_i uint8:
 *     sum_i x_i*c_i = xs * ( sum_i(u_i*c_i) - z*sum_i(c_i) )
 *                     ^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^
 *                     exactly one UDOT chain   precomputed rsum
 * so the whole group reduces to one integer dot plus two scalars. rsum is
 * computed once at load, which is why it lives in the format.
 *
 * ACCURACY
 * Activations are quantized to uint8 per group of gs with a real min/max, so
 * the error is bounded by the activation range within a 64-wide group rather
 * than across the whole row. Verified token-exact against the f32 path on the
 * transformers-oracle fixtures; see docs/oq-optimization-rounds.md.
 */
#ifndef COLI_Q8R_H
#define COLI_Q8R_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include "oq.h"   /* oq_unpack for the non-2-bit widths */

/* Codes stay in the oQ packing. One byte per code would be 31.4 GB of routed
 * experts for Laguna-XS (measured, not estimated) and does not fit in 32 GB;
 * 2-bit packing is 7.9 GB and the unpack feeds UDOT directly. `bits` selects
 * the unpack, so a checkpoint mixing widths still works. */
typedef struct {
    uint32_t *codes;     /* packed, [rows][in*bits/32]                     */
    float   *scale;      /* [rows][ngroups]                                */
    float   *bias;       /* [rows][ngroups]                                */
    float   *rsum;       /* [rows][ngroups]   sum of codes per group       */
    int rows, in, gs, ng, bits;
} Q8R;

static inline int q8r_valid(const Q8R *t){ return t && t->codes; }

static void q8r_free(Q8R *t){
    free(t->codes); free(t->scale); free(t->bias); free(t->rsum);
    memset(t, 0, sizeof(*t));
}

/* Activation block: uint8 codes + per-group scale/zero, laid out to match the
 * weight's group size so the kernel can pair them index-for-index. */
typedef struct {
    uint8_t *u;          /* [rows][in]      */
    float   *xs;         /* [rows][ngroups] */
    float   *xz;         /* [rows][ngroups] zero point, in real units */
    float   *xsum;       /* [rows][ngroups] sum of the ORIGINAL floats */
    int rows, in, gs, ng;
} Q8Act;

static void q8act_alloc(Q8Act *a, int rows, int in, int gs){
    a->rows=rows; a->in=in; a->gs=gs; a->ng=(in+gs-1)/gs;
    a->u    = malloc((size_t)rows*in);
    a->xs   = malloc((size_t)rows*a->ng*sizeof(float));
    a->xz   = malloc((size_t)rows*a->ng*sizeof(float));
    a->xsum = malloc((size_t)rows*a->ng*sizeof(float));
    if(!a->u||!a->xs||!a->xz||!a->xsum){ fprintf(stderr,"OOM q8act\n"); exit(1); }
}
static void q8act_free(Q8Act *a){ free(a->u); free(a->xs); free(a->xz); free(a->xsum); memset(a,0,sizeof(*a)); }

/* Quantize activations to uint8 per group. Keeps xsum (sum of the true floats)
 * because the weight bias term needs it exactly, not a requantized version. */
static void q8act_fill(Q8Act *a, const float *x){
    int gs=a->gs, ng=a->ng, in=a->in;
    /* `if` guard: the MoE resident path calls this from inside a parallel
     * region, where an inner region would only add fork/join cost. */
    #pragma omp parallel for schedule(static) if(!omp_in_parallel() && a->rows>1)
    for(int r=0;r<a->rows;r++){
        const float *xr = x + (int64_t)r*in;
        uint8_t *ur = a->u + (int64_t)r*in;
        for(int g=0; g<ng; g++){
            int b0=g*gs, n=(b0+gs<=in)?gs:(in-b0);
            float mn=xr[b0], mx=xr[b0], sm=0.f;
            for(int i=0;i<n;i++){ float v=xr[b0+i]; if(v<mn)mn=v; if(v>mx)mx=v; sm+=v; }
            float s=(mx-mn)/255.f; if(s<=0.f) s=1e-8f;
            float inv=1.f/s;
            for(int i=0;i<n;i++){
                int q=(int)lrintf((xr[b0+i]-mn)*inv);
                ur[b0+i]=(uint8_t)(q<0?0:(q>255?255:q));
            }
            for(int i=n;i<gs && b0+i<in;i++) ur[b0+i]=0;
            a->xs  [(int64_t)r*ng+g]=s;
            a->xz  [(int64_t)r*ng+g]=mn;
            a->xsum[(int64_t)r*ng+g]=sm;
        }
    }
}

/* ---- 2-bit codes -> uint8 lanes, straight into UDOT --------------------------
 * The naive Q8R layout (one byte per code) is 31.4 GB of routed experts for
 * Laguna-XS, which does not fit in 32 GB. Keeping the oQ 2-bit packing costs
 * 7.9 GB instead, and the unpack to bytes is 3 NEON ops per 16 values -- far
 * cheaper than the f32 dequant it replaces, and it feeds UDOT directly.
 *
 * A 32-bit word holds 16 2-bit codes, LSB-first (see docs/oq-format.md). */
#if defined(__ARM_FEATURE_DOTPROD)
static inline uint32_t q8r_udot_2bit(const uint32_t *w, const uint8_t *u, int n){
    const uint8x16_t m3 = vdupq_n_u8(3);
    uint32x4_t a0=vdupq_n_u32(0), a1=vdupq_n_u32(0);
    int i=0;
    for(; i+32<=n; i+=32, w+=2){
        /* 2 words = 32 codes. Spread each word's bytes 4x then mask the pair
         * position out of each lane. */
        uint8x16_t raw0 = vreinterpretq_u8_u32(vdupq_n_u32(w[0]));
        uint8x16_t raw1 = vreinterpretq_u8_u32(vdupq_n_u32(w[1]));
        /* byte b of the word supplies codes 4b..4b+3 */
        static const uint8_t idxb[16] = {0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3};
        static const uint8_t shft[16] = {0,2,4,6,0,2,4,6,0,2,4,6,0,2,4,6};
        uint8x16_t sel = vld1q_u8(idxb);
        int8x16_t  sh  = vreinterpretq_s8_u8(vld1q_u8(shft));
        uint8x16_t b0 = vqtbl1q_u8(raw0, sel);
        uint8x16_t b1 = vqtbl1q_u8(raw1, sel);
        uint8x16_t c0 = vandq_u8(vshlq_u8(b0, vnegq_s8(sh)), m3);
        uint8x16_t c1 = vandq_u8(vshlq_u8(b1, vnegq_s8(sh)), m3);
        a0 = vdotq_u32(a0, c0, vld1q_u8(u+i));
        a1 = vdotq_u32(a1, c1, vld1q_u8(u+i+16));
    }
    uint32_t s = vaddvq_u32(vaddq_u32(a0,a1));
    for(; i<n; i++){ int wi=i>>4, off=(i&15)*2; s += ((w[wi]>>off)&3u) * (uint32_t)u[i]; }
    return s;
}
#endif

/* one group: integer dot of uint8 codes with uint8 activations */
static inline uint32_t q8r_udot(const uint8_t *w, const uint8_t *u, int n){
#if defined(__ARM_FEATURE_DOTPROD)
    uint32x4_t a0=vdupq_n_u32(0), a1=vdupq_n_u32(0);
    int i=0;
    for(; i+32<=n; i+=32){
        a0=vdotq_u32(a0, vld1q_u8(w+i),    vld1q_u8(u+i));
        a1=vdotq_u32(a1, vld1q_u8(w+i+16), vld1q_u8(u+i+16));
    }
    for(; i+16<=n; i+=16) a0=vdotq_u32(a0, vld1q_u8(w+i), vld1q_u8(u+i));
    uint32_t s=vaddvq_u32(vaddq_u32(a0,a1));
    for(; i<n; i++) s += (uint32_t)w[i]*(uint32_t)u[i];
    return s;
#else
    uint32_t s=0; for(int i=0;i<n;i++) s+=(uint32_t)w[i]*(uint32_t)u[i]; return s;
#endif
}

/* y[S,O] = A @ dequant(W)^T   with W in Q8R and A pre-quantized to uint8.
 *
 * Per (row o, group g):
 *   contribution = s_w*xs*(UDOT(c,u) - z*rsum) + s_w*z*rsum ... expanded below
 * Derivation, with w_i = c_i*sw + bw and x_i = u_i*xs + xz:
 *   sum w_i x_i = sw*xs*sum(c_i u_i) + sw*xz*sum(c_i) + bw*sum(x_i)
 * The three terms are: one UDOT, one precomputed rsum, one precomputed xsum.
 */
static void q8r_gemm(float *y, const Q8Act *a, const Q8R *W){
    int O=W->rows, in=W->in, gs=W->gs, ng=W->ng, bits=W->bits;
    int64_t rw = (int64_t)in*bits/32;
    int wpg = gs*bits/32;
    #pragma omp parallel if(!omp_in_parallel())
    {
        uint8_t scratch[512];
        #pragma omp for schedule(static)
        for(int o=0;o<O;o++){
            const uint32_t *wc = W->codes + (int64_t)o*rw;
            const float *sw = W->scale + (int64_t)o*ng;
            const float *bw = W->bias  + (int64_t)o*ng;
            const float *rs = W->rsum  + (int64_t)o*ng;
            for(int r=0;r<a->rows;r++){
                const uint8_t *ur = a->u + (int64_t)r*in;
                const float *xs = a->xs   + (int64_t)r*ng;
                const float *xz = a->xz   + (int64_t)r*ng;
                const float *xm = a->xsum + (int64_t)r*ng;
                float acc=0.f;
                for(int g=0; g<ng; g++){
                    int b0=g*gs;
                    const uint32_t *wg = wc + (int64_t)g*wpg;
                    uint32_t dd;
#if defined(__ARM_FEATURE_DOTPROD)
                    if(bits==2) dd = q8r_udot_2bit(wg, ur+b0, gs);
                    else { oq_unpack(wg,bits,gs,scratch); dd = q8r_udot(scratch, ur+b0, gs); }
#else
                    oq_unpack(wg,bits,gs,scratch); dd = q8r_udot(scratch, ur+b0, gs);
#endif
                    acc += sw[g]*xs[g]*(float)dd + sw[g]*xz[g]*rs[g] + bw[g]*xm[g];
                }
                y[(int64_t)r*O+o]=acc;
            }
        }
    }
}

#endif /* COLI_Q8R_H */
