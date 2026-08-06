/* ---- oQ (oMLX affine, fmt=101): per-TENSOR bits + group size ---------------
 * fmt=101 is a PRIVATE ORDINAL: 0-8 are upstream-assigned and 8 is already
 * native FP8-e4m3, so this fork mints itself a number in the 100+ experimental
 * block per colibri.c's PRIVATE ORDINAL BLOCK CONVENTION. It gets a public
 * ordinal only if a maintainer assigns one.
 * Layout per quantized weight, three safetensors tensors sharing a stem:
 *   .weight U32  [N, I*bits/32]   dense LSB-first code stream, unsigned
 *   .scales BF16 [N, I/gs]
 *   .biases BF16 [N, I/gs]        additive, already in weight space
 *   w[i] = code[i]*scale[i/gs] + bias[i/gs]
 * bits and gs come from config.json per tensor, so one reader covers oQ2e..oQ8e
 * and every third-party respin. Widths in the wild: 2,3,4,5,6,8; gs 64 or 128.
 * Verified byte-exact vs mlx.core.dequantize, incl. 3/6-bit (tools/validate_oq_dequant.py).
 * Spec + measurements: docs/oq-format.md.
 *
 * 32%bits!=0 for 3/5/6, so codes straddle words; gs*bits%32==0 for every pair
 * that occurs, so a group always starts word-aligned and the unpacker never
 * carries bits across groups. */
#ifndef COLI_OQ_H
#define COLI_OQ_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define OQ_MAX_GROUP 512                 /* unpack scratch bound; gs is 64/128 */
/* Rows processed per unpack. The batch accumulator is stack-resident and the
 * activations for these rows want to stay in L1 alongside the unpacked codes;
 * 32 keeps both true while still amortizing the unpack ~32x. */
#define OQ_MAX_BATCH 32
static inline int64_t oq_words(int I, int bits){ return (int64_t)I*bits/32; }
static inline int64_t oq_groups(int I, int gs){ return (int64_t)I/gs; }
static inline int64_t oq_rowbytes(int I, int bits, int gs){
    return oq_words(I,bits)*4 + oq_groups(I,gs)*8;   /* codes + f32 scale/bias */
}

/* one word-aligned group of gs codes -> out[0..gs). Power-of-two widths need no
 * straddle handling; 3 and 6 are specialized because 3 words hold exactly 32
 * resp. 16 values, which turns every shift into a constant (the generic cursor
 * below measured 2.4x slower, and 3-bit dominates an oQ3e checkpoint). */
static inline void oq_unpack(const uint32_t *w, int bits, int gs, uint8_t *out){
    switch(bits){
        case 8:
            for(int i=0;i<gs;i+=4){ uint32_t v=w[i>>2];
                out[i]=(uint8_t)v; out[i+1]=(uint8_t)(v>>8);
                out[i+2]=(uint8_t)(v>>16); out[i+3]=(uint8_t)(v>>24); }
            return;
        case 4:
            for(int i=0;i<gs;i+=8){ uint32_t v=w[i>>3];
                for(int j=0;j<8;j++) out[i+j]=(uint8_t)((v>>(4*j))&0xF); }
            return;
        case 2:
            for(int i=0;i<gs;i+=16){ uint32_t v=w[i>>4];
                for(int j=0;j<16;j++) out[i+j]=(uint8_t)((v>>(2*j))&0x3); }
            return;
        case 3:
            for(int i=0;i<gs;i+=32){ const uint32_t *p=w+(i>>5)*3;
                uint64_t lo=(uint64_t)p[0]|((uint64_t)p[1]<<32);
                for(int j=0;j<21;j++) out[i+j]=(uint8_t)((lo>>(3*j))&7);
                out[i+21]=(uint8_t)(((lo>>63)|((uint64_t)p[2]<<1))&7);   /* straddles */
                uint32_t hi=p[2]>>2;
                for(int j=22;j<32;j++) out[i+j]=(uint8_t)((hi>>(3*(j-22)))&7); }
            return;
        case 6:
            for(int i=0;i<gs;i+=16){ const uint32_t *p=w+(i>>4)*3;
                uint64_t lo=(uint64_t)p[0]|((uint64_t)p[1]<<32);
                for(int j=0;j<10;j++) out[i+j]=(uint8_t)((lo>>(6*j))&63);
                out[i+10]=(uint8_t)(((lo>>60)|((uint64_t)p[2]<<4))&63);  /* straddles */
                uint32_t hi=p[2]>>2;
                for(int j=11;j<16;j++) out[i+j]=(uint8_t)((hi>>(6*(j-11)))&63); }
            return;
        default: break;                  /* 5, and any future width */
    }
    uint32_t mask=(1u<<bits)-1u;
    for(int i=0,bit=0;i<gs;i++,bit+=bits){
        int wi=bit>>5, off=bit&31;
        uint64_t win=(uint64_t)w[wi];
        if(off+bits>32) win|=(uint64_t)w[wi+1]<<32;
        out[i]=(uint8_t)((win>>off)&mask);
    }
}

/* one group: dot(x, codes) and sum(x) in a single pass. The codes are uint8, so
 * widening is two vmovl steps and the multiply is plain f32 FMA. */
static inline void oq_group_dot(const uint8_t *c, const float *x, int gs,
                                float *dot_out, float *sum_out) {
#ifdef __ARM_NEON
    float32x4_t d0=vdupq_n_f32(0), d1=vdupq_n_f32(0);
    float32x4_t s0=vdupq_n_f32(0), s1=vdupq_n_f32(0);
    int i = 0;
    for (; i + 8 <= gs; i += 8) {
        uint16x8_t c16 = vmovl_u8(vld1_u8(c + i));
        float32x4_t cl = vcvtq_f32_u32(vmovl_u16(vget_low_u16(c16)));
        float32x4_t ch = vcvtq_f32_u32(vmovl_u16(vget_high_u16(c16)));
        float32x4_t x0 = vld1q_f32(x + i), x1 = vld1q_f32(x + i + 4);
        d0 = vfmaq_f32(d0, x0, cl); d1 = vfmaq_f32(d1, x1, ch);
        s0 = vaddq_f32(s0, x0);     s1 = vaddq_f32(s1, x1);
    }
    float dot = vaddvq_f32(vaddq_f32(d0, d1)), sum = vaddvq_f32(vaddq_f32(s0, s1));
    for (; i < gs; i++) { dot += x[i]*(float)c[i]; sum += x[i]; }
    *dot_out = dot; *sum_out = sum;
#else
    float dot = 0, sum = 0;
    for (int i = 0; i < gs; i++) { dot += x[i]*(float)c[i]; sum += x[i]; }
    *dot_out = dot; *sum_out = sum;
#endif
}

/* Does this width need an unpack pass at all? At 8 bits the packed stream IS a
 * byte array, so the codes can be read straight from the weight buffer and the
 * copy into scratch is pure overhead -- which matters because oQ8e checkpoints
 * are 8-bit for every tensor. */
static inline int oq_is_byte_aligned(int bits) { return bits == 8; }

/* y[S,O] = x[S,I] @ dequant(W)^T, W packed [O, I*bits/32].
 * Dequantizes inside the loop: materializing f32 would spend the whole point of
 * the format. Per group the affine form factors,
 *   sum x_i*(c_i*s + b) = s*sum(x_i*c_i) + b*sum(x_i)
 * so scale/bias apply once per group, not once per weight.
 *
 * The group loop is OUTSIDE the batch loop on purpose: a row's codes are the
 * same for every token, so unpacking once and reusing it across all S rows is
 * the difference between O(S*ng) and O(ng) unpacks. With the batched MoE path
 * feeding S=59 rows per expert, hoisting this took oq_unpack from 29% of total
 * runtime to noise (docs/oq-format.md, round 4). */
static void matmul_oq(float *y, const float *x, const uint32_t *q,
                      const float *scale, const float *bias,
                      int S, int I, int O, int bits, int gs){
    int ng=(int)oq_groups(I,gs), wpg=gs*bits/32;
    int64_t rw=oq_words(I,bits);
    const int byte_aligned = oq_is_byte_aligned(bits);
    #pragma omp parallel
    {
        uint8_t c[OQ_MAX_GROUP];
        float accs[OQ_MAX_BATCH];
        #pragma omp for schedule(static)
        for(int o=0;o<O;o++){
            const uint32_t *w=q+(int64_t)o*rw;
            const float *scl=scale+(int64_t)o*ng, *bi=bias+(int64_t)o*ng;
            for(int s0=0;s0<S;s0+=OQ_MAX_BATCH){
                int nb = S-s0 < OQ_MAX_BATCH ? S-s0 : OQ_MAX_BATCH;
                for(int s=0;s<nb;s++) accs[s]=0.f;
                for(int g=0;g<ng;g++){
                    /* ROUND 9: at 8 bits the stream is already bytes, so point at
                     * it instead of memcpy-ing every group into scratch. */
                    const uint8_t *cp;
                    if(byte_aligned) cp = (const uint8_t*)(w) + (int64_t)g*gs;
                    else { oq_unpack(w+(int64_t)g*wpg,bits,gs,c); cp = c; }
                    float sc=scl[g], bs=bi[g];
                    for(int s=0;s<nb;s++){
                        float dot, xsum;
                        oq_group_dot(cp, x+(int64_t)(s0+s)*I+(int64_t)g*gs, gs, &dot, &xsum);
                        accs[s]+=sc*dot+bs*xsum;
                    }
                }
                for(int s=0;s<nb;s++) y[(int64_t)(s0+s)*O+o]=accs[s];
            }
        }
    }
}

/* one row of dequantized weights, for callers that need f32 (oracle paths) */
static void oq_dequant_row(const uint32_t *q, const float *scale, const float *bias,
                           int row, int I, int bits, int gs, float *out){
    int ng=(int)oq_groups(I,gs), wpg=gs*bits/32;
    const uint32_t *w=q+(int64_t)row*oq_words(I,bits);
    const float *scl=scale+(int64_t)row*ng, *bi=bias+(int64_t)row*ng;
    uint8_t c[OQ_MAX_GROUP];
    for(int g=0;g<ng;g++){
        oq_unpack(w+(int64_t)g*wpg,bits,gs,c);
        float s=scl[g], b=bi[g]; float *o=out+(int64_t)g*gs;
        for(int i=0;i<gs;i++) o[i]=(float)c[i]*s+b;
    }
}

#endif /* COLI_OQ_H */
