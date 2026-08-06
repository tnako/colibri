/* oQ: oMLX per-tensor mixed-precision affine quantization.
 *
 * Format spec and its verification live in docs/oq-format.md. Short version:
 *   .weight  U32  [N, K*bits/32]      dense LSB-first bitstream of unsigned codes
 *   .scales  BF16 [N, K/group_size]
 *   .biases  BF16 [N, K/group_size]
 *   w[i] = code[i] * scale[i/gs] + bias[i/gs]
 *
 * Bit width and group size are PER TENSOR, listed in config.json's
 * "quantization" map with a global scalar default for anything unlisted. That is
 * the whole point: one loader reads oQ2e / oQ3e / oQ4e / oQ5e / oQ6e / oQ8e and
 * every third-party respin of them, because the width is data, not code.
 *
 * Widths seen in the wild: 2, 3, 4, 5, 6, 8. For 3, 5 and 6 the codes STRADDLE
 * 32-bit word boundaries (32 % bits != 0), so there is no per-word padding to
 * skip. What saves the implementation is that every (bits, group_size) pair that
 * occurs has group_size*bits % 32 == 0, i.e. a group always starts on a word
 * boundary -- verified for all of them by c/tools/probe_oq.py. So the unpacker
 * can restart its bit cursor at every group and never carry state across groups.
 *
 * The dequant here was checked against mlx.core.dequantize on real downloaded
 * bytes and matches EXACTLY, including bits=3 and bits=6
 * (c/tools/validate_oq_dequant.py).
 */
#ifndef OQ_H
#define OQ_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define OQ_MAX_BITS 8

/* One quantized tensor: packed codes plus per-group affine parameters. */
typedef struct {
    uint32_t *code;      /* [rows, words_per_row]                      */
    float    *scale;     /* [rows, ngroups] widened from BF16 at load  */
    float    *bias;      /* [rows, ngroups]                            */
    int32_t   rows;      /* output dim N                               */
    int32_t   K;         /* input dim                                  */
    int32_t   bits, gs;
    int32_t   words;     /* per row: K*bits/32                         */
    int32_t   ngroups;   /* per row: K/gs                              */
} OQTensor;

static inline int oq_valid(const OQTensor *t) { return t && t->code != NULL; }

/* Bytes held resident by one oQ tensor: codes plus f32-widened scale/bias.
 * Scales and biases are kept as f32 rather than BF16 because they are touched
 * once per group per dot product, not once per weight -- widening them costs
 * K/gs floats per row (0.8% of the codes at gs=128) and saves a conversion in
 * the inner loop. */
static inline int64_t oq_bytes(const OQTensor *t) {
    if (!oq_valid(t)) return 0;
    return (int64_t)t->rows * t->words * 4 + (int64_t)t->rows * t->ngroups * 8;
}

static inline void oq_free(OQTensor *t) {
    if (!t) return;
    free(t->code); free(t->scale); free(t->bias);
    memset(t, 0, sizeof(*t));
}

/* ---- unpack one group of `gs` codes starting at a word boundary ----
 * `w` points at the first word of the group; `bits` values are packed
 * LSB-first, low bit of value 0 in bit 0 of w[0]. Returns codes in out[0..gs).
 *
 * Specialized for the power-of-two widths because those need no straddle
 * handling at all (a value never crosses a word), which is most of the volume
 * in practice: the routed experts of every oQ<N>e checkpoint are 2, 4 or 8 bit.
 */
static inline void oq_unpack_group(const uint32_t *w, int bits, int gs, uint8_t *out) {
    switch (bits) {
        case 8: {
            for (int i = 0; i < gs; i += 4) {
                uint32_t v = w[i >> 2];
                out[i]   = (uint8_t)(v & 0xFF);
                out[i+1] = (uint8_t)((v >> 8) & 0xFF);
                out[i+2] = (uint8_t)((v >> 16) & 0xFF);
                out[i+3] = (uint8_t)(v >> 24);
            }
            return;
        }
        case 4: {
            for (int i = 0; i < gs; i += 8) {
                uint32_t v = w[i >> 3];
                for (int j = 0; j < 8; j++) out[i+j] = (uint8_t)((v >> (4*j)) & 0xF);
            }
            return;
        }
        case 2: {
            for (int i = 0; i < gs; i += 16) {
                uint32_t v = w[i >> 4];
                for (int j = 0; j < 16; j++) out[i+j] = (uint8_t)((v >> (2*j)) & 0x3);
            }
            return;
        }
        case 3: {
            /* 3 words = 96 bits = exactly 32 values, so a whole number of
             * values fits in a fixed window and the per-value shift is a
             * compile-time constant. The generic cursor path below measured
             * 2.4x slower than the power-of-two widths on an M-series CPU
             * (1.92 vs 0.80 ms per 3072x3072 matvec) because every value cost a
             * division-shaped index computation plus a branch; this costs a
             * 64-bit load and a mask. 3-bit is the dominant width in oQ3e, so
             * it is worth the specialization. */
            for (int i = 0; i < gs; i += 32) {
                const uint32_t *p = w + (i >> 5) * 3;
                uint64_t lo = (uint64_t)p[0] | ((uint64_t)p[1] << 32);
                for (int j = 0; j < 21; j++) out[i+j] = (uint8_t)((lo >> (3*j)) & 7);
                /* value 21 straddles p[1]/p[2]: bits 63..65 of the group */
                out[i+21] = (uint8_t)(((lo >> 63) | ((uint64_t)p[2] << 1)) & 7);
                uint32_t hi = p[2] >> 2;
                for (int j = 22; j < 32; j++) out[i+j] = (uint8_t)((hi >> (3*(j-22))) & 7);
            }
            return;
        }
        case 6: {
            /* 3 words = 96 bits = exactly 16 values; same reasoning as bits=3. */
            for (int i = 0; i < gs; i += 16) {
                const uint32_t *p = w + (i >> 4) * 3;
                uint64_t lo = (uint64_t)p[0] | ((uint64_t)p[1] << 32);
                for (int j = 0; j < 10; j++) out[i+j] = (uint8_t)((lo >> (6*j)) & 63);
                /* value 10 straddles: bits 60..65 */
                out[i+10] = (uint8_t)(((lo >> 60) | ((uint64_t)p[2] << 4)) & 63);
                uint32_t hi = p[2] >> 2;
                for (int j = 11; j < 16; j++) out[i+j] = (uint8_t)((hi >> (6*(j-11))) & 63);
            }
            return;
        }
        default: break;                     /* 5 and anything new: generic path */
    }
    /* Generic straddling reader. The group is word-aligned, so the cursor starts
     * at 0 and the last value ends exactly on a word boundary; a 64-bit window
     * covers any value that spans two words (bits <= 8 << 32). */
    uint32_t mask = (1u << bits) - 1u;
    int64_t bit = 0;
    for (int i = 0; i < gs; i++) {
        int widx = (int)(bit >> 5), off = (int)(bit & 31);
        uint64_t win = (uint64_t)w[widx];
        if (off + bits > 32) win |= (uint64_t)w[widx + 1] << 32;
        out[i] = (uint8_t)((win >> off) & mask);
        bit += bits;
    }
}

/* Dequantize one full row into f32. Used by the loader for tensors small enough
 * to keep dense, and by the reference path. */
static inline void oq_dequant_row(const OQTensor *t, int row, float *out) {
    const uint32_t *w = t->code + (int64_t)row * t->words;
    const float *sc = t->scale + (int64_t)row * t->ngroups;
    const float *bi = t->bias  + (int64_t)row * t->ngroups;
    uint8_t codes[512];                       /* gs is 64 or 128 in practice */
    int gs = t->gs;
    int wpg = gs * t->bits / 32;              /* whole words per group */
    for (int g = 0; g < t->ngroups; g++) {
        oq_unpack_group(w + (int64_t)g * wpg, t->bits, gs, codes);
        float s = sc[g], b = bi[g];
        float *o = out + (int64_t)g * gs;
        for (int i = 0; i < gs; i++) o[i] = (float)codes[i] * s + b;
    }
}

/* ---- packed matmul: y[S,N] = x[S,K] @ dequant(W)^T ----
 *
 * Dequantizes in the INNER LOOP rather than materializing f32 weights, which is
 * the entire reason oQ saves memory. Per group the affine form factors out of
 * the dot product:
 *
 *     sum_i x_i * (code_i * s + b) = s * sum_i x_i*code_i  +  b * sum_i x_i
 *
 * so each group costs one integer-ish accumulation plus a running sum of x,
 * and the scale/bias multiply happens once per group instead of once per weight.
 * The sum of x over each group is loop-invariant across output rows, so it is
 * hoisted and computed once per (token, group) into xg.
 */
static void oq_matmul(float *y, const float *x, const OQTensor *t, int S) {
    int K = t->K, N = t->rows, gs = t->gs, ng = t->ngroups;
    int wpg = gs * t->bits / 32;
    /* per-token, per-group sum of activations: reused by every output row */
    float *xg = malloc((size_t)S * ng * sizeof(float));
    if (!xg) { fprintf(stderr, "OOM oq_matmul xg (%d x %d)\n", S, ng); exit(1); }
    for (int s = 0; s < S; s++) {
        const float *xs = x + (int64_t)s * K;
        for (int g = 0; g < ng; g++) {
            const float *xp = xs + (int64_t)g * gs;
            float acc = 0.f;
            for (int i = 0; i < gs; i++) acc += xp[i];
            xg[(int64_t)s * ng + g] = acc;
        }
    }
    #pragma omp parallel
    {
        uint8_t codes[512];
        #pragma omp for schedule(static)
        for (int n = 0; n < N; n++) {
            const uint32_t *w = t->code  + (int64_t)n * t->words;
            const float    *sc = t->scale + (int64_t)n * ng;
            const float    *bi = t->bias  + (int64_t)n * ng;
            for (int s = 0; s < S; s++) {
                const float *xs = x + (int64_t)s * K;
                const float *xgs = xg + (int64_t)s * ng;
                float total = 0.f;
                for (int g = 0; g < ng; g++) {
                    oq_unpack_group(w + (int64_t)g * wpg, t->bits, gs, codes);
                    const float *xp = xs + (int64_t)g * gs;
                    float dot = 0.f;
                    for (int i = 0; i < gs; i++) dot += xp[i] * (float)codes[i];
                    total += sc[g] * dot + bi[g] * xgs[g];
                }
                y[(int64_t)s * N + n] = total;
            }
        }
    }
    free(xg);
}

/* Same, but for a single token (S == 1): skips the xg scratch entirely, which is
 * the decode-time path and by far the hottest. */
static void oq_matvec(float *y, const float *x, const OQTensor *t) {
    int N = t->rows, gs = t->gs, ng = t->ngroups;
    int wpg = gs * t->bits / 32;
    #pragma omp parallel
    {
        uint8_t codes[512];
        #pragma omp for schedule(static)
        for (int n = 0; n < N; n++) {
            const uint32_t *w  = t->code  + (int64_t)n * t->words;
            const float    *sc = t->scale + (int64_t)n * ng;
            const float    *bi = t->bias  + (int64_t)n * ng;
            float total = 0.f;
            for (int g = 0; g < ng; g++) {
                oq_unpack_group(w + (int64_t)g * wpg, t->bits, gs, codes);
                const float *xp = x + (int64_t)g * gs;
                float dot = 0.f, xsum = 0.f;
                for (int i = 0; i < gs; i++) { dot += xp[i] * (float)codes[i]; xsum += xp[i]; }
                total += sc[g] * dot + bi[g] * xsum;
            }
            y[n] = total;
        }
    }
}

#endif /* OQ_H */
