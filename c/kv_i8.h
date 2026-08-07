/* ---- KV cache in int8 (LAGUNA-FORK) ----------------------------------------
 *
 * WHY: the KV cache is what makes long context expensive, and it was f32.
 * Laguna-S at 262144 tokens (12 full-attention layers of 48, sliding window 512):
 *
 *   f32   24.14 GiB      int8   6.04 GiB
 *
 * The whole 20 GB budget cannot hold an f32 cache at that length, so this is the
 * difference between 256k being reachable and not.
 *
 * FORMAT: symmetric per-row int8. One row is one (layer, kv_head, position)
 * vector of head_dim values, quantized with a single scale:
 *
 *   s = max|x| / 127        q[i] = round(x[i] / s)        x'[i] = q[i] * s
 *
 * Per-row rather than per-tensor because attention rows have very different
 * magnitudes across positions, and per-row costs only 4 bytes per 128 values
 * (3% overhead) while keeping the error local. Symmetric rather than affine
 * because K and V are near zero-centred and a zero point would cost another 4
 * bytes per row for no accuracy gain here.
 *
 * ACCURACY: int8 keeps ~2.4 decimal digits per value, and attention sums hd=128
 * of them, so per-row errors partially cancel. Verified token-exact on the
 * transformers-oracle fixtures, including the Laguna-S fixture whose sliding
 * window wraps the ring 50 times.
 */
#ifndef COLI_KV_I8_H
#define COLI_KV_I8_H

#include <stdint.h>
#include <math.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* Quantize one row of hd values. Writes hd int8 codes plus the scale. */
static inline void kv_i8_pack(int8_t *q, float *scale, const float *x, int hd) {
    float mx = 0.f;
    for (int i = 0; i < hd; i++) { float a = fabsf(x[i]); if (a > mx) mx = a; }
    if (mx == 0.f) { *scale = 0.f; for (int i = 0; i < hd; i++) q[i] = 0; return; }
    float s = mx / 127.f, inv = 1.f / s;
    *scale = s;
    for (int i = 0; i < hd; i++) {
        int v = (int)lrintf(x[i] * inv);
        q[i] = (int8_t)(v < -127 ? -127 : (v > 127 ? 127 : v));
    }
}

/* Dequantize one row into f32. Staged per chunk in attention() so the inner
 * loops stay plain f32 and every kernel below them is unchanged. */
static inline void kv_i8_unpack(float *out, const int8_t *q, float scale, int hd) {
#ifdef __ARM_NEON
    float32x4_t vs = vdupq_n_f32(scale);
    int i = 0;
    for (; i + 16 <= hd; i += 16) {
        int8x16_t c = vld1q_s8(q + i);
        int16x8_t lo = vmovl_s8(vget_low_s8(c)), hi = vmovl_s8(vget_high_s8(c));
        vst1q_f32(out+i,    vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),  vs));
        vst1q_f32(out+i+4,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), vs));
        vst1q_f32(out+i+8,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),  vs));
        vst1q_f32(out+i+12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), vs));
    }
    for (; i < hd; i++) out[i] = (float)q[i] * scale;
#else
    for (int i = 0; i < hd; i++) out[i] = (float)q[i] * scale;
#endif
}

#endif /* COLI_KV_I8_H */
