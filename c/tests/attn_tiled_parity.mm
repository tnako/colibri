/* Direct GPU-attention parity test (Phase 1 tiled path).
 *
 * Binds a deterministic int8 KV cache and runs lg_metal_attn (the tiled
 * full-attention path) against a plain-C f32 reference of the same softmax+
 * PV recurrence. A real bug (wrong tile math, head indexing, masking, gate)
 * shows up as a max-abs-diff >> 1e-3; rounding alone stays well under that.
 *
 * Args: S nkey H KV hd [KTILE] [pos0] [boost]
 *   boost scales the int8 values so dot products land in the regime of real
 *   hidden states (scores ~ +-10..1000), catching overflow bugs that
 *   tiny-magnitude random data hides.
 *
 * Build (from c/):
 *   clang++ -std=gnu++17 -fobjc-arc -O2 -arch arm64 \
 *       tests/attn_tiled_parity.mm laguna_metal.o laguna_attn_metal.o \
 *       -framework Metal -framework MetalPerformanceShaders \
 *       -framework Foundation -o /tmp/attn_parity
 */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "../laguna_metal.h"

static float frand(uint32_t *s) {
    *s = *s * 1103515245 + 12345;
    return (float)((*s >> 8) & 0xffffff) / 8388608.0f - 1.0f;
}
static float softplusf(float x) { return x > 20.f ? x : log1pf(expf(x)); }

int main(int argc, char **argv) {
    int S = 64, H = 8, KV = 2, hd = 32, pos0 = 0, boost = 0;
    if (argc >= 2) S = atoi(argv[1]);
    if (argc >= 4) H = atoi(argv[3]);
    if (argc >= 5) KV = atoi(argv[4]);
    if (argc >= 6) hd = atoi(argv[5]);
    if (argc >= 7) setenv("LG_KTILE", argv[6], 1);
    if (argc >= 8) pos0 = atoi(argv[7]);
    if (argc >= 9) boost = atoi(argv[8]);
    if (!lg_metal_init() || !lg_metal_available()) { printf("no metal; skip\n"); return 0; }

    int G = H / KV;
    int qdim = H * hd;
    int cache = pos0 + S + 2;  /* full layers read keys [0, pos0+S) */
    int ctxcap = cache;
    if (ctxcap & 15) ctxcap = (ctxcap + 15) & ~15;
    if (ctxcap < 4096) ctxcap = 4096;   /* page-quantized bind needs >=1 page */
    const float scale = 1.f / sqrtf((float)hd);
    size_t kvbytes = (size_t)KV * ctxcap * hd;
    size_t scalesz = (size_t)KV * ctxcap * sizeof(float);

    uint8_t *K  = (uint8_t*)aligned_alloc(16384, (kvbytes + 16383) & ~(size_t)16383);
    uint8_t *V  = (uint8_t*)aligned_alloc(16384, (kvbytes + 16383) & ~(size_t)16383);
    float   *Ks = (float*)aligned_alloc(16384, (scalesz + 16383) & ~(size_t)16383);
    float   *Vs = (float*)aligned_alloc(16384, (scalesz + 16383) & ~(size_t)16383);
    float   *q  = (float*)calloc(S * qdim, sizeof(float));
    float   *gt = (float*)calloc(S * H, sizeof(float));
    float   *out_gpu = (float*)calloc(S * qdim, sizeof(float));
    float   *out_cpu = (float*)calloc(S * qdim, sizeof(float));

    uint32_t rnd = 7;
    for (int i = 0; i < KV*ctxcap*hd; i++) { K[i] = (uint8_t)(frand(&rnd)*40*boost); V[i] = (uint8_t)(frand(&rnd)*40*boost); }
    for (int i = 0; i < KV*ctxcap; i++) { Ks[i] = 0.5f + frand(&rnd); Vs[i] = 0.5f + frand(&rnd); }
    for (int i = 0; i < S*qdim; i++) q[i] = frand(&rnd);
    for (int i = 0; i < S*H; i++)   gt[i] = frand(&rnd) * 0.5f;

    if (!lg_metal_attn_bind(1, 0, KV, ctxcap, hd, K, V, Ks, Vs,
                            kvbytes, scalesz, 0)) { printf("bind failed\n"); return 1; }
    fprintf(stderr, "harness: call S=%d H=%d KV=%d hd=%d scale=%g\n", S, H, KV, hd, scale);
    if (!lg_metal_attn(0, out_gpu, q, gt, S, pos0, H, KV, hd, scale, 0)) {
        printf("attn returned 0 (fell back)\n"); return 1; }

    /* CPU reference: full-row softmax per query, same recurrence.
     * Keys used by head hq (kv head kh) are rows [kbase, kbase+n). */
    for (int s = 0; s < S; s++) {
        for (int hq = 0; hq < H; hq++) {
            int kh = hq / G;
            int kbase = kh * ctxcap;
            int n = pos0 + s + 1; if (n > cache) n = cache; if (n < 1) n = 1;
            float *sc  = (float*)calloc(n, sizeof(float));
            float *acc = (float*)calloc(hd, sizeof(float));
            float mx = -INFINITY;
            for (int c = 0; c < n; c++) {
                float dot = 0.0f;
                for (int d = 0; d < hd; d++) {
                    float kf = (float)(int8_t)K[(kbase + c)*hd + d] * Ks[kbase + c];
                    dot += q[(long)s*qdim + hq*hd + d] * kf;
                }
                sc[c] = dot * scale;
                if (sc[c] > mx) mx = sc[c];
            }
            float sum = 0.0f;
            for (int c = 0; c < n; c++) {
                float e = expf(sc[c] - mx);
                sum += e;
                for (int d = 0; d < hd; d++) {
                    float vf = (float)(int8_t)V[(kbase + c)*hd + d] * Vs[kbase + c];
                    acc[d] += e * vf;
                }
            }
            float gate = softplusf(gt[(long)s*H + hq]);
            float inv = sum > 0.0f ? gate / sum : 0.0f;
            for (int d = 0; d < hd; d++) out_cpu[(long)s*qdim + hq*hd + d] = acc[d] * inv;
            free(sc); free(acc);
        }
    }

    double maxerr = 0.0; int worst = -1;
    for (int i = 0; i < S*qdim; i++) {
        double e = fabs((double)out_gpu[i] - (double)out_cpu[i]);
        if (e > maxerr) { maxerr = e; worst = i; }
    }
    int ws = worst / qdim, wh = (worst % qdim) / hd, wd = worst % hd;
    if (worst < 0) { ws = wh = wd = -1; }
    for (int k = 0; k < 4 && S > k; k++) {
        int s0 = (k*S)/4, h0 = k*H/4, d0 = 5;
        double g = out_gpu[(long)s0*qdim + h0*hd + d0];
        double c = out_cpu[(long)s0*qdim + h0*hd + d0];
        printf("  [s=%d h=%d d=%d] gpu=%12.6f cpu=%12.6f diff=%.4f\n",
               s0, h0, d0, g, c, g-c);
    }
    printf("S=%d cache=%d H=%d KV=%d hd=%d pos0=%d boost=%d: max |gpu-cpu| = %.6g at (s=%d h=%d d=%d)\n",
           S, cache, H, KV, hd, pos0, boost, maxerr, ws, wh, wd);
    if (getenv("LG_HDUMP")) {
        FILE *f = fopen("/tmp/h_q.bin", "wb");  if (f) { fwrite(q, 1, (size_t)S*qdim*4, f); fclose(f); }
        f = fopen("/tmp/h_K.bin", "wb");        if (f) { fwrite(K, 1, (size_t)KV*ctxcap*hd, f); fclose(f); }
        f = fopen("/tmp/h_Ks.bin", "wb");       if (f) { fwrite(Ks, 1, (size_t)KV*ctxcap*4, f); fclose(f); }
        f = fopen("/tmp/h_V.bin", "wb");        if (f) { fwrite(V, 1, (size_t)KV*ctxcap*hd, f); fclose(f); }
        f = fopen("/tmp/h_Vs.bin", "wb");       if (f) { fwrite(Vs, 1, (size_t)KV*ctxcap*4, f); fclose(f); }
        f = fopen("/tmp/h_gt.bin", "wb");       if (f) { fwrite(gt, 1, (size_t)S*H*4, f); fclose(f); }
    }
    return maxerr < 1e-2 ? 0 : 1;
}