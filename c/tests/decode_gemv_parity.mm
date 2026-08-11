/* Direct lg_decode_gemv / lg_decode_silu parity test (Phase 2 decode path).
 *
 * Builds a small random oQ/dense model, runs the persistent Metal decode GEMV
 * (all four formats: OQF32, OQBF16, F32, BF16) against a plain-C f32 reference
 * that mirrors matmul_oq's affine group factorisation (sc*dot + bi*xsum), and
 * reports max-abs-diff per format at S=1 and S=8. A real kernel bug (packing
 * order, group stride, scale type) shows up as a max-abs-diff >> 2e-4; f32
 * rounding alone stays well under that.
 *
 * Build (from c/):
 *   c++ -x objective-c++ -std=gnu++17 -fobjc-arc -O3 -mcpu=native \
 *       tests/decode_gemv_parity.mm laguna_metal.o \
 *       -framework Metal -framework MetalPerformanceShaders -framework Foundation \
 *       -o /tmp/decode_parity
 */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "../laguna_metal.h"

static uint32_t frand(uint32_t *s) { *s = *s * 1103515245 + 12345; return *s; }
static float frandf(uint32_t *s) { return (float)(frand(s) >> 8) / 8388608.0f - 1.0f; }

static uint16_t f32_to_bf16(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }
static float bf16_to_f32(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

/* LSB-first unpack, mirroring oq.h's oq_unpack layout. */
static unsigned unpack_code(uint32_t word, int bits, int idx) {
    int per = 32 / bits;
    return (word >> ((idx % per) * bits)) & ((1u << bits) - 1u);
}

int main(int argc, char **argv) {
    if (!lg_metal_init() || !lg_metal_available()) { printf("no metal; skip\n"); return 0; }
    LgDecode *d = lg_decode_new();
    if (!d) { printf("lg_decode_new failed\n"); return 1; }

    const int N = 128, K = 768, gs = 64;
    const int ng = K / gs;
    const int Svals[2] = { 1, 8 };
    uint32_t rnd = 12345;

    struct Combo { int fmt; int bits; const char *name; } combos[] = {
        { LG_DEC_OQF32,  2, "OQF32"  },
        { LG_DEC_OQF32,  4, "OQF32"  },
        { LG_DEC_OQF32,  8, "OQF32"  },
        { LG_DEC_OQBF16, 2, "OQBF16" },
        { LG_DEC_OQBF16, 8, "OQBF16" },
        { LG_DEC_F32,    0, "F32"    },
        { LG_DEC_BF16,   0, "BF16"   },
    };
    const int ncombos = (int)(sizeof(combos) / sizeof(combos[0]));

    double worst_total = 0.0;
    int failed = 0;

    for (int c = 0; c < ncombos; c++) {
        int fmt = combos[c].fmt, bits = combos[c].bits;
        const char *name = combos[c].name;
        int wwords = (K * bits + 31) / 32;
        size_t wrowb = (size_t)wwords * 4;
        uint32_t *codes = NULL;
        float   *sc32 = NULL, *bi32 = NULL;
        uint16_t *sc16 = NULL, *bi16 = NULL;
        float   *Wf = NULL;
        uint16_t *Wb = NULL;

        if (fmt == LG_DEC_OQF32 || fmt == LG_DEC_OQBF16) {
            posix_memalign((void**)&codes, 16384, (size_t)N * wwords * 4);
            memset(codes, 0, (size_t)N * wwords * 4);
            if (fmt == LG_DEC_OQF32) {
                posix_memalign((void**)&sc32, 16384, (size_t)N * ng * 4);
                posix_memalign((void**)&bi32, 16384, (size_t)N * ng * 4);
                memset(sc32, 0, (size_t)N * ng * 4); memset(bi32, 0, (size_t)N * ng * 4);
            } else {
                posix_memalign((void**)&sc16, 16384, (size_t)N * ng * 2);
                posix_memalign((void**)&bi16, 16384, (size_t)N * ng * 2);
                memset(sc16, 0, (size_t)N * ng * 2); memset(bi16, 0, (size_t)N * ng * 2);
            }
            for (int i = 0; i < N * wwords; i++) codes[i] = frand(&rnd);
            for (int i = 0; i < N * ng; i++) {
                float s = frandf(&rnd) * 0.25f, b = frandf(&rnd) * 0.25f;
                if (sc32) { sc32[i] = s; bi32[i] = b; }
                else { sc16[i] = f32_to_bf16(s); bi16[i] = f32_to_bf16(b); }
            }
        } else if (fmt == LG_DEC_F32) {
            posix_memalign((void**)&Wf, 16384, (size_t)N * K * 4);
            for (int i = 0; i < N * K; i++) Wf[i] = frandf(&rnd);
        } else {
            posix_memalign((void**)&Wb, 16384, (size_t)N * K * 2);
            for (int i = 0; i < N * K; i++) Wb[i] = f32_to_bf16(frandf(&rnd));
        }

        size_t rbytes = (size_t)8 * N * 4;
        size_t ralloc = (rbytes + 16383) & ~(size_t)16383;
        if (ralloc < 16384) ralloc = 16384;
        float *yhost = (float*)aligned_alloc(16384, ralloc);
        int ry = lg_decode_region(d, yhost, rbytes);
        if (ry < 0) { printf("region failed\n"); return 1; }

        float *xs = (float*)malloc((size_t)8 * K * 4);
        float *y_cpu = (float*)malloc(rbytes);
        float *y_gpu = yhost;

        for (int si = 0; si < 2; si++) {
            int S = Svals[si];
            for (int i = 0; i < S * K; i++) xs[i] = frandf(&rnd);
            memset(y_gpu, 0, (size_t)S * N * 4);
            memset(y_cpu, 0, (size_t)S * N * 4);

            /* GPU */
            lg_decode_begin(d, S);
            if (fmt == LG_DEC_OQF32 || fmt == LG_DEC_OQBF16) {
                const void *sc = sc32 ? (const void*)sc32 : (const void*)sc16;
                const void *bi = bi32 ? (const void*)bi32 : (const void*)bi16;
                if (!lg_decode_gemv(d, ry, 0, xs, codes, 0, sc, 0, bi, 0, N, K, bits, gs, fmt)) {
                    printf("%s bits=%d S=%d: lg_decode_gemv returned 0\n", name, bits, S);
                    failed = 1; continue;
                }
            } else {
                const void *W = fmt == LG_DEC_F32 ? (const void*)Wf : (const void*)Wb;
                if (!lg_decode_gemv(d, ry, 0, xs, W, 0, NULL, 0, NULL, 0, N, K, 0, 0, fmt)) {
                    printf("%s S=%d: lg_decode_gemv returned 0\n", name, S);
                    failed = 1; continue;
                }
            }
            if (!lg_decode_run(d)) { printf("%s S=%d: lg_decode_run failed\n", name, S); failed = 1; continue; }

            /* CPU reference */
            for (int n = 0; n < N; n++) {
                if (fmt == LG_DEC_OQF32 || fmt == LG_DEC_OQBF16) {
                    int wpg = gs * bits / 32;
                    const uint32_t *wrow = codes + (size_t)n * wwords;
                    for (int g = 0; g < ng; g++) {
                        float sc, bi;
                        if (sc32) { sc = sc32[n * ng + g]; bi = bi32[n * ng + g]; }
                        else { sc = bf16_to_f32(sc16[n * ng + g]); bi = bf16_to_f32(bi16[n * ng + g]); }
                        int per = 32 / bits;
                        for (int s = 0; s < S; s++) {
                            float dot = 0.f, xsum = 0.f;
                            const float *xr = xs + (size_t)s * K + (size_t)g * gs;
                            for (int i = 0; i < gs; i++) {
                                uint32_t word = wrow[(size_t)g * wpg + i / per];
                                unsigned code = unpack_code(word, bits, i);
                                float xv = xr[i];
                                dot += (float)code * xv;
                                xsum += xv;
                            }
                            y_cpu[(size_t)s * N + n] += sc * dot + bi * xsum;
                        }
                    }
                } else {
                    for (int s = 0; s < S; s++) {
                        float acc = 0.f;
                        const float *xr = xs + (size_t)s * K;
                        for (int k = 0; k < K; k++) {
                            float wv = fmt == LG_DEC_F32 ? Wf[(size_t)n * K + k] : bf16_to_f32(Wb[(size_t)n * K + k]);
                            acc += xr[k] * wv;
                        }
                        y_cpu[(size_t)s * N + n] = acc;
                    }
                }
            }

            double maxerr = 0.0;
            for (int i = 0; i < S * N; i++) {
                double e = fabs((double)y_gpu[i] - (double)y_cpu[i]);
                if (e > maxerr) maxerr = e;
            }
            if (maxerr > worst_total) worst_total = maxerr;
            if (maxerr > 2e-4) failed = 1;
            printf("%s bits=%d gs=%d S=%d: max|gpu-cpu| = %.6g\n", name, bits, gs, S, maxerr);
        }

        free(xs); free(y_cpu); free(yhost);
        free(codes); free(sc32); free(bi32); free(sc16); free(bi16); free(Wf); free(Wb);
    }

    /* silu fusion: gb[i] = silu(gb[i]) in place on a region-backed host buffer */
    {
        const int I = 512, S = 8, n = S * I;
        size_t bytes = (size_t)n * 4;
        size_t ralloc = (bytes + 16383) & ~(size_t)16383;
        if (ralloc < 16384) ralloc = 16384;
        float *gb = (float*)aligned_alloc(16384, ralloc);
        int ry = lg_decode_region(d, gb, bytes);
        for (int i = 0; i < n; i++) gb[i] = frandf(&rnd);
        float *ref = (float*)malloc(bytes);
        for (int i = 0; i < n; i++) ref[i] = gb[i] / (1.0f + expf(-gb[i]));
        lg_decode_begin(d, S);
        if (!lg_decode_silu(d, gb, 0, n)) { printf("lg_decode_silu returned 0\n"); failed = 1; }
        else if (!lg_decode_run(d)) { printf("silu run failed\n"); failed = 1; }
        else {
            double maxerr = 0.0;
            for (int i = 0; i < n; i++) {
                double e = fabs((double)gb[i] - (double)ref[i]);
                if (e > maxerr) maxerr = e;
            }
            if (maxerr > worst_total) worst_total = maxerr;
            if (maxerr > 2e-4) failed = 1;
            printf("SILU S=%d n=%d: max|gpu-cpu| = %.6g\n", S, n, maxerr);
        }
        free(ref); free(gb);
    }

    lg_decode_free(d);
    printf(failed ? "DECODE PARITY FAIL\n" : "DECODE PARITY OK (worst=%.6g)\n", worst_total);
    return failed ? 1 : 0;
}
