/* Laguna (poolside Laguna-XS / Laguna-S) forward pass, shared by both sizes.
 *
 * The two checkpoints are ONE architecture at two scales: every code path here
 * is identical, only config numbers differ (D 2048/3072, L 40/48, topk 8/10,
 * moe_inter 512/1024, sliding-layer head count 64/72, YaRN factor 32/128). So
 * the engine is written once and c/laguna_xs.c / c/laguna_s.c include it; see
 * docs/laguna.md.
 *
 * What is genuinely new versus every existing Colibri engine:
 *  - PER-HEAD ATTENTION OUTPUT GATE. g_proj is Linear(D, n_heads) and the
 *    attention context is multiplied by softplus(g) per head before o_proj.
 *    No other engine gates attention output this way.
 *  - HALF-SPLIT (HF `rotate_half`) ROPE, not the interleaved scheme colibri.c
 *    and deepseek_v4.c implement. The reference says so explicitly: "Removes
 *    the interleaving of cos and sin from GLM".
 *  - PARTIAL ROTARY: full_attention layers rotate only the first
 *    head_dim * 0.5 dims and pass the rest through; sliding_attention layers
 *    rotate all of them. Two rope tables, one per layer type, with different
 *    theta (500000 vs 10000) and different rope_type (yarn vs default).
 *  - PER-LAYER ATTENTION HEAD COUNT from num_attention_heads_per_layer, with
 *    KV heads fixed at 8 (so the GQA group size differs per layer).
 *
 * What is shared rather than copied:
 *  - the MoE router (sigmoid + e_score_correction_bias top-k, renormalize,
 *    routed_scaling_factor) comes from coli_moe_route.h, the same header
 *    colibri.c's GLM-5.2 router calls.
 *
 * Weights: dense parts (attention, norms, router, shared expert, layer-0 dense
 * MLP) resident, keeping their on-disk dtype where it is bf16. Routed experts
 * are streamed per-expert from the separate
 * model.layers.N.mlp.experts.<e>.{gate,up,down}_proj.weight tensors and held in
 * an LRU cache, optionally runtime-quantized to int8 (bits=0 keeps f32 for
 * bit-exact oracle validation).
 */
#ifndef LAGUNA_COMMON_H
#define LAGUNA_COMMON_H

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#endif
#include "st.h"
#include "tok.h"
#include "route_trace.h"
#include "coli_moe_route.h"
#include "oq.h"                   /* LAGUNA-FORK: oMLX oQ packed-weight kernels */
#include "q8r.h"                  /* LAGUNA-FORK: UDOT-native resident format   */
#include "kv_i8.h"                /* LAGUNA-FORK: int8 KV cache                 */
#ifdef LAGUNA_METAL
#include "laguna_metal.h"         /* LAGUNA-FORK: MPS f16 prefill GEMM          */
#endif
#if defined(__APPLE__)
#include <mach/mach.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_in_parallel(void) { return 0; }
#endif

#ifndef LAGUNA_NAME
#define LAGUNA_NAME "Laguna"
#endif
#ifndef LAGUNA_REF_DEFAULT
#define LAGUNA_REF_DEFAULT "ref_laguna.json"
#endif

#define LG_MAXL 128
#define LG_FULL 0                 /* layer_types[i] == "full_attention"    */
#define LG_SLIDE 1                /* layer_types[i] == "sliding_attention" */

/* Routed-expert tensor layout, probed at load — all four exist in the wild and
 * none is guessed. */
enum {
    EXP_PER   = 0,   /* mlp.experts.<e>.{gate,up,down}_proj.weight (released HF) */
    EXP_FUSED = 1,   /* mlp.experts.{gate_up_proj,down_proj}, no .weight suffix  */
    EXP_FUSEDW= 2,   /* same, with .weight (what save_pretrained writes)         */
    EXP_OQ    = 3,   /* mlp.switch_mlp.*, oQ-packed [E, ...] (MLX)               */
};

/* ---------- config ---------- */
typedef struct {
    double theta, factor, attn_factor, beta_fast, beta_slow;
    int    orig_max, yarn, rot_dim;        /* rot_dim = head_dim * partial_rotary_factor */
} RopeCfg;

typedef struct {
    int hidden, n_layers, vocab;
    int n_kv, head_dim, window;
    int n_experts, topk, moe_inter, shared_inter, dense_inter;
    int n_eos, eos[8];
    float eps, routed_scale, softcap;
    int heads[LG_MAXL];                    /* num_attention_heads_per_layer */
    unsigned char slide[LG_MAXL];          /* 1 = sliding_attention layer   */
    unsigned char sparse[LG_MAXL];         /* 1 = MoE layer, 0 = dense MLP  */
    RopeCfg rope[2];                       /* [LG_FULL], [LG_SLIDE]         */
} Cfg;

/* ---------- weights ---------- */
/* f32, raw bf16, or oQ-packed (qbits>0). Same shape as inkling.c's Wt: the
 * quantized fields live inline rather than in a nested struct. */
typedef struct { float *f; uint16_t *h;
                 uint32_t *q32;      /* oQ codes, [rows * I*qbits/32]        */
                 float *qs, *qb;     /* per-group scale and bias, [rows*ng]  */
                 int gs, qbits;      /* 0 = not quantized                    */
                 int rows, in;       /* O and I, for the kernel call         */
                 void *gpu;          /* LAGUNA-FORK: f16 copy on the GPU     */
} Wt;

typedef struct {
    float *in_ln, *post_ln;
    Wt q, k, v, g, o;
    float *qn, *kn;                        /* per-head rmsnorm [head_dim] */
    Wt dg, du, dd;                         /* dense layer: gate/up/down   */
    float *router, *rbias;                 /* [E,D], [E]                  */
    Wt sh_g, sh_u, sh_d;                   /* shared expert               */
} Layer;

/* config.json "quantization": scalar bits/group_size are the default, object
 * values override per tensor stem. Declared here because Model holds one. */
typedef struct { char *name; int bits, gs; } OQOverride;

typedef struct {
    int on;                        /* checkpoint is oQ-quantized              */
    int bits, gs;                  /* defaults for anything unlisted          */
    OQOverride *ov; int n, cap;
} OQMap;

/* ---------- routed-expert LRU cache ---------- */
typedef struct {
    int eid; uint64_t used; int filled;
    float *fg, *fu, *fd;                   /* bits == 0: f32 (oracle)     */
    int8_t *qg, *qu, *qd; float *sg, *su, *sd;   /* bits > 0: int8 + row scales */
    Wt wg, wu, wd;                         /* oQ: codes stay packed       */
} Slot;
typedef struct { Slot *slots; int n, cap; } LCache;

typedef struct {
    Cfg c;
    shards S;
    int quant_bits;
    int experts;                   /* EXP_* : routed-expert tensor layout    */
    /* Resident Q8R expert bank (LAGUNA_RESIDENT=1, the default when the whole
     * model fits). eg/eu/ed are [n_layers][n_experts]; when populated the LRU
     * cache, slot_fill and all expert disk IO are bypassed entirely. */
    Q8R *eg, *eu, *ed;
    int  resident;
    /* GPU expert bank: one f16 upload per (layer, matrix), all E experts stacked
     * so a whole layer is 3 handles instead of 3*E. Populated only when the GPU
     * is present and the budget allows. */
    double mem_budget, mem_used, kv_bytes, proj_reserve;
    int ctx_hint;                  /* max context, for KV headroom accounting */
    int gpu_attn, gpu_attn_cap;    /* GPU flash attention for full layers      */
    OQMap oq;                      /* per-tensor bits/group_size from config */
    int   oq_tensors;              /* weights read oQ-packed                 */
    Wt embed, lm_head;
    float *final_norm;
    Layer *L;
    LCache *cache;
    uint32_t **eusage;
    uint64_t clock, hits, miss;
    double t_attn, t_fill, t_expert, t_shared, dense_load_s;
    /* rope tables, [pos][rot_dim], grown on demand, one pair per layer type */
    float *cos_t[2], *sin_t[2]; int rope_pos[2];
    /* KV cache: sliding layers keep only `window` slots (ring), full layers
     * keep max_t. Laid out [kv_head][kvcap][head_dim].
     *
     * int8 codes with one f32 scale per row (see kv_i8.h): 4x smaller than f32,
     * which is what makes 256k context fit in a 20 GB budget. */
    int8_t **K, **V; float **Ks, **Vs; int *kvcap; int kv_len, max_t;
} Model;

/* ---------- utility ---------- */
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
#if defined(__APPLE__)
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0*1024.0); }
#else
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0); }
#endif
/* Prefill chunk size. Defined here because both the memory budget and the
 * attention path need it; the chunking itself is in step() far below. */
#ifndef LG_CHUNK
#define LG_CHUNK 256
#endif

/* ---- per-step scratch arena (LAGUNA-FORK) ----------------------------------
 * attention() and moe() allocate ~26 S-sized buffers per layer with malloc and
 * free them again. Over 40 layers per pass that churns thousands of large
 * transient blocks, and the allocator neither coalesces nor returns them: vmmap
 * on a 6144-token prefill showed MALLOC_SMALL at 7.7 GB virtual across 1976
 * regions plus 730 MB of MALLOC_LARGE(empty), with 7.4 GB swapped out. The tell
 * was a contended run finishing the same work in 8.50 GiB where the clean run
 * reported 18.88 GiB -- a config that fits in 8.5 GiB does not need 19.
 *
 * A bump arena, reset once per layer, replaces those transient blocks with one
 * long-lived region.
 *
 * CONTRACT -- read before adding a call:
 *   1. NOT THREAD SAFE. Only call arena_alloc/afloat from single-threaded code.
 *      Buffers allocated INSIDE an `omp parallel` region are per-thread and must
 *      keep using malloc; a first version of this change converted them too and
 *      concurrent threads got overlapping memory (fixtures fell 24/24 -> 8/24).
 *   2. Lifetime ends at the next arena_reset(), which runs per layer in
 *      step_raw(). Anything that must survive the layer boundary (x, nrm, tmp,
 *      weights, KV) stays malloc'd.
 */
typedef struct {
    char  *base;
    size_t cap, used, peak;
} Arena;

static Arena g_arena;

static void arena_reset(void) { g_arena.used = 0; }

static void *arena_alloc(size_t bytes) {
    bytes = (bytes + 63) & ~(size_t)63;              /* 64B aligned */
    if (g_arena.used + bytes > g_arena.cap) {
        /* Grow to cover this layer, then never again for this S. The old region
         * may still hold live pointers from the current layer, so it is retained
         * rather than freed; growth happens a handful of times at startup. */
        size_t want = (g_arena.used + bytes) * 2;
        char *nb = (char*)malloc(want);
        if (!nb) { fprintf(stderr, "OOM arena (%.2f GB)\n", want/1e9); exit(1); }
        g_arena.base = nb; g_arena.cap = want; g_arena.used = 0;
    }
    void *p = g_arena.base + g_arena.used;
    g_arena.used += bytes;
    if (g_arena.used > g_arena.peak) g_arena.peak = g_arena.used;
    return p;
}

static float *afloat(int64_t n) {
    return (float*)arena_alloc((size_t)n * sizeof(float));
}

static float *falloc(int64_t n) {
    float *p = malloc((size_t)n*sizeof(float));
    if (!p) { fprintf(stderr, "OOM %lld floats\n", (long long)n); exit(1); }
    return p;
}
static float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }

/* pread that survives short reads and >2 GB requests. Defined here because both
 * the oQ loader and load_w need it. */
static void pread_all(int fd, void *buf, int64_t nb, int64_t off) {
    char *p = buf;
    while (nb > 0) {
        int64_t chunk = nb < (1<<30) ? nb : (1<<30);
        ssize_t got = pread(fd, p, (size_t)chunk, off);
        if (got <= 0) { perror("pread chunk"); exit(1); }
        p += got; off += got; nb -= got;
    }
}
static float siluf(float x) { return x / (1.f + expf(-x)); }
/* softplus in f32 with the large-x guard the reference's float() path gets for
 * free: expf overflows to inf above ~88 and log1pf(inf) is inf, so the gate
 * would become inf*0 = NaN on a well-behaved head. */
static float softplusf(float x) { return x > 20.f ? x : log1pf(expf(x)); }

/* f32 dot and AXPY. Attention calls these once per (query,key) pair with
 * hd=128, so at long context they run millions of times per layer; leaving them
 * as scalar loops made attention the largest phase once the MoE path was fixed
 * (docs/oq-format.md, round 5). */
#ifdef __ARM_NEON
static inline float dot_f32(const float *a, const float *b, int n) {
    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        s0 = vfmaq_f32(s0, vld1q_f32(a+i),   vld1q_f32(b+i));
        s1 = vfmaq_f32(s1, vld1q_f32(a+i+4), vld1q_f32(b+i+4));
    }
    float r = vaddvq_f32(vaddq_f32(s0, s1));
    for (; i < n; i++) r += a[i]*b[i];
    return r;
}
static inline void axpy_f32(float *y, float a, const float *x, int n) {
    float32x4_t va = vdupq_n_f32(a);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(y+i,   vfmaq_f32(vld1q_f32(y+i),   va, vld1q_f32(x+i)));
        vst1q_f32(y+i+4, vfmaq_f32(vld1q_f32(y+i+4), va, vld1q_f32(x+i+4)));
    }
    for (; i < n; i++) y[i] += a*x[i];
}
#else
static inline float dot_f32(const float *a, const float *b, int n) {
    float r = 0; for (int i = 0; i < n; i++) r += a[i]*b[i]; return r;
}
static inline void axpy_f32(float *y, float a, const float *x, int n) {
    for (int i = 0; i < n; i++) y[i] += a*x[i];
}
#endif

/* y[S,O] = x[S,I] @ W^T, W row-major [O,I] */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
#ifdef __ARM_NEON
            float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
            int i = 0;
            for (; i + 8 <= I; i += 8) {
                a0 = vfmaq_f32(a0, vld1q_f32(xs + i),     vld1q_f32(w + i));
                a1 = vfmaq_f32(a1, vld1q_f32(xs + i + 4), vld1q_f32(w + i + 4));
            }
            float acc = vaddvq_f32(vaddq_f32(a0, a1));
            for (; i < I; i++) acc += xs[i] * w[i];
#else
            float acc = 0.f;
            for (int i = 0; i < I; i++) acc += xs[i] * w[i];
#endif
            y[(int64_t)s * O + o] = acc;
        }
    }
}

/* bf16 x f32 dot. bf16->f32 IS a 16-bit left shift, so vshll_n_u16(...,16) is
 * the whole conversion -- no BF16 extension needed, works on any NEON. The
 * per-element shift+memcpy this replaced measured 19.7 GB/s vs 102.6 GB/s here
 * (5.2x) on an M5; the conversion, not memory, was the limit.
 *
 * Deliberately NOT BFDOT (FEAT_BF16): that needs BOTH operands bf16, so the f32
 * activations would be rounded to bf16 first. Measured 132 GB/s but changed the
 * result (1024.91 vs 1024.62 on the bench), and the tiny fixtures are
 * token-exact against a transformers oracle -- an f32-accurate accumulation is
 * worth more than the remaining 30%. */
#ifdef __ARM_NEON
static inline float dot_bf16_f32(const uint16_t *w, const float *x, int n) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t v = vld1q_u16(w + i);
        a0 = vfmaq_f32(a0, vld1q_f32(x + i),
                       vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v), 16)));
        a1 = vfmaq_f32(a1, vld1q_f32(x + i + 4),
                       vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(v), 16)));
    }
    float a = vaddvq_f32(vaddq_f32(a0, a1));
    for (; i < n; i++) a += x[i] * bf16_to_f32(w[i]);
    return a;
}
#else
static inline float dot_bf16_f32(const uint16_t *w, const float *x, int n) {
    float a = 0;
    for (int i = 0; i < n; i++) a += x[i] * bf16_to_f32(w[i]);
    return a;
}
#endif

static void matmul_h(float *y, const float *x, const uint16_t *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint16_t *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++)
            y[(int64_t)s * O + o] = dot_bf16_f32(w, x + (int64_t)s * I, I);
    }
}

/* Rows below which the GPU loses. A Metal dispatch round-trip measures 0.327 ms
 * on M5, so a small GEMM is pure latency; upstream colibri.c gates its own GPU
 * GEMM at 16 rows for the same reason. LG_METAL_MIN overrides for experiments. */
static int g_metal_min = 32;

static void matmul_w(float *y, const float *x, Wt W, int S, int I, int O) {
#ifdef LAGUNA_METAL
    if (W.gpu && S >= g_metal_min && !omp_in_parallel() &&
        lg_metal_gemm(W.gpu, y, x, S)) return;
#endif
    if (W.qbits) matmul_oq(y, x, W.q32, W.qs, W.qb, S, I, O, W.qbits, W.gs);
    else if (W.f) matmul(y, x, W.f, S, I, O);
    else          matmul_h(y, x, W.h, S, I, O);
}

/* y[1,O] = x @ q^T, int8 rows + per-row scale */
static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float acc = 0.f;
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
}

/* Runtime int8 row-symmetric quantization for streamed routed experts.
 *
 * BITS is 0 (keep f32) or 8. Sub-byte widths are deliberately NOT offered here:
 * the codes are stored one-per-int8_t, so a 4-bit request cost exactly the same
 * RAM as int8 while being measurably less accurate (measured on the tiny
 * fixture: bits=8 matched the oracle 12/12, bits=4 matched 11/12). That is a
 * pure loss, so it is rejected at the CLI rather than silently accepted.
 *
 * Real sub-byte weights come from oQ instead (c/oq.h), where the codes are
 * genuinely bit-packed and the width is chosen per tensor by an importance
 * matrix rather than uniformly by a flag. */
static void quantize_rows(const float *w, int8_t *q, float *scale, int O, int I, int bits) {
    int qmax = (1 << (bits - 1)) - 1;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *wr = w + (int64_t)o * I;
        float amax = 0.f;
        for (int i = 0; i < I; i++) { float a = fabsf(wr[i]); if (a > amax) amax = a; }
        float s = amax / qmax; if (s < 1e-8f) s = 1e-8f;
        scale[o] = s;
        int8_t *qr = q + (int64_t)o * I;
        for (int i = 0; i < I; i++) {
            int v = (int)lrintf(wr[i] / s);
            if (v >  qmax)   v =  qmax;
            if (v < -qmax-1) v = -qmax-1;
            qr[i] = (int8_t)v;
        }
    }
}

static void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps) {
    double ms = 0; for (int i = 0; i < D; i++) ms += (double)x[i]*x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * r * w[i];
}

static void softmax_row(float *x, int n) {
    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* ---------- rope: YaRN / default inverse frequencies, half-split apply ----------
 * Transcribed from the reference this fork was validated against:
 * transformers' _compute_yarn_parameters (modeling_rope_utils.py) and Laguna's
 * own compute_default_rope_parameters override, which differs from Gemma3's
 * only by taking `dim` from head_dim * partial_rotary_factor. Both branches are
 * here rather than reused from deepseek_v4.c's precompute: that one is written
 * against an interleaved apply step and a single layer type, and confirming the
 * formulas matched cost more than transcribing the 20 lines that do. */
static double yarn_correction_dim(double rot, int dim, double base, int max_pos) {
    return (dim * log(max_pos / (rot * 2 * M_PI))) / (2 * log(base));
}

static void rope_inv_freq(const RopeCfg *r, double *inv) {
    int dim = r->rot_dim, half = dim / 2;
    for (int i = 0; i < half; i++)
        inv[i] = 1.0 / pow(r->theta, (double)(2*i) / dim);
    if (!r->yarn) return;
    /* interpolation = divide the frequency by `factor`; extrapolation keeps it.
     * The ramp between low/high mixes the two per dimension. `truncate` is the
     * reference's default (true), hence floor/ceil here. */
    double low  = floor(yarn_correction_dim(r->beta_fast, dim, r->theta, r->orig_max));
    double high = ceil (yarn_correction_dim(r->beta_slow, dim, r->theta, r->orig_max));
    if (low < 0) low = 0;
    if (high > dim - 1) high = dim - 1;
    if (low == high) high += 0.001;
    for (int i = 0; i < half; i++) {
        double ramp = ((double)i - low) / (high - low);
        if (ramp < 0) ramp = 0;
        if (ramp > 1) ramp = 1;
        double extrap_f = 1.0 - ramp;              /* 1 - linear_ramp_factor */
        inv[i] = (inv[i] / r->factor) * (1.0 - extrap_f) + inv[i] * extrap_f;
    }
}

/* Grow cos/sin tables for layer type `lt` to cover positions [0, npos).
 * emb = cat(freqs, freqs) so cos has rot_dim entries per position, the second
 * half repeating the first — that is what makes rotate_half work. */
static void rope_grow(Model *m, int lt, int npos) {
    if (npos <= m->rope_pos[lt]) return;
    const RopeCfg *r = &m->c.rope[lt];
    int dim = r->rot_dim, half = dim / 2;
    double *inv = malloc((size_t)half * sizeof(double));
    rope_inv_freq(r, inv);
    m->cos_t[lt] = realloc(m->cos_t[lt], (size_t)npos * dim * sizeof(float));
    m->sin_t[lt] = realloc(m->sin_t[lt], (size_t)npos * dim * sizeof(float));
    if (!m->cos_t[lt] || !m->sin_t[lt]) { fprintf(stderr, "OOM rope tables\n"); exit(1); }
    for (int p = m->rope_pos[lt]; p < npos; p++) {
        float *cp = m->cos_t[lt] + (int64_t)p*dim, *sp = m->sin_t[lt] + (int64_t)p*dim;
        for (int i = 0; i < half; i++) {
            double a = (double)p * inv[i];
            float c = (float)(cos(a) * r->attn_factor), s = (float)(sin(a) * r->attn_factor);
            cp[i] = cp[i+half] = c;
            sp[i] = sp[i+half] = s;
        }
    }
    m->rope_pos[lt] = npos;
    free(inv);
}

/* in-place half-split rope on one head vector of length head_dim; only the
 * first rot_dim entries rotate, the tail passes through untouched */
static void rope_apply(float *v, const float *cs, const float *sn, int rot_dim) {
    int half = rot_dim / 2;
    for (int i = 0; i < half; i++) {
        float a = v[i], b = v[i+half];
        v[i]      = a * cs[i]      - b * sn[i];
        v[i+half] = b * cs[i+half] + a * sn[i+half];
    }
}

/* ---------- config loading ---------- */
static double jnum(jval *o, const char *k, double dflt) {
    jval *v = json_get(o, k);
    return (v && v->t == J_NUM) ? v->num : dflt;
}

static const char *jstr_at(jval *arr, int i) {
    if (!arr || arr->t != J_ARR || i >= arr->len) return NULL;
    return (arr->kids[i] && arr->kids[i]->t == J_STR) ? arr->kids[i]->str : NULL;
}

static void load_rope(RopeCfg *r, jval *o, int head_dim, double dflt_theta, double dflt_prf) {
    const char *ty = NULL;
    if (o) { jval *t = json_get(o, "rope_type"); if (t && t->t == J_STR) ty = t->str; }
    r->theta       = o ? jnum(o, "rope_theta", dflt_theta) : dflt_theta;
    double prf     = o ? jnum(o, "partial_rotary_factor", dflt_prf) : dflt_prf;
    r->rot_dim     = (int)(head_dim * prf);
    if (r->rot_dim % 2) r->rot_dim--;              /* rotate_half needs an even split */
    r->yarn        = ty && !strcmp(ty, "yarn");
    r->factor      = o ? jnum(o, "factor", 1.0) : 1.0;
    r->orig_max    = o ? (int)jnum(o, "original_max_position_embeddings", 8192) : 8192;
    r->beta_fast   = o ? jnum(o, "beta_fast", 32.0) : 32.0;
    r->beta_slow   = o ? jnum(o, "beta_slow", 1.0)  : 1.0;
    /* attention_factor: the checkpoints ship the computed value; when absent
     * the reference infers 0.1*log(factor)+1 from `factor` (get_mscale). */
    jval *af = o ? json_get(o, "attention_factor") : NULL;
    if (af && af->t == J_NUM)      r->attn_factor = af->num;
    else if (r->yarn && r->factor > 1) r->attn_factor = 0.1 * log(r->factor) + 1.0;
    else                           r->attn_factor = 1.0;
}

/* ---- oQ quantization map (config.json) ------------------------------------ */
static void oq_map_add(OQMap *m, const char *name, int bits, int gs) {
    if (m->n == m->cap) {
        m->cap = m->cap ? m->cap*2 : 256;
        m->ov = realloc(m->ov, (size_t)m->cap * sizeof(OQOverride));
        if (!m->ov) { fprintf(stderr, "OOM oq map\n"); exit(1); }
    }
    m->ov[m->n].name = strdup(name);
    m->ov[m->n].bits = bits; m->ov[m->n].gs = gs; m->n++;
}

static void oq_map_load(OQMap *m, jval *root) {
    memset(m, 0, sizeof(*m));
    jval *q = json_get(root, "quantization");
    if (!q) q = json_get(root, "quantization_config");
    if (!q || q->t != J_OBJ) return;
    m->on   = 1;
    m->bits = (int)jnum(q, "bits", 4);
    m->gs   = (int)jnum(q, "group_size", 64);
    for (int i = 0; i < q->len; i++) {
        jval *v = q->kids[i];
        if (!v || v->t != J_OBJ) continue;          /* scalars = the defaults */
        jval *md = json_get(v, "mode");
        const char *mode = (md && md->t == J_STR) ? md->str : NULL;
        /* w = c*s + b is affine-only; a future mode would need its own kernel */
        if (mode && strcmp(mode, "affine")) {
            fprintf(stderr, "oQ: '%s' has mode '%s', only 'affine' is implemented\n",
                    q->keys[i], mode);
            exit(1);
        }
        oq_map_add(m, q->keys[i], (int)jnum(v, "bits", m->bits),
                                 (int)jnum(v, "group_size", m->gs));
    }
}

/* MLX prefixes stems with "language_model."; try both spellings. Unlisted =
 * defaults, which is how routed experts are described in every oQ<N>e. */
static void oq_lookup(const OQMap *m, const char *stem, int *bits, int *gs) {
    *bits = m->bits; *gs = m->gs;
    char alt[352];
    snprintf(alt, sizeof(alt), "language_model.%s", stem);
    for (int i = 0; i < m->n; i++)
        if (!strcmp(m->ov[i].name, stem) || !strcmp(m->ov[i].name, alt)) {
            *bits = m->ov[i].bits; *gs = m->ov[i].gs; return;
        }
}

static void load_cfg(Cfg *c, const char *snap, OQMap *oq) {
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    fseek(f,0,SEEK_END); long n = ftell(f); fseek(f,0,SEEK_SET);
    char *buf = malloc((size_t)n+1);
    if (fread(buf,1,(size_t)n,f) != (size_t)n) { fprintf(stderr,"%s: short read\n",path); exit(1); }
    buf[n] = 0; fclose(f);
    char *arena = NULL; jval *r = json_parse(buf, &arena);

    c->hidden       = (int)jnum(r,"hidden_size",2048);
    c->n_layers     = (int)jnum(r,"num_hidden_layers",40);
    c->vocab        = (int)jnum(r,"vocab_size",100352);
    c->n_kv         = (int)jnum(r,"num_key_value_heads",8);
    c->head_dim     = (int)jnum(r,"head_dim",128);
    c->window       = (int)jnum(r,"sliding_window",512);
    c->n_experts    = (int)jnum(r,"num_experts",256);
    c->topk         = (int)jnum(r,"num_experts_per_tok",8);
    c->moe_inter    = (int)jnum(r,"moe_intermediate_size",512);
    c->shared_inter = (int)jnum(r,"shared_expert_intermediate_size",c->moe_inter);
    c->dense_inter  = (int)jnum(r,"intermediate_size",8192);
    c->eps          = (float)jnum(r,"rms_norm_eps",1e-6);
    c->routed_scale = (float)jnum(r,"moe_routed_scaling_factor",1.0);
    c->softcap      = (float)jnum(r,"moe_router_logit_softcapping",0.0);
    if (c->n_layers > LG_MAXL) { fprintf(stderr,"num_hidden_layers %d > %d\n", c->n_layers, LG_MAXL); exit(1); }

    /* eos_token_id is a LIST in both released checkpoints ([2, 24]); a tiny
     * fixture may write a scalar. Both are accepted, and generation stops on
     * any of them. */
    c->n_eos = 0;
    jval *eo = json_get(r,"eos_token_id");
    if (eo && eo->t == J_NUM) c->eos[c->n_eos++] = (int)eo->num;
    else if (eo && eo->t == J_ARR)
        for (int i = 0; i < eo->len && c->n_eos < 8; i++)
            if (eo->kids[i] && eo->kids[i]->t == J_NUM) c->eos[c->n_eos++] = (int)eo->kids[i]->num;

    int default_heads = (int)jnum(r,"num_attention_heads",48);
    jval *lt  = json_get(r,"layer_types");
    jval *mt  = json_get(r,"mlp_layer_types");
    jval *hpl = json_get(r,"num_attention_heads_per_layer");
    for (int i = 0; i < c->n_layers; i++) {
        const char *s = jstr_at(lt, i);
        c->slide[i] = s ? (strcmp(s,"sliding_attention") == 0) : 0;
        const char *mp = jstr_at(mt, i);
        /* config default (LagunaConfig.__post_init__): layer 0 dense, rest sparse */
        c->sparse[i] = mp ? (strcmp(mp,"sparse") == 0) : (i > 0);
        int h = default_heads;
        if (hpl && hpl->t == J_ARR && i < hpl->len && hpl->kids[i] && hpl->kids[i]->t == J_NUM)
            h = (int)hpl->kids[i]->num;
        if (h % c->n_kv) { fprintf(stderr,"layer %d: %d heads is not a multiple of %d kv heads\n", i, h, c->n_kv); exit(1); }
        c->heads[i] = h;
    }

    jval *rp = json_get(r,"rope_parameters");
    load_rope(&c->rope[LG_FULL],  rp ? json_get(rp,"full_attention")    : NULL, c->head_dim, 500000.0, 0.5);
    load_rope(&c->rope[LG_SLIDE], rp ? json_get(rp,"sliding_attention") : NULL, c->head_dim,  10000.0, 1.0);
    /* oQ map must be read before any weight load; it decides whether a tensor is
     * looked up as a packed triple or a plain float array. */
    if (oq) oq_map_load(oq, r);
    free(buf); free(arena);
}

/* Read the oQ triple at `stem` into `w`. Returns 0 if not oQ-packed, so callers
 * fall through to the plain float path. expert>=0 slices one routed expert out
 * of the leading [E, ...] axis. */
static int oq_load(Model *m, const char *stem, Wt *w, int expert) {
    char nm[384];
    snprintf(nm, sizeof(nm), "%s.weight", stem);
    st_tensor *tw = st_find(&m->S, nm);
    if (!tw || tw->dtype != 7) return 0;               /* not U32 -> not oQ */
    snprintf(nm, sizeof(nm), "%s.scales", stem);
    st_tensor *ts = st_find(&m->S, nm);
    snprintf(nm, sizeof(nm), "%s.biases", stem);
    st_tensor *tb = st_find(&m->S, nm);
    if (!ts || !tb) { fprintf(stderr, "oQ: %s: codes without scales/biases\n", stem); exit(1); }

    int bits, gs;
    oq_lookup(&m->oq, stem, &bits, &gs);
    /* I comes from the scales shape, not the config: those are two independent
     * fields and a disagreeing group_size would mis-stride every row. */
    int ng = (int)ts->shape[ts->rank-1], I = ng * gs;
    int words = (int)tw->shape[tw->rank-1], rows = (int)tw->shape[tw->rank-2];
    if ((int64_t)I*bits != (int64_t)words*32) {
        fprintf(stderr, "oQ: %s: bits=%d gs=%d ng=%d -> I=%d wants %lld words, file has %d\n",
                stem, bits, gs, ng, I, (long long)oq_words(I,bits), words); exit(1); }
    if ((gs*bits) % 32) {
        fprintf(stderr, "oQ: %s: gs*bits=%d is not whole words; unpack assumes group alignment\n",
                stem, gs*bits); exit(1); }
    if (gs > OQ_MAX_GROUP) {
        fprintf(stderr, "oQ: %s: gs=%d over OQ_MAX_GROUP\n", stem, gs); exit(1); }

    w->qbits = bits; w->gs = gs; w->rows = rows; w->in = I;
    int64_t nc = (int64_t)rows*words, nsb = (int64_t)rows*ng;
    w->q32 = malloc((size_t)nc*4);
    if (!w->q32) { fprintf(stderr, "OOM oQ codes %s\n", stem); exit(1); }
    w->qs = falloc(nsb); w->qb = falloc(nsb);
    pread_all(tw->fd, w->q32, nc*4, tw->off + (expert>=0 ? (int64_t)expert*nc : 0)*4);
    /* widen scale/bias to f32 once: read per group, not per weight */
    int64_t soff = (expert>=0 ? (int64_t)expert*nsb : 0)*2;
    uint16_t *t16 = malloc((size_t)nsb*2);
    if (!t16) { fprintf(stderr, "OOM oQ scales %s\n", stem); exit(1); }
    pread_all(ts->fd, t16, nsb*2, ts->off + soff);
    for (int64_t i = 0; i < nsb; i++) w->qs[i] = bf16_to_f32(t16[i]);
    pread_all(tb->fd, t16, nsb*2, tb->off + soff);
    for (int64_t i = 0; i < nsb; i++) w->qb[i] = bf16_to_f32(t16[i]);
    free(t16);
    return 1;
}

/* ---- resident Q8R expert bank (LAGUNA-FORK) --------------------------------
 * Expand every routed expert from packed oQ into UDOT-native Q8R, once, at
 * load. This deletes the entire streaming design at runtime: no LRU, no
 * slot_fill, no per-token disk reads, no eviction correctness hazard.
 *
 * It is affordable because expanding 2-bit codes to one byte each still lands
 * the whole XS model near 9 GiB -- BELOW the 11.05 GiB the streaming cache was
 * measured at, since that peak was dominated by cache slots and scratch. The
 * memory is spent where it converts directly into arithmetic throughput.
 *
 * Fill is parallel over experts and reads through the safetensors fd; the OS
 * page cache turns 40 layers x 256 experts of pread into mostly sequential IO.
 */
static void wt_free_oq(Wt *w) {
    free(w->q32); free(w->qs); free(w->qb);
    w->q32 = NULL; w->qs = w->qb = NULL; w->qbits = 0;
}

static int q8r_from_oq(Model *m, const char *stem, int expert, Q8R *out) {
    Wt w = {0};
    if (!oq_load(m, stem, &w, expert)) return 0;
    int O = w.rows, I = w.in, gs = w.gs, ng = I / gs;
    out->rows = O; out->in = I; out->gs = gs; out->ng = ng; out->bits = w.qbits;
    int64_t rw = oq_words(I, w.qbits);
    int wpg = gs * w.qbits / 32;
    /* codes are ADOPTED from the Wt, not copied: oq_load already malloc'd them
     * in exactly the packed layout the kernel wants */
    out->codes = w.q32;  w.q32 = NULL;
    out->scale = w.qs;   w.qs = NULL;
    out->bias  = w.qb;   w.qb = NULL;
    out->rsum  = malloc((size_t)O * ng * sizeof(float));
    if (!out->rsum) { fprintf(stderr, "OOM rsum for %s\n", stem); exit(1); }
    uint8_t sc[512];
    for (int o = 0; o < O; o++) {
        const uint32_t *src = out->codes + (int64_t)o * rw;
        for (int g = 0; g < ng; g++) {
            oq_unpack(src + (int64_t)g * wpg, w.qbits, gs, sc);
            float s = 0;
            for (int i = 0; i < gs; i++) s += sc[i];
            out->rsum[(int64_t)o*ng + g] = s;
        }
    }
    wt_free_oq(&w);
    return 1;
}

static void load_resident_experts(Model *m) {
    Cfg *c = &m->c;
    int L = c->n_layers, E = c->n_experts;
    m->eg = calloc((size_t)L*E, sizeof(Q8R));
    m->eu = calloc((size_t)L*E, sizeof(Q8R));
    m->ed = calloc((size_t)L*E, sizeof(Q8R));
    if (!m->eg || !m->eu || !m->ed) { fprintf(stderr, "OOM expert bank\n"); exit(1); }

    int fm = 0;
    while (fm < L && !c->sparse[fm]) fm++;
    char probe[384];
    const char *pfx = "";
    snprintf(probe, sizeof(probe), "model.layers.%d.mlp.switch_mlp.gate_proj.weight", fm);
    if (!st_find(&m->S, probe)) pfx = "language_model.";

    double t0 = now_s();
    int done = 0;
    for (int li = 0; li < L; li++) {
        if (!c->sparse[li]) continue;
        char sg[352], su[352], sd[352];
        snprintf(sg,sizeof(sg),"%smodel.layers.%d.mlp.switch_mlp.gate_proj",pfx,li);
        snprintf(su,sizeof(su),"%smodel.layers.%d.mlp.switch_mlp.up_proj",  pfx,li);
        snprintf(sd,sizeof(sd),"%smodel.layers.%d.mlp.switch_mlp.down_proj",pfx,li);
        #pragma omp parallel for schedule(dynamic,4)
        for (int e = 0; e < E; e++) {
            int64_t k = (int64_t)li*E + e;
            if (!q8r_from_oq(m, sg, e, &m->eg[k]) ||
                !q8r_from_oq(m, su, e, &m->eu[k]) ||
                !q8r_from_oq(m, sd, e, &m->ed[k])) {
                fprintf(stderr, "resident: layer %d expert %d missing\n", li, e); exit(1);
            }
        }
        done++;
        (void)0;
    }
    fprintf(stderr, "[mem] %d layers x %d experts unpacked in %.1fs\n", done, E, now_s() - t0);
    m->resident = 1;
}



/* ---------- weight loading ---------- */
static float *load_t(Model *m, const char *name) {
    char resolved[384];
    snprintf(resolved, sizeof(resolved), "%s", name);
    if (!st_find(&m->S, resolved)) {
        char alt[384];
        snprintf(alt, sizeof(alt), "language_model.%s", name);
        if (st_find(&m->S, alt)) snprintf(resolved, sizeof(resolved), "%s", alt);
    }
    int64_t n = st_numel(&m->S, resolved);
    if (n < 0) { fprintf(stderr, "missing %s\n", name); exit(1); }
    float *p = falloc(n);
    st_read_f32(&m->S, resolved, p, 0);
    return p;
}

/* Resolve a logical tensor stem to whatever this checkpoint actually calls it.
 * MLX prefixes everything with "language_model." and renames a few modules
 * (docs/oq-format.md has the table). Returns a pointer into `buf`. */
static const char *lg_name(Model *m, char *buf, size_t n, const char *stem, const char *suffix) {
    snprintf(buf, n, "%s%s", stem, suffix);
    if (st_find(&m->S, buf)) return buf;
    char alt[384];
    snprintf(alt, sizeof(alt), "language_model.%s%s", stem, suffix);
    if (st_find(&m->S, alt)) { snprintf(buf, n, "%s", alt); return buf; }
    /* also try the oQ triple's stem, whose ".weight" sibling may be U32 */
    snprintf(buf, n, "%s%s", stem, suffix);
    return buf;
}

/* bf16 tensors stay bf16 in RAM (Laguna S's dense set is ~10 GB at bf16 and
 * would double as f32); anything else is expanded to f32, which is what the
 * tiny oracle fixtures ship so parity checks stay bit-exact. */
static Wt load_w(Model *m, const char *name) {
    Wt w = {0};
    /* strip a trailing ".weight" to get the oQ stem; oQ stores the triple
     * <stem>.{weight,scales,biases} and `name` already carries ".weight". */
    char stem[384];
    snprintf(stem, sizeof(stem), "%s", name);
    size_t sl = strlen(stem);
    if (sl > 7 && !strcmp(stem + sl - 7, ".weight")) stem[sl-7] = 0;
    if (m->oq.on) {
        char mlx[384];
        snprintf(mlx, sizeof(mlx), "language_model.%s", stem);
        const char *cand[2] = { stem, mlx };
        for (int i = 0; i < 2; i++)
            if (oq_load(m, cand[i], &w, -1)) { m->oq_tensors++; return w; }
    }
    char resolved[384];
    lg_name(m, resolved, sizeof(resolved), stem, ".weight");
    st_tensor *t = st_find(&m->S, resolved);
    if (!t) { fprintf(stderr, "missing %s\n", name); exit(1); }
    if (t->dtype == 0) {
        w.h = malloc(t->nbytes);
        if (!w.h) { fprintf(stderr,"OOM %s\n",name); exit(1); }
        pread_all(t->fd, w.h, t->nbytes, t->off);
    } else {
        w.f = falloc(t->numel);
        st_read_f32(&m->S, resolved, w.f, 0);
    }
    return w;
}

/* one row of a resident weight as f32. embed_tokens is a Wt like any other, and
 * in an oQ checkpoint it is PACKED (8-bit in every variant seen), so the row has
 * to be dequantized rather than copied -- reading it as bf16 walks off the end
 * of a buffer that is bits/16 of the size the f32 view assumes. */

/* ---- GPU upload of the big resident projections (LAGUNA-FORK) ---------------
 * Only the per-layer attention projections and the shared expert go to the GPU.
 * They are the same weights for every token, so uploading once amortizes over
 * the whole prefill, and together they are 40% of prefill FLOPs.
 *
 * The routed experts deliberately do NOT go here: 256 per layer at f16 is 63 GB
 * for XS, far past the machine. They stay on the CPU's UDOT path.
 *
 * A GPU copy is ADDITIONAL memory, so it is only taken when it fits in the
 * budget the user allowed; otherwise the weight stays CPU-only and matmul_w
 * falls back automatically. */
#ifdef LAGUNA_METAL
static void wt_to_gpu(Wt *w, int rows, int in, double *spent, double budget) {
    if (!w || w->gpu) return;
    double need = (double)rows * in * 2;              /* f16 */
    if (*spent + need > budget) return;
    float *tmp = NULL;
    if (w->f) {
        w->gpu = lg_metal_upload(w->f, rows, in);
    } else if (w->qbits) {
        tmp = malloc((size_t)rows * in * sizeof(float));
        if (!tmp) return;
        for (int o = 0; o < rows; o++)
            oq_dequant_row(w->q32, w->qs, w->qb, o, in, w->qbits, w->gs,
                           tmp + (int64_t)o * in);
        w->gpu = lg_metal_upload(tmp, rows, in);
        free(tmp);
    } else if (w->h) {
        tmp = malloc((size_t)rows * in * sizeof(float));
        if (!tmp) return;
        for (int64_t i = 0; i < (int64_t)rows * in; i++) tmp[i] = bf16_to_f32(w->h[i]);
        w->gpu = lg_metal_upload(tmp, rows, in);
        free(tmp);
    }
    if (w->gpu) *spent += need;
}

static void load_gpu_weights(Model *m) {
    Cfg *c = &m->c;
    if (!lg_metal_init()) { fprintf(stderr, "[metal] unavailable, CPU only\n"); return; }
    /* Spend the reserve set aside before the expert cache was sized, so the
     * projections cannot be crowded out by a cache that already took the room. */
    double budget = m->proj_reserve > 0 ? m->proj_reserve : 4e9;
    double spent = 0;
    int D = c->hidden;
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        int qd = c->heads[i] * c->head_dim, kvd = c->n_kv * c->head_dim;
        wt_to_gpu(&l->q, qd,  D, &spent, budget);
        wt_to_gpu(&l->k, kvd, D, &spent, budget);
        wt_to_gpu(&l->v, kvd, D, &spent, budget);
        wt_to_gpu(&l->o, D,  qd, &spent, budget);
        wt_to_gpu(&l->sh_g, c->shared_inter, D, &spent, budget);
        wt_to_gpu(&l->sh_u, c->shared_inter, D, &spent, budget);
        wt_to_gpu(&l->sh_d, D, c->shared_inter, &spent, budget);
    }
    /* the reserve was already charged before the cache was sized; settle up with
     * what was really spent so later consumers see the truth either way */
    m->mem_used += spent - m->proj_reserve;
    fprintf(stderr, "[mem] gpu projections %.2f GB (reserved %.2f)\n",
            spent/1e9, m->proj_reserve/1e9);
}
#endif

/* one row of a resident weight as f32 (see the note above wt_to_gpu's block). */

static void wt_row_f32(Wt w, int64_t row, float *out, int n) {
    if (w.qbits)  oq_dequant_row(w.q32, w.qs, w.qb, (int)row, n, w.qbits, w.gs, out);
    else if (w.f) memcpy(out, w.f + row*n, (size_t)n * sizeof(float));
    else for (int i = 0; i < n; i++) out[i] = bf16_to_f32(w.h[row*n + i]);
}

static double mem_avail_bytes(void) {
#if defined(__linux__)
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char ln[256]; double kb = 0;
    while (fgets(ln, sizeof(ln), f)) if (sscanf(ln, "MemAvailable: %lf", &kb) == 1) break;
    fclose(f);
    return kb * 1024.0;
#elif defined(__APPLE__)
    vm_size_t page = 0; host_page_size(mach_host_self(), &page);
    vm_statistics64_data_t vs; mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vs, &n) != KERN_SUCCESS)
        return 0;
    return (double)(vs.free_count + vs.inactive_count + vs.purgeable_count) * page;
#else
    return 0;
#endif
}

static void model_init(Model *m, const char *snap, int cap, int bits) {
    memset(m, 0, sizeof(*m));
    m->quant_bits = bits;
    load_cfg(&m->c, snap, &m->oq);
    st_init(&m->S, snap);
    Cfg *c = &m->c;
    int D = c->hidden;
    double t0 = now_s();

    m->embed      = load_w(m, "model.embed_tokens.weight");
    m->final_norm = load_t(m, "model.norm.weight");
    m->lm_head    = load_w(m, "lm_head.weight");
    m->L = calloc(c->n_layers, sizeof(Layer));
    char nm[352];
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        #define LD(field, suffix)  snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); l->field = load_t(m,nm)
        #define LDW(field, suffix) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); l->field = load_w(m,nm)
        LD(in_ln,   "input_layernorm.weight");
        LD(post_ln, "post_attention_layernorm.weight");
        LDW(q, "self_attn.q_proj.weight"); LDW(k, "self_attn.k_proj.weight");
        LDW(v, "self_attn.v_proj.weight"); LDW(o, "self_attn.o_proj.weight");
        LDW(g, "self_attn.g_proj.weight");
        LD(qn, "self_attn.q_norm.weight"); LD(kn, "self_attn.k_norm.weight");
        if (!c->sparse[i]) {
            LDW(dg, "mlp.gate_proj.weight"); LDW(du, "mlp.up_proj.weight"); LDW(dd, "mlp.down_proj.weight");
        } else {
            /* MLX renames the router projection to mlp.gate.proj.weight; both
             * forms stay BF16 and unquantized in an oQ checkpoint. */
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.gate.weight",i);
            if (!st_find(&m->S, nm)) {
                char alt[384];
                snprintf(alt,sizeof(alt),"language_model.model.layers.%d.mlp.gate.proj.weight",i);
                if (st_find(&m->S, alt)) snprintf(nm,sizeof(nm),"%s",alt);
                else snprintf(nm,sizeof(nm),"model.layers.%d.mlp.gate.proj.weight",i);
            }
            l->router = load_t(m, nm);
            /* The bias hangs off the router module in transformers
             * (mlp.gate.e_score_correction_bias) but off the expert block in
             * the released checkpoints (mlp.experts.e_score_correction_bias).
             * Same tensor, two homes. */
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.e_score_correction_bias",i);
            if (!st_find(&m->S, nm))
                snprintf(nm,sizeof(nm),"model.layers.%d.mlp.gate.e_score_correction_bias",i);
            l->rbias = load_t(m, nm);
            /* Released checkpoints name it shared_expert (singular); the HF
             * module is shared_experts. Accept both instead of guessing. */
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.shared_expert.gate_proj.weight",i);
            const char *sh = "shared_expert";
            if (!st_find(&m->S, nm)) {
                char alt[384];
                snprintf(alt,sizeof(alt),"language_model.model.layers.%d.mlp.shared_expert.gate_proj.weight",i);
                if (!st_find(&m->S, alt)) sh = "shared_experts";
            }
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.%s.gate_proj.weight",i,sh); l->sh_g = load_w(m,nm);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.%s.up_proj.weight",  i,sh); l->sh_u = load_w(m,nm);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.%s.down_proj.weight",i,sh); l->sh_d = load_w(m,nm);
        }
        #undef LD
        #undef LDW
    }

    int nsp = 0; for (int i = 0; i < c->n_layers; i++) nsp += c->sparse[i];
    /* expert layout probe: look for the fused tensor on the first sparse layer */
    for (int i = 0; i < c->n_layers; i++) if (c->sparse[i]) {
        /* MLX packs routed experts as switch_mlp with a leading [E, ...] axis;
         * probe that first since an oQ checkpoint has no per-expert tensors. */
        snprintf(nm,sizeof(nm),"language_model.model.layers.%d.mlp.switch_mlp.gate_proj.weight",i);
        if (st_find(&m->S, nm)) { m->experts = EXP_OQ; break; }
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.switch_mlp.gate_proj.weight",i);
        if (st_find(&m->S, nm)) { m->experts = EXP_OQ; break; }
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.gate_up_proj",i);
        if (st_has(&m->S, nm)) m->experts = EXP_FUSED;
        else {
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.gate_up_proj.weight",i);
            if (st_has(&m->S, nm)) m->experts = EXP_FUSEDW;
        }
        break;
    }
    /* ---- ONE BUDGET, RESERVED IN PRIORITY ORDER (LAGUNA-FORK) ---------------
     * LAGUNA_MEM_GB is the whole tuning surface; everything else is derived.
     *
     * ORDER MATTERS AND WAS A BUG. The streaming cache used to be sized first,
     * from the FULL budget, and only then did GPU attention, the GPU projections
     * and the CPU KV cache take their share -- so the same 20 GB was handed out
     * three times. On Laguna-S that was cap 8.0 GB + attention 0.33 + projections
     * 3.33 + KV, against a 20 GB budget the OS then had to make up with ~10 GB of
     * swap (visible in htop with no other process to blame).
     *
     * Now every consumer is reserved against m->mem_used BEFORE the cache is
     * sized, and the cache gets only what is genuinely left. */
    {
        const char *mg = getenv("LAGUNA_MEM_GB");
        m->mem_budget = (mg ? atof(mg) : 20.0) * 1e9;
        const char *cm = getenv("CTX_MAX");
        m->ctx_hint = cm ? atoi(cm) : 8192;
        /* CPU KV cache: int8 codes + one f32 scale per row, k and v, full layers
         * at ctx and sliding layers at their window ring. */
        double kvb = 0;
        for (int i = 0; i < c->n_layers; i++) {
            double rows = (c->slide[i] && c->window > 0 && c->window < m->ctx_hint)
                        ? c->window : m->ctx_hint;
            kvb += rows * c->n_kv * (c->head_dim + 4.0) * 2;
        }
        m->mem_used += kvb;
        m->kv_bytes = kvb;
    }
#ifdef LAGUNA_METAL
    /* GPU attention K/V next: measured 3.66x on the largest phase at 30k, so it
     * outranks the expert cache. Capped at half the budget so it can never
     * starve everything else. */
    if (lg_metal_init()) {
        int gcap = m->ctx_hint > 0 ? m->ctx_hint : 8192;
        int nfull = 0, nslide = 0;
        for (int i = 0; i < c->n_layers; i++) { if (c->slide[i]) nslide++; else nfull++; }
        int sring = 2 * (c->window + LG_CHUNK);
        if (sring > gcap) sring = gcap;
        double need = ((double)nfull * gcap + (double)nslide * sring)
                    * c->n_kv * c->head_dim * 2 * 2;
        if (need <= m->mem_budget * 0.5 && need <= m->mem_budget - m->mem_used) {
            m->gpu_attn = 1; m->gpu_attn_cap = gcap;
            m->mem_used += need;
            fprintf(stderr, "[mem] gpu attention %.2f GB (%d full + %d sliding, ctx %d)\n",
                    need/1e9, nfull, nslide, gcap);
        } else {
            fprintf(stderr, "[mem] gpu attention needs %.2f GB, not affordable -> CPU\n",
                    need/1e9);
        }
    }
    /* GPU projections and the shared expert.
     *
     * UNIFIED MEMORY: on Apple silicon an MTLResourceStorageModeShared buffer is
     * ordinary RAM. Calling it "GPU memory" does not make it free, and an earlier
     * version reserved a guess (attention projections only) while the loader
     * actually uploaded the shared expert too -- dense=12288 on Laguna-S, which is
     * far larger than the projections. The cache had already taken the remainder,
     * so the total over-committed and the OS made up ~5 GB in swap.
     *
     * Reserve the exact figure the loader will spend, including the shared expert,
     * and cap it at a third of what is left. */
    m->proj_reserve = 0;
    for (int i = 0; i < c->n_layers; i++) {
        double qd = (double)c->heads[i] * c->head_dim;
        m->proj_reserve += ((double)D*qd + 2.0*D*(c->n_kv*c->head_dim) + qd*D) * 2;
        if (c->shared_inter > 0)
            m->proj_reserve += (2.0*D*c->shared_inter + (double)c->shared_inter*D) * 2;
    }
    double left_for_proj = (m->mem_budget - m->mem_used) / 3.0;
    if (m->proj_reserve > left_for_proj) m->proj_reserve = left_for_proj;
    m->mem_used += m->proj_reserve;
#endif

    int64_t I = c->moe_inter;
    /* bytes per cached expert: 3 matrices of I rows. oQ rows are packed, so an
     * oQ slot is bits/8 per weight plus the group scale/bias pair -- ~13x under
     * the f32 slot at 2-bit gs128, which is what lets a useful cache fit on S. */
    int64_t slotb = (m->experts == EXP_OQ)
        ? 3 * I * oq_rowbytes((int)D, m->oq.bits, m->oq.gs)
        : (bits ? 3*I*D + (2*I+D)*4 : 3*I*D*4);
    if (cap <= 0) {
        /* Whatever the budget has left after KV, GPU attention and projections,
         * minus a margin for the arena, page cache of the mmap'd checkpoint, and
         * the f32 dequant scratch the cache itself needs while filling. Taking
         * the whole remainder is what tipped Laguna-S into swap. */
        double bud = (m->mem_budget - m->mem_used) * 0.75;
        double avail = mem_avail_bytes();
        if (avail > 0 && avail * 0.70 < bud) bud = avail * 0.70;
        cap = (int)(bud / ((double)slotb * (nsp ? nsp : 1)));
        if (cap < 4) cap = 4;
        if (cap > c->n_experts) cap = c->n_experts;
        fprintf(stderr, "[mem] expert cache %d/layer (%.1f GB of %.1f GB left)\n",
                cap, (double)cap*slotb*nsp/1e9, bud/1e9);
    }
    m->mem_used += (double)cap * slotb * (nsp ? nsp : 1);
    m->cache = calloc(c->n_layers, sizeof(LCache));
    for (int i = 0; i < c->n_layers; i++) { m->cache[i].cap = cap; m->cache[i].slots = calloc(cap, sizeof(Slot)); }

    /* LAGUNA-FORK: when the expert bank fits, hold all of it in RAM as Q8R and
     * skip the streaming machinery. Default ON for oQ checkpoints; LAGUNA_RESIDENT=0
     * forces the streaming path (needed for Laguna-S, which does not fit). */
    if (m->experts == EXP_OQ) {
        {
            /* packed codes at the checkpoint's own bit width + f32 scale/bias/rsum
             * per group. Measured, not guessed: one-byte-per-code would be 31 GB. */
            int bpw = m->oq.bits ? m->oq.bits : 4, g = m->oq.gs ? m->oq.gs : 64;
            int64_t nw = (int64_t)3 * c->moe_inter * c->hidden;
            double need = (double)nsp * c->n_experts *
                          (nw * bpw / 8.0 + (double)(nw / g) * 3 * 4);
            /* what is left after KV, GPU attention, projections and the cache */
            double room = m->mem_budget - m->mem_used;
            double avail = mem_avail_bytes();
            if (avail > 0 && avail * 0.9 < room) room = avail * 0.9;
            if (need > room) {
                fprintf(stderr, "[mem] expert bank %.1f GB > %.1f GB free budget -> streaming\n",
                        need/1e9, room/1e9);
            } else {
                fprintf(stderr, "[mem] expert bank %.1f GB resident\n", need/1e9);
                load_resident_experts(m);
                m->mem_used += need;
            }
        }
    }

#ifdef LAGUNA_METAL
    load_gpu_weights(m);
#endif

    rt_init(LAGUNA_NAME, c->n_layers, c->n_experts);
    for (int i = 0; i < c->n_layers; i++) if (!c->sparse[i]) rt_drop_row(i);
    rt_drop_row(c->n_layers);                     /* no MTP row */
    m->eusage = rt_counts_all();
    m->dense_load_s = now_s() - t0;
}

/* ---------- routed-expert slots ---------- */
static Slot *slot_find(Model *m, int layer, int eid) {
    LCache *lc = &m->cache[layer];
    for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid == eid) {
        lc->slots[i].used = ++m->clock;
        return &lc->slots[i];
    }
    return NULL;
}

static Slot *slot_acquire(Model *m, int layer, int eid) {
    LCache *lc = &m->cache[layer]; Cfg *c = &m->c;
    int64_t D = c->hidden, I = c->moe_inter;
    Slot *s;
    if (lc->n < lc->cap) {
        s = &lc->slots[lc->n++];
        if (m->experts == EXP_OQ) {
            /* oq_load sizes its own buffers */
        } else if (m->quant_bits) {
            s->qg = malloc((size_t)I*D); s->qu = malloc((size_t)I*D); s->qd = malloc((size_t)D*I);
            if (!s->qg || !s->qu || !s->qd) { fprintf(stderr,"OOM expert slot\n"); exit(1); }
            s->sg = falloc(I); s->su = falloc(I); s->sd = falloc(D);
        } else {
            s->fg = falloc(I*D); s->fu = falloc(I*D); s->fd = falloc(D*I);
        }
    } else {
        int lru = 0;
        for (int i = 1; i < lc->n; i++) if (lc->slots[i].used < lc->slots[lru].used) lru = i;
        s = &lc->slots[lru];
    }
    s->eid = eid; s->used = ++m->clock; s->filled = 0;
    return s;
}

/* pure I/O (+ optional requant): safe to run in parallel across slots */
static void slot_fill(Model *m, int layer, Slot *s) {
    Cfg *c = &m->c;
    int64_t D = c->hidden, I = c->moe_inter;
    char nm[352];
    if (m->experts == EXP_OQ) {
        /* slice this expert out of switch_mlp's [E, ...] axis, codes stay packed.
         * A reused slot still holds the previous expert's buffers. */
        wt_free_oq(&s->wg); wt_free_oq(&s->wu); wt_free_oq(&s->wd);
        char stem[352], probe[384];
        const char *pfx = "";
        snprintf(probe,sizeof(probe),"model.layers.%d.mlp.switch_mlp.gate_proj.weight",layer);
        if (!st_find(&m->S, probe)) pfx = "language_model.";
        #define OQEXP(field, which) \
            snprintf(stem,sizeof(stem),"%smodel.layers.%d.mlp.switch_mlp." which "_proj",pfx,layer); \
            if (!oq_load(m, stem, &s->field, s->eid)) { fprintf(stderr,"oQ: %s missing\n",stem); exit(1); }
        OQEXP(wg, "gate"); OQEXP(wu, "up"); OQEXP(wd, "down");
        #undef OQEXP
        s->filled = 1;
        return;
    }
    float *tmp = falloc(2*I*D > D*I ? 2*I*D : D*I);
    float *gp, *up, *dp;                       /* f32 views of this expert */
    if (m->experts != EXP_PER) {
        /* fused [E,2I,D] gate_up (gate rows then up rows) and [E,D,I] down */
        const char *sfx = m->experts == EXP_FUSEDW ? ".weight" : "";
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.gate_up_proj%s",layer,sfx);
        st_read_slice_f32(&m->S, nm, (int64_t)s->eid*2*I*D, 2*I*D, tmp, 1);
        gp = tmp; up = tmp + I*D;
        dp = falloc(D*I);
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.down_proj%s",layer,sfx);
        st_read_slice_f32(&m->S, nm, (int64_t)s->eid*D*I, D*I, dp, 1);
    } else {
        gp = tmp; up = falloc(I*D); dp = falloc(D*I);
        #define EXP(which, dst) \
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d." which "_proj.weight",layer,s->eid); \
            st_read_f32(&m->S, nm, dst, 1)
        EXP("gate", gp); EXP("up", up); EXP("down", dp);
        #undef EXP
    }
    if (m->quant_bits) {
        quantize_rows(gp, s->qg, s->sg, I, D, m->quant_bits);
        quantize_rows(up, s->qu, s->su, I, D, m->quant_bits);
        quantize_rows(dp, s->qd, s->sd, D, I, m->quant_bits);
    } else {
        memcpy(s->fg, gp, (size_t)I*D*sizeof(float));
        memcpy(s->fu, up, (size_t)I*D*sizeof(float));
        memcpy(s->fd, dp, (size_t)D*I*sizeof(float));
    }
    if (m->experts == EXP_PER) free(up);
    free(dp); free(tmp);
    s->filled = 1;
}

/* ---------- attention ----------
 * GQA over a per-layer head count, q/k per-head RMSNorm, partial half-split
 * rope from the layer type's table, sliding window on sliding layers, and the
 * per-head softplus output gate before o_proj. */
static void attention(Model *m, Layer *l, int li, float *x, int S, int pos0, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden, H = c->heads[li], KV = c->n_kv, hd = c->head_dim;
    int lt = c->slide[li] ? LG_SLIDE : LG_FULL, rot = c->rope[lt].rot_dim;
    int qdim = H*hd, kvdim = KV*hd, group = H/KV;
    int kvcap = m->kvcap[li];
    /* arena: single-threaded, dead at this layer's arena_reset() */
    float *q  = afloat((int64_t)S*qdim);
    float *k  = afloat((int64_t)S*kvdim);
    float *vv = afloat((int64_t)S*kvdim);
    float *gt = afloat((int64_t)S*H);
    matmul_w(q,  x, l->q, S, D, qdim);
    matmul_w(k,  x, l->k, S, D, kvdim);
    matmul_w(vv, x, l->v, S, D, kvdim);
    matmul_w(gt, x, l->g, S, D, H);
    rope_grow(m, lt, pos0 + S);
    for (int s = 0; s < S; s++) {
        int pos = pos0 + s;
        const float *cs = m->cos_t[lt] + (int64_t)pos*rot, *sn = m->sin_t[lt] + (int64_t)pos*rot;
        for (int h = 0; h < H; h++) {
            float *qh = q + (int64_t)s*qdim + h*hd;
            rmsnorm_row(qh, qh, l->qn, hd, c->eps);
            rope_apply(qh, cs, sn, rot);
        }
        for (int h = 0; h < KV; h++) {
            float *kh = k + (int64_t)s*kvdim + h*hd;
            rmsnorm_row(kh, kh, l->kn, hd, c->eps);
            rope_apply(kh, cs, sn, rot);
        }
    }
    /* Scoring reads this batch's own rows (t >= pos0) straight out of the k/vv
     * scratch and older history out of the cache. The scratch holds exactly the
     * bytes the cache would hold (post-rmsnorm, post-rope), so the arithmetic is
     * unchanged — and it is what lets the append happen AFTER this loop.
     *
     * That ordering is required, not stylistic: with a `window`-row ring,
     * appending the whole batch up front overwrites history rows that earlier
     * queries in the SAME batch still need, for any prefill with S > window.
     * Silent, too — no crash, just attention built from future keys. Same
     * hazard and same resolution as upstream PR #830 for inkling.c. */
    float scale = 1.f / sqrtf((float)hd);
    float *ctx = afloat((int64_t)S*qdim);
    /* ROUND 6+7 (docs/oq-optimization-rounds.md): tile the score loop over a
     * block of QUERIES per KV head, with a CHUNKED online softmax.
     *
     * Round 6 - reuse. The old shape was one query-head x one query at a time,
     * re-walking the whole K/V history for each. With group = H/KV = 6 query
     * heads sharing a KV head, every K row was read 6 times per query and
     * re-read for all S queries. Tiling loads each K/V row once per
     * (KV head, query block) and reuses it across LG_QB queries.
     *
     * Round 7 - rescale frequency. A naive online softmax renormalizes whenever
     * the running max grows, and each renormalize is O(hd) over the accumulator.
     * Early in a row the max grows constantly, so round 6 paid that O(hd) on a
     * large fraction of keys and only bought 1.10x. Processing keys in chunks of
     * LG_KC and taking the chunk max first drops it to at most ONE rescale per
     * chunk per query: the exp() count is unchanged, the rescales fall by ~LG_KC.
     *
     * The score row is never materialized for the whole context, so the
     * per-thread scratch is O(LG_QB*LG_KC) instead of O(context) -- attention
     * scratch stops growing with prompt length. */
    #define LG_QB 8
    #define LG_KC 64

#ifdef LAGUNA_METAL
    /* ---- GPU path, full-attention layers only (LAGUNA-FORK) -----------------
     * These layers are O(S^2) and were 90.7% of the attention work at 30k
     * context (110 of 122 TFLOP), running at ~145 GFLOP/s on the CPU. Sliding
     * layers are O(S*window) and stay on the CPU, where they are already cheap.
     *
     * The GPU keeps its own f16 K/V for these layers, appended once per chunk,
     * so keys are uploaded once rather than re-read per query. Falls through to
     * the CPU path on any failure (no device, context beyond the allocation),
     * which is why the CPU KV append below still runs unconditionally. */
    /* Both layer kinds, now that the kernel is BANDED. A first attempt sent
     * sliding layers through the dense path, which computed the whole S x nkey
     * matrix and masked it away: 6x wasted work at 6k, and it regressed
     * 32.4 -> 53.1 s. The band restricts the GEMM to the window+chunk-1 columns
     * a chunk can actually reach, which is constant in context, so sliding layers
     * are now O(S*window) on the GPU exactly as they were on the CPU. */
    /* Sliding layers only go to the GPU once the context is long enough to pay
     * for the dispatch. Measured on XS: at 2k the band is 767 of 2000 columns, so
     * the CPU still wins (attn 9.8 s CPU-sliding vs 13.0 s all-GPU); by 6k the
     * band is 767 of 6000 and the GPU wins (32.2 -> 21.8 s). The crossover sits
     * near 4x the window, which is where the band stops being most of the row. */
    int gpu_ok = m->gpu_attn && (!c->slide[li] || pos0 + S >= 4 * c->window);
    if (gpu_ok &&
        lg_metal_attn_alloc(c->n_layers, li, KV, m->gpu_attn_cap, hd,
                            c->slide[li] ? c->window + LG_CHUNK : 0)) {
        lg_metal_attn_append(li, pos0, S, k, vv, kvdim);
        int win = c->slide[li] ? c->window : 0;
        /* LG_FA2=1 selects the streaming FlashAttention-2 kernel over the GEMM
         * path. Both are token-exact; they differ in memory (FA2 is O(tile) and
         * never materializes scores) and in speed, so the choice is measured. */
        static int fa2 = -1;
        if (fa2 < 0) { const char *e = getenv("LG_FA2"); fa2 = e ? atoi(e) : 0; }
        if (fa2 && lg_metal_attn2(li, ctx, q, gt, S, pos0, H, KV, hd, scale, win))
            goto attn_out;
        if (lg_metal_attn(li, ctx, q, gt, S, pos0, H, KV, hd, scale, win))
            goto attn_out;   /* the output gate is applied by scatter_o */
    }
#endif
    #pragma omp parallel
    {
        float mx[LG_QB], den[LG_QB];
        float *accum = falloc((int64_t)LG_QB * hd);
        float *sbuf  = falloc((int64_t)LG_QB * LG_KC);
        float *qt    = falloc((int64_t)LG_QB * hd);   /* contiguous query tile */
        float *kstage = falloc((int64_t)LG_KC * hd);  /* int8 KV -> f32, per chunk */
        float *vstage = falloc((int64_t)LG_KC * hd);
        #pragma omp for collapse(2) schedule(static)
        for (int kh = 0; kh < KV; kh++) {
            for (int sb = 0; sb < S; sb += LG_QB) {
                int nb = S - sb < LG_QB ? S - sb : LG_QB;
                const int8_t *Kh = m->K[li] + (int64_t)kh*kvcap*hd;
                const int8_t *Vh = m->V[li] + (int64_t)kh*kvcap*hd;
                const float  *Kq = m->Ks[li] + (int64_t)kh*kvcap;
                const float  *Vq = m->Vs[li] + (int64_t)kh*kvcap;
                /* Cached rows are int8; stage each score chunk to f32 once so the
                 * dot/axpy kernels below stay f32 and untouched. LG_KC rows is
                 * 32 KB at hd=128, i.e. L1-resident, so the dequant is amortized
                 * over all nb queries in the tile rather than done per query. */
                #define LG_KROW(t) ((t) >= pos0 ? k  + (int64_t)((t)-pos0)*kvdim + kh*hd \
                                                : kstage + (int64_t)((t) - tc)*hd)
                #define LG_VROW(t) ((t) >= pos0 ? vv + (int64_t)((t)-pos0)*kvdim + kh*hd \
                                                : vstage + (int64_t)((t) - tc)*hd)
                #define LG_KSLOT(t) ((int64_t)(c->slide[li] ? (t) % kvcap : (t)))
                for (int hq = kh*group; hq < (kh+1)*group; hq++) {
                    int hi = pos0 + sb + nb - 1;
                    int t0 = 0;
                    if (c->slide[li]) { t0 = pos0 + sb - c->window + 1; if (t0 < 0) t0 = 0; }
                    for (int b = 0; b < nb; b++) {
                        mx[b] = -INFINITY; den[b] = 0.f;
                        memset(accum + (int64_t)b*hd, 0, (size_t)hd*sizeof(float));
                    }
                    /* Hoist the query rows into a small contiguous tile. The old
                     * shape re-read q[(sb+b)*qdim + hq*hd] from a strided
                     * location inside the innermost loop; qdim is 6144 floats so
                     * consecutive b are 24 KB apart and every access missed. */
                    for (int b = 0; b < nb; b++)
                        memcpy(qt + (int64_t)b*hd, q + (int64_t)(sb+b)*qdim + hq*hd,
                               (size_t)hd*sizeof(float));
                    for (int tc = t0; tc <= hi; tc += LG_KC) {
                        int tn = hi - tc + 1; if (tn > LG_KC) tn = LG_KC;
                        /* Dequantize the cached part of this chunk once. Rows at
                         * t >= pos0 are this batch's own k/vv, still f32, and are
                         * read directly by the macros. */
                        for (int j = 0; j < tn; j++) {
                            int t = tc + j;
                            if (t >= pos0) break;
                            int64_t sl = LG_KSLOT(t);
                            kv_i8_unpack(kstage + (int64_t)j*hd, Kh + sl*hd, Kq[sl], hd);
                            kv_i8_unpack(vstage + (int64_t)j*hd, Vh + sl*hd, Vq[sl], hd);
                        }
                        for (int j = 0; j < tn; j++) {
                            const float *kv = LG_KROW(tc + j);
                            int t = tc + j;
                            /* one K row, all nb queries: K stays in L1 across
                             * the whole inner loop instead of being re-fetched */
                            for (int b = 0; b < nb; b++) {
                                int qpos = pos0 + sb + b;
                                float v = -INFINITY;
                                if (t <= qpos && !(c->slide[li] && t < qpos - c->window + 1))
                                    v = dot_f32(qt + (int64_t)b*hd, kv, hd) * scale;
                                sbuf[(int64_t)b*LG_KC + j] = v;
                            }
                        }
                        /* one rescale per query per chunk, then accumulate V */
                        for (int b = 0; b < nb; b++) {
                            float *row = sbuf + (int64_t)b*LG_KC;
                            float cmax = -INFINITY;
                            for (int j = 0; j < tn; j++) if (row[j] > cmax) cmax = row[j];
                            if (cmax == -INFINITY) continue;      /* nothing in range */
                            float nmax = mx[b] > cmax ? mx[b] : cmax;
                            if (nmax != mx[b]) {
                                float r = (mx[b] == -INFINITY) ? 0.f : expf(mx[b] - nmax);
                                den[b] *= r;
                                float *ac = accum + (int64_t)b*hd;
                                if (r == 0.f) memset(ac, 0, (size_t)hd*sizeof(float));
                                else for (int d = 0; d < hd; d++) ac[d] *= r;
                                mx[b] = nmax;
                            }
                            float *ac = accum + (int64_t)b*hd;
                            for (int j = 0; j < tn; j++) {
                                if (row[j] == -INFINITY) continue;
                                float w = expf(row[j] - mx[b]);
                                den[b] += w;
                                axpy_f32(ac, w, LG_VROW(tc + j), hd);
                            }
                        }
                    }
                    for (int b = 0; b < nb; b++) {
                        float *cx = ctx + (int64_t)(sb+b)*qdim + hq*hd;
                        float *ac = accum + (int64_t)b*hd;
                        /* per-head output gate: softplus of g_proj, one per head */
                        float gate = softplusf(gt[(int64_t)(sb+b)*H + hq]);
                        float inv = den[b] > 0.f ? gate / den[b] : 0.f;
                        for (int d = 0; d < hd; d++) cx[d] = ac[d] * inv;
                    }
                }
                #undef LG_KROW
                #undef LG_VROW
                #undef LG_KSLOT
            }
        }
        free(accum); free(sbuf); free(qt); free(kstage); free(vstage);
    }
    #undef LG_QB
    #undef LG_KC
#ifdef LAGUNA_METAL
attn_out: ;   /* empty statement: a label must precede a statement, not a decl */
#endif
    /* Append now that every query has been scored. Sliding layers skip the rows
     * this same batch would immediately overwrite: a skipped row is at position
     * < (pos0+S) - window, and no later query ever attends earlier than
     * (pos0+S) - window + 1, so it is dead on arrival. */
    int s0 = 0;
    if (c->slide[li] && S > kvcap) s0 = S - kvcap;
    for (int s = s0; s < S; s++) {
        int pos = pos0 + s, slot = c->slide[li] ? pos % kvcap : pos;
        for (int h = 0; h < KV; h++) {
            int64_t r = (int64_t)h*kvcap + slot;
            kv_i8_pack(m->K[li] + r*hd, &m->Ks[li][r], k  + (int64_t)s*kvdim + h*hd, hd);
            kv_i8_pack(m->V[li] + r*hd, &m->Vs[li][r], vv + (int64_t)s*kvdim + h*hd, hd);
        }
    }
    matmul_w(out, ctx, l->o, S, qdim, D);
    /* q/k/vv/gt/ctx are arena-owned; reclaimed by arena_reset() per layer */
}

/* ---------- dense MLP (layer 0) ---------- */
static void dense_mlp(Model *m, Layer *l, float *x, int S, float *out) {
    Cfg *c = &m->c; int D = c->hidden, I = c->dense_inter;
    float *g = falloc((int64_t)S*I), *u = falloc((int64_t)S*I);
    matmul_w(g, x, l->dg, S, D, I);
    matmul_w(u, x, l->du, S, D, I);
    for (int64_t i = 0; i < (int64_t)S*I; i++) g[i] = siluf(g[i]) * u[i];
    matmul_w(out, g, l->dd, S, I, D);
    free(g); free(u);
}

/* ---------- MoE ----------
 * Router selection and weighting come from coli_moe_route.h (shared with
 * colibri.c's GLM-5.2 router — same sigmoid + e_score_correction_bias top-k,
 * same renormalize-then-scale). Laguna always renormalizes, so norm_topk is 1.
 *
 * Order matters and follows LagunaSparseMoeBlock.forward: the ROUTED sum is
 * multiplied by moe_routed_scaling_factor, and the shared expert is added
 * AFTERWARDS, unscaled.
 *
 * Expert compute runs in rounds of at most `cap` (token, expert) pairs, so a
 * slot cannot be evicted while it is still needed — with a cache smaller than
 * the batch's distinct expert count, acquiring everything up front would hand
 * out slots that later hold a different expert's weights, silently.
 */
static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden, E = c->n_experts, K = c->topk, I = c->moe_inter;
    /* arena: all single-threaded, all dead at this layer's arena_reset() */
    float *logits = afloat((int64_t)S*E);
    matmul(logits, x, l->router, S, D, E);
    memset(out, 0, (size_t)S*D*sizeof(float));
    int   *idx = (int*)arena_alloc((size_t)S*K*sizeof(int));
    float *wgt = afloat((int64_t)S*K);
    float *choice = afloat(E);
    Slot **use  = (Slot**)arena_alloc((size_t)S*K*sizeof(Slot*));
    Slot **fill = (Slot**)arena_alloc((size_t)S*K*sizeof(Slot*));

    for (int s = 0; s < S; s++) {
        float *lg = logits + (int64_t)s*E;
        for (int e = 0; e < E; e++) {
            float z = lg[e];
            if (c->softcap > 0.f) z = tanhf(z / c->softcap) * c->softcap;
            lg[e] = sigmoidf_(z);                  /* unbiased score: the WEIGHT */
            choice[e] = lg[e] + l->rbias[e];       /* biased score: the SELECTOR */
        }
        int *si = idx + (int64_t)s*K; float *w = wgt + (int64_t)s*K;
        coli_moe_pick_topk(choice, lg, E, K, si, w, rt_router_pick, layer);
        coli_moe_norm_scale(w, K, 1, c->routed_scale);
        for (int kk = 0; kk < K; kk++) if (m->eusage[layer]) m->eusage[layer][si[kk]]++;
    }

    int cap = m->cache[layer].cap; if (cap < 1) cap = 1;
    int64_t npair = (int64_t)S*K;
    int64_t *visit = NULL;         /* streaming visit order; NULL on the resident path */

    /* ---- RESIDENT PATH (LAGUNA-FORK) ---------------------------------------
     * With the whole expert bank in RAM as Q8R there is no cache, no eviction
     * and no IO: gather each expert's tokens, run three UDOT GEMMs, scatter.
     * One pass over the experts that this batch actually touched. */
    if (m->resident) {
        int64_t *vis = (int64_t*)arena_alloc((size_t)npair * sizeof(int64_t));
        int *cnt = (int*)arena_alloc((size_t)(E + 1) * sizeof(int));
        memset(cnt, 0, (size_t)(E + 1) * sizeof(int));
        for (int64_t t = 0; t < npair; t++) cnt[idx[t] + 1]++;
        for (int e = 0; e < E; e++) cnt[e+1] += cnt[e];
        int *estart = (int*)arena_alloc((size_t)(E + 1) * sizeof(int));
        memcpy(estart, cnt, (size_t)(E + 1) * sizeof(int));
        for (int64_t t = 0; t < npair; t++) vis[cnt[idx[t]]++] = t;

        double te = now_s();
        /* Experts stay on the CPU. A GPU version was built and measured: at
         * LG_CHUNK=256 each expert sees ~8 rows, and an 8x2048 @ 2048x512 GEMM
         * is far too small for MPS -- dispatch dominates. It cost 8 GB of f16
         * weights to make expert-mm 2.5x SLOWER (13.7 -> 33.8 s), so it was
         * removed rather than left behind a flag. UDOT needs no dispatch. */
        #pragma omp parallel
        {
            int64_t rcap = 0;
            float *xb=NULL,*gb=NULL,*ub=NULL,*hb=NULL;
            Q8Act ax={0}, ah={0};
            #pragma omp for schedule(dynamic,1)
            for (int e = 0; e < E; e++) {
                int g0 = estart[e], g1 = estart[e+1], nr = g1 - g0;
                if (nr <= 0) continue;
                int64_t k = (int64_t)layer*E + e;
                /* Realloc when the row count grows OR the group size changes:
                 * oQ checkpoints mix group sizes per tensor, and Q8Act's ng
                 * (hence every metadata array) is derived from gs. Keying the
                 * cache on rows alone silently reused a buffer sized for a
                 * different ng. */
                if (nr > rcap || ax.gs != m->eg[k].gs || ah.gs != m->ed[k].gs) {
                    if (nr > rcap) rcap = nr;
                    free(xb);free(gb);free(ub);free(hb);
                    q8act_free(&ax); q8act_free(&ah);
                    xb=falloc(rcap*D); gb=falloc(rcap*I);
                    ub=falloc(rcap*I); hb=falloc(rcap*D);
                    q8act_alloc(&ax,(int)rcap,D,m->eg[k].gs);
                    q8act_alloc(&ah,(int)rcap,I,m->ed[k].gs);
                }
                for (int r = 0; r < nr; r++)
                    memcpy(xb + (int64_t)r*D, x + (vis[g0+r]/K)*D, (size_t)D*sizeof(float));
                /* one activation quantization feeds BOTH gate and up */
                /* CPU UDOT path. The GPU equivalent is the batched dispatch
                 * above, issued single-threaded; calling Metal from in here raced
                 * on its shared scratch. */
                {
                ax.rows = nr; q8act_fill(&ax, xb);
                q8r_gemm(gb, &ax, &m->eg[k]);
                q8r_gemm(ub, &ax, &m->eu[k]);
                for (int64_t i = 0; i < (int64_t)nr*I; i++) gb[i] = siluf(gb[i]) * ub[i];
                ah.rows = nr; q8act_fill(&ah, gb);
                q8r_gemm(hb, &ah, &m->ed[k]);
                }
                for (int r = 0; r < nr; r++) {
                    int64_t t = vis[g0+r];
                    int s = (int)(t / K), kk = (int)(t % K);
                    float sc = wgt[(int64_t)s*K + kk];
                    float *os = out + (int64_t)s*D, *hr = hb + (int64_t)r*D;
                    for (int d = 0; d < D; d++) {
                        #pragma omp atomic
                        os[d] += sc * hr[d];
                    }
                }
            }
            free(xb);free(gb);free(ub);free(hb);
            q8act_free(&ax); q8act_free(&ah);
        }
        m->t_expert += now_s() - te;
        m->hits += npair;
        visit = NULL;              /* streaming path never allocated it here */
        goto moe_shared;
    }


    /* ROUND 8 (docs/oq-optimization-rounds.md): walk the pairs in EXPERT order.
     *
     * The pair list is naturally in token order, so a chunk of `cap` pairs holds
     * up to `cap` DIFFERENT experts; the next chunk needs a different set, evicts
     * them, and the one after reloads what the first already had. At 6K tokens
     * with cap=48 that produced a 57% hit rate and made slot_fill (disk IO) a
     * top phase, even though only 256 distinct experts exist in the whole layer.
     *
     * Sorting the visit order by expert id means each expert is loaded at most
     * once per layer per call: all of its tokens are consumed while it is
     * resident. Pure scheduling change -- the arithmetic per pair, and the
     * `cap`-sized eviction safety property, are untouched. */
    visit = (int64_t*)arena_alloc((size_t)npair * sizeof(int64_t));
    {   /* counting sort over expert id: O(npair + E), no comparator */
        int *cnt = calloc((size_t)E + 1, sizeof(int));
        if (!cnt) { fprintf(stderr, "OOM moe counting sort\n"); exit(1); }
        for (int64_t t = 0; t < npair; t++) cnt[idx[t] + 1]++;
        for (int e = 0; e < E; e++) cnt[e+1] += cnt[e];
        for (int64_t t = 0; t < npair; t++) visit[cnt[idx[t]]++] = t;
        free(cnt);
    }

    for (int64_t base = 0; base < npair; base += cap) {
        int64_t end = base + cap < npair ? base + cap : npair;
        int nfill = 0;
        for (int64_t vi = base; vi < end; vi++) {
            int64_t t = visit[vi];
            int eid = idx[t];
            Slot *e = slot_find(m, layer, eid);
            if (e) m->hits++;
            else { m->miss++; e = slot_acquire(m, layer, eid); fill[nfill++] = e; }
            use[vi - base] = e;
        }
        if (nfill) {
            double tf = now_s();
            #pragma omp parallel for schedule(dynamic,1)
            for (int j = 0; j < nfill; j++) slot_fill(m, layer, fill[j]);
            m->t_fill += now_s() - tf;
        }
        for (int64_t vi = base; vi < end; vi++) {
            if (use[vi - base]->eid != idx[visit[vi]]) {
                fprintf(stderr, "layer %d: cache served expert %d for requested expert %d\n",
                        layer, use[vi - base]->eid, (int)idx[visit[vi]]);
                exit(1);
            }
        }
        double te = now_s();
        /* ROUND 1+3 (docs/oq-format.md): batch the pairs BY EXPERT, then run one
         * matmul per expert over all its tokens.
         *
         * Round 1 replaced per-matmul OpenMP regions (52% of time was
         * __psynch_cvwait: ~1.8M barriers per layer, each guarding I=512 rows)
         * with one region over the (token,expert) pairs.
         *
         * Round 3 fixes the bigger waste that exposed: with S=1 per call, an
         * expert's packed weights were unpacked once PER TOKEN. At S=1902/K=8
         * over 256 experts that is ~59 tokens per expert, so oq_unpack ran ~59x
         * more than necessary (28% of all time). Gathering each expert's tokens
         * into a contiguous batch turns the GEMV into a GEMM: the unpack cost is
         * paid once per group and amortizes across the batch, and the inner loop
         * gets row reuse it never had.
         *
         * Same shape as upstream's coli_metal_moe_gemv (packed activations +
         * per-expert row offsets), so the two stay comparable.
         *
         * omp_in_parallel(): nesting is off, so an inner region from an
         * already-parallel caller would serialize anyway. */
        int npair_c = (int)(end - base);
        /* order the pairs by slot so each expert's tokens are contiguous */
        int *ord = (int*)arena_alloc((size_t)npair_c * sizeof(int));
        int *estart = (int*)arena_alloc((size_t)(npair_c + 1) * sizeof(int));
        for (int i = 0; i < npair_c; i++) ord[i] = i;
        /* insertion sort by slot pointer: npair_c <= cap (a few hundred) and the
         * list is already clustered, so this beats pulling in a qsort callback */
        for (int i = 1; i < npair_c; i++) {
            int v = ord[i]; Slot *sv = use[v]; int j = i - 1;
            while (j >= 0 && (uintptr_t)use[ord[j]] > (uintptr_t)sv) { ord[j+1] = ord[j]; j--; }
            ord[j+1] = v;
        }
        int ngrp = 0;
        for (int i = 0; i < npair_c; ) {
            estart[ngrp++] = i;
            Slot *e = use[ord[i]];
            while (i < npair_c && use[ord[i]] == e) i++;
        }
        estart[ngrp] = npair_c;

        int par = ngrp > 1 && !omp_in_parallel();
        /* One result row per pair in this chunk. npair_c <= cap (a few hundred),
         * so this is ~1 MB and lets the scatter run serially afterwards -- no
         * atomics and no per-thread copy of out[]. */
        float *res = afloat((int64_t)npair_c * D);
        #pragma omp parallel if(par)
        {
            /* scratch sized for the largest group seen by this thread */
            int64_t rcap = 0;
            float *xb = NULL, *gb = NULL, *ub = NULL, *hb = NULL;
            #pragma omp for schedule(dynamic,1)
            for (int gi = 0; gi < ngrp; gi++) {
                int g0 = estart[gi], g1 = estart[gi+1], nr = g1 - g0;
                Slot *e = use[ord[g0]];
                if (nr > rcap) {
                    rcap = nr;
                    free(xb); free(gb); free(ub); free(hb);
                    xb = falloc(rcap*D); gb = falloc(rcap*I);
                    ub = falloc(rcap*I); hb = falloc(rcap*D);
                }
                for (int r = 0; r < nr; r++) {
                    int64_t t = visit[base + ord[g0 + r]];
                    memcpy(xb + (int64_t)r*D, x + (t / K)*D, (size_t)D*sizeof(float));
                }
                /* down_proj also goes through the batched kernel: writing into a
                 * scratch then permuting is cheaper than S separate GEMVs, each
                 * of which would re-unpack the whole weight. */
                if (m->experts == EXP_OQ) {
                    matmul_w(gb, xb, e->wg, nr, D, I);
                    matmul_w(ub, xb, e->wu, nr, D, I);
                    for (int64_t i = 0; i < (int64_t)nr*I; i++) gb[i] = siluf(gb[i]) * ub[i];
                    matmul_w(hb, gb, e->wd, nr, I, D);
                } else if (m->quant_bits) {
                    for (int r = 0; r < nr; r++) {
                        matmul_q(gb + (int64_t)r*I, xb + (int64_t)r*D, e->qg, e->sg, D, I);
                        matmul_q(ub + (int64_t)r*I, xb + (int64_t)r*D, e->qu, e->su, D, I);
                    }
                    for (int64_t i = 0; i < (int64_t)nr*I; i++) gb[i] = siluf(gb[i]) * ub[i];
                    for (int r = 0; r < nr; r++)
                        matmul_q(hb + (int64_t)r*D, gb + (int64_t)r*I, e->qd, e->sd, I, D);
                } else {
                    matmul(gb, xb, e->fg, nr, D, I);
                    matmul(ub, xb, e->fu, nr, D, I);
                    for (int64_t i = 0; i < (int64_t)nr*I; i++) gb[i] = siluf(gb[i]) * ub[i];
                    matmul(hb, gb, e->fd, nr, I, D);
                }
                /* place each row where the serial scatter expects it */
                for (int r = 0; r < nr; r++)
                    memcpy(res + (int64_t)ord[g0+r]*D, hb + (int64_t)r*D,
                           (size_t)D*sizeof(float));
            }
            free(xb); free(gb); free(ub);
        }
        /* serial weighted scatter: cheap next to the matmuls and collision-free */
        for (int i = 0; i < npair_c; i++) {
            int64_t t = visit[base + i];
            int s = (int)(t / K), kk = (int)(t % K);
            float sc = wgt[(int64_t)s*K + kk];
            float *os = out + (int64_t)s*D, *hr = res + (int64_t)i*D;
            for (int d = 0; d < D; d++) os[d] += sc * hr[d];
        }
        /* res/ord/estart are arena-owned */
        m->t_expert += now_s() - te;
    }
    /* g/u/hh are arena-owned */

moe_shared:
    /* shared expert: every token, unscaled, added on top of the routed sum */
    { double ts = now_s();
    int SI = c->shared_inter;
    float *sg = afloat((int64_t)S*SI), *su = afloat((int64_t)S*SI), *sd = afloat((int64_t)S*D);
    matmul_w(sg, x, l->sh_g, S, D, SI);
    matmul_w(su, x, l->sh_u, S, D, SI);
    for (int64_t i = 0; i < (int64_t)S*SI; i++) sg[i] = siluf(sg[i]) * su[i];
    matmul_w(sd, sg, l->sh_d, S, SI, D);
    for (int64_t i = 0; i < (int64_t)S*D; i++) out[i] += sd[i];
    /* sg/su/sd are arena-owned */
    m->t_shared += now_s() - ts; }

    /* logits/idx/wgt/choice/use/fill/visit are arena-owned (arena_reset per layer) */
}

/* ---------- one forward pass over S new tokens ----------
 * Returns malloc'd logits for the last position. tf_out, when non-NULL, also
 * receives the per-position argmax (teacher-forcing parity check). */
static float *step_raw(Model *m, const int *ids, int S, int pos0, int *tf_out) {
    Cfg *c = &m->c; int D = c->hidden;
    float *x = falloc((int64_t)S*D);
    /* SEC: ids come from tok_encode, i.e. from tokenizer.json, while the row
     * count of embed_tokens comes from config.json. Those are two files and
     * nothing makes them agree — a tokenizer paired with the wrong checkpoint
     * (or a snapshot assembled by hand) yields ids past the table and this
     * reads off the end of the embedding allocation. Refuse by name instead. */
    for (int s = 0; s < S; s++) {
        if (ids[s] < 0 || ids[s] >= c->vocab) {
            fprintf(stderr, "token id %d at position %d is outside vocab_size %d "
                            "(tokenizer.json and config.json disagree)\n",
                    ids[s], pos0 + s, c->vocab);
            exit(1);
        }
    }
    for (int s = 0; s < S; s++) wt_row_f32(m->embed, ids[s], x + (int64_t)s*D, D);
    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        /* Arena scratch from the previous layer is dead once its residual has
         * been added, so one reset per layer caps the high-water mark at a single
         * layer's scratch. x/nrm/tmp outlive the loop and stay malloc'd. */
        arena_reset();
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->in_ln, D, c->eps);
        double ta = now_s();
        attention(m, l, i, nrm, S, pos0, tmp);
        m->t_attn += now_s() - ta;
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->post_ln, D, c->eps);
        if (c->sparse[i]) moe(m, l, i, nrm, S, tmp);
        else              dense_mlp(m, l, nrm, S, tmp);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
    }
    m->kv_len = pos0 + S;
    float *last = falloc(D);
    float *logit = falloc(c->vocab);
    if (tf_out) {
        for (int s = 0; s < S; s++) {
            rmsnorm_row(last, x + (int64_t)s*D, m->final_norm, D, c->eps);
            matmul_w(logit, last, m->lm_head, 1, D, c->vocab);
            int best = 0; for (int i = 1; i < c->vocab; i++) if (logit[i] > logit[best]) best = i;
            tf_out[pos0 + s] = best;
        }
    }
    rmsnorm_row(last, x + (int64_t)(S-1)*D, m->final_norm, D, c->eps);
    matmul_w(logit, last, m->lm_head, 1, D, c->vocab);
    free(x); free(nrm); free(tmp); free(last);
    return logit;
}

/* Feed an arbitrary number of positions. Sliding layers stay correct for any S
 * because attention() appends after scoring (see there).
 *
 * CHUNKED (LAGUNA-FORK): a single 262144-token call would need f32 scratch that
 * scales with S -- q and ctx alone are S*heads*head_dim*4, which is 6 GiB each at
 * 256k for Laguna-S. Splitting the prompt into fixed-size chunks makes every
 * scratch buffer O(chunk) instead of O(S), so peak memory stops depending on
 * context length at all. The KV cache still holds the full history, so results
 * are identical: each chunk attends to everything before it exactly as a single
 * big call would.
 *
 * The chunk is large enough that per-call overhead is amortized and the GPU GEMMs
 * still see plenty of rows (LG_CHUNK=1024 keeps MPS well above its efficiency
 * threshold, measured 15.5 TFLOP/s at S>=1024). */
/* LG_CHUNK is defined at the top of this file. Swept with c/tools/tune_chunk.sh
 * on XS at 1902 tokens: prefill is flat across 256..4096 (29.7-32.5 s, i.e.
 * noise) but peak RSS is not -- 16.5 GB at 256 against 20.2 GB at 4096, because
 * every scratch buffer is O(chunk). The goal is long context inside a fixed
 * budget, so take the smallest chunk that costs no speed. */

static float *step(Model *m, const int *ids, int S, int pos0, int *tf_out) {
    if (S <= LG_CHUNK) return step_raw(m, ids, S, pos0, tf_out);
    float *logit = NULL;
    for (int off = 0; off < S; off += LG_CHUNK) {
        int n = S - off < LG_CHUNK ? S - off : LG_CHUNK;
        free(logit);
        /* tf_out (teacher-forcing argmax per position) is filled per chunk at the
         * matching offset so chunking is invisible to the fixtures. */
        logit = step_raw(m, ids + off, n, pos0 + off, tf_out ? tf_out + off : NULL);
    }
    return logit;
}

static void kv_alloc(Model *m, int max_t) {
    Cfg *c = &m->c;
    if (m->K && max_t <= m->max_t) return;
    if (m->K) {
        for (int i = 0; i < c->n_layers; i++) {
            free(m->K[i]); free(m->V[i]); free(m->Ks[i]); free(m->Vs[i]);
        }
        free(m->K); free(m->V); free(m->Ks); free(m->Vs); free(m->kvcap);
    }
    m->max_t = max_t; m->kv_len = 0;
    m->K  = calloc(c->n_layers, sizeof(int8_t*));
    m->V  = calloc(c->n_layers, sizeof(int8_t*));
    m->Ks = calloc(c->n_layers, sizeof(float*));
    m->Vs = calloc(c->n_layers, sizeof(float*));
    m->kvcap = calloc(c->n_layers, sizeof(int));
    for (int i = 0; i < c->n_layers; i++) {
        /* sliding layers only ever read the last `window` positions, so the
         * ring is exactly `window` rows — the post-scoring append in
         * attention() is what makes that safe during prefill (PR #830). */
        int cap = (c->slide[i] && c->window > 0 && c->window < max_t) ? c->window : max_t;
        m->kvcap[i] = cap;
        int64_t nrow = (int64_t)c->n_kv * cap;
        m->K[i]  = malloc((size_t)nrow * c->head_dim);
        m->V[i]  = malloc((size_t)nrow * c->head_dim);
        m->Ks[i] = falloc(nrow);
        m->Vs[i] = falloc(nrow);
        if (!m->K[i] || !m->V[i]) { fprintf(stderr, "OOM kv cache\n"); exit(1); }
    }
}

static int is_eos(Cfg *c, int tok) {
    for (int i = 0; i < c->n_eos; i++) if (c->eos[i] == tok) return 1;
    return 0;
}

/* greedy generation into out[] (prompt copied in first) */
static void generate(Model *m, const int *prompt, int np, int n_new, int *out, int *n_out) {
    for (int i = 0; i < np; i++) out[i] = prompt[i];
    float *logit = step(m, prompt, np, 0, NULL);
    int len = np;
    Cfg *c = &m->c;
    for (int s = 0; s < n_new; s++) {
        int best = 0; float bv = logit[0];
        for (int i = 1; i < c->vocab; i++) if (logit[i] > bv) { bv = logit[i]; best = i; }
        free(logit);
        out[len++] = best;
        if (s == n_new - 1) break;
        int one = best;
        logit = step(m, &one, 1, len - 1, NULL);
    }
    *n_out = len;
}

/* ---------- interactive prompt: greedy, streaming, stop on eos ---------- */
static void generate_stream(Model *m, Tok *T, const char *prompt, int n_new) {
    Cfg *c = &m->c;
    int cap = (int)strlen(prompt) + 16;
    int *ids = malloc((size_t)cap * sizeof(int));
    int np = tok_encode(T, prompt, (int)strlen(prompt), ids, cap);
    if (np <= 0) { fprintf(stderr, "empty prompt after tokenization\n"); free(ids); return; }
    kv_alloc(m, np + n_new + 8);
    printf("[%d prompt tokens] %s", np, prompt);
    fflush(stdout);
    double t0 = now_s(), t1 = 0;
    float *logit = step(m, ids, np, 0, NULL);
    int len = np;
    char buf[512];
    for (int s = 0; s < n_new; s++) {
        int best = 0; float bv = logit[0];
        for (int i = 1; i < c->vocab; i++) if (logit[i] > bv) { bv = logit[i]; best = i; }
        free(logit);
        if (s == 0) t1 = now_s();
        if (is_eos(c, best)) { printf("\n[eos after %d tokens]", s); break; }
        int nb = tok_decode(T, &best, 1, buf, sizeof(buf)-1);
        buf[nb] = 0; fputs(buf, stdout); fflush(stdout);
        len++;
        if (s == n_new - 1) break;
        int one = best;
        logit = step(m, &one, 1, len - 1, NULL);
    }
    double dt = now_s() - t1;
    int gen = len - np;
#ifdef LAGUNA_METAL
    lg_metal_prof_dump();
#endif
    printf("\n[prefill %.1fs | %d tokens in %.1fs = %.2f tok/s | RSS %.1f GB]\n",
           t1 - t0, gen, dt, gen > 1 ? (gen-1)/dt : 0.0, rss_gb());
    double tot = m->hits + m->miss;
    printf("[phases] fill %.1fs | expert-mm %.1fs | shared %.1fs | attn %.1fs | expert cache hit %.1f%%\n",
           m->t_fill, m->t_expert, m->t_shared, m->t_attn, tot ? 100.0*m->hits/tot : 0.0);
    free(ids);
}

/* ---------- serve mode: openai_server.py engine protocol ----------
 * Byte-identical to colibri.c's and inkling.c's protocol so the shared gateway
 * drives these engines unchanged:
 *   stdin:  SUBMIT <id> <slot> <len> <max_tokens> <temp> <top_p>\n<payload>\n
 *           CANCEL <id>\n
 *   stdout: READY sentinel, then DATA <id> <size>\n<bytes>\n frames and a final
 *           DONE <id> STAT <tok> <tps> <hit%> <rss> <prompt_tok> <len_limited>\n
 * One request at a time; the KV slot argument is accepted and ignored, which is
 * why openai_server pins these arches to kv_slots == 1. */
static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static double rng_next(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (double)(g_rng >> 11) / 9007199254740992.0;
}

typedef struct { float p; int i; } PI;
static int pi_desc(const void *a, const void *b) {
    float d = ((const PI*)b)->p - ((const PI*)a)->p;
    return d > 0 ? 1 : d < 0 ? -1 : 0;
}

/* temperature + top-p nucleus sampling; temp <= 0 = greedy (the oracle path) */
static int sample_logits(const float *logit, int n, float temp, float top_p) {
    int best = 0;
    for (int i = 1; i < n; i++) if (logit[i] > logit[best]) best = i;
    if (temp <= 0.f) return best;
    PI *c = malloc((size_t)n * sizeof(PI));
    double sum = 0;
    for (int i = 0; i < n; i++) {
        c[i].p = expf((logit[i] - logit[best]) / temp);
        c[i].i = i; sum += c[i].p;
    }
    qsort(c, n, sizeof(PI), pi_desc);
    double cut = (top_p > 0.f && top_p < 1.f) ? top_p * sum : sum;
    double acc = 0; int k = 0;
    while (k < n && acc < cut) acc += c[k++].p;
    double r = rng_next() * acc, run = 0;
    int pick = c[0].i;
    for (int i = 0; i < k; i++) { run += c[i].p; if (run >= r) { pick = c[i].i; break; } }
    free(c);
    return pick;
}

static void apply_rep_penalty(float *logit, int n, const int *hist, int nhist, float pen) {
    if (pen <= 1.f) return;
    for (int i = 0; i < nhist; i++) {
        int t = hist[i];
        if (t < 0 || t >= n) continue;
        logit[t] = logit[t] > 0 ? logit[t] / pen : logit[t] * pen;
    }
}

typedef struct { char id[64]; int max_tok; float temp, top_p; char *payload; int plen; } SReq;
#define LG_SRV_QMAX 16
static SReq g_q[LG_SRV_QMAX]; static int g_qn = 0;

/* read one control line (+ payload for SUBMIT). cur_id: request in flight;
 * returns 1 if that request was cancelled, 0 otherwise, -1 on stdin EOF. */
static int serve_read_cmd(const char *cur_id) {
    char ln[512];
    if (!fgets(ln, sizeof(ln), stdin)) return -1;
    char cmd[16], id[64];
    if (sscanf(ln, "%15s %63s", cmd, id) < 2) return 0;
    if (!strcmp(cmd, "CANCEL")) return cur_id && !strcmp(id, cur_id);
    if (!strcmp(cmd, "SUBMIT")) {
        int slot, plen, max_tok; float temp, top_p;
        int nf = sscanf(ln, "%*s %*s %d %d %d %f %f", &slot, &plen, &max_tok, &temp, &top_p);
        /* Validate max_tok as well as plen: kv_alloc is sized on
         * np + max_tok + 8, so a negative value makes the buffer shorter than
         * the prompt and prefill writes past the end of the KV cache. The
         * official gateway always sends a positive integer, but the SERVE
         * protocol is public and anything bridging it reaches this directly.
         * Same reasoning, and same check, as inkling.c's serve_read_cmd. */
        if (nf < 5 || plen < 0 || plen > (1<<22) || max_tok < 1 || max_tok > (1<<20)) {
            printf("ERROR %s bad submit header\n", id); fflush(stdout); return 0; }
        (void)slot;
        char *pl = malloc((size_t)plen + 1);
        if (fread(pl, 1, (size_t)plen, stdin) != (size_t)plen) { free(pl); return -1; }
        pl[plen] = 0;
        int nl = fgetc(stdin); (void)nl;
        if (g_qn < LG_SRV_QMAX) {
            SReq *q = &g_q[g_qn++];
            snprintf(q->id, sizeof(q->id), "%s", id);
            q->max_tok = max_tok; q->temp = temp; q->top_p = top_p;
            q->payload = pl; q->plen = plen;
        } else { printf("ERROR %s queue full\n", id); fflush(stdout); free(pl); }
    }
    return 0;
}

/* reject a prompt that would overrun the served KV bound (CTX_MAX, default 8192) */
static const char *prompt_reject(int np, int want) {
    const char *cm = getenv("CTX_MAX");
    int ctx_max = cm ? atoi(cm) : 8192;
    if (np + want > ctx_max) return "context exceeds CTX_MAX";
    return NULL;
}

static void serve_one(Model *m, Tok *T, SReq *q) {
    Cfg *c = &m->c;
    int cap = q->plen + 16;
    int *ids = malloc((size_t)cap * sizeof(int));
    int np = tok_encode(T, q->payload, q->plen, ids, cap);
    if (np <= 0) { printf("ERROR %s empty prompt\n", q->id); fflush(stdout); free(ids); return; }
    const char *bad = prompt_reject(np, q->max_tok);
    if (bad) { printf("ERROR %s %s\n", q->id, bad); fflush(stdout); free(ids); return; }
    kv_alloc(m, np + q->max_tok + 8);
    m->kv_len = 0;
    double t0 = now_s();
    uint64_t h0 = m->hits, m0 = m->miss;
    double f0 = m->t_fill, e0 = m->t_expert, s0 = m->t_shared, a0 = m->t_attn;
    float *logit = step(m, ids, np, 0, NULL);
    int len = np, gen = 0, limited = 1, cancelled = 0;
    char buf[512];
    float rep = getenv("REP_PEN") ? (float)atof(getenv("REP_PEN")) : 1.1f;
    int hist[128], nhist = 0;
    for (int i = (np > 128 ? np - 128 : 0); i < np; i++) hist[nhist++] = ids[i];
    for (int s = 0; s < q->max_tok && !cancelled; s++) {
        apply_rep_penalty(logit, c->vocab, hist, nhist, rep);
        int tk = sample_logits(logit, c->vocab, q->temp, q->top_p);
        free(logit); logit = NULL;
        if (is_eos(c, tk)) { limited = 0; break; }
        if (nhist < 128) hist[nhist++] = tk;
        else { memmove(hist, hist+1, 127*sizeof(int)); hist[127] = tk; }
        int nb = tok_decode(T, &tk, 1, buf, sizeof(buf)-1);
        printf("DATA %s %d\n", q->id, nb);
        fwrite(buf, 1, (size_t)nb, stdout);
        fputc('\n', stdout); fflush(stdout);
        gen++; len++;
        while (coli_stdin_readable()) {
            int r = serve_read_cmd(q->id);
            if (r < 0) { free(ids); free(logit); return; }
            if (r > 0) { cancelled = 1; limited = 0; }
        }
        if (cancelled || s == q->max_tok - 1) break;
        logit = step(m, &tk, 1, len - 1, NULL);
    }
    free(logit);
    double dt = now_s() - t0;
    double tot = (double)(m->hits - h0 + m->miss - m0);
    printf("DONE %s STAT %d %.3f %.1f %.2f %d %d\n", q->id, gen,
           dt > 0 ? gen/dt : 0.0, tot ? 100.0*(m->hits-h0)/tot : 0.0, rss_gb(), np, limited);
    printf("PROF %.3f %d %d %.3f %.3f %.3f %.3f %.3f %d\n", dt, np, gen,
           m->t_fill - f0, m->t_shared - s0, m->t_expert - e0, m->t_attn - a0, 0.0, gen + 1);
    fflush(stdout);
    free(ids);
}

static void serve_loop(Model *m, Tok *T) {
    coli_serve_binary_mode();
    setvbuf(stdin, NULL, _IONBF, 0);
    const char *sd = getenv("SEED");
    if (sd) g_rng ^= (uint64_t)strtoull(sd, NULL, 10);
    else g_rng ^= (uint64_t)time(NULL) * 2654435761u;
    fputs("\x01\x01READY\x01\x01\n", stdout);
    printf("STAT 0 0.0 0.0 %.2f 0 0\n", rss_gb());
    fflush(stdout);
    for (;;) {
        while (!g_qn) if (serve_read_cmd(NULL) < 0) return;
        SReq q = g_q[0];
        memmove(g_q, g_q+1, (size_t)(--g_qn) * sizeof(SReq));
        serve_one(m, T, &q);
        free(q.payload);
    }
}

/* ---------- ref_laguna.json parity harness ---------- */
static int *read_int_array(jval *o, const char *key, int *n_out) {
    jval *a = json_get(o, key);
    if (!a || a->t != J_ARR) { *n_out = 0; return NULL; }
    int *r = malloc((size_t)a->len * sizeof(int));
    for (int i = 0; i < a->len; i++) r[i] = (int)a->kids[i]->num;
    *n_out = a->len; return r;
}

static void print_cfg(Model *m) {
    Cfg *c = &m->c;
    int nfull = 0, nsp = 0;
    for (int i = 0; i < c->n_layers; i++) { nfull += !c->slide[i]; nsp += c->sparse[i]; }
    printf("cfg: D=%d L=%d(%d full/%d sliding, %d MoE) V=%d kv=%d hd=%d heads=%d/%d win=%d\n",
           c->hidden, c->n_layers, nfull, c->n_layers-nfull, nsp, c->vocab,
           c->n_kv, c->head_dim, c->heads[0], c->heads[c->n_layers-1], c->window);
    printf("     E=%d topk=%d moe_inter=%d shared=%d dense=%d scale=%.2f\n",
           c->n_experts, c->topk, c->moe_inter, c->shared_inter, c->dense_inter, c->routed_scale);
    for (int t = 0; t < 2; t++) {
        RopeCfg *r = &c->rope[t];
        printf("     rope[%s]: %s theta=%.0f rot_dim=%d factor=%.1f attn_factor=%.6f beta=%.0f/%.0f orig_max=%d\n",
               t == LG_FULL ? "full   " : "sliding", r->yarn ? "yarn   " : "default",
               r->theta, r->rot_dim, r->factor, r->attn_factor, r->beta_fast, r->beta_slow, r->orig_max);
    }
    if (m->oq.on)
        printf("     oQ: default %d-bit gs%d, %d per-tensor overrides%s\n",
               m->oq.bits, m->oq.gs, m->oq.n,
               m->experts == EXP_OQ ? ", experts packed (switch_mlp)"
                                    : " (expert layout probed at load)");
}

int main(int argc, char **argv) {
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "set SNAP=<snapshot directory>\n"); return 1; }
    const char *prompt = NULL, *refpath = LAGUNA_REF_DEFAULT;
    int cap = -1, bits = 0, n_new = 256, npos = 0, cfg_only = 0, chat = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-p") && i+1 < argc) prompt = argv[++i];
        /* -f reads the prompt from a file: a long-context prompt is tens of KB,
         * which is awkward on argv and hits ARG_MAX past a few hundred KB. */
        else if (!strcmp(argv[i], "-f") && i+1 < argc) {
            const char *pf = argv[++i];
            FILE *f = fopen(pf, "rb");
            if (!f) { perror(pf); return 1; }
            fseek(f, 0, SEEK_END); long fn = ftell(f); fseek(f, 0, SEEK_SET);
            char *pb = malloc((size_t)fn + 1);
            if (!pb) { fprintf(stderr, "OOM prompt file\n"); return 1; }
            if (fread(pb, 1, (size_t)fn, f) != (size_t)fn) { perror(pf); return 1; }
            pb[fn] = 0;
            while (fn > 0 && (pb[fn-1] == '\n' || pb[fn-1] == '\r')) pb[--fn] = 0;
            fclose(f);
            prompt = pb;
        }
        else if (!strcmp(argv[i], "-n") && i+1 < argc) n_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chat")) chat = 1;
        else if (!strcmp(argv[i], "--config")) cfg_only = 1;
        else if (npos == 0) { cap = atoi(argv[i]); npos++; }
        else if (npos == 1) { bits = atoi(argv[i]); npos++; }
        else refpath = argv[i];
    }
    /* --chat: wrap the prompt in Laguna's own template (the text subset of
     * tokenizer_config.json's chat_template, same rendering openai_server.py
     * does for the served path): leading EOS, optional <system> block, the turn
     * as <user>…</user>, then <assistant> plus the closing </think> that a
     * thinking-disabled turn emits. Without the template an instruct model gets
     * out-of-distribution text. THINK=1 opens a <think> block instead. */
    char *chat_buf = NULL;
    if (chat && prompt) {
        const char *think = getenv("THINK");
        int thinking = think && *think == '1';
        size_t need = strlen(prompt) + 320;
        chat_buf = malloc(need);
        if (!chat_buf) { fprintf(stderr, "OOM chat template\n"); return 1; }
        snprintf(chat_buf, need,
                 "\u3008|EOS|\u3009<user>%s</user>\n<assistant>%s",
                 prompt, thinking ? "<think>" : "</think>");
        prompt = chat_buf;
    }
    if (cap < 0) cap = prompt ? 0 : 16;
    if (bits && bits != 8) {
        fprintf(stderr,
            "BITS must be 0 (f32 experts) or 8 (runtime int8).\n"
            "  Sub-byte values are rejected on purpose: the runtime quantizer stores one\n"
            "  code per byte, so BITS=%d would use the SAME memory as BITS=8 and be less\n"
            "  accurate. For real sub-byte weights use an oQ checkpoint (docs/oq-format.md),\n"
            "  which is bit-packed and picks a width per tensor.\n", bits);
        return 1;
    }

    /* SERVE=1: the openai_server.py gateway drives the engine over stdin/stdout
     * (READY handshake, SUBMIT/CANCEL, DATA/DONE frames) — the same protocol
     * colibri.c and inkling.c speak, so `coli serve` / `coli chat` work. */
    if (getenv("SERVE") && getenv("SERVE")[0] == '1') {
        Model m; model_init(&m, snap, cap, bits);
        char tkp[2048]; snprintf(tkp, sizeof(tkp), "%s/tokenizer.json", snap);
        Tok T; tok_load(&T, tkp);
        serve_loop(&m, &T);
        return 0;
    }

    /* --config: parse and print the checkpoint geometry, load no weights. The
     * first thing to run against a new checkpoint, and the cheapest way to see
     * whether a config drifted from what this engine expects. */
    if (cfg_only) {
        Model m; memset(&m, 0, sizeof(m));
        load_cfg(&m.c, snap, &m.oq);
        printf("== " LAGUNA_NAME " C engine, config only ==\n");
        print_cfg(&m);
        return 0;
    }

    if (prompt) {
        Model m; model_init(&m, snap, cap, bits);
        printf("== " LAGUNA_NAME " C engine, %d layers, experts @ %s, cache %d/layer ==\n",
               m.c.n_layers, bits ? "int" : "f32", m.cache[0].cap);
        print_cfg(&m);
        printf("resident weights loaded in %.1fs | RSS %.2f GB\n", m.dense_load_s, rss_gb());
        char tkp[2048]; snprintf(tkp, sizeof(tkp), "%s/tokenizer.json", snap);
        Tok T; tok_load(&T, tkp);
        generate_stream(&m, &T, prompt, n_new);
        return 0;
    }

    FILE *f = fopen(refpath, "rb"); if (!f) { perror(refpath); return 1; }
    fseek(f,0,SEEK_END); long n = ftell(f); fseek(f,0,SEEK_SET);
    char *buf = malloc((size_t)n+1);
    if (fread(buf,1,(size_t)n,f) != (size_t)n) { fprintf(stderr,"%s: short read\n",refpath); return 1; }
    buf[n] = 0; fclose(f);
    char *arena = NULL; jval *ref = json_parse(buf, &arena);
    int np, nfull, ntf;
    int *pids  = read_int_array(ref,"prompt_ids",&np);
    int *full  = read_int_array(ref,"full_ids",&nfull);
    int *tfref = read_int_array(ref,"tf_pred",&ntf);
    if (!pids || !full) { fprintf(stderr,"%s: needs prompt_ids and full_ids\n",refpath); return 1; }
    int ngen = nfull - np;

    Model m; model_init(&m, snap, cap, bits);
    printf("== " LAGUNA_NAME " C engine, cache %d experts/layer, experts @ %s ==\n",
           m.cache[0].cap, bits ? "int (runtime quant)" : "f32");
    print_cfg(&m);
    printf("resident weights loaded in %.1fs | RSS %.2f GB\n", m.dense_load_s, rss_gb());
    kv_alloc(&m, nfull + 8);

    int tf_ok = 1;
    if (tfref && ntf == nfull) {
        int *tf = malloc((size_t)nfull * sizeof(int));
        free(step(&m, full, nfull, 0, tf));
        int ok = 0; for (int i = 0; i < nfull; i++) ok += (tf[i] == tfref[i]);
        printf("teacher-forced argmax: %d/%d match\n", ok, nfull);
        tf_ok = (ok == nfull);
        free(tf);
        kv_alloc(&m, nfull + 8); m.kv_len = 0;
    }

    int *out = malloc((size_t)nfull * sizeof(int));
    int nout = 0;
    double t = now_s();
    generate(&m, pids, np, ngen, out, &nout);
    double dt = now_s() - t;
    int match = 0;
    printf("Reference: "); for (int i = np; i < nfull; i++) printf("%d ", full[i]);
    printf("\nC engine : "); for (int i = np; i < nfull; i++) { printf("%d ", out[i]); if (out[i] == full[i]) match++; }
    printf("\nMatching tokens: %d/%d\n", match, ngen);
    double tot = m.hits + m.miss;
    printf("PEAK RSS: %.2f GB | expert cache hit %.1f%% | %.2f tok/s\n",
           rss_gb(), tot ? 100.0*m.hits/tot : 0.0, ngen/dt);
    free(buf); free(arena);
    return (match == ngen && tf_ok) ? 0 : 1;
}

#endif /* LAGUNA_COMMON_H */
