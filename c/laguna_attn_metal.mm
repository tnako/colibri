/* ---- GPU flash attention for full-attention prefill (LAGUNA-FORK) -----------
 *
 * WHY: at 30k context the attention phase was 839.8 s of a 1110 s prefill (76%),
 * running at ~145 GFLOP/s. The full-attention layers are 90.7% of that work
 * (110 of 122 TFLOP) and are the only part that grows as O(S^2); the sliding
 * layers are O(S*512) and stay on the CPU.
 *
 * DESIGN: a persistent per-layer f16 K/V cache lives on the GPU and is appended
 * once per chunk, so keys are never re-uploaded. One dispatch then does the whole
 * operation -- scores, online softmax, and the V accumulation -- without ever
 * materializing the S x keys score matrix in memory. Downloading that matrix to
 * softmax on the CPU would be 1.4 GB per chunk per layer, which is why the
 * softmax has to be inside the kernel.
 *
 * The kernel is deliberately simple: one thread per (query, head), stepping over
 * keys. All threads in a group read the same K row at the same time, so the read
 * is broadcast and cache-friendly without explicit threadgroup staging. Accum
 * lives in threadgroup memory because 128 floats per thread would spill.
 */
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "laguna_metal.h"

extern id<MTLDevice>       lg_metal_device(void);
extern id<MTLCommandQueue> lg_metal_queue(void);

/* Mirrors LG_ATTN_CHUNK in laguna_common.h (max tokens scored in one prefill call).
 * The budget loop in the engine uses its own LG_ATTN_CHUNK for the same number;
 * byte-exact agreement is not required, keeping it within ~1 chunk is. */
#define LG_ATTN_CHUNK 8192

static const char *ATTN_SRC = R"(
#include <metal_stdlib>
using namespace metal;

/* Row-wise causal softmax over a [S, nkey] score matrix, in place.
 * One threadgroup per row: max, then exp-and-sum, then normalize. Keeping this
 * on the GPU is the whole point -- downloading the score matrix to softmax on
 * the CPU would be 256 MiB per head at 256k context. */
/* `k0` is the absolute position of score column 0. It is 0 for a full-attention
 * layer (columns are absolute positions) and the band origin for a sliding layer,
 * where only `window + chunk - 1` columns are materialized. */
kernel void softmax_causal(device float*       Sc   [[buffer(0)]],
                           constant int&       nkey [[buffer(1)]],
                           constant int&       pos0 [[buffer(2)]],
                           constant int&       win  [[buffer(3)]],
                           constant int&       k0   [[buffer(4)]],
                           uint  row  [[threadgroup_position_in_grid]],
                           uint  lane [[thread_position_in_threadgroup]],
                           uint  W    [[threads_per_threadgroup]]) {
    device float* r = Sc + (long)row * nkey;
    int qpos = pos0 + int(row);
    int lo = 0;
    if (win > 0) { lo = qpos - win + 1; if (lo < 0) lo = 0; }
    lo -= k0; qpos -= k0;             /* into band-local column space */

    threadgroup float red[32];
    float m = -INFINITY;
    for (int t = int(lane); t <= qpos; t += int(W))
        if (t >= lo) { float v = r[t]; if (v > m) m = v; }
    red[lane] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { float g = -INFINITY;
        for (uint i = 0; i < W; i++) if (red[i] > g) g = red[i];
        red[0] = g; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = red[0];

    float s = 0.0f;
    for (int t = int(lane); t < nkey; t += int(W)) {
        if (t > qpos || t < lo) { r[t] = 0.0f; continue; }
        float e = exp(r[t] - m);
        r[t] = e; s += e;
    }
    red[lane] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { float g = 0.0f;
        for (uint i = 0; i < W; i++) g += red[i];
        red[0] = g; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = red[0] > 0.0f ? 1.0f / red[0] : 0.0f;
    for (int t = int(lane); t < nkey; t += int(W)) r[t] *= inv;
}

/* Dequantize a [nkey, hd] band of one kv head from the int8 cache into f32.
 * w = code * scale, one scale per row (kv_i8.h). This is the only place the GPU
 * touches the cache, and it reads it in place -- no upload. */
kernel void deq_kv(device const char*   C   [[buffer(0)]],
                   device const float*  SC  [[buffer(1)]],
                   device float*        O   [[buffer(2)]],
                   constant int&        k0  [[buffer(3)]],
                   constant int&        n   [[buffer(4)]],
                   constant int&        hd  [[buffer(5)]],
                   constant int&        base[[buffer(6)]],
                   uint2 gid [[thread_position_in_grid]]) {
    int r = int(gid.y), d = int(gid.x);
    if (r >= n || d >= hd) return;
    long src = (long)(base + k0 + r);
    O[(long)r*hd + d] = float(C[src*hd + d]) * SC[src];
}

/* f32 -> f16 copy of one query head's rows, into a [S, hd] contiguous tile. */
kernel void gather_q(device const float* Q  [[buffer(0)]],
                     device float*       QT [[buffer(1)]],
                     constant int&       qdim [[buffer(2)]],
                     constant int&       off  [[buffer(3)]],
                     constant int&       hd   [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    int s = int(gid.y), d = int(gid.x);
    if (d >= hd) return;
    QT[(long)s*hd + d] = Q[(long)s*qdim + off + d];
}

/* scatter a [S, hd] f16 result into the strided ctx output, applying the gate. */
kernel void scatter_o(device const float* OT [[buffer(0)]],
                      device float*       O  [[buffer(1)]],
                      device const float* GT [[buffer(2)]],
                      constant int&       qdim [[buffer(3)]],
                      constant int&       off  [[buffer(4)]],
                      constant int&       hd   [[buffer(5)]],
                      constant int&       H    [[buffer(6)]],
                      constant int&       hq   [[buffer(7)]],
                      uint2 gid [[thread_position_in_grid]]) {
    int s = int(gid.y), d = int(gid.x);
    if (d >= hd) return;
    float g = GT[(long)s*H + hq];
    /* softplus, matching the CPU path */
    float gate = g > 20.0f ? g : log(1.0f + exp(g));
    O[(long)s*qdim + off + d] = OT[(long)s*hd + d] * gate;
}

/* --------------------------------------------------------------------------
 * PHASE 1 (256k-rework): tiled online-softmax kernels for FULL layers.
 *
 * The banded path above sizes one score tile as S x nkey -- 8.6 GB at the
 * last 256k chunk -- so full layers stream keys in tiles of `kt` columns
 * instead. Each tile contributes through the flash-attention recurrence
 * (online softmax), so no full score matrix ever exists:
 *
 *   scores[tile] = Q @ K_tile^T            (MPS, alpha = scale)
 *   online_chunk: cmax over the tile, rescale running AC/MD, write e = exp,
 *                 fold the tile's exp-sum into the running denominator
 *   AC          += e @ V_tile               (MPS, beta = 1)
 *   fin_scatter: out = AC * softplus(gate) / den
 *
 * Query heads of one kv head (the "group") are stacked into [G*S, hd] rows so
 * the K/V tile, which all G heads share, is fetched and staged once per tile.
 * AC/MD accumulate across tiles for the whole group before the final divide.
 */

/* Stacked gather: rows [G*S, hd] = q[s*qdim + (hq0+g)*hd + d], g = row/S. */
kernel void gather_g(device const float* Q  [[buffer(0)]],
                     device float*       QT [[buffer(1)]],
                     constant int&       qdim [[buffer(2)]],
                     constant int&       hd   [[buffer(3)]],
                     constant int&       S    [[buffer(4)]],
                     constant int&       hq0  [[buffer(5)]],
                     uint2 gid [[thread_position_in_grid]]) {
    int r = int(gid.y), d = int(gid.x);
    if (d >= hd) return;
    int g = r / S, s = r % S;
    int hq = hq0 + g;
    QT[(long)r*hd + d] = Q[(long)s*qdim + (long)hq*hd + d];
}

/* One AC/MD row: AC[row*hd..] = 0, MD[2*row] = -INF, MD[2*row+1] = 0.
 * Run once per lg_metal_attn before the first group's tiles. */
kernel void init_md(device float* AC [[buffer(0)]],
                    device float* MD [[buffer(1)]],
                    constant int& rows [[buffer(2)]],
                    constant int& hd   [[buffer(3)]],
                    uint2 gid [[thread_position_in_grid]]) {
    int r = int(gid.y), d = int(gid.x);
    if (r >= rows) return;
    MD[(long)r*2 + 0] = -INFINITY;
    MD[(long)r*2 + 1] = 0.0f;
    if (d < hd) AC[(long)r*hd + d] = 0.0f;
}

/* Online-softmax update for one score tile [rows, ktile] (in place -> e).
 * One threadgroup per row; `kabs` is the absolute column of score column 0.
 * `kt2` is the actual number of valid columns in this tile (== ktile except the
 * last, partial tile). A row's valid tile columns are [0, qc] where
 * qc = qpos - kabs (qpos = pos0 + row%S, causal); all other columns become
 * e = 0 so PV ignores them. Rows with no valid column in this tile (qc < 0)
 * have their whole row zeroed too -- otherwise the following PV GEMM would
 * accumulate raw, non-softmax scores for those rows. */
kernel void online_chunk(device float* Sc  [[buffer(0)]],
                         device float* AC  [[buffer(1)]],
                         device float* MD  [[buffer(2)]],
                         constant int& rows [[buffer(3)]],
                         constant int& kt   [[buffer(4)]],
                         constant int& kabs [[buffer(5)]],
                         constant int& pos0 [[buffer(6)]],
                         constant int& S    [[buffer(7)]],
                         constant int& hd   [[buffer(8)]],
                         constant int& kt2  [[buffer(9)]],
                         uint  r    [[threadgroup_position_in_grid]],
                         uint  lane [[thread_position_in_threadgroup]],
                         uint  W    [[threads_per_threadgroup]]) {
    if (r >= (uint)rows) return;
    device float* sc = Sc + (long)r * kt;
    int s = int(r) % S;
    int qc = (pos0 + s) - kabs;          /* last valid tile column (inclusive) */
    if (qc < 0) {                         /* no valid key in this tile          */
        for (int c = int(lane); c < kt; c += int(W)) sc[c] = 0.0f;
        return;
    }
    if (qc >= kt2) qc = kt2 - 1;

    threadgroup float red[1024];
    float cmax = -INFINITY;
    for (int c = int(lane); c <= qc; c += int(W)) if (sc[c] > cmax) cmax = sc[c];
    red[lane] = cmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { float g = -INFINITY;
        for (uint i = 0; i < W; i++) if (red[i] > g) g = red[i];
        red[0] = g; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    cmax = red[0];
    if (!(cmax > -1.0e35f)) return;       /* nothing valid in this tile        */

    float m = MD[(long)r*2 + 0], d = MD[(long)r*2 + 1];
    float nm = m > cmax ? m : cmax;
    if (nm != m) {
        float rs = (m == -INFINITY) ? 0.0f : exp(m - nm);
        if (rs == 0.0f) {
            for (int j = int(lane); j < hd; j += int(W)) AC[(long)r*hd + j] = 0.0f;
            d = 0.0f;
        } else {
            for (int j = int(lane); j < hd; j += int(W)) AC[(long)r*hd + j] *= rs;
            d *= rs;
        }
        m = nm;
    }
    float sacc = 0.0f;
    for (int c = int(lane); c < kt; c += int(W)) {
        if (c <= qc) { float e = exp(sc[c] - m); sc[c] = e; sacc += e; }
        else         sc[c] = 0.0f;
    }
    red[lane] = sacc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { float g = 0.0f;
        for (uint i = 0; i < W; i++) g += red[i];
        red[0] = g; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    MD[(long)r*2 + 0] = m;
    MD[(long)r*2 + 1] = d + red[0];
}

/* Finalize one group's accumulated rows: out[s*qdim + hq*hd + d] =
 * AC[row*hd+d] * softplus(gate) / den, matching the CPU path's gate+divide. */
kernel void fin_scatter(device const float* AC [[buffer(0)]],
                        device const float* MD [[buffer(1)]],
                        device const float* GT [[buffer(2)]],
                        device float*       O  [[buffer(3)]],
                        constant int& qdim [[buffer(4)]],
                        constant int& hd   [[buffer(5)]],
                        constant int& S    [[buffer(6)]],
                        constant int& H    [[buffer(7)]],
                        constant int& hq0  [[buffer(8)]],
                        constant int& rbase [[buffer(9)]],
                        constant int& rows [[buffer(10)]],
                        uint2 gid [[thread_position_in_grid]]) {
    int r = int(gid.y), d = int(gid.x);
    if (d >= hd) return;
    long row = r;              /* AC/MD are bound at rbase already */
    int s = r % S, g = r / S;
    int hq = hq0 + g;
    float den = MD[(long)row*2 + 1];
    float v = AC[(long)row*hd + d];
    if (den > 0.0f) {
        float gg = GT[(long)s*H + hq];
        float gate = gg > 20.0f ? gg : log(1.0f + exp(gg));
        v = v * gate / den;
    } else v = 0.0f;
    O[(long)s*qdim + (long)hq*hd + d] = v;
}
)";

typedef struct {
    void *K, *V;          /* int8 codes  [KV][ctxcap][hd], bound zero-copy */
    void *Ks, *Vs;        /* f32 scales  [KV][ctxcap],     bound zero-copy */
    int kv, ctxcap, hd;
    int ring;             /* kept for the sliding path; 0 = linear         */
    int hheads;           /* query heads, learned at first lg_metal_attn   */
} AttnLayer;

/* A sliding layer only ever reads the last `window` positions, so its GPU cache
 * needs window+chunk rows, not the whole context: at 250k that is the difference
 * between 45.8 GiB for 48 layers and 11.4 GiB for the 12 full ones.
 *
 * But the banded GEMM addresses its band as ONE contiguous row range, and a plain
 * ring would split it at the wrap. The fix is the standard double-mapped ring:
 * physical capacity is 2*ring and every row is written at both `r` and `r+ring`,
 * so any window-length span starting anywhere in [0,ring) is contiguous. Costs 2x
 * a ring (1536 rows) instead of 250000. */

/* Scores and PV are plain GEMMs, so MPSMatrixMultiplication does them: measured
 * 15572 GFLOP/s on this device against ~81 GFLOP/s for a hand-written
 * one-thread-per-query kernel (the first version of this file). Only the softmax
 * and the gather/scatter need custom shaders. */
static void *g_pipe_sm = NULL, *g_pipe_gq = NULL, *g_pipe_so = NULL, *g_pipe_dq = NULL;
static void *g_pipe_gg = NULL, *g_pipe_init = NULL, *g_pipe_oc = NULL, *g_pipe_fin = NULL;
static void *g_qt = NULL, *g_sc = NULL, *g_ot = NULL;
static size_t g_qtcap = 0, g_sccap = 0, g_otcap = 0;
static void *g_kf = NULL;  /* f32 K+V staging for the band actually scored */
static size_t g_kfcap = 0;
static void *g_md = NULL;  /* online-softmax running max/den [H*S][2] */
static size_t g_mdcap = 0;

static id<MTLComputePipelineState> mk_pipe(id<MTLLibrary> lib, const char *name) {
    NSError *e = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) { fprintf(stderr, "[metal] no fn %s\n", name); return nil; }
    id<MTLComputePipelineState> ps =
        [lg_metal_device() newComputePipelineStateWithFunction:fn error:&e];
    if (!ps) fprintf(stderr, "[metal] pipeline %s: %s\n", name, [[e description] UTF8String]);
    return ps;
}

static void *ensure(void **slot, size_t *cap, size_t need) {
    if (*cap >= need && *slot) return *slot;
    id<MTLBuffer> nb = [lg_metal_device() newBufferWithLength:need
                                                     options:MTLResourceStorageModeShared];
    if (!nb) return NULL;
    if (*slot) CFRelease((CFTypeRef)*slot);
    *slot = (void*)CFBridgingRetain(nb);
    *cap = need;
    return *slot;
}

static AttnLayer *g_al = NULL;
static int g_al_n = 0;

/* GPU-side profiling. cb.GPUStartTime/GPUEndTime are the same timestamps
 * Instruments reports; reading them in-process avoids needing Xcode (not
 * installed here -- only CommandLineTools, so no xctrace). LAGUNA_GPU_PROF=1. */
static int    g_prof = -1;
static double g_gpu_s = 0, g_wall_s = 0;
static long   g_calls = 0;

static int prof_on(void) {
    if (g_prof < 0) { const char *e = getenv("LAGUNA_GPU_PROF"); g_prof = e ? atoi(e) : 0; }
    return g_prof;
}

void lg_metal_prof_dump(void) {
    if (!prof_on() || g_calls == 0) return;
    fprintf(stderr,
        "[gpuprof] attn: %ld dispatches, GPU busy %.2fs, wall %.2fs (%.0f%% busy)\n",
        g_calls, g_gpu_s, g_wall_s, 100.0*g_gpu_s/g_wall_s);
}

static int attn_pipeline(void) {
    if (g_pipe_sm) return 1;
    id<MTLDevice> d = lg_metal_device();
    if (!d) return 0;
    NSError *e = nil;
    id<MTLLibrary> lib = [d newLibraryWithSource:[NSString stringWithUTF8String:ATTN_SRC]
                                        options:nil error:&e];
    if (!lib) { fprintf(stderr, "[metal] attn compile: %s\n",
                        [[e description] UTF8String]); return 0; }
    id<MTLComputePipelineState> sm = mk_pipe(lib, "softmax_causal");
    id<MTLComputePipelineState> gq = mk_pipe(lib, "gather_q");
    id<MTLComputePipelineState> so = mk_pipe(lib, "scatter_o");
    if (!sm || !gq || !so) return 0;
    id<MTLComputePipelineState> dq = mk_pipe(lib, "deq_kv");
    if (!dq) return 0;
    g_pipe_dq = (void*)CFBridgingRetain(dq);
    g_pipe_sm = (void*)CFBridgingRetain(sm);
    g_pipe_gq = (void*)CFBridgingRetain(gq);
    g_pipe_so = (void*)CFBridgingRetain(so);
    id<MTLComputePipelineState> gg = mk_pipe(lib, "gather_g");
    id<MTLComputePipelineState> im = mk_pipe(lib, "init_md");
    id<MTLComputePipelineState> oc = mk_pipe(lib, "online_chunk");
    id<MTLComputePipelineState> fs = mk_pipe(lib, "fin_scatter");
    if (!gg || !im || !oc || !fs) return 0;
    g_pipe_gg  = (void*)CFBridgingRetain(gg);
    g_pipe_init = (void*)CFBridgingRetain(im);
    g_pipe_oc   = (void*)CFBridgingRetain(oc);
    g_pipe_fin  = (void*)CFBridgingRetain(fs);
    return 1;
}

/* ZERO-COPY BIND (LAGUNA-FORK).
 *
 * The engine already maintains an int8 KV cache with one f32 scale per row
 * (kv_i8.h). Previously this file kept a SECOND copy in f16 on the GPU, which at
 * 262144 context on Laguna-S was 14.24 GB -- more than half the whole budget, so
 * the allocation was declined and those layers fell back to the CPU.
 *
 * Instead of uploading tiles of a duplicate, bind the cache itself: it is
 * page-aligned (see kv_aligned) and unified memory means the GPU reads the very
 * same bytes the CPU wrote. GPU K/V allocation becomes ZERO, the per-chunk append
 * disappears, and there is one source of truth for KV so the two paths cannot
 * drift. The kernel dequantizes int8 -> f16 in the gather. */
int lg_metal_attn_bind(int layers, int layer, int kv, int ctxcap, int hd,
                       const void *kcodes, const void *vcodes,
                       const float *kscale, const float *vscale, size_t code_bytes,
                       size_t scale_bytes, int ring) {
    if (!lg_metal_device() || !attn_pipeline()) return 0;
    if (!g_al) { g_al = (AttnLayer*)calloc(layers, sizeof(AttnLayer)); g_al_n = layers; }
    if (layer < 0 || layer >= g_al_n) return 0;
    AttnLayer *L = &g_al[layer];
    if (L->K) return 1;                       /* already bound */
    id<MTLDevice> d = lg_metal_device();
    size_t cb = code_bytes  & ~(size_t)16383;
    size_t sb = scale_bytes & ~(size_t)16383;
    if (!cb || !sb) return 0;
    id<MTLBuffer> kb = [d newBufferWithBytesNoCopy:(void*)kcodes length:cb
                        options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> vb = [d newBufferWithBytesNoCopy:(void*)vcodes length:cb
                        options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> ks = [d newBufferWithBytesNoCopy:(void*)kscale length:sb
                        options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> vs = [d newBufferWithBytesNoCopy:(void*)vscale length:sb
                        options:MTLResourceStorageModeShared deallocator:nil];
    if (!kb || !vb || !ks || !vs) return 0;
    L->K  = (void*)CFBridgingRetain(kb);
    L->V  = (void*)CFBridgingRetain(vb);
    L->Ks = (void*)CFBridgingRetain(ks);
    L->Vs = (void*)CFBridgingRetain(vs);
    /* `ctxcap` here is the PHYSICAL per-head row stride (kvphys): equal to the
     * true context cap for full (linear) layers, or a small double-mapped
     * ring for sliding layers -- see kv_alloc's comment in laguna_common.h.
     * `ring` is the logical ring modulus (0 for full/linear layers, > 0 for
     * sliding layers), used below to fold an absolute band origin back into
     * the physical buffer. */
    L->kv = kv; L->ctxcap = ctxcap; L->hd = hd; L->ring = ring;
    return 1;
}

/* One (layer, chunk) attention: per query head, QK^T then softmax then PV, with
 * both GEMMs on MPS. win>0 restricts to a sliding window. */
int lg_metal_attn(int layer, float *ctx_out, const float *q, const float *gt,
                  int S, int pos0, int H, int KV, int hd, float scale, int window) {
    if (!g_al || !g_pipe_sm || layer < 0 || layer >= g_al_n) return 0;
    AttnLayer *L = &g_al[layer];
    if (!L->K) return 0;
    if (!L->hheads) L->hheads = H;
    /* BANDED (LAGUNA-FORK): a sliding layer's queries in this chunk span absolute
     * positions [pos0, pos0+S), so the only keys any of them can attend are
     * [pos0-window+1, pos0+S). That band is window+S-1 wide -- CONSTANT in
     * context -- while a full layer needs all pos0+S keys. At 262144 tokens with
     * window 512 and chunk 256 that is 341x fewer score columns, which is what
     * makes the GPU viable for sliding layers at all: the dense version computed
     * the whole matrix and masked it away (measured 32.4 -> 53.1 s at 6k).
     *
     * k0/nkey below are ABSOLUTE positions, used for the causal/window mask in
     * softmax_causal. For a full (linear) layer the cache is addressed by that
     * same absolute position, so k0 doubles as the physical offset too. For a
     * sliding (ring) layer the cache is only `L->ring` rows physically, so the
     * absolute band origin is folded into the ring with k0_phys = k0 % ring --
     * safe because the band (window+S-1 columns) is always <= L->ring by
     * construction (see kv_alloc), so it never wraps mid-read once double-mapped. */
    int k0 = 0, nkey = pos0 + S;
    if (window > 0) {
        k0 = pos0 - window + 1;
        if (k0 < 0) k0 = 0;
        nkey = pos0 + S - k0;
    }
    int band = nkey - k0;
    if (L->ring > 0) {
        if (band > L->ring) return 0;         /* safety net; should not happen */
    } else if (pos0 + S > L->ctxcap) {
        return 0;                             /* full layer: linear cap check */
    }
    int k0_phys = (L->ring > 0) ? (k0 % L->ring) : k0;
    int group = H / KV, qdim = H * hd;
    if (getenv("LG_DUMP")) fprintf(stderr, "entry: layer=%d S=%d pos0=%d H=%d KV=%d hd=%d window=%d L->ctxcap=%d L->ring=%d group=%d nkey=%d k0=%d\n",
            layer, S, pos0, H, KV, hd, window, L->ctxcap, L->ring, group, nkey, k0);

    /* ---- PHASE 1 tiled path: FULL layers only (window == 0) ---------------
     * Full layers score the whole prefix, so a banded tile would be the whole
     * S x nkey matrix (8.6 GB at the last 256k chunk). Instead stream keys in
     * tiles of `ktile` columns: per tile a score GEMM, an online-softmax update
     * (rescale the running AC/MD in device memory), and a PV GEMM with beta=1.
     * Peak GPU staging is then O(S*hd*H + S*ktile), independent of context.
     * Sliding layers below keep the existing banded path (window > 0). */
    if (window == 0) {
        id<MTLDevice> d2 = lg_metal_device();
        id<MTLCommandQueue> cq2 = lg_metal_queue();
        @autoreleasepool {
            long rows = (long)group * S;            /* stacked head-group rows */
            long ktile = ((384LL << 20) / (rows * 4));   /* ~384 MiB score tile */
            { const char *e = getenv("LG_KTILE");
              if (e) { long v = atol(e); if (v > 0) ktile = v;
                       if (getenv("LG_DUMP")) fprintf(stderr, "tiled: env[%s]=%ld\n", e, v); } }
            if (ktile < 256) ktile = 256;
            if (ktile > 65536) ktile = 65536;
            if (ktile > nkey) ktile = nkey;
            if (ktile < 1) ktile = 1;
            if (getenv("LG_DUMP")) fprintf(stderr, "tiled: ktile=%ld\n", ktile);

            if (!ensure(&g_qt, &g_qtcap, (size_t)rows * hd * 4)) { fprintf(stderr, "tiled: g_qt fail rows=%ld hd=%d dev=%s\n", rows, hd, lg_metal_device()? "yes":"nil"); return 0; }
            if (!ensure(&g_sc, &g_sccap, (size_t)rows * ktile * 4)) { fprintf(stderr, "tiled: g_sc fail\n"); return 0; }
            if (!ensure(&g_ot, &g_otcap, (size_t)H * S * hd * 4)) { fprintf(stderr, "tiled: g_ot fail\n"); return 0; }
            if (!ensure(&g_md, &g_mdcap, (size_t)H * S * 8)) { fprintf(stderr, "tiled: g_md fail\n"); return 0; }
            if (!ensure(&g_kf, &g_kfcap, (size_t)ktile * hd * 4 * 2)) { fprintf(stderr, "tiled: g_kf fail\n"); return 0; }
            static void *qs2 = NULL, *od2 = NULL, *gs2 = NULL;
            static size_t qsc2 = 0, odc2 = 0, gsc2 = 0;
            if (!ensure(&qs2, &qsc2, (size_t)S * qdim * 4)) { fprintf(stderr, "tiled: qs2 fail\n"); return 0; }
            if (!ensure(&od2, &odc2, (size_t)S * qdim * 4)) { fprintf(stderr, "tiled: od2 fail\n"); return 0; }
            if (!ensure(&gs2, &gsc2, (size_t)S * H * 4)) { fprintf(stderr, "tiled: gs2 fail\n"); return 0; }
            memcpy([(__bridge id<MTLBuffer>)qs2 contents], q,  (size_t)S*qdim*4);
            memcpy([(__bridge id<MTLBuffer>)gs2 contents], gt, (size_t)S*H*4);

            id<MTLBuffer> QS = (__bridge id<MTLBuffer>)qs2;
            id<MTLBuffer> OD = (__bridge id<MTLBuffer>)od2;
            id<MTLBuffer> GS = (__bridge id<MTLBuffer>)gs2;
            id<MTLBuffer> QT = (__bridge id<MTLBuffer>)g_qt;
            id<MTLBuffer> SC = (__bridge id<MTLBuffer>)g_sc;
            id<MTLBuffer> AC = (__bridge id<MTLBuffer>)g_ot;
            id<MTLBuffer> MD = (__bridge id<MTLBuffer>)g_md;
            id<MTLBuffer> KF = (__bridge id<MTLBuffer>)g_kf;
            long KVD = (long)ktile * hd * 4;            /* V tile byte offset */

            MPSMatrixDescriptor *dq = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:hd
                                        rowBytes:(size_t)hd*4 dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *dk = [MPSMatrixDescriptor matrixDescriptorWithRows:ktile columns:hd
                                        rowBytes:(size_t)hd*4 dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *ds = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:ktile
                                        rowBytes:(size_t)ktile*4 dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *dv = [MPSMatrixDescriptor matrixDescriptorWithRows:ktile columns:hd
                                        rowBytes:(size_t)hd*4 dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *da = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:hd
                                        rowBytes:(size_t)hd*4 dataType:MPSDataTypeFloat32];
            MPSMatrixMultiplication *qk =
                [[MPSMatrixMultiplication alloc] initWithDevice:d2 transposeLeft:NO transposeRight:YES
                    resultRows:rows resultColumns:ktile interiorColumns:hd alpha:scale beta:0.0];
            MPSMatrixMultiplication *pv =
                [[MPSMatrixMultiplication alloc] initWithDevice:d2 transposeLeft:NO transposeRight:NO
                    resultRows:rows resultColumns:hd interiorColumns:ktile alpha:1.0 beta:1.0];

            id<MTLComputePipelineState> pgg  = (__bridge id<MTLComputePipelineState>)g_pipe_gg;
            id<MTLComputePipelineState> pinit= (__bridge id<MTLComputePipelineState>)g_pipe_init;
            id<MTLComputePipelineState> poc  = (__bridge id<MTLComputePipelineState>)g_pipe_oc;
            id<MTLComputePipelineState> pfin = (__bridge id<MTLComputePipelineState>)g_pipe_fin;
            id<MTLComputePipelineState> pdeq = (__bridge id<MTLComputePipelineState>)g_pipe_dq;

            id<MTLCommandBuffer> cb = [cq2 commandBuffer];
            int ninit = H * S;
            {   /* one online-softmax state per (head, row): m=-INF, d=0, AC=0 */
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:pinit];
                [e setBuffer:AC offset:0 atIndex:0];
                [e setBuffer:MD offset:0 atIndex:1];
                [e setBytes:&ninit length:4 atIndex:2];
                [e setBytes:&hd   length:4 atIndex:3];
                [e dispatchThreads:MTLSizeMake((NSUInteger)hd, (NSUInteger)ninit, 1)
              threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                [e endEncoding];
            }
            for (int kh = KV - 1; kh >= 0; kh--) {
                int hq0 = kh * group;               /* first query head of group */
                long rbase = (long)hq0 * S;         /* AC/MD row base            */

                {   /* stack this group's query heads: [G*S, hd] */
                    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                    [e setComputePipelineState:pgg];
                    [e setBuffer:QS offset:0 atIndex:0];
                    [e setBuffer:QT offset:0 atIndex:1];
                    [e setBytes:&qdim length:4 atIndex:2];
                    [e setBytes:&hd   length:4 atIndex:3];
                    [e setBytes:&S    length:4 atIndex:4];
                    [e setBytes:&hq0  length:4 atIndex:5];
                    [e dispatchThreads:MTLSizeMake((NSUInteger)hd, (NSUInteger)rows, 1)
                  threadsPerThreadgroup:MTLSizeMake(hd < 64 ? hd : 64, 1, 1)];
                    [e endEncoding];
                }
                int kbase = kh * L->ctxcap;
                for (long t = 0; t < nkey; t += ktile) {
                    long kt = nkey - t; if (kt > ktile) kt = ktile;
                    int kabs = (int)(k0 + t);
                    /* dequantize this tile's K and V into the tile staging */
                    for (int rep = 0; rep < 2; rep++) {
                    id<MTLComputeCommandEncoder> ed = [cb computeCommandEncoder];
                    [ed setComputePipelineState:pdeq];
                    [ed setBuffer:(__bridge id<MTLBuffer>)L->K  offset:0 atIndex:0];
                    [ed setBuffer:(__bridge id<MTLBuffer>)L->Ks offset:0 atIndex:1];
                    [ed setBuffer:KF offset:0 atIndex:2];
                    [ed setBytes:&kabs length:4 atIndex:3];
                    int kth = (int)kt;
                    [ed setBytes:&kth  length:4 atIndex:4];
                    [ed setBytes:&hd   length:4 atIndex:5];
                    [ed setBytes:&kbase length:4 atIndex:6];
                    [ed dispatchThreads:MTLSizeMake((NSUInteger)hd, (NSUInteger)kt, 1)
                  threadsPerThreadgroup:MTLSizeMake(hd < 64 ? hd : 64, 1, 1)];
                    [ed setBuffer:(__bridge id<MTLBuffer>)L->V  offset:0 atIndex:0];
                    [ed setBuffer:(__bridge id<MTLBuffer>)L->Vs offset:0 atIndex:1];
                    [ed setBuffer:KF offset:(NSUInteger)KVD atIndex:2];
                    [ed dispatchThreads:MTLSizeMake((NSUInteger)hd, (NSUInteger)kt, 1)
                  threadsPerThreadgroup:MTLSizeMake(hd < 64 ? hd : 64, 1, 1)];
                    [ed endEncoding];
                    }
                    if (getenv("LG_DUMP_STEP")) {
                        [cb commit];
                        [cb waitUntilCompleted];
                        if (!getenv("LG_KF_LATE")) {
                        char pth[1024];
                        snprintf(pth, sizeof pth, "%s/LG_kf_t%ld_kh%d.bin", getenv("LG_DUMP_STEP"), t, kh);
                        FILE *f = fopen(pth, "wb"); if (f) { fwrite([KF contents], 1, (size_t)ktile*hd*4*2, f); fclose(f); }
                        }
                        cb = [cq2 commandBuffer];
                    }

                    MPSMatrix *mq = [[MPSMatrix alloc] initWithBuffer:QT descriptor:dq];
                    MPSMatrix *mk = [[MPSMatrix alloc] initWithBuffer:KF descriptor:dk];
                    MPSMatrix *ms = [[MPSMatrix alloc] initWithBuffer:SC descriptor:ds];
                    [qk encodeToCommandBuffer:cb leftMatrix:mq rightMatrix:mk resultMatrix:ms];
                    if (getenv("LG_DUMP_STEP")) {
                        [cb commit];
                        [cb waitUntilCompleted];
                        fprintf(stderr, "qk t=%ld kh=%d cb.status=%ld\n", t, kh, (long)cb.status);
                        cb = [cq2 commandBuffer];
                        char pth[1024];
                        snprintf(pth, sizeof pth, "%s/LG_kf_t%ld_kh%d.bin", getenv("LG_DUMP_STEP"), t, kh);
                        FILE *f = fopen(pth, "wb"); if (f) { fwrite([KF contents], 1, (size_t)ktile*hd*4*2, f); fclose(f); }
                        snprintf(pth, sizeof pth, "%s/LG_qk_t%ld_kh%d.bin", getenv("LG_DUMP_STEP"), t, kh);
                        f = fopen(pth, "wb"); if (f) { fwrite([SC contents], 1, (size_t)rows*ktile*4, f); fclose(f); }
                    }

                    int rrows = (int)rows;
                    id<MTLComputeCommandEncoder> eo = [cb computeCommandEncoder];
                    [eo setComputePipelineState:poc];
                    [eo setBuffer:SC offset:0 atIndex:0];
                    [eo setBuffer:AC offset:(NSUInteger)(rbase*hd*4) atIndex:1];
                    [eo setBuffer:MD offset:(NSUInteger)(rbase*8) atIndex:2];
                    int ktl = (int)ktile;
                    [eo setBytes:&rrows length:4 atIndex:3];
                    [eo setBytes:&ktl  length:4 atIndex:4];
                    [eo setBytes:&kabs length:4 atIndex:5];
                    [eo setBytes:&pos0 length:4 atIndex:6];
                    [eo setBytes:&S    length:4 atIndex:7];
                    [eo setBytes:&hd   length:4 atIndex:8];
                    int kt2 = (int)kt;
                    [eo setBytes:&kt2  length:4 atIndex:9];
                    [eo dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    [eo endEncoding];

                    MPSMatrix *me = [[MPSMatrix alloc] initWithBuffer:SC descriptor:ds];
                    MPSMatrix *mv = [[MPSMatrix alloc] initWithBuffer:KF offset:(NSUInteger)KVD descriptor:dv];
                    MPSMatrix *ma = [[MPSMatrix alloc] initWithBuffer:AC offset:(NSUInteger)(rbase*hd*4) descriptor:da];
                    [pv encodeToCommandBuffer:cb leftMatrix:me rightMatrix:mv resultMatrix:ma];
                    if (getenv("LG_DUMP_STEP")) {
                        [cb commit];
                        [cb waitUntilCompleted];
                        cb = [cq2 commandBuffer];
                        char pth[1024];
                        snprintf(pth, sizeof pth, "%s/LG_sc_t%ld_kh%d.bin", getenv("LG_DUMP_STEP"), t, kh);
                        FILE *f = fopen(pth, "wb"); if (f) { fwrite([SC contents], 1, (size_t)rows*ktile*4, f); fclose(f); }
                        snprintf(pth, sizeof pth, "%s/LG_md_t%ld_kh%d.bin", getenv("LG_DUMP_STEP"), t, kh);
                        f = fopen(pth, "wb"); if (f) { fwrite([MD contents], 1, (size_t)H*S*8, f); fclose(f); }
                    }
                }
                {   /* finalize: out = AC * softplus(gate) / den, straight to ctx */
                    int rrows = (int)rows;
                    id<MTLComputeCommandEncoder> ef = [cb computeCommandEncoder];
                    [ef setComputePipelineState:pfin];
                    [ef setBuffer:AC offset:(NSUInteger)(rbase*hd*4) atIndex:0];
                    [ef setBuffer:MD offset:(NSUInteger)(rbase*8) atIndex:1];
                    [ef setBuffer:GS offset:0 atIndex:2];
                    [ef setBuffer:OD offset:0 atIndex:3];
                    [ef setBytes:&qdim  length:4 atIndex:4];
                    [ef setBytes:&hd    length:4 atIndex:5];
                    [ef setBytes:&S     length:4 atIndex:6];
                    [ef setBytes:&H     length:4 atIndex:7];
                    [ef setBytes:&hq0   length:4 atIndex:8];
                    long rb = rbase;
                    [ef setBytes:&rb    length:8 atIndex:9];
                    [ef setBytes:&rrows length:4 atIndex:10];
                    [ef dispatchThreads:MTLSizeMake((NSUInteger)hd, (NSUInteger)rows, 1)
                  threadsPerThreadgroup:MTLSizeMake(hd < 64 ? hd : 64, 1, 1)];
                    [ef endEncoding];
                }
            }
            double w0 = prof_on() ? CFAbsoluteTimeGetCurrent() : 0;
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status == MTLCommandBufferStatusError) {
                fprintf(stderr, "tiled: cb error\n");
                return 0;
            }
            if (prof_on()) {
                g_wall_s += CFAbsoluteTimeGetCurrent() - w0;
                g_gpu_s  += cb.GPUEndTime - cb.GPUStartTime;
                g_calls++;
            }
            memcpy(ctx_out, [OD contents], (size_t)S*qdim*4);
            if (getenv("LG_DUMP")) {
                const char *dir = getenv("LG_DUMP");
                char path[1024];
                snprintf(path, sizeof path, "%s/LG_qt.bin", dir);
                FILE *f = fopen(path, "wb"); if (f) { fwrite([QT contents], 1, (size_t)rows*hd*4, f); fclose(f); }
                snprintf(path, sizeof path, "%s/LG_kf.bin", dir);
                f = fopen(path, "wb"); if (f) { fwrite([KF contents], 1, (size_t)ktile*hd*4*2, f); fclose(f); }
                snprintf(path, sizeof path, "%s/LG_sc.bin", dir);
                f = fopen(path, "wb"); if (f) { fwrite([SC contents], 1, (size_t)rows*ktile*4, f); fclose(f); }
                snprintf(path, sizeof path, "%s/LG_md.bin", dir);
                f = fopen(path, "wb"); if (f) { fwrite([MD contents], 1, (size_t)H*S*8, f); fclose(f); }
                snprintf(path, sizeof path, "%s/LG_ac.bin", dir);
                f = fopen(path, "wb"); if (f) { fwrite([AC contents], 1, (size_t)H*S*hd*4, f); fclose(f); }
                snprintf(path, sizeof path, "%s/LG_od.bin", dir);
                f = fopen(path, "wb"); if (f) { fwrite([OD contents], 1, (size_t)S*qdim*4, f); fclose(f); }
                snprintf(path, sizeof path, "%s/LG_gt.bin", dir);
                f = fopen(path, "wb"); if (f) { fwrite([GS contents], 1, (size_t)S*H*4, f); fclose(f); }
            }
        }
        return 1;
    }

    id<MTLDevice> d = lg_metal_device();
    id<MTLCommandQueue> cq = lg_metal_queue();

    @autoreleasepool {
        /* Q for one head as f16 [S,hd]; scores [S,nkey] f16; out [S,hd] f16. */
        if (!ensure(&g_qt, &g_qtcap, (size_t)S*hd*4)) return 0;
        if (!ensure(&g_sc, &g_sccap, (size_t)S*nkey*4)) return 0;
        if (!ensure(&g_ot, &g_otcap, (size_t)S*hd*4)) return 0;
        /* the strided f32 Q and the f32 ctx/gate live in shared buffers too */
        static void *qsrc = NULL, *odst = NULL, *gsrc = NULL;
        static size_t qsc = 0, odc = 0, gsc = 0;
        if (!ensure(&qsrc, &qsc, (size_t)S*qdim*4)) return 0;
        if (!ensure(&odst, &odc, (size_t)S*qdim*4)) return 0;
        if (!ensure(&gsrc, &gsc, (size_t)S*H*4))    return 0;
        /* f32 staging for the band of K and V actually scored this chunk. This is
         * the ONLY GPU-side KV memory now: O(band), not O(context).
         * Shared across layers -- layers are processed sequentially, so only one
         * layer's staging is live at a time. Previously each AttnLayer held its
         * own kf buffer, which at 256k on Laguna-S (12 full layers x ~2.1 GB
         * each) silently grew to 25 GB of unbudgeted unified memory. */
        if (!ensure(&g_kf, &g_kfcap, (size_t)KV*nkey*hd*4*2)) return 0;
        memcpy([(__bridge id<MTLBuffer>)qsrc contents], q,  (size_t)S*qdim*4);
        memcpy([(__bridge id<MTLBuffer>)gsrc contents], gt, (size_t)S*H*4);

        id<MTLBuffer> QS = (__bridge id<MTLBuffer>)qsrc;
        id<MTLBuffer> OD = (__bridge id<MTLBuffer>)odst;
        id<MTLBuffer> QT = (__bridge id<MTLBuffer>)g_qt;
        id<MTLBuffer> SC = (__bridge id<MTLBuffer>)g_sc;
        id<MTLBuffer> OT = (__bridge id<MTLBuffer>)g_ot;

        /* f32 end to end: K/V come from an int8 cache, so a second f16 rounding
         * here broke token-exactness (see the staging note above). */
        MPSMatrixDescriptor *dq = [MPSMatrixDescriptor matrixDescriptorWithRows:S columns:hd
                                    rowBytes:(size_t)hd*4 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dk = [MPSMatrixDescriptor matrixDescriptorWithRows:nkey columns:hd
                                    rowBytes:(size_t)hd*4 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *ds = [MPSMatrixDescriptor matrixDescriptorWithRows:S columns:nkey
                                    rowBytes:(size_t)nkey*4 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dv = [MPSMatrixDescriptor matrixDescriptorWithRows:nkey columns:hd
                                    rowBytes:(size_t)hd*4 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *do_ = [MPSMatrixDescriptor matrixDescriptorWithRows:S columns:hd
                                    rowBytes:(size_t)hd*4 dataType:MPSDataTypeFloat32];

        MPSMatrixMultiplication *qk =
            [[MPSMatrixMultiplication alloc] initWithDevice:d transposeLeft:NO transposeRight:YES
                resultRows:S resultColumns:nkey interiorColumns:hd alpha:scale beta:0.0];
        MPSMatrixMultiplication *pv =
            [[MPSMatrixMultiplication alloc] initWithDevice:d transposeLeft:NO transposeRight:NO
                resultRows:S resultColumns:hd interiorColumns:nkey alpha:1.0 beta:0.0];

        id<MTLComputePipelineState> psm = (__bridge id<MTLComputePipelineState>)g_pipe_sm;
        id<MTLComputePipelineState> pgq = (__bridge id<MTLComputePipelineState>)g_pipe_gq;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)g_pipe_so;
        int win = window;

        id<MTLCommandBuffer> cb = [cq commandBuffer];
        {   /* dequantize every kv head's band once: int8 cache -> f16, in place */
            id<MTLComputeCommandEncoder> ed = [cb computeCommandEncoder];
            [ed setComputePipelineState:(__bridge id<MTLComputePipelineState>)g_pipe_dq];
            size_t kband = (size_t)nkey*hd*4;
            for (int kh = 0; kh < KV; kh++) {
                int kbase = kh * L->ctxcap;
                [ed setBytes:&k0_phys length:4 atIndex:3];
                [ed setBytes:&nkey  length:4 atIndex:4];
                [ed setBytes:&hd    length:4 atIndex:5];
                [ed setBytes:&kbase length:4 atIndex:6];
                [ed setBuffer:(__bridge id<MTLBuffer>)L->K  offset:0 atIndex:0];
                [ed setBuffer:(__bridge id<MTLBuffer>)L->Ks offset:0 atIndex:1];
                [ed setBuffer:(__bridge id<MTLBuffer>)g_kf offset:(size_t)kh*kband atIndex:2];
                [ed dispatchThreads:MTLSizeMake((NSUInteger)hd, (NSUInteger)nkey, 1)
              threadsPerThreadgroup:MTLSizeMake(hd < 64 ? hd : 64, 1, 1)];
                [ed setBuffer:(__bridge id<MTLBuffer>)L->V  offset:0 atIndex:0];
                [ed setBuffer:(__bridge id<MTLBuffer>)L->Vs offset:0 atIndex:1];
                [ed setBuffer:(__bridge id<MTLBuffer>)g_kf
                       offset:(size_t)KV*kband + (size_t)kh*kband atIndex:2];
                [ed dispatchThreads:MTLSizeMake((NSUInteger)hd, (NSUInteger)nkey, 1)
              threadsPerThreadgroup:MTLSizeMake(hd < 64 ? hd : 64, 1, 1)];
            }
            [ed endEncoding];
        }
        for (int hq = 0; hq < H; hq++) {
            int kh = hq / group, off = hq * hd;
            id<MTLBuffer> KF = (__bridge id<MTLBuffer>)g_kf;
            size_t kband = (size_t)nkey*hd*4;
            size_t koff  = (size_t)kh * kband;              /* this head's K tile */
            size_t voff  = (size_t)KV * kband + koff;       /* V tiles follow K   */

            id<MTLComputeCommandEncoder> e1 = [cb computeCommandEncoder];
            [e1 setComputePipelineState:pgq];
            [e1 setBuffer:QS offset:0 atIndex:0];
            [e1 setBuffer:QT offset:0 atIndex:1];
            [e1 setBytes:&qdim length:4 atIndex:2];
            [e1 setBytes:&off  length:4 atIndex:3];
            [e1 setBytes:&hd   length:4 atIndex:4];
            [e1 dispatchThreads:MTLSizeMake((NSUInteger)hd, (NSUInteger)S, 1)
          threadsPerThreadgroup:MTLSizeMake(hd < 64 ? hd : 64, 1, 1)];
            [e1 endEncoding];

            MPSMatrix *mq = [[MPSMatrix alloc] initWithBuffer:QT descriptor:dq];
            MPSMatrix *mk = [[MPSMatrix alloc] initWithBuffer:KF offset:koff descriptor:dk];
            MPSMatrix *ms = [[MPSMatrix alloc] initWithBuffer:SC descriptor:ds];
            [qk encodeToCommandBuffer:cb leftMatrix:mq rightMatrix:mk resultMatrix:ms];

            id<MTLComputeCommandEncoder> e2 = [cb computeCommandEncoder];
            [e2 setComputePipelineState:psm];
            [e2 setBuffer:SC offset:0 atIndex:0];
            [e2 setBytes:&nkey length:4 atIndex:1];
            [e2 setBytes:&pos0 length:4 atIndex:2];
            [e2 setBytes:&win  length:4 atIndex:3];
            [e2 setBytes:&k0   length:4 atIndex:4];
            [e2 dispatchThreadgroups:MTLSizeMake((NSUInteger)S, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [e2 endEncoding];

            MPSMatrix *mv = [[MPSMatrix alloc] initWithBuffer:KF offset:voff descriptor:dv];
            MPSMatrix *mo = [[MPSMatrix alloc] initWithBuffer:OT descriptor:do_];
            [pv encodeToCommandBuffer:cb leftMatrix:ms rightMatrix:mv resultMatrix:mo];

            id<MTLComputeCommandEncoder> e3 = [cb computeCommandEncoder];
            [e3 setComputePipelineState:pso];
            [e3 setBuffer:OT offset:0 atIndex:0];
            [e3 setBuffer:OD offset:0 atIndex:1];
            [e3 setBuffer:(__bridge id<MTLBuffer>)gsrc offset:0 atIndex:2];
            [e3 setBytes:&qdim length:4 atIndex:3];
            [e3 setBytes:&off  length:4 atIndex:4];
            [e3 setBytes:&hd   length:4 atIndex:5];
            [e3 setBytes:&H    length:4 atIndex:6];
            [e3 setBytes:&hq   length:4 atIndex:7];
            [e3 dispatchThreads:MTLSizeMake((NSUInteger)hd, (NSUInteger)S, 1)
          threadsPerThreadgroup:MTLSizeMake(hd < 64 ? hd : 64, 1, 1)];
            [e3 endEncoding];
        }
        double w0 = prof_on() ? CFAbsoluteTimeGetCurrent() : 0;
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) return 0;
        if (prof_on()) {
            g_wall_s += CFAbsoluteTimeGetCurrent() - w0;
            g_gpu_s  += cb.GPUEndTime - cb.GPUStartTime;
            g_calls++;
        }
        memcpy(ctx_out, [OD contents], (size_t)S*qdim*4);
    }
    return 1;
}

size_t lg_metal_attn_bytes(int layer) {
    if (!g_al || layer < 0 || layer >= g_al_n || !g_al[layer].K) return 0;
    /* Phase 1: the GPU attention footprint is bounded by the chunk, not the
     * context (see the budget formula in laguna_common.h) and shared across
     * layers, so this is the same tiled peak the budget reserves. */
    long H = g_al[layer].hheads ? g_al[layer].hheads : (long)g_al[layer].kv;
    long hd = g_al[layer].hd;
    double g_ = H / (double)(g_al[layer].kv ? g_al[layer].kv : 1);
    double rows = g_ * LG_ATTN_CHUNK;
    double kt = (384.0 * 1048576.0) / (rows * 4.0);
    if (kt < 256) kt = 256;
    if (kt > 65536) kt = 65536;
    return (size_t)(rows * hd * 4 + H * LG_ATTN_CHUNK * hd * 4 + rows * kt * 4 +
                    kt * hd * 8 + H * LG_ATTN_CHUNK * 8 + 2.0 * LG_ATTN_CHUNK * H * hd * 4);
}
