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
kernel void softmax_causal(device half*        Sc   [[buffer(0)]],
                           constant int&       nkey [[buffer(1)]],
                           constant int&       pos0 [[buffer(2)]],
                           constant int&       win  [[buffer(3)]],
                           constant int&       k0   [[buffer(4)]],
                           uint  row  [[threadgroup_position_in_grid]],
                           uint  lane [[thread_position_in_threadgroup]],
                           uint  W    [[threads_per_threadgroup]]) {
    device half* r = Sc + (long)row * nkey;
    int qpos = pos0 + int(row);
    int lo = 0;
    if (win > 0) { lo = qpos - win + 1; if (lo < 0) lo = 0; }
    lo -= k0; qpos -= k0;             /* into band-local column space */

    threadgroup float red[32];
    float m = -INFINITY;
    for (int t = int(lane); t <= qpos; t += int(W))
        if (t >= lo) { float v = float(r[t]); if (v > m) m = v; }
    red[lane] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { float g = -INFINITY;
        for (uint i = 0; i < W; i++) if (red[i] > g) g = red[i];
        red[0] = g; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = red[0];

    float s = 0.0f;
    for (int t = int(lane); t < nkey; t += int(W)) {
        if (t > qpos || t < lo) { r[t] = half(0.0f); continue; }
        float e = exp(float(r[t]) - m);
        r[t] = half(e); s += e;
    }
    red[lane] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { float g = 0.0f;
        for (uint i = 0; i < W; i++) g += red[i];
        red[0] = g; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = red[0] > 0.0f ? 1.0f / red[0] : 0.0f;
    for (int t = int(lane); t < nkey; t += int(W)) r[t] = half(float(r[t]) * inv);
}

/* f32 -> f16 copy of one query head's rows, into a [S, hd] contiguous tile. */
kernel void gather_q(device const float* Q  [[buffer(0)]],
                     device half*        QT [[buffer(1)]],
                     constant int&       qdim [[buffer(2)]],
                     constant int&       off  [[buffer(3)]],
                     constant int&       hd   [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    int s = int(gid.y), d = int(gid.x);
    if (d >= hd) return;
    QT[(long)s*hd + d] = half(Q[(long)s*qdim + off + d]);
}

/* scatter a [S, hd] f16 result into the strided ctx output, applying the gate. */
kernel void scatter_o(device const half*  OT [[buffer(0)]],
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
    O[(long)s*qdim + off + d] = float(OT[(long)s*hd + d]) * gate;
}
)";

typedef struct {
    void *K, *V;          /* CFBridgingRetain'd MTLBuffer, f16 [KV][phys][hd] */
    int kv, ctxcap, hd;
    int ring;             /* 0 = linear (full layers); else rows before wrap    */
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
static void *g_pipe_sm = NULL, *g_pipe_gq = NULL, *g_pipe_so = NULL;
static void *g_qt = NULL, *g_sc = NULL, *g_ot = NULL;
static size_t g_qtcap = 0, g_sccap = 0, g_otcap = 0;

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
    g_pipe_sm = (void*)CFBridgingRetain(sm);
    g_pipe_gq = (void*)CFBridgingRetain(gq);
    g_pipe_so = (void*)CFBridgingRetain(so);
    return 1;
}

int lg_metal_attn_alloc(int layers, int layer, int kv, int ctxcap, int hd, int ring) {
    if (!lg_metal_device() || !attn_pipeline()) return 0;
    if (!g_al) { g_al = (AttnLayer*)calloc(layers, sizeof(AttnLayer)); g_al_n = layers; }
    if (layer < 0 || layer >= g_al_n) return 0;
    AttnLayer *L = &g_al[layer];
    if (ring > 0 && ring >= ctxcap) ring = 0;      /* fits anyway: stay linear */
    if (L->K && L->ctxcap >= ctxcap && L->ring == ring) return 1;
    int phys = ring > 0 ? 2*ring : ctxcap;
    size_t bytes = (size_t)kv * phys * hd * 2;
    id<MTLDevice> d = lg_metal_device();
    id<MTLBuffer> kb = [d newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> vb = [d newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!kb || !vb) return 0;
    if (L->K) CFRelease((CFTypeRef)L->K);
    if (L->V) CFRelease((CFTypeRef)L->V);
    L->K = (void*)CFBridgingRetain(kb);
    L->V = (void*)CFBridgingRetain(vb);
    L->kv = kv; L->ctxcap = ctxcap; L->hd = hd; L->ring = ring;
    return 1;
}

/* Append this batch's rows. k/vv are f32 [S][kvdim] as attention() produces them. */
void lg_metal_attn_append(int layer, int pos0, int S, const float *k, const float *vv,
                          int kvdim) {
    if (!g_al || layer < 0 || layer >= g_al_n) return;
    AttnLayer *L = &g_al[layer];
    if (!L->K) return;
    __fp16 *kb = (__fp16*)[(__bridge id<MTLBuffer>)L->K contents];
    __fp16 *vb = (__fp16*)[(__bridge id<MTLBuffer>)L->V contents];
    int hd = L->hd;
    int R = L->ring, phys = R > 0 ? 2*R : L->ctxcap;
    for (int h = 0; h < L->kv; h++) {
        for (int s = 0; s < S; s++) {
            int t = pos0 + s;
            if (R == 0 && t >= L->ctxcap) break;
            const float *ks = k  + (int64_t)s*kvdim + (int64_t)h*hd;
            const float *vs = vv + (int64_t)s*kvdim + (int64_t)h*hd;
            int slot = R > 0 ? t % R : t;
            for (int rep = 0; rep < (R > 0 ? 2 : 1); rep++) {
                int64_t row = (int64_t)h*phys + slot + (int64_t)rep*R;
                __fp16 *kd = kb + row*hd, *vd = vb + row*hd;
                for (int d = 0; d < hd; d++) { kd[d] = (__fp16)ks[d]; vd[d] = (__fp16)vs[d]; }
            }
        }
    }
}

/* One (layer, chunk) attention: per query head, QK^T then softmax then PV, with
 * both GEMMs on MPS. win>0 restricts to a sliding window (unused for now: only
 * full-attention layers take this path). */
int lg_metal_attn(int layer, float *ctx_out, const float *q, const float *gt,
                  int S, int pos0, int H, int KV, int hd, float scale, int window) {
    if (!g_al || !g_pipe_sm || layer < 0 || layer >= g_al_n) return 0;
    AttnLayer *L = &g_al[layer];
    if (!L->K || pos0 + S > L->ctxcap) return 0;
    /* BANDED (LAGUNA-FORK): a sliding layer's queries in this chunk span absolute
     * positions [pos0, pos0+S), so the only keys any of them can attend are
     * [pos0-window+1, pos0+S). That band is window+S-1 wide -- CONSTANT in
     * context -- while a full layer needs all pos0+S keys. At 262144 tokens with
     * window 512 and chunk 256 that is 341x fewer score columns, which is what
     * makes the GPU viable for sliding layers at all: the dense version computed
     * the whole matrix and masked it away (measured 32.4 -> 53.1 s at 6k).
     *
     * The cache is linear per kv head, so the band is a contiguous row range and
     * costs only an offset -- no gather. */
    int k0 = 0, nkey = pos0 + S;
    if (window > 0) {
        k0 = pos0 - window + 1;
        if (k0 < 0) k0 = 0;
        nkey = pos0 + S - k0;
    }
    int group = H / KV, qdim = H * hd;
    id<MTLDevice> d = lg_metal_device();
    id<MTLCommandQueue> cq = lg_metal_queue();

    @autoreleasepool {
        /* Q for one head as f16 [S,hd]; scores [S,nkey] f16; out [S,hd] f16. */
        if (!ensure(&g_qt, &g_qtcap, (size_t)S*hd*2)) return 0;
        if (!ensure(&g_sc, &g_sccap, (size_t)S*nkey*2)) return 0;
        if (!ensure(&g_ot, &g_otcap, (size_t)S*hd*2)) return 0;
        /* the strided f32 Q and the f32 ctx/gate live in shared buffers too */
        static void *qsrc = NULL, *odst = NULL, *gsrc = NULL;
        static size_t qsc = 0, odc = 0, gsc = 0;
        if (!ensure(&qsrc, &qsc, (size_t)S*qdim*4)) return 0;
        if (!ensure(&odst, &odc, (size_t)S*qdim*4)) return 0;
        if (!ensure(&gsrc, &gsc, (size_t)S*H*4))    return 0;
        memcpy([(__bridge id<MTLBuffer>)qsrc contents], q,  (size_t)S*qdim*4);
        memcpy([(__bridge id<MTLBuffer>)gsrc contents], gt, (size_t)S*H*4);

        id<MTLBuffer> QS = (__bridge id<MTLBuffer>)qsrc;
        id<MTLBuffer> OD = (__bridge id<MTLBuffer>)odst;
        id<MTLBuffer> QT = (__bridge id<MTLBuffer>)g_qt;
        id<MTLBuffer> SC = (__bridge id<MTLBuffer>)g_sc;
        id<MTLBuffer> OT = (__bridge id<MTLBuffer>)g_ot;

        MPSMatrixDescriptor *dq = [MPSMatrixDescriptor matrixDescriptorWithRows:S columns:hd
                                    rowBytes:(size_t)hd*2 dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *dk = [MPSMatrixDescriptor matrixDescriptorWithRows:nkey columns:hd
                                    rowBytes:(size_t)hd*2 dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *ds = [MPSMatrixDescriptor matrixDescriptorWithRows:S columns:nkey
                                    rowBytes:(size_t)nkey*2 dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *dv = [MPSMatrixDescriptor matrixDescriptorWithRows:nkey columns:hd
                                    rowBytes:(size_t)hd*2 dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *do_ = [MPSMatrixDescriptor matrixDescriptorWithRows:S columns:hd
                                    rowBytes:(size_t)hd*2 dataType:MPSDataTypeFloat16];

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
        for (int hq = 0; hq < H; hq++) {
            int kh = hq / group, off = hq * hd;
            int phys = L->ring > 0 ? 2*L->ring : L->ctxcap;
            int kbase = L->ring > 0 ? (k0 % L->ring) : k0;
            size_t koff = ((size_t)kh * phys + (size_t)kbase) * hd * 2;

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
            MPSMatrix *mk = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)L->K
                                                      offset:koff descriptor:dk];
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

            MPSMatrix *mv = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)L->V
                                                      offset:koff descriptor:dv];
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
    return (size_t)g_al[layer].kv * g_al[layer].ctxcap * g_al[layer].hd * 2 * 2;
}
