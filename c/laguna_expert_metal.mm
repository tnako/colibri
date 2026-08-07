/* ---- routed experts on the GPU, straight from the mmap'd checkpoint ---------
 *
 * This replaces the whole CPU expert path: the Q8R resident bank (39 GB on
 * Laguna-S, which is why it never fit), the streaming LRU cache, and the UDOT
 * kernels. Expert matmul was 70% of Laguna-S prefill (438.7 s of 615 s).
 *
 * THE KEY FACT, measured before writing this (c/tools/nocopy.mm):
 * `newBufferWithBytesNoCopy` accepts an mmap'd safetensors shard. The GPU then
 * reads the SAME physical pages the CPU mapped -- zero copy, and those pages are
 * evictable page cache rather than dirty RSS. So the 29 GB of 2-bit expert
 * weights never enter the memory budget at all. That is what makes Laguna-S fit
 * in 20 GB: the weights are simply not resident.
 *
 * The kernel dequantizes oQ in-register (affine: w = q*scale + bias, per group of
 * `gs` along K) and accumulates with simdgroup_matrix 8x8 tiles, which drive the
 * GPU's matrix units. A scalar kernel loses to MPS by 5-6x -- measured three
 * times on this project -- so the tiles are the point.
 *
 * Grouping: tokens are sorted by expert on the CPU (already done for the cache),
 * so each expert's rows are contiguous and one threadgroup handles one
 * (expert, row-tile, col-tile).
 */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "laguna_metal.h"

extern id<MTLDevice>       lg_metal_device(void);
extern id<MTLCommandQueue> lg_metal_queue(void);

static const char *EXP_SRC = R"(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

/* One threadgroup computes a 32x32 output tile for one expert.
 * A = x[rows, Kd] f32 (activations), B = expert weight [N, Kd] oQ-packed.
 * Output y[rows, N] f32.
 *
 * oQ layout (verified in docs/oq-format.md): codes are LSB-first packed into
 * uint32 along K; scales/biases are bf16, one pair per group of `gs` along K. */
#define TM 64
#define TN 32
#define TK 32
#define NSG 8          /* simdgroups per threadgroup; each owns one 8-row band */

struct ExpArgs {
    uint Kd, N, gs, bits;
    uint rows;            /* max rows over all experts (grid sizing)   */
    uint row0;            /* unused in grouped mode                    */
    uint wwords;          /* uint32 words per weight row               */
    uint ngroups;         /* groups per row = Kd/gs                    */
    uint wslab;           /* bytes per expert in the weight tensor     */
    uint sslab;           /* bytes per expert in scales/biases         */
};

static inline float deq(uint code, float s, float b) { return fma((float)code, s, b); }

kernel void expert_gemm(device const float*    X    [[buffer(0)]],
                        device const uint*     W    [[buffer(1)]],
                        device const ushort*   SC   [[buffer(2)]],  /* bf16 scales */
                        device const ushort*   BI   [[buffer(5)]],  /* bf16 biases */
                        device float*          Y    [[buffer(3)]],
                        constant ExpArgs&      a    [[buffer(4)]],
                        device const uint*     OFF  [[buffer(6)]],  /* [E+1] row starts */
                        uint3 tg   [[threadgroup_position_in_grid]],
                        uint  sidx [[simdgroup_index_in_threadgroup]],
                        uint  lane [[thread_index_in_simdgroup]]) {
    /* GROUPED: grid.z selects the expert, so ONE dispatch covers all of them and
     * the GPU schedules every expert's threadgroups concurrently. Per-expert
     * dispatches left the GPU idle (profile: 35330 samples in __psynch_cvwait). */
    uint ex = tg.z;
    uint xs = OFF[ex], xe = OFF[ex+1];
    uint erows = xe - xs;
    uint r0 = tg.x * TM;
    uint n0 = tg.y * TN;
    if (r0 >= erows || n0 >= a.N) return;
    /* weight slab for this expert */
    W  = (device const uint*)  ((device const uchar*)W  + (ulong)ex * a.wslab);
    SC = (device const ushort*)((device const uchar*)SC + (ulong)ex * a.sslab);
    BI = (device const ushort*)((device const uchar*)BI + (ulong)ex * a.sslab);

    /* f32 staging, deliberately. f16 staging measured 1479 vs 1125 GFLOP/s but
     * broke accuracy (max rel err 5.5e-2, 313/2560 values over 1e-3): expert
     * output feeds silu*up and then down_proj, so f16's 11-bit mantissa compounds
     * through three matmuls before the residual. simdgroup_multiply_accumulate
     * needs matching operand types, so mixed f16-weight/f32-activation is not
     * expressible either. Exact at 1125 beats fast and wrong. */
    threadgroup float As[TM * TK];
    threadgroup float Bs[TK * TN];

    simdgroup_float8x8 acc[4];
    for (uint j = 0; j < 4; j++) acc[j] = make_filled_simdgroup_matrix<float,8,8>(0.0f);

    uint tid = sidx * 32 + lane;                 /* 0..NSG*32-1 */
    uint NT  = NSG * 32;

    for (uint k0 = 0; k0 < a.Kd; k0 += TK) {
        /* stage A: TM x TK activations */
        for (uint e = tid; e < TM*TK; e += NT) {
            uint rr = e / TK, kk = e % TK;
            uint gr = r0 + rr;
            As[e] = (gr < erows && k0+kk < a.Kd)
                  ? X[(ulong)(xs + gr) * a.Kd + k0 + kk] : 0.0f;
        }
        /* stage B: TK x TN weights, dequantized from oQ on the fly */
        for (uint e = tid; e < TK*TN; e += NT) {
            uint kk = e / TN, nn = e % TN;
            uint gn = n0 + nn, gk = k0 + kk;
            float v = 0.0f;
            if (gn < a.N && gk < a.Kd) {
                uint per = 32u / a.bits;                       /* codes per word */
                uint widx = gk / per, sh = (gk % per) * a.bits;
                uint word = W[(ulong)gn * a.wwords + widx];
                uint code = (word >> sh) & ((1u << a.bits) - 1u);
                uint g = gk / a.gs;
                ushort sh_ = SC[(ulong)gn * a.ngroups + g];
                ushort bh_ = BI[(ulong)gn * a.ngroups + g];
                float s = as_type<float>((uint)sh_ << 16);     /* bf16 -> f32 */
                float b = as_type<float>((uint)bh_ << 16);
                v = deq(code, s, b);
            }
            Bs[kk * TN + nn] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* Each of the 4 simdgroups owns one 8-row band of the 32-row tile, and
         * walks the 4 column tiles. acc[0][j] is that band's j-th 8x8 tile. */
        for (uint kk = 0; kk < TK; kk += 8) {
            simdgroup_float8x8 ma, mb;
            simdgroup_load(ma, As + (ulong)(sidx*8)*TK + kk, TK);
            for (uint j = 0; j < 4; j++) {
                simdgroup_load(mb, Bs + (ulong)kk*TN + j*8, TN);
                simdgroup_multiply_accumulate(acc[j], ma, mb, acc[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* store: simdgroup sidx owns rows [sidx*8, sidx*8+8) */
    threadgroup float Cs[TM * TN];
    for (uint j = 0; j < 4; j++)
        simdgroup_store(acc[j], Cs + (ulong)(sidx*8)*TN + j*8, TN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < TM*TN; e += NT) {
        uint rr = e / TN, nn = e % TN;
        if (r0 + rr < erows && n0 + nn < a.N)
            Y[(ulong)(xs + r0 + rr) * a.N + n0 + nn] = Cs[e];
    }
}
)";

typedef struct { void *buf; size_t len; } MapBuf;
static MapBuf *g_maps = NULL;
static int g_nmap = 0, g_capmap = 0;
static void *g_exp_pipe = NULL;

static int exp_pipeline(void) {
    if (g_exp_pipe) return 1;
    id<MTLDevice> d = lg_metal_device();
    if (!d) return 0;
    NSError *e = nil;
    id<MTLLibrary> lib = [d newLibraryWithSource:[NSString stringWithUTF8String:EXP_SRC]
                                         options:nil error:&e];
    if (!lib) { fprintf(stderr, "[metal] expert compile: %s\n",
                        [[e description] UTF8String]); return 0; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"expert_gemm"];
    id<MTLComputePipelineState> ps = [d newComputePipelineStateWithFunction:fn error:&e];
    if (!ps) { fprintf(stderr, "[metal] expert pipeline: %s\n",
                       [[e description] UTF8String]); return 0; }
    g_exp_pipe = (void*)CFBridgingRetain(ps);
    return 1;
}

/* Wrap an mmap'd region with no copy. Returns an opaque handle, or NULL.
 * `base` must be page aligned, which mmap guarantees. */
extern "C" void *lg_metal_map(const void *base, size_t len) {
    if (!lg_metal_device() || !exp_pipeline()) return NULL;
    size_t use = len & ~(size_t)16383;
    if (!use) return NULL;
    id<MTLBuffer> b = [lg_metal_device() newBufferWithBytesNoCopy:(void*)base
                                                          length:use
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
    if (!b) return NULL;
    if (g_nmap == g_capmap) {
        g_capmap = g_capmap ? g_capmap*2 : 8;
        g_maps = (MapBuf*)realloc(g_maps, (size_t)g_capmap*sizeof(MapBuf));
    }
    g_maps[g_nmap].buf = (void*)CFBridgingRetain(b);
    g_maps[g_nmap].len = use;
    return g_maps[g_nmap++].buf;
}

extern "C" size_t lg_metal_map_count(void) { return (size_t)g_nmap; }

/* Persistent shared-storage scratch buffers, indexed by slot. Shared storage on
 * unified memory means the CPU writes activations and the GPU reads the same
 * pages with no copy in either direction. */
static void  *g_scr[8];
static size_t g_scrlen[8];
extern "C" void *lg_metal_scratch(int which, size_t bytes) {
    if (which < 0 || which >= 8 || !lg_metal_device()) return NULL;
    if (g_scr[which] && g_scrlen[which] >= bytes) return g_scr[which];
    id<MTLBuffer> b = [lg_metal_device() newBufferWithLength:bytes
                                                     options:MTLResourceStorageModeShared];
    if (!b) return NULL;
    if (g_scr[which]) CFRelease((CFTypeRef)g_scr[which]);
    g_scr[which] = (void*)CFBridgingRetain(b);
    g_scrlen[which] = bytes;
    return g_scr[which];
}
extern "C" void *lg_metal_scratch_ptr(void *h) {
    return h ? [(__bridge id<MTLBuffer>)h contents] : NULL;
}

/* NOTE: an earlier per-expert batched API (begin/add/end) lived here. It issued
 * one dispatch per expert -- 768 per layer, 36,864 per chunk -- and profiling
 * showed the process 87% blocked in __psynch_cvwait with the GPU idle. The
 * grouped kernel below replaced it and it was deleted rather than kept as a
 * second path. See docs/gpu-expert-grouped-gemm.md. */

/* GROUPED: all E experts in ONE dispatch. `offs` is E+1 row starts into the
 * expert-sorted activation buffer; maxrows sizes the grid. */
extern "C" int lg_metal_expert_grouped(void *wmap, size_t woff, void *smap, size_t soff,
                    void *bmap, size_t boff, void *xbuf, void *ybuf, void *offbuf,
                    int E, int maxrows, int Kd, int N, int gs, int bits,
                    size_t wslab, size_t sslab) {
    if (!g_exp_pipe || !wmap || !smap || !bmap || maxrows <= 0) return 0;
    @autoreleasepool {
        struct { unsigned Kd,N,gs,bits,rows,row0,wwords,ngroups,wslab,sslab; } a;
        a.Kd=Kd; a.N=N; a.gs=gs; a.bits=bits; a.rows=maxrows; a.row0=0;
        a.wwords=((unsigned)Kd*(unsigned)bits+31u)/32u; a.ngroups=(unsigned)(Kd/gs);
        a.wslab=(unsigned)wslab; a.sslab=(unsigned)sslab;
        id<MTLCommandBuffer> cb = [lg_metal_queue() commandBuffer];
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        [en setComputePipelineState:(__bridge id<MTLComputePipelineState>)g_exp_pipe];
        [en setBuffer:(__bridge id<MTLBuffer>)xbuf   offset:0    atIndex:0];
        [en setBuffer:(__bridge id<MTLBuffer>)wmap   offset:woff atIndex:1];
        [en setBuffer:(__bridge id<MTLBuffer>)smap   offset:soff atIndex:2];
        [en setBuffer:(__bridge id<MTLBuffer>)bmap   offset:boff atIndex:5];
        [en setBuffer:(__bridge id<MTLBuffer>)ybuf   offset:0    atIndex:3];
        [en setBuffer:(__bridge id<MTLBuffer>)offbuf offset:0    atIndex:6];
        [en setBytes:&a length:sizeof(a) atIndex:4];
        [en dispatchThreadgroups:MTLSizeMake((maxrows+63)/64, (N+31)/32, E)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [en endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return cb.status != MTLCommandBufferStatusError;
    }
}

/* Single-expert convenience for the standalone checkers (c/tools/chk_expert.mm,
 * bench_expert.mm). The engine uses the grouped entry point above. */
extern "C" int lg_metal_expert(void *wmap, size_t woff, void *smap, size_t soff,
                    void *bmap, size_t boff, void *xbuf, void *ybuf,
                    int rows, int row0, int Kd, int N, int gs, int bits) {
    static void *offh = NULL;
    unsigned *o = NULL;
    if (!offh) offh = lg_metal_scratch(7, 2*sizeof(unsigned));
    o = (unsigned*)lg_metal_scratch_ptr(offh);
    if (!o) return 0;
    o[0] = (unsigned)row0; o[1] = (unsigned)(row0 + rows);
    return lg_metal_expert_grouped(wmap, woff, smap, soff, bmap, boff,
                                   xbuf, ybuf, offh, 1, rows, Kd, N, gs, bits, 0, 0);
}
