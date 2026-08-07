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
    uint rows;            /* rows for THIS expert            */
    uint row0;            /* first row in the packed x buffer */
    uint wwords;          /* uint32 words per weight row      */
    uint ngroups;         /* groups per row = Kd/gs           */
};

static inline float deq(uint code, float s, float b) { return fma((float)code, s, b); }

kernel void expert_gemm(device const float*    X    [[buffer(0)]],
                        device const uint*     W    [[buffer(1)]],
                        device const ushort*   SB   [[buffer(2)]],  /* bf16 scale,bias */
                        device float*          Y    [[buffer(3)]],
                        constant ExpArgs&      a    [[buffer(4)]],
                        uint3 tg   [[threadgroup_position_in_grid]],
                        uint  sidx [[simdgroup_index_in_threadgroup]],
                        uint  lane [[thread_index_in_simdgroup]]) {
    uint r0 = tg.x * TM;
    uint n0 = tg.y * TN;
    if (r0 >= a.rows || n0 >= a.N) return;

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
            As[e] = (gr < a.rows && k0+kk < a.Kd)
                  ? X[(ulong)(a.row0 + gr) * a.Kd + k0 + kk] : 0.0f;
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
                ushort sh_ = SB[((ulong)gn * a.ngroups + g) * 2 + 0];
                ushort bh_ = SB[((ulong)gn * a.ngroups + g) * 2 + 1];
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
        if (r0 + rr < a.rows && n0 + nn < a.N)
            Y[(ulong)(r0 + rr) * a.N + n0 + nn] = Cs[e];
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

/* y[rows, N] = x[row0.., Kd] @ dequant(W)^T for ONE expert. */
extern "C" int lg_metal_expert(void *wmap, size_t woff, void *sbmap, size_t sboff,
                    void *xbuf, void *ybuf,
                    int rows, int row0, int Kd, int N, int gs, int bits) {
    if (!g_exp_pipe || !wmap || !sbmap || rows <= 0) return 0;
    @autoreleasepool {
        id<MTLCommandQueue> cq = lg_metal_queue();
        struct { unsigned Kd, N, gs, bits, rows, row0, wwords, ngroups; } a;
        a.Kd = Kd; a.N = N; a.gs = gs; a.bits = bits;
        a.rows = rows; a.row0 = row0;
        a.wwords = ((unsigned)Kd * (unsigned)bits + 31u) / 32u;
        a.ngroups = (unsigned)(Kd / gs);

        id<MTLCommandBuffer> cb = [cq commandBuffer];
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        [en setComputePipelineState:(__bridge id<MTLComputePipelineState>)g_exp_pipe];
        [en setBuffer:(__bridge id<MTLBuffer>)xbuf  offset:0     atIndex:0];
        [en setBuffer:(__bridge id<MTLBuffer>)wmap  offset:woff  atIndex:1];
        [en setBuffer:(__bridge id<MTLBuffer>)sbmap offset:sboff atIndex:2];
        [en setBuffer:(__bridge id<MTLBuffer>)ybuf  offset:0     atIndex:3];
        [en setBytes:&a length:sizeof(a) atIndex:4];
        [en dispatchThreadgroups:MTLSizeMake((rows+63)/64, (N+31)/32, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [en endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "[metal] expert dispatch failed\n");
            return 0;
        }
    }
    return 1;
}
