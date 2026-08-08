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

/* One threadgroup computes a TMxTN output tile for one expert.
 * A = x[rows, Kd] f32 (activations), B = expert weight [N, Kd] oQ-packed.
 * Output y[rows, N] f32.
 *
 * oQ layout (verified in docs/oq-format.md): codes are LSB-first packed into
 * uint32 along K; scales/biases are bf16, one pair per group of `gs` along K.
 *
 * TM=128 was tried (to amortize the B dequant over more rows per weight-tile
 * decode -- see the "dequant is redundant across row-tiles" note in
 * docs/gpu-expert-grouped-gemm.md) and MEASURED WORSE: 284.6s vs 199.7s
 * prefill wall on Laguna-S at 7370 tokens. Halving dequant work per row also
 * halved the number of threadgroups dispatched (half as many row-tiles per
 * expert), and the occupancy loss outweighed the compute saving -- consistent
 * with the doc's own note that 32->64 helped but going further was untested
 * and, now measured, does not. Reverted to 64. */
#define TM 64
#define TN 32
#define TK 32
#define NSG 8          /* simdgroups per threadgroup; each owns one 8-row band */
#define RB (TM / (NSG * 8))  /* 8-row bands per simdgroup (1 at TM=64) */

struct ExpArgs {
    uint Kd, N, gs, bits;
    uint rows;            /* max rows over all experts (grid sizing)   */
    uint row0;            /* unused in grouped mode                    */
    uint wwords;          /* uint32 words per weight row               */
    uint ngroups;         /* groups per row = Kd/gs                    */
    uint wslab;           /* bytes per expert in the weight tensor     */
    uint sslab;           /* bytes per expert in scales/biases         */
    /* Tensor byte offsets inside the shard, carried as scalars and applied in
     * uchar space instead of via setBuffer:offset:. safetensors puts its data at
     * header_len+8 -- 9794 here, i.e. 2 mod 16 -- so EVERY tensor offset is
     * unaligned for a uint/ushort binding. Metal's unaligned loads return
     * shifted data rather than faulting: layers 2..18 happened to survive, layer
     * 19 produced 3.4e38 and the router collapsed to expert 0 for every token. */
    uint woff_lo, woff_hi;
    uint soff_lo, soff_hi;
    uint boff_lo, boff_hi;
};

static inline float deq(uint code, float s, float b) { return fma((float)code, s, b); }

/* silu(gate) * up, in place on GPU. Chaining this in the SAME command buffer as
 * gate/up/down removes the CPU sync point that forced one commit+wait per
 * matrix per layer -- measured: 234 dispatches at 61ms GPU busy each but 170ms
 * WALL each (36% busy overall), because each commit+wait pays full command
 * buffer scheduling cost for a short kernel. Doing all layers in one buffer
 * amortizes that overhead across the whole prefill chunk. */
kernel void silu_mul(device float* G [[buffer(0)]],
                     device const float* U [[buffer(1)]],
                     constant uint& n [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    float g = G[gid];
    float s = g / (1.0f + exp(-g));
    G[gid] = s * U[gid];
}

kernel void expert_gemm(device const float*    X    [[buffer(0)]],
                        device const uint*     W    [[buffer(1)]],
                        device const ushort*   SC   [[buffer(2)]],  /* bf16 scales */
                        device const ushort*   BI   [[buffer(5)]],  /* bf16 biases */
                        device float*          Y    [[buffer(3)]],
                        constant ExpArgs&      a    [[buffer(4)]],
                        device const uint*     OFF  [[buffer(6)]],  /* [E+1] row starts */
                        device const uint2*    TILES[[buffer(7)]],  /* [ntiles] (expert, r0) */
                        uint3 tg   [[threadgroup_position_in_grid]],
                        uint  sidx [[simdgroup_index_in_threadgroup]],
                        uint  lane [[thread_index_in_simdgroup]]) {
    /* COMPACTED TILE LIST (LAGUNA-FORK): the old grid was E x ceil(maxrows/TM),
     * so an imbalanced router made every expert pay for the LARGEST expert's
     * row count. Measured on a real Laguna-XS layer: maxrows swung 698-3863
     * while the mean was 128 -- a 14.7x waste factor, and padding was nearly
     * the entire cost (engine expert-mm 49.8s vs an isolated uniform-routing
     * bench of ~11s for the same total FLOPs). grid.x now indexes a CPU-built
     * list with exactly one entry per real (expert, row-tile) pair, so padding
     * drops to at most TM-1 rows on the last tile of each expert. */
    uint2 tile = TILES[tg.x];
    uint ex = tile.x, r0 = tile.y;
    uint xs = OFF[ex], xe = OFF[ex+1];
    uint erows = xe - xs;
    uint n0 = tg.y * TN;
    if (r0 >= erows || n0 >= a.N) return;
    /* weight slab for this expert */
    ulong wo = ((ulong)a.woff_hi << 32) | a.woff_lo;
    ulong so = ((ulong)a.soff_hi << 32) | a.soff_lo;
    ulong bo = ((ulong)a.boff_hi << 32) | a.boff_lo;
    device const uchar* Wb = (device const uchar*)W  + wo + (ulong)ex * a.wslab;
    device const uchar* Sb = (device const uchar*)SC + so + (ulong)ex * a.sslab;
    device const uchar* Bb = (device const uchar*)BI + bo + (ulong)ex * a.sslab;

    /* f32 staging, deliberately. f16 staging measured 1479 vs 1125 GFLOP/s but
     * broke accuracy (max rel err 5.5e-2, 313/2560 values over 1e-3): expert
     * output feeds silu*up and then down_proj, so f16's 11-bit mantissa compounds
     * through three matmuls before the residual. simdgroup_multiply_accumulate
     * needs matching operand types, so mixed f16-weight/f32-activation is not
     * expressible either. Exact at 1125 beats fast and wrong. */
    threadgroup float As[TM * TK];
    threadgroup float Bs[TK * TN];

    /* acc[b][j]: b indexes this simdgroup's RB row-bands (each 8 rows), j
     * indexes the 4 8-wide column tiles inside TN. Was acc[4] (RB==1) when
     * TM==64; widening TM to amortize the B dequant (see the comment above
     * expert_gemm) over more rows per weight-tile decode needs each
     * simdgroup to cover RB bands instead of exactly one. */
    simdgroup_float8x8 acc[RB][4];
    for (uint b = 0; b < RB; b++)
        for (uint j = 0; j < 4; j++) acc[b][j] = make_filled_simdgroup_matrix<float,8,8>(0.0f);

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
        /* stage B: TK x TN weights, dequantized from oQ on the fly. This is
         * the expensive part (bit-unpack + bf16->f32 per element) and is now
         * shared across RB row-bands per threadgroup instead of just one --
         * doubling TM/RB halves how many times each weight tile gets
         * re-dequantized across an expert's row-tiles. */
        for (uint e = tid; e < TK*TN; e += NT) {
            uint kk = e / TN, nn = e % TN;
            uint gn = n0 + nn, gk = k0 + kk;
            float v = 0.0f;
            if (gn < a.N && gk < a.Kd) {
                uint per = 32u / a.bits;                       /* codes per word */
                uint widx = gk / per, sh = (gk % per) * a.bits;
                /* byte-wise reads: the base is unaligned, so never form a typed pointer */
                device const uchar* wp = Wb + ((ulong)gn * a.wwords + widx) * 4;
                uint word = (uint)wp[0] | ((uint)wp[1] << 8)
                          | ((uint)wp[2] << 16) | ((uint)wp[3] << 24);
                uint code = (word >> sh) & ((1u << a.bits) - 1u);
                uint g = gk / a.gs;
                device const uchar* sp = Sb + ((ulong)gn * a.ngroups + g) * 2;
                device const uchar* bp = Bb + ((ulong)gn * a.ngroups + g) * 2;
                ushort sh_ = (ushort)sp[0] | ((ushort)sp[1] << 8);
                ushort bh_ = (ushort)bp[0] | ((ushort)bp[1] << 8);
                float s = as_type<float>((uint)sh_ << 16);     /* bf16 -> f32 */
                float b = as_type<float>((uint)bh_ << 16);
                v = deq(code, s, b);
            }
            Bs[kk * TN + nn] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* Each simdgroup owns RB 8-row bands (rows [sidx*RB*8 + b*8, ...+8)
         * for b in [0,RB)) and walks the 4 column tiles for each. */
        for (uint kk = 0; kk < TK; kk += 8) {
            simdgroup_float8x8 mb[4];
            for (uint j = 0; j < 4; j++)
                simdgroup_load(mb[j], Bs + (ulong)kk*TN + j*8, TN);
            for (uint b = 0; b < RB; b++) {
                simdgroup_float8x8 ma;
                simdgroup_load(ma, As + (ulong)(sidx*RB*8 + b*8)*TK + kk, TK);
                for (uint j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(acc[b][j], ma, mb[j], acc[b][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* store: TM*TN threadgroup memory for Cs pushed this over Metal's 32KB
     * threadgroup-memory limit at TM=128 (36864 > 32768), which silently
     * failed pipeline compilation and fell back to CPU experts -- caught by
     * actually running it, not by review. Fix: Cs holds only ONE row-band
     * per simdgroup at a time (NSG*8 rows, not TM rows), flushed to device
     * memory and reused across the RB bands each simdgroup owns. */
    threadgroup float Cs[NSG * 8 * TN];
    for (uint b = 0; b < RB; b++) {
        for (uint j = 0; j < 4; j++)
            simdgroup_store(acc[b][j], Cs + (ulong)(sidx*8)*TN + j*8, TN);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < NSG*8*TN; e += NT) {
            uint rr = e / TN, nn = e % TN;
            uint sg = rr / 8, local = rr % 8;
            uint grow = sg*RB*8 + b*8 + local;
            if (r0 + grow < erows && n0 + nn < a.N)
                Y[(ulong)(xs + r0 + grow) * a.N + n0 + nn] = Cs[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
)";

typedef struct { void *buf; size_t len; } MapBuf;
static MapBuf *g_maps = NULL;
static int g_nmap = 0, g_capmap = 0;
static void *g_exp_pipe = NULL;
static void *g_silu_pipe = NULL;

static int exp_pipeline(void) {
    if (g_exp_pipe && g_silu_pipe) return 1;
    id<MTLDevice> d = lg_metal_device();
    if (!d) return 0;
    NSError *e = nil;
    id<MTLLibrary> lib = [d newLibraryWithSource:[NSString stringWithUTF8String:EXP_SRC]
                                         options:nil error:&e];
    if (!lib) { fprintf(stderr, "[metal] expert compile: %s\n",
                        [[e description] UTF8String]); return 0; }
    if (!g_exp_pipe) {
        id<MTLFunction> fn = [lib newFunctionWithName:@"expert_gemm"];
        id<MTLComputePipelineState> ps = [d newComputePipelineStateWithFunction:fn error:&e];
        if (!ps) { fprintf(stderr, "[metal] expert pipeline: %s\n",
                           [[e description] UTF8String]); return 0; }
        g_exp_pipe = (void*)CFBridgingRetain(ps);
    }
    if (!g_silu_pipe) {
        id<MTLFunction> fn2 = [lib newFunctionWithName:@"silu_mul"];
        id<MTLComputePipelineState> ps2 = [d newComputePipelineStateWithFunction:fn2 error:&e];
        if (!ps2) { fprintf(stderr, "[metal] silu pipeline: %s\n",
                            [[e description] UTF8String]); return 0; }
        g_silu_pipe = (void*)CFBridgingRetain(ps2);
    }
    return 1;
}

/* Wrap an mmap'd region with no copy. Returns an opaque handle, or NULL.
 * `base` must be page aligned, which mmap guarantees. */
extern "C" void *lg_metal_map(const void *base, size_t len) {
    if (!lg_metal_device() || !exp_pipeline()) return NULL;
    size_t use = (len + 16383) & ~(size_t)16383;   /* round UP, never truncate */
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

/* GPU busy vs wall time for expert dispatches (LAGUNA_GPU_PROF=1). */
static double g_exp_gpu_s = 0, g_exp_wall_s = 0;
static long   g_exp_calls = 0;
static int exp_prof_on(void) {
    static int v = -1;
    if (v < 0) v = getenv("LAGUNA_GPU_PROF") ? 1 : 0;
    return v;
}
extern "C" void lg_metal_expert_prof_dump(void) {
    if (!exp_prof_on() || g_exp_calls == 0) return;
    fprintf(stderr, "[gpuprof] expert: %ld dispatches, GPU busy %.2fs, wall %.2fs (%.0f%% busy)\n",
            g_exp_calls, g_exp_gpu_s, g_exp_wall_s, 100.0*g_exp_gpu_s/g_exp_wall_s);
}

/* GROUPED: all E experts in ONE dispatch, using a COMPACTED TILE LIST so an
 * imbalanced router does not force every expert to pay for the largest one's
 * row count (see the kernel comment above `expert_gemm`). `tiles` is built by
 * the caller from `offs`: one (expert, row-tile-origin) pair per real tile,
 * `ntiles` entries total. */
extern "C" int lg_metal_expert_grouped(void *wmap, size_t woff, void *smap, size_t soff,
                    void *bmap, size_t boff, void *xbuf, void *ybuf, void *offbuf,
                    void *tilesbuf, int ntiles, int Kd, int N, int gs, int bits,
                    size_t wslab, size_t sslab) {
    if (!g_exp_pipe || !wmap || !smap || !bmap || ntiles <= 0) return 0;
    @autoreleasepool {
        struct { unsigned Kd,N,gs,bits,rows,row0,wwords,ngroups,wslab,sslab,
                          wlo,whi,slo,shi,blo,bhi; } a;
        a.Kd=Kd; a.N=N; a.gs=gs; a.bits=bits; a.rows=0; a.row0=0;
        a.wwords=((unsigned)Kd*(unsigned)bits+31u)/32u; a.ngroups=(unsigned)(Kd/gs);
        a.wslab=(unsigned)wslab; a.sslab=(unsigned)sslab;
        a.wlo=(unsigned)(woff & 0xffffffffu); a.whi=(unsigned)(woff >> 32);
        a.slo=(unsigned)(soff & 0xffffffffu); a.shi=(unsigned)(soff >> 32);
        a.blo=(unsigned)(boff & 0xffffffffu); a.bhi=(unsigned)(boff >> 32);
        id<MTLCommandBuffer> cb = [lg_metal_queue() commandBuffer];
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        [en setComputePipelineState:(__bridge id<MTLComputePipelineState>)g_exp_pipe];
        [en setBuffer:(__bridge id<MTLBuffer>)xbuf   offset:0    atIndex:0];
        [en setBuffer:(__bridge id<MTLBuffer>)wmap   offset:0 atIndex:1];
        [en setBuffer:(__bridge id<MTLBuffer>)smap   offset:0 atIndex:2];
        [en setBuffer:(__bridge id<MTLBuffer>)bmap   offset:0 atIndex:5];
        [en setBuffer:(__bridge id<MTLBuffer>)ybuf   offset:0    atIndex:3];
        [en setBuffer:(__bridge id<MTLBuffer>)offbuf offset:0    atIndex:6];
        [en setBuffer:(__bridge id<MTLBuffer>)tilesbuf offset:0  atIndex:7];
        [en setBytes:&a length:sizeof(a) atIndex:4];
        [en dispatchThreadgroups:MTLSizeMake((NSUInteger)ntiles, (NSUInteger)((N+31)/32), 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [en endEncoding];
        double w0 = exp_prof_on() ? CFAbsoluteTimeGetCurrent() : 0;
        [cb commit];
        [cb waitUntilCompleted];
        if (exp_prof_on()) {
            g_exp_wall_s += CFAbsoluteTimeGetCurrent() - w0;
            g_exp_gpu_s  += cb.GPUEndTime - cb.GPUStartTime;
            g_exp_calls++;
        }
        return cb.status != MTLCommandBufferStatusError;
    }
}

/* ONE COMMAND BUFFER for a whole MoE layer: gate, up, silu(on GPU), down.
 * Replaces 3 separate commit+wait round trips (previously measured 36% GPU
 * busy: 14.38s real work inside 39.77s wall for 234 dispatches, i.e. ~108ms of
 * scheduling overhead per short kernel). One round trip per layer instead of
 * three cuts that overhead by roughly 3x. */
extern "C" int lg_metal_moe_layer(
        void *gwmap, size_t gwoff, void *gsmap, size_t gsoff, void *gbmap, size_t gboff,
        void *uwmap, size_t uwoff, void *usmap, size_t usoff, void *ubmap, size_t uboff,
        void *dwmap, size_t dwoff, void *dsmap, size_t dsoff, void *dbmap, size_t dboff,
        void *xbuf, void *gbuf, void *ubuf, void *ybuf, void *offbuf, void *tilesbuf,
        int ntiles, int npair, int D, int I, int gs, int bits,
        size_t wslabGU, size_t sslabGU, size_t wslabD, size_t sslabD) {
    if (!g_exp_pipe || !g_silu_pipe) return 0;
    @autoreleasepool {
        struct { unsigned Kd,N,gs,bits,rows,row0,wwords,ngroups,wslab,sslab,
                          wlo,whi,slo,shi,blo,bhi; } ag, au, ad;
        auto fill = [&](decltype(ag)& a, unsigned Kd, unsigned N, size_t wslab, size_t sslab,
                       size_t woff, size_t soff, size_t boff) {
            a.Kd=Kd; a.N=N; a.gs=(unsigned)gs; a.bits=(unsigned)bits; a.rows=0; a.row0=0;
            a.wwords=(Kd*(unsigned)bits+31u)/32u; a.ngroups=Kd/(unsigned)gs;
            a.wslab=(unsigned)wslab; a.sslab=(unsigned)sslab;
            a.wlo=(unsigned)(woff&0xffffffffu); a.whi=(unsigned)(woff>>32);
            a.slo=(unsigned)(soff&0xffffffffu); a.shi=(unsigned)(soff>>32);
            a.blo=(unsigned)(boff&0xffffffffu); a.bhi=(unsigned)(boff>>32);
        };
        fill(ag, D, I, wslabGU, sslabGU, gwoff, gsoff, gboff);
        fill(au, D, I, wslabGU, sslabGU, uwoff, usoff, uboff);
        fill(ad, I, D, wslabD,  sslabD,  dwoff, dsoff, dboff);

        id<MTLCommandBuffer> cb = [lg_metal_queue() commandBuffer];
        id<MTLComputePipelineState> ps = (__bridge id<MTLComputePipelineState>)g_exp_pipe;
        id<MTLComputePipelineState> sp = (__bridge id<MTLComputePipelineState>)g_silu_pipe;
        NSUInteger cols = (NSUInteger)((I+31)/32);

        id<MTLComputeCommandEncoder> e1 = [cb computeCommandEncoder];
        [e1 setComputePipelineState:ps];
        [e1 setBuffer:(__bridge id<MTLBuffer>)xbuf offset:0 atIndex:0];
        [e1 setBuffer:(__bridge id<MTLBuffer>)gwmap offset:0 atIndex:1];
        [e1 setBuffer:(__bridge id<MTLBuffer>)gsmap offset:0 atIndex:2];
        [e1 setBuffer:(__bridge id<MTLBuffer>)gbmap offset:0 atIndex:5];
        [e1 setBuffer:(__bridge id<MTLBuffer>)gbuf offset:0 atIndex:3];
        [e1 setBuffer:(__bridge id<MTLBuffer>)offbuf offset:0 atIndex:6];
        [e1 setBuffer:(__bridge id<MTLBuffer>)tilesbuf offset:0 atIndex:7];
        [e1 setBytes:&ag length:sizeof(ag) atIndex:4];
        [e1 dispatchThreadgroups:MTLSizeMake((NSUInteger)ntiles, cols, 1)
           threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e1 endEncoding];

        id<MTLComputeCommandEncoder> e2 = [cb computeCommandEncoder];
        [e2 setComputePipelineState:ps];
        [e2 setBuffer:(__bridge id<MTLBuffer>)xbuf offset:0 atIndex:0];
        [e2 setBuffer:(__bridge id<MTLBuffer>)uwmap offset:0 atIndex:1];
        [e2 setBuffer:(__bridge id<MTLBuffer>)usmap offset:0 atIndex:2];
        [e2 setBuffer:(__bridge id<MTLBuffer>)ubmap offset:0 atIndex:5];
        [e2 setBuffer:(__bridge id<MTLBuffer>)ubuf offset:0 atIndex:3];
        [e2 setBuffer:(__bridge id<MTLBuffer>)offbuf offset:0 atIndex:6];
        [e2 setBuffer:(__bridge id<MTLBuffer>)tilesbuf offset:0 atIndex:7];
        [e2 setBytes:&au length:sizeof(au) atIndex:4];
        [e2 dispatchThreadgroups:MTLSizeMake((NSUInteger)ntiles, cols, 1)
           threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e2 endEncoding];

        id<MTLComputeCommandEncoder> e3 = [cb computeCommandEncoder];
        [e3 setComputePipelineState:sp];
        [e3 setBuffer:(__bridge id<MTLBuffer>)gbuf offset:0 atIndex:0];
        [e3 setBuffer:(__bridge id<MTLBuffer>)ubuf offset:0 atIndex:1];
        unsigned n = (unsigned)npair * (unsigned)I;
        [e3 setBytes:&n length:4 atIndex:2];
        [e3 dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e3 endEncoding];

        NSUInteger colsD = (NSUInteger)((D+31)/32);
        id<MTLComputeCommandEncoder> e4 = [cb computeCommandEncoder];
        [e4 setComputePipelineState:ps];
        [e4 setBuffer:(__bridge id<MTLBuffer>)gbuf offset:0 atIndex:0];
        [e4 setBuffer:(__bridge id<MTLBuffer>)dwmap offset:0 atIndex:1];
        [e4 setBuffer:(__bridge id<MTLBuffer>)dsmap offset:0 atIndex:2];
        [e4 setBuffer:(__bridge id<MTLBuffer>)dbmap offset:0 atIndex:5];
        [e4 setBuffer:(__bridge id<MTLBuffer>)ybuf offset:0 atIndex:3];
        [e4 setBuffer:(__bridge id<MTLBuffer>)offbuf offset:0 atIndex:6];
        [e4 setBuffer:(__bridge id<MTLBuffer>)tilesbuf offset:0 atIndex:7];
        [e4 setBytes:&ad length:sizeof(ad) atIndex:4];
        [e4 dispatchThreadgroups:MTLSizeMake((NSUInteger)ntiles, colsD, 1)
           threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e4 endEncoding];

        double w0 = exp_prof_on() ? CFAbsoluteTimeGetCurrent() : 0;
        [cb commit];
        [cb waitUntilCompleted];
        if (exp_prof_on()) {
            g_exp_wall_s += CFAbsoluteTimeGetCurrent() - w0;
            g_exp_gpu_s  += cb.GPUEndTime - cb.GPUStartTime;
            g_exp_calls++;
        }
        return cb.status != MTLCommandBufferStatusError;
    }
}

/* Single-expert convenience for the standalone checkers (c/tools/chk_expert.mm,
 * bench_expert.mm). The engine uses the grouped entry point above. */
extern "C" int lg_metal_expert(void *wmap, size_t woff, void *smap, size_t soff,
                    void *bmap, size_t boff, void *xbuf, void *ybuf,
                    int rows, int row0, int Kd, int N, int gs, int bits) {
    static void *offh = NULL, *tileh = NULL;
    if (!offh) offh = lg_metal_scratch(6, 2*sizeof(unsigned));
    unsigned *o = (unsigned*)lg_metal_scratch_ptr(offh);
    if (!o) return 0;
    o[0] = (unsigned)row0; o[1] = (unsigned)(row0 + rows);
    int ntiles = (rows + LG_EXP_TM - 1) / LG_EXP_TM;
    if (!tileh) tileh = lg_metal_scratch(7, (size_t)4096 * 2 * sizeof(unsigned));
    unsigned *t = (unsigned*)lg_metal_scratch_ptr(tileh);
    if (!t) return 0;
    for (int i = 0; i < ntiles; i++) { t[i*2] = 0; t[i*2+1] = (unsigned)(i * LG_EXP_TM); }
    return lg_metal_expert_grouped(wmap, woff, smap, soff, bmap, boff,
                                   xbuf, ybuf, offh, tileh, ntiles, Kd, N, gs, bits, 0, 0);
}

