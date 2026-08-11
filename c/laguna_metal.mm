/* ---- Metal/MPS backend for Laguna prefill (LAGUNA-FORK) --------------------
 *
 * WHY: measured on M5, the CPU tops out at 59 GFLOP/s (f32 FMA) while Apple's
 * MPSMatrixMultiplication reaches 15572 GFLOP/s in f16 -- 264x. Prefill at 6144
 * tokens is 35.8 TFLOP, so the CPU has a ~600s floor and the GPU a ~2.3s one.
 * Attention (projections + scores) was 81% of prefill time after the expert path
 * was fixed, so that is what moves here.
 *
 * WHAT IT DOES NOT DO: decode. A Metal dispatch round-trip measures 0.327 ms on
 * this machine and a decode step needs hundreds of matmuls, so single-token
 * decode stays on the CPU (see docs/redesign-roofline.md). The gate is
 * LG_METAL_MIN rows, mirroring upstream colibri.c's own S>=16 GEMM gate.
 *
 * PRECISION: weights are uploaded once as f16, activations converted per call.
 * f16 has 11 bits of mantissa; the tiny fixtures are checked token-exact against
 * the transformers oracle with this path active, and the engine falls back to the
 * CPU whenever a shape or dtype is not handled, so nothing silently degrades.
 */
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "laguna_metal.h"

static id<MTLDevice>       g_dev = nil;
static id<MTLCommandQueue> g_q   = nil;
static int g_ok = 0;

/* A weight resident on the GPU as f16, plus the MPS descriptor cache. */
typedef struct LgMetalW {
    void *raw;              /* CFBridgingRetain'd id<MTLBuffer>, [rows][cols] f16 */
    int rows, cols;
    /* Cached MPS kernel + descriptors. Rebuilding MPSMatrixMultiplication on every
     * call leaked badly: it caches internal pipeline/scratch state per object, and
     * prefill makes ~280 calls per step, which grew RSS from 2 GB to 14.5 GB over
     * one 6144-token prefill. Cached per (weight, S) instead. */
    void *mm;               /* MPSMatrixMultiplication for cached_S              */
    void *db;               /* right-matrix descriptor                           */
    int cached_S, cached_N;
} LgMetalW;

int lg_metal_init(void) {
    if (g_ok) return 1;
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) return 0;
        g_q = [g_dev newCommandQueue];
        if (!g_q) return 0;
        g_ok = 1;
    }
    return g_ok;
}

int lg_metal_available(void) { return g_ok; }

/* Shared with laguna_attn_metal.mm: one device and one queue for the process. */
id<MTLDevice>       lg_metal_device(void) { return g_dev; }
id<MTLCommandQueue> lg_metal_queue(void)  { return g_q; }

const char *lg_metal_name(void) {
    return g_ok ? [[g_dev name] UTF8String] : "none";
}

/* Upload a row-major f32 weight as f16. Returns NULL if Metal is unavailable,
 * so callers keep their CPU copy and fall back. */
void *lg_metal_upload(const float *w, int rows, int cols) {
    if (!g_ok) return NULL;
    @autoreleasepool {
        size_t n = (size_t)rows * cols;
        id<MTLBuffer> b = [g_dev newBufferWithLength:n*2 options:MTLResourceStorageModeShared];
        if (!b) return NULL;
        __fp16 *dst = (__fp16*)b.contents;
        for (size_t i = 0; i < n; i++) dst[i] = (__fp16)w[i];
        LgMetalW *h = (LgMetalW*)calloc(1, sizeof(LgMetalW));
        /* malloc'd memory is not ARC-managed, so retain explicitly */
        h->raw = (void*)CFBridgingRetain(b);
        h->rows = rows; h->cols = cols;
        return h;
    }
}

void lg_metal_free(void *handle) {
    if (!handle) return;
    LgMetalW *h = (LgMetalW*)handle;
    if (h->mm)  CFRelease((CFTypeRef)h->mm);
    if (h->db)  CFRelease((CFTypeRef)h->db);
    if (h->raw) CFRelease((CFTypeRef)h->raw);
    free(h);
}

size_t lg_metal_bytes(void *handle) {
    if (!handle) return 0;
    LgMetalW *h = (LgMetalW*)handle;
    return (size_t)h->rows * h->cols * 2;
}

/* Scratch buffers, grown on demand and reused across calls so a prefill does not
 * allocate per layer. Held as void* (CFRetain'd) rather than ARC locals because
 * ARC refuses to take the address of a global __strong id for write-back. Not
 * thread-safe by design: called from the serial part of the layer loop only. */
static void  *g_a = NULL, *g_c = NULL;
static size_t g_acap = 0, g_ccap = 0;
/* cached MPSMatrix wrappers over the scratch buffers */
static void *g_ma = NULL, *g_mc = NULL, *g_ma_buf = NULL, *g_mc_buf = NULL;
static int g_ma_S = -1, g_ma_K = -1, g_mc_S = -1, g_mc_N = -1;

/* Returns the buffer, and sets *grew when the underlying MTLBuffer was replaced
 * (callers cache MPSMatrix wrappers over it and must invalidate them). */
static void *ensure_buf(void **slot, size_t *cap, size_t need) {
    if (*cap >= need && *slot) return *slot;
    id<MTLBuffer> nb = [g_dev newBufferWithLength:need options:MTLResourceStorageModeShared];
    if (!nb) return NULL;
    if (*slot) CFRelease((CFTypeRef)*slot);
    *slot = (void*)CFBridgingRetain(nb);
    *cap = need;
    return *slot;
}

/* y[S,rows] = x[S,cols] @ W[rows,cols]^T   via MPS f16.
 * Returns 0 when it declines (no device, buffer failure) so the caller runs the
 * CPU path. */
int lg_metal_gemm(void *handle, float *y, const float *x, int S) {
    if (!g_ok || !handle) return 0;
    LgMetalW *h = (LgMetalW*)handle;
    int K = h->cols, N = h->rows, row0 = 0;
    @autoreleasepool {
        void *a_before = g_a, *c_before = g_c;
        if (!ensure_buf(&g_a, &g_acap, (size_t)S*K*2)) return 0;
        if (!ensure_buf(&g_c, &g_ccap, (size_t)S*N*2)) return 0;
        /* invalidate cached wrappers if the buffer object itself was replaced */
        if (g_a != a_before && g_ma) { CFRelease((CFTypeRef)g_ma); g_ma = NULL; g_ma_S = -1; }
        if (g_c != c_before && g_mc) { CFRelease((CFTypeRef)g_mc); g_mc = NULL; g_mc_S = -1; }
        id<MTLBuffer> ab = (__bridge id<MTLBuffer>)g_a;
        id<MTLBuffer> cbuf = (__bridge id<MTLBuffer>)g_c;
        __fp16 *ap = (__fp16*)ab.contents;
        for (size_t i = 0, n = (size_t)S*K; i < n; i++) ap[i] = (__fp16)x[i];

        /* W is [N,K] row-major, i.e. B^T with B[K,N]; transposeRight handles it
         * without a repack, which matters because repacking every call would
         * cost more than the GEMM. */
        if (h->cached_S != S || h->cached_N != N) {
            if (h->mm) { CFRelease((CFTypeRef)h->mm); h->mm = NULL; }
            if (h->db) { CFRelease((CFTypeRef)h->db); h->db = NULL; }
            MPSMatrixDescriptor *nd = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K
                                        rowBytes:(size_t)K*2 dataType:MPSDataTypeFloat16];
            MPSMatrixMultiplication *nm =
                [[MPSMatrixMultiplication alloc] initWithDevice:g_dev
                    transposeLeft:NO transposeRight:YES
                    resultRows:S resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];
            if (!nd || !nm) return 0;
            h->db = (void*)CFBridgingRetain(nd);
            h->mm = (void*)CFBridgingRetain(nm);
            h->cached_S = S; h->cached_N = N;
        }
        /* Cache the activation/result MPSMatrix wrappers too. They are cheap
         * objects but Metal keeps per-object state alive, and recreating them on
         * every call still grew RSS ~0.5 GB per 15 s of prefill. */
        if (g_ma_S != S || g_ma_K != K || g_ma_buf != g_a) {
            if (g_ma) { CFRelease((CFTypeRef)g_ma); g_ma = NULL; }
            MPSMatrixDescriptor *da = [MPSMatrixDescriptor matrixDescriptorWithRows:S columns:K
                                        rowBytes:(size_t)K*2 dataType:MPSDataTypeFloat16];
            MPSMatrix *nm = [[MPSMatrix alloc] initWithBuffer:ab descriptor:da];
            if (!nm) return 0;
            g_ma = (void*)CFBridgingRetain(nm);
            g_ma_S = S; g_ma_K = K; g_ma_buf = g_a;
        }
        /* Result wrapper. Its row count must match this call's N exactly, not a
         * previously cached larger N: MPS asserts "Number of requested rows in
         * result exceeds result matrix size" otherwise. That fired as soon as the
         * per-expert path started passing different N values through one handle,
         * so the cache key includes N and the buffer identity. */
        if (g_mc_S != S || g_mc_N != N || g_mc_buf != g_c) {
            if (g_mc) { CFRelease((CFTypeRef)g_mc); g_mc = NULL; }
            MPSMatrixDescriptor *dc = [MPSMatrixDescriptor matrixDescriptorWithRows:S columns:N
                                        rowBytes:(size_t)N*2 dataType:MPSDataTypeFloat16];
            /* rowBytes = N*2 means the wrapper spans exactly S*N*2 bytes, which is
             * what ensure_buf just guaranteed. */
            MPSMatrix *nm = [[MPSMatrix alloc] initWithBuffer:cbuf descriptor:dc];
            if (!nm) return 0;
            g_mc = (void*)CFBridgingRetain(nm);
            g_mc_S = S; g_mc_N = N; g_mc_buf = g_c;
        }
        MPSMatrix *ma = (__bridge MPSMatrix*)g_ma;
        MPSMatrix *mc = (__bridge MPSMatrix*)g_mc;
        MPSMatrix *mb = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)h->raw
                                                   offset:(size_t)row0*K*2
                                               descriptor:(__bridge MPSMatrixDescriptor*)h->db];
        MPSMatrixMultiplication *mm = (__bridge MPSMatrixMultiplication*)h->mm;
        id<MTLCommandBuffer> cb = [g_q commandBuffer];
        [mm encodeToCommandBuffer:cb leftMatrix:ma rightMatrix:mb resultMatrix:mc];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) return 0;
        const __fp16 *cp = (const __fp16*)cbuf.contents;
        for (size_t i = 0, n = (size_t)S*N; i < n; i++) y[i] = (float)cp[i];
    }
    return 1;
}

/* ---- persistent decode GEMV path (LAGUNA-FORK, Phase 2) --------------------
 * A small batch of GEMVs per layer, all encoded into ONE command buffer so the
 * CPU pays a single commit+wait per layer. Decode is memory-bound over the
 * resident oQ2/Q8R weights: one thread per output row, loop over the S token
 * rows in-thread, dequantizing oQ codes in-kernel with the same affine group
 * factorisation as matmul_oq (sc*dot + bi*xsum).
 *
 * The session owns the pipeline and a cache of host-memory MTLBuffer wraps
 * keyed by (ptr, bytes). Output regions are page-aligned host memory bound
 * no-copy in shared mode so the CPU sees GPU writes after the commit.
 */
#define DEC_NOCOPY_ROUND 16384
#define DEC_MAX_THREADS  256

static const char *DEC_MSL = R"DEC(
#include <metal_stdlib>
using namespace metal;

struct DecArgs {
    uint S, N, K, gs, bits, fmt;
    uint wwords, ng;
    uint xoff_lo, xoff_hi, yoff_lo, yoff_hi;
    uint woff_lo, woff_hi, soff_lo, soff_hi, boff_lo, boff_hi;
};

static inline float dec_bf16(ushort h) {
    return as_type<float>((uint)h << 16);
}

kernel void dec_gemv(device const float*   X  [[buffer(0)]],
                     device float*         Y  [[buffer(1)]],
                     device const uchar*   W  [[buffer(2)]],
                     device const uchar*   SC [[buffer(3)]],
                     device const uchar*   BI [[buffer(4)]],
                     constant DecArgs&     a  [[buffer(5)]],
                     uint row [[thread_position_in_grid]]) {
    if (row >= a.N) return;
    ulong wo = ((ulong)a.woff_hi << 32) | a.woff_lo;
    ulong so = ((ulong)a.soff_hi << 32) | a.soff_lo;
    ulong bo = ((ulong)a.boff_hi << 32) | a.boff_lo;
    ulong xo = ((ulong)a.xoff_hi << 32) | a.xoff_lo;
    ulong yo = ((ulong)a.yoff_hi << 32) | a.yoff_lo;
    device const float* Xp = (device const float*)((device const uchar*)X + xo);
    device float* Yp = (device float*)((device uchar*)Y + yo);
    const uint S = a.S, K = a.K;

    if (a.fmt == 2u) {
        device const float* Wr = (device const float*)(W + wo + (ulong)row * K * 4u);
        for (uint s = 0; s < S; s++) {
            device const float* xs = Xp + (ulong)s * K;
            float acc = 0.0f;
            for (uint k = 0; k < K; k++) acc += xs[k] * Wr[k];
            Yp[(ulong)s * a.N + row] = acc;
        }
        return;
    }
    if (a.fmt == 3u) {
        device const uchar* Wr = W + wo + (ulong)row * K * 2u;
        for (uint s = 0; s < S; s++) {
            device const float* xs = Xp + (ulong)s * K;
            float acc = 0.0f;
            for (uint k = 0; k < K; k++) {
                ushort h = (ushort)((ushort)Wr[k*2u] | ((ushort)Wr[k*2u+1u] << 8));
                acc += xs[k] * dec_bf16(h);
            }
            Yp[(ulong)s * a.N + row] = acc;
        }
        return;
    }
    /* oQ: affine group factorisation, per group g of gs inputs:
     * sum_i x_i*(c_i*s + b) = s*sum(x_i*c_i) + b*sum(x_i) = s*dot + b*xsum */
    const uint ww = a.wwords, ng = a.ng, gs = a.gs, bits = a.bits;
    const uint per = 32u / bits;
    const uint mask = (1u << bits) - 1u;
    const uint sw = (a.fmt == 0u) ? 4u : 2u;
    device const uchar* Wr = W + wo + (ulong)row * ww * 4u;
    device const uchar* Sr = SC + so + (ulong)row * ng * sw;
    device const uchar* Br = BI + bo + (ulong)row * ng * sw;
    for (uint s = 0; s < S; s++) {
        device const float* xs = Xp + (ulong)s * K;
        float acc = 0.0f;
        for (uint g = 0; g < ng; g++) {
            float dot = 0.0f, xsum = 0.0f;
            for (uint i = 0; i < gs; i++) {
                uint el = g * gs + i;
                uint widx = el / per;
                uint sh = (el % per) * bits;
                device const uchar* wp = Wr + (ulong)widx * 4u;
                uint word = (uint)wp[0] | ((uint)wp[1] << 8)
                          | ((uint)wp[2] << 16) | ((uint)wp[3] << 24);
                uint code = (word >> sh) & mask;
                float xv = xs[g * gs + i];
                dot   += (float)code * xv;
                xsum  += xv;
            }
            float sc, bi;
            if (a.fmt == 0u) {
                device const uchar* sp = Sr + (ulong)g * 4u;
                device const uchar* bp = Br + (ulong)g * 4u;
                uint sv = (uint)sp[0] | ((uint)sp[1] << 8) | ((uint)sp[2] << 16) | ((uint)sp[3] << 24);
                uint bv = (uint)bp[0] | ((uint)bp[1] << 8) | ((uint)bp[2] << 16) | ((uint)bp[3] << 24);
                sc = as_type<float>(sv); bi = as_type<float>(bv);
            } else {
                device const uchar* sp = Sr + (ulong)g * 2u;
                device const uchar* bp = Br + (ulong)g * 2u;
                ushort hs = (ushort)((ushort)sp[0] | ((ushort)sp[1] << 8));
                ushort hb = (ushort)((ushort)bp[0] | ((ushort)bp[1] << 8));
                sc = dec_bf16(hs); bi = dec_bf16(hb);
            }
            acc += sc * dot + bi * xsum;
        }
        Yp[(ulong)s * a.N + row] = acc;
    }
}

struct SiluArgs {
    uint n, bo_lo, bo_hi;
};

kernel void dec_silu(device float* G [[buffer(0)]],
                     constant SiluArgs& a [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= a.n) return;
    ulong bo = ((ulong)a.bo_hi << 32) | a.bo_lo;
    device float* Gb = (device float*)((device uchar*)G + bo);
    float v = Gb[i];
    Gb[i] = v / (1.0f + exp(-v));
}
)DEC";

typedef struct DecArgsH {
    unsigned S, N, K, gs, bits, fmt;
    unsigned wwords, ng;
    unsigned xoff_lo, xoff_hi, yoff_lo, yoff_hi;
    unsigned woff_lo, woff_hi, soff_lo, soff_hi, boff_lo, boff_hi;
} DecArgsH;

typedef struct SiluArgsH {
    unsigned n, bo_lo, bo_hi;
} SiluArgsH;

typedef struct DecWrap {
    const void *ptr;
    void *buf;              /* CFBridgingRetain'd MTLBuffer */
    size_t cap;
    size_t copied;          /* bytes of ptr content present in the buffer */
    int nocopy;             /* buffer aliases ptr (newBufferWithBytesNoCopy) */
} DecWrap;

typedef struct DecRegion {
    float *host;
    size_t bytes;
    void *buf;              /* CFBridgingRetain'd shared no-copy MTLBuffer */
} DecRegion;

typedef struct DecOp {
    int kind;               /* 0 = gemv, 1 = silu */
    void *xb, *yb, *wb, *sb, *bb;
    size_t xoff, yoff, woff, soff, boff;
    unsigned S, N, K, bits, gs, fmt, wwords, ng;
    size_t silu_n;
    const void *xsrc;       /* pending copyback for volatile x, deferred to run */
    size_t xbytes;
    int xcopy;
} DecOp;

struct LgDecode {
    int S;
    int active;
    DecRegion *regs; int nreg, creg;
    DecWrap *wraps; int nwrap, cwrap;
    DecOp *ops; int nops, cops;
    void **refs; int nrefs, cref;   /* per-op retained buffers, released at run */
};

static void *g_dec_pipe = NULL, *g_dec_spipe = NULL;

static int dec_pipeline(void) {
    if (g_dec_pipe && g_dec_spipe) return 1;
    id<MTLDevice> d = lg_metal_device();
    if (!d) return 0;
    NSError *e = nil;
    MTLCompileOptions *co = [MTLCompileOptions new];
    co.mathMode = MTLMathModeSafe;      /* keep matmul_oq bit-exact: strict fp, no fma contraction */
    id<MTLLibrary> lib = [d newLibraryWithSource:[NSString stringWithUTF8String:DEC_MSL]
                                         options:co error:&e];
    if (!lib) { fprintf(stderr, "[metal] decode compile: %s\n",
                        [[e description] UTF8String]); return 0; }
    if (!g_dec_pipe) {
        id<MTLFunction> fn = [lib newFunctionWithName:@"dec_gemv"];
        id<MTLComputePipelineState> ps = [d newComputePipelineStateWithFunction:fn error:&e];
        if (!ps) { fprintf(stderr, "[metal] decode gemv pipeline: %s\n",
                           [[e description] UTF8String]); return 0; }
        g_dec_pipe = (void*)CFBridgingRetain(ps);
    }
    if (!g_dec_spipe) {
        id<MTLFunction> fn = [lib newFunctionWithName:@"dec_silu"];
        id<MTLComputePipelineState> ps = [d newComputePipelineStateWithFunction:fn error:&e];
        if (!ps) { fprintf(stderr, "[metal] decode silu pipeline: %s\n",
                           [[e description] UTF8String]); return 0; }
        g_dec_spipe = (void*)CFBridgingRetain(ps);
    }
    return 1;
}

static void *dec_hold(LgDecode *d, void *buf) {
    if (!buf) return NULL;
    if (d->nrefs == d->cref) {
        int nc = d->cref ? d->cref * 2 : 16;
        void **nr = (void**)realloc(d->refs, (size_t)nc * sizeof(void*));
        if (!nr) return NULL;
        d->refs = nr; d->cref = nc;
    }
    void *h = (void*)CFBridgingRetain((__bridge id<MTLBuffer>)buf);
    d->refs[d->nrefs++] = h;
    return h;
}

static DecOp *dec_op(LgDecode *d) {
    if (d->nops == d->cops) {
        int nc = d->cops ? d->cops * 2 : 64;
        DecOp *no = (DecOp*)realloc(d->ops, (size_t)nc * sizeof(DecOp));
        if (!no) return NULL;
        d->ops = no; d->cops = nc;
    }
    DecOp *o = &d->ops[d->nops++];
    memset(o, 0, sizeof(*o));
    return o;
}

/* Bind a host pointer as a device buffer. Region aliases bind the region's own
 * buffer (so GPU writes to a region are visible to a later op in the same
 * command buffer); otherwise an entry is cached keyed by ptr. Page-aligned
 * pointers are wrapped no-copy; everything else is copied into a shared buffer.
 * `write` requests a buffer that reflects GPU writes back to host: only region
 * or no-copy binds qualify. Volatile (per-layer) content is copied in run()
 * so it is still current after the last CPU write before the commit. */
static int dec_bind(LgDecode *d, const void *ptr, size_t bytes, int vol, int write,
                    void **buf, size_t *off, int *copy_needed) {
    *buf = NULL; *off = 0; *copy_needed = 0;
    for (int i = 0; i < d->nreg; i++) {
        DecRegion *r = &d->regs[i];
        uintptr_t b = (uintptr_t)r->host, p = (uintptr_t)ptr;
        if (p >= b && p < b + r->bytes) {
            *buf = r->buf; *off = p - b; return 1;
        }
    }
    int wi = -1;
    for (int i = 0; i < d->nwrap; i++) if (d->wraps[i].ptr == ptr) { wi = i; break; }
    if (wi < 0) {
        if (d->nwrap == d->cwrap) {
            int nc = d->cwrap ? d->cwrap * 2 : 8;
            DecWrap *nw = (DecWrap*)realloc(d->wraps, (size_t)nc * sizeof(DecWrap));
            if (!nw) return 0;
            d->wraps = nw; d->cwrap = nc;
        }
        wi = d->nwrap++;
        memset(&d->wraps[wi], 0, sizeof(d->wraps[wi]));
        d->wraps[wi].ptr = ptr;
    }
    DecWrap *w = &d->wraps[wi];
    NSUInteger pg = NSPageSize();
    if (((uintptr_t)ptr & (pg - 1)) == 0 && !w->nocopy) {
        size_t len = (bytes + DEC_NOCOPY_ROUND - 1) & ~(size_t)(DEC_NOCOPY_ROUND - 1);
        id<MTLBuffer> b = [lg_metal_device() newBufferWithBytesNoCopy:(void*)ptr length:len
                                    options:MTLResourceStorageModeShared deallocator:nil];
        if (b) {
            if (w->buf) CFRelease((CFTypeRef)w->buf);
            w->buf = (void*)CFBridgingRetain(b);
            w->cap = len; w->nocopy = 1; w->copied = 0;
        }
    }
    if (w->nocopy) {
        if (w->cap < bytes) {
            size_t len = (bytes + DEC_NOCOPY_ROUND - 1) & ~(size_t)(DEC_NOCOPY_ROUND - 1);
            id<MTLBuffer> b = [lg_metal_device() newBufferWithBytesNoCopy:(void*)ptr length:len
                                        options:MTLResourceStorageModeShared deallocator:nil];
            if (!b) return 0;
            if (w->buf) CFRelease((CFTypeRef)w->buf);
            w->buf = (void*)CFBridgingRetain(b);
            w->cap = len; w->copied = 0;
        }
        *buf = w->buf; return 1;
    }
    if (write) return 0;
    if (w->cap < bytes) {
        id<MTLBuffer> b = [lg_metal_device() newBufferWithLength:bytes
                                    options:MTLResourceStorageModeShared];
        if (!b) return 0;
        if (w->buf) CFRelease((CFTypeRef)w->buf);
        w->buf = (void*)CFBridgingRetain(b);
        w->cap = bytes; w->copied = 0;
    }
    if (!vol) {
        if (w->copied < bytes) {
            memcpy([(__bridge id<MTLBuffer>)w->buf contents], ptr, bytes);
            w->copied = bytes;
        }
        *buf = w->buf; return 1;
    }
    *buf = w->buf; *copy_needed = 1;
    return 1;
}

LgDecode *lg_decode_new(void) {
    if (!g_ok) return NULL;
    if (!dec_pipeline()) return NULL;
    return (LgDecode*)calloc(1, sizeof(LgDecode));
}

void lg_decode_free(LgDecode *d) {
    if (!d) return;
    for (int i = 0; i < d->nreg; i++) if (d->regs[i].buf) CFRelease((CFTypeRef)d->regs[i].buf);
    for (int i = 0; i < d->nwrap; i++) if (d->wraps[i].buf) CFRelease((CFTypeRef)d->wraps[i].buf);
    for (int i = 0; i < d->nrefs; i++) CFRelease((CFTypeRef)d->refs[i]);
    free(d->regs); free(d->wraps); free(d->ops); free(d->refs);
    free(d);
}

int lg_decode_region(LgDecode *d, float *host, size_t bytes) {
    if (!g_ok || !d) return -1;
    if (!host || bytes == 0) return -1;
    NSUInteger pg = NSPageSize();
    if (((uintptr_t)host & (pg - 1)) != 0) {
        fprintf(stderr, "[metal] lg_decode_region: host %p not page-aligned\n", host);
        return -1;
    }
    size_t len = (bytes + (size_t)pg - 1) & ~((size_t)pg - 1);
    id<MTLBuffer> b = [lg_metal_device() newBufferWithBytesNoCopy:host length:len
                                options:MTLResourceStorageModeShared deallocator:nil];
    if (!b) return -1;
    if (d->nreg == d->creg) {
        int nc = d->creg ? d->creg * 2 : 8;
        DecRegion *nr = (DecRegion*)realloc(d->regs, (size_t)nc * sizeof(DecRegion));
        if (!nr) return -1;
        d->regs = nr; d->creg = nc;
    }
    DecRegion *r = &d->regs[d->nreg++];
    r->host = host; r->bytes = bytes;
    r->buf = (void*)CFBridgingRetain(b);
    return d->nreg - 1;
}

void lg_decode_begin(LgDecode *d, int S) {
    if (!d) return;
    for (int i = 0; i < d->nrefs; i++) CFRelease((CFTypeRef)d->refs[i]);
    d->nrefs = 0;
    d->nops = 0;
    d->S = S > 0 ? S : 0;
    d->active = 1;
}

int lg_decode_gemv(LgDecode *d, int ry, size_t rOff,
                   const float *x, const void *W, size_t woff,
                   const void *Sc, size_t soff, const void *Bi, size_t boff,
                   int N, int K, int bits, int gs, int fmt) {
    if (!g_ok || !d || !d->active) return 0;
    if (ry < 0 || ry >= d->nreg) return 0;
    if (!x || !W || N < 1 || K < 1 || d->S < 1) return 0;
    if (rOff & 3) return 0;
    size_t wneed = 0, sneed = 0;
    unsigned wwords = 0, ng = 0;
    switch (fmt) {
    case LG_DEC_OQF32:
    case LG_DEC_OQBF16:
        if (bits != 1 && bits != 2 && bits != 4 && bits != 8) return 0;
        if (gs < 1 || K % gs) return 0;
        if ((gs * bits) % 32) return 0;
        if (!Sc || !Bi) return 0;
        wwords = (unsigned)(((int64_t)K * bits + 31) / 32);
        ng = (unsigned)(K / gs);
        wneed = woff + (size_t)N * wwords * 4;
        sneed = soff + (size_t)N * ng * (fmt == LG_DEC_OQF32 ? 4 : 2);
        break;
    case LG_DEC_F32:
        wneed = woff + (size_t)N * K * 4;
        break;
    case LG_DEC_BF16:
        wneed = woff + (size_t)N * K * 2;
        break;
    default:
        return 0;
    }
    if (rOff + (size_t)d->S * (size_t)N * 4 > d->regs[ry].bytes) return 0;

    void *xb = NULL, *wb = NULL, *sb = NULL, *bb = NULL;
    size_t xoff = 0, woff2 = 0, soff2 = 0, boff2 = 0;
    int xcopy = 0, wcopy = 0, scopy = 0, bcopy = 0;
    if (!dec_bind(d, x, (size_t)d->S * K * 4, 1, 0, &xb, &xoff, &xcopy)) return 0;
    if (!dec_bind(d, W, wneed, 0, 0, &wb, &woff2, &wcopy)) return 0;
    if (fmt == LG_DEC_OQF32 || fmt == LG_DEC_OQBF16) {
        if (!dec_bind(d, Sc, sneed, 0, 0, &sb, &soff2, &scopy)) return 0;
        if (!dec_bind(d, Bi, sneed, 0, 0, &bb, &boff2, &bcopy)) return 0;
    }
    DecOp *o = dec_op(d);
    if (!o) return 0;
    o->kind = 0;
    o->xb = dec_hold(d, xb); if (!o->xb) return 0;
    o->yb = dec_hold(d, d->regs[ry].buf); if (!o->yb) return 0;
    o->wb = dec_hold(d, wb); if (!o->wb) return 0;
    if (sb) { o->sb = dec_hold(d, sb); if (!o->sb) return 0; }
    if (bb) { o->bb = dec_hold(d, bb); if (!o->bb) return 0; }
    o->xoff = xoff; o->yoff = rOff;
    o->woff = woff2 + woff; o->soff = soff2 + soff; o->boff = boff2 + boff;
    o->S = (unsigned)d->S; o->N = (unsigned)N; o->K = (unsigned)K;
    o->bits = (unsigned)bits; o->gs = (unsigned)gs; o->fmt = (unsigned)fmt;
    o->wwords = wwords; o->ng = ng;
    o->xsrc = x; o->xbytes = (size_t)d->S * K * 4; o->xcopy = xcopy;
    return 1;
}

int lg_decode_silu(LgDecode *d, float *gb, size_t gboff, size_t n) {
    if (!g_ok || !d || !d->active) return 0;
    if (!gb || n < 1) return 0;
    if (gboff & 3) return 0;
    void *buf = NULL; size_t off = 0; int copy_needed = 0;
    if (!dec_bind(d, gb, gboff + n * 4, 1, 1, &buf, &off, &copy_needed)) return 0;
    DecOp *o = dec_op(d);
    if (!o) return 0;
    o->kind = 1;
    o->xb = dec_hold(d, buf); if (!o->xb) return 0;
    o->xoff = off + gboff;
    o->silu_n = n;
    return 1;
}

int lg_decode_run(LgDecode *d) {
    if (!g_ok || !d) return 0;
    if (!d->active) return 0;
    if (d->nops == 0) { d->active = 0; return 1; }
    int ok = 0;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [lg_metal_queue() commandBuffer];
        if (cb) {
            for (int i = 0; i < d->nops; i++) {
                DecOp *o = &d->ops[i];
                if (o->xcopy && o->xsrc)
                    memcpy([(__bridge id<MTLBuffer>)o->xb contents], o->xsrc, o->xbytes);
            }
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (int i = 0; i < d->nops; i++) {
                DecOp *o = &d->ops[i];
                if (o->kind == 0) {
                    if (!o->xb || !o->yb || !o->wb) continue;
                    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)g_dec_pipe];
                    DecArgsH h;
                    h.S = o->S; h.N = o->N; h.K = o->K; h.gs = o->gs;
                    h.bits = o->bits; h.fmt = o->fmt; h.wwords = o->wwords; h.ng = o->ng;
                    h.xoff_lo = (unsigned)(o->xoff & 0xffffffffu);
                    h.xoff_hi = (unsigned)(o->xoff >> 32);
                    h.yoff_lo = (unsigned)(o->yoff & 0xffffffffu);
                    h.yoff_hi = (unsigned)(o->yoff >> 32);
                    h.woff_lo = (unsigned)(o->woff & 0xffffffffu);
                    h.woff_hi = (unsigned)(o->woff >> 32);
                    h.soff_lo = (unsigned)(o->soff & 0xffffffffu);
                    h.soff_hi = (unsigned)(o->soff >> 32);
                    h.boff_lo = (unsigned)(o->boff & 0xffffffffu);
                    h.boff_hi = (unsigned)(o->boff >> 32);
                    [enc setBuffer:(__bridge id<MTLBuffer>)o->xb offset:0 atIndex:0];
                    [enc setBuffer:(__bridge id<MTLBuffer>)o->yb offset:0 atIndex:1];
                    [enc setBuffer:(__bridge id<MTLBuffer>)o->wb offset:0 atIndex:2];
                    [enc setBuffer:(__bridge id<MTLBuffer>)(o->sb ? o->sb : o->wb) offset:0 atIndex:3];
                    [enc setBuffer:(__bridge id<MTLBuffer>)(o->bb ? o->bb : o->wb) offset:0 atIndex:4];
                    [enc setBytes:&h length:sizeof(h) atIndex:5];
                    [enc dispatchThreads:MTLSizeMake((NSUInteger)o->N, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(DEC_MAX_THREADS, 1, 1)];
                } else {
                    if (!o->xb) continue;
                    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)g_dec_spipe];
                    SiluArgsH sh;
                    sh.n = (unsigned)o->silu_n;
                    sh.bo_lo = (unsigned)(o->xoff & 0xffffffffu);
                    sh.bo_hi = (unsigned)(o->xoff >> 32);
                    [enc setBuffer:(__bridge id<MTLBuffer>)o->xb offset:0 atIndex:0];
                    [enc setBytes:&sh length:sizeof(sh) atIndex:1];
                    [enc dispatchThreads:MTLSizeMake((NSUInteger)o->silu_n, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(DEC_MAX_THREADS, 1, 1)];
                }
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            ok = (cb.status != MTLCommandBufferStatusError) ? 1 : 0;
        }
        for (int i = 0; i < d->nrefs; i++) CFRelease((CFTypeRef)d->refs[i]);
        d->nrefs = 0;
    }
    d->active = 0;
    return ok;
}

float *lg_decode_region_ptr(LgDecode *d, int ry) {
    if (!d || ry < 0 || ry >= d->nreg) return NULL;
    return d->regs[ry].host;
}

size_t lg_decode_bytes(LgDecode *d) {
    if (!d) return 0;
    size_t t = 0;
    for (int i = 0; i < d->nreg; i++) t += d->regs[i].bytes;
    return t;
}

int lg_decode_active(LgDecode *d) {
    return d && d->nreg > 0;
}
