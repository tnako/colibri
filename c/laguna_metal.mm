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
 * decode stays on the CPU (see docs/ENGINEERING.md). Prefill projection
 * dispatches use a fixed 32-row gate because smaller GEMMs are latency-bound.
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
         * previously cached larger N: MPS asserts otherwise. */
        if (g_mc_S != S || g_mc_N != N || g_mc_buf != g_c) {
            if (g_mc) { CFRelease((CFTypeRef)g_mc); g_mc = NULL; }
            MPSMatrixDescriptor *dc = [MPSMatrixDescriptor matrixDescriptorWithRows:S columns:N
                                        rowBytes:(size_t)N*2 dataType:MPSDataTypeFloat16];
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
        if (!mb) return 0;
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
