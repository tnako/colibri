// Standalone check of the GPU expert kernel against an exact CPU oQ dequant.
// Random oQ-format weights, random activations, compare y.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/mman.h>

extern "C" {
void  *lg_metal_map(const void *base, size_t len);
int    lg_metal_expert(void *wmap, size_t woff, void *smap, size_t soff, void *bmap, size_t boff,
                       void *xbuf, void *ybuf,
                       int rows, int row0, int Kd, int N, int gs, int bits);
int    lg_metal_init(void);
}
id<MTLDevice> lg_metal_device(void);
id<MTLCommandQueue> lg_metal_queue(void);

static float bf16f(unsigned short h){ unsigned u=(unsigned)h<<16; float f; memcpy(&f,&u,4); return f; }
static unsigned short fbf16(float f){ unsigned u; memcpy(&u,&f,4); return (unsigned short)(u>>16); }

int main(void) {
    if (!lg_metal_init()) { printf("no metal\n"); return 1; }
    const int Kd = 256, N = 64, gs = 64, bits = 2, rows = 40;
    int per = 32/bits, wwords = (Kd*bits+31)/32, ng = Kd/gs;

    // page-aligned allocations so they can be wrapped no-copy
    size_t wbytes = (size_t)N*wwords*4, sbytes = (size_t)N*ng*2*2;
    void *wmem = mmap(NULL, (wbytes+16383)&~(size_t)16383, PROT_READ|PROT_WRITE,
                      MAP_PRIVATE|MAP_ANON, -1, 0);
    void *smem = mmap(NULL, (sbytes+16383)&~(size_t)16383, PROT_READ|PROT_WRITE,
                      MAP_PRIVATE|MAP_ANON, -1, 0);
    void *bmem = mmap(NULL, (sbytes+16383)&~(size_t)16383, PROT_READ|PROT_WRITE,
                      MAP_PRIVATE|MAP_ANON, -1, 0);
    unsigned *W = (unsigned*)wmem;
    unsigned short *SB = (unsigned short*)smem;
    unsigned short *BI = (unsigned short*)bmem;

    srandom(7);
    for (size_t i = 0; i < (size_t)N*wwords; i++) W[i] = (unsigned)random();
    for (int n = 0; n < N; n++) for (int g = 0; g < ng; g++) {
        SB[n*ng+g] = fbf16(0.02f + 0.001f*(float)((n+g)%7));
        BI[n*ng+g] = fbf16(-0.03f + 0.002f*(float)((n*g)%5));
    }
    float *x = (float*)malloc((size_t)rows*Kd*sizeof(float));
    for (size_t i = 0; i < (size_t)rows*Kd; i++) x[i] = ((float)random()/RAND_MAX - 0.5f);

    // exact CPU reference
    float *ref = (float*)calloc((size_t)rows*N, sizeof(float));
    for (int r = 0; r < rows; r++)
      for (int n = 0; n < N; n++) {
        double s = 0;
        for (int k = 0; k < Kd; k++) {
            unsigned word = W[(size_t)n*wwords + k/per];
            unsigned code = (word >> ((k%per)*bits)) & ((1u<<bits)-1u);
            float sc = bf16f(SB[(size_t)n*ng + k/gs]);
            float bi = bf16f(BI[(size_t)n*ng + k/gs]);
            s += (double)x[(size_t)r*Kd+k] * ((float)code*sc + bi);
        }
        ref[(size_t)r*N+n] = (float)s;
      }

    @autoreleasepool {
        id<MTLDevice> d = lg_metal_device();
        void *wm = lg_metal_map(wmem, (wbytes+16383)&~(size_t)16383);
        void *sm = lg_metal_map(smem, (sbytes+16383)&~(size_t)16383);
        void *bm = lg_metal_map(bmem, (sbytes+16383)&~(size_t)16383);
        if (!wm || !sm) { printf("map failed\n"); return 2; }
        id<MTLBuffer> xb = [d newBufferWithBytes:x length:(size_t)rows*Kd*4
                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> yb = [d newBufferWithLength:(size_t)rows*N*4
                                          options:MTLResourceStorageModeShared];
        void *xh = (void*)CFBridgingRetain(xb), *yh = (void*)CFBridgingRetain(yb);
        if (!lg_metal_expert(wm, 0, sm, 0, bm, 0, xh, yh, rows, 0, Kd, N, gs, bits)) {
            printf("dispatch failed\n"); return 3;
        }
        float *y = (float*)yb.contents;
        double maxrel = 0; int bad = 0;
        for (size_t i = 0; i < (size_t)rows*N; i++) {
            double den = fabs(ref[i]) > 1e-3 ? fabs(ref[i]) : 1e-3;
            double rel = fabs(y[i]-ref[i])/den;
            if (rel > maxrel) maxrel = rel;
            if (rel > 1e-3) bad++;
        }
        printf("rows=%d N=%d Kd=%d bits=%d gs=%d\n", rows, N, Kd, bits, gs);
        printf("max rel err %.3e, %d/%d over 1e-3 -> %s\n",
               maxrel, bad, rows*N, (bad==0 && maxrel < 1e-3) ? "OK" : "MISMATCH");
        printf("sample ref %.5f got %.5f | ref %.5f got %.5f\n",
               ref[0], y[0], ref[rows*N-1], y[rows*N-1]);
    }
    return 0;
}
