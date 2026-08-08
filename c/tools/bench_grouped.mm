// Is the SLOWNESS in the kernel, or in the grouped dispatch shape?
// Mimics the engine exactly: 256 experts, ~23 rows each, XS geometry.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <mach/mach_time.h>

extern "C" {
void  *lg_metal_map(const void *base, size_t len);
int    lg_metal_expert_grouped(void *wmap, size_t woff, void *smap, size_t soff,
            void *bmap, size_t boff, void *xbuf, void *ybuf, void *offbuf,
            int E, int maxrows, int Kd, int N, int gs, int bits,
            size_t wslab, size_t sslab);
void  *lg_metal_scratch(int which, size_t bytes);
void  *lg_metal_scratch_ptr(void *h);
int    lg_metal_init(void);
}
id<MTLDevice> lg_metal_device(void);
id<MTLCommandQueue> lg_metal_queue(void);

static double now_s(void){ static mach_timebase_info_data_t tb; if(!tb.denom) mach_timebase_info(&tb);
                           return mach_absolute_time()*(double)tb.numer/tb.denom/1e9; }
static unsigned short fbf16(float f){ unsigned u; memcpy(&u,&f,4); return (unsigned short)(u>>16); }

int main(int argc, char **argv) {
    if (!lg_metal_init()) { printf("no metal\n"); return 1; }
    // XS MoE geometry
    const int Kd = 2048, N = 512, gs = 64, bits = 2, E = 256;
    int S = argc > 1 ? atoi(argv[1]) : 726, topk = 8;
    int npair = S * topk;

    int wwords = (Kd*bits+31)/32, ng = Kd/gs;
    size_t wslab = (size_t)N*wwords*4, sslab = (size_t)N*ng*2;
    size_t wtot = ((size_t)E*wslab + 16383) & ~(size_t)16383;
    size_t stot = ((size_t)E*sslab + 16383) & ~(size_t)16383;

    void *wmem = mmap(NULL, wtot, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
    void *smem = mmap(NULL, stot, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
    void *bmem = mmap(NULL, stot, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
    unsigned *W = (unsigned*)wmem;
    unsigned short *SC = (unsigned short*)smem, *BI = (unsigned short*)bmem;
    srandom(5);
    for (size_t i = 0; i < (size_t)E*wslab/4; i++) W[i] = (unsigned)random();
    for (size_t i = 0; i < (size_t)E*sslab/2; i++) { SC[i]=fbf16(0.01f); BI[i]=fbf16(-0.004f); }

    @autoreleasepool {
        void *wm = lg_metal_map(wmem, wtot);
        void *sm = lg_metal_map(smem, stot);
        void *bm = lg_metal_map(bmem, stot);

        void *hx = lg_metal_scratch(0, (size_t)npair*Kd*4);
        void *hy = lg_metal_scratch(1, (size_t)npair*N*4);
        void *ho = lg_metal_scratch(4, (size_t)(E+1)*4);
        float *x = (float*)lg_metal_scratch_ptr(hx);
        unsigned *offs = (unsigned*)lg_metal_scratch_ptr(ho);
        for (size_t i = 0; i < (size_t)npair*Kd; i++) x[i] = 0.01f*(float)(i%13);

        // realistic near-uniform routing
        int per = npair / E, maxrows = 0, acc = 0;
        for (int e = 0; e <= E; e++) {
            offs[e] = acc;
            if (e < E) { int r = per + (e % 3); acc += r; if (r > maxrows) maxrows = r; }
        }
        offs[E] = acc;

        printf("XS MoE shape: Kd=%d N=%d %d-bit E=%d | S=%d topk=%d\n", Kd,N,bits,E,S,topk);
        printf("npair=%d, rows/expert=%.1f, maxrows=%d\n", acc, (double)acc/E, maxrows);

        lg_metal_expert_grouped(wm,0,sm,0,bm,0,hx,hy,ho,E,maxrows,Kd,N,gs,bits,wslab,sslab);
        int reps = 10;
        double t0 = now_s();
        for (int r = 0; r < reps; r++)
            lg_metal_expert_grouped(wm,0,sm,0,bm,0,hx,hy,ho,E,maxrows,Kd,N,gs,bits,wslab,sslab);
        double dt = (now_s()-t0)/reps;
        double fl = 2.0*acc*Kd*N;
        printf("\ngrouped call: %.2f ms -> %.1f GFLOP/s (%.2f%% of 15572 peak)\n",
               dt*1e3, fl/dt/1e9, 100.0*fl/dt/1e9/15572);
        printf("weight bytes touched: %.1f MB -> %.1f GB/s\n",
               (double)E*wslab/1e6, (double)E*wslab/dt/1e9);

        // What does ONE big dense GEMM of the same total FLOPs cost? (upper bound)
        printf("\nfor reference, the same FLOPs as one dense GEMM would need %.2f ms at peak\n",
               fl/15572e9*1e3);
    }
    return 0;
}
