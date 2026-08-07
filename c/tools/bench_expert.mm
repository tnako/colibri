// Throughput of the GPU expert kernel at real Laguna-S shapes, vs the measured
// CPU UDOT rate. Kd=3072 (hidden), N=1024 (moe_inter), 2-bit gs128.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <mach/mach_time.h>

extern "C" {
void  *lg_metal_map(const void *base, size_t len);
int    lg_metal_expert(void *wmap, size_t woff, void *sbmap, size_t sboff,
                       void *xbuf, void *ybuf,
                       int rows, int row0, int Kd, int N, int gs, int bits);
int    lg_metal_init(void);
}
id<MTLDevice> lg_metal_device(void);

static double now_s(void){ static mach_timebase_info_data_t tb; if(!tb.denom) mach_timebase_info(&tb);
                           return mach_absolute_time()*(double)tb.numer/tb.denom/1e9; }
static unsigned short fbf16(float f){ unsigned u; memcpy(&u,&f,4); return (unsigned short)(u>>16); }

int main(void) {
    if (!lg_metal_init()) { printf("no metal\n"); return 1; }
    const int Kd = 3072, N = 1024, gs = 128, bits = 2;
    int wwords = (Kd*bits+31)/32, ng = Kd/gs;
    size_t wb = ((size_t)N*wwords*4 + 16383) & ~(size_t)16383;
    size_t sb = ((size_t)N*ng*2*2   + 16383) & ~(size_t)16383;
    void *wmem = mmap(NULL, wb, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
    void *smem = mmap(NULL, sb, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
    unsigned *W = (unsigned*)wmem; unsigned short *SB = (unsigned short*)smem;
    srandom(3);
    for (size_t i = 0; i < (size_t)N*wwords; i++) W[i] = (unsigned)random();
    for (int i = 0; i < N*ng; i++) { SB[i*2]=fbf16(0.01f); SB[i*2+1]=fbf16(-0.005f); }

    @autoreleasepool {
        id<MTLDevice> d = lg_metal_device();
        void *wm = lg_metal_map(wmem, wb), *sm = lg_metal_map(smem, sb);
        printf("shape Kd=%d N=%d %d-bit gs=%d  (one Laguna-S expert matrix)\n", Kd,N,bits,gs);
        printf("%8s %10s %12s %12s\n", "rows", "ms", "GFLOP/s", "vs CPU UDOT");
        int rowset[] = {8, 32, 128, 512, 2048};
        for (int ri = 0; ri < 5; ri++) { int rows = rowset[ri];
            float *x = (float*)calloc((size_t)rows*Kd, sizeof(float));
            for (size_t i = 0; i < (size_t)rows*Kd; i++) x[i] = 0.01f*(float)(i%7);
            id<MTLBuffer> xb = [d newBufferWithBytes:x length:(size_t)rows*Kd*4
                                             options:MTLResourceStorageModeShared];
            id<MTLBuffer> yb = [d newBufferWithLength:(size_t)rows*N*4
                                              options:MTLResourceStorageModeShared];
            void *xh=(void*)CFBridgingRetain(xb), *yh=(void*)CFBridgingRetain(yb);
            lg_metal_expert(wm,0,sm,0,xh,yh,rows,0,Kd,N,gs,bits);   // warm
            int reps = rows <= 128 ? 50 : 10;
            double t0 = now_s();
            for (int r = 0; r < reps; r++)
                lg_metal_expert(wm,0,sm,0,xh,yh,rows,0,Kd,N,gs,bits);
            double dt = (now_s()-t0)/reps;
            double fl = 2.0*rows*Kd*N;
            printf("%8d %9.3f %11.1f %11.1fx\n", rows, dt*1e3, fl/dt/1e9, (fl/dt/1e9)/298.0);
            free(x);
        }
        printf("\nCPU UDOT baseline on this machine: 298 GOP/s (c/tools/bench_isa.c)\n");
    }
    return 0;
}
