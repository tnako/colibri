// Same shape/data-size as bench_grouped, but reads weights from the REAL
// checkpoint's mmap'd file instead of anonymous memory, to isolate whether the
// 3x/threadgroup gap seen in the engine is a data-source effect.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <mach/mach_time.h>

extern "C" {
void  *lg_metal_map(const void *base, size_t len);
int    lg_metal_expert_grouped(void *wmap, size_t woff, void *smap, size_t soff,
            void *bmap, size_t boff, void *xbuf, void *ybuf, void *offbuf,
            void *tilesbuf, int ntiles, int Kd, int N, int gs, int bits,
            size_t wslab, size_t sslab);
void  *lg_metal_scratch(int which, size_t bytes);
void  *lg_metal_scratch_ptr(void *h);
int    lg_metal_init(void);
}
id<MTLDevice> lg_metal_device(void);

static double now_s(void){ static mach_timebase_info_data_t tb; if(!tb.denom) mach_timebase_info(&tb);
                           return mach_absolute_time()*(double)tb.numer/tb.denom/1e9; }

#include <sys/mman.h>
int main(int argc, char **argv) {
    if (!lg_metal_init()) { printf("no metal\n"); return 1; }
    const char *path = argc>1 ? argv[1] : "models/Laguna-XS-2.1-oQ2/model-00001-of-00003.safetensors";
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st);
    size_t len = (size_t)st.st_size;
    void *p = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); return 1; }
    printf("mapped real shard: %.2f GB\n", len/1e9);

    const int Kd=2048, N=512, gs=64, bits=2, E=256;
    int S = 7077, topk = 8;
    int npair = S*topk;
    int wwords=(Kd*bits+31)/32, ng=Kd/gs;
    size_t wslab=(size_t)N*wwords*4, sslab=(size_t)N*ng*2;
    // point INTO the real mapped file, well clear of the header, for E experts
    size_t wtot = (size_t)E*wslab, stot = (size_t)E*sslab;
    if (wtot + stot*2 + 1000000 > len) { printf("shard too small\n"); return 1; }
    size_t wbase = 1000000, sbase = wbase+wtot, bbase = sbase+stot;

    @autoreleasepool {
        void *wm = lg_metal_map(p, len);
        void *hx = lg_metal_scratch(0, (size_t)npair*Kd*4);
        void *hy = lg_metal_scratch(1, (size_t)npair*N*4);
        void *ho = lg_metal_scratch(4, (size_t)(E+1)*4);
        void *ht = lg_metal_scratch(5, (size_t)8192*2*4);
        float *x = (float*)lg_metal_scratch_ptr(hx);
        unsigned *offs = (unsigned*)lg_metal_scratch_ptr(ho);
        unsigned *tiles = (unsigned*)lg_metal_scratch_ptr(ht);
        for (size_t i = 0; i < (size_t)npair*Kd; i++) x[i] = 0.01f*(float)(i%13);

        int per = npair/E, maxrows=0, acc=0;
        for (int e = 0; e <= E; e++) { offs[e]=acc; if (e<E){ int r=per+(e%3); acc+=r; if(r>maxrows)maxrows=r;} }
        int ti=0;
        for (int e = 0; e < E; e++) {
            int nr = offs[e+1]-offs[e];
            for (int r0=0; r0<nr; r0+=64) { tiles[ti*2]=e; tiles[ti*2+1]=r0; ti++; }
        }
        printf("npair=%d rows/expert=%.1f ntiles=%d\n", acc, (double)acc/E, ti);

        lg_metal_expert_grouped(wm,wbase,wm,sbase,wm,bbase,hx,hy,ho,ht,ti,Kd,N,gs,bits,wslab,sslab);
        int reps=10; double t0=now_s();
        for (int r=0;r<reps;r++)
            lg_metal_expert_grouped(wm,wbase,wm,sbase,wm,bbase,hx,hy,ho,ht,ti,Kd,N,gs,bits,wslab,sslab);
        double dt=(now_s()-t0)/reps;
        double fl=2.0*acc*Kd*N;
        printf("REAL FILE mmap: %.2f ms -> %.1f GFLOP/s\n", dt*1e3, fl/dt/1e9);
    }
    return 0;
}
