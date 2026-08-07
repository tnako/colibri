// Does newBufferWithBytesNoCopy work on an mmap'd file? If yes, the GPU can read
// the 2-bit checkpoint directly out of page cache: zero copy, zero dirty RSS.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <mach/mach_time.h>
static double now_s(void){ static mach_timebase_info_data_t tb; if(!tb.denom) mach_timebase_info(&tb); return mach_absolute_time()*(double)tb.numer/tb.denom/1e9; }

static const char *SRC = R"(
#include <metal_stdlib>
using namespace metal;
// sum bytes so the compiler cannot elide the read
kernel void touch(device const uchar* p [[buffer(0)]],
                  device atomic_uint* out [[buffer(1)]],
                  constant ulong& n [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {
    ulong stride = 4096;                 // one sample per page
    ulong i = (ulong)gid * stride;
    if (i >= n) return;
    atomic_fetch_add_explicit(out, (uint)p[i], memory_order_relaxed);
}
)";

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <file>\n", argv[0]); return 1; }
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st);
    size_t n = st.st_size;
    void *p = mmap(NULL, n, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); return 1; }
    printf("mapped %.2f GB at %p (page aligned: %s)\n",
           n/1e9, p, ((uintptr_t)p & 0x3fff) ? "NO" : "yes");

    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        printf("device: %s\n", [[d name] UTF8String]);
        printf("max buffer length: %.2f GB\n", d.maxBufferLength/1e9);

        // The critical call: wrap existing mmap'd pages with no copy.
        size_t len = n & ~(size_t)16383;          // must be page-multiple
        id<MTLBuffer> b = [d newBufferWithBytesNoCopy:p
                                              length:len
                                             options:MTLResourceStorageModeShared
                                         deallocator:nil];
        if (!b) { printf("RESULT: newBufferWithBytesNoCopy FAILED\n"); return 2; }
        printf("RESULT: wrapped %.2f GB with NO COPY, gpuAddress=%llu\n",
               len/1e9, (unsigned long long)b.gpuAddress);

        NSError *e = nil;
        id<MTLLibrary> lib = [d newLibraryWithSource:[NSString stringWithUTF8String:SRC]
                                             options:nil error:&e];
        id<MTLFunction> fn = [lib newFunctionWithName:@"touch"];
        id<MTLComputePipelineState> ps = [d newComputePipelineStateWithFunction:fn error:&e];
        id<MTLCommandQueue> q = [d newCommandQueue];
        id<MTLBuffer> out = [d newBufferWithLength:4 options:MTLResourceStorageModeShared];

        size_t pages = len / 4096;
        double t0 = now_s();
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        [en setComputePipelineState:ps];
        [en setBuffer:b offset:0 atIndex:0];
        [en setBuffer:out offset:0 atIndex:1];
        [en setBytes:&len length:8 atIndex:2];
        [en dispatchThreads:MTLSizeMake(pages,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [en endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        double dt = now_s() - t0;
        printf("GPU touched %zu pages in %.3fs (checksum %u)\n",
               pages, dt, *(unsigned*)out.contents);
        printf("status: %s\n", cb.status == MTLCommandBufferStatusCompleted ? "COMPLETED" : "ERROR");
    }
    return 0;
}
