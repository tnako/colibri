/* Apple's own tuned GEMM (MPSMatrixMultiplication) vs my hand-rolled kernel.
 * If MPS is much faster, the right move is to lean on it rather than hand-tune
 * Metal shaders -- it uses the AMX/matrix units and Apple's own tiling.
 * Build: clang++ -x objective-c++ -std=gnu++17 -fobjc-arc -O2 mps_gemm.mm \
 *   -o mps_gemm -framework Metal -framework MetalPerformanceShaders -framework Foundation */
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

static void bench(id<MTLDevice> d,id<MTLCommandQueue> q,int M,int N,int K,int fp16,const char*tag){
  MPSDataType dt = fp16?MPSDataTypeFloat16:MPSDataTypeFloat32;
  size_t es = fp16?2:4;
  id<MTLBuffer> A=[d newBufferWithLength:(size_t)M*K*es options:MTLResourceStorageModeShared];
  id<MTLBuffer> B=[d newBufferWithLength:(size_t)K*N*es options:MTLResourceStorageModeShared];
  id<MTLBuffer> C=[d newBufferWithLength:(size_t)M*N*es options:MTLResourceStorageModeShared];
  memset(A.contents,0x3c,(size_t)M*K*es); memset(B.contents,0x3c,(size_t)K*N*es);
  MPSMatrixDescriptor *da=[MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:K*es dataType:dt];
  MPSMatrixDescriptor *db=[MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N rowBytes:N*es dataType:dt];
  MPSMatrixDescriptor *dc=[MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N*es dataType:dt];
  MPSMatrix *ma=[[MPSMatrix alloc] initWithBuffer:A descriptor:da];
  MPSMatrix *mb=[[MPSMatrix alloc] initWithBuffer:B descriptor:db];
  MPSMatrix *mc=[[MPSMatrix alloc] initWithBuffer:C descriptor:dc];
  MPSMatrixMultiplication *mm=[[MPSMatrixMultiplication alloc] initWithDevice:d
      transposeLeft:NO transposeRight:NO resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];
  for(int r=0;r<4;r++){
    double t0=now();
    id<MTLCommandBuffer> cb=[q commandBuffer];
    [mm encodeToCommandBuffer:cb leftMatrix:ma rightMatrix:mb resultMatrix:mc];
    [cb commit];[cb waitUntilCompleted];
    double dt=now()-t0;
    if(r==3) printf("%-8s M=%5d N=%5d K=%5d  %8.2f ms  %8.1f GFLOP/s\n",tag,M,N,K,dt*1e3,2.0*M*N*K/dt/1e9);
  }
}
int main(){
 @autoreleasepool{
  id<MTLDevice> d=MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> q=[d newCommandQueue];
  printf("device: %s\n",[[d name] UTF8String]);
  bench(d,q,6144,2048,2048,0,"MPS f32");
  bench(d,q,6144,2048,2048,1,"MPS f16");
  bench(d,q,6144,6144,2048,1,"MPS f16");   /* q_proj shape */
  bench(d,q,4096,4096,4096,1,"MPS f16");   /* square, peak-ish */
 }
 return 0;
}
