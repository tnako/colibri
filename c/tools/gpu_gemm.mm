/* What does the M5 GPU deliver on the shapes this engine needs?
 * The earlier Metal probe measured only DISPATCH latency (0.327 ms) and I
 * concluded "GPU cannot win at decode". That conclusion stands for one matmul
 * per dispatch, but says nothing about PREFILL, where S=6144 rows make a single
 * dispatch do enormous work. 35.8 TFLOP in 20s needs ~1.8 TFLOP/s, so the
 * question is simply whether this GPU can sustain that.
 * Build: clang++ -x objective-c++ -std=gnu++17 -fobjc-arc -O2 gpu_gemm.mm \
 *   -o gpu_gemm -framework Metal -framework Foundation
 */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

static const char *SRC = R"(
#include <metal_stdlib>
using namespace metal;
/* C[M,N] = A[M,K] * B[N,K]^T, f32, 4x4 register tile */
kernel void gemm_f32(device const float* A [[buffer(0)]],
                     device const float* B [[buffer(1)]],
                     device float* C       [[buffer(2)]],
                     constant int& M [[buffer(3)]],
                     constant int& N [[buffer(4)]],
                     constant int& K [[buffer(5)]],
                     uint2 gid [[thread_position_in_grid]]) {
  int m0 = int(gid.y)*4, n0 = int(gid.x)*4;
  if (m0>=M || n0>=N) return;
  float acc[4][4] = {{0}};
  for (int k=0;k<K;k++){
    float a[4], b[4];
    for(int i=0;i<4;i++) a[i] = (m0+i<M)? A[(m0+i)*K+k] : 0.0f;
    for(int j=0;j<4;j++) b[j] = (n0+j<N)? B[(n0+j)*K+k] : 0.0f;
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) acc[i][j] = fma(a[i],b[j],acc[i][j]);
  }
  for(int i=0;i<4;i++) for(int j=0;j<4;j++)
    if(m0+i<M && n0+j<N) C[(m0+i)*N+n0+j] = acc[i][j];
}
/* 2-bit weights, per-group affine, f32 activations: the actual oQ shape */
kernel void gemm_oq2(device const uint*  W  [[buffer(0)]],  // [N, K/16]
                     device const float* S_ [[buffer(1)]],  // [N, K/gs]
                     device const float* Bi [[buffer(2)]],  // [N, K/gs]
                     device const float* A  [[buffer(3)]],  // [M, K]
                     device float* C        [[buffer(4)]],
                     constant int& M [[buffer(5)]],
                     constant int& N [[buffer(6)]],
                     constant int& K [[buffer(7)]],
                     constant int& gs[[buffer(8)]],
                     uint2 gid [[thread_position_in_grid]]) {
  int n = int(gid.x), m = int(gid.y);
  if (n>=N || m>=M) return;
  int words = K/16, ng = K/gs;
  device const uint* w = W + (long)n*words;
  float acc = 0.0f;
  for (int g=0; g<ng; g++){
    float sc = S_[(long)n*ng+g], bi = Bi[(long)n*ng+g];
    float dot=0.0f, xs=0.0f;
    int base=g*gs;
    for (int i=0;i<gs;i++){
      int idx=base+i;
      uint code=(w[idx>>4] >> ((idx&15)*2)) & 3u;
      float x=A[(long)m*K+idx];
      dot=fma(x,float(code),dot); xs+=x;
    }
    acc += sc*dot + bi*xs;
  }
  C[(long)m*N+n]=acc;
}
)";

int main(){
 @autoreleasepool{
  id<MTLDevice> d=MTLCreateSystemDefaultDevice();
  printf("device: %s\n",[[d name] UTF8String]);
  NSError *e=nil;
  id<MTLLibrary> lib=[d newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&e];
  if(!lib){ printf("compile: %s\n",[[e description] UTF8String]); return 1; }
  id<MTLCommandQueue> q=[d newCommandQueue];

  /* --- f32 GEMM, attention-projection shape at prefill --- */
  int M=6144,N=2048,K=2048;
  id<MTLComputePipelineState> ps=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gemm_f32"] error:&e];
  id<MTLBuffer> A=[d newBufferWithLength:(size_t)M*K*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> B=[d newBufferWithLength:(size_t)N*K*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> C=[d newBufferWithLength:(size_t)M*N*4 options:MTLResourceStorageModeShared];
  float *ap=(float*)A.contents,*bp=(float*)B.contents;
  for(long i=0;i<(long)M*K;i++)ap[i]=0.001f*(i&63);
  for(long i=0;i<(long)N*K;i++)bp[i]=0.001f*(i&31);
  for(int rep=0;rep<3;rep++){
    double t0=now();
    id<MTLCommandBuffer> cb=[q commandBuffer];
    id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
    [en setComputePipelineState:ps];
    [en setBuffer:A offset:0 atIndex:0];[en setBuffer:B offset:0 atIndex:1];[en setBuffer:C offset:0 atIndex:2];
    [en setBytes:&M length:4 atIndex:3];[en setBytes:&N length:4 atIndex:4];[en setBytes:&K length:4 atIndex:5];
    [en dispatchThreads:MTLSizeMake((N+3)/4,(M+3)/4,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    [en endEncoding];[cb commit];[cb waitUntilCompleted];
    double dt=now()-t0;
    if(rep==2) printf("f32  GEMM M=%d N=%d K=%d: %7.1f ms  %7.1f GFLOP/s\n",M,N,K,dt*1e3,2.0*M*N*K/dt/1e9);
  }

  /* --- oQ 2-bit GEMM, expert shape --- */
  int M2=6144,N2=512,K2=2048,gs=64;
  id<MTLComputePipelineState> p2=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gemm_oq2"] error:&e];
  id<MTLBuffer> W=[d newBufferWithLength:(size_t)N2*(K2/16)*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> Sc=[d newBufferWithLength:(size_t)N2*(K2/gs)*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> Bi=[d newBufferWithLength:(size_t)N2*(K2/gs)*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> C2=[d newBufferWithLength:(size_t)M2*N2*4 options:MTLResourceStorageModeShared];
  for(long i=0;i<(long)N2*(K2/16);i++)((uint32_t*)W.contents)[i]=(uint32_t)(i*2654435761u);
  for(long i=0;i<(long)N2*(K2/gs);i++){((float*)Sc.contents)[i]=0.002f;((float*)Bi.contents)[i]=-0.05f;}
  for(int rep=0;rep<3;rep++){
    double t0=now();
    id<MTLCommandBuffer> cb=[q commandBuffer];
    id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
    [en setComputePipelineState:p2];
    [en setBuffer:W offset:0 atIndex:0];[en setBuffer:Sc offset:0 atIndex:1];[en setBuffer:Bi offset:0 atIndex:2];
    [en setBuffer:A offset:0 atIndex:3];[en setBuffer:C2 offset:0 atIndex:4];
    [en setBytes:&M2 length:4 atIndex:5];[en setBytes:&N2 length:4 atIndex:6];
    [en setBytes:&K2 length:4 atIndex:7];[en setBytes:&gs length:4 atIndex:8];
    [en dispatchThreads:MTLSizeMake(N2,M2,1) threadsPerThreadgroup:MTLSizeMake(32,8,1)];
    [en endEncoding];[cb commit];[cb waitUntilCompleted];
    double dt=now()-t0;
    if(rep==2) printf("oQ2  GEMM M=%d N=%d K=%d: %7.1f ms  %7.1f GOP/s\n",M2,N2,K2,dt*1e3,2.0*M2*N2*K2/dt/1e9);
  }
 }
 return 0;
}
