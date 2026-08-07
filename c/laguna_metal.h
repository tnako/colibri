/* C interface to the Metal/MPS prefill backend (LAGUNA-FORK).
 * Implementation in laguna_metal.mm. Every entry point is safe to call when
 * Metal is unavailable: init returns 0, upload returns NULL, gemm returns 0 and
 * the caller uses its CPU path. */
#ifndef LAGUNA_METAL_H
#define LAGUNA_METAL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int         lg_metal_init(void);
int         lg_metal_available(void);
const char *lg_metal_name(void);

/* Upload row-major f32 [rows][cols] as f16. NULL on failure. */
void  *lg_metal_upload(const float *w, int rows, int cols);
void   lg_metal_free(void *handle);
size_t lg_metal_bytes(void *handle);

/* y[S,rows] = x[S,cols] @ W^T. Returns 1 if it ran on the GPU, 0 to fall back. */
int    lg_metal_gemm(void *handle, float *y, const float *x, int S);

/* ---- GPU flash attention (full-attention layers only) --------------------
 * Persistent per-layer f16 K/V on the GPU; append per chunk, one dispatch per
 * (layer, chunk) does scores + softmax + PV without materializing the score
 * matrix. All return 0 / do nothing when Metal is unavailable. */
int    lg_metal_attn_alloc(int layers, int layer, int kv, int ctxcap, int hd, int ring);
void   lg_metal_attn_append(int layer, int pos0, int S, const float *k,
                            const float *vv, int kvdim);
int    lg_metal_attn(int layer, float *ctx_out, const float *q, const float *gt,
                     int S, int pos0, int H, int KV, int hd, float scale, int window);
size_t lg_metal_attn_bytes(int layer);
/* FlashAttention-2 streaming variant: no score matrix, O(tile) memory. */
int    lg_metal_attn2(int layer, float *ctx_out, const float *q, const float *gt,
                      int S, int pos0, int H, int KV, int hd, float scale, int window);
/* GPU busy vs wall for the attention dispatches (LAGUNA_GPU_PROF=1). */
void   lg_metal_prof_dump(void);
/* GPU busy vs wall time for the attention dispatches (LAGUNA_GPU_PROF=1). */
void   lg_metal_prof_dump(void);

#ifdef __cplusplus
}
#endif
#endif
