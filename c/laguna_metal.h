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
/* Same, but the source is already f16 (one big staging buffer, no conversion). */
void  *lg_metal_upload_f16(const void *w16, int rows, int cols);
void   lg_metal_free(void *handle);
size_t lg_metal_bytes(void *handle);

/* y[S,rows] = x[S,cols] @ W^T. Returns 1 if it ran on the GPU, 0 to fall back. */
int    lg_metal_gemm(void *handle, float *y, const float *x, int S);
/* N rows starting at row0 -- for a stacked per-expert weight bank. */
int    lg_metal_gemm_rows(void *handle, float *y, const float *x, int S,
                          int row0, int nrows);

#ifdef __cplusplus
}
#endif
#endif
