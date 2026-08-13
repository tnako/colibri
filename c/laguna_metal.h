/* C interface to the Metal/MPS prefill backend (LAGUNA-FORK).
 * Implementation in laguna_metal.mm. Every entry point is safe to call when
 * Metal is unavailable: init returns 0, upload returns NULL, gemm returns 0 and
 * the caller uses its CPU path. */
#ifndef LAGUNA_METAL_H
#define LAGUNA_METAL_H

#include <stddef.h>

/* Row-tile size the GPU expert kernel groups per threadgroup (laguna_expert_metal.mm's
 * TM, compiled at runtime as Metal Shading Language source -- this macro is NOT
 * visible there, so that literal must be kept in sync with this one by hand).
 * The CPU-side tile-list builder in laguna_common.h chunks experts' row ranges
 * using THIS constant, and it must match the kernel's TM or the (expert, row0)
 * pairs handed to the kernel won't align with what TM expects.
 *
 * TM=128 was tried (larger row-tile amortizes the oQ dequant over more rows)
 * and measured WORSE: 284.6s vs 199.7s prefill wall on Laguna-S at 7370
 * tokens -- fewer threadgroups dispatched cost more than the reduced dequant
 * work saved. See laguna_expert_metal.mm's TM comment. Kept at 64. */
#define LG_EXP_TM 32

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
/* Bind the engine's int8 KV cache directly (zero copy). No GPU-side KV alloc.
 * `ctxcap` is the PHYSICAL per-head row stride (kvphys); `ring` is the logical
 * ring modulus for sliding layers (0 for full/linear layers). */
int    lg_metal_attn_bind(int layers, int layer, int kv, int ctxcap, int hd,
                          const void *kcodes, const void *vcodes,
                          const float *kscale, const float *vscale,
                          size_t code_bytes, size_t scale_bytes, int ring);
void   lg_metal_attn_append(int layer, int pos0, int S, const float *k,
                            const float *vv, int kvdim);
int    lg_metal_attn(int layer, float *ctx_out, const float *q, const float *gt,
                     int S, int pos0, int H, int KV, int hd, float scale, int window);
/* PHASE 7 (LAGUNA-FORK): selective-prefill variant of lg_metal_attn. When
 * `sel` is non-NULL and sel->engaged, the tiled full-layer path (window==0)
 * skips tiles with no selected column and masks non-selected columns in the
 * others, so a LATE full layer (layer > sel->sel_base) scores O(cap) columns
 * instead of the whole prefix. NULL has exactly the behavior of lg_metal_attn. */
typedef struct {
    const int *sel_idx[32];   /* per KV head, sorted absolute positions  */
    int        sel_n[32];     /* per KV head, selected count (<= cap)    */
    int sel_base;             /* scoring (first full) layer index        */
    int engaged;              /* nonzero => use selection on this call   */
} LgAttnSel;
int    lg_metal_attn_sel(int layer, float *ctx_out, const float *q,
                         const float *gt, int S, int pos0, int H, int KV, int hd,
                         float scale, int window, const LgAttnSel *sel);
size_t lg_metal_attn_bytes(int layer);
/* GPU busy vs wall for the attention dispatches (LAGUNA_GPU_PROF=1). */
void   lg_metal_prof_dump(void);

/* ---- routed experts on the GPU, from mmap'd weights ----------------------
 * lg_metal_map wraps an mmap'd region with no copy (verified: c/tools/nocopy.mm).
 * lg_metal_expert runs y[rows,N] = x[row0..,Kd] @ dequant(oQ weight)^T using
 * simdgroup matrix tiles, dequantizing 2-bit codes in-kernel. */
void  *lg_metal_map(const void *base, size_t len);
size_t lg_metal_map_count(void);
int    lg_metal_expert(void *wmap, size_t woff, void *smap, size_t soff,
                       void *bmap, size_t boff, void *xbuf, void *ybuf,
                       int rows, int row0, int Kd, int N, int gs, int bits);
/* All experts in one dispatch (grouped GEMM), using a compacted tile list so an
 * imbalanced router does not pay for the largest expert's row count everywhere.
 * offs = E+1 row starts; tiles = ntiles x (expert, row-tile-origin) pairs. */
int    lg_metal_expert_grouped(void *wmap, size_t woff, void *smap, size_t soff,
                    void *bmap, size_t boff, void *xbuf, void *ybuf, void *offbuf,
                    void *tilesbuf, int ntiles, int Kd, int N, int gs, int bits,
                    size_t wslab, size_t sslab);
/* One MoE layer (gate, up, silu, down) in a single command buffer -- see the
 * comment in laguna_expert_metal.mm for why this replaced 3 separate ones. */
int    lg_metal_moe_layer(
        void *gwmap, size_t gwoff, void *gsmap, size_t gsoff, void *gbmap, size_t gboff,
        void *uwmap, size_t uwoff, void *usmap, size_t usoff, void *ubmap, size_t uboff,
        void *dwmap, size_t dwoff, void *dsmap, size_t dsoff, void *dbmap, size_t dboff,
        void *xbuf, void *gbuf, void *ubuf, void *ybuf, void *offbuf, void *tilesbuf,
        int ntiles, int npair, int D, int I, int gs, int bits,
        size_t wslabGU, size_t sslabGU, size_t wslabD, size_t sslabD);
/* shared GPU scratch for the expert path */
void  *lg_metal_scratch(int which, size_t bytes);
void  *lg_metal_scratch_ptr(void *h);
/* GPU busy vs wall time for the attention dispatches (LAGUNA_GPU_PROF=1). */
void   lg_metal_prof_dump(void);
void   lg_metal_expert_prof_dump(void);

/* ---- persistent decode GEMV path (LAGUNA-FORK, Phase 2) ---------------------
 * Decode is a per-token memory-bound GEMV stream over the resident oQ2/Q8R
 * weights (attention projections + shared expert + routed experts), so running
 * it through the prefill GEMM path (gated S>=64) or the f16 MPS uploads is
 * wrong twice over. This is a small batch of GEMVs per layer, all encoded into
 * ONE command buffer so the CPU pays a single commit+wait per layer instead of
 * one round trip per matrix (measured ~0.33 ms each on M5 -- at 40 layers that
 * is the entire 140 tok/s budget, which is why per-matrix is banned).
 *
 * The session owns nothing but the pipeline and a cache of host-memory
 * MTLBuffer wraps. The C engine declares persistent page-aligned output regions
 * (lg_decode_region), then per layer begins a session, enqueues one GEMV per
 * matrix with lg_decode_gemv, and commits with lg_decode_run. Every entry point
 * returns 0 / does nothing when Metal is unavailable so the CPU path is the
 * floor; a failed gemv is reported so the caller can fall back per matrix. */
typedef struct LgDecode LgDecode;
LgDecode   *lg_decode_new(void);
void        lg_decode_free(LgDecode *d);
/* Declare a persistent output region hosted at page-aligned `host` (bytes).
 * Returns a small non-negative slot id, or -1. The GPU kernel writes into the
 * region by slot; the CPU reads `host` directly after the commit. */
int         lg_decode_region(LgDecode *d, float *host, size_t bytes);
/* Start a layer's session for S rows. CPU may write x / read stale regions
 * until lg_decode_run(). */
void        lg_decode_begin(LgDecode *d, int S);
/* One matrix: y[ry + rOff*4 bytes, S*N] = x[S,K] @ dequant(W)^T.
 *   W/Sc/Bi are HOST base pointers (wrapped and cached); woff/soff/boff are
 *   byte offsets into them (expert slabs). For dense weights pass Sc=Bi=NULL,
 *   fmt=LG_DEC_F32 / LG_DEC_BF16 and bits=0. bit-exactness matches matmul_oq
 *   (affine group factorisation: sc*dot + bi*xsum). Returns 1 if enqueued,
 *   0 if the matrix shape/dtype is not representable (caller runs CPU). */
enum {
    LG_DEC_OQF32  = 0,   /* oQ codes, f32 scales/biases (Wt from oq_load) */
    LG_DEC_OQBF16 = 1,   /* oQ codes, bf16 scales/biases (mmap'd gx slabs) */
    LG_DEC_F32    = 2,   /* dense f32 weights */
    LG_DEC_BF16   = 3,   /* dense bf16 weights */
};
int         lg_decode_gemv(LgDecode *d, int ry, size_t rOff,
                           const float *x,
                           const void *W, size_t woff,
                           const void *Sc, size_t soff, const void *Bi, size_t boff,
                           int N, int K, int bits, int gs, int fmt);
/* Enqueue an elementwise gate*up -> (silu) fusion in place on gb (S*I floats). */
int         lg_decode_silu(LgDecode *d, float *gb, size_t gboff, size_t n);
/* Commit the session's command buffer and wait. Returns 1 on success. */
int         lg_decode_run(LgDecode *d);
/* Per-region host pointer (for the engine to address outputs after a run). */
float      *lg_decode_region_ptr(LgDecode *d, int ry);
/* Total bytes declared; decode is zero-cost until the engine declares regions. */
size_t      lg_decode_bytes(LgDecode *d);
int         lg_decode_active(LgDecode *d);

#ifdef __cplusplus
}
#endif
#endif
