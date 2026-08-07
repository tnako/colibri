# 256k on the GPU inside 20 GB: bind the KV cache, don't copy it

## The blocker

GPU attention kept its own f16 copy of K/V, sized to `CTX_MAX`. On Laguna-S at
262144 context that was **14.24 GB** -- more than half a 20 GB budget -- so the
engine declined the allocation and every full-attention layer fell back to the
CPU. 256k "fit in memory" only by not using the GPU.

## What was actually done

The plan was per-tile upload: stream K/V to the GPU in tiles so the allocation is
O(tile). Once written out, a better option was obvious. The engine **already**
maintains an int8 KV cache with one f32 scale per row (`kv_i8.h`). Uploading tiles
of a *second* copy is wasted work when unified memory lets the GPU read the first
one.

So the cache itself is bound with `newBufferWithBytesNoCopy`, the same mechanism
already proven for the expert weights. It required only that the four KV buffers
be page-aligned (`kv_aligned`, 16 KB) instead of plain `malloc`. The kernel gained
a `deq_kv` shader that dequantizes `code * scale` into an f16 tile of just the band
being scored, which MPS then multiplies as before.

| | before | after |
|---|---|---|
| GPU K/V allocation @262144 | 14.24 GB | **0.02 GB** (band staging only) |
| scales with context | yes | **no** |
| per-chunk upload | full f16 append | **none** |
| copies of KV | 2 (int8 CPU + f16 GPU) | **1** |

The staging buffer is `KV * band * hd * 2 * 2` bytes -- a function of window and
chunk, not of context.

## Ordering: a real correctness trap

The CPU appends this chunk's K/V *after* scoring, which is safe because the CPU
kernels read `k`/`vv` directly for the current chunk. The GPU now reads the cache
instead, so the append had to move **before** the GPU block or the current chunk's
own keys would be missing. The sliding-layer skip logic moved with it unchanged.

Caught by reasoning about the read/write order rather than by a failing fixture,
because the tiny fixtures are single-chunk and would not have exposed it.

## Result

Laguna-S 2.1 oQ2e-fast, `CTX_MAX=262144`, default 20 GB budget:

```
[mem] gpu attention 0.02 GB staging (12 full + 36 sliding, ctx 262144, KV bound in place)
[mem] gpu experts: 28.4 GB mapped zero-copy in 0.1s (page cache, evictable)
[mem] gpu projections 4.43 GB (reserved 4.43)
resident weights loaded in 0.9s | RSS 7.24 GB
```

**256k context configures and runs on the GPU at 7.24 GB.** Both of the model's
large weight sets -- 28.4 GB of experts and the KV cache -- are now read in place
rather than copied.

A 1433-token prompt at that setting: prefill 97.8 s, expert-mm 48.7 s,
attention 41.6 s, peak RSS 8.2 GB.

## Honest cost

The same prompt at `CTX_MAX=4096` runs in 74.5 s with attention at 22.2 s. Raising
`CTX_MAX` to 262144 costs ~23 s even though the prompt is identical.

The cause is layout, not extra work. The cache is `[KV][ctxcap][hd]`, so a larger
`ctxcap` scatters each head's band across a much wider address range -- same
number of bytes read, far worse locality. Hoisting the dequant out of the
query-head loop (it was redundantly dequantizing each KV head `group`=6 times)
recovered only 43.4 -> 41.6 s, which confirms the cost is memory locality rather
than dequant volume.

Fixing it properly means making the KV cache head-minor (`[ctxcap][KV][hd]`) or
tiling the dequant by page. Not done.

## Correctness

XS 24/24 + 12/12, S 208/208 + 8/8, on both CPU and Metal builds. `task test` 5/5,
`task test:baseline` clean, 0 build warnings. The int8 cache is now the single
source of truth for KV, so the CPU and GPU paths cannot drift apart.
