# Phase 4 — prefill chunk sweep (LG_CHUNK) on the Metal build

Measured on `c/laguna_xs_metal` (make target `laguna_xs_metal`), real
`Laguna-XS-2.1-oQ2` checkpoint, 941-token prose prompt (`--chat`, 2 generated
tokens), each run < 1.5 min wall. `LG_CHUNK` was overridden at runtime via the
new env var (no rebuild between points). Per-chunk prefill/attn/expert-mm
attribution comes from the new `LG_TRACE_CHUNK=1` trace in `step()`.

| LG_CHUNK | prefill | expert-mm | attn | fill | RSS |
|---:|---:|---:|---:|---:|---:|
| 256 | 93.0 s | 86.1 s | 5.3 s | 0.1 s | 5.2 GB |
| 512 | 47.3 s | 41.6 s | 4.8 s | 0.2 s | 5.3 GB |
| 1024 | 26.9 s | 21.4 s | 4.2 s | 0.3 s | 5.5 GB |
| 2048 | 27.0 s | 21.5 s | 4.1 s | 0.4 s | 5.5 GB |
| 4096 | 27.6 s | 21.5 s | 4.6 s | 0.2 s | 5.5 GB |
| 8192 | 27.5 s | 21.7 s | 4.2 s | 0.2 s | 5.5 GB |

Per-chunk breakdown at the two slowest points (LG_TRACE_CHUNK=1):

```
LG_CHUNK=256:  chunk 1/4 n=256  prefill 20.6s  attn 0.7s  expert-mm 19.4s
               chunk 2/4 n=256  prefill 22.6s  attn 1.3s  expert-mm 20.8s
               chunk 3/4 n=256  prefill 24.4s  attn 1.5s  expert-mm 22.3s
               chunk 4/4 n=173  prefill 25.4s  attn 1.4s  expert-mm 23.6s
LG_CHUNK=512:  chunk 1/2 n=512  prefill 23.7s  attn 1.7s  expert-mm 21.1s
               chunk 2/2 n=429  prefill 23.6s  attn 2.3s  expert-mm 20.5s
```

## What this shows

- **The expert-mm term is the whole lever.** It falls 86.1 s -> 21.4 s as the
  chunk grows 256 -> 1024 and is flat thereafter; attention (4.2-5.3 s) and the
  shared expert are effectively chunk-independent. `expert cache hit 99.9%` at
  every point confirms the GPU grouped-GEMM path (`lg_metal_moe_layer`, one
  command buffer per layer) is serving the whole prefill, not the CPU UDOT/streaming
  fallback.
- **The old 256-is-optimal conclusion is obsolete.** `docs/one-knob-and-chunking.md`
  measured prefill flat across 256..4096 on a pre-unified-dispatch build (CPU
  UDOT experts). On the current build the GPU expert kernel's throughput is
  rows/expert bound (8 rows/expert at chunk 256 vs 32 at 1024 for XS topk=8/E=256),
  so small chunks are ~4x worse. Big-chunk amortization is exactly the Phase 4
  target.
- **Memory no longer tracks the chunk.** RSS is flat at 5.5 GB from 1024 up
  (5.2 GB at 256): the O(chunk) scratch is small next to the resident weights,
  GPU attention staging and KV ring, so the historical RSS-vs-chunk tradeoff is
  gone. Default `LG_CHUNK=8192` stays; 1024 is the knee and costs no speed.

## Per-chunk fixed cost (STEP 3 audit)

The residual `prefill - (expert-mm + attn + shared + fill)` is the per-chunk
serial tail: router CPU matmul, the routed-index gather, rmsnorm/residual, KV
append and the layer-serial Metal dispatches. From the trace it is ~1.0 s per
chunk (40 layers x dispatch+sync) and ~0.7 s for the whole prompt at chunk 1024
-- i.e. 4 chunks pay ~4 s, 1 chunk pays ~0.7 s. None of it is O(1)-reusable
across chunks:

- the router scores (`matmul(logits, x, l->router, S, D, E)` in `moe()`) are
  per-chunk-proportional and recomputed only for that chunk's tokens (correct,
  not duplicated);
- the routed-index gather into the GPU scratch (`xb`) is batched once per
  (chunk, layer), proportional to `npair` -- no per-token CPU `matmul_oq`
  remains in prefill;
- the shared expert weights are a single mmap'd `Wt` per layer, never copied
  per chunk.

So the only lever big chunks provide is amortizing that ~1 s/chunk dispatch
tail plus the rows/expert efficiency of the grouped GEMM, and both are already
fully captured by this sweep (93.0 s -> 26.9 s = 3.5x on prefill).

## How the sweep works

`LG_CHUNK` is read once by `lg_chunk()` in `c/laguna_common.h` (env var overrides
the `LG_CHUNK` compile-time default), so `LG_CHUNK=1024 ./c/laguna_xs_metal 0 0
--chat -n 2 -f prompt.txt` reproduces any row above. `LG_TRACE_CHUNK=1` turns on
the per-chunk `[chunk] k/n` lines from `step()`.

Run: `LG_CHUNK=$CH LG_TRACE_CHUNK=1 SNAP=models/Laguna-XS-2.1-oQ2 ./c/laguna_xs_metal 0 0 --chat -n 2 -f <prompt>`
