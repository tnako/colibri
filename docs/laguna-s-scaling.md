# Banded sliding attention, ring KV, and Laguna-S at scale

## The banded kernel

A sliding layer's queries in one chunk span `[pos0, pos0+S)`, so the only keys any
of them can reach are `[pos0-window+1, pos0+S)`. That band is `window+chunk-1`
columns — **767, constant in context** — while the dense path computed all
`pos0+S` columns and masked the rest away.

| context | dense columns | banded | waste avoided |
|---|---|---|---|
| 2,000 | 2000 | 767 | 2.6x |
| 6,000 | 6000 | 767 | 7.8x |
| 30,000 | 30000 | 767 | 39x |
| 262,144 | 262144 | 767 | **342x** |

This is what made sliding layers viable on the GPU. The dense attempt regressed
6k from 32.4 to 53.1 s; banded takes it to **21.8 s**.

Measured on Laguna-XS at 12k: **attention 74.4 -> 42.7 s, prefill 197.8 -> 139.4 s.**

Gated at `pos0+S >= 4*window`: below that the band is most of the row anyway and
the CPU wins (at 2k, CPU-sliding 9.8 s vs all-GPU 13.0 s).

## Double-mapped ring KV

Sliding layers need only `window+chunk` rows of GPU K/V, but a plain ring splits a
band at the wrap and the GEMM needs one contiguous range. Every row is therefore
written at both `r` and `r+ring` in a buffer of `2*ring` rows, so any
window-length span starting anywhere is contiguous.

GPU attention K/V for Laguna-S, 48 layers:

| ctx | linear (all layers) | ring (measured) |
|---|---|---|
| 4,096 | 0.75 GB | **0.43 GB** |
| 65,536 | 12.00 GB | **3.45 GB** |
| 262,144 | 48.00 GB | **13.11 GB** |

Without this, 250k context needed 45.8 GiB of GPU K/V and was simply impossible.

## Laguna-S 2.1 measured

`mlx-community/Laguna-S-2.1-oQ2e-fast`, 32 GB on disk, 48 layers, D=3072,
256 experts, topk 10, 12 full / 36 sliding. Default 20 GB budget, one command:

| ctx | tokens | prefill | attn | expert-mm | peak RSS |
|---|---|---|---|---|---|
| 100 | 112 | 20.6 s | 3.7 s | 13.6 s | 11.8 GB |
| 200 | 195 | 38.3 s | 5.8 s | 27.3 s | 13.3 GB |
| 400 | 378 | 77.4 s | 11.6 s | 54.0 s | 13.5 GB |
| 800 | 733 | 159.2 s | 25.2 s | 112.6 s | 13.9 GB |
| 1,600 | 1433 | 322.7 s | 56.4 s | 217.6 s | 12.7 GB |
| 3,200 | 2845 | 615.0 s | 91.0 s | 438.7 s | 13.6 GB |

**Memory scales: it does not.** 11.8 -> 13.9 GB across a 25x context range, inside
the 20 GB budget throughout. That is the chunked prefill plus int8 KV plus the
ring working as intended.

**Time scales linearly in this range**, 194-225 ms/token, because expert-mm is
70% of prefill and is linear in tokens. Attention's quadratic term has not bitten
yet at 3k.

## Why Laguna-S experts cannot be resident

The Q8R bank is **39 GB**, against 32 GB on disk:

| | |
|---|---|
| packed 2-bit codes | 26.44 GiB |
| per-group f32 scale + bias + rsum | **9.91 GiB** |
| total | 36.35 GiB |

The metadata is 47% of the codes, because gs=128 means 12 bytes of f32 metadata
per 32 bytes of weight. Even storing all three as bf16 gives 31.4 GiB, still over
a 20 GB budget. So Laguna-S streams its experts, at a measured 90.3% cache hit,
and that is the correct configuration rather than a failure.

## 250,000-token estimate

Per-phase extrapolation from the measured rates (not a curve fit through short
prompts, which extrapolates the quadratic term badly):

| phase | scaling | at 250k |
|---|---|---|
| expert-mm | linear | 10.7 h |
| disk fill | linear | 1.6 h |
| attention, sliding | linear | 1.2 h |
| **attention, full (12 layers)** | **quadratic** | **93.9 h** |
| **total** | | **~107 h** |

Memory at 250k:

| | |
|---|---|
| CPU KV cache (int8) | 5.94 GiB |
| GPU attention K/V (ring) | 13.11 GiB |
| projections + embed | ~3.3 GiB |
| chunked scratch | ~0.03 GiB |
| experts | streaming, evictable page cache |

**Memory: fits.** **Time: does not.** ~107 hours, dominated by the 12
full-attention layers at O(S²).

And there is a second bind: at 250k the 13.11 GB of GPU K/V is more than half a
20 GB budget, so the engine refuses it and those layers fall back to the CPU,
which makes the quadratic term worse still. Using the GPU at 250k needs
`LAGUNA_MEM_GB` around 30, which a 32 GB machine cannot give without swapping.

The honest summary: **Laguna-S at 250k is a memory success and a latency
failure.** Every structure now scales, and the remaining wall is 12 layers of
genuine O(S²) attention on hardware that gives ~27 GFLOP/s on that shape.

What would actually move it: chunked/blockwise softmax over the full layers so
they stream K in tiles (bounded memory, GPU-resident, no 13 GB allocation), which
is the FlashAttention-2 structure rather than the per-head GEMM used here.

## Correctness

XS 24/24 + 12/12, S 208/208 + 8/8 on CPU and Metal builds, `task test` 5/5,
`task test:baseline` clean, 0 warnings.
