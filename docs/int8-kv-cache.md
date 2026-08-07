# int8 KV cache, and what 256k actually costs

## The change

`c/kv_i8.h`. The KV cache was f32; it is now symmetric per-row int8 — one f32
scale per (layer, kv_head, position) row of `head_dim` values:

```
s = max|x| / 127     q[i] = round(x[i]/s)     x'[i] = q[i]*s
```

Per-row rather than per-tensor because attention row magnitudes vary a lot across
positions, and the scale costs 4 bytes per 128 values (3% overhead) while keeping
error local. Symmetric rather than affine because K and V are near zero-centred,
so a zero point would cost another 4 bytes per row for nothing.

Read path: each score chunk (`LG_KC` rows, 32 KB at hd=128) is dequantized into a
staging buffer once per tile, so every kernel below stays f32 and untouched, and
the dequant is amortized over all `LG_QB` queries in the tile instead of repeated
per query.

**Measured 3.88x smaller** in every configuration:

| model | context | f32 | int8 |
|---|---|---|---|
| Laguna-XS | 6144 | 0.59 GiB | 0.15 GiB |
| Laguna-XS | 262144 | 20.12 GiB | **5.19 GiB** |
| Laguna-S | 262144 | 24.14 GiB | **6.22 GiB** |

Cost: attention +5% at 2k, +11% at 6k, from the per-chunk dequant. That buys 3.88x
on the structure that blocks long context, so it is taken unconditionally rather
than made a flag.

## Measured context scaling on Laguna-XS

One command, no environment variables beyond `SNAP`:

| tokens | prefill | peak RSS | s per 1k tokens |
|---|---|---|---|
| 2,000 | 37.4 s | 12.2 GB | 18.7 |
| 6,000 | 123.0 s | 17.5 GB | 20.5 |
| **29,967** | **1110.4 s** | **17.6 GB** | 37.1 |

**Peak RSS is flat from 6k to 30k** — 17.5 vs 17.6 GB across a 5x context
increase, and 12.2 GB at 2k. Combined with chunked prefill making scratch O(chunk)
rather than O(context), memory no longer scales with context length. That is the
property 256k needs, and it is now measured rather than projected.

## The 256k verdict, honestly

**Memory: yes.** With int8 KV the Laguna-S dirty footprint at 262144 tokens is:

| | |
|---|---|
| KV cache (int8) | 6.22 GiB |
| expert working set (topk10, 2-bit) | 1.05 GiB |
| resident projections + embed | 0.64 GiB |
| chunked scratch | 0.03 GiB |
| **total dirty** | **7.94 GiB** |
| budget | 20.00 GiB |

leaving ~12 GiB for evictable mmap'd weight pages. The 27.6 GiB of Laguna-S
weights cannot be resident, but they do not need to be: page-cached mmap measured
126.7 GB/s here, identical to RAM, and the OS reclaims those pages under pressure
instead of the process dying.

**Time: no.** Attention was 839.8 s of the 1110 s at 30k — 76%, and it is O(S²) on
the 10 full-attention layers. Scaling that to 262144 tokens is roughly 72x the
attention work, i.e. **~18 hours**. 256k fits in the memory budget on this
hardware; it does not finish in a useful time.

So this is a memory result, not a "256k works" result. Making 256k time-feasible
needs the attention *scores* on the GPU (the projections are already there), which
is the same item the roofline analysis flagged: 3.87 TFLOP of scores at 6k runs at
~14 GFLOP/s on the CPU against 15572 GFLOP/s measured for MPS f16.

## Correctness

Token-exact on every fixture with int8 KV active, on both CPU and Metal builds:
XS 24/24 teacher-forced + 12/12 generated, S 208/208 + 8/8 (that fixture wraps the
sliding ring 50 times, which is exactly what a KV format change can break), the
`cap=1` eviction path, `task test` 3/3, `task test:baseline` clean.

Quantizing the KV cache to int8 changing no output token was the result I least
expected, and it is why the per-row scale is worth its 3% overhead.
