# Long-context optimization rounds (oQ, Laguna-XS)

Profile-fix-measure rounds against real oQ checkpoints, driven by
`c/tools/stress_laguna.sh`. Every run uses a fixed-seed prompt, so numbers are
comparable within a context size. Hardware: M5 (4P+6E), 32 GB.

The harness runs the engine under macOS `sample` at 1 ms and reports SELF time
per symbol, plus peak RSS and major page faults from `/usr/bin/time -l`. Self
time matters: `sample` prints an indented call tree where a parent's count
includes its children, so summing lines naively makes `main` look like 100%.

## Rounds 1-5 (1902-token prompt, oQ2)

| round | change | prefill | expert-mm | attn | top symbol |
|---|---|---|---|---|---|
| 0 | baseline (after NEON + `-mcpu=native`) | 301.0 s | 182.5 s | 105.8 s | `__psynch_cvwait` 52.4% |
| 1 | one OpenMP region over (token,expert) pairs | 180.1 s | 63.6 s | 106.0 s | `matmul_oq` 38.8% |
| 2 | compact per-thread accumulator | 173.4 s | 59.4 s | 104.4 s | `matmul_oq` 40.3% |
| 3 | group pairs by expert | 173.6 s | 57.9 s | 106.2 s | `matmul_oq` 40.7% |
| 4 | hoist `oq_unpack` out of the batch loop | 146.3 s | 42.3 s | 95.6 s | `oq_unpack` 28.8%→6.7% |
| 5 | NEON `dot_f32` / `axpy_f32` in attention | **136.1 s** | 42.7 s | 84.9 s | attn 12.0%→6.9% |

Prefill 301 -> 136 s (2.2x). Expert matmul 4.3x. Detail on each round is in the
git history; the short version is that round 1 killed a barrier storm (1.8M
barriers per layer, each guarding 512 rows), round 3 alone did nothing until
round 4 fixed the kernel it fed, and round 2 undid a 12.7 GB RSS regression that
round 1 introduced.

## Rounds 6-9 (6144-token prompt)

16K was the original target but a single 16K run takes 35+ minutes at this
prefill speed, so the sweep moved to 6K where a round is ~8 minutes. The
scaling behaviour being optimized (attention going quadratic, expert cache
thrashing) is fully visible at 6K.

### oQ2 (11 GB checkpoint, 2-bit default)

| round | change | prefill | attn | fill (IO) | cache hit | peak RSS |
|---|---|---|---|---|---|---|
| 5 | entering this sweep | 520.8 s | 327.3 s | 24.2 s | 56.9% | 11.81 GiB |
| 6 | flash-style query tiling + online softmax | 490.9 s | 297.7 s | 21.7 s | 56.9% | 11.82 GiB |
| 7 | chunked softmax rescale (`LG_KC = 64`) | 465.8 s | 280.5 s | 21.5 s | 56.9% | 11.07 GiB |
| 8 | visit expert pairs in EXPERT order | **465.1 s** | 279.5 s | **6.0 s** | **99.4%** | **11.05 GiB** |

### oQ8e (33 GB checkpoint, 8-bit everywhere)

| round | change | prefill | attn | decode | peak RSS |
|---|---|---|---|---|---|
| 8 | state entering the oQ8e work | 482.6 s | 288.9 s | 2.51 tok/s | 12.27 GiB |
| 9 | skip the unpack pass for 8-bit codes | **469.6 s** | 281.4 s | **2.80 tok/s** | **10.93 GiB** |

## What each round found

**Round 6 - K/V reuse.** Attention scored one query-head x one query at a time,
re-walking the whole K/V history for each. With `group = H/KV = 6` query heads
per KV head, every K row was read 6 times per query and re-read for all S
queries. Tiling over `LG_QB = 8` queries per KV head loads each row once per
tile. Worth only 1.10x on its own, which pointed at the next problem.

**Round 7 - rescale frequency.** A naive online softmax renormalizes whenever
the running max grows, and each renormalize is O(hd) over the accumulator. Early
in a row the max grows on most keys, so round 6 was paying that O(hd) constantly.
Scoring keys in chunks of `LG_KC = 64` and taking the chunk max first caps it at
one rescale per chunk per query. Attention 297.7 -> 280.5 s and, as a side
effect, peak RSS fell 0.75 GiB because the per-thread score buffer went from
O(context) to O(LG_QB*LG_KC) -- attention scratch no longer grows with prompt
length.

**Round 8 - the IO win.** The pair list is in token order, so a chunk of `cap`
pairs holds up to `cap` DIFFERENT experts; the next chunk evicts them and the one
after reloads what the first had. At 6K/cap=48 that was a 56.9% hit rate against
only 256 distinct experts in the layer. Visiting pairs in expert order (counting
sort, O(npair + E)) means each expert loads at most once per layer per call:
**hit rate 56.9% -> 99.4%, fill 21.5 s -> 6.0 s, 3.6x less disk IO.** Pure
scheduling change; the `cap`-sized eviction safety property is untouched.

**Round 9 - 8-bit needs no unpack.** At `bits == 8` the packed stream is already
a byte array, so `oq_unpack`'s copy into scratch is pure overhead. Pointing
directly at the weight buffer cut peak RSS 12.27 -> 10.93 GiB (-11%) and decode
2.51 -> 2.80 tok/s (1.12x) on oQ8e. Verified bit-exact against
`mlx.core.dequantize` at both gs=64 and gs=128.

## Cumulative result

Against the round-0 baseline at 1902 tokens, prefill went **301.0 s -> 136.1 s
(2.2x)**. Across rounds 6-9 at 6144 tokens, prefill went **520.8 s -> 465.1 s**
with the real wins concentrated in IO (fill 24.2 -> 6.0 s, 4.0x) and memory
(11.81 -> 11.05 GiB on oQ2, 12.27 -> 10.93 GiB on oQ8e).

Three axes, honestly separated:

- **CPU**: 2.2x on prefill in rounds 1-5; rounds 6-9 added ~12% more. The MoE
  matmul is now 64% of self time and is the remaining target.
- **IO**: expert cache hit rate 56.9% -> 99.4%, fill phase 4.0x faster. Major
  page faults stay in single digits, so the streaming path is not touching disk
  beyond the initial expert loads.
- **Memory**: peak RSS down 6-11% depending on quant, and more importantly the
  attention scratch no longer scales with context length.

## Not a leak: RSS is the expert cache

RSS tracks the expert cache, which `cap=0` sizes from available RAM. All 6K runs
above pin `CAP=48` for comparability. `CAP=4` runs the same model at 9.3 GiB.

## Correctness gate, every round

Every round kept the tiny fixtures token-exact against the transformers oracle:
XS 24/24 teacher-forced + 12/12 generated, S 208/208 + 8/8 (that one wraps the
sliding-window ring 50x, which is exactly what a flash-attention rewrite can
break), plus the `cap=1` eviction path and `BITS=8`. The C-vs-MLX unpack check
stayed bit-exact. `task test` 3/3 and the upstream engines still build.

Rounds 6 and 8 are the two that could plausibly have changed numerics -- round 6
reorders the softmax accumulation, round 8 reorders expert visits -- and both
came out token-identical.

## Reproduce

```
CAP=48 ./c/tools/stress_laguna.sh models/Laguna-XS-2.1-oQ2  6144 8 mytag
CAP=48 ./c/tools/stress_laguna.sh models/Laguna-XS-2.1-oQ8e 6144 8 q8
```

## Where the time goes now

`matmul_oq` dominates at ~64% of self time, with `__psynch_cvwait` around 19%.
That residual wait is load imbalance in the per-expert `schedule(dynamic,1)`
region: a hot expert can hold 10x the rows of a cold one, so threads finish
unevenly. Splitting large expert groups across threads is the next lever, and it
is a scheduling change rather than a kernel change.
