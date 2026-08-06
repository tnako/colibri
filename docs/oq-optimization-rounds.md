# Long-context optimization rounds (oQ, Laguna-XS)

Five profile-fix-measure rounds against real oQ checkpoints, driven by
`c/tools/stress_laguna.sh`. Every round used the identical 1902-token prompt
(fixed seed) and 16 generated tokens on an M5 (4P+6E, 32 GB), so the numbers are
comparable across rows.

The harness runs the engine under macOS `sample` at 1 ms and reports SELF time
per symbol. Self time matters: `sample` prints an indented call tree where a
parent's count includes all its children, so naively summing lines makes
`main` look like 100% of the program. The reporter subtracts each frame's
immediate children.

## Results

| round | change | prefill | expert-mm | attn | top symbol |
|---|---|---|---|---|---|
| 0 | baseline (after the NEON/`-mcpu=native` pass) | 301.0 s | 182.5 s | 105.8 s | `__psynch_cvwait` 52.4% |
| 1 | one OpenMP region over (token,expert) pairs | 180.1 s | 63.6 s | 106.0 s | `matmul_oq` 38.8% |
| 2 | compact per-thread accumulator | 173.4 s | 59.4 s | 104.4 s | `matmul_oq` 40.3% |
| 3 | group pairs by expert, batch the GEMM | 173.6 s | 57.9 s | 106.2 s | `matmul_oq` 40.7% |
| 4 | hoist `oq_unpack` out of the batch loop | 146.3 s | 42.3 s | 95.6 s | `matmul_oq` ~65% |
| 5 | NEON `dot_f32` / `axpy_f32` in attention | **136.1 s** | **42.7 s** | **84.9 s** | `matmul_oq` 64.2% |

**Prefill 301 s -> 136 s, 2.2x.** Expert matmul 182.5 s -> 42.7 s, 4.3x.
Attention 105.8 s -> 84.9 s, 1.25x.

Decode on this workload moved 3.87 -> 6.79 tok/s (1.75x), but decode is
cache-hit-dominated at S=1 and varies +/-10% run to run; prefill is the reliable
signal at long context.

## What each round found

**Round 1 - the barrier storm.** 52% of all wall time was `__psynch_cvwait`,
i.e. OpenMP threads waiting rather than working. At prefill the MoE loop runs
S*K = 15,216 times per layer, and each of the three `matmul_w` calls inside was
opening its own parallel region over just I=512 rows. That is roughly 1.8M
barriers per layer, each guarding less work than the barrier costs. Hoisting one
region out to the pair loop cut prefill 40%.

**Round 2 - a memory regression I caused.** Round 1's per-thread accumulator was
`S*D` floats, which at S=1902 with 10 threads pushed RSS from 5.7 GB to 12.7 GB.
Replaced with one row per token the thread actually touched. Speed unchanged,
which is the point: it removed a regression rather than adding a win. (RSS stayed
high for an unrelated reason -- see "not a leak" below.)

**Round 3 - grouping alone did nothing.** Sorting pairs by expert so each expert
gets one batched call looked obviously right and moved prefill by 0.2 s, inside
noise. Worth recording as a negative result: the batching was necessary but not
sufficient, because the kernel it fed still worked row-at-a-time internally.

**Round 4 - the real waste, exposed by round 3.** `matmul_oq` had its group loop
INSIDE its batch loop, so a weight row's codes were unpacked once per token.
With ~59 tokens per expert that is 59x more unpacking than needed. Swapping the
loops (`OQ_MAX_BATCH = 32` rows per unpack) took `oq_unpack` from 28.8% of
runtime to 6.7% and prefill down another 27 s. This is the round that justified
round 3.

**Round 5 - attention was left holding the bag.** Once the MoE path was 4x
faster, attention was the largest phase. Its inner loops were scalar `for d <
hd` dot products and AXPYs with hd=128, called once per (query,key) pair.
NEON versions took attention 95.6 -> 84.9 s and its profile share 12.0% -> 6.9%.

## Verified on a second quant

`mlx-coders/Laguna-XS-2.1-oQ4e` (18 GB, 4-bit default gs64) runs the same
harness: prefill 144.8 s, 4.02 tok/s, RSS 11.8 GB, coherent output. It also
exercises a different code path by accident, which is useful: its tensors use
plain HF naming (`model.layers.N...`) with no `language_model.` prefix, so the
dual-name lookup in `oq_load`/`load_w` is covered by a real checkpoint rather
than only by construction.

## Not a leak: RSS is the expert cache

RSS grows to 12-14 GB at long context because `cap=0` sizes the expert cache
from available RAM (~7.4 GB budget, 161 experts/layer here). That is the cache
doing its job at a 98.5% hit rate. Pass `CAP=<n>` to bound it; `CAP=4` runs the
same model at 9.3 GB.

## Correctness gate, every round

All five rounds kept the tiny fixtures token-exact against the transformers
oracle: XS 24/24 teacher-forced + 12/12 generated, S 208/208 + 8/8, plus the
`cap=1` eviction path and `BITS=8`. The C-vs-MLX unpack check stayed bit-exact.
No round shipped without that passing, which is what makes the speedups
trustworthy rather than merely fast.

## Where the time goes now

`matmul_oq` is 64% of self time and `__psynch_cvwait` is still 19.6%. The
remaining wait is the per-expert `schedule(dynamic,1)` region: expert groups have
very uneven row counts (one hot expert can take 10x the tokens of a cold one), so
threads finish at different times. The next lever is splitting large groups
across threads rather than assigning whole groups, which is a load-balancing
change, not a kernel change.

Reproduce any row:

```
./c/tools/stress_laguna.sh models/Laguna-XS-2.1-oQ2 2048 16 mytag
CAP=4 ./c/tools/stress_laguna.sh models/Laguna-XS-2.1-oQ4e 4096 32 oq4e-long
```
