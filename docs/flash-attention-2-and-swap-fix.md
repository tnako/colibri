# The swap bug, and FlashAttention-2

## The swap bug was real

You saw ~10 GB of swap with nothing else to blame. It was mine, and it was a
budget over-commit.

The order of decisions was wrong. The streaming expert cache was sized **first**,
from the *full* `LAGUNA_MEM_GB`, and only afterwards did GPU attention, the GPU
projections and the CPU KV cache take their share. The same 20 GB was handed out
three or four times over. On Laguna-S:

```
[cap auto]  58 experts/layer (8.0 GB cache budget)   <- took 8 GB of 20
[metal] flash attention ... (0.33 GB)                <- then this
[mem] expert bank 39.0 GB > 13.2 GB free budget      <- "13.2 free" was fiction
[metal] Apple M5: 3.33 GB projections on GPU         <- and this
```

Two compounding mistakes:

1. **Order.** Every consumer must be reserved before the cache is sized, because
   the cache is the one that can shrink.
2. **Unified memory.** I had been treating GPU buffers as if they were separate.
   On Apple silicon a `MTLResourceStorageModeShared` buffer is ordinary RAM, so
   "reserving it for the GPU" does not make it free. The reserve also omitted the
   shared expert, which on Laguna-S (`dense=12288`) is larger than the attention
   projections it did count.

Fixed: `mem_used` accumulates KV cache, then GPU attention, then an exact
projection reserve (including the shared expert), and the cache gets 75% of what
genuinely remains. The loader settles up with actual spend afterwards.

Measured on the same 800-token Laguna-S run, `sudo purge` before each:

| | swap growth | peak RSS |
|---|---|---|
| before | **+4915 MB** | 13.5 GB |
| after | **+119 MB** | 12.2 GB |

Reserve accuracy now: reserved 6.49 GB, actually spent 6.47 GB.

## FlashAttention-2: implemented, token-exact, and not faster

`flash_attn2` in `c/laguna_attn_metal.mm`, selected with `LG_FA2=1`. One
threadgroup per (query, head); keys stream in tiles of 64 staged into threadgroup
memory; running `(max, denom, accum)` with one `exp(m-m')` correction per tile
rather than per key, which is the FA2 formulation.

Back-to-back on Laguna-XS, same prompts:

| tokens | GEMM attn | GPU busy | FA2 attn | GPU busy | speedup |
|---|---|---|---|---|---|
| 2,000 | 12.9 s | 0.76 s | **11.8 s** | 0.77 s | 1.09x |
| 6,000 | 32.0 s | 6.43 s | **30.2 s** | 6.45 s | 1.06x |
| 12,000 | **62.6 s** | 16.11 s | 67.4 s | 18.74 s | 0.93x |

**Read these as a ratio, not as absolutes.** This batch ran under heavy CPU
contention (a browser at 85%, later a background shortcut runner at 66%), so its
prefill times are ~2.9x the clean figures elsewhere in these docs -- 72.5 s at 2k
here against 25.1 s clean, 428.7 s at 12k against 139.4 s. The comparison is still
sound because both arms ran back-to-back under identical load, and the GPU-busy
column is GPU-side and barely moves with CPU load. But do not compare an attn
number from this table against a clean one from another document and conclude
there was a regression.

**A wash, trending worse with context.** GPU busy time is slightly *higher* for
FA2 at 12k (18.74 vs 16.11 s), so it is doing more work, not less.

Why: the kernel is scalar. One threadgroup of 64 threads per (query, head), where
only `hd` lanes carry accumulator state and the rest merely help stage tiles, and
each lane recomputes dot products instead of sharing them through simdgroup matrix
ops. MPS's GEMM drives the matrix units. This is the third time on this project
that a hand-written kernel lost to MPS:

| attempt | mine | MPS |
|---|---|---|
| oQ2 tiled GEMM | 144.6 GOP/s | 191.7 (naive) |
| first attention kernel | 81 GFLOP/s | 480 (GEMM path) |
| FA2 streaming | 0.93-1.09x | baseline |

It is kept behind `LG_FA2=1`, off by default, because it does buy one real thing.

## What FA2 actually buys

It never materializes the score matrix:

| ctx | score matrix (GEMM) | FA2 tile memory |
|---|---|---|
| 65,536 | 33.6 MB per head per chunk | 32 KB per threadgroup |
| 262,144 | **134.2 MB** per head per chunk | 32 KB per threadgroup |

That removes an allocation ceiling rather than time.

## Where this leaves 250k, honestly

**FA2 does not unlock 250k, and I want to be exact about why.** The blocker is the
GPU K/V cache, not the scores:

| ctx | GPU K/V (both paths) |
|---|---|
| 65,536 | 3.45 GB |
| 262,144 | **13.11 GB** |

13.11 GB is more than half a 20 GB budget, so the engine declines it and the full
layers fall back to the CPU. FA2 shrinks the scores to nothing but reads the same
K/V, so it changes neither the allocation nor the verdict.

Sliding layers already have bounded K/V via the double-mapped ring. Full-attention
layers genuinely read every previous position, so the only way to bound *their* GPU
K/V is to upload K/V per tile as it is scored, instead of keeping the whole cache
resident. That is the remaining piece of real FA2 structure and it is not done —
it would make GPU K/V O(tile) and let 250k run on the GPU inside 20 GB.

## Correctness

Token-exact on both paths and both models: XS 24/24 + 12/12, S 208/208 + 8/8,
`task test` 5/5, 0 build warnings. `LG_FA2=1` and the default GEMM path produce
identical tokens.
