# Does the GPU expert work help Laguna-XS too?

Yes, and the memory win is larger than on Laguna-S.

Back-to-back A/B on the same machine, same prompts: `e899eec` (the commit before
the GPU expert kernel) against `3305bbc` (current). Both Metal builds, default
20 GB budget, `CTX_MAX=16384`.

| tokens | prefill | expert-mm | attention | peak RSS |
|---|---|---|---|---|
| 2,000 | 66.9 -> **34.5 s** (1.94x) | 50.2 -> 23.5 s (2.14x) | 11.8 -> 7.4 s | 15.6 -> **5.7 GB** (2.7x) |
| 6,000 | 205.4 -> **71.5 s** (2.87x) | 160.3 -> 45.1 s (3.55x) | 32.6 -> 20.9 s | 15.8 -> **7.6 GB** (2.1x) |
| 12,000 | 415.2 -> **112.4 s** (3.69x) | 327.5 -> 68.1 s (4.81x) | 63.0 -> 22.0 s | 22.7 -> **7.2 GB** (3.2x) |

The speedup grows with context because rows-per-expert is `chunk*topk/E`, and
`LG_CHUNK` is now 4096 instead of 256 -- at 2k tokens the prompt is smaller than
one chunk, so the kernel sees fewer rows than it could. By 12k the expert phase
is 4.8x faster and RSS has stopped growing with context entirely (7.2 GB at 12k
against 7.6 GB at 6k), where the old path was still climbing (22.7 GB at 12k).

## Why memory falls so much further on XS

XS's expert bank is 7.9 GB of 2-bit weights. Previously that was either a 13.7 GB
resident Q8R bank or a multi-GB streaming LRU cache, both charged to the process.
Now it is mmap'd zero-copy and read by the GPU straight from page cache, so it is
evictable and leaves the budget entirely:

```
[mem] gpu experts: 7.9 GB mapped zero-copy in 0.0s (page cache, evictable)
```

Laguna-S sees the same mechanism at 28.4 GB, which is what made it runnable at
all; XS was already runnable, so the benefit shows up as headroom instead.

## Both models via the Taskfile

Verified end to end, no downloads, GPU expert path active in both:

```
task laguna:xs Q=oQ2       NGEN=6 SYNC=0
task laguna:s  Q=oQ2e-fast NGEN=6 SYNC=0
```

| | prefill | peak RSS | output |
|---|---|---|---|
| Laguna-XS (oQ2) | 2.2 s | 6.5 GB | coherent |
| Laguna-S (oQ2e-fast) | 8.8 s | 5.3 GB | coherent |

Both resolve their local `models/` directory by `basename` of the HF repo, so the
already-downloaded checkpoints are reused rather than re-fetched.

Task surface is 3 knobs plus the model selector:

| | |
|---|---|
| `LAGUNA_MEM_GB` / `MEM_GB` | total memory budget, default 20 |
| `CTX_MAX` / `CTX` | max context, default 8192 |
| `LAGUNA_GPU_PROF` | GPU busy vs wall reporting |
| `Q=` | quant variant per model |
| `METAL=0` | force the CPU engines |

`task test` covers both models on both CPU and Metal builds (5 checks).
