# One knob, chunked prefill, batched GPU dispatch

Target set by the user: **Laguna-S at 256k context in at most 20 GB**, one good
configuration rather than a pile of environment variables.

## The honest answer on Laguna-S at 256k in 20 GB

It does not fit, and no amount of optimization makes it fit. Real config, fetched
from the checkpoint (`48` layers, `D=3072`, `moe_inter=1024`, `256` experts,
`topk=10`, 12 full / 36 sliding@512):

| component | 2-bit oQ |
|---|---|
| routed experts | **27.00 GiB** |
| attention projections | 0.49 GiB |
| embed + lm_head | 0.14 GiB |
| **weights total** | **27.64 GiB** |
| KV cache @256k, f16 | 12.07 GiB |
| KV cache @256k, int8 | 6.04 GiB |

The weights alone exceed 20 GB at the smallest quantization that exists. With a
zero-byte KV cache and zero scratch it still does not fit.

What *does* work is the streaming path, which is why it stays the default: weights
live in the mmap'd checkpoint (measured at 126.7 GB/s when page-cached, identical
to RAM) and the OS evicts pages under pressure instead of the process dying. The
dirty, non-evictable requirement at 256k is:

| | |
|---|---|
| KV cache (int8) | 6.04 GiB |
| expert working set (topk10, 2-bit) | 1.05 GiB |
| resident projections + embed | 0.64 GiB |
| scratch (chunked, context-independent) | ~0.01 GiB |
| **total dirty** | **7.73 GiB** |

So 256k in 20 GB is reachable *as a streaming configuration*, with the remaining
~12 GB as evictable page cache. Two of the three prerequisites are now done
(chunked prefill, budget-driven cache); **int8 KV is not** — the cache is still
f32, which is 24 GiB at 256k for S and the remaining blocker.

Laguna-XS at 256k fits comfortably: 7.68 GiB of weights plus 10.06 GiB of f16 KV.

## One knob

`LAGUNA_MEM_GB` (default 20) is now the entire tuning surface. It replaced four
separate variables (`LAGUNA_RESIDENT`, `LAGUNA_GPU_BUDGET_GB`,
`LAGUNA_GPU_EXPERT_GB`, `LG_METAL_MIN`) and drives, in priority order:

1. **GPU attention projections** — cheapest win per byte (4.6x on attention), capped at 1/6 of the budget
2. **Resident Q8R expert bank** — if the whole bank fits in what is left; removes all expert IO
3. **GPU f16 expert bank** — most memory-hungry, takes the remainder
4. **Streaming cache size** — when the bank does not fit

Measured on XS at ~2000 tokens, one command, no other variables:

| `LAGUNA_MEM_GB` | prefill | peak RSS |
|---|---|---|
| 8 | 128.5 s | **11.1 GB** |
| 14 | 92.7 s | 11.3 GB |
| 20 | **31.4 s** | 15.5 GB |

A bug surfaced while checking this: the streaming cache sized itself from *free
RAM* and ignored the budget, so `LAGUNA_MEM_GB=8` still peaked near 17 GB. The
knob looked like it worked (prefill changed) while not actually capping memory.
Now every path reads the same budget.

## Chunked prefill

`step()` splits any prompt into `LG_CHUNK`-sized calls to `step_raw()`. The KV
cache holds full history, so each chunk attends to everything before it and
results are bit-identical — the 208-position teacher-forced S fixture passes
unchanged.

This is what makes long context possible at all: previously `q` and `ctx` were
each `S*heads*head_dim*4`, which at 256k is 6 GiB *each* for Laguna-S. Now every
scratch buffer is O(chunk), so **peak memory no longer depends on context
length**. Verified: 18.5 GB at 2k versus 17.6 GB at 6k, i.e. flat across 3x.

`LG_CHUNK` was swept with `c/tools/tune_chunk.sh`:

| chunk | prefill | peak RSS |
|---|---|---|
| 256 | 29.7 s | **16.5 GB** |
| 512 | 32.5 s | 18.8 GB |
| 1024 | 29.8 s | 19.1 GB |
| 2048 | 30.2 s | 18.6 GB |
| 4096 | 30.2 s | 20.2 GB |

Prefill is flat (noise) and memory is not, so **256** is the default: the smallest
chunk that costs no speed.

## Batched GPU expert dispatch

The previous attempt called Metal from inside `#pragma omp parallel` and threads
raced on the shared MPS scratch and wrapper cache, tripping `"Number of requested
rows in result exceeds result matrix size"`. It shipped inert behind a guard.

Now the whole layer's expert GEMMs are issued from **one thread**, which removes
the race by construction and costs nothing — the GPU is the parallel device, so
the CPU only gathers rows and enqueues. Falls back to the CPU UDOT path if any
dispatch declines, zeroing the output first so partial work cannot double-count.

This also repaired a regression chunking introduced. At `LG_CHUNK=256` each expert
sees ~40 rows instead of ~240, and the CPU kernel lost its batching:
**expert-mm 15.1 s -> 71.2 s**. A 40-row GEMM against a resident f16 weight does
not care, and with the GPU dispatch active prefill went **89.6 s -> 31.4 s**.

## Result on XS at ~2000 tokens

| | before today | now |
|---|---|---|
| prefill | 89.2 s | **31.4 s** |
| peak RSS | 18.9 GB | 15.5 GB (11.1 GB at `LAGUNA_MEM_GB=8`) |
| scratch scaling | O(context) | **O(chunk)** |

## Correctness

XS 24/24 teacher-forced + 12/12 generated, S 208/208 + 8/8, the `cap=1` eviction
path, `task test` 3/3, `task test:baseline` clean, on both the CPU and Metal
builds. Chunking and the GPU dispatch are both invisible to the fixtures.

## Still open

- **int8 KV cache** — the last prerequisite for S at 256k. f32 today.
- **Laguna-S has never been run end to end here**; the 256k analysis above is
  arithmetic from the real config plus measured per-phase rates, not a run. The S
  fixtures pass, so the code path is exercised, but no full-size S checkpoint has
  been downloaded (it is ~235 GB at bf16, 27 GiB at oQ2).
