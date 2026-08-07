# Budget priority, GPU profiling, and two things removed

## Budget priority fixed

GPU flash attention is now reserved **before** the resident expert bank, capped at
half the budget. Previously the bank claimed memory first, so a 20 GB run — the
stated target — silently fell back to CPU attention, the slower configuration.

| `LAGUNA_MEM_GB` | GPU attention | prefill @2k |
|---|---|---|
| 20 (default) | **yes, 10 layers** | 35.5 s |
| 22 | yes, 10 layers | 29.9 s |

Priority is measured, not assumed: GPU attention is 3.66x on the largest phase,
the resident bank ~1.5x on a smaller one.

## GPU profiling without Xcode

Only CommandLineTools is installed here, so `xctrace` and Instruments are
unavailable. `cb.GPUStartTime` / `GPUEndTime` are the same timestamps Instruments
reports, so reading them in-process gives the same answer:

```
LAGUNA_GPU_PROF=1 ./c/laguna_xs_metal ...
[gpuprof] attn: 240 dispatches, GPU busy 3.16s, wall 3.42s (92% busy)
```

That immediately settled where the time goes, and contradicted what I expected:

| 6k prompt | |
|---|---|
| attention phase | 33.3 s |
| GPU dispatch (wall) | **3.42 s** |
| GPU busy | 3.16 s (92% occupancy) |
| **CPU-side remainder** | **~30 s** |

The GPU part is already efficient at 92% occupancy. **90% of the attention phase
is the 30 sliding-window layers still on the CPU** — they are 3x as numerous as
the full layers, and at 2k they carry more total FLOPs (755 vs 491 GFLOP).

## Two negative results, both removed rather than kept behind a flag

### Batched per-head GEMMs: slower

The theory was that 144 command encoders per layer per chunk (48 heads x 3) was
the reason attention reached only 3% of the MPS ceiling. Batching all heads into
one `MPSMatrixMultiplication` with `batchSize` should have removed that.

Measured, twice, at 6k: **37.6 s and 38.2 s batched against 32.4 s per-head.**
Reverted. The encoder overhead was not the bottleneck — the profiler shows GPU
occupancy was already 92%, so there was nothing to reclaim.

### Sliding layers on the GPU: much slower at long context

Since sliding layers dominate CPU time, sending them to the GPU looked obvious.
But a GEMM cannot skip out-of-window keys: it computes the whole S x nkey score
matrix and lets the mask discard it.

| context | waste factor | result |
|---|---|---|
| 2k | 2.0x | 9.8 -> 8.6 s (better) |
| 6k | 5.9x | 32.4 -> **53.1 s** (worse) |
| 30k | 29.3x | not run, clearly worse |

Reverted. The CPU loop skips those keys outright, which no dense GEMM can.

### GPU expert bank: 8 GB to be 2.5x slower

Removed entirely, along with `lg_metal_upload_f16` and `lg_metal_gemm_rows`.
With `LG_CHUNK=256` each expert sees ~8 rows, and an 8x2048 @ 2048x512 GEMM is far
too small for MPS — dispatch dominates.

| | expert-mm @2k |
|---|---|
| CPU UDOT | **13.7 s** |
| GPU (5 layers, 8.05 GB) | 33.8 s |

It cost 8 GB of f16 weights to run 2.5x slower. Deleted rather than left behind a
flag, since a knob nobody should ever set is just a trap.

## Current state, default configuration

`SNAP=<model> ./c/laguna_xs_metal 0 0 --chat -n N -f prompt.txt`, no other
variables:

| context | prefill | attention | expert-mm | peak RSS |
|---|---|---|---|---|
| 2,000 | 25.1 s | 9.8 s | 12.1 s | 18.1 GB |
| 6,000 | 81.0 s | 32.2 s | 38.4 s | 18.3 GB |

Attention and experts are now balanced, and both are CPU-bound: attention by the
sliding layers, experts by the UDOT kernel.

## Remaining knobs

`LAGUNA_MEM_GB` (budget, default 20), `CTX_MAX` (context bound, default 8192),
`LAGUNA_GPU_PROF` (profiling). The tuning knobs that were removed this session:
`LAGUNA_RESIDENT`, `LAGUNA_GPU_BUDGET_GB`, `LAGUNA_GPU_EXPERT_GB`, `LG_METAL_MIN`.

## What would actually help next

The profiler says the remaining attention cost is 30 sliding layers on the CPU,
and a dense GPU GEMM cannot serve them. The options that respect that:

1. A **banded** Metal kernel that only materializes the `window`-wide diagonal
   band, not the full matrix. That is the shape the CPU loop already exploits.
2. Leave attention alone and attack `expert-mm` (38.4 s at 6k), which is now the
   larger phase.

## Correctness

XS 24/24 + 12/12 and S 208/208 + 8/8 on both the CPU and Metal builds after every
change including the removals, `task test` 3/3, `task test:baseline` clean.
