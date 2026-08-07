# Metal/MPS prefill port, and why 300 tok/s is impossible here

## 300 tok/s: a bandwidth wall, not a code problem

The condition was "16 GiB is acceptable only at 300 tok/s". That cannot be met on
this machine at any bit width, so the memory increase is **not** taken: the
resident bank is now opt-in (`LAGUNA_RESIDENT=1`) and default behaviour is
streaming.

Single-stream decode touches 2.49 G params per token (attention 1.17 G, top-8 of
256 experts 0.98 G, lm_head 0.21 G, shared 0.13 G). Measured RAM read bandwidth
on this M5 is 126 GB/s, so:

| config | bytes/token | ceiling |
|---|---|---|
| everything at 2-bit | 622 MB | **203 tok/s** |
| everything at 1-bit (does not exist) | 311 MB | 405 tok/s |
| **300 tok/s would require** | 420 MB | **187 GB/s** |

300 tok/s needs 1.35 bits per parameter at 100% bandwidth efficiency with zero
compute time. The hardware supplies 126 GB/s. This is arithmetic; no kernel,
quantization scheme or GPU port changes it.

300 tok/s *aggregate* is reachable with batching, because weights are then read
once per batch rather than per token (batch 8 -> ~1600 tok/s aggregate at 2-bit).
That is a different feature (continuous batching), not single-stream latency.

## The Metal port

`c/laguna_metal.mm` + `c/laguna_metal.h`, built by `make laguna_xs_metal` /
`laguna_s_metal`. The plain CPU targets are untouched and stay dependency-free.

Design, driven by the measurements in `docs/redesign-roofline.md`:

- **MPS, not hand-written shaders.** `MPSMatrixMultiplication` in f16 measured
  15572 GFLOP/s against 1087 for my own tiled Metal kernel and 59 for the CPU.
  Hand-tuning was strictly worse than calling Apple's.
- **Projections only.** Per-layer q/k/v/o and the shared expert are uploaded once
  as f16 (3.10 GB for XS, capped by `LAGUNA_GPU_BUDGET_GB`, default 4). Routed
  experts are excluded on purpose: 256 per layer at f16 is 63 GB.
- **Gated at S >= 32** (`LG_METAL_MIN`). A Metal round-trip is 0.327 ms, so small
  GEMMs lose; upstream `colibri.c` gates its own GPU GEMM at 16 for the same
  reason. Decode stays entirely on the CPU.
- **Falls back silently.** No device, no buffer, unhandled shape -> returns 0 and
  the CPU path runs. `[metal] unavailable, CPU only` when there is no GPU.

## Results, 6144-token prefill, oQ2

| config | prefill | attn | expert-mm | shared | peak RSS |
|---|---|---|---|---|---|
| round 8 (CPU) | 465.1 s | 279.5 s | 153.5 s | 16.2 s | 11.05 GiB |
| Q8R resident (CPU) | 339.9 s | 274.8 s | 39.3 s | 15.2 s | 16.54 GiB |
| **Metal, streaming** | **249.6 s** | **60.5 s** | 165.7 s | **1.1 s** | 18.9 GiB @CAP=48 |
| Metal + Q8R resident | **115.8 s** | 60.1 s | 43.0 s | 1.0 s | 18.4 GiB |

The GPU did what the roofline predicted for the phases it covers:

- **attention 279.5 -> 60.5 s (4.6x)**
- **shared expert 16.2 -> 1.1 s (14.7x)**

Metal and Q8R are complementary -- Metal fixes attention and shared, Q8R fixes
the expert path and IO -- and combining them gives **115.8 s, 4.0x off round 8**.
That combination needs the resident bank, so it is opt-in for the reason above.

## A leak I introduced and fixed

The first working Metal build grew RSS from 2 GB to 14.5 GB over a single
6144-token prefill. Diagnosis was a straight A/B: the CPU build held ~2 GB on the
same prompt while the Metal build climbed monotonically, which localized it to my
code rather than the engine.

Two causes, both "cheap object created per call":

1. A fresh `MPSMatrixMultiplication` per `lg_metal_gemm` call. Prefill makes ~280
   calls per step and each object retains internal pipeline state. Now cached per
   (weight, S).
2. Fresh `MPSMatrix` wrappers over the scratch buffers, same story. Now cached
   per (S, K) / (S, N) and invalidated when the underlying buffer is reallocated.

After both: RSS flat at ~2 GiB for the same run, matching the CPU build.

Important caveat on that number: it was measured at `CAP=16` on a 512-token
prompt, where the expert cache is small enough for the leak to be visible. In the
`CAP=48` 6144-token runs the cache dominates RSS completely, so before/after the
fix those runs measure 15.9 and 18.9 GiB respectively -- the fix does not reduce
RSS there and was never going to. Both statements are true; the leak was real
(a monotonic ~12 GB climb) and is gone, but eliminating it does not by itself
make the long-context CAP=48 configuration fit a memory budget. `CAP` is the
lever for that. Timings are unaffected either way: 247.8 s before, 249.6 s after.

## Where the remaining memory actually goes

The 18.9 GiB in the streaming rows above is **not** the Metal port. Measured on a
512-token prompt, varying only the cache size:

| CAP | peak RSS |
|---|---|
| 4 | 6.9 GB |
| 16 | 7.3 GB |
| 48 | 8.8 GB |

There is a **4.98 GB floor** reported at load, which is page-cached safetensors
data: the OS counts mapped file pages against RSS, and they are evictable rather
than committed. The variable part above that floor is the f32 expert cache, which
`CAP` controls directly.

**But `CAP` stops being an effective lever at long context**, and the 512-token
table above does not generalize. Measured at 6144 tokens with the same binary:

| CAP | peak RSS @6144 |
|---|---|
| 16 | 19.68 GiB |
| 48 | 18.88 GiB |

Effectively identical, and not ordered the way CAP would predict. At long context
the per-call scratch (q/ctx are S*qdim f32 = 151 MB each, plus nrm/tmp/shared) and
the KV cache dominate, so shrinking the expert cache reclaims little. Reducing
peak RSS at 6K therefore needs the scratch itself addressed -- f16 or chunked
prefill -- not a smaller cache.

Honest summary of the memory picture: the Metal port adds a bounded 3.10 GB
(`LAGUNA_GPU_BUDGET_GB`), the resident bank adds 13.7 GB and is opt-in, and the
~19 GiB seen in long-context streaming runs is mostly scratch plus page cache
that none of this work reduced.

## Current recommended configurations

```
# lowest memory on SHORT prompts (CAP is only effective there -- see above)
CAP=8 ./c/laguna_xs_metal ...

# fastest prefill, needs ~18 GiB
LAGUNA_RESIDENT=1 ./c/laguna_xs_metal ...

# no GPU at all (other platforms, or to A/B the port)
./c/laguna_xs ...
```

At 6144 tokens every streaming configuration lands near 19 GiB regardless of
`CAP`, so there is currently no long-context configuration that fits a ~13 GiB
budget. That is an open problem, not a solved one: the fix is f16 or chunked
prefill scratch, which is not done.

## Correctness

`laguna_xs_metal` and `laguna_s_metal` both reproduce the transformers-oracle
fixtures token-exactly: XS 24/24 teacher-forced + 12/12 generated, S 208/208 +
8/8, with f16 GEMM active on every layer. The CPU builds are unchanged and still
exact. `task test` 3/3, `task test:baseline` clean, upstream engines build.

f16 has an 11-bit mantissa and the fixtures pass regardless, but the gate is
`S >= 32`, so short prompts and all decode run in f32 on the CPU -- the precision
change only ever applies to batched prefill.

## Honest status against the original targets

- **prefill < 20 s**: not yet. 115.8 s at 6144 tokens (from 465 s). The remaining
  time is 60 s attention and 43 s experts. Getting under 20 s needs the routed
  experts and the attention *scores* on the GPU too, which means keeping K/V in
  f16 GPU-side and dequantizing experts per layer on the fly -- the plan is in
  `docs/redesign-roofline.md`, it is a substantial further build.
- **50 tok/s**: not yet, and not from the GPU. Decode is 5-14 tok/s and is
  overhead-bound, not bandwidth-bound (the bandwidth ceiling is ~200 tok/s). The
  work is reducing per-token overhead on the CPU path.
- **300 tok/s**: impossible on this hardware, shown above.
- **100x**: no. Cumulative prefill improvement is 4.0x at 6144 tokens and 3.4x at
  1902 tokens.
