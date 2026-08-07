# Aggressive redesign: roofline, Q8R, and where the 100x actually lives

Rounds 10-13 of the optimization work, aimed at "100x faster, prefill under 20s,
50 tok/s". This file records what the hardware can actually do, what was built,
and which targets are reachable on which processor. Hardware: M5 (4P+6E), 32 GB.

## Start from the roofline, not from a kernel

Before writing anything, measured the ceilings. Every number here is from a
benchmark in `c/tools/`, not a datasheet.

| unit / path | measured | vs CPU f32 |
|---|---|---|
| CPU f32 FMA GEMV, 10 threads | 59 GFLOP/s | 1.0x |
| CPU UDOT (int8, FEAT_DotProd) | 298 GOP/s | **5.0x** |
| CPU UDOT on packed 2-bit codes | ~198 GOP/s | 3.4x |
| CPU SMMLA (I8MM) as tested | 90 GOP/s | 1.5x (my shape was wrong) |
| GPU hand-written Metal f32 GEMM | 1087 GFLOP/s | 18x |
| **GPU MPS f16 GEMM** | **15572 GFLOP/s** | **264x** |
| RAM read, 10 threads | 126 GB/s | |
| mmap of page-cached file | 127 GB/s | same as RAM |
| NVMe pread, 1 thread | 11.2 GB/s | |
| NVMe pread, 10 threads | 30.6 GB/s | queue depth matters |
| Metal dispatch round-trip | 0.327 ms | |

Two immediate consequences:

- **mmap costs nothing over RAM** once pages are resident, so explicit `pread`
  into private buffers buys nothing on this machine.
- **NVMe needs parallelism**: 11 GB/s single-threaded, 31 GB/s at depth 10. Any
  storage-bound design must issue reads from many threads.

### The workload, in FLOPs and bytes

Laguna-XS at 6144 tokens: **35.8 TFLOP** of prefill (attention projections 40%,
MoE 34%, scores 22%, shared 4%). Decode touches **2.49 G params/token** -- only
top-8 of 256 experts, which is the fact that makes the bandwidth arithmetic work
out.

## Q8R: a UDOT-native expert format (shipped)

`c/q8r.h`. The oQ kernels dequantize to f32 and run f32 FMA, capped at 59
GFLOP/s. Q8R quantizes activations to uint8 per group and runs the integer dot
directly, using the affine factorization:

```
w_i = c_i*sw + bw ,  x_i = u_i*xs + xz
sum w_i x_i = sw*xs*UDOT(c,u) + sw*xz*rsum + bw*xsum
```

`rsum` (sum of codes per group) is precomputed at load; `xsum` comes from the
activation quantizer. One UDOT chain plus two scalars per group.

Measured against the shipping oQ kernel on real shapes: **1.7-4.5x**, peaking at
442 GOP/s on the lm_head shape. Accuracy against the exact f32 path on real oQ
bytes: relative error 4.8e-5 (2-bit) to 6.6e-4 (8-bit), **zero argmax
mismatches**, and both tiny fixtures stay token-exact.

### A sizing error worth recording

The first version stored one byte per code and I claimed the whole model would
fit in 8.9 GiB. That was wrong: 8.9 GiB is the *2-bit packed* size, but
one-byte-per-code is 4x that for a 2-bit checkpoint -- **31.4 GB of routed
experts**, which does not fit in 32 GB. The engine printed
`[resident] need 37.3 GB, only 17.5 GB available -> streaming` and fell back,
which is how the mistake surfaced.

Fix: keep codes in the oQ packing and unpack 16 codes per word straight into a
UDOT register (`q8r_udot_2bit`, verified bit-identical to the byte path). The
resident bank is then **13.7 GB and loads in 2.6 s**.

Interesting negative result: the packed path measures **0.67x of the byte path**
per dot. It is still the right choice, because the byte path cannot be resident
at all -- and being resident is worth more than the 33%.

## Result: expert IO eliminated

With the whole bank resident there is no cache, no LRU, no `slot_fill`, no
eviction hazard. At 6144 tokens:

| phase | round 8 | Q8R resident |
|---|---|---|
| prefill | 465.1 s | **339.9 s** |
| expert-mm | 153.5 s | **39.3 s** (3.9x) |
| fill (IO) | 6.0 s | **0.0 s** |
| cache hit | 99.4% | 100% (no cache) |
| peak RSS | 11.05 GiB | 16.54 GiB (+50%) |

At 1902 tokens, prefill is **89.2 s** against the round-0 baseline of 301.0 s and
round 5's 136.1 s -- **3.4x cumulative**.

RSS went up 50%, past the +20% budget. That is the deliberate trade and it can be
tuned back: `LAGUNA_RESIDENT=0` restores streaming, and a hot/cold split
(resident top-N experts by usage, stream the tail) would land between the two.

## Attention is now the whole problem, and CPU cannot fix it

Attention is **274.8 s of the 339.9 s prefill (81%)**. Its scores are 3.87 TFLOP
at 6144 tokens, and it currently runs at ~14 GFLOP/s -- so there is roughly 4x of
headroom to the 59 GFLOP/s CPU f32 ceiling, and hoisting the query tile into
contiguous memory recovered part of it (attn 84.9 -> 69.7 s at 1902 tokens).

But the ceiling itself is the wall:

```
scores 3.87 TFLOP / 59 GFLOP/s  = 66 s   <- CPU floor, unreachable below
scores 3.87 TFLOP / 15.5 TFLOP/s = 0.25 s <- GPU
```

## Honest verdict on the targets

**prefill < 20 s: not reachable on the CPU.** The attention scores alone have a
66 s floor at the measured f32 ceiling, and total prefill is 35.8 TFLOP which
needs ~1.8 TFLOP/s. The CPU tops out at 0.06 TFLOP/s (f32) or ~0.3 TOP/s (UDOT).
This is arithmetic, not tuning.

**prefill < 20 s IS reachable on the GPU.** 35.8 TFLOP at the measured MPS f16
rate of 15.5 TFLOP/s is 2.3 s of GEMM. Adding dequant passes (~0.6 s), dispatch
overhead and sync, a realistic target is 5-10 s -- comfortably inside 20 s.

**50 tok/s: reachable, and NOT bandwidth-limited.** Decode touches 1751 MB/token
with experts at 2-bit and an 8-bit lm_head, which at 126 GB/s allows **72 tok/s**.
Current decode is 10-14 tok/s, so the gap is per-token overhead (dispatch,
scratch allocation, small-GEMM inefficiency), not memory. The resident bank
already moved decode 10.2 -> 13.9 tok/s.

**100x overall: no, not on the CPU.** The honest ceiling from here is roughly
5-7x cumulative on prefill. 100x requires the GPU, and specifically:

1. Port attention scores + projections to MPS f16 GEMM (biggest single win, 81%
   of remaining prefill).
2. Keep K/V in f16 on the GPU so scores never round-trip.
3. Batch MoE per-expert GEMMs into one dispatch per layer, dequantizing one
   layer's experts to f16 on the fly (1.6 GB/layer, fits).
4. Leave decode on the CPU with the resident UDOT bank -- at S=1 the 0.327 ms
   dispatch cost still dominates, which the earlier Metal measurement showed.

That is a substantial build, not a round of tuning. It is the only route to the
requested numbers, and the measurements above are what justify saying so.

## Reproduce

```
c/tools/bench_isa.c    # CPU f32 vs UDOT vs SMMLA
c/tools/bench_mem.c    # RAM, mmap, NVMe at depth 1 and 10
c/tools/gpu_gemm.mm    # hand-written Metal f32/oQ2 kernels
c/tools/mps_gemm.mm    # Apple MPS f32/f16 -- the 15.5 TFLOP/s number
c/tools/chk_q8r.c      # Q8R vs exact f32 oQ, error + argmax
CAP=48 c/tools/stress_laguna.sh models/Laguna-XS-2.1-oQ2 6144 8 tag
```

## Correctness

Both paths stay token-exact on the transformers-oracle fixtures: XS 24/24 + 12/12,
S 208/208 + 8/8 (the S fixture wraps the sliding ring 50x). `task test` 3/3.
`LAGUNA_RESIDENT=0` exercises the streaming path, default exercises Q8R, and both
were run for every result above.

Two real bugs the resident path exposed, both caught by ASan rather than by
inspection: scratch buffers keyed on row count alone were reused across experts
with different group sizes (Q8Act's `ng` derives from `gs`), and the resident
branch double-freed the visit-order array with the shared-expert tail.
