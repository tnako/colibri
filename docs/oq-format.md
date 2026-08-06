# oQ format (oMLX per-tensor mixed-precision affine quant)

Decoded from a real checkpoint, not from documentation. Source of truth:
`mlx-community/Laguna-S-2.1-oQ2e-fast` (7 shards, 35 GB). Every claim below was
verified against that repo's `config.json` and safetensors headers; the
verification is reproducible with `c/tools/probe_oq.py`.

Implemented as **fmt=101**, a private ordinal: `colibri.c` reserves 0-8 for
maintainer-assigned public formats (8 is already native FP8-e4m3) and keeps
100+ as the experimental block for in-flight proposals.

oQ is MLX's `affine` quantization with a **per-tensor** bits/group_size choice
driven by an importance matrix. That is exactly the "lots of models at 3, 3.5, 4,
5, 6 bits" surface — one loader covers all of them, because the bit width is data
in the checkpoint rather than a property of the format.

## Where the bit widths live

`config.json` carries both a global default and a per-tensor override map. The
same content appears under `quantization` and `quantization_config`.

```json
"quantization": {
  "group_size": 128, "bits": 2, "mode": "affine",
  "language_model.lm_head":                       {"bits": 8, "group_size": 64,  "mode": "affine"},
  "language_model.model.embed_tokens":            {"bits": 8, "group_size": 64,  "mode": "affine"},
  "language_model.model.layers.0.mlp.gate_proj":  {"bits": 3, "group_size": 64,  "mode": "affine"},
  "language_model.model.layers.0.mlp.down_proj":  {"bits": 6, "group_size": 64,  "mode": "affine"},
  "language_model.model.layers.1.mlp.shared_expert.gate_proj": {"bits": 8, "group_size": 128, "mode": "affine"}
}
```

The scalar keys are the fallback for any tensor with no entry. In the `oQ2e-fast`
checkpoint that default is `bits: 2, group_size: 128` and it applies to the
routed experts (`switch_mlp.*`), which are the overwhelming majority of the
weights — hence "2e" in the name. The 386 explicit overrides are the tensors the
importance matrix flagged as sensitive.

Distribution of `(bits, group_size)` across the 386 overrides:

| bits | group_size | tensors |
|---|---|---|
| 8 | 128 | 141 |
| 3 | 64 | 137 |
| 4 | 64 | 59 |
| 8 | 64 | 34 |
| 6 | 64 | 15 |

So a single checkpoint mixes 2, 3, 4, 6 and 8 bit tensors, with two different
group sizes. A loader that assumes one uniform bit width cannot read it.

Naming: the fractional names (`oQ3.5`, `oQ4e`) describe the *average* bits per
weight across the model, not any single tensor's width. There is no 3.5-bit
encoding — no `bits` value is ever fractional.

## On-disk tensor layout

Each quantized weight is three tensors sharing a stem:

| suffix | dtype | shape | meaning |
|---|---|---|---|
| `.weight` | `U32` | `[N, K*bits/32]` | packed codes, dense bitstream |
| `.scales` | `BF16` | `[N, K/group_size]` | per-group multiplier |
| `.biases` | `BF16` | `[N, K/group_size]` | per-group offset (zero point, pre-scaled) |

Worked example, `layers.0.self_attn.q_proj` at bits=3, gs=64, K=3072:

```
.weight  U32  [6144, 288]     288 = 3072*3/32
.scales  BF16 [6144, 48]       48 = 3072/64
.biases  BF16 [6144, 48]
```

Routed experts add a leading expert axis — `switch_mlp.down_proj.biases` is
`BF16 [256, 3072, 8]`, i.e. `[E, N, K/gs]`. Convenient: the per-expert slice is
contiguous, so the existing per-expert streaming design still works unchanged.

**Verified**: for all 396 quantized tensors in shard 1, `words == K*bits/32` and
`K == ngroups*group_size` held exactly, where `K` was derived independently from
the scales shape. No padding, no exceptions.

### Packing is a dense bitstream

For bits=3 and bits=6, `32 % bits != 0`, so values straddle 32-bit word
boundaries. Confirmed arithmetically: bits=3 with K=3072 gives 288 words =
9216 bits = 3072x3 exactly. There is no per-word padding to unpack around.

The saving grace is that **every group is word-aligned**, checked for all six
`(bits, group_size)` combinations that actually occur:

| bits | gs | group size in bits | whole words |
|---|---|---|---|
| 3 | 64 | 192 | 6 |
| 6 | 64 | 384 | 12 |
| 2 | 128 | 256 | 8 |
| 4 | 64 | 256 | 8 |
| 8 | 64 | 512 | 16 |
| 8 | 128 | 1024 | 32 |

So the unpacker can process one group at a time from a word boundary and never
needs to carry state across groups. That is what makes a clean C implementation
practical for the non-power-of-two widths.

### Dequantization

MLX affine mode, per group `g` of a row:

```
w[i] = q[i] * scales[i / group_size] + biases[i / group_size]
```

`biases` is an additive offset already in weight space (not an integer zero
point needing multiplication by the scale). Codes are unsigned: `q` in
`[0, 2^bits - 1]`, confirmed by observing the full range in real tensors
(bits=3 -> 0..7, bits=6 -> 0..63, bits=8 -> 0..255).

Note both `scales` and `biases` are BF16, so a bit-exact reimplementation needs
the same BF16 -> f32 widening the engine already has in `st.h`.

Bit order within the stream is **LSB-first, little-endian words**: value `i`
occupies bits `[i*bits, (i+1)*bits)` of the concatenated little-endian
bitstream, low bit first.

**This was verified, not inferred.** An independent numpy unpacker built from
this description was compared against `mlx.core.dequantize` on real downloaded
bytes:

| tensor | bits | gs | result |
|---|---|---|---|
| `layers.0.self_attn.q_proj` | 3 | 64 | exact, maxdiff 0.0 |
| `layers.0.mlp.down_proj` | 6 | 64 | exact, maxdiff 0.0 |
| `lm_head` | 8 | 64 | exact, maxdiff 0.0 |
| `layers.1.mlp.shared_expert.gate_proj` | 8 | 128 | exact, maxdiff 0.0 |

Both non-word-aligned widths (3 and 6) are covered, so the straddling logic is
confirmed rather than assumed.

## Tensor names differ from the HF checkpoint

MLX rewrites the tree. Mapping needed by any loader that already reads the
bf16 release:

| HF / released checkpoint | oQ (MLX) |
|---|---|
| `model.layers.N...` | `language_model.model.layers.N...` |
| `lm_head.weight` | `language_model.lm_head.weight` |
| `mlp.experts.<e>.{gate,up,down}_proj.weight` | `mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}` with a leading `[E, ...]` axis |
| `mlp.gate.weight` | `mlp.gate.proj.weight` |
| `mlp.experts.e_score_correction_bias` | `mlp.gate.e_score_correction_bias` |
| `mlp.shared_expert.*` | `mlp.shared_expert.*` (unchanged) |

Norms, the router projection and `e_score_correction_bias` stay **BF16 and
unquantized** — `mlp.gate.proj.weight` is `BF16 [256, 3072]`, not packed. Only
`Linear`/`SwitchLinear` weights get quantized, which matches the imatrix
report's module census (`Linear: 47`, `QuantizedLinear: 385`,
`QuantizedSwitchLinear: 141`).

## What the imatrix report tells us

`oq_imatrix_report.json` ships alongside the weights and documents how the
per-tensor widths were chosen: 1024 calibration samples of
`oqe_code_multilingual` at seq_length 512, 573 instrumented modules, per-expert
activation counts (`total_experts: 36096`, `active_experts: 36084`, median
activation count 14013).

Two things worth knowing from it:

- 12 experts had a zero activation count across the whole calibration set. They
  are still present in the weights, just never exercised during calibration.
- It records the *provenance* of the width choices but is **not needed at
  inference time**. The widths that matter are all in `config.json`.

## Implementation consequences for this engine

- **One loader covers every oQ variant.** Bit width is per-tensor data, so
  `oQ2e`, `oQ3.5`, `oQ4e`, `oQ6` need no separate code paths. This is the direct
  answer to wanting to test many bit widths cheaply.
- **Five unpack widths needed**: 2, 3, 4, 6, 8. Group-alignment means each can be
  a small loop over whole words with no cross-group carry.
- **Do not reuse `quantize_rows`.** That is per-row symmetric int8 with no zero
  point; oQ is per-group asymmetric affine with a BF16 additive bias. Different
  arithmetic, different metadata shape.
- **Routed experts stay streamable.** The `[E, N, K/gs]` layout keeps each
  expert's slice contiguous, so the existing LRU + per-expert read design holds.

## Measured: oQ is smaller AND faster, once the kernels are vectorized

M5 (4P+6E, 32 GB), K=3072 N=12288 (`down_proj` shape, 75 MB at bf16 so it misses
cache), single-token matvec. The "scalar" column is the first working version,
"NEON" is what ships:

| weights | scalar ms | NEON ms | speedup | resident MB | vs bf16 size |
|---|---|---|---|---|---|
| f32 | 3.18 | 3.06 | 1.0x | 151 | 0.5x |
| bf16 | 3.10 | 3.05 | 1.0x | 75 | 1.0x |
| oQ 8-bit | 2.85 | **0.95** | 3.0x | 42.5 | 1.78x smaller |
| oQ 6-bit | 3.68 | **1.48** | 2.5x | 33.0 | 2.29x smaller |
| oQ 4-bit | 5.03 | **1.14** | 4.4x | 23.6 | 3.20x smaller |
| oQ 3-bit | 4.00 | **1.36** | 2.9x | 18.9 | 4.00x smaller |
| oQ 2-bit | 4.29 | **1.02** | 4.2x | 14.2 | 5.33x smaller |

**This reverses the earlier conclusion in this file.** Before vectorizing, oQ at
2-6 bits was 1.2-1.4x slower than bf16 and the honest advice was "oQ buys memory,
not speed". Vectorized, every width is 2.0-3.3x FASTER than bf16 as well as
smaller, because the unpack was never memory-bound -- it was scalar-code-bound.

Two things did the work, both in the code that touches every weight:

1. `oq_group_dot` widens the uint8 codes with `vmovl` and accumulates with f32
   FMA, computing `dot(x,c)` and `sum(x)` in one pass over the group.
2. `-mcpu=native` for the Laguna targets (`LG_ARCH` in the Makefile). The shared
   Darwin CFLAGS pass no `-mcpu` on purpose, which left everything at baseline
   armv8.

### bf16 was the worse bottleneck: 5.2x

`bf16 -> f32` is exactly a 16-bit left shift, so `vshll_n_u16(v, 16)` IS the
conversion. The old per-element shift-into-`uint32_t`-then-`memcpy` measured
**19.7 GB/s; the NEON form measures 102.6 GB/s, a 5.2x speedup** on the same
data. That was pure conversion overhead, not memory.

`FEAT_BF16`'s BFDOT is available on M5 and reached 132 GB/s, but it is **not
used**: BFDOT needs both operands in bf16, so the f32 activations would be
rounded first, and the result changed (1024.91 vs 1024.62 on the bench). The
fixtures are token-exact against a transformers oracle; an f32-accurate
accumulation is worth more than the last 30%.

### 3-bit and 6-bit needed specializing first

They hit the generic bit cursor (32 % bits != 0) and measured 2.4x slower than
the power-of-two widths, 1.92 vs 0.80 ms at 3072x3072. Both are now specialized
on 3 words holding exactly 32 resp. 16 values, which makes the shifts constant
and leaves one straddling value per window: 3-bit 1.92 -> 1.08 ms, 6-bit
1.93 -> 0.97 ms. 3-bit is the dominant width in oQ3e.

### Still true regardless of speed

For Laguna-S (235 GB bf16 against 32 GB) every routed expert is read from disk on
a cache miss, so 5.33x less data per expert is a 5.33x smaller read on the
critical path against a device orders of magnitude slower than any of this.
Fitting remains the main reason to pick oQ; being faster is a bonus.

Dequant-on-load stays rejected: it would forfeit the entire memory saving to land
on the f32 row above, which is now the SLOWEST row in the table.

Every number here is reproducible with `c/tools/bench_oq.c` and
`c/tools/bench_bf16.c`.

## End to end on a real oQ checkpoint

`mlx-works/Laguna-XS-2.1-oQ2` (11 GB on disk, 2-bit default gs64, 322 per-tensor
overrides), M5/32 GB, `task laguna:xs Q=oQ2`, 30 tokens, auto cache sizing:

| build | prefill | decode | expert-mm | attn |
|---|---|---|---|---|
| scalar | 4.5 s | 2.70 tok/s | 6.4 s | 5.4 s |
| **NEON + `-mcpu=native`** | **2.3 s** | **5.26 tok/s** | **3.5 s** | **2.0 s** |
| | 2.0x | **1.95x** | 1.8x | 2.7x |

RSS 6.8 GB for a 22B model with 161 experts/layer cached. Output is coherent
prose, and the same binary still reproduces both bf16 tiny fixtures token-exactly
(24/24 and 208/208), so the vectorization did not trade accuracy for speed.

Two bugs the first real run exposed, neither reachable from the tiny fixtures:

- `embed_tokens` is oQ-packed too (8-bit in every variant seen). `wt_row_f32`
  read it as bf16 and walked off the end of a buffer 1/2 the assumed size —
  instant SEGV on the first token. It now dequantizes the row.
- The same function took a flat element offset while the packed path needs a row
  index, so the callers had to change with it.

Worth stating plainly: the fixtures are synthetic f32, so no amount of fixture
passing would have caught either. Running the real checkpoint was the test.

## Metal: measured, then deliberately not used for oQ decode

Asked for, benchmarked, rejected on evidence. `c/backend_metal.mm` already has an
`mm_gemv` shader handling fmt 1-8, so adding fmt=101 would be a shader plus
dispatch plumbing. The reason not to, on this M5:

| | measured |
|---|---|
| Metal round-trip (encode + commit + wait, trivial kernel) | **0.327 ms** |
| CPU oQ 2-bit matvec, K=3072 N=12288 | **1.02 ms** |
| matmuls per Laguna-S decode token (48 layers x ~7) | ~336 |
| floor per token if each one round-trips | **~110 ms** |

A GPU dispatch costs a third of what the whole CPU kernel now costs, so a
per-matmul offload cannot win at decode (S=1) — and 336 sequential round-trips
per token is 110 ms of pure latency before any arithmetic. This is exactly why
`colibri.c` gates its own Metal GEMM at `S >= g_metal_gemm_min` (default 16,
`COLI_METAL_GEMM_MIN`): the GPU is for batched prefill, not token-by-token decode.

Where Metal would genuinely pay for this engine, in order:

1. **Batched prefill** (S >= 16), following upstream's existing gate. One
   dispatch amortizes over many rows.
2. **A resident expert tier**, like `coli_metal_moe_gemv` does for other formats:
   keep hot experts in GPU-visible memory and batch the routed GEMVs for all
   selected experts into one dispatch, instead of one per expert per token.

Both are real work and neither is done. The honest current state is that the
NEON CPU path is fast enough that Metal has to clear a much higher bar than it
did before this optimization pass — the 4.2x CPU speedup moved the goalposts.

M5 hardware notes gathered while measuring (`sysctl hw.optional.arm.*`):
`FEAT_BF16`, `FEAT_EBF16`, `FEAT_I8MM`, `FEAT_DotProd`, `FEAT_SME`/`SME2` all
present, 4 performance + 6 efficiency cores, unified memory. BFDOT and I8MM are
reachable but unused for the accuracy reason above; SME is untouched (it needs
streaming-mode setup that only pays off on much larger tiles than a decode GEMV).
