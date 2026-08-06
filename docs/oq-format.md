# oQ format (oMLX per-tensor mixed-precision affine quant)

Decoded from a real checkpoint, not from documentation. Source of truth:
`mlx-community/Laguna-S-2.1-oQ2e-fast` (7 shards, 35 GB). Every claim below was
verified against that repo's `config.json` and safetensors headers; the
verification is reproducible with `c/tools/probe_oq.py`.

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

## Measured: oQ buys memory, NOT arithmetic speed

This is the finding that matters most, and it goes against the intuition that
"bf16 is slow so a smaller quant will be faster". Measured with
`c/oq.h`'s kernel on an M-series CPU, K=3072 N=12288 (a `down_proj` shape, 75 MB
at bf16 so it does not fit in cache), single-token matvec, best of repeated runs:

| weights | ms/matvec | resident MB | vs bf16 size |
|---|---|---|---|
| f32 | 3.18 | 151 | 0.5x |
| **bf16 (today's engine)** | **3.10** | **75** | **1.0x** |
| oQ 8-bit | 2.85 | 42.5 | 1.78x smaller |
| oQ 6-bit | 3.68 | 33.0 | 2.29x smaller |
| oQ 3-bit | 4.00 | 18.9 | 4.00x smaller |
| oQ 2-bit | 4.29 | 14.2 | 5.33x smaller |

So per-matvec, oQ at 2-6 bits is **1.2-1.4x SLOWER** than bf16, not faster. Only
8-bit edges ahead. The unpack is compute-bound: the arithmetic saved by touching
fewer bytes is smaller than the arithmetic added to unpack them.

Two consequences that should shape expectations:

1. **For a model that already fits in RAM, oQ will not speed up decode.** Choose
   it to fit a bigger model, or to leave RAM for something else, not for tok/s.
2. **For a model that does NOT fit, oQ is the only thing that matters.** Laguna-S
   is 235 GB at bf16 against a 32 GB box, so every routed expert is read from
   disk on a cache miss. There, 5.33x less data per expert is a 5.33x smaller
   disk read on the critical path, and disk is orders of magnitude slower than
   the unpack. That is the regime this engine actually runs in.

### 3-bit and 6-bit needed specializing

The generic bit-cursor loop (needed because 32 % bits != 0) measured **2.4x
slower** than the power-of-two widths: 1.92 ms vs 0.80 ms on the smaller
3072x3072 shape. Every value cost a division-shaped index computation plus a
straddle branch.

Both are now specialized on the observation that 3 words hold exactly 32 values
at 3-bit and exactly 16 at 6-bit, so shifts become compile-time constants and
only one value per window straddles. That took 3-bit from 1.92 to 1.08 ms
(1.8x) and 6-bit from 1.93 to 0.97 ms (2.0x), verified still bit-exact against
`mlx.core.dequantize`. Worth doing because 3-bit is the dominant width in an
oQ3e checkpoint.

### A separate bottleneck this exposed: the bf16 path itself

bf16 moves half the bytes of f32 but takes the same time (3.10 vs 3.18 ms), i.e.
23.9 GB/s vs 47.5 GB/s of effective bandwidth. The per-element
`bf16_to_f32` (shift into a `uint32_t`, `memcpy` into a float) is eating the
entire bandwidth advantage. Vectorizing that conversion is a real, independent
speedup for every bf16 checkpoint, worth more than further oQ tuning for anyone
running the original release. Not done yet.

### Dequant-on-load was rejected

Dequantizing to f32 at load would make the matmul as fast as the f32 row above
and throw away 100% of the memory saving, which is the only reason to use oQ.
The kernel therefore keeps codes packed and unpacks per group inside the dot
product. The affine form factors out per group,
`sum_i x_i*(c_i*s + b) = s*sum_i x_i*c_i + b*sum_i x_i`, so the scale/bias
multiply happens once per group rather than once per weight, and the activation
sum is hoisted out of the output-row loop.
