# Laguna (Poolside Laguna-XS / Laguna-S)

Two checkpoints, one architecture, two binaries: `c/laguna_xs.c` and
`c/laguna_s.c` are one-line wrappers around `c/laguna_common.h`, which holds the
entire forward pass. Every code path is identical between the sizes; only config
numbers differ, and all of them are read from `config.json` at load time. There
are no compile-time geometry constants.

    make -C c laguna_xs
    make -C c laguna_s

    SNAP=<checkpoint> ./c/laguna_xs --config          # geometry only, loads no weights
    SNAP=<checkpoint> ./c/laguna_xs --chat -p "hello" # generate
    coli chat  --model <checkpoint>                   # launcher picks the binary
    coli serve --model <checkpoint>

## Geometry, from the released checkpoints

Read from `poolside/Laguna-XS-2.1` and `poolside/Laguna-S-2.1` — not from
`LagunaConfig`'s class defaults, which differ from what shipped.

| | XS | S |
|---|---|---|
| hidden_size | 2048 | 3072 |
| num_hidden_layers | 40 (10 full / 30 sliding) | 48 (12 full / 36 sliding) |
| heads per layer | 48 full, 64 sliding | 48 full, 72 sliding |
| num_key_value_heads | 8 | 8 |
| head_dim | 128 | 128 |
| num_experts / per token | 256 / 8 | 256 / 10 |
| moe_intermediate_size | 512 | 1024 |
| shared_expert_intermediate_size | 512 | 1024 |
| intermediate_size (layer 0, dense) | 8192 | 12288 |
| moe_routed_scaling_factor | 2.5 | 2.5 |
| sliding_window | 512 | 512 |
| max_position_embeddings | 262144 | 1048576 |
| YaRN factor / beta_fast | 32 / 64 | 128 / 32 |
| YaRN attention_factor | 1.3465735902799727 | 1.4852030263919618 |

`layer_types` repeats one full-attention layer then three sliding ones.
`mlp_layer_types` is layer 0 dense, everything else sparse. Both sizes use
`rope_type: "yarn"` on full-attention layers (`partial_rotary_factor` 0.5,
theta 500000) and `rope_type: "default"` on sliding layers (full rotary,
theta 10000).

Note XS is **not** `rope_type: "default"` on its full layers, which the class
defaults suggest. It ships YaRN, with its own factor.

## What is new versus the other engines

- **Per-head attention output gate.** `g_proj` is `Linear(hidden_size, n_heads)`
  and the attention context for head *h* is multiplied by `softplus(g[h])`
  before `o_proj`. No other Colibri engine gates attention output per head. It
  is not a router variant, despite `"gating": "per-head"` in the config.
- **Half-split rope.** Laguna uses HF's standard `rotate_half`, not the
  interleaved scheme `colibri.c` and `deepseek_v4.c` implement. The reference's
  own docstring says so: *"Removes the interleaving of cos and sin from GLM"*.
- **Partial rotary.** Full-attention layers rotate only the first
  `head_dim * 0.5` dimensions; the rest pass through and concatenate back.
  Sliding layers rotate everything. No precedent in the other engines.
- **Per-layer head count.** `num_attention_heads_per_layer` varies by layer
  type, with KV heads fixed at 8, so the GQA group size differs per layer.

## Checkpoint layout quirks

The loader accepts both layouts rather than guessing, because the released
checkpoints and a `save_pretrained` fixture disagree:

| | released checkpoint | transformers in-memory |
|---|---|---|
| routed experts | `mlp.experts.<e>.{gate,up,down}_proj.weight` | fused `mlp.experts.gate_up_proj` `[E,2I,D]` + `down_proj` `[E,D,I]` |
| shared expert | `mlp.shared_expert.*` | `mlp.shared_experts.*` |
| router bias | `mlp.experts.e_score_correction_bias` | `mlp.gate.e_score_correction_bias` |

A related trap: `load_state_dict` does **not** apply the per-expert to fused
conversion, so it silently leaves the whole MoE randomly initialized. Use
`from_pretrained` when building a reference to compare against.

## Sliding-window KV ring

Sliding layers keep only `window` rows, addressed `pos % window`. The append
happens **after** the scoring loop, and scoring reads the current batch's own
keys out of the projection scratch rather than the cache. That ordering is
required: appending the whole batch up front overwrites history rows that
earlier queries in the same batch still need, for any prefill with
`S > window` — silently, producing attention over future keys rather than
crashing.

This is the same hazard and the same resolution as upstream
[PR #830](https://github.com/JustVugg/colibri/pull/830) for `inkling.c`. When
that PR lands, the two implementations should be compared and, if a shared shape
emerges, extracted — see the migration table below.

## Shared migration status

Following the convention `docs/deepseek-v4.md` uses for its own shared-versus-
private code.

| Component | Status | Notes |
|---|---|---|
| MoE router (sigmoid + `e_score_correction_bias` top-k, renormalize, scale) | **shared / done** | `c/coli_moe_route.h`, also called by `colibri.c`'s GLM-5.2 router. Byte-identical logic; extracted so an upstream fix reaches both. |
| Router renorm + `routed_scaling_factor` | **shared / partial** | `coli_moe_norm_scale` exists and Laguna calls it; `colibri.c` still has it inline so its ablation and top-p bookkeeping stay untouched. Diff the two on every sync. |
| Sliding-window KV ring | **private, converging** | Upstream PR #830 does the same thing for `inkling.c`. Revisit once it merges. |
| Half-split partial rope apply | **private** | Nothing to share: every other engine's rope is interleaved. |
| YaRN frequency precompute | **private** | Transcribed from `modeling_rope_utils.py` directly. `deepseek_v4.c`'s version is written against one layer type and an interleaved apply; sharing it would need that generalized first. |
| Per-head attention gate | **private** | New primitive, no second consumer. |
| Per-layer head-count access | **private, deliberately** | Three one-line lookups. Extracting them for two consumers is not worth the indirection. |

## Validation

    python3 c/tools/make_laguna_tiny.py --output ./laguna_tiny --force
    python3 c/tests/test_laguna_tiny.py --binary ./c/laguna_xs --fixture ./laguna_tiny

The fixture is random-weight but exercises both layer types, a per-layer head
count that differs between them, a dense layer 0, a non-1.0 routed scaling
factor, a non-zero router bias, and a window smaller than the prompt so the ring
wraps. Requires PyTorch and transformers >= 5.12 (for `LagunaForCausalLM`); the
safetensors it writes is never committed.

Measured on this fork, f32 experts:

| fixture | engine | teacher-forced | generated |
|---|---|---|---|
| 4L, 8E, topk 2, window 4, 12-token prompt | `laguna_xs` | 24/24 | 12/12 |
| same | `laguna_s` | 24/24 | 12/12 |
| same, int8 experts (`bits=8`) | `laguna_xs` | 24/24 | 12/12 |
| same, `cap=1` (forces per-round eviction) | `laguna_xs` | 24/24 | 12/12 |
| 4L, 12E, topk 10, window 4, 200-token prompt (ring wraps 50x) | `laguna_s` | 208/208 | 8/8 |

## Not done

- No Metal / CUDA / Vulkan path. CPU only; profile before adding one.
- No int4 container conversion. Routed experts are read from the bf16 checkpoint
  and optionally runtime-quantized to int8. A `convert_laguna_int4.py` in the
  shape of `convert_inkling_int4.py` is the obvious next step for Laguna-S.
- No KV prefix reuse (`kv_prefix.h`), so a served conversation re-prefills every
  turn.
- No tool-call rendering in the served chat template.
