#!/usr/bin/env python3
"""Generate a tiny random-weight Laguna fixture plus its reference token oracle.

Reference tokens come from the official transformers LagunaForCausalLM — there
is no C-engine fallback, because the point of the fixture is to disagree with
the C engine when the C engine is wrong. The generated safetensors is never
committed; regenerate it with `task laguna:fixture`.

The fixture is deliberately small (128 vocab, 64 hidden, 4 layers) but exercises
every path the real checkpoints do:
  - both layer types, so both rope tables are used (full_attention with YaRN and
    partial_rotary_factor 0.5, sliding_attention with default rope and full
    rotary)
  - a per-layer attention head count that DIFFERS between the two types, so a
    hardcoded head count fails
  - layer 0 dense, the rest sparse
  - a non-1.0 moe_routed_scaling_factor and a non-zero e_score_correction_bias,
    so dropping either is visible
  - a sliding window smaller than the prompt, so the ring-buffer KV path and the
    window mask are actually hit

Usage:
  python3 tools/make_laguna_tiny.py --output ./laguna_tiny --force
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SEED = 1234
VOCAB = 128
HIDDEN = 64
LAYERS = 4
HEAD_DIM = 16
KV_HEADS = 2
HEADS_FULL = 4
HEADS_SLIDE = 6           # deliberately != HEADS_FULL
EXPERTS = 8
TOPK = 2
MOE = 32
SHARED = 32
DENSE = 128
WINDOW = 4                # smaller than the prompt on purpose
ROUTED_SCALE = 2.5
PROMPT_LEN = 12
NEW_TOKENS = 12


def build_config() -> dict:
    layer_types = ["full_attention" if i % 2 == 0 else "sliding_attention" for i in range(LAYERS)]
    return {
        "architectures": ["LagunaForCausalLM"],
        "model_type": "laguna",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": DENSE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS_FULL,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "max_position_embeddings": 512,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "rms_norm_eps": 1e-6,
        "num_experts": EXPERTS,
        "num_experts_per_tok": TOPK,
        "moe_intermediate_size": MOE,
        "shared_expert_intermediate_size": SHARED,
        "moe_routed_scaling_factor": ROUTED_SCALE,
        "moe_router_logit_softcapping": 0.0,
        "moe_apply_router_weight_on_input": False,
        "sliding_window": WINDOW,
        "layer_types": layer_types,
        "mlp_layer_types": ["dense"] + ["sparse"] * (LAYERS - 1),
        "num_attention_heads_per_layer": [
            HEADS_FULL if t == "full_attention" else HEADS_SLIDE for t in layer_types
        ],
        "rope_parameters": {
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500000.0,
                "factor": 32.0,
                "original_max_position_embeddings": 128,
                "beta_fast": 64.0,
                "beta_slow": 1.0,
                "attention_factor": 1.3465735902799727,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            },
        },
        "bos_token_id": 1,
        "eos_token_id": [VOCAB - 1],
        "pad_token_id": 0,
        "tie_word_embeddings": False,
        "use_cache": True,
        "torch_dtype": "float32",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="./laguna_tiny")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    try:
        import torch
        from transformers import LagunaConfig, LagunaForCausalLM
    except Exception as exc:
        raise SystemExit(
            "Laguna fixture generation needs PyTorch and a transformers build "
            "with LagunaForCausalLM (transformers >= 5.12)"
        ) from exc

    out = Path(args.output)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    cfg_dict = build_config()
    (out / "config.json").write_text(json.dumps(cfg_dict, indent=2) + "\n")

    torch.manual_seed(SEED)
    cfg = LagunaConfig(**{k: v for k, v in cfg_dict.items()
                          if k not in ("architectures", "model_type", "torch_dtype")})
    model = LagunaForCausalLM(cfg).to(torch.float32).eval()
    # Random init leaves e_score_correction_bias at zero (LagunaPreTrainedModel
    # zeroes it deliberately), which would hide a router that ignores the bias
    # entirely. Give it real values.
    with torch.no_grad():
        for layer in model.model.layers:
            gate = getattr(layer.mlp, "gate", None)
            if gate is not None and hasattr(gate, "e_score_correction_bias"):
                gate.e_score_correction_bias.copy_(
                    torch.randn_like(gate.e_score_correction_bias) * 0.1)

    model.save_pretrained(out, safe_serialization=True)
    # save_pretrained rewrites config.json from the config object; put ours back
    # so the C engine reads exactly the fields this script documents.
    (out / "config.json").write_text(json.dumps(cfg_dict, indent=2) + "\n")

    g = torch.Generator().manual_seed(SEED)
    prompt = torch.randint(2, VOCAB - 1, (1, PROMPT_LEN), generator=g)

    with torch.no_grad():
        full = model.generate(prompt, max_new_tokens=NEW_TOKENS, do_sample=False,
                              num_beams=1, use_cache=True)
        tf = model(full).logits.argmax(-1)[0].tolist()

    ref = {
        "prompt_ids": prompt[0].tolist(),
        "full_ids": full[0].tolist(),
        # tf_pred[i] is the argmax AT position i (what the engine's teacher-forced
        # pass reports), not shifted — the C harness compares position for position.
        "tf_pred": tf,
    }
    (out / "ref_laguna.json").write_text(json.dumps(ref, indent=2) + "\n")
    print(f"wrote {out}/  ({LAYERS} layers, {EXPERTS} experts, window {WINDOW})")
    print(f"prompt {PROMPT_LEN} tok -> generated {NEW_TOKENS}: {ref['full_ids'][PROMPT_LEN:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
