#!/usr/bin/env python3
"""Phase 3 selection-pass functional gate.

Drives an engine binary in prompt mode (-f) against a tokenized model dir and
asserts the post-prefill selection pass behaves as designed:

  1. Baseline (no LG_SEL): capture the generated assistant output.
  2. LG_SEL=1 LG_SEL_MIN=1 LG_SEL_CAP=<huge>: the pass engages but selects every
     position, so the generated output must be byte-identical to the baseline
     (a strong check that the selection walk computes the same attention when
     nothing is dropped). Must print the "[sel] cap=..." line.
  3. LG_SEL=1 LG_SEL_MIN=1 LG_SEL_CAP=64: engages with a real cap; must print
     the "[sel] cap=..." line, run to completion, and exit 0.

Usage:
  python3 tests/test_selection.py --binary ./laguna_xs_metal --model ./models/Laguna-XS-2.1-oQ2
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

NOISE_RE = re.compile(r"^(\[sel\] |\[prefill .*\]|\[.*prompt tokens\]|"
                      r"\[phases\] .*|\[spec\] .*|\s*sel: ON .*|"
                      r"resident weights loaded in .*\| RSS .*)")


def make_prompt(words: int) -> str:
    rng_lo, rng_hi = 11000, 12000
    import random
    rng = random.Random(7)
    vocab = ["kernel", "matrix", "tensor", "stream", "decode", "attention",
             "cache", "prefill", "vector", "neural"]
    return " ".join(rng.choice(vocab) for _ in range(words)) + " Summarize:"


def run(binary, model, prompt, extra_env) -> subprocess.CompletedProcess:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        pf = f.name
    env = dict(os.environ, SNAP=str(model), **extra_env)
    try:
        return subprocess.run([str(binary), "0", "0", "--chat", "-n", "8",
                               "-f", pf], env=env, capture_output=True,
                              text=True, timeout=3600)
    finally:
        os.unlink(pf)


def strip_noise(out: str) -> str:
    return "\n".join(l for l in out.splitlines() if not NOISE_RE.match(l))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=1024,
                    help="synthetic prompt length above the sel_min override")
    args = ap.parse_args()

    binary, model = Path(args.binary).resolve(), Path(args.model).resolve()
    if not binary.exists() or not (model / "config.json").exists():
        print("missing --binary or --model", file=sys.stderr)
        return 2

    prompt = make_prompt(args.tokens)
    # Obs window must cover the whole context so every block scores > 0 and a
    # huge cap can select ALL positions (byte-exact vs baseline). sel_min must
    # stay <= kv_len or sel_pass bails; the chat wrap adds a few tokens.
    sel_min = str(args.tokens)

    base = run(binary, model, prompt, {})
    if base.returncode != 0:
        print("FAIL: baseline run exited non-zero", file=sys.stderr)
        print(base.stdout, file=sys.stderr)
        return 1
    base_out = strip_noise(base.stdout)
    print("[check] baseline ok")

    full = run(binary, model, prompt,
               {"LG_SEL": "1", "LG_SEL_MIN": sel_min, "LG_SEL_CAP": "1048576"})
    if full.returncode != 0:
        print("FAIL: full-cap selection run exited non-zero", file=sys.stderr)
        print(full.stdout, file=sys.stderr)
        return 1
    if "sel] cap=" not in full.stdout:
        print("FAIL: full-cap run did not engage selection ([sel] cap= missing)",
              file=sys.stderr)
        return 1
    if strip_noise(full.stdout) != base_out:
        print("FAIL: full-cap selection changed generation (must be byte-exact "
              "when every position is selected)", file=sys.stderr)
        return 1
    print("[check] full-cap selection byte-exact vs baseline: ok")

    small = run(binary, model, prompt,
                {"LG_SEL": "1", "LG_SEL_MIN": sel_min, "LG_SEL_CAP": "64"})
    if small.returncode != 0:
        print("FAIL: capped selection run exited non-zero", file=sys.stderr)
        print(small.stdout, file=sys.stderr)
        return 1
    if "sel] cap=" not in small.stdout:
        print("FAIL: capped run did not engage selection", file=sys.stderr)
        return 1
    print("[check] capped selection engages and completes: ok")

    print("selection gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
