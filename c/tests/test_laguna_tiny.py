#!/usr/bin/env python3
"""Run a Laguna engine against a tiny fixture and assert token-exact parity.

Mirrors tests/test_deepseek_v4_tiny.py: generate the fixture with
tools/make_laguna_tiny.py, then run the engine over it and require every
teacher-forced argmax and every generated token to match the transformers
oracle. Exits non-zero on any mismatch, so it works as a CI gate.

Usage:
  python3 tests/test_laguna_tiny.py --binary ./laguna_xs --fixture ./laguna_tiny
  python3 tests/test_laguna_tiny.py --binary ./laguna_s  --fixture ./laguna_tiny --bits 8
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--cap", default="8", help="expert cache slots per layer")
    ap.add_argument("--bits", default="0", help="0 = f32 experts (bit-exact), 2..8 = int")
    args = ap.parse_args()

    binary, fixture = Path(args.binary).resolve(), Path(args.fixture).resolve()
    ref = fixture / "ref_laguna.json"
    for path in (binary, ref):
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            print("build the engine (make -C c laguna_xs) and generate the fixture "
                  "(python3 c/tools/make_laguna_tiny.py --output ./laguna_tiny --force)",
                  file=sys.stderr)
            return 2

    env = dict(os.environ, SNAP=str(fixture))
    proc = subprocess.run([str(binary), args.cap, args.bits, str(ref)],
                          env=env, capture_output=True, text=True, timeout=1800)
    out = proc.stdout
    print(out, end="")
    if proc.stderr.strip():
        print(proc.stderr, file=sys.stderr, end="")

    ok = True
    # Teacher forcing is optional in a fixture (tf_pred may be absent), but when
    # the engine reports it, every position must match — a partial match means
    # the forward pass is wrong somewhere the greedy check may not reach.
    tf = re.search(r"teacher-forced argmax: (\d+)/(\d+) match", out)
    if tf:
        got, want = int(tf.group(1)), int(tf.group(2))
        print(f"[check] teacher-forced {got}/{want}")
        ok &= got == want
    else:
        print("[check] teacher-forced pass not reported (fixture has no tf_pred)")

    gen = re.search(r"Matching tokens: (\d+)/(\d+)", out)
    if not gen:
        print("FAIL: engine printed no 'Matching tokens' line", file=sys.stderr)
        return 1
    got, want = int(gen.group(1)), int(gen.group(2))
    print(f"[check] generated {got}/{want}")
    ok &= got == want

    if proc.returncode != 0:
        print(f"FAIL: engine exited {proc.returncode}", file=sys.stderr)
        ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
