#!/usr/bin/env python3
"""Probe an oQ (oMLX affine-quant) checkpoint and verify the layout assumptions.

Reads only config.json and the safetensors HEADERS -- a ranged GET of a few
hundred KB, not the multi-GB weights -- so it is cheap to run against any oQ
repo on the Hub before committing to a download.

What it checks, per quantized tensor:
  words == K*bits/32          (dense bitstream, no per-word padding)
  K     == ngroups*group_size (scales/biases shape agrees with the packed width)
  group_size*bits % 32 == 0   (every group starts on a word boundary)

The third is the one that makes a C unpacker tractable for bits=3 and bits=6,
where values straddle word boundaries; if it ever fails for some checkpoint the
unpacker needs cross-group carry and this script should say so loudly.

Usage:
  python3 tools/probe_oq.py mlx-community/Laguna-S-2.1-oQ2e-fast
  python3 tools/probe_oq.py /path/to/local/oq-model
  python3 tools/probe_oq.py <repo> --shards 3      # inspect more shards
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import urllib.request
from collections import Counter
from pathlib import Path


def fetch_bytes(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def read_json_url(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def st_header_remote(url: str) -> dict:
    n = struct.unpack("<Q", fetch_bytes(url, 0, 7))[0]
    if not 0 < n < 200_000_000:
        raise SystemExit(f"implausible safetensors header length {n} at {url}\n"
                         "  (a 404/HTML error page parses as garbage here -- check the filename)")
    return json.loads(fetch_bytes(url, 8, 8 + n - 1))


def st_header_local(path: Path) -> dict:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="HF repo id or local directory")
    ap.add_argument("--shards", type=int, default=1, help="how many shards to inspect")
    args = ap.parse_args()

    local = Path(args.model)
    if local.is_dir():
        cfg = json.loads((local / "config.json").read_text())
        shards = sorted(p.name for p in local.glob("*.safetensors"))
        get_header = lambda name: st_header_local(local / name)
    else:
        repo = args.model
        cfg = read_json_url(f"https://huggingface.co/{repo}/raw/main/config.json")
        api = read_json_url(f"https://huggingface.co/api/models/{repo}")
        shards = sorted(s["rfilename"] for s in api.get("siblings", [])
                        if s["rfilename"].endswith(".safetensors"))
        base = f"https://huggingface.co/{repo}/resolve/main"
        get_header = lambda name: st_header_remote(f"{base}/{name}")

    q = cfg.get("quantization") or cfg.get("quantization_config")
    if not isinstance(q, dict):
        print("no `quantization` block: this is not an oQ/MLX-quantized checkpoint")
        return 1

    default = {"bits": q.get("bits"), "group_size": q.get("group_size"),
               "mode": q.get("mode")}
    overrides = {k: v for k, v in q.items() if isinstance(v, dict)}
    print(f"model: {args.model}")
    print(f"global default: {default}")
    print(f"per-tensor overrides: {len(overrides)}")

    combos = Counter((v.get("bits"), v.get("group_size"), v.get("mode"))
                     for v in overrides.values())
    print("\n(bits, group_size, mode) across overrides:")
    for c, n in combos.most_common():
        print(f"  {c}: {n}")

    modes = {v.get("mode") for v in overrides.values()} | {default["mode"]}
    if modes - {"affine"}:
        print(f"\n!! non-affine modes present: {modes - {'affine'}}")
        print("   the dequant formula w = q*scale + bias applies to `affine` only")

    checked = fails = 0
    widths, unaligned = Counter(), set()
    for name in shards[: args.shards]:
        hdr = get_header(name)
        for k in sorted(hdr):
            if k == "__metadata__" or not k.endswith(".weight"):
                continue
            if hdr[k]["dtype"] != "U32":
                continue                      # unquantized (norms, router proj)
            stem = k[: -len(".weight")]
            sc = hdr.get(stem + ".scales")
            if not sc:
                print(f"  !! {stem}: U32 weight with no .scales")
                fails += 1
                continue
            c = overrides.get(stem, default)
            bits, gs = c["bits"], c["group_size"]
            words = hdr[k]["shape"][-1]
            K = sc["shape"][-1] * gs
            widths[bits] += 1
            if (gs * bits) % 32:
                unaligned.add((bits, gs))
            if K * bits // 32 != words or K * bits % 32:
                print(f"  MISMATCH {stem}: bits={bits} gs={gs} K={K} "
                      f"words={words} expected={K*bits/32}")
                fails += 1
            checked += 1

    print(f"\nchecked {checked} quantized tensors across {min(args.shards, len(shards))} "
          f"shard(s) of {len(shards)}")
    print(f"bit widths seen: {dict(sorted(widths.items()))}")
    if unaligned:
        print(f"!! groups NOT word-aligned for {sorted(unaligned)} -- a C unpacker "
              f"would need cross-group bit carry")
    else:
        print("all groups word-aligned: per-group unpack needs no cross-group carry")
    print("PASS" if fails == 0 else f"FAIL ({fails} inconsistencies)")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
