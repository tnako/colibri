#!/bin/bash
# Sweep LG_CHUNK on Laguna-XS at ~2k tokens. Quick (each run is under a minute),
# which is the point: chunk size trades scratch memory against how many rows each
# expert GEMM sees, and the optimum is empirical.
#
# Uses the RUNTIME override (LG_CHUNK env var, lg_chunk() in laguna_common.h) on
# the prebuilt Metal binary -- no recompile per point. Requires
# `make -C c laguna_xs_metal` first.
#
# Usage: c/tools/tune_chunk.sh [model] [prompt]
set -e
cd "$(dirname "$0")/../.."
MODEL="${1:-models/Laguna-XS-2.1-oQ2}"
PROMPT="${2:-/tmp/lgstress/round5/prompt.txt}"
ENGINE="${ENGINE:-./c/laguna_xs_metal}"

[ -x "$ENGINE" ] || { echo "no engine at $ENGINE (run: make -C c laguna_xs_metal)"; exit 1; }

printf "%8s %10s %10s %10s %10s %9s\n" chunk prefill expert-mm attn RSS tok/s
for CH in 256 512 1024 2048 4096 8192; do
  OUT=$(LG_CHUNK=$CH SNAP="$MODEL" "$ENGINE" 0 0 --chat -n 4 -f "$PROMPT" 2>&1 | tr '\r' '\n')
  P=$(echo "$OUT"  | grep -oE "prefill [0-9.]+s"       | grep -oE "[0-9.]+" | tail -1)
  E=$(echo "$OUT"  | grep -oE "expert-mm [0-9.]+s"     | grep -oE "[0-9.]+" | tail -1)
  A=$(echo "$OUT"  | grep -oE "attn [0-9.]+s"          | grep -oE "[0-9.]+" | tail -1)
  R=$(echo "$OUT"  | grep -oE "RSS [0-9.]+ GB"         | grep -oE "[0-9.]+" | tail -1)
  T=$(echo "$OUT"  | grep -oE "= [0-9.]+ tok/s"        | grep -oE "[0-9.]+" | tail -1)
  printf "%8d %9ss %9ss %9ss %8sGB %9s\n" "$CH" "${P:--}" "${E:--}" "${A:--}" "${R:--}" "${T:--}"
done
