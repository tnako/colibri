#!/usr/bin/env bash
# sweep_spec.sh — A/B sweep of speculative-decode depth (LG_SPEC_MAX) on one
# model. Reuses bench_laguna.sh, prints a compact table: for each depth the
# decode tok/s and the draft acceptance rate.
#
# Usage:
#   tools/sweep_spec.sh <model-dir> [prompt-tokens] [gen-tokens] [tag]
#
# Env:
#   PROSE=1        prose prompt (n-gram-poor) vs code-like (n-gram-rich)
#   DEPTHS="0 2 4 8 12 16 24"   depths to sweep (0 = plain loop, LG_SPEC=0)
#   ENGINE, CAP, CTX_MAX, OUT   forwarded to bench_laguna.sh
#   LG_DEC_GPU=1 LG_DEC_EXP_ON=1   (optional, Metal decode session batching)
set -euo pipefail

MODEL="${1:?usage: sweep_spec.sh <model-dir> [prompt-tokens] [gen-tokens] [tag]}"
PTOK="${2:-2048}"
NGEN="${3:-128}"
TAG="${4:-sweep-$(date +%H%M%S)}"
PROSE="${PROSE:-1}"
DEPTHS="${DEPTHS:-0 2 4 8 12 16 24}"
OUT="${OUT:-/tmp/lgbench}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

[ -f "$MODEL/config.json" ] || { echo "!! no checkpoint at $MODEL" >&2; exit 1; }

D="$OUT/$TAG"
mkdir -p "$D"
echo "# spec-depth sweep $MODEL  prompt~$PTOK gen=$NGEN prose=$PROSE" | tee "$D/sweep.txt"
printf '%-6s %-10s %-10s %-12s %-10s\n' depth toks acc% rounds | tee -a "$D/sweep.txt"

for d in $DEPTHS; do
  round="depth$d"
  if [ "$d" = "0" ]; then
    LG_SPEC=0 OUT="$OUT" bash "$ROOT/tools/bench_laguna.sh" "$MODEL" "$PTOK" "$NGEN" "$TAG/$round" \
      > "$D/$round.out" 2>&1 || true
    tok=$(grep -oE 'tok/s=[0-9.]+' "$D/$round.out" | sed 's/tok\/s=//' | tail -1 || true)
    acc=-
  else
    LG_SPEC_MAX="$d" OUT="$OUT" bash "$ROOT/tools/bench_laguna.sh" "$MODEL" "$PTOK" "$NGEN" "$TAG/$round" \
      > "$D/$round.out" 2>&1 || true
    tok=$(grep -oE 'tok/s=[0-9.]+' "$D/$round.out" | sed 's/tok\/s=//' | tail -1 || true)
    acc=$(grep -oE 'accepted \([0-9.]+%\)' "$D/$round.out" | tail -1 | sed -E 's/.*\(([0-9.]+)%\).*/\1/' || true)
  fi
  rnds=$(grep -oE '[0-9]+ rounds' "$D/$round.out" | tail -1 | awk '{print $1}' || true)
  printf '%-6s %-10s %-10s %-12s\n' "$d" "${tok:--}" "${acc:--}" "${rnds:-0}" | tee -a "$D/sweep.txt"
done

echo "==> sweep results: $D/sweep.txt"