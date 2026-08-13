#!/usr/bin/env bash
# bench_laguna.sh — prefill + decode benchmark for the Laguna oQ engines.
#
# Measures one configuration and prints a machine-readable summary plus the
# engine's own phase timers. Two runs per invocation: a prefill-only run
# (-n 2) and a decode run (-n $NGEN), so a subagent or the sweep harness can
# subtract the prefill phase split and isolate the per-token decode costs.
#
# Usage:
#   tools/bench_laguna.sh <model-dir> [prompt-tokens] [gen-tokens] [tag]
#   tools/bench_laguna.sh models/Laguna-XS-2.1-oQ2 2048 128 round1
#
# Env:
#   ENGINE=./c/laguna_xs_metal   engine binary (default: laguna_xs_metal,
#                                falls back to laguna_xs)
#   CAP=0                        expert cache slots per layer, 0 = auto
#   CTX_MAX=8192                 context bound (default 8192; 0 = engine default)
#   LAGUNA_MEM_GB=20             memory knob (0 = engine default)
#   PROSE=1                      prompt flavor: 1 prose (n-gram-poor, hide from
#                                spec) or 0 code-like (n-gram-rich, spec flatters)
#   SEED=1234                    prompt RNG seed (identical prompt across rounds)
#   OUT=/tmp/lgbench              output dir
#   Any LG_SPEC_MAX / LG_SPEC / LG_* value is inherited by the engine.
set -euo pipefail

MODEL="${1:?usage: bench_laguna.sh <model-dir> [prompt-tokens] [gen-tokens] [tag]}"
PTOK="${2:-2048}"
NGEN="${3:-128}"
TAG="${4:-$(date +%H%M%S)}"
if [ -x "$(dirname "$0")/../laguna_xs_metal" ]; then
  ENGINE="${ENGINE:-$(dirname "$0")/../laguna_xs_metal}"
else
  ENGINE="${ENGINE:-$(dirname "$0")/../laguna_xs}"
fi
CAP="${CAP:-0}"
CTX="${CTX_MAX:-8192}"
MEM="${LAGUNA_MEM_GB:-20}"
SEED="${SEED:-1234}"
PROSE="${PROSE:-1}"
OUT="${OUT:-/tmp/lgbench}"

[ -x "$ENGINE" ] || { echo "!! no engine at $ENGINE (make -C c laguna_xs laguna_xs_metal)" >&2; exit 1; }
[ -f "$MODEL/config.json" ] || { echo "!! no checkpoint at $MODEL" >&2; exit 1; }

D="$OUT/$TAG"
mkdir -p "$D"

# --- deterministic prompt: ~PTOK tokens, prose or code-like -----------------
python3 - "$PTOK" "$PROSE" "$SEED" > "$D/prompt.txt" <<'PY'
import sys, random
want, prose, seed = int(sys.argv[1]), int(sys.argv[2]) == 1, int(sys.argv[3])
random.seed(seed)
prose_topics = [
 "The ring buffer wraps when the write index passes the window length, so a naive prefill that appends the whole batch before scoring can overwrite rows an earlier query still needs.",
 "Grouped affine quantization stores one scale and one bias per group of inputs, which means the dequantization can be factored out of the inner product and applied once per group.",
 "A mixture-of-experts router picks the top k experts per token, so the working set at decode time is k matrices per layer rather than the full expert bank.",
 "Sliding-window attention bounds the key-value cache to a fixed number of rows per layer, trading unlimited history for predictable memory.",
 "YaRN rescales rotary position frequencies by interpolating the wavelength across dimensions, extending usable context far past the original training length.",
 "When weights are streamed from disk the critical path is dominated by read bandwidth, so a smaller on-disk representation matters more than the arithmetic cost of unpacking it.",
 "Speculative decoding drafts several tokens with a small model and verifies them in one forward pass of the large model, trading extra compute for lower latency.",
 "Unified memory on Apple silicon removes the host-to-device copy, but a kernel dispatch still costs hundreds of microseconds of round-trip latency.",
]
code_topics = [
 "def ring_push(buf, w, r, win):\n    if r >= win: r = 0\n    buf[r] = w; return r + 1",
 "scale = w_off * 0x3f + 1\nk = (x << 8) >> 6\nacc += scale * k",
 "router = softmax(nn.Linear(D, E)(x))\ntop = router.topk(k).indices\nunion = set(top.tolist())",
 "k = (offset + i) & mask\nv = cache[k, :]\nscore = q.dot(k) * rsqrt(hd)",
 "freq = theta * (1 / base ** (arange(0, hd, 2) / hd))\npos *= freq",
 "evict = lru.popitem(last=False)\nprefetch(queue.get(block=False))",
 "for d in draft:\n    batch.append(tok(d))\nlogit = model(batch)",
 "mtl = MetalDevice()\nmtl.commit(use_fence=True)\nwait = mtl.fence.wait()",
]
if not prose:
    # code-like: repeat structured snippets so the n-gram draft fires often
    snips = code_topics * 3
    lines = [random.choice(snips) for _ in range(max(1, want // 12))]
    sys.stdout.write("\n".join(lines) + "\n")
    raise SystemExit
words = []
while len(words) < want * 0.8:
    words.extend(random.choice(prose_topics).split())
sys.stdout.write(" ".join(words))
PY

WORDS=$(wc -w < "$D/prompt.txt" | tr -d ' ')
echo "==> bench '$TAG': $MODEL, ~$PTOK tok prompt ($WORDS words), gen=$NGEN, cap=$CAP, ctx=$CTX, mem=$MEM, engine=$(basename "$ENGINE")"

result="$D/summary.txt"
: > "$result"

run_one() {  # $1=ngen  $2=outfile  $3=label
  local ngen="$1" outf="$2" label="$3"
  local pre="" envstr=()
  [ "$CTX" != "0" ] && envstr+=(CTX_MAX="$CTX")
  [ "$MEM" != "0" ] && envstr+=(LAGUNA_MEM_GB="$MEM")
  GLOBIGNORE=*
  env "${envstr[@]}" SNAP="$MODEL" "$ENGINE" "$CAP" 0 --chat -n "$ngen" -f "$D/prompt.txt" \
    > "$outf.log" 2> "$outf.err" || { echo "  !! run failed (exit $?)"; return 1; }
  pre=$(grep -oE '[0-9]+ prompt tokens' "$outf.log" | head -1 || true)
  local pf tt gen tok spec
  pf=$(grep -oE '\[prefill [0-9.]+s' "$outf.log" | head -1 | sed 's/\[prefill //;s/s//' || true)
  tt=$(grep -E 'tok/s' "$outf.log" | grep -oE '[0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s' | head -1 || true)
  gen=$(grep -oE 'in [0-9.]+s = [0-9.]+ tok/s' <<<"$tt" | sed 's/in //;s/ = .*//;s/s//' || true)
  tok=$(grep -oE '= [0-9.]+ tok/s' <<<"$tt" | sed 's/= //;s/ tok\/s//' || true)
  spec=$(grep -hE '\[spec\]' "$outf.log" "$outf.err" 2>/dev/null | head -2 || true)
  local phases fill exp sh attn hit rss
  phases=$(grep -oE '\[phases\].*' "$outf.log" | head -1 || true)
  fill=$(grep -oE 'fill [0-9.]+s' <<<"$phases" | head -1 | sed 's/fill //;s/s//' || true)
  exp=$(grep -oE 'expert-mm [0-9.]+s' <<<"$phases" | head -1 | sed 's/expert-mm //;s/s//' || true)
  sh=$(grep -oE 'shared [0-9.]+s' <<<"$phases" | head -1 | sed 's/shared //;s/s//' || true)
  attn=$(grep -oE 'attn [0-9.]+s' <<<"$phases" | head -1 | sed 's/attn //;s/s//' || true)
  hit=$(grep -oE 'hit [0-9.]+%' <<<"$phases" | head -1 | sed 's/hit //;s/%//' || true)
  echo "  $label: prefill=$pf s | $gen gen in $tt | tok/s=$tok | fill=$fill expert=$exp shared=$sh attn=$attn hit=$hit%"
  if [ -n "$spec" ]; then
    echo "    $(echo "$spec" | tr '\n' ' ')"
    sacc=$(echo "$spec" | head -1 | sed -E 's/.*accepted \(([0-9.]+)%\).*/\1/')
    sacc=${sacc:-'?'}
  else
    sacc=0
  fi
  rss=$(grep -oE 'RSS [0-9.]+ GB' "$outf.log" | tail -1 | awk '{print $2}')
  echo "$label prefill $pf gen $gen toks $tok fill $fill expert $exp shared $sh attn $attn hit $hit rss $rss specacc $sacc" >> "$result"
  return 0
}

run_one 2      "$D/prefill"   "prefill-only"
run_one "$NGEN" "$D/decode"   "decode"

echo "  summary: $result"
if [ -s "$result" ]; then
  # decode per-token phase deltas = decode-run phases minus prefill-run phases
  python3 - "$result" "$NGEN" <<'PY'
import sys
lines = [l.split() for l in open(sys.argv[1])]
ng = int(sys.argv[2])
def row(kind):
    for l in lines:
        if l[0] == kind:
            return {l[i]: float(l[i+1]) for i in range(1, len(l)-1, 2)}
p, d = row("prefill-only"), row("decode")
print("  per-token decode (deltas): tok/s=%.2f  fill=%.1fms  expert=%.1fms  shared=%.1fms  attn=%.1fms"
      % (d["toks"], 1e3*(d["fill"]-p["fill"])/ng, 1e3*(d["expert"]-p["expert"])/ng,
         1e3*(d["shared"]-p["shared"])/ng, 1e3*(d["attn"]-p["attn"])/ng))
PY
fi
echo "==> done: results in $D"