#!/usr/bin/env bash
# Long-context stress + sampled profile for the Laguna oQ engines.
#
# Runs a real oQ checkpoint at a configurable prompt length under macOS's
# `sample` profiler, then reports the engine's own phase timers alongside the
# profiler's hottest symbols. Built for repeated optimization rounds: each run
# writes a numbered directory so rounds can be diffed.
#
# Long context is the point. A short prompt hides everything the KV cache and the
# sliding-window ring do; XS has window=512, so a prompt over that exercises the
# wrap path, and full-attention layers grow their score loop with every token.
#
# Usage:
#   tools/stress_laguna.sh <model-dir> [prompt-tokens] [gen-tokens] [tag]
#   tools/stress_laguna.sh models/Laguna-XS-2.1-oQ2 4096 32 round1
#
# Env:
#   ENGINE=./c/laguna_xs   which binary (default laguna_xs)
#   CAP=0                  expert cache slots per layer, 0 = auto
#   OUT=/tmp/lgstress      where to write results
set -euo pipefail

MODEL="${1:?usage: stress_laguna.sh <model-dir> [prompt-tokens] [gen-tokens] [tag]}"
PTOK="${2:-4096}"
NGEN="${3:-32}"
TAG="${4:-$(date +%H%M%S)}"
ENGINE="${ENGINE:-./c/laguna_xs}"
CAP="${CAP:-0}"
OUT="${OUT:-/tmp/lgstress}"

[ -x "$ENGINE" ] || { echo "no engine at $ENGINE (run: make -C c laguna_xs)"; exit 1; }
[ -f "$MODEL/config.json" ] || { echo "no checkpoint at $MODEL"; exit 1; }

D="$OUT/$TAG"
mkdir -p "$D"

# Build a prompt of roughly PTOK tokens. Real prose, not a repeated phrase: a
# repetitive filler prompt makes the model degenerate and also gives the expert
# router an unnaturally narrow expert distribution, which would flatter the LRU
# cache hit rate and hide the streaming cost we are trying to measure.
python3 - "$PTOK" > "$D/prompt.txt" <<'PY'
import sys, random
want = int(sys.argv[1])
random.seed(1234)   # fixed: every round must use the identical prompt
topics = [
 "The ring buffer wraps when the write index passes the window length, so a naive prefill that appends the whole batch before scoring can overwrite rows an earlier query still needs.",
 "Grouped affine quantization stores one scale and one bias per group of inputs, which means the dequantization can be factored out of the inner product and applied once per group.",
 "A mixture-of-experts router picks the top k experts per token, so the working set at decode time is k matrices per layer rather than the full expert bank.",
 "Sliding-window attention bounds the key-value cache to a fixed number of rows per layer, trading unlimited history for predictable memory.",
 "YaRN rescales rotary position frequencies by interpolating the wavelength across dimensions, extending usable context far past the original training length.",
 "When weights are streamed from disk the critical path is dominated by read bandwidth, so a smaller on-disk representation matters more than the arithmetic cost of unpacking it.",
 "Speculative decoding drafts several tokens with a small model and verifies them in one forward pass of the large model, trading extra compute for lower latency.",
 "Unified memory on Apple silicon removes the host-to-device copy, but a kernel dispatch still costs hundreds of microseconds of round-trip latency.",
]
# ~1.3 tokens/word is a safe estimate for this tokenizer; overshoot slightly.
words = []
while len(words) < want * 0.8:
    words.extend(random.choice(topics).split())
print(" ".join(words))
PY

WORDS=$(wc -w < "$D/prompt.txt" | tr -d ' ')
echo "==> stress $TAG: $MODEL, ~$PTOK tokens ($WORDS words), $NGEN generated, cap=$CAP"

# Run the engine in the background so `sample` can attach to a live pid, and let
# the profiler cover the whole run rather than a fixed slice.
# /usr/bin/time -l gives peak RSS + page-in/fault counters on macOS. It must NOT
# be the process `sample` attaches to, or the profile is 100% __sigsuspend in the
# time wrapper; run the engine directly and read RSS from the engine's own line
# plus a footprint poll.
SNAP="$MODEL" "$ENGINE" "$CAP" 0 --chat -n "$NGEN" -f "$D/prompt.txt" \
     > "$D/engine.log" 2> "$D/engine.err" &
PID=$!
# poll peak footprint while it runs (cheap, 1 Hz)
( peak=0
  while kill -0 "$PID" 2>/dev/null; do
    r=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ')
    [ -n "$r" ] && [ "$r" -gt "$peak" ] && peak=$r
    sleep 1
  done
  echo "peak_rss_kb $peak" > "$D/peak_rss" ) &
RSSPID=$!

# 1 ms sampling for up to an hour: a 16K-token prefill runs many minutes and the
# old 600s cap silently ended the profile (and this script) mid-run.
sample "$PID" "${SAMPLE_SECS:-3600}" 1 -file "$D/sample.txt" >/dev/null 2>&1 &
SPID=$!

wait "$PID" 2>/dev/null || { echo "!! engine exited nonzero"; tail -20 "$D/engine.err"; }
wait "$SPID" 2>/dev/null || true

echo
echo "--- engine ---"
# The engine echoes the whole prompt; cut it to the counters we care about.
grep -hoE "\[[0-9]+ prompt tokens\]|prefill [0-9.]+s.*|\[phases\].*|oQ: .*" \
  "$D/engine.log" "$D/engine.err" 2>/dev/null | sort -u || true

wait "$RSSPID" 2>/dev/null || true
echo
echo "--- memory ---"
if [ -f "$D/peak_rss" ]; then
  awk '{printf "  peak RSS: %.2f GiB\n", $2/1048576}' "$D/peak_rss"
fi
grep -hoE "RSS [0-9.]+ GB" "$D/engine.log" 2>/dev/null | tail -1 | sed 's/^/  final /' || true

echo
echo "--- self time by symbol (sampled) ---"
# `sample` prints an indented call TREE, so a parent's count includes all its
# children -- summing every line double-counts massively. Self time is
# (own count) - (sum of children's counts), which needs the indent depth.
python3 - "$D/sample.txt" <<'PY'
import re, sys, collections
lines = open(sys.argv[1], errors="ignore").read().splitlines()
# keep only the call-graph section
try:
    start = next(i for i,l in enumerate(lines) if l.startswith("Call graph"))
    end   = next(i for i,l in enumerate(lines) if l.startswith("Total number in stack"))
except StopIteration:
    print("  (unexpected sample format)"); raise SystemExit
pat = re.compile(r'^(?P<pre>[\s+!:|]*)(?P<n>\d+)\s+(?P<sym>.+?)\s+\(in (?P<bin>[^)]+)\)')
rows = []
for l in lines[start:end]:
    m = pat.match(l)
    if m:
        rows.append((len(m.group('pre')), int(m.group('n')),
                     m.group('sym').strip(), m.group('bin')))
# self = own samples minus the samples of its immediate children
self_t = collections.Counter()
for i,(d,n,sym,bin_) in enumerate(rows):
    kids = 0; child_depth = None
    for d2,n2,_,_ in rows[i+1:]:
        if d2 <= d: break                  # left this subtree
        if child_depth is None: child_depth = d2
        if d2 == child_depth: kids += n2   # immediate children only
    self_t[(sym,bin_)] += max(0, n - kids)
tot = sum(self_t.values()) or 1
print(f"  (total self samples {tot})")
for (sym,bin_),w in self_t.most_common(16):
    tag = "" if 'laguna' in bin_ else f"  [{bin_.split('/')[-1]}]"
    print(f"  {100.0*w/tot:5.1f}%  {w:7d}  {sym}{tag}")
PY
echo
echo "results in $D"
