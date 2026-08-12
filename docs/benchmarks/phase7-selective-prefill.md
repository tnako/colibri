# Phase 7 — Selective-propagation prefill (LG_SELP) measured table

Model `Laguna-XS-2.1-oQ2`, Metal build, `LG_SPEC=0`, via `c/tools/stress_laguna.sh`.
Machine shared → single-run variance ±15%; the 32k runs are the most stable.

| config | prompt | prefill attn | prefill wall | peak RSS | decode (16 tok) |
|---|---|---|---|---|---|
| baseline (`LG_SELP=0`) | 16k | 141.3 s | 273.4 s | 8.70 GiB | 2.24 tok/s |
| `LG_SELP=1 cap=4096` | 16k | 116.5 s | 227.8 s | 9.03 GiB | 2.35 tok/s |
| `LG_SELP=1 cap=8192` | 16k | ~136 s | ~254 s | 9.45 GiB | 2.53 tok/s |
| baseline | 32k | 370.4 s | 649.9 s | ~9.7 GiB | 1.79 tok/s |
| `LG_SELP=1 cap=8192` | 32k | 297.1 s | 568.7 s | ~9.7 GiB | 2.26 tok/s |

32k clean A/B: attn 370.4 → 297.1 s (−20%), wall 649.9 → 568.7 s.

Remaining prefill cost is the scoring layer's full O(S²) pass + expert-MM
(153-165 s), not the skipped late layers.

Quality: ~2k prompt, greedy -n 32, first 32 tokens `cap=8192` vs baseline →
46/54 = 85.2% token agreement.
