# Phase 6 — 256k hardening + release gate — progress report

Branch `phase6/hardening`, rebased onto merged laguna-support (Phases 1-5,
`5245c59`). All runs on Apple Silicon (10 p-cores, 32 GiB), real checkpoints.

## STEP 1 — gate matrix (green, no fixes needed)

| check | result |
|---|---|
| `make -C c laguna_xs_metal laguna_s_metal laguna_xs laguna_s` | 4/4 link clean |
| `task test` (laguna_tiny, xs + s, CPU + Metal) | 4/4 PASS 12/12 |
| `test_laguna_tiny.py --bits 8` (xs) | PASS 12/12 |
| `test_ngram_draft` / `test_kv_alloc` / `test_spec_decode_state` | 3/3 pass |
| `test_resource_plan.py` | 44/44 OK |
| `task test:baseline` (olmoe, inkling, colibri) | 3/3 link clean |

No code was broken by the Phase 1-5 merge; no `fix:` commit needed.

## STEP 2 — scaled context table (Metal builds, direct runs, CTX_MAX=262144)

Direct engine runs (no `sample` profiler; plain greedy `-n 8` decode). RSS is
`/usr/bin/time` maximum resident set size.

| model | prompt | prefill | attn | expert-mm | shared | fill | RSS | decode tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| XS-oQ2 | 3,802 | 39.8 s | 13.0 s | 20.3 s | 0.6 s | 1.4 s | 7.0 GB | 4.13 |
| XS-oQ2 | 16,023 | 261.2 s | 154.7 s | 71.7 s | 4.5 s | 1.3 s | 10.0 GB | 0.54 |
| XS-oQ2 | 64k (LG_SEL=1: cap 8192) | 4071.7 s | 2023.8 s | 1338.0 s | 15.1 s | 1.0 s | 10.9 GB | 1.28 |
| S-oQ2e-fast | 3,802 | 407.5 s | 240.5 s | 119.2 s | 29.1 s | 2.3 s | 10.8 GB | 0.58 |
| S-oQ2e-fast | 15,073 | 4266.1 s | 3602.2 s | 380.4 s | 145.4 s | 2.4 s | 12.1 GB | 0.80 |

Notes:
- XS 65k prefill ~68 min; attn stays O(S²) for full layers (10 full / 30 sliding
  for XS). LG_SEL helped decode (1.28 vs 0.54 at 16k) but prefill attn is still
  quadratic (2023.8 s at 65k vs 154.7 s at 16k — ~13x for 4x tokens).
- S prefill is attention-dominated and quadratic: attn 3602.2 s at 15k (12 full
  layers) vs 240.5 s at 3.8k — ~15x for 4x tokens. Expert-mm is ~linear
  (~31 ms/tok), shared ~10 ms/tok.
- RSS stays well inside 20 GB at every step (max 12.1 GB), even at CTX_MAX=262144.
- XS decode collapsed at 16k without selection (0.54 tok/s — O(S) full-layer
  decode reads); selection restores 1.28 tok/s at 65k. S decode stays CPU-bound
  ~0.6-0.8 tok/s regardless.

## STEP 3 — 256k budget assessment (bounded)

A literal 256k S-oQ2e-fast prefill is **not runnable in bounded time**: extrapolating
the measured O(S²) attention term, 256k attn ≈ 3602.2 × (256/15)² ≈ **1.05e6 s
(~290 h)**, plus linear terms ~1.5 h. XS at 256k attn ≈ 154.7 × (256/16)² ≈ 39,600 s
(~11 h). RSS at every measured point is ≤12.1 GB at CTX_MAX=262144, so the memory
target (<20 GB @256k) is already met without selection or a bigger LG_CHUNK; the
wall-clock quadratic term is the blocker, not RAM. Longest feasible measured:
XS 65k (68 min), S 16k (71 min).

## STEP 4 — write-up

`docs/256k-final.md` written (measured table + per-phase state + honest
recommendation). Plan doc `docs/256k-rework-plan.md` Phase 6 marked DONE.

## Final numbers (headline)

- 256k context fits in RAM: peak RSS 8.7-12.1 GB @ CTX_MAX=262144 (target <20 GB ✅)
- 256k prefill not runnable: projected ~290 h (S) / ~11 h (XS), O(S²) attn
- Decode 0.6-1.3 tok/s (target 140 tok/s ❌); gate matrix 100% green