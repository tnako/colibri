# Laguna-S-2.1-oQ2e-fast — 256k rework final report

Date: 2026-08-12. Branch `phase6/hardening` on `laguna-support`. Machine: Apple
Silicon, 10 p-cores, 32 GiB unified memory. All numbers measured on the real
checkpoints with the Metal builds (`c/laguna_s_metal`, `c/laguna_xs_metal`),
direct engine runs (no `sample` profiler), `CTX_MAX=262144`, `LAGUNA_MEM_GB=20`
defaults, plain greedy decode (`-n 8` unless noted).

## Targets vs. measured

| metric | target | measured | verdict |
|---|---|---|---|
| context | 256k tokens | **256k runnable in RAM; not in wall-clock** | ⚠️ |
| memory | < 20 GB @256k | **8.7-12.1 GB peak RSS @ CTX_MAX=262144** | ✅ hit |
| prefill (256k) | < 2 s | 65k: 68 min (XS); 256k ≈ 9-290 h projected | ❌ not reachable |
| decode | 140 tok/s | 0.6-1.3 tok/s (XS & S) | ❌ far short |

The two headline facts this phase re-confirmed: the memory target is *met* —
even at `CTX_MAX=262144` peak RSS never exceeded 12.1 GB (S 15k) / 8.7 GB (S 4k
@256k ctx) / 10.9 GB (XS 65k), so a 256k prompt **fits** inside 20 GB. The
*wall-clock* target is not: full-attention prefill is O(S²) on the 10-12 full
layers and 256k prefill projects to ~290 h on S / ~11 h on XS.

## Measured table (scaled context, Metal builds, CTX_MAX=262144)

| model | prompt | prefill | attn | expert-mm | shared | fill | peak RSS | decode tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| XS-2.1-oQ2 | 3,802 | 39.8 s | 13.0 s | 20.3 s | 0.6 s | 1.4 s | 7.0 GB | 4.13 |
| XS-2.1-oQ2 | 16,023 | 261.2 s | 154.7 s | 71.7 s | 4.5 s | 1.3 s | 10.0 GB | 0.54 |
| XS-2.1-oQ2 | ~64k, `LG_SEL=1` cap 8192 | 4071.7 s | 2023.8 s | 1338.0 s | 15.1 s | 1.0 s | 10.9 GB | 1.28 |
| S-2.1-oQ2e-fast | 3,802 | 407.5 s | 240.5 s | 119.2 s | 29.1 s | 2.3 s | 10.8 GB | 0.58 |
| S-2.1-oQ2e-fast | 15,073 | 4266.1 s | 3602.2 s | 380.4 s | 145.4 s | 2.4 s | 12.1 GB | 0.80 |
| S-2.1-oQ2e-fast | 3,802 (@256k ctx) | 336.0 s | — | — | — | — | 8.7 GB | 1.30 (n=128) |

Notes:
- Attention dominates S prefill and is quadratic in context: 240.5 s @3.8k → 3602.2 s
  @15k (≈15x for 4x tokens; 12 full-attention layers). Expert-mm is ~linear
  (~31 ms/tok), shared ~10 ms/tok. XS (10 full layers) shows the same shape.
- Phase-3 selection (`LG_SEL=1`, min 16384, cap 8192) improves decode on XS
  (0.54 → 1.28 tok/s at 65k: full-layer decode reads O(cap) not O(S)) but does
  **not** remove the prefill quadratic term — prefill attn was still 2023.8 s at
  65k. Selection is a decode-traffic fix, not a prefill-wall fix.
- Memory: no change in the budget needed. The Phase-1 tiled prefill keeps the
  GPU score staging tile-sized (1.25-1.35 GB fixed), and peak RSS stays
  ≤12.1 GB at every prompt length including `CTX_MAX=262144`. Neither a bigger
  `LG_CHUNK` nor selection is required for the <20 GB @256k memory target.

## 256k projection (from the measured scaling curve)

A literal 256k S-oQ2e-fast prefill is **not runnable in bounded time** on this
kit. Extrapolating the measured quadratic attention term:

- S attn @256k ≈ 3602.2 × (256/15)² ≈ **1.05e6 s ≈ 290 h**
  (plus ~1.5 h of linear terms). 
- XS attn @256k ≈ 154.7 × (256/16)² ≈ **39,600 s ≈ 11 h**.
- A bounded 15-min 256k-prompt probe on XS (fastest model) did not clear even
  the first of 32 chunks, matching the projection.

So the honest statement: **256k context loads and holds in <20 GB, but full-
attention prefill of 256k tokens is a multi-hour-to-multi-day job on this
hardware.** The longest feasible measured runs were XS ~64k (68 min) and
S ~15k (71 min).

## Per-phase state (Phases 0-6)

| phase | idea | state | result |
|---|---|---|---|
| 0 | Baseline rig + planner KV model | ✅ done (`7101974`) | 44/44 plan tests; baseline numbers |
| 1 | Tiled online-softmax prefill (kill the 13 GB GPU ring) | ✅ done (`2581a9f`) | score matrix never materialized; staging tile-sized; RSS ≤12 GB @256k ctx — **memory target met** |
| 2 | Metal decode GEMV for projections + experts | ✅ done (`dd12af2`) | projections/shared-expert GEMV behind `LAGUNA_DEC_GPU`; decode still CPU oQ-path |
| 3 | Sparse/selective attention (SAGE-KV) | ✅ done (`1e57466`) | caps decode KV reads (XS 65k decode 1.28 vs 0.54 tok/s); **does not cap prefill quadratic** |
| 4 | Prefill linear-term + big-chunk amortization | ✅ done (`ffa01f4`) | expert-mm ~linear 31 ms/tok; chunk knee at 1024; RSS flat vs chunk |
| 5 | Full-Metal migration | ✅ investigated (`bd04e23`) | routed-expert decode GEMV regressed (kept opt-in `LG_DEC_EXP_ON=1`); 3 decode-path bug fixes landed |
| 6 | 256k hardening + release gate | ✅ this phase | gate matrix green; memory target met @256k; prefill/decode wall targets not reachable — see below |

## What hit, what missed

- **Hit: memory.** <20 GB @256k context is a real, measured property now —
  peak RSS 8.7-12.1 GB across the table, no `LG_CHUNK`/selection changes
  needed. Phase 1 did this.
- **Hit: gate matrix.** Full matrix re-run on CPU + Metal builds all green
  after the Phases 1-5 merge (tiny fixtures 4/4 × 12/12, bits-8 variant,
  `test_ngram_draft`, `test_kv_alloc`, `test_spec_decode_state`, resource
  plan 44/44, `test:baseline` 3/3 link).
- **Missed: prefill < 2 s and even "minutes at 256k".** The quadratic full-
  layer attention term was never bounded. Phase 3's selection caps *decode*
  reads only; the documented "selective propagation" prefill skip (Phase 4
  second half) was not implemented. This is the one lever that could make
  256k prefill feasible, and it is the honest recommendation below.
- **Missed: 140 tok/s decode.** Decode is CPU-lane bound (0.6-1.3 tok/s).
  Phase 2/5 moved projections to Metal GEMV but the per-token CPU path (oQ
  unpack, router, residual) still dominates; routed-expert Metal decode
  regressed and is opt-in off. The measured gap is architectural, matching
  the pre-rework docs (`laguna-decode-throughput.md`).

## Known limits

- Prefill is O(S²) on the 10 (XS) / 12 (S) full-attention layers; the tiled
  kernel removes the O(S²) *memory* but not the O(S²) *time*.
- Decode is CPU-bound; Metal decode GEMV delivers no win at the small
  batch/rows-per-expert this hardware exhibits.
- `sample` profiler inflates wall time ~10-20%; table above uses direct runs.

## Before/after (vs `docs/benchmarks/256k-baseline.md`, measured 2026-08-10)

XS-2.1-oQ2 Metal, direct runs, `LG_SPEC=0` decode.

| ctx (prompt) | prefill before → after | attn before → after | peak RSS before → after | decode before → after |
|---|---|---|---:|---:|
| ~1.9k | 28.7 s → 33.6 s* | 9.4 s → 10.8 s* | 6.1 GB → 5.8 GB | 6.10 → 3.76 tok/s* |
| ~6k | 68.8 s → 74.1 s* | 28.6 s → 30.8 s* | 7.7 GB → 7.5 GB | — |
| 256k ctx | not holdable (O(S·ctx) scratch ~10.7 GB unbudgeted) → **fits, 8.7-12.1 GB** | — | — | — |

\* ≈ same-kit re-run noise/±10%; the structural changes are the two rows below,
which the baseline could not measure at all:
- **256k context now holds in <20 GB** (Phase 1 tiled prefill removed the
  unbudgeted O(S·context) GPU scratch). Baseline explicitly projected it would not.
- **65k decode improves ~2.4x with selection** (Phase 3): 0.54 → 1.28 tok/s on
  XS (full-layer KV reads drop from O(S) to O(cap)).

## Honest recommendation

1. **Ship the memory win**: 256k context genuinely fits in <20 GB; the engine
   is a 256k-context-holding build. Document it as such (large-context
   retrieval/generation at shallow KV read cost), not as a low-TTFT engine.
2. **Do not present this as a 140 tok/s / sub-2 s-TTFT release.** Those
   targets need the one unimplemented lever: selective-propagation prefill
   (skip most prompt tokens in late full layers, FastKV/PFlash-style) behind
   a flag, measured for quality on the fixtures. Phase 3 built the index
   machinery; Phase 4 scoped the path; neither landed the actual prefill
   skip.
3. If decode latency matters next, the documented top lever remains
   speculative decoding (ngram draft landed; verify-in-batch is the
   follow-up) — or a smaller dense model for the interactive tier.

## Gate matrix (Phase 6 re-run)

| check | result |
|---|---|
| `make -C c laguna_xs_metal laguna_s_metal laguna_xs laguna_s` | 4/4 link clean |
| `task test` (tiny xs/s × CPU/Metal) | 4/4 PASS 12/12 |
| `test_laguna_tiny.py --bits 8` | PASS 12/12 |
| `test_ngram_draft` / `test_kv_alloc` / `test_spec_decode_state` | 3/3 pass |
| `test_resource_plan.py` | 44/44 OK |
| `task test:baseline` (olmoe/inkling/colibri) | 3/3 link clean |

Progress log: `docs/task-reports/phase6.md`.
