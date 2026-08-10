# 256k rework — Phase 0 baseline (measured 2026-08-10, Apple Silicon, 10 p-cores, 32 GiB)

One reproducible command per row: `c/tools/stress_laguna.sh <model> <prompt-tok> <gen> <tag>`
(ENGINE=.../laguna_xs|laguna_s for CPU, ..._metal for the GPU build). Prompts are the
harness's fixed-seed prose (seed 1234), so rounds are comparable. Decode rows are
`LG_SPEC=0` (plain greedy loop; the ngram spec'd path is a separate lever). The
`sample` sample-profiler adds ~10-20% wall time to stress_laguna rows; "clean" rows
(half duplex decode / non-sampled runs) are labelled.

Targets to beat: prefill 256k < 2 s (not physically reachable in full attention — see
plan), decode 140 tok/s, memory < 20 GB.

## Laguna-XS-2.1-oQ2 (10 full / 30 sliding, D=2048, kv=8, hd=128, topk=8)

| build | ctx (prompt) | prefill | attn | expert-mm | fill | peak RSS | decode tok/s (LG_SPEC=0) |
|---|---|---|---|---|---|---|---|
| CPU   | 1,869 | 184.6 s | 89.6 s | 82.8 s | 4.5 s | 15.1 GB | 5.88 |
| Metal | 1,869 | 28.7 s (37.7 sampled) | 9.4 s | 19.2 s | 2.0 s | 6.1 GB | **6.10** |
| Metal | 6,000  | 68.8 s | 28.6 s | 26.3 s | 2.3 s | 7.7 GB | _tbd_ |
| Metal | 30,000 | 436.4 s | **231.4 s** | 108.4 s | 3.2 s | 9.3 GB | _tbd_ |

Metal matches the docs' 30k attention (229.7 s) within noise; the expert term is
already ~2x better than the docs' 228 s (compacted tile list). Decode matches the
docs' 6.1-6.6 tok/s — it is the CPU per-token path whether the build is Metal or not.

## Laguna-S-2.1-oQ2e-fast (12 full / 36 sliding, D=3072, kv=8, hd=128, topk=10)

| build | ctx (prompt) | prefill | attn | expert-mm | fill | peak RSS | decode tok/s (LG_SPEC=0) |
|---|---|---|---|---|---|---|---|
| Metal | 800 | 60.7 s | 17.4 s | 43.0 s | 3.3 s | 9.5 GB | **0.86** |

S prefill is ~75 ms/token (expert-dominated, streaming 27 GB bank); decode is ~1.16 s/
token on the CPU lanes — 7x slower per token than XS (3x params, 48 vs 40 layers).

## Key structural facts re-confirmed

- The docs' "13 GB GPU KV ring" is already gone (commit 32592a1 bound the int8 cache
  zero-copy). The remaining GPU-side prefill scratch is **unbudgeted and O(S·context)**:
  the shared score tile `g_sc = S*nkey*4` reaches **8.6 GB** and the f32 band
  `g_kf = KV*nkey*hd*4*2` **2.1 GB** at the last 256k chunk (S=8192, nkey=262144) —
  ~10.7 GB the budget does not reserve. Phase 1 tiles this away.
- Decode never touches the GPU (S<64 gate); it is the CPU streaming-expert path
  (6.1 XS / 0.86 S tok/s). Phase 2 moves it to Metal GEMV.

## Planner sanity (Phase 0 fix)

`c/resource_plan.py` now mirrors the engine's single-source budget
(`laguna_common.h:1186-1207`): int8 CPU KV (`rows * n_kv * (head_dim + 4) * 2`) with
Metal sliding rings widened to `window + LG_CHUNK` and double-mapped. Tests:
`test_resource_plan.py` 44/44 (2 new Laguna KV expectations).

| model | ctx | KV (Metal build) | KV (CPU build) |
|---|---|---|---|
| Laguna-S-2.1-oQ2e-fast | 262,144 | 7.97 GB | 6.68 GB |