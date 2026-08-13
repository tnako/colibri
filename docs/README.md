# colibri-laguna documentation index

The docs were consolidated (~10x smaller) into five master documents. Every
original file is mapped below; no content was dropped outright — each source was
condensed into the corresponding master doc, preserving exact values (env
defaults, flags, measured numbers, commits).

| master doc | covers |
|---|---|
| [`REFERENCE.md`](REFERENCE.md) | HTTP/serve API + wire protocol, env vars + CLI settings, quant/format specs (FORMATS, oQ, int4-rans256-g0), KV/cache notes, routing telemetry, model notes (DeepSeek-V4, Kimi K3, corpus/grammar drafts) |
| [`RUNBOOK.md`](RUNBOOK.md) | quickstart + build/run per backend (Windows/CUDA/Metal/Vulkan), tuning knobs, benchmarking + measured tables, Metal perf reports, Laguna XS/S one-knob section |
| [`ENGINEERING.md`](ENGINEERING.md) | 256k rework phases + final results, Metal port, decode-throughput gap, attention/selection design, KV/memory, quantization perf history, profiling, engine notes (Inkling, Laguna geometry), phase task reports |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | dated multi-GPU experiment reports (CNRE simulator, 6×5090, 4×A6000, continuous batching, split-entropy, decode failure ledger, inference-paper claim matrix) |

## Absorption map (original → master doc)

| original file | absorbed into |
|---|---|
| `api.md`, `serve_protocol.md`, `ENVIRONMENT.md`, `SETTINGS.md`, `FORMATS.md`, `oq-format.md`, `int4-rans256-g0.md`, `kv-bind-zero-copy.md`, `int8-kv-cache.md`, `CACHE_ROUTE.md`, `routing-telemetry.md`, `deepseek-v4.md`, `deepseek-v4.zh-CN.md`, `kimi_k3.md`, `corpus-draft.md`, `grammar-draft.md` | `REFERENCE.md` |
| `quickstart.md`, `windows.md`, `cuda.md`, `BUILD-cuda-glibc241.md`, `metal.md`, `vulkan.md`, `tuning.md`, `benchmarks.md`, `benchmarks/256k-baseline.md`, `benchmarks/phase7-selective-prefill.md`, `METAL-M5MAX-PERF-REPORT.md`, `METAL-M1ULTRA-FMT2-REPORT.md`, `laguna-xs-gpu-experts.md`, `one-knob-and-chunking.md` | `RUNBOOK.md` |
| `256k-rework-plan.md`, `256k-final.md`, `metal-port.md`, `laguna-decode-throughput.md`, `laguna-s-scaling.md`, `oq-optimization-rounds.md`, `phase3-selection-design.md`, `flash-attention-2-and-swap-fix.md`, `ane-investigation.md`, `gpu-attention.md`, `gpu-profiling-and-cleanup.md`, `redesign-roofline.md`, `gpu-expert-grouped-gemm.md`, `inkling.md`, `laguna.md` | `ENGINEERING.md` |
| `task-reports/phase4-chunk-sweep.md` (via `benchmarks/`), `task-reports/phase5.md`, `task-reports/phase5-routed-expert-decode-session.md`, `task-reports/phase6.md`, `task-reports/phase7.md` | `ENGINEERING.md` (Phase task reports section) |
| `experiments/cnre-offline-simulator.md`, `experiments/glm52-6x5090-2026-07-12.md`, `experiments/glm52-4xa6000-2026-08-02.md`, `experiments/glm52-continuous-batching-2026-07-31.md`, `experiments/glm52-split-entropy-2026-07-31.md`, `experiments/glm52-decode-failure-ledger-2026-07-31.md`, `experiments/inference-paper-test-matrix-2026-07-28.md` | `EXPERIMENTS.md` |
| `MAINTAINING-DOCS.md` | below |

## Keeping ENVIRONMENT.md / SETTINGS.md honest

`ENVIRONMENT.md` and `SETTINGS.md` rows in `REFERENCE.md` are **generated from
source** and drift as code changes (`REFERENCE.md` carries the "Generated from"
hash). Truth lives at:
- env vars → `getenv("…")` sites in `c/*.c|h|cu|mm` (`:!c/tests/*`)
- CLI flags → `add_parser`/`add_argument` in `c/coli`, `c/openai_server.py`

Refresh procedure: diff code vars vs documented vars then update tables in place
(`comm` on backticked `VAR` cells), keep grouping (Common/Performance/Backend/
Advanced/Set-by-CLI), record the owner engine (colibri/kimi_k3/inkling/olmoe do
**not** share knob sets), defaults come from the ternary, and bump the hash.
Four binaries: `colibri` (most vars), `kimi_k3` (`K3_*`), `inkling` (`INK_*`,
`CTX_MAX`, `PIN_N`, `REP_PEN`, `GPU_DEV`, `NOGPU`), `olmoe` (`HOT`, `WIDE`,
`SMOOTH`, `CONF_LIMIT`, `MAX_NEW`, `CHAT`, `EXPERT_DROP`, `WARMUP`).

## Other housekeeping

- `docs/media/` (images) kept; referenced by README/site.
- `media/` OG images: `site/index.html` links
  `docs/media/colibri-atlas.png` (unchanged).
- `docs/tuning-9950x3d-5090.md` (referenced by `c/coli`) never existed — dead
  link removed; the tune guide now points to `RUNBOOK.md` → Tuning.