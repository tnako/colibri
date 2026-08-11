# Laguna-S 256k rework plan: goal, phases, and verification

> **Status (updated by the assistant, as of this session):**
> - ✅ **Phase 0 — done** (commit `7101974`)
> - ✅ **Phase 1 — done** (commit `2581a9f`)
> - 🚧 **Phase 2 — in progress**: API contract for the persistent decode GEMV path is
>   drafted and awaiting implementation (`c/laguna_metal.h` `lg_decode_*`, working
>   tree, uncommitted). Implementation and wiring are the current next step.
> - ⬜ Phases 3-6 not started.

Targets (user-set):

| metric | target |
|---|---|
| context | 256k tokens |
| memory | < 20 GB (inside `LAGUNA_MEM_GB=20`) |
| prefill (256k prompt) | < 2 s |
| decode | 140 tok/s |

Honest calibration against measured reality (`docs/laguna-s-scaling.md`,
`docs/gpu-attention.md`, `docs/laguna-decode-throughput.md`, `docs/one-knob-and-chunking.md`):

- **Weights alone** (oQ2e-fast) are ~27.6 GiB resp. the dirty working set is
  ~7.7 GiB with a *streamed* expert bank; KV is CPU-int8 (~6 GiB @256k) + GPU
  ring (~13 GiB @256k for the 12 full layers). Memory fit requires deleting the
  13 GB GPU ring and keeping experts streamed/hot-pinned.
- **< 2 s prefill of 256k tokens is not physically reachable** on this hardware
  in any full-attention configuration: 12 full layers of O(S²) attention is
  ~1e17 FLOP ≈ 1000 h at the measured ~27 GFLOP/s, and even the linear expert
  term is ~0.2 s/token. Every plan below is honest that the achievable target
  is *minutes*, and "2 s" only becomes conceivable with aggressive selective
  (sparse) attention that skips most prompt tokens in late layers.
- **140 tok/s decode** is reachable in principle: decode is memory-bandwidth
  bound (~500 MB weight reads/token at oQ2 ⇒ ~70 GB/s at 140 tok/s, inside
  unified-memory bandwidth) but only if decode moves on to Metal GEMV +
  speculative decoding. The current CPU path is ~6 tok/s on XS and the docs
  conclude the gap is architectural, with spec decode as the top lever — which
  has since landed (`c/laguna_common.h`).

Phase ordering principle: each phase must (a) preserve token-exactness on the
existing fixtures and (b) leave the tree measurable with one command, so
"check performance and memory usage after every phase" is a fixed ritual.

## Crash course: what to read first

- `c/laguna_common.h` — the whole CPU engine: `attention()` at :1453 (GPU
  flash path behind `#ifdef LAGUNA_METAL` at :1551), `moe()` at :1740,
  `step_raw()` at :2134, `step()` chunker at :2205, budget reservation at
  :1157, `kv_alloc` around :2238.
- `c/laguna_attn_metal.mm` — GPU full/sliding attention kernel.
- `c/laguna_expert_metal.mm` — GPU grouped-GEMM MoE dispatch.
- `c/resource_plan.py` — planner; `kv_bytes` at :573-574 does **not** model
  Laguna KV today (0 bytes) — fix early.
- `Models: models/Laguna-S-2.1-oQ2e-fast/` (real 35 GB checkpoint is present).
- Tests: `c/tests/test_laguna_tiny.py` (token-exact gate), `c/tests/test_resource_plan.py`, `c/tests/test_ngram_draft.c`, `c/tests/test_kv_alloc.c`, `c/tests/test_spec_decode_state.c`.
- Bench: `c/tools/stress_laguna.sh <model> <prompt-tok> <gen> <tag>`; produces
  per-phase timers + `peak_rss` + `sample` profile. Build/run: `task laguna:s`
  or `make -C c laguna_s_metal` then run `./c/laguna_s_metal ...` with `SNAP=`.
- Taskfile: `laguna:s` (Q=oQ2e-fast default via resolve), `laguna:s:serve`.

## The verification ritual (every phase)

For each phase, from a clean tree:

1. Build both CPU and Metal engines and run the byte-exact gates:
   `task test` (or the explicit python runs in Taskfile `test`), i.e.
   `c/tests/test_laguna_tiny.py --binary c/laguna_s_metal --fixture laguna_tiny`
   must stay token-exact (24/24 + 12/12 for S path), plus
   `./c/tests/test_ngram_draft`, `test_kv_alloc`, `test_spec_decode_state`,
   and `c/tests/test_resource_plan.py` all pass.
2. Measure on the **real checkpoint** with `c/tools/stress_laguna.sh` at a
   fixed token budget (e.g. 2000/6000/30000/65536 as units reach their useful
   range), XS first when a phase is generic, S-oQ2e-fast for the 256k-bound
   phases. Record prefill / attn / expert-mm / RSS, plus peak RSS from the
   poller.
3. For 256k-specific claims, use a **padded/repeated-prompt** harness that
   measures wall prefill for T tokens and peak RSS, with `LAGUNA_MEM_GB=20`
   and `CTX_MAX=262144`.
4. Commit per phase with a `perf:`/`feat:` subject + the "measured" table in
   the body, mirroring the repo's style (see `git log --oneline -20`).

## Phase 0 — Baseline rig + memory accounting (no behaviour change) — ✅ DONE

Status: `perf:`/`feat:` commit `7101974`. `c/resource_plan.py` now models Laguna KV
(S KV @256k = 7.97 GB Metal / 6.68 GB CPU); `test_resource_plan.py` 44/44;
`docs/benchmarks/256k-baseline.md` records the fresh baselines (XS 2k prefill ~68.8 s
attn-dominated, decode XS 6.1 tok/s @1869, S 0.86 tok/s @800, both Metal builds).
Ritual gates green (fixtures on xs_metal/s_metal, test_ngram_draft, test_kv_alloc).

Goal: one reproducible command that reports every number the targets need, and
a planner that models Laguna KV correctly.

- Measure XS and S at 2k/6k/30k (+65k for S) prefill/attn/expert-mm/RSS and
  decode tok/s; save as `docs/benchmarks/256k-baseline.md` (new dir).
- Fix `c/resource_plan.py:573-574` to compute Laguna KV bytes from the real
  config (int8 CPU KV + double-mapped Metal ring formula already in
  `laguna_common.h:1186-1207`), feed into plans and tests
  (`test_resource_plan.py` gets a co-located expectation).
- Baselines to beat (from docs, re-measured fresh):
  - XS prefill 2k ≈ 31.4 s, 6k ≈ 82 s, 30k ≈ 513 s; decode ≈ 6-6.6 tok/s.
  - S prefill ≈ 195-620 s at 100-3200 tokens (linear ~200 ms/token); decode
    slower than XS (bigger model). Memory inside 20 GB at ≤3k already.

Acceptance: same numbers as the docs within noise; planner prints non-zero KV
for Laguna; all gates pass.

## Phase 1 — Kill the 13 GB GPU KV ring for the 12 full layers (FlashAttention-style tiled prefill) — ✅ DONE

Target memory: < 20 GB @256k. This is the single highest-value change.

- Replace the persistent f16 per-layer full-attention K/V appends in
  `c/laguna_attn_metal.mm` with **tiled online-softmax prefill**: stream K in
  tiles of `LG_KC` from the int8 CPU KV cache (already the single source of
  truth — see `attention()`'s "APPEND FIRST" comment at `laguna_common.h:1518`)
  or a compressed GPU-side copy, computing causal scores + chunk max/rescale
  per tile, never materializing `S × S`.
- This is exactly the docs' own recommendation (`laguna-s-scaling.md:114`,
  `gpu-attention.md:92`). It removes the 13.11 GB allocation for 12 full layers
  at 256k and lets the GPU full-attention path of a 256k prompt run on-ring
  instead of being vetoed by the budget and dropping to the CPU.
- Keep the int8 CPU KV as the source of truth; upload per tile. Keep sliding
  layers' banded path as-is (already O(S·window) and budgeted).
- The MPS batched-GEMM experiment (tried, slower, reverted per
  `gpu-attention.md:61-67`) should NOT be resurrected for head-batching; the
  win here is streaming the KV, not batching heads.

Verify (phase ritual + specifically):
- Full-layer attention memory at 256k drops from ~13 GB to tile-sized; peak
  RSS for a 256k (or the longest feasible) padded run inside 20 GB.
- Attention phase time after the change vs. before at 6k and 30k.

Status: `perf:` commit `2581a9f`. Tiled online-softmax prefill in
`c/laguna_attn_metal.mm` for full layers (`gather_g`, `init_md`, `online_chunk`,
`fin_scatter` kernels; tile ~384 MiB; `LG_KTILE` override). The S×nkey score
matrix (8.6 GB at the last 256k chunk) is never materialized; staging is now
tile-sized (1.25 GB fixed at `CTX_MAX=262144`, peak RSS 7.9 GB). Parity harness
`c/tests/attn_tiled_parity.mm` + `test_gpu_attn_parity.py` (max |diff| 2.4e-4 at
H=48 KV=8 hd=128 S=256). Sliding layers keep the banded path. Fixtures green on
xs_metal/s_metal. XS Metal @6k: prefill 67.9 s / attn 28.5 s / RSS 7.5 GB.

## Phase 2 — Decode on Metal: batched GEMV for projections + experts — ✅ DONE (merged dd12af2)

Target: 140 tok/s decode.

Status: the API contract is drafted in `c/laguna_metal.h` (working tree,
uncommitted) — `LgDecode` session, `lg_decode_new/free/region/begin/gemv/silu/run/
region_ptr/bytes/active`, formats `LG_DEC_OQF32/OQBF16/F32/BF16`. NOT yet
implemented in `c/laguna_metal.mm` and NOT yet wired into `c/laguna_common.h`
(attention projections / shared expert / routed experts / spec-verify batched
forward). Decode still runs the CPU `matmul_oq`/UDOT path per token.

- Current decode runs CPU per-token (`S=1`); GPU dispatch is gated off at
  `S < 64` (`LG_GPU_EXP_MIN`) because per-row round trips are catastrophic
  (documented, `laguna-decode-throughput.md:21-28`).
- Add a **persistent Metal decode path**: resident oQ2 weights for the
  attention projections (+ embed/lm_head) and the hot experts, executed as
  memory-bound GEMVs, with one dispatch per layer per token (or per draft
  batch), not per head. The ngram spec decoder (`laguna_common.h` `ngram_draft`
  + accept/reject in `generate_stream`) should verify the draft in ONE batched
  forward so it rides the bulk-token GPU code paths.
- Hot-expert residency: keep the top-routed experts pinned on GPU (usage/pin
  machinery exists in the repo) so decode has ~topk reads resident instead of
  streaming.
- Do NOT just remove the `S>=64` gate on the prefill grouped-GEMM kernel —
  that specific regression is documented twice.

Verify: decode tok/s on XS and S at each context size; phase time budget
(full-layer decode attention must read a bounded KV set once Phase 1/3 make it
so).

## Phase 3 — Sparse/selective attention for the full layers (bounded effective KV) — ✅ DONE (merged 1e57466)

Target: keep the quadratic term from growing with context and cap triple
decode attention traffic + KV memory.

- Full-attention layers attend to the *whole* prefix (12 layers @256k), which
  is the O(S²) wall. Implement a **selection pass** (SAGE-KV/SnapKV-style
  top-k per head group, or block-level Quest-style) run once after prefill so
  the 12 full layers' effective KV is capped (e.g. 4k-32k tokens):
  - KV memory for the 12 full layers becomes O(cap) not O(S).
  - Decode attention for full layers reads O(cap) not O(S).
  - Prefill's quadratic term becomes O(S·cap) instead of O(S²).
- Sliding layers already bounded; leave alone.
- Offline K/V index must be **shared per head group** (GQA) to keep the
  index overhead small — matches FastKV/ChunkKV layer-wise index reuse.

Verify: accuracy gate on the fixtures + a real long-context sample (padded
prompt baseline before/after token agreement); memory + prefill wall at 65k/256k.

## Phase 4 — Prefill linear-term reduction + big-chunk amortization — ✅ DONE (merged ffa01f4)

Target: shrink the ~200 ms/token linear expert term and attention fixed cost.

- Sweep prefill on S with `LG_CHUNK` (8192 current default; `tune_chunk.sh`
  pattern) on the *Metal* build — the previous sweep predates the unified
  GPU dispatch and the shared `g_kf` staging.
- Push the prefill MoE path through `laguna_expert_metal.mm` grouped GEMM for
  the whole chunk (needs resident or streamed awings) and check the streaming
  bank doesn't dominate disk service (`stress_laguna.sh` reports it).
- If prefill at 256k is a hard wall, this phase also wires up the
  Phase-3-selected-token prefill (skip tokens in late layers after the
  selection layer) as the only known path toward single-digit-second TTFT.
  This is explicitly the "selective propagation" of FastKV / the PFlash-style
  bandit in the literature; keep it behind a flag and measure quality.

Verify: prefill wall and per-phase split at 6k/30k/65k (+256k if safely
run); memory must stay <20 GB.

## Phase 5 — Full-Metal migration (only if the above cannot meet targets) — ⬜ NOT STARTED

The user has pre-authorized: "You can always fully migrate to metal if this
requires."

- If CPU lanes still cap decode (<140 tok/s) or the chunked CPU paths cap
  prefill, move the remaining per-token CPU work (rmsnorm, rope, gate, output
  projection, logits) into the same Metal dispatch so the whole
  forward-per-token is one persistent command buffer, eliminating CPU↔GPU
  ping-pong per layer.
- Keep the byte-exact CPU fallback path intact (current architecture already
  falls back on any GPU failure).
- This is the "BaseRT-style" end state from the literature (routing the
  compute-bound prefill through matmul engines, keeping memory-bound decode on
  tuned memory kernels).

## Phase 6 — 256k hardening + release gate — ⬜ NOT STARTED

- Full 256k prefill+decode run on `Laguna-S-2.1-oQ2e-fast`, inside 20 GB,
  measured decode 140 tok/s target + TTFT.
- Write up `docs/256k-final.md` with the measured table and the state of each
  phase's idea.
- Re-run the full test matrix (`task test`, `task test:baseline`) on CPU and
  Metal builds.

## Known landmines (from docs, do not re-trip)

- Do not resurrect the MPS batched-per-head attention GEMM — measured slower,
  reverted (`gpu-attention.md:61-67`).
- Do not just remove the `gpu_exp` residency gate at decode — regressed twice
  (`laguna-decode-throughput.md:39-80`).
- Do not just drop the `S>=64` GPU-expert gate — prefill kernel at 1 row is a
  disaster (`laguna-decode-throughput.md:21-28`).
- The budget is reserved once in priority order; feeding a consumer the *full*
  budget is the historical double-count bug (`laguna_common.h:1157-1168`).
- KV ring must stay `window+LG_CHUNK` rows, double-mapped, on the Metal build
  or a banded read splits at the wrap (`laguna_common.h:1174-1204`).
- `===` in zsh is an error; the repeatable harness must use `echo "==="`.

## Branch and commit convention

- Work on `laguna-support` (or a derived `256k` branch for the risky phases).
- One commit per phase, subject style `perf:`/`feat:`/`fix:`, body = exact
  measured table, per repo history convention.
- The `.o` build artifacts are untracked/ignored (`.gitignore` `*.o`).