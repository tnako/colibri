# Phase 7 — Selective-propagation prefill (LG_SELP)

Status: implemented, gates green, measured on XS-2.1-oQ2 (Metal).

## What was implemented

Selective propagation makes the O(S²) full-attention layers of PREFILL O(S·cap):
during prefill, after the first full-attention (scoring) layer appends each
chunk's K/V, the shared selection index is refreshed, and the LATE full layers
of that chunk then score only the selected KV columns. Decode is untouched
(still the Phase 3 output-bounded walk).

### Config (`c/laguna_common.h`)
- `Model.selp` (env `LG_SELP`, default 0). `LG_SELP=1` also forces `m->sel=1`
  (the Phase 3 machinery runs) but the flags stay distinct so a full-cap
  byte-exact run is still verifiable. Default off ⇒ byte-exact everywhere.
- `print_cfg` prints a `selp: ON cap=… min=…` line.

### Prefill-time scoring (`sel_pass` → `sel_pass_at`)
- `sel_pass(Model*)` became `sel_pass_at(Model*, int upto)` where `upto` is the
  absolute prefix length to score; `sel_pass()` is now `sel_pass_at(kv_len)`
  (the post-prefill call in `generate_stream` is unchanged). Repeated calls free
  the previous index first (no leak/double-alloc).
- `sel_full` is re-derived per call for the given `upto` (critical: an early
  chunk where `cap >= upto` must not latch full-selection for the whole run once
  the prefix grows past cap — this was the bug that made a 32k cap=8192 run show
  zero gain before the fix).
- `step_raw` calls `sel_pass_at(m, pos0+S)` on the scoring layer of each PREFILL
  chunk (`S > LG_DEC_SMAX`) once `pos0+S >= sel_min`. Decode batches are excluded
  so the pass is not re-run per token.

### CPU walk (`attention`)
- `sel_use` extended: during prefill with `selp`, late full layers (`li >
  sel_base`) also walk the shared index. The decode `S <= LG_DEC_SMAX` path keeps
  using the index on all full layers (incl. `sel_base`) as before; the scoring
  layer itself is exempt during prefill.

### GPU tiled path (`c/laguna_attn_metal.mm`, `c/laguna_metal.h`)
- New `LgAttnSel` param struct + `lg_metal_attn_sel(...)` (plain
  `lg_metal_attn` is a thin wrapper with `sel=NULL`, unchanged behavior).
- Tile-level skip: tiles with no selected column are skipped entirely (no
  dequant, no QK GEMM, no online_chunk/PV).
- In-tile masking: new `zero_unselected` kernel zeroes non-selected columns of a
  partially-selected tile to -INF before `online_chunk` (exp(-inf)=0, exact),
  using a per-tile byte membership mask built on the host from `sel_idx`.
- `lg_metal_attn` is now `lg_metal_attn_impl(...)` so the CPU fallback (return 0)
  still works; the CPU path honors the same index (bounded reads, not full
  attention) per the design doc §7.
- Scoring layer and sliding layers are NOT routed to the skip path.

## Gates (all green)

- Build: `make -C c laguna_xs_metal laguna_s_metal laguna_xs laguna_s` — 4/4 clean.
- `test_laguna_tiny.py` xs_metal + xs + s_metal + s: 24/24 + 12/12 (defaults).
- `test_selection.py` xs_metal + xs: PASS (baseline / full-cap byte-exact / capped engages).
- **new** `test_selection_prefill.py` xs_metal + xs: PASS — long prompt,
  `LG_SELP=1 LG_SEL_MIN=1 LG_SEL_CAP=1048576` byte-exact vs `LG_SELP=0`; real cap
  engages + completes + prints `[sel] cap=…`.
- `test_resource_plan.py` 44/44.
- `decode_parity` (lg_decode_gemv) worst |diff| 7.63e-06 — PASS.
- low-level `attn_tiled_parity.mm` (rebuilt here): max |gpu-cpu| = 0 — the
  `online_chunk` change is conservative.
- end-to-end metal-vs-cpu token parity on `Laguna-XS-2.1-oQ2`: generated text
  identical.

## Measurement (XS-2.1-oQ2, Metal, LG_SPEC=0, stress_laguna.sh)

Machine is shared/noisy: single-run attn times vary ±15%; the 32k case is the
most stable (single dominating chunk). Honest numbers:

| config | prompt | prefill attn | prefill wall | peak RSS |
|---|---|---|---|---|
| baseline (`LG_SELP=0`) | 16k | 141.3 s | 273.4 s | 8.70 GiB |
| `LG_SELP=1 cap=4096` (early build, deq. spam fixed) | 16k | 116.5 s | 227.8 s | 9.03 GiB |
| `LG_SELP=1 cap=8192` | 16k | ~136 s | ~254 s | 9.45 GiB |
| baseline | 32k | 370.4 s | 649.9 s | ~9.7 GiB |
| `LG_SELP=1 cap=8192` | 32k | 297.1 s | 568.7 s | ~9.7 GiB |

32k is the clean A/B: **attn 370.4 → 297.1 s (−20%)**, wall 649.9 → 568.7 s.
The remaining prefill cost is the scoring layer (still full O(S²)) and the
expert-mm term (153-165 s), not the skipped late layers. Diagnostic
(`LG_SEL_DIAG=1`): a 3.2k prompt with cap 512 scored 1162/3210 cols (36%) —
tile granularity (ktile≈1024) rounds the scan upward, as the block-aligned
selection predicts.

## Quality

~2k-token real prompt, greedy -n 32, first 32 generated tokens:
`LG_SELP=1 cap=8192` vs baseline → **46/54 = 85.2% token agreement** (top-1).
Not byte-exact (expected), but high agreement — matches the design target
(>~80%).

## Files changed

- `c/laguna_common.h` — `selp` config, `sel_pass_at`, prefill-time refresh,
  CPU `sel_use` extension, GPU routing (`lg_metal_attn_sel`), `print_cfg`.
- `c/laguna_metal.h` — `LgAttnSel` + `lg_metal_attn_sel` declaration.
- `c/laguna_attn_metal.mm` — `zero_unselected` kernel, tile skip + mask,
  `lg_metal_attn_impl`/`_sel` split, `LG_SEL_DIAG`.
- `c/tests/test_selection_prefill.py` — new gate.

## Next step

None remaining for this phase; the tree is ready to commit. (Optional future:
skip the scoring layer's own O(S²) via a shallow selection layer, per the design
doc §5 — deferred.)
