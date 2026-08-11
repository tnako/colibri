# Phase 3 — Post-prefill selection pass (bounded effective KV for full layers)

Design draft for `phase3/sparse-attn`. Status: **config platform in; selection
pass + decode walk implemented** (`sel_pass` in `laguna_common.h`, CPU decode
walk honors the shared index; prefill tile-skip is exempt per §5 and lands with
Phase 4's selective propagation).

## 1. Why

Laguna-S has 12 full-attention layers that score against the *whole* prefix
(O(S²) prefill, O(S) KV read at decode) and 36 sliding-window layers that are
already O(window). The 256k wall is the full layers. We cap their effective KV
with a SAGE-KV / SnapKV-style post-prefill selection pass:

| term | today | with selection |
|---|---|---|
| prefill full-layer work | O(S²) | O(S·cap) + one cheap scoring layer |
| decode full-layer KV reads | O(S) | O(cap) |
| full-layer KV memory | O(S) | O(cap) (index + GPU staging) |

## 2. Knobs (config platform, implemented)

| env | default | meaning |
|---|---|---|
| `LG_SEL` | `0` (off) | master switch; off ⇒ byte-exact full attention everywhere |
| `LG_SEL_CAP` | `8192` | max selected KV positions **per head group** (effective KV ceiling) |
| `LG_SEL_MIN` | `16384` | selection engages only when the prefilled context > this |

Semantics:
- `LG_SEL=0` (default): nothing changes; the engine is byte-exact, fixtures stay
  green. This is the CPU fallback *by construction*.
- `LG_SEL=1` **and** prompt length `> LG_SEL_MIN`: the pass runs; short fixture
  runs (far below `LG_SEL_MIN`) remain byte-exact even with `LG_SEL=1`.
- `LG_SEL_CAP` and `LG_SEL_MIN` are read in `model_init()` and printed in
  `print_cfg()` (a `sel: ON cap=… min=…` line only when `LG_SEL=1`).

`Model` gains `int sel, sel_cap, sel_min` (`c/laguna_common.h`).

## 3. Scoring — SnapKV pooled attention over an observation window

Scores are the **pooled softmax attention weights** that the last
`LG_SEL_OBW = 64` query positions of the prompt put on every prefix key
position, exactly SnapKV's observation-window aggregation:

```
importance[kh][p] = Σ_{q ∈ last LG_SEL_OBW queries} softmax_w(q, p, kh)
```

- **Where the weights come from**: only the *last* prefill chunk contains the
  observation-window queries. During that chunk's tiled GPU attention
  (`lg_metal_attn`, full layer, `online_chunk`), the obs-window rows of each
  score tile are `exp(score − running_m)`; the running `m`/`den` converge to
  their final values by the last tile. We harvest these rows' exp values and
  normalize by the final `den` once the chunk finishes, giving the true final
  softmax weights for the obs window against the whole prefix. Cost:
  O(LG_SEL_OBW · S) dequant-ish work on ONE layer — negligible next to one
  full attention pass.
- **Which layer**: the **first full-attention layer** (lowest `li` with
  `!slide[li]`) is the scoring layer. Its pooled weights select the KV set;
  the *same index is shared by all 12 full layers* (FastKV/ChunkKV layer-wise
  index reuse), which keeps the index memory O(cap) not O(12·cap).
- **Per head group (GQA)**: one score vector per KV head `kh`
  (`KV = n_kv = 8`). Query heads in a group share the same index — index
  overhead is `KV · cap · 4 bytes` per layer, i.e. 256 KiB at cap=8192.
- Block-level fallback (Quest-style, `LG_SEL_BLK = 128`): score each block by
  the pooled weight of its positions; a block is selected iff its score
  crosses the top-k threshold. Kept as an alternative that bounds the *tile*
  granularity too; the per-position SnapKV scoring is the default.

## 4. Selection pass (run once after prefill)

1. From the harvested pooled weights, per head group `kh` of the scoring layer,
   pick the top `(cap − LG_SEL_MARGIN)` positions by weight.
2. **Always keep** the last `LG_SEL_MARGIN` positions (default 512; recent
   tokens dominate attention and decoding starts there) and the observation
   window itself.
3. Union + clamp to `cap`, sort ascending → the **selection index**.

`cap >= context length` selects **every** position for every head group (no
sparsification), making full-cap runs byte-exact vs selection off.

### Index location and layout

Stored on `Model` (allocated in `kv_alloc`, alive as long as the KV):

```
Model.sel_idx[li]    →  int  [n_kv][cap]      sorted absolute positions
Model.sel_n[li]      →  int  [n_kv]           count (≤ cap) per head group
Model.sel_lo[li]     →  int                   0 = no pass run / full attention
```

Only full layers (`!slide[li]`) get an index; sliding layers are untouched.
A per-position `uint8_t` membership bitmask over `[0, S)` is the compact
form used to mask tiles cheaply (S bytes for S=256k).

## 5. Prefill: skip non-selected tiles

The tiled GPU path (`window == 0` branch of `lg_metal_attn`) streams K/V in
tiles of `ktile` columns. With an active selection index:

- **Tile skipping**: a tile `[t, t+ktile)` is processed only if it contains at
  least one selected position `p ∈ index[kh]` with `p ≤ pos0+S` (causal). Tiles
  with no selected column are skipped entirely — no `deq_kv`, no score GEMM, no
  PV. This makes the total scored columns O(cap + margin) instead of O(S).
- **Masking within a tile**: a per-tile 64-bit bitmap (or the index itself) is
  passed to a selection-aware `online_chunk` variant that zeroes the scores of
  unselected columns (in addition to today's causal mask), so the PV GEMM sees
  zero weight for dropped positions. Block-aligned tiles (block = `LG_SEL_BLK`)
  make whole tiles either fully selected or fully skipped, which is both the
  cheapest and the exact case.
- **The scoring layer is exempt**: it must see the whole prefix during the
  final chunk to harvest the obs-window weights, so it keeps today's full tile
  walk. This is the only full layer that pays O(S²) on the first prefill;
  Phase 4 (`256k-rework-plan.md` §Phase 4, "selective propagation") replaces it
  with a single shallow selection layer. Note the effective-KV *bound* (decode
  reads, KV memory) does not depend on this exemption.
- Sliding layers keep the banded path untouched.

## 6. Decode: read only the selected rows

Decode (`S = 1`) never dispatches the GPU (the `S >= 64` gate at
`laguna_common.h:1595`), so the decode win is on the **CPU** attention loop:
for a full layer, the key walk iterates `index[kh]` (O(cap)) instead of
`[0, pos0)` (O(S)). Implementation: when a layer's index is active, the CPU
tile loop (`LG_KC` chunks) skips `t` not in the index/mask, or — cleaner —
walks a compact list of selected runs. The int8 KV cache stays the single
source of truth; nothing is copied. The GPU ring/staging for full layers holds
only the selected rows (the tiled path already bounds staging by the chunk;
the selection bound makes the *reads* O(cap)).

## 7. CPU fallback and exactness

- **Selection off** (`LG_SEL=0`, or context ≤ `LG_SEL_MIN`, or the pass has not
  run for a fresh `kv_alloc`): `sel_lo[li] == 0` and every layer takes today's
  full-attention path byte-for-byte. The fixtures (24/24 + 12/12 on all four
  binaries) must stay green — they are far below `LG_SEL_MIN`, so they never
  engage selection even with `LG_SEL=1`.
- **Selection on**: results are *not* byte-exact vs full attention (that is
  expected and acceptable); the parity gate for selection runs under the same
  selection on both sides. The long-context accuracy gate is token agreement on
  a padded/repeated-prompt sample (the plan's fixture + real long-context
  sample gate).
- Any GPU failure falls back to the CPU path exactly as today; with selection
  active the CPU path also honors the index, so the fallback keeps the bounded
  reads rather than silently reverting to full attention.

## 8. Verification plan (ritual)

1. `make -C c laguna_xs_metal laguna_s_metal laguna_xs laguna_s` clean.
2. `c/tests/test_laguna_tiny.py --binary … --fixture laguna_tiny` → 24/24 + 12/12
   on all four binaries (defaults, selection off).
3. Functional gate `c/tests/test_selection.c` (+ python runner): a long
   synthetic prompt above `LG_SEL_MIN` with `LG_SEL=1`; asserts the printed
   `[sel] effective KV = cap` and that the tiled GPU path never scores more
   than `cap + margin` rows per full layer.
4. `c/resource_plan.py` KV math models full layers as O(cap) when `LG_SEL` is
   on (planner change to land with the implementation).
