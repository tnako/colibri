# Phase 5 — routed experts on the persistent decode session

Status: opt-in, correct, but a measured throughput regression vs the CPU oQ-cache
path. Kept behind `LG_DEC_EXP_ON=1` pending a batched multi-row kernel.

## What was built

`moe()` in `c/laguna_common.h` now has a routed-expert block that routes decode
(S==1) routed-expert pairs through the persistent `LgDecode` Metal session
(`g_dec`, the same session as attention/shared):

- visit-order (expert-sorted) gather of each pair's token row into a gather
  region; per-pair one S=1 `dec_gemv` (oQ bf16 format, `LG_DEC_OQBF16`) for
  gate and up in one commit, CPU-side silu fusion `siluf(g)*u` (identical to
  the CPU path), then down in a second commit, then weighted scatter into
  `out`. Falls back to the CPU paths (`memset(out,0)` then the grouped
  prefill / streaming resident code) on any failure.
- `dec_exp_alloc(m, maxrows)`: lazily registers 4 scratch regions
  `[0]=gather`, `[1]=gate`, `[2]=up`, `[3]=down` (posix_memalign 16384).
- `dec_exp_add(...)`: enqueues one routed-expert gemv; wraps page-ALIGNED
  tensor bases and folds the alignment delta into the offsets so dec_bind's
  no-copy path stays engaged.

## Bugs fixed along the way

1. **Command buffer never ran.** `dec_batch_end()` early-returned when
   `n_dec_copies == 0`; the routed block enqueues gemvs via `dec_exp_add`
   which adds no copy entry, so `lg_decode_run()` was never called and the
   regions stayed uninitialized garbage. Fixed by tracking `dec_exp_ops` and
   running the batch when either copies or exp ops are pending.
2. **Command buffer OOM** (layers 1/4/13/14 initially): wrapping the shard
   base pointer with `wneed = tensor_off + ...` created no-copy buffers sized
   up to the STRIDED tensor offset (~1.9-3.1 GB) which hit a Metal command
   buffer out-of-memory. Wrapping a bare `wbase+woff` tensor start fell off
   the no-copy path (unaligned) into per-gemv full-buffer copies (2-15 s per
   layer). Final fix: page-align the base and offset-shift (above).
3. (Pre-existing, Phase 2) `dec_reg_for` used 4096-aligned posix_memalign but
   `lg_decode_region` requires `NSPageSize()` (16384 on this M5), silently
   masking the entire GPU decode path; fixed earlier (bumped to 16384).
4. (Pre-existing, Phase 2) heap-use-after-free in `dec_bind`: growing wrap
   buffers released a buffer an earlier bind of the same shard base had
   already returned; freed via `dec_relinquish` (moves old buf into `d->refs`).

## Verification

- Engine output with `LG_DEC_EXP_ON=1` is byte-identical to the CPU default
  (diff of `<assistant>` text across a 16-token run: OUTPUTS IDENTICAL).
- `tests/decode_gemv_parity.mm` DECODE PARITY OK (worst=7.63e-06) — base
  lanes untouched.
- `LG_DEC_EXP_OFF` env removed; the path now gates on `LG_DEC_EXP_ON`.

## Measured (Laguna-XS-2.1-oQ2, 40-token decode A/B, 2026-08-12)

| mode | tok/s | fill | expert-mm |
|---|---|---|---|
| default (CPU oQ cache) | 3.25 | 4.1s | 3.8s |
| `LG_DEC_EXP_ON=1` | 1.88 | 1.6s | 15.2s |

16-token run: default 3.29 tok/s, exp-on 1.29 tok/s (same text). The path
collapses slot-fill as intended but the S=1 scalar gemv dispatch (24 tiny
gemvs + 2 commit/wait per layer × 39 layers) is ~4x slower on the expert-mm
side, matching the pre-existing warning that 8 rows/expert is the kernel's
worst regime (31.6 GFLOP/s at 8 rows vs 1515 at 2048).

## Next step

Make it faster or retire it: batch all of a layer's pairs into ONE gemv per
matrix using the compacted-tile kernel (`lg_metal_moe_layer`) via a persistent
ring of regions, i.e. only ~3-6 dispatches+1 commit per layer instead of 24+2.
Until then the default stays on the CPU path; the block is opt-in.