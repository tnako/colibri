# Phase 5 — routed experts on the persistent decode session

Now: **batched one-dispatch-per-matrix decode routed experts** (opt-in
`LG_DEC_EXP_ON=1`), replacing the per-pair S=1 gemv loop that measured 1.88
vs 3.25 tok/s.

## Design (this session)

Feed a layer's decode routed pairs through `lg_metal_moe_layer` exactly like
prefill — ONE command buffer, 4 dispatch groups (gate+up+silu+down), ONE
compact 128-row tile list per layer, ONE commit+wait:

1. Sort npair pairs by expert (`vis`, `estart`) — already done at the top of
   the routed block; the kernel requires rows per expert contiguous in the x
   buffer, so reuse the same visit-order gather into `lg_metal_scratch(0)`.
2. Build `offs[E+1]` (`estart`) into scratch(4) and the compacted tile list
   `(expert, r0)` into scratch(5) — identical to the prefill block (lines
   2219-2239). ntiles = sum of ceil(nr/TM) over touched experts.
3. One `lg_metal_moe_layer(G[0..2].wmap/woff/..., hx,hg,hu,hh,ho,ht,
   ntiles,npair,D,I,gs,bits,wslabGU,sslabGU,wslabD,sslabD)` — the scatter is
   the same CPU loop reading `hh` (down out) with routing weights.
4. This makes the decode session (`dec_exp_*`, per-pair `dec_exp_add`) dead
   for the routed block; remove it. `lg_metal_moe_layer` uses the same
   `expert_gemm` kernel and GPU silu_mul (`x/(1+exp(-x))` == `siluf`), so
   per-row FMA order is unchanged and the block stays byte-identical to the
   CPU default path.

## S=25 reality check

At S=1, npair=8-10 (topk), so ntiles≈that many tiles → one threadgroup per
(expert,row-tile) — same tiny kernel. The win only appears when the ngram
spec-verify batch rides the same path: S up to LG_DEC_SMAX=25, npair up to
~200 → tiles pack to ≥ TM=64 rows/expert, i.e. the 31→1000+ GFLOP/s regime.
If `LG_DEC_EXP_ON=1` still regresses at S=1 after this change, record it and
keep opt-in.

## Measured (Laguna-XS-2.1-oQ2, 40-token decode A/B, 2026-08-12)

Baseline prompt "Write a short list of three things to do in Paris." (this session):

| mode | tok/s | fill | expert-mm | attn |
|---|---|---|---|---|
| default (CPU oQ cache, spec on) | 12.28 | 0.5s | 1.3s | 1.7s |
| default (CPU, `LG_SPEC=0`) | 10.06 | 0.5s | 1.5s | 2.1s |
| `LG_DEC_EXP_ON=1` (batched, spec on) | 0.49 | 0.4s | 72.2s | 6.3s |
| `LG_DEC_EXP_ON=1` (batched, `LG_SPEC=0`) | 0.06 | 0.1s | 649.5s | 17.1s |

The batched kernel is a clear regression in every regime — even with the ngram
spec batch it stays slow, because rows/expert at DECODE scale is ~1 no matter
whether S=1 or S=25 (npair=200 over 256 experts) and the `expert_gemm` TM=64
simdgroup tile wastes 64x of the GPU FLOPs padding a 1-row expert to a full
row-tile. Instrumented: one layer = 177ms GPU busy (the 64x-wasted tiles) +
~150ms commit+wait that cannot be amortized across layers (rmsnorm/residuals
stay CPU-owned). The CPU slot cache is resident-hot at decode and needs no
dispatch, so it wins by a wide margin. **Verdict: keep opt-in.**

## Verification

- Generated text byte-identical to CPU default path (n=16, same prompt, text
  after the `[N prompt tokens]` marker diffs empty).
- `tests/decode_gemv_parity.mm` DECODE PARITY OK (worst=7.63e-06).
- Four tiny gates PASS: decode-parity, laguna_tiny on laguna_xs_metal, on
  laguna_s_metal (--bits 8), on CPU laguna_xs; `make test-c` ALL PASS.
- `dec_exp_*` per-pair S=1 decode-session helpers removed: the routed block now
  uses `lg_metal_moe_layer` + scratch slots (same shape as prefill), so the
  routed decode path no longer needs `LgDecode` regions / wrap buffers at all;
  the attention/shared decode batch (`dec_batch_*`) is untouched.