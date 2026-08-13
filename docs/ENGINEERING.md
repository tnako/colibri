# colibri-laguna engineering/design/performance — condensed master doc

Condensed from: 256k-rework-plan, 256k-final, metal-port, laguna-decode-throughput, laguna-s-scaling,
oq-optimization-rounds, phase3-selection-design, flash-attention-2-and-swap-fix, ane-investigation,
gpu-attention, gpu-profiling-and-cleanup, redesign-roofline, gpu-expert-grouped-gemm, inkling, laguna.
Hardware: Apple Silicon M5 (4P+6E), 32 GiB unified RAM, ~126 GB/s RAM read. Numbers measured on real
checkpoints unless "projected".

## 256k rework (Laguna-S)

**Targets (user-set):** context 256k · memory <20 GB (`LAGUNA_MEM_GB=20`) · prefill (256k) <2 s · decode 140
tok/s. Calibration: weights ~27.6 GiB (oQ2e-fast), ~7.7 GiB dirty streaming; KV CPU-int8 ~6 GiB @256k + GPU
ring ~13 GiB; O(S²) full-layer prefill ≈ 1e17 FLOP ≈ 1000 h at ~27 GFLOP/s; 140 tok/s bandwidth-reachable only
via Metal GEMV + spec decode.

**Phases (status + commit):**
| # | idea | status / commit | result |
|---|---|---|---|
| 0 | Baseline rig + planner KV model | ✅ `7101974` | `resource_plan.py` models Laguna KV (S @256k = 7.97 GB Metal / 6.68 GB CPU); 44/44 plan tests; XS 2k ~68.8 s prefill, XS decode 6.1 tok/s @1869, S 0.86 tok/s @800 |
| 1 | Kill 13 GB GPU KV ring: tiled online-softmax prefill | ✅ `2581a9f` | S×nkey score matrix never materialized; staging tile-sized (peak RSS 7.9 GB); parity max \|diff\| 2.4e-4; XS Metal @6k prefill 67.9 / attn 28.5 s / RSS 7.5 GB; sliding keeps banded path |
| 2 | Metal decode: batched GEMV projections + experts | ✅ `dd12af2` merged | projections/shared-expert GEMV behind `LAGUNA_DEC_GPU`; decode still CPU oQ path (GPU gated `S>=64`) |
| 3 | Sparse/selective attention (SAGE-KV) | ✅ `1e57466` merged | caps decode KV reads (XS 65k decode 1.28 vs 0.54 tok/s); **doesn't cap prefill quadratic** |
| 4 | Prefill linear-term + big-chunk amortization | ✅ `ffa01f4` merged | expert-mm ~linear ~31 ms/tok; chunk knee at 1024; RSS flat vs chunk |
| 5 | Full-Metal migration | ✅ investigated `bd04e23` merged | routed-expert decode GEMV regressed, opt-in `LG_DEC_EXP_ON=1` (spec-on 12.28 tok/s vs opt-in 0.49); 3 decode-path bug fixes (`dec_batch_end` silent no-op, `dec_reg_for` 4096→16384 align masking path, `dec_bind` use-after-free) |
| 6 | 256k hardening + release gate | ✅ done | memory target **met** (peak RSS 8.7–12.1 GB @CTX_MAX=262144); gate matrix green CPU+Metal |
| 7 | Selective-propagation prefill (O(S·cap)) | ✅ `aca7080` merged (plan `acb0aed`) | `LG_SELP` (default 0 ⇒ byte-exact); late full layers score only selected KV, refreshed via `sel_pass_at(m, upto)`; `sel_full` re-derived per prefix. 32k prefill attn **370.4→297.1 s (−20%)**, wall 649.9→568.7 s; 85.2% top-1 agreement @cap=8192; scoring layer still O(S²) (deferred) |

**Final results** (2026-08-12, branch `phase6/hardening`, direct runs, CTX_MAX=262144): context: 256k runnable
in RAM not wall-clock ⚠️ · memory: <20 GB → **8.7–12.1 GB peak RSS** ✅ · prefill 256k: <2 s → 65k 68 min (XS),
256k ≈ 9–290 h projected ❌ · decode: 140 tok/s → 0.6–1.3 tok/s ❌.

Measured (prefill / attn / peak RSS / decode tok/s): XS 3,802: 39.8 s / 13.0 / 7.0 GB / 4.13 · XS 16,023:
261.2 / 154.7 / 10.0 GB / 0.54 · XS ~64k `LG_SEL=1` cap 8192: 4071.7 / 2023.8 / 10.9 GB / 1.28 · S 3,802:
407.5 / 240.5 / 10.8 GB / 0.58 · S 15,073: 4266.1 / 3602.2 / 12.1 GB / 0.80 · S 3,802 @256k ctx: 336.0 s, 8.7
GB, 1.30 tok/s (n=128). Attention quadratic (S: 240.5 @3.8k → 3602.2 s @15k, ≈15x for 4x tokens); expert-mm
~linear ~31 ms/tok; shared ~10 ms/tok.

256k projection: S ≈ 3602.2×(256/15)² ≈ **290 h**; XS ≈ 154.7×(256/16)² ≈ **11 h**; a 15-min 256k XS probe
didn't clear chunk 1 of 32. Longest feasible: XS ~64k (68 min), S ~15k (71 min). Before/after: 256k was "not
holdable (O(S·ctx) scratch ~10.7 GB unbudgeted)" → **fits 8.7–12.1 GB** (Phase 1); 65k decode 0.54→1.28 tok/s
(Phase 3).

**Known limits:** prefill O(S²) on 10 (XS) / 12 (S) full layers — tiled kernel removes O(S²) *memory*, not
*time*; decode CPU-bound, Metal decode GEMV no win at small batch; `sample` profiler inflates wall ~10–20%.
Landmines: don't resurrect MPS batched per-head attention GEMM (slower, reverted); don't drop the `gpu_exp`
residency gate (regressed twice) or the `S>=64` GPU-expert gate (1-row prefill disaster); budget reserved once
in priority order (full-budget-to-consumer = double-count); KV ring stays `window+LG_CHUNK` rows double-mapped
on Metal; `===` is a zsh error (use `echo "==="`).

## Metal port findings

- **300 tok/s impossible**: decode touches 2.49 G params/token (attn 1.17 G, top-8/256 experts 0.98 G, lm_head 0.21 G, shared 0.13 G); 126 GB/s ⇒ 2-bit = 622 MB/token → **203 tok/s**; 300 tok/s needs 420 MB/token = 187 GB/s = 1.35 bits/param @100% bandwidth. Aggregate reachable with batching (batch 8 ≈ 1600 tok/s @2-bit).
- **MPS over hand-written shaders** (`c/laguna_metal.mm`): MPS f16 = 15572 GFLOP/s vs 1087 own kernel vs 59 CPU. Projections only (q/k/v/o + shared as f16, 3.10 GB XS, `LAGUNA_GPU_BUDGET_GB` default 4); routed experts excluded (63 GB f16). Gated `S>=32` (round-trip 0.327 ms); silent CPU fallback.
- 6144 prefill oQ2: round-8 CPU 465.1 s → Q8R resident 339.9 s → **Metal streaming 249.6 s** (attn 279.5→60.5, **4.6x**; shared 16.2→1.1, **14.7x**) → Metal+Q8R **115.8 s (4.0x)**; that combo needs the resident bank ⇒ opt-in (`LAGUNA_RESIDENT=1`).
- Leak fixed: fresh `MPSMatrixMultiplication` + `MPSMatrix` per call retained state (~280 calls/step) → cached per (weight,S)/(S,K)/(S,N). RSS 14.5 GB → ~2 GB @CAP=16, monotonic climb gone; timings unchanged (247.8→249.6 s). 4.98 GB floor = page-cached safetensors. CAP stops being a lever at long context (6144: CAP16 19.68 vs CAP48 18.88 GiB) — scratch + KV dominate.
- **Arena fix** (`arena_alloc`/`afloat`): ~26 transient S-sized buffers/layer in `attention()`/`moe()` moved to one region reset per layer → 18.88 → **12.89 GiB (−32%)**, `swapped_out` 7.4 GB → **0**, `MALLOC_LARGE (empty)` 730 MB gone, prefill unchanged. First regex attempt broke fixtures (24/24→8/24: `accum`/`sbuf`/`qt` are per-thread inside `omp parallel`) — reverted; second pass converted only function-scope single-threaded buffers.
- Correctness: XS 24/24+12/12, S 208/208+8/8 token-exact with f16 every layer; f16 only on batched prefill (`S>=32`, 11-bit mantissa). Status: prefill <20 s not yet (115.8 s @6144); 50 tok/s not yet (5–14 tok/s, overhead-bound); 300 tok/s impossible; "100x" no (4.0x @6144, 3.4x @1902).

## Decode throughput (Laguna-XS)

Opening 6.1–6.6 tok/s (2044-token prompt, oQ2, Metal). Roofline ~147 tok/s (298 GFLOP/s UDOT ÷ ~2
GFLOP/token). Current: **~9.5 tok/s** — headline change is spec decode (ngram draft, `ca7374b`, on by
default).

Ruled out: disk streaming (cache already holds 256/layer, hit 99.3–99.5%, zero `pread`) · GPU dispatch at S=1
(`LG_GPU_EXP_MIN=1`: >300 s for 30 tokens vs 29 s) · OMP spin-wait (`OMP_WAIT_POLICY=active` inert;
`OMP_NUM_THREADS=1` worse, 1.87 tok/s). Physical-core sizing was missing (laguna never included `omp_tune.h`)
— fixed at v1.6.0 merge (2026-08-13); parity fixture measured 3.4x faster decode (453→1528 tok/s) than the
SMT-wide default team.

Tried/reverted: **resident Q8R bank at decode** (removed `gpu_exp` gate + fixed budget-ordering bug): decode
WORSE, 3.06–3.43 vs 6.1–6.6 tok/s (~2x). Root cause: OMP loop `for e in E=256` scans all experts/token
(`__psynch_cvwait` ~67% vs ~45% streaming); compacting the active list didn't recover it — abandoned,
reverted, **do not re-attempt** · **batching projections into one OMP region** (`matmul_w_group`, bit-exact):
spec off 8.22→8.37 (+2%), default spec 9.60→9.49 (−1.1%), deep spec 9.55→9.43 (−3%); slower at S>1, reverted —
region forks aren't the bottleneck.

Landed: **`matmul_oq` nested-parallel bug** (unconditional `#pragma omp parallel` from inside the MoE parallel
region → wasted fork+join): expert-mm 18.3–18.6 → 17.2–18.3 s (~4–5%).

Gap to 147 tok/s is architectural: ~10 regions/layer (~400/token), each tiny (topk=8-row / 1-row-per-kv-head),
far from UDOT compute density; `__psynch_cvwait` dominance is legitimately idle workers. Spec decode is the
lever that landed; deeper draft + trained draft head unstarted.

## Attention & selection design

**Phase-3 selection** (`sel_pass` in `c/laguna_common.h`): knobs `LG_SEL` 0 off ⇒ byte-exact · `LG_SEL_CAP`
8192 (max selected KV per head group) · `LG_SEL_MIN` 16384 (engages above only); `cap >= context` selects
everything ⇒ byte-exact. Scoring = **SnapKV pooled softmax attention** over the last `LG_SEL_OBW=64` queries,
harvested from the last prefill chunk's tiled GPU attention (exp ÷ final den); scoring layer = **first full
layer**, index shared by all 12 full layers, per head group (KV=8); overhead KV·cap·4 B = 256 KiB @cap=8192;
Quest-style block fallback `LG_SEL_BLK=128`. Pass: top `(cap − LG_SEL_MARGIN=512)` + always-keep last 512 +
obs window; union, clamp, sort. Layout: `sel_idx[li] int[n_kv][cap]`, `sel_n`, `sel_lo`. Prefill tile-skip:
tiles with no selected column skipped; per-tile 64-bit bitmap zeroes unselected scores in `online_chunk`;
**scoring layer exempt** (pays O(S²) once — replaced by Phase 7 `LG_SELP`). Decode (CPU): key walk iterates
index O(cap) instead of [0,pos0); int8 CPU KV stays source of truth; GPU failure falls back to a CPU path that
still honors the index.

**GPU attention** (`c/laguna_attn_metal.mm`): 29,967-token prefill **1110.4 → 513.1 s (2.16x)**; attention
839.8 → 229.7 s (**3.66x**); expert-mm co-dominant (228.3 s). 2k attn 16.5→10.3 s; 6k 63.9→32.4 s. Only **full
layers** to GPU (90.7% of 122 TFLOP @30k; sliding O(S·512) stay CPU). Persistent per-layer f16 K/V cache
appended per chunk, sized to CTX_MAX, charged to budget (1.64 GB @40k XS); silently CPU if it doesn't fit.
`QK^T → causal softmax → PV` in one command buffer, both GEMMs MPS, softmax + gather/scatter custom shaders;
score matrix never leaves GPU (256 MiB/head @256k); output gate softplus folded into scatter. Hand-written
kernel: 81 GFLOP/s (~1% of ceiling); rewrite as MPS GEMMs: **480 GFLOP/s (5.9x)** — express work as a GEMM.
MPS **batched** per-head GEMM slower (37.6/38.2 vs 32.4 s @6k, 92% occupancy), reverted. f16 score matrix
token-exact; RSS flat ~20 GB (+1.64 GB GPU K/V).

**FlashAttention-2 + swap fix:** swap bug = budget over-commit — streaming cache sized FIRST from the full
budget, then GPU attention/projections/KV; 20 GB handed out 3–4x. Fix: `mem_used` accumulates KV, GPU
attention, exact projection reserve (incl. shared expert); cache gets 75% of what remains. 800-token S: swap
**+4915 → +119 MB**, RSS 13.5 → 12.2 GB; reserved 6.49 vs spent 6.47 GB. **FA2** (`flash_attn2`, `LG_FA2=1`):
threadgroup per (query,head), 64-key tiles, one `exp(m−m')` correction per tile. Back-to-back vs GEMM
(ratio-only): 2k 1.09x, 6k 1.06x, **12k 0.93x** (GPU busy 18.74 vs 16.11 s); kept off by default. Buys: never
materializes the score matrix (65k: 33.6 MB/head/chunk; 262k: 134.2 MB vs 32 KB/threadgroup). FA2 does NOT
unlock 250k: the blocker is GPU K/V (**13.11 GB @262k**), read identically; >half the 20 GB budget ⇒ declined
⇒ CPU fallback worsens the quadratic term. Per-tile K/V upload is the unfinished real-FA2 piece. Third
hand-written kernel to lose to MPS.

## KV / memory

**Laguna-S scaling:** banded sliding kernel — band = `window+chunk−1` = **767 columns, constant** vs dense
262144 @262k (**342x**). Dense attempt regressed 6k 32.4→53.1 s; banded → 21.8 s. XS @12k: attention
74.4→42.7, prefill 197.8→139.4 s. Gated `pos0+S >= 4*window` (below, CPU wins). Double-mapped ring KV (rows at
`r` and `r+ring`): any window span contiguous. S @48 layers: 4k→0.43, 65k→3.45, 262k→13.11 GB (vs 48 GB
linear); without it 250k = 45.8 GiB, impossible.

S measured: RSS 11.8→13.9 GB across 25x context, inside 20 GB; time linear 194–225 ms/token (expert-mm 70% of
prefill). S experts can't be resident: 2-bit codes 26.44 GiB + f32 metadata **9.91 GiB** = 36.35 GiB (metadata
47% of codes, gs=128); bf16 → 31.4 GiB still over budget; streams at 90.3% hit — correct config.

250k estimate: expert-mm 10.7 h + fill 1.6 h + sliding attn 1.2 h (linear) + **full attn 93.9 h (quadratic)**
≈ ~107 h. Memory @250k: CPU int8 KV 5.94 GiB, GPU ring 13.11 GiB, projections+embed ~3.3 GiB, scratch ~0.03
GiB. **Memory fits; time does not.** 13.11 GB > half of 20 GB ⇒ engine refuses, falls to CPU (worse
quadratic); needs `LAGUNA_MEM_GB` ≈30, impossible on 32 GB without swapping. Fix identified (blockwise softmax
streaming K in tiles) — done by Phase 1.

**ANE (investigated, unusable):** "38 TOPS INT8" isn't compute — dequantizes INT8→fp16; peak ~19 TFLOPS ≈ 1.2x
the GPU's *measured* 15572 GFLOP/s; below it with the SRAM cliff. Weights **baked at compile time** — fatal
for MoE: 9,984 distinct (layer,expert) sets in one 6144 prefill; recompile 494 ms–4200 ms each (total
4,932–41,933 s); ~119 compilations/process limit. **32 MB SRAM cliff** (~30% drop beyond): one layer's 256
experts = 1611 MB (50x over); projections 58.7 MB (1.8x over, ~13 TFLOPS = slower than GPU). Decode worse:
Orion GPT-2 124M = CPU 283 vs ANE 170 tok/s; ~49 KB min IOSurface pads 3072 B → 24576 B. Access requires
private `_ANEClient`/`ANECompiler`. Verdict: no help for experts, projections, or decode; GPU is the right
target. If revisited: 1x1 convs ≈ 3x matmul; fused graphs 94% vs ~30% single. Built instead:
`lg_metal_upload_f16` + `lg_metal_gemm_rows` (`LAGUNA_GPU_EXPERT_GB`, default off) — **inert**, gated
`!omp_in_parallel()` (shared cached MPS state raced across threads: `MPSMatrixMultiplication.mm:3240`). f16 =
1.6 GB/layer ⇒ only 4/39 MoE layers fit an 8 GB budget.

**Roofline (redesign):** ceilings (measured in `c/tools/`): CPU f32 FMA 59 GFLOP/s · CPU UDOT 298 GOP/s · UDOT
packed 2-bit ~198 GOP/s · SMMLA 90 GOP/s · hand Metal f32 1087 · **MPS f16 15572 GFLOP/s** · RAM 126 GB/s ·
mmap page-cached 127 GB/s · NVMe 11.2/30.6 (depth 10) GB/s · dispatch 0.327 ms. mmap costs nothing over RAM;
NVMe needs parallelism. Workload: XS @6144 = **35.8 TFLOP** prefill (attn projections 40%, MoE 34%, scores
22%, shared 4%); decode 2.49 G params/token (only top-8/256). **Q8R** (`c/q8r.h`, shipped): uint8 activations
+ UDOT via affine factorization (`w·x = sw·xs·UDOT(c,u) + sw·xz·rsum + bw·xsum`); 1.7–4.5x vs oQ, peak 442
GOP/s (lm_head); rel err 4.8e-5 (2-bit)–6.6e-4 (8-bit), zero argmax mismatches.

**Sizing error:** one-byte-per-code claim of 8.9 GiB was wrong (4x for 2-bit ⇒ 31.4 GB); engine printed
`[resident] need 37.3 GB, only 17.5 GB available -> streaming`. Fix: packed 16 codes/word (`q8r_udot_2bit`,
bit-identical) ⇒ resident bank **13.7 GB, loads 2.6 s**; packed is 0.67x the byte path per dot, but residency
wins. Expert IO eliminated @6144: prefill 465.1→339.9 s, expert-mm 153.5→39.3 s (3.9x), fill 6.0→0.0 s, RSS
11.05→**16.54 GiB (+50%)** (deliberate; `LAGUNA_RESIDENT=0` restores streaming). @1902: 89.2 s = **3.4x
cumulative**. Attention = 81% of remaining prefill (274.8/339.9 s); scores 3.87 TFLOP at ~14 GFLOP/s ⇒ CPU
floor 66 s, GPU 0.25 s. Verdict: prefill <20 s **not reachable on CPU** (35.8 TFLOP ⇒ ~1.8 TFLOP/s needed),
**reachable on GPU** (2.3 s GEMM + ~0.6 s dequant ⇒ 5–10 s). **50 tok/s reachable, not bandwidth-limited**
(1751 MB/token ⇒ 72 tok/s ceiling; resident bank moved decode 10.2→13.9). 100x: no on CPU (~5–7x honest); path
= scores+projections on MPS, f16 K/V GPU, MoE one dispatch/layer, decode stays CPU. ASan caught: scratch keyed
on rows reused across differing group sizes; double-free of visit-order array.

## Quantization perf history

**oQ rounds:** R1–5 @1902 (oQ2): prefill **301.0 → 136.1 s (2.2x)**, expert-mm 4.3x. R1 one OMP region over
(token,expert) killed a barrier storm (1.8M barriers/layer, 512 rows); R2 compact per-thread accumulator
(undid R1's 12.7 GB RSS regression); R3 group-by-expert did nothing until R4 hoisted `oq_unpack` (28.8%→6.7%);
R5 NEON `dot_f32`/`axpy_f32`.

R6–9 @6144: oQ2 520.8 → **465.1 s**. R6 query tiling + online softmax (`LG_QB=8`) → 490.9 s; R7 chunked
rescale `LG_KC=64` → 465.8 s + RSS −0.75 GiB (scratch no longer O(context)); R8 **expert-order pair visits**
(counting sort): hit **56.9→99.4%**, fill 21.5→6.0 s (**4.0x less disk IO**). oQ8e: R9 skip unpack for 8-bit
codes: 482.6→469.6 s, decode 2.51→2.80 tok/s, RSS 12.27→10.93 GiB; bit-exact vs `mlx.core.dequantize` gs
64/128. Cumulative: CPU 2.2x + ~12%; IO 4.0x; memory −6–11%. Remaining: `matmul_oq` ~64% self time,
`__psynch_cvwait` ~19% (per-expert `schedule(dynamic,1)` load imbalance). `CAP=48` pinned (CAP=4 ⇒ 9.3 GiB).
Every round token-exact (XS 24/24+12/12, S 208/208+8/8, ring wraps 50x).

**GPU grouped-GEMM experts:** unlock — `newBufferWithBytesNoCopy` over mmap'd safetensors (5.21 GB wrapped,
dispatch completed); GPU reads the same physical pages, clean file-backed pages are evictable page cache.
**Laguna-S's 28.4 GB of 2-bit expert weights map in 0.1 s at 0 GB budget.** Retires Q8R-resident (39 GB) and
streaming-LRU (8–14 GB + IO). Kernel dequantizes oQ in-register per gs group, simdgroup 8x8 tiles; max rel err
5.7e-05, 0/2560 >1e-3. Throughput: 8 rows 31.6 → 2048 rows **1515.4 GFLOP/s** (vs 298 CPU UDOT). Tile
32x32/4→64x32/8: 1125→1515. f16 staging broke accuracy (rel err 5.5e-02, 313/2560 — silu·up then down_proj
compounds 11-bit mantissa); f32 stayed.

Dispatch trap (twice): 36,864 dispatches/chunk hung (13.8% CPU, `__psynch_cvwait` 35330); one command
buffer/layer didn't fix it (serial). Fix: **grouped GEMM, `grid.z`=expert**, offsets buffer, 3
dispatches/layer. Gated `S>=64`. "Broken" counter was real: expert-mm 470.5 s in a 77.1 s prefill included
decode (~390 s for 4 tokens). After gate: expert-mm 48.9 s, decode 8 tokens in 8.6 s. Result (S @1433):
prefill **322.7 → 74.5 s (4.33x)**, expert-mm 48.9, attn 22.2 s, RSS 12.7 → 9.9 GB, 0 GB weights, no disk IO.
@262k ctx: 7.24 GB resident; GPU attention declines its 14.24 GB allocation ⇒ those layers use CPU. Honest
gap: **4.2x, not 100x** (needs 6.97 TFLOP/s = 45% of the 15.57 ceiling; kernel at 1515 ≈ 10%). Un-done: B
staged through threadgroup memory; rows/expert = chunk·topk/E (160 @LG_CHUNK=4096 vs 2048 peak); attn @256k
still asks 14.24 GB. Cleanup: removed FA2 streaming kernel (161 lines) + per-expert batched API (45 lines),
net **−206 lines**.

## Profiling (`gpu-profiling-and-cleanup.md`)

Budget priority fixed: GPU attention reserved **before** the resident bank, capped at half the budget
(previously bank-first ⇒ 20 GB run silently fell back to CPU attention). `LAGUNA_MEM_GB` 20 → prefill @2k 35.5
s; 22 → 29.9 s. Priority measured: GPU attention 3.66x on the largest phase vs bank ~1.5x on a smaller one.
`LAGUNA_GPU_PROF=1` reads `cb.GPUStartTime/GPUEndTime` (same timestamps as Instruments). @6k: attention phase
33.3 s = GPU 3.42 s wall / 3.16 s busy (**92% occupancy**) + **~30 s of 30 sliding layers on CPU** (3x the
full layers; more FLOPs at 2k).

Negative results (removed, not flagged): batched per-head GEMMs 37.6/38.2 vs 32.4 s @6k · sliding layers on
GPU (dense GEMM computes full S×nkey and masks it): 2k better 9.8→8.6 s but 6k worse 32.4→**53.1 s** (5.9x
waste) · GPU expert bank (`lg_metal_upload_f16` / `lg_metal_gemm_rows`): 8 GB of f16 for 2.5x slower (CPU 13.7
s vs GPU 33.8 s @2k) — deleted.

Default state: 2k prefill 25.1 s / attn 9.8 / expert 12.1 / RSS 18.1 GB; 6k 81.0 / 32.2 / 38.4 / 18.3 GB.
Knobs removed: `LAGUNA_RESIDENT`, `LAGUNA_GPU_BUDGET_GB`, `LAGUNA_GPU_EXPERT_GB`, `LG_METAL_MIN` (some
re-added by later work). This doc predates the grouped-GEMM re-add of the expert path.

## Engine notes

**Inkling** (Thinking Machines 975B MoE, `c/inkling.c`): 975B total / 41B active, Apache 2.0; dense resident
(RAM/VRAM), routed experts streamed from disk, LRU + pinned cache. Vision encoder + MTP head not loaded.
Pre-converted int4 ~469 GiB; CPU build ~86 GB bf16 residents (f32 expand ⇒ peak ~99 GB, dies below ~64 GB
RAM); CUDA +~37 GB VRAM; ~120 GB RAM / NVMe required. `convert_inkling_dense_int4.py` (int4-gs64): dense 49.4
GB → **15.3 GB**, fits 25 GB, ~14 min. Error: dense/attn/shared ~11% (quant noise ≈0.135σ, 16 levels, gs64 —
expected); embed/lm_head int8 ~0.9%; norms/router 0; `ATTN_BITS=8` ⇒ 1.1% at +4 GB. With ~8 GB cache left,
residency ~1.7% of 464 GB ⇒ decode disk-bound (tens of s/token) — runnable, not fast.

Audio (DMel): 80 mel bands / 50 ms, 16 levels; frame embedding = RMSNorm of 80 table rows at the `<|audio|>`
position; `--keep-audio` or an `audio.safetensors` sidecar; 16 kHz input; WAV via gateway (numpy), raw frames
via `--audio`. ~5 s speech ≈ 100 positions — inflates prefill.

Expert cache: cap = experts/layer, ~28 MB/slot on 975B ⇒ cap~2 on 25 GB. Small caps correct-but-slow (rounds
of cap); used to silently evict in-use slots (18-token prefill @topk=6 needs up to 108 distinct experts/layer
→ incoherent output, no error); fixed by acquiring all S×topk slots up front. Cache warming
(`SNAP/.coli_usage`): per-(layer,expert) counts accumulate across runs; top `PIN_N` (default cap/2) pinned per
layer. Envs: `PIN`, `PIN_N`, `USAGE_SAVE=0`, `NOGPU=1`, `GPU_DEV=<n>`, `IDOT=0`, `TOPP=<p>` (routing trim,
off).

Perf (975B, Ryzen 9 7900/24t, 187 GB DDR5, RTX A6000): plain LRU 150.2 s / 0.06 tok/s / ~0% → +packed-int4,
parallel fills, pins: 21.1 s / 0.25 / 81.5% → +CUDA resident: 18.4 s / 0.32 / 83.6% → +deep pins, trained
prompt: **1.9 s / 2.51 tok/s / 100%** → deep pins, novel prompt (overfit): 33.8 s / 0.17 / 79.8% → steady
state (11-prompt diverse history, novel): **35.4 s / 0.25 tok/s / 82.2%**. At high hit ~90% CPU expert matmul
— next lever is GPU expert compute. Validation: token-exact vs HF oracle in f32, int4 (VNNI and `IDOT=0`
scalar), bf16 CPU + CUDA; o200k tokenizer 357/357; converter `--selftest-e2e`.

**Laguna geometry & validation** (`c/laguna_common.h`): two checkpoints (XS/S), one architecture; no
compile-time geometry constants — config.json from the *released* checkpoints, not class defaults. Geometry:
hidden 2048/3072 · layers 40 (10 full/30 sliding) / 48 (12 full/36 sliding) · heads 48/64 & 48/72
(full/sliding) · KV heads 8 · head_dim 128 · experts 256, topk 8/10 · moe_intermediate 512/1024 · shared
512/1024 · dense layer-0 8192/12288 · window 512 · max_pos 262144/1048576 · YaRN factor 32/128, beta_fast
64/32, attn_factor 1.3465735902799727 / 1.4852030263919618. Layer types: one full + three sliding; rope yarn
on full (partial 0.5, theta 500000), default on sliding (theta 10000). **XS full layers ship YaRN.**

Unique vs other engines: per-head attention output gate (`g_proj` → `softplus(g[h])` × context); half-split
`rotate_half` rope (not interleaved); partial rotary on full layers (first head_dim/2); per-layer head count
(GQA group varies, KV=8). Checkpoint quirks: released vs transformers in-memory differ (per-expert vs fused
`gate_up_proj`/`down_proj`; `shared_expert` vs `shared_experts`; router bias under `mlp.experts.` vs
`mlp.gate.`); loader accepts both. **`load_state_dict` skips the per-expert→fused conversion ⇒ MoE silently
random** — use `from_pretrained`.

Sliding KV ring: `window` rows, `pos % window`; **append AFTER the scoring loop**, scoring reads the batch's
own keys from scratch (appending up front overwrites history rows still needed when S > window — attention
over future keys, silently). Same hazard as upstream PR #830 (inkling.c). Migration: MoE router shared/done
(`c/coli_moe_route.h`, also used by colibri.c); router renorm + `routed_scaling_factor` shared/partial
(`coli_moe_norm_scale`); sliding ring private/converging (revisit PR #830); half-split rope, YaRN precompute,
per-head gate, per-layer head count = private (no second consumer / needs generalizing).

Validation: `make_laguna_tiny.py` fixture (both layer types, differing per-layer heads, dense layer 0, non-1.0 scale, nonzero router bias, ring-wrap window; needs torch + transformers ≥5.12). 4L/8E/topk2/window4: XS & S 24/24 + 12/12 (also bits=8, cap=1). 4L/12E/topk10/window4/200-token (ring wraps 50x): S 208/208 + 8/8. Not done (as of `laguna.md`): no GPU path at that time (later landed), no int4 container conversion, no KV prefix reuse (`kv_prefix.h` — served convs re-prefill every turn), no tool-call rendering in the served template.



---

# Phase task & benchmark reports

Condensed from `docs/task-reports/phase4-chunk-sweep.md` (via `benchmarks/`), `phase5*.md`, `phase6.md`, `phase7.md`. All numbers as measured.


Condensed from `docs/task-reports/phase5*.md`, `phase6.md`, `phase7.md`,
`docs/benchmarks/phase4-chunk-sweep.md`. All numbers as measured.

## Phase 4 — prefill chunk sweep (LG_CHUNK, Metal)

`c/laguna_xs_metal`, Laguna-XS-2.1-oQ2, 941-token prompt, `-n 2`, runtime `LG_CHUNK` override, `LG_TRACE_CHUNK=1` attribution.

| LG_CHUNK | prefill | expert-mm | attn | fill | RSS |
|---:|---:|---:|---:|---:|---:|
| 256 | 93.0 s | 86.1 s | 5.3 s | 0.1 s | 5.2 GB |
| 512 | 47.3 s | 41.6 s | 4.8 s | 0.2 s | 5.3 GB |
| 1024 | 26.9 s | 21.4 s | 4.2 s | 0.3 s | 5.5 GB |
| 2048 | 27.0 s | 21.5 s | 4.1 s | 0.4 s | 5.5 GB |
| 4096 | 27.6 s | 21.5 s | 4.6 s | 0.2 s | 5.5 GB |
| 8192 | 27.5 s | 21.7 s | 4.2 s | 0.2 s | 5.5 GB |

Lever is **expert-mm** (86.1→21.4 s 256→1024, flat after); attn (4.2–5.3 s) + shared chunk-independent; `expert cache hit 99.9%` ⇒ GPU `lg_metal_moe_layer` serves all prefill. Old "256 optimal" obsolete (CPU-UDOT build); kernel rows/expert bound (8 rows/expert @256 vs 32 @1024, topk=8/E=256). RSS flat 5.5 GB from 1024 up. Per-chunk serial tail ≈ 1.0 s/chunk, 0.7 s whole prompt @1024 — amortized by big chunks (93.0→26.9 s = 3.5x). Default `LG_CHUNK=8192` stays; 1024 the knee. Repro: `LG_CHUNK=$CH LG_TRACE_CHUNK=1 SNAP=models/Laguna-XS-2.1-oQ2 ./c/laguna_xs_metal 0 0 --chat -n 2 -f <prompt>`.

## Phase 5 — routed experts on the persistent decode session

Status: **opt-in, correct, but a regression vs the CPU oQ-cache path** (`LG_DEC_EXP_ON=1`). `moe()` routes decode (S==1) routed pairs through persistent Metal session `g_dec`: expert-sorted gather, per-pair S=1 `dec_gemv` (oQ bf16, `LG_DEC_OQBF16`) gate+up in one commit, CPU silu `siluf(g)*u`, down in a second commit, weighted scatter; CPU fallback on failure. `dec_exp_alloc` registers scratch `[0]gather [1]gate [2]up [3]down` (posix_memalign 16384); `dec_exp_add` page-aligns bases, folds delta into offsets (keeps dec_bind no-copy). `LG_DEC_EXP_OFF` removed; gates on `LG_DEC_EXP_ON`.

Bugs fixed: (1) cmd buffer never ran — `dec_batch_end()` early-returned when `n_dec_copies==0`; fixed via `dec_exp_ops`; (2) cmd-buffer OOM from no-copy buffers sized to strided offsets (~1.9–3.1 GB) → page-align base + offset-shift; (3, P2) `dec_reg_for` 4096-align vs required `NSPageSize()` 16384; (4, P2) UAF in `dec_bind` wrap buffers — `dec_relinquish`.

Measured (XS-2.1-oQ2, 40-token A/B, 2026-08-12):

| mode | tok/s | fill | expert-mm | attn |
|---|---:|---:|---:|---:|
| default (CPU oQ cache, spec on) | 12.28 | 0.5 s | 1.3 s | 1.7 s |
| default (CPU, `LG_SPEC=0`) | 10.06 | 0.5 s | 1.5 s | 2.1 s |
| `LG_DEC_EXP_ON=1` (batched, spec on) | 0.49 | 0.4 s | 72.2 s | 6.3 s |
| `LG_DEC_EXP_ON=1` (batched, `LG_SPEC=0`) | 0.06 | 0.1 s | 649.5 s | 17.1 s |

Earlier per-pair S=1 path: 3.25 vs 1.88 tok/s (40-tok); 3.29 vs 1.29 (16-tok, byte-identical text).

### v2 — one-dispatch-per-matrix batched routed decode

Routes decode routed pairs through `lg_metal_moe_layer` like prefill — ONE command buffer, 4 dispatch groups (gate+up+silu+down), ONE compact 128-row tile list/layer, ONE commit+wait. Same `expert_gemm` + GPU silu → byte-identical to CPU default. Still a regression (0.49 / 0.06 tok/s): rows/expert ≈1 at decode scale whether S=1 or S=25 (npair=200 over 256 experts); TM=64 tile wastes 64x FLOPs padding 1-row experts. One layer = 177 ms GPU busy + ~150 ms commit+wait (rmsnorm/residual CPU-owned). CPU slot cache resident-hot → **verdict: keep opt-in.** Routed decode no longer needs `LgDecode` regions/wrap buffers; `dec_batch_*` untouched.

Verification (both): text byte-identical to CPU default (n=16); `decode_gemv_parity` DECODE PARITY OK (worst=7.63e-06); 4 tiny gates + `make test-c` ALL PASS; `dec_exp_*` per-pair helpers removed.

## Phase 6 — 256k hardening + release gate

Branch `phase6/hardening`, rebased on merged laguna-support (Phases 1–5, `5245c59`). Apple Silicon 10 p-cores/32 GiB, real checkpoints.

Gates: 4/4 builds link clean; `task test` 4/4 12/12; `--bits 8` 12/12; ngram/kv_alloc/spec_decode 3/3; resource_plan 44/44; baseline (olmoe, inkling, colibri) 3/3. No `fix:` needed.

Scaled context (Metal, CTX_MAX=262144, greedy `-n 8`; RSS = `/usr/bin/time` max):

| model | prompt | prefill | attn | expert-mm | shared | fill | RSS | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| XS-oQ2 | 3,802 | 39.8 s | 13.0 s | 20.3 s | 0.6 s | 1.4 s | 7.0 GB | 4.13 |
| XS-oQ2 | 16,023 | 261.2 s | 154.7 s | 71.7 s | 4.5 s | 1.3 s | 10.0 GB | 0.54 |
| XS-oQ2 | 64k (`LG_SEL=1` cap 8192) | 4071.7 s | 2023.8 s | 1338.0 s | 15.1 s | 1.0 s | 10.9 GB | 1.28 |
| S-oQ2e-fast | 3,802 | 407.5 s | 240.5 s | 119.2 s | 29.1 s | 2.3 s | 10.8 GB | 0.58 |
| S-oQ2e-fast | 15,073 | 4266.1 s | 3602.2 s | 380.4 s | 145.4 s | 2.4 s | 12.1 GB | 0.80 |

Attn O(S²) on full layers (XS 10 full/30 sliding, ~13x per 4x tokens; S 12 full, ~15x); S expert-mm ~linear (~31 ms/tok), shared ~10 ms/tok. RSS ≤12.1 GB every point. XS decode collapsed at 16k w/o selection (0.54 tok/s); selection restores 1.28 @65k. S decode CPU-bound ~0.6–0.8 tok/s always.

256k budget: S prefill not runnable — projected attn ≈ 3602.2 × (256/15)² ≈ **1.05e6 s (~290 h)** + ~1.5 h linear; XS ≈ 154.7 × (256/16)² ≈ 39,600 s (~11 h). RSS <20 GB @256k met (peak 8.7–12.1 GB); quadratic attn is the blocker, not RAM. Longest feasible: XS 65k (68 min), S 16k (71 min). Headline: 256k fits RAM (✅); prefill not runnable (~290 h S / ~11 h XS); decode 0.6–1.3 tok/s (target 140 ❌); gates 100% green. Doc: `docs/256k-final.md`.

## Phase 7 — selective-propagation prefill (LG_SELP)

Implemented, gates green, XS-2.1-oQ2 (Metal). O(S²) full-layer prefill → O(S·cap): after the first full (scoring) layer appends each chunk's K/V, the shared selection index refreshes and LATE full layers score only selected KV columns. Decode untouched (Phase 3 walk).

Config (`c/laguna_common.h`): `Model.selp` (env `LG_SELP`, default 0); `LG_SELP=1` forces `m->sel=1` too, flags distinct for byte-exact runs; `print_cfg` prints `selp: ON cap=… min=…`. Scoring: `sel_pass` → `sel_pass_at(Model*, int upto)`; `sel_full` re-derived per `upto` (fix: early chunk `cap >= upto` must not latch full-selection — the 32k cap=8192 zero-gain bug). `step_raw` calls `sel_pass_at(m, pos0+S)` on the scoring layer of each PREFILL chunk (`S > LG_DEC_SMAX`) once `pos0+S >= sel_min`; decode batches excluded.

CPU: `sel_use` extended — prefill+selp, late full layers (`li > sel_base`) walk index; decode keeps index on all full layers; scoring layer exempt during prefill. GPU: new `LgAttnSel` + `lg_metal_attn_sel(...)` (`lg_metal_attn` = wrapper `sel=NULL`); fully-unselected tiles skipped; `zero_unselected` -INF-masks partial tiles (exp(-inf)=0, exact); `lg_metal_attn_impl` split keeps CPU fallback (bounded reads). Scoring + sliding layers NOT routed to skip path.

Gates (all green): 4/4 builds; `test_laguna_tiny.py` 24/24 + 12/12; `test_selection.py` PASS; **new** `test_selection_prefill.py` PASS — `LG_SELP=1 LG_SEL_MIN=1 LG_SEL_CAP=1048576` byte-exact vs `LG_SELP=0`, cap engages + prints `[sel] cap=…`; resource_plan 44/44; decode_parity worst 7.63e-06; `attn_tiled_parity.mm` max |gpu-cpu|=0; metal-vs-cpu parity identical.

Measurement (XS-2.1-oQ2, Metal, `LG_SPEC=0`, stress_laguna.sh; noisy ±15%):

| config | prompt | prefill attn | prefill wall | peak RSS |
|---|---|---|---|---|
| baseline (`LG_SELP=0`) | 16k | 141.3 s | 273.4 s | 8.70 GiB |
| `LG_SELP=1 cap=4096` (early build) | 16k | 116.5 s | 227.8 s | 9.03 GiB |
| `LG_SELP=1 cap=8192` | 16k | ~136 s | ~254 s | 9.45 GiB |
| baseline | 32k | 370.4 s | 649.9 s | ~9.7 GiB |
| `LG_SELP=1 cap=8192` | 32k | 297.1 s | 568.7 s | ~9.7 GiB |

32k clean A/B: **attn 370.4 → 297.1 s (−20%)**, wall 649.9 → 568.7 s. Remaining cost: scoring layer (full O(S²)) + expert-mm (153–165 s). `LG_SEL_DIAG=1`: 3.2k prompt cap 512 scored 1162/3210 cols (36%) — ktile≈1024 rounds scan up. Quality: ~2k prompt, greedy -n 32, first 32 tokens: `LG_SELP=1 cap=8192` vs baseline → **46/54 = 85.2% token agreement** (top-1); matches >~80% target.

Files changed: `c/laguna_common.h` (`selp`, `sel_pass_at`, prefill refresh, CPU `sel_use`, GPU routing, `print_cfg`); `c/laguna_metal.h` (`LgAttnSel`); `c/laguna_attn_metal.mm` (`zero_unselected`, tile skip + mask, impl/_sel split, `LG_SEL_DIAG`); `c/tests/test_selection_prefill.py`.

Next: none for this phase; tree ready to commit. (Optional: skip scoring layer's own O(S²) via shallow selection layer, design doc §5 — deferred.)
