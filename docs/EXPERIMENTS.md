# Colibri GLM-5.2 experiment reports (condensed)

Compressed master doc. Sources: `docs/experiments/*.md`. All tok/s are
single-request decode unless stated. Keep sources for full tables/log paths.

---

## 1. CNRE Phase 0 — offline residency-policy simulator

Date: ongoing (supports Discussion #884). No engine change, no runtime speedup;
replays `ROUTE_TRACE` files through bounded expert caches to reject weak
policies before coding/renting hardware.

**What it models** (`c/tools/residency_sim.py`): first-seen expert unions per
`(call, layer)` batch; promotion across GLM's 64-expert blocks; per-layer
resident bytes, read bytes, felt-miss cost; fixed expert-RAM budget; equal vs
trace-trained per-layer budgets; policies GLM LRU, half-capacity pins+LRU,
decayed LFU, segmented LRU, decayed-frequency admission. Does **not** model
prefetch, routing substitution, GPU kernels, NUMA, page cache, locks, startup
read, or tok/s. Read-ahead of the next 64-expert block is not modeled →
long-prefill felt-wait estimates conservative (check with low
`felt_fraction`).

**Trace contract**: `<call> <row> <layer> <expert>:<gate>`; contiguous
`(call, layer)` groups = one batch; rejects malformed/missing/reordered rows,
non-advancing call IDs. Kimi K3 traces unsafe (no call-ID advance); Inkling and
OLMoE emit no route records — each needs an engine profile first. Unset
`EXPERT_BUDGET`/`ABLATE_*` during collection; record `COLI_PREFILL_CHUNK`.
Training and eval traces must be separate files; policy state resets per file.

**Manifest**: fixed geometry required (inferring `max_experts` from held-out
IDs leaks eval data). `felt_miss_us` default = `read_bytes/(read_gbps*1e9)*1e6
* felt_fraction`. `resident_bytes` consumes budget; `read_bytes` is demand I/O.

**Phase-0 decision gate** (advance only if all hold on category-held-out real
traces):
- ≥10% lower predicted felt wait;
- no principal held-out category worse than 3% vs uniform LRU;
- bounded churn + exact byte-budget compliance;
- survives per-layer felt-cost sensitivity.

`--policies` must include `lru`. Categories equal-weight; derived by stripping
final numeric replicate (e.g. `chat_2.trace`→`chat`) unless `--eval-category`.
`--prof-log` calibrates felt wait from the physical `N load` field; legacy
logs → `felt_us_per_physical_miss=null`. A withdrawn calibration: earlier
"19.6 s / 2,625 loads" summed `loads/token`, not physical misses.

**Deterministic fixtures**: stationary (cacheable) + shifted hot-set (shows
overfit). Uniform-frequency admission fails shifted-category guard (−4.76%);
dynamic frequency passes nominal (−1.72%) but fails layer sensitivity at 8 MB.
Budget sweep 2–32 MB shows robustness is an operating region, not a
policy-wide property — real traces must be swept across RAM budgets.

**H200 pilot** (DO H200, GLM container rev `95bb5f03e3e0ca16b4c711394d0461fa86a3cfcb`,
model 429.3 GB, 24 cores/247 GB RAM/143.8 GiB VRAM):

| budget | candidate | mean held-out gain | worst cat. |
|---:|---|---:|---:|
| 120 GB | uniform freq | +39.91% | +34.84% |
| 120 GB | dynamic freq | +40.45% | +35.41% |
| 160 GB | uniform freq | +53.35% | +49.92% |
| 160 GB | dynamic freq | +54.50% | +50.66% |
| 200 GB | uniform freq | +65.31% | +62.64% |
| 200 GB | dynamic freq | +66.29% | +63.07% |

All frequency candidates passed nominal + sampled sensitivity (layers 3/30/60);
LRU dynamic failed the 10% gate (≈0.0–0.4%). Dynamic rows predate exhaustive
capacity eval → re-derive; uniform rows fine. Runtime pilot: `RAM_GB=120 PIPE=0`
1.66 tok/s @57.2% hit; `RAM_GB=200 PIPE=0` 1.31 @73.5%; `PIPE=1` 2.00 @57.2%.
**Hit rate alone ≠ tok/s**: overlap/disk/compute matter. No runtime
admission policy was implemented (cache admits every demand miss in `moe()`);
unrecognized `COLI_CACHE_ADMISSION` has no effect.

**Open action items**: no complete GLM routing traces in repo — need coding/
chat/multilingual/reasoning/long-context train+held-out, matching `PROF=1`
logs, exact commit/config/cache state; runtime prototype only after real-trace
evidence + A/B. Negative result = valid published experiment.

---

## 2. GLM-5.2 on 6× RTX 5090 (Blackwell) — 2026-07-12

Host: 6× RTX 5090 (32 GiB ea), dual Intel Xeon Silver 4510 (24c/48t),
251 GiB RAM, NVMe. Reference vLLM-Moet = ~28–32 tok/s on 2× RTX PRO 6000
(96 GiB ea); 6 smaller memory islands ≠ 2 large ones.

**Models/artifacts**: official NVFP4 `/data/models/GLM-5.2-NVFP4` (434 GB, 47
shards, via `hf-mirror.com HF_HUB_DISABLE_XET=1`, ~22–46 MB/s); colibri INT4
(144 shards); vLLM-Moet TP4 pack (190 GB); TP2/PP3 pack (190 GB); image
`vllm-moet-sm120:v024`.

**Kernels OK on SM120** (not an unsupported-GPU fallback): decode T=2 max rel
err 2.086e-2 cos 0.999878; NVFP4 delta 1.818e-2/0.999884; AFRAG prefill T=128
2.591e-2/0.999883.

**Real generation (single request)**:

| runtime/layout | TTFT | decode |
|---|---|---:|
| colibri INT4 morning (partial residency, dense on CPU) | ~42 s | 0.12 tok/s |
| colibri INT4 afternoon **full residency** | — | **6.28–6.84 tok/s** |
| vLLM-Moet TP4, 12 GiB/card (25.3% cov) | 2.03 s | median 2.5 |
| vLLM-Moet TP4, 14 GiB/card (29.5% cov) | — | median 2.6 |
| vLLM-Moet TP2×PP3 (55–62% cov) | 5.39 s | 1.78 |

Morning: 16 tok in 132.46 s, hot tier 77.48 GB, RAM 175.94 GB, 71% hit; costs
matmul 81.37 s, disk 29.31 s, attn 14.28 s → disk loading, not CUDA, was the
problem. TP4 every miss replays whole step → +cache only ~+0.1 tok/s.

**Full-resident ladder** (PR #80; all `TEMP=0`, fixed prompt, cumulative):
baseline 150+150 GB fixed → 2.30 (4.15 s disk/20 tok); all 19,456 experts in
VRAM+RAM → 5.77 (0 disk); `REPIN=16` dynamic repin → 6.00; 24 physical cores
pinned (`OMP_PROC_BIND=spread OMP_PLACES=cores`) → +39.6% (3.64→5.08); prefill
corrects all 75 MoE layers once (454 ms, in TTFT) → 6.05–6.08; swap cap 32→16
per round → **6.10–6.28**; 256-token run 6.84. Winning layout: 9,343/19,456
experts on GPU (176.73 GB), 10,113 RAM (~191.3 GB), 0 s disk.

Rejected: faster/slower repin & swap counts (16/16 local optimum); second
prefill pass (5.86); lazy demotion (4.77); D2H recovery (6.15); extra pthreads
overlap (no gain); OpenMP restructuring; `numactl --interleave`, 2 MB THP,
12-core, VNNI int4×int8, normalized profiles (neutral); next-layer prediction
w/ GPU staging (recall 70.6–78.9%, +7.8% GPU cov, but PCIe contention → 5.39–
5.44; revisit only w/ dedicated streams).

**AVX-512 int4 kernel (candidate, later qualified)** — see §"AVX-512" below.

**Time at 6.00 tok/s**: expert matmul 5.96 s (56%), attention 2.62 s (25%),
other 19%. gprof 84.9% CPU in `matmul_qt`; per-thread RoPE sin/cos cache cut
projection/RoPE 13.2%.

**MTP spec decode retest (07-13)**: stock MTP head unusable (acceptance 0–4%,
int4-head defect, issue #8). Swapped to community int8 shards
(`mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp`). Speculation still lost at
all depths (int8 head good: 69–79% chained acceptance): water-cycle D0 6.79 /
D1 6.45 (79%) / D2 5.65 (64%) / D3 6.13 (69%); degenerate sky D0 6.12 / D1 5.51
(73%) / D2 4.42 (44%) / D3 3.86 (38%). Structural: S-position verify batch
routes to mostly distinct experts → per-forward expert time ~linear in S (80/168/
306 ms at S=1/2/4); no amortization. Set `DRAFT=0` explicitly (default `-1`
auto-enables 3 drafts and silently costs 10–37%). Revisit only after
Tensor-Core grouped GEMM makes S=4 verify near S=1 cost.

**AVX-512 int4 kernel qualification (07-13, `I4_ACC512`, default on)**: numeric
error 2–4× LOWER than scalar-f32 order (gate/up max rel 2.6e-4 vs 1.1e-3);
quality SCORE logprob −449.30 vs −448.70, PPL 5.99 vs 5.98 (0.24%, 4/4 sign
split); throughput +4% mean (+7.0–7.4% on CPU-heavy prompts, 0% when routed
from GPU tier) — earlier +15% retired. Merged (outputs not bit-identical to
the worse order; open policy call).

**Conclusions**: whole int4 model fits VRAM+RAM; full-resident colibri
6.28–6.84 tok/s beats every vLLM-Moet layout (≤2.6). Bottleneck is now the CPU
int4 expert matmul (56% / 84.9% samples). Ceiling on this machine 6.84 tok/s
(256-token greedy); 20–30 tok/s never demonstrated. Next lever: GPU experts as
a compute tier (Tensor-Core grouped GEMM), which would also reopen MTP
speculation. Actions: finer CPU-expert counters; attention/score-softmax-value
optimization once expert path >7 tok/s; keep A/B on fixed `TEMP=0` + fixed
token counts. Lab state: fastest = colibri full-resident
`/data/test/colibri-full-resident`; fastest vLLM-Moet = TP4 port 8000 GPUs 0–3
14 GiB/card.

---

## 3. GLM-5.2 on 4× RTX A6000 (Ampere) + EPYC 7402P — 2026-08-02

Engine colibri v1.3.0 @ `ecade075cfc2eae684097ea7de5570c3786ce199`; model
`mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp` (142 shards, 429.3 GB);
`make glm CUDA=1 CUDA_ARCH=sm_86`. Main contribution = **methodological**: naive
A/B is invalid on this engine (§6). Earlier claims retracted inline: prefill I/O
claim and the issue #776 CPU-fallback claim; §5–§8 re-measured under
snapshot/restore protocol.

**Host**: EPYC 7402P Zen 2 24c/48t, AVX2 no AVX-512, single NUMA; 8×32 GB
DDR4-2400 (263.8 GB, ~153 GB/s); 4× A6000 48 GB sm_86 PCIe Gen4 x16, driver
580.173.02; NVMe A Gen3 x4 → `/`, NVMe B Gen3 x2 → model dir. Storage: NVMe A
randread 4M QD4 2619 MB/s/5.79 ms, iobench 19M 16t 2.86 GB/s/7.0 ms per expert;
NVMe B 1521 MB/s/10.44 ms, 1.60 GB/s/12.2 ms.

**Protocol**: restart → warm-up → 5 measured gens; greedy `COLI_TEMP=0`,
`max_tokens=64`, single client, `--ctx 32768`, median. Concurrent clients
invalid (engine serializes → 0.43–1.04 tok/s). §5+ add: snapshot
`$MODEL/.coli_usage` once, restore byte-for-byte + `sleep 35` before every
config. Rules: never read cumulative log counters (report deltas); never sample
during cold start (429 GB load takes minutes).

**9-config matrix (1 session)**: E1 `RAM_GB=205` **5.46** (190 GB RSS / 177 GB
VRAM); D1 = C2 minus URING/PILOT* 5.38; D4 4.45; D2/D3/D5 (+TC_INT4 / +OMP
close / no CUDA_PIPE) 4.30; C2 `CUDA_DENSE=1` 4.26; C0 start (cap 64, no PIN)
1.53; C1 max RAM/VRAM w/o CUDA_DENSE 1.41; `SERVE_BATCH=1 KV_SLOTS=4` 1/2/4
concurrent 1.32/1.51/1.46.

Levers that mattered: **`CUDA_DENSE=1` = ×2.8** (1.53→4.26; moves dense+attn off
CPU; not in README); **remove URING+PILOT+PILOT_REAL+PILOT_TWO = +26%**
(4.26→5.38; once resident, disk reads 0 MB/s, prefetch only burns CPU). Neutral:
TC_INT4, OMP close, CUDA_PIPE. Counter-intuitive: max RAM+VRAM degrades when
dense stays CPU (C1 1.41 < C0 1.53); gains don't compose (RAM 190→205 +4.5% on
C2 base, +1.5% after URING removal); concurrency doesn't scale.

**Bottleneck**: nothing saturated (CPU 292–490%, GPU 20–22%/131 W, disk 0 MB/s);
limiter is serialization latency — 78 sequential layers, CPU↔GPU round trips,
S=1 matmuls. PROF at 64 gen tok/CTX 32768 (best config, 29.76 s): expert-matmul
13.75 s 46.2%, attention 7.80 s 26.2%, disk wait 3.94 s 13.3%, other 14.3%. At
CTX 8192 & ~750 tok: attention 64.2% (score-softmax-value 41.9%) — share grows
with generation length. **Ceiling setter**: CPU routed-expert path 19.67 GB/s =
23% of this host's ~85 GB/s; 8 experts × 20.1 MB × 75 layers = 12.1 GB/token,
6.6 GB non-resident → 333 ms/token = **3.0 tok/s ceiling today**; at full
DDR4-2400 → 13.0 tok/s hard ceiling. Open question: is 19.67 GB/s a broken
parallelization (OMP_NUM_THREADS sweep against that counter = single most
valuable outstanding measurement).

Prefill is attention-bound, not I/O-bound (issue #153: prompt doubles → disk
19.4→21.2 s flat, attention 49.2→118.0 s); `COLI_PREFILL_CHUNK` 2048 worth
taking. Controlled prefill sweep (`.coli_usage` restored): CTX 32768 198 tok/s
prefill / 1.56 decode; 65536 170 / 1.42; 98304 198 / 1.33; 131072 170 /
**1.20** (−23%); 196608 148 / 1.03 (−34%). No cliff, zero OOM at any level.
Earlier 2.6–4.3 prefill belonged to `--ctx 262144` + `RAM_GB=215` (cache
collapses to `cap=1`). Recommended `CTX=131072` (4× context, −23% decode).

**§6 — A/B invalidation**: identical memory placement measured 2.56 vs 5.46
tok/s. Hidden var = `.coli_usage` persistent profile (58,240 selections at
first start → 238,840 after 9-config matrix → **17,697,640** after a day of
real use). Pinned set re-tuned to real workload → benchmark became
out-of-distribution. **`.coli_usage` persists across restarts and grows
monotonically** — snapshot/restore required; absolute numbers are within-session
rankings only. Hypotheses refuted: "total residency governs" (98% residency
slower than 91%) and "VRAM residency governs" (identical-placement result);
bandwidth differential ≈1.3 ms of 77 ms/token (<2%). Under snapshot protocol:
A VRAM-heavy 188/200 2.81; B balanced 176/205 2.78 (indistinguishable); C
RAM-heavy 176/**235** → **3.51** (+25%) → **RAM budget, not split, is what
matters**. The 5.46 is not reproducible: config B now gives 2.78; a specialised
profile is worth ≈×2 and dilutes with use → motivates `COLI_USAGE_DECAY`
(merged #780): typical turn ≈38 k selections vs 18.19 M history = 0.2%.

**§7 Context vs residency**: 192 VRAM+251 RAM=443 GB vs 512 GB required at
262144 (−69 GB). At 262144 `RAM_GB=215` → `cap=1` (one slot/layer; every token
needs 8 disk reads) = budget failure, not context failure; at RAM_GB=235 reaches
196608 without collapse. MLA KV: `kv_lora_rank 512 + rope 64 = 576`/token/layer
× 78 = 44,928/token; fp32 179.7 KB → 47.1 GB @262144 (engine uses fp32,
`colibri.c:2277`); `kvb_all` 114.7 KB/position → 30.1 GB @262144 (defeats MLA's
purpose; `colibri.c:2568` notes O(T·kv_lora) alternative). llama.cpp `-ctk f16`
0.098 MB/token vs colibri 0.316 MB/token (×3.2 ≈57 GB avoidable = 83% of the
deficit). **Open**: fp16 KV + no `kvb_all` → long context + full residency
simultaneously reachable.

**§8 Profile**: 19,324 experts/76 layers, 18,189,640 selections. Not a power
law: top 1% (194) carries 6.8% (a power law would give 30–50%); max/median 18.7;
no expert unused (`noaux_tc` defeats hot/cold tiering). Heat pinning works with
diminishing returns (42.9% of model / 72.1% accesses ×1.68; 20.7% / 47.8% ×2.31).
Uniform per-layer cap ≈ global top-k (71.5% vs 72.1% = 0.6 pts) — not worth the
complexity. The lever is how many fit + profile/workload match, not which.

**§9 DRAFT self-speculation**: `--auto-tier` forces DRAFT=0; retest (restored
profile): DRAFT=0 1.70; DRAFT=3 0.85 (−50%); DRAFT=3+CUDA_MTP 0.96 (−44%) —
correct decision. But n-gram lookup only copies from context; novel-text prompt
measures structural worst case. Untested on copy-heavy workloads (code edit,
summarise, translate, structured output). Retest needs copy-heavy prompt +
logged acceptance (harness defect: acceptance not reported).

**§10 shard flags**: measured 4 conditions (restored profile): S0 neither 1.97;
S1 ATTN_SHARD 2.05 (+4.1%); S2 PIPE_SHARD 2.06 (+4.6%); **S3 both 2.15 (+9.1%)**.
Earlier "49% gap" was an artefact (750 vs 64 generated tokens; NGEN ignored,
ran to EOS).

**§11 Determinism**: 6 identical temp-0 requests → 1 distinct output (md5
e86d393c506f) — token-identical on 4 GPUs; enables byte-for-byte acceptance
tests (#399).

**§12 vs llama.cpp** (same KINGSTON SNV2S1000G class disk):
`unsloth/GLM-5.2-GGUF UD-IQ4_NL` (348 GB) `-cmoe` = prefill 239 tok/s, decode
1.58, VRAM 26 GB vs colibri 198 / 1.56 / 176 GB. llama.cpp matches decode, beats
prefill 20%, uses 26 GB. Per-expert placement buys nothing here — limiter is the
19.67 GB/s CPU read both suffer. llama.cpp `-ncmoe` splits by layer not bytes
(73 GB buffer on 48 GB card → fails to load).

**§13 Tuning 2.35→3.64**: baseline (48 threads, no CACHE_ROUTE) 2.35 → 2.70
`OMP_NUM_THREADS=24` (physical; never set on Linux, #805; libgomp falls back to
logical nproc = 2× oversubscription; 24thr 30.10 GB/s/2.90 tok/s vs 48thr
17.35/2.06; ~1.4 GB/s per Zen 2 core is the kernel ceiling → 34 GB/s host cap;
CPU expert path compute-bound per core, not memory-bound) → 3.59 `CACHE_ROUTE=1
ROUTE_M=16 ROUTE_J=1` (99.1% route_agree, 32.98 GB/s; #811; off by default;
M 12/16/24 within 1% — defaults right, decision is on/off) → **3.64**
`CUDA_EXPERT_GB=188` (tier saturates ~183 GB; no OOM any level). Ceiling:
6.6 GB non-resident/token / 32.98 GB/s = 200 ms of 275 ms/token (73%);
per-core compute limit → **realistic ceiling ~4.5 tok/s**; beyond needs more
VRAM or narrower quantization.

**§14 Build**: nvcc12.6+gcc13 fails (`_Float32`); `BUILD-cuda-glibc241.md`
recipe: `micromamba create -p ~/cuda13 cuda-nvcc=13 cuda-cudart-dev=13`, symlink
lib64, `make glm CUDA=1 CUDA_ARCH=sm_86`. Dashboard needs Node ≥20.12 (Ubuntu
24.04 ships 18.19).

**§15 Final config**: `CTX=32768 COLI_CUDA=1 COLI_GPUS=0,1,2,3 CUDA_DENSE=1
COLI_CUDA_ATTN=1 COLI_CUDA_PIPE=2 COLI_CUDA_TC_W4A16=1 COLI_CUDA_ATTN_SHARD=1
COLI_CUDA_PIPE_SHARD=1 CUDA_EXPERT_GB=188 OMP_NUM_THREADS=24 CACHE_ROUTE=1
ROUTE_M=16 ROUTE_J=1 PIN=auto PIN_GB=150 RAM_GB=235 COLI_USAGE_DECAY=0.9
DIRECT=1 PIPE=1 COLI_PREFILL_CHUNK=2048`. `MemoryMax=238G` on systemd unit.

**Open/operational**: `PIN_GB=all` unsafe steady state — after 2 days the
engine refused to start (26.4 M selections, projected 315 GB > 258 GB). Fix:
bound `PIN_GB` + `COLI_USAGE_DECAY`. Refusal is correct behaviour since #793
(under-counting previously → OOM-kill, #305/#766). Engine doesn't release VRAM
on stop (~30 s restart fails); `coli plan` computes vs *currently free* VRAM →
meaningless while engine runs.

---

## 4. GLM-5.2 continuous batching — 2026-07-31

Question: can the mux scheduler raise aggregate decode by evaluating 1 token
from several independent KV slots in one forward? Host 6×5090/251 GiB, int4
model, `DRAFT=0`, full residency (9,335 VRAM + 10,121 RAM, zero disk), frozen
placement, 8 KV slots `CTX=256`, batch order 1,2,4,8,4,2,1; timing starts at
last request's first token (`c/tools/benchmark_continuous_batch.py`).

| sessions | aggregate tok/s | per-session median |
|---:|---:|---:|
| 1 | 4.84 | 4.84 |
| 2 | 6.33 | 3.16 |
| 4 | 8.14 | 2.04 |
| 8 | 8.30 | 1.04 |

**No breakthrough**: saturates ~8–9 tok/s by 4 sessions ≈ existing single-stream
baseline (8.90), not a multiple. 8-session TBT median 0.852 s, p95 1.306 s —
trades latency without more total work. Profile (1,4,8): forward p50 267→522→813
ms; CPU expert BW 26.57→~31–35→~33–35 GB/s vs 66.88 GB/s single-stream; expert
matmul 69–72% of time; CPU side 5.6–11.1× slower than GPU critical path.
Multi-row CPU expert path loses ~half the RAM bandwidth — grouping doesn't turn
into weight reuse.

**Memory trap**: 8 default `CTX=4096` slots → only 9,335+5,321 pinned, 96.6–
99.8% hits but medians 3.64/3.51/5.48/6.00 — mixed batching with disk/LRU
warm-up, invalid. `KV_SLOTS`, `CTX`, expert placement are ONE capacity decision.

**Verdict/action**: functionally implemented but **not a throughput multiplier
— do not advertise**. Next bounded experiment: dedicated multi-row INT4 CPU
expert kernel (decode weights once, accumulate all rows); integrate only if it
restores ≥60+ GB/s single-row BW and pushes full-resident 4/8-session aggregate
above 8.9 tok/s. Artifacts under
`/data/test/colibri-contbatch-20260731-IucO7Z/`.

---

## 5. Lossless VRAM-base / RAM-residual split — 2026-07-31

Question: represent each routed INT4 weight as a small VRAM base + exact RAM
residual, restored losslessly on GPU? `c/tools/analyze_split_entropy.py` over
3,840 routed tensors (24.16 GB) from layers 3/10/30/50/76 + all headers:
routed packed 372.052 GB; F32 scales 0.797 GB; symbol entropy 2.9147 bits/weight
(72.87% of raw INT4); per-layer 2.7698–2.9505. At 176.57 GB expert-VRAM budget
(+ scales + 32-bit tile offset per 4096) base may average 1.882 bits/weight;
ideal entropy residual 96.06 GB (not practical). Tested global unequal-group
codebook: 1-bit base 2.559 residual bits; 2-bit base 1.258; 11.8%/88.2% 1-bit/
2-bit tiles fit VRAM → practical RAM residual **131.31 GB** = 3.294 bits/weight
(82.34% of raw INT4). Lossless symbol coding; info remains in RAM.

RTX 5090 microbench (exact reconstruction checked): 1-bit base+2.560-bit
residual decode 184.0 Gweight/s (92.0 raw GB/s), pinned H2D+decode 65.8 (32.9);
2-bit 184.3 / 97.8 (48.9). Capacity-weighted ≈91.9 Gweight/s = 46.0 GB/s/GPU →
balanced 6-GPU transfer/decode ceiling ≈ **23.8 tok/s** (upper bound, excludes
matmul/routing/attn/sync).

**Fused matvec follow-up killed it** (`benchmark_split_matvec.cu`, real
geometries 2048×6144 & 6144×2048 vs resident raw INT4 W4A32): final direct-
bitplane+per-warp-exception format (2-bit residual 1.491 bits/weight) still
3.4–3.8× slower before transfer, 7.6–9.5× with transfer; global-scan exact
variant bit-identical FP32. **Rejected for engine integration** — exchanges RAM
capacity for an address-dependent codec that destroys the cheap raw-INT4 loop,
loses existing CPU/GPU overlap. Capacity result still useful. Real claim would
need fused residual-decode+INT4 matvec, 6-GPU placement, 131 GB pinned store,
ABBA vs 8.90 tok/s baseline. Artifacts on `rs-yuesheng-gpu`.

---

## 6. GLM-5.2 decode failure ledger — 2026-07-31

Stop-doing list. Reference host: 6×5090, dual-socket DDR5, int4, greedy
`COLI_TEMP=0 DRAFT=0`, fully resident. Rules: no reopen because a microbenchmark
is faster; no cross-process compare without interleaved ABBA; compute experiments
must report residency/disk/CPU-BW/output equality/token count; "conditional" ≠
safe default; a new name/implementation doesn't invalidate the reopen gate.

**Rejected directions (key numbers)**:
- Whole-expert NUMA ownership: interleaved 8.84/8.92 tok/s 64.74/64.89 GB/s vs
  node-local 6.16/30.97 GB/s. Reject on this topology.
- One CPU task/expert-parallel OpenMP: ~23% regress; ~2.5 GB/s/expert. Reject.
- Impact/cost-aware placement w/o hard residency: fixed-replay 7.46→4.01
  (−46.2%), 490–491 disk loads/9.28 GB/4.29–4.46 s wait; capacity-preserving
  6.88 vs 7.26; held-out layout ~38% regression. Reject as implemented.
- Cross-layer prefetch (`PREFETCH`/`PILOT`/learned coupling): `PREFETCH=1`
  4.01→3.91 (−2.5%); learned coupling found 6 nonresident hints. Predictable
  experts already resident; staging contends with PCIe/attention. Reject for
  resident decode.
- WarpDecode low-row kernel: isolated 8-expert group block 0.248 vs warp 0.326
  ms (−31%). Reject on sm_120.
- Fused ANS decompress+matvec: same-process ABBA 6.505 vs 5.285 (−18.8%);
  cross-process +7.1% was drift. Reject; ANS residency still valid.
- GPU shared-expert microkernels / routed scatter-add in isolation: 110.08 s /
  135.27 s vs 94.05 s control. Reject in current split layer.
- GPU-resident residual across PCIe star: parallel dispatch +9.1% (6.82→7.44)
  kept; cross-layer extension rejected (no NVLink serializes).
- GPU zero-copy reads of RAM experts: 4.24 vs 1.71 tok/s; PCIe zero-copy ~2.8
  GB/s vs 56 GB/s bulk DMA; double-buffered parity only; registration +0.57 RSS
  byte/byte; cudaHostAlloc broke NUMA (CPU leg 38 GB/s). Reject on this host;
  reopen at 384–512 GiB RAM headroom or coherent interconnect.
- Static expert pruning/`EXPERT_BUDGET` as lossless speed: no win or incoherent;
  changes the model. Reject as lossless/default.
- Cross-expert common-basis extraction: shared mean energy 0.38–0.69% (~random
  1/256); 90% energy needs ~192–224/256 bases; 35% cut → 18–30% recon error.
  Reject engine-only.
- Permutation neuron alignment: nearest cos 0.246–0.253 vs random 0.245. Reject.
- SERE functional substitution: mean sim 0.0846/0.0213/0.0728, max 0.303/0.061/
  0.284 at layers 3/30/77, none ≥0.5. Reject.
- MTP as global default: full-resident D0 8.90, D1 7.51 (−15.6%, 60% acc), D3
  3.95 (−55.6%, 23%); predictable prompt D1 +9.2% @91%. Rejected drafts widen
  expert union. **Conditional, never default.**
- Adding GPUs w/o fixing placement: TP4 2.6 vs TP2×PP3 1.78. Reject as remedy.
- Repin/promotion sweeps: REPIN 8/32, 64 or 8 swaps, 2nd prefill pass 5.86,
  lazy demotion 4.77, D2H recovery 6.15 — all regress. Optimum 16 tok/16 swaps.
- Extra pthreads CPU/GPU overlap: 6.09–6.30, no stable gain. Reject.
- Generic CPU/JIT tuning: NUMA interleave, huge pages, 12-core, VNNI, OMP
  restructuring neutral/negative; AVX-512 changed accumulation order, diverged.
- Existing continuous batching as multiplier: 4.84/6.33/8.14/8.30 aggregate;
  CPU BW 33–35 vs 66.88 GB/s. Reject as speed claim, retain serving mechanism.

**Keep**: full residency (disk avoidance dominates); frequency/hotness
placement; selective NUMA interleave for RAM experts; parallel per-device
dispatch + async CPU/GPU path; ANS compressed resident tier (only fused
consumption rejected); mux scheduler + paged/ragged KV + prefix reuse + prefill
accel as serving infra (no throughput claim until multi-row CPU expert path
passes the gate); MTP as opt-in workload-specific feature.

**Remaining honest directions**: (1) persistent/whole-layer expert kernel
removing launches/host round-trips/reductions; (2) stronger trained draft with
joint acceptance + expert-union control; (3) retrained/distilled locality-aware
MoE derivative (model research); (4) hardware raising capacity/BW without worse
topology.

**Software ceiling**: best full-resident = **8.90 tok/s** (9,335 VRAM + 10,121
RAM, 100% hit, 0 disk, CPU BW 66.88 GB/s; ≈44% expert matmul / 34% attn / 22%
other). Every identified path to a general 1.5–2× engine-only win is closed.
Remaining headroom incremental: ~10–25%; practical stable target ~9.5–10.5
tok/s, exceptional ~11–12 (planning bounds, not measured). Beyond needs new
boundary: more VRAM, higher/coherent BW, trained route-aware draft, or
retrained MoE. Priority: stability, auto hardware planning, cross-machine
portability unless a proposal cuts bytes/token or a measured whole-layer cost.
Evidence index under `/data/test/glm52-cross-expert-structure-20260731.json`,
`/data/test/glm52-neuron-alignment-20260731.json`,
`/data/test/sere-calib-output/`, `/data/test/mtp-retest-20260731/`.

---

## 7. Inference-paper claim test matrix — 2026-07-28

Ledger of paper claims tested against controlled baselines (same HW, model,
prompt, token count, precision). Verdicts: confirmed / conditional /
rejected / pending-runtime / pending-model / not-applicable. Primary: 6×5090 +
dual Xeon Silver 4510 + 251 GiB; GLM-5.2 int4; greedy `TEMP=0 DRAFT=0`; fixed
prompt+length, warm-up, ABBA/rotated, ≥3 runs; median + breakdown, not best;
correctness gate = identical replay tokens (lossless claims) or quality+error
bound; batch-1 decode can't validate serving claims. Corpus: Kimi K3, Kimi
Linear, Mooncake, HybriMoE, SparseSpec, PagedAttention, Splitwise, HCAttention.

**Headline verdicts**:
- **M1** CPU/GPU intra-layer overlap: **confirmed, conditional** (+7.1% no NUMA,
  +2.0% after NUMA makes GPU tier the straggler).
- **M2** impact-aware vs frequency placement: **rejected on this host** — fixed-
  replay 7.46→4.01 (−46.2%), 490–491 loads/9.28 GB/4.29–4.46 s wait; capacity-
  aware bound collapses to frequency fill (1.000×). Replay oracle 13.01 (+48.9%)
  is in-sample only (38% held-out regression). Robust lambda=0.25 candidate
  −11.4%/−26.1% real A/B → rejected. `CUDA_EXPERT_LOAD_BALANCE=1` retained as
  experimental opt-in (+2.9% median, one workload).
- **M3** cross-layer prefetch: **rejected for available policies** (PREFETCH=1
  −2.5%; learned coupling ≈6 nonresident hints, misses/wait worsened).
- **M5** WarpDecode low-row kernel: **rejected on dev/RTX 5090** — isolated
  block 0.248 vs warp 0.326 ms (−31.5%, bit-exact both); apparent end-to-end
  wins were CPU-bandwidth drift. Not merged.
- **A4** decoupled page granularities: partly supported, model-limited (64-token
  page Pareto point; no KDA checkpoint).
- **K1** PagedAttention: **partly confirmed by prototype** — 64-token physical
  pages, copy-free growth, exact output; skewed 9-slot trace reserves 76.4%
  less than fixed slots (2,176 vs 9,216 positions, 8.1% fragmentation); page
  alias/refcount sharing + concurrency curve absent. `COLI_KV_PAGE_TOKENS`
  override; 64 retained as Pareto (16 minimizes waste but +47% growth time).
- **K2** prefix reuse: **confirmed** — second-request TTFT 17.689→0.569 s
  (31.1×), 257/264 tokens reused, 7 prefilled.
- **K3** page-alias (memfd/mmap) instead of prefix memcpy: **rejected** —
  coalesced alias 4.051 vs 3.156 s (+28.3%); existing exact memcpy adoption
  stays.
- **C1** ANS compressed resident experts: **confirmed** — +13.9% capacity;
  6.19→7.12 (32 tok), 6.75→7.31 (128 tok), output byte-identical.
- **C2** sidecar loading: **confirmed** — ~197 s buffered, ~157 s pinned,
  80–103 s aligned direct.
- **C3** fused ANS decode+matvec: prototype +7.1%/−11.3% → integrated ABBA
  6.505 vs 5.285 (**−18.8%**) → **rejected, not merged** (cross-process +7.1%
  was machine state). MTP layer-78 int8 must not enter fmt2 ANS sidecar.
- **P1** more GPUs don't help with per-rank serial placement: **confirmed**
  (TP4 2.6 vs TP2×PP3 1.78).
- **P2** parallel per-device dispatch: **confirmed** (+9.1%, 6.82→7.44); forced
  serialization control removes overlap (0.0% ≥2-GPU time). Residual-broadcast
  across PCIe **rejected** (~45 ms wire for 1.38 GB vs ~6 s CPU work — not a
  dominant cost, architecture rewrite).
- **P3** NUMA-aware placement: **confirmed** — 5.67→7.89 (no async),
  6.07→8.05 (async) in first controlled replicate; NUMA (+39.2%) ≫ overlap.
- **D1** split prefill/decode goodput: interference **confirmed** (501-token
  prefill raises active decode max TBT 0.188→44.205 s, TTFT 44.199 s); split
  benefit **pending** (no disaggregated path). Scheduler chunk prototypes
  rejected (32-token: max TBT −53.9% but median TBT 4.992 s / TTFT 204 s;
  no useful operating point; needs pipeline/layer preemption or separate
  workers).
- **D2** layer-wise KV transfer: supported only by calibrated model (111.5 MB KV,
  ideal streaming hides 98.7% of link time; no transfer runtime exists).
- **D3** prefix-affinity scheduler: mechanism confirmed (saves 17.12 s TTFT),
  multi-instance scheduler unvalidated.
- **D4** Splitwise break-even: calibrated only (501-token request exposes
  11.29 ms @1 Gb/s, 0.11 ms @100 Gb/s under ideal streaming; RPC/queueing
  unmeasured).
- **S1** MTP speculation: **confirmed, strongly conditional** — after grouped/
  async/full residency: D1 +9.2% (91% acc, 8.79 vs 8.05), D2 −6.5% (73%), D3
  +5.7% (77%); on explanatory water-cycle text D1 **−17.7%** (62% acc, +2,846
  CPU rows). Old "MTP always loses" verdict is dead; acceptance alone
  insufficient — routed-expert union cost is the missing variable. Runtime
  guard: 70% acceptance / 24 proposals default (`COLI_MTP_GUARD_PCT/WINDOW`),
  cuts worst regression 17.7%→7.8%, preserves predictable-text win.
- **M4/M6/A1/A2/A3/S3/K3-runtime**: pending-model (LatentMoE, MXFP4 QAT,
  KDA-MLA, KDA prefill, typed unified pages, EAGLE-3 draft, HCAttention
  quality) — need checkpoints/training; microbenchmarks can't validate quality.
- **S2** grouped-verify reversal: partly confirmed (D1/D3 now positive, but
  deeper verification grows expert work, prompt-sensitive).
- **S4** delayed verification: pending-runtime (needs S2/S3 draft).

**Other controlled results**: CPU thread-count sweep — 24 physical threads
retained (20 thr 8.09 vs 24 thr 8.35 tok/s; 48 SMT 6.19; broad-sweep "20 wins"
was drift). Fused FP32 gate/up pair retained (separate VNNI IDOT 1.5% worse,
alters routing). Per-device CUDA group telemetry retained (measurement-only).

**References**: Kimi K3 (2026), Kimi Linear (2025), Mooncake (2024),
HybriMoE (2025), SparseSpec (2025), PagedAttention (2023), Splitwise (2023),
HCAttention (2025).

---

## 12. Laguna Engine Optimization & 2k Prefill Sweeps — 2026-08-13

Host: Apple M5 (4P+6E CPU, 32 GB RAM, ~126 GB/s), Metal build `c/laguna_xs_metal` / `c/laguna_s_metal`. Model: `Laguna-XS-2.1-oQ2` (39 MoE layers, $E=256$, top-$K=8$).

### Summary of Results
- **2k Prompt Prefill**: Wall time reduced from **37.5 s → 29.2 s** (**22% overall speedup**).
- **M3 (Sliding Attention GPU Gate Threshold $K=1$)**: `c/laguna_common.h` updated gate threshold to $K=1$ (`pos0 + S >= c->window`), dropping prefill attention phase from **10.6 s → 4.2 s** (**60.4% faster**).
- **M6 (Grouped Expert Metal Occupancy & Unaligned Vector Loads)**: `c/laguna_expert_metal.mm` retained tile geometry `TM=32, TN=32, TK=64, NSG=4` with 32-bit `ushort2` vector loads. The measured expert delta improvement applied to the prefill/grouped candidate path; decode remains on CPU because the S=1 GPU route regressed.
- **Parallel Token-Indexed OpenMP Gather/Scatter**: `c/laguna_common.h` parallelized token gather and scatter-add with `pos_map` lookups, eliminating atomic contention and single-threaded 163 MB memory copies per layer.
- **M4 (Decode Attention UDOT `LG_DEC_Q8`) candidate**: int8 query scoring dropped one measured 4k attention phase from **97 ms → 72 ms**, but the numeric path and end-to-end result did not justify another decode mode. Removed in `fc20f78`.
- **Adaptive speculative decode candidate (`LG_SPEC_ADAPT=1`)**: depth-survival counters and rolling acceptance were implemented and measured, then removed in `fc20f78` with the regressive speculative decode path. It is not a current engine feature.

### Prefill Ceiling & Hardware Analysis
- On Apple M5 with 39 MoE layers, 2k prompt prefill has a hard hardware/architectural floor (~2.5–4 s). Inter-layer execution is sequential because layer $l+1$'s router needs layer $l$'s output ($X_{l+1} = X_l + \text{MoE}_l$). FP16 expert staging introduced $5.5 \times 10^{-2}$ relative error through gate/up/down, so the retained expert kernel uses FP32 staging to maintain fixture parity.

---

## 13. Laguna-XS Real-Model 4k Decode Audit — 2026-08-13

**Question:** can semantics-preserving tuning bring Laguna-XS oQ2 close to 140 tok/s at a 4k context window?

**Host/model:** Apple M5 (4P+6E CPU, 10-core GPU), 32 GiB unified memory; real `Laguna-XS-2.1-oQ2`; Metal engine; commit `fc20f78`. Deterministic prose seed 1234 produced 4,084 actual prompt tokens. Runs used `CTX_MAX=8192`, `LAGUNA_MEM_GB=20`, 128 generated tokens, automatic four-P-core OpenMP tuning, and plain greedy decode. The benchmark launches separate prefill-only and decode processes; phase deltas subtract the former from the latter.

| configuration | prefill | prefill tok/s | decode tok/s | result |
|---|---:|---:|---:|---|
| baseline | 51.1 s | 79.92 | **4.70** | reference |
| harness-only rerun | 50.3 s | 81.19 | **4.63** | neutral; variance, no engine change |
| `OMP_NUM_THREADS=10` | 49.9 s | 81.84 | **4.84** | +3%; insufficient |
| `CAP=256` | 49.3 s | 82.84 | **4.40** | regression |
| `LAGUNA_MEM_GB=24` (auto cap 256) | 46.1 s | 88.59 | **3.95** | regression |
| `LG_SEL=1 LG_SEL_MIN=1 LG_SEL_CAP=512`, 10 threads | 47.9 s | 85.26 | **5.94** | changes attention semantics |

**Baseline phase deltas:** expert fill 45.3 ms/token, routed expert 56.2 ms, shared expert 6.2 ms, attention 82.0 ms. The post-harness run measured 54.7/54.7/4.7/81.2 ms respectively. A 32-token sampled profile reported 38.0% self samples in `__workq_kernreturn`, 34.3% in `__psynch_cvwait`, 14.6% in `matmul_oq.omp_outlined`, 3.4% in `matmul.omp_outlined`, and 1.2% in `attention.omp_outlined`; profiler wall is not used as the throughput number.

**Decision:** 140 tok/s needs 7.1 ms/token; the baseline needs about 213 ms/token. Even removing all measured fills leaves a roughly 6 tok/s projection. Full cache, extra memory, and extra CPU threads do not alter that order of magnitude. The only larger measured gain used sparse attention and is not a faithful default. No inference optimization landed from this audit; the useful change is a reproducible harness that defaults to 4k/128, reports actual token count/prefill rate, records commit/dirty state/binary SHA/config/hardware, rejects reused result tags, and serializes bench/stress through one lock.

**Next credible experiment:** one resident whole-forward Metal decode graph, not per-matrix dispatch: fuse or batch attention, projections, routed/shared experts, and LM head so command-buffer waits are counted per token rather than per layer/matrix. Gate it on real-model end-to-end throughput and the existing 24/24 teacher-forced plus 12/12 generated Metal fixture. Earlier per-layer Metal decode implementations measured 3.41-3.48 vs 4.98 CPU and 0.49 tok/s for routed experts, then were removed.

Raw local artifacts from this run: `/tmp/lgbench/baseline-4k`, `/tmp/lgbench/after-harness-4k`, and `/tmp/lgstress/after-harness-4k`. They are not committed; the harness metadata makes future shared artifacts attributable.
