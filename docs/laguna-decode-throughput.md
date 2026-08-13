# Laguna-XS decode throughput: what was tried, what worked

Starting point: `task laguna:xs Q=oQ2`, LAGUNA_METAL, real Laguna-XS-2.1-oQ2
checkpoint, 2044-token prompt, decode measured at 6.1-6.6 tok/s. The user's
own reference point was omlx's ~100 tok/s on the same class of hardware, and
a roofline estimate from CPU UDOT's own measured throughput (298 GFLOP/s)
against Laguna-XS's ~2 GFLOP/token puts a ceiling near 147 tok/s. This
documents the gap-hunting, most of which did not pan out.

## Ruled out: disk streaming

The initial theory was that the SSD-backed streaming expert cache was the
bottleneck. Measured directly: at decode, `[mem] expert cache 256/layer`
already holds the WHOLE bank (256 of 256 experts), hit rate 99.3-99.5%,
essentially zero `pread` during steady-state decode. This model's expert bank
(11.8-13.7 GB) fits comfortably inside a 20 GB budget, so it was never
actually streaming from disk during decode. (Laguna-S is genuinely
disk-bound -- its bank does not fit -- but that's a different model and this
investigation used Laguna-XS throughout.)

## Ruled out: forcing GPU dispatch at decode's S=1

Tried `LG_GPU_EXP_MIN=1` to route decode's single-token batch through the
same Metal grouped-GEMM kernel used at prefill. Measured: catastrophically
worse (>300s for 30 tokens vs a 29s baseline). Confirmed the existing code's
own reasoning (`S >= 64` gate) still holds after this session's prefill
tile-list/chunk-size work: GPU dispatch round-trip overhead dominates when
each dispatch covers ~1 row.

## Ruled out: OMP spin-wait / thread count tuning

> Status correction (v1.6.0 merge, 2026-08-13): the wording below said the team
> was "sized to physical cores already by omp_tune.h", but laguna_common.h then
> did NOT include or call omp_tune.h — every other engine did. That gap is now
> closed: `coli_omp_tune_threads(LAGUNA_NAME)` runs at main() entry. The
> spin-wait half of the family tuning stays off for laguna (measured inert/regres-
> sion on Apple Silicon, #707); only the physical-core sizing applies.

- `OMP_WAIT_POLICY=active`: no measurable change (5.5-5.8 vs 6.1-6.6 tok/s,
  within run-to-run noise). Once the missing physical-core sizing landed, the
  tiny fixture parity harness measured 3.4x faster decode (453 -> 1528 tok/s)
  than the SMT-wide default team an uninstrumented launch would otherwise use;
  spin-wait's documented win (docs cite a -2.2x case) was on a different, more
  disk-bound regime.
- `OMP_NUM_THREADS=1`: measurably WORSE (1.87 tok/s). Confirms the existing
  parallel regions are net-positive, not a source of pure overhead to strip.

## Tried and reverted: resident Q8R bank at decode instead of the streaming LRU cache

The resident path (`m->resident`) is unconditionally faster at PREFILL, but
was gated off whenever the GPU expert path (`m->gpu_exp`) was active, which
it always is on a Metal build. Reasoning at the time: the GPU path only ever
runs prefill (`S>=64`), so decode always falls to the CPU regardless, and
gating residency on `gpu_exp` looked like an oversight rather than a real
memory conflict (GPU-mapped weights are zero-copy/evictable page cache, not
charged against the resident-bank budget).

Removing the gate and fixing the budget-ordering bug it exposed (the
streaming cache was sized and charged to the budget BEFORE the resident-bank
fit check ran, so a bank that only fit once the now-unnecessary cache didn't
exist was wrongly declined) was implemented and measured. Result: decode got
WORSE, not better -- 3.06-3.43 tok/s vs 6.1-6.6 tok/s streaming, a ~2x
regression.

Root cause, confirmed via `sample`-based profiling: the resident path's OMP
loop is `for (e = 0; e < E; e++)`, i.e. it scans every one of E=256 experts
every call to find the ~8 (topk) that have rows this token, via a
`schedule(dynamic,1)` team dispatch. That is free at prefill (hundreds of
tokens spread across most of E, so few iterations are wasted), but at
decode's tiny batch it is 248 near-instant no-op iterations of scheduling
overhead per layer per token.

A follow-up fix (compact the active-experts list before the OMP loop, so it
only iterates the ~8 that actually have rows -- mirroring the "compacted
tile list" pattern the GPU expert path already uses for the same reason) was
implemented and re-measured: it did NOT recover the regression (still
~3.0-3.4 tok/s). The dominant cost at decode's resident path is not the
E-scan itself; a `sample` profile of the resident path showed
`__psynch_cvwait` at ~67% of samples (78948/117000+), vs ~45% for the
streaming path's profile in the same session -- i.e. the resident path pays
MORE synchronization overhead per token even after removing the wasted
iterations, likely because its inner allocation/quantization work per active
expert is structured differently (per-expert `Q8Act` alloc/realloc inside the
parallel region) than the streaming path's pre-batched approach. This was not
root-caused further; the resident-at-decode idea was abandoned and both
changes (gate removal + budget-ordering fix + compact list) were reverted.
**Do not re-attempt "just remove the gpu_exp gate" without addressing
whatever makes the resident loop's per-token overhead structurally higher
than the streaming path's -- it has now regressed decode twice.**

## Landed: fix a genuine nested-parallel bug in matmul_oq

`matmul_oq` (oq.h) had an unconditional `#pragma omp parallel`, unlike its
sibling `q8r_gemm` (q8r.h) and `matmul_w`'s other call sites, all of which
guard with `if (!omp_in_parallel())`. The streaming MoE path calls
`matmul_oq` (via `matmul_w`) from inside its own
`#pragma omp for schedule(dynamic,1)` region, so every one of those calls
attempted true OMP nesting, which is disabled by default and degrades to a
wasted fork+join per call instead of a no-op.

Fixed to match the existing guard convention. Measured impact, clean A/B (3
runs each side, Laguna-XS-oQ2, LAGUNA_METAL, identical 40-token decode):
expert-mm phase 18.3-18.6s -> 17.2-18.3s. Real, but modest (~4-5%), not the
dominant cost.

## Where the gap to 147 tok/s actually is (best evidence so far, not fully closed)

`sample` profiling consistently shows `__psynch_cvwait` (OMP worker threads
idle between dispatches) as the largest single bucket at decode, both before
and after the matmul_oq fix. Back-of-envelope check: 40 layers x ~2 OMP
parallel regions (attention + moe) = 80 regions/decode-token; at ~164ms/token
(6.1 tok/s) that is up to ~2ms/region if ALL of it were fork/join overhead,
which is 40-200x a normal OMP fork/join cost (10-50us) -- so the profiler's
`__psynch_cvwait` dominance mostly reflects worker threads legitimately
idle between short real-work bursts (attention's per-kv-head loop, the
expert GEMM's per-active-expert loop), not a fixable synchronization bug at
that scale. The remaining gap to the 147 tok/s roofline is most likely
architectural: 40 sequential per-layer OMP regions each doing tiny
(topk=8-row, or 1-row-per-kv-head) work is inherently far from the compute
density that 298 GFLOP/s UDOT throughput assumes.

The single most promising remaining lever, not yet attempted in this
session: **speculative decoding** (draft several tokens cheaply, verify them
in one batched forward that reaches the efficient bulk-token code paths this
session already optimized for prefill). This repo already implements the
mechanism for `deepseek_v4.c` (MTP head + verified draft/accept) but has none
for Laguna. This is a real, substantial engineering effort (new draft
mechanism + verification path + KV rollback on rejected drafts), not a
config knob, and was intentionally not started without confirming the
tradeoff with the user first.
