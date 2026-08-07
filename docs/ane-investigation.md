# Apple Neural Engine: researched, and why it cannot help this engine

Asked to investigate the ANE. Conclusion up front: **the ANE is unusable for this
workload**, for three independent structural reasons, not for lack of effort or a
missing API binding. This file records the evidence so the question does not get
re-opened on marketing numbers.

## Sources

- Orion: *Characterizing and Programming Apple's Neural Engine for LLM Training
  and Inference*, arXiv 2603.06728 — the first open end-to-end system driving the
  ANE directly via the private `_ANEClient`/`ANECompiler` APIs.
- maderix, *Inside the M4 Apple Neural Engine (part 2): benchmarking, SRAM
  characterization, and the 38 TOPS myth* — hardware-level characterization.
- Apple Developer Forums thread 833036: "There is no public API or language for
  programming the ANE directly, equivalent to Metal Shading Language for the GPU."

Local confirmation that the hardware is present and reachable in principle:
`/usr/libexec/aned` is running and ioreg exposes ANE nodes. So availability is
not the issue.

## Finding 1: "38 TOPS INT8" is not a compute number

The ANE **dequantizes INT8 to fp16 before computation**. Measured peak is
**~19 TFLOPS fp16**; INT8 saves memory bandwidth only, not cycles.

That matters because it removes the headline reason to want the ANE. Against what
this machine actually delivers, measured in `c/tools/`:

| unit | measured | source |
|---|---|---|
| CPU f32 FMA | 59 GFLOP/s | `bench_isa.c` |
| CPU UDOT int8 | 298 GOP/s | `bench_isa.c` |
| **GPU MPS f16** | **15572 GFLOP/s** | `mps_gemm.mm` |
| ANE fp16 | ~19000 GFLOP/s (published peak, M4/16-core) | Orion |

Best case the ANE is **1.2x the GPU**, and that compares its *peak* against the
GPU's *measured* rate. With the SRAM cliff below it lands under the GPU.

## Finding 2: weights are baked at compile time — fatal for MoE

> "the ANE bakes weights at compile time: weight tensors are embedded in the
> compiled program and cannot be mutated post-compilation"

Laguna picks **top-8 of 256 experts per token**, so the weight set changes
constantly. Each distinct weight set needs its own compiled program:

| | |
|---|---|
| distinct (layer, expert) weight sets in one 6144-token prefill | **9,984** |
| ANE recompile cost each | 494 ms optimized … 4200 ms naive |
| total compile time | **4,932 s … 41,933 s** |
| current CPU cost for *all* expert math | 43 s resident / 168 s streaming |
| ANE compilations-per-process limit | **~119** |

9,984 programs needed against a ~119 limit. This is not a tuning problem; the
hardware model is incompatible with per-token expert selection.

## Finding 3: 32 MB SRAM cliff

Throughput drops ~30% once the working set exceeds 32 MB.

| tensor set | fp16 size | vs 32 MB |
|---|---|---|
| one expert (gate+up+down) | 6.3 MB | fits |
| one layer's 256 experts | 1611 MB | **50x over** |
| one layer's attention projections | 58.7 MB | 1.8x over |

Even the static attention projections — the one part with no recompile problem —
exceed the budget, so the ANE would run them at roughly 13 TFLOPS, i.e. **slower
than the GPU path already shipping**.

## Finding 4: decode would get worse, not better

Orion measured GPT-2 124M at **CPU 283 tok/s vs ANE 170 tok/s**, attributing the
loss to per-dispatch IOSurface overhead. There is also a ~49 KB minimum IOSurface,
so a single-token tensor must be padded (their example: 3072 bytes padded to
24576).

This matches what I measured independently for Metal earlier in this work: a GPU
dispatch round-trip is 0.327 ms on this machine, which is why decode stays on the
CPU. The ANE has the same problem with an extra padding tax.

## Finding 5: access requires private APIs

Core ML is the only public path, and it does not let a caller force ANE placement,
inspect programs, or control residency. Direct access means `_ANEClient` /
`ANECompiler`, i.e. private APIs — unshippable, and they are what Orion had to
reverse-engineer.

## Verdict

| phase | current | could ANE help? |
|---|---|---|
| routed experts (168 s / 43 s) | CPU UDOT | **No** — baked weights vs per-token selection; 9,984 programs vs 119 limit; 50x SRAM |
| attention projections (61 s) | GPU MPS f16 | **No** — 1.8x SRAM cliff puts it below the GPU it would replace |
| decode (5-14 tok/s) | CPU | **No** — measured slower than CPU on the same class of workload |

The one genuinely useful thing the research surfaced is a *negative* that
redirects effort: since INT8 buys bandwidth rather than cycles on fixed-function
hardware, and the ANE's peak barely exceeds the GPU's measured rate, **the GPU
remains the right target for everything the CPU cannot do**. The next optimization
is therefore the routed experts on Metal, not the ANE.

One ANE detail worth remembering if this ever changes: 1x1 convolutions measure
~3x the throughput of an equivalent matmul on that hardware, and deep op graphs
(16-64 ops) reach 94% utilization versus ~30% for single operations. Any future
ANE attempt must submit fused graphs, never individual GEMMs.

## What was built instead, and what it cost

Acting on the research's conclusion, the routed experts were wired for the GPU:
`lg_metal_upload_f16` + `lg_metal_gemm_rows` stack all E experts of a layer into
one f16 buffer so a per-expert GEMM is an offset into it, uploaded once at load
(`LAGUNA_GPU_EXPERT_GB`, opt-in, default off).

**It is present but inert, and the guard saying so is deliberate.** The MoE expert
loop runs inside `#pragma omp parallel`, and the Metal wrapper shares global
scratch buffers plus cached MPSMatrix objects across calls. Concurrent threads
raced on them, which surfaced as:

```
MPSMatrixMultiplication.mm:3240: failed assertion
`Number of requested rows in result exceeds result matrix size.'
```

One thread rebuilt the cached result wrapper for its own N while another was
mid-encode with a different N. I chased two wrong theories first (stale descriptor
after buffer growth, result-row mismatch) before the actual cause -- the shared
state being touched from a parallel region -- became clear.

Rather than ship a racy fast path, the call is gated behind `!omp_in_parallel()`,
so it compiles, uploads, and never fires. Making it pay off needs either per-thread
Metal state or the expert loop restructured to gather all groups and issue one
batched dispatch from a single thread. That is real work and it is not done.

Bounded by memory anyway: at 1.6 GB per layer in f16, only **4 of 39 MoE layers**
fit in an 8 GB budget, so even a working version would cover a tenth of the phase.
The honest conclusion from the ANE investigation stands -- the GPU is the right
target -- but the win needs a restructured dispatch, not just a kernel call.
