# GPU flash attention for prefill

## Result

Laguna-XS, 29,967-token prompt, same prompt and settings, CPU attention versus GPU:

| phase | CPU | GPU | |
|---|---|---|---|
| **prefill total** | 1110.4 s | **513.1 s** | **2.16x** |
| attention | 839.8 s | **229.7 s** | **3.66x** |
| expert-mm | 214.3 s | 228.3 s | 0.94x |
| shared expert | 12.3 s | 11.8 s | 1.04x |

Shorter contexts:

| tokens | attn before | attn after | prefill now |
|---|---|---|---|
| 2,000 | 16.5 s | **10.3 s** | 26.0 s |
| 6,000 | 63.9 s | **32.4 s** | 82.2 s |
| 29,967 | 839.8 s | **229.7 s** | 513.1 s |

Attention is no longer the dominant phase — 229.7 s against expert-mm's 228.3 s.
The profile is balanced now, which changes where the next win has to come from.

## Design

`c/laguna_attn_metal.mm`. Only the **full-attention layers** go to the GPU: at 30k
they were 110 of 122 TFLOP (90.7%) and are the only part that grows as O(S²).
Sliding layers are O(S·512) and stay on the CPU where they are already cheap.

A persistent per-layer f16 K/V cache lives on the GPU, appended once per chunk, so
keys are uploaded once rather than re-read per query. Sized to `CTX_MAX` and
charged against `LAGUNA_MEM_GB` like everything else (1.64 GB at 40k context for
XS); if it does not fit, the layer silently uses the CPU path.

Per query head: `QK^T` → causal softmax → `PV`, all three encoded into one command
buffer, with the softmax and the gather/scatter as custom shaders and **both GEMMs
on MPSMatrixMultiplication**. The score matrix never leaves the GPU — downloading
it to softmax on the CPU would be 256 MiB per head at 256k context.

The output gate (softplus of `g_proj`) is folded into the scatter shader, so the
result comes back ready to use.

## A rewrite that was worth doing

The first version was a hand-written kernel: one thread per (query, head), walking
keys serially with an online softmax. It worked and was token-exact, but measured
**81 GFLOP/s — about 1% of the 15572 GFLOP/s this device demonstrably reaches**.
Attention only went 63.9 → 54.5 s at 6k.

The cause was structural, not a tuning gap: 12288 threads each running a
6000-iteration dependent loop, with no cooperation, no vectorization and no reuse.
No amount of tweaking fixes that shape.

Rewriting it so scores and PV are plain GEMMs handed to MPS took it to
**480 GFLOP/s** on the same work, a 5.9x improvement over my own kernel. The
lesson is the same one the ANE investigation produced: on this hardware, hand-
written kernels lose to Apple's tuned GEMM, so the job is to express the work as a
GEMM rather than to write a better loop.

480 GFLOP/s is still only 3% of the MPS ceiling, because at chunk=256 the GEMMs are
small and there are three dispatches per head per chunk (48 heads x 3 = 144 command
encoders per layer per chunk). Batching all heads into one MPS batched GEMM is the
obvious next step and was not done.

## Correctness

Token-exact with GPU attention active on both fixture models: XS 24/24
teacher-forced + 12/12 generated, S 208/208 + 8/8. The CPU builds are unchanged and
still exact, and the GPU path falls through to the CPU on any failure, which is why
the CPU KV append still runs unconditionally.

Worth noting: the GPU path runs in f16 with an f16 score matrix and still produces
identical tokens, on top of the int8 KV cache also being token-exact.

## Where prefill stands now

| context | prefill | peak RSS |
|---|---|---|
| 2,000 | 26.0 s | 20.1 GB |
| 6,000 | 82.2 s | 20.4 GB |
| 29,967 | 513.1 s | 19.9 GB |

RSS is flat across a 15x context range. Note it now sits near 20 GB rather than the
17.6 GB before this change, because the GPU K/V cache is real additional memory
(1.64 GB at 40k) — it is inside the default budget but it is not free.

## Still open

- **Batch the per-head GEMMs.** 144 command encoders per layer per chunk is the
  main reason this reaches 3% rather than 30% of the MPS ceiling.
- **Expert path is now co-dominant** (228.3 s vs attention's 229.7 s). The batched
  GPU expert dispatch exists but only 1-4 layers fit in the budget at f16.
- **256k**: memory fits (7.94 GiB dirty for Laguna-S), and the time projection
  improves from ~18 h to roughly 5 h with this change. Still not practical, and
  still bounded by O(S²) on the full-attention layers.
