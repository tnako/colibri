# GPU experts via grouped GEMM, from mmap'd weights

## The unlock: weights the GPU reads but the process never owns

Measured first, before any of this was built (`c/tools/nocopy.mm`):
`newBufferWithBytesNoCopy` accepts an mmap'd safetensors shard. 5.21 GB wrapped,
a compute dispatch ran against it, `MTLCommandBufferStatusCompleted`.

That single fact changes what "fits in 20 GB" means. On Apple silicon the GPU
reads the *same physical pages* the CPU mapped, and clean file-backed pages are
evictable page cache rather than dirty RSS. So Laguna-S's **28.4 GB of 2-bit
expert weights are mapped in 0.1 s and never enter the memory budget at all**.

This retires the whole problem that blocked Laguna-S:

| approach | resident cost | verdict |
|---|---|---|
| Q8R resident bank | 39 GB | impossible in 20 GB |
| streaming LRU cache | 8-14 GB + disk IO | slow, and still charged to the budget |
| **mmap + GPU dequant** | **0 GB** | what ships |

The kernel dequantizes oQ in-register (`w = q*scale + bias`, per group of `gs`
along K) and accumulates with `simdgroup_matrix` 8x8 tiles. Validated standalone
against an exact CPU dequant: **max rel err 5.7e-05, 0/2560 values over 1e-3**.

## Throughput, and why the tile shape mattered

`c/tools/bench_expert.mm`, one real Laguna-S expert matrix (Kd=3072, N=1024,
2-bit gs128):

| rows | GFLOP/s |
|---|---|
| 8 | 31.6 |
| 32 | 181.8 |
| 128 | 835.1 |
| 512 | 1386.6 |
| 2048 | **1515.4** |

vs 298 GOP/s for the CPU UDOT kernel it replaces.

Two things were tried to get there. Widening the threadgroup tile from 32x32 with
4 simdgroups to 64x32 with 8 took it 1125 -> 1515 GFLOP/s. Staging in f16 looked
better still (1479 at the narrower tile) but **broke accuracy**: max rel err
5.5e-02, 313/2560 values off, because expert output feeds `silu*up` and then
`down_proj`, so f16's 11-bit mantissa compounds through three matmuls before the
residual. `simdgroup_multiply_accumulate` needs matching operand types, so mixed
f16-weight/f32-activation is not expressible either. f32 staging stayed.

## The dispatch-granularity trap, twice

The first integration issued **one dispatch per (expert, matrix)**: 256 experts x
3 matrices x 48 layers = **36,864 dispatches per chunk**, each with its own
`commit` + `waitUntilCompleted`. The run appeared to hang. `sample` on the live
process was unambiguous:

```
13.8% CPU
35330 samples  __psynch_cvwait
 3533 samples  iokit_user_client_trap
```

Blocked on dispatch, GPU idle.

Batching into one command buffer per layer did **not** fix it -- dispatches inside
a single encoder still execute sequentially, so 768 tiny dispatches per layer just
became 2 waits around the same serial work.

The fix is the grouped-GEMM shape the MoE literature already describes (PyTorch's
Triton grouped-GEMM post, DeepGEMM): **one dispatch covering all experts**, with
`grid.z` indexing the expert and each threadgroup reading its own `(row0, rows)`
from an offsets buffer. Three dispatches per layer instead of 768, and the GPU
schedules every expert's threadgroups concurrently.

Worth recording plainly: that pattern was in the research read at the start of
this work, described correctly in the plan, and then not implemented for two
rounds. The profiler, not the reasoning, is what forced the correction.

## Prefill only, and the measurement that forced it

The kernel is gated to `S >= 64`. Decode (S=1) presents topk=10 rows spread over
10 experts -- the kernel's worst regime (31.6 GFLOP/s at 8 rows) plus 3 dispatch
round-trips per layer with nothing to amortize them.

This was found by trusting a counter that looked broken. `expert-mm` reported
**470.5 s inside a 77.1 s prefill**, which I first assumed was a timer-scope bug.
It was not: the `[phases]` line prints after generation, so it legitimately
included decode, and decode really was spending ~390 s in the expert phase for 4
tokens. Gating to prefill:

| | before gate | after gate |
|---|---|---|
| expert-mm (prefill+decode) | 470.5 s | **48.9 s** |
| decode | 4 tokens, most of 470 s | 8 tokens in **8.6 s** |

The phases now reconcile: 48.9 + 22.2 = 71.1 s of a 74.5 s prefill.

## Result

Laguna-S 2.1 oQ2e-fast, 1433-token prompt, default 20 GB budget:

| | CPU baseline | GPU grouped |
|---|---|---|
| **prefill** | 322.7 s | **74.5 s (4.33x)** |
| expert-mm | 217.6 s | 48.9 s |
| attention | 56.4 s | 22.2 s |
| peak RSS | 12.7 GB | **9.9 GB** |
| expert weights resident | 8-14 GB cache | **0 GB** (28.4 GB mapped) |
| disk IO during prefill | yes | none |

At 262144 context the engine configures inside the budget at **7.24 GB** resident
(GPU attention declines its 14.24 GB allocation and those layers use the CPU).

Token-exact throughout: XS 24/24 + 12/12, S 208/208 + 8/8, `task test` 5/5.

## Honest gap to the 100x target

**4.2x, not 100x.** The arithmetic for 100x was checked and is not absurd -- it
needs 6.97 TFLOP/s against a 15.57 TFLOP/s measured MPS ceiling, i.e. 45% of peak.
The expert kernel currently runs at **1515 GFLOP/s, about 10% of that ceiling**, so
the compute headroom is real but unclaimed.

Where the remaining time goes at 1433 tokens (76.1 s prefill): attention 23.5 s
(31%), the rest experts + shared expert + projections. Reaching 100x would require
roughly 24x more from *every* phase simultaneously.

Known, specific, not done:
- The kernel stages B through threadgroup memory after dequant, so every 8x8 tile
  costs a round-trip and the dequant is redundant across row-tiles.
- Rows-per-expert is `chunk*topk/E`; at `LG_CHUNK=4096` that is 160, still well
  short of the 2048 rows where the kernel peaks.
- Attention at 256k still asks for 14.24 GB of GPU K/V in one allocation, is
  declined against a 20 GB budget, and falls back to the CPU. Per-tile K/V upload
  is the fix and is not implemented.

## Cleanup done alongside

Removed, having been measured as not worth their weight:

| removed | why |
|---|---|
| FlashAttention-2 streaming kernel (161 lines) | token-exact but 0.93-1.09x, never faster; opt-in flag nobody should set |
| per-expert batched API `begin/add/end` (45 lines) | orphaned by the grouped kernel |

Net **-206 lines** across the two Metal translation units, with the findings kept
here rather than as dead code behind flags.
