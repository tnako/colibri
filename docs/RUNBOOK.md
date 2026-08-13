# colibri-laguna runbook (condensed)

Condensed from docs/quickstart, windows, cuda, BUILD-cuda-glibc241, metal, vulkan, tuning, benchmarks, benchmark reports, METAL-* reports, laguna-xs-gpu-experts, one-knob-and-chunking. Numbers are measured unless marked "estimated". Engine runs a 744B MoE model (GLM-5.2 int4) by streaming experts from disk; single C program, Python only for model prep.

## Quickstart

### Prereqs
- RAM: 16 GB min / 24 GB+ recommended · disk: ~380 GB free for int4 (~372 GB) · fast NVMe SSD = token speed. No GPU needed (CPU-only default).

### Install
- **Shortcut:** prebuilt archives on Releases page (Linux/macOS/Windows). `mkdir colibri && tar xzf colibri-v1.1.0-linux-x86_64.tar.gz -C colibri && cd colibri && python3 coli info`. ARM64 Linux: no prebuilt → build from source.
- **From source:** `git clone https://github.com/JustVugg/colibri.git && cd colibri/c && ./setup.sh`. Expect self-test `~30-32/32` (near-ties are toolchain-dependent).
- Linux: `sudo apt install -y build-essential git python3`; runtime-only host: `sudo apt install -y libgomp1`. macOS: `xcode-select --install && brew install libomp git python`. Windows: prebuilt zip or MSYS2 UCRT64 `pacman -S --needed mingw-w64-ucrt-x86_64-gcc make git python`.
- Tools: C compiler + `make` + `git` + `python3`.

### Get model
- Recommended: download **group-scaled gs64 int8-MTP** container `https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp` (~372 GB). Per-row int4 containers are ~9pp worse; plain int4 heads disable MTP (#455, #8).
- Or convert: `./coli convert --model /nvme/glm52_i4` (resumable, never needs full 756 GB).

### Run
```bash
COLI_MODEL=/nvme/glm52_i4 ./coli chat
COLI_MODEL=/nvme/glm52_i4 ./coli doctor   # what's missing
COLI_MODEL=/nvme/glm52_i4 ./coli plan     # RAM/disk/GPU placement
COLI_MODEL=/nvme/glm52_i4 ./coli chat --topp 0.85   # 30-40% less disk, same quality
```
- First launch loads ~10 GB resident weights. Slow disk ⇒ <1 tok/s expected; placement never changes answers.

## Build backends

### Windows (native, no WSL)
- Core Ultra 9 285K / RTX 5080 / 128 GB / Win11 24H2 walkthrough (#306); also validated on Thinkpad P16v (RTX 2000 Ada sm_89).
- CPU build: `make colibri.exe ARCH=native` (AVX-VNNI on Alder Lake+; banner prints `idot: avx-vnni`) + `make iobench.exe`.
- CUDA DLL (needs MSVC + CUDA Toolkit ≥12.8): from "x64 Native Tools Command Prompt" with `set PATH=%PATH%;C:\msys64\usr\bin` (no sh.exe → `'printf' is not recognized`, #478):
```cmd
make cuda-dll CUDA_ARCH=sm_120        # Blackwell sm_120, Hopper sm_90, Ada sm_89, Ampere sm_80/86, Turing sm_75, Volta sm_70, Pascal sm_60, Maxwell sm_50
make colibri.exe CUDA_DLL=1 ARCH=native
```
- Run: `$env:COLI_CUDA="1"; $env:COLI_GPU="0"; $env:CUDA_DENSE="1"; $env:CUDA_EXPERT_GB="4"`. Size `CUDA_EXPERT_GB` so dense (~10 GB)+experts < VRAM. MTP off by default under CUDA (#293); `COLI_CUDA_MTP=1` opts in.
- Measured (285K/5080): 0.26 cold → 0.30 warm CPU → **0.42 tok/s** GPU. RTX 5070 Ti + Ultra 9 32GB: CPU 0.63 → 0.72 → **1.07 tok/s** decode (#273/#274).
- Smart App Control blocks self-built binaries: HKLM `...\CI\Policy\VerifiedAndReputablePolicyState` =0 then **reboot** (one-way). Windows Store python alias = common trap.
- AMD HIP: `make -C c colibri.exe HIP_DLL=1` (no SDK needed) then `make -C c hip-dll HIP_DLL=1 HIP_SDK_ROOT=<root> HIP_ARCH=gfxNNNN`; run needs `coli_hip.dll` next to exe + `$env:COLI_HIP_RUNTIME_DIR="C:\path\to\hip\bin"`. Validated: Radeon 8060S gfx1151, MSVC 14.44.35207, WinSDK 10.0.26100.0. Pin `VCToolsVersion=14.44.35207` if VS2026 present. Verify residency via `[CUDA] resident set: N tensors` (N>0), not the device line. Runtime loading of `coli_hip.dll` not implemented yet — GPU won't engage.

### CUDA (Linux)
```bash
make CUDA=1                          # CUDA_HOME override; CUDA_ARCH=native
COLI_CUDA=1 COLI_GPU=0 CUDA_DENSE=1 SNAP=/nvme/glm52_i4 ./colibri 64 4 4
```
- VRAM expert tier: collect `STATS=stats.txt` run, then `COLI_CUDA=1 COLI_GPU=0 CUDA_EXPERT_GB=16 PIN=stats.txt PIN_GB=160`. Multi-GPU: `COLI_GPUS=0,1,2,3,4,5`; `CUDA_EXPERT_GB` is total budget; `auto` fills free VRAM; `PIN_GB=all` + `RAM_GB=auto` = full residency.
- **6× RTX 5090, 251 GB host:** `CUDA_EXPERT_GB=auto PIN_GB=all COLI_CUDA_PIPE=2 COLI_CUDA_TC_W4A16=1` → 176.7 GB VRAM + 191.3 GB RAM, all 19,456 experts resident, **5.8–6.8 tok/s** (TTFT ~13 s; 1571-token prefill ~122 s → 4.2 tok/s).
- `COLI_CUDA_PIPE=2` keeps residual stream on-device (+49% single-GPU at S=1). `COLI_CUDA_TC_W4A16=1` Tensor-Core int4×fp16 (pays ≥16 rows). Experimental DietGPU ANS tier: `make CUDA=1 COLI_ANS=1 DIETGPU_ROOT=/opt/dietgpu`; sidecar tied to exact placement order; +13.9% VRAM experts, up to +15% tok/s.
- Same 150 GB tier: 0.94–1.64 tok/s hot-first vs 0.29 filled without heat — profile quality > VRAM. AVX-512 CPU can match a 5090 on expert matmul (#101). 313M bench fixture: `python tools/make_glm_bench_model.py --device cuda` + `python tools/benchmark_cuda_fixture.py`.

### CUDA on glibc ≥2.41 (Debian trixie)
- `cospi`/`sinpi` clash → CUDA ≤12.9 headers fail vs glibc 2.41. Fix = **CUDA 13.x**. Conda route (no root, no distro driver):
```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /opt bin/micromamba
/opt/bin/micromamba create -y -p /opt/cuda13 -c nvidia -c conda-forge cuda-nvcc=13 cuda-cudart-dev=13
ln -s /opt/cuda13/lib /opt/cuda13/lib64
make glm CUDA=1 CUDA_ARCH=sm_86 CUDA_HOME=/opt/cuda13
```
- Ship `libcudart.so.13` with binary (LD_LIBRARY_PATH) or `/usr/local/lib`. Avoid: Debian `nvidia-cuda-toolkit` (driver clash), pip `nvidia-cuda-nvcc-cu12` wheel (ptxas only).

### Metal (Apple Silicon, experimental)
```bash
cd c && make colibri METAL=1 && make metal-test
COLI_METAL=1 COLI_MODEL=/path/glm52_i4 ./coli chat --ram 96
```
- Runs routed-expert SwiGLU, fused decode attention (S≤4), prefill GEMMs on GPU. Decode token-exact; prefill near-tie top-token can differ (`COLI_METAL_GEMM_MIN=100000` for bit-exact prefill; default GEMM_MIN=16). `COLI_METAL_PREFILL=1` (off by default) cuts 544-token prefill attention 35.9→9.0 s.
- Measured: M4 Max CPU 0.30 → **0.42 tok/s**; M5 Max (46.9 GB learned pin) **2.06 tok/s** (#103). Requires macOS 15 SDK for `COLI_METAL_RESSET` path (else compiled out).

### Vulkan (any GPU, Vulkan 1.2 ICD + GL_KHR_shader_subgroup_arithmetic)
```bash
cd c && make glm VK=1                 # needs libvulkan + glslc
COLI_VULKAN=1 COLI_VK_DENSE=1 COLI_VK_ATTN=1 PIN=<model>/.coli_usage PIN_GB=0 COLI_NO_OMP_TUNE=1 \
  ./coli run "Hello" --topp 0.7
```
- `COLI_VK_EXPERTS=N` (default 320) hot set in VRAM; `COLI_VK_DENSE=1` projections; `COLI_VK_ATTN=1` MLA core fused with o-projection. `PIN_GB=0` but keep `PIN` set (RAM pin → LRU cache).
- Set `COLI_NO_OMP_TUNE=1` (spin starves async I/O: 28→5 GB/s). **Discrete cards need Resizable BAR/SAM** (else 0.11 vs 0.24 tok/s). Shaders via `COLI_VK_SHADERS` or `shaders/` next to binary.
- Measured RX 9070 (RADV): expert 0.11–0.13 ms vs HIP 0.179 ms (~35% faster); decode **1.7–1.8 tok/s** (64-tok) / 1.58 sustained vs HIP 1.5–1.55. Buffers CPU reads must be HOST_CACHED; rest HOST_VISIBLE|DEVICE_LOCAL.
- Benchmarking vs CUDA/HIP: pin `DRAFT=0` (CUDA/HIP default 0, CPU/VK keep 3) and GPU clocks `power_dpm_force_performance_level=high` on both arms.

## Tuning knobs
- `--temp T` (default 0.7 + nucleus 0.90; 0=greedy) · `--topp` (adaptive expert top-p; lossy) · `--ngen N` · `--repin N` (live hot-expert adapt, 25% hysteresis, 4-swap limit) · `RAM_GB=<n>` / `--ram` · `PIN=stats PIN_GB=g` · `DRAFT=n` (MTP depth) · `GRAMMAR=g.gbnf` · `THINK=1` · `PILOT=1` (router-lookahead prefetch; +11pp hit with `PILOT_REAL=1`, +3% recall `PILOT_TWO=1`) · `URING=1` (Linux) · `PIPE=0` disables async load pool (default ON, −18% disk service) · `DIRECT=1` O_DIRECT (+65% alone on Strix Halo) · `COLI_NUMA=1` (+13% 2-socket / +40% 4-socket CPU-only; up to 10× regression if blanket on GPU hosts) · `CACHE_ROUTE=1` · `AUTOPIN=0` · `CAP_RAISE=0` · `KVSAVE=0` · `TF=1`.
- `coli plan` (--policy quality|balanced; `--policy experimental-fast --topk 4` research-only) · `coli tune` then `coli run --auto-tier` (saves only if ≥3% throughput gain, hit rate within 0.5pp, p99 within 20%; `--no-tune-profile` bypasses). Profiles in `~/.config/colibri/tuning`. Auto-tier caps OpenMP to physical cores; `OMP_NUM_THREADS`/`COLI_NO_OMP_TUNE` override. `COLI_OMP_TUNED=1` skips the re-exec.
- Learning cache: `.coli_usage` next to model auto-pins hottest experts; auto-sizes to `--ram` (2026-07-10; rerun old benchmarks). KV-cache persisted per-turn (`.coli_kv`, ~182 KB/token, crash-safe) → chats reopen warm, zero re-prefill; `:reset` clears.
- Byte-exact reproducibility: `DRAFT=0` (+`IDOT=0 COLI_CUDA=0` for GPU independence). `SPEC_PIN=1` (default) pins draft/verify to S=1 kernels.

## Benchmarking
- Build + measure in order: `./setup.sh`; `gcc -O2 -fopenmp iobench.c -o iobench`; `./iobench <shard> 19 64 8 0` (buffered) / `... 8 1` (**O_DIRECT** = true number; buffered reports page cache, #86; macOS uses F_NOCACHE, reboot for real cold read); `./coli chat`; `STATS=stats.txt` then `PIN=stats.txt PIN_GB=20`; `./coli bench` (hellaswag/arc/mmlu, 40 each; `--limit 200`, `--ram 100`).
- Reference (dev box WSL2, 12c, 25 GB): model ~370 GB disk, 9.9 GB resident RAM, ~30 s load, ~20 GB RSS, cold ~11 GB reads/token (75 layers × 8 experts), ~1 GB/s disk → 0.05–0.1 tok/s cold; MTP 2.2–2.8 tok/forward.
- Estimates: PCIe4 NVMe 3–5 GB/s + 32 GB → ~0.5–1 tok/s; PCIe5/RAID0 8–12 GB/s + 64 GB (PIN ~40 GB) → ~2–4; 128–256 GB + 12c → ~2–4 (matmul-bound); +24–32c or AVX-512 → ~5–15.
- **Community (measured, greedy, --ngen 32, MTP active unless noted):** Core Ultra 7 270K WSL2 24GB: 0.07 → 0.11 (`--topp 0.7`) · M5 Max 128GB: 1.06 CPU → **1.83 Metal** (--ram 96) → **2.06** (46.9 GB pin) · M1 Ultra fmt=2 Metal: **1.50** (--ram 125 --cap 33) · Mac Mini M4 Pro 48GB Metal: 0.30 (vs 0.18 CPU) · Epyc 9654 ES: 0.31 · Ryzen AI 9 HX 370: 0.37 (2.59 tok/fw) · Ryzen 9 9950X: 0.10 QLC → 0.28 PCIe5 (profile flips 66% disk → 57% matmul) · Ryzen AI Max+ 395: 0.16 fresh → 0.40 learned pin → 1.10 sustained (DIRECT+PIPE) → 1.83 (dev, +PILOT) · 9800X3D + 5090: 0.41 (CUDA tier ≈0%, AVX-512 matches) · EPYC 7443 430GB: **1.00** (98% hit, RAM-bound) · i5-12600K native Win: 0.08 · 9950X3D2 PCIe5 + 5090: **1.23** (`MTP=0 DIRECT=1 PIPE_WORKERS=16 PREFETCH=1`) · 185H + 5070 Ti: 0.03→0.5→**1.07** · DGX Spark GB10: 0.50 warm → 2.4 (full-k8) → **3.33** (CACHE_ROUTE) · i5-13600K + 5070 Ti: 0.56→**0.98** (PIN=auto DIRECT=1 PIPE=1) · **6×5090: 5.8–6.8 tok/s**.
- Takeaways: on small-RAM machines the RAM cap (not disk) binds; `--topp 0.7` ≈ 1.6×; ×5.8 disk BW → ×2.9 tok/s; when experts stream, the drive sets the rate. NUMA interleave: 42.42→58.26/65.89 GB/s CPU-expert BW, decode 7.66→9.02/9.17 tok/s (2-socket Xeon Silver 4510).

### Quality benchmark
- Quality: int4 gs64 container **62.5% mean acc_norm** (hellaswag/arc/mmlu 0-shot, n=40); quantization cost −8.2pp vs fp16 (per-row scales; grouped recovers ~63%).

## Metal perf reports
### M5 Max (18c / 40c GPU / 128 GB, 1024-tok run, GLM-5.2 int4)
- Flags held: `COLI_METAL=1 DIRECT=1 MTP=0 --ram 110` (~607 experts/token, ~74-75% hit, RSS ~97.9 GB).
- Configs (tok/s): A old-base 2.06 · B rebased defaults **1.25** (regression) · C +`PIPE=1` 1.30 · D +`COLI_NO_OMP_TUNE=1` 1.90 · **E NO_OMP+PIPE (winner) 2.24**.
- Cause: OMP hot-team active-spin steals shared SoC power → GPU throttles (attention kernel 76→223 s); `NO_OMP` restores clocks, `PIPE=1 PIPE_WORKERS=8` hides the CPU→GPU dispatch latency. **Use both.**
- Winner: `COLI_METAL=1 DIRECT=1 MTP=0 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8 ./coli run --model ... --ram 110`.
- `--ram 110` safe; `--ram 120` crosses into memory compression. `DIRECT=1` required (~2× slower without, 2.16→1.15). Not run-to-run bit-reproducible (parallel reductions flip argmax ~every 7 tokens); throughput stable.

### M1 Ultra (20c / 48c GPU / 128 GB, fmt=2 per-row int4, frozen 46.9 GB pin)
- Disk 6.89 GB/s F_NOCACHE (93% ceiling during decode). Best **1.50 tok/s** vs M5 Max 2.24 (−33% at +20% GPU cores) — **the SSD, not the GPU, sets speed** (disk wait 55-60% of decode wall, budget fully serial; zero-miss asymptote ~3.5 tok/s).
- Winner: `COLI_METAL=1 DIRECT=1 MTP=0 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8 ./coli run --model ... --ram 125 --cap 33`.
- `PIPE` is the only lever that matters (+6.9%); NO_OMP neutral here; PIPE_WORKERS=8 sweet spot (16 → 1.37). OMP spin trap does NOT reproduce on M1 Ultra. v1.2.0→v1.4.0 rebase neutral; watch v1.4.0 fast-SSD default drops cap to 1 (0.98 tok/s) — check `--cap` first. **MTP is a strict loss at 128 GB** (swap thrash). fmt=2 costs ~9pp quality (Metal fmt=4 dispatch open: #585/#587).
- Spend on storage, not GPU, for this class.

## Model-specific (Laguna)
### GPU expert work helps Laguna-XS (Metal, default 20 GB budget, CTX_MAX=16384)
- Commit A/B `e899eec`→`3305bbc`: prefill 2k 66.9→34.5 s (1.94×) / 6k 205.4→71.5 s (2.87×) / 12k 415.2→**112.4 s** (3.69×); expert-mm 4.8× at 12k; peak RSS 22.7→**7.2 GB** (3.2×) and stops growing with context. XS 2-bit bank (7.9 GB) now mmap'd zero-copy page-cache (evictable, leaves budget). Laguna-S same mechanism at 28.4 GB.
- Taskfile: `task laguna:xs Q=oQ2 NGEN=6 SYNC=0` (prefill 2.2 s, 6.5 GB) · `task laguna:s Q=oQ2e-fast NGEN=6 SYNC=0` (8.8 s, 5.3 GB) — both coherent, reuse local `models/` by basename. Knobs: `LAGUNA_MEM_GB`/`MEM_GB` (default 20) · `CTX_MAX`/`CTX` (default 8192) · `LAGUNA_GPU_PROF` · `Q=` · `METAL=0` forces CPU. `task test` = 5 checks.

### One knob + chunked prefill (Laguna, 256k target ≤20 GB)
- **Honest answer: Laguna-S at 256k does not fit.** Weights alone 27.64 GiB (oQ2; experts 27.00 GiB), KV f16 12.07 / int8 6.04 GiB. Streaming works: dirty total 7.73 GiB (KV int8 6.04 + expert WS 1.05 + projections 0.64 + scratch ~0.01); **int8 KV not done (f32 = 24 GiB at 256k) = remaining blocker**. XS at 256k fits (7.68 GiB weights + 10.06 GiB f16 KV).
- **`LAGUNA_MEM_GB` (default 20) = the one knob** (replaced 4 vars). Priority: GPU attn projections (4.6× on attn, ≤1/6 budget) → resident Q8R bank → GPU f16 expert bank → streaming cache. Measured XS 2k-tok: 8→**11.1 GB** (128.5 s) · 14→11.3 GB (92.7 s) · 20→15.5 GB (**31.4 s**). Bug fixed: cache sized from free RAM ignored budget.
- **Chunked prefill:** `step()` splits into `LG_CHUNK` (default **256**) → all scratch O(chunk), peak mem context-independent (18.5 GB@2k vs 17.6 GB@6k), bit-identical (208-position S fixture passes). At LG_CHUNK=256 expert mm regressed 15.1→71.2 s (40 rows/expert) until GPU dispatch fixed it → prefill 89.6→**31.4 s**.
- **Batched GPU expert dispatch:** all expert GEMMs from one thread (removes MPS race, "Number of requested rows in result exceeds result matrix size"); falls back to CPU UDOT, zeroes output first.
- Result XS ~2k tok: prefill 89.2→31.4 s, RSS 18.9→15.5 GB (11.1 at MEM=8), scratch O(chunk). Correctness: XS 24/24 + 12/12, S 208/208 + 8/8, task test 3/3.
- Open: int8 KV; full-size S never run (arithmetic from config + measured per-phase rates; S ~235 GB bf16 / 27 GiB oQ2).

## Laguna 256k rework phases (measured 2026-08-10, 10 p-cores, 32 GiB)
- Harness: `c/tools/stress_laguna.sh <model> <prompt-tok> <gen> <tag>`; `LG_SPEC=0` for plain greedy decode. Targets: prefill 256k <2 s (unreachable full-attn), decode 140 tok/s, mem <20 GB.
- **Laguna-XS-2.1-oQ2** (10 full/30 sliding, D=2048, kv=8, hd=128, topk=8): CPU 1,869-tok: prefill 184.6 s / 15.1 GB / 5.88 tok/s; Metal 1,869: 28.7 s / 6.1 GB / **6.10 tok/s**; Metal 30k: prefill 436.4 s (attn 231.4 s) / 9.3 GB. Decode = CPU per-token path (6.1-6.6 tok/s) regardless of build.
- **Laguna-S-2.1-oQ2e-fast** (12/36, D=3072, topk=10): Metal 800-tok: 60.7 s / 9.5 GB / **0.86 tok/s** (decode ~1.16 s/token, 7× slower than XS; 3× params).
- Prefill scratch unbudgeted ~10.7 GB at last 256k chunk (score tile 8.6 GB + f32 band 2.1 GB); decode never touches GPU (S<64 gate). KV: Metal build 7.97 GB vs CPU 6.68 GB @ 262,144 (Laguna-S).

### Phase 7 selective prefill (`LG_SELP`, Laguna-XS, Metal, `LG_SPEC=0`)
- 16k: baseline attn 141.3 s / wall 273.4 s / 2.24 tok/s → `LG_SELP=1 cap=4096`: 116.5 / 227.8 / 2.35 → `cap=8192`: ~136 / ~254 / **2.53**.
- 32k: baseline 370.4 / 649.9 / 1.79 → **cap=8192: 297.1 (−20%) / 568.7 / 2.26**. Remaining cost = full O(S²) scoring pass + expert-MM (153-165 s). Quality: first 32 tokens, 85.2% token agreement vs baseline. Single-run variance ±15%.
