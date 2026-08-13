# colibri-laguna condensed reference

Dense master doc distilled from `docs/*` (API, serve protocol, env vars, CLI
settings, quant formats, KV/cache, routing telemetry, model notes). Exact values
preserved; anything trimmed is noted inline.

## API & serve protocol

### `coli serve` (OpenAI-compatible HTTP, text-only)

- `coli serve --host 127.0.0.1 --port 8000 --model-id glm-5.2-colibri`; requires
  `COLI_MODEL` and optionally `COLI_API_KEY` (bearer, default localhost-only).
- Endpoints: `GET /v1/models`, `GET /v1/models/{model}`, `POST /v1/chat/completions`,
  legacy `POST /v1/completions`, `POST /v1/messages` (Anthropic), `GET /health`.
- Chat/completion: JSON + SSE streaming, usage counts, `max_tokens`/
  `max_completion_tokens`, `temperature`, `top_p`, up to **4** custom `stop` seqs
  (stripped from output, end generation early).
- Extensions: `x_colibri_ignore_leading_stop:true` drops leading stop seqs until
  first non-whitespace content; GLM chat with no client `stop` auto-uses
  `<|user|>`/`<|observation|>` markers (ignoring only leading ones); Inkling/legacy
  get no implicit stops. `enable_thinking:true` or `reasoning_effort` (≠`none`)
  enables GLM-5.2 reasoning.
- Tool-calling supported (OpenAI `tools`/`tool_choice`); image/audio input, logprobs,
  token penalties → explicit error. One generation at a time; concurrent requests
  queue. Gateway is stdlib-only Python; inference in the C engine.
- Admission queue: `--max-queue N` (default 8), `--queue-timeout S` (default 300);
  env `COLI_MAX_QUEUE`/`COLI_QUEUE_TIMEOUT`. Saturated/timeout → OpenAI-shaped HTTP
  429 before streaming headers. `GET /health` shows active/queued/completed/rejected;
  responses carry `x-colibri-queue-wait-ms`.
- Host allowlist: repeat `--allowed-host` or comma-separated `COLI_ALLOWED_HOSTS`
  (DNS-rebinding guard, no wildcard, independent of CORS). Repeat `--cors-origin`
  or `'*'` on trusted local net only; Vite/Tauri local origins allowed by default.

### Anthropic endpoint (`/v1/messages`)

- Same port, nothing to enable. Auth `x-api-key` (Bearer also works).
- Claude Code: `ANTHROPIC_BASE_URL=http://localhost:8000`, `ANTHROPIC_API_KEY=local`
  (only enforced if `COLI_API_KEY` set), `ANTHROPIC_MODEL=glm-5.2-colibri`.
- Full named-event streaming: `message_start` → `content_block_*` →
  `message_delta` → `message_stop` + `ping` keepalives; `stop_reason`, Anthropic
  `usage`, `{"thinking":{"type":"enabled"}}` extended thinking (per-architecture
  reasoning protocol). Native chat template per engine — GLM/Inkling/Kimi K3/DSV4
  prompts not interchangeable.
- Tool use GLM-only; Inkling/K3/DSV4 reject explicitly. Refused explicitly:
  `stop_sequences`, `top_k`, non-text content blocks. Errors use Anthropic
  `{"type":"error","error":{...}}` envelope.
- ⚠ Prefill cost (hardest for Claude Code): CPU-streaming prefill ≈ a few tok/s; a
  15k-token agent preamble ≈ an hour of silent "thinking". Decode ≈ 1 tok/s for large
  models. Trim/disable client preambles.

### Client wiring

- Base URL `http://localhost:8000/v1`, model `glm-5.2-colibri`, any dummy key.
- aider: `OPENAI_API_BASE=http://localhost:8000/v1`, `OPENAI_API_KEY=local`,
  `--model openai/glm-5.2-colibri`.
- crush: provider `{"type":"openai-compat","base_url":"http://localhost:8000/v1/",
  "api_key":"local","models":[{...,"context_window":131072,"default_max_tokens":1024}]}`
  — `context_window` is client-side display only.
- Smoke test: `curl http://localhost:8000/v1/chat/completions -d '{"model":"glm-5.2-colibri","messages":[{"role":"user","content":"hi"}]}'`.

### KV slots & web dashboard

- `coli serve --kv-slots N` (env `COLI_KV_SLOTS=N`), up to **16** independent slots;
  requests pick one via integer `cache_slot` (default 0). Each slot owns history,
  compressed MLA/DSA KV, MTP window, crash-safe `.coli_kv` persistence; common KV
  prefix reused across requests/restarts. Default 4096-token context: each slot costs
  hundreds of MB.
- `coli web` = `coli serve` + browser open (`--no-browser` for headless). Dashboard:
  chat with live tok/s/TTFT/queue metrics, runtime panel (19,456 experts; tier bar
  VRAM/RAM/disk), Brain (76×256 cortex, per-expert tier/heat/topic affinity), Atlas
  (3-D galaxy from `tools/expert_atlas/analyze.py --web` → `experts.json`). `web/` is
  a pure OpenAI-API React/TS client, works against any compatible endpoint.

### serve protocol (engine ⇄ server, mux; defined in `glm.c`/`openai_server.py`)

- Line protocols over stdin/stdout, `fflush` per line. Mux `SERVE_BATCH=1`
  (`run_serve_mux`, ≤16 KV slots, used by `openai_server.py`/`coli web`);
  legacy `run_serve`/`SERVE=1` used by `coli chat`.
- Engine→server: `\x01\x01READY\x01\x01`, then `STAT`, `HWINFO <cores> <ram> <avail>
  <ngpu> <vram> <cpu|gpu>`, `TIERS <vram_experts> <ram_experts> <disk_experts> <vram_gb>
  <ram_gb>`, `EMAP <rows> <cols> <hex>`. Requests: `SUBMIT <id> <slot> <bytes>
  <max_tokens> <temperature> <top_p>\n<payload>\n` (rendered prompt; server owns
  template), `STOP <id>` (persists stats/usage/KV), `CANCEL <id>`. `slot` 0…KV_SLOTS-1.
- Responses: `DATA <id> <n>\n<bytes>`, `TOPK` (`SERVE_TOPK=1`), `HITS` — header for the
  expert map, `REPIN <layer> <eid> <old_tier> <gpu>`, `DONE <id> STAT <emitted> <tok_s>
  <hit_pct> <rss_gb> <prompt_tokens> <length_limited>`. Errors replace stream:
  `ERROR <id> <CODE>`: `BAD_FRAME`/`BAD_REQUEST`/`SLOT_BUSY`/`DUPLICATE_ID`/
  `EMPTY_PROMPT`/`NOT_FOUND`/`CANCELLED`.
- Telemetry before DONE: `PERF <dt> <t_edisk> <t_ewait> <t_emm> <t_attn> <t_kvb>`,
  `ENTROPY` (per-layer bits), `GPUS`, `TIERS`/`EMAP`/`HITS`; `EMAP` byte =
  `(tier<<6)|heat` (2-bit tier, 6-bit log₂ heat); `.coli_usage` persisted each turn end.
- HTTP: SSE emits `data: {"colibri":{stats,perf,topk,entropy,gpus,repin}}` before
  `data: [DONE]`; non-streaming attaches a `"colibri"` field. `GET /experts` → expert
  map; `GET /*` serves `web/dist` (SPA). Legacy: `\x02PROMPT <bytes> <max_tokens>
  <temp> <top_p> [kv_slot]\n<prompt>\n`, `\x02RESET`/`\x02MORE`, raw text +
  `\x01\x01END\x01\x01`.

## Environment variables

**Generated from `dev @ 7fb1159`** by scanning `getenv()` sites in `c/*.c`, `c/*.h`,
`c/*.cu`, `c/*.mm`. Regenerate via README.md (Keeping ENVIRONMENT/SETTINGS honest) after code changes.

Four engine binaries, non-shared knobs: `colibri` (`c/colibri.c`, formerly glm.c) reads
most; `kimi_k3` reads `K3_*`; `inkling` reads `INK_*` + `CTX_MAX`, `PIN_N`, `REP_PEN`,
`GPU_DEV`, `NOGPU`; `olmoe` reads `HOT`, `WIDE`, `SMOOTH`, `CONF_LIMIT`, `MAX_NEW`,
`CHAT`, `EXPERT_DROP`, `WARMUP`; `deepseek_v4` reads `CTX` (default 4096). Shared via
headers: `COLI_USAGE`, `USAGE_SAVE`, `COLI_USAGE_DECAY` (route_trace.h); `RANS_*`
(rans.h); `COLI_NO_OMP_TUNE`/`OMP_NUM_THREADS` (omp_tune.h). CLI usually translates
flags → vars (see SETTINGS).

### Common
| Var | Default | Effect |
|---|---|---|
| `RAM_GB` | 0 (auto ≈ 88% free) | RAM budget (GB) for expert working set |
| `CTX` | 4096 | Max context (tokens) KV is sized for |
| `COLI_PREFILL_CHUNK` | 0 (off) | N-token slice prefill; shrinks S-scaled buffers; byte-identical (N=256); skipped under MTP |
| `NGEN` | 256 (engine; `coli --ngen` 1024) | Max gen tokens |
| `COLI_TEMP` | -1 (auto: 1.0 chat/text, greedy elsewhere) | Temperature; `0`=greedy/deterministic; `TEMP` deprecated alias (numeric only) |
| `NUCLEUS` | 0.90 | top-p mass |
| `TOPK` / `TOPP` | 0 (off) | filters (`TOPP=0`→NUCLEUS) |
| `SEED` | clock+PID | unset = different every run |
| `KVSAVE` | 1 | persist KV to `<model>/.coli_kv` |
| `KV_SLOTS` | 1 | serve KV slots (1–16) |
| `THINK` | 0 | `<think>` reasoning block |
| `MTP` | on | multi-token prediction draft |

### Performance / tuning (subset — full list in source)
- Metal: `COLI_METAL` (off; needs `make METAL=1`), `COLI_METAL_GEMM_MIN`=16,
  `COLI_METAL_SPIN`=off, `COLI_METAL_PREFILL`=0 (CPU bit-exact), `COLI_METAL_RESSET`
  (macOS 15+ residency set), `COLI_METAL_UNTRACKED`=off.
- I/O: `PIPE`=0 (overlap disk load; `PIPE=1` opts in), `PIPE_WORKERS`=8 (cap 64),
  `COLI_PIPE_BLOCK`=0 spin, `PILOT_WORKERS`=1 (clamped [1,16]), `PILOT_EVICT_GUARD`=1,
  `URING`=0 (Linux io_uring; implies `PIPE=1`; incompatible with `COLI_MMAP=1`),
  `DIRECT`=0 (O_DIRECT; drive-dependent, measured +34% decode on NVMe), `COLI_MMAP`=0,
  `PREFETCH`=0, `MLOCK`=-1 (auto-on macOS), `RSS_GUARD_GB`=RAM budget.
- Cache pinning: `PIN` (path or `auto` — seeds from `<model>/.coli_usage`, fallback
  `stats.txt`), `PIN_GB`=10.0, `AUTOPIN`=1 (≥5000 selections), `REPIN`=0,
  `CAP` (unset; precedence `--cap` > `CAP` > platform default), `CAP_RAISE`=1
  (0 on Metal+macOS+fast volume), `COLI_SSD_FAST_GBS`=4.0.
- Routing: `CACHE_ROUTE`=0, `ROUTE_J`=2, `ROUTE_M`=12, `ROUTE_P`=0, `ROUTE_ALPHA`=1,
  `ROUTE_AGREE`=auto, `ROUTE_TRACE`=unset, `COLI_KV_SHARE`=0 (slot TTFT 50.1s→1.7s @6×5090).
- Misc: `COLI_GEMM_CHUNK`=1 (≤2^25-thread chunks), `COLI_RTOP8`=1, `XEXP`=0 (opt-in,
  +11.6% on 48-core Ice Lake), `COLI_GROUP_ASYNC`=0, `COLI_NO_OMP_TUNE`=off
  (kill-switch for OMP hot-thread tuning), `COLI_NUMA`=auto, `ABSORB`=-1, `IDOT`=1,
  `COLI_POLICY`=`quality`|`balanced`|`experimental-fast`, `PROF`=0, `SPEC_PIN`=1,
  `COLI_RAM_OVERCOMMIT`=off, `I4S`=unset, `COLI_DISKCLASS_WINDOW`=see source.

`.coli_ssd` probe cache (Metal+macOS): F_NOCACHE random-read probe cached in
`<model>/.coli_ssd`; format v2 one line `v2 <gbs> <st_dev>`; veto below 64 MB cold
windows; honored only while `st_dev` matches volume; delete is always safe (~0.35 s
re-probe). Dual-SSD: `COLI_MODEL_DIRS`/`COLI_MODEL_MIRROR` (`;`/`,` lists),
`COLI_DISK_WEIGHTS` ratio, `SNAP_MIRROR` (legacy alias), `COLI_MIR_STRIPE`=see source;
`MIRROR:` stats line; pair with `DIRECT=1`.

### CUDA / Vulkan
- CUDA: `COLI_CUDA`=off (or `--gpu none`), `COLI_GPU`/`COLI_GPUS`=unset (`auto`,
  `none`, list `0,1`), `CUDA_DENSE`=0 (measured x2.8 @4×A6000), `CUDA_EXPERT_GB`=0,
  `CUDA_RESERVE_GB`=2.0, `CUDA_RELEASE_HOST`=auto, `COLI_CUDA_ROUTER`=0 (≤4096
  experts/topk 64), `COLI_CUDA_RESID`=0, `COLI_DSA_GATHER`=0, `COLI_CUDA_ATTN`=off,
  `COLI_CUDA_ATTN_PREFIX`/`_SHARD`=off, `COLI_CUDA_PROFILE`=off, `COLI_MTP_GUARD_PCT`=70,
  `COLI_MTP_GUARD_WINDOW`=24, `COLI_CUDA_PIPE`=0 (1/2), `COLI_CUDA_PIPE_SHARD`=off,
  `COLI_CUDA_PIPE_S_MIN`=1 single/8 multi-GPU, `COLI_CUDA_MTP`=0 (opt-in; ~85% hit
  break-even), `COLI_CUDA_ASYNC`=on, `COLI_CUDA_DUAL_PROJ`=on, `COLI_CUDA_W4_PACKED`=on,
  `COLI_CUDA_TC_INT4`=off, `COLI_CUDA_TC_MIN_ROWS`=8, `COLI_CUDA_TC_W4A16`=off (cc≥7),
  `COLI_CUDA_TC_W4A16_MIN`=16, `COLI_CUDA_SHARED_W4A16`=off (min rows 32),
  `CUDA_EXPERT_LOAD_BALANCE`=0. Windows: bare chat/run/serve auto-enables GPU on CUDA
  build + `nvidia-smi`; `--vram N` or add to PATH; `--gpu none` forces CPU.
- Vulkan (`make VK=1`; fails loudly if missing): `COLI_VULKAN`=off, `COLI_VK_DEV`,
  `COLI_VK_SHADERS`=auto (`qmatmul.spv`), `COLI_VK_EXPERTS`=320 (~19 MB VRAM/expert),
  `COLI_VK_DENSE`=0, `COLI_VK_ATTN`=0, `COLI_VK_QPREP`=1, `COLI_VK_RESERVE_GB`=3.0,
  `COLI_VK_SPIN_US`=300. Second device: `COLI_VK_DEV2` (index/`auto`), `COLI_VK_EXPERTS2`=512,
  `COLI_VK_RESERVE2_GB`=0.5. Multi-core → also `COLI_NO_OMP_TUNE=1`.

### Advanced / experimental (subset)
`SPEC`=1, `DRAFT`=-1 (auto: 3 with MTP), `GRAMMAR`/`SCHEMA` (GBNF/JSON-schema; GRAMMAR
wins), `GRAMMAR_DRAFT`=unset, `COLI_DRAFT_CORPUS`=unset, `COLI_CORPUS_K`=8 (max 48),
`COLI_CORPUS_MINACC`=50, `EXPERT_BUDGET`=0 (quarantined; needs
`EXPERT_BUDGET_EXPERIMENTAL`), `DSA`=on, `DSA_FORCE`=0, `DSA_TOPK`=model value,
`I4_ACC512`/`I4_ACC512_TEST`=off, `NOPACK`=off, `CHAT_TEMPLATE`=1, `STATS`/`TOKENS`/
`SCORE`/`REPLAY`/`TF`/`REF`/`REF_FORCE`/`ABLATE_SCORE`/`ABLATE_OUT`/`DEBUG_LOGITS`,
`I3_AVX512`=auto, `COLI_GPU_FAIL_AFTER`, `COLI_SERVE_ALL_STOPS`, `VK_PROF`,
`COLI_LOGIT_DUMP`, `COLI_USAGE`=`<model>/.coli_usage`, `COLI_USAGE_DECAY`=1.0 (∈(0,1]),
`USAGE_SAVE`=1, `RANS_PATH`=auto, `RANS_NEON`/`RANS_AVX512`=on where built,
`OMP_NUM_THREADS`=unset (disables engine tuning).

### K3 / Inkling / OLMoE (engine-specific; subsets)
- K3: `K3_BITS`=4, `K3_MLA_BITS`=8, `K3_HEAD_BITS`=8, `K3_EXPERT_GB`=8.0,
  `K3_LAYERS`=0 (all), `K3_MAXT`=np+ngen / 8192 serve, `K3_CHUNK`=32 (clamp [1,512]),
  `K3_DIRECT`=1, `K3_IDOT`=1, `K3_PIPE`=1, `K3_LOAD_THREADS`=4, `K3_DIRS`,
  `K3_TOPP`=0, `K3_THINK`=1, `K3_VK`=1, `K3_VK_GB`, `K3_VK_UP`=8, `K3_PREFIX_LOG`,
  `K3_CHAT_IDS`, `K3_TRACE`, `K3_LOGITS`, `K3_X0`.
- Inkling: `CTX_MAX`=8192 (reject not truncate), `PIN_N`=cap/2 (clamp cap−8), `REP_PEN`=1.1
  (128-token history), `INK_DENSE_Q4`=auto, `INK_METAL_MIN_S`=1, `INK_PREFIX_LOG`,
  `GPU_DEV`=0, `NOGPU`=unset.
- OLMoE: `CHAT`=unset, `MAX_NEW`=512, `HOT`=0, `WARMUP`=5, `WIDE`=1 (clamp [1,4]),
  `SMOOTH`=0.3 (clamp [0,0.95]), `CONF_LIMIT`=0.92 (clamp [0.1,1.0]), `EXPERT_DROP`=0.

### Server/CLI (Python-side)
`COLI_DEBUG`=0 (1=model output stream, 2=prompt+output transcript),
`COLI_TOOL_SALVAGE`=0 (int4 tool-call de-mangler), `COLI_THINK`=0, `COLI_MODEL`,
`COLI_MODEL_ID`=`glm-5.2-colibri`, `COLI_API_KEY`, `COLI_ALLOWED_HOSTS`,
`COLI_MAX_QUEUE`=8, `COLI_QUEUE_TIMEOUT`=300, `COLI_KV_SLOTS`=1, `COLI_POLICY`=quality,
`COLI_COLOR`=auto(TTY), `COLI_RAW`=0. Set by CLI internally: `SNAP`, `SERVE`,
`SERVE_BATCH`, `PROMPT`/`COLI_PROMPT`, `COLI_OMP_TUNED`.

Fast reproducible Apple-Silicon recipe:
`COLI_METAL=1 DIRECT=1 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=6 MTP=0 ./coli run --model ... --ram 113 "..."`; greedy variant adds `COLI_TEMP=0`.

## CLI settings (SETTINGS.md)

**Updated for contribution based on `upstream/dev @ 21e7a35`** (argparse in
`c/coli`, `c/openai_server.py`).

- `coli` subcommands: `build`, `info`, `plan`, `doctor` (`--json`, `--deep` strict
  preflight: safetensors headers, shard completeness, `COLI_MODEL_MIRROR` admission;
  no payload hashing), `tune` (`--prompt`, `--tokens 16`, `--repeats 2`, `--timeout 900`,
  `--min-gain 0.03`), `run "<prompt>"`, `chat`, `serve`, `bench [tasks]` (`--limit`,
  `--data`), `convert`.
- Common flags → env: `--model`→`SNAP`, `--ram`→`RAM_GB`, `--ctx`→`CTX`, `--cap`→argv
  (0=auto: 8 historically, 1 on Metal+macOS fast volume), `--ngen`=1024→`NGEN`,
  `--temp`→`TEMP`, `--topp`→`TOPP`, `--topk`→`TOPK`, `--repin`→`REPIN`,
  `--policy`→`COLI_POLICY`, `--gpu`→`COLI_GPU(S)`, `--vram`, `--auto-tier`,
  `--no-tune-profile`.
- `serve` flags: `--host` 127.0.0.1, `--port` 8000, `--model-id` ($COLI_MODEL_ID/
  glm-5.2-colibri), `--api-key`, `--cors-origin` (repeat), `--allowed-host` (repeat),
  `--max-queue` (8), `--queue-timeout` (300), `--kv-slots` (1).
- `convert` flags: `--repo` `zai-org/GLM-5.2-FP8`, `--ebits` 4 (streamed experts),
  `--io-bits` 8 (resident), `--xbits` 0, `--no-mtp`.
- `openai_server.py` adds `--engine` `./glm`, `--max-tokens` 1024; rest mirror serve.
- Precedence: flag for knobs with a flag; env var for the rest (`COLI_METAL`, `PIPE`,
  `DIRECT`, `MLOCK`, `CAP_RAISE`, `KVSAVE`, `SEED`, `NUCLEUS`, …). CLI passes whole
  environment through to `glm`.

## Quant / format specs

### FORMATS.md registry (`QT.fmt`; verified at PR-pair restack, base dev `292ed4c`)
| ord | name | weight bytes | scale layout | status |
|---|---|---|---|---|
| 0 | `f32` | O*I×4 | none | stable |
| 1 | `int8-row` | O*I×1 | f32/row | stable |
| 2 | `int4-row` | O*ceil(I/2) | f32/row | stable |
| 3 | `int2-row` | O*ceil(I/4) | f32/row | stable |
| 4 | `int4-grouped` | O*ceil(I/2) | f32/group (gs per-tensor) | stable (#242) |
| 5 | `int3-g64` | O*ceil(I/64)*24 (I3_GROUP=64, I3_GBYTES=24) | f32/64-input group | stable |
| 6 | `e8-iq3-lattice` | O*ceil(I/256)*98 (E8_QK=256, E8_BBYTES=98) | inside super-blocks (ns==4 tag) | stable, upstream #465 merged 2026-07-21 |
| 7 | `mxfp4` | O*ceil(I/2) e2m1 | UE8M0 per gs=32 group (host gate gs≥8, gs%8==0) | stable upstream, Vulkan-only (#676/#705) |
| 8 | `fp8-e4m3-b128` | O*I×1 (same layout as fmt 1) | declared property: f32 per 128×128 block (impl); UE8M0 recognized+refused | this PR pair |
| *(none)* | `int4-rans256-g0` | data-dependent (stamp mandatory) | per-row f32 .qs | merged tools-only |

- `qt_resolve_fmt` (`c/colibri.c`) infers format from byte arithmetic; container never
  carries an ordinal. Next free public ordinal: **9**. ID assignment = first merge into
  dev (no reservation); private ordinals 100+ for in-flight branches.
- fmt=8: f32 scales = `ceil(O/128)*ceil(I/128)` floats (GLM-5.2-FP8 ships this);
  UE8M0 (1 byte/block) is DeepSeek-V4's geometry — recognized, refused by name, invited
  follow-up. LANDMINE disambiguation vs fmt=1 and fmt=6 in `qt_resolve_fmt`
  (`c/colibri.c:1356`); unstamped fmt=1-collision resolves to int8-row (INVERSION).
- Metadata stamp `__metadata__["colibri.fmt"]`: JSON map tensor→format NAME; writer
  `repack_fp8_passthrough.py`; reader TRUST-VERIFY-REFUSE (disagree → exit). Not
  retroactive; conflicting claims refuse; agreeing duplicates tolerated; only
  `.qs`-backed tensors consulted. Ingest cap `ST_FMT_STAMP_MAX`=4096 (exit 1); abort at
  container-discovery time. Sources: `qt_alloc` colibri.c:1105, `qt_bytes` colibri.c:183,
  `FMT_NAMES` colibri.c:1316, `matmul_q` quant.h:105, `matmul_i4` :125, `matmul_i2` :251,
  `matmul_i4_grouped` :168, `matmul_i3` :354, `matmul_fp8` quant.h:491, `FP8_BLOCK`=128.

### oQ format (fmt=101 private; `mlx-community/Laguna-S-2.1-oQ2e-fast`, 7 shards 35 GB)
- MLX `affine` quant, per-tensor bits/group_size from `config.json` (`quantization`/
  `quantization_config`). Default 2-bit/gs128 for routed experts; 386 overrides:
  8/128×141, 3/64×137, 4/64×59, 8/64×34, 6/64×15. Fractional names (`oQ3.5`) = average
  bits, never an actual width.
- Layout per weight: `.weight` U32 `[N, K*bits/32]` (dense bitstream, LSB-first LE),
  `.scales` BF16 `[N, K/gs]`, `.biases` BF16 `[N, K/gs]` (additive offset in weight
  space, not zero-point). Routed experts add leading `[E,...]` axis → per-expert slice
  contiguous (streaming-friendly). Dequant: `w[i]=q[i]*scales[i/gs]+biases[i/gs]`,
  q∈[0,2^bits−1]. Verified exact vs `mlx.core.dequantize` (maxdiff 0.0).
- Name mapping from HF: prefix `model.layers.N`→`language_model.model.layers.N`,
  `lm_head.weight`→`language_model.lm_head.weight`, experts→`switch_mlp.{gate,up,down}_proj`
  `[E,...]`; router/norms/`e_score_correction_bias` stay BF16 unquantized.
- Perf (M5, K=3072 N=12288, single-token): NEON vs bf16: 8-bit 0.95 ms/42.5 MB, 6-bit
  1.48/33.0, 4-bit 1.14/23.6, 3-bit 1.36/18.9, 2-bit 1.02/14.2 (bf16 3.05 ms/75 MB) —
  every width faster AND smaller (2.0–3.3× vs bf16). bf16→f32 `vshll_n_u16` 19.7→102.6
  GB/s. Metal rejected: 0.327 ms/dispatch vs 1.02 ms CPU kernel → ~110 ms/token floor.
- End-to-end `Laguna-XS-2.1-oQ2`: NEON+`-mcpu=native` prefill 4.5→2.3 s, decode
  2.70→5.26 tok/s; RSS 6.8 GB (22B). Bugs caught: `embed_tokens` is oQ-packed too.

### int4-rans256-g0 (no ordinal; PR 1 of 3-PR ladder)
- Lossless rANS entropy codec for per-row int4 routed experts; single static table per
  shard; 256 round-robin streams per tensor. Ratio ~0.76 (≈24% fewer bytes), byte-exact
  reconstruction. Name: int4 codes (15 effective symbols), rANS 256-way, shared table g0.
- Container: ordinary safetensors; weight tensors keep original name, dtype `U8` shape
  `[record_len]`; `.qs` scales unchanged raw F32. Chunk record (all LE): `n_symbols u64`,
  `packed_bytes u64 (=ceil(n/2))`, `stream_offsets[N+1] u32`, zero-pad to 16B, payload
  (N streams), zero-pad. Streams ≥4 bytes each; amplification bound `payload_len*8*M_max`
  (M_max=2^15) vs decompression bombs. Interleave: nibble j→stream j%256, round-robin.
- Codec: ryg_rans, 32-bit state, L=2^23, scale_bits=14, M=16384; encode backwards,
  worst `ceil(n*14/8)+4` bytes; decode forward with `else break` on x<L; final state =L.
- Table in `__metadata__["colibri.int4-rans256-g0.table"]`: `{table_id:"g0", n_streams:256,
  scale_bits:14, M:16384, freq[16], start[16], slot_to_symbol_b64, table_crc32}`.
- Stamp `__metadata__["colibri.fmt"]` → `int4-rans256-g0` is **MANDATORY** (no byte
  arithmetic possible); future decode needs stamp-gated dispatch ahead of inference.
  Scope v1: per-row gs=0 only; g64 refused by name (ratio ~0.89).
- Files: `c/rans.h` (C99 header-only, RANS_NSTREAMS 256, scalar/NEON/AVX-512 arms,
  `RANS_NEON=0`/`RANS_AVX512=0`/`RANS_PATH=`), `make rans` → `tools/librans_c.*`,
  `c/tools/repack_rans.py`, `c/tools/rans_verify.py`, `c/tests/test_rans{,_repack}.py`.
- Integrity: `repack-manifest.json` — per-shard whole-file sha256 + per-record digests;
  refusals `E_SHARD_DIGEST_MISMATCH`, `E_DIGEST_MISMATCH`, `E_DIGEST_MISSING`,
  `E_MANIFEST_MALFORMED`; `--manifest-only` retro-generates.

## KV & cache

### int8 KV cache (`c/kv_i8.h`)
- Symmetric per-row int8: `s=max|x|/127`, one f32 scale per (layer, kv_head, position)
  row — 3% overhead (4B/128 values), symmetric (no zero-point). Dequant once per
  `LG_KC`-row chunk into staging, amortized over `LG_QB` queries.
- **3.88× smaller**: Laguna-XS 6144→0.15 GiB, 262144→5.19 GiB; Laguna-S 262144→6.22 GiB.
  Cost: attention +5% @2k, +11% @6k — taken unconditionally.
- RSS flat 6k→30k (17.5→17.6 GB) — memory no longer scales with context. 256k dirty
  footprint (Laguna-S, topk10 2-bit): KV 6.22 + experts 1.05 + resident 0.64 + scratch
  0.03 = **7.94 GiB** of 20 GiB. Time: attention = 839.8 s of 1110 s at 30k (76%,
  O(S²), 10 full-attn layers) → ~18 h at 256k. Memory win, not a "256k works" result;
  fix needs attention scores on GPU (3.87 TFLOP @6k vs CPU 14 GFLOP/s / MPS 15572).

### KV bind zero-copy (Metal)
- Blocker: f16 GPU KV copy @CTX_MAX=262144 = 14.24 GB > budget → CPU fallback. Fix:
  bind int8 CPU cache in place with `newBufferWithBytesNoCopy` (page-aligned `kv_aligned`,
  16 KB), `deq_kv` shader dequantizes `code*scale` band on demand. GPU KV alloc 14.24→
  **0.02 GB**; 1 copy instead of 2; staging = `KV*band*hd*2*2` bytes. Correctness trap:
  CPU appends chunk K/V **before** the GPU block. Result: 256k @7.24 GB RSS; 1433-token
  prompt prefill 97.8 s / expert-mm 48.7 s / attention 41.6 s / peak 8.2 GB. Honest cost:
  4096 ctx = 74.5 s vs 262144 = ~97.8 s (locality: `[KV][ctxcap][hd]` layout scatters
  bands; fix = head-minor layout, not done). Fixtures token-exact, CPU+Metal.

### CACHE_ROUTE (`CACHE_ROUTE=1`, default OFF)
- Max-rank cache-aware MoE routing (arXiv:2412.00099): keep true top-J, fill rest
  preferring resident (pin∪LRU) experts inside top-M. Routing-side only; complements
  PILOT (prefetch, doesn't change expert IDs). Flags: `ROUTE_J`=2, `ROUTE_M`=12,
  `ROUTE_P`=0 (cumulative mass; 0=fixed M), `ROUTE_ALPHA`=1, `ROUTE_AGREE`=auto.
- Stats: `swap N%` (fraction not in true top-K), `route_swaps`/`route_slots`,
  `route_agree` (|chosen∩top-K|/K), `route_kl`, `hit N%`. Experimental — do not default
  on; A/B via `CACHE_ROUTE=1 PILOT=0|1` combos.

## Routing telemetry (`route_trace.h`)

- One header, four engines (`colibri`, `kimi_k3`, `inkling`, `olmoe`); format lives in
  route_trace.h, deps: C lib + `compat.h` (Windows rename shim). History + trace stream
  from a handful of calls; kimi_k3 needs five.
- History file (`.coli_usage` / `stats.txt` / `PIN=<file>`): text, sparse, one record/line:
  `-1 <n_layers> <n_experts>`, `-2 <format_version> <engine_id>`, then `<layer> <expert>
  <count>`. Negative layer = header record (old readers skip via `l>=0` guard). Rules:
  exactly 3 numeric fields (no strings — `fscanf` <3 silently drops rest), hash stays in
  field 3 (field 2 parsed as `%d`; `glm_moe_dsa`→3815245270 > INT_MAX). Identity mismatch
  refused by name ("pass PIN=<path> to use it anyway"). Empty file = zero-byte; `PIN=auto`
  falls back to `stats.txt`. Legacy: plain triples accepted; `IKU1` (inkling binary:
  `uint32{magic,n_layers,n_experts}` + `uint32[n_layers][n_experts]`) read by inkling only.
- Trace (`ROUTE_TRACE=<path>`): one line per (moe call, batch row): `<call> <row> <layer>
  <expert>:<gate>…` (post-normalization gates). Input to `tools/route_pairs.py` → `.coli_pairs`
  for `COUPLE=` prefetch. Disables device-side router. Measurement only — never changes output.
- API: `rt_init("engine", n_layers, n_experts)`, `rt_drop_row(layer)` for non-routing
  layers (dense, MTP), `rt_route(layer,row,ids,gates,k)`, `rt_count()`/`rt_trace()`,
  `rt_trace_end()`, `rt_save(path,1)`, `rt_load`, `rt_read(path,cb,ud)` / trusted
  `rt_read_ex(...,1)`. Trust = how the path was reached: explicit `PIN=` trusted;
  `PIN=auto`/`AUTOPIN` full checks. Parse-geometry checks never relaxed.

## Model notes

### DeepSeek V4 (target-only Flash engine; `c/deepseek_v4.c` / `.h`)
- DSpark speculation excluded (stacked follow-up). `--no-dspark` is a compat no-op.
  `c/coli` routes run/chat/serve/web to V4; serve keeps engine+caches warm. Targets
  x86-64/aarch64 Linux + Windows/MSYS2. Destroy sessions before engine.
- Migration: st.h reads done; fmt7 MXFP4 via quant.h done; fmt7 rows16 resident cache =
  V4-private `TODO(upstream-fmt7-rows16)`; fmt8 E4M3+UE8M0 128×128 via `st_read_scale_f32`
  + `matmul_fp8` done.
- Memory: 43 layers, hidden 4096, 256 experts/layer topk-6; dense ≈6.27 GiB, BF16 head
  ≈1.06 GiB; experts streamed per RAM budget. `--ram GiB` = planner budget, not OS limit.
- Use: `make deepseek-v4`; `hf download deepseek-ai/DeepSeek-V4-Flash-0731 --local-dir …`;
  run/chat/serve/web with `--model … --ram 32`. Native serving: greedy only, one KV slot;
  tools/grammar rejected; requests re-prefill (process/weights/dense/head/expert cache warm).
- Validate: `make deepseek-v4-tiny-check` (local fixture, covers SUBMIT/DATA/DONE x2);
  `make deepseek-v4-oracle MODEL=… MEMORY_GB=32 ORACLE_TEACHER_FORCING=32 ORACLE_GREEDY=20`.
- (Chinese mirror `deepseek-v4.zh-CN.md` is the same content in 简体中文.)

### Kimi K3 (`c/kimi_k3.c`)
- 2.8T params, 104B active, 93 layers. `make kimi_k3`; text-only (vision shards 95–96
  never read). New architecture: hybrid **KDA** (69 layers) + gated **MLA** (24, every
  4th + final), NoPE anywhere; per-layer full-rank sigmoid gate `σ(W_g x)` before o_proj.
  KDA: q/k/v = SiLU(causal-conv4(Wx)), L2-normed, delta-rule state `S=(I−βkkᵀ)·Diag(e^gk)·
  S+βkvᵀ`, decode state 96×128×128 f32/layer. **AttnRes** replaces residual stream
  (snapshots at layers 0,12,…,84; softmax mix scored by `(v·(res_norm.w⊙res_proj.w))/rms(v)`).
  **Stable LatentMoE**: sigmoid router [896,7168], top-16, latent-space 16 experts (GLU
  inter 3072), 2 fused shared experts (inter 6144), SiTU-GLU `4·tanh(g/4)·σ(g)·25·tanh(u/25)`.
- Weights: routed experts = **MXFP4** (e2m1, ue8m0 scale per 32 cols, `w=v·2^(scale-127)`),
  17.55 MB/expert × 82,432 ≈ 1.45 TB; `matmul_mxfp4` (scalar+AVX2) computes on layout
  directly (never re-encoded — QAT). Rest BF16 → load-time int8/int4-g64.
- Streaming: per-layer LRU (`K3_EXPERT_GB`); experts back-to-back (one pread), loads in
  disk-offset order, parallel + O_DIRECT (1.8→6.3 GB/s, decode 21→9.4 s/token);
  `K3_PIPE` overlap; `K3_IDOT` int8-activation dots; Quantile-Balanced flat routing ⇒
  structurally lower LRU hits, bandwidth-bound.
- Repack `tools/k3_repack.py`: source 1.56 TB → ≈1.50 TB (8-bit) / 1.48 TB (4-bit);
  single streaming read, byte-identical experts, resume per shard, atomic index rebuild,
  `--verify-full`. Container auto-detected (U8+.qs); `K3_BITS=4` downcasts (35 vs 57 GiB
  resident); startup 30.4 s→0.6 s (2 layers). Sub-4-bit expert re-encode not offered
  (double quant of QAT).
- Tokenizer: `tools/k3_tokenizer.py` writes tokenizer.json from raw tiktoken vocab;
  kimi pre-tokenizer family (sniffed via `\p{Han}`) + rank-BPE when `model.merges` empty
  (tiktoken-exact by construction); matches tiktoken on all 18 corpus cases.
- Validation: `tools/k3_ref.py` numpy oracle — C engine matches to rel-L2 ≤ 2.2e-6 across
  all 4 layer types. Chat XTML format (4 special tokens `<|open|> <|close|> <|sep|>
  <|end_of_msg|>`); Moonshot-identical numbers on 77 conversations. Reasoning as
  `reasoning_content`, `<|end_of_msg|>` = model-owned stop.
- Vulkan tier (`make VK=1 kimi_k3`): fmt7 MXFP4 decode (ue8m0→f32 per-32-group at upload;
  kernel rel_l2 2.2e-07), shared experts uploaded once (7.5 GB/92 layers @int4) + fill-once
  routed tier (`K3_VK_GB`/`K3_VK_UP`). Limits: no speculative decode, no tools/images,
  CPU+Vulkan only.

### corpus-draft (`COLI_DRAFT_CORPUS=`)
- Retrieval-based speculative draft: proposes continuation following longest suffix
  (lengths 8→3, most recent first) of live context in a frozen file of token ids;
  verified in the same batch-union forward as MTP/grammar (proposal-only, lossless).
- Measured (GLM-5.2 int4, 96-token greedy): H200 resident 3.69→4.52 tok/s (+22%, 90%
  acc); CPU 0.82→1.00 tok/s (+22%, 100% acc). 3–4× fewer forwards ≈ only +22% wall clock
  (MoE: expert work scales with rows). Best case = replayed/benchmark text; novel text
  ≈ no gain. Deep drafts at S≥8 can hit #689 (CUDA near-tie divergence).
- `COLI_CORPUS_K`=8 (cap 48), `COLI_CORPUS_MINACC`=50 (pause 256 tok, re-arm), file = ws-
  separated ids, `-1` = span separator. Off by default; takes priority over MTP. CPU path
  byte-exact; CUDA may diverge by a near-tie token.

### grammar-draft (`GRAMMAR=`; method F)
- Byte-level GBNF subset → set-of-stacks PDA; exactly-one legal next byte ⇒ forced span
  pre-accepted as draft into the batch-union verify forward. Never constrains sampling
  (verified by target model); composes with DRAFT/MTP. Win denominated in expert I/O
  (disk reads avoided), not FLOPs. Adaptive guard disables below 50% acceptance.
- Usage: `GRAMMAR=<file.gbnf>` (root rule must be `root`; literals/escapes `\" \\ \n \r
  \t \xHH`, classes, postfix `? * +`, `|`, `#`), `GRAMMAR_DRAFT`=24 (max 48). Lazy arm +
  desync-tolerant. Measured: conforming NDJSON 1.60 tok/fwd (large win); sloppy NDJSON
  1.21–1.22 @87% (~+5%); prose ~1.0. Levers: compact output; whitespace-tolerant grammars.
- Lossless by construction (greedy byte-identical; sampling distribution preserved).
- Server: `response_format` → per-request grammar over SUBMIT (optional 7th field
  `gbytes`; 6-field old headers still valid). `json_object`/`json_schema` (via
  schema_gbnf.h)/`gbnf` extension. Greedy only (temp 0); ~8 µs/request typical, ~18 µs at
  32-level cap; payloads capped 1 MiB. Drafted greedy may differ in near-tie tokens.
