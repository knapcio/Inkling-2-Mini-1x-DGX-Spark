# Running it

Requirements, the API, the fallback profiles, every flag of `scripts/launch.sh`, the memory budget and longer contexts. The commands to download, start and stop are in the README's [Quick start](../README.md#quick-start); the commands that reproduce the published numbers are in its [Reproduce](../README.md#reproduce) section.

- [Requirements and first start](#requirements-and-first-start)
- [Using it](#using-it)
- [Fallback: the DSpark profile](#fallback-the-dspark-profile)
- [Every flag explained](#every-flag-explained)
- [Memory budget](#memory-budget)
- [What differs from the published recipes, and why](#what-differs-from-the-published-recipes-and-why)
- [Longer context: 512K and 1M](#longer-context-512k-and-1m)
- [Reproducing the numbers](#reproducing-the-numbers)

## Requirements and first start

Requirements: one DGX Spark with DGX OS, Docker with the NVIDIA runtime (the default on DGX OS), ~80 GB of disk for
the image (56 GB), the weights (20 GB) and the drafter (~1 GB), and nothing else large running on the box.
The download, start and stop commands are in the README's [Quick start](../README.md#quick-start).

`scripts/download.sh` fetches the drafter only for the DFlash2 profile: `NO_DFLASH2=1`, `SPEC=dspark` or
`ENGINE=vllm` skip it, and `scripts/download.sh --dflash2` fetches only the drafter. A file that fails its checksum is
removed (run the script again to fetch it), and `.revision` is written only after every file has been verified. The
drafter directory may hold only the files the profile reads (`config.json`, `model.safetensors`,
`sglang_inkling_dflash2.py`, `README.md`, `.revision`); `scripts/launch.sh` refuses to start on any other file, a
different revision or a sha256 mismatch, checked on the host and again inside the container.

The Chat Completions API is at `http://127.0.0.1:8000/v1`, model name `inkling-2-mini`. The server listens on
loopback only; from another machine use `ssh -L 8000:127.0.0.1:8000 <spark>`, or set `HOST=0.0.0.0` on a trusted LAN.
Do not forward the port through your router.

The first start builds the overlay CUDA extensions (pico12, wide, certified head, prefill attention; ~30 s each) and
compiles the overlay Triton kernels; all are cached in `~/.cache/inkling-2-mini-sglang` like SGLang's own JIT caches.

Do not start the server while a large download or copy is still running: on GB10 the page cache and the GPU share
the same memory (the "page-cache trap", see [CREDITS.md](../CREDITS.md)). `scripts/launch.sh up` refuses to start next to another model
server or with less than `MIN_AVAIL_GIB` + 8 GiB MemAvailable (for example while a download fills the page cache),
and runs `scripts/memguard.sh` for the first 20 minutes. The guard stops the container if MemAvailable falls
below 3.5 GiB for two consecutive samples, 2 s apart. An out-of-memory on GB10 can hang the box until a power cycle.

## Using it

Reasoning effort goes into the chat template, either as the top-level `reasoning_effort` field or as
`chat_template_kwargs: {"reasoning_effort": ...}`; both reach the template. Values: `none` (0.0), `minimal` (0.1),
`low` (0.2), `medium` (0.7), `high` (0.9, the default), `xhigh` / `max` (0.99), or a number in [0.0, 0.99]. The model
always writes a reasoning block, even at `none`; the reasoning parser returns it as `reasoning_content`, the answer as
`content`. Tools use `tools` / `tool_calls`. Structured output uses `response_format` with a JSON schema (xgrammar).

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini",
  "messages": [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
  "reasoning_effort": "low", "max_tokens": 1024}'
```

The checkpoint's `generation_config.json` sets temperature 1.0; pass `temperature` per request if you want
something else. Reasoning tokens count toward `max_tokens`: at effort high, leave several thousand.

## Fallback: the DSpark profile

`SPEC=dspark scripts/launch.sh up` serves with the model's own DSpark draft (`mtp.safetensors`) and its overlays:
the v2.7.2 profile (verify-width policy `r2-theta004.json` up to 16 requests, draft byte cuts, certified heads for
<= 16 and 17-192 rows, block verification, fused sampled accept, dead-row expert skip, packed KV store at
`MEM_FRACTION=0.89`), plus the exact changes made since: `RSFIX=1`, the compact hot-vocabulary base logits
(`hotbase`) and the Markov-walk hoist up to 32 rows (`RNNHOIST_MAX_BS=32`; v2.7.2: 16). It does not need the drafter
download. `SPEC=dspark` starts the v2.7.2 profile (with the v2.8 exactness fixes); speedbench quick 63.7 tok/s on
box A, 0 failed prompts, median TTFT 157 ms (`results/V28dsp-s05/`; the default profile 70.61 and `DFCAP=off` 62.49
in the same mode). Its v2.7.2 two-box medians: speedbench full c1 63.8, prose decode c4 / c8 / c16 113.3 / 164.6 /
244.1, thinking c1 / c8 / c24 72.2 / 233.6 / 437.1, decode at 32K / 128K 68.4 / 45.5 tok/s, 2.13-2.17M global-layer KV
tokens ([CHANGELOG, v2.7.2](../CHANGELOG.md#v272-2026-11-24)).

`ENGINE=vllm scripts/launch.sh up` runs vLLM with the DSpark draft (the alternative profile, no overlays);
`SPEC=dflash2` is refused there.

## Every flag explained

`scripts/launch.sh up` runs one container (the command it prints, with the defaults):

```bash
docker run -d --name inkling-2-mini --gpus all --ipc=host --network host --shm-size 16g --restart no \
  -v ~/models/Inkling-2-Mini:/model:ro -v ~/.cache/inkling-2-mini-sglang:/root/.cache \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v $PWD/overlay:/opt/inkling-overlay:ro -e PYTHONPATH=/opt/inkling-overlay \
  -e INKLING_VERIFY_CAP=dfpolicy:/opt/inkling-overlay/inkling_vcap/policies/df-SLDF2x.json,bsmax=8 \
  -e INKLING_VERIFY_CAP_LOG_EVERY=0 -e INKLING_VERIFY_CAP_STEPLOG= -e INKLING_VERIFY_CAP_REMAP=1 \
  -e INKLING_DRAFT_BYTES= \
  -e INKLING_DRAFT_HOTVOCAB_FILE=/opt/inkling-overlay/inkling_draftbytes/hotvocab/inkling2-mini-hot-r2.json \
  -e INKLING_KERNELS=moe_tiles,qkvr_gemv \
  -e INKLING_MOE_TILES='1@2304:swap_ab=1,tile_n=32,tile_k=256;1@1152:swap_ab=1,tile_n=32,tile_k=256;8@2304:swap_ab=1,tile_n=32,tile_k=256;8@1152:swap_ab=1,tile_n=32,tile_k=256;64@2304:swap_ab=0,tile_n=64,tile_k=256;64@1152:swap_ab=1,tile_n=32,tile_k=256' \
  -e INKLING_PICO12=dense,lmhead,stack1 \
  -e INKLING_PREFILL=qkvr_split,bias_layout,quant_once,partial_bf16 -e INKLING_PREFILL_QKVR_OUT=strided \
  -e INKLING_CERT_HEAD=1 -e INKLING_WIDE=dense,lmhead -e INKLING_WIDE_SCOPE=verify -e INKLING_VATTN=1 \
  -e INKLING_ATTN=prefill_swa,prefill_global -e INKLING_PFUSE=all \
  -e INKLING_PFUSE_HOIST=1 -e INKLING_ATTN_GATE=all -e INKLING_PROLOGUE=1 -e INKLING_MOE2=tiles -e INKLING_SCODES=1 \
  -e INKLING_VERIFY_CAP_FASTPROBS=0 -e INKLING_VERIFY_CAP_FUSED_ACCEPT=0 -e INKLING_NOEXP=1 -e INKLING_PADX=1 \
  -e INKLING_DFLASH_BYTES=mx8,mx8w,hot8 -e INKLING_DFLASH_HOT_N= \
  -e INKLING_DFLASH_PREFIX_TAIL=2048 -e INKLING_DFLASH_PREFIX_TAIL_MAX_INPUT=4096 -e INKLING_DFLASH_TAIL_SNAPSHOTS=8 \
  -e INKLING_DFLASH_PCOPY=12 -e INKLING_DFLASH_PCOPY_MIN_CTX=8192 -e INKLING_DFLASH_CTX8=0 -e INKLING_DFLASH_CTX8_MAX_M= -e INKLING_PROMPT_TOUCH=1 \
  -e INKLING_DFLASH_FUSED_ACCEPT=1 -e INKLING_RSFIX=1 \
  -e INKLING_KV12=0 -e INKLING_KV12_MAX_PAGES=2048 \
  -e INKLING_CERT_WIDE=1 -e INKLING_CERT_WIDE_MAX_ROWS= -e INKLING_CERT_WIDE_STATS=0 \
  -e INKLING_DRAFT_RNNHOIST_MAX_BS=32 \
  -e INKLING_KV12S=1 -e INKLING_KV12S_OVERFLOW_HEADS=2048 -e INKLING_KV12S_FALLBACK_PAGES=64 \
  -e SGLANG_RAGGED_VERIFY_MODE=static -e INKLING_KV12S_POOL=static -e INKLING_KV12S_DFLASH_SETTLE=shared \
  -e MAX_JOBS=4 \
  -e SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1 -e SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1 \
  -e SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE=1 -e SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS=1 \
  -e MALLOC_MMAP_THRESHOLD_=1048576 \
  --rm -e SGLANG_ENABLE_DFLASH_DRAFT_KV_RING=1 -e DFLASH2_SHA256="<the three pinned sha256 lines>" \
  -v ~/models/Inkling-2-Mini-DFlash2:/draft:ro \
  --entrypoint bash lmsysorg/sglang@sha256:efac853b46811ccb760545b7b29e7119bdf907537ccf6e863fac3011fe8b98de \
  -c 'mkdir -p /root/.cache/dflash2-logs; exec > >(tee -a /root/.cache/dflash2-logs/server-$(date -u +%Y%m%dT%H%M%SZ).log) 2>&1
      (cd /draft && printf "%s\n" "$DFLASH2_SHA256" | sed "s/ /  /" | sha256sum -c --quiet -) || {
        echo "DFlash2 drafter files do not match the pinned sha256" >&2; exit 1; }
      python3 -I /draft/sglang_inkling_dflash2.py && exec python3 "$@"' dflash2 \
  -m sglang.launch_server --model-path /model --served-model-name inkling-2-mini \
  --tp 1 --ep 1 --attention-backend fa4 --dtype bfloat16 --page-size 128 \
  --speculative-algorithm DFLASH --speculative-draft-model-path /draft --speculative-num-draft-tokens 8 \
  --speculative-draft-attention-backend fa4 --speculative-use-block-verification \
  --max-running-requests 48 --cuda-graph-max-bs-decode=48 \
  --mem-fraction-static 0.88 --context-length 262144 \
  --grammar-backend xgrammar --reasoning-parser inkling --tool-call-parser inkling2 \
  --enable-metrics --host 127.0.0.1 --port 8000 --weight-loader-disable-mmap --swa-full-tokens-ratio 0.037 \
  --max-mamba-cache-size 512 --max-prefill-tokens 8192
```

With `SPEC=dspark` the container starts with `--entrypoint python3` (no drafter mount, no patch), the DSpark switches
take their own defaults (table below) and the speculative flags are the cookbook's
`--speculative-algorithm DSPARK --speculative-num-draft-tokens 8 --speculative-use-block-verification`.

Engine flags (the SGLang cookbook's DGX Spark cell, text only, with the DFlash2 drafter):

| flag / variable | why |
|---|---|
| `lmsysorg/sglang@sha256:efac853b…` | the `inkling-2-mini` tag pinned by digest (multi-arch; the Spark pulls arm64 `sha256:bc811ac7…`); SGLang `0.0.0.dev1+g03fde9cc3` |
| `--gpus all --ipc=host --shm-size 16g` | the GB10, and shared memory for the scheduler / detokenizer processes |
| `--network host` + `--host 127.0.0.1` | the API listens on loopback only |
| `--restart no` | nothing restarts behind your back after an out-of-memory |
| `--rm` (DFlash2) | the drafter's compatibility patch rewrites installed SGLang files inside this container; `--rm` discards the container, and with it the patched files, when it stops |
| `--entrypoint bash -c '... sha256sum -c ... python3 -I /draft/sglang_inkling_dflash2.py && exec python3 "$@"'`, `DFLASH2_SHA256` (DFlash2) | checks the drafter's sha256 again on the read-only mount the patch and weights load from, applies the drafter's SGLang compatibility patch, then starts the server; the server log is also written to `~/.cache/inkling-2-mini-sglang/dflash2-logs/` |
| `-v ~/models/Inkling-2-Mini-DFlash2:/draft:ro`, `--speculative-draft-model-path /draft` | the drafter at the pinned revision (`DFLASH2_DIR`), read-only. With `SPEC=dflash2`, extra `--speculative-draft-model-path` or `--context-length` flags are refused: set `DFLASH2_DIR` and `CTX` (the inputs the sha256 check and the memory budget are made for) |
| `--speculative-algorithm DFLASH --speculative-num-draft-tokens 8` | SGLang's DFLASH worker with the drafter's block size of 8 (7 drafts per verify step) |
| `--speculative-draft-attention-backend fa4` | FlashAttention-4 for the drafter's attention (the drafter's model card) |
| `SGLANG_ENABLE_DFLASH_DRAFT_KV_RING=1` | the drafter's KV is a fixed ring over its 2,048-token window (2.16 GB), not a per-token pool; the packed store requires it |
| `--speculative-use-block-verification` | block verification for sampled requests (`DFLASH_BV=1`, only with the overlays and `RSFIX`'s `reject` part; `VCAP_BV=1` with `SPEC=dspark`; not part of the cookbook cell) |
| `-v …:/model:ro`, `HF_HUB_OFFLINE=1` | local weights at the pinned revision; no Hugging Face access at start |
| `-v ~/.cache/inkling-2-mini-sglang:/root/.cache` | keeps the JIT / Triton / FlashInfer caches and the overlay extension builds for warm starts |
| `MAX_JOBS=4` | caps parallel JIT compile jobs during the first start (cookbook) |
| `SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1` | target verify attention through the split-KV relative-bias decode kernel (cookbook) |
| `SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1`, `…_PREPARE_INPUTS_ONCE=1`, `…_SKIP_PREFILLED_INPUTS=1` | DSpark step glue inside the CUDA graphs (cookbook; DSpark draft path) |
| `--tp 1 --ep 1` | one GPU |
| `--attention-backend fa4` | FlashAttention-4 path with Inkling's relative bias (the only backend Inkling2 accepts) |
| `--dtype bfloat16` | activations; the weights stay NVFP4 / MXFP8 as shipped |
| `--page-size 128` | KV page size (cookbook) |
| `--max-running-requests 48 --cuda-graph-max-bs-decode=48` | up to 48 running requests, decode CUDA graphs captured to 48 (set with the packed store) |
| `MEM_FRACTION=0.88` (`--mem-fraction-static 0.88`) | static pool for weights, KV and CUDA graphs. With DFlash2 up to `CTX=262144`: 0.88 with any prefill lever on, as with the default sink gate (the levers add up to ~0.6-0.7 GB of transient memory during long prefills; at 0.89 the guard stopped the server under the worst-case load), 0.89 with `PFUSE_HOIST=0 ATTN_GATE=0 PROLOGUE=0`; 0.85 above 262,144 (the prefill scratch grows with `CTX`). `SPEC=dspark` keeps 0.89. Global-layer KV tokens at 262,144: 2,371,200 on v2.9, box B (v2.8.2: 2,349,056, -2.1 % against 0.89; see Memory budget) |
| `SWA_RATIO=0.037` (`--swa-full-tokens-ratio 0.037`) | sliding-window pool as a fraction of the global-layer pool, enough for 48 requests' windows plus one prefill chunk (0.05 with `KV12S=0`; the image defaults to 0.1) |
| `MIN_AVAIL_GIB=3.5` | host MemAvailable floor for the memory guard. Accepts decimal GiB |
| `--max-mamba-cache-size 512`, `--max-prefill-tokens 8192` | conv-state slots (the image's default sized 1,232; peak use measured 76) and one 8,192-token prefill chunk in flight instead of two: +2.2 % KV tokens, prefill and decode unchanged |
| `LOAD_NOMMAP=1` (`--weight-loader-disable-mmap`, `MALLOC_MMAP_THRESHOLD_=1048576`) | one-pass safetensors read; large frees return to the OS before SGLang sizes the KV pool; `0` uses the stock mmap loader |
| `--context-length 262144` | per-request limit; the KV pool is shared, so this reserves nothing per request. The model supports 1,048,576 |
| `--grammar-backend xgrammar` | JSON-schema / regex constrained output |
| `--reasoning-parser inkling`, `--tool-call-parser inkling2` | thinking to `reasoning_content`, answer to `content`, tool calls to `tool_calls` |
| `--enable-metrics` | Prometheus `/metrics` (speedbench reads them) |

Profile and drafter switches:

| launch.sh variable | default | what |
|---|---|---|
| `SPEC` | `dflash2` (`dspark` with `ENGINE=vllm`) | `dflash2`: the DFlash2 drafter; `dspark`: the model's own DSpark draft (v2.7.2 profile); `none`: no speculation |
| `ENGINE` | `sglang` | `vllm`: the vLLM alternative profile (DSpark, no overlays) |
| `DFLASH2_DIR` | `~/models/Inkling-2-Mini-DFlash2` | drafter directory; revision and sha256 checked on every start |
| `DFLASH_BYTES` (`INKLING_DFLASH_BYTES`), `DFLASH_HOT_N` | `mx8,mx8w,hot8` with the overlays, `65536` | drafter byte cuts: `mx8` MXFP8 drafter linears; `mx8w` (needs `mx8`) FlashInfer's b12x MXFP8 GEMM on the same bytes for the linears whose K and N are multiples of 128, up to `INKLING_DFLASH_MX8W_MAX_M` (256) rows, E8M0 scales swizzled once (+11.2 MB); `hot8` FP8 hot-vocabulary plane (first `DFLASH_HOT_N` ids of `HOTVOCAB_FILE`) for the selector's candidate projection; `hot16` the same in BF16; empty = stock drafter |
| `CERT_DFLASH` (`INKLING_CERT_HEAD`) | `1` with the overlays | certified greedy LM head on DFLASH verify steps of <= 16 rows; `check` serves the full head and counts mismatches; `0` = off. Needs `PICO12` with `lmhead` |
| `CERT_WIDE_DFLASH` (`INKLING_CERT_WIDE`) | `1` with `CERT_DFLASH=1` | certified head for 17-192-row DFLASH verify batches; its token ids go straight to the DFLASH accept (needs `CERT_DFLASH=1`) |
| `DFCAP` (`INKLING_VERIFY_CAP`) | `dfpolicy:/opt/inkling-overlay/inkling_vcap/policies/df-SLDF2x.json,bsmax=8` with the overlays | DFLASH verify-width cap: calibrated survival policy priced separately for greedy and sampled requests, with a context bucket (`theta_x`, `ctx_edges` 16,384 / 65,536); all 8 rows above 8 requests, and for sampled requests from 8 requests up; the v2.8.1 policy is `df-SLDF1-sm.json`; `fixed:K` and `conf:T` exist for measurement; `off` = full width |
| `DFCAP_NOEXP` (`INKLING_NOEXP`), `DFCAP_REMAP` (`INKLING_VERIFY_CAP_REMAP`) | `1`, `1` | dead rows read the anchor row's experts (remap) and then get no experts at all (`inkling_noexp`) |
| `DFLASH_BV`, `DFCAP_BV` | `1`, `DFLASH_BV` | block verification on sampled steps (`--speculative-use-block-verification`); under the cap the cut drafts are masked as a stopped draft. Only with `OVERLAY=1` and `RSFIX` including `reject` (exact only with that fix); otherwise it stays off |
| `DFLASH_FUSED` (`INKLING_DFLASH_FUSED_ACCEPT`) | `1` with the overlays | sampled accept from the logits with a sparse draft q (`inkling_dfaccept`) for batches of `INKLING_DFLASH_FUSED_MIN_BS` (16) requests or more; smaller batches and calls with a verify-cap cutoff keep the stock accept; `0` = off |
| `DFLASH_PREFIX_TAIL` (`INKLING_DFLASH_PREFIX_TAIL`) | `2048` | on a prefix-cache hit, prefill the last N prompt tokens again so the drafter sees them (`inkling_dftail`; DFLASH only, not streaming sessions; pinned to the image's source); `0` = off. Set by the launcher only: the launcher refuses `INKLING_DFLASH_PREFIX_TAIL*` in `SGLANG_ENV` / `SGLANG_ARGS` |
| `DFLASH_PREFIX_TAIL_MAX_INPUT` (`INKLING_DFLASH_PREFIX_TAIL_MAX_INPUT`) | `4096` | only prompts of at most this many tokens are bounded; longer prompts keep the cache as before. Raise it (e.g. 8192) for long-output or document-QA traffic; at a raised limit, agent loops with short turns on 5-8K histories lose ~5-9 % end to end |
| `DFLASH_TAIL_SNAPSHOTS` (`INKLING_DFLASH_TAIL_SNAPSHOTS`) | `8` | prompt-end drafter windows kept for later prefix-cache hits (40 MiB per slot, 42.5 MiB with `DFLASH_TAIL_DEFER=1`, charged to the KV budget; restored only on matching content and salt; cleared by weight updates and `flush_cache`); `0` = off, for example where tenants must not observe each other's cache use. Set by the launcher only |
| `DFLASH_TAIL_DEFER` (`INKLING_DFLASH_TAIL_DEFER`) | `1` | the prefix tail defers to a draft-tail snapshot that already covers the drafter's window: the request keeps its prefix-cache hit and the window is restored instead of recomputed (prompts of at most `DFLASH_PREFIX_TAIL_MAX_INPUT` tokens; needs `DFLASH_PREFIX_TAIL` and `DFLASH_TAIL_SNAPSHOTS` on; not for logprob requests, more than one rank or hierarchical / external KV storage). `0` = v2.9.3's behaviour. Set by the launcher only: the launcher refuses `INKLING_DFLASH_TAIL_DEFER` in `SGLANG_ENV` / `SGLANG_ARGS` |
| `RSFIX` (`INKLING_RSFIX`) | `1` | exact sampled-verification fixes: `reject`, `selector`, `markov` (SGLang PR #51681), `blockwise` (SGLang PR #51692); both profiles |
| `STEPLOG` | empty | `STEPLOG=<label>`: full-width DFLASH step capture to `~/.cache/inkling-2-mini-sglang/steplog/<label>/` for fitting verify-width policies (measurement only; excludes `DFCAP`) |

With `SPEC=dflash2` the launcher turns off the switches that act on the DSpark draft or its verify epilogue, whatever
they are set to: `VERIFY_CAP`, `DRAFT_BYTES`, `VCAP_BV`, `VCAP_FASTPROBS`, `VCAP_FUSED`, `NOEXP`, `CERT_HEAD` and
`CERT_WIDE` are replaced by the DFLASH switches above, `KV12` is off, and `PICO12=...,drafthead` is refused.
`DFLASH_BYTES` and `DFLASH_FUSED` are refused with `SPEC=dspark` or `OVERLAY=0`, and `KV12S_POOL=unified` with
`SPEC=dflash2`. `OVERLAY=0` with `SPEC=dflash2` is plain stock DFLASH (no overlays, no block verification, BF16 KV store,
`MEM_FRACTION=0.85`).

Overlay switches (`OVERLAY=0` turns all of them off; each is described in `overlay/<package>/README.md`):

| launch.sh variable (container env) | default | what |
|---|---|---|
| `OVERLAY_DIR` (mounted at `/opt/inkling-overlay`, on `PYTHONPATH`) | `overlay/` of this checkout | the overlay packages and the loader |
| `VERIFY_CAP` (`INKLING_VERIFY_CAP`) | `SPEC=dspark`: `policy:/opt/inkling-overlay/inkling_vcap/policies/r2-theta004.json,bsmax=16` | calibrated verify-width rule (policy file: per-position survival tables and the price theta); `bsmax=16`: all 8 rows above 16 requests; the round-1 `cost:1=0.30,2=0.25,4=0.15,8=0.0` still works; `off` = stock |
| `VCAP_BV` (`--speculative-use-block-verification`, `INKLING_VERIFY_CAP_BV`) | `SPEC=dspark`: `1` (`mask`) | block verification for sampled requests under the verify cap; `mask` hands the kernel the dead drafts as a stopped draft, `stock` runs the kernel over all rows and caps after (both exact); `0` = chain accept |
| `VCAP_FASTPROBS` (`INKLING_VERIFY_CAP_FASTPROBS`) | `SPEC=dspark`: `1` | max draft probabilities for the verify rule and the sampled accept's draft probabilities from the BF16 draft logits in one or two reads |
| `VCAP_FUSED` (`INKLING_VERIFY_CAP_FUSED_ACCEPT`) | `SPEC=dspark`: `1` | sampled accept (chain or block verification) computed from the logits for decode batches of >= 3 requests (`INKLING_VERIFY_CAP_FUSED_ACCEPT_MIN_BS`); self-checked at start-up, stock accept otherwise |
| `DRAFT_BYTES` (`INKLING_DRAFT_BYTES`), `HOTVOCAB_FILE` | `SPEC=dspark`: `gemv,fold8,head8,draftmx8,hotvocab:65536,hotbase,wide8,w8tile,rnnhoist`; `.../inkling_draftbytes/hotvocab/inkling2-mini-hot-r2.json` (both profiles) | draft logits on the GEMV, FP8 Markov fold, FP8 draft LM head, MXFP8 draft layers, draft vocabulary = the first 65,536 ids of the ranked list, compact hot base logits in the Markov walk, single-read FP8 draft GEMM above 64 rows, narrow-N tiles for the MXFP8 draft layers, precomputed Markov projection partials; `dwin:W[:S]` is optional (off by default) |
| `RNNHOIST_MAX_BS` (`INKLING_DRAFT_RNNHOIST_MAX_BS`) | `32` | largest decode batch the Markov-walk hoist serves (v2.7.2: 16; 32 changes c17-c32 DSpark drafts by FP32 order, proposals only) |
| `KERNELS` (`INKLING_KERNELS`), `MOE_TILES` | `moe_tiles,qkvr_gemv` | GB10 NVFP4 tile table for decode; stacked Q/K/V on one GEMV |
| `PICO12` (`INKLING_PICO12`) | `dense,lmhead,stack1` | pico's 12-bit packed GEMV for qkvr / wo and the target LM head at <= 16 rows; `stack1`: the stacked q/k/v of the looped layers in one pass over the coded matrix (+1.7 GB memory for the packed copies; the extensions build in ~30-35 s each on the first start) |
| `PREFILL` (`INKLING_PREFILL`) | `qkvr_split,bias_layout,quant_once,partial_bf16` | prefill: per-source qkvr projection, relative bias written in FA4's layout, MoE input quantised once per token, BF16 MoE combine |
| `CERT_HEAD` (`INKLING_CERT_HEAD`) | `SPEC=dspark`: `1` | certified greedy LM-head argmax (6-bit screen + pico's exact GEMV on the candidate tiles; +358 MB); `check` serves the full head and counts mismatches; `0` = off. Needs `PICO12` with `lmhead` |
| `WIDE` (`INKLING_WIDE`), `WIDE_SCOPE` | `dense,lmhead`, `verify` | packed 12-bit GEMM for 17-256-row calls (verify batches of 3-32 requests) on decode / verify forwards; per-kind row caps `INKLING_WIDE_MAX_M_KIND` (default `qkvr=96,wo=32,head=256`) |
| `VATTN` (`INKLING_VATTN`) | `1` | verify attention on the split-KV relative-bias kernel for up to 48 requests, bias in-kernel above 32 rows (`cap` = only the row cap) |
| `KV12` (`INKLING_KV12`), `KV12_SLOTS` (`INKLING_KV12_MAX_PAGES`) | `1` only with `KV12S=0` and `SPEC=dspark`, `2048` | packed 12-bit copy of full, committed global-layer KV pages, read by decode / verify attention; 2,048 slots per global layer = 6.1 GiB, enough for one 256K-token request (`0` = one slot per page of the pool) |
| `ATTN` (`INKLING_ATTN`) | `prefill_swa,prefill_global` | fused relative-bias attention for prefill / extend (CUDA extension, builds in ~30 s on the first start) |
| `SCODES` (`INKLING_SCODES`) | `1` | NVFP4 expert scale codes (+0.81 GB: the stock scales stay resident for escapes and fallback; first start builds the modified FlashInfer kernels, ~4.5 min, then cached) |
| `MOE2` (`INKLING_MOE2`) | `tiles` | tile tables for the routed-expert GEMMs at verify batch sizes (over `moe_tiles`); `INKLING_MOE2_NVFP4` / `INKLING_MOE2_MXFP8` override the tables |
| `PFUSE` (`INKLING_PFUSE`) | `all` | prefill glue fusion (`loopmix,resnorm,sconvres`), extend batches of >= 512 rows only |
| `PFUSE_HOIST` (`INKLING_PFUSE_HOIST`) | `1` | loop controller's source projection prepared once per loop pass (`inkling_phoist`; changes the FP32 summation order, not the math; quality bracketed equal; start-up self-test with fallback); `0` = off (v2.9.1 behaviour); `exact` = the bit-identical, slower variant; needs `PFUSE` with `loopmix` |
| `DFLASH_PCOPY` (`INKLING_DFLASH_PCOPY`) | `12` | `SPEC=dflash2`: prompt copy, the key length of the match (12 is the only supported value); `0` = off (v2.9.2 behaviour). Set by the launcher only |
| `DFLASH_PCOPY_MIN_CTX` (`INKLING_DFLASH_PCOPY_MIN_CTX`) | `8192` | the prompt copy runs only for requests whose prompt has at least this many tokens |
| `DFLASH_CTX8` (`INKLING_DFLASH_CTX8`) | `0` (opt-in since v2.9.7) | `SPEC=dflash2` with the overlays: the drafter's two context projections, which write the committed rows into the draft window after every verify (the context `fc` 9216 -> 2304 and the five layers' stacked K/V 2304 -> 10240), read MXFP8 copies on FlashInfer's b12x GEMM for 1-256 rows (larger calls keep BF16; +48 MB of copies, every b12x tile plan compiled at start-up); draft proposals only; online weight updates are refused while it is on (the copies are derived weights); not armed when SGLang's `SGLANG_OPT_DFLASH_AUX_GRAPH` is set; `0` = BF16 (v2.9.5 behaviour). Set by the launcher only |
| `DFLASH_CTX8_MAX_M` (`INKLING_DFLASH_CTX8_MAX_M`) | empty (= 256) | largest context append (rows) that `DFLASH_CTX8` runs on MXFP8, 1-4096 |
| `PROMPT_TOUCH` (`INKLING_PROMPT_TOUCH`) | `1` | after a request finishes, re-touch its prompt path in the prefix cache (`inkling_ptouch`; eviction order only, nothing computed changes); `0` = off (v2.9.1 behaviour) |
| `ATTN_GATE` (`INKLING_ATTN_GATE`) | `all` | sink gate in the prefill attention epilogue on the sliding-window and global layers (`inkling_attn`; bit-identical to `gate_sink`); `1` = sliding-window layers only; off by itself while `SGLANG_KERNEL_API_LOGLEVEL` is non-zero (that logging reads the output buffer the gate leaves unwritten); `0` = off |
| `PROLOGUE` (`INKLING_PROLOGUE`) | `1` | sliding-window prefill K conv + L2 clip and V conv + value norm, one kernel each (`inkling_prologue`; bit-identical, stock row-sum order); `0` = off |
| `CERT_WIDE` (`INKLING_CERT_WIDE`), `CERT_WIDE_MAX_ROWS` | `SPEC=dspark`: `1`, overlay default | certified greedy head for 17-192-row verify batches (needs `CERT_HEAD`, `PICO12` and `WIDE` with `lmhead`) |
| `INKLING_ATTN_W8_FROM` | `131072` | prefix length (keys) from which global-layer prefill uses the 8-warp fused kernel; `0` = one 4-warp launch |
| `KV12S` (`INKLING_KV12S`), `KV12S_POOL` (`INKLING_KV12S_POOL`) | `1`, `static` | packed global KV store (see [Packed global KV store](levers.md#packed-global-kv-store-default-kv12s1)); turns `KV12` off. `static` runs on SGLang's stock pools with prefill CUDA graphs, `MEM_FRACTION=0.89` and `SWA_RATIO=0.037` unless set; `unified` (`SPEC=dspark` only) is the v2.7.1 binding (`--enable-unified-memory --cuda-graph-backend-prefill=disabled`, 0.91 / 0.035). `0` = the BF16 store (`MEM_FRACTION` 0.85 with DFlash2, 0.93 with DSpark); vLLM and `OVERLAY=0` always use `0`. Set the store with these variables, not through `SGLANG_ENV` / `SGLANG_ARGS` (the launcher refuses the store's variables there, and with `KV12S=1` also the pool's engine flags, so the engine flags and the overlay always agree). `KV12S_OVERFLOW_HEADS` (2,048) and `KV12S_FALLBACK_PAGES` (64) size the exact overflow arena and fallback pool per layer visit |
| `KV12S_SETTLE` (`INKLING_KV12S_DFLASH_SETTLE`) | `shared` | `SPEC=dflash2` with `KV12S=1`: after each verify step the 8 packed global visits settle the step's tentative writes. `shared` builds the page candidates once for all visits (as the DSpark path does since v2.7.3) instead of once per visit: 77 fewer launches per step, single-request decode step -0.25 ms (v2.9.5). Exact: the same K/V bits and attention outputs. `1` also runs the trim and commit kernels once over all visits (14 launches per step instead of 112); it is exact as well, but in the v2.9.5 measurements it was slower at 4 and 8 requests in flight, so it is opt-in. `0` = each visit on its own (v2.9.4). The store's bytes are identical in every mode except where one visit's side-slot claims race near arena exhaustion, which the per-visit path resolves in scheduling order as well |
| `NOEXP` (`INKLING_NOEXP`) | `SPEC=dspark`: `1` | dead verify rows get no experts (inactive when `VERIFY_CAP=off` or its remap is off); `0` = off |
| `PADX` (`INKLING_PADX`) | `1` | padding rows of decode / verify CUDA-graph batches route to the phantom expert, so the expert GEMMs skip them (`inkling_padx`; per graph at capture: graphs of 10 requests and up; live rows bit-identical; composes with the dead-verify-row skip); `0` = off |

Left at SGLang's defaults (measured, see [experiments.md](experiments.md)): chunked prefill 8,192 tokens (16,384: +2-3 % at
32K-128K, more memory), radix prefix cache on, overlap scheduler on, prefill CUDA graphs on (off only with
`KV12S_POOL=unified`).

Extra engine flags go in `SGLANG_ARGS` (or after `scripts/launch.sh up`), extra container variables in `SGLANG_ENV`
(`-e NAME=value ...`); both are split on whitespace, so their values cannot contain spaces.

## Memory budget

GB10 has one 121.6 GiB pool for the GPU and the operating system. `MEM_FRACTION=0.88` budgets for the weights,
KV and CUDA graphs. With `LOAD_NOMMAP=1`, large frees from checkpoint loading return to the OS before SGLang
sizes the static pool. The packed store's fixed buffers (prefill scratch, overflow arena, fallback pool) and the
drafter's KV ring are charged to that budget before the pool is sized.

| | |
|---|---|
| KV cache, global layers (packed 12-bit target visits) | 2,371,584-2,416,896 tokens (box B, across boots) |
| drafter KV ring (2,048-token window) | 2.16 GB, fixed |
| prefill scratch, overflow arena, fallback pool | 1.03 GiB at `CTX=262144` (4.03 GiB at 1M), 8 x 2,048 x 32 KiB, 8 x 64 x 512 KiB |
| KV cache, sliding-window layers | ~88-89K tokens at `SWA_RATIO=0.037` (computed from the ratio). With radix caching on, 48 requests at one 1,024-token window plus a 128-token page each and one 8,192-token prefill chunk need ~63K tokens |
| MemAvailable when the API is ready / worst case (thinking c24 + cold 240K needle) | 4.5-5.5 / 4.9 GiB (box B) |

The defaults target at least 3.5 GiB MemAvailable under stress, a level that held up in memory-cliff measurements
on other GB10 recipes; MemAvailable after start varies by up to ~1.3 GiB between boots.
[Full memory sweep (DSpark profile)](experiments.md#v26-kv-pool-sizing)

To leave more RAM for other work, lower `MEM_FRACTION`. Each 0.01 frees ~1.09 GiB, taken from the KV pools
(the weights, graphs and fixed buffers do not shrink).

## What differs from the published recipes, and why

| | SGLang cookbook (DGX Spark, DSpark) | vLLM recipe (DGX Spark) | here |
|---|---|---|---|
| engine | SGLang | vLLM | **SGLang**: the stock engines decode at the same speed on one Spark (c1 within 3 %, c16 / c24 within run noise); SGLang prefills 1.3-3.4x faster, keeps long prompts in its prefix cache more reliably and leaves twice the host memory ([experiments.md](experiments.md#sglang-vs-vllm-same-benchmark-both-engines-on-both-boxes)) |
| draft | DSpark (bundled) | DSpark (bundled) | **DFlash2** (incoai, pinned revision, sha256-checked); DSpark with `SPEC=dspark` or `ENGINE=vllm` |
| memory | `--mem-fraction-static 0.45` (518K KV tokens) | `--gpu-memory-utilization` unset (vLLM's own suggestion would leave the host 0-2 GiB) | `MEM_FRACTION=0.88`, `SWA_RATIO=0.037` and the packed global KV store give 2,371,200 global-layer KV tokens, 4.6 GiB MemAvailable under stress and a 3.5 GiB guard floor (v2.9, box B) |
| context | not set (model default) | 262,144 | 262,144 (the model supports 1M; see [Longer context](#longer-context-512k-and-1m)) |
| multimodal | `--enable-multimodal` | on | off (text only; add `--enable-multimodal` for images) |
| images, weights | pulled by tag, weights from the Hub at start | same | pinned image digest, weights and drafter at pinned revisions, sha256-checked, offline at start |
| bind address | 0.0.0.0 | 0.0.0.0 | 127.0.0.1 |
| speed overlays | - | - | twenty-four packages (`overlay/`), on by default; `OVERLAY=0` disables them |

## Longer context: 512K and 1M

The model supports 1,048,576 tokens; `CTX` sets the per-request limit (default 262,144). The KV pool is shared, not
reserved per request. With the packed store, `CTX=1048576` sizes the prefill scratch for 1M tokens (4.03 GiB instead
of 1.03 GiB). Above 262,144 the DFlash2 profile uses `MEM_FRACTION=0.85`.

`CTX=1048576 scripts/launch.sh up` on v2.8 (DFlash2 profile), box A (`results/V28m1-s05/`): needle PASS at 999,296
tokens; decode at 1,000K context 23.3 tok/s (one repetition); pool 2.14M global-layer tokens; lowest MemAvailable
8.9 GiB. 512K with the DFlash2 profile is part of the v2.8.x qualification. The measurements below are the DSpark
profile's.

`SPEC=dspark CTX=1048576 scripts/launch.sh up`, v2.7.2 defaults, box B (`results/Q272M-s06/`): needle PASS at 999,296
tokens (prefill and answer in 931 s); a cold 1,024,202-token document prefilled in 971 s (1,055 tok/s, TTFT cached
1.7 s), then decoded at 13.8 tok/s (194 ms step, 2.7 tokens per step on that text); pool 2,057,856 tokens; lowest
MemAvailable 6.5 GiB.

Earlier measurements with the v2.6 build and the BF16 store (`KV12S=0`), each needle also with thinking c24 running
alongside:

| `CTX` | needle | cold prefill | decode at that context | lowest MemAvailable |
|---|---|---:|---:|---:|
| 524288 | PASS at 499,995 tokens (also with thinking c24 alongside) | 499,995 tokens in 332-351 s | 24.5 tok/s at 491K (112 ms step) | 5.5 GiB |
| 1048576 | PASS at 999,296 tokens (thinking c24 alongside) | 1,024,204 tokens in 1,285 s | 14.0 tok/s at 1.02M (200 ms step) | 4.9 GiB |

A single 1M prompt takes about half of the packed pool (two thirds of the BF16 one) and, while it prefills,
concurrent requests slow down (thinking c24 fell to 62 tok/s aggregate in the v2.6 run). 512K and the 1M needle with
concurrent thinking were not re-measured with the packed store.

## Reproducing the numbers

The benchmark commands (speedbench at `6733653`, `bench/full.sh`, `bench/report.py`, `bench/screen.sh`, and the
launch lines for the stock cells) are in the README's [Reproduce](../README.md#reproduce) section.

The overlays' own checks run inside the image with the server stopped: `overlay/tools/gpu_check_qkvr.py`,
`gpu_check_pico12.py`, `gpu_check_prefill.py`, `gpu_check.py` (draft package), `gpu_check_bv.py`,
`gpu_check_fastprobs.py`, `gpu_check_fused_accept.py` (sampled accept), `gpu_check_rsfix.py`, `gpu_check_kv12s.py`, and
the CPU unit tests in `overlay/tests/` (see `overlay/README.md`). The DFLASH sampled-path law test under the verify
cap is `tools/gpu_check_dflash2_cap_law.py` (inside the image with `INKLING_RSFIX=1`); `tools/dfreplay.py` fits and
replays verify-width policies from `STEPLOG` captures.
