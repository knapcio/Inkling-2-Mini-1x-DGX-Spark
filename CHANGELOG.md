# Changelog

## v2.8.1 (2026-11-24)

Basic tests on one box; full two-box qualification follows in v2.8.x.

Fixes prefix-hit draft starvation of the DFlash2 drafter. The drafter's KV is a ring over the last 2,048 positions,
filled from the target's hidden states as the target prefills or verifies them. On a prefix-cache hit the target
skips the cached prefix, so the ring held only the uncached suffix and the drafter drafted with little or no context.
speedbench primes every prompt, and real chat and agent turns re-hit cached prefixes, so v2.8's speedbench
long-context (-16 %) and agentic (-4 %) deficits against v2.7.2 came from this.

- `inkling_dftail` (new; launcher `DFLASH_PREFIX_TAIL=2048`, `DFLASH_PREFIX_TAIL_MAX_INPUT=4096`): on a prefix-cache
  hit of a prompt of at most 4,096 tokens, the last 2,048 prompt tokens are prefilled again (SGLang's own prefix bound,
  as for logprob requests), so the drafter sees them. Prefill-only (exact up to the floating-point order of the
  prefill chunking), DFLASH only, never streaming sessions, pinned to the image's source; allocates no memory.
  `DFLASH_PREFIX_TAIL=0` turns it off.
- speedbench full, release defaults on box B (`results/V281s-s06/`, `MEM_FRACTION=0.89`), against v2.8 on the same
  box (`MEM_FRACTION=0.85`): all prompts 69.59 -> 72.86, long context 63.09 -> 73.53, agentic 73.44 -> 82.74, code
  79.84 -> 81.88; prose, structured and multilingual move by under 0.5 %. longctx-02 (4,930 tokens) is above the
  limit and runs as on v2.8. With the radix cache off, the investigation runs reached 72.82 / 75.00 / 82.42 (all /
  long context / agentic). Against the published v2.7.2 (same harness; v2.8.1 one box, v2.7.2 a two-box median):
  all +14.2 %, agentic +7.9 %, long context on par (73.53 vs 75.1, -2.1 %). End to end (time to first token plus decode, 256-token outputs): long context +6.2 %, agentic +1.2 %, code
  +1.6 %.
- Trade-off: median time to first token on cache hits rises (speedbench long-context prompts 186 -> 540 ms, agentic
  170 -> 235 ms). The first hit of a prompt re-prefills it fully, because the hybrid state cache resumes only at
  checkpoints; later hits reuse a checkpoint planted at the bound. Cached hits at 32K / 128K are above the limit and
  unchanged (0.24 / 0.46 s; without the limit 6.38 / 35.3 s). Decode at 32K / 128K on v2.8.1: 64.3 / 50.7 tok/s. Multi-turn agent sessions (8 conversations of 4-8 turns, 5-12K-token histories) behave
  exactly as on v2.8 at the default limit; raising it to 6,144 / 8,192 / 12,288 costs +5.0 / +8.7 / +14.8 %
  end-to-end time at 160-token turns and is about break-even at 512-token turns. Raise it only for long-output or
  document-QA traffic.
- Exactness: greedy output with the tail is bit-identical to the same server with the radix cache off (43/43 prompts,
  top-5 logprobs); against the default cached server it differs only by near-ties, the same ones the stock server
  shows between cache on and cache off (`results/T81EX2*-s06/`).
- The 32K / 128K decode caveat stays: it is the drafter's 2,048-token window, not the cache (no gain at 32K / 128K
  with the radix cache off either).
- Memory: worst case (thinking c24 alongside a cold 240K needle) 4.1 GiB, guard 3.5 GiB; the tail allocates no
  memory, and the run generated 4 % more tokens than the v2.8 run. Needles at 128K and 240K PASS.
- 1M on v2.8 (`CTX=1048576`, box A, `results/V28m1-s05/`): needle PASS at 999,296 tokens, decode at 1,000K context
  23.3 tok/s (one repetition), pool 2.14M global-layer tokens, lowest MemAvailable 8.9 GiB.
- Results: `results/V281s-s06/` (v2.8.1 release defaults), `results/T81a1-s06/` (tail 2,048, limit 8,192),
  `results/N81sbfc-s06/` (v2.8), `results/N81sbf-s06/` (radix cache off),
  `results/T81t1k-s06/` (1,024-token tail), `results/T81t2k-s06/` (no input limit),
  `results/T81kv-s06/`, `results/A81off-s06/`, `results/A81m6k-s06/`, `results/A81m8k-s06/`, `results/A81m12k-s06/`,
  `results/V28m1-s05/`, `results/T81EX2nc-s06/`, `results/T81EX2tail-s06/`, `results/T81EX2ref-s06/`.

## v2.8 (2026-11-24)

Basic tests on one box; full two-box qualification follows in v2.8.x.

The default drafter is the public third-party DFlash2 drafter
[incoai/Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) (Apache-2.0), with every exact
lever of this recipe on top of it. The model's own DSpark draft stays available as `SPEC=dspark`. v2.7.3 and v2.7.4
were never released separately; their changes are folded in here. One GB10 Spark, release defaults, against the
v2.7.2 two-box medians: speedbench full 63.8 -> 69.66 tok/s (+9.2 %), prose +20.6 %, multilingual +30.9 %, code
+10.8 %, structured +13.4 %; the long-context category is -16.0 % (the
drafter's 2,048-token window, see the long-context caveat below) and agentic trails slightly (-4.1 %); prose decode
c1 / c4 / c8 / c16 65.1 / 140.5 / 203.5 / 288.3 (52.6 / 113.3 / 164.6 / 244.1); code
79.0 / 141.6 / 211.3 / 298.2 (62.2 / 124.8 / 186.6 / 268.2); thinking c8 / c24 277.4 / 515.5 aggregate, 48 x 2,048
tokens (235.3 / 438.8 same protocol); 2,371,584-2,416,896 global-layer KV tokens (2.13-2.17M); MemAvailable 4.5-5.5 GiB
at ready, 4.9 GiB in the worst case. Long context: at 32K context the DFlash2 profile decodes about 12 % slower
than v2.7.2's DSpark profile at c1 (60-63 vs 72 tok/s, same box). Its acceptance falls with context (3.10 vs 3.36
tokens per step at 32K), because the drafter sees only a 2,048-token window. At 128K both decode at about 50 tok/s.
For mostly long-context single-user work, `SPEC=dspark` is the faster choice. Prefill is unchanged within noise (8K
5,383 tok/s). Speedbench quick (cap on vs off): 70.61 vs 62.49. Results: `results/V28sbf-s05/`,
`results/V28M1v28-s05/`, `results/V28M1off-s05/`, `results/V28M1nonx-s05/`, `results/V28kv-s06/`, `results/V28lc-s05/`, `results/R272lc-s05/`,
`results/V28dsp-s05/`.

- Default `SPEC=dflash2`: a 5-layer block-diffusion draft with a 2,048-token draft window and a 16-candidate selector
  (7 drafts per verify step), pinned at revision `4f9088ea66c21f2b99ed657d1941c306b03e84d2`. `scripts/launch.sh`
  checks the revision, the file list and the sha256 of `config.json`, `model.safetensors` and
  `sglang_inkling_dflash2.py` on every start, on the host and again inside the container. The drafter's SGLang
  compatibility patch runs inside the serving container only, which is started with `--rm`. The drafter's KV is a
  fixed ~2.16 GB ring over its window (`SGLANG_ENABLE_DFLASH_DRAFT_KV_RING=1`).
- **Verify-width cap on the DFlash2 drafter** (`inkling_vcap`, on by default with `SPEC=dflash2` for batches of 1-8
  requests; default `DFCAP=dfpolicy:.../df-SLDF1-sm.json,bsmax=8`). Each request verifies only the drafted tokens a
  calibrated survival rule finds worth their MoE cost; unverified rows reuse their request's anchor experts and, with
  `DFCAP_NOEXP=1` (default), get no expert work. The accept stops at each request's live length. Greedy output is the
  target argmax (differences only at floating-point near-ties) and the sampled law is unchanged (Monte Carlo law
  tests). Speedbench quick on one box, cap on vs off: 62.49 -> 70.61 tok/s (+13.0 %); decode c1 prose +17.5 %, code
  +16.7 %. `DFCAP=off` restores full-width verification.
- On by default with the drafter, besides the cap:
  - `inkling_dfbytes` (new, `DFLASH_BYTES=mx8,hot8`): MXFP8 drafter linears and a 64K FP8 hot-vocabulary plane for the
    selector's candidate projection. Draft proposals only; speedbench quick +8.4 % in development.
  - `inkling_cert` on DFLASH: the certified greedy LM head on verify steps of up to 16 rows (`CERT_DFLASH=1`; c1
    +2.8 %, 0 mismatches in 34,808 checked rows) and, with it, the wide certified head for 17-192-row batches, whose
    token ids go straight to the DFLASH accept (`CERT_WIDE_DFLASH`; code c8 +3 %, greedy texts identical 30 / 30).
  - Block verification on sampled steps (`DFLASH_BV=1`); under the cap the cut drafts are masked as a stopped draft.
    It needs `RSFIX`'s `reject` part and stays off without it.
  - `inkling_dfaccept` (new, `DFLASH_FUSED=1`): the sampled accept from the logits with a sparse draft q, at 16 or more
    requests (`INKLING_DFLASH_FUSED_MIN_BS=16`; thinking c24 per-request decode +2.3 %; c8 stays on the stock accept).
  - `inkling_kv12s` accepts a DFLASH draft on the static pools: the packed store is the default in both profiles.
    `MEM_FRACTION=0.89` up to `CTX=262144` and 0.85 above, `SWA_RATIO=0.037`.
- `inkling_rsfix` (new, `RSFIX=1`, both profiles): the residual / bonus CDF pass of SGLang's reject sampler no longer
  emits a zero-mass padded row or an out-of-range lane under FP32 rounding; the DFLASH selector walk no longer clamps a
  no-match to a zero-probability last candidate; SGLang PR #51681 (DSpark Markov sampler) and PR #51692 (blockwise
  rejection, off the recipe's path). The law is unchanged in exact arithmetic; the Monte Carlo law tests of the DFLASH
  sampled path (chain and block verification, T 1.0 / 0.7, top-k 5, with and without the cap) show 0 invalid tokens
  in about 1M steps per case.
- Checks on one box: needles at 128K and 240K PASS; greedy texts cap on vs off identical on 28 / 30 speedbench prompts
  (the other two first diverge at top-2 near-ties, reduction order only), NOEXP on vs off 30 / 30.
- Development measurement, stock DFlash2 against stock DSpark: tokens per step x1.25, prose decode c1 / c4 / c8 / c16
  32.4 / 74.8 / 115.2 / 183.3 -> 46.2 / 109.4 / 161.4 / 254.8, code x1.22-1.25, thinking c8 187.2 -> 223.1.
- Folded in from v2.7.3: `RSFIX=1` (above) and, for `SPEC=dspark`, `DRAFT_BYTES` gains `hotbase` (the hot-vocabulary
  base logits stay compact inside the captured Markov walk; bit-identical step logits and tokens).
- Folded in from v2.7.4: `RNNHOIST_MAX_BS` 16 -> 32 (the Markov-walk hoist up to 32 rows; changes c17-c32 DSpark drafts
  by FP32 order, proposals only); the DFlash2 profile, then opt-in, is now the default; `scripts/launch.sh down` waits
  for the `--rm` container to be removed.
- Fallbacks: `SPEC=dspark` is the v2.7.2 profile with the changes above (speedbench quick 63.7 tok/s on box A); `ENGINE=vllm` keeps DSpark. `OVERLAY=0` with
  the default `SPEC` is plain stock DFLASH (no block verification); the stock cookbook cell is now
  `SPEC=dspark OVERLAY=0 LOAD_NOMMAP=0 MEM_FRACTION=0.62 SWA_RATIO=0.1`.
- `scripts/launch.sh`: `SPEC=dflash2` replaces the DSpark draft and verify switches (`VERIFY_CAP`, `DRAFT_BYTES`,
  `VCAP_*`, `NOEXP`, `CERT_HEAD`, `CERT_WIDE`) by the DFLASH ones and refuses `KV12S_POOL=unified`,
  `PICO12=...,drafthead`, `ENGINE=vllm`, and `--context-length` / `--speculative-draft-model-path` among the extra
  flags (set `CTX` / `DFLASH2_DIR`); it refuses to start with less than `MIN_AVAIL_GIB` + 8 GiB MemAvailable;
  `STEPLOG=<label>` captures full-width DFLASH steps for policy fitting.
- `scripts/download.sh` fetches the drafter (~1 GB) only for the DFlash2 profile (`NO_DFLASH2=1`, `SPEC=dspark` or
  `ENGINE=vllm` skip it; `--dflash2` fetches only the drafter), removes a file that fails its checksum (run it again
  to fetch it) and writes `.revision` only after every file has been verified.
- New checks: `tools/gpu_check_dflash2_cap_law.py` (law test under the cap), `overlay/tools/gpu_check_rsfix.py`,
  `tools/dfreplay.py` (cross-fitted replay of verify-width policies).
- Not measured in this release (v2.8.x qualification): the quality guard and sanity suite, 200K-1M context,
  structured / JSON decode.

## v2.7.2 (2026-11-24)

The packed global KV store moves from SGLang's unified memory pool to its stock pools, with prefill CUDA graphs back:
greedy outputs are now bit-identical to the BF16 store (v2.7.1 differed on 5 of 32 prompts). Verify rows the verify
cap has cut get no experts (+3 to +6 % concurrent decode). Full runs on both boxes (`results/Q272R-s05/`,
`results/Q272R-s06/`).

- `inkling_kv12s` on the static pools (`KV12S_POOL=static`, default): the packed target envelopes replace the global
  BF16 buffers inside SGLang's hybrid sliding-window pools, pages are released on the forward stream, and the draft
  keeps its own BF16 pool and captured fold. 2,171,392 / 2,126,720 global-layer KV tokens (v2.7 store 1,593,088,
  v2.7.1 2,264,704); `MEM_FRACTION=0.89`, `SWA_RATIO=0.037`; the draft's two window lanes are charged per window token.
  Lowest MemAvailable 5.6-6.9 GiB in the full runs and 5.4 GiB in the worst case (thinking c24 alongside a cold 240K
  needle: needle 92.5 s, thinking 304.6 tok/s).
- Captured prefill reads the packed prefix through the fixed BF16 scratch, gathered on the device inside the graph
  (the packed loader when a prefix does not fit); a first build that read every captured 8K chunk with the packed
  loader was 2.2x slower on a cold 240K prompt. Cold prefill 1K-128K 2,905 / 5,439 / 5,085 / 3,737 tok/s.
- Exactness: greedy speedbench quick texts identical to the BF16 store on 32 of 32 prompts (two runs, with and without
  `NOEXP`). v2.7.1's 5 differing prompts came from eager prefill (no padded prefill-graph buckets); the same tree with
  prefill graphs disabled reproduces exactly those 5.
- `inkling_noexp` (new, `NOEXP=1`): cut verify rows route to a phantom expert that every MoE kernel skips. Greedy
  decode prose c4 / c8 / c16 +2.9 / +5.0 / +5.3 %, code c16 +5.1 % in paired screens on both boxes; c1 and thinking
  above 16 requests unchanged. Inactive when the verify cap or its remap is off.
- Two-box medians against v2.7.1: speedbench full 63.8 (63.8), prose decode c4 / c8 / c16 113.3 / 164.6 / 244.1
  (109.9 / 160.5 / 230.9), thinking c1 / c8 / c24 72.2 / 233.6 / 437.1 (71.8 / 237.6 / 436.9), c48 596.4. Warm
  start-up 122-192 s (prefill graphs are captured again; v2.7.1 86-91 s).
- `scripts/launch.sh`: `KV12S_POOL=static|unified`; `KV12S_POOL=unified NOEXP=0` reproduces v2.7.1. The launcher
  refuses packed-store variables or pool flags passed through `SGLANG_ENV` / `SGLANG_ARGS`, so the engine flags and
  the overlay agree.
- Screened, not adopted: quant-once routed NVFP4 quantisation and a GB10 launch config for the loop-controller mix
  (bit-identical, flat in serving at c4-c24); a pfuse attention-side mix launch config (+0.7 % prefill, different
  FP32 order on prompts of 512 tokens or more).
- 1M (`CTX=1048576`, box B, `results/Q272M-s06/`): needle PASS at 999,296 tokens in 931 s; a cold 1,024,202-token
  prefill in 971 s (v2.7.1 1,040 s), decode 13.8 tok/s at a 194 ms step (v2.7.1 16.0 at 215 ms; tokens per step differ
  with the generated text); pool 2,057,856 tokens; lowest MemAvailable 6.5 GiB.

## v2.7.1 (2026-11-24)

The packed global KV store is the default (`KV12S=1`): 2.26M global-layer KV tokens instead of 1.59M (+42 %), with
outputs bit-identical to stock SGLang on its unified memory pool. Full runs on both boxes (`results/Q271R-s05/`,
`results/Q271R-s06/`); `KV12S=0` restores the v2.7 store.

- `inkling_kv12s` fixes since v2.7:
  - The draft's global layer is stored raw BF16 (the packed draft lane overflowed and needed a 10 GiB workspace).
  - Prefill reads packed prefixes through a fixed BF16 scratch charged to the KV budget and sized for `CTX`
    (the first build was ~2x slower on cold prefills above 120K tokens and grew memory per chunk).
  - The sliding-window band of the unified pool is laid out page-major and written by a Triton store kernel: the
    token-major layout spread one window over ~80 GPU pages and cost 22 % of sampled thinking throughput at c24.
  - Exhaustion of the exact overflow arena falls back to a 64-page raw pool and then to the staging slot; only if
    all three are full does the server abort the requests that hold rows (HTTP 503) instead of stopping. A host-side
    guard evicts cached prefixes, defers prefill and retracts decode before that point. Measured use: 0.
  - Startup refuses unsupported combinations (streaming sessions, multimodal, parallelism above 1, non-DSpark
    speculation, prefill CUDA graphs, verify shapes above the compact attention cutoff) and invalid sizes.
- Defaults with `KV12S=1`: `MEM_FRACTION=0.91`, `SWA_RATIO=0.035`, `--enable-unified-memory`, prefill CUDA graphs
  off. MemAvailable 7.3-7.4 GiB after start, 4.3-4.5 GiB lowest in the full runs and 4.5 GiB in the worst case
  (thinking c24 alongside a cold 240K needle, needle PASS); 3.5 GiB guard floor.
- Speed against v2.7's store on the same box: speedbench full 63.91 vs 64.27, decode c1-c16 and prefill 1K-128K within
  run noise, thinking c48 596.5-596.6; thinking c1 / c8 / c24 71.8 / 237.6 / 436.9 vs 71.5 / 236.4 / 451.9 in the v2.6 full
  runs: c24 -3 %, the cost of running the draft commit outside the CUDA graph.
- Exactness: stock SGLang's unified and default pools differ on 5 of 32 greedy speedbench quick texts. Both hand the
  first verify step bitwise identical layer inputs, and both stay equally close to an exact prefill of the same text;
  the differences are floating-point order in later steps (docs/experiments.md).
- 1M with the packed store (`CTX=1048576`, box A): needle PASS at 999,296 tokens; a cold 1,024,203-token prefill in
  1,040 s (v2.6: 1,285 s), decode 16.0 tok/s at that context (v2.6: 14.0), lowest MemAvailable 5.1 GiB; pool 2,154,112
  tokens with the 4.03 GiB scratch. 512K and the 1M needle under concurrent thinking were not re-measured.
- `scripts/launch.sh`: `KV12S` accepts only 0 / 1; the redundant `--cuda-graph-backend-decode=full` is dropped.
- `overlay/tools/gpu_check_kv12s.py` covers the release scratch path, overflow fallback, side-slot exhaustion,
  the page-major window band and the raw draft lane.

## v2.7 (2026-11-23)

Certified head for wider verify batches, a long-prefix prefill fix, a measured 512K / 1M option, and an opt-in packed
global KV store. All defaults are exact; the default store's greedy speedbench quick texts are identical to v2.6 on
32/32 prompts. Full v2.7 benchmark runs follow; the speed tables still show the v2.6 runs.

- `inkling_cert` `CERT_WIDE=1` (default): certified greedy head for 17-192-row verify batches (one 6-bit screen pass,
  exact packed wide arithmetic on the candidate-tile union, reduced to ids). Head + softcap + argmax 3.99 -> 1.72 ms
  at 64 rows, 5.09 -> 3.19 ms at 128 rows; serving greedy decode c4 +1.6 to +2.1 %, c8 +1.0 to +1.5 % (2 on / 2 off
  per box), c1 and c16 within noise.
- `inkling_attn`: global-layer prefill over prefixes of >= 131,072 keys runs the existing 8-warp fused kernel
  (`INKLING_ATTN_W8_FROM`), bit-identical to the single launch; 1.39x / 1.47x per global layer and 8K chunk at
  512K / 1M prefixes.
- Memory flags `--max-mamba-cache-size 512 --max-prefill-tokens 8192`: KV pool 1,593,088 tokens on box A
  (+2.2 %), worst case 5.2 GiB, prefill and decode unchanged.
- 512K / 1M context option documented: needles PASS at 499,995 and 999,296 tokens (with thinking c24 alongside),
  decode 24.5 tok/s at 491K and 14.0 tok/s at 1.02M, lowest MemAvailable 4.9-5.5 GiB (default store).
- `inkling_kv12s` (new, opt-in `KV12S=1`): global-layer KV stored only in the exactly coded 12-bit form, 2,168,320
  KV tokens (+40 %), speedbench quick 63.16 vs 63.49, sanity 11/11, needles 120K / 240K PASS, lowest MemAvailable
  4.2 GiB. Every packed store and read is bit-identical in the GPU checks, but in live serving it computes different
  target logits in decode / verify (found after the release; experimental, not for real work), and long cold prefills and the 1M option are not yet measured with it;
  therefore not the default. Fixes follow in v2.7.1.
- Screened, not adopted: verify-policy refit on fresh full-width v2.6 logs (-0.22 %, CI -0.52..+0.71 %), split-K
  small kernels for router / conv projection / controllers (flat in serving), INT8 draft weights (flat), dropping
  the 12-bit side copy for +8 % KV (-9 % decode at 128K-250K).

## v2.6 (2026-11-23)

Larger exact KV pools. `MEM_FRACTION` goes from 0.81 to 0.93 and `SWA_RATIO=0.05` halves the image's
sliding-window ratio. The global-layer pool grows by 40 %, from 1,105,792 / 1,106,816 to 1,551,104 / 1,557,120
tokens (box A / box B). MemAvailable is 5.8 / 6.1 GiB after start and 5.2 / 5.4 GiB under the combined
thinking and cold needle load. The memory guard defaults to `MIN_AVAIL_GIB=3.5` and accepts decimal GiB.
The [memory sweep](docs/experiments.md#v26-kv-pool-sizing) covers the choice of defaults.

Throughput stays within run-to-run variation. The full runs used `MEM_FRACTION=0.94` on both boxes
(`bench/full.sh`, `results/V26-s05/`, `results/V26-s06/`). Median speedbench full c1 was 64.4 (v2.5 64.2),
prose decode c4 / c8 / c16 was 107.5 / 161.8 / 239.1 tok/s and json c16 was 280.5 tok/s. Thinking c1 / c8 / c24
reached 71.5 / 236.4 / 451.9 tok/s (70.5 / 235.6 / 445.2). Cold prefill was within run noise and 32K / 128K
decode was 71.0 / 46.4 tok/s. Warm start-up took 121-137 s. Both boxes scored 30/30 on the quality guard and
11/11 on sanity, and passed the needles at 120,097 and 240,104 tokens.

## v2.5 (2026-11-23)

One-pass checkpoint loading, a cheaper draft Markov walk and full-context draft attention by default. Median of
both boxes, `bench/full.sh` (`results/V25-s05/`, `results/V25-s06/`), against v2.4: speedbench full c1 64.2 (64.1)
at 2.383 tokens per step in both releases; decode prose c1 / c4 / c8 / c16 53.0 / 107.7 / 161.9 / 238.9
(52.9 / 107.3 / 159.6 / 236.4), code / json c16 260.4 / 277.1 (261.3 / 275.4); thinking c1 / c8 / c24
70.5 / 235.6 / 445.2 (68.2 / 239.9 / 439.4); cold prefill 1K / 8K / 32K / 128K 2,893 / 5,464 / 5,143 / 3,806
tok/s (2,939 / 5,456 / 5,148 / 3,805); decode at 32K / 128K 71.0 / 46.4 (63.3 / 48.2). At 200K / 250K on box B,
c1 decode 43.8 / 37.9 tok/s, cold prefill 3,105 / 2,655 tok/s. Warm start-up 137 / 131 s and KV pool 1,105,792 /
1,106,816 global-layer tokens (box A / box B); 16.5-16.6 GiB MemAvailable after start. The stress runs
reached 14.1 / 14.5 GiB lowest MemAvailable (box A / box B) with thinking c24 at 310.7 / 309.2 tok/s alongside a
cold 240K-token needle, PASS. Quality guard
30/30, sanity 11/11, needles 120K / 240K PASS on both boxes.

- `inkling_draftbytes` `rnnhoist`: precompute the known 90 % of the Markov projection in FP32, run the 768x512
  dependent slice per step, then round once to BF16 and call the stock gates. Draft proposals only. The seven-step
  walk takes 320 / 454 / 596 us at 1 / 8 / 16 rows (584 / 661 / 782 us, cold L2, CUDA graph).
- `LOAD_NOMMAP=1`: SGLang's `--weight-loader-disable-mmap` reads the checkpoint in one pass. Warm start-up
  263-268 -> 131-137 s, target weight load 160 -> 22-34 s in the screen. `MALLOC_MMAP_THRESHOLD_=1048576`
  returns large frees to the OS before KV allocation, reducing the pool loss from 6 % to 2.6 % at the same fraction.
  `MEM_FRACTION` default 0.80 -> 0.81.
- `dwin:16384` is off by default again. The 16K window loses far-context copying. Controlled A/Bs on both boxes
  put net decode at -11 / -7 / -3 / +7 % at 32K / 64K / 128K / 250K. The option stays available for workloads
  mostly above 200K tokens.
- Screened, not adopted: Humming MoE, per-row INT8 draft weights, context n-gram proposals, extra draft vocabulary
  rows per step. Details and the same-harness vLLM comparison are in docs/experiments.md "v2.5".

## v2.4 (2026-11-22)

Sampled-path verification, a drafter attention window and a packed global-layer KV; all exact. Median of both boxes,
`bench/full.sh` (`results/V24-s05/`, `results/V24-s06/`), against v2.3: thinking c1 / c8 / c24 68.2 / 239.9 / 439.4
(69.8 / 236.5 / 427.9); decode prose c8 / c16 159.6 / 236.4 (156.5 / 232.9), code c16 261.3 (256.0), json c16 275.4
(268.3); decode at 128K context 48.2 (43.1, +12 %), at 32K the same step time as v2.4 with its levers off (47.2 vs 47.3 ms, same session; the full-run tok/s cell, 63.3 vs
69.6, follows the accepted tokens of the generated texts); 200K / 250K c1 decode 44.1 / 38.9
on box B (v2.3 38.1 / 35.1 on box A); cold prefill 1K / 8K / 32K / 128K 2,939 / 5,456 / 5,148 / 3,805 tok/s
(2,914 / 5,352 / 5,036 / 3,751, run noise); speedbench full c1 64.1 (64.7) at identical tokens per step (2.383): a
same-session ablation on both boxes puts the c1 step with all v2.4 levers on and off within 0.6 %, so the -0.9 % is
run-to-run variation, not the levers; KV pool 1.11-1.12M tokens (-6 %: `MEM_FRACTION` 0.84 -> 0.80 for the 6.1 GiB
packed side pool), lowest MemAvailable 14.8-15.6 GiB under thinking c24 with a cold 240K-token prompt; quality guard
30/30, sanity 11/11, needles 120K / 240K PASS on both boxes.

- `inkling_vcap`: block verification (Sun et al., ICLR 2025) for sampled requests under the verify cap
  (`VCAP_BV=1` -> `--speculative-use-block-verification`, `INKLING_VERIFY_CAP_BV=mask`: dead drafts enter as a
  stopped draft); thinking c1 / c8 / c24 +3.0 / +4.8 / +4.0 % in the screen.
- `inkling_vcap` `VCAP_FASTPROBS=1`: the verify rule's max draft probabilities and the sampled accept's q from one
  read of the BF16 draft logits; +1.7 to +3.5 % (thinking c8, greedy c8 / c16).
- `inkling_vcap` `VCAP_FUSED=1`: sampled accept computed from the logits for >= 3 requests; code c16 +2 %.
- `inkling_draftbytes` `dwin:16384`: the draft's global layer reads a 128-token sink + the last 16,384 tokens; c1 step
  at 128K / 250K -1.4 to -2.4 / -4.0 to -4.8 ms (+2 to +6 %, both boxes), tokens per step unchanged.
- `inkling_kv12` (new, `KV12=1`, `KV12_SLOTS=2048`): full, committed global-layer KV pages also kept as exactly coded
  12-bit pages (2,048 slots per layer, 6.1 GiB) and decoded in the decode / verify attention; bit-identical partials;
  c1 step at 128K 62.2 -> 56.8 ms, 250K 80.1 -> 75.2 ms.
- `MEM_FRACTION` default 0.84 -> 0.80.
- Rejected: draft sampling temperature, 4-bit draft MLP (tokens-per-step-only gain within boot-to-boot variation, step
  time unchanged), late kill of doomed verify rows, shared-prefix KV reads, fused routed epilogue, streaming NVFP4 MoE
  kernel, centroid draft vocabulary, verify-policy retunes, 80K hot vocabulary. Details: docs/experiments.md "v2.4".

## v2.3 (2026-11-22)

Three exact levers. Median of both boxes, `bench/full.sh` (`results/V23-s05/`, `results/V23-s06/`), against v2.2:
speedbench full c1 64.7 (61.8, +5 %), prose 54.4 (51.9); decode prose c1 / c2 / c4 / c8 / c16 53.4 / 72.5 / 110.0 /
156.5 / 232.9 (51.0 / 69.6 / 107.0 / 154.1 / 228.8), code c8 178.3 (171.4); thinking c1 / c8 / c24 69.8 / 236.5 /
427.9 (67.1 / 219.6 / 419.6); prefill within run noise; KV pool 1.18-1.19M tokens (-2 %: +0.81 GB resident scales);
quality guard 30/30, sanity 11/11, needles PASS on both boxes.

- `inkling_scodes` (new, `SCODES=1`): NVFP4 expert block scales as row base + 5-bit codes, expanded inside a
  modified FlashInfer SM120 grouped-GEMM mainloop; -4.06 % of expert bytes, bit-identical outputs.
- `inkling_pico12` `stack1`: the stacked q/k/v of the looped layers in one pass over the coded weight (no kv pack).
- `inkling_draftbytes` `w8tile`: narrow-N tiles for the MXFP8 draft layers (bit-identical).
- Screened, not adopted: stream-K packed GEMM for 97-192 rows, prefill glue round 3 (both flat in serving).

## v2.2 (2026-11-22)

GB10 tile tables for the routed-expert GEMMs (`overlay/inkling_moe2`, default `MOE2=tiles`; bit-identical outputs).
Median of both boxes, `bench/full.sh` (`results/V22-s05/`, `results/V22-s06/`), against v2.1: decode prose c2 / c4 /
c8 / c16 69.6 / 107.0 / 154.1 / 228.8 (68.0 / 103.2 / 149.8 / 216.7), code c4 / c16 122.2 / 250.5 (115.8 / 240.3);
thinking c8 / c24 219.6 / 419.6 (220.8 / 405.4); c1 unchanged (speedbench 61.8, the c1 path does not use these tiles);
quality guard 30/30, sanity 11/11, needles PASS on both boxes. Same-day A/B on box A (control / moe2 / control):
decode +2 to +6 % at c2-c16. Prefill numbers move within run noise.

## v2.1 (2026-11-21)

Hot-vocabulary draft. Median of both boxes, `bench/full.sh` (`results/V21-s05/`, `results/V21-s06/`), against v2:
speedbench full c1 61.7 (58.1, +6 %), prose 51.7 (48.5), multilingual 42.3 (39.4); decode prose c1 / c4 / c16 51.0 /
103.2 / 216.7 (47.5 / 99.7 / 212.1), code c1 60.2 (56.7), json c8 / c16 173.6 / 265.6 (164.6 / 251.8); thinking c1 /
c8 / c24 67.8 / 220.8 / 405.4 (65.8 / 211.1 / 402.2); prefill and memory unchanged within run noise (1K / 8K / 32K /
128K 2,914 / 5,293 / 4,995 / 3,726 tok/s; 1.20M KV tokens, 17.5-17.8 GiB lowest MemAvailable); quality guard 30/30,
sanity 11/11, needles PASS on both boxes.

- `inkling_draftbytes`: default `hotvocab:65536` with the shipped list
  `overlay/inkling_draftbytes/hotvocab/inkling2-mini-hot-r2.json` (ranked on 1.95M tokens of the model's own outputs,
  `tools/hotvocab/`), and `wide8` (FP8 draft GEMMs read their weight once above 64 rows; bit-identical).
- Same-day A/B on both boxes: speedbench quick +6.0 / +6.1 %, greedy texts byte-identical 30/30.
- Rejected: KV page size 16 (flat decode; the fused prefill attention needs page 128).

## v2 (2026-11-21)

Five new overlay packages, a new verify-width rule and a larger KV pool; all levers exact (same weights, same target
distribution). Median of both boxes, `bench/full.sh`, against v1:

- speedbench full c1 58.1 (v1 55.8, +4 %); decode table prose c4 / c8 / c16 99.7 / 149.1 / 212.1 (90.1 / 121.5 /
  190.6: +11 / +23 / +11 %), code c4 / c8 / c16 +10 / +11 / +7 %; thinking c1 / c8 / c24 65.8 / 211.1 / 402.2
  (62.8 / 193.8 / 372.8: +5 / +9 / +8 %); cold prefill 1K / 8K / 32K / 128K 2,821 / 5,317 / 5,060 / 3,761 tok/s
  (+13 / +25 / +28 / +34 %); KV pool 1.19-1.21M global-layer tokens (794K, +52 %) with 17.5-17.9 GiB lowest
  MemAvailable; quality guard 30/30, sanity 11/11, needles 120K / 240K PASS.
- `inkling_vcap`: `policy:` mode, a calibrated left-to-right verify-width rule (per-position survival tables from 36K
  logged request-steps, one price for all batch sizes); default `policy:.../r2-theta004.json,bsmax=16`.
- `inkling_cert` (new): certified greedy LM-head argmax (6-bit screen + exact pico GEMV on the candidate tiles).
- `inkling_wide` (new): packed 12-bit GEMM for 17-256-row verify batches, decode / verify forwards only.
- `inkling_vattn` (new): verify attention on the split-KV relative-bias kernel up to 48 requests, bias in-kernel.
- `inkling_attn` (new): fused relative-bias prefill attention (sliding-window and global layers).
- `inkling_pfuse` (new): prefill glue fusion (loop-message mixes, attention-residual mix + norm, short conv + residual).
- `--mem-fraction-static` 0.62 -> 0.84 (worst case with thinking c24 and a 240K-token prompt measured at 0.72-0.86).
- Rejected: L2 prefetch / warm kernels before the small per-visit kernels, a native layer-stack backend (1.13x upper
  bound), the verify policy above 16 requests. Details: docs/experiments.md "Round 2 (v2)".

## v1 (2026-11-21)

- 2026-11-21: first release. SGLang `lmsysorg/sglang:inkling-2-mini` (`sha256:efac853b…`) with the cookbook's DGX
  Spark DSpark cell (`--mem-fraction-static 0.62`, 262K context, text only) plus five import-time overlays
  (`overlay/`), all on by default in `scripts/launch.sh` (`OVERLAY=0` for the stock engine):
  - `inkling_vcap`: verify-width cap with dead-row expert remap and a batch-aware threshold rule
    (`cost:1=0.30,2=0.25,4=0.15,8=0.0`);
  - `inkling_draftbytes`: draft logits on the GEMV, FP8 Markov fold, FP8 draft LM head, MXFP8 draft layers;
  - `inkling_kernels`: GB10 NVFP4 tile table at decode, stacked Q/K/V on one GEMV;
  - `inkling_pico12`: pico's exactly coded 12-bit BF16 GEMV for the dense projections and the LM head;
  - `inkling_prefill`: per-source qkvr projection, relative bias written in FA4's layout, MoE input quantised once,
    BF16 MoE combine.
  Speed and quality numbers: README; every screen, the engine comparison and the rejected levers:
  docs/experiments.md.
- 2026-11-21: SGLang vs vLLM on the same benchmark, both engines on both test boxes: decode tied at c1 (within
  3 %), SGLang 1.3-3.4x faster cold prefill and twice the free host memory -> SGLang is the base.
- 2026-11-21: repository created. Weights revision `ed40e7da`, vLLM image `sha256:c2dc759f…` for the comparison.
  Launcher with a start-up memory guard, download script with sha256 checks, benchmarks (`bench/`: decode table,
  thinking load, sanity checks, screening and full runs).
