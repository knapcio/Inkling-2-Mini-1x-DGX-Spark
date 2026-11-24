# Inkling-2-Mini prefill on one GB10: roofline, serving path, levers

Status 2026-11-21:

- Parts 1-2 are done.
- The bench ran on one Spark; results are in this directory (`bench.log`, `bench.json`, `roofline.log`)
  (bench.log, bench.json) and are folded in below (sections 1.3, 2.2, 3).
- The 34K prefill trace holds no prefill GPU work (section 4). The per-layer-type table in
  section 1.3 is therefore built from benchmarked kernels (`compose.py`), and it reproduces the
  measured TTFT.
- A recapture recipe is in section 4.

Source: the `sglang` Python package of `lmsysorg/sglang:inkling-2-mini` (sglang 0.0.0.dev1+g03fde9cc3.d20261120). All paths below are
relative to it. Files in this directory:

- `bench_tc.py`: GPU microbenchmark (cuBLAS BF16, NVFP4/MXFP8 grouped GEMM, the whole MoE layer,
  FA4 SM120 attention with rel-bias, DRAM bandwidth).
- `roofline.py`: per-token FLOP/byte model (stdlib only). `--rates bench.json` replaces the
  placeholder rates with the measured ones.
- `compose.py`: rebuilds the cold TTFT from the benchmarked per-visit kernel times; output in
  `compose.txt`.
- `analyze_prefill.py`: trace analyzer for part 3.

## 0. Run the benchmark

Stop the model server first. The benchmark allocates about 6 GB of unified memory and never
changes clocks or power limits (it only reads them with `nvidia-smi`).

```
scripts/launch.sh down                      # recipe repo; one GPU job at a time
mkdir -p ~/prefill-bench && cp bench_tc.py roofline.py ~/prefill-bench/
docker run --rm --gpus all --ipc=host \
  -v ~/prefill-bench:/opt/x \
  -v ~/.cache/inkling-2-mini-sglang:/root/.cache \
  --entrypoint python3 lmsysorg/sglang:inkling-2-mini \
  /opt/x/bench_tc.py --json /opt/x/bench.json 2>&1 | tee ~/prefill-bench/bench.log
python3 -I ~/prefill-bench/roofline.py --rates ~/prefill-bench/bench.json
```

- The `/root/.cache` mount reuses the FlashInfer JIT cache from the K1/K1b tile sweeps, so most
  NVFP4 tile configs are already compiled. With a cold cache, the full run takes about 10-20 min.
  Most of that is JIT, and each FA4 rel-bias variant compiles once.
- If the image is only present under the recipe digest, use
  `lmsysorg/sglang@sha256:efac853b46811ccb760545b7b29e7119bdf907537ccf6e863fac3011fe8b98de`.
- Quick subset (~5 min): `--sections env,bw,gemm,moe,attn --no-sweep`. Wider NVFP4 sweep that
  also covers `tile_m` 64/128/256: `--full-sweep`.
- Each section runs in its own subprocess, and so does each NVFP4 tile config, so a rejected
  config or a CUDA fault is reported and the run carries on.

What it prints:

- achieved TFLOP/s for each GEMM and attention case;
- the bitwise checks: stacked vs column-split qkvr, rows at M=16384 vs M=8192, and every NVFP4
  config vs the stock `_tiles()` choice;
- the full MoE layer cost per stage;
- a `rates` line for `roofline.py`.

## 1. Roofline per prompt token

### 1.1 What one prompt token costs

- 48 layer visits: layers 0-5, then 6-17 three times, then 18-23.
- Attention: 40 SWA visits and 8 global visits. The global visits are layer 5, layer 11 x3,
  layer 17 x3 and layer 23.
- MoE: 44 NVFP4 visits and 4 MXFP8 visits.
- **28 of the 48 qkvr projections run on 3 stacked sources (3M rows x all 8600 columns).**
  - Which visits: the 8 global visits always, plus the 20 local core visits in passes 2-3.
    In those visits Q, K and V have separate inputs: `models/inkling2.py:126-151`,
    `inkling2_attn.py:423-441`.
  - Only q|rel|gate|sinks of the first third is used (6552 cols), plus k of the second third and
    v of the third. So **2/3 of those FLOPs are discarded**.

GFLOP per prompt token, as served. Derivation and constants are in `roofline.py`.

| class | 1K | 8K | 32K | 128K | precision / kernel |
|---|---:|---:|---:|---:|---|
| MoE NVFP4, 44 visits x 8 experts x (w13 2304x2304 + w2 2304x1152) | 5.61 | 5.61 | 5.61 | 5.61 | W4A4, CUTLASS `group_gemm_nvfp4_nt_groupwise` |
| MoE MXFP8, 4 visits | 0.51 | 0.51 | 0.51 | 0.51 | W8A8, CuTe `moe_gemm_mxfp8_nt_groupwise` |
| dense qkvr (20 x 39.6 + **28 x 118.9** MFLOP) | 4.12 | 4.12 | 4.12 | 4.12 | BF16 cuBLAS `F.linear` |
| dense wo, router (FP32 out), dyn-conv proj, loop controllers | 0.82 | 0.82 | 0.82 | 0.82 | BF16 cuBLAS |
| attention SWA (40 visits, ≤1024 keys) | 0.24 | 0.47 | 0.50 | 0.50 | FA4 SM120 + rel-bias |
| attention global (8 visits, mean N/2 keys) | 0.05 | 0.40 | 1.61 | 6.44 | FA4 SM120 + rel-bias |
| rel-bias einsum (bandwidth-bound) | 0.04 | 0.04 | 0.04 | 0.04 | cuBLAS, writes 48 KB/token/visit |
| DSpark draft context K/V (3 x 2304→2048) | 0.03 | 0.03 | 0.03 | 0.03 | BF16 cuBLAS |
| LM head (last position only) | ~0 | ~0 | ~0 | ~0 | sm12x GEMV, FP32 |
| **total** | **11.4** | **12.0** | **13.2** | **18.1** | |
| with the qkvr column split (section 3, C1) | 9.2 | 9.8 | 11.0 | 15.9 | |

### 1.2 Activation traffic, the cost a FLOP roofline misses

The prefill path is mostly unfused, so every intermediate goes to DRAM and back. At 8192
tokens a chunk's tensors are tens to hundreds of MB, far over the 24 MB L2.

Estimated bytes read and written per token per layer visit. These are read off the eager path;
expect about ±30 % until the trace confirms them.

| item | KB |
|---|---:|
| rel-bias einsum output `[t,24,1024]` bf16 | 50 |
| rel-bias **shear copy** into `[t,24,1280]` (`flash_attn/cute/interface.py:1170-1290`) | 110 |
| FA4: q + sheared bias + **FP32** output + K/V window | 116 |
| gate_sink (FP32 in, gate in, bf16 out) | 25 |
| MoE routed quantize (8 gathered rows) | 47 |
| MoE gate_up rows BF16 out + swiglu-quantize in/out | 79 |
| MoE down rows BF16 out + `routed_partial` (FP32) + bf16 cast | 97 |
| attn-res mixes x2, norms x2, q/k/rel L2 clips, v_norm, 3 short convs, KV store, residual | 143 |
| **per visit** | **~667** |

- Per token: 48 visits x 667 KB, plus 24 loop-message visits, the 2 unused thirds of the
  stacked qkvr outputs, and the DSpark aux capture. That is **~36 MB/token, ~163 us/token at
  220 GB/s**.
- Measured 8K prefill is 293 us/token. If this estimate holds, more than half of prefill is
  activation traffic, not tensor-core time.

### 1.3 Measured: rates, per-layer-type budget, ceiling

**Rates on GB10 (clocks untouched, SM 2177 MHz at idle).**

| what | measured |
|---|---|
| DRAM copy | 224 GB/s |
| fp32*bf16 elementwise | 224 GB/s |
| cuBLAS BF16 8192^3 | 92.7 TFLOP/s |
| qkvr, M=8192 | 87.6 TFLOP/s |
| qkvr stacked, M=24576 | 91.9 TFLOP/s |
| wo, M=8192 | 83.6 TFLOP/s |
| router, FP32 out | 23.7 TFLOP/s (small) |
| NVFP4 grouped GEMM at T=8192 (stock tiles, uniform routing) | w13 186, w2 133, **together 164 TFLOP/s** |
| NVFP4 grouped GEMM at T=16384 | 215 / 155 TFLOP/s |
| NVFP4 grouped GEMM at T=1024 (weight-bound) | 45 TFLOP/s, 198 GB/s of weights |
| MXFP8 grouped GEMM at T=8192 | 110 / 89 TFLOP/s (102 together); barely above BF16 |
| FA4 SM120 attention, **no rel-bias** | SWA 55-56, global 67-70 TFLOP/s; global reaches 76 % of the BF16 cuBLAS peak |
| FA4 SM120 attention, **with rel-bias, as served** | **SWA 13 TFLOP/s, about 4.3x slower**; global 28 at prefix 0, 39 at 8K, 44 at 24K, 46 at 120K |

**Per-layer-type budget of one 8192-token chunk at prefix 0.** Benchmarked kernels, each times
its visit count (`compose.py`). The measured TTFT is 2.40 s.

| class | ms per chunk | share of 2.40 s | us/token |
|---|---:|---:|---:|
| MoE grouped GEMMs, NVFP4 x44 + MXFP8 x4 | 318 | 13.2 % | 38.8 |
| MoE glue: router, sort, routed quantize, swiglu-quantize, combine + cast | 302 | 12.6 % | 36.9 |
| dense GEMMs, useful work: qkvr as if column-split, wo, dyn-conv | 251 | 10.5 % | 30.6 |
| **dense qkvr stacking waste** (28 x (10.60 - 3.82) ms) | **190** | **7.9 %** | 23.2 |
| attention mainloop without rel-bias, SWA x40 + global x8 | 119 | 5.0 % | 14.6 |
| **rel-bias**: einsum (98) + shear and bias inside FA4 (306) | **404** | **16.8 %** | 49.3 |
| DSpark context K/V | 1 | 0.0 % | 0.1 |
| **not benchmarked**: attn-res mixes, norms, q/k/rel clips, convs, KV store, gate_sink, loop controllers, host | **815** | **34 %** | 99.5 |

Whole prompts (`compose.txt`):

- **The model reproduces the measured TTFTs.**
  - The benchmarked kernels give 1.58 / 7.15 / 42.0 s at 8K / 32K / 128K.
  - The unbenchmarked remainder is nearly the same per chunk: 0.82 / 0.74 / 0.75 s. So it really
    is the per-token elementwise work and not noise.
  - At 1K the remainder is 0.16 s: about 0.09 s of per-token elementwise work plus about 0.06 s of
    fixed per-request cost.
- **At 128K**, global attention is the largest item: 18.5 s of 54 s (34 %).
  - Without rel-bias the same attention would take 13.2 s at 70 TFLOP/s.
  - Rel-bias in total costs 11.8 s at 128K (22 %): 1.6 s einsum, 6.5 s extra in global attention,
    3.8 s extra in SWA attention.

**Checking section 1.2's byte model against the measurements:**

- Rel-bias: modelled 47 us/token, measured 49.
- MoE glue: modelled ~32 us/token, measured 37. The quantize, swiglu-quantize and combine stages
  run at 204-224 GB/s, so they are purely bandwidth-bound and they cost as much as the GEMMs.
- The not-benchmarked elementwise ops: modelled ~48 us/token, measured remainder ~92-100.
  These ops move twice the bytes I counted, or run at half bandwidth. Only a working trace can
  split it (section 4).

**Ceiling.**

- "Ideal" here counts only the tensor kernels at measured rates: MoE GEMMs, the dense GEMMs with
  the column split, and attention without rel-bias.
- In other words, all glue fused away and the bias free. Nothing lossy.

| prompt | measured | ideal | measured / ideal |
|---|---:|---:|---:|
| 1K | 0.46 s (2,190 tok/s) | 0.20 s (5,100 tok/s; MoE weight streaming 153 us/token is the floor) | 43 % |
| 8K | 2.40 s (3,400 tok/s) | 0.69 s (11,900 tok/s) | 29 % |
| 32K | 10.1 s (3,230 tok/s) | 3.3 s (9,900 tok/s) | 33 % |
| 128K | 54 s (2,420 tok/s) | 22.3 s (5,900 tok/s) | 41 % |

**Reading.**

1. The hardware can do 2.4-3.5x more prefill than the stock engine gets.
2. The gap is not slow tensor-core kernels. cuBLAS runs at 88-92 TFLOP/s, against a 92.7 peak.
   FA4 without bias reaches 70 TFLOP/s. NVFP4 runs at 164 TFLOP/s; its tile sweep found at most
   2-3 % (section 2.2).
3. About 70 % of an 8K chunk is not useful tensor work:
   - the elementwise and glue passes (47 %: MoE glue + not-benchmarked remainder);
   - the rel-bias materialisation (17 %);
   - the stacked qkvr (8 %).
## 2. The prefill path at chunk 8192 (from source)

### 2.1 Per layer visit

All of the following is eager Python inside one full-graph replay (section 2.3).
`VERIFY_ROUTED_EPILOGUE` applies to decode/verify only.

1. **Residual mix.** `Inkling2AttnResState.attention_input` → `attn_res` Triton kernel, which
   reads all blocks plus partial.
   - Global layers produce 3 streams (q/k/v queries) and use 3 RMSNorms.
   - Core layers in passes 2-3 also run `LoopMessageBank.mix`: an `attn_res` over the
     previous-pass bank, then `message_controllers` (rank-64 controllers).
   - Files: `models/inkling2.py:126-151`, `inkling2_loop.py`.
2. **qkvr.** `MergedColumnParallelLinear` → `UnquantizedLinearMethod.apply` → **`F.linear`
   (cuBLAS)**. The sm12x GEMV only takes ≤16 rows (`unquant.py:1152-1158`).
   - Stacked 3M rows on 28 visits; `_stacked` is copy-free because the sources are adjacent.
3. **Eager `_prologue`** (`inkling2_attn.py:305-336`). The fused `attention_prologue` kernel is
   decode/verify only.
   - k/v `ShortConvolution`: `_prepare_extend_sconv_cache`, `short_convolution`,
     `update_sconv_cache`. Global layers first compute the dynamic conv weights,
     `F.linear(src, weight_proj[32,2304])`.
   - CuTe-DSL `qk_l2_clip` for q, k and rel. The log-scaling tau is folded into the clip on
     global layers.
   - V RMSNorm.
4. **Rel-bias.** `RelLogitsProj` einsum `thd,de->the`: cuBLAS, bf16 `[t,24,1024]`.
5. **KV store, then FA4 SM120.** `flash_attn_with_kvcache` → `flash_attn_varlen_func`
   (`kernels/ops/attention/flash_attention_v4_sm120.py`; dispatch in
   `flashattention_backend.py:1643-2090`).
   - Paged KV (page 128); causal; SWA `window_size=(1023, 0)`.
   - `num_splits=1`, `pack_gqa=False`, **FP32 output, LSE returned**.
   - rel-bias passes through the `CuSeqlensToBlocks` prep and a **shear kernel that writes a
     `[round_up(t,128),24,1280]` copy**. Pre-shearing is disabled on SM120
     (`inkling2_attn.py:500-512`).
   - Inkling2 asserts `fa4` (`inkling2_attn.py:460-463`). No other attention backend is valid.
6. **gate_sink** (sigmoid gate + sink using the LSE, FP32 in, bf16 out), then **wo** with
   `F.linear` (cuBLAS), then the attn short conv.
7. **MLP side.** `attn_res` mlp mix, the loop message mix in passes 2-3, then mlp RMSNorm.
8. **MoE** (`inkling2_moe.py:827-937`):
   - router: `torch.mm(x, W[128,2304], out_dtype=fp32)`, then Triton `route_topk`;
   - 65,536 ids > `ROUTE_MAX_IDS`=4096, so it takes the **sort path**: `torch.sort(int16, stable)`,
     `get_src2dst`, `compute_grouped_gemm_metadata`, gamma gather;
   - JIT CUDA `nvfp4_routed_quantize`, then CUTLASS grouped GEMM w13, then
     `nvfp4_routed_swiglu_quantize`, then CUTLASS grouped GEMM w2, then Triton `routed_partial`
     (FP32), then `.to(bf16)`;
   - layers 20-23 use the MXFP8 twins with the CuTe grouped GEMM.
9. mlp short conv, then the residual finalize.

**After the stack:**

- the final `attn_res` output mix and RMSNorm;
- the 4 aux captures that DSpark needs (layer visits 5, 23, 41 and 48), copied for every token;
- the LM head on the last position only (FP32 out), softcap, sampling.

### 2.2 MoE tile choice at large M

`_tiles(tokens, width)` (`kernels/ops/gemm/inkling2_grouped_sm120.py:9-18`) is keyed on the
**token count of the forward**. It was tuned on an RTX 5090 with cold weights. `tile_m` is never
passed (FlashInfer default 128). There is no env var, flag or autotune for it.

| tokens in the forward | swap_ab, tile_n, tile_k | rows per expert (uniform) | concern |
|---|---|---:|---|
| ≤128 | True, 32, 128 | ≤8 | decode/verify, swept in K1b (tile_k 256 was +6 %, bit-exact) |
| ≤1536 (the 1K prompt) | True, 32, 256 | ~64 | Measured: stock streams expert weights at 198 GB/s (88 % of copy bandwidth). The double-read concern is **refuted**. Alternatives are within ±1 % uniform and up to -5..-6 % only under heavy skew |
| ≤3072 | True, 64, 256 | ~128-192 | |
| >3072, w13 (K 2304) | **False, 128, 256** | ~512 | Measured: best of the sweep at T=4096/8192. 186 TFLOP/s at 8K, 215 at 16K |
| >3072, w2 (K 1152) | **True, 128, 128** | ~512 | Measured: `swap_ab=0,128,128` is 2.3 % faster at 8K uniform (bit-exact). 133 TFLOP/s; K=1152 is the less efficient half |

- Bench result: all 12 configs are bit-equal to stock at every T and routing.
- The best bit-exact alternative saves 0-2.3 % of a GEMM at 8K/16K, which is ~0.1 % of TTFT.
  **Tile retuning is closed for prefill.**
- MXFP8 (`moe_gemm_mxfp8_nt_groupwise`) has no tile arguments.

### 2.3 Dense GEMMs and prefill CUDA graphs

**Dense GEMMs:** all prefill dense GEMMs are cuBLAS `F.linear`. No slower fallback is involved.
`--bf16-gemm-backend` choices do not help on sm12x (`unquant.py:584-597`; `cutedsl` raises off
SM100). The waste is in the shapes (stacked qkvr), not the library.

**Prefill CUDA graphs: on, backend FULL, and they cover every 8192-token chunk**, including the
chunks of a 128K prompt.

- Inkling2 opts into `Backend.FULL` (`arg_groups/cuda_graph_hook.py:349-378`). The server logs
  "experimental".
- Buckets come from `generate_prefill_cuda_graph_batch_sizes(max_bs)` with max_bs =
  `chunked_prefill_size` = 8192 (`memory_hook.py:246-252`, `cuda_graph_hook.py:517-534`):
  - 4..32 step 4, 48..256 step 16, 288..512 step 32, 576..1024 step 64;
  - 1280..4096 step 256, 4608..8192 step 512;
  - about 50 graphs, each with 16 request slots (8192 // 512, `cuda_graph_setup.py:623-632`).
- Chunks with a prefix still replay:
  - The chunked-prefix capture topology is MLA-only.
  - `maybe_disable_chunked_prefix_cache` forces `disable_chunked_prefix_cache=True` for this
    non-MLA model in `ModelRunner.__init__` (`model_runner.py:465`, `misc_utils.py:28-50`),
    before the runner is built.
  - So `_has_uncapturable_chunked_prefix` is inert. The FA backend sets no `max_seq_len`.
- Eager fallbacks:
  - `return_logprob`, input_embeds/multimodal, LoRA;
  - padding of more than 2x the bucket;
  - more than 16 requests.
- At 8K chunks a graph saves only launch latency, roughly ~1,500 launches x a few us, which is
  ~5 ms of 2.4 s. It matters a little at 1K and more for many short prompts.

### 2.4 DSpark during prefill: no full-prompt draft forward

`DSparkWorkerV2._forward_prefill` (`speculative/dspark_components/dspark_worker_v2.py:825-910`)
does the following:

1. Runs the target with `CaptureHiddenMode.FULL`, so the aux hidden states of 4 layer visits are
   kept for every token (18 KB/token).
2. `TargetHiddenKvInjector.inject_prefill` → `Inkling2ForConditionalGenerationMTP.write_target_hidden_kv`
   (`models/inkling2_mtp.py:284-344`). For each of the 3 draft layers it runs, on all chunk tokens:
   - mix-RMSNorm of the 4 features;
   - `kv_proj_ctx` (2304→2048, cuBLAS);
   - k RMSNorm and RoPE;
   - the KV store.

   That totals 28 MFLOP/token, about 0.2 % of prefill FLOPs, and runs eagerly after the target
   forward.
3. No draft layer, draft LM head or Markov fold runs at prefill. The first draft and verify happen
   in the first decode step.
4. Local draft layers 0 and 2 compute K/V for every prompt token although only the last 1024 are
   read. Skipping them would save about 1 ms per 8K chunk, which is not worth a patch.

### 2.5 Caches per chunk

**KV.** Hybrid SWA pool: `swa_full_tokens_ratio` 0.1 and page 128, both Inkling2 overrides in
`arg_groups/model_overrides/inkling.py`.

- SWA layers read through the SWA page table, so each chunk reads at most the window plus the
  chunk.
- Global layers read the whole prefix from the full pool.
- Each loop pass has its own KV layers (`cache_layer_id`). `--inkling2-loop-share-cache` is lossy
  and is not used.

**Conv states.** Per layer visit (4 short convs x 48 visits): the last 3 tokens, kept in the
"mamba" pool. Strategy `extra_buffer`, forced by the override. For each conv, each chunk runs:

1. `_prepare_extend_sconv_cache`, which loads the previous chunk's tail;
2. the conv;
3. `update_sconv_cache`.

`mamba_track_mask` checkpoints conv states for the radix tree at track intervals. These ops are
tiny per chunk.

**Radix.** Unified radix tree (`uses_mamba_radix_cache=True`). A cold prompt matches nothing. A
cached prefix is reused only at page and track-interval granularity, because the conv-state
checkpoint must exist.

### 2.6 The fixed per-request cost at 1K (measured kernels)

- **Benchmarked kernels for a 1K chunk: 0.30 s.**
  - MoE takes 0.19 s of it, weight-streaming bound: NVFP4 at 198 GB/s and MXFP8 at 186 GB/s of
    touched weights.
  - Dense 0.06 s, rel-bias and attention 0.05 s.
- **Remainder to the measured 0.46 s: 0.16 s.** At the per-token rate of the 8K-128K remainder
  (~92 us/token), about 0.09 s of that is elementwise work, leaving **~0.06 s fixed per request**.
  That covers host tokenize, schedule, detokenize, the HTTP first chunk, and any GPU clock ramp
  after idle.
- The MoE tile hypothesis (a) is refuted (section 2.2).
- A 1K trace captured the section 4 way would split host time from GPU time.

## 3. Prefill levers (numbers from the benchmark; % of the 8K TTFT of 2.40 s)

Exactness key:

| key | meaning |
|---|---|
| E0 | bit-identical |
| E2 | same math, different FP32 accumulation order |
| L | lossy |

### 3.1 Flags and env (runnable now)

| # | lever | exact setting | measured basis, expected effect | exact? | risk |
|---|---|---|---|---|---|
| F1 | Bigger chunk | `--chunked-prefill-size 16384` (`--max-prefill-tokens` is already 16384) | MoE layer per token -5 % (NVFP4 1.524 → 1.445 us/token per visit; MXFP8 2.16 → 2.02) and dense +3 %: **~-2 % TTFT** for prompts ≥16K, 0 below | **E0 confirmed**. qkvr/wo rows at M=16384 and 24576 are bit-equal to M=8192, and MoE stays in the same tile bucket | **Memory.** Activations roughly double (+2-3 GB); prefill graphs are captured up to 16384. Use with F3 and the memory guard |
| F2 | Prefill graphs off (control) | `--disable-prefill-cuda-graph` | ~0 at ≥8K; launch latency is ~5 ms per 2.4 s chunk. Frees the ~50 graphs' memory, which pays for F1 | E0 | low |
| F3 | Fewer prefill buckets | `--cuda-graph-bs-prefill 1024 2048 4096 8192` | memory and start-up only | E0 | padding ≤2x |
| F4 | Spec off (bracket) | `SPEC=none` | DSpark's prefill share: the context K/V is 1 ms per chunk, plus the aux capture. <1 % | E0 for the target | none |
| F5 | Mixed chunk | `--enable-mixed-chunk` | c>1 only; degrades concurrent decode steps to 1-token | E1 | decode cost |
| F6 | FP8 KV / attention | `--kv-cache-dtype mxfp8` | FP8 QK^T, with the bias via `score_mod` instead of the shear copy. Aims at the 128K global-attention 34 % | **L** | disables the compact verify kernel |

Not valid or no-ops for prefill:

- `--attention-backend triton/flashinfer`: Inkling2 asserts fa4.
- `--moe-runner-backend`, `--fp4/fp8-gemm-runner-backend`, FlashInfer autotune: never reach the
  native SM120 experts.
- `--bf16-gemm-backend gemv|cutedsl`.
- `--enable-dynamic-chunking`: PP only.
- The rel-proj env flags: already on.
- **NVFP4 tile retuning**: measured ≤0.1 % of TTFT.
- Nothing gates the drafter's prefill (<1 %).

### 3.2 Code levers, ranked by measured gain per effort

| # | lever | patch point | gain at 8K (and 128K) | exact? | effort |
|---|---|---|---|---|---|
| C1 | **qkvr column split** for the 28 stacked visits. Permute qkvr weight rows once at load to [q, rel, gate, sinks, k, v]; the q source uses rows 0:6552, k 6552:7576, v 7576:8600 (views). The same change fixes decode (LEVERS #7). | `Inkling2Attention` split_sizes + `_project_sources` (`inkling2_attn.py:139-153, 423-441`), loader permutation | **-190 ms per chunk = -7.9 %** (10.60 → 3.82 ms per stacked visit). 128K: -3.0 s (-5.6 %) | **E0 confirmed**: bit-equal at M=1024/8192/16384 | small |
| C2b | **Rel-bias computed inside the FA4 SM120 kernel** from r (24x16 per token) and the 16x1024 table in shared memory. Removes the einsum, the `[t,24,1024]` tensor, the shear copy and the bias reads | SM120 FA4 mainloop + `inkling2_attn.py` | up to **-404 ms per chunk = -16.8 %** (bias-free FA4 is 1.76 vs 7.67 ms per SWA visit). 128K: up to -11.8 s (-22 %) | E2 unless the 16-term FP32 dot and its bf16 rounding are replicated (then E0) | large |
| C2a | Cheaper subset of C2b: einsum writes the sheared `[round_up(t,128),24,1280]` layout directly (`rel_bias_presheared`, which SM120 currently refuses) | `RelLogitsProj` + SM120 FA4 host | ~-4 ms per visit (the ~0.9 GB shear copy at 224 GB/s): **~-190 ms per chunk, -8 %** (estimate; the bench cannot separate shear from mainloop) | E0 (layout only) | medium |
| C3d | `routed_partial` writes bf16 directly instead of FP32 followed by `.to(bf16)` | `kernels/ops/moe/inkling2_moe.py:369-395`, `inkling2_moe.py:851` | ~-0.6 ms per visit: **~-29 ms per chunk, -1.2 %** | E0 (same FP32 sum, same rounding) | trivial |
| C3a | Quantize each token once (T rows) and gather packed NVFP4/MXFP8 rows, instead of quantizing 8 gathered BF16 copies | `experts_sm120` quantizers | quantize stage 1.87 → ~0.7 ms per visit: **~-53 ms per chunk, -2.2 %** | E0 (row-wise quantization of identical rows) | small-medium |
| C3b | GEMM1 epilogue = capped swiglu + gammas + quantize (drop the BF16 gate_up round trip) | CUTLASS/CuTe epilogue | swiglu-quant stage 1.54 ms per visit: **~-72 ms per chunk, -3 %** | E0 if it keeps "BF16-round the accumulator, then scale" | large |
| C3c | Combine in the GEMM2 epilogue | CUTLASS epilogue | up to -2.2 ms per visit: -105 ms, -4.4 % | E2 (atomic order) unless ordered | large |
| C6 | Fuse the 34 % remainder: attn-res + norm, q/k/rel clips + v_norm + k/v conv (a prefill `attention_prologue`), gate_sink into the FA4 epilogue, loop message mixes | several | the largest pool (815 ms per chunk), but **unattributed until a trace splits it**. If its true traffic matches the 1.2 model, fusion halves it: **~-15-20 %** | E0 if the rounding points are kept | medium-large |
| C7 | Long-context global attention | SM120 FA4 | without bias it already runs at 70 TFLOP/s (76 % of BF16 peak). The remaining exact lever at 128K is C2b. Beyond that only F6 / FP8 attention (lossy) | | |

**Stacking (exact only, 8K).**

| combination | saves per chunk | 8K TTFT |
|---|---:|---|
| C1 + C2b + C3a/b/d + F1 | ~-0.79 s | 2.40 → **~1.6 s (+49 %)** |
| + halving the remainder (C6) | about another -0.4 s | **~1.2 s (2x)** |

For 128K, C1 + C2b + C3 + F1 give ~54 → ~37 s (+45 %).

Reaching the 3.5x "ideal" needs everything except the GEMMs and the attention mainloop fused away.

Lossy tier (optional, not part of the recipe):

- FP8 dense qkvr/wo. The `_scaled_mm` probe failed on N=8600 not being a multiple of 16; it is
  fixed in `bench_tc.py` (8608) and not rerun. Since MXFP8 grouped GEMMs reach only ~102 TFLOP/s
  on GB10, expect well under 2x.
- F6 for 128K.

## 4. Part 3: trace status and recapture

**The 34K trace (`traces/P2-prefill-profile-pf32k`) contains no prefill GPU work.**

- It holds 16 kernels in total, all batch-prep: `alloc_extend`, `write_req_to_token_pool`.
- `scheduler.run_batch` of the prefill shows `_profile_batch_predicate → _stop_profile`. The
  `num_steps` window closed at the launch of the first prefill batch, before its GPU work
  (`profiler_manager.py:446-457`).
- The 136 MB size is 5.1 M `python_function` events from the idle scheduler loop. SGLang
  defaults to `with_stack=True` (`profiler_manager.py:240`).
- The same likely happened at 128K: the window never covered the forward, and Python stack
  tracing slowed the run to 134.8 s.

**Recapture: manual start/stop, no stacks.** Without `num_steps`/`start_step` the profile starts
at once and stops only on `/stop_profile`, which first runs `torch.cuda.synchronize`.

```
# server up, idle, prompt not cached (unique prefix)
curl -s -X POST localhost:8000/start_profile -H 'Content-Type: application/json' \
  -d '{"output_dir":"/root/.cache/prof/pf32k","activities":["CPU","GPU"],"with_stack":false,"record_shapes":false}'
# send the cold 32K request with max_tokens 1 and wait for the response
curl -s -X POST localhost:8000/stop_profile
python3 -I analyze_prefill.py ~/.cache/inkling-2-mini-sglang/prof/pf32k --tokens <prompt tokens>
```

- Repeat for 128K, then for a 1K prompt; the 1K trace splits host time from GPU time.
- Expected size is a few MB: ~1,500 kernels per chunk and no Python events.
- Alternative: `"profile_by_stage": true, "num_steps": 20`. This starts at the first prefill batch
  and stops at the first decode batch after it.
- Neither variant changes clocks or the engine config.

What `analyze_prefill.py` will add from a valid trace:

- the 815 ms-per-chunk remainder split by kernel (attn-res, norms, clips, convs, KV store,
  gate_sink, loop controllers, casts);
- the shear vs FA4 mainloop split, which sizes C2a vs C2b;
- per-chunk graph replay vs eager;
- global attention vs prefix.
