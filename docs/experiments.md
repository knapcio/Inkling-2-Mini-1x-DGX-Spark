# Experiments: engines and levers

Everything that was measured to pick the profile in the README: the SGLang vs vLLM comparison, every lever that
was screened, and what was rejected and why. Raw files are in `results/screen/` (screening) and `results/<label>/`
(full runs).

## Setup

- Two DGX Spark (GB10, 121.6 GiB unified memory, driver 580.178.04, kernel 6.17.0-1032-nvidia), used standalone
  (box A and box B; result directories end in `-s05` / `-s06`); one model server per box, nothing else running. GPU clock capped at 2200 MHz by the
  hosts' own timer (not changed for these runs).
- Model `thinkingmachines/Inkling-2-Mini` revision `ed40e7da` (sha256-checked), bundled DSpark draft.
- SGLang image `lmsysorg/sglang:inkling-2-mini` (`sha256:efac853b…`, arm64 `sha256:bc811ac7…`).
- vLLM image `vllm/vllm-openai:inkling-2-mini` (`sha256:c2dc759f…`, arm64 `sha256:ecdf0ccc…`, vLLM
  `0.1.dev25855+g7cb5dae22`).
- Benchmarks run on the box itself against 127.0.0.1.

### Screening set (`bench/screen.sh`, ~10 min per arm)

1. speedbench 0.3 quick: 30 frozen prompts x 160 tokens, c1, temperature 0, reasoning effort low. Geometric mean of
   decode tok/s over the prompts; the engine counters give tokens per step and step time.
2. Thinking load: 8 concurrent requests, 16 seeded arithmetic puzzles, 2,048 tokens each, effort high,
   temperature 0.8 (`bench/think_load.py`). Aggregate completion tok/s.
3. speedbench longctx at 32K: cold time to first token (prefill tok/s), cached TTFT, decode at 32K context
   (2 documents).

A screen is one run per arm and is only a filter. Differences under ~3 % at c1 (speedbench quick's MDE) and under
~5 % on the sampled thinking load are noise at this stage; winners were confirmed with repeated full runs on both
boxes (below).

## Screening results

One run per arm, `bench/screen.sh`. speedbench quick = geometric mean of decode tok/s over 30 prompts (c1, T=0,
effort low); thinking c8 = aggregate tok/s; prefill / decode at 32K from speedbench longctx (cold TTFT of a
32,702-token prompt; decode over 2 documents).

| arm | box | speedbench quick all (prose) | tokens / step | thinking c8 | 32K prefill tok/s | 32K decode tok/s | KV pool, tokens | MemAvailable after start | start-up |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 vLLM, recipe settings (util 0.80) | B | 42.82 (33.32) | 2.448 | 199.6 | 2,429 | 53.7 | 1,347,359 | 20.0 GiB | 313 s (cold) |
| V2 vLLM + `--max-num-batched-tokens 8192` | B | 41.47 (32.45) | 2.448 | 195.0 | 2,763 | 53.1 | 693,420 | 24.9 GiB | 360 s |
| S1 SGLang, cookbook cell, mem 0.62 | A | 41.86 (33.18) | 2.448 | 189.3 | 3,245 | 51.9 | 844,928 | 39.0 GiB | 378 s (cold) |
| S2 SGLang + `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=1` | A | 41.07 (32.59) | 2.448 | 191.6 | 3,255 | 57.2 | 845,952 | 39.9 GiB | 237 s (warm) |

- vLLM's `--max-num-batched-tokens 8192` lifts 32K prefill by 14 % (vLLM caps scheduled tokens at 2048 with
  speculative decoding) but halves the reported KV pool (the per-request sliding-window and conv blocks grow with
  the scheduled tokens); decode unchanged. Not adopted (vLLM is not the default engine).
- The NV-mega bundle (glue graphs, folded accept tail, deferred draft KV, blockwise rejection) is flat: the step is
  GPU-bound (profile: GPU busy 99.4 %, host gap 0.39 ms per step).

## SGLang vs vLLM, same benchmark, both engines on both boxes

Full runs (`bench/full.sh`), stock recipe settings of each engine (SGLang: cookbook DSpark cell with mem 0.62;
vLLM: official recipe + `--gpu-memory-utilization 0.80`, `--enable-auto-tool-choice --tool-call-parser inkling`),
F1 = SGLang on box A / vLLM on box B, F2 = swapped.

speedbench full (73 prompts x 256 tokens, c1, T=0, effort low), decode tok/s geomean:

| engine | box | prose | code | structured | agentic | multilingual | long context | all 73 | TTFT median |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SGLang | A | 33.0 | 48.5 | 44.9 | 49.9 | 25.6 | 51.5 | **41.09** | 182 ms |
| SGLang | B | 32.3 | 47.5 | 43.8 | 48.7 | 25.0 | 50.4 | **40.16** | 186 ms |
| vLLM | B | 32.8 | 47.9 | 42.4 | 52.7 | 25.0 | 54.4 | **41.11** | 163 ms |
| vLLM | A | 33.0 | 48.2 | 42.7 | 53.2 | 25.2 | 54.8 | **41.38** | 163 ms |

Decode table (T=0, 256 tokens, effort none; c1 per-stream, higher concurrency aggregate):

| | prose c1 / c4 / c16 | code c1 / c4 / c16 | json c1 / c4 / c16 |
|---|---|---|---|
| SGLang 05 | 33.6 / 76.0 / 190.8 | 44.1 / 89.0 / 221.3 | 47.0 / 90.6 / 236.3 |
| SGLang 06 | 32.8 / 74.4 / 181.5 | 43.1 / 85.3 / 221.3 | 46.0 / 88.1 / 228.0 |
| vLLM 06 | 32.6 / 75.6 / 197.1 | 38.5 / 89.7 / 231.2 | 41.1 / 93.0 / 240.8 |
| vLLM 05 | 32.8 / 73.9 / 194.2 | 38.7 / 87.4 / 214.5 | 40.5 / 93.6 / 236.6 |

Thinking load (effort high, T=0.8, aggregate tok/s): SGLang c1 53.3 / 51.4, c8 198.1 / 189.1, c24 380.9 / 368.9;
vLLM c1 54.7 / 54.7, c8 201.8 / 213.7, c24 397.1 / 381.0 (05 / 06 order as above; one run each, sampled).

Cold prefill (speedbench longctx, tok/s; TTFT in brackets):

| engine | 1K | 8K | 32K | 128K | 128K cached TTFT |
|---|---:|---:|---:|---:|---:|
| SGLang (A / B) | 2,190 / 2,137 (0.5 s) | 3,400 / 3,399 (2.4 s) | 3,234 / 3,230 (10.1 s) | 2,419 / 2,426 (54 s) | 27 s (1 of 2 hits) |
| vLLM (B / A) | 644 / 649 (1.6 s) | 1,803 / 1,762 (4.5 s) | 2,437 / 2,427 (13.4 s) | 1,979 / 1,952 (66 s) | 66 s (0 of 2 hits) |

Memory after start-up: SGLang ~39-40 GiB MemAvailable with an 845K-token KV pool; vLLM ~20 GiB with a 1.35M-token
pool (its own suggestion, `--gpu-memory-utilization` unset, would leave the host 0-2 GiB). Start-up: vLLM 5.2-5.6 min,
SGLang 6.2 min cold / 3.7-4 min warm.

**Verdict.** Decode is a tie: c1 within 3 % on both boxes (vLLM +0.7 %/+3 %), c16 and thinking c24 within run
noise, vLLM slightly ahead at high concurrency in some cells and SGLang ahead on code/json c1. SGLang prefills
1.3x (32K) to 3.4x (1K) faster, serves cached long prompts more reliably, and leaves twice the host memory. SGLang is
the base of this recipe; all further work is on SGLang. This agrees with thefullctx's head-to-head (vLLM +5 % on
short c1 decode, SGLang ~2x long prefill, much less memory).

## Where the decode time goes (c1)

Torch profile of the live engine (SGLang `/start_profile`, 40 steps, greedy prose, DSpark on; analysis script and
tables in `docs/profile/`): a c1 step is 60.4 ms = draft 9.7 + target verify 50.5 ms; the GPU is busy 99.4 % of the
wall time (host gap 0.39 ms per step, no graph breaks). Per step: routed MoE 30.2 ms (NVFP4 26.4 + MXFP8 3.8; both
native grouped GEMMs at ~220-230 GB/s, no dequantise or BF16 fallback), BF16 dense projections 11.8 ms, target LM
head 3.8 ms, draft LM head 3.9 ms, draft Markov fold (7 reads of a 103 MB matrix) 3.0 ms, draft layers 2.2 ms,
~1,000 small kernels 4.9 ms.

Bytes per c1 step: dense BF16 2.79 GB (48 layer applications), LM head 0.93 GB, draft ~1.65 GB, MoE u x 0.23 GB
where u = distinct experts touched per layer visit. Measured on the live server (counter in the verify-cap overlay):
u = 28.3 at 8 verify rows (8 independent top-8 draws would give 51.6), 19.2 at 4 rows, 8 at 1 row. Total ~13.4 GB per
step: the measured 60 ms is 1.08x the floor at 240 GB/s. Every kernel is already near the memory bandwidth, so only
fewer bytes per accepted token can make decode faster: verify fewer rows (MoE union), read fewer draft bytes,
read the dense weights in fewer bits (exactly), or accept more tokens per step (a better draft, which needs training
and is out of scope).

## Decode levers (all screens on box A unless noted)

`vscreen`: 16 greedy prompts dumped for a text comparison against the base, speedbench quick (30 x 160, c1, T=0,
effort low), thinking load c8 (16 x 2048, T=0.8, effort high); thinking c1 (4 x 2048) for the combined arms.

| arm | speedbench quick all (prose) | step ms | tokens/step | thinking c8 | thinking c1 | greedy text = base |
|---|---:|---:|---:|---:|---:|---:|
| B0 base | 40.68 (32.26) | 61.4 | 2.448 | 187.8 | - | - |
| B0b base, 2 h later | 40.42 (32.06) | 61.7 | 2.448 | 198.5 | 52.6 | 16/16 |
| verify cap fixed:8 + union counter (inert check) | 39.63 (31.41) | 63.0 | 2.448 | 187.7 | - | 16/16 |
| verify cap fixed:4 + union counter | 41.97 (35.40) | 52.9 | 2.210 | 174.9 | - | 15/16 |
| verify cap fixed:4, remap off (identity check) | - | - | - | - | - | 16/16 vs fixed:4 remap on |
| verify cap v2 fixed:8 fused remap (overhead) | 40.38 (32.01) | 61.9 | 2.448 | 190.9 | - | 16/16 |
| verify cap cost 0.30/0.25/0.15/0 | 46.39 (38.53) | 50.9 | 2.356 | 186.6 | - | 15/16 |
| verify cap fixed:4 only at bs 1 | 43.01 (36.40) | 51.6 | 2.210 | 185.6 | - | 15/16 |
| draft gemv + fold8 + head8 (box B) | 41.96 (33.68) | 59.1 | 2.487 | 187.5 | - | 15/16 |
| draft + MXFP8 draft layers (box B) | 42.44 (34.21) | 58.6 | 2.451 | 192.3 | - | 14/16 |
| MoE tiles (box B) | 40.84 (32.39) | 61.2 | 2.448 | 186.5 | - | 16/16 |
| MoE tiles + QKVR GEMV (box B) | 41.10 (32.56) | 60.7 | 2.448 | 190.8 | - | 16/16 |
| C1: cost 0.30 + draft (no mx8) + tiles + QKVR | 51.22 (42.73) | 46.5 | 2.409 | 189.6 | 57.6 | 15/16 |
| C2: C1 + MXFP8 draft layers | 51.90 (43.57) | 45.8 | 2.356 | 199.9 | 58.7 | 14/16 |
| C3: C2 with cost 0.40/0.30/0.20/0 | 52.14 (44.01) | 44.2 | 2.255 | 192.3 | 59.2 | 13/16 |
| C4: C2 with cost 0.20/0.15/0.10/0 | 50.95 (42.53) | 47.7 | 2.431 | 199.5 | 60.8 | 14/16 |
| C5: C2 with cost 0.50/0.40/0.25/0 | 51.42 (43.35) | 42.4 | 2.106 | - | - | 14/16 |
| C6: C2 + pico12 dense,lmhead + prefill qkvr_split, partial_bf16 | 54.95 (48.06) | 43.1 | 2.239 | 195.7 | 61.1 | 0/16 (diverges in the first 20-370 characters of the reasoning; see below) |

Reading the table:

- C6 changes the floating-point summation order of every dense projection (pico's GEMV accumulates in a different
  order than SGLang's Triton GEMV; its packed weights decode to the same BF16 bits), so every greedy text leaves the
  base's trajectory early, at a word choice inside the reasoning, and continues coherently. Quality is checked with
  the speedbench quality guard and the sanity set in the final runs (README).

- The base is deterministic across boots (B0 and B0b: identical greedy text, speed within 0.6 %).
- Greedy text identity is not the exactness gate for this model: any change of floating-point summation order or
  of the verify block boundaries moves near-tie tokens through the W4A4 activation quantisers (equally correct
  implementations agree on ~95 % of tokens teacher-forced). Exactness is established per lever: bit-identical
  kernel outputs in GPU checks (tiles, QKVR GEMV, prefill levers, packed BF16 == raw BF16), byte-identical text
  between remap on and off for the verify cap (so the dead rows never reach a live row), and the stopping-rule
  argument plus a Monte Carlo test for sampled requests.
- Thresholds: greedy speedbench peaks around 0.40, the sampled thinking c1 runs prefer lower thresholds (4 requests
  each, noisy). 0.30 is the default; `VERIFY_CAP` sets it.
- Offline replay of 18.7K logged c1 steps (`results/steplog/`, logged with a read-only step-log overlay and replayed offline):
  greedy fixed:8 37.6 -> best fixed width (5) 38.8 -> left-to-right cost policy 43.4 tok/s (+15 %; oracle 49.4);
  sampled 50.0 -> 53.3 (+6.6 %). The cost rule in serving (+14 % greedy) captures about what the best replayed
  policy does.

Rejected or parked:

- `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=1`: flat (GPU-bound step).
- Fixed verify caps: lose on thinking at c8 (-7 % at fixed:4); the batch-aware cost rule replaced them.
- Hot-vocabulary draft: from 135K tokens of generated traces a 32K-token list covers only 92.7 % of held-out target
  tokens (94-95 % padded with low ids), and every miss is a forced rejection. Needs a multi-million-token trace
  corpus first.
- Exact 12-bit packed BF16 GEMV as a Triton port of Christopher Owen's packed format (`results/kbench/K3-*`):
  x1.11 on qkvr, x1.16 on the head, below the 1.15x kill line on qkvr; superseded by pico's CUDA kernel.
- `--speculative-num-draft-tokens` other than 8: refused by the trained draft (block size 8).

## Prefill

Where the time goes (one 8,192-token chunk at prefix 0, 2.40 s; composed from GB10 microbenchmarks of every kernel
class, `docs/prefill/`): MoE grouped GEMMs 13 %, MoE glue (activation quantise, SwiGLU-quantise, combine) 13 %,
useful dense GEMMs 10 %, the stacked Q/K/V projection's discarded rows 8 %, attention main loop 5 %, relative
position bias (built as `[t, 24, 1024]`, copied into FA4's sheared layout, then applied) 17 %, other elementwise work
34 %. The kernels themselves run near their peaks (cuBLAS BF16 88-92 TFLOP/s, NVFP4 grouped GEMM 164, FA4 without
bias 70, DRAM 224 GB/s); FA4 with Inkling's bias runs at 13 (sliding window) to 46 (global) TFLOP/s. Measured prefill
equals the ceiling that includes today's memory round trips and is 29-43 % of a GEMM+attention-only ideal.

Exact prefill levers (`overlay/inkling_prefill/`, all bit-identical to stock in the GPU check on real weights,
216/216 comparisons incl. the attention output and LSE for the bias lever):

| arm (speedbench longctx, cold, tok/s) | 1K | 8K | 32K | 128K |
|---|---:|---:|---:|---:|
| base (PF0, box B) | 2,151 | 3,408 | 3,235 | 2,409 |
| qkvr_split + partial_bf16 (PFO2) | 2,336 | 3,781 | 3,558 | 2,537 |
| all four: + bias_layout + quant_once (PFO3) | **2,363** | **4,253** | **3,958** | **2,749** |
| `--chunked-prefill-size 16384` alone (PF1) | 2,160 | 3,399 | 3,310 | 2,493 |

- `qkvr_split`: the looped layers project three different inputs (q, k, v sources) through one stacked 8600x2304
  matrix as 3M rows and keep a third of the output; computing each source only against its own column block cuts
  that projection by 63 % (8K chunk: 10.55 -> 3.77 ms per call). Bit-identical from 128 rows (it is off below).
- `bias_layout`: one Triton kernel computes the relative-position bias directly in FA4's sheared layout.
- `quant_once`: each token's MoE input is quantised to NVFP4 once and gathered to its 8 experts, instead of 8 times.
- `partial_bf16`: the MoE combine writes BF16 directly.
- A suspected stock failure of SGLang's `route_metadata` above ~2,370 routed ids (shared-memory limit) shows up in a
  standalone call but not in serving: cold prompts of 328-774 tokens all answer on the stock server.
- Not adopted: chunk 16K (+2-3 % at 32K-128K, more activation memory), NVFP4 tile retune for prefill (<= 2 % of a
  GEMM, all configs bit-identical), prefill CUDA graphs off and 4K / 32K chunks were not screened (out of time).

## Final confirmation (both boxes)

The shipped configuration (`scripts/launch.sh` defaults) ran `bench/full.sh` once on each box
(`results/FINAL-s06/`, `results/FINAL-s05/`), against the stock engine's full runs on the same boxes
(`results/F2-sglang-base-s06/`, `results/F1-sglang-base-s05/`):

| | stock (B / A) | final (B / A) |
|---|---|---|
| speedbench full, all 73 | 40.16 / 41.09 | 55.51 / 56.06 |
| speedbench prose | 32.3 / 33.0 | 46.4 / 46.8 |
| step / tokens per step (c1) | 61.6 / 60.1 ms, 2.45 | 43.9 / 43.4 ms, 2.48 |
| decode table prose c1 / c4 / c16 | 32.8 / 74.4 / 181.5 ; 33.6 / 76.0 / 190.8 | 47.3 / 90.6 / 188.9 ; 47.8 / 89.6 / 192.3 |
| thinking c1 / c8 / c24 | 51.4 / 189.1 / 368.9 ; 53.3 / 198.1 / 380.9 | 63.9 / 192.8 / 371.8 ; 61.8 / 194.8 / 373.8 |
| cold prefill 1K / 8K / 32K / 128K | 2,137 / 3,399 / 3,230 / 2,426 ; 2,190 / 3,400 / 3,234 / 2,419 | 2,510 / 4,256 / 3,903 / 2,759 ; 2,474 / 4,275 / 3,979 / 2,836 |
| quality guard / sanity / needles 120K + 240K | (guard 30/30 on the base control) / 11/11 / PASS | 30/30 / 11/11 / PASS on both |
| MemAvailable after start, lowest | ~39-40 GiB | 39.8-40.1, lowest 39.1-39.3 GiB |

Where the remaining c1 time goes after the overlays (from the screens): the dense BF16 projections and LM head
(now 12-bit packed), the routed experts of the verified rows, and the draft. Doubling c1 speed would need
more accepted tokens per step (a better or longer-trained draft) or a lossy weight format; neither is in this recipe.

## Round 2 (v2)

Same boxes, same image, same harness. Every lever below was screened alone against a control booted the same day on
the same box (`results/screen-r2/`, one line per arm in `summary.jsonl`); the stack was then confirmed with
`bench/full.sh` on both boxes (`results/V2-s05/`, `results/V2-s06/`). Kernel checks, microbenchmarks and analysis
scripts are in `results/kbench/round2/<lever>/`.

### Where the time goes with the round-1 stack

Torch profile of the shipped round-1 stack (greedy prose, 40 steps each; `results/kbench/round2/residual/` has the
corrected analyser `analyze2.py` and per-class tables, every kernel attributed, GPU idle 0.4-0.6 ms per step):

| ms per step | c1 | c4 | c8 | c24 |
|---|---:|---:|---:|---:|
| step | 39.9 | 85.3 | 117.1 | 171.8 |
| routed experts (grouped GEMMs) | 16.5 | 53.0 | 75.8 | 98.8 |
| dense qkvr / wo (pico at <= 16 rows, cuBLAS above) | 9.5 | 12.9 | 14.8 | 16.7 |
| target LM head | 2.9 | 4.4 | 5.0 | 7.0 |
| verify attention (SWA + global) | 1.2 | 2.3 | 6.8 | 20.6 |
| draft (layers, FP8 head, FP8 fold) | 5.2 | 5.8 | 6.7 | 11.4 |
| norms, attn-res, short conv, loop controllers, MoE glue | 2.9 | 4.4 | 6.0 | 13.9 |

Measured expert unions (stock engine with the read-only counter, 36K logged request-steps,
`results/screen-r2/R2SL-s06/`): per MoE visit U(1) / U(4) / U(8) rows per request = 32.8 / 56.7 / 68.8 at 8 requests
and 51.4 / 79.0 / 91.2 at 16 requests, far from the 128 experts. The round-1 rule verified all 8 rows from 5 requests
up because the union was assumed saturated; it is not.

### Decode levers (screens on box A, same day)

Controls: C0a and C0b (round-1 stack, booted before and after the arms) agree within 1-3 % except the sampled thinking
cells (+-5 %).

| arm | sbquick c1 | thinking c4 / c8 / c24 | prose c4 / c8 / c16 | code c4 / c8 / c16 |
|---|---:|---|---|---|
| C0a control | 54.93 | 126.0 / 196.2 / 374.3 | 92.1 / 121.7 / 195.3 | 103.1 / 148.7 / 225.2 |
| C0b control | 54.97 | 131.4 / 197.8 / 373.4 | 91.8 / 120.2 / 192.1 | 102.8 / 143.6 / 224.7 |
| V1a policy theta 0.04 | 55.58 | 130.8 / 200.3 / 346.1 | 96.9 / 137.4 / 205.3 | 105.4 / 157.0 / 228.6 |
| V1b policy theta 0.04 (repeat) | 55.46 | 133.6 / 188.9 / 360.3 | 96.1 / 135.6 / 202.8 | 105.0 / 152.4 / 230.2 |
| V2a cost:1=0.30,24=0.25 | 55.01 | 133.8 / 191.0 / 359.2 | 94.3 / 134.3 / 202.2 | 107.0 / 152.1 / 233.5 |
| V3a policy, bsmax=12 | 55.55 | 143.5 / 196.9 / 372.2 | 96.4 / 138.8 / 192.1 | - / 154.7 / 231.3 |
| CERT certified greedy head | 56.70 (56.65) | - | 90.2 / 122.9 / 194.5 | 107.1 / 146.2 / 227.9 |
| VAT verify attention | 55.07 | - / 195.9 / 398.0 | 92.1 / 125.8 / 204.0 | 102.8 / 142.7 / 235.2 |
| WID wide GEMM (all forwards) | 54.41 | - / 213.5 / 367.0 | 95.8 / 126.4 / 197.3 | 106.9 / 145.3 / 227.9 |

- **Verify policy** (`inkling_vcap` `policy:` mode). Offline replay on the logged steps (`results/kbench/round2/policy/`)
  priced a calibrated left-to-right rule with one price for all batch sizes at +0.5 / +2 / +5 / +6 % at 1 / 4 / 8 / 16
  requests. Live: greedy decode +5 / +13 / +5 % at c4 / c8 / c16, thinking c4 / c8 +4 / +2 % on average; at 24
  thinking requests both policy runs were below both controls (346 / 360 vs 374 / 373), the bsmax=12 run equal
  (372). Shipped with `bsmax=16`, which keeps the c16 greedy gain and verifies all rows above 16 requests.
- **Certified head**. Serving check mode first (`R2CHK`): 5,670 certified steps, 45,584 rows, 0 mismatches against
  the full head; 216 candidate tiles of 12,564 per step on average. Then the A/B: +3.1 % at c1, nothing above 16 rows
  (not eligible).
- **Verify attention**. Above 32 verify rows the stock path padded q and the bias to 4 heads per KV group, sheared the
  bias to [t, 32, 1280] and ran FA4; the split-KV kernel used at <= 4 requests has no batch-size dependence. The cap
  alone (32 -> 384 rows) gives most of the gain, the in-kernel bias the rest (visit 1.4-1.7x faster at 8-48
  requests). Long-context c1-c4 verify is already at the DRAM floor (222-235 GB/s at 32K-128K).
- **Wide GEMM**. In this screen it also took 17-256-row prefill chunks of short prompts (the c1 text changed:
  2.304 tokens per step vs 2.272 in every other arm); it ships restricted to decode / verify forwards
  (`INKLING_WIDE_SCOPE=verify`), confirmed in the full runs (c1 decode unchanged).

### Prefill levers (box A, same day, speedbench longctx, cold)

| arm | 1K | 8K | 32K | 128K | guard / sanity / needles |
|---|---:|---:|---:|---:|---|
| C0b control (round-1 prefill levers) | 2,522 | 4,265 | 3,972 | 2,805 | - |
| ATT fused relative-bias attention | 2,694 | 4,887 | 4,669 | 3,535 | 30/30, 11/11, 120K + 240K PASS |
| PFA attention + glue fusion | 2,881 | 5,487 | 5,179 | 3,818 | 30/30, 11/11, 120K + 240K PASS |

The 240K needle's prefill took 89 s with ATT vs 133 s on the control. Decode tok/s at 32K / 128K moved in these arms
(68 / 45 control, 56-62 / 39-43 with the levers), but the time per verify step did not (32K 54-56 ms, 128K 66-69 ms in
all three): the two decode documents produce different text when the prefill numerics change, and the accepted tokens
per step range from 2.2 to 5.3 between documents. Prefill budget per 8K chunk before these levers (rebuilt with the
measured 164 TFLOP/s NVFP4 rate, `results/kbench/round2/residual/out/prefill.txt`): elementwise remainder 744 ms
(loop-message mixes 233, attention-residual mixes 143, norms 74, ...), routed-expert GEMMs 317, dense 250, MoE glue 211,
relative bias 219, attention main loop 119.

### KV pool (box B, MemAvailable logged every 5 s from container start)

Worst case per setting: thinking load c24 (48 x 2,048 tokens) together with a cold 240,104-token needle prompt, then
40 min of 28 concurrent mixed requests.

| `MEM_FRACTION` | global-KV tokens | MemAvailable after start | lowest | needle 240K |
|---:|---:|---:|---:|---|
| 0.62 (round 1) | 794K | ~40 GiB | 39.1 GiB (full run) | PASS |
| 0.72 | 985,088 | 29.4 GiB | 26.9 GiB | PASS |
| 0.80 | 1,132,800 | 22.1 GiB | 18.5 GiB | PASS |
| 0.86 | 1,257,856 | 15.6 GiB | 12.1 GiB | PASS |

0.84 is the default: 1.19-1.21M tokens and 17.5-17.9 GiB lowest during the full runs, ~14 GiB expected in the worst
case above. The start-up peak is not lower than the steady state at any setting.

### Not adopted in round 2

- Loop-controller / small-kernel launch fusion and L2 warming: SGLang PR #51926 (fused loop controllers) is already in
  the image. An overlay that issues `prefetch.L2` for the next projection's weights (exact, bit-identical outputs) saves
  nothing: GB10 drops most of these instructions (an 8 MB read takes 45 us cold, ~40 us after an L2 prefetch, 14 us
  after a normal read). A side-stream read-and-discard warm before the consumer moves the same DRAM bytes and contends
  with its tail: -0.13 ms per step at c1, 0 at c24.
- A native layer-stack backend (SGLang keeps the API, scheduler and KV; the backend owns the 48-visit core): after
  crediting the round-2 overlay levers, removing every boundary-specific cost (cross-op fusion to the weight-byte floor,
  launch chaining, eager glue) gives at most 1.13x geometric mean at c4 / c8 / c24 (1.14x at c1), below the 1.20x we
  require before building it (`results/kbench/round2/residual/out/budget.txt`).
- The verify policy above 16 requests (above).
- Long-context decode beyond the DRAM floor: the remaining lever is an FP8 KV cache (~+18-20 % at 128K by bytes), which
  changes the numerics and is outside this recipe's exact default.

### Confirmation (both boxes, `bench/full.sh`)

| | round 1 (B / A) | v2 (A / B) |
|---|---|---|
| speedbench full, all 73 | 55.51 / 56.06 | 58.19 / 58.10 |
| decode table prose c1 / c4 / c8 / c16 | 47.3 / 90.6 / 120.3 / 188.9 ; 47.8 / 89.6 / 122.8 / 192.3 | 47.5 / 98.7 / 148.1 / 213.2 ; 47.5 / 100.6 / 150.2 / 211.0 |
| thinking c1 / c8 / c24 | 63.9 / 192.8 / 371.8 ; 61.8 / 194.8 / 373.8 | 63.5 / 210.7 / 406.7 ; 68.2 / 211.5 / 397.7 |
| cold prefill 1K / 8K / 32K / 128K | 2,510 / 4,256 / 3,903 / 2,759 ; 2,474 / 4,275 / 3,979 / 2,836 | 2,833 / 5,393 / 5,115 / 3,785 ; 2,809 / 5,241 / 5,004 / 3,737 |
| KV pool / MemAvailable lowest | 793,728 / 39.1-39.3 GiB | 1,206,016 ; 1,187,840 / 17.5-17.9 GiB |
| quality guard / sanity / needles 120K + 240K | 30/30 / 11/11 / PASS | 30/30 / 11/11 / PASS on both |

## v2.1: hot-vocabulary draft

The draft's vocabulary-sized reads (draft LM head in FP8 and seven Markov fold rows per step) cost 3.5-8 ms per step
at every batch size in the v2 profile. `hotvocab:N` (round-1 code in `inkling_draftbytes`) restricts the draft to the
first N ids of a ranked list; every target token outside the list is a forced rejection at that position, so the list
must cover the model's real output.

Corpus (`tools/hotvocab/`): 46,479 prompts from 14 open datasets (chat 47 %, code 16 %, math 13 %, tool use 9 %,
structured output 9 %, summaries 7 %; 22 % non-English), speedbench prompts excluded by n-gram containment; 2,311 of
them generated on the recipe's own server (28 concurrent, mixed effort and temperature, exact output token ids):
1.95M tokens. Held-out coverage (split by prompt, `results/kbench/round2/hotvocab/cov19m.txt`):

| N | 32K | 48K | 64K | 96K |
|---|---:|---:|---:|---:|
| all held-out | 94.4 | 97.5 | 98.8 | 99.8 |
| English / Polish / German / Chinese | 96.0 / 90.9 / 86.0 / 91.5 | 98.3 / 96.6 / 92.9 / 96.8 | 99.3 / 98.6 / 96.9 / 98.7 | 99.9 / 99.8 / 99.6 / 99.7 |
| speedbench greedy texts (never ranked on) | 98.0 | 99.1 | 99.6 | 99.8 |

The modelled c1 optimum is flat between 48K and 80K (+5.4 % at 64K); 64K was screened. Same-day A/B
(`results/screen-r2/V21H-*`, `V2C-*`; v2 + `hotvocab:65536,wide8` vs v2):

| box | sbquick | prose c1 / c4 / c8 / c16 | thinking c1 / c4 / c8 / c24 | greedy texts identical |
|---|---|---|---|---|
| box A | 61.08 vs 57.64 | 51.0 / 103.2 / 150.5 / 219.3 vs 47.7 / 100.2 / 145.6 / 216.2 | 68.2 / 157.1 / 220.5 / 408.4 vs 62.3 / 146.9 / 212.4 / 388.8 | 30 / 30 |
| box B | 60.70 vs 57.20 | 50.6 / 102.2 / 149.5 / 218.1 vs 47.4 / 101.2 / 150.4 / 212.7 | 66.8 / 143.3 / 214.7 / 399.7 vs 65.5 / 143.0 / 205.9 / 292.1 (control c24 outlier) | 30 / 30 |

Speedbench quick c1 step 37.5 vs 39.8 ms at equal tokens per step (2.14 median in both; multilingual 1.74 vs 1.73).
`wide8` (in the same arm) only changes calls above 64 rows (c9+ draft head) and is bit-identical to the previous
kernel. Confirmed with `bench/full.sh` on both boxes (CHANGELOG v2.1).

KV page size 16 (thefullctx measured +19 % decode at 15.6K context on their stack), on v2, box B: speedbench quick
57.05 vs 57.20, thinking c8 211.6, decode at 32K / 128K 66.0 / 42.5 (no gain), cold prefill 8K 4,210 vs ~5,300 (the
fused prefill attention covers page 128 only and falls back). Not adopted.

## v2.2: routed-expert GEMM tiles at verify batch sizes

Round 1 tuned the NVFP4 tiles for decode on evenly spread synthetic routing. Replayed with real top-8 routing (16
greedy speedbench texts through a CPU reference of the model, verify blocks of B requests x 8 positions, the verify
cap's dead-row remap applied; unions 52 / 74 / 103 at 4 / 8 / 16 requests, close to the served 57 / 69 / 91), cold L2,
real layer-0 NVFP4 and layer-20 MXFP8 weights: all 12 NVFP4 tile configurations and all 6 MXFP8 tactics give outputs
bit-identical to the shipped choice. The fastest ones: NVFP4 `swap_ab=1, tile_n=32, tile_k=256` at 9-128 tokens per
call (w13 +3.4 to +8 %), MXFP8 CuTe tactic `tile_m=32, tile_n=128` at 512-1536 routed rows (w2 +2 to +19 %). MoE time
per verify step: c4 55.6 -> 54.1, c8 77.8 -> 75.0, c16 112.9 -> 105.8, c24 125.0 -> 123.4 ms
(`results/kbench/round2/moe/`). A multi-row SwiGLU-quantize kernel (bit-identical) saved only 0.13 ms at c24 and was
slower below; not adopted.

Same-day A/B on box A (`results/screen-r2/R3C-s05`, `R3M-s05`, `R3Cb-s05`; v2.1 control / + moe2 / control):

| | control (mean of 2) | moe2 |
|---|---|---|
| decode prose c2 / c4 / c8 / c16 | 68.1 / 101.2 / 150.1 / 219.0 | 69.8 / 107.1 / 153.1 / 231.7 |
| decode code c2 / c4 / c8 / c16 | 80.9 / 114.7 / 167.7 / 242.3 | 83.7 / 119.2 / 168.0 / 255.4 |
| decode json c4 / c8 / c16 | 121.9 / 175.0 / 260.7 | 128.0 / 181.3 / 272.2 |
| thinking c4 / c8 / c24 | 141.4 / 213.6 / 411.8 | 141.1 / 216.3 / 418.7 |
| speedbench quick c1 | 61.07 | 60.93 |

Confirmed with full runs on both boxes (CHANGELOG v2.2). During the box B full run an unrelated CPU compile job
ran on the host for 8 minutes (thinking c24 and long-context cells; MemAvailable down to 4.05 GiB); those cells agree
with box A within run noise and every check passed, but they are the less clean of the two.

## v2.3: scale codes and two c1 byte leaks

**NVFP4 expert scale codes** (`results/kbench/round2/scodes/`). On all 20 NVFP4 layers, 11,794,049 of 11,796,480
expert rows use at most 31 distinct scale bytes; they are stored as a row base byte plus 5-bit codes. The other 2,431
rows (0.02 %) mark the words they cannot code with an escape code, read from the stock scales that stay resident.
Scales 1,274.0 -> 808.1 MB (-4.06 % of NVFP4 expert bytes); encode / decode round trip exact on all 40 tensors. The
patch replaces only the collective mainloop of FlashInfer's SM120 NVFP4 grouped GEMM (CUTLASS): the producer loads a
640-byte code chunk per 128 rows x 128 K instead of the 1,024-byte scale block, the consumer threads expand it into
the same shared-memory scale bytes, and the MMAs run unchanged. 350 / 350 tile comparisons bit-identical to stock on
real weights and routing at 8 / 64 / 192 tokens (layers 0, 6, 17, 19, w13 and w2, all five tiles, escapes forced),
CUDA-graph replay included. Cold-L2 GEMM time -3.5 to -4.5 % per call (layer 17, with the most escape rows, -2.2 to
-3 %).

A/B on box A (`results/screen-r2/R4C-s05`, `R4S-s05`, `R4Cb-s05`):

| | control (2 runs) | scale codes |
|---|---|---|
| speedbench quick c1 | 61.10 / 60.69 | 61.47 |
| decode prose c8 / c16 | 159.7, 153.7 / 227.5, 227.3 | 161.8 / 232.3 |
| decode code / structured / json c8 | 171.6, 171.4 / 167.5, 171.7 / 177.3, 178.8 | 176.4 / 180.4 / 180.2 |
| thinking c8 / c24 | 223.1, 215.2 / 410.6, 316.5 (second run's c24 an outlier) | 225.5 / 427.6 |

**pico stacked q/k/v** (`results/kbench/round2/c1leak/`). At c1 the looped layers' stacked projection has 3 x 8 rows;
the q-source call read all 8,600 coded rows, including the k / v rows it discards, and the k / v call then read a
second coded copy of those rows (7.3 MB x 28 visits = 204 MB per step). `stack1` runs pico's unmodified kernel body
over column ranges of the one coded matrix: 178 -> 144 us per stacked call at 8 rows (layer 0), bit-identical.
**Draft MXFP8 layers**: the o / down shapes (2,304 outputs) streamed at 117-124 GB/s with 32-wide output tiles; 16-wide
tiles reach 198 GB/s per layer, bit-identical (48 / 48 configurations checked).

A/B on box A (`R6C-s05` control, `R6L-s05` stack1 + w8tile): speedbench quick 62.35 / 62.22 vs 60.48 / 61.05
(+2.5 %), decode c1 prose / code / structured / json 52.1 / 61.3 / 68.9 / 61.2 vs 51.1 / 60.3 / 67.8 / 60.0, c4-c16
unchanged (these paths only run at <= 16 rows per source).

**Screened, not adopted**

- Stream-K packed GEMM for 97-192 rows (`R5W-s05` vs `R5C-s05`): 1.21-1.45x cuBLAS on qkvr in isolation, in serving
  speedbench quick 60.15 vs 60.81, thinking c8 / c24 214.1 / 402.5 vs 220.6 / 426.2, decode c8 -1 to -3 %, c16 +1 to
  +3 %. A shared-memory tile design (weights decoded once into shared memory, cp.async pipeline) lost to cuBLAS at
  every row count. The released kernel stays at <= 96 rows.
- Prefill glue round 3 (faster mix-kernel configs, gate + sink in the attention epilogue, K/V conv + clip + norm in
  one kernel; -70 ms per 8K chunk in isolation): `R4P-s06` vs `R4PC-s06` cold prefill 1K / 8K / 32K / 128K
  2,870 / 5,523 / 5,200 / 3,884 vs 2,928 / 5,619 / 5,214 / 3,853 tok/s, flat.

Confirmed with full runs on both boxes (CHANGELOG v2.3).

## v2.4: sampled-path verification, drafter window, packed global KV

Same boxes, same image, same harness. Every lever was screened against controls booted the same day on the same box,
then confirmed on the second box, then the stack ran `bench/full.sh` on both boxes (`results/V24-s05/`,
`results/V24-s06/`). Method change in this round: c1 and long-context A/Bs are judged by step time = tokens per step /
decode tok/s, not by tok/s alone. The same configuration gives 2.224 or 2.194 tokens per step on speedbench quick
depending on what ran earlier in the boot (every arm of the ablation below shows both), so a whole-set tokens-per-step
shift of ~1.4 % is W4A4 text variation, not acceptance.

### Sampled path (thinking load, T=0.8)

Where the sampled step spends its extra time (torch profile, thinking c8 / c24): the max draft probability that the
verify rule reads costs 2.0 ms per step at c8 in its torch form (~8 full-width FP32 passes over [bs x 7, vocabulary]);
the stock sampled accept copies the BF16 draft logits to FP32, runs two softmaxes and materialises the [bs x 8,
vocabulary] target and [bs x 7, vocabulary] draft probabilities, about 1.07 GB and 5 ms per step at c24.

- **Block verification under the cap** (`VCAP_BV=1`). SGLang's `--speculative-use-block-verification` implements
  Sun et al., "Block Verification Accelerates Speculative Decoding" (ICLR 2025, Algorithm 2) for sampled rows. Under
  the verify cap a request verifies only L rows, and the dead rows beyond L carry wrong target rows (expert remap).
  `mask` (default) zeroes the draft probability of every dead draft, which is exactly a draft that stops after L-1
  tokens; over that draft the kernel is the stock fixed-length algorithm, so Sun et al.'s theorem applies directly,
  and dead levels are never scanned. The proof that `P(tau >= i | X_1..X_i) = p_i` depends only on live rows, and the
  Monte Carlo test (joint law of the first three tokens, 300K sequences, garbage dead rows, max error <= 1.2e-3 against
  a 3.5e-3 tolerance; a dead-row leak and a look-ahead cut are both detected) are in `overlay/inkling_vcap/README.md`.
- **FASTPROBS** (`VCAP_FASTPROBS=1`): the max probability and the draft q of the sampled accept from one or two reads
  of the BF16 draft logits with the image's split softmax kernels; the same function up to FP32 summation order.
- **FUSED_ACCEPT** (`VCAP_FUSED=1`): p and q of the drafted tokens from a per-row (max, log-sum-exp), the accept walk
  (chain or block verification) and the bonus sample on [bs, 7] tensors, without the full-width probability tensors;
  from 3 requests up, self-checked against the stock accept at start-up.

Screens (aggregate tok/s; two runs per arm, two controls):

| arm (box) | thinking c1 / c8 / c24 | greedy prose c8 / c16 | code c16 |
|---|---|---|---|
| controls T0c, T0d (box A) | 69.3 / 220.1 / 423.7 ; 69.8 / 221.1 / 415.8 | - | - |
| block verification BVa, BVb (box A) | 69.5 / 227.7 / 429.3 ; 73.8 / 234.7 / 443.4 | - | - |
| BV + draft temperature 0.7 (box A) | 67.2 / 229.1 / 428.6 ; 67.8 / 211.1 / 418.8 | - | - |
| controls FPc, FPcb (box B) | - / 225.3 / 428.8 ; - / 213.2 / 425.2 | 156.3, 158.8 / 232.7, 229.4 | 255.9 / 256.6 |
| FASTPROBS FPa, FPb (box B) | - / 228.3 / 435.6 ; - / 224.6 / 418.9 | 160.8, 159.6 / 235.5, 238.9 | 262.1 / 261.1 |
| FASTPROBS + FUSED_ACCEPT FAa, FAb (box B) | - / 229.2 / 418.9 ; - / 234.0 / 436.0 | - | 269.5 / 266.2 |

Block verification: thinking c1 / c8 / c24 +3.0 / +4.8 / +4.0 %; with a draft temperature of 0.7 the gain is gone.
FASTPROBS: thinking c8 +3.5 %, c24 flat, greedy prose c8 / c16 +1.7 / +2.6 %, code c16 +2 % (the max-probability
path also runs for greedy requests). FUSED_ACCEPT on top: code c16 +2 %, thinking c8 up, c24 within noise.

Confirmation of block verification + FASTPROBS on both boxes (`U2a` vs control `U2c`):

| box | thinking c1 / c8 / c24 | greedy prose c8 / c16 | code c16 |
|---|---|---|---|
| box B | 68.5 / 234.1 / 440.6 vs 70.5 / 226.7 / 426.2 | - / 238.3 vs - / 230.0 | 265.4 vs 255.6 |
| box A | 70.5 / 227.3 / 422.0 vs 68.3 / 234.9 / 418.2 (sampled c8 noisy) | 161.1 / 235.1 vs 155.1 / 229.7 | - |

### Drafter global-attention window (`dwin:16384`)

The DSpark drafter has three layers; layer 1 is global and, in every draft forward, attends over the request's whole
context through the drafter's own KV (4,096 bytes per token: 0.54 GB at 128K, 1.05 GB at 255K). `dwin:W[:S]`
(`overlay/inkling_draftbytes/dwin.py`) gives that layer a page table of the first S tokens (the attention sink,
default 128) plus every page that holds one of the last W tokens; whole pages only, so FA4 keeps its paged path, and
the call is the identity below W + 2 pages (16,640 tokens). Draft-only: the target verifies every proposal as
before.

| box | context | step ms, controls | step ms, `dwin:16384` | tokens per step, dwin vs controls |
|---|---|---:|---:|---|
| box A | 128K | 60.1 / 60.2 | **57.7** | 2.44 vs 2.45 / 2.66 |
| box A | 250K | 80.0 / 79.7 | **76.0** | 2.54 vs 2.48 / 2.47 |
| box B | 128K | 59.5 | **58.1** | unchanged |
| box B | 250K | 80.0 | **75.2** | unchanged |

32K step time is flat (the window saves at most 67 MB per step there).
`dwin:8192` gave a similar step time with fewer tokens per step at 32K.

### Packed 12-bit global-layer KV (`inkling_kv12`)

Long-context decode is at the DRAM floor for the BF16 KV of the 8 global-layer visits (layers 5 / 11 / 17 / 23, each
pass of the looped core with its own cache). On K / V from a CPU reference run of the model, all 16 global tensors (K after the L2 clip, V
after the gain-free RMSNorm) use the same 15 exponents 116..130; 2.8e-4 of the values fall outside
(tiny values and exact zeros). `inkling_kv12` keeps the BF16 pool and stores full, committed pages a second time as
sign + 7-bit mantissa bytes, a 4-bit exponent code (15 = escape) and a sparse escape stream: 193 instead of 256 bytes
per key and head (1.33x fewer). The decode / verify attention is the stock split-KV kernel with the K / V tile load
replaced: a packed tile is decoded in registers, any other tile (open page, unpacked or overflowed) is read raw. Every
write path clears the page's packed flag, so a packed page always equals its raw bytes. Design, invalidation and
CUDA-graph safety: `overlay/inkling_kv12/README.md`.

Exactness: decoded tiles equal the BF16 cache for all 65,536 BF16 patterns, inf / nan, escape-dense rows and real
K / V; attention partials are bit-identical to stock at the stock launch configuration (packed, mixed and raw tiles,
eager and graph-replayed, 60-step verify loop with page reuse).

Microbenchmark, one global verify visit (find / pack steady state + attention + combine), box B, cold, CUDA-graph
medians:

| shape | stock us | kv12 us | speedup |
|---|---:|---:|---:|
| 128K, 1 request x 8 rows, synthetic KV (measured exponent histogram) | 2,296 | 1,809 | 1.27 |
| 128K, 1 x 8, real KV (layer-5 bank, the most escapes) | 2,297 | 1,996 | 1.15 |
| 128K, 4 x 8, synthetic | 9,123 | 7,149 | 1.28 |
| 256K, 1 x 8, synthetic | 4,572 | 3,623 | 1.26 |

Serving screens (box A, c1 step ms from speedbench longctx):

| arm | 128K | 250K | note |
|---|---:|---:|---|
| first version, side pool capped at 3,000 pages by physical page id (mem 0.76) | 59.5 / 59.1 vs 62.4 / 59.7 | 79.3 / 80.2 vs 80.1 / 79.6 | no clear gain: long requests held page ids above the cap |
| every page covered (`MAX_PAGES=0`, mem 0.62, 18 GiB side pool) | 57.5 vs 60.2 | 72.7 vs 79.5 | +4.5 / +9.5 %: the coverage was the cause |
| slot indirection, 2,048 slots per layer (6.1 GiB, mem 0.80), released | **56.8** vs 62.2 | **75.2** vs 80.1 | +9.5 / +6.5 %; 92.6-99.9 % of tiles read packed, 0 overflow |

The released version gives any page a slot from a free stack, then by LRU stealing; 2,048 slots cover one 256K-token
request or the recently decoded pages of many shorter ones. Sanity 11/11 and needles 120K / 240K PASS with it.
`MEM_FRACTION` went from 0.84 to 0.80 to make room for the side pool.

### Same-session ablation (both boxes)

Four arms per box, booted back to back the same day (`results/screen-r2/AB1all-*`, `AB1nokv-*`, `AB1nodw-*`,
`AB1v23-*`; all at mem 0.80). Step ms = tokens per step / speedbench quick tok/s, c1; greedy decode table at c16:

| arm | c1 step ms (box B / box A) | prose c16 (B / A) | code c16 (B / A) |
|---|---|---|---|
| all v2.4 levers | 34.78 / 34.72 | 239.3 / 235.8 | 263.4 / 262.3 |
| without kv12 | 34.78 / 34.75 | 236.8 / 239.0 | 263.6 / 262.0 |
| without dwin | 35.05 / 34.57 | 235.2 / 238.0 | 257.0 / 261.7 |
| all v2.4 levers off (the v2.3 configuration) | 34.56 / 34.77 | 230.3 / 228.0 | 260.4 / 256.7 |

The levers cost nothing at c1 (+0.6 / -0.1 %, within noise); with all of them on, greedy c16 is +3.9 / +3.4 % on
prose and +1.2 / +2.2 % on code.

### Confirmation (both boxes, `bench/full.sh`)

| | v2.3 (A / B) | v2.4 (A / B) |
|---|---|---|
| speedbench full, all 73 | 64.88 / 64.54 | 63.88 / 64.28 |
| tokens per step / step ms (speedbench, c1) | 2.383 / 36.94 ; 2.383 / 37.14 | 2.383 / 37.51 ; 2.383 / 37.25 |
| decode table prose c1 / c4 / c8 / c16 (median of both) | 53.4 / 110.0 / 156.5 / 232.9 | 52.9 / 107.3 / 159.6 / 236.4 |
| decode table code / json c16 (median of both) | 256.0 / 268.3 | 261.3 / 275.4 |
| thinking c1 / c8 / c24 | 68.1 / 223.6 / 423.0 ; 71.5 / 249.4 / 432.9 | 68.5 / 243.4 / 444.6 ; 68.0 / 236.5 / 434.1 |
| cold prefill 1K / 8K / 32K / 128K | 2,933 / 5,238 / 4,926 / 3,694 ; 2,896 / 5,466 / 5,146 / 3,808 | 2,913 / 5,452 / 5,145 / 3,799 ; 2,965 / 5,461 / 5,152 / 3,811 |
| decode at 128K | 42.8 / 43.4 | 48.2 / 48.2 |
| decode at 32K (256 tokens x 2 documents) | 72.0 / 67.2 | 63.7 / 62.9 |
| KV pool / MemAvailable lowest | 1,191,936 ; 1,175,808 / 17.8-18.2 GiB | 1,113,984 ; 1,117,184 / 15.6-16.0 GiB |
| quality guard / sanity / needles 120K + 240K | 30/30 / 11/11 / PASS on both | 30/30 / 11/11 / PASS on both |

The full-run speedbench is 0.9 % below v2.3 at identical tokens per step. The same-session ablation shows that this is
run-to-run variation between boots, not the levers. Decode at 32K reads 63.3 vs 69.6 (median of both boxes); these
cells are 256 tokens x 2 documents with a confidence interval of about +-25 %. A same-session 32K ablation on box A
(`results/screen-r2/AB32*`: 4 reps x cold + cached document, 512 tokens each, step time from the engine metrics) settles it:

| arm | step ms (median of 8) | tokens per step (median) | decode tok/s |
|---|---:|---:|---:|
| v2.4 | 47.2 | 3.23 | 69.2 |
| no drafter window | 47.7 | 3.49 | 75.3 |
| no kv12 | 47.5 | 3.24 | 67.1 |
| all four levers off (v2.3) | 47.3 | 3.16 | 71.6 |

The step time is the same in all four arms. Tokens per step vary with the salted documents' generated texts (2.3
to 5.5 per request). The v2.5 window screen below measures the effect on acceptance with more requests per cell.

Long context (`results/V24-s06/K/`, `results/V24-s05/K/`): at 200K / 250K box B decodes 44.1 / 38.9 tok/s at c1
(v2.3 on box A: 38.1 / 35.1), cold prefill 3,142 / 2,799 tok/s, TTFT 65.4 / 91.2 s; box A's 40.9 / 46.7 come
from one cold and one cached request each and have confidence intervals too wide to use. Worst case for memory
(thinking c24, 48 x 2,048 tokens, together with a cold 240,104-token needle prompt): lowest MemAvailable 14.8 GiB
(box B) / 15.6 GiB (box A), KV pool 1,121,792 / 1,121,280 tokens, thinking 308.2 / 309.6 tok/s alongside the
prefill, needle PASS on both.

### Not adopted in v2.4

- Draft sampling temperature below the request's (box B): tau 0.7 66.5 / 218.7 / 417.6 and 67.8 / 223.6 / 415.0,
  tau 0.8 70.0 / 216.1 / 413.2 vs controls 69.9 / 240.4 / 415.6 and 68.3 / 225.6 / 425.0 (thinking c1 / c8 / c24): no
  gain or worse, also on top of block verification (above).
- 4-bit draft MLP: speedbench quick 64.12 / 64.56 vs 63.84 / 63.65 (+0.9 %) and decode c1 +1-3 % on box A, the same
  pattern on box B, but the gain is tokens per step only (2.225 vs 2.194); the step time is unchanged (box B
  34.70 / 34.86 vs 34.69 / 34.75 ms, box A 34.72 / 34.47 vs 34.39 / 34.47 ms). The same configuration moves between
  2.224 and 2.194 tokens per step from boot to boot, and an offline acceptance check on 1.6M tokens gives -0.17 % with a
  confidence interval that includes 0.
- Late kill of doomed verify rows from a mid-stack LM-head readout (after layer visit 17 / 29 / 41 of 48): on T=0
  traces the readouts barely separate rows whose draft will be rejected (AUC 0.59-0.74, recall <= 1 % at 90 %
  precision), so no rows can be dropped early.
- Shared-prefix KV reads (one read of a common prefix for concurrent requests): L2 already absorbs the reuse.
- Fused routed-expert epilogue: already fused on GB10.
- Streaming NVFP4 MoE kernel: bit-identical and 2-5 % faster than the stock kernel per call, but not faster than the
  shipped scale-code kernel (+0.14 / +0.53 / +1.63 ms per step at c1 / c8 / c24).
- Centroid-clustered draft vocabulary: at most ~0.2 ms per c1 step saved in the byte model.
- Verify-policy retunes on v2.3 (both boxes): theta 0.03 speedbench quick -1.4 %, theta 0.06 +0.2 to +0.9 % (noise),
  the policy up to 48 requests with theta 0.02 above 16 no gain at thinking c24 (417.5 / 413.5 vs 419-423); decode
  c4-c16 within +-3 %. theta 0.04 and `bsmax=16` stay.
- 80K hot vocabulary: held-out coverage 99.5 vs 98.6 % at 64K, paid for with 25 % more draft head and fold rows.

## v2.5: draft Markov walk, one-pass checkpoint load, full draft context

Same boxes, image and harness. Full runs with the release defaults (`results/V25-s05/`, `results/V25-s06/`) use
`rnnhoist`, `LOAD_NOMMAP=1`, `MEM_FRACTION=0.81` and full-context draft attention.

### Draft Markov walk (`rnnhoist`)

Each of the seven steps projects [state, previous-token embedding, hidden_s, f_rnn] through `rnn_proj`. The
`hidden_s` of every step and `f_rnn` are known before the walk (4,608 of the 5,120 input columns). Compute their
GEMM contribution in FP32 up front. Each step runs the 768x512 slice for the state and previous-token embedding,
adds the precomputed partial in FP32, rounds once to BF16 as stock does, and calls the stock gates. Against FP64,
99.9 % of projection elements round correctly (stock cuBLAS 62 % at 8-16 rows). The target distribution is unchanged.

| rows | stock walk us | `rnnhoist` walk us |
|---:|---:|---:|
| 1 | 584 | 320 |
| 8 | 661 | 454 |
| 16 | 782 | 596 |

Seven steps, CUDA graph, cold L2. Same-session serving (`sbquick`, c1 step ms = tokens per step / tok/s):

| box | `rnnhoist` step ms | control step ms |
|---|---|---|
| box B | 34.46 / 34.55 | 34.79 / 34.82 / 35.12 |
| box A | 34.53 / 34.78 / 34.67 / 34.59 | 34.61 / 34.65 / 35.04 / 34.90 |

Our implementation follows the walk-fusion idea of the one-kernel DSpark Markov walk in
[vLLM PR #62958](https://github.com/vllm-project/vllm/pull/62958)
([tie-pilot-qxw/dspark-markov-walk](https://github.com/tie-pilot-qxw/dspark-markov-walk), MIT).

### One-pass checkpoint load (`LOAD_NOMMAP=1`)

`--weight-loader-disable-mmap` is SGLang's one-pass safetensors read. In the screen, warm start-up went from
263-268 s to 131-137 s and target weight load from 160 s to 22-34 s. With the flag alone the KV pool lost 6 %.
`MALLOC_MMAP_THRESHOLD_=1048576` lets large frees return to the OS before SGLang sizes the pool, leaving a 2.6 %
loss at the same fraction. The default `MEM_FRACTION` goes from 0.80 to 0.81. The load strategy follows sxuff's
127 -> 45 s warm start with vLLM's `--safetensors-load-strategy eager` on this checkpoint, found with 0xSero's
inference-optimisation skill (CREDITS.md).

### Drafter global-attention window off by default

Both boxes, 2-3 rounds of on/off, 6-24 requests of 512 tokens per cell with salted documents
(`results/screen-r2/DW*`, `DL*`, `AB32*`). The v2.4 screen judged tokens per step from 1-2 requests per cell.

| context | tokens per step, `dwin:16384` / off | step ms, `dwin:16384` / off | decode tok/s change |
|---|---:|---:|---:|
| 32K | 3.16 / 3.54 | 45.6 / 45.6 | -11 % |
| 64K | 3.10 / 3.34 | 48.6 / 50.3 | -7 % |
| 128K | 2.88 / 3.09 | 55.7 / 57.8 | -3 % |
| 250K | 2.75 / 2.75 | 69.6 / 74.6 | +7 % |

The draft's global layer loses far-context copying with the 16K window. At 32K-128K the drop in tokens per step
outweighs the time saved per step. With `dwin:65536`, decode equals off through 64K, is -2 % at 128K and +3 % at
250K. The default uses the whole context. `DRAFT_BYTES=...,dwin:16384` remains an option for workloads mostly above
200K tokens.

### Confirmation (both boxes, `bench/full.sh`)

Medians of both boxes except where noted. Tables from `python3 bench/report.py results/V25-s05 results/V25-s06`
and `python3 bench/report.py results/V24-s05 results/V24-s06`.

| | v2.4 | v2.5 |
|---|---|---|
| speedbench full, all 73 | 64.1 | 64.2 |
| tokens per step / step ms (speedbench, c1) | 2.383 / 37.4 | 2.383 / 37.1 |
| decode table prose c1 / c4 / c8 / c16 | 52.9 / 107.3 / 159.6 / 236.4 | 53.0 / 107.7 / 161.9 / 238.9 |
| decode table code / json c16 | 261.3 / 275.4 | 260.4 / 277.1 |
| thinking c1 / c8 / c24 | 68.2 / 239.9 / 439.4 | 70.5 / 235.6 / 445.2 |
| cold prefill 1K / 8K / 32K / 128K | 2,939 / 5,456 / 5,148 / 3,805 | 2,893 / 5,464 / 5,143 / 3,806 |
| decode at 32K / 128K | 63.3 / 48.2 | 71.0 / 46.4 |
| KV pool (box A / box B) | 1,113,984 / 1,117,184 | 1,105,792 / 1,106,816 |
| lowest MemAvailable in the full runs | 15.6-16.0 GiB | 16.1-16.3 GiB |
| quality guard / sanity / needles 120K + 240K | 30/30 / 11/11 / PASS on both | 30/30 / 11/11 / PASS on both |

Warm start-up was 137 / 131 s (box A / box B), with 16.5-16.6 GiB MemAvailable after start. The global-layer
KV pool ranges from 1.11M to 1.13M tokens across boots. On box B (`results/V25-s06/K/`), c1 at 200K / 250K
decoded at 43.8 / 37.9 tok/s and cold prefill reached 3,105 / 2,655 tok/s. With thinking c24 (48 x 2,048 tokens)
alongside a cold 240K-token needle prefill, MemAvailable reached 14.5 GiB, thinking ran at 309.2 tok/s, and the
needle passed.

### Screened and not adopted

- **Humming MoE** ([vllm-project/humming](https://github.com/vllm-project/humming) 0.1.20, Apache-2.0, vLLM
  `--moe-backend humming`): same W4A4 block-scaled MMA, 0 mismatches in 102 comparisons against FlashInfer on real
  weights and routing. Per call 1.5-2.4 % faster than stock FlashInfer, generally 0-3 % slower than scale codes,
  at 223-234 GB/s. Layer 0, cold L2, Humming vs scale codes: c1 w13 471 vs 464 us, c1 w2 243 vs 236 us, c8 w13
  1,259 vs 1,223 us, c24 w2 835 vs 843-864 us. The tuning sweep was within ~1 % of scale codes.
- **INT8 per-row draft weights** (sxuff's format, in place of FP8 / MXFP8): 2-3x lower weight error (head 0.94
  vs 2.64 % relative error). On 12,358 real draft blocks the best combination gave +0.0015 tokens per step
  (95 % CI -0.0048 to +0.0076), and BF16 gave -0.0028. The byte count equals FP8, leaving no byte-driven step-time gain.
- **Context n-gram proposals spliced into DSpark** (prompt lookup / suffix decoding): offline replay at full width
  with the rule fixed in advance gave +0.046 tokens per step overall (CI +0.01 to +0.09) and -0.04 on code. The
  best-of-both oracle gave +0.17. The rule fell below our +0.15 bar and was not built.
- **Extra draft vocabulary rows per step** ([vLLM PR #67238](https://github.com/vllm-project/vllm/pull/67238)):
  only 3.1 % of rejections are hot-vocabulary misses, bounding the gain at +0.058 tokens per step. Not built.

### Other recipes, same harness

[sxuff/inkling-2-mini-spark](https://github.com/sxuff/inkling-2-mini-spark) (vLLM + DSpark + INT8 draft,
`serve.sh q8`, pinned image) and our v2.4 recipe ran on box B the same day
(`results/screen-r2/SXQ8-s06`, `results/screen-r2/SXOUR-s06`).

| | sxuff q8 | our v2.4 |
|---|---|---|
| sxuff `bench.py` median tok/s (code / prose / math) | 65.45 (72.1 / 44.8 / 65.5) | 67.77 (67.8 / 63.6 / 77.7) |
| speedbench quick tok/s | 45.84 / 45.95 | 63.10 / 63.05 |
| speedbench quick step ms / tokens per step | 54.6 / 2.41 | 36.5 / 2.19 |
| decode prose c1 / c4 / c8 / c16 | 36.1 / 81.4 / 123.2 / 133.6 | 52.7 / 107.9 / 156.1 / 239.3 |

`bench.py` sends 9 greedy requests x 160 tokens at effort low. The 65.45 tok/s median reproduces the published
66.72 within 2 %. The q8 recipe's `max-num-seqs` of 8 caps the c16 cell. Thinking c1 / c8 on q8 was 59.6 /
216.2 tok/s.

## v2.6: KV pool sizing

With the radix cache on, budget one 1,024-token window plus one 128-token page per running request and two
8,192-token prefill chunks in flight. At 48 requests that is `48 x (1,024 + 128) + 2 x 8,192 = 71,680` tokens.
The image's 0.1 ratio gave 110,592 sliding-window tokens from a 1.1M-token global-layer pool (16.9 GB target +
0.84 GB draft in the sliding-window pool), with ~35K tokens never needed. `SWA_RATIO=0.05`
(`--swa-full-tokens-ratio 0.05`) gives 75-79K sliding-window tokens in the larger pools.

The stress run pairs thinking c24 (48 x 2,048 tokens) with a cold 240,104-token needle prefill. MemAvailable was
logged every 5 s from container start. The sliding-window ratio was 0.1 at 0.81, 0.055 at 0.86 and 0.88, and 0.05
from 0.91 up. The pairs below are box A / box B. Raw files are in `results/screen-r2/`: `KVa-s0*` (0.86), `KVb-s0*` (0.88), `KVd-s0*` (0.91), `KVf093-s0*`
(0.93) and `KVg-s0*` (0.945), plus `results/V25-s05/`, `results/V25-s06/`, `results/V26-s05/` and `results/V26-s06/`. The 0.94 stress
runs are in the `K/` subdirectories of the v2.6 runs.

| `MEM_FRACTION` | global-layer pool tokens | MemAvailable after start (GiB) | lowest under stress (GiB) | needle (both boxes) |
|---|---:|---:|---:|---|
| 0.81 (v2.5) | 1,105,792 / 1,106,816 | 16.6 / 16.5 | 14.1 / 14.5 | PASS |
| 0.86 | 1,376,256 / 1,380,992 | 13.5 / 12.9 | 12.1 / 12.1 | PASS |
| 0.88 | 1,424,256 / 1,422,848 | 11.0 / 10.3 | 10.2 / 10.1 | PASS |
| 0.91 | 1,513,344 / 1,508,352 | 7.4 / 8.0 | 6.8 / 7.9 | PASS |
| **0.93 (v2.6)** | **1,551,104 / 1,557,120** | **5.8 / 6.1** | **5.2 / 5.4** | PASS |
| 0.94 | 1,568,384-1,568,896 / 1,582,720-1,587,840 | 5.0-5.3 / 4.3-4.6 | 4.9 / 3.7 | PASS |
| 0.945 | 1,590,656 / 1,588,224 | 4.8 / 4.7 | 4.8 / 4.0 | PASS |

Throughput was flat across the sweep. Thinking c24 alongside the prefill ran at 297-314 tok/s and thinking c8
at 221-238 tok/s. At 200K / 250K, c1 decode was 40-47 / 34-40 tok/s, varying with the generated text. The full
`bench/full.sh` speed and quality runs used `MEM_FRACTION=0.94` on both boxes.

Other GB10 memory-cliff measurements put the worst-case target at 3.5 GiB MemAvailable. At 0.94, box B had
only 0.2 GiB of margin, and MemAvailable after start varied by up to ~1 GiB between boots at the same fraction.
The release uses 0.93 for that spread, with a 3.5 GiB memory guard floor and 40 % more global-layer KV than v2.5.
Both pool sizing changes preserve exactness.

### Screened on v2.5

- **INT8 draft weights.** The speedbench quick differences tracked the generated text. Thinking and decode
  throughput were flat, so the INT8 variant was dropped.
- **Small kernels.** Router / wproj split-K and controller re-tiling saved 0.35 ms per c1 step in the microbench.
  Serving was flat on box A and box B, so the kernel variant was dropped. The screen drew on Thien Tran's
  [vLLM PR #69168](https://github.com/vllm-project/vllm/pull/69168).

### Future work

A packed-only global KV store is in development. The design uses ~24.6 KB per token vs 32 KB for BF16, making
the lossless packed format the primary store for committed global-layer pages. The serving check packed 15.5M
page-heads with 0 escape overflows.

## v2.7: wider certified head, long-prefix prefill, 512K / 1M, packed KV store

Certified head for 17-192 rows (GPU check on real hidden states, every row band certified, FP64 error inside the
budget; cold timing head + softcap + argmax vs certified): 64 rows 3.99 -> 1.72 ms, 128 rows 5.09 -> 3.19 ms,
192 rows 6.14 -> 6.02 ms (the screen is compute-bound there); overflow fallback costs +1.5 to +5.7 ms but did not
trigger on real states. Serving A/B on v2.6 (`results/screen-r2/CW*`), greedy decode table, mean over four prompt
types, on vs off: c1 +0.3 / +0.1 %, c4 +2.1 / +1.6 %, c8 +1.0 / +1.5 %, c16 -0.3 / -0.7 % (box A / box B).

Long-prefix prefill: 512K / 1M cold prefill timings fitted the fused kernel slowing down, not a fallback. Per global
layer and 8K chunk (cold L2, box B): 4-warp kernel 70.7 TF/s at 120-256K, 49.2 at 512K, 46.5 at 1M; 8-warp variant
68.4-68.9 TF/s at every prefix. Splitting the launch at 131,072 keys: 512K 1,082 -> 777 ms, 1M 2,281 -> 1,547 ms,
bit-identical to the single launch.

512K / 1M option (v2.6 build, default store, box A): see the README table; 1M cold prefill 1,285 s, 512K 358 s.

Memory flags (`results/screen-r2/K1a-s05`): conv-state slots 1,232 -> 512 (peak use 76), one prefill chunk in flight:
pool 1,559,424 -> 1,593,088 tokens, worst case 5.2 GiB, sbquick 63.56, prefill 1K-128K 2,962 / 5,490 / 5,163 / 3,808.

Packed KV store (`KV12S=1`, opt-in): format and checks in `overlay/inkling_kv12s/`; first serving build 2,168,320
tokens, speedbench quick 63.16 (v2.6 63.49), sanity 11/11, needles PASS, worst run 4.2 GiB. Greedy speedbench quick
hashes: a v2.7 reference with `KV12S=0` and the same flags matches v2.6 on 32/32; `KV12S=1` on 27/32 (5 deterministic
differences). Ruled out so far: the eager draft commit (32/32 with `KV12S=0`), FA4 split counts depending on the
cache extent (bit-identical across extents). Resolved in v2.7.1 (below): the differences come from SGLang's unified
memory pool, not from the packed store.

Screened and not adopted (v2.6 base):

- Verify policy refit with top-2-gap features on a fresh full-width capture (65,088 request-steps): -0.22 %
  [-0.52, +0.71] vs the shipped policy.
- Split-K / re-tiled small kernels for the router, global conv projection and loop controllers (after vLLM PR
  #69168, Thien Tran): -0.35 ms per c1 step in isolation, flat in serving on both boxes.
- INT8 per-row draft weights: flat (same bytes as FP8, acceptance unchanged).
- Dropping the 12-bit side copy (`KV12=0`, MEM 0.99): +8 % KV tokens but decode steps 63.7 / 80.0 ms vs 57.8 / 73.2 ms
  at 128K / 250K.

## v2.7.1: packed global KV store as the default

Results: `results/Q271R-s05/`, `results/Q271R-s06/` (full runs), `results/Q271RS-s05/` (worst case, 200K / 250K).

Exactness. A server with `KV12S=1` gives bit-identical greedy speedbench quick texts and logprobs to stock SGLang
with `--enable-unified-memory` and the BF16 store; each of our overlays toggled off left the outputs unchanged. The
packed store needs the unified pool, and stock SGLang's unified pool and its default hybrid pool differ on 5 of 32
greedy texts. Checks on that difference:

- At the first verify step after prefill, a dump of every attention and short-conv layer's inputs and outputs (row 0)
  was bitwise identical between the two pools on 3 prompts (96 / 96 calls), and the teacher-forced extend of the same
  position was identical across pools (240 / 240 calls).
- 24 prompts (18 synthetic of 32 to 12.6K tokens spread over prompt length mod 128, plus 6 speedbench prompts):
  chosen-token logprob error against a teacher-forced prefill at the first verify positions ~0 on both pools (mean
  0.0000 / 0.0010 default, 0.0000 / 0.0007 unified), noise floor at positions 8-40 0.034 / 0.026.
- Teacher-forced re-scoring of whole texts: 0.051 (default) vs 0.049 (unified) nat/token.

Neither pool is closer to an exact prefill. v2.7.2 traced the difference to prefill CUDA graphs (below).

Sampled throughput. The first packed build lost 22 % of thinking throughput at c24 on the unified pool. Each token's
sliding-window band entry held all 40 layer visits, so one layer's 1,024-key window touched ~80 GPU pages per request
and missed the TLB after the MoE weight reads; a probe measured 1,170 vs 468 us per call. Laying the band out
page-major (each visit's rows of a 128-token page together) with a Triton store kernel keeps the same bytes and
greedy hashes (32 / 32) and removed the loss. What remains at c24 (436.9 vs 451.9 tok/s in the v2.6 full runs) is the
draft commit running outside the CUDA graph.

Memory. Fixed buffers are charged to the KV budget before the pool is sized: the prefill scratch (2,112 pages or
`ceil(CTX/128) + 65`, whichever is larger), the overflow arena (2,048 heads per visit) and the fallback pool (64 pages
per visit). An earlier build grew the scratch per prefill chunk, and the allocator kept every old copy; that pushed
MemAvailable through the guard at `MEM_FRACTION=0.91`. `SWA_RATIO=0.035` leaves 79,232 sliding-window tokens; thinking
c48 and the worst case ran clean with it.

Side-slot exhaustion. Forced on purpose (4 overflow heads, 1 fallback page, every page-head sent through overflow): a
~40K-token prompt was refused with HTTP 503, the server kept serving, short greedy prompts before and after matched
the normal boot exactly, and 8 concurrent requests all completed. In all normal runs the high water was 0 overflow
heads and 0 fallback pages.

Retraction. 48 concurrent greedy requests with 63K-token prompts sharing a 16K radix prefix completed 48 / 48 under
forced retraction; re-scored against prefill, the texts averaged 0.0057 nat/token.

Logprobs. The certified heads step aside for any batch that asks for logprobs, so clients get the full head's values.
Runner-up logprobs from a verify step differ from a teacher-forced prefill by ~0.1-0.2 nat above -3 and 1-2 nats
around -15 to -20, on both pools: the verify and prefill forwards round BF16 differently.

## v2.7.2: packed store on the stock pools, dead-row expert skip

Results: `results/Q272R-s05/`, `results/Q272R-s06/` (full runs), `results/Q272S-s05/` (worst case, 200K / 250K).

Where v2.7.1's 5 / 32 came from. The packed store on SGLang's static pools with prefill CUDA graphs gives greedy
speedbench quick texts identical to the BF16 store on 32 / 32 prompts (two runs). The same tree with
`--cuda-graph-backend-prefill=disabled` gives 27 / 32, and the 5 prompts are exactly v2.7.1's (agentic-07, longctx-02,
longctx-07 in both the measured and sentinel slots, structured-02). The captured prefill graphs run each chunk padded
to its bucket size; eager prefill runs it unpadded, which changes the floating-point order. The unified storage and
the eager draft commit play no part.

Captured prefill over a packed prefix. The first static build captured prefill graphs up to 8,192 rows, so every 8K
chunk of a long prompt read its prefix with the kernel's packed loader instead of the BF16 scratch: a cold 240K needle
alongside thinking c24 took 189.9 s (thinking 224.7 tok/s) against 86.5 s (309.0) on the BF16 store. The release
gathers the prefix into the fixed scratch on the device inside the graph (fixed-shape inputs and a fits flag; two
launches into one output, each gated to zero query rows when it does not apply). Both readers replay bitwise against
the BF16 kernel, including a single graph replayed across fits -> overflow -> fits. After the fix: needle 92.5 s,
thinking 304.6 tok/s; cold prefill at parity with v2.7.1.

Sizing. Stock SGLang charges the DSpark draft at three full lanes per global token, but allocates its two
sliding-window lanes at window size. Charging those lanes per window token (startup refuses if the draft is not one
global plus two window layers) and `MEM_FRACTION=0.89` gives 2.13-2.17M global-layer tokens with 5.4 GiB at worst.

Dead-row expert skip (`inkling_noexp`). Paired screens on the release tree, two boxes (ABBA, then a component split
on box B):

| arm | prose c4 / c8 / c16 | code c16 | thinking c24 |
|---|---|---:|---:|
| control (2 runs) | 110.2 / 160.1 / 234.2 | 262.7 | 438.1 (3 runs) |
| NOEXP | 113.3 / 168.1 / 246.6 | 276.2 | 438.1 |
| quant-once + controller-mix launch | 112.4 / 159.7 / 234.3 | 268.0 | 438.6 |
| all four candidate overlays (2 runs) | 112.4 / 164.8 / 247.8 | 272.6 | - |

Box A confirmed the bundle against its own controls: prose c4 / c8 / c16 +3.5 / +2.3 / +5.8 %. Thinking at c4
looked 3 % lower in aggregate but its per-request decode was 1.8 % higher (8 requests in two waves; the wall time
follows the slowest request). NOEXP shipped; the quant-once and controller-mix launch configs (bit-identical) and the
pfuse mix launch config (+0.7 % prefill, different FP32 order on long prompts) did not.

## v2.8: DFlash2 drafter as the default

Basic tests on one box; full two-box qualification follows in v2.8.x. Results: `results/V28sbf-s05/` (speedbench full),
`results/V28M1v28-s05/` (release defaults), `results/V28M1off-s05/` (`DFCAP=off`), `results/V28M1nonx-s05/` (NOEXP off), `results/V28lc-s05/` and
`results/R272lc-s05/` (long context, v2.8 and v2.7.2 on the same box), `results/V28dsp-s05/` (`SPEC=dspark`), all on
box A; `results/V28kv-s06/` (memory and worst case) on box B.

Drafter: [incoai/Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) at revision
`4f9088ea`, a 5-layer block-diffusion draft over a 2,048-token sliding window of target features (layer visits 5,
23, 41, 48), 16 candidates per position and a selector that traces one path (block size 8, 7 drafts per verify step).
It runs on SGLang's DFLASH worker after the drafter's compatibility patch, applied inside the serving container only.

Development measurement, stock against stock (`OVERLAY=0`, DFlash2 vs DSpark, same harness):

| | stock DSpark | stock DFlash2 | ratio |
|---|---:|---:|---:|
| tokens per verify step | - | - | x1.25 |
| prose decode c1 / c4 / c8 / c16, tok/s | 32.4 / 74.8 / 115.2 / 183.3 | 46.2 / 109.4 / 161.4 / 254.8 | x1.43 / x1.46 / x1.40 / x1.39 |
| code decode, c1-c16 | - | - | x1.22-1.25 |
| thinking c8, aggregate tok/s | 187.2 | 223.1 | x1.19 |

Long context, release stack (box A, speedbench longctx, 512 decode tokens, c1; v2.7.2 control on the same box, 3
reps; v2.8 two passes, the second with 2 reps; confidence intervals in brackets):

| | v2.7.2 (DSpark) | v2.8, first pass | v2.8, second pass |
|---|---:|---:|---:|
| decode at 32K, tok/s | 71.9 [58.7, 88.1] | 63.1 [57.7, 69.0] | 59.9 [53.9, 66.5] |
| tokens per step / step ms at 32K | 3.36 / 47.4 | 3.10 / 48.3 | - |
| decode at 128K, tok/s | 49.0 [41.4, 57.9] | 50.2 [38.1, 66.1] | 49.7 [39.5, 62.5] |
| tokens per step / step ms at 128K | 3.20 / 60.8 | 2.68 / 59.0 | - |
| cold prefill 32K / 128K, tok/s | 5,162 / 3,719 | 5,132 / 3,714 | 5,083 / 3,738 |

At 32K context the DFlash2 profile decodes about 12 % slower than v2.7.2's DSpark profile at c1 (60-63 vs 72 tok/s, same box). Its acceptance falls with context (3.10 vs 3.36 tokens per step at 32K), because the drafter sees only a 2,048-token window. At 128K both decode at about 50 tok/s. For mostly long-context single-user work, `SPEC=dspark` is the faster choice. v2.8 cold prefill 1K / 8K: 2,753 / 5,383 tok/s, unchanged within noise (prefill
is target-side).

Levers on the DFLASH path in development screens, each on against the same build with it off:

| lever | switch | measured |
|---|---|---|
| drafter byte cuts: MXFP8 drafter linears, FP8 64K hot-vocabulary plane for the candidate projection | `DFLASH_BYTES=mx8,hot8` | speedbench quick +8.4 %, prose c1 +6.5 % (ABBA, box A) |
| certified greedy head on verify steps of <= 16 rows | `CERT_DFLASH=1` | c1 +2.8 %; check mode (`CERT_DFLASH=check`): 0 mismatches in 34,808 rows (box A) |
| certified head for 17-192-row verify batches | `CERT_WIDE_DFLASH` (on with `CERT_DFLASH`) | code c8 +3 %, no regression elsewhere; greedy texts identical 30 / 30 |
| block verification on sampled steps | `DFLASH_BV=1` | thinking c1 +5.9 % |
| fused sampled accept with a sparse draft q, 16 or more requests | `DFLASH_FUSED=1` | thinking c24 per-request decode +2.3 % (ABBA, cap off); c8 not shown to gain, so batches below 16 keep the stock accept |

### Verify-width cap on DFlash2

**Mechanism.** DFlash2 drafts 7 tokens per step by walking a 16-candidate lattice; the row walked at slot j is chosen
by the token drafted at slot j-1. Per slot the cap reads the walked row's maximum probability (temperature 1 for
greedy requests, the request temperature for sampled ones) and looks up a survival table (slot x probability bin,
separate for greedy and sampled): the probability that draft j is accepted given that drafts 0..j-1 were. Draft j stays
live while the product of these probabilities is at least a per-batch-size price times the modelled cost of
verifying row j+1 (milliseconds per expert of MoE union times the row's expected new experts). Whether draft j is
verified depends only on drafts 0..j-1, so the cut is a stopping rule. Dead rows are routed to their anchor row's
experts (the expert union shrinks to that of the live rows) or, with NOEXP, to no expert. The accept takes at most L-1
drafts and the bonus from the last live row; block verification sees the dead drafts as a stopped draft. All of it runs
inside the static CUDA graphs. Batches above 8 verify every row (`bsmax=8`).

**Fitting.** A measurement-only capture (`STEPLOG`) of 15,400 decode steps (89,098 request-steps: chat, code,
thinking, agentic, long context; greedy and sampled; 0.5 % overhead) recorded the walked rows, full-width acceptance,
per-visit expert unions for widths 1-8 and per-step GPU times. Verify time against expert union gives about 0.02 ms
per expert at batch 1. Tables, prices and row costs were fitted and scored with two-fold cross-fitting over request
clusters (`tools/dfreplay.py`; held-out batch-1 projection +8.5 to +11 %). The shipped policy is
`overlay/inkling_vcap/policies/df-SLDF1-sm.json`.

**Sampled law.** Monte Carlo through the engine's own selector walk and sampled accept
(`tools/gpu_check_dflash2_cap_law.py`, `RSFIX=1`; about 1M steps per arm, a 24-token Markov target, deliberately wrong
values in every dead target row): chain and block verification with the stopped-draft mask at temperature 1.0 and
0.7, target top-k 5, block verification without the mask and a look-ahead cut rule all pass (chi-square z from -1.3 to
+0.9, no invalid token); withholding the cutoff fails as it must (z = 142).

**Greedy exactness** (30 prompts, greedy, per-token logprobs).
- The expert remap never changes a live row: capped runs with and without it are bit-identical on all 30 prompts.
- Differences from uncapped serving come from the step schedule: under the cap a token position lands in a different
  row of a differently placed verify block, which changes the floating-point reduction order. With the remap disabled
  (same schedule) the shipped policy is bit-identical to the remapped run on all 30 prompts (box B).
- Full v2.8 stack, cap on vs off (box A): 28/30 identical. The other two first diverge where the capped run takes the
  uncapped run's second-ranked token, at top-2 log-probability gaps of 0.026 and 0.367 nats, after a gradual
  sub-nat drift on the shared prefix. NOEXP on vs off: output hashes identical on 30/30 (logprobs within 0.12 nats on
  one prompt; MoE tile tactics depend on per-expert token counts).

**Speed** (box A, v2.8 stack, cap on vs off; NOEXP and block verification on).

| | cap off | cap on | change |
|---|---:|---:|---:|
| speedbench quick, tok/s | 62.49 | 70.61 | +13.0 % |
| median step, ms (tokens/step) | 49.51 (3.000) | 40.54 (2.741) | -8.97 ms |
| decode prose c1 / c4 / c8 | 55.4 / 122.8 / 190.6 | 65.1 / 140.5 / 203.5 | +17.5 / +14.4 / +6.8 % |
| decode code c1 / c4 / c8 | 67.7 / 119.2 / 187.9 | 79.0 / 141.6 / 211.3 | +16.7 / +18.8 / +12.5 % |
| decode prose / code c16 (not capped) | 292.3 / 297.5 | 288.3 / 298.2 | within noise |
| thinking c8, sampled: aggregate (per-request median) | 280.1 (39.1) | 277.4 (38.2) | -1.0 % (one run each) |

NOEXP adds +0.6 % to speedbench quick with the cap on (70.20 -> 70.61 tok/s). A second boot of the release defaults
on box B gave 70.17 (prose 65.48).

### Release stack, one box

speedbench full (73 prompts x 256 tokens, box A, 0 failed prompts), v2.8 against the published v2.7.2 two-box median,
v2.8 tokens per step in brackets: prose 64.66 vs 53.6 (+20.6 %, 2.50), code 79.89 vs 72.1 (+10.8 %, 3.32), structured
77.14 vs 68.0 (+13.4 %, 3.40), agentic 73.57 vs 76.7 (-4.1 %, 3.18), multilingual 57.21 vs 43.7 (+30.9 %, 2.23), long
context 63.12 vs 75.1 (-16.0 %, 2.66), all 73 prompts 69.66 vs 63.8 (+9.2 %, 2.77); median TTFT 162.8 ms. The
long-context category trails because the drafter sees only a 2,048-token window (32K: 3.10 vs 3.36 tokens per step,
long-context table above); agentic trails slightly (-4.1 %).

speedbench quick per category, release defaults (cap off in brackets): prose 65.67 (55.88), code 80.16 (73.69),
structured 77.26 (69.00), agentic 75.72 (68.21), multilingual 64.47 (56.50), long context 63.00 (56.02); 0 failed
prompts. Thinking (T=0.8, 48 x 2,048 max tokens at c8 and c24): c8 277.4 aggregate (38.2 per request), c24 515.5
(24.0); v2.7.2 with the same protocol 235.3 / 438.8. An earlier boot of the same stack with the cap off gave c1 / c4 /
c8 / c24 82.5 / 177.9 / 288.4 / 518.9 aggregate. The aggregate depends on how long the completions run, so the
per-request decode is the better comparison.

### Sampled-verification fixes (`RSFIX`)

SGLang's reject sampler picks the residual / bonus token as the first lane whose running sum exceeds the coin, with no
range or positive-mass condition. FP32 rounding between the normalizer and the scan can let a zero-mass lane match: a
padded vocabulary row (ids 200,058-201,023 of the 201,024-wide logits have p = 0) or a lane past the vocabulary (seen
on the GPU at a 24-token test vocabulary). The fix takes the first in-range lane with positive mass, then the last
positive residual lane, then the last positive target lane. The DFLASH selector walk clamped a coin past the end of
its scan to the last candidate, which can have q = 0; the fix takes the last candidate with q > 0. SGLang PR #51681
(DSpark Markov sampler) and PR #51692 (blockwise rejection, not on the recipe's path) are carried verbatim. None of
these changes the law in exact arithmetic. Block verification on DFLASH is enabled only together with the `reject`
fix.

### Memory

The drafter's KV is a fixed ~2.16 GB ring over its window instead of a per-token draft lane, so the packed store holds
more global-layer tokens at the same fraction than with DSpark. Box B, `MEM_FRACTION=0.89`, `CTX=262144`:
2,371,584-2,416,896 tokens across boots, MemAvailable 4.5-5.5 GiB when the API is ready, 4.9 GiB at worst with thinking
c24 (48 x 2,048 tokens) alongside a cold 240K needle (guard floor 3.5 GiB); start-up 131-275 s. In development,
`MEM_FRACTION=0.90` gave 2.42M tokens with 4.6 GiB at worst. Above `CTX=262144` the release keeps 0.85 (larger
prefill scratch) until measured.

### Folded in

v2.7.3 and v2.7.4 were not released separately: `RSFIX=1` in both profiles; for `SPEC=dspark`, `hotbase` (compact
hot-vocabulary base logits inside the captured Markov walk, bit-identical step logits and tokens) and
`RNNHOIST_MAX_BS=32` (the Markov-walk hoist for up to 32 rows; c17-c32 drafts change by FP32 order, proposals only).

## v2.8.1: prefix-hit draft starvation and the prefix tail

Basic tests on one box; full two-box qualification follows in v2.8.x. Results on box B: `results/V281s-s06/`
(release defaults), `results/T81a1-s06/`
(speedbench full, tail 2,048 with limit 8,192), `results/N81sbfc-s06/` (v2.8 default), `results/N81sbf-s06/` (radix
cache off), `results/T81t1k-s06/` (tail 1,024 with limit 8,192), `results/T81t2k-s06/` (no input limit),
`results/T81kv-s06/` (worst case, shipping default), `results/A81off-s06/`, `results/A81m6k-s06/`,
`results/A81m8k-s06/`, `results/A81m12k-s06/` (multi-turn agent sessions); on box A: `results/V28m1-s05/` (1M).

Cause. DFlash2's draft KV is a ring over the last 2,048 positions, filled from the target's hidden states as the
target prefills or verifies them. A radix-cache hit skips the target prefill of the matched prefix, so the ring holds
only the uncached suffix and the drafter proposes with little or no context. speedbench primes every prompt before it
measures, and multi-turn traffic re-hits its prefix on every turn.

Cache off against cache on (speedbench full, c1, same box and session, `MEM_FRACTION=0.85`):

| | all 73 prompts | long context | agentic |
|---|---:|---:|---:|
| v2.8 (radix cache on) | 69.59 | 63.09 | 73.44 |
| tail 2,048, limit 8,192 (`T81a1`) | 72.49 | 73.85 | 82.25 |
| radix cache off (upper bound) | 72.82 | 75.00 | 82.42 |

Prose, structured and multilingual do not move; code 79.84 -> ~81.5. A 1,024-token tail was also screened
(`results/T81t1k-s06/`). Per prompt: longctx-01 2.84 -> 4.20 tokens per
step, agentic-01 10.8 -> 17.0. The tail recovers almost all of the cache-off upper bound. The v2.8.1 row was measured
with the 2,048-token tail at a limit of 8,192; every speedbench prompt in those categories that gains is at most 4,096
tokens long except longctx-02 (4,930 tokens), so the shipped limit of 4,096 gives the same categories. End to end
(time to first token plus decode, 256-token outputs): long context +6.2 %, agentic +1.2 %, code +1.6 %.

Release defaults (box B, `results/V281s-s06/`, `MEM_FRACTION=0.89`, limit 4,096), speedbench full, tok/s with the
median tokens per step in brackets: prose 64.92 (2.50), code 81.88 (3.62), structured 77.46 (3.40), agentic 82.74
(3.38), multilingual 57.46 (2.23), long context 73.53 (3.27), all 73 prompts 72.86 (2.80); median time to first token
162.1 ms. v2.8 on the same box (`MEM_FRACTION=0.85`): 64.57 / 79.84 / 77.10 / 73.44 / 57.19 / 63.09 / 69.59.
longctx-02 (4,930 tokens) is above the 4,096 limit and runs as on v2.8. At 32K / 128K: decode 64.3 tok/s (3.17
tokens per step) / 50.7 tok/s, cached time to first token 0.24 / 0.46 s.

Long context is a different matter. A 32K test refuted starvation there: with the radix cache off, DFlash2 at 32K
drafted 2.99 tokens per step against 3.04-3.08 with the cache, and DSpark 3.29-3.37 either way. Without the input
limit (`results/T81t2k-s06/`) the tail also bounds 32K / 128K hits, which raised their cached time to first token
from 0.26 to 6.38 s and from 0.45 to 35.3 s (a full re-prefill, see below) for no acceptance gain; with the limit they
stay at 0.24 / 0.45 s. The 32K decode deficit against DSpark is the drafter's 2,048-token window.

Elastic draft window (box A, up to 8 requests, drafter config copies with a larger `sliding_window`):

| drafter `sliding_window` | speedbench quick | tokens per step at 32K | tokens per step at 128K |
|---|---:|---:|---:|
| 2,048 (as shipped) | 69.9 | 3.26 | 3.42 |
| 8,192 | 70.4 | 3.29 | 2.97 |
| 16,384 | 70.4 | 3.11 | 2.89 |

No gain: the drafter was trained at a 2,048-token window, and a wider window lowers its acceptance at 128K.

Gate design. The tail bounds the prefix match (`Req._compute_max_prefix_len`, the bound SGLang's own logprob path
uses for the match, the prefetch and the storage linker) to at most `input_len - 2048`; the cache picks the longest
reusable prefix within the bound, page- and checkpoint-aligned, as for a logprob request. When the uncached suffix is
already >= 2,048 tokens nothing changes. Inkling's hybrid state is cached at checkpoints, so the first bounded hit of
a prompt usually resumes from the root (a full re-prefill); that re-prefill plants a checkpoint at the bound, and later
hits of the same prompt redo only ~2K tokens. The planted checkpoint replaces the request's end-of-prompt checkpoint,
so the next turn re-prefills from the bound once even when it is not bounded itself. Hence the input limit: only
prompts of at most 4,096 tokens are bounded (`DFLASH_PREFIX_TAIL_MAX_INPUT`). Scope: DFLASH only (read from SGLang's
runtime context), never a streaming-session continuation; the stock bound and its lookup are pinned by source hash.
Measured cost on cache hits: median time to first token 186 -> 540 ms on speedbench's long-context prompts, 170 -> 235
ms on its agentic prompts; 32K / 128K hits are above the limit and stay at 0.24 / 0.45 s.

Multi-turn agent sessions (speedbench agent mode, 8 conversations of 4-8 turns, 5-12K-token histories), end-to-end
time against the tail off:

| limit (`DFLASH_PREFIX_TAIL_MAX_INPUT`) | 160-token turns | 512-token turns |
|---|---:|---:|
| 4,096 (default) | same as v2.8 | same as v2.8 |
| 6,144 | +5.0 % | about break-even |
| 8,192 | +8.7 % | about break-even |
| 12,288 | +14.8 % | about break-even |

At 512-token turns decode gains +3-8 %, about what the re-prefill costs. Raise the limit only for long-output or
document-QA traffic.

Memory. The tail allocates no memory; the worst case (thinking c24 alongside a cold 240K needle, shipping default)
was 4.1 GiB, in a run that generated 4 % more tokens than the v2.8 run (guard
3.5 GiB), and the needles at 128K and 240K pass.

1M on v2.8 (`CTX=1048576`, box A): needle PASS at 999,296 tokens; decode at 1,000K context 23.3 tok/s (one
repetition); pool 2.14M global-layer tokens; lowest MemAvailable 8.9 GiB.

## v2.8.2: prefill levers and the verify-cap policy df-SLDF2x

Basic tests on one box; full two-box qualification follows in v2.8.x. Release-defaults runs on box B:
`results/PF282-s06/`, `results/REL282-s06/` (speedbench full, prefill, worst case), `results/KVS282a-s06/`,
`results/KVS282b-s06/` (worst case, needles), `results/KVS282c-s06/` (`MEM_FRACTION=0.89`), `results/PHAc-s06/`,
`results/PHAc2-s06/` (prefill control on the v2.8.1 tree), `results/SBF0-s06/` (cap
off), `results/SBF1-s06/` (v2.8.1's cap policy), exactness `results/XGc-s06/`, `results/XGc2-s06/`, `results/XGg-s06/`,
`results/XGgp-s06/`, `results/XGag-s06/`, `results/XGn-s06/`, `results/CAP282-s06/`. v2.8.2 is a prefill and
verify-policy release: speedbench decode is unchanged (72.78 / 72.82 against v2.8.1's 72.86).

### Where prefill time goes (v2.8 defaults, box A)

Cold prefill in the same boot: 1K / 8K / 32K / 128K 2,856 / 5,425 / 5,111 / 3,743 tok/s; profiler overhead < 1 %.
One 8,119-token chunk is one captured graph: GPU wall 1,436 ms, busy 1,416 ms, host 20 ms.

| class (8K chunk) | ms | % |
|---|---:|---:|
| MoE GEMMs: NVFP4 w13 182.9 (166 TF/s), w2 114.5 (133 TF/s), MXFP8 44.2 (94 TF/s) | 341.6 | 23.8 |
| dense GEMMs (qkvr, wo, router, dynamic conv) | 279.1 | 19.4 |
| loop message path (fused mix 113.8, statistics 82.0, split down projection 29.7) | 225.5 | 15.7 |
| MoE glue (combine 71.6, SwiGLU + quantise 67.5, quantise-once gather 23.4, quantise 10.8, MXFP8 17.2, routing 4.6) | 196.5 | 13.7 |
| attention (sliding window 79.9 at ~48 TF/s, global 49.8 at ~65 TF/s) | 131.1 | 9.1 |
| short conv 64.1, residual norm 49.2, gate_sink 39.3, L2 clips 31.8, KV store 26.2, RMSNorm 20.1, other ~10 | 240 | 16.8 |
| host | 20.1 | 1.4 |

At 128K the chunks grow from 1,403 to 2,846 ms; global attention is 12.2 s of 34.7 s at ~69 TF/s (the kernel's peak).
A prompt of 16 x 8,192 + 25 tokens pays a 220 ms forward for its 25-token tail, so prefill runs are compared only at
equal prompt lengths.

### Prefill levers

- `PFUSE_HOIST` (`overlay/inkling_phoist`, opt-in, default 0): the loop bank's attention-residual weights, the message rstd and the
  message-only K groups of the controller down projection are prepared once per loop pass. 99.96 % of elements
  bit-equal to the fused path, 168 / 168 FP64-reference comparisons within 1.5x of its error (512-8,192 rows). One 8K
  loop pass 104.5 -> 93.0-94.0 ms (-21 to -23 ms per 8K chunk). A start-up self-test (before graph capture) checks it
  against the fused path and falls back on any failure; only the checked row-count regimes are armed. In the
  prefill exactness gate, 2 long prompts diverged with the hoist on at points that are not near-ties; it ships
  opt-in until a bracket places them within the normal floating-point spread.
- `ATTN_GATE=1`: gate_sink in the epilogue of the sliding-window prefill attention kernel. Bit-identical (output and
  LSE) to SGLang's gate_sink, including graph replay and padded multi-request batches. The epilogue stages the gate
  tile in the free K / V shared memory (coalesced 16-byte loads and stores): 234 vs 232 registers, no spills, 2 CTAs
  per SM. 8K: attention + gate_sink 2,759 -> 1,944 us per sliding-window visit (-32.6 ms per chunk); a first,
  uncoalesced version saved 20.6 ms.
- `PROLOGUE` (`overlay/inkling_prologue`, opt-in, default 0): the sliding-window visit's K conv + L2 clip and V conv +
  value norm as one kernel each. 227 -> 122 us per call, -8.4 ms per 8K chunk. It flips single elements by one BF16
  ulp (the order of the sum of squares), so it is in the floating-point-order class like the hoist: the gate + prologue
  serving run did not match the control in the exactness gate, and it ships opt-in.

Exactness gate (30 speedbench quick prompts, greedy, against a control on the v2.8.1 tree):

| arm | identical | near-ties | not near-ties |
|---|---:|---:|---:|
| control repeat (`XGc2` vs `XGc`) | 30 / 30 (4,691 tokens bit-equal) | 0 | 0 |
| sink gate, the default (`XGg`) | 30 / 30, max log-probability difference 0 | 0 | 0 |
| gate + prologue (`XGgp`) | 27 / 30 | 1 | 2 |
| hoist + gate (`XGag`) | 26 / 30 | 2 | 2 |
| the FP-order prefill fusion shipped since v2.8 turned off (`XGn`) | 26 / 30 | 3 | 1 |

Turning off the floating-point-order prefill fusion that has shipped since v2.8 moves the same four long prompts as
the hoist does, so these long-prompt divergences follow any change of floating-point order in prefill, not a defect in
one lever. With the v2.8.2 cap on, the release matches v2.8.1 on 29 / 30; the difference (longctx-09) comes from the
cap: with the gate off and the cap on (`CAP282`) the outputs equal the release on 30 / 30. It is the four-way near-tie
described under the policy below.

The gated epilogue turns itself off while `SGLANG_KERNEL_API_LOGLEVEL` is non-zero (logging "gated epilogue OFF
..."): that kernel logging reads the FA4 FP32 output buffer, which the gate leaves unwritten. With the variable unset
nothing changes (4 unit tests; checked live in the release run).

Serving, cold prefill tok/s (box B, same prompts, mean of 2 runs):

| | 1K | 8K | 32K | 128K |
|---|---:|---:|---:|---:|
| v2.8.1 control (same box; a repeat: 2,812 / 5,408 / 5,077 / 3,726) | 2,836 | 5,411 | 5,073 | 3,749 |
| opt-in: hoist + gate | 2,833 | 5,599 (+3.5 %) | 5,254 (+3.5 %) | 3,844 (+2.6 %) |
| opt-in: hoist + gate + prologue | - | 5,656 | - | - |
| gate only (release defaults) | 2,821 (flat) | 5,527 (+2.1 %) | 5,189 (+2.2 %) | 3,785 (+1.0 %) |

Memory: the levers deepen the MemAvailable dip of a long prefill by ~0.6-0.7 GB, so `SPEC=dflash2` up to
`CTX=262144` sizes at `MEM_FRACTION=0.88` with any of them on, including the default gate. Release defaults: KV pool
2,349,056 tokens (v2.8.1 at 0.89: 2,400,384, -2.1 %); cold time to first token 1.48 / 6.30 / 34.6 s at 8K / 32K /
128K. Worst case (thinking c24, 48 x 2,048 tokens, alongside a cold 240K needle; 48 / 48 completed, 343.8 / 341.8
tok/s aggregate): lowest MemAvailable 5.7 / 5.0 / 5.9 / 6.4 GiB over four runs (guard 3.5 GiB). At 0.89 the guard
stopped the server 25 s into the same load (3.3 GiB), so 0.88 stays. Needles at 128K and 240K PASS.

Speedbench full, release defaults, two runs (`PF282` / `REL282`), against v2.8.1 (`V281s`):

| | prose | code | structured | agentic | multilingual | long context | all 73 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2.8.2 | 64.99 / 64.96 | 81.52 / 81.43 | 77.62 / 77.56 | 83.32 / 83.21 | 57.63 / 57.56 | 72.15 / 72.94 | 72.78 / 72.82 |
| v2.8.1 | 64.92 | 81.88 | 77.46 | 82.74 | 57.46 | 73.53 | 72.86 |

Same tree, other cap settings: no cap (`SBF0`) 64.06, long context 67.31; v2.8.1's `df-SLDF1-sm` policy (`SBF1`)
72.42, long context 73.05. The new policy is +0.4 on all prompts against the old one, and the long-context spread
(72.15-73.53) is run-to-run variation.

### Closed prefill candidates (measured, not adopted)

| idea | measured | verdict |
|---|---|---|
| single-kernel loop message path | correct against FP64, but one 8K pass 137-163 ms against the fused path's 104 ms; an all-controllers variant needs 104-115 KB of shared memory (99 KB available) | slower: the mix is bound by compute and parallelism (~82 GB/s effective), not by bytes |
| 8-warp sliding-window prefill attention | 2,152 vs 1,877 us per 8K visit | slower |
| Triton BF16 GEMM instead of cuBLAS for the prefill linears | cuBLAS runs at 79-92 TF/s on the served shapes; Triton wins only wo (+5.6 %) and k / v (+6 %), ~-5 ms per chunk | cuBLAS is already the best choice; not worth an overlay |
| MXFP8 MoE tactics at prefill row counts | stock is already the best tactic (128, 128) at 109-126 TF/s; the others are bit-equal but 20-85 % slower | closed |
| one-pass loop mix with an L2-resident FP32 slab | L2 probe: reads up to 20 MB hit L2 at 1.5-1.6 TB/s, but repeated writes run at 658 / 766 / 578 / 321 GB/s for 2 / 4 / 8 / 12 MB and at DRAM speed from 16 MB; a 4 / 8 MB slab read back after 32 MB of other traffic costs +22 / +44 us | the write-back L2 retains only ~8 MB of dirty data: the 14 MB slab under ~50 MB of concurrent traffic goes to DRAM and costs about what the recompute saves |
| mlp-side RMSNorm folded into the loop mix | 3,866 -> 4,186 us per mlp call | slower (+7.7 ms per chunk) |

### Verify-width policy df-SLDF2x

`df-SLDF2x.json` (the new `DFCAP` default, `bsmax=8`) prices greedy and sampled requests separately and adds a
context bucket (`theta_x` per batch band, greedy / sampled and context bucket; `ctx_edges` 16,384 / 65,536). It was
fitted on a larger full-width capture (all batch bands, greedy and sampled thinking, 32K-128K prompts) with the same
survival-table form (slot x selector confidence) and cross-fitted on held-out requests. Sampled requests verify all
rows from 8 requests up; a single greedy request at 64K tokens of context or more caps less (price 0.057 instead of
0.095). Policy files without the new fields behave as before.

On the v2.8.1 stack (box A, ABBA, two runs per policy), v2.8.1 policy against `df-SLDF2x`:

| | v2.8.1 policy | df-SLDF2x |
|---|---|---|
| prose c4 | 141.2 / 138.0 | 140.2 / 140.6 |
| prose c8 | 201.9 / 205.3 | 201.6 / 202.4 |
| code c4 | 137.1 / 137.0 | 139.9 / 134.2 |
| code c8 | 204.4 / 199.5 | 201.3 / 199.2 |
| speedbench quick | 73.17 | 73.75 |
| decode at 32K | 61.27 | 66.05 (+7.8 %) |
| decode at 128K | 51.83 | 50.91 |

Earlier, on the v2.8 stack before the prefix-tail fix (off / v2.8 policy / new): speedbench quick 62.20 / 70.65 /
70.79; decode at 32K 58.97 / 60.13 / 63.85, at 128K 47.02 / 50.72 / 50.78. Sampled thinking: two identical c8
cohorts on the same server differ by up to 6.7 %, so one cohort cannot resolve a few percent; over three paired c8
cohorts the new policy against off was 0.973 / 1.056 / 1.038 in aggregate tok/s (mean 1.022), c1 per-request median
78.8 off, 79.5 new. A variant that also capped sampled requests at 8 requests (price 0.338) measured 1.042 / 0.953 /
1.024 against the new policy (mean 1.006, spread 0.089), within the cohort spread, so it was not adopted.

Exactness against the uncapped server (30 speedbench quick prompts, top-2 logprobs): 27 / 30 identical, 2 top-2
near-ties and 1 three-way near-tie (longctx-09 at token 93: four candidates within 0.1 nat, ' It' -1.939, ' self'
-1.991, ' In' -2.033, ' Need' -2.035; the capped run took the third at 0.094 nat). Unit tests check that the runtime
widths equal the fitting tool's per-request price lookup. The fitting tool (`tools/dfreplay.py --per-mode
--ctx-price`) scores sampled requests by the exact expected yield of block verification when the capture has the
target's candidate probabilities (`tools/bvexpect.py`, capture field `tgt_cand_p`, not recorded for top-k / top-p /
min-p filtered requests); on the capture this changed sampled scores by under 0.5 percentage points and left the
prices unchanged.

## v2.8.3: padding rows without experts, bit-identical prologue

Basic tests on one box; full two-box qualification follows in v2.8.x. Release runs: on box A `results/PADX283g-s05/`
and `results/PADX283off2-s05/` (released padx version against padx off, same session: speedbench full, thinking c9 /
c15), `results/PADX283-s05/` and `results/PADX283off-s05/` (first padx version: thinking c16, worst-case load) and the
padx screen
`results/PXa0-s05/`, `results/PXa1-s05/`, `results/PXb1-s05/`, `results/PXb0-s05/`; on box B `results/KVS283a-s06/`, `results/KVS283b-s06/` (memory,
needles) and `results/PF283-s06/` (prefill), `results/REL283-s06/` (speedbench full).

### Padding rows (`inkling_padx`)

A decode or target-verify CUDA graph replays at its captured batch size. With fewer live requests the tail rows are
padding: the graph runner zeroes their positions, cache locations and request indices but keeps the previous replay's
input ids, so every padding row still routes to 8 experts. Measured on the v2.8.1 stack (sampled thinking): a batch of 9
replays the graph for 10 with padding rows, and its step took 125.3 ms against 116.1 ms for a full batch of 10 (one
more real request); the difference was in the NVFP4 / MXFP8 grouped GEMMs (+8.1 / +1.9 ms). With `PADX=1` the padding rows' expert ids become the phantom expert (the out-of-range id
that the dead-verify-row skip uses), so the SM120 kernels never touch them and the combine gives them 0. The live row
count is written on the device on every replay and read inside the graph; eager calls and other graphs keep stock
routing. A row is routed only if it is live for both padx and the dead-verify-row skip.

Exactness: live rows bit-identical in the GPU check (14 shapes, alone and with the dead-verify-row path). Concurrent
greedy output on this stack varies with the batch composition: two runs with padx off already differ on 57 of 60
outputs, and padx on against off shows the same first-divergence class.

Screen on box A, thinking (T=0.8), aggregate tok/s, ABBA, two runs per arm:

| requests | padx off | padx on | aggregate | per-request median |
|---|---|---|---:|---:|
| 9 | 269.5 / 270.3 | 292.0 / 291.3 | +8 % | +6 % |
| 10 | - | - | flat | - |
| 15 | 354.9 / 375.9 | 400.5 / 392.3 | +8 % | +7 % |
| 16 | 401.2 / 397.5 | 410.6 / 407.9 | +2.4 % | - |

At 16 requests the gain comes from the end of the run, when requests drain and the batch pads.

Gating per graph. The first version masked every captured graph and added 4 small kernels per MoE visit to each: 192
per step, +0.36 ms (0.8 %) at c1, where nothing pads. Graphs of 1-8 requests can never replay with padding (the runner
pads n live requests up to the smallest captured size >= n, and 1-8 are all captured), so the released version decides
at capture time: graphs of 1-8 requests keep the stock route with zero extra kernels, graphs of 10 and up get the mask.
Live rows are bit-identical in every shape; the small graphs are bit-identical to stock on every row (GPU check
against written-out expectations).

Released version against padx off, box A, same session (`PADX283g` / `PADX283off2`):

| | padx off | padx on | change |
|---|---:|---:|---:|
| speedbench full, all 73 prompts | 73.25 | 73.17 | flat |
| median step (tokens per step) | 39.46 ms (2.742) | 39.59 ms (2.742) | two padx-off runs differ by 0.09 ms |
| thinking c9, aggregate (per-request median) | 269.3 (34.3) | 282.2 (35.6) | +4.8 % |
| thinking c15, aggregate (per-request median) | 373.9 (27.9) | 392.5 (29.6) | +5.0 % (+6.1 %) |

With the first version (`PADX283` / `PADX283off`; graphs of 16 and 24 requests are masked in both versions): thinking
c16 415.5 vs 403.4 (+3.0 %), the worst-case load (thinking c24) 331.6 vs 324.6 (+2.2 %). The released version's c1
greedy outputs equal v2.8.2's release run on 30 / 30 prompts, and the 128K needle passes.

### Prologue in the stock row-sum order

v2.8.2's fused prologue (K conv + L2 clip, V conv + value norm) flipped single elements by one BF16 ulp because it
summed the squares in a different order, and it shipped opt-in. The v2.8.3 kernel reduces in the stock kernel's
row-sum order: 0 ulp at 2,048 / 2,049 / 3,001 / 8,192 rows, including adversarial rows built to be sensitive to the
summation order. Serving dump: greedy outputs identical to both controls on 30 / 30 prompts, log-probabilities
bit-equal over 4,691 tokens. About -8.5 ms per 8K prefill chunk. It is on by default (`PROLOGUE=1`).

Cold prefill, release defaults (box B, `PF283`, mean of 2 runs) against v2.8.2: 1K 2,864 vs 2,821 (+1.5 %), 8K 5,553
vs 5,527 (+0.5 %), 32K 5,217 vs 5,189 (+0.6 %), 128K 3,788 vs 3,785 (+0.1 %); against the v2.8.1-tree control (mean of
`PHAc` / `PHAc2`) +2.7 % at 8K, +2.8 % at 32K, +1.4 % at 128K.

Memory (box B, `MEM_FRACTION=0.88`): KV pool 2,352,000 tokens; lowest MemAvailable under the worst-case load (thinking
c24, 48 x 2,048 tokens, alongside a cold 240K needle) 5.6 / 5.5 / 5.4 / 5.3 GiB in `KVS283a` / `KVS283b` / `PF283` /
`REL283` (48 / 48 requests completed, 340.1 / 357.4 tok/s aggregate; guard 3.5 GiB). Needles at 128K and 240K PASS;
c1 greedy outputs equal v2.8.2's release run on 30 / 30 prompts in all four runs.

Speedbench full, release defaults (`REL283`, box B, one run, 0 failed): all 72.90 (v2.8.2 72.78 / 72.82), prose
65.02, code 81.56, structured 77.63, agentic 83.28, multilingual 57.65, long context 72.99; median time to first
token 163.8 ms, median step 39.71 ms, 2.742 tokens per step. c1 decode is unchanged within the run-to-run spread;
v2.8.3's gains are at concurrency (padding rows) and in prefill (the prologue).

## v2.8.4: drafter MXFP8 GEMM (`mx8w`)

Basic tests on one box; full two-box qualification follows in v2.8.x. Runs on box A: ABBA `results/MXa0-s05/`
(off), `results/MXa1-s05/` (on), `results/MXb1-s05/` (on), `results/MXb0-s05/` (off); memory `results/MXkv-s05/`;
release check `results/REL284-s05/`.

Since v2.8, `mx8` runs the drafter's linears on MXFP8 copies of their BF16 weights through a GEMV (up to 16 rows)
and a Triton kernel. `mx8w` hands the 20 of the 30 linears whose K and N are multiples of 128 to FlashInfer's stock
b12x MXFP8 GEMM (`mm_mxfp8`, activations quantised per call by `mxfp8_quantize`) for every row count up to 256, on the
same FP8 bytes; the E8M0 scales are swizzled once into the layout the GEMM reads (+11.2 MB, no second FP8 copy). The
10 576-wide conv projections keep `mx8`. It changes draft proposals only; greedy output under the lossless
verification was identical in every arm (30 / 30, no near-ties).

Microbench, the drafter's 30 layer linears in one graph, cold L2:

| rows | BF16 cuBLAS | `mx8` GEMV | `mx8w` (b12x) |
|---:|---:|---:|---:|
| 8 | 3.33 ms | 2.23 ms | 1.90 ms |
| 32 | 3.42 ms | - | 1.96 ms |
| 64 | 3.48 ms | - | 2.00 ms |
| 192 | 4.05 ms | - | 2.34 ms |

1.73-1.76x over BF16; 0.33-0.36 ms faster than the `mx8` GEMV at 8 / 16 rows.

Serving ABBA (box A), speedbench full:

| | prose | code | structured | agentic | multilingual | long context | all 73 | median step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| on (two runs, mean) | 65.97 | 82.73 | 78.03 | 83.97 | 58.24 | 74.46 | 73.78 (73.75 / 73.81) | 39.21 / 39.22 ms |
| off (two runs, mean) | 65.14 | 81.58 | 77.57 | 83.32 | 57.70 | 72.96 | 72.93 (73.13 / 72.73) | 39.57 / 39.91 ms |
| change | +1.3 % | +1.4 % | +0.6 % | +0.8 % | +0.9 % | +2.1 % | +1.2 % | |

Per-stream decode at c1: prose 66.7 / 66.9 against 65.9 / 65.7 (+1.4 %), code 79.4 / 79.5 against 78.7 / 78.8
(+0.9 %). At c4-c16 mostly up but noisy, from code c8 -1.2 % to prose c4 +3.8 %. Thinking, per-request median: c8
+1.8 %, c24 +2.4 %. No acceptance change is claimed; the gain is in the drafter's GEMM time.

Memory (`MXkv`, box A, `mx8w` on): lowest MemAvailable under the worst-case load 4.8 GiB, thinking c24 aggregate
345.8 tok/s, needles at 128K and 240K PASS; KV pool unchanged (about 2.36M tokens on box A).

Release check (`REL284`, box A, release defaults, one run, 0 failed): speedbench full 73.53 (prose 65.75, code
82.46, structured 77.73, agentic 83.69, multilingual 58.01, long context 74.20); median time to first token 164.8 ms,
median step 39.27 ms, 2.833 tokens per step; same-box reference with `mx8w` off 73.13 / 72.73. v2.8.3's 72.90 was
measured on box B and is not compared as a gain. Per-stream decode, prose c1 / c4 / c8 / c16 66.7 / 37.0 / 27.9 /
20.4, code 79.1 / 38.6 / 30.0 / 22.0. Greedy outputs identical to `MXa0` (`mx8w` off) on 30 / 30 prompts; the sink
gate, the verify cap, padding rows, the prologue and `mx8w` (20 of 30 drafter linears on the MXFP8 GEMM) all
reported armed at start-up. Lowest MemAvailable 4.9 GiB; the v2.8.4 worst case stays 4.8 GiB (`MXkv`).

## v2.9: draft-tail snapshots

Basic tests on one box; v2.8.4 (the base of v2.9) passed a two-box qualification: Q284R and Q284S on both boxes. Runs on box B: release `results/V29R-s06/`
(speedbench full and agent mode), `results/V29S-s06/` (worst case, needles); screen `results/S81off-s06/`,
`results/S81a-s06/`, `results/S81b-s06/`, `results/S81ab-s06/`; slots `results/K81b8-s06/`, `results/K81b16-s06/`;
exactness `results/BEXoffref-s06/`, `results/BEXoffb-s06/`, `results/BEXonref-s06/`, `results/BEXonb-s06/`,
`results/BEXschref4-s06/`. v2.8.4 qualification: `results/Q284R-s06/`, `results/Q284R-s05/`,
`results/Q284S-s06/`, `results/Q284S-s05/`.

Mechanism. The v2.8.1 prefix tail re-prefills the last 2,048 prompt tokens on a prefix-cache hit of a prompt of at most
4,096 tokens, so the drafter's ring holds them; longer histories (agent loops) keep the cache and draft from the
uncached suffix only. Snapshots keep the drafter's 2,048-position window at the end of each completed prompt in one of
8 slots, keyed by the prompt's content and cache salt, and copy it back into the ring when a later request hits the
same prefix, with no re-prefill. Draft-only state: the target model's computation is untouched. 8 slots x 2,048
positions x 5 draft layers = 320 MiB, charged to the KV budget (0.312 GiB). Restored only on matching content and salt;
weight updates and `flush_cache` clear the slots. The LRU is shared across tenants, so `DFLASH_TAIL_SNAPSHOTS=0` is the
setting where tenants must not observe each other's cache use.

Screen (box B, v2.8.1 base), change against both off (A = prefix tail, B = snapshots):

| | speedbench full, long context | agentic | code | agent loops, 160-token turns | 512-token turns |
|---|---:|---:|---:|---:|---:|
| A | +6.6 % | +1.1 % | +1.7 % | -0.7 % | +0.3 % |
| B | +14.4 % | +0.2 % | +0.1 % | -4.0 % | -5.4 % |
| A + B | +6.7 % | +1.3 % | +1.7 % | -4.4 % | -4.9 % |

Agent loops: speedbench agent mode, 8 conversations of 4-8 turns on 5-16K-token histories, end-to-end time at equal
output length (negative is faster). With A + B decode goes from 73.9 to 78.3 tok/s; median time to first token stays
at 0.242-0.247 s. B alone wins most on cached long prompts (no re-prefill) but misses A's gain on short agentic and code
prompts below its 1,024-token capture minimum, so the release ships A + B.

Exactness (greedy, top-k dumps of 30 quick prompts, 13 long prompts and 48 agent turns):

| comparison | result |
|---|---|
| verify cap off, reference vs snapshots | no token outside the reference's top-k; 27 / 30 quick prompts identical; long prompts and agent turns near-ties only |
| verify cap on, reference vs snapshots | the same picture |
| control: reference vs itself with only the verify width fixed at 4 (same drafts, no snapshots) | diverges on the same three prompts; largest shift 1.65 nat (longctx-02), larger than any snapshot difference |

So the snapshots' differences are in the verify-schedule class. The release run's greedy outputs equal v2.8.4's on
30 / 30 prompts.

Slots and memory: 8 slots, worst MemAvailable 4.6 GiB; 16 slots, 4.0 GiB with no speed gain, so the default is 8.
Release defaults (`V29S`): KV pool 2,371,200 tokens, worst 4.6 GiB, needles at 128K and 240K PASS, the worst-case load
(thinking c24 alongside a cold 240K needle) completed 48 / 48 requests.

Release run (`V29R`, box B): speedbench full 73.34 against v2.8.4's 73.43 on the same box (`Q284R`), flat; speedbench
agent mode 76.33 (160-token turns) and 78.86 (512-token turns), geometric mean tok/s.

## v2.9.1: sink gate on the global layers

Basic tests on one box; v2.8.4 (the base of v2.9) passed a two-box qualification: Q284R and Q284S on both boxes. Runs
on box B: release `results/PF291-s06/` (prefill, speedbench full), `results/KVS291-s06/` (worst case, needles);
exactness `results/XG291r-s06/` (gate on the sliding-window layers), `results/XG291g-s06/` (gate on all layers),
`results/XGGa-s06/`, `results/XGG1-s06/`; A/B `results/PFGa-s06/`, `results/PFG1-s06/` (prefill),
`results/KVSGa-s06/`, `results/KVSG1-s06/` (memory).

v2.8.2 moved the sink gate (`gate_sink`) into the epilogue of the sliding-window prefill attention kernel. v2.9.1 does
the same on the global layers, on both the BF16 and the packed 12-bit KV paths (`ATTN_GATE=all`; `1` keeps the
sliding-window-only gate). Kernel: -840 us per 8K global-layer visit, -6.7 ms per 8K chunk.

Exactness: the GPU check finds y and LSE bit-identical to stock `gate_sink` on every path, including captured pairs
and replays. In serving, the log-probability dump with the gate on all layers matches the sliding-window-only gate on
30 / 30 records, and greedy outputs equal v2.9's release run on 30 / 30 prompts.

Prefill, cold, tok/s (box B, mean of 2 runs):

| context | v2.9.1 release defaults | v2.8.3 release defaults | change | A/B, all layers against sliding-window only |
|---|---:|---:|---:|---:|
| 1K | 2,821 | 2,864 | ~flat | - |
| 8K | 5,595 | 5,553 | +0.8 % | +0.4 % |
| 32K | 5,236 | 5,217 | +0.4 % | +0.4 % |
| 128K | 3,824 | 3,788 | +1.0 % | +0.5 % |

Speedbench full: 73.12 against v2.9's 73.34 on the same box, decode unchanged within the run-to-run spread.

Memory: release defaults worst MemAvailable 4.8 GiB, needles at 128K and 240K PASS; `MEM_FRACTION` stays 0.88. The
A/B memory pair (gate on all layers / sliding-window only) gave 5.2 / 5.4 GiB. Its worst-case aggregate (326.7 vs
348.0 tok/s) reflects a different number of generated tokens at T=0.8 (86,182 vs 91,254), not speed: wall time,
per-request decode and time to first token were equal or better with the gate on all layers.

## v2.9.2 draft: the prefill loop-bank hoist under a quality gate (2026-11-25)

The hoist (`PFUSE_HOIST=1`, `inkling_phoist`) changes the FP32 summation order of the loop controller's source
projection, not the math: 99.96 % of elements bit-equal to the fused path, FP64-reference error within 1.5x, startup
self-test with fallback. In the exactness gate it moved 2 of 30 long prompts without a near-tie; removing the shipped
fused loop mix (`PFUSE` without `loopmix`, the same class against stock) moved the same 4 long prompts with 1
non-near-tie, so the hoist is in the class v2.8 already ships. The owner set the gate for such levers to quality.

Eval `inkq` (not shipped in this repo): 575 automatically graded items, every prompt >= 1.5K tokens so the hoist runs (it arms
from ~600 prefill rows): GSM8K test 0-399 10-shot (final number), 100 synthetic long-context lookups at 4 / 8 / 16 /
32K tokens (exact code), 75 code / JSON / math / format tasks behind a fixed 1.1K-token preamble (their own checkers). T = 0, effort
low, one box (box A), arms interleaved.

| run | gsm8k | lc | qeval | total | cut-off answers |
|---|---:|---:|---:|---:|---:|
| v1 QE291a (v2.9.1) | 382 | 98 | 64 | 544 | 30 |
| v1 QE292h (hoist) | 382 | 97 | 63 | 542 | - |
| v1 QE291b (v2.9.1) | 383 | 99 | 64 | 546 | - |
| v2 QA1 (v2.9.1) | 390 | 100 | 66 | 556 | 7 |
| v2 QL1 (hoist) | 392 | 100 | 70 | 562 | 5 |
| v2 QA2 (v2.9.1) | 389 | 100 | 70 | 559 | 5 |
| v2 QL2 (hoist) | 391 | 100 | 68 | 559 | 5 |
| v2 QA3 (v2.9.1) | 387 | 100 | 68 | 555 | 5 |

Rules were registered before each protocol's data. v1 (A / A' / lever): flips 18 / 18 <= 27, McNemar p 0.82 / 0.48,
families within 2, total 542 outside [543, 547]: FAIL by one item; the hoist's 6 lone misses were all answers cut off
at the token limit. v2 (three baselines, two lever runs, token budgets raised): baseline flips 11 / 13 / 10, budget
22.5, total range [554, 560]. QL2 passes everything (flips 7 / 10 / 10, p 0.45 / 1.0 / 0.34). QL1: flips 10 / 9 / 7
pass, families pass, but total 562 is above the range and McNemar against QA3 is p 0.016 with 7 hoist-only wins and
0 losses: the two-sided rule fails it on the upside only. Across the six lever-baseline pairs: 38 hoist-only correct,
15 baseline-only correct. The owner's criterion is one-sided ("fine if model quality does not drop"); the two-sided
bracket was our own construct and does not block a measured win, so the hoist ships on in v2.9.2.

Serving with the hoist (box B): sanity 11 / 11, needles 128K / 240K PASS, speedbench full 73.44, worst MemAvailable
4.0 GiB at `MEM_FRACTION` 0.88 (`KVS292-s06`; a repeat `KVS292b-s06` 5.7 GiB with time to first token 1.08 s and
per-request decode 17.2 tok/s, so the first run's 1.51 s / 16.3 tok/s was run-to-run spread; v2.9.1 4.8 / 5.2).
Cold prefill against v2.9.1 (`PF292-s06` vs `PF291-s06`, box B, mean of 2 runs): 1K / 8K / 32K / 128K 2,857 / 5,657
/ 5,325 / 3,875 against 2,821 / 5,595 / 5,236 / 3,824 (+1.3 / +1.1 / +1.7 / +1.4 %).

## v2.9.2: prompt-end cache touch for agent loops (2026-11-25)

AG64 (24 agents x ~72K-token distinct contexts, two turns) showed intermittent whole-context prefix-cache misses on
turn 2: the sliding-window state behind a finished request's prompt end got its LRU recency at prefill time, and the
final insert refreshes only the window behind the output end, so after the request finished its prompt-end window was
the oldest evictable one. Under window-pool pressure the earliest conversations lost it first, their next turn fell
back to the root and re-prefilled ~72K tokens, and each re-prefill evicted the next-oldest prompt end. Observed in the
cache dumps of stock misses: the prompt-end node had lost its sliding window and with it the saved state (11 of 12
dumped misses); in the remaining one part of the window was gone.
`inkling_ptouch` (`PROMPT_TOUCH`, on) touches the prompt path with the cache's own `touch_prefix` right after the
final insert (no allocation, no lock, no state slot): eviction order only.

| arms (24 x 72K, `--prefill-decode-interval 2`) | turn-2 cache misses | turn-2 TTFT | turn-2 wall |
|---|---|---|---|
| stock, 3 runs | 3 / 3 runs with 9-10 agents missing | 163-186 s | ~370-380 s |
| prompt-end touch, 2 runs | 0 / 2 (24 / 24 hits) | 3.0 s | ~210 s |

Default-interval runs showed the same intermittent misses (5 of 24 agents in AG64 and AGD; none in AGD2, AGD3, AG64b).
Combined-tree release check: AG292 (default interval) 24 / 24 hits, time to first token 3.0 s, turn-2 wall 211 s;
KVS292c worst MemAvailable 4.8 GiB, 48 / 48 at c24, needles 128K / 240K PASS; REL292 speedbench full 73.19
(hoist-only Q292h 73.44), greedy outputs identical to Q292h on 73 / 73 speedbench and 30 / 30 warm prompts.

## v2.9.3: prompt copy for long prompts (2026-11-25)

The DFlash2 drafter drafts from a 2,048-token window, so on long documents it cannot see a passage the answer is
quoting. `inkling_pcopy` looks up the most recent earlier occurrence of the last 12 committed tokens in the request's
own context; at a match of 16 tokens or more the tokens that followed it replace the drafter's proposals for the rest
of the block (draft proposals only). Offline, on step logs replayed against the token history with the exact
alignment, the rule raised tokens per step by about 12 % on requests of 16K tokens or more; the gain came from a
minority of requests (one 32K request alone +78 %), and requests under 16K gained nothing.

Kernels: the first version ran its work as separate small launches and added 0.3 ms per single-request decode step,
so it was rebuilt as one fused launch and then, for long contexts, as two (a scan split over 2,048-token chunks with
an atomic maximum, then the per-row copy). GPU checks: bit-identical to the reference step on 160 random steps,
directed rows (coverage gaps, clipped writes, ids past the vocabulary, sampled copies in and off the candidate set),
competing hits across chunk boundaries and a fault between the two launches; sampled-law Monte Carlo tests pass,
including a wrong-q control that the test detects. GPU time of a no-hit step, single request: 12.3 us at 128K
context (the first fused version 307 us), 10-14 us at 4-32K; host enqueue about 34 us.

Two ABBA runs on box A (`warm`, speedbench full, `lc32`, agent mode 160 / 512, thinking c8), off = `DFLASH_PCOPY=0`.
ABBA 1 ran the separate-launch version, ABBA 2 the fused one; neither had the 8K gate. `lc32` decode is the sum of
decoded tokens over the sum of decode seconds of the six requests per size (3 repeats, cold and cached, 512 tokens);
the step is the median of the per-request `step_ms` over the 73 measured speedbench full requests (the 4 sentinel
requests excluded), and speedbench full is the run summary's geometric mean of per-request decode tok/s.

| arm | lc32 32K tok/s (tok/step) | lc32 128K tok/s (tok/step) | step ms | speedbench full tok/s (summary geomean) | agent 160 / 512 decode tok/s | thinking c8 tok/s |
|---|---:|---:|---:|---:|---:|---:|
| ABBA 1 off (PCa1) | 65.5 (3.10) | 55.8 (3.40) | 39.61 | 72.82 | 74.7 / 77.1 | 270.9 |
| ABBA 1 on (PCb1r) | 74.5 (3.56) | 58.6 (3.68) | 39.81 | 72.40 | 73.7 / 76.1 | 255.0 |
| ABBA 1 on (PCb2) | 66.0 (3.08) | 56.1 (3.47) | 39.83 | 72.56 | 74.0 / 76.8 | 274.5 |
| ABBA 1 off (PCa2) | 64.0 (2.99) | 57.7 (3.52) | 39.35 | 73.56 | 75.4 / 75.3 | 262.8 |
| ABBA 2 off (PCrA1) | 60.5 (2.76) | 50.7 (3.08) | 39.46 | 73.42 | 77.8 / 80.0 | 275.8 |
| ABBA 2 on (PCrB1) | 67.5 (3.10) | 50.3 (3.04) | 39.22 | 73.63 | 77.3 / 79.2 | 276.9 |
| ABBA 2 on (PCrB2) | 73.8 (3.45) | 55.9 (3.45) | 39.63 | 72.96 | 78.0 / 81.4 | 280.4 |
| ABBA 2 off (PCrA2) | 66.6 (3.08) | 51.4 (3.10) | 39.52 | 73.38 | 77.2 / 81.1 | 275.5 |

Mean of the four runs per arm: 32K 64.2 -> 70.5 tok/s (+9.8 %; 2.98 -> 3.30 tokens per step), 128K 53.9 -> 55.2
(+2.4 %; 3.27 -> 3.41). Within each ABBA every run with the copy was faster at 32K than every run without it. The
step: ABBA 1 +0.34 ms (the separate-launch version, rejected), ABBA 2 -0.06 ms. Greedy speedbench full outputs were
identical in all four arms of ABBA 2. Copies were forced on about 2.2-2.5 % of decode steps (422 / 487 forced rows in
19,456 steps), almost all in `lc32`.

Speedbench full tokens per step (deterministic across repeats) showed one cost: 13 of 73 requests changed by a
verify step or so, and one long context prompt of 2.7K tokens needed 67 steps instead of 61 (long context category
-0.8 % tokens per step). Every speedbench prompt is shorter than 5K tokens and the offline gain sat at 16K and above,
so v2.9.3 runs the copy only for prompts of at least 8,192 tokens (`DFLASH_PCOPY_MIN_CTX`); below that the step
returns before any launch and speedbench runs exactly as without the copy.

### Release checks (this tree, 8K gate, both boxes)

An ABBA on each box: box A in the order on, off, off, on; box B off, on, on, off. Same measures as above.

| arm | lc32 32K tok/s, cold / cached | lc32 128K tok/s, cold / cached | step ms (median per-request `step_ms`, 73 measured requests) | speedbench full tok/s (summary geomean) | thinking c8 tok/s |
|---|---:|---:|---:|---:|---:|
| box A, off (A1, A2) | 62.1 / 63.5 | 60.6 / 49.9 | 39.42, 39.26 | 73.21, 73.52 | 279.2, 284.3 |
| box A, on (B1, B2) | 67.8 / 70.5 | 65.5 / 48.2 | 39.53, 39.49 | 73.15, 72.99 | 281.0, 268.5 |
| box B, off (A1, A2) | 64.3 / 62.5 | 56.8 / 48.2 | 39.45, 39.62 | 73.01, 72.82 | 282.0, 268.6 |
| box B, on (B1, B2) | 74.7 / 69.9 | 54.4 / 54.8 | 39.47, 39.47 | 73.09, 73.06 | 272.9, 279.1 |

(`lc32` cells: the pair of runs pooled, cold and cached separately.) At 32K: box A +10.1 % (62.8 -> 69.1 tok/s,
tokens per step +12.8 %), box B +14.0 % (63.4 -> 72.2, +18.0 %). At 128K: box A +1.5 %, box B +4.6 %. The 128K cells
hold 3 requests each and are noisy: box A's cached cell fell 3.4 % and box B's cold cell 4.2 %, while the other cell
on each box rose (box A cold +8.2 %, box B cached +13.5 %).

The gate: in all four arms on both boxes every one of the 73 speedbench full requests took the same number of verify
steps as the first off arm, and greedy outputs were identical on 73 / 73 speedbench and 30 / 30 warm prompts. With the
gate, speedbench never reaches the copy; its speed differences are run-to-run timing.

Agent mode (speedbench agent histories are 5-16K tokens, so the 8K gate admits part of them). Pooled over the turns of
8K-16K tokens (sum of tokens over sum of verify steps; 36 turns at 160 tokens, 18 at 512):

| | box A off | box A on | box B off | box B on |
|---|---|---|---|---|
| 160-token turns, tokens per step | 4.03, 3.93, 3.72, 3.61 | 3.90, 3.89 | 3.96, 3.80 | 3.81, 3.92 |
| 512-token turns, tokens per step | 3.94, 3.95, 3.95, 4.10 | 4.12, 4.15 | 3.77, 3.75 | 3.82, 3.69 |
| 160-token turns, end-to-end s (normalized to the first off arm's tokens) | 215.8, 215.5, 217.2, 218.2 | 217.5, 217.0 | 214.6, 216.1 | 218.5, 216.9 |
| 512-token turns, end-to-end s | 181.9, 186.3, 185.4, 183.4 | 188.3, 185.3 | 188.5, 189.7 | 188.7, 187.4 |

(Box A's off column includes the two off arms of the gate screen below.) Tokens per step with the copy stays inside the
off arms' range; end-to-end is +0.3 / +1.4 % on box A and +1.1 / -0.6 % on box B, against off-arm spreads of 1.2 % and
2.4 % on box A. The gate stays at 8,192 tokens.

Memory (box B, `KVS293`): thinking c24 (48 x 2,048 tokens) together with a cold 240K-token needle, worst MemAvailable
4.9 GiB, 48 / 48 completed, needles 128K and 240K PASS. Agent loops (24 x ~72K-token contexts, two turns): box A 24 / 24
turn-2 prefix-cache hits, time to first token 3.0 s, turn-2 wall 217.2 s (`AG293-s05`); box B 24 / 24, 2.9 s, 214.5 s,
worst MemAvailable 5.2 GiB (`AG293-s06`).

Gate screen (box A, report only): the same tree with `DFLASH_PCOPY_MIN_CTX=16384` (C1, C2) against off (A3, A4),
each warm, `lc32` and agent mode. `lc32` at 32K: off 64.1 / 65.1 tok/s (2.98 / 3.05 tokens per step), 16K gate 71.3 /
71.0 (3.43 / 3.35), +10.2 %; at 128K off 53.1 / 51.5, 16K gate 55.4 / 49.5 (flat within the cell noise). No agent
turn reaches 16K tokens (the longest is 15,873), so under the 16K gate the agent-mode arms run exactly the off path;
their end-to-end times (raw sums of time to first token and decode) still spread from 215.2 to 223.2 s at 160-token
turns and from 185.6 to 195.0 s at 512-token turns, about 4 % between arms of identical behaviour. That is the noise
floor of agent-mode end-to-end against which the 8K gate's +0.3 to +1.4 % was read; the gate stays at 8,192 tokens.

## v2.9.4: the prefix tail defers to a draft-tail snapshot (2026-11-25)

The prefix tail (v2.8.1) and the draft-tail snapshots (v2.9) were applied independently: a cached prompt of at most
4,096 tokens had its last 2,048 tokens re-prefilled by the target even when a snapshot already held the drafter's window
for that prefix. In the v2.9 screen, snapshots alone gave speedbench long context +14.4 % against +6.7 % with both.
`DFLASH_TAIL_DEFER=1` checks the snapshot slots before the tail bounds a request: a snapshot of the same prefix (tokens,
cache salt, extra key) that ends past the tail's bound and whose window covers every position the tail would recompute
is reserved, and the request keeps the stock prefix bound, capped at the snapshot's end. The radix lookup then decides
the reusable prefix; the reservation (slot version, content digest, end, salt / extra key, snapshot generation) is
checked after the lookup and again in the drafter's floor hook before the copy, and the ring floor is published only
after the copy is enqueued. A slot that changed in between is never restored: the request keeps its cache hit and the
drafter starts from the ordinary snapshot search. Ring floors are page-aligned (128 tokens), so with the switch on a
slot holds one extra page starting on a page boundary; the restored window then starts on the page the tail would have
recomputed from. Only SGLang's device-only radix cache qualifies (storage-backed trees stage loads outside the lookup),
only a single-rank process (ranks would decide on their own slots), and never a logprob request.

CPU checks: a replay of 400 random repeated-prompt steps through the overlay gives prefixes, ring floors, ring contents
and counters identical to v2.9.3 with the switch off; with it on, 65 % more prefix tokens come from the cache and every
drafter row read is the right position's.

Speed (box B, on, off, off, on; warm, speedbench full, 32K / 128K decode, agent mode):

| | on | off | off | on |
|---|---:|---:|---:|---:|
| speedbench full, all prompts, tok/s | 73.31 | 72.92 | 72.93 | 73.11 |
| long context, tok/s | 75.01 | 73.39 | 73.52 | 74.94 |
| long context, tokens per verify step (pooled) | 3.083 | 2.972 | 2.972 | 3.083 |
| long context, median time to first token, s | 0.189 | 0.448 | 0.443 | 0.189 |
| single-request step (median of 73), ms | 39.05 | 39.62 | 39.58 | 39.17 |

Agentic, code, prose, structured and multilingual stay within 0.4 % of the off runs. Speedbench agent mode is
unaffected: its histories are 5-16K tokens, all above the tail's 4,096-token limit, so the deferral never runs (end to
end 217.4 / 218.0 s on against 220.9 / 215.0 s off at 160-token turns). Decode at 32K and 128K was repeated to four runs
each way because the first two on runs read lower at 32K: 32K on 67.4 / 68.2 / 68.4 / 73.1 tok/s, off 70.0 / 70.2 / 63.4
/ 69.4; 128K on 55.2 / 52.0 / 51.1 / 54.7, off 58.0 / 53.8 / 50.0 / 56.8. The cell is flat: its outputs differ even
between two off runs, and same-prompt step counts range 100-215.

Greedy: the on runs differ from the off runs on 7 / 73 speedbench and 3 / 30 warm prompts, the same prompts in both on
runs (off against off: identical). A cached prefix replaces a recomputed one, the prefill floating-point-order class.
Quality bracket (box A, inkq2, off / on / off / on / off): totals 559 / 559 / 557 / 561 / 557 of 575 (gsm8k 392 / 391 /
389 / 391 / 388 of 400, long-context 100 / 100 everywhere, qeval 67 / 68 / 68 / 70 / 69 of 75); flips against each off
run 4-10 (budget 18), McNemar p >= 0.219; no drop.

Agent loops (box A, 24 x ~72K contexts, two turns; two pairs, the second in reverse order): 24 / 24 turn-2 hits in all
four runs (time to first token 2.88-2.95 s). Only the short first phase (~1.1K-token cached prompts) can defer: its
turn-2 time to first token was 4.84 / 4.81 s on against 5.21 / 5.23 s off, and the phase took 81.6 / 79.4 s against
87.3 / 81.8 s (83.0 s on v2.9.3, `AG293-s05`).

Memory (box B, thinking c24 with a cold 240K needle, `KV294`): worst MemAvailable 4.1 GiB, 48 / 48 completed, needles
128K and 240K PASS. The global-layer KV pool is sized at start-up from the memory free at that moment, and the size
varies between boots of the same configuration: over 40 boots on box A it ranged from 2,289K to 2,377K tokens, and
MemAvailable when the server is ready ran inversely, 7.3 to 4.3 GiB. A large-pool boot lowers the worst case by up to
~1 GiB on its own, with or without this change; compare worst cases at matched pool sizes. In the bracket arms at
matched pools (2,338K-2,343K tokens) the worst MemAvailable was 4.8-4.9 GiB with and without the deferral. In the agent
loops the first pair showed a large gap (on 3.9 GiB at a 2,354K-token pool, off 5.1 GiB at 2,325K); the reversed
repeat came out the other way (on 5.1 GiB at 2,334K, off 4.6 GiB at 2,351K). The drop from the post-start plateau to
the worst point, which removes the pool size, was 1.1 / 0.5 GiB on, 0.8 / 0.6 GiB off and 0.7 / 0.6 GiB on v2.9.3; it
happens once, when the first long prompts prefill (the deferral does not act on them), and stays flat afterwards.

Known limitation: a waiting request that holds a snapshot reservation but is not admitted in that scheduling round
keeps the slot pinned for one forward, so one capture into that slot can be skipped. It costs at most one missed
snapshot and never restores a wrong window.

## v2.9.5: one set of page candidates for the packed store's verify settle (2026-11-26)

After each DFlash2 verify step the 8 packed global visits settle the step's tentative writes: trim the rejected rows,
then commit the pages the accepted rows completed. In v2.9.4 each visit first rebuilt the same page candidates from
the verify locations (11 small ATen launches, including a device assert), then ran its own trim and commit kernels:
112 launches per step, about 0.48 ms of the post-verify eager region at c1 in the v2.9.3 profile. The DSpark path has
shared the candidates since v2.7.3; the DFlash2 hook did not.

Two variants, both exact:
- `shared` (the default): the candidates are built once and handed to every visit, 35 launches per step.
- `1` (opt-in): also the trim and commit kernels run once over all visits (`device.LaneGroup`: each visit's buffers
  are addressed from the first visit's by a fixed offset table, built when the pool is created), 14 launches.

GPU check (`tools/gpu_check_settle_lanes.py`, box B, no server): copies of the static pool's packed store get the same
verify steps from CUDA graphs (tentative writes of all 8 visits with different values per visit, packed attention of
all 8), then the eager settle with random accepted lengths, every third step ragged. Arms: v2.9.4's own kv12s code
(loaded from the previous release), per visit, shared and batched.

| case | requests | store bytes | attention | accepted rows |
|---|---|---|---|---|
| plain values | 1, 4, 8, 24, 48 | identical in all four arms | identical, and equal to the stock BF16 kernel | equal to the written K/V |
| page-heads spilling, side arenas exhausted | 1 | identical | identical | equal |
| page-heads spilling, side arenas exhausted | 4 | differ in slot ids (claim order, also between per-visit runs) | identical | equal, no row lost |

Eager settle of 8 visits, every request sealing a page (us per step):

| requests | per visit | shared | batched |
|---|---:|---:|---:|
| 1 | 645 | 322 | 115 |
| 8 | 839 | 541 | 450 |
| 24 | 1,532 | 1,222 | 1,172 |
| 48 | 2,557 | 2,247 | 2,230 |

Serving, ABBA per box (greedy warm = speedbench quick, 30 prompts; `dtp` = decode table, prose and code, c1-c16;
speedbench full = 73 prompts, c1). Single-request step = median per-request `step_ms`, the two arms on each side
averaged per prompt; every pair had identical greedy outputs and verify-step counts.

| | box | single-request step, ms | decode table vs off |
|---|---|---|---|
| batched (`1`), on / off / off / on, speedbench full | B | 38.80 -> 38.43 (-0.31; 65 of 73 faster) | code c4 -5.1 % (off spread 1.8 %), other cells within |
| batched (`1`), on / off / off / on, warm | B | 37.83 -> 37.45 (-0.36; 30 of 30) | |
| batched (`1`), on / off / off / on, warm + `dtp` | A | 37.69 -> 37.38 (-0.27; 30 of 30) | c4 code -4.1 %, prose -3.0 %; c8 prose -1.4 %, code -0.9 % (off spreads 0.1-0.5 %); geomean -1.0 % |
| shared, shared / off / off / shared, warm + `dtp` | A | 37.62 -> 37.37 (-0.25; 29 of 30) | every cell at or above the off range; geomean +1.3 % |

The batched loss at 4 and 8 requests is not a slower step. From the engine's decode log (one line per 40 steps, about
360 steps at 4 requests per arm), box A at 4 requests: step about 73 ms on vs 78 ms off, accepted tokens per step
2.78 vs 3.07, mean verify width 4.5 vs 4.7. The verify-width cap reads only draft confidence, so the batched arms
decoded different greedy texts with several requests in flight, and fewer of their draft tokens were accepted. Greedy
texts with several requests in flight also differ between runs of one setting (box B prose c16: one off arm stopped a
request 3 tokens early, the other did not), so this does not by itself show a store difference; the GPU check found
none. The shared arms run the per-visit kernels and lost nothing; `shared` is the default and `1` stays opt-in.

Greedy dumps with several requests in flight (box A, `tools/lc_dump_conc.py`: the 30 speedbench quick prompts, 160
tokens, top-2 logprobs, two rounds each, at 4 and at 8 in flight; arms `1`, `0`, `0`, `1`, `results/KV295lc-*-s05/`)
do not separate `1` from `0`, because no setting reproduces its own greedy texts there. At 4 in flight, 9-12 of 60
requests were identical between any two arms (off against off: 10), and within one arm only 4-5 of 30 were identical
between its two rounds; at 8 in flight, 2-7 of 60 between arms (off against off: 3) and 1-2 of 30 within an arm. The
texts part at a median of about 50 tokens in, at tokens whose top two candidates were close (median gap 0.18 nat at 4
in flight, 0.15 at 8; about 92 % under 0.5 nat): with several requests in flight the greedy text depends on how the
requests share each step. The dumps show no difference beyond the off arms' own, and the GPU check found none in the
store; `1` stays opt-in because of the decode-table loss, not because of a difference found.

Memory (box A, `KV295mem`, booted with `1`; `shared` allocates the same less a 768-byte table): thinking c24
(48 x 2,048 tokens) together with a cold 240K-token needle, worst MemAvailable 4.9 GiB (v2.9.3, same box and recipe:
4.6 GiB), 48 / 48 completed, 337.3 tok/s aggregate; needles 128K and 240K pass.

## v2.9.6: MXFP8 context projections for the DFlash2 drafter (2026-11-26)

After each verify step the DFLASH worker appends the committed rows' target features to the drafter's context window,
eagerly. That append is the context `fc` (four 2,304-wide target features -> 2,304, then `hidden_norm`), then the five
layers' stacked K/V projection (2,304 -> 10,240), then a fused K RMSNorm + RoPE and one prefix-valid ring write per
layer. The two BF16 GEMMs read 89.7 MB per step. In the v2.9.3 profile the region cost 0.466 ms at greedy c1
(1.2 % of the step): 230 us for the `fc` and 207 us for the K/V GEMM.

Two ways to make it cheaper:
- Run the append inside a CUDA graph. SGLang already has a flag for that (`SGLANG_OPT_DFLASH_AUX_GRAPH`), which was
  screened flat earlier. The kernels stay the same, so only the region's 11 launch gaps (0.025 ms) could go.
- MXFP8 copies of the two weights on the b12x GEMM the drafter's layers already use (`mx8w`); built as `DFLASH_CTX8`.

`inkling_dfbytes/ctx8.py` wraps `DFlashDraftModel.project_target_hidden` and `FusedKVMaterializeHelper.materialize`.
On the MXFP8 path the latter runs the stock body line by line with only its `torch.mm` replaced; the three stock
functions it relies on are pinned by source hash. Rows 1-256 run on b12x; larger calls (long prefills) and any input
the quantizer cannot take run the stock function.

GPU check (`tools/gpu_check_ctx8.py`, box B, real drafter weights, no server):

| check | result |
|---|---|
| ctx8 append with its GEMM swapped back to `torch.mm`, 1 / 8 / 32 / 64 / 192 / 256 rows | K/V bytes identical to stock |
| 257 rows | stock path, bit for bit |
| b12x writes the stock workspace rows in place (1 / 3 / 127 / 129 / 255 rows) | yes |
| CUDA graph replay against eager, repeat runs | bitwise equal |
| b12x kernels compiled after the start-up warm-up, rows 1-256 | none (4 tile plans) |
| fc output against BF16 (relative error / lowest row cosine) | 3.6-3.8 % / 0.9992 |
| K and V end to end against BF16 | 5.1-5.5 % / 0.998 |

Eager fc + K/V append with ring writes, L2 flushed (ms):

| rows | BF16 | MXFP8 |
|---:|---:|---:|
| 8 | 0.447 | 0.269 |
| 32 | 0.467 | 0.275 |
| 64 | 0.467 | 0.283 |
| 192 | 0.527 | 0.346 |

Serving, box B, two ABBA runs (on, off, off, on). The first ran speedbench full, the decode table and `longctx`
(lc32); the repeat ran the decode table and lc32. Both used the v2.9.5 candidate tree with `KV12S_SETTLE=1` on both
sides.
- Single-request decode step, paired per request over the 73 speedbench full requests: -0.195 ms median, -0.182 ms
  mean, 54 / 73 faster. Speedbench full: 74.02 / 74.48 with the lever against 73.62 / 73.66 without; tokens per step
  2.849 against 2.842.
- Greedy outputs:
  - Speedbench quick: 30 / 30 identical.
  - Speedbench full: 68 / 73 identical; the same 5 prompts (code-06, longctx-01, -02, -07, -08) differ in both on
    arms. Off against off and on against on are 73 / 73.
  - The drafter changes which tokens are accepted, the verify blocks then fall elsewhere, and a near-tie can resolve
    the other way: the class the verify-width cap and the draft-tail snapshots also show.

Decode table, four arms each side (tok/s; c1 per stream, c4-c16 aggregate):

| | on, mean | off, range | on vs off mean |
|---|---:|---:|---:|
| code c1 | 80.1 | 79.2-79.5 | +1.0 % |
| code c4 | 140.2 | 133.4-142.1 | +0.9 % |
| code c8 | 205.7 | 202.6-212.4 | -1.2 % |
| code c16 | 310.1 | 291.8-308.2 | +2.6 % |
| prose c1 | 66.9 | 66.5-66.8 | +0.3 % |
| prose c4 | 141.4 | 138.5-143.9 | +0.1 % |
| prose c8 | 207.3 | 203.8-214.1 | -0.4 % |
| prose c16 | 296.9 | 293.9-297.5 | +0.6 % |

lc32, four arms each side:
- Tokens per step: 3.037 against 3.138 at 32K and 3.260 against 3.323 at 128K.
- Decode tok/s: 67.8 against 68.8 at 32K and 54.6 against 55.1 at 128K.
- All are inside the off range; per-arm spread is about ±5 %.
- lc32 writes new documents in every run, so these do not pair. To check whether the drafter loses acceptance at long
  context, box A ran the same 32K and 128K sizes with fixed documents and a step capture (4 documents each, cold and
  cached, 1,024 tokens, greedy, full-width verify; on / off / off / on; `tools/ctx8_lcpair.py`).
- Repeat runs of one setting were bit-identical. Between settings, the requests are paired over the tokens both arms
  generated identically (up to the first differing token):

| | paired requests | tokens | tokens per step, on | off | on / off | first-slot agreement |
|---|---:|---:|---:|---:|---:|---:|
| 32K | 8 / 8 | 4,401 | 4.0525 | 4.0525 | 1.000 | 97.5 % |
| 128K | 8 / 8 | 3,693 | 3.4482 | 3.4450 | 1.001 | 98.2 % |

- Over whole requests the lever arms had more tokens per step: 3.669 against 3.518 at 32K and 3.561 against 3.379 at
  128K. Those texts part at the first differing token, though: 6 of 8 requests per length, after 27 to 832 tokens.
- On the 30 speedbench quick prompts (steplog pair, box A), drafted tokens agree on 97.1 / 93.1 / 90.6 / 87.5 / 85.4 /
  82.4 / 79.8 % of slots 1-7. Accepted drafts per matched step: 2.137 with the lever against 2.136 without.

Quality: the inkq2 bracket (box B, 575 items: gsm8k 400, long-context retrieval 100, qeval 75;
A L A L A, A = `DFLASH_CTX8=0`, L = `DFLASH_CTX8=1`) passes under protocol v2.

| arm | total | gsm8k | lc | qeval | flips vs A1 / A2 / A3 | McNemar p (lowest) |
|---|---:|---:|---:|---:|---|---:|
| A1 | 557 | 388 | 100 | 69 | | |
| A2 | 558 | 390 | 100 | 68 | | |
| A3 | 561 | 392 | 100 | 69 | | |
| L1 | 557 | 391 | 100 | 66 | 10 / 9 / 8 | 0.289 |
| L2 | 557 | 390 | 100 | 67 | 10 / 7 / 10 | 0.344 |

The baselines flip 7-10 items between themselves (budget 18). Both lever runs sit at the low end of the baselines'
range, and qeval is 1-3 below them; every criterion passes. Results: `results/QB296A1-s06/` ... `results/QB296A3-s06/`
(the compare output is in `results/QB296L2-s06/inkq2-compare.txt`).

Memory (box B, `MEM296`, lever on):
- Thinking c24 (48 x 2,048 tokens) together with a cold 240K-token needle: lowest MemAvailable 4.2 GiB, 48 / 48
  completed, 334.4 tok/s aggregate. Needles 128K and 240K pass.
- The KV pool sized 2,341,632 tokens (v2.9.4 on the same box: 2,351,872).

## SWA KV packing (closed experiment)

Closed 2026-11-26.

The packed global KV store's exact 12-bit format (sign + mantissa byte, 4-bit exponent code, escapes, raw overflow)
was extended to the target's 40 sliding-window visits.
- **Written through.** Every write packs at once, and a writer drops the escapes of the rows it rewrites; there is no
  staging, seal or settle.
- **Readers.** The stock windowed compact kernel (up to 32 verify rows) and the in-kernel-bias verify kernel (above
  32) are used as they are, with only the K/V tile load replaced.
- **Prefill.** Each request's window span is decoded into the existing BF16 prefix scratch.
- **Capacity.** Planned at +4.5-4.8 % global KV tokens (window tokens at 124,320 instead of 163,840 bytes).
- **Speed.** Expected from the 24 % fewer K/V bytes in the verify window reads, the larger share at thinking c8.

GPU check (`gpu_check_swa12s.py --quick`, box B, no server; `results/GKSWA-s06/`): everything bitwise.
- All 65,536 BF16 words, through the packed path and the overflow path.
- The writer against its Torch model, byte for byte.
- Both verify readers against the stock kernels: partials and gate/sink output, with overflowed pages and chunks of
  more than 16 escapes.
- Malformed descriptors are counted for the abort path; prefill warm-up dummies count nothing.

The verify visit itself (store of the 8-row block + attention + gate/sink, CUDA-graph replay, L2 flushed before each
sample) was slower or equal at every size:

| requests | full window (1,031 keys): BF16 / packed us | speed | half window: BF16 / packed us | speed |
|---:|---:|---:|---:|---:|
| 1 | 43.8 / 58.6 | 0.747x | 33.6 / 50.8 | 0.661x |
| 4 | 139.9 / 138.9 | 1.007x | 71.3 / 89.8 | 0.795x |
| 8 | 227.1 / 227.0 | 1.001x | 149.0 / 174.0 | 0.856x |
| 16 | 386.8 / 473.1 | 0.818x | 214.0 / 269.9 | 0.793x |
| 24 | 531.5 / 633.9 | 0.838x | 335.9 / 487.1 | 0.690x |
| 48 | 1,027.7 / 1,219.3 | 0.843x | 621.2 / 820.1 | 0.757x |

The time box required at least 1.12x at 8 requests and no more than 3 % loss at 1 request; the lever stopped there,
before any serving run.
- **Why.** The BF16 window read runs at 97-197 GB/s, well under the ~235 GB/s the packed global reader approaches
  on long contexts, so fewer bytes do not pay for the decode on a 1,024-key window.
- **Where the time goes.** Per 32-key chunk the decode adds a chain of dependent loads (page, overflow reference,
  chunk offsets, escape words) and the escape patch.
- **Same limit as before.** This is the limit the global 12-bit reader already noted for short contexts: its gain
  needs long contexts (1.15-1.27x at 128K).
- **Prefill cost.** The window gather would also have cost 13.4 ms per 8K prefill chunk.
- **Status.** The build stays out of the release; `KV12S_SWA` does not exist in the recipe.
