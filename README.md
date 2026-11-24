# Inkling-2-Mini on one NVIDIA DGX Spark

Serving [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on a single DGX Spark
(GB10, sm_121, 121.6 GiB unified memory) with the model's own NVFP4 / MXFP8 weights and the public DFlash2 draft model
[incoai/Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2), on SGLang, plus a set of
import-time speed overlays that leave the target model's output distribution as it is.

**Status (2026-11-24, v2.8.1):** basic tests on one box; full two-box qualification follows in v2.8.x.

One default profile: SGLang `lmsysorg/sglang:inkling-2-mini`, pinned by digest, drafting with DFlash2, a 5-layer
block-diffusion draft with a 2,048-token draft window and a 16-candidate selector, pinned at revision `4f9088ea`.
`scripts/launch.sh` checks the drafter's sha256 on the host and again inside the container on every start, and the
drafter's SGLang compatibility patch runs only inside the throwaway (`--rm`) serving container. Nineteen overlay
packages (`overlay/`) run on top of it by default: drafter byte cuts (MXFP8 drafter layers, an FP8 64K hot-vocabulary
plane for the candidate projection), the prefix tail that keeps the drafter's window filled on prefix-cache hits
(v2.8.1, see "Prefix tail"), the certified greedy LM head on verify steps, the packed 12-bit global KV store,
the exact sampled-verification fixes, block verification and a fused accept on sampled steps, a calibrated
verify-width cap with dead-row expert skip, and the target-side kernel, prefill and memory levers of v2.7.2. The
target distribution is unchanged. `SPEC=dspark` runs the model's own DSpark draft instead (the v2.7.2 profile, see
"Fallback: the DSpark profile").

Long context: at 32K context the DFlash2 profile decodes about 11-17 % slower than v2.7.2's DSpark profile at c1
(v2.8.1 64.3, v2.8 60-63, v2.7.2 72 tok/s). Its acceptance falls with context (3.10-3.17 vs 3.36 tokens per step at
32K), because the drafter sees only a 2,048-token window. At 128K both decode at about 50 tok/s. For mostly long-context single-user
work, `SPEC=dspark` is the faster choice.

Basic tests on one box; full two-box qualification follows in v2.8.x. The v2.8.1 column is one GB10 Spark on
2026-11-24 with the release defaults (`SPEC=dflash2`, verify-width cap on, `CTX=262144`); the stock and v2.7.2
columns are the earlier two-box medians. v2.8.1 changes only prefix-cache hits on prompts of at most 4,096 tokens:
speedbench full, decode at 32K / 128K and the worst case were measured on v2.8.1, the other v2.8.1 cells are the
v2.8 runs.

| | stock SGLang cookbook cell (DSpark, mem 0.62) | v2.7.2 | **v2.8.1** |
|---|---:|---:|---:|
| speedbench quick, 30 prompts, c1, tok/s (prose) | 41.86 (33.18) | - | **70.61 (65.67)** |
| speedbench full, 73 prompts, c1, tok/s | 40.6 | 63.8 | **72.86** (v2.8: 69.59-69.66) |
| decode, prose, tok/s at c1 / c4 / c8 / c16 | 33.2 / 75.2 / 117.0 / 186.2 | 52.6 / 113.3 / 164.6 / 244.1 | **65.1 / 140.5 / 203.5 / 288.3** |
| decode, code, tok/s at c1 / c4 / c8 / c16 | 43.6 / 87.2 / 134.8 / 221.3 | 62.2 / 124.8 / 186.6 / 268.2 | **79.0 / 141.6 / 211.3 / 298.2** |
| thinking c8 / c24, aggregate tok/s (48 x 2,048 tokens, effort high, T=0.8) | - | 235.3 / 438.8 | **277.4 / 515.5** |
| cold prefill, 8K tokens, tok/s | 3,399 | 5,439 | **5,383** |
| decode at 32K / 128K context, c1, tok/s (512 tokens) | 55.1 / 37.1 | 71.9 / 49.0 (one box) | **64.3 / 50.7** |
| KV pool, global-layer tokens | 843K | 2.13-2.17M | **2.37-2.42M** |
| MemAvailable after start / worst case (thinking c24 + cold 240K needle) | ~40 GiB / - | 6.4-7.7 / 5.4 GiB | **4.5-5.5 / 4.1 GiB** |

Speedbench full (73 prompts x 256 tokens) is the release-to-release comparison; quick (30 prompts x 160 tokens) is a
different prompt set and length, used here for the cap on / off screens (the stock cell is the stock engine's screen on
one box, docs/experiments.md, "Screening results"). Do not compare across the two rows. In full mode v2.8.1 leads
v2.7.2 in every category except long context, where it is on par (73.53 vs 75.1). v2.8 trailed there (-16 %) and on
agentic prompts (-4 %) because the drafter drafted from an empty window after a prefix-cache hit, which speedbench
causes on every prompt; v2.8.1 fixes that (see "Prefix tail").

Not measured for v2.8 / v2.8.1 (v2.8.x qualification): structured / JSON decode and c2, thinking c1 / c4 with the
cap on, 200K-512K context, the quality guard and the sanity suite.

## Speed

Measured 2026-11-24 on one Spark per run, GPU clock capped at 2200 MHz by the host's own timer, fresh start of the
container, release defaults. Speed runs on box A (speedbench full: `results/V28sbf-s05/`; `results/V28M1v28-s05/`, cap
off: `results/V28M1off-s05/`, NOEXP off: `results/V28M1nonx-s05/`), memory and the worst case on box B
(`results/V28kv-s06/`), long context on box A (`results/V28lc-s05/`, v2.7.2 on the same box: `results/R272lc-s05/`;
`SPEC=dspark`: `results/V28dsp-s05/`). The stock engine was measured on both boxes with the same harness
(`results/F1-sglang-base-s05/`, `results/F2-sglang-base-s06/`).

### speedbench

[speedbench](https://github.com/knapcio/speedbench) 0.3 (`6733653`), full mode: 73 frozen prompts in six categories,
c1, temperature 0, 256 output tokens, reasoning effort low (the model always thinks; its reasoning tokens count as
decode tokens). Decode tok/s, geometric mean per category; 0 failed prompts.

| | prose | code | structured | agentic | multilingual | long context | all 73 prompts |
|---|---:|---:|---:|---:|---:|---:|---:|
| **v2.8.1** (box B) | **64.92** | 81.88 | 77.46 | 82.74 | 57.46 | 73.53 | **72.86** |
| v2.8.1 tokens per step (median) | 2.50 | 3.62 | 3.40 | 3.38 | 2.23 | 3.27 | 2.80 |
| v2.8, same box (box B, `MEM_FRACTION=0.85`) | 64.57 | 79.84 | 77.10 | 73.44 | 57.19 | 63.09 | 69.59 |
| v2.8 (box A) | 64.66 | 79.89 | 77.14 | 73.57 | 57.21 | 63.12 | 69.66 |
| v2.7.2 (two-box median) | 53.6 | 72.1 | 68.0 | 76.7 | 43.7 | 75.1 | 63.8 |
| stock engine | 32.6 | 48.0 | 44.3 | 49.3 | 25.3 | 51.0 | 40.6 |

v2.8.1 is the release defaults on box B (`results/V281s-s06/`, `MEM_FRACTION=0.89`, limit 4,096), median time to
first token 162.1 ms; v2.8 on the same box is `results/N81sbfc-s06/`, v2.8 on box A `results/V28sbf-s05/` (median
time to first token 162.8 ms; v2.7.2 156.5-157.3 ms, stock 184 ms). Long context and agentic gain; prose, structured
and multilingual move by under 0.5 %. longctx-02 (4,930 tokens) is above the 4,096 limit and runs as on v2.8. Against
the published v2.7.2 (same harness, but v2.8.1 is one box and v2.7.2 a two-box median): all prompts +14.2 %, agentic
+7.9 %, long context 73.53 vs 75.1 (-2.1 %, on par). With the radix cache off, the investigation runs reached 72.82
(long context 75.00, agentic 82.42; docs/experiments.md). Per prompt, the tail lifts tokens per step from 2.84 to 4.20 on
longctx-01 and from 10.8 to 17.0 on agentic-01. Counted end to end (time to first token plus decode, 256-token
outputs): long context +6.2 %, agentic +1.2 %, code +1.6 %.

### Prefix tail

`SPEC=dflash2` prefix tail: on a prefix-cache hit, a prompt of at most 4,096 tokens (`DFLASH_PREFIX_TAIL_MAX_INPUT`) has
its last 2,048 tokens (`DFLASH_PREFIX_TAIL`) prefilled again so the drafter sees them. The first hit of a prompt
re-prefills it fully, because the hybrid state cache resumes only at checkpoints. Cost: +0.1-0.5 s TTFT. It pays back
after ~25-450 output tokens on prompts the drafter copies from (speedbench long context +16 % decode tok/s, +6 % end
to end). Prompts above the limit keep the cache as before. The first turn above the limit after a bounded turn
re-prefills ~2K tokens once. Raise the limit (e.g. 8192) for long-output or document-QA traffic; at a raised limit,
agent loops with short turns on 5-8K histories lose ~5-9 % end to end. `DFLASH_PREFIX_TAIL=0` turns it off.

Measured: median time to first token on cache hits rises from 186 to 540 ms on speedbench's long-context prompts and
from 170 to 235 ms on its agentic prompts; later hits of the same prompt reuse a state checkpoint planted at the
bound and redo only ~2K tokens. Cached hits at 32K and 128K are above the limit and stay at 0.24 / 0.46 s on v2.8.1 (bounded, they would
take 6.38 / 35.3 s); the drafter's window does not reach far context, and decode at 32K / 128K does not gain even with
the radix cache off. Multi-turn
agent sessions (speedbench agent mode, 8 conversations of 4-8 turns on 5-12K-token histories) run exactly as on v2.8
at the default limit; raising it to 6,144 / 8,192 / 12,288 costs +5.0 / +8.7 / +14.8 % end-to-end time at 160-token
turns and is about break-even at 512-token turns (decode +3-8 %). The tail recomputes positions a cold prefill would
compute (prefill floating-point order only) and allocates no memory; it applies to DFLASH only, never to streaming
sessions, and is pinned to the image's source (a different image refuses to start; `DFLASH_PREFIX_TAIL=0` runs
without it).

Quick mode, cap on against cap off (30 prompts, 160 output tokens, same settings otherwise; box A):

| | prose | code | structured | agentic | multilingual | long context | all 30 prompts | tokens / step | step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **v2.8** | **65.67** | 80.16 | 77.26 | 75.72 | 64.47 | 63.00 | **70.61** | 2.741 | 40.5 ms |
| v2.8, `DFCAP=off` | 55.88 | 73.69 | 69.00 | 68.21 | 56.50 | 56.02 | 62.49 | 3.000 | 49.5 ms |

Tokens per step and step time are medians. A second boot on box B gave 70.17 (prose 65.48).

The cap verifies fewer rows per step (2.741 instead of 3.000 accepted tokens per step) at a 9 ms shorter step.

### Decode by prompt type and concurrency

`bench/decode_table.py`: speedbench's prompts split by type, temperature 0, 256 new tokens, reasoning effort none (the
model still writes a short thought). c1 is the decode speed of one stream (median of three requests); above c1 the
aggregate over the level's wall time.

| | c1 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|
| prose, v2.8 | **65.1** | **140.5** | **203.5** | **288.3** |
| prose, v2.8 `DFCAP=off` | 55.4 | 122.8 | 190.6 | 292.3 |
| prose, v2.7.2 | 52.6 | 113.3 | 164.6 | 244.1 |
| prose, stock engine | 33.2 | 75.2 | 117.0 | 186.2 |
| code, v2.8 | **79.0** | **141.6** | **211.3** | **298.2** |
| code, v2.8 `DFCAP=off` | 67.7 | 119.2 | 187.9 | 297.5 |
| code, v2.7.2 | 62.2 | 124.8 | 186.6 | 268.2 |
| code, stock engine | 43.6 | 87.2 | 134.8 | 221.3 |

v2.7.2 and the stock engine are two-box medians from earlier runs; v2.8 is one box. Against v2.7.2, prose is +24 % at
c1-c8 and +18 % at c16, code +27 % at c1 and +11 to +13 % at c4-c16. The cap is on up to 8 requests: at c16 both v2.8
rows verify every row and agree within noise.

### Thinking workloads

`bench/think_load.py`: seeded arithmetic puzzles, reasoning effort high, temperature 0.8, aggregate completion tok/s
with the median per-request decode in brackets. c8 and c24: 48 requests x 2,048 max tokens.

| | c1 | c4 | c8 | c24 |
|---|---:|---:|---:|---:|
| v2.8 | - | - | **277.4** (38.2) | **515.5** (24.0) |
| v2.8, earlier boot, cap off | 82.5 | 177.9 | 288.4 | 518.9 |
| v2.7.2, same protocol | - | - | 235.3 | 438.8 |

One run per cell (sampled). The aggregate depends on how long the completions run, so compare the per-request decode
where it is given; c8 and c24 vary ~5 % from run to run.

### Prefill and long context

speedbench `longctx` on box A: a generated document with a unique salt (no prefix-cache hit) plus a question; cold
prefill, then decode of 512 tokens at c1 at 32K and 128K context. v2.7.2 ran on the same box as the control (3 reps in
one run, `results/R272lc-s05/`); v2.8 ran two passes (`results/V28lc-s05/`; the second with 2 reps). Decode in tok/s
with the confidence interval in brackets.

| | v2.7.2 (DSpark), same box | **v2.8**, first pass | v2.8, second pass |
|---|---:|---:|---:|
| cold prefill 32K / 128K, tok/s | 5,162 / 3,719 | **5,132 / 3,714** | 5,083 / 3,738 |
| decode at 32K context | 71.9 [58.7, 88.1] | **63.1** [57.7, 69.0] | 59.9 [53.9, 66.5] |
| tokens per step / step ms at 32K | 3.36 / 47.4 | 3.10 / 48.3 | - |
| decode at 128K context | 49.0 [41.4, 57.9] | **50.2** [38.1, 66.1] | 49.7 [39.5, 62.5] |
| tokens per step / step ms at 128K | 3.20 / 60.8 | 2.68 / 59.0 | - |

v2.8.1 (release defaults, box B, `results/V281s-s06/`): decode at 32K 64.3 tok/s (3.17 tokens per step), at 128K
50.7 tok/s; cached time to first token at 32K / 128K 0.24 / 0.46 s (the prefix tail does not act above 4,096 tokens).
v2.8 cold prefill at 1K / 8K / 32K / 128K (second pass): 2,753 / 5,383 / 5,083 / 3,738 tok/s. Prefill runs on the
target-side levers only and is unchanged within noise (8K: 5,383 against the published v2.7.2 5,439).

Stock engine (two-box median): 3,399 tok/s cold at 8K, decode 55.1 at 32K and 37.1 at 128K. Decode at long context is
dominated by the global layers reading the whole context (4.3 GB of BF16 KV per step at 128K, at the memory-bandwidth
floor); the packed store reads the committed pages in an exactly coded 12-bit form (1.33x fewer bytes). The drafter
attends only to the last 2,048 tokens of the target features, so its acceptance falls with context: at 32K the
DFlash2 profile decodes about 12 % slower than v2.7.2's DSpark profile at c1 (60-63 vs 72 tok/s), and at 128K both
decode at about 50 tok/s. For mostly long-context single-user work, `SPEC=dspark` is the faster choice.

### Memory and start-up

Box B (`results/V28kv-s06/`), release defaults (`SPEC=dflash2`, `KV12S=1`, `KV12S_POOL=static`, `MEM_FRACTION=0.89`,
`SWA_RATIO=0.037`, `CTX=262144`):

| | |
|---|---|
| global-layer KV pool | 2,371,584-2,416,896 tokens (packed store; the drafter's KV ring is fixed at ~2.16 GB) |
| MemAvailable when the API is ready | 4.5-5.5 GiB |
| lowest MemAvailable with thinking c24 (48 x 2,048 tokens) alongside a cold 240K-token needle prefill | 4.1 GiB on v2.8.1 (`results/T81kv-s06/`; v2.8: 4.9 GiB); guard floor 3.5 GiB. The tail allocates no memory; the run generated 4 % more tokens than the v2.8 run |
| start-up to the first answer | about 2-4.5 min (131-275 s; the first start after an update includes JIT compilation) |

The pool differs between boots at the same fraction because the memory left after the weights and graphs differs.

### Quality checks

- Needles with the passphrase at 50 % depth at 128K and 240K tokens: PASS on v2.8 and v2.8.1.
- Prefix tail: greedy output with the tail is bit-identical to the same server with the radix cache off (43/43
  prompts with top-5 logprobs: 30 speedbench quick, 13 long-context and agentic). Against the default cached server
  it differs only by near-ties (13 long prompts: 3 identical, 9 top-2 swaps at gaps 0.006-0.25 nat, 1 three-way tie
  within 0.1 nat), exactly the differences the stock server shows between cache on and cache off
  (results/T81EX2nc-s06/, T81EX2tail-s06/, T81EX2ref-s06/).
- Sampled law: Monte Carlo tests through the engine's own DFLASH selector walk and sampled accept with `RSFIX` (chain
  and block verification at temperature 1.0 and 0.7, target top-k 5, and the same arms under the verify-width cap):
  about 1M steps per case, 0 invalid tokens, all within the law (docs/experiments.md).
- Greedy exactness of the cap: the cap changes the verify schedule, so a token can land in a different row of a
  differently placed verify block, and the floating-point reduction order changes. On 30 speedbench prompts, cap on
  against cap off: 28 / 30 greedy texts bit-identical; the other two first diverge at a top-2 near-tie (log-probability
  gaps 0.026 and 0.367 nat) after a gradual drift of at most 0.17 nat on the shared prefix. With the expert remap off
  on the same schedule the outputs are bit-identical. NOEXP on against off: greedy texts identical on 30 / 30,
  log-probabilities at rounding level (the MoE tile tactics depend on per-expert token counts).
- What "exact" means here: no lever changes the target model's weights or its output distribution. The drafter only
  proposes tokens; the target verifies them (greedy output is the target's argmax; sampled requests are accepted by
  SGLang's DFLASH accept, whose output law is the target's). The drafter byte cuts change proposals only: the
  selector's draft probabilities are computed from the same candidates the path is drawn from. The verify-width cap
  only decides how many draft tokens to verify, by a left-to-right stopping rule on the drafter's own probabilities
  (each row's width is decided before its token is drawn); rows the cap has cut get no experts and are never
  accepted. The certified LM heads return the exact argmax of the full head (proved per step; 0 mismatches in 34,808
  checked rows on DFLASH). The sampled-verification fixes (`RSFIX`) do not change the law in exact arithmetic; they
  stop FP32 rounding from emitting a zero-probability or out-of-range token. The prefix tail only recomputes prompt positions that a cold prefill
  would compute (prefill floating-point order only). The packed 12-bit KV decodes to the same
  BF16 bits. The kernel levers compute the same math, bit-identical where stated, otherwise with a different
  floating-point summation order (each checked against an FP32/FP64 reference to be no less accurate than stock). Any
  such change moves near-tie tokens on this model (its W4A4 activation quantisers amplify last-bit differences;
  equally correct implementations agree on ~95 % of tokens teacher-forced), so greedy texts are not byte-identical
  to the stock engine; the stock engine itself is deterministic from boot to boot.

## What the overlays do

The overlays cut bytes and kernel time per decode step without touching the target's math. The analysis behind them
was done on the DSpark profile: decode at c1 is bound by memory bandwidth (a step read ~13.4 GB: BF16 dense projections
2.8 GB over 48 layer applications, LM head 0.93 GB, DSpark draft ~1.65 GB, and the routed experts touched by the
8 verify rows, ~6.5 GB) and the stock engine ran at 1.08x the 240 GB/s floor with the GPU 99 % busy
([profile](docs/profile/PROFILE.md)). With several requests in flight the step is still mostly bytes (c8: 117 ms, of
which 76 ms routed experts), but kernels built for one row lose time there: cuBLAS streams the dense projections at
150-200 GB/s, and above 32 verify rows the attention left the split-KV kernel for padded FA4 with a sheared bias. At
long context the global layers' BF16 KV dominates the step (4.3 GB at 128K). Prefill is the opposite: the kernels run
near their peaks, but intermediates make round trips through memory (FP32 [T, 2304] activations between the
loop-message mixes, norms and residual updates), the relative position bias was materialised as a [t, 24, 1024]
tensor, and the stacked Q/K/V projection computed three times the rows it keeps
([prefill analysis](docs/prefill/PREFILL.md), [round-2 profiles](docs/experiments.md#round-2-v2)).

The levers are switched by environment variables of `scripts/launch.sh`. The overlays are import-time patches in
`overlay/` (`OVERLAY=0` disables them). Effects are from screens against same-box, same-day controls (speedbench
quick: 30 prompts x 160 tokens, c1, T=0; decode table; thinking load; at long context the step time = tokens per step
/ decode tok/s). Raw files: `results/screen/`, `results/screen-r2/`; details and rejected levers:
[docs/experiments.md](docs/experiments.md).

### Drafter and DFLASH levers (default profile)

| lever | switch (default) | effect, alone | output | credit |
|---|---|---|---|---|
| DFlash2 drafter: a 5-layer block-diffusion draft that proposes a block of 8 positions in one pass over a 2,048-token sliding window of target features (layer visits 5, 23, 41, 48), keeps 16 candidates per position, and a selector traces one path through them (7 drafts per verify step) | `SPEC=dflash2` (`--speculative-algorithm DFLASH`) | stock against stock DSpark: tokens per step x1.25, prose decode c1 / c4 / c8 / c16 +43 / +46 / +40 / +39 %, code x1.22-1.25, thinking c8 +19 % | draft proposals only; the target verifies every token | **incoai**, [Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) and its SGLang compatibility patch (Apache-2.0); block-diffusion drafting from the **DFlash** authors ([z-lab/dflash](https://github.com/z-lab/dflash)); SGLang's DFLASH worker |
| Drafter byte cuts: MXFP8 copies of the drafter's five layers' linears (the W8 kernels of the DSpark draft cuts), and an FP8 row-scaled copy of the target head restricted to the 65,536 most frequent ids for the selector's candidate projection (instead of the full 200,058-row BF16 head) | `DFLASH_BYTES=mx8,hot8`, `DFLASH_HOT_N` (65536) | speedbench quick +8.4 %, prose c1 +6.5 % (ABBA, one box) | draft proposals only (the selector's q is computed from the candidates the path is drawn from) | ours; the hot-vocabulary list is the v2.1 one (idea: FR-Spec, Zhao et al.) |
| Certified greedy LM head on DFLASH verify steps of <= 16 rows (c1 / c2): armed per step when the stock accept would take the plain argmax (no grammar, logprobs or logit adjustments) | `CERT_DFLASH=1` | c1 +2.8 % | identical argmax by construction (check mode: 34,808 rows, 0 mismatches) | ours (the v2 certified head, armed from the DFLASH worker) |
| Certified greedy LM head for 17-192-row DFLASH verify batches: the certified token ids go straight to the DFLASH accept | `CERT_WIDE_DFLASH` (on with `CERT_DFLASH`) | code c8 +3 %, no regression elsewhere | greedy texts identical on 30 / 30 prompts; per-row proof | ours (the v2.7 wide certified head) |
| Verify-width cap on DFLASH: per request and step, verify only the first L of the 8 rows; L grows left to right while the drafter's calibrated survival (per slot and selector-confidence bin, fitted on full-width DFLASH step logs) pays for the row; cut rows are remapped to the anchor row and get no experts at all; above 8 requests all rows are verified | `DFCAP` unset = `dfpolicy:/opt/inkling-overlay/inkling_vcap/policies/df-SLDF1-sm.json,bsmax=8`, `DFCAP_NOEXP=1`, `DFCAP_REMAP=1`; `DFCAP=off` disables it | release stack, cap on vs off, box A: speedbench quick 62.49 -> 70.61 (+13.0 %); prose c1 / c4 / c8 +17.5 / +14.4 / +6.8 %, code +16.7 / +18.8 / +12.5 %; c16 not capped | target distribution unchanged (a stopping rule on draft-side information; GPU Monte Carlo law test, chain and block verification, with wrong values on the cut rows); greedy texts differ only by reduction order (28 / 30 identical) | ours (port of the DSpark verify-width policy) |
| Block verification on sampled DFLASH steps; under the cap the cut drafts enter as a stopped draft (sparse-q mask) | `DFLASH_BV=1` (`--speculative-use-block-verification`; `DFCAP_BV` under the cap); needs `RSFIX` with its `reject` part, off otherwise | thinking c1 +5.9 % (development) | target distribution unchanged (Sun et al.'s theorem; law test above) | **Sun et al.**, "Block Verification Accelerates Speculative Decoding" (ICLR 2025); SGLang's kernel |
| Fused sampled accept for the DFlash2 selector draft: the accept computed from the logits with a sparse draft q (no dense [requests, 7, vocabulary] q or [requests, 8, vocabulary] p), at 16 or more requests without a verify-cap cutoff | `DFLASH_FUSED=1` (`INKLING_DFLASH_FUSED_MIN_BS=16`) | thinking c24 per-request decode +2.3 % (ABBA, cap off); below 16 requests the stock accept | same law as the stock accept | ours |
| Prefix tail: on a prefix-cache hit of a prompt of at most 4,096 tokens, the last 2,048 prompt tokens are prefilled again (SGLang's own prefix bound, as for logprob requests), so the drafter's 2,048-token window is filled instead of drafting from the uncached suffix only | `DFLASH_PREFIX_TAIL=2048`, `DFLASH_PREFIX_TAIL_MAX_INPUT=4096`; `0` = off | speedbench full 69.59 -> 72.86 (long context 63.09 -> 73.53, agentic 73.44 -> 82.74), same box (v2.8.1 at `MEM_FRACTION=0.89`, v2.8 at 0.85); +0.1-0.5 s time to first token on bounded hits | prefill-only: the same positions a cold prefill computes (floating-point order of the prefill chunking only); draft-side benefit | ours |
| Exact sampled verification: the residual / bonus CDF pass of SGLang's reject sampler takes the first lane above the coin that is inside the vocabulary and has positive mass (stock could emit a zero-mass padded row or an out-of-range lane under FP32 rounding); the DFLASH selector walk no longer clamps a no-match to a zero-probability last candidate; the DSpark Markov sampler's Philox fold (SGLang PR #51681) and the blockwise-rejection fixes (SGLang PR #51692, a path the recipe does not use) | `RSFIX=1` | correctness fix, both profiles | law unchanged in exact arithmetic; only rounding events that emitted a zero-probability token change | ours (`reject`, `selector`); **SGLang PR #51681 and PR #51692** |

The drafter, byte-cut and certified-head effects are development screens on one box (each on against the same build
with it off; the drafter row is stock against stock DSpark); the cap row is the release stack. `DFLASH_BYTES=...,hot16`
(BF16 hot rows instead of FP8) is built and off by default.

### Target-side levers (both profiles) and DSpark-only levers

Rows marked `SPEC=dspark` act only on the DSpark draft and are off in the default profile; their effects were measured
on the DSpark profile and confirmed in the v2.7.2 full runs (`hotbase` and `RNNHOIST_MAX_BS=32` came after v2.7.2).

| lever | switch (default) | effect, alone | output | credit |
|---|---|---|---|---|
| Verify-width policy with dead-row expert remap: per request and step, verify only the first L of the 8 DSpark rows; L grows left to right while the draft's calibrated survival (per position and confidence bin, greedy and sampled tables) pays for the row's marginal expert reads (one price, theta 0.04, for every batch size up to 16); dead rows reuse the anchor row's experts, so the MoE reads fewer distinct experts | `SPEC=dspark`: `VERIFY_CAP=policy:.../r2-theta004.json,bsmax=16` | vs the round-1 batch-size threshold rule, same day, same box: greedy decode prose c4 / c8 / c16 +5 / +13 / +5 %, code +2 / +6 / +2 %, thinking c4 / c8 +4 / +2 %; c1 flat (+14 % at c1 vs no cap, round 1). Above 16 requests all 8 rows are verified (thinking c24 lost 4-7 % with the policy on) | target distribution unchanged (the width uses only drafts before the cut; Monte Carlo law test) | ours (idea from our DeepSeek-V4.1 verify cap; calibrated per-row cost rule fitted on 36K logged request-steps, after our GLM-5.3-Flash pay-for-rows rule) |
| Block verification of sampled requests under the verify cap: the accepted prefix is drawn by block verification instead of the token-by-token chain; dead drafts beyond the verify width enter the kernel as a stopped draft (zero draft probability), so dead levels are never scanned | `SPEC=dspark`: `VCAP_BV=1` (`--speculative-use-block-verification`, `INKLING_VERIFY_CAP_BV=mask`) | thinking c1 / c8 / c24 +3.0 / +4.8 / +4.0 % (two runs vs two controls); greedy requests unchanged | target distribution unchanged (Sun et al.'s theorem over the stopped draft; joint law of the first tokens Monte Carlo tested with garbage dead rows) | **Sun et al.**, "Block Verification Accelerates Speculative Decoding" (ICLR 2025) for the algorithm; SGLang's block-verification kernel; integration under the cap ours |
| Draft statistics in one read: the per-position max draft probability that the verify rule reads and the draft probabilities q of the sampled accept, from one or two passes over the BF16 draft logits (the image's split softmax kernels) instead of FP32 copies and full-width temporaries | `SPEC=dspark`: `VCAP_FASTPROBS=1` | thinking c8 +3.5 %, greedy decode prose c8 / c16 +1.7 / +2.6 %, code c16 +2 % | same function, FP32 summation order only; the max probability only sets the verify width | ours |
| Fused sampled accept: p and q of the drafted tokens from a per-row log-sum-exp, the accept walk (chain or block verification) and the bonus sample without materialising the [rows, vocabulary] probability tensors; from 3 requests up, self-checked against the stock accept at start-up | `SPEC=dspark`: `VCAP_FUSED=1` | with FASTPROBS: code c16 266-270 vs 256-262 tok/s (+2 %), thinking c8 +1 to +4 % | same law; accept probabilities equal to FP32 rounding | ours |
| Draft logits through the bandwidth-efficient GEMV, FP8 copy of the draft's Markov fold matrix and of the LM head for the draft only, MXFP8 copies of the 3 draft layers | `SPEC=dspark`: `DRAFT_BYTES=gemv,fold8,head8,draftmx8` | 40.7 -> 42.4, step -2.8 ms, acceptance unchanged (2.45-2.49 tokens/step) | draft proposals only | ours; FP8 draft head after our DeepSeek-V4.1 `DSV41_DRAFT_HEAD_FP4` |
| NVFP4 expert GEMM tiles for GB10 at decode (`tile_k` 256 instead of the RTX 5090 table's 128) | `KERNELS=moe_tiles` | MoE GEMM +5-7 % (microbench), step -0.2 to -0.5 ms | bit-identical (bench + end-to-end text) | ours; FlashInfer / CUTLASS kernels |
| Stacked Q/K/V projection of the looped layers on one Triton GEMV instead of cuBLAS at <= 16 rows per source | `KERNELS=qkvr_gemv` | 187 -> 173 us per call, step -0.4 ms | bit-identical in the GPU check | ours; SGLang's sm12x GEMV kernel |
| pico's exactly coded 12-bit BF16 GEMV (`bf16x`) for qkvr / wo and the target LM head at <= 16 rows | `PICO12=dense,lmhead` | qkvr 130 vs 174 us, wo 52 vs 66 us, LM head 2.90 vs 3.96 ms per call (M=8, cold); with the stack, step 45.8 -> 43.1 ms, speedbench quick 51.9 -> 55.0 | weights decode to the same BF16 bits; FP32 summation order differs from SGLang's GEMV | **Andy Shu (@blaesus)**, [dsgb10](https://github.com/blaesus/dsgb10) / [pico](https://github.com/Picom-code/pico-dgx-ds4) (MIT), vendored unmodified; integration ours |
| Prefill: stacked qkvr projection computed per source column block (no wasted 3x rows) | `PREFILL=qkvr_split` | that projection -63 % at 8K chunks | bit-identical at >= 128 rows | ours (found by our prefill roofline) |
| Hot-vocabulary draft: the draft proposes only from the 65,536 most frequent target tokens (ranked on 1.95M tokens of the model's own outputs for 46K prompts from open datasets, unseen ids filled by prompt-text counts), so its LM head and Markov fold read 1/3 of the rows; FP8 draft-head GEMM reads its weight once for > 64 rows (`wide8`) | `SPEC=dspark`: `DRAFT_BYTES=...,hotvocab:65536,wide8`, `HOTVOCAB_FILE` | held-out coverage 98.8 % (speedbench 99.6 %); c1 step 39.8 -> 37.5 ms with tokens per step unchanged; speedbench quick +6 %, thinking c1 / c8 / c24 +2-9 / +4 / +5 % | draft proposals only; greedy texts byte-identical in the A/B | FR-Spec (Zhao et al., 2025) and SGLang's EAGLE token map for the idea; list, corpus and kernels ours |
| Hot-vocabulary base logits kept compact inside the captured draft sampler's Markov walk (no full-vocabulary buffer of -inf plus a scatter of the hot columns) | `SPEC=dspark`: `DRAFT_BYTES=...,hotbase` | not screened alone | bit-identical step logits and tokens | ours |
| Drafter global-attention window: the draft's global layer reads a 128-token sink + the last 16,384 tokens (identity below 16,640 tokens) | `SPEC=dspark`: `DRAFT_BYTES=...,dwin:16384` (off by default) | decode at 32K / 64K / 128K / 250K -11 / -7 / -3 / +7 %; option for workloads mostly above 200K tokens | draft proposals only; fewer far-context copies at 32K-128K | ours; attention sink from StreamingLLM (Xiao et al., 2023) |
| Routed-expert GEMM tile tables measured with real routing: NVFP4 `swap_ab=1, tile_n=32, tile_k=256` at 9-128 tokens per call, MXFP8 CuTe tactic 32x128 at 512-1536 routed rows | `MOE2=tiles` | MoE time per step c4 / c8 / c16 -3 / -4 / -6 % (microbench); greedy decode c2-c16 +2 to +6 %, thinking c8 / c24 +1 / +2 % | bit-identical (every config's outputs equal the shipped choice on real weights and routing) | ours; FlashInfer / CUTLASS kernels and tactics |
| NVFP4 expert scale codes: each expert row's E4M3 block scales stored as a row base + 5-bit codes (rare rows with > 31 distinct scales use an escape to the resident stock scales); FlashInfer's SM120 grouped-GEMM mainloop expands them into the same shared-memory scale bytes | `SCODES=1` | -4.06 % of NVFP4 expert bytes; GEMM -3 to -4.5 % per call; serving c2-c24 +1 to +3 %, thinking c24 +4 % (A/B) | bit-identical (350 / 350 tile comparisons on real weights and routing; first eager calls self-checked against stock) | ours (format idea: Christopher Owen / B12X MXFP4-CSF); FlashInfer and CUTLASS kernels (mainloop modified, see NOTICE) |
| pico stacked q/k/v without the second pass over the k/v rows: one call over the coded matrix, each column range for its own source | `PICO12=...,stack1` | -0.98 ms per c1 step (204 MB less read), 175 MB less memory | bit-identical (pico's kernel body, column ranges) | ours; pico (Andy Shu) kernel unchanged |
| Draft MXFP8 layers with narrow-N tiles (BN 16) | `SPEC=dspark`: `DRAFT_BYTES=...,w8tile` | draft layers 155 -> 198 GB/s, -0.36 ms per c1 step | bit-identical | ours |
| Draft Markov walk: precompute the FP32 contribution of 90 % of `rnn_proj` input columns once per walk, leaving a 768x512 projection per step, for up to 32 rows | `SPEC=dspark`: `DRAFT_BYTES=...,rnnhoist`, `RNNHOIST_MAX_BS=32` | 7-step walk, cold L2, 1 / 8 / 16 rows: 584 -> 320 / 661 -> 454 / 782 -> 596 us | draft proposals only; 99.9 % of elements correctly rounded vs FP64 | ours; walk-fusion idea from [vLLM PR #62958](https://github.com/vllm-project/vllm/pull/62958) (MIT) |
| Certified greedy LM head: a 6-bit copy of the head and a rigorous per-logit error bound select the 16-row vocabulary tiles that can hold the argmax; pico's exact GEMV computes only those; everything else is -inf before the stock softcap and argmax | `SPEC=dspark`: `CERT_HEAD=1` (on DFLASH: `CERT_DFLASH`, above) | head 2.90 -> 1.67 ms per 8-row step; speedbench quick 55.0 -> 56.7 (+3 %), decode c1 +3.5 %; only greedy steps of <= 16 rows without logprobs / grammar (sampled steps run the full head) | identical argmax by construction (serving check: 45,584 rows, 0 mismatches) | ours (ported from our GLM-5.3-Flash certified head; exact pass = pico's kernel) |
| Certified greedy LM head for 17-192-row verify batches: one 6-bit screen pass over all rows, the exact packed wide-head arithmetic only on the union of candidate tiles, reduced straight to token ids | `SPEC=dspark`: `CERT_WIDE=1` (`INKLING_CERT_WIDE`) | head + softcap + argmax at 64 rows 3.99 -> 1.72 ms, 128 rows 5.09 -> 3.19 ms (cold); serving greedy decode c4 / c8 +1.6-2.1 / +1.0-1.5 %, c1 and c16 within noise; every eligible step certified on real traffic | identical argmax to the shipped wide head (per-row proof; the GB10 tensor-core error model is qualified inside the GPU check; dense fallback when unsafe) | ours (extends the <= 16-row certified head) |
| Long-prefix fused prefill: on global layers, requests whose prefix is >= 131,072 keys run the fused prefill kernel's existing 8-warp variant (128 query rows per block), the rest the 4-warp variant, in two graph-safe launches | `INKLING_ATTN_W8_FROM` (default 131072; 0 = one launch) | the 4-warp kernel loses L2 reuse of K/V on long sweeps (70.7 TF/s at 120-256K, 46.5 at 1M); per global layer and 8K chunk: 512K 1,082 -> 777 ms (1.39x), 1M 2,281 -> 1,547 ms (1.47x) | bit-identical to the single 4-warp launch (all rows, output and LSE, 256K-1M, mixed batches, graph replay) | ours |
| Packed 12-bit GEMM for 17-256-row verify batches: reads pico's coded weights once per call, mma.sync over all rows (qkvr <= 96 rows per source, wo_ud <= 32, LM head <= 256; verify/decode forwards only) | `WIDE=dense,lmhead` | qkvr 1.24-1.34x and LM head 1.34-1.67x faster than cuBLAS at 17-64 rows; thinking c8 +8 %, greedy decode c4 / c8 / c16 +4 / +5 / +3 % | weights decode to the same BF16 bits; FP32 summation order differs from cuBLAS (error vs FP64 equal or lower) | ours; pico's `bf16x` reader (Andy Shu, MIT) |
| Verify attention on the split-KV relative-bias kernel at every batch size (cap 32 -> 384 rows) and the bias computed in-kernel above 32 rows, instead of padded FA4 with a sheared [t, 32, 1280] bias | `VATTN=1` | verify-attention visit 1.4-1.7x faster at 8-48 requests; thinking c24 +7 %, greedy decode c8 / c16 +5 / +6 % | same math; bias bit-identical, output within 1 BF16 ulp of the stock compact path | ours; SGLang's Inkling2 verify kernel |
| Packed 12-bit global-layer KV: full, committed KV pages of the 8 global-layer visits are also kept as sign + 7-bit mantissa bytes and a 4-bit exponent code against a fixed 15-exponent table (escape stream for the ~0.03 % of values outside it), 193 instead of 256 bytes per key and head; decode / verify attention decodes packed tiles in registers; 2,048 side slots per layer follow the pages being decoded (LRU) | `KV12=1` with `KV12S=0` and `SPEC=dspark`, `KV12_SLOTS=2048` | global verify visit 1.15-1.28x at 128K-256K (microbench); c1 step at 128K 62.2 -> 56.8 ms, at 250K 80.1 -> 75.2 ms; 92.6-99.9 % of tiles read packed; +6.1 GiB | bit-identical: decoded tiles equal the BF16 cache, attention partials equal stock's | ours, on SGLang's split-KV decode kernel; inspired by lossless exponent coding of floats (ZipNN, SplitZip) and by pico's exactly coded 12-bit `bf16x` weight format (Andy Shu); no code taken from either |
| Packed global KV store: the target's global-layer KV lives only in the 12-bit form above (no BF16 global pool, no side copy) inside SGLang's stock (static) pools; open pages stay BF16 until committed, the draft keeps its own KV (DFlash2: a fixed ring over its window; DSpark: its own BF16 pool), prefill (also inside its captured CUDA graphs) reads packed prefixes through a fixed BF16 scratch gathered on the device, and a host guard keeps the exact overflow slots from running out | `KV12S=1`, `KV12S_POOL=static` | DSpark profile (v2.7.2): global-layer KV 1.59M -> 2.13-2.17M tokens (+33-36 %), speed within run noise of the BF16 store; DFlash2 profile (v2.8, one box): 2.37-2.42M tokens | bit-identical: greedy texts equal to the BF16 store on 32 of 32 prompts (DSpark), every packed store and read bitwise in the GPU checks | ours, on SGLang's pools |
| Dead verify rows get no experts: rows the verify cap has cut route to a phantom expert that every MoE kernel skips, so quantisation, both grouped GEMMs, SwiGLU and the combine do no work for them | `NOEXP=1` (`SPEC=dspark`; on DFLASH `DFCAP_NOEXP=1`, above); needs the verify cap's remap | DSpark: greedy decode prose c4 / c8 / c16 +2.9 / +5.0 / +5.3 %, code c16 +5.1 % (paired screens, both boxes); c1 and above 16 requests unchanged | bit-identical for every row that can be accepted (live rows bitwise in the GPU check; cut rows are never read) | ours |
| Prefill attention with the relative-position bias computed inside the kernel (FA2-style, mma.sync, paged KV, sliding window and global layers), instead of a materialised [t, 24, 1024] bias read by FA4 | `ATTN=prefill_swa,prefill_global` | SWA attention 3.3x vs the round-1 bias_layout path; cold prefill 8K / 32K / 128K +15 / +18 / +26 % | same math; error vs FP32 reference equal to stock's | ours (FlashAttention-2 algorithm, Tri Dao) |
| Prefill glue fusion: the loop-message mixes, attention-residual mix + RMSNorm and the MLP short conv + residual update in fused kernels that skip the FP32 [T, 2304] round trips | `PFUSE=all` | -187 ms per 8K chunk (microbench); with the fused attention: cold prefill 1K / 8K / 32K / 128K +14 / +29 / +30 / +36 % | same per-element math; only FP32 row-sum order differs (error vs FP64 equal or lower), short conv bit-identical | ours |
| Prefill: relative-position bias written directly in FA4's sheared layout | `PREFILL=bias_layout` | with quant_once: 8K prefill 3,781 -> 4,253 tok/s | bit-identical (attention output and LSE) | ours |
| Prefill: MoE input quantised to NVFP4 once per token, then gathered to its 8 experts | `PREFILL=quant_once` | (with bias_layout, above) | bit-identical | ours |
| Prefill: MoE combine writes BF16 directly | `PREFILL=partial_bf16` | combine -22 to -32 %; with qkvr_split: 8K 3,408 -> 3,781 tok/s | bit-identical | ours |
| SGLang DSpark cookbook cell (FA4, page 128, 8 draft tokens, 4 `SGLANG_OPT_*` switches) | `SPEC=dspark` | base of the fallback profile | - | SGLang |
| One-pass checkpoint load: SGLang reads the safetensors file directly and large frees return to the OS before KV allocation | `LOAD_NOMMAP=1` | 121-137 s warm start-up in the v2.6 full benchmarks | weights unchanged | SGLang's loader flag; [sxuff](https://github.com/sxuff/inkling-2-mini-spark) and [0xSero](https://github.com/sybil-solutions/inference-optimisation-skill) for the load-strategy idea |
| KV pool sizing | `MEM_FRACTION=0.89` (0.85 above `CTX=262144` with DFlash2), `SWA_RATIO=0.037` | 2.37-2.42M global-layer KV tokens with 4.5-5.5 GiB MemAvailable after start, 4.1 GiB at worst (one box) | exact | ours |

Tried and not adopted (details in [docs/experiments.md](docs/experiments.md)): vLLM (same decode, slower prefill,
half the free memory), `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION` (flat), fixed verify caps (lose on concurrent
thinking), the Triton port of a packed-BF16 GEMV (superseded by pico's kernel), NVFP4 prefill tile retunes (<= 2 %),
other draft block sizes (refused by the draft), L2 prefetch / side-stream L2 warming before the small per-visit kernels
(exact, but GB10 drops `prefetch.L2` and a warm read only moves the same DRAM bytes: -0.1 ms per step), a native
layer-stack backend under SGLang (best case 1.13x at c4-c24 after the overlays, below our 1.20x bar), the verify
policy above 16 requests (thinking c24 -4 to -7 %), KV page size 16 (flat decode; the fused prefill attention needs
page 128), a stream-K packed GEMM for 97-192-row calls (1.2-1.45x cuBLAS in isolation, flat to negative in serving),
further prefill glue fusion (-70 ms per 8K chunk in isolation, flat end to end), a draft sampling temperature below
the request's (no gain or worse on thinking), a 4-bit draft MLP (same step time; its tokens-per-step gain is within
boot-to-boot variation and the offline acceptance change is -0.17 % with a confidence interval that includes 0),
late kill of doomed verify rows from a mid-stack readout (the readouts barely separate doomed rows: AUC 0.59-0.74),
shared-prefix KV reads for concurrent requests (L2 already absorbs the reuse), a fused routed-expert epilogue (already
fused on GB10), a streaming NVFP4 MoE kernel (bit-identical and 2-5 % faster than stock per call, not faster than the
scale codes), a centroid-clustered draft vocabulary (at most ~0.2 ms per c1 step in the byte model), verify-policy
retunes (theta 0.03 / 0.06, a second price above 16 requests: -1.4 % to flat), an 80K-token hot vocabulary.

## The model

Read from the checkpoint's `config.json` at revision `ed40e7da`:

| | |
|---|---|
| Parameters | 26.4B total, 3.4B active per pass (model card) |
| Layers | 24 stored; layers 6-17 form a looped core that runs 3 times, so a token passes 48 layer applications |
| Experts | 128 routed experts per MoE layer, top-8, no shared expert |
| Attention | 20 sliding-window layers (window 1,024) and 4 global layers (5, 11, 17, 23); 24 query heads, 8 KV heads, head dim 128, relative position bias; a short convolution (kernel 4) in every layer. Each pass of the looped core keeps its own KV and convolution state |
| Context | up to 1,048,576 tokens |
| Precision | routed experts NVFP4 in layers 0-19 and MXFP8 in layers 20-23 (`quant_method: inkling2`, ModelOpt); attention, embeddings and LM head BF16 |
| Bundled draft model | DSpark, 3 layers, block size 8 (`mtp.safetensors`, 0.64 GB); used with `SPEC=dspark` and `ENGINE=vllm` |
| Inputs | text and images (23-layer vision tower); text out. This recipe serves text only |
| Reasoning | always thinks; `reasoning_effort` none / minimal / low / medium / high / xhigh / max or 0.0-0.99 (default high = 0.9) |
| On disk | 20.2 GB |
| Licence | Apache-2.0 |

The default drafter, read from its `config.json` at revision `4f9088ea`:

| | |
|---|---|
| Drafter | DFlash2 by incoai, [incoai/Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) |
| Layers | 5 sliding-window attention layers (window 2,048), hidden size 2,304, BF16 |
| Drafting | block diffusion: a block of 8 positions per pass (7 drafts per verify step), 16 candidates per position, a selector traces one path |
| Inputs | target features from layer visits 5, 23, 41 and 48 (the last after the final norm) |
| On disk | ~1 GB |
| Licence | Apache-2.0 |

## Running it

Requirements: one DGX Spark with DGX OS, Docker with the NVIDIA runtime (the default on DGX OS), ~80 GB of disk for
the image (56 GB), the weights (20 GB) and the drafter (~1 GB), and nothing else large running on the box.

```bash
git clone https://github.com/knapcio/Inkling-2-Mini-1x-DGX-Spark && cd Inkling-2-Mini-1x-DGX-Spark
scripts/download.sh                       # weights -> ~/models/Inkling-2-Mini, drafter -> ~/models/Inkling-2-Mini-DFlash2,
                                          # both at pinned revisions, sha256-checked
docker pull lmsysorg/sglang@sha256:efac853b46811ccb760545b7b29e7119bdf907537ccf6e863fac3011fe8b98de
scripts/launch.sh up                      # SGLang + DFlash2 + overlays on 127.0.0.1:8000, packed KV store, mem 0.89
#   or: SPEC=dspark scripts/launch.sh up   # the model's own DSpark draft (v2.7.2 profile)
scripts/launch.sh status                  # container, API, MemAvailable
python3 bench/sanity.py all               # chat, effort ladder, tool call, JSON schema, 120K needle
scripts/launch.sh down
```

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
the same memory (the "page-cache trap", see Credits). `scripts/launch.sh up` refuses to start next to another model
server or with less than `MIN_AVAIL_GIB` + 8 GiB MemAvailable (for example while a download fills the page cache),
and runs `scripts/memguard.sh` for the first 20 minutes. The guard stops the container if MemAvailable falls
below 3.5 GiB for two consecutive samples, 2 s apart. An out-of-memory on GB10 can hang the box until a power cycle.

### Using it

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

### Fallback: the DSpark profile

`SPEC=dspark scripts/launch.sh up` serves with the model's own DSpark draft (`mtp.safetensors`) and its overlays:
the v2.7.2 profile (verify-width policy `r2-theta004.json` up to 16 requests, draft byte cuts, certified heads for
<= 16 and 17-192 rows, block verification, fused sampled accept, dead-row expert skip, packed KV store at
`MEM_FRACTION=0.89`), plus the exact changes made since: `RSFIX=1`, the compact hot-vocabulary base logits
(`hotbase`) and the Markov-walk hoist up to 32 rows (`RNNHOIST_MAX_BS=32`; v2.7.2: 16). It does not need the drafter
download. `SPEC=dspark` starts the v2.7.2 profile (with the v2.8 exactness fixes); speedbench quick 63.7 tok/s on
box A, 0 failed prompts, median TTFT 157 ms (`results/V28dsp-s05/`; the default profile 70.61 and `DFCAP=off` 62.49
in the same mode). Its v2.7.2 two-box medians: speedbench full c1 63.8, prose decode c4 / c8 / c16 113.3 / 164.6 /
244.1, thinking c1 / c8 / c24 72.2 / 233.6 / 437.1, decode at 32K / 128K 68.4 / 45.5 tok/s, 2.13-2.17M global-layer KV
tokens (CHANGELOG, v2.7.2).

`ENGINE=vllm scripts/launch.sh up` runs vLLM with the DSpark draft (the alternative profile, no overlays);
`SPEC=dflash2` is refused there.

### Every flag explained

`scripts/launch.sh up` runs one container (the command it prints, with the defaults):

```bash
docker run -d --name inkling-2-mini --gpus all --ipc=host --network host --shm-size 16g --restart no \
  -v ~/models/Inkling-2-Mini:/model:ro -v ~/.cache/inkling-2-mini-sglang:/root/.cache \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v $PWD/overlay:/opt/inkling-overlay:ro -e PYTHONPATH=/opt/inkling-overlay \
  -e INKLING_VERIFY_CAP=dfpolicy:/opt/inkling-overlay/inkling_vcap/policies/df-SLDF1-sm.json,bsmax=8 \
  -e INKLING_VERIFY_CAP_LOG_EVERY=0 -e INKLING_VERIFY_CAP_STEPLOG= -e INKLING_VERIFY_CAP_REMAP=1 \
  -e INKLING_DRAFT_BYTES= \
  -e INKLING_DRAFT_HOTVOCAB_FILE=/opt/inkling-overlay/inkling_draftbytes/hotvocab/inkling2-mini-hot-r2.json \
  -e INKLING_KERNELS=moe_tiles,qkvr_gemv \
  -e INKLING_MOE_TILES='1@2304:swap_ab=1,tile_n=32,tile_k=256;1@1152:swap_ab=1,tile_n=32,tile_k=256;8@2304:swap_ab=1,tile_n=32,tile_k=256;8@1152:swap_ab=1,tile_n=32,tile_k=256;64@2304:swap_ab=0,tile_n=64,tile_k=256;64@1152:swap_ab=1,tile_n=32,tile_k=256' \
  -e INKLING_PICO12=dense,lmhead,stack1 \
  -e INKLING_PREFILL=qkvr_split,bias_layout,quant_once,partial_bf16 -e INKLING_PREFILL_QKVR_OUT=strided \
  -e INKLING_CERT_HEAD=1 -e INKLING_WIDE=dense,lmhead -e INKLING_WIDE_SCOPE=verify -e INKLING_VATTN=1 \
  -e INKLING_ATTN=prefill_swa,prefill_global -e INKLING_PFUSE=all -e INKLING_MOE2=tiles -e INKLING_SCODES=1 \
  -e INKLING_VERIFY_CAP_FASTPROBS=0 -e INKLING_VERIFY_CAP_FUSED_ACCEPT=0 -e INKLING_NOEXP=1 \
  -e INKLING_DFLASH_BYTES=mx8,hot8 -e INKLING_DFLASH_HOT_N= \
  -e INKLING_DFLASH_PREFIX_TAIL=2048 -e INKLING_DFLASH_PREFIX_TAIL_MAX_INPUT=4096 \
  -e INKLING_DFLASH_FUSED_ACCEPT=1 -e INKLING_RSFIX=1 \
  -e INKLING_KV12=0 -e INKLING_KV12_MAX_PAGES=2048 \
  -e INKLING_CERT_WIDE=1 -e INKLING_CERT_WIDE_MAX_ROWS= -e INKLING_CERT_WIDE_STATS=0 \
  -e INKLING_DRAFT_RNNHOIST_MAX_BS=32 \
  -e INKLING_KV12S=1 -e INKLING_KV12S_OVERFLOW_HEADS=2048 -e INKLING_KV12S_FALLBACK_PAGES=64 \
  -e SGLANG_RAGGED_VERIFY_MODE=static -e INKLING_KV12S_POOL=static \
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
  --mem-fraction-static 0.89 --context-length 262144 \
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
| `MEM_FRACTION=0.89` (`--mem-fraction-static 0.89`) | static pool for weights, KV and CUDA graphs. With DFlash2, 0.89 up to `CTX=262144` and 0.85 above (the prefill scratch grows with `CTX`; not yet measured at 0.89 there); 2.37-2.42M global-layer KV tokens at 262,144 (see Memory budget) |
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
| `DFLASH_BYTES` (`INKLING_DFLASH_BYTES`), `DFLASH_HOT_N` | `mx8,hot8` with the overlays, `65536` | drafter byte cuts: `mx8` MXFP8 drafter linears; `hot8` FP8 hot-vocabulary plane (first `DFLASH_HOT_N` ids of `HOTVOCAB_FILE`) for the selector's candidate projection; `hot16` the same in BF16; empty = stock drafter |
| `CERT_DFLASH` (`INKLING_CERT_HEAD`) | `1` with the overlays | certified greedy LM head on DFLASH verify steps of <= 16 rows; `check` serves the full head and counts mismatches; `0` = off. Needs `PICO12` with `lmhead` |
| `CERT_WIDE_DFLASH` (`INKLING_CERT_WIDE`) | `1` with `CERT_DFLASH=1` | certified head for 17-192-row DFLASH verify batches; its token ids go straight to the DFLASH accept (needs `CERT_DFLASH=1`) |
| `DFCAP` (`INKLING_VERIFY_CAP`) | `dfpolicy:/opt/inkling-overlay/inkling_vcap/policies/df-SLDF1-sm.json,bsmax=8` with the overlays | DFLASH verify-width cap: calibrated survival policy, all 8 rows above 8 requests; `fixed:K` and `conf:T` exist for measurement; `off` = full width |
| `DFCAP_NOEXP` (`INKLING_NOEXP`), `DFCAP_REMAP` (`INKLING_VERIFY_CAP_REMAP`) | `1`, `1` | dead rows read the anchor row's experts (remap) and then get no experts at all (`inkling_noexp`) |
| `DFLASH_BV`, `DFCAP_BV` | `1`, `DFLASH_BV` | block verification on sampled steps (`--speculative-use-block-verification`); under the cap the cut drafts are masked as a stopped draft. Only with `OVERLAY=1` and `RSFIX` including `reject` (exact only with that fix); otherwise it stays off |
| `DFLASH_FUSED` (`INKLING_DFLASH_FUSED_ACCEPT`) | `1` with the overlays | sampled accept from the logits with a sparse draft q (`inkling_dfaccept`) for batches of `INKLING_DFLASH_FUSED_MIN_BS` (16) requests or more; smaller batches and calls with a verify-cap cutoff keep the stock accept; `0` = off |
| `DFLASH_PREFIX_TAIL` (`INKLING_DFLASH_PREFIX_TAIL`) | `2048` | on a prefix-cache hit, prefill the last N prompt tokens again so the drafter sees them (`inkling_dftail`; DFLASH only, not streaming sessions; pinned to the image's source); `0` = off. Set by the launcher only: the launcher refuses `INKLING_DFLASH_PREFIX_TAIL*` in `SGLANG_ENV` / `SGLANG_ARGS` |
| `DFLASH_PREFIX_TAIL_MAX_INPUT` (`INKLING_DFLASH_PREFIX_TAIL_MAX_INPUT`) | `4096` | only prompts of at most this many tokens are bounded; longer prompts keep the cache as before. Raise it (e.g. 8192) for long-output or document-QA traffic; at a raised limit, agent loops with short turns on 5-8K histories lose ~5-9 % end to end |
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
| `CERT_WIDE` (`INKLING_CERT_WIDE`), `CERT_WIDE_MAX_ROWS` | `SPEC=dspark`: `1`, overlay default | certified greedy head for 17-192-row verify batches (needs `CERT_HEAD`, `PICO12` and `WIDE` with `lmhead`) |
| `INKLING_ATTN_W8_FROM` | `131072` | prefix length (keys) from which global-layer prefill uses the 8-warp fused kernel; `0` = one 4-warp launch |
| `KV12S` (`INKLING_KV12S`), `KV12S_POOL` (`INKLING_KV12S_POOL`) | `1`, `static` | packed global KV store (see "Packed global KV store"); turns `KV12` off. `static` runs on SGLang's stock pools with prefill CUDA graphs, `MEM_FRACTION=0.89` and `SWA_RATIO=0.037` unless set; `unified` (`SPEC=dspark` only) is the v2.7.1 binding (`--enable-unified-memory --cuda-graph-backend-prefill=disabled`, 0.91 / 0.035). `0` = the BF16 store (`MEM_FRACTION` 0.85 with DFlash2, 0.93 with DSpark); vLLM and `OVERLAY=0` always use `0`. Set the store with these variables, not through `SGLANG_ENV` / `SGLANG_ARGS` (the launcher refuses the store's variables there, and with `KV12S=1` also the pool's engine flags, so the engine flags and the overlay always agree). `KV12S_OVERFLOW_HEADS` (2,048) and `KV12S_FALLBACK_PAGES` (64) size the exact overflow arena and fallback pool per layer visit |
| `NOEXP` (`INKLING_NOEXP`) | `SPEC=dspark`: `1` | dead verify rows get no experts (inactive when `VERIFY_CAP=off` or its remap is off); `0` = off |

Left at SGLang's defaults (measured, see docs/experiments.md): chunked prefill 8,192 tokens (16,384: +2-3 % at
32K-128K, more memory), radix prefix cache on, overlap scheduler on, prefill CUDA graphs on (off only with
`KV12S_POOL=unified`).

Extra engine flags go in `SGLANG_ARGS` (or after `scripts/launch.sh up`), extra container variables in `SGLANG_ENV`
(`-e NAME=value ...`); both are split on whitespace, so their values cannot contain spaces.

### Memory budget

GB10 has one 121.6 GiB pool for the GPU and the operating system. `MEM_FRACTION=0.89` budgets for the weights,
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
[Full memory sweep (DSpark profile)](docs/experiments.md#v26-kv-pool-sizing)

To leave more RAM for other work, lower `MEM_FRACTION`. Each 0.01 frees ~1.09 GiB, taken from the KV pools
(the weights, graphs and fixed buffers do not shrink).

### What differs from the published recipes, and why

| | SGLang cookbook (DGX Spark, DSpark) | vLLM recipe (DGX Spark) | here |
|---|---|---|---|
| engine | SGLang | vLLM | **SGLang**: the stock engines decode at the same speed on one Spark (c1 within 3 %, c16 / c24 within run noise); SGLang prefills 1.3-3.4x faster, keeps long prompts in its prefix cache more reliably and leaves twice the host memory ([docs/experiments.md](docs/experiments.md)) |
| draft | DSpark (bundled) | DSpark (bundled) | **DFlash2** (incoai, pinned revision, sha256-checked); DSpark with `SPEC=dspark` or `ENGINE=vllm` |
| memory | `--mem-fraction-static 0.45` (518K KV tokens) | `--gpu-memory-utilization` unset (vLLM's own suggestion would leave the host 0-2 GiB) | `MEM_FRACTION=0.89`, `SWA_RATIO=0.037` and the packed global KV store give 2.37-2.42M global-layer KV tokens, 4.1 GiB MemAvailable under stress and a 3.5 GiB guard floor |
| context | not set (model default) | 262,144 | 262,144 (the model supports 1M; see "Longer context") |
| multimodal | `--enable-multimodal` | on | off (text only; add `--enable-multimodal` for images) |
| images, weights | pulled by tag, weights from the Hub at start | same | pinned image digest, weights and drafter at pinned revisions, sha256-checked, offline at start |
| bind address | 0.0.0.0 | 0.0.0.0 | 127.0.0.1 |
| speed overlays | - | - | nineteen packages, on by default (`overlay/`); `OVERLAY=0` disables them |

### Longer context: 512K and 1M

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

### Packed global KV store (default, `KV12S=1`)

`KV12S=1` stores the target's global-layer KV only in the exactly coded 12-bit form that `KV12` reads (sign and
mantissa bytes, a 4-bit exponent code with 64 escape entries per page and head) and drops both the BF16 global pool
and the 6.1 GiB side copy. The open page of each request stays BF16 until it is full and committed. It lives in
SGLang's stock (static) pools, and prefill keeps its CUDA graphs. The draft keeps its own KV: DFlash2 a fixed ring over
its 2,048-token window, DSpark its own BF16 pool and captured fold. Inside a captured prefill, a chunk whose prefix
fits the fixed BF16 scratch gathers it on the device and runs the BF16 attention kernel; a prefix that does not fit
is read by the kernel's packed loader. `KV12S=0` is the BF16 store; with `SPEC=dspark`, `KV12S_POOL=unified NOEXP=0`
is the v2.7.1 binding (the same store on SGLang's unified memory pool, without prefill graphs).

| | BF16 store (`KV12S=0`) | packed store (default) |
|---|---:|---:|
| global-layer KV tokens, DFlash2 profile (v2.8, one box) | - | **2,371,584-2,416,896** |
| global-layer KV tokens, DSpark profile (v2.7.2) | 1,593,088 | 2,171,392 / 2,126,720 |
| `MEM_FRACTION` / `SWA_RATIO` defaults | 0.85 (DFlash2) or 0.93 (DSpark) / 0.05 | 0.89 / 0.037 |
| lowest MemAvailable, worst case (thinking c24 + cold 240K needle), DSpark profile | 5.2-5.6 GiB | 5.4 GiB |
| greedy texts vs the BF16 store (speedbench quick, DSpark profile) | - | **32 / 32 identical** |

`results/Q272R-*`, `results/Q272S-s05/`.

Exactness: every packed store and read is bit-identical to SGLang's BF16 path in the GPU checks
(`overlay/tools/gpu_check_kv12s.py`, including captured prefill graphs on both readers), and on the DSpark profile
the server's greedy texts are identical to the BF16 store's on 32 of 32 speedbench quick prompts, in two runs, with
and without `NOEXP`. Captured prefill graphs run each chunk padded to its bucket size, which changes the
floating-point order against eager prefill; both orders are equally close to an exact prefill (teacher-forced
re-scoring 0.049 vs 0.051 nat/token).

Limits and refusals:

- A page-head with more than 64 exponents outside its window uses an exact raw overflow arena (2,048 heads per
  layer visit); if that is full, the page stays raw in a 64-page fallback pool, then in its staging slot, all exact.
  Only if all three are exhausted does the server abort the requests that hold rows (HTTP 503) and keep running.
  A host-side guard evicts cached prefixes, defers prefill and retracts decode before that point. Measured use in
  all qualification runs: 0 overflow heads, 0 fallback pages.
- The prefill scratch is sized for `CTX`: 1.03 GiB at 256K, 2.03 GiB at 512K, 4.03 GiB at 1M.
- Startup refuses combinations it does not support: tensor / pipeline / data / expert parallelism above 1, context
  parallelism, streaming sessions, multimodal input, hierarchical / external KV caches and disaggregation,
  speculation other than a linear DSpark draft or a DFLASH draft (the latter only on the static pools and with its
  draft KV ring), breakable prefill graphs, ragged verify modes other than `static`, captured verify shapes above the
  compact attention cutoff (48 requests x 8 rows), quantised global KV, and `--enable-unified-memory` without
  `INKLING_KV12S_POOL=unified` (or the reverse). `scripts/launch.sh` sets the engine flags and the overlay's variables
  together; when starting the container by hand, keep them consistent with the command above.
- Prompt-logprob requests over thousands of tokens make SGLang hold FP32 logits for every scored row (2.5 GB for
  3,073 rows in one request, on either store); request them in windows of a few hundred rows on this box.

### FP8 KV

More concurrent long requests can also use an FP8 KV cache. FP8 KV changes the model's numerics (lossy) and is
not part of this recipe's measured profile; Tech2Wild's vLLM recipe measured an FP8 tier (3.3M tokens, needle PASS at
~950K, its tool-calling eval 87.0 vs 88.4 with BF16 KV) and documents the memory budget for it.

## Reproducing the numbers

```bash
git clone https://github.com/knapcio/speedbench speedbench && git -C speedbench checkout 6733653
scripts/launch.sh up                      # release defaults, as in the full benchmarks
#   or: SPEC=dspark OVERLAY=0 LOAD_NOMMAP=0 MEM_FRACTION=0.62 SWA_RATIO=0.1 scripts/launch.sh up  # stock cookbook cell
#   or: OVERLAY=0 scripts/launch.sh up     # stock DFLASH with the drafter (no block verification, BF16 store, mem 0.85)
bench/full.sh MYRUN                       # ~75 min, writes results/MYRUN/ (the v2.8.x qualification protocol)
python3 bench/report.py results/MYRUN      # README-style tables
bench/screen.sh MYSCREEN                   # ~10 min screen: speedbench quick, thinking c8, 32K prefill
```

The overlays' own checks run inside the image with the server stopped: `overlay/tools/gpu_check_qkvr.py`,
`gpu_check_pico12.py`, `gpu_check_prefill.py`, `gpu_check.py` (draft package), `gpu_check_bv.py`,
`gpu_check_fastprobs.py`, `gpu_check_fused_accept.py` (sampled accept), `gpu_check_rsfix.py`, `gpu_check_kv12s.py`, and
the CPU unit tests in `overlay/tests/` (see `overlay/README.md`). The DFLASH sampled-path law test under the verify
cap is `tools/gpu_check_dflash2_cap_law.py` (inside the image with `INKLING_RSFIX=1`); `tools/dfreplay.py` fits and
replays verify-width policies from `STEPLOG` captures.

## Licences

Repository-authored scripts, overlays, tests and documentation are MIT (see `LICENSE`). `overlay/inkling_pico12/
third_party/pico/` holds the GEMV sources by Andy Shu (dsgb10 / pico), unmodified, under their MIT licence (`LICENSE` next to
them); `overlay/inkling_wide/csrc/bf16x_format.cuh` copies pico's `bf16x` reader under the same licence
(`overlay/inkling_wide/third_party/pico/LICENSE`); the packed-KV attention kernel in `overlay/inkling_kv12/ops.py`
follows SGLang's Inkling2 split-KV decode kernel (Apache-2.0) with its K / V tile loads replaced; `overlay/inkling_rsfix/`
carries the code of SGLang PRs #51681 and #51692 (Apache-2.0); see `NOTICE`. The serving images (SGLang, vLLM) are
Apache-2.0 and are used as published, pinned by digest. The Inkling-2-Mini weights are Apache-2.0 from Thinking
Machines Lab. The DFlash2 drafter and its SGLang compatibility patch (incoai/Inkling-2-Mini-DFlash2) are Apache-2.0;
`scripts/download.sh` fetches them at a pinned revision, and nothing of them is distributed here.

See [CREDITS.md](CREDITS.md) for the work this builds on.
