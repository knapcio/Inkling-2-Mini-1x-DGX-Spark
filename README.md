# Inkling-2-Mini on one NVIDIA DGX Spark

Serving [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on a single DGX Spark
(GB10, sm_121, 121.6 GiB unified memory) with the model's own NVFP4 / MXFP8 weights and the public DFlash2 draft model
[incoai/Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2), on SGLang, plus a set of
import-time speed overlays that leave the target model's output distribution as it is.

**Status (2026-11-26, v2.9.6.1):** v2.9.3 passed a two-box qualification: Q293R and Q293S on both boxes; v2.9.4's prefix-tail deferral was release-checked on both boxes; v2.9.5's shared verify settle is bit-exact in the GPU check and was measured on box A, and v2.9.5 and v2.9.6 passed two-box qualifications (Q295R / Q295S, Q296R / Q296S); v2.9.6.1 turns v2.9.6's MXFP8 drafter context projections off by default (opt-in, `DFLASH_CTX8=1`).

One default profile: SGLang `lmsysorg/sglang:inkling-2-mini`, pinned by digest, drafting with DFlash2, a 5-layer
block-diffusion draft with a 2,048-token draft window and a 16-candidate selector, pinned at revision `4f9088ea`.
`scripts/launch.sh` checks the drafter's sha256 on the host and again inside the container on every start, and the
drafter's SGLang compatibility patch runs only inside the throwaway (`--rm`) serving container. Twenty-four overlay
packages (`overlay/`) run on top of it by default: drafter byte cuts (MXFP8 drafter layers, since v2.8.4 on FlashInfer's MXFP8 GEMM where
the shapes allow, and an FP8 64K hot-vocabulary plane for the candidate projection), the prefix tail and draft-tail snapshots that keep the drafter's window
filled on prefix-cache hits (v2.8.1 / v2.9, see "Prefix tail" and "Draft-tail snapshots"), the certified greedy LM head on verify steps, the packed 12-bit global KV store,
the exact sampled-verification fixes, block verification and a fused accept on sampled steps, a calibrated
verify-width cap with dead-row expert skip (v2.8.2: priced separately for greedy and sampled requests, with a context
bucket), no expert work for the padding rows of CUDA-graph batches (v2.8.3, see "Padding rows"), two bit-identical
prefill kernels, the sink gate (sliding-window layers since v2.8.2, global layers since v2.9.1) and the sliding-window
prologue (v2.8.3) and the loop-bank hoist (v2.9.2), see "Prefill levers"; the prompt-end cache touch for agent loops
(v2.9.2, see "Prefix cache in agent loops"), the prompt copy for long prompts (v2.9.3, see "Prompt copy for long
documents"), and the target-side kernel, prefill and
memory levers of v2.7.2. The
target distribution is unchanged; the hoist changes only the floating-point summation order of one projection (quality
checked with a five-run bracket, no drop). `SPEC=dspark` runs the model's own DSpark draft instead (the v2.7.2 profile, see
"Fallback: the DSpark profile").

Long context: at 32K context the DFlash2 profile decodes about 8-17 % slower than v2.7.2's DSpark profile at c1
(v2.8.2's verify-cap policy 66.1, v2.8.1 64.3, v2.8 60-63, v2.7.2 72 tok/s; different runs and boxes). Its acceptance
falls with context (3.10-3.17 vs 3.36 tokens per step at 32K), because the drafter sees only a 2,048-token window. At
128K both decode at about 50 tok/s. For mostly long-context single-user
work, `SPEC=dspark` is the faster choice. Since v2.9.3 the prompt copy narrows the gap where the answer quotes the
document: on speedbench `longctx` decode at 32K rises 10-14 % and at 128K 1.5-4.6 % (two boxes) (see "Prompt copy for long
documents").

V2.9.3 passed a two-box qualification: Q293R and Q293S on both boxes; v2.9.4's prefix-tail deferral was release-checked on both boxes; v2.9.5's shared verify settle is bit-exact in the GPU check and was measured on box A. The current column is one GB10 Spark on
2026-11-24 with the release defaults (`SPEC=dflash2`, verify-width cap on, `CTX=262144`); the stock and v2.7.2
columns are the earlier two-box medians. Cells marked v2.9.6, v2.9.5, v2.9.4, v2.9.3, v2.9.2, v2.9.1 or v2.9 were measured on those release defaults
(box B unless stated); the others are the v2.8.x or v2.8 runs. v2.9 added draft-tail snapshots for agent loops; v2.9.1 runs the sink
gate on the global layers too (prefill); v2.9.2 adds the loop-bank hoist (prefill) and the prompt-end cache touch
(agent-loop prefix hits); v2.9.3 adds the prompt copy for prompts of 8K tokens or more (long-context decode); v2.9.4 lets the prefix tail defer to
a draft-tail snapshot on repeated cached prompts of up to 4K tokens (speedbench long context +2.1 %); v2.9.5 builds the
packed store's verify-settle page candidates once for all 8 global visits (exact; single-request step -0.25 ms); v2.9.6
runs the drafter's two context projections on MXFP8 copies (single-request step -0.2 ms; drafts only), opt-in since v2.9.6.1.
Single-request decode on short prompts is unchanged.

| | stock SGLang cookbook cell (DSpark, mem 0.62) | v2.7.2 | **current (v2.9.6.1)** |
|---|---:|---:|---:|
| speedbench quick, 30 prompts, c1, tok/s (prose) | 41.86 (33.18) | - | **70.61 (65.67)** |
| speedbench full, 73 prompts, c1, tok/s | 40.6 | 63.8 | **73.21** (v2.9.4, box B, mean of 2 runs; the same tree with the deferral off 72.93, v2.9.3 73.08) |
| agent loops (speedbench agent mode, 5-16K-token histories), end-to-end time against prefix tail and snapshots off, 160- / 512-token turns | - | - | **-4.4 % / -4.9 %** (v2.9 screen; decode 73.9 -> 78.3 tok/s) |
| decode, prose, tok/s at c1 / c4 / c8 / c16 | 33.2 / 75.2 / 117.0 / 186.2 | 52.6 / 113.3 / 164.6 / 244.1 | **65.1 / 140.5 / 203.5 / 288.3** |
| decode, code, tok/s at c1 / c4 / c8 / c16 | 43.6 / 87.2 / 134.8 / 221.3 | 62.2 / 124.8 / 186.6 / 268.2 | **79.0 / 141.6 / 211.3 / 298.2** |
| thinking c8 / c24, aggregate tok/s (48 x 2,048 tokens, effort high, T=0.8) | - | 235.3 / 438.8 | **277.4 / 515.5** (v2.8) |
| thinking c9 / c15, aggregate tok/s | - | - | **282.2 / 392.5** (v2.8.3; padx off 269.3 / 373.9) |
| cold prefill, 8K tokens, tok/s | 3,399 | 5,439 | **5,657** (v2.9.2; v2.9.1: 5,595) |
| decode at 32K / 128K context, c1, tok/s (512 tokens) | 55.1 / 37.1 | 71.9 / 49.0 (one box) | **69.1-72.2 / 54.5-55.5** (v2.9.3 with the prompt copy, boxes A / B, 2 runs each; copy off: 62.8-63.4 / 52.1-54.7) |
| KV pool, global-layer tokens | 843K | 2.13-2.17M | **2,371,200** (v2.9, box B, `MEM_FRACTION=0.88`) |
| MemAvailable after start / worst case (thinking c24 + cold 240K needle) | ~40 GiB / - | 6.4-7.7 / 5.4 GiB | **4.5-5.5 (v2.8.1) / 4.8 GiB** (v2.9.2, box B; v2.9.1: 4.8 GiB) |

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
| **v2.9.2** (box B, release defaults) | - | - | - | - | - | - | **73.19** |
| v2.9.1 (box B, release defaults) | - | - | - | - | - | - | 73.12 |
| v2.9 (box B, release defaults) | - | - | - | - | - | - | 73.34 |
| v2.8.4 (box B, qualification run) | - | - | - | - | - | - | 73.43 |
| v2.8.4 (box A, release defaults) | 65.75 | 82.46 | 77.73 | 83.69 | 58.01 | 74.20 | 73.53 |
| v2.8.4 drafter MXFP8 GEMM on, mean of two runs (box A) | 65.97 | 82.73 | 78.03 | 83.97 | 58.24 | 74.46 | 73.78 |
| the same, off (v2.8.3 defaults), mean of two runs (box A) | 65.14 | 81.58 | 77.57 | 83.32 | 57.70 | 72.96 | 72.93 |
| v2.8.3 (box B, release defaults) | 65.02 | 81.56 | 77.63 | 83.28 | 57.65 | 72.99 | 72.90 |
| v2.8.2 (box B, two runs) | 64.99 / 64.96 | 81.52 / 81.43 | 77.62 / 77.56 | 83.32 / 83.21 | 57.63 / 57.56 | 72.15 / 72.94 | 72.78 / 72.82 |
| v2.8.2 without the cap (box B) | - | - | - | - | - | 67.31 | 64.06 |
| v2.8.2 with v2.8.1's cap policy (box B) | - | - | - | - | - | 73.05 | 72.42 |
| v2.8.1 (box B) | 64.92 | 81.88 | 77.46 | 82.74 | 57.46 | 73.53 | 72.86 |
| v2.8.1 tokens per step (median) | 2.50 | 3.62 | 3.40 | 3.38 | 2.23 | 3.27 | 2.80 |
| v2.8, same box (box B, `MEM_FRACTION=0.85`) | 64.57 | 79.84 | 77.10 | 73.44 | 57.19 | 63.09 | 69.59 |
| v2.8 (box A) | 64.66 | 79.89 | 77.14 | 73.57 | 57.21 | 63.12 | 69.66 |
| v2.7.2 (two-box median) | 53.6 | 72.1 | 68.0 | 76.7 | 43.7 | 75.1 | 63.8 |
| stock engine | 32.6 | 48.0 | 44.3 | 49.3 | 25.3 | 51.0 | 40.6 |

v2.9.2 is the release defaults on box B (`results/REL292-s06/`): 73.19 against v2.9.1's 73.12 on the same box (the
hoist-only tree 73.44, `results/Q292h-s06/`); greedy outputs identical to the hoist-only tree on 73 / 73 prompts, so the
cache touch changes no output. v2.9.1 is the release defaults on box B (`results/PF291-s06/`): 73.12 against v2.9's 73.34 on the same box, so
decode is unchanged within the run-to-run spread; greedy outputs equal v2.9's on 30 / 30 prompts. v2.9 is the
release defaults on box B (`results/V29R-s06/`): all 73 prompts 73.34 against v2.8.4's 73.43 on the same
box (`results/Q284R-s06/`), flat; greedy outputs identical to v2.8.4 on 30 / 30 prompts. In speedbench agent mode the
release run scored 76.33 (160-token turns) and 78.86 (512-token turns), geometric mean tok/s. The v2.8.4 qualification
runs on both boxes passed (`results/Q284R-s06/`, `results/Q284R-s05/`, `results/Q284S-s06/`,
`results/Q284S-s05/`). The v2.9.2 qualification passed on both boxes (`results/Q292R-s05/`, `results/Q292R-s06/`,
`results/Q292S-s05/`, `results/Q292S-s06/`): speedbench full 72.95 / 73.12 on box A / B (v2.8.4: 72.95 / 73.43), quality
guard 30 / 30 and sanity on both, cold prefill +1.6 to +6.4 % against v2.8.4 at 1K-128K, thinking c1 / c8 / c24 / c48
per-request decode within run spread, worst MemAvailable 4.2 / 5.6 GiB over three thinking-c24 rounds with a cold
240K needle after each (3 / 3 PASS per box), 200K and 250K contexts 2 / 2 each. v2.8.4 is the release defaults on box A
(`results/REL284-s05/`, one run, 73 prompts, 0 failed): median time to
first token 164.8 ms, median step 39.27 ms, 2.833 tokens per step. The same-box reference with `mx8w` off is 73.13 /
72.73 (`results/MXa0-s05/`, `results/MXb0-s05/`); v2.8.3's 72.90 was measured on box B, so the release rows of the two
versions are not a like-for-like comparison. Per-stream decode in the release run: prose c1 / c4 / c8 / c16 66.7 /
37.0 / 27.9 / 20.4, code 79.1 / 38.6 / 30.0 / 22.0 tok/s. Greedy outputs are identical to the `mx8w`-off run on 30 / 30
prompts. The drafter's MXFP8 GEMM (`mx8w`) was
measured as an ABBA on box A (`results/MXa0-s05/`, `results/MXa1-s05/`, `results/MXb1-s05/`, `results/MXb0-s05/`,
off / on / on / off): all prompts 73.75 / 73.81 against 73.13 / 72.73 (+1.2 % on the means), median step 39.21 /
39.22 ms against 39.57 / 39.91 ms; every category gains, long context most (+2.1 %), structured least (+0.6 %).
Per-stream decode at c1: prose 66.7 / 66.9 against 65.9 / 65.7 (+1.4 %), code 79.4 / 79.5 against 78.7 / 78.8
(+0.9 %); at c4-c16 mostly up but noisy (from code c8 -1.2 % to prose c4 +3.8 %); thinking per-request median c8
+1.8 %, c24 +2.4 %. It makes the drafter's own GEMMs faster; no acceptance change is claimed. Greedy outputs are identical with
and without it on 30 / 30 prompts in every arm.

v2.8.3 is the release defaults on box B (`results/REL283-s06/`, one run, 73 prompts, 0 failed): median time to
first token 163.8 ms, median step 39.71 ms, 2.742 tokens per step. c1 decode is unchanged within the run-to-run
spread (v2.8.2 72.78 / 72.82); v2.8.3's gains are at concurrency (padding rows) and in prefill (the prologue). Greedy
outputs equal v2.8.2's release run on 30 / 30 prompts. v2.8.2 is the release defaults on box B in two runs (`results/PF282-s06/`, `results/REL282-s06/`, 73 prompts, 0
failed); the same tree with the cap off is `results/SBF0-s06/` and with v2.8.1's `df-SLDF1-sm` policy
`results/SBF1-s06/`. Decode speed on speedbench is unchanged against v2.8.1: v2.8.2 is a prefill and verify-policy
release. The new policy is +0.4 on all prompts against the old one in the same tree; the long-context spread
(72.15-73.53) is run-to-run variation. v2.8.1 is the release defaults on box B (`results/V281s-s06/`,
`MEM_FRACTION=0.89`, limit 4,096), median time to
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

Since v2.9.4 the tail defers to a draft-tail snapshot (`DFLASH_TAIL_DEFER=1`, the default; it needs the snapshots on).
Before a prompt of at most 4,096 tokens is bounded, the snapshot slots are checked for one of the same prefix (same
tokens, cache salt and extra key) that ends past the tail's bound and already holds the drafter's whole window. If
there is one, the request keeps its ordinary prefix-cache hit, capped at the snapshot's end, and the window is copied
back from the slot instead of being recomputed by the target; otherwise the tail applies as before. The slot is
reserved from the cache lookup to the copy and checked again just before it, and a slot that changed in between is
not used. Not for logprob requests, servers with more than one rank, or hierarchical / external KV storage.
`DFLASH_TAIL_DEFER=0` runs v2.9.3's behaviour exactly.

Measured on box B, two runs each way (on, off, off, on), same tree: speedbench full long context +2.1 % (75.01 / 74.94
against 73.39 / 73.52 tok/s; tokens per verify step 3.08 against 2.97; median time to first token on these cache hits
0.19 against 0.45 s), all prompts +0.4 %, the other categories flat; single-request decode step 39.05 / 39.17 against
39.62 / 39.58 ms. Speedbench agent mode is unaffected: its histories (5-16K tokens) are all above the tail's limit, so
the deferral never runs there; the gain is on repeated cached prompts of 1K-4K tokens. Decode at 32K / 128K is flat
(four runs each way: 32K on 67.4-73.1, off 63.4-70.2 tok/s). Greedy outputs differ from the deferral-off runs on 7 / 73
speedbench and 3 / 30 warm prompts, the same ones in both runs with it on: a cached prefix in place of a recomputed
one, the prefill floating-point-order class. Quality bracket (box A, off / on / off / on / off): 559 / 559 / 557 / 561 /
557 of 575 correct, no drop. Agent loops (24 x ~72K): 24 / 24 turn-2 cache hits with and without it. Worst
MemAvailable with thinking c24 and a cold 240K needle: 4.1 GiB.

### Draft-tail snapshots

`SPEC=dflash2` draft-tail snapshots (`DFLASH_TAIL_SNAPSHOTS`, default 8 slots, v2.9): when a request's prefill
completes its prompt, the drafter's 2,048-position window at the prompt end is copied into a slot, keyed by the
prompt's content and cache salt. A later request that hits the same prefix in the cache gets that window copied back
into its drafter ring, instead of a ring rebuilt from only the uncached suffix and without a re-prefill. Snapshots are
restored only when content and salt match. Draft-only state: the target model's computation is untouched. 8 slots of
2,048 positions x 5 draft layers take 320 MiB, charged to the KV budget (0.312 GiB); with the prefix tail's deferral on
(the default since v2.9.4, see "Prefix tail") a slot holds one 128-token page more, starting on a page boundary, so a
restore can start on the page the tail would have recomputed from: 340 MiB (0.332 GiB). The slots are one shared LRU: in
a server shared by several tenants, one tenant's requests can evict another's snapshots, which shows only as that
tenant's draft acceptance and latency (the radix cache's own eviction is the same kind of channel). Run with
`DFLASH_TAIL_SNAPSHOTS=0` where tenants must not observe each other's cache use. Any weight update or `flush_cache`
clears every snapshot. With more than 8 concurrent long conversations only the most recently prefilled ones keep a snapshot;
`DFLASH_TAIL_SNAPSHOTS` raises the count (up to 64) at 40 MiB of KV budget per slot (42.5 MiB with the deferral). Snapshots touch only draft state
and never change outputs.

Measured on box B against the same tree with both levers off (A = the prefix tail shipped since v2.8.1, B =
snapshots, on the v2.8.1 base). The release ships both.

| | speedbench full, long context | agentic | code | agent loops, 160-token turns | 512-token turns |
|---|---:|---:|---:|---:|---:|
| A (prefix tail) | +6.6 % | +1.1 % | +1.7 % | -0.7 % | +0.3 % |
| B (snapshots) | +14.4 % | +0.2 % | +0.1 % | -4.0 % | -5.4 % |
| A + B (release) | +6.7 % | +1.3 % | +1.7 % | -4.4 % | -4.9 % |

Speedbench columns are the change in end-to-end speed; agent loops (speedbench agent mode, 8 conversations of 4-8 turns
on 5-16K-token histories) are the change in end-to-end time at equal output length, so negative is faster. With A + B
decode in the agent loops goes from 73.9 to 78.3 tok/s, and median time to first token stays at 0.242-0.247 s.
Snapshots alone do best on cached long prompts (no re-prefill) but miss the tail's gain on short agentic and code
prompts below the 1,024-token capture minimum. 16 slots cost 0.6 GiB more MemAvailable in the worst case (4.0 against
4.6 GiB) for no speed gain, so the default is 8.

### Prefix cache in agent loops (prompt-end touch)

`PROMPT_TOUCH` (`overlay/inkling_ptouch`, default on since v2.9.2). In multi-turn agent loops with long contexts, some
turn-2 requests could miss the prefix cache entirely. When a request finished, the sliding-window state behind its
prompt end became the oldest evictable entry in the cache, so under load the earliest conversations lost it first;
their next turn re-prefilled the whole context, which evicted the next-oldest prompt end, and the misses cascaded. In
the dumped stock misses the prompt-end node had lost its sliding window and with it the saved state (11 of 12); in
the twelfth part of the window was gone. Reported upstream as
[sgl-project/sglang#53149](https://github.com/sgl-project/sglang/issues/53149).
After a request finishes, the overlay touches its prompt path with SGLang's own `touch_prefix` (the walk the scheduler
uses for waiting requests: no allocation, no lock). It changes the eviction order only; nothing the model computes
changes.

Measured on box B with 24 agents of ~72K-token contexts (`--prefill-decode-interval 2`): stock missed in 3 of 3 runs
(9 or 10 agents per run re-prefilled; turn-2 time to first token 163-186 s, turn-2 wall about 370-380 s); with
the fix 2 of 2 runs had 24 / 24 cache hits (turn-2 time to first token 3.0 s, wall about 210 s). The default
scheduling showed the same intermittent misses (5 of 24 agents in 2 of 5 runs). Release check on the combined tree:
24 / 24 hits at the default interval (AG292: time to first token 3.0 s, turn-2 wall 211 s); worst MemAvailable 4.8 GiB
with 48 / 48 requests at c24 and needles at 128K / 240K passing (KVS292c); greedy outputs identical to the hoist-only
tree on 73 / 73 speedbench and 30 / 30 warm prompts (REL292).

A larger sliding-window pool is not a substitute for the touch. With `SWA_RATIO=0.074` instead of 0.037 (145K instead
of 87K window tokens, at the cost of 16 % of the global-layer KV tokens) the same stress configuration (24 x ~72K,
`--prefill-decode-interval 2`, without the touch) still missed the prefix cache for 8 of 24 agents on box A and 9 of 24
on box B on turn 2 (one run on each box; time to first token 149 and 164 s), about as many as with the default pool
(9-10 of 24 in three runs). What decides it is which windows are evicted first, not how many fit, so the prompt-end
touch is the fix and the default pool stays as it is (`results/SWA1i2-s05/`, `results/SWA1i2-s06/`).

Interactive streaming with many concurrent long prompts: `SGLANG_ARGS="--prefill-decode-interval 1"` shows the first
tokens earlier (time to first token p50 373 -> 213 s on 24 x 72K) but finishes every response later (completion
+2-3 %, p90 time to first token +3 %, tail +15 %). It is off by default and does not help agent loops, which wait for
the whole response.

### Prompt copy for long documents

`DFLASH_PCOPY=12` (`overlay/inkling_pcopy`, default on since v2.9.3, `SPEC=dflash2` only). The drafter sees a
2,048-token window, so when an answer quotes a long document it cannot see the passage it is quoting. Each decode
step, the prompt copy looks for the most recent earlier occurrence of the last 12 committed tokens in the request's
own context (prompt and output so far); when that match extends to at least 16 tokens, the tokens that followed it
replace the drafter's proposals for the rest of the draft block. These are draft proposals only: the target verifies
every token as before. Sampled requests give each copied slot a point mass, which the exact acceptance rule needs, and
the verify-width cap keeps the copied rows live. The step is two small kernels (a scan of the context split over
2,048-token chunks, then the copy) with no host synchronisation.

The copy is armed only for requests whose prompt has at least `DFLASH_PCOPY_MIN_CTX` tokens (8,192). On short
prompts true copies are rare and a forced copy that is wrong costs acceptance: without the gate one speedbench long
context prompt of 2.7K tokens needed 67 verify steps instead of 61 for the same output. Every speedbench prompt is
shorter than 5K tokens, so with the gate speedbench runs exactly as without the copy; the gain sits at 16K tokens of
context and above.

Measured on box A, two ABBA runs (off, on, on, off), speedbench `longctx` (`lc32`: 32K and 128K documents, 3 repeats
cold and cached, 512 decoded tokens): decode in tok/s is the sum of decoded tokens over the sum of decode seconds of
the six requests per size; tokens per step likewise. The first ABBA ran an earlier version of the copy (one kernel
launch per piece of work), the second the shipped kernels; neither had the 8K gate, which only affects prompts
shorter than 8K.

| | off | on |
|---|---:|---:|
| decode at 32K, tok/s, ABBA 1 / ABBA 2 | 65.5, 64.0 / 60.5, 66.6 | 74.5, 66.0 / 67.5, 73.8 |
| mean of the four runs, 32K: decode tok/s / tokens per step | 64.2 / 2.98 | **70.5 / 3.30** (+9.8 % / +10.5 %) |
| decode at 128K, tok/s, ABBA 1 / ABBA 2 | 55.8, 57.7 / 50.7, 51.4 | 58.6, 56.1 / 50.3, 55.9 |
| mean of the four runs, 128K: decode tok/s / tokens per step | 53.9 / 3.27 | **55.2 / 3.41** (+2.4 % / +4.2 %) |
| single-request decode step, ms (median of the per-request `step_ms` over the 73 measured speedbench full requests), ABBA 2 | 39.46, 39.52 | 39.22, 39.63 |

Within each ABBA every run with the copy decoded faster at 32K than every run without it. Speedbench full, agent
mode and thinking c8 were flat (docs/experiments.md has every arm); greedy outputs of speedbench full were identical
in all four arms of the second ABBA. Across speedbench, copies were forced on about 2 % of decode steps, almost all
in the long documents. With the earlier one-launch-per-piece version the decode step was 0.3 ms longer, which is why
the shipped version runs two fused kernels (the extra host launch, about 10 us, is hidden behind the GPU: -0.06 ms
in ABBA 2). `DFLASH_PCOPY=0` turns it off.

Release check on this tree, with the gate, one ABBA per box: `longctx` decode at 32K +10.1 % (box A) and +14.0 %
(box B); speedbench full took the same verify steps and gave identical greedy outputs on 73 / 73 prompts in every arm;
agent mode and thinking c8 within the off arms' spread (docs/experiments.md, "Release checks").

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

### Padding rows

A CUDA graph replays at its captured batch size; with fewer live requests the remaining rows are padding, and in the
stock engine each padding row still routes its stale input to 8 experts. From v2.8.3 (`PADX=1`,
`overlay/inkling_padx`) the padding rows route to the phantom expert that the dead-verify-row skip already uses, so
the expert GEMMs do no work for them. The choice is made per captured graph, at capture time: graphs of 1-8 requests
can never replay with padding (the runner pads n live requests up to the smallest captured size >= n, and 1-8 are all
captured), so they keep the stock route with no extra kernels; graphs of 10 requests and up get the mask. A first
version masked every graph and added 4 small kernels per MoE visit (192 per step, +0.36 ms or 0.8 % at c1, where
nothing pads); the per-graph choice removes that cost. Live rows are bit-identical in every shape, and the small
graphs are bit-identical to stock on every row (GPU check against written-out expectations).

Release defaults against padx off, box A, same session (`results/PADX283g-s05/`, `results/PADX283off2-s05/`):

| | padx off | padx on (release) | change |
|---|---:|---:|---:|
| speedbench full, all 73 prompts | 73.25 | 73.17 | flat |
| median step / tokens per step | 39.46 ms / 2.742 | 39.59 ms / 2.742 | two padx-off runs differ by 0.09 ms |
| thinking c9, aggregate (per-request median) | 269.3 (34.3) | **282.2** (35.6) | +4.8 % |
| thinking c15, aggregate (per-request median) | 373.9 (27.9) | **392.5** (29.6) | +5.0 % (+6.1 %) |

Measured with the first, ungated version (`results/PADX283-s05/`, `results/PADX283off-s05/`; graphs of 16 and 24
requests are masked in both versions): thinking c16 415.5 vs 403.4 (+3.0 %), the worst-case load (thinking c24) 331.6
vs 324.6 (+2.2 %). Screen, box A, ABBA, two runs each (`results/PXa0-s05/`, `results/PXa1-s05/`, `results/PXb1-s05/`,
`results/PXb0-s05/`), aggregate tok/s, off -> on: c9 269.5 / 270.3 -> 292.0 / 291.3 (+8 %), c15 354.9 / 375.9 ->
400.5 / 392.3 (+8 %), c16 401.2 / 397.5 -> 410.6 / 407.9 (+2.4 %), c10 flat.

The released version's greedy outputs at c1 equal v2.8.2's release run on 30 / 30 prompts, and its 128K needle
passes. Concurrent greedy output depends on the batch composition on this stack: two runs with padx off already
differ on 57 of 60 outputs, and padx on against off shows the same first-divergence class.

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
v2.8 cold prefill at 1K / 8K / 32K / 128K (second pass): 2,753 / 5,383 / 5,083 / 3,738 tok/s, unchanged within noise
against v2.7.2 (8K: 5,383 against the published 5,439).

Prefill levers, cold prefill tok/s (box B, same prompts, mean of 2 runs; v2.9.1 release defaults `results/PF291-s06/`;
v2.8.3 release defaults `results/PF283-s06/`;
v2.8.2 release defaults
`results/PF282-s06/`; control on the v2.8.1 tree on the same box `results/PHAc-s06/`, repeat `results/PHAc2-s06/`
2,812 / 5,408 / 5,077 / 3,726):

| | 1K | 8K | 32K | 128K |
|---|---:|---:|---:|---:|
| v2.8.1 control (same box) | 2,836 | 5,411 | 5,073 | 3,749 |
| screen on the v2.8.1 tree: hoist + gate (FP32-order class) | 2,833 | 5,599 (+3.5 %) | 5,254 (+3.5 %) | 3,844 (+2.6 %) |
| screen on the v2.8.1 tree: hoist + gate + the v2.8.2 prologue (FP32-order class) | - | 5,656 | - | - |
| v2.8.2 release defaults (gate only) | 2,821 (flat) | 5,527 (+2.1 %) | 5,189 (+2.2 %) | 3,785 (+1.0 %) |
| v2.8.3 release defaults (gate + prologue) | 2,864 (+1.5 %) | 5,553 (+0.5 %) | 5,217 (+0.6 %) | 3,788 (+0.1 %) |
| v2.9.1 release defaults (gate on all layers + prologue) | 2,821 (~flat) | 5,595 (+0.8 %) | 5,236 (+0.4 %) | 3,824 (+1.0 %) |
| **v2.9.2 release defaults (+ loop-bank hoist)** | **2,857** (+1.3 %) | **5,657** (+1.1 %) | **5,325** (+1.7 %) | **3,875** (+1.4 %) |

v2.8.3's changes are against v2.8.2, v2.9.1's against v2.8.3, v2.9.2's against v2.9.1 (`results/PF292-s06/`). Against the v2.8.1-tree control (mean of the two control runs) the release
defaults gain +2.7 % at 8K, +2.8 % at 32K and +1.4 % at 128K.

### Prefill levers

Two bit-identical prefill levers are on by default, the sink gate (v2.8.2 on the sliding-window layers: +2.1 % at 8K,
+2.2 % at 32K, +1.0 % at 128K in serving; v2.9.1 also on the global layers) and the sliding-window prologue (v2.8.3),
and since v2.9.2 a third lever that changes the floating-point summation order but not the math:

- `PFUSE_HOIST` (`overlay/inkling_phoist`), default `1` since v2.9.2: the loop
  controller's linear source projection (the loop bank's attention-residual weights, the message normalisation and
  the message-only part of the controller's down projection) is prepared once per loop pass instead of inside every
  fused visit. Same operations, different FP32 summation order: 99.96 % of elements bit-equal to the existing fused
  path and every FP64-reference comparison within 1.5x of its error (512-8,192 rows), so greedy outputs can move on
  long prompts exactly as any reduction-order change does (2 of 30 long prompts in the exactness gate, the same class
  as the existing fused loop mix against stock). Quality is verified equal by a bracket of repeated eval runs (see
  Correctness). One 8K loop pass 104.5 -> 93-94 ms; cold prefill on the v2.9.2 release defaults against v2.9.1 +1.3 / +1.1 /
  +1.7 / +1.4 % at 1K / 8K / 32K / 128K. A start-up self-test checks it against the existing path
  and falls back to that path on any failure, and only the checked row-count regimes are armed. `PFUSE_HOIST=0`
  restores v2.9.1; `PFUSE_HOIST=exact` is a bit-identical, slower variant. It needs `PFUSE` with `loopmix` (on in
  `all`).
- `ATTN_GATE=all` (`overlay/inkling_attn`): the sink gate (`gate_sink`) computed in the epilogue of the prefill
  attention kernel, with coalesced loads and stores: on the sliding-window layers since v2.8.2, and since v2.9.1 also
  on the global layers (BF16 and packed 12-bit KV paths). `ATTN_GATE=1` keeps it on the sliding-window layers only,
  `0` turns it off. On the global layers: -840 us per 8K visit, -6.7 ms per 8K chunk; same-tree A/B against
  sliding-window only +0.4 / +0.4 / +0.5 % at 8K / 32K / 128K. On the sliding-window layers: Bit-identical to SGLang's `gate_sink` (output and LSE,
  graph replay, padded multi-request batches, padded rows zeroed; on every global-layer path too, including captured
  pairs and replays). Attention plus gate 2,759 -> 1,944 us per
  sliding-window visit at 8K (-32.6 ms per 8K chunk). In serving: cold prefill 8K / 32K / 128K +2.1 / +2.2 / +1.0 %,
  cold time to first token 1.48 / 6.30 / 34.6 s. The gated epilogue turns itself off while `SGLANG_KERNEL_API_LOGLEVEL`
  is non-zero (it logs "gated epilogue OFF ..."), because that kernel logging reads the FA4 FP32 output buffer that
  the gate leaves unwritten; with the variable unset nothing changes.
- `PROLOGUE=1` (`overlay/inkling_prologue`, on since v2.8.3): the sliding-window visit's K short conv + L2 clip and
  V short conv + value norm as one kernel each, reducing in the stock kernel's row-sum order. Bit-identical: 0 ulp
  at 2,048 / 2,049 / 3,001 / 8,192 rows, including adversarial rows that are sensitive to the summation order; in
  serving, greedy outputs identical to both controls on 30 / 30 prompts with log-probabilities bit-equal over 4,691
  tokens. About -8.5 ms per 8K prefill chunk.

All three are pinned to the image's source; with any of them on (the gate and the prologue are on by default), `scripts/launch.sh`
refuses `--enable-mixed-chunk` (exit 2: mixed batches would give decode rows prefill-only math). The levers add up to
~0.6-0.7 GB of transient memory during long prefills, so `MEM_FRACTION` is 0.88 with any of them on (0.89 with all
three off; 0.85 above `CTX=262144`). With v2.8.2's set (the gate alone), 0.89 was tested: the memory guard stopped
the server 25 s into the worst-case load (MemAvailable 3.3 GiB), so 0.88 stays.

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

v2.8.2 (box B, release defaults, `MEM_FRACTION=0.88`): global-layer KV pool 2,349,056 tokens (v2.8.1 at 0.89:
2,400,384, -2.1 %). Lowest MemAvailable over 4 release-defaults runs, each with thinking c24 (48 x 2,048 tokens)
alongside a cold 240K needle: 5.0 GiB (5.7 / 5.0 / 5.9 / 6.4 GiB; `results/KVS282a-s06/`, `results/KVS282b-s06/`,
`results/PF282-s06/`, `results/REL282-s06/`; 48 / 48 requests completed, 343.8 / 341.8 tok/s aggregate), guard
floor 3.5 GiB. At 0.89 (`results/KVS282c-s06/`) the guard stopped the server 25 s into that load at 3.3 GiB.

v2.9.2 (box B, release defaults): lowest MemAvailable in the worst-case load 4.8 GiB on the release tree
(`results/KVS292c-s06/`; 48 / 48 requests, 340.1 tok/s aggregate) and 4.0 / 5.7 GiB in two runs with the hoist alone
(`results/KVS292-s06/`, `results/KVS292b-s06/`), guard floor 3.5 GiB; `MEM_FRACTION` stays 0.88. The cache touch
allocates nothing.

v2.9.1 (box B, `results/KVS291-s06/`, release defaults): lowest MemAvailable in the worst-case load 4.8 GiB, guard
floor 3.5 GiB; `MEM_FRACTION` stays 0.88. The gate on all layers against sliding-window only, as a memory pair: 5.2 /
5.4 GiB (`results/KVSGa-s06/`, `results/KVSG1-s06/`).

v2.9 (box B, `results/V29S-s06/`, release defaults with 8 snapshot slots): global-layer KV pool 2,371,200 tokens;
lowest MemAvailable in the worst-case load 4.6 GiB, 48 / 48 requests completed, guard floor 3.5 GiB. With 16 slots
(`results/K81b16-s06/`) the worst case was 4.0 GiB (8 slots: 4.6 GiB, `results/K81b8-s06/`).

v2.8.4 (box A, `results/MXkv-s05/`, release defaults with the drafter MXFP8 GEMM): lowest MemAvailable in the
worst-case load 4.8 GiB (thinking c24 aggregate 345.8 tok/s; the release run `REL284`: 4.9 GiB), guard floor 3.5 GiB; KV pool unchanged (about 2.36M
tokens on box A; the swizzled scales add 11.2 MB).

v2.8.3 (box B, release defaults, `MEM_FRACTION=0.88`): global-layer KV pool 2,352,000 tokens; lowest MemAvailable in
the same worst-case load 5.3 GiB (5.6 / 5.5 / 5.4 / 5.3 GiB; `results/KVS283a-s06/`, `results/KVS283b-s06/`,
`results/PF283-s06/`, `results/REL283-s06/`; 48 / 48 requests completed, 340.1 / 357.4 tok/s aggregate), guard floor
3.5 GiB.

The pool differs between boots at the same fraction because the memory left after the weights and graphs differs.

### Quality checks

- Needles with the passphrase at 50 % depth at 128K and 240K tokens: PASS on v2.8, v2.8.1, v2.8.2, v2.8.3, v2.8.4, v2.9, v2.9.1 and v2.9.2.
- Prefill levers: exactness gate (FP64 reference, near-tie check of greedy outputs, needles). v2.9.1's gate on all
  layers: y and LSE bit-identical to stock `gate_sink` on every path (GPU check, including captured pairs and
  replays); in serving its log-probability dump matches the sliding-window-only gate on 30 / 30 records, and greedy
  outputs equal v2.9's on 30 / 30 prompts. The default set (sink
  gate and prologue) is bit-identical in serving: greedy outputs identical to the controls on 30 / 30 prompts,
  log-probabilities bit-equal over 4,691 tokens (a control repeat is also 30 / 30 bit-equal). The hoist (v2.9.2)
  changes the FP32 summation order: 2 long prompts diverged without a near-tie (26 / 30 identical, 2 near-ties), the
  same class as the existing fused loop mix against stock (1 non-near-tie). For such levers the gate is model quality:
- Hoist quality bracket (`inkq` v2: 575 automatically graded items, every prompt >= 1.5K tokens so the hoist runs: 400
  GSM8K 10-shot, 100 long-context lookups at 4-32K tokens, 75 code / JSON / math / format tasks; T = 0, one box, five
  interleaved runs): v2.9.1 three times 556 / 559 / 555, the hoist twice 562 / 559 (gsm8k 392 / 391 against 387-390,
  long context 100 / 100 against 100, the 75 tasks 70 / 68 against 66-70). No drop: one hoist run (562) scored above
  the spread of the three baselines, the other inside it; over the six hoist-baseline pairs the hoist alone was right
  on 38 items and a baseline alone on 15. Results: `results/QA1-s05/`, `results/QA2-s05/`, `results/QA3-s05/`,
  `results/QL1-s05/`, `results/QL2-s05/`; the first, smaller bracket is in `docs/experiments.md`.
- Padding rows (`PADX`): live rows bit-identical in the GPU check; concurrent greedy outputs vary with the batch
  composition (two runs with padx off differ on 57 / 60), and padx on against off shows the same first-divergence
  class. With the v2.8.2 cap, greedy outputs match v2.8.1 on 29 / 30; the one
  difference (longctx-09) comes from the cap, not the gate (gate off, cap on gives the same outputs on 30 / 30), and
  is the near-tie described below.
- Draft-tail snapshots: the target path is untouched, so greedy outputs differ from the snapshot-free server only in
  the verify-schedule class. With the verify cap off, no generated token fell outside the reference's top-k; 27 / 30
  quick prompts were identical, and the long prompts and 48 agent turns showed near-ties only. With the cap on, the
  same. A control without snapshots that only fixes the verify width at 4 (same drafts) diverges on the same three
  prompts, with a larger shift (1.65 nat on longctx-02) than any the snapshots produced. The v2.9 release run's greedy
  outputs equal v2.8.4's on 30 / 30 prompts.
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
  log-probabilities at rounding level (the MoE tile tactics depend on per-expert token counts). v2.8.2's policy
  against uncapped: 27 / 30 identical, 2 top-2 near-ties and 1 three-way near-tie (longctx-09 at token 93: four
  candidates within 0.1 nat, ' It' -1.939, ' self' -1.991, ' In' -2.033, ' Need' -2.035; the capped run took the
  third at 0.094 nat), the same class as before.
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
| Drafter byte cuts: MXFP8 copies of the drafter's five layers' linears (the W8 kernels of the DSpark draft cuts); since v2.8.4 (`mx8w`) the 20 of 30 linears whose K and N are multiples of 128 run FlashInfer's stock b12x MXFP8 GEMM on the same MXFP8 bytes for 1-256 rows (activations quantised to MXFP8 per call; the 10 576-wide conv projections keep `mx8`); and an FP8 row-scaled copy of the target head restricted to the 65,536 most frequent ids for the selector's candidate projection (instead of the full 200,058-row BF16 head) | `DFLASH_BYTES=mx8,mx8w,hot8`, `DFLASH_HOT_N` (65536) | `mx8,hot8`: speedbench quick +8.4 %, prose c1 +6.5 % (ABBA, one box). `mx8w`: drafter linears 1.73-1.76x over BF16 cuBLAS (microbench), 0.33-0.36 ms faster than `mx8`'s GEMV at 8 / 16 rows; speedbench full +1.2 % (ABBA, box A) | draft proposals only (the selector's q is computed from the candidates the path is drawn from); greedy outputs identical 30 / 30 | ours; the hot-vocabulary list is the v2.1 one (idea: FR-Spec, Zhao et al.); `mx8w` runs **FlashInfer**'s b12x MXFP8 GEMM and quantiser |
| Drafter context projections on MXFP8 (opt-in since v2.9.6.1, `DFLASH_CTX8=1`): after every verify the drafter writes the committed rows into its context window through two BF16 projections (the context `fc` 9216 -> 2304 and the five layers' stacked K/V 2304 -> 10240, 89.7 MB); they read MXFP8 copies (47.7 MB) on FlashInfer's b12x GEMM for 1-256 rows, the rest of the append is the stock code | `DFLASH_CTX8=1`, `DFLASH_CTX8_MAX_M` (256); `0` = BF16 | fc + K/V append 0.447 -> 0.269 ms at 8 rows (eager, real weights); single-request decode step -0.195 ms (median per request, paired over 73 speedbench full requests, 54 / 73 faster; box B); speedbench tokens per step 2.849 vs 2.842; long context with fixed documents: tokens per step on the identically generated tokens 1.000 (32K) and 1.001 (128K) of the BF16 path (box A) | draft proposals only; greedy outputs identical 30 / 30 quick, 68 / 73 full (the other 5 deterministic, verify-schedule class); quality bracket quality bracket passes (inkq2: 557 / 557 against 557-561) | ours; runs **FlashInfer**'s b12x MXFP8 GEMM and quantiser (as `mx8w`) |
| Certified greedy LM head on DFLASH verify steps of <= 16 rows (c1 / c2): armed per step when the stock accept would take the plain argmax (no grammar, logprobs or logit adjustments) | `CERT_DFLASH=1` | c1 +2.8 % | identical argmax by construction (check mode: 34,808 rows, 0 mismatches) | ours (the v2 certified head, armed from the DFLASH worker) |
| Certified greedy LM head for 17-192-row DFLASH verify batches: the certified token ids go straight to the DFLASH accept | `CERT_WIDE_DFLASH` (on with `CERT_DFLASH`) | code c8 +3 %, no regression elsewhere | greedy texts identical on 30 / 30 prompts; per-row proof | ours (the v2.7 wide certified head) |
| Verify-width cap on DFLASH: per request and step, verify only the first L of the 8 rows; L grows left to right while the drafter's calibrated survival (per slot and selector-confidence bin, fitted on full-width DFLASH step logs) pays for the row; cut rows are remapped to the anchor row and get no experts at all. v2.8.2 policy `df-SLDF2x`: greedy and sampled requests are priced separately, with a context bucket; sampled requests verify all rows from 8 requests up, and a single greedy request at 64K tokens of context or more caps less; above 8 requests all rows are verified | `DFCAP` unset = `dfpolicy:/opt/inkling-overlay/inkling_vcap/policies/df-SLDF2x.json,bsmax=8`, `DFCAP_NOEXP=1`, `DFCAP_REMAP=1`; `DFCAP=off` disables it | v2.8 stack, cap on vs off, box A: speedbench quick 62.49 -> 70.61 (+13.0 %); prose c1 / c4 / c8 +17.5 / +14.4 / +6.8 %, code +16.7 / +18.8 / +12.5 %; c16 not capped. `df-SLDF2x` against the v2.8.1 policy (box A, ABBA): speedbench quick 73.17 -> 73.75, decode at 32K 61.27 -> 66.05 (+7.8 %), c4 / c8 and 128K within noise | target distribution unchanged (a stopping rule on draft-side information; GPU Monte Carlo law test, chain and block verification, with wrong values on the cut rows); greedy texts differ only at near-ties (`df-SLDF2x`: 27 / 30 identical) | ours (port of the DSpark verify-width policy) |
| Block verification on sampled DFLASH steps; under the cap the cut drafts enter as a stopped draft (sparse-q mask) | `DFLASH_BV=1` (`--speculative-use-block-verification`; `DFCAP_BV` under the cap); needs `RSFIX` with its `reject` part, off otherwise | thinking c1 +5.9 % (development) | target distribution unchanged (Sun et al.'s theorem; law test above) | **Sun et al.**, "Block Verification Accelerates Speculative Decoding" (ICLR 2025); SGLang's kernel |
| Fused sampled accept for the DFlash2 selector draft: the accept computed from the logits with a sparse draft q (no dense [requests, 7, vocabulary] q or [requests, 8, vocabulary] p), at 16 or more requests without a verify-cap cutoff | `DFLASH_FUSED=1` (`INKLING_DFLASH_FUSED_MIN_BS=16`) | thinking c24 per-request decode +2.3 % (ABBA, cap off); below 16 requests the stock accept | same law as the stock accept | ours |
| Prefix tail: on a prefix-cache hit of a prompt of at most 4,096 tokens, the last 2,048 prompt tokens are prefilled again (SGLang's own prefix bound, as for logprob requests), so the drafter's 2,048-token window is filled instead of drafting from the uncached suffix only | `DFLASH_PREFIX_TAIL=2048`, `DFLASH_PREFIX_TAIL_MAX_INPUT=4096`; `0` = off | speedbench full 69.59 -> 72.86 (long context 63.09 -> 73.53, agentic 73.44 -> 82.74), same box (v2.8.1 at `MEM_FRACTION=0.89`, v2.8 at 0.85); +0.1-0.5 s time to first token on bounded hits | prefill-only: the same positions a cold prefill computes (floating-point order of the prefill chunking only); draft-side benefit | ours |
| Draft-tail snapshots: the drafter's 2,048-position window at the end of each prompt is kept in one of 8 slots, keyed by content and cache salt, and copied back into the drafter ring on a later prefix-cache hit instead of a ring rebuilt from the uncached suffix | `DFLASH_TAIL_SNAPSHOTS=8`; `0` = off | with the prefix tail, against both off: agent loops -4.4 / -4.9 % end-to-end time at 160- / 512-token turns (decode 73.9 -> 78.3 tok/s); speedbench full long context +6.7 %, agentic +1.3 %, code +1.7 %; time to first token unchanged | draft-only state; greedy differences in the verify-schedule class (no token outside the reference's top-k); 320 MiB charged to the KV budget | ours |
| Prefix tail defers to a snapshot (v2.9.4): a prompt of at most 4,096 tokens whose prefix has a draft-tail snapshot covering the drafter's whole window keeps its prefix-cache hit, and the window is restored instead of re-prefilled | `DFLASH_TAIL_DEFER=1`; `0` = v2.9.3 | box B, 2 runs each way: speedbench full long context +2.1 % (tokens per step 2.97 -> 3.08, time to first token on these hits 0.45 -> 0.19 s), all prompts +0.4 %; agent mode unaffected (histories above 4,096 tokens); decode at 32K / 128K flat | cached prefix in place of recomputed prefill (floating-point order); one extra 128-token page per snapshot slot (+20 MiB of KV budget) | ours |
| Prompt copy: for prompts of 8K tokens or more, when the last 12 committed tokens recur earlier in the request's context with a match of at least 16 tokens, the tokens that followed replace the drafter's proposals for the rest of the block (two fused kernels per decode step, no host sync) | `DFLASH_PCOPY=12`, `DFLASH_PCOPY_MIN_CTX=8192`; `DFLASH_PCOPY=0` = off | speedbench `longctx` decode at 32K +9.8 %, 128K +2.4 % (mean of four runs per arm in two ABBAs, box A); speedbench full, agent mode and thinking c8 flat; single-request step within 0.1 ms | draft proposals only; sampled copies carry a point mass in the draft q (exact acceptance law, GPU Monte Carlo test); greedy outputs identical on speedbench full | ours; the copy rule is n-gram prompt lookup (idea: prompt lookup decoding, **Apoorv Saxena**, 2023) |
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
| Padding rows of CUDA-graph batches get no experts: a graph replays at its captured batch size, and its padding rows (which carry the previous replay's inputs) route to the phantom expert instead of 8 real ones; decided per graph at capture (graphs of 1-8 requests never pad and keep the stock route; 10 and up get the mask) | `PADX=1` | release, same session: thinking c9 / c15 +4.8 / +5.0 % aggregate (c15 per-request +6.1 %), speedbench full flat | bit-identical for live rows; small graphs bit-identical to stock on every row; padding rows are never read | ours |
| Prefill attention with the relative-position bias computed inside the kernel (FA2-style, mma.sync, paged KV, sliding window and global layers), instead of a materialised [t, 24, 1024] bias read by FA4 | `ATTN=prefill_swa,prefill_global` | SWA attention 3.3x vs the round-1 bias_layout path; cold prefill 8K / 32K / 128K +15 / +18 / +26 % | same math; error vs FP32 reference equal to stock's | ours (FlashAttention-2 algorithm, Tri Dao) |
| Prefill glue fusion: the loop-message mixes, attention-residual mix + RMSNorm and the MLP short conv + residual update in fused kernels that skip the FP32 [T, 2304] round trips | `PFUSE=all` | -187 ms per 8K chunk (microbench); with the fused attention: cold prefill 1K / 8K / 32K / 128K +14 / +29 / +30 / +36 % | same per-element math; only FP32 row-sum order differs (error vs FP64 equal or lower), short conv bit-identical | ours |
| Prefill loop-bank hoist: the loop controller's source projection (attention-residual weights, message normalisation, the message-only part of the controller down projection) prepared once per loop pass | `PFUSE_HOIST=1` (v2.9.2; `0` = off) | one 8K loop pass 104.5 -> 93-94 ms; cold prefill +1.1 / +1.7 / +1.4 % at 8K / 32K / 128K against v2.9.1 | changes the FP32 summation order, not the math (99.96 % of elements bit-equal; FP64-reference error within 1.5x of the fused path); quality bracket: no drop (see Correctness) | ours |
| Sink gate in the prefill attention epilogue, coalesced: sliding-window layers (v2.8.2) and global layers, BF16 and packed paths (v2.9.1) | `ATTN_GATE=all` (`1` = sliding-window only) | attention + gate 2,759 -> 1,944 us per sliding-window visit at 8K (-32.6 ms per chunk); global layers -840 us per 8K visit (-6.7 ms per chunk), +0.4-0.5 % prefill against sliding-window only | bit-identical to SGLang's `gate_sink`, padded rows zeroed; serving 30 / 30 identical with the gate on all layers | ours |
| Sliding-window prefill prologue: K conv + L2 clip and V conv + value norm, one kernel each, in the stock kernel's row-sum order | `PROLOGUE=1` | about -8.5 ms per 8K prefill chunk | bit-identical (0 ulp at 2,048-8,192 rows; serving 30 / 30 identical, log-probabilities bit-equal) | ours |
| Prefill: relative-position bias written directly in FA4's sheared layout | `PREFILL=bias_layout` | with quant_once: 8K prefill 3,781 -> 4,253 tok/s | bit-identical (attention output and LSE) | ours |
| Prefill: MoE input quantised to NVFP4 once per token, then gathered to its 8 experts | `PREFILL=quant_once` | (with bias_layout, above) | bit-identical | ours |
| Prefill: MoE combine writes BF16 directly | `PREFILL=partial_bf16` | combine -22 to -32 %; with qkvr_split: 8K 3,408 -> 3,781 tok/s | bit-identical | ours |
| SGLang DSpark cookbook cell (FA4, page 128, 8 draft tokens, 4 `SGLANG_OPT_*` switches) | `SPEC=dspark` | base of the fallback profile | - | SGLang |
| One-pass checkpoint load: SGLang reads the safetensors file directly and large frees return to the OS before KV allocation | `LOAD_NOMMAP=1` | 121-137 s warm start-up in the v2.6 full benchmarks | weights unchanged | SGLang's loader flag; [sxuff](https://github.com/sxuff/inkling-2-mini-spark) and [0xSero](https://github.com/sybil-solutions/inference-optimisation-skill) for the load-strategy idea |
| KV pool sizing | `MEM_FRACTION=0.88` with DFlash2 and the prefill levers on (0.89 with all three off; 0.85 above `CTX=262144`), `SWA_RATIO=0.037` | v2.9: 2,371,200 global-layer KV tokens (box B; the snapshot slots take 0.312 GiB of the budget), 4.6 GiB MemAvailable at worst (v2.8.3: 2,352,000 and 5.3 GiB; 0.89 with the gate stopped at the guard) | exact | ours |

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
scripts/launch.sh up                      # SGLang + DFlash2 + overlays on 127.0.0.1:8000, packed KV store, mem 0.88
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
| `DFLASH_CTX8` (`INKLING_DFLASH_CTX8`) | `0` (opt-in since v2.9.6.1) | `SPEC=dflash2` with the overlays: the drafter's two context projections, which write the committed rows into the draft window after every verify (the context `fc` 9216 -> 2304 and the five layers' stacked K/V 2304 -> 10240), read MXFP8 copies on FlashInfer's b12x GEMM for 1-256 rows (larger calls keep BF16; +48 MB of copies, every b12x tile plan compiled at start-up); draft proposals only; online weight updates are refused while it is on (the copies are derived weights); not armed when SGLang's `SGLANG_OPT_DFLASH_AUX_GRAPH` is set; `0` = BF16 (v2.9.5 behaviour). Set by the launcher only |
| `DFLASH_CTX8_MAX_M` (`INKLING_DFLASH_CTX8_MAX_M`) | empty (= 256) | largest context append (rows) that `DFLASH_CTX8` runs on MXFP8, 1-4096 |
| `PROMPT_TOUCH` (`INKLING_PROMPT_TOUCH`) | `1` | after a request finishes, re-touch its prompt path in the prefix cache (`inkling_ptouch`; eviction order only, nothing computed changes); `0` = off (v2.9.1 behaviour) |
| `ATTN_GATE` (`INKLING_ATTN_GATE`) | `all` | sink gate in the prefill attention epilogue on the sliding-window and global layers (`inkling_attn`; bit-identical to `gate_sink`); `1` = sliding-window layers only; off by itself while `SGLANG_KERNEL_API_LOGLEVEL` is non-zero (that logging reads the output buffer the gate leaves unwritten); `0` = off |
| `PROLOGUE` (`INKLING_PROLOGUE`) | `1` | sliding-window prefill K conv + L2 clip and V conv + value norm, one kernel each (`inkling_prologue`; bit-identical, stock row-sum order); `0` = off |
| `CERT_WIDE` (`INKLING_CERT_WIDE`), `CERT_WIDE_MAX_ROWS` | `SPEC=dspark`: `1`, overlay default | certified greedy head for 17-192-row verify batches (needs `CERT_HEAD`, `PICO12` and `WIDE` with `lmhead`) |
| `INKLING_ATTN_W8_FROM` | `131072` | prefix length (keys) from which global-layer prefill uses the 8-warp fused kernel; `0` = one 4-warp launch |
| `KV12S` (`INKLING_KV12S`), `KV12S_POOL` (`INKLING_KV12S_POOL`) | `1`, `static` | packed global KV store (see "Packed global KV store"); turns `KV12` off. `static` runs on SGLang's stock pools with prefill CUDA graphs, `MEM_FRACTION=0.89` and `SWA_RATIO=0.037` unless set; `unified` (`SPEC=dspark` only) is the v2.7.1 binding (`--enable-unified-memory --cuda-graph-backend-prefill=disabled`, 0.91 / 0.035). `0` = the BF16 store (`MEM_FRACTION` 0.85 with DFlash2, 0.93 with DSpark); vLLM and `OVERLAY=0` always use `0`. Set the store with these variables, not through `SGLANG_ENV` / `SGLANG_ARGS` (the launcher refuses the store's variables there, and with `KV12S=1` also the pool's engine flags, so the engine flags and the overlay always agree). `KV12S_OVERFLOW_HEADS` (2,048) and `KV12S_FALLBACK_PAGES` (64) size the exact overflow arena and fallback pool per layer visit |
| `KV12S_SETTLE` (`INKLING_KV12S_DFLASH_SETTLE`) | `shared` | `SPEC=dflash2` with `KV12S=1`: after each verify step the 8 packed global visits settle the step's tentative writes. `shared` builds the page candidates once for all visits (as the DSpark path does since v2.7.3) instead of once per visit: 77 fewer launches per step, single-request decode step -0.25 ms (v2.9.5). Exact: the same K/V bits and attention outputs. `1` also runs the trim and commit kernels once over all visits (14 launches per step instead of 112); it is exact as well, but in the v2.9.5 measurements it was slower at 4 and 8 requests in flight, so it is opt-in. `0` = each visit on its own (v2.9.4). The store's bytes are identical in every mode except where one visit's side-slot claims race near arena exhaustion, which the per-visit path resolves in scheduling order as well |
| `NOEXP` (`INKLING_NOEXP`) | `SPEC=dspark`: `1` | dead verify rows get no experts (inactive when `VERIFY_CAP=off` or its remap is off); `0` = off |
| `PADX` (`INKLING_PADX`) | `1` | padding rows of decode / verify CUDA-graph batches route to the phantom expert, so the expert GEMMs skip them (`inkling_padx`; per graph at capture: graphs of 10 requests and up; live rows bit-identical; composes with the dead-verify-row skip); `0` = off |

Left at SGLang's defaults (measured, see docs/experiments.md): chunked prefill 8,192 tokens (16,384: +2-3 % at
32K-128K, more memory), radix prefix cache on, overlap scheduler on, prefill CUDA graphs on (off only with
`KV12S_POOL=unified`).

Extra engine flags go in `SGLANG_ARGS` (or after `scripts/launch.sh up`), extra container variables in `SGLANG_ENV`
(`-e NAME=value ...`); both are split on whitespace, so their values cannot contain spaces.

### Memory budget

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
[Full memory sweep (DSpark profile)](docs/experiments.md#v26-kv-pool-sizing)

To leave more RAM for other work, lower `MEM_FRACTION`. Each 0.01 frees ~1.09 GiB, taken from the KV pools
(the weights, graphs and fixed buffers do not shrink).

### What differs from the published recipes, and why

| | SGLang cookbook (DGX Spark, DSpark) | vLLM recipe (DGX Spark) | here |
|---|---|---|---|
| engine | SGLang | vLLM | **SGLang**: the stock engines decode at the same speed on one Spark (c1 within 3 %, c16 / c24 within run noise); SGLang prefills 1.3-3.4x faster, keeps long prompts in its prefix cache more reliably and leaves twice the host memory ([docs/experiments.md](docs/experiments.md)) |
| draft | DSpark (bundled) | DSpark (bundled) | **DFlash2** (incoai, pinned revision, sha256-checked); DSpark with `SPEC=dspark` or `ENGINE=vllm` |
| memory | `--mem-fraction-static 0.45` (518K KV tokens) | `--gpu-memory-utilization` unset (vLLM's own suggestion would leave the host 0-2 GiB) | `MEM_FRACTION=0.88`, `SWA_RATIO=0.037` and the packed global KV store give 2,371,200 global-layer KV tokens, 4.6 GiB MemAvailable under stress and a 3.5 GiB guard floor (v2.9, box B) |
| context | not set (model default) | 262,144 | 262,144 (the model supports 1M; see "Longer context") |
| multimodal | `--enable-multimodal` | on | off (text only; add `--enable-multimodal` for images) |
| images, weights | pulled by tag, weights from the Hub at start | same | pinned image digest, weights and drafter at pinned revisions, sha256-checked, offline at start |
| bind address | 0.0.0.0 | 0.0.0.0 | 127.0.0.1 |
| speed overlays | - | - | twenty-two packages (`overlay/`), on by default; `OVERLAY=0` disables them |

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
| `MEM_FRACTION` / `SWA_RATIO` defaults | 0.85 (DFlash2) or 0.93 (DSpark) / 0.05 | 0.88 (DFlash2 with the prefill levers) or 0.89 / 0.037 |
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
