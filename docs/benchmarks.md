# Benchmarks

Every speed, prefill, memory and start-up number of the recipe, with the run directories under `results/`. What each lever does is in [levers.md](levers.md); quality and qualification runs are in [quality.md](quality.md); the release-by-release history is in [CHANGELOG.md](../CHANGELOG.md).

- [Summary: stock, v2.7.2 and current](#summary-stock-v272-and-current)
- [How the runs were made](#how-the-runs-were-made)
- [speedbench](#speedbench)
- [Decode by prompt type and concurrency](#decode-by-prompt-type-and-concurrency)
- [Thinking workloads](#thinking-workloads)
- [Padding rows](#padding-rows)
- [Prefill and long context](#prefill-and-long-context)
- [Memory and start-up](#memory-and-start-up)

## Summary: stock, v2.7.2 and current

The current column is one GB10 Spark on 2026-11-24 with the release defaults (`SPEC=dflash2`, verify-width cap on, `CTX=262144`); the stock and v2.7.2
columns are the earlier two-box medians. Cells marked v2.9.6, v2.9.5, v2.9.4, v2.9.3, v2.9.2, v2.9.1 or v2.9 were measured on those release defaults
(box B unless stated); the others are the v2.8.x or v2.8 runs. v2.9 added draft-tail snapshots for agent loops; v2.9.1 runs the sink
gate on the global layers too (prefill); v2.9.2 adds the loop-bank hoist (prefill) and the prompt-end cache touch
(agent-loop prefix hits); v2.9.3 adds the prompt copy for prompts of 8K tokens or more (long-context decode); v2.9.4 lets the prefix tail defer to
a draft-tail snapshot on repeated cached prompts of up to 4K tokens (speedbench long context +2.1 %); v2.9.5 builds the
packed store's verify-settle page candidates once for all 8 global visits (exact; single-request step -0.25 ms); v2.9.6
runs the drafter's two context projections on MXFP8 copies (single-request step -0.2 ms; drafts only), opt-in since v2.9.7.
Single-request decode on short prompts is unchanged.

| | stock SGLang cookbook cell (DSpark, mem 0.62) | v2.7.2 | **current (v2.9.7)** |
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
one box, [experiments.md, "Screening results"](experiments.md#screening-results)). Do not compare across the two rows. In full mode v2.8.1 leads
v2.7.2 in every category except long context, where it is on par (73.53 vs 75.1). v2.8 trailed there (-16 %) and on
agentic prompts (-4 %) because the drafter drafted from an empty window after a prefix-cache hit, which speedbench
causes on every prompt; v2.8.1 fixes that (see [Prefix tail](levers.md#prefix-tail)).

Qualification status per release, and what v2.8 / v2.8.1 did not measure: [quality.md](quality.md#qualification-status).

### Long context vs the DSpark profile

At 32K context the DFlash2 profile decodes about 8-17 % slower than v2.7.2's DSpark profile at c1
(v2.8.2's verify-cap policy 66.1, v2.8.1 64.3, v2.8 60-63, v2.7.2 72 tok/s; different runs and boxes). Its acceptance
falls with context (3.10-3.17 vs 3.36 tokens per step at 32K), because the drafter sees only a 2,048-token window. At
128K both decode at about 50 tok/s. For mostly long-context single-user
work, `SPEC=dspark` is the faster choice. Since v2.9.3 the prompt copy narrows the gap where the answer quotes the
document: on speedbench `longctx` decode at 32K rises 10-14 % and at 128K 1.5-4.6 % (two boxes) (see [Prompt copy for long
documents](levers.md#prompt-copy-for-long-documents)).

## How the runs were made

Measured 2026-11-24 on one Spark per run, GPU clock capped at 2200 MHz by the host's own timer, fresh start of the
container, release defaults. Speed runs on box A (speedbench full: `results/V28sbf-s05/`; `results/V28M1v28-s05/`, cap
off: `results/V28M1off-s05/`, NOEXP off: `results/V28M1nonx-s05/`), memory and the worst case on box B
(`results/V28kv-s06/`), long context on box A (`results/V28lc-s05/`, v2.7.2 on the same box: `results/R272lc-s05/`;
`SPEC=dspark`: `results/V28dsp-s05/`). The stock engine was measured on both boxes with the same harness
(`results/F1-sglang-base-s05/`, `results/F2-sglang-base-s06/`).

## speedbench

speedbench 0.3 (`6733653`; repository link and checkout in the README's [Reproduce](../README.md#reproduce) section), full mode: 73 frozen prompts in six categories,
c1, temperature 0, 256 output tokens, reasoning effort low (the model always thinks; its reasoning tokens count as
decode tokens). Decode tok/s, geometric mean per category; 0 failed prompts.

| | prose | code | structured | agentic | multilingual | long context | all 73 prompts |
|---|---:|---:|---:|---:|---:|---:|---:|
| **v2.9.7** (release defaults, box A / box B) | **65.73 / 65.35** | **82.40 / 82.00** | **77.76 / 77.54** | **82.81 / 82.75** | **58.07 / 57.41** | **75.38 / 74.84** | **73.54 / 73.17** |
| v2.9.6 = v2.9.7 with `DFLASH_CTX8=1` (box A / box B) | 66.52 / 66.44 | 83.70 / 83.75 | 78.39 / 78.45 | 83.41 / 83.47 | 58.38 / 58.40 | 75.94 / 75.94 | 74.28 / 74.28 |
| v2.9.3 (box A / box B) | 65.05 / 65.21 | 81.63 / 81.81 | 77.05 / 77.16 | 82.04 / 82.21 | 57.44 / 57.58 | 73.26 / 73.30 | 72.65 / 72.79 |
| v2.9.2 (box B, release defaults) | 65.56 | 82.24 | 77.52 | 82.69 | 57.92 | 73.75 | 73.19 |
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

v2.9.7 is the v2.9.5 / v2.9.6 two-box qualification runs on its serving path (`results/Q295R-s05/`,
`results/Q295R-s06/`; v2.9.7's defaults run the same serving code). v2.9.6's row is its qualification
(`results/Q296R-s05/`, `results/Q296R-s06/`). Its speedbench is higher, but its drafter projections lost 2.3-2.6 % on
code at 8 requests in flight on box A in two sets, so v2.9.7 makes them opt-in. v2.9.3 is its qualification
(`results/Q293R-s05/`, `results/Q293R-s06/`).
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
(long context 75.00, agentic 82.42; [experiments.md](experiments.md#v281-prefix-hit-draft-starvation-and-the-prefix-tail)). Per prompt, the tail lifts tokens per step from 2.84 to 4.20 on
longctx-01 and from 10.8 to 17.0 on agentic-01. Counted end to end (time to first token plus decode, 256-token
outputs): long context +6.2 %, agentic +1.2 %, code +1.6 %.

## Decode by prompt type and concurrency

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

## Thinking workloads

`bench/think_load.py`: seeded arithmetic puzzles, reasoning effort high, temperature 0.8, aggregate completion tok/s
with the median per-request decode in brackets. c8 and c24: 48 requests x 2,048 max tokens.

| | c1 | c4 | c8 | c24 |
|---|---:|---:|---:|---:|
| v2.8 | - | - | **277.4** (38.2) | **515.5** (24.0) |
| v2.8, earlier boot, cap off | 82.5 | 177.9 | 288.4 | 518.9 |
| v2.7.2, same protocol | - | - | 235.3 | 438.8 |

One run per cell (sampled). The aggregate depends on how long the completions run, so compare the per-request decode
where it is given; c8 and c24 vary ~5 % from run to run.

## Padding rows

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

## Prefill and long context

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

## Memory and start-up

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
