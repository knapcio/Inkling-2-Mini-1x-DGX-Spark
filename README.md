# ⚡ Inkling-2-Mini on one DGX Spark

Serving [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on a single NVIDIA
DGX Spark (GB10, sm_121, 121.6 GiB unified memory) with the model's own NVFP4 / MXFP8 weights and the public DFlash2
draft model [incoai/Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) by **incoai**, on
SGLang, plus a set of import-time speed overlays that leave the target model's output distribution as it is
([what "exact" means here](docs/quality.md#quality-checks)).

**Current release: v2.9.7 (2026-11-26)**, the final release of this round of speed work.
[Changelog](CHANGELOG.md) · [Benchmarks](docs/benchmarks.md) · [Levers](docs/levers.md) ·
[Running it](docs/running.md) · [Quality](docs/quality.md) · [Experiments](docs/experiments.md)

## Headline numbers

One Spark per run, release defaults (`SPEC=dflash2`, `CTX=262144`); v2.9.7 values are the mean of two Sparks. "Stock" is
SGLang's DGX Spark cookbook cell (DSpark draft), measured with the same harness. All numbers, their `results/`
directories and [how the runs were made](docs/benchmarks.md#how-the-runs-were-made) are in
[docs/benchmarks.md](docs/benchmarks.md).

**speedbench full** (73 frozen prompts, c1, T=0, 256 tokens; decode tok/s, geometric mean per category):

| | prose | code | structured | agentic | multilingual | long context | **all 73** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **v2.9.7** | **65.54** | **82.20** | **77.65** | **82.78** | **57.74** | **75.11** | **73.36** |
| stock SGLang | 32.6 | 48.0 | 44.3 | 49.3 | 25.3 | 51.0 | 40.6 |

All 73 prompts: **+81 %** over stock's 40.6.

Against the other public single-Spark recipes, measured with one harness on the same hardware (speed, lossy or not,
quality, context, KV, prefill): [inkling-2-mini-spark-bench](https://github.com/knapcio/inkling-2-mini-spark-bench).

**Decode by concurrency** (`bench/decode_table.py`, T=0, 256 tokens; c1 is one stream, above c1 the aggregate;
v2.9.7 = mean of two Sparks, from the qualification runs on its serving path; stock is a two-box median; cells hold
3-32 requests, so single cells swing by a few %):

| tok/s | c1 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|
| prose, v2.9.7 | **66.2** | **138.5** | **214.4** | **296.2** |
| prose, stock | 33.2 | 75.2 | 117.0 | 186.2 |
| code, v2.9.7 | **74.4** | **139.6** | **212.2** | **302.2** |
| code, stock | 43.6 | 87.2 | 134.8 | 221.3 |
| structured, v2.9.7 | 75.4 | 143.2 | 209.7 | 316.5 |
| json, v2.9.7 | 80.4 | 154.1 | 217.4 | 340.5 |

**Agents, memory and context:**

- **Agent loops:** 24 agents with ~72K-token prompts on v2.9.7's defaults, 0 failures; the second turn's first token
  in ~3 s against ~370 s cold, on both Sparks. A 12-turn soak (288 requests, 0 failures) served
  every later turn from the prefix cache.
- **Memory:** worst MemAvailable 4.6 GiB over three rounds of thinking c24 with a cold 240K needle
  after each (v2.9.5 qualification, the serving code v2.9.7 runs); the lowest single sample in the 12-turn soak was
  3.8 GiB. Guard floor 3.5 GiB.
- **Context:** 262,144 tokens per request by default, from a shared pool of 2,371,200 global-layer KV tokens (v2.9);
  the model supports 1,048,576 (`CTX=1048576` on v2.8: needle PASS at 999,296 tokens). See
  [Longer context](docs/running.md#longer-context-512k-and-1m).
- **Prefill:** 5,657 tok/s cold at 8K tokens (v2.9.2; stock 3,399).

> **Long context, single user:** the drafter sees only a 2,048-token window, so at 32K context the DFlash2 profile
> decodes about 8-17 % slower than the DSpark profile at c1; at 128K both decode at about 50 tok/s. For mostly
> long-context single-user work, `SPEC=dspark` is the faster choice
> ([details](docs/benchmarks.md#long-context-vs-the-dspark-profile)).

## Quick start

Needs one DGX Spark with DGX OS and ~80 GB of disk; details in [docs/running.md](docs/running.md).

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

The Chat Completions API is at `http://127.0.0.1:8000/v1`, model name `inkling-2-mini`, on loopback only. Do not
start the server while a large download or copy is still running: on GB10 the page cache and the GPU share the same
memory. [Using it](docs/running.md#using-it) covers reasoning effort, tools and structured output.

## What's inside

One default profile: SGLang `lmsysorg/sglang:inkling-2-mini`, pinned by digest, drafting with DFlash2 (pinned at
revision `4f9088ea`, sha256-checked on the host and inside the container), and twenty-four overlay packages
(`overlay/`, on by default, `OVERLAY=0` disables them). The main levers:

- **DFlash2 drafter** (incoai): a 5-layer block-diffusion draft with a 2,048-token window and a 16-candidate
  selector; tokens per step x1.25 against stock DSpark. [Drafter levers](docs/levers.md#drafter-and-dflash-levers-default-profile)
- **Drafter byte cuts and certified heads:** MXFP8 drafter layers on FlashInfer's MXFP8 GEMM, an FP8 64K
  hot-vocabulary plane, and certified greedy LM heads that return the exact argmax.
  [Drafter levers](docs/levers.md#drafter-and-dflash-levers-default-profile)
- **Verify-width cap with no expert work for dead rows:** a calibrated stopping rule on the drafter's own
  probabilities; cut rows and the padding rows of CUDA-graph batches route to a phantom expert.
  [Padding rows](docs/benchmarks.md#padding-rows)
- **Agent loops and the prefix cache:** the [prefix tail](docs/levers.md#prefix-tail),
  [draft-tail snapshots](docs/levers.md#draft-tail-snapshots) and the
  [prompt-end cache touch](docs/levers.md#prefix-cache-in-agent-loops-prompt-end-touch).
- **Prompt copy for long documents:** decode at 32K +10-14 %.
  [Prompt copy](docs/levers.md#prompt-copy-for-long-documents)
- **Packed 12-bit global KV store:** exactly coded, bit-identical KV with 1.33x fewer bytes.
  [Packed KV store](docs/levers.md#packed-global-kv-store-default-kv12s1)
- **Prefill kernels:** fused attention with the relative bias in-kernel, glue fusion, sink gate, sliding-window
  prologue and the loop-bank hoist. [Prefill levers](docs/levers.md#prefill-levers)
- **Decode kernels:** pico's 12-bit GEMV, NVFP4 expert scale codes, GB10 tile tables and the split-KV verify
  attention. [Target-side levers](docs/levers.md#target-side-levers-both-profiles-and-dspark-only-levers)
- **Exact sampled verification:** `RSFIX`, block verification and a fused sampled accept.
  [Quality checks](docs/quality.md#quality-checks)

What was measured and dropped: [Tried and not adopted](docs/levers.md#tried-and-not-adopted) and
[Open ideas, not pursued](docs/levers.md#open-ideas-not-pursued).

## Version history

speedbench full, all 73 prompts, c1 tok/s (mean of the Sparks it was run on). Every release in detail:
[CHANGELOG.md](CHANGELOG.md); every historical speedbench row: [docs/benchmarks.md](docs/benchmarks.md#speedbench).

| release | speedbench all-73 | headline change |
|---|---:|---|
| [v2.9.7](CHANGELOG.md#v297-2026-11-26) | **73.36** | MXFP8 drafter context projections opt-in (`DFLASH_CTX8=1`); serving code = v2.9.5 |
| [v2.9.6](CHANGELOG.md#v296-2026-11-26) | 74.28 | MXFP8 copies of the drafter's context projections (code c8 -2.3 to -2.6 % on one Spark) |
| [v2.9.5](CHANGELOG.md#v295-2026-11-26) | 73.36 | packed store's verify settle builds page candidates once for all 8 global visits |
| [v2.9.4](CHANGELOG.md#v294-2026-11-25) | 73.21 | prefix tail defers to a draft-tail snapshot (long context +2.1 %) |
| [v2.9.3](CHANGELOG.md#v293-2026-11-25) | 72.72 | prompt copy for prompts of 8K tokens or more |
| [v2.9.2](CHANGELOG.md#v292-2026-11-25) | 73.04 | prefill loop-bank hoist; prompt-end cache touch for agent loops |
| [v2.9.1](CHANGELOG.md#v291-2026-11-25) | 73.12 | sink gate on the global layers too (prefill) |
| [v2.9](CHANGELOG.md#v29-2026-11-25) | 73.34 | draft-tail snapshots (agent loops -4.4 / -4.9 % end-to-end time) |
| [v2.8.4](CHANGELOG.md#v284-2026-11-25) | 73.48 | drafter MXFP8 linears on FlashInfer's MXFP8 GEMM (+1.2 %) |
| [v2.8.3](CHANGELOG.md#v283-2026-11-25) | 72.90 | no expert work for padding rows (thinking c9 / c15 +4.8 / +5.0 %); bit-identical prologue |
| [v2.8.2](CHANGELOG.md#v282-2026-11-24) | 72.80 | prefill sink gate; verify-cap policy `df-SLDF2x` |
| [v2.8.1](CHANGELOG.md#v281-2026-11-24) | 72.86 | prefix tail fixes prefix-hit draft starvation |
| [v2.8](CHANGELOG.md#v28-2026-11-24) | 69.63 | DFlash2 drafter becomes the default |
| [v2.7.2](CHANGELOG.md#v272-2026-11-24) | 63.8 | DSpark profile: packed KV store on the stock pools; dead verify rows get no experts |

## Reproduce

```bash
git clone https://github.com/knapcio/speedbench speedbench && git -C speedbench checkout 6733653
scripts/launch.sh up                      # release defaults, as in the full benchmarks
#   or: SPEC=dspark OVERLAY=0 LOAD_NOMMAP=0 MEM_FRACTION=0.62 SWA_RATIO=0.1 scripts/launch.sh up  # stock cookbook cell
#   or: OVERLAY=0 scripts/launch.sh up     # stock DFLASH with the drafter (no block verification, BF16 store, mem 0.85)
bench/full.sh MYRUN                       # ~75 min, writes results/MYRUN/ (the v2.8.x qualification protocol)
python3 bench/report.py results/MYRUN      # README-style tables
bench/screen.sh MYSCREEN                   # ~10 min screen: speedbench quick, thinking c8, 32K prefill
```

The overlays' GPU checks and unit tests: [docs/running.md](docs/running.md#reproducing-the-numbers).

## Documentation

| file | contents |
|---|---|
| [docs/benchmarks.md](docs/benchmarks.md) | speedbench (all historical rows), decode by concurrency, thinking, padding rows, prefill, long context, memory and start-up |
| [docs/levers.md](docs/levers.md) | what every overlay does, with switch, effect, exactness and credit; prefix tail, snapshots, prompt copy, packed KV store; tried and not adopted; open ideas |
| [docs/running.md](docs/running.md) | requirements, the API, the DSpark and vLLM fallbacks, every flag, memory budget, 512K / 1M context, differences from the published recipes, checks |
| [docs/quality.md](docs/quality.md) | qualification status per release and every quality / exactness check |
| [docs/model.md](docs/model.md) | the model and the drafter, read from their configs |
| [docs/experiments.md](docs/experiments.md) | every screen, the engine comparison, rejected levers and closed experiments |
| [docs/profile/PROFILE.md](docs/profile/PROFILE.md), [docs/prefill/PREFILL.md](docs/prefill/PREFILL.md) | decode profile and prefill roofline analyses |
| [docs/sglang-source-survey.md](docs/sglang-source-survey.md), [docs/weight-bytes.md](docs/weight-bytes.md) | serving-source survey and weight bytes per layer |
| [CHANGELOG.md](CHANGELOG.md) | every release, with a table of releases at the top |
| [overlay/README.md](overlay/README.md) | the overlay packages and how they load |

## Credits and licences

Repository-authored scripts, overlays, tests and documentation are MIT (see `LICENSE`). `overlay/inkling_pico12/third_party/pico/`
holds the GEMV sources by Andy Shu (dsgb10 / pico), unmodified, under their MIT licence (`LICENSE` next to
them); `overlay/inkling_wide/csrc/bf16x_format.cuh` copies pico's `bf16x` reader under the same licence
(`overlay/inkling_wide/third_party/pico/LICENSE`); the packed-KV attention kernel in `overlay/inkling_kv12/ops.py`
follows SGLang's Inkling2 split-KV decode kernel (Apache-2.0) with its K / V tile loads replaced; `overlay/inkling_rsfix/`
carries the code of SGLang PRs #51681 and #51692 (Apache-2.0); see `NOTICE`. The serving images (SGLang, vLLM) are
Apache-2.0 and are used as published, pinned by digest. The Inkling-2-Mini weights are Apache-2.0 from Thinking
Machines Lab. The DFlash2 drafter and its SGLang compatibility patch (incoai/Inkling-2-Mini-DFlash2) are Apache-2.0;
`scripts/download.sh` fetches them at a pinned revision, and nothing of them is distributed here.

See [CREDITS.md](CREDITS.md) for the work this builds on.
