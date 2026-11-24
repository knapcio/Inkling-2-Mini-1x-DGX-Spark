# Inkling-2-Mini + DSpark on one GB10: live-engine profile analysis (c1, c8)

Inputs: torch-profiler traces of the live SGLang engine (`/start_profile`, 40 steps, CPU+GPU, record_shapes),
one GB10 Spark, 2026-11-21 08:48 (c1) and 08:49 (c8); greedy prose, effort none, T=0. Server log of the same boot.
Script: `analyze_trace.py` (stdlib only). Full outputs: `report_c1.txt`,
`work/report_c8.txt` (+ `.json`). Reproduce:

```
python3 -I analyze_trace.py traces/P1-base-profile-c1 --label c1 --log traces/P1-base-profile.server.log --per-layer --ctx-tokens 384
python3 -I analyze_trace.py traces/P1-base-profile-c8 --label c8 --per-layer --ctx-tokens 3072
```

Method: steps are cut at the GPU-side `step[DRAFT bs=N]` annotations (one draft graph + one verify graph per
step). Forwards counted by the embedding kernel (draft 2, verify 1 per step) and by `_attention_prologue`
(exactly 48 per verify = one 48-visit target forward), never by MoE kernels. No kernel > 5 ms exists (nothing
dropped). Busy time is the union of kernel/memcpy intervals; per-class "excl ms" is exclusive timeline occupancy
(PDL overlaps inside the graphs are counted once), "raw" is summed duration (used for GB/s). Graph kernels are
identified by name, grid and their position in the layer sequence (qkvr -> attention -> wo -> router -> MoE ->
next qkvr; last GEMM = target LM head; after argmax = draft context K/V). Bytes come from `weights.md` / the
safetensors headers.

**Caveat on the workload:** the profiled prose accepts **1.89 tokens/step** (c1), not the 2.45 of speedbench.
Step time is nearly acceptance-independent (fixed 8-row verify), so the ms tables transfer; tok/s does not.

---

## 1. Step time, GPU busy vs wall

| ms per step | c1 mean | c1 p10 / p50 / p90 (min-max) | c8 mean | c8 p10 / p50 / p90 (min-max) |
|---|---:|---|---:|---|
| wall period (draft start -> next draft start) | **60.39** | 57.8 / 59.8 / 64.0 (53.3-67.4) | **126.29** | 122.5 / 126.7 / 130.1 (120.9-134.5) |
| GPU busy (union) | 59.99 | | 125.70 | |
| GPU idle in period (host gap) | **0.39** | 0.38 / 0.38 / 0.39 (max 0.80) | **0.59** | 0.58 / 0.59 / 0.59 |
| draft graph window | 9.72 | 9.66 / 9.68 / 9.79 | 11.05 | 11.00 / 11.04 / 11.09 |
| verify graph window | 50.52 | 48.0 / 50.0 / 54.1 | 114.89 | 111.1 / 115.3 / 118.7 |
| MoE grouped GEMMs (in verify) | 30.19 | 27.6 / 29.7 / 33.7 | 81.63 | 77.8 / 82.2 / 85.3 |
| busy minus MoE GEMMs | 29.80 | 29.69 / 29.77 / 30.00 | 44.07 | 43.83 / 44.03 / 44.41 |

- Steps captured: 40 draft + 40 verify windows per trace (39 complete periods). Each window is one CUDA graph
  (c1: draft gid 311, verify gid 242; c8: 290 / 221). No graph break inside either graph.
- **All step-to-step variance is MoE**: d(period)/d(MoE ms) = 1.01 (c1) and 0.99 (c8); the non-MoE part is
  constant to +-0.2 ms. The variance is the expert union changing with the routed tokens.
- **There is no host gap.** Period minus GPU busy is 0.39 ms at c1 (0.65 %) and 0.59 ms at c8. The overlap
  scheduler runs one step ahead: per step the CPU does 2 `cudaGraphLaunch` (~1.0-1.3 ms each, hidden), one
  `cudaEventSynchronize` (CPU *waits* ~54 ms of 60 for the GPU) and 3 async DtoH copies on a side stream.
  ~21-23 eager (non-graph) kernels per step cost 0.08 ms (c1) / 0.27 ms (c8). The 4-8 ms "wall vs graph"
  residue assumed in an earlier estimate does not exist in this engine: the step *is* GPU time.
- The profile run is not distorted: the log shows 29.4 tok/s at 1.77 accept during the window (= 60 ms/step)
  and ~31 tok/s at 1.93 outside it.

## 2. Kernel classes per step

### c1 (8 verify rows; mean period 60.39 ms)

| class | n/step | excl ms | % step | MB/step | GB/s |
|---|---:|---:|---:|---:|---:|
| MoE grouped GEMM NVFP4, layers 0-19 (44 visits) | 88 | **26.45** | 43.8 | u x 197 | see below |
| MoE grouped GEMM MXFP8, layers 20-23 (4 visits) | 8 | 3.83 | 6.3 | u x 33 | see below |
| dense qkvr, cuBLAS `nvjet` M=24 stacked (28 visits) | 28 | 5.13 | 8.5 | 1110 | **215** |
| dense qkvr, sm12x Triton GEMV M=8 (20 visits) | 20 | 3.43 | 5.7 | 793 | 231 |
| dense wo, sm12x GEMV | 48 | 2.92 | 4.8 | 679 | 233 |
| MoE router gate (b12x GEMV, 0.59 MB) | 48 | 0.32 | 0.5 | 28 | 89 (latency-bound) |
| target LM head (sm12x GEMV, FP32 out) | 1 | 3.80 | 6.3 | 926 | **244** |
| draft LM head / base logits (cuBLAS `cutlass_80_wmma` 16x16, M=7) | 1 | 3.90 | 6.5 | 926 | 238 |
| draft Markov fold `addmm` x7 (cuBLAS wmma 32x32, M=1) | 7 | 3.02 | 5.0 | 720 | 238 |
| draft layers (3 x qkv/o/gate_up/down GEMV + fa4 + norms) | 35 | 2.22 | 3.7 | 495 | 223 |
| draft Markov rnn_proj (cuBLAS `gemvx`, 768x5120) | 7 | 0.36 | 0.6 | 55 | **154** |
| draft context K/V proj (runs in the verify graph) | 3 | 0.15 | 0.3 | 28 | 185 |
| draft sampling (`_markov_sample_*`) + rnn gates/inputs | 28 | 0.08 | 0.1 | | |
| attention, sliding-window (40 visits; compact rel-bias verify kernel `_decode_partials` + `_combine_gate_sink`) | 160 | 0.95 | 1.6 | | |
| attention, global (8 visits) | 38 | 0.22 | 0.4 | | |
| MoE args / route top-k+metadata / act quant / combine | 376 | 1.02 | 1.7 | | |
| loop controllers (core passes 2-3) | 144 | 0.83 | 1.4 | 113 | 86 |
| attn-res mix | 144 | 0.50 | 0.8 | | |
| short conv | 96 | 0.40 | 0.7 | | |
| norms | 121 | 0.29 | 0.5 | | |
| softcap + argmax (target) | 3 | 0.03 | 0.1 | | |
| accept/commit + prep glue (in graphs and eager) | ~95 | 0.27 | 0.4 | | |
| **total** | ~1500 | **60.10** | 99.5 | 5875 + MoE + KV | |

Draft total = 9.72 ms window + 0.15 ms context K/V = **~9.9 ms (16 %)**: LM head 3.90, Markov fold 3.02,
layers 2.22, rnn_proj 0.36, sampling 0.08. Dense target BF16 (qkvr+wo+router) = **11.8 ms (20 %)**.

### c8 (64 verify rows; mean period 126.29 ms)

| class | n/step | excl ms | % step | MB/step | GB/s |
|---|---:|---:|---:|---:|---:|
| MoE grouped GEMM NVFP4 | 88 | **70.89** | 56.1 | u x 197 | see below |
| MoE grouped GEMM MXFP8 | 8 | 10.72 | 8.5 | u x 33 | see below |
| dense qkvr: 28 x `nvjet` M=192 stacked (231 us) + 20 x `cutlass_80_wmma` M=64 (200 us) | 48 | 10.39 | 8.2 | 1902 | **171 / 198** |
| dense wo, cuBLAS `cutlass_80_wmma` M=64 | 48 | 4.51 | 3.6 | 679 | **151** |
| router gate, cuBLAS `nvjet` split-K + reduce | 96 | 0.35 | 0.3 | 28 | 64 |
| target LM head, cuBLAS `nvjet` M=64 FP32 out | 1 | 4.49 | 3.6 | 926 | 206 |
| logits softcap + argmax (51 MB FP32 logits, twice) | 3 | 0.64 | 0.5 | | |
| draft LM head, cuBLAS `nvjet` M=56 | 1 | 4.40 | 3.5 | 926 | 211 |
| draft Markov fold x7 (wmma, M=8) | 7 | 3.38 | 2.7 | 720 | 213 |
| draft layers | 34 | 2.62 | 2.1 | 495 | 189 |
| draft rnn_proj + ctx K/V + sampling | 45 | 0.64 | 0.5 | 83 | 198-214 |
| attention, sliding-window (padded FA4 + ShearingBias + gate_sink: 64 tokens > `VERIFY_REL_BIAS_MAX_TOKENS`=32) | 200 | 5.01 | 4.0 | ~500 KV | ~90 |
| attention, global | 54 | 1.12 | 0.9 | ~100 KV | ~75 |
| MoE args/route/quant/combine | 376 | 2.31 | 1.8 | | |
| loop controllers | 144 | 1.49 | 1.2 | 113 | 47 |
| attn-res mix | 144 | 1.10 | 0.9 | | |
| short conv / norms / glue | ~330 | 1.49 | 1.2 | | |
| **total** | ~1650 | **125.65** | 99.5 | | |

### Dense projections: GEMV vs cuBLAS

- c1: qkvr runs on the sm12x Triton GEMV for 20 visits and on **cuBLAS for 28 visits** (the 3M=24-row stacked
  input of the 4 global layers in pass 1 and all 24 core visits of passes 2-3; exactly the `MAX_M=16` cut).
  wo (48) and target LM head stay on the GEMV. Other cuBLAS calls per step: draft LM head (1), Markov fold (7),
  rnn_proj gemvx (7), draft ctx K/V is GEMV. Total 43 cuBLAS calls/step.
- **The cuBLAS penalty at c1 is small**: nvjet M=24 reads 39.6 MB in 184.5 us (215 GB/s) vs 171.2 us (231 GB/s)
  on the GEMV, i.e. 13 us x 28 = **0.37 ms/step**, not 1-2 ms.
- c8: every dense GEMM is cuBLAS (M=64 / 192). wo runs at **151 GB/s**, stacked qkvr at **171 GB/s**.

### MoE: union back-solve and the MXFP8 check

Bytes per expert: NVFP4 4.479 MB (w13 2.986 + w2 1.493), MXFP8 8.211 MB (5.474 + 2.737). MoE bytes per
forward = u x 229.9 MB.

| | c1 NVFP4 | c1 MXFP8 | c8 NVFP4 | c8 MXFP8 |
|---|---:|---:|---:|---:|
| us per visit (w13 + w2) | 410 + 192 = 602 | 641 + 317 = 958 | 1092 + 520 = 1611 | 1766 + 913 = 2679 |
| w13/w2 time ratio (bytes ratio 2.00) | 2.14 | 2.02 | 2.10 | 1.94 |
| implied u at 220 GB/s | **29.5** | 25.7 | **79.1** | 71.8 |
| implied u at 240 GB/s | 32.2 | 28.0 | 86.3 | 78.3 |
| hard upper bound u (at 273 GB/s peak) | 36.7 | 31.8 | 98.2 | 89.1 |
| random union, independent top-8 draws from 128 | **51.6** (8 rows) | | **125.9** (64 rows) | |

- **Assumption:** the grouped GEMM streams each touched expert once per call at the same 220-240 GB/s the
  vocab-sized kernels reach in this trace (LM head GEMV 244, draft head 238, Markov fold 238 GB/s). Then
  **u ~ 30-32 at c1**, 57-62 % of a random union, and **u ~ 79-86 at c8** (requests share hot experts).
  Even at 273 GB/s the union cannot exceed 36.7 (c1): routing inside a verify block is strongly correlated.
- Evidence the MoE kernel is bandwidth-proportional, not overhead-bound: (1) step variance is 100 % MoE;
  (2) per visit, the implied u of w13 and w2 agree (regression u13 = 1.03 u2 + 1.2 at c1, 1.05 u2 + 0.4 at c8);
  (3) the cheapest visit of 1760 at c1 costs 222 us = u 11.9 at 240 GB/s (u >= 8 is forced by the anchor row).
- **Per-byte, w13 is 5-7 % slower than w2** (time ratio 2.10-2.14 vs bytes 2.00). That is the one concrete
  tile-tuning signal in the trace.
- Per stored layer (c1, u at 240): L0-5 37-42 (early layers spread most), core L6-16 26-37, **L17 24** (pass 1
  only 17), L18-23 27-30. c8: L0-5 101-111, core 80-93, L17 57, L18-23 71-82. Full table in the reports.
- **MXFP8 is a native FP8 grouped GEMM**: `flashinfer::gemm::cute_sm12x_gemm::sm120_blockscaled::
  SM120BlockScaledMoeGemmKernel<SM120BlockScaledBuilder<128, 8, 128, 4, 32, GemmType 5, true>>`, grid 48
  (persistent), fed by `mxfp8_quantize_routed<2304>` / `mxfp8_swiglu_quantize_routed<1152>`: W8A8 block-scaled
  MMA, no dequant, BF16 or Triton kernel anywhere in the layer 20-23 sequence. The averaged ratio is 1.59x (c1) /
  1.66x (c8) an NVFP4 visit, below the 1.83x bytes ratio, **only because layers 20-23 have smaller unions than
  layers 0-5**. Against the adjacent NVFP4 layers 18-19 it is **1.84x (c1) and 1.88x (c8)**: same cost per byte
  as the NVFP4 kernel.

## 3. Below-bandwidth kernels, syncs, graph breaks

| where | kernel | GB/s | ms/step | note |
|---|---|---:|---:|---|
| c1 | draft Markov rnn_proj, cuBLAS `gemvx` M=1, 7.9 MB | 154 | 0.36 | streaming GEMV would save ~0.12 ms |
| c1 | MoE router gate b12x GEMV, 0.59 MB | 89 | 0.32 | latency-bound (6.6 us per call) |
| c1 | loop controllers (`_controller_mix/down/norm`) | 86 | 0.83 | 144 tiny kernels |
| c1 | draft ctx K/V GEMV (N=2048) | 185 | 0.15 | borderline |
| c8 | dense wo, cuBLAS `cutlass_80_wmma` M=64 | **151** | 4.51 | at 230 GB/s: 2.95 ms |
| c8 | dense qkvr stacked, `nvjet` M=192 | **171** | 6.48 | unstacked M=64 runs 198 GB/s |
| c8 | router `nvjet` split-K + reduce | 64-79 | 0.35 | |
| c8 | loop controllers | 47 | 1.49 | |
| c8 | attention (padded FA4 + shearing) | ~75-90 | 6.13 | ~2.5x the 0.6 GB KV floor at 240 GB/s |
| c8 | target / draft LM head on `nvjet` | 206 / 211 | 8.9 | the c1 GEMV path reaches 238-244 |

Not below bandwidth (corrects earlier estimates): the Markov fold `addmm` runs at **238 GB/s** on cuBLAS
(an earlier estimate assumed ~175 GB/s and 4.1 ms; measured 3.02 ms), the draft LM head on cuBLAS at 238 GB/s,
and the cuBLAS qkvr at c1 at 215 GB/s.

Syncs / host round trips: exactly one `cudaEventSynchronize` per step (overlap scheduler collecting the
previous step's result; the GPU already holds the next step) and one `cudaDeviceSynchronize` in the whole trace
(profiler stop). No graph break inside draft or verify graphs; ~21-23 eager kernels per step between graphs
(accept/commit/prep, 0.08-0.27 ms). The idle inside the verify graph is 0.25 ms (c1) / 0.45 ms (c8) over
~1450 nodes.

## 4. Bytes-per-step roofline

Known bytes per step (exact, from headers): dense BF16 2.58 GB (qkvr 1.90 + wo 0.68) + router 0.03 +
controllers 0.11 + target LM head 0.93 + draft LM head 0.93 + Markov fold 0.72 + draft layers 0.50 + rnn_proj
0.06 + ctx K/V 0.03 = **5.88 GB**. KV reads: 48 visits x context tokens x 4 KiB (c1 384 tokens: 0.08 GB;
c8 3072 tokens total: 0.60 GB). MoE = u x 0.230 GB.

| | u | total GB | floor @220 | floor @240 | measured busy | busy / floor@240 |
|---|---:|---:|---:|---:|---:|---:|
| c1, u from MoE time @240 | 32.2 | 13.36 | 60.7 | 55.7 | 60.0 | 1.08 |
| c1, u from MoE time @220 | 29.5 | 12.74 | 57.9 | 53.1 | 60.0 | 1.13 |
| c1, earlier fitted u | 28.0 | 12.39 | 56.3 | 51.6 | 60.0 | 1.16 |
| c8, u from MoE time @240 | 86.3 | 26.33 | 119.7 | 109.7 | 125.7 | 1.15 |
| c8, u from MoE time @220 | 79.1 | 24.68 | 112.2 | 102.8 | 125.7 | 1.22 |

Non-MoE part, which has exact bytes: **c1 29.8 ms vs 24.8 ms floor at 240 (27.1 at 220)**; **c8 44.0 ms vs
27.0 ms (29.5)**. At c1 the 5 ms excess is almost entirely ~1000 small latency-bound kernels (attention 1.2,
MoE glue 1.0, controllers 0.8, attn-res 0.5, sconv 0.4, router 0.3, norms 0.3, rnn_proj, glue). At c8 the
17 ms excess is cuBLAS dense at 151-198 GB/s (~4.3 ms), cuBLAS LM heads at 206-211 (~1.2 ms), attention (~3.6 ms
over its KV floor) and the small-kernel tail (~8 ms).

## 5. Server log

`decode_log_interval=40`; only window means are logged (no per-step distribution). `accept len` = tokens per
step per request including the bonus token (accept rate = (len-1)/7).

| running reqs | 40-step windows | accept len mean | min | p50 | max | gen tok/s (median) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 20 | **1.89** | 1.43 | 1.90 | 2.48 | 29.7 |
| 8 | 18 | **2.00** (1.90 median; first window 3.74 is the prefix) | 1.75 | 1.90 | 3.74 | 124.1 |

Tokens per step: c1 1.89 (31.3 tok/s at 60.4 ms); c8 8 x ~1.95 = ~15.6 (124 tok/s at 126.3 ms).
Geometric fit: p = 0.47 per draft position (speedbench 2.45 fits p = 0.60, thinking 3.1 fits p = 0.695).

## 6. Where the remaining ms are, and which lever hits them

ms are per step at unchanged acceptance unless stated; c1 base 60.4 ms, c8 base 126.3 ms.

| rank | bucket (c1 ms) | lever | c1 estimate | c8 estimate | basis |
|---:|---|---|---:|---:|---|
| 1 | MoE expert union, 30.3 ms (50 %) | **(a) verify-width cap with dead-row expert remap** | static cap k=4: step -10.9 ms (49.5 ms) but tokens 2.46 -> 2.18: **+8 %** speedbench, +18 % prose (k=3), +2 % thinking; oracle rows = accepted+1: step 43 ms, **+37-46 %** | cap 4 rows/req: step 106.5 ms (-20 ms): +5 % (p=.6) to +15 % (p=.47, k=3) | union U(R)=128(1-exp(-c R^b)) fitted to U(1)=8, U(8)=32.2 (c1, b=0.72) and U(8)/U(64) (c8); 0.94 ms per expert of union; geometric acceptance. Adaptive cuts land between static and oracle. |
| 2 | draft, 9.9 ms (16 %) | **(b) draft byte cuts** | MXFP8 draft LM head **-1.9**; FP8 Markov fold **-1.5**; MXFP8 draft layers **-1.1**; rnn_proj on the GEMV -0.1; total **~-4.6 ms (+8 %)**. Hot-vocab (32K) instead of the first two: -5.8 alone, ~-7 with MXFP8 layers, costs some acceptance | -2.1 / -1.7 / -1.3: **~-5 ms (+4 %)** | measured 238 GB/s kept for the quantized kernels (risk: `mm_mxfp8` cute-dsl speed on sm_121 at M=7 is unmeasured) |
| 3 | dense target BF16, 11.8 ms (20 %) | **(e) exact 12-bit packed BF16 GEMV (dense + LM head)** | dense -2.9, target LM head -0.95: **~-3.8 ms (+7 %)**; +0.5 draft layers, +1.0 draft head if (b) is not taken | needs a >16-row kernel: dense 15.0 -> 8.4 ms **-6.6**, LM head -1.5 | 0.75x bytes at the measured 231-244 GB/s; c8 number includes replacing cuBLAS |
| 4 | target LM head 3.8 ms (6 %) | **(f) certified-argmax LM-head screen (T=0)** | **-1.9 (8-bit screen) to -2.8 ms (4-bit)** on certified steps | -2.5 to -3.6 (also removes softcap+argmax 0.64) | do not add to (e)'s LM-head part: combined LM-head saving <= ~2.9 ms at c1 |
| 5 | small-kernel tail incl. attention: ~4.9 ms c1 / ~14 ms c8 | not in the list: fuse router+topk+metadata+quant, controllers, attn-res/norms; c8 attention back to a compact kernel above 32 tokens | ~-1 to -2 ms (guess) | ~-3 to -6 ms (guess) | ~1000 kernels of 3-12 us each; measured occupancy only |
| 6 | MoE kernel efficiency | **(d) NVFP4 tile retune** | **~-1.2 ms** if w13 reaches w2's per-byte rate; more only if the true union is < 32 | **~-2.3 ms** | w13/w2 time ratio 2.10-2.14 vs bytes 2.00; the union counter decides the rest |
| 7 | 28 stacked qkvr on cuBLAS | **(c) QKVR on the GEMV (column split)** | **-0.37 ms** | column split -0.9; with a streaming >16-row GEMV for all dense -3.8 (overlaps (e)) | 184.5 vs 171.2 us per 39.6 MB call |
| - | host gap / wall-vs-graph | (none needed) | 0.39 ms total idle | 0.59 | nothing to win |

Stacked at unchanged acceptance, c1: (b) 4.6 + (e) 3.8 + (f) ~1.9 net of (e) + (d) 1.2 + (c) 0.4 (part of (e)
if the 12-bit kernel takes 24 rows) ~ **-12 ms -> ~48.5 ms/step (+24 %)**; (a) multiplies on top, because it
removes MoE bytes the others do not touch (MoE is 50 % of c1 and 65 % of c8).

## Findings that change earlier documents

1. No host gap: the step is 99.4 % GPU busy. The 4-8 ms host residue estimated earlier does not exist.
2. Markov fold and draft LM head already stream at 238 GB/s on cuBLAS. moving them to another GEMV is worth ~0.1 ms
   (rnn_proj only), not ~1 ms.
3. LEVERS #7 (QKVR off cuBLAS) is worth 0.37 ms at c1, not 1-2 ms; it matters at c8 (0.9-3.8 ms).
4. MoE union at c1 is ~30-32 per visit (hard bound 36.7), 60 % of random; MXFP8 cost per byte equals NVFP4.
5. c8 is where kernel inefficiency lives: non-MoE 44 ms vs a 27 ms byte floor (cuBLAS dense at 151-198 GB/s,
   padded FA4 attention above 32 verify tokens, small-kernel tail).
