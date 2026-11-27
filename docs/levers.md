# Levers: what the overlays do

How the default profile is built, what every overlay changes and what it was measured to gain. The measurements behind each lever are in [benchmarks.md](benchmarks.md) and [experiments.md](experiments.md); the switches are listed in [running.md](running.md#every-flag-explained).

- [The default profile](#the-default-profile)
- [Why these levers](#why-these-levers)
- [Drafter and DFLASH levers (default profile)](#drafter-and-dflash-levers-default-profile)
- [Target-side levers (both profiles) and DSpark-only levers](#target-side-levers-both-profiles-and-dspark-only-levers)
- [Prefix tail](#prefix-tail)
- [Draft-tail snapshots](#draft-tail-snapshots)
- [Prefix cache in agent loops (prompt-end touch)](#prefix-cache-in-agent-loops-prompt-end-touch)
- [Prompt copy for long documents](#prompt-copy-for-long-documents)
- [Prefill levers](#prefill-levers)
- [Packed global KV store (default, `KV12S=1`)](#packed-global-kv-store-default-kv12s1)
- [Image input (`MM=1`)](#image-input-mm1)
- [FP8 KV](#fp8-kv)
- [Tried and not adopted](#tried-and-not-adopted)
- [Open ideas, not pursued](#open-ideas-not-pursued)

## The default profile

One default profile: SGLang `lmsysorg/sglang:inkling-2-mini`, pinned by digest, drafting with DFlash2, a 5-layer
block-diffusion draft with a 2,048-token draft window and a 16-candidate selector, pinned at revision `4f9088ea`.
`scripts/launch.sh` checks the drafter's sha256 on the host and again inside the container on every start, and the
drafter's SGLang compatibility patch runs only inside the throwaway (`--rm`) serving container. Twenty-four overlay
packages (`overlay/`) run on top of it by default: drafter byte cuts (MXFP8 drafter layers, since v2.8.4 on FlashInfer's MXFP8 GEMM where
the shapes allow, and an FP8 64K hot-vocabulary plane for the candidate projection), the prefix tail and draft-tail snapshots that keep the drafter's window
filled on prefix-cache hits (v2.8.1 / v2.9, see [Prefix tail](#prefix-tail) and [Draft-tail snapshots](#draft-tail-snapshots)), the certified greedy LM head on verify steps, the packed 12-bit global KV store,
the exact sampled-verification fixes, block verification and a fused accept on sampled steps, a calibrated
verify-width cap with dead-row expert skip (v2.8.2: priced separately for greedy and sampled requests, with a context
bucket), no expert work for the padding rows of CUDA-graph batches (v2.8.3, see [Padding rows](benchmarks.md#padding-rows)), two bit-identical
prefill kernels, the sink gate (sliding-window layers since v2.8.2, global layers since v2.9.1) and the sliding-window
prologue (v2.8.3) and the loop-bank hoist (v2.9.2), see [Prefill levers](#prefill-levers); the prompt-end cache touch for agent loops
(v2.9.2, see [Prefix cache in agent loops](#prefix-cache-in-agent-loops-prompt-end-touch)), the prompt copy for long prompts (v2.9.3, see [Prompt copy for long
documents](#prompt-copy-for-long-documents)), and the target-side kernel, prefill and
memory levers of v2.7.2. The
target distribution is unchanged; the hoist changes only the floating-point summation order of one projection (quality
checked with a five-run bracket, no drop). `SPEC=dspark` runs the model's own DSpark draft instead (the v2.7.2 profile, see
[Fallback: the DSpark profile](running.md#fallback-the-dspark-profile)).

## Why these levers

The overlays cut bytes and kernel time per decode step without touching the target's math. The analysis behind them
was done on the DSpark profile: decode at c1 is bound by memory bandwidth (a step read ~13.4 GB: BF16 dense projections
2.8 GB over 48 layer applications, LM head 0.93 GB, DSpark draft ~1.65 GB, and the routed experts touched by the
8 verify rows, ~6.5 GB) and the stock engine ran at 1.08x the 240 GB/s floor with the GPU 99 % busy
([profile](profile/PROFILE.md)). With several requests in flight the step is still mostly bytes (c8: 117 ms, of
which 76 ms routed experts), but kernels built for one row lose time there: cuBLAS streams the dense projections at
150-200 GB/s, and above 32 verify rows the attention left the split-KV kernel for padded FA4 with a sheared bias. At
long context the global layers' BF16 KV dominates the step (4.3 GB at 128K). Prefill is the opposite: the kernels run
near their peaks, but intermediates make round trips through memory (FP32 [T, 2304] activations between the
loop-message mixes, norms and residual updates), the relative position bias was materialised as a [t, 24, 1024]
tensor, and the stacked Q/K/V projection computed three times the rows it keeps
([prefill analysis](prefill/PREFILL.md), [round-2 profiles](experiments.md#round-2-v2)).

The levers are switched by environment variables of `scripts/launch.sh`. The overlays are import-time patches in
`overlay/` (`OVERLAY=0` disables them). Effects are from screens against same-box, same-day controls (speedbench
quick: 30 prompts x 160 tokens, c1, T=0; decode table; thinking load; at long context the step time = tokens per step
/ decode tok/s). Raw files: `results/screen/`, `results/screen-r2/`; details and rejected levers:
[experiments.md](experiments.md).

## Drafter and DFLASH levers (default profile)

| lever | switch (default) | effect, alone | output | credit |
|---|---|---|---|---|
| DFlash2 drafter: a 5-layer block-diffusion draft that proposes a block of 8 positions in one pass over a 2,048-token sliding window of target features (layer visits 5, 23, 41, 48), keeps 16 candidates per position, and a selector traces one path through them (7 drafts per verify step) | `SPEC=dflash2` (`--speculative-algorithm DFLASH`) | stock against stock DSpark: tokens per step x1.25, prose decode c1 / c4 / c8 / c16 +43 / +46 / +40 / +39 %, code x1.22-1.25, thinking c8 +19 % | draft proposals only; the target verifies every token | **incoai**, [Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) and its SGLang compatibility patch (Apache-2.0); block-diffusion drafting from the **DFlash** authors ([z-lab/dflash](https://github.com/z-lab/dflash)); SGLang's DFLASH worker |
| Drafter byte cuts: MXFP8 copies of the drafter's five layers' linears (the W8 kernels of the DSpark draft cuts); since v2.8.4 (`mx8w`) the 20 of 30 linears whose K and N are multiples of 128 run FlashInfer's stock b12x MXFP8 GEMM on the same MXFP8 bytes for 1-256 rows (activations quantised to MXFP8 per call; the 10 576-wide conv projections keep `mx8`); and an FP8 row-scaled copy of the target head restricted to the 65,536 most frequent ids for the selector's candidate projection (instead of the full 200,058-row BF16 head) | `DFLASH_BYTES=mx8,mx8w,hot8`, `DFLASH_HOT_N` (65536) | `mx8,hot8`: speedbench quick +8.4 %, prose c1 +6.5 % (ABBA, one box). `mx8w`: drafter linears 1.73-1.76x over BF16 cuBLAS (microbench), 0.33-0.36 ms faster than `mx8`'s GEMV at 8 / 16 rows; speedbench full +1.2 % (ABBA, box A) | draft proposals only (the selector's q is computed from the candidates the path is drawn from); greedy outputs identical 30 / 30 | ours; the hot-vocabulary list is the v2.1 one (idea: FR-Spec, Zhao et al.); `mx8w` runs **FlashInfer**'s b12x MXFP8 GEMM and quantiser |
| Drafter context projections on MXFP8 (opt-in since v2.9.7, `DFLASH_CTX8=1`): after every verify the drafter writes the committed rows into its context window through two BF16 projections (the context `fc` 9216 -> 2304 and the five layers' stacked K/V 2304 -> 10240, 89.7 MB); they read MXFP8 copies (47.7 MB) on FlashInfer's b12x GEMM for 1-256 rows, the rest of the append is the stock code | `DFLASH_CTX8=1`, `DFLASH_CTX8_MAX_M` (256); `0` = BF16 | fc + K/V append 0.447 -> 0.269 ms at 8 rows (eager, real weights); single-request decode step -0.195 ms (median per request, paired over 73 speedbench full requests, 54 / 73 faster; box B); speedbench tokens per step 2.849 vs 2.842; long context with fixed documents: tokens per step on the identically generated tokens 1.000 (32K) and 1.001 (128K) of the BF16 path (box A) | draft proposals only; greedy outputs identical 30 / 30 quick, 68 / 73 full (the other 5 deterministic, verify-schedule class); quality bracket quality bracket passes (inkq2: 557 / 557 against 557-561) | ours; runs **FlashInfer**'s b12x MXFP8 GEMM and quantiser (as `mx8w`) |
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

## Target-side levers (both profiles) and DSpark-only levers

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
| Prefill loop-bank hoist: the loop controller's source projection (attention-residual weights, message normalisation, the message-only part of the controller down projection) prepared once per loop pass | `PFUSE_HOIST=1` (v2.9.2; `0` = off) | one 8K loop pass 104.5 -> 93-94 ms; cold prefill +1.1 / +1.7 / +1.4 % at 8K / 32K / 128K against v2.9.1 | changes the FP32 summation order, not the math (99.96 % of elements bit-equal; FP64-reference error within 1.5x of the fused path); quality bracket: no drop (see [quality.md](quality.md#quality-checks)) | ours |
| Sink gate in the prefill attention epilogue, coalesced: sliding-window layers (v2.8.2) and global layers, BF16 and packed paths (v2.9.1) | `ATTN_GATE=all` (`1` = sliding-window only) | attention + gate 2,759 -> 1,944 us per sliding-window visit at 8K (-32.6 ms per chunk); global layers -840 us per 8K visit (-6.7 ms per chunk), +0.4-0.5 % prefill against sliding-window only | bit-identical to SGLang's `gate_sink`, padded rows zeroed; serving 30 / 30 identical with the gate on all layers | ours |
| Sliding-window prefill prologue: K conv + L2 clip and V conv + value norm, one kernel each, in the stock kernel's row-sum order | `PROLOGUE=1` | about -8.5 ms per 8K prefill chunk | bit-identical (0 ulp at 2,048-8,192 rows; serving 30 / 30 identical, log-probabilities bit-equal) | ours |
| Prefill: relative-position bias written directly in FA4's sheared layout | `PREFILL=bias_layout` | with quant_once: 8K prefill 3,781 -> 4,253 tok/s | bit-identical (attention output and LSE) | ours |
| Prefill: MoE input quantised to NVFP4 once per token, then gathered to its 8 experts | `PREFILL=quant_once` | (with bias_layout, above) | bit-identical | ours |
| Prefill: MoE combine writes BF16 directly | `PREFILL=partial_bf16` | combine -22 to -32 %; with qkvr_split: 8K 3,408 -> 3,781 tok/s | bit-identical | ours |
| SGLang DSpark cookbook cell (FA4, page 128, 8 draft tokens, 4 `SGLANG_OPT_*` switches) | `SPEC=dspark` | base of the fallback profile | - | SGLang |
| One-pass checkpoint load: SGLang reads the safetensors file directly and large frees return to the OS before KV allocation | `LOAD_NOMMAP=1` | 121-137 s warm start-up in the v2.6 full benchmarks | weights unchanged | SGLang's loader flag; [sxuff](https://github.com/sxuff/inkling-2-mini-spark) and [0xSero](https://github.com/sybil-solutions/inference-optimisation-skill) for the load-strategy idea |
| KV pool sizing | `MEM_FRACTION=0.88` with DFlash2 and the prefill levers on (0.89 with all three off; 0.85 above `CTX=262144`), `SWA_RATIO=0.037` | v2.9: 2,371,200 global-layer KV tokens (box B; the snapshot slots take 0.312 GiB of the budget), 4.6 GiB MemAvailable at worst (v2.8.3: 2,352,000 and 5.3 GiB; 0.89 with the gate stopped at the guard) | exact | ours |

## Prefix tail

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

## Draft-tail snapshots

`SPEC=dflash2` draft-tail snapshots (`DFLASH_TAIL_SNAPSHOTS`, default 8 slots, v2.9): when a request's prefill
completes its prompt, the drafter's 2,048-position window at the prompt end is copied into a slot, keyed by the
prompt's content and cache salt. A later request that hits the same prefix in the cache gets that window copied back
into its drafter ring, instead of a ring rebuilt from only the uncached suffix and without a re-prefill. Snapshots are
restored only when content and salt match. Draft-only state: the target model's computation is untouched. 8 slots of
2,048 positions x 5 draft layers take 320 MiB, charged to the KV budget (0.312 GiB); with the prefix tail's deferral on
(the default since v2.9.4, see [Prefix tail](#prefix-tail)) a slot holds one 128-token page more, starting on a page boundary, so a
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

## Prefix cache in agent loops (prompt-end touch)

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

## Prompt copy for long documents

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
mode and thinking c8 were flat ([experiments.md](experiments.md#v293-prompt-copy-for-long-prompts-2026-11-25) has every arm); greedy outputs of speedbench full were identical
in all four arms of the second ABBA. Across speedbench, copies were forced on about 2 % of decode steps, almost all
in the long documents. With the earlier one-launch-per-piece version the decode step was 0.3 ms longer, which is why
the shipped version runs two fused kernels (the extra host launch, about 10 us, is hidden behind the GPU: -0.06 ms
in ABBA 2). `DFLASH_PCOPY=0` turns it off.

Release check on this tree, with the gate, one ABBA per box: `longctx` decode at 32K +10.1 % (box A) and +14.0 %
(box B); speedbench full took the same verify steps and gave identical greedy outputs on 73 / 73 prompts in every arm;
agent mode and thinking c8 within the off arms' spread ([experiments.md, "Release checks"](experiments.md#release-checks-this-tree-8k-gate-both-boxes)).

## Prefill levers

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
  [quality.md](quality.md#quality-checks)). One 8K loop pass 104.5 -> 93-94 ms; cold prefill on the v2.9.2 release defaults against v2.9.1 +1.3 / +1.1 /
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

## Packed global KV store (default, `KV12S=1`)

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

Results: `results/Q272R-*`, `results/Q272S-s05/`.

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
  parallelism, streaming sessions, hierarchical / external KV caches and disaggregation,
  speculation other than a linear DSpark draft or a DFLASH draft (the latter only on the static pools and with its
  draft KV ring), breakable prefill graphs, ragged verify modes other than `static`, captured verify shapes above the
  compact attention cutoff (48 requests x 8 rows), quantised global KV, and `--enable-unified-memory` without
  `INKLING_KV12S_POOL=unified` (or the reverse). `scripts/launch.sh` sets the engine flags and the overlay's variables
  together; when starting the container by hand, keep them consistent with the command above.
- Prompt-logprob requests over thousands of tokens make SGLang hold FP32 logits for every scored row (2.5 GB for
  3,073 rows in one request, on either store); request them in windows of a few hundred rows on this box.

## Image input (`MM=1`)

`MM=1` (opt-in since v2.9.8; the default `MM=0` is text only) serves images: SGLang's `--enable-multimodal` loads the 23-layer
vision tower (0.59 GiB) and the image processor. Images go in as `image_url` parts; each 40 x 40-pixel cell becomes
one prompt token at the image's native size.

Image tokens are ordinary causal positions for the language model: the vision tower only supplies their input
embeddings, and no attention mask changes (FA4's image masks, `image_token_ranges`, belong to other models). So image
prefill reads the packed store through the same fused prefill readers as text: image chunks run eagerly (SGLang
never replays prefill graphs for them), on the pool-charged BF16 prefix scratch or, when a batch references more
pages than it holds, the kernel's packed loader. No image path reaches a BF16 gather of the store
(`tools/gpu_check_kv12s.py` checks an image batch of three long-prefix requests bitwise on both readers with the
gather refused; the servers logged no gather). Up to v2.9.7, `KV12S=1` refused `--enable-multimodal` at startup;
nothing else in the store had to change.

The vision side has no limit in Inkling2's processor: each request's preprocessed patches (19.2 KB per image token,
held in host memory until the request finishes) and the vision tower's activations come on top of the KV budget.
With images at 16,320 tokens per request arriving 8 at a time next to thinking c24 and a cold 240K needle,
MemAvailable fell to 2.4 GiB at 0.88 and to 3.4 / 2.7 GiB at 0.86 with one preprocessing thread; the memory guard
stopped the server each time (`results/MM298-W1-s06/`, `-W2-`, `-W3-`). So `MM=1` adds:

- `MM_MAX_IMAGE_TOKENS=16384` (`inkling_mm`): a request whose images hold more vision tokens together gets HTTP 400
  naming the sizes, decided from the image headers before any pixel is decoded; the server keeps running. Images
  within it are processed exactly as stock (no resize). Two 12 MP photos (4032 x 3024) are 15,352 tokens.
- `MM_INFLIGHT_TOKENS=16384` (`inkling_mm`): image requests are admitted in the tokenizer, before their images are
  decoded, while the image tokens of all requests in flight stay within 16,384; a later image request waits until an
  earlier one finishes (a lone request is always admitted, text-only requests never wait). This bounds the host
  memory the patches hold to ~0.4 GB and the vision tower's batch to 16,384 tokens; with large images, image requests
  run about one at a time (small ones still run together).
- `--mm-io-worker-num 1`: one request's images are preprocessed at a time.
- `MEM_FRACTION` defaults 0.03 lower (0.85 for the default profile, ~3.3 GiB).

With the in-flight bound at 32,768 and the fraction 0.02 lower (box A, 2026-11-27), the same worst case ran 16 / 16
requests with the needle PASS and a lowest MemAvailable of 3.34 GiB in one boot; in a second boot it touched 3.22 GiB
and the guard stopped the server (`results/MM298-W4-s05/`, `results/MM298-W5-s05/`). The drops line up with image chunks reaching the vision
tower, whose activations the CUDA caching allocator keeps. The shipped defaults halve the in-flight bound (the
vision batch and the held patches with it) and lower the fraction by another 0.01 (+1.1 GiB of headroom); this
combination was not run again, so treat ~4.3 GiB as the expected lowest MemAvailable in that worst case, not a
measurement. For more headroom lower `MEM_FRACTION` further or `MM_INFLIGHT_TOKENS` / `MM_MAX_IMAGE_TOKENS`.

Measured on box B (2026-11-27, ABBA, `MM=0` / `MM=1` at the same fraction 0.88, so the only differences are the
vision tower, the multimodal embedding cache and the prefill graphs' embedding slot):

| | `MM=0` (A1 / A2) | `MM=1` (B1 / B2) |
|---|---:|---:|
| speedbench full c1, all 73 | 73.54 / 73.95 | 73.56 / 73.81 |
| speedbench full c8, per-request geomean | 24.46 / 24.20 | 24.16 / 24.31 |
| thinking c24, aggregate | 530.2 / 531.1 | 535.8 / 522.1 |
| prefill 8K / 128K, tok/s | 5,718 / 5,726, 3,859 / 3,862 | 5,750 / 5,710, 3,861 / 3,848 |
| greedy texts, speedbench full c1 | 73 / 73 identical across all four runs | |
| global-layer KV tokens | 2,299,904 / 2,317,056 | 2,287,104 / 2,298,752 |

Text is bit-identical and as fast; at the same fraction the tower and cache cost ~15K KV tokens (-0.7 %, within the
boot-to-boot spread of 17K). The lower fraction costs more: at 0.02 lower `MM=1` sized 2,231,552 / 2,212,736 tokens
against 2,305,792 for `MM=0` on box A (-3.2 / -4.0 %); at 0.03 lower expect about -5 %. Text speed with `MM=1` there:
speedbench full c1 73.26 against 73.54, c8 per-request 23.29 against 23.03 (single runs). That KV cost is why images
stay opt-in.

Images: eight greedy questions with known answers (`bench/image_check.py run`; numbers, shapes, a bar chart, a
note, a 4,950-token image, two images in one request, a text-only control) gave identical texts with `KV12S=1`
and `KV12S=0`, at 0.88 / 262,144 and at 0.60 / 524,288 (8 / 8 in every pair; `results/MM298-B1-s06/`,
`-B2-`, `-K0-`, `-R0-`, `-R1-`). Seven answers were right; the eighth correctly said the note's text was cut at the
image edge (a drawing error in that run's test image, fixed since). At the
reporter's sizing (0.60, `CTX=524288`), thinking c24, a cold 480K needle and 4 concurrent requests of ~230K tokens
with two 8,160-token images each ran with 0 failures, needle PASS, lowest MemAvailable 34.2 GiB
(`results/MM298-R2-s06/`).

Results: `results/MM298-*` (box B: `-s06`, box A: `-s05`).

## FP8 KV

More concurrent long requests can also use an FP8 KV cache. FP8 KV changes the model's numerics (lossy) and is
not part of this recipe's measured profile; Tech2Wild's vLLM recipe measured an FP8 tier (3.3M tokens, needle PASS at
~950K, its tool-calling eval 87.0 vs 88.4 with BF16 KV) and documents the memory budget for it.

## Tried and not adopted

Details in [experiments.md](experiments.md): vLLM (same decode, slower prefill,
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

## Open ideas, not pursued

This round of speed work ended with v2.9.7.

- **MXFP8 or NVFP4 copies of the target's dense projections.** This is the one sizeable lever left: the dense
  QKVR + WO projections take 8.5 of a 37.4 ms greedy single-request step. It would change the target's arithmetic
  (lossy), and the recipe keeps the original weights, so it was not built.
- **Bypassing the drafter when the prompt-copy rule fills all seven draft slots.** The offline screen forced only
  5-9 % of rows on long-copy traffic, below the 8-14 % needed to pay for the change.
- **Fusing the sampled-probability construction with a captured acceptance tail.** The estimate was 0.35-0.60 ms
  per sampled step at 8 requests, against a 0.54 ms bar, for several days of work with high correctness risk.
- **SWA KV packing.** Bit-exact, but slower on the verify window reads (see
  [experiments.md](experiments.md#swa-kv-packing-closed-experiment)).
