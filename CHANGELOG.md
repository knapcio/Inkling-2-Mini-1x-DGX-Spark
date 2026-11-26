# Changelog

## v2.9.7 (2026-11-26)

First tagged as v2.9.6.1 (same code, renamed; that tag is withdrawn). The final release of this round.

v2.9.6 with `DFLASH_CTX8=0` as the default: the MXFP8 drafter context projections become opt-in. Serving code with the
default is v2.9.5's (the projections run in BF16); nothing else changed, so v2.9.6's qualification stands.

- Why: v2.9.6 saves ~0.2 ms per single-request step (~0.5 %), but the decode table at c8 came out lower on box A in two
  separate repeats while box B stayed inside its spread. Box A, four runs each way over two ABBAs: code c8 on 210.8 /
  202.5 and 203.6 / 208.4 against off 212.1 / 212.1 and 211.8 / 209.7 tok/s (-2.6 % and -2.3 %, below the off range both
  times); code c4 -2.7 % in the first repeat, +1.5 % in the second; prose c16 -1.0 % and -1.6 %. Box B, eight runs each
  way: code c8 -1.2 % and +0.0 %, code c4 +0.9 % and +3.5 %, prose c4 +0.1 % and -1.6 %. A loss of ~2 % at c8 outweighs
  the c1 gain, so the default goes back to BF16. Long-context acceptance (paired greedy steplogs) and the quality bracket
  were unaffected; set `DFLASH_CTX8=1` for single-stream use.
- v2.9.5 and v2.9.6 both passed two-box qualifications: speedbench full 73.54 / 73.17 and 74.28 / 74.28 (box A / B),
  quality guard 30 / 30 in all four, worst MemAvailable 4.6 / 4.8 and 5.0 / 4.4 GiB over three thinking-c24 rounds with a
  cold 240K needle after each (3 / 3 PASS per run), 200K / 250K contexts 2 / 2 each (`results/Q295R-s05/`,
  `results/Q295R-s06/`, `results/Q295S-s05/`, `results/Q295S-s06/`, `results/Q296R-s05/`, `results/Q296R-s06/`,
  `results/Q296S-s05/`, `results/Q296S-s06/`).
- Agent checks on the final defaults (2026-11-26), 24 agents with ~72K-token prompts:
  - **Both boxes:** 0 failures. The second turn's first token came in 2.8 / 3.1 s against 371 / 369 s cold (box A /
    B), and the worst MemAvailable was 5.0 / 4.5 GiB.
  - **12-turn soak on box B (26 min, 288 requests):** 0 failures. Every turn after the first was served from the prefix
    cache (first token 3.1-6.6 s against 369 s cold). The median MemAvailable stayed at 4.6 GiB from turn 2 to turn 12;
    the lowest single sample was 3.8 GiB, at the start of turn 12.
  - **Results:** `results/AG2961-s05/`, `results/AG2961-s06/`, `results/SOAK2961-s06/`.
- docs/experiments.md: closed experiment, lossless packing of the sliding-window KV (bit-exact, but the packed visit
  was 0.66-0.84x of BF16 at 1 and at 16-48 requests and ~1.0x at 4-8; `results/GKSWA-s06/`).
- Results: `results/CTX296a-on1-s05/`, `results/CTX296a-off1-s05/`, `results/CTX296a-off2-s05/`, `results/CTX296a-on2-s05/`,
  `results/CTX296b-on1-s05/`, `results/CTX296b-off1-s05/`, `results/CTX296b-off2-s05/`, `results/CTX296b-on2-s05/`,
  `results/CTX296c-on1-s06/`, `results/CTX296c-off1-s06/`, `results/CTX296c-off2-s06/`, `results/CTX296c-on2-s06/`,
  `results/GKSWA-s06/`.

## v2.9.6 (2026-11-26)

v2.9.5 plus MXFP8 context projections for the DFlash2 drafter. Draft proposals only.

- `DFLASH_CTX8=1` (`inkling_dfbytes` ctx8, new default with `SPEC=dflash2`): after every verify, the drafter writes
  the committed rows into its context window. That append runs two BF16 projections: the context `fc` (9216 -> 2304)
  and the five layers' stacked K/V (2304 -> 10240), 89.7 MB per step, 0.47 ms of a c1 step in the v2.9.3 profile.
  - They now read MXFP8 copies (47.7 MB, the `mx8` format) on FlashInfer's b12x GEMM for 1-256 rows
    (`DFLASH_CTX8_MAX_M`); larger calls keep BF16.
  - Everything after the two GEMMs is the stock code, pinned by source hash.
  - Every b12x tile plan is compiled at start-up: 2.0-2.1 s on a serving boot, 46.6 s in a cold check container that
    also built FlashInfer's quantizer.
  - While the lever is on, online weight updates are refused (the copies are derived weights).
  - It is not armed when SGLang's aux CUDA graphs are requested. `DFLASH_CTX8=0` gives the BF16 path.
- GPU check (`tools/gpu_check_ctx8.py`, box B, real drafter weights, no server):
  - With its GEMM swapped back, the append writes bytes identical to stock at 1-256 rows; 257 rows take the stock
    path bit for bit. CUDA graph replay equals eager. No kernel compiles after the start-up warm-up.
  - Error against BF16: 3.6-3.8 % on the fc output, 5.1-5.5 % end to end on K and V (row cosine >= 0.998).
  - Eager fc + K/V append with ring writes: 0.447 -> 0.269 ms at 8 rows, -0.18 to -0.19 ms from 8 to 192 rows.
- Box B, ABBA (on, off, off, on: speedbench full, decode table, `longctx` lc32), then a second ABBA (decode table,
  lc32). Both ran on the v2.9.5 candidate tree with `KV12S_SETTLE=1` on both sides.
  - Single-request decode step: -0.195 ms median per request, paired over the 73 speedbench full requests (54 / 73
    faster); median per-request step 38.33 / 38.06 ms with the lever, 38.55 / 38.48 without.
  - Speedbench full: 74.02 / 74.48 with the lever, 73.62 / 73.66 without; tokens per step 2.849 against 2.842.
  - Decode table, mean of four arms against four (code / prose at c1, c4, c8, c16): code +1.0 / +0.9 / -1.2 / +2.6 %,
    prose +0.3 / +0.1 / -0.4 / +0.6 %; every cell inside the off arms' range.
  - lc32, mean of four against four: tokens per step 3.037 vs 3.138 at 32K and 3.260 vs 3.323 at 128K; decode tok/s
    67.8 vs 68.8 and 54.6 vs 55.1. All inside the off range (per-arm spread about ±5 %). lc32 draws new documents in
    every run; the fixed-document runs below pair the same tokens.
  - Greedy outputs: 30 / 30 speedbench quick prompts identical. On speedbench full, 68 / 73 are identical; the same 5
    differ in both on arms (code-06, longctx-01, -02, -07, -08), while off against off and on against on are 73 / 73.
    That is the verify-schedule floating-point class: a different acceptance moves the verify blocks.
- Box A, long context with fixed documents (speedbench `longctx`, 4 documents at 32K and 4 at 128K, each cold and
  cached, 1,024 tokens, greedy, steplog capture, on / off / off / on, `tools/ctx8_lcpair.py`):
  - Repeat runs are bit-identical (off against off and on against on).
  - Tokens per step over the tokens both arms generated identically, paired per request (8 / 8 per length): 32K
    4.0525 against 4.0525 (1.000), 128K 3.4482 against 3.4450 (1.001).
  - Drafted tokens agree on 97.5 % (32K) and 98.2 % (128K) of first slots.
  - Whole requests: 3.669 against 3.518 tokens per step at 32K and 3.561 against 3.379 at 128K; the texts differ after
    the first differing token, so these do not pair.
  - Greedy texts: 6 of 8 requests per length differ within 1,024 tokens, deterministically, in the same
    verify-schedule class.
- Box A, steplog pair on speedbench quick (greedy, full-width verify, `tools/ctx8_agree.py`): drafted tokens agree on
  97.1 % of first slots and 70.6 % of whole blocks. Accepted drafts per matched step: 2.137 against 2.136.
- Quality (box B, inkq2 bracket A L A L A, lever off / on): PASS.
  - Totals 557 / 558 / 561 without the lever and 557 / 557 with it, of 575.
  - gsm8k 388 / 390 / 392 against 391 / 390; long-context retrieval 100 / 100 in every arm; qeval 69 / 68 / 69 against
    66 / 67.
  - Flips 7-10 against each baseline (budget 18), McNemar p >= 0.29.
  - Both lever runs sit at the low end of the baselines' range, and qeval is 1-3 below them, inside the bracket's
    tolerance (protocol v2, one-sided).
- Memory (box B, lever on): lowest MemAvailable 4.2 GiB under thinking c24 with a cold 240K needle prefill; kvstress
  48 / 48 (334.4 tok/s); needles 128K and 240K pass.
- `inkling_kv12s`: with `INKLING_KV12S_DFLASH_SETTLE` unset, the overlay now settles with shared page candidates
  (`shared`), the launcher's default; before, it fell back to `1`.
- Results:
  - `results/CTX296on1-s06/`, `results/CTX296off1-s06/`, `results/CTX296off2-s06/`, `results/CTX296on2-s06/`;
  - `results/CTX296r-on3-s06/`, `results/CTX296r-off3-s06/`, `results/CTX296r-off4-s06/`, `results/CTX296r-on4-s06/`;
  - `results/LCS296on1-s05/`, `results/LCS296off1-s05/`, `results/LCS296off2-s05/`, `results/LCS296on2-s05/`;
  - `results/SL296on-s05/`, `results/SL296off-s05/`;
  - `results/QB296A1-s06/`, `results/QB296L1-s06/`, `results/QB296A2-s06/`, `results/QB296L2-s06/`,
    `results/QB296A3-s06/`;
  - `results/MEM296-s06/`.

## v2.9.5 (2026-11-26)

v2.9.4 plus a cheaper settle of the packed global KV store after each DFlash2 verify step. Exact.

- `KV12S_SETTLE=shared` (`INKLING_KV12S_DFLASH_SETTLE`, new default): the DFlash2 hook settled each of the 8 packed
  global visits on its own and rebuilt the same page candidates for every visit (11 small launches each). They are now
  built once for all visits, as the DSpark path does since v2.7.3: 77 fewer launches per step. The trim and commit
  kernels still run per visit, unchanged. `0` is the v2.9.4 settle.
- `KV12S_SETTLE=1` (opt-in): also runs the trim and commit kernels once over all visits (`device.LaneGroup`; 14
  launches per step instead of 112). Exact as well, but slower at 4 and 8 requests in flight in the measurements below,
  so it is not the default.
- Exactness, every mode: the same K/V bits and attention outputs. The store's bytes are identical as well, except where
  one visit's side-slot claims race near overflow-arena exhaustion, which the per-visit path also resolves in
  scheduling order (every outcome exact). GPU check (`tools/gpu_check_settle_lanes.py`, box B): v2.9.4's own kv12s
  code against per-visit, shared and batched, 1 / 4 / 8 / 24 / 48 requests, verify writes and attention replayed from
  CUDA graphs, every third step ragged; every byte of the store and every attention output identical, attention equal
  to the stock BF16 kernel, every accepted row equal to the written K/V. With page-heads spilling and both side arenas
  exhausted (4 requests): attention and accepted rows identical, no row lost.
- Eager settle of 8 visits with every request sealing a page (the check's microbenchmark), per visit / shared /
  batched: 645 / 322 / 115 us at 1 request, 839 / 541 / 450 us at 8, 1,532 / 1,222 / 1,172 us at 24, 2,557 / 2,247 /
  2,230 us at 48.
- Shared, box A, ABBA (shared, off, off, shared; warm and the decode table): greedy outputs and verify-step counts
  identical on 30 / 30 speedbench quick prompts. Single-request decode step 37.62 -> 37.37 ms (median per-request
  `step_ms` over the 30 prompts, mean of the two arms each side; -0.25 ms, -0.7 %; 29 of 30 prompts faster). Decode
  table: every cell at or above the off arms' range; c1 +0.5 % prose, +0.6 % code; geometric mean of the eight cells
  +1.3 % (off-vs-off spread up to 2.9 % per cell).
- Batched (`1`), box B, ABBA (on, off, off, on; speedbench full and the decode table): greedy outputs and verify-step
  counts identical on 73 / 73 speedbench full and 30 / 30 quick prompts; single-request decode step 38.80 -> 38.43 ms
  (-0.31 ms, -0.8 %; 65 of 73 faster); speedbench full 72.96 / 73.25 off, 73.89 / 73.26 on. But the decode table fell
  at 4 requests in flight, code -5.1 % (off spread 1.8 %). A repeat on box A (on, off, off, on) confirmed it: code c4
  -4.1 %, prose c4 -3.0 %, prose c8 -1.4 %, code c8 -0.9 %, against off spreads of 0.1-0.5 %; geometric mean -1.0 %.
  The engine log shows the steps themselves were not slower (box A, 4 requests: about 73 ms on vs 78 ms off). Fewer
  tokens were accepted per step (2.78 vs 3.07) under narrower verify widths, so the batched arms decoded different
  greedy texts with several requests in flight. With several requests in flight the greedy texts differ between runs of
  the same setting too (for example prose at c16, box B, off vs off), so this alone does not show a store difference;
  the shared arms, which run the per-visit kernels, lost nothing in the decode table.
  Greedy dumps at 4 and 8 requests in flight (box A, `results/KV295lc-*-s05/`) differ between any two arms about as
  much as between two off arms or two rounds of one arm, so they show no store difference either; `1` stays opt-in for
  the decode-table loss.
- Memory, box A (warm, kvstress, 128K needle; booted with `1`, which allocates the same as `shared` plus a 768-byte
  table): lowest MemAvailable 4.9 GiB (v2.9.3 on the same box and recipe: 4.6 GiB); kvstress 48 of 48 thinking
  requests at 24 in flight together with the cold 240K needle, 337.3 tok/s aggregate (v2.9.3: 325.8); 240K and 128K
  needles pass.
- Results: `results/KV295sh1-s05/`, `results/KV295sh2-s05/`, `results/KV295dt-off3-s05/`, `results/KV295dt-off4-s05/`
  (shared); `results/KV295on1-s06/`, `results/KV295off1-s06/`, `results/KV295off2-s06/`, `results/KV295on2-s06/`,
  `results/KV295dt-on1-s05/`, `results/KV295dt-off1-s05/`, `results/KV295dt-off2-s05/`, `results/KV295dt-on2-s05/`
  (batched); `results/KV295mem-s05/`.

## v2.9.4 (2026-11-25)

v2.9.3 plus the prefix tail deferring to draft-tail snapshots, on by default with `SPEC=dflash2`. v2.9.3 (the base of v2.9.4) passed a two-box qualification: Q293R and Q293S on both boxes; v2.9.4's prefix-tail deferral was release-checked on both boxes.
The gain applies to repeated cached prompts of 1K-4K tokens; agent mode (histories above 4K tokens) is unaffected.

- `DFLASH_TAIL_DEFER=1` (`inkling_dftail`, new default): before the prefix tail bounds a cached prompt of at most
  4,096 tokens, the draft-tail snapshot slots are checked for one of the same prefix (tokens, cache salt, extra key)
  that ends past the tail's bound and covers the drafter's whole window. If there is one, the request keeps its
  ordinary prefix-cache hit, capped at the snapshot's end, and the window is restored from the slot instead of the
  target re-prefilling the last 2,048 tokens; otherwise the tail applies as before. The slot is reserved from the
  lookup to the restore (slot version, content digest, end, salt / extra key, snapshot generation) and checked after
  the lookup and again before the restore; a changed slot is never restored. To cover exactly the positions the tail
  would recompute, a slot now holds one extra 128-token page starting on a page boundary (8 slots: 340 MiB, 0.332 GiB
  of KV budget, was 320 MiB). Not for logprob requests, more than one rank, or hierarchical / external KV storage.
  `DFLASH_TAIL_DEFER=0` runs v2.9.3's code path exactly. The launcher refuses `INKLING_DFLASH_TAIL_DEFER` in
  `SGLANG_ENV` / `SGLANG_ARGS`.
- Box B, two runs each way (on, off, off, on): speedbench full long context +2.1 % (75.01 / 74.94 against 73.39 /
  73.52 tok/s; tokens per verify step 2.97 -> 3.08; median time to first token on these hits 0.45 -> 0.19 s), all
  prompts +0.4 %, other categories flat; single-request step 39.05 / 39.17 against 39.62 / 39.58 ms. Speedbench agent
  mode is unaffected (every history is above 4,096 tokens). Decode at 32K / 128K flat over four runs each way (32K: on
  67.4-73.1, off 63.4-70.2 tok/s). Greedy outputs differ from the off runs on 7 / 73 speedbench and 3 / 30 warm
  prompts, identically in both on runs (cached prefix in place of recomputed prefill). Quality bracket (box A, inkq2,
  off / on / off / on / off): totals 559 / 559 / 557 / 561 / 557 of 575, flips within budget, McNemar p >= 0.219; no
  drop.
- Agent loops (box A, 24 x ~72K contexts, two turns, two on / off pairs): 24 / 24 turn-2 hits in all four runs (time
  to first token 2.88-2.95 s). Only the short first-phase turns (~1.1K-token cached prompts) defer: their turn-2 time to
  first token was 4.84 / 4.81 s against 5.21 / 5.23 s. Worst MemAvailable follows the KV pool size each boot picks
  (pairs: on 3.9 / 5.1 GiB, off 5.1 / 4.6 GiB); the drop from the post-start plateau to the worst point was 1.1 / 0.5
  GiB on and 0.8 / 0.6 GiB off.
- Memory (box B, thinking c24 with a cold 240K needle): worst MemAvailable 4.1 GiB, 48 / 48 completed, needles 128K and
  240K PASS.
- Docs: the eviction-order cause of the agent-loop prefix-cache misses that the prompt-end touch works around is
  reported upstream as [sgl-project/sglang#53149](https://github.com/sgl-project/sglang/issues/53149). A larger
  sliding-window pool does not prevent the misses: with `SWA_RATIO=0.074` (twice the default) and without the touch,
  24 x ~72K agents at `--prefill-decode-interval 2` still missed on turn 2 for 8 of 24 (box A) and 9 of 24 (box B),
  about as many as with the default pool; the default pool stays.
- Results: `results/AB294on1-s06/`, `results/AB294off1-s06/`, `results/AB294off2-s06/`, `results/AB294on2-s06/`
  (speedbench full, long context, agent mode); `results/LC294on3-s06/`, `results/LC294off3-s06/`,
  `results/LC294off4-s06/`, `results/LC294on4-s06/` (32K / 128K repeats); `results/KV294-s06/` (memory);
  `results/AG294on-s05/`, `results/AG294off-s05/`, `results/AG294off2-s05/`, `results/AG294on2-s05/` (agent loops); `results/QB294A1-s05/`, `results/QB294L1-s05/`,
  `results/QB294A2-s05/`, `results/QB294L2-s05/`, `results/QB294A3-s05/` (quality bracket);
  `results/SWA1i2-s05/`, `results/SWA1i2-s06/` (window-pool contrast); `results/KVS293-s05/`; v2.9.3 qualification:
  `results/Q293R-s05/`, `results/Q293R-s06/`, `results/Q293S-s05/`, `results/Q293S-s06/`.

## v2.9.3 (2026-11-25)

v2.9.2 plus the prompt copy for long prompts, on by default with `SPEC=dflash2`. v2.9.2 passed a two-box qualification
(Q292R and Q292S on both boxes); the prompt copy was release-checked on both boxes.

- `DFLASH_PCOPY=12` (`inkling_pcopy`, new default): each decode step, the most recent earlier occurrence of the last
  12 committed tokens in the request's own context is looked up; at a match of 16 tokens or more, the tokens that
  followed it replace the drafter's proposals for the rest of the draft block. Draft proposals only: the target
  verifies every token; sampled requests give each copied slot a point mass (exact acceptance law, GPU Monte Carlo
  test), and the verify-width cap keeps the copied rows live. Two fused kernels per step (a scan split over
  2,048-token chunks with an atomic maximum, then the per-row copy), bit-identical to the reference step on random and
  directed cases; no-hit cost 12 us GPU per step at 128K context.
- `DFLASH_PCOPY_MIN_CTX=8192`: the copy runs only for prompts of at least 8,192 tokens. Without it one speedbench long
  context prompt (2.7K tokens) needed 67 verify steps instead of 61; every speedbench prompt is under 5K tokens, so
  speedbench now runs exactly as without the copy.
- Box A, two ABBA runs, speedbench `longctx`: decode at 32K 64.2 -> 70.5 tok/s (+9.8 %, tokens per step 2.98 ->
  3.30), at 128K 53.9 -> 55.2 (+2.4 %); within each ABBA every run with the copy was faster at 32K. Speedbench full,
  agent mode and thinking c8 flat; single-request decode step 39.49 ms off vs 39.43 ms on (median per-request `step_ms` of the 73
  measured requests, second ABBA); greedy speedbench outputs identical in all four arms. Copies were forced on about 2 % of speedbench
  decode steps.
- Release checks on this tree with the 8K gate, an ABBA on each box (copy off / on: speedbench full, `longctx`, agent
  mode, thinking c8). `longctx` decode at 32K: box A 62.8 -> 69.1 tok/s (+10.1 %; cold +9.2 %, cached +11.0 %), box B
  63.4 -> 72.2 (+14.0 %; cold +16.2 %, cached +11.9 %); at 128K +1.5 % / +4.6 % (cells of 3 requests, noisy: box A
  cached -3.4 %, box B cold -4.2 %). The gate check: speedbench full step counts identical to the off arm on 73 / 73
  requests in every arm on both boxes, greedy outputs identical on 73 / 73 speedbench and 30 / 30 warm prompts.
  Single-request step (median per-request `step_ms`, 73 measured requests): box A 39.34 ms off vs 39.51 on, box B 39.54 vs 39.47; speedbench
  full and thinking c8 within the off arms' spread. Agent mode: on the 8-16K-token turns the gate admits, tokens per
  step with the copy stayed inside the off arms' range on both boxes; end-to-end +0.3 / +1.4 % (box A, 160 / 512-token
  turns) and +1.1 / -0.6 % (box B), against an off-arm spread of up to 2.4 %. Memory (box B, thinking c24 + cold 240K
  needle): worst MemAvailable 4.9 GiB, 48 / 48 at c24, needles 128K / 240K PASS. Agent loops (24 x 72K, two turns):
  box A and box B 24 / 24 turn-2 hits (time to first token 3.0 / 2.9 s).
- Results: `results/PCa1-s05/`, `results/PCa2-s05/`, `results/PCb1r-s05/`, `results/PCb2-s05/` (first ABBA),
  `results/PCrA1-s05/`, `results/PCrA2-s05/`, `results/PCrB1-s05/`, `results/PCrB2-s05/` (second ABBA); release
  checks: `results/V293A1-s05/`, `results/V293A2-s05/`, `results/V293B1-s05/`, `results/V293B2-s05/`,
  `results/V293A1-s06/`, `results/V293A2-s06/`, `results/V293B1-s06/`, `results/V293B2-s06/`, `results/KVS293-s06/`,
  `results/AG293-s05/`, `results/AG293-s06/`; gate screen: `results/V293C1-s05/`, `results/V293C2-s05/`,
  `results/V293A3-s05/`, `results/V293A4-s05/`; v2.9.2 qualification: `results/Q292R-s05/`, `results/Q292R-s06/`,
  `results/Q292S-s05/`, `results/Q292S-s06/`; prompt-end touch repeats and window-pool contrasts (default interval):
  `results/PT1-s06/`, `results/PT2-s06/`, `results/SWA1-s06/`, `results/SWA5-s06/`.

## v2.9.2 (2026-11-25)

v2.9.1 plus the prefill loop-bank hoist and the prompt-end cache touch, both on by default. Quality bracket on one box
(`inkq`); memory, needles and agent loops on box B.

- `PFUSE_HOIST=1` (`inkling_phoist`, new default): the loop controller's linear source projection is prepared once per
  loop pass instead of inside every fused visit. It changes the FP32 summation order, not the math (99.96 % of
  elements bit-equal to the fused path, FP64-reference error within 1.5x), so greedy outputs can move on long
  prompts like any reduction-order change. Cold prefill against v2.9.1 (box B, mean of 2 runs): 1K 2,857 vs 2,821
  (+1.3 %), 8K 5,657 vs 5,595 (+1.1 %), 32K 5,325 vs 5,236 (+1.7 %), 128K 3,875 vs 3,824 (+1.4 %).
- Quality, the gate for summation-order levers ("fine if model quality does not drop"): `inkq` v2, 575 graded items
  with prompts >= 1.5K tokens, five interleaved runs on one box: v2.9.1 556 / 559 / 555, the hoist 562 / 559. No
  drop on any family; one hoist run scored above the baseline spread; over the six hoist-baseline pairs 38 items
  were right only with the hoist and 15 only with the baseline.
- `PROMPT_TOUCH` (`inkling_ptouch`, new default): after a request finishes, its prompt path is touched with SGLang's
  `touch_prefix`, so the sliding-window state behind a finished prompt end is no longer the first entry evicted. In
  long-context agent loops this removes intermittent whole-context cache misses on the next turn (24 x 72K, interval
  2: stock 3 / 3 runs with 9-10 misses, turn-2 time to first token 163-186 s, wall ~370-380 s; with the touch 2 / 2
  runs 24 / 24 hits, 3.0 s, ~210 s). Eviction order only: nothing the model computes changes. Combined-tree release
  check: AG292 24 / 24 hits at the default interval (3.0 s, turn-2 wall 211 s); KVS292c worst MemAvailable 4.8 GiB,
  48 / 48 at c24, needles 128K / 240K PASS; REL292 speedbench full 73.19 (hoist-only tree 73.44) with greedy outputs
  identical to the hoist-only tree on 73 / 73 and 30 / 30 warm.
- README: `--prefill-decode-interval 1` documented as an opt-in for interactive streaming (earlier first tokens,
  later completion; not for agent loops); draft-tail snapshot slots beyond 8 concurrent conversations.
- Memory with the hoist: worst MemAvailable 4.0 / 5.7 GiB in two runs (v2.9.1: 4.8), needles at 128K and 240K
  PASS; `MEM_FRACTION` stays 0.88. Sanity 11 / 11; speedbench full 73.44 (v2.9.1 on the other box 73.19 / 73.24).
- The `--enable-mixed-chunk` refusal now states the real reasons (it moves chunk boundaries and drops speculation for
  running requests), and the prologue's docstring says bit-identical.
- Results: `results/Q292h-s06/`, `results/PF292-s06/`, `results/KVS292-s06/`, `results/KVS292b-s06/`, `results/QE291a-s05/`, `results/QE292h-s05/`,
  `results/QE291b-s05/`, `results/QA1-s05/`, `results/QA2-s05/`, `results/QA3-s05/`, `results/QL1-s05/`,
  `results/QL2-s05/`, `results/AG292-s06/`, `results/KVS292c-s06/`, `results/REL292-s06/`; agent-loop A/B:
  `results/AG64-s06/`, `results/AGD-s06/`, `results/AGD2-s06/`, `results/AGD3-s06/`, `results/AG64b-s05/`,
  `results/DYI1-s06/`, `results/DYI2-s06/`, `results/STd-s06/`, `results/STd2-s06/`, `results/PTd-s06/`, `results/PTd2-s06/`.

## v2.9.1 (2026-11-25)

Basic tests on one box; v2.8.4 (the base of v2.9) passed a two-box qualification: Q284R and Q284S on both boxes.

The sink gate now also runs on the global layers in prefill; decode is unchanged. Box B, release defaults: cold
prefill 8K / 32K / 128K +0.8 / +0.4 / +1.0 % against v2.8.3; speedbench full 73.12 (v2.9 on the same box 73.34);
worst case 4.8 GiB. Results (box B): `results/PF291-s06/`, `results/KVS291-s06/`, `results/XG291r-s06/`,
`results/XG291g-s06/`, `results/KVSGa-s06/`, `results/KVSG1-s06/`, `results/PFGa-s06/`, `results/PFG1-s06/`,
`results/XGGa-s06/`, `results/XGG1-s06/`.

- `ATTN_GATE=all` (`inkling_attn`, the new default): the sink gate (`gate_sink`), which v2.8.2 computes in the
  epilogue of the sliding-window prefill attention kernel, now also runs on the global layers, on both the BF16 and the
  packed 12-bit KV paths. Bit-identical: in the GPU check, y and LSE equal stock `gate_sink` on every path, including
  captured pairs and replays; in serving, the log-probability dump with the gate on all layers matches the
  sliding-window-only gate on 30 / 30 records, and greedy outputs equal v2.9's on 30 / 30 prompts. `ATTN_GATE=1`
  keeps the v2.9 behaviour (sliding-window only), `0` turns it off.
- Kernel: -840 us per 8K global-layer visit, -6.7 ms per 8K chunk. Same-tree A/B, all layers against
  sliding-window only: +0.4 / +0.4 / +0.5 % at 8K / 32K / 128K.
- Cold prefill, release defaults (box B, mean of 2 runs) against v2.8.3: 1K 2,821 vs 2,864 (about flat), 8K 5,595
  vs 5,553 (+0.8 %), 32K 5,236 vs 5,217 (+0.4 %), 128K 3,824 vs 3,788 (+1.0 %).
- Speedbench full 73.12 against v2.9's 73.34 on the same box: decode unchanged within the run-to-run spread.
- Memory: worst MemAvailable 4.8 GiB, needles at 128K and 240K PASS; gate on all layers against sliding-window only
  5.2 / 5.4 GiB; `MEM_FRACTION` stays 0.88.

## v2.9 (2026-11-25)

Basic tests on one box; v2.8.4 (the base of v2.9) passed a two-box qualification: Q284R and Q284S on both boxes.

Draft-tail snapshots for agent loops; single-request decode is unchanged. Box B, release defaults: speedbench full
73.34 against v2.8.4's 73.43 on the same box (flat); agent loops -4.4 / -4.9 % end-to-end time at 160- / 512-token
turns (screen); KV pool 2,371,200 tokens; worst case 4.6 GiB. Results (box B): `results/V29R-s06/`,
`results/V29S-s06/`, `results/S81off-s06/`, `results/S81a-s06/`, `results/S81b-s06/`, `results/S81ab-s06/`,
`results/K81b8-s06/`, `results/K81b16-s06/`, `results/BEXoffref-s06/`, `results/BEXoffb-s06/`,
`results/BEXonref-s06/`, `results/BEXonb-s06/`, `results/BEXschref4-s06/`; v2.8.4 qualification:
`results/Q284R-s06/`, `results/Q284R-s05/`, `results/Q284S-s06/`, `results/Q284S-s05/`.

- Draft-tail snapshots (`inkling_dftail`, `DFLASH_TAIL_SNAPSHOTS=8`, on by default with `SPEC=dflash2`): when a
  prefill completes its prompt, the drafter's 2,048-position window at the prompt end is kept in one of 8 slots, keyed
  by content and cache salt. A later request that hits the same prefix in the cache gets it copied back into its draft
  ring instead of a ring rebuilt from only the uncached suffix. Restored only when content and salt match. Draft-only
  state; the target path is untouched. 8 slots of 2,048 positions x 5 draft layers take 320 MiB, charged to the KV
  budget (0.312 GiB). Any weight update or `flush_cache` clears them. The slots are one shared LRU, so in a multi-tenant
  server one tenant's requests can evict another's snapshots (visible only as draft acceptance and latency, like the
  radix cache's own eviction); use `DFLASH_TAIL_SNAPSHOTS=0` where tenants must not observe each other's cache use.
- Screen on box B (v2.8.1 base), against both levers off (A = the prefix tail shipped since v2.8.1, B = snapshots):
  speedbench full long context / agentic / code: A +6.6 / +1.1 / +1.7 %, B +14.4 / +0.2 / +0.1 %, A + B +6.7 / +1.3 /
  +1.7 %. Agent loops (speedbench agent mode, 5-16K-token histories, end-to-end time at equal output length) at 160- /
  512-token turns: A -0.7 / +0.3 %, B -4.0 / -5.4 %, A + B -4.4 / -4.9 %; decode 73.9 -> 78.3 tok/s with A + B; median
  time to first token unchanged at 0.242-0.247 s. The release ships A + B.
- Release run (`V29R`, box B): speedbench full 73.34 (v2.8.4 on the same box 73.43, flat); speedbench agent mode 76.33
  at 160-token turns and 78.86 at 512-token turns (geometric mean tok/s); greedy outputs equal v2.8.4's on 30 / 30
  prompts.
- Exactness: with the verify cap off, the snapshots put no token outside the reference's top-k (27 / 30 quick prompts
  identical; the long prompts and 48 agent turns show near-ties only); with the cap on, the same. A control without
  snapshots that only fixes the verify width at 4 (same drafts) diverges on the same three prompts, with a larger shift
  (1.65 nat on longctx-02) than any the snapshots produced: the snapshots' differences are in the verify-schedule
  class.
- Memory: 8 slots, worst MemAvailable 4.6 GiB; 16 slots 4.0 GiB with no speed gain, so 8 is the default. Release
  defaults (`V29S`): worst 4.6 GiB, needles at 128K and 240K PASS, the worst-case load completed 48 / 48 requests.
- Inert (logged once, never failing a request) on PD disaggregation, DCP > 1, a quantized or non-NHD draft pool, an
  uncharged pool, or any hook exception. Pinned to the image's source (exit 78 on drift; `DFLASH_PREFIX_TAIL=0
  DFLASH_TAIL_SNAPSHOTS=0` runs without the tail and snapshots).

## v2.8.4 (2026-11-25)

Basic tests on one box; full two-box qualification follows in v2.8.x.

The drafter's MXFP8 linears run FlashInfer's MXFP8 GEMM where the shapes allow. Speedbench full +1.2 % in an ABBA on
box A; greedy outputs identical. Release run (box A): speedbench full 73.53; the same box with `mx8w`
off 73.13 / 72.73. Results (box A): `results/MXa0-s05/`,
`results/MXa1-s05/`, `results/MXb1-s05/`, `results/MXb0-s05/`, `results/MXkv-s05/`, `results/REL284-s05/`.

- `DFLASH_BYTES` default `mx8,hot8` -> `mx8,mx8w,hot8` (`inkling_dfbytes`). With `mx8w`, the 20 of the drafter's 30
  linears whose K and N are multiples of 128 run FlashInfer's stock b12x MXFP8 GEMM (`mm_mxfp8` with
  `mxfp8_quantize` for the activations) for 1-256 rows; the other 10 (the 576-wide conv projections) keep `mx8`. It
  reuses `mx8`'s FP8 bytes, with the E8M0 scales swizzled once (+11.2 MB, no second FP8 copy). Draft proposals only:
  under the lossless verification greedy output is unchanged (30 / 30 identical in every arm, no near-ties).
- Microbench (the drafter's 30 layer linears, cold L2, one graph): 1.73-1.76x over BF16 cuBLAS (8 rows 3.33 ->
  1.90 ms, 192 rows 4.05 -> 2.34 ms), 0.33-0.36 ms faster than `mx8`'s GEMV at 8 / 16 rows.
- Serving ABBA, box A (off / on / on / off): speedbench full 73.75 / 73.81 on against 73.13 / 72.73 off (+1.2 % on the
  means); median step 39.21 / 39.22 against 39.57 / 39.91 ms. Categories, on against off: long context 74.46 vs 72.96
  (+2.1 %), code 82.73 vs 81.58 (+1.4 %), prose 65.97 vs 65.14 (+1.3 %), multilingual 58.24 vs 57.70 (+0.9 %),
  agentic 83.97 vs 83.32 (+0.8 %), structured 78.03 vs 77.57 (+0.6 %). Per-stream decode at c1: prose 66.7 / 66.9
  against 65.9 / 65.7 (+1.4 %), code 79.4 / 79.5 against 78.7 / 78.8 (+0.9 %); c4-c16 mostly up but noisy (code c8
  -1.2 % to prose c4 +3.8 %); thinking per-request median c8 +1.8 %, c24 +2.4 %.
- Memory (box A, release defaults with `mx8w`): lowest MemAvailable under the worst-case load 4.8 GiB (thinking c24
  aggregate 345.8 tok/s; guard 3.5 GiB); needles at 128K and 240K PASS; KV pool unchanged (about 2.36M tokens on box
  A).
- Release run (box A, `results/REL284-s05/`, one run, 0 failed): speedbench full 73.53; prose 65.75, code 82.46,
  structured 77.73, agentic 83.69, multilingual 58.01, long context 74.20; median time to first token 164.8 ms,
  median step 39.27 ms, 2.833 tokens per step. Same-box reference with `mx8w` off: 73.13 / 72.73 (v2.8.3's 72.90 was
  on box B, not comparable as a gain). Per-stream decode, prose c1 / c4 / c8 / c16 66.7 / 37.0 / 27.9 / 20.4, code
  79.1 / 38.6 / 30.0 / 22.0. Greedy outputs identical to the `mx8w`-off run on 30 / 30 prompts; lowest MemAvailable
  4.9 GiB.

## v2.8.3 (2026-11-25)

Basic tests on one box; full two-box qualification follows in v2.8.x.

Padding rows of CUDA-graph batches get no expert work, and the sliding-window prefill prologue is now bit-identical
and on by default. Release defaults: thinking c9 / c15 +4.8 / +5.0 % against padx off (box A); cold prefill 1K / 8K /
32K / 128K +1.5 / +0.5 / +0.6 / +0.1 % against v2.8.2 (box B); KV pool 2,352,000 tokens; worst case 5.3 GiB; speedbench full 72.90
(unchanged).
Results: box A `results/PADX283g-s05/`, `results/PADX283off2-s05/` (release version), `results/PADX283-s05/`,
`results/PADX283off-s05/` (first version), screen `results/PXa0-s05/`, `results/PXa1-s05/`, `results/PXb1-s05/`,
`results/PXb0-s05/`; box B `results/KVS283a-s06/`, `results/KVS283b-s06/`, `results/PF283-s06/`,
`results/REL283-s06/`.

- `inkling_padx` (new, `PADX=1`): a decode or verify CUDA graph replays at its captured batch size, and with fewer
  live requests its padding rows still carried the previous replay's inputs and routed to 8 experts each. They now
  route to the phantom expert that the dead-verify-row skip already uses, so the expert GEMMs skip them. The choice is
  made per captured graph: graphs of 1-8 requests can never replay with padding (n live requests use the smallest
  captured size >= n, and 1-8 are all captured), so they keep the stock route with no extra kernels; graphs of 10
  requests and up get the mask. A first version masked every graph and added 4 small kernels per MoE visit (192 per
  step, +0.36 ms or 0.8 % at c1, where nothing pads). Live rows are bit-identical in every shape, small graphs are
  bit-identical to stock on every row, and the c1 greedy outputs equal v2.8.2's release run on 30 / 30 prompts.
  - Release version against padx off, box A, same session: speedbench full 73.17 vs 73.25 (flat; median step 39.59
    vs 39.46 ms, two padx-off runs differ by 0.09 ms; 2.742 tokens per step in both); thinking c9 282.2 vs 269.3
    aggregate (+4.8 %, per-request median 35.6 vs 34.3), c15 392.5 vs 373.9 (+5.0 %, per request 29.6 vs 27.9,
    +6.1 %). Needle at 128K PASS.
  - First version (graphs of 16 and 24 requests are masked in both): thinking c16 415.5 vs 403.4 (+3.0 %), worst-case
    load (thinking c24) 331.6 vs 324.6 (+2.2 %). Screen, two runs each: c9 +8 %, c15 +8 %, c16 +2.4 %, c10 flat.
  - Concurrent greedy output depends on the batch composition on this stack (two runs with padx off differ on 57 / 60
    outputs); padx on against off shows the same first-divergence class. `PADX=0` turns it off.
- `PROLOGUE=1` (`inkling_prologue`) is now on: the fused K conv + L2 clip and V conv + value norm reduce in the stock
  kernel's row-sum order, so the v2.8.2 one-ulp differences are gone. 0 ulp at 2,048 / 2,049 / 3,001 / 8,192 rows,
  including adversarial rows that are sensitive to the summation order; in serving, greedy outputs identical to both
  controls on 30 / 30 prompts, log-probabilities bit-equal over 4,691 tokens. About -8.5 ms per 8K prefill chunk.
  Cold prefill, release defaults, box B, mean of 2 runs, against v2.8.2: 1K 2,864 vs 2,821 (+1.5 %), 8K 5,553 vs
  5,527 (+0.5 %), 32K 5,217 vs 5,189 (+0.6 %), 128K 3,788 vs 3,785 (+0.1 %); against the v2.8.1-tree control +2.7 /
  +2.8 / +1.4 % at 8K / 32K / 128K.
- The prefill hoist (`PFUSE_HOIST`) stays opt-in.
- Memory and needles, release defaults (box B, `MEM_FRACTION=0.88`): KV pool 2,352,000 tokens; lowest MemAvailable
  under the worst-case load 5.3 GiB over four runs (5.6 / 5.5 / 5.4 / 5.3 GiB; 48 / 48 requests completed, 340.1 /
  357.4 tok/s aggregate; guard 3.5 GiB); needles at 128K and 240K PASS; c1 greedy outputs equal v2.8.2's release run
  on 30 / 30 prompts in all four runs.
- Speedbench full, release defaults (box B, `results/REL283-s06/`, one run, 0 failed): all 72.90 (v2.8.2 72.78 /
  72.82): prose 65.02, code 81.56, structured 77.63, agentic 83.28, multilingual 57.65, long context 72.99; median
  time to first token 163.8 ms, median step 39.71 ms, 2.742 tokens per step. c1 decode is unchanged within the
  run-to-run spread; the gains of this release are at concurrency (padding rows) and in prefill (the prologue).

## v2.8.2 (2026-11-24)

Basic tests on one box; full two-box qualification follows in v2.8.x.

A bit-identical prefill lever on by default, two more opt-in, and a refitted DFLASH verify-width policy: a prefill
and verify-policy release, with decode speed on speedbench unchanged. Box B, release defaults: speedbench full 72.78
/ 72.82 in two runs (v2.8.1 72.86); cold prefill 8K / 32K / 128K +2.1 / +2.2 / +1.0 %; KV pool 2,349,056 tokens at
`MEM_FRACTION=0.88`; worst case 5.0 GiB. Results: `results/PF282-s06/`, `results/REL282-s06/`,
`results/KVS282a-s06/`, `results/KVS282b-s06/`, `results/KVS282c-s06/`, `results/SBF0-s06/`, `results/SBF1-s06/`,
`results/PHAc-s06/`, `results/PHAc2-s06/` (v2.8.1-tree prefill control),
`results/XGc-s06/`, `results/XGc2-s06/`, `results/XGg-s06/`,
`results/XGgp-s06/`, `results/XGag-s06/`, `results/XGn-s06/`, `results/CAP282-s06/`.

- Prefill levers, each pinned to the image's source:
  - `ATTN_GATE=1` (`inkling_attn`, on): the sink gate in the sliding-window prefill attention epilogue, coalesced.
    Bit-identical to SGLang's `gate_sink`, padded rows zeroed; in serving, greedy outputs identical to the control on
    30 / 30 prompts with a maximum log-probability difference of 0. The gated epilogue turns itself off while
    `SGLANG_KERNEL_API_LOGLEVEL` is non-zero (that kernel logging reads the output buffer the gate leaves
    unwritten); with the variable unset nothing changes (4 unit tests, checked live).
  - `PFUSE_HOIST` (`inkling_phoist`, new, opt-in, default `0`): the loop controller's linear source projection is
    prepared once per loop pass instead of inside every fused visit (99.96 % of elements bit-equal, FP64-reference
    error within 1.5x of the fused path; a start-up self-test with fallback). It ships off because two long prompts
    gave different greedy text with it on, and the check of whether that is ordinary rounding is not finished.
    `PFUSE_HOIST=1` turns it on; `PFUSE_HOIST=exact` is a bit-identical, slower variant.
  - `PROLOGUE` (`inkling_prologue`, new, opt-in, default `0`): the sliding-window visit's K conv + L2 clip and V
    conv + value norm fused, one kernel each. It ships off because it changes single values by the smallest BF16 step
    (a different order of the sum of squares), and greedy outputs then did not match the control.
  - With any of them on, `scripts/launch.sh` refuses `--enable-mixed-chunk` (exit 2).
  - Cold prefill, release defaults against a control on the v2.8.1 tree, same box, mean of 2 runs: 1K 2,821 vs 2,836
    (flat), 8K 5,527 vs 5,411 (+2.1 %), 32K 5,189 vs 5,073 (+2.2 %), 128K 3,785 vs 3,749 (+1.0 %); cold time to first
    token 1.48 / 6.30 / 34.6 s at 8K / 32K / 128K. Opt-in, same box: hoist + gate 8K 5,599 (+3.5 %), 32K 5,254
    (+3.5 %), 128K 3,844 (+2.6 %); hoist + gate + prologue 8K 5,656.
- Memory: the prefill levers add up to ~0.6-0.7 GB of transient memory during long prefills, so `SPEC=dflash2` up to
  `CTX=262144` sizes at `MEM_FRACTION=0.88` with any of them on, as with the default gate (0.89 with all three off;
  0.85 above 262,144 as before; `SPEC=dspark` keeps 0.89). KV pool 2,349,056 tokens (v2.8.1 at 0.89: 2,400,384,
  -2.1 %). Lowest MemAvailable over 4 release-defaults runs with thinking c24 (48 x 2,048 tokens) alongside a cold
  240K needle: 5.0 GiB (guard 3.5 GiB; 48 / 48 requests completed). At 0.89 the guard stopped the server 25 s into
  that load (3.3 GiB), so 0.88 stays. Needles at 128K and 240K PASS.
- Speedbench full, release defaults (two runs, 0 failed): prose 64.99 / 64.96, code 81.52 / 81.43, structured 77.62 /
  77.56, agentic 83.32 / 83.21, multilingual 57.63 / 57.56, long context 72.15 / 72.94, all 72.78 / 72.82 (v2.8.1
  72.86). Same tree without the cap: 64.06 (long context 67.31); with v2.8.1's policy: 72.42 (73.05).
- The DFLASH verify-width policy now prices greedy and sampled requests separately and has a context bucket
  (`df-SLDF2x.json`, the new `DFCAP` default; `bsmax=8` as before). It was fitted on a larger full-width capture (all
  batch bands, greedy and sampled thinking, 32K-128K prompts) with the same survival-table form (slot x selector
  confidence) and cross-fitted on held-out requests. Sampled requests verify all rows from 8 requests up; a single
  greedy request at 64K tokens of context or more caps less (price 0.057 instead of 0.095). The policy file adds two
  optional fields, `theta_x` (price per batch band, greedy / sampled, context bucket) and `ctx_edges` (16,384 /
  65,536); files without them behave as before. The rule is still a stopping rule on draft-side information, so the
  target distribution is unchanged.
  - Measured on box A on the v2.8.1 stack, ABBA, two runs per policy, v2.8.1 policy -> `df-SLDF2x`: greedy decode
    prose c4 141.2 / 138.0 -> 140.2 / 140.6, c8 201.9 / 205.3 -> 201.6 / 202.4; code c4 137.1 / 137.0 -> 139.9 /
    134.2, c8 204.4 / 199.5 -> 201.3 / 199.2 (within run noise); speedbench quick 73.17 -> 73.75; decode at 32K
    61.27 -> 66.05 (+7.8 %), at 128K 51.83 -> 50.91.
  - Exactness against the uncapped server, 30 speedbench quick prompts: 27 / 30 identical, 2 top-2 near-ties and 1
    three-way near-tie (longctx-09 at token 93: four candidates within 0.1 nat, ' It' -1.939, ' self' -1.991,
    ' In' -2.033, ' Need' -2.035; the capped run took the third at 0.094 nat). The same class as previous releases.
    Against v2.8.1 the release defaults give identical greedy outputs on 29 / 30 prompts; the one difference,
    longctx-09, is this near-tie and comes from the cap, not the sink gate (gate off with the cap on gives the same
    outputs as the release on 30 / 30, `results/CAP282-s06/`).
  - The fitting tool (`tools/dfreplay.py --per-mode --ctx-price`) scores sampled requests by the exact expected yield
    of block verification when the capture has the target's candidate probabilities (`tools/bvexpect.py`).
- Closed prefill candidates (measured, not adopted): docs/experiments.md, "v2.8.2".

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
