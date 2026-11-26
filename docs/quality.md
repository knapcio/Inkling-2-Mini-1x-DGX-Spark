# Quality and qualification

How each release was qualified, and every check behind the claim that no lever changes the target model's output
distribution. Speed and memory numbers are in [benchmarks.md](benchmarks.md); per-release details are in
[CHANGELOG.md](../CHANGELOG.md).

- [Qualification status](#qualification-status)
- [Quality checks](#quality-checks)

## Qualification status

| release | qualification | results |
|---|---|---|
| v2.9.7 | serving code = v2.9.5 (v2.9.6 with `DFLASH_CTX8=0`). 24-agent check on both boxes and a 12-turn, 24-agent soak on box B: 0 failures; lowest MemAvailable 3.8 GiB | `results/AG2961-s05/`, `results/AG2961-s06/`, `results/SOAK2961-s06/` |
| v2.9.6 | two-box: speedbench full 74.28 / 74.28 (box A / B), quality guard 30 / 30, worst MemAvailable 5.0 / 4.4 GiB over three thinking-c24 rounds with a cold 240K needle after each (3 / 3 PASS per run), 200K / 250K contexts 2 / 2 each | `results/Q296R-s05/`, `results/Q296R-s06/`, `results/Q296S-s05/`, `results/Q296S-s06/` |
| v2.9.5 | two-box: speedbench full 73.54 / 73.17, quality guard 30 / 30, worst MemAvailable 4.6 / 4.8 GiB (same load), 200K / 250K contexts 2 / 2 each | `results/Q295R-s05/`, `results/Q295R-s06/`, `results/Q295S-s05/`, `results/Q295S-s06/` |
| v2.9.4 | prefix-tail deferral release-checked on both boxes (base v2.9.3) | [CHANGELOG](../CHANGELOG.md#v294-2026-11-25) |
| v2.9.3 | two-box: Q293R and Q293S on both boxes; speedbench full 72.65 / 72.79 | `results/Q293R-s05/`, `results/Q293R-s06/`, `results/Q293S-s05/`, `results/Q293S-s06/` |
| v2.9.2 | two-box: speedbench full 72.95 / 73.12, quality guard 30 / 30 and sanity, worst MemAvailable 4.2 / 5.6 GiB, 200K / 250K contexts 2 / 2 each | `results/Q292R-s05/`, `results/Q292R-s06/`, `results/Q292S-s05/`, `results/Q292S-s06/` |
| v2.9, v2.9.1 | basic tests on one box; base v2.8.4 | [CHANGELOG](../CHANGELOG.md#v29-2026-11-25) |
| v2.8.4 | two-box: Q284R and Q284S on both boxes | `results/Q284R-s05/`, `results/Q284R-s06/`, `results/Q284S-s05/`, `results/Q284S-s06/` |
| v2.8 - v2.8.3 | basic tests on one box; full two-box qualification in v2.8.x | [CHANGELOG](../CHANGELOG.md#v28-2026-11-24) |

**Status (2026-11-26, v2.9.7, the final release of this round):** v2.9.3 passed a two-box qualification: Q293R and Q293S on both boxes; v2.9.4's prefix-tail deferral was release-checked on both boxes; v2.9.5's shared verify settle is bit-exact in the GPU check and was measured on box A, and v2.9.5 and v2.9.6 passed two-box qualifications (Q295R / Q295S, Q296R / Q296S); v2.9.7 turns v2.9.6's MXFP8 drafter context projections off by default (opt-in, `DFLASH_CTX8=1`). On v2.9.7's defaults, the 24-agent check on both boxes and a 12-turn, 24-agent soak on box B ran with 0 failures; every later turn hit the prefix cache, and the lowest MemAvailable was 3.8 GiB (`results/AG2961-s05/`, `results/AG2961-s06/`, `results/SOAK2961-s06/`).

Not measured for v2.8 / v2.8.1 (v2.8.x qualification): structured / JSON decode and c2, thinking c1 / c4 with the
cap on, 200K-512K context, the quality guard and the sanity suite.

## Quality checks

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
  `results/QL1-s05/`, `results/QL2-s05/`; the first, smaller bracket is in [experiments.md](experiments.md#v292-draft-the-prefill-loop-bank-hoist-under-a-quality-gate-2026-11-25).
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
  about 1M steps per case, 0 invalid tokens, all within the law ([experiments.md](experiments.md#verify-width-cap-on-dflash2)).
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
