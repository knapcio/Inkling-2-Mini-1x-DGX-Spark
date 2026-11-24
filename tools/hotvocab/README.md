# Hot-vocabulary trace corpus

Goal: a multi-million-token corpus of Inkling-2-Mini's own outputs, so the `hotvocab:N` draft list
(`overlay/inkling_draftbytes`) can be ranked on real target tokens and N chosen from held-out coverage.

| file | what |
|---|---|
| `prompts.jsonl` | 46,479 prompts (70 MB, md5 `bf395e313dc21f5e906d5b74f83ba771`), pre-shuffled, so any prefix has the full mix. Fields: `id, source, license, category, language, messages, tools?, reasoning_effort, max_tokens` |
| `prompts.stats.json` | mix and drop counts |
| `gen_corpus.py` | generator, runs ON the Spark, stdlib only, resumable |
| `coverage.py` | held-out coverage + acceptance/speed model + writes the hot list (needs `tokenizers`) |
| `tokenizer/` | tokenizer files of `thinkingmachines/Inkling-2-Mini` @ `ed40e7da` (no weights); `tokenizer.json` sha256 `7f332727...` = the copy in `engine/.weights` |
| `build_prompts.py`, `fetch_sources.py` | how `prompts.jsonl` was made (Mac, pyarrow + huggingface_hub) |
| `validation/` | coverage reports on the round-1 traces |

## Prompt corpus

Mix: chat 46.6 %, code 15.8 %, math 12.7 %, agentic (OpenAI `tools`, some with tool results) 8.9 %, structured
output 8.5 %, long-context summaries 7.4 % (news and Wikipedia articles up to ~12K chars). Answer language: en 77.7 %,
pl 4.3, es 3.0, zh 3.0, de 2.7, ja 2.5, fr 2.0, ru 2.0, pt/it/uk/ko/ca 2.8 (multilingual 22 %; slightly above the
15-20 % target on purpose, as multilingual is where coverage is weakest). 3,207 multi-turn prompts. Effort suggestions
low 28 / medium 37 / high 34 %.

| source | licence | prompts |
|---|---|---:|
| allenai/WildChat-1M (2 of 14 shards; toxic/redacted rows dropped; real user chats, 12 languages) | ODC-BY-1.0 | 8,826 |
| OpenAssistant/oasst2 (root prompts, 15 % second turns; 13 languages) | Apache-2.0 | 5,196 |
| NousResearch/hermes-function-calling-v1 (tool calls, tool results, JSON-schema mode) | Apache-2.0 | 5,096 |
| bigcode/self-oss-instruct-sc2-exec-filter-50k (code; 1,463 as coding-agent tasks with a repo tool set) | ODC-BY | 4,263 |
| HuggingFaceH4/ultrachat_200k (test_sft) | MIT | 3,955 |
| ise-uiuc/Magicoder-OSS-Instruct-75K | MIT | 3,500 |
| openai/gsm8k (train) | MIT | 2,774 |
| databricks/databricks-dolly-15k | CC-BY-SA-3.0 | 2,638 |
| EleutherAI/hendrycks_math (train) | MIT | 2,598 |
| abisee/cnn_dailymail 3.0.0 (test; summaries and JSON/YAML/CSV extraction) | Apache-2.0 | 2,500 |
| CohereLabs/aya_dataset (pl 900, ja, zh, de, es, ...) | Apache-2.0 | 2,453 |
| wikimedia/wikipedia 20231101 (en/pl/de/es/zh/ja/fr/ru articles, summary tasks in the article's language) | CC-BY-SA-4.0 | 1,892 |
| juletxara/mgsm (de/es/fr/ru/zh/ja) | CC-BY-SA-4.0 | 540 |
| llm-jp/oasst2-33k-ja | Apache-2.0 | 248 |

965 of the English prompts carry an instruction in the target language to answer in it (mostly pl, then de, es, zh, ja, fr, it, ru).
Only the model's outputs are used; the prompts are not redistributed with any recipe.

Exclusions: speedbench prompts (`prompts/*.json`, 203 texts) by word-6-gram / CJK-8-gram containment >= 0.25 in either
direction (2 dropped: "What is the capital city of ...?"), the `bench/think_load.py` puzzle shape (regex, 6 dropped),
exact duplicates (3,673, mostly WildChat).

## Commands

```bash
# 1. copy (workstation -> an idle Spark, <box>)
rsync -a tools/hotvocab/{gen_corpus.py,prompts.jsonl} <box>:inkling/hotvocab/

# 2. server: the normal recipe launch (scripts/launch.sh, served name inkling-2-mini, reasoning + tool parsers on).
#    The draft config does not matter: outputs follow the target distribution exactly (T=0 = target greedy).

# 3. smoke test (30 requests, ~1-2 min), check the first records have response_token_ids
ssh <box> 'cd ~/inkling/hotvocab && python3 gen_corpus.py --prompts prompts.jsonl --out traces.jsonl --limit 30 --concurrency 16 \
  && python3 -c "import json; r=[json.loads(l) for l in open(\"traces.jsonl\")]; print(len(r), sum(bool(x[\"response_token_ids\"]) for x in r), r[0][\"usage\"])"'

# 4. the run (same command resumes after any stop; Ctrl-C/SIGTERM lets in-flight requests finish)
ssh <box> 'cd ~/inkling/hotvocab && nohup python3 gen_corpus.py --prompts prompts.jsonl --out traces.jsonl \
  --concurrency 28 --target-tokens 4000000 > gen.log 2>&1 &'
ssh <box> 'tail -3 ~/inkling/hotvocab/gen.log'     # progress line every 60 s: M tok/h, ETA, finish=length share

# 5. back to the Mac and score (traces only on the Mac; ~1-3 min for 4M tokens)
rsync -a <box>:inkling/hotvocab/traces.jsonl tools/hotvocab/
cd tools/hotvocab
PY=/path/to/python-with-tokenizers   # e.g. any venv with `pip install tokenizers`
$PY -I coverage.py --tokenizer tokenizer traces.jsonl --prior prompts.jsonl \
    --eval-only ../../results/screen/B0-base-s05-greedy.jsonl \
    --tokens-per-step <measured> --step-ms <measured c1 step> \
    --json cov4m.json --write-hot hot.json --top 98304 | tee cov4m.txt

# 6. arm: INKLING_DRAFT_BYTES=hotvocab:<N>,gemv,fold8,head8,draftmx8  INKLING_DRAFT_HOTVOCAB_FILE=<path>/hot.json
```

Generator defaults: concurrency 28; effort = the prompt's suggestion 70 % of the time, else uniform
low/medium/high; temperature 0 for 30 % of prompts, else 0.8 or 1.0 (top_p 1, per-prompt seed); max_tokens 1536 /
2048 / 3072 by effort; non-streaming with `return_token_ids` (SGLang returns `response_token_ids` = the exact
sampled ids; if the server refuses the field, the script continues on text). All choices are hashed from the prompt
id, so a resumed run makes the same choices. A dropped server makes workers wait on `/v1/models` without consuming
prompts; real failures go to `traces.jsonl.errors.jsonl` and are retried on the next start (3 attempts per id).
Tested on the Mac against a fake server (kill/restart mid-run, resume, 400 on `return_token_ids`, target stop).
At ~370 tok/s (thinking c24) 4M tokens take ~3 h; the 46K prompts hold several such runs (`--category`,
`--language` filters exist for a top-up, e.g. `--language pl,de,es,zh,ja`).

Building the list: use `coverage.py --write-hot` (exact ids, EOS first like `build_hotvocab.py`, then ids by trace
count, then ids never seen in traces back-filled by their count in the prompt text). `build_hotvocab.py` also
works on `traces.jsonl` but it reads `reasoning_content` + `content` as one concatenated string (it does not know the
`response_token_ids` key) and has no back-fill; adding `"response_token_ids"` to its `ID_KEYS` makes it use the ids.

Choosing N: take the N with the best modelled tok/s in the `acceptance / speed model` table, as long as held-out
miss <= ~1 % overall and no large language bucket (pl/de/es/zh) is above ~3 %; between Ns within ~0.3 % of the best,
take the smaller. Then measure: speedbench quick c1/c4 and the mean accept length from the server log against the
same-box base. The model is an estimate (see the `coverage.py` docstring): it assumes misses are independent of
whether the draft would have been right (pessimistic) and ignores the verify-cap interaction (the cap reads the
draft's confidence, which the hot set changes).

## Validation on the existing small sets

Inputs: round-1 traces `results/screen/T0-*.jsonl` (27 ok) + `T0b-*.jsonl` (274) = 301 records, 274 prompts,
191,650 target tokens, 12,791 distinct ids; eval-only: the speedbench greedy dumps (`results/screen/*greedy*.jsonl`
and an earlier hot list's copies, deduplicated to 14,331 tokens; never ranked on). The text rebuild of the emitted
ids matches `usage.completion_tokens` exactly on 297/301 records (191,650 vs 191,636 tokens).

Held-out coverage, split by prompt, 2 folds pooled (%):

| ranking | 16K | 24K | 32K | 48K | 64K | 96K |
|---|---:|---:|---:|---:|---:|---:|
| traces only (as round 1; flat above 12.8K distinct ids) | 93.45 | 93.45 | 93.45 | 93.45 | 93.45 | 93.45 |
| traces + prompt-text back-fill (`--prior prompts.jsonl`) | 96.11 | 97.39 | **98.18** | **98.97** | 99.46 | 99.87 |
| same, reasoning tokens | 96.75 | 97.77 | 98.39 | 99.10 | 99.51 | 99.88 |
| same, content tokens | 95.24 | 96.86 | 97.88 | 98.79 | 99.40 | 99.85 |
| same, speedbench greedy (eval-only, list from all traces) | 94.60 | 96.67 | 98.08 | 98.93 | 99.59 | 99.87 |

T0b alone, traces only: 93.10 % at 32K (round 1 reported 92.7 % on its 135K tokens). By language with back-fill
at 32K: en 98.35, ja 98.6, pl 94.1, zh 94.8, fr 92.9, es 89.1, de 86.3, pt 85.4 (each only 300-1,900 tokens).

Learning curve (fold 0 held out; traces only -> with back-fill, 32K): 10.5K train tokens 72.5 -> 95.8, 19.9K 80.6 ->
96.6, 42K 88.2 -> 97.6, 93K 93.3 -> 98.2. The traces-only miss roughly halves per doubling and is nowhere near
flat at 93K tokens, so a 4M corpus should move the curve well past these numbers.

Speed model at c1 (shipped `fold8,head8` on both sides: full-vocab draft vocab 830 MB/step; step 43.1 ms;
a = 0.602 from 2.47 tokens/step; with back-fill):

| N | miss % | draft MB saved | ms saved | tokens/step | tok/s | break-even miss % |
|---:|---:|---:|---:|---:|---:|---:|
| 16K | 3.89 | 762 | 3.46 | -4.9 % | +3.4 % | 6.3 |
| 32K | 1.82 | 695 | 3.16 | -2.4 % | +5.4 % | 5.7 |
| 48K | 1.03 | 627 | 2.85 | -1.4 % | **+5.6 %** | 5.1 |
| 64K | 0.54 | 559 | 2.54 | -0.7 % | +5.5 % | 4.5 |
| 96K | 0.13 | 424 | 1.93 | -0.2 % | +4.5 % | 3.4 |

On these weak data the optimum is flat between 40K and 64K (+5.5-5.6 %). With a = 0.7 / 0.8 the 48K figure is
+5.1 / +4.6 %. Each 1 % of misses costs ~1.3 % of tokens/step at a = 0.6.
Full reports: `validation/round1-coverage{,-prior}.{txt,json}`.

Reproduce (Mac):

```bash
R=results/screen
$PY -I coverage.py --tokenizer tokenizer $R/T0-sglang-traces-traces.jsonl $R/T0b-sglang-traces-traces.jsonl \
    --eval-only $R/*-greedy.jsonl --prior prompts.jsonl --json validation/round1-coverage-prior.json
# rebuild the prompt file: python3 -I fetch_sources.py DL; python3 -I build_prompts.py --dl DL \
#   --speedbench ../speedbench --out prompts.jsonl --stats prompts.stats.json
```

## What ships

`overlay/inkling_draftbytes/hotvocab/inkling2-mini-hot-r2.json` is the list built with these tools on 2026-11-21:
2,311 records (1.95M completion tokens, effort low / medium / high, temperature 0 / 0.8 / 1.0) generated by the
recipe's own server from prompts of the datasets listed above, ranked by `coverage.py --write-hot` with prompt-text
counts as the fill for ids never seen in the outputs. Neither the prompts (`prompts.jsonl`, 70 MB) nor the traces are
distributed here; `build_prompts.py` / `fetch_sources.py` rebuild the prompt file from the public datasets.
Coverage report: `results/kbench/round2/hotvocab/cov19m.txt`.
