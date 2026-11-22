# Parallel vs rag-all-fields (96 PDFs)

On the same 96 PDFs, `rag-all-fields` has the higher local exact-match score. The chat tokens behind the scored output are about one fifth of Parallel (3.04M versus 14.26M), and the LLM calls are 142 versus 649.

The Parallel score comes from the one run, among fourteen retries, that finished and wrote `predictions.jsonl`. The `run_metrics.json` in that directory records only 414,425 fresh prompt tokens, because 623 of 649 calls were cache hits. Deduplicating the fourteen directories by `request_hash`, the billed cost is 656 requests and 14,516,858 chat tokens.

The scores are a local diagnostic from each run's `metrics.json`, using the gold rows in `data/dev_references.jsonl` whose ids appear in the predictions. The official course scorer was not part of this comparison.

## Setup

Both runs were launched from `taskc_chunkSys_baseline`. PDFs come from `data/pdfs_dev_100/`, and the id list is `data/ids_100.txt`. That file is now 96 lines and matches the paper ids of both scored runs exactly. The chat model is `deepseek-v4.1-flash-1017` on Academic Cloud. GROBID is local at `http://localhost:8070`. The LLM cache directory is `.cache/taskc` for both.

| Item | Parallel | rag-all-fields |
| --- | --- | --- |
| Scored directory | `runs/pdfs_96_parallel_deepseek4.1_pages_a` | `runs/pdfs_96_rag_all_fields_deepseek4.1` |
| Mode | One all-field extraction per 10-page chunk, merged in document order | Top 4 pages retrieved for each of four field groups, then one all-field extraction on the union |
| `--min-votes` | 2 | 1 |
| `--workers` | 1 at first, then 2 from `pdfs_98_…_pages_b` | 4 (this mode does not use workers to choose pages) |
| `--context-budget` | 65000 on the first attempt, then 131072 | 131072 |
| `--max-output` | 16384 | 16384 |
| `--top-k` | unused | 4 |
| Embedding model | none | `qwen3-embedding-4b`, same `--base-url` as chat |
| Completion | Twelve of fourteen directories stopped early; one has no `failure.json`; one finished 96 papers | One run finished all 96 papers |

The code also requires at least 3 votes for `biodiversity_level`, above `--min-votes`. Parallel recall is lower than rag overall. The vote threshold is one part of that gap, and retrieval coverage is another.

## Commands

Run these from the project directory. `LLM_API_KEY` lives only in the shell and is not written to `run_config.json`. The flags below match the `run_config.json` saved in the corresponding directory.

```bash
cd /Users/yuquancui/Desktop/aNLP/taskc_chunkSys_baseline
curl -sS -o /dev/null -w "%{http_code}\n" http://localhost:8070/api/isalive
export LLM_API_KEY="your-api-key"
```

A healthy GROBID prints `200`. `--output` must be an empty directory. A directory that already failed cannot be reused.

### rag-all-fields

Finished in one process. Wall clock was 1 hour 48 minutes (2026-11-21 11:20–13:09).

```bash
.venv/bin/python -m core_pipeline run \
  --input data/pdfs_dev_100 \
  --ids data/ids_100.txt \
  --input-format pdf \
  --output runs/pdfs_96_rag_all_fields_deepseek4.1 \
  --mode rag-all-fields \
  --model deepseek-v4.1-flash-1017 \
  --base-url https://chat-ai.academiccloud.de/v1 \
  --grobid-url http://localhost:8070 \
  --cache .cache/taskc \
  --workers 4 \
  --chunk-by pages \
  --pages-per-chunk 10 \
  --quote-threshold 1.0 \
  --min-votes 1 \
  --context-budget 131072 \
  --max-output 16384 \
  --embedding-model qwen3-embedding-4b \
  --embedding-max-bytes 6000 \
  --embedding-batch-size 16 \
  --top-k 4
```

`rag` and `rag-all-fields` set `chunk_by` to `pages` before extraction, so GROBID is asked for TEI with coordinates. `--pages-per-chunk` does not decide which pages reach the model. Each of the four groups keeps at most 4 pages. Those pages are deduplicated by page id, sorted by page number, and sent through one all-field extraction.

Evaluation (the current `metrics.json` includes BERTScore, so `--bert-model` was set):

```bash
.venv/bin/python -m core_pipeline evaluate \
  --gold data/dev_references.jsonl \
  --predictions runs/pdfs_96_rag_all_fields_deepseek4.1/predictions.jsonl \
  --output runs/pdfs_96_rag_all_fields_deepseek4.1/metrics.json \
  --bert-model bert-base-multilingual-cased
```

### Parallel

The scored run is the last one. `--workers 2`, `--context-budget 131072`, and `--min-votes` left at its default of 2. That process took 18 minutes 38 seconds. Of its calls, 623/649 hit the cache left by earlier retries.

```bash
.venv/bin/python -m core_pipeline run \
  --input data/pdfs_dev_100 \
  --ids data/ids_100.txt \
  --input-format pdf \
  --output runs/pdfs_96_parallel_deepseek4.1_pages_a \
  --mode parallel \
  --model deepseek-v4.1-flash-1017 \
  --base-url https://chat-ai.academiccloud.de/v1 \
  --grobid-url http://localhost:8070 \
  --cache .cache/taskc \
  --workers 2 \
  --chunk-by pages \
  --chunk-chars 5000 \
  --overlap 300 \
  --pages-per-chunk 10 \
  --quote-threshold 1.0 \
  --min-votes 2 \
  --context-budget 131072 \
  --max-output 16384
```

```bash
.venv/bin/python -m core_pipeline evaluate \
  --gold data/dev_references.jsonl \
  --predictions runs/pdfs_96_parallel_deepseek4.1_pages_a/predictions.jsonl \
  --output runs/pdfs_96_parallel_deepseek4.1_pages_a/metrics.json \
  --bert-model bert-base-multilingual-cased
```

The first thirteen attempts used the same input and cache. They changed `--output`, and the `--workers` / `--context-budget` values in the table below. Each attempt started again from the first id. Requests that had already succeeded hit the cache. A failure stops the run and does not write `predictions.jsonl`. Audit files span 2026-11-19 13:57 to 2026-11-20 12:18, including idle gaps.

The first attempt was stopped on paper `25ABQZIH` by the local context guard (input estimate 50007 bytes + `max-output` 16384 > budget 65000). It sent no LLM request:

```bash
.venv/bin/python -m core_pipeline run \
  --input data/pdfs_dev_100 \
  --ids data/ids_100.txt \
  --input-format pdf \
  --output runs/pdfs_100_parallel_deepseek4.1_pages \
  --mode parallel \
  --model deepseek-v4.1-flash-1017 \
  --base-url https://chat-ai.academiccloud.de/v1 \
  --grobid-url http://localhost:8070 \
  --cache .cache/taskc \
  --workers 1 \
  --chunk-by pages \
  --pages-per-chunk 10 \
  --quote-threshold 1.0 \
  --min-votes 2 \
  --context-budget 65000 \
  --max-output 16384
```

From `pdfs_100_…_pages_b` onward the budget is 131072. `--workers` is 1 from `pages_b` through `pdfs_98_…_pages_a`. From `pages_b` through `pdfs_96_…_pages_a`, the remaining flags match the scored command above; only `--output` and `--workers` change.

| Order | `--output` | workers | Budget | Papers reached | New billed requests | New chat tokens | Stop |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `runs/pdfs_100_parallel_deepseek4.1_pages` | 1 | 65000 | 0 | 0 | 0 | Context guard, `25ABQZIH` |
| 2 | `runs/pdfs_100_parallel_deepseek4.1_pages_b` | 1 | 131072 | 3 | 4 | 70,634 | Server disconnected, `2E9XWUUE` |
| 3 | `runs/pdfs_100_parallel_deepseek4.1_pages_c` | 1 | 131072 | 3 | 0 | 0 | Connection refused, `2E9XWUUE` |
| 4 | `runs/pdfs_100_parallel_deepseek4.1_pages_d` | 1 | 131072 | 3 | 0 | 0 | Server disconnected, `2E9XWUUE` |
| 5 | `runs/pdfs_100_parallel_deepseek4.1_pages_e` | 1 | 131072 | 3 | 0 | 0 | Server disconnected, `2E9XWUUE` |
| 6 | `runs/pdfs_100_parallel_deepseek4.1_pages_f` | 1 | 131072 | 17 | 185 | 4,171,112 | GROBID HTTP 500, `3WEEGFGW` |
| 7 | `runs/pdfs_100_parallel_deepseek4.1_pages_g` | 1 | 131072 | 17 | 0 | 0 | Timed out, `3WEEGFGW` |
| 8 | `runs/pdfs_99_parallel_deepseek4.1_pages_a` | 1 | 131072 | 20 | 28 | 547,271 | Incomplete response: length, `46RX4AEN` |
| 9 | `runs/pdfs_99_parallel_deepseek4.1_pages_b` | 1 | 131072 | 20 | 0 | 0 | Incomplete response: length, `46RX4AEN` |
| 10 | `runs/pdfs_99_parallel_deepseek4.1_pages_c` | 1 | 131072 | 33 | 102 | 2,261,671 | Timed out, `5SIYLM9W` |
| 11 | `runs/pdfs_98_parallel_deepseek4.1_pages_a` | 1 | 131072 | 44 | 173 | 3,586,636 | HTTP 429, `6JGB9KZ2` |
| 12 | `runs/pdfs_98_parallel_deepseek4.1_pages_b` | 2 | 131072 | 52 | 57 | 1,447,544 | No `failure.json` |
| 13 | `runs/pdfs_97_parallel_deepseek4.1_pages_a` | 2 | 131072 | 85 | 81 | 1,821,371 | GROBID HTTP 500, `V2BVMBPF` |
| 14 | `runs/pdfs_96_parallel_deepseek4.1_pages_a` | 2 | 131072 | 96 | 26 | 610,619 | Finished; this directory is scored |

"New billed requests" counts a `request_hash` only when it first appears in that directory with `cache_hit` false. Rows with zero requests failed again on a cached prefix. The fourteen runs sum to 14,516,858 tokens. Of those, 51 requests (1,159,528 tokens) do not appear in the final 96 audits. No hash was billed twice.

Directory names count down from 100 to 96. Three papers that caused a failure are absent from the final 96: `3WEEGFGW`, `5SIYLM9W`, and `V2BVMBPF`. The current `data/ids_100.txt` is those 96 ids.

## Quality

Gold support (TP+FN, the same on both sides): habitat 133, biodiversity level 63, ecosystem category 45, ecosystem term 52, species group 104, scientific name 195, German name 229, service category 8, service term 8.

| Summary | rag-all-fields | Parallel | rag − parallel |
| --- | ---: | ---: | ---: |
| C1 macro F1 | 0.549 | 0.455 | +0.094 |
| C2b macro F1 | 0.257 | 0.198 | +0.060 |
| All categorical macro F1 | 0.366 | 0.289 | +0.078 |
| All categorical micro F1 | 0.415 | 0.327 | +0.088 |
| biodiversity_variable BERTScore | 0.217 | 0.164 | +0.053 |

C1 is the unweighted mean of habitat, biodiversity_level, ecosystem_type.category, ecosystem_type.term, and taxa.species_group. C2b is the unweighted mean of scientific name, German name, species group, biodiversity level, and both ecosystem-service fields. BERTScore is the best-match F1 of `biodiversity_variable` strings with `bert-base-multilingual-cased`. Unmatched strings count as zero.

| Field | RAG P | RAG R | RAG F1 | RAG TP/FP/FN | Par P | Par R | Par F1 | Par TP/FP/FN |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| habitat | 0.750 | 0.699 | 0.724 | 93/31/40 | 0.757 | 0.654 | 0.702 | 87/28/46 |
| biodiversity_level | 0.326 | 0.492 | 0.392 | 31/64/32 | 0.386 | 0.429 | 0.406 | 27/43/36 |
| ecosystem_type.category | 0.423 | 0.911 | 0.577 | 41/56/4 | 0.326 | 0.644 | 0.433 | 29/60/16 |
| ecosystem_type.term | 0.333 | 0.692 | 0.450 | 36/72/16 | 0.198 | 0.462 | 0.277 | 24/97/28 |
| taxa.species_group | 0.563 | 0.644 | 0.601 | 67/52/37 | 0.435 | 0.481 | 0.457 | 50/65/54 |
| taxa.scientific_name | 0.524 | 0.169 | 0.256 | 33/30/162 | 0.189 | 0.072 | 0.104 | 14/60/181 |
| taxa.german_name | 0.182 | 0.009 | 0.017 | 2/9/227 | 0.271 | 0.057 | 0.094 | 13/35/216 |
| ecosystem_service.category | 0.171 | 0.750 | 0.279 | 6/29/2 | 0.075 | 0.375 | 0.125 | 3/37/5 |
| ecosystem_service.term | 0 | 0 | 0 | 0/0/8 | 0 | 0 | 0 | 0/3/8 |

Rag is higher on seven of the nine fields. The largest gaps are ecosystem term (+0.173), service category (+0.154), scientific name (+0.152), species group (+0.144), and ecosystem category (+0.145). Parallel is higher on German name (0.094 versus 0.017, 13 true positives versus 2) and biodiversity level (0.406 versus 0.392). Service term has 0 true positives on both sides.

Pooled counts: rag 309 TP / 343 FP / 528 FN; Parallel 247 / 428 / 590.

## Efficiency

The two "scored output" columns sum usage over every call in the final 96 audits, including cache replays. That is the same definition as `logical_prompt_tokens` / `logical_completion_tokens` in the rag `run_metrics.json` (checked: 2,016,597 and 1,021,949). The Parallel `run_metrics.json` was written by an earlier version of the code and has no logical fields. The table rebuilds them from the usage on 649 calls.

"Fourteen runs, billed" adds only requests with `cache_hit == false`, once per hash.

| Metric | rag-all-fields | Parallel, scored output | Parallel, fourteen runs billed |
| --- | ---: | ---: | ---: |
| Papers | 96 | 96 | 96 finished; twelve runs stopped early |
| LLM calls | 142 | 649 | 656 unique fresh requests |
| Calls per paper, median / mean | 1 / 1.5 | 2 / 6.8 | — |
| Prompt tokens | 2,016,597 | 10,364,815 | 10,464,477 |
| Completion tokens | 1,021,949 | 3,894,280 | 4,052,381 |
| of which reasoning | 947,490 | 3,512,903 | 3,656,748 |
| Total chat tokens | 3,038,546 | 14,259,095 | 14,516,858 |
| Chat tokens per paper, median / mean | 26,653 / 31,652 | 43,789 / 148,532 | — |
| Sum of call latency | 52 min | 3.5 h | 3.6 h (fresh only) |
| Wall clock of the finishing process | 1 h 48 min | 18.6 min (96% cache hits) | Audit files span 22.4 h, including gaps |
| Embedding HTTP calls | 324 | 0 | 0 |
| Embedding tokens (qwen3-embedding-4b) | 3,817,591 | 0 | 0 |
| Cached embedding items | 565 | 0 | 0 |
| Truncated LLM calls | 23 | 0 | — |

On 84 of 96 papers, rag makes a single LLM call. Its maximum is 9 calls and 122,817 tokens. Parallel call counts run from 1 to 50. Parallel uses more chat tokens on 76 of 96 papers. The median gap is 1.6×. The mean gap is 4.7×, because long papers get one call per 10 pages.

The five longest papers account for 26% of the 14.26M tokens in the Parallel scored output. Rag makes one call on each of them:

| Paper | Parallel calls | Parallel tokens | RAG calls | RAG tokens |
| --- | ---: | ---: | ---: | ---: |
| 2E9XWUUE | 40 | 1,059,750 | 1 | 38,729 |
| 4YXRYRJC | 40 | 863,336 | 1 | 38,988 |
| 2P53UVJA | 32 | 685,171 | 1 | 33,862 |
| 3LGPK6BL | 25 | 567,259 | 1 | 30,224 |
| 2EUNPHDZ | 30 | 513,223 | 1 | 29,575 |

The 23 truncated rag calls have no usage, so 3.04M is a lower bound. Adding the 3.82M embedding tokens to the chat tokens gives 6.86M, still below Parallel's 14.26M chat tokens. The models differ, so that sum is only a count comparison.

Eleven request hashes in the Parallel scored output appear in two papers, which counts 112,569 prompt tokens twice. By unique hash the total is 638 requests and 14,146,151 tokens. Another 33 final calls (788,821 tokens) never appear as fresh requests in these fourteen directories. Their usage comes from an earlier cache. They are included in "scored output" and excluded from "fourteen runs, billed".

## How to read the token columns

- Scored-output tokens answer how much this extraction costs. The fourteen-run column answers what these fourteen directories actually billed to the API.
- Fresh tokens in the final Parallel directory (prompt 414,425, completion 196,194) cover only the 26 cache misses. They are not the cost of the 96 papers.
- Reasoning tokens are already inside completion tokens and are not added again.
- Embedding tokens are listed separately from chat tokens.
