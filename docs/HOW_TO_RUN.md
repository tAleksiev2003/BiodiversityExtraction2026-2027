# How to run

Run every command from the repository root. Each prediction command creates a new
folder `runs/<timestamp>_<name>/` with `predictions.jsonl` and `predictions.zip`.

## Two dev sets

| Set | Papers | Used for |
|---|---|---|
| `with_pdf` | 100 with a PDF | **the dev set**: developing and comparing our systems (default) |
| `full` | all 900 in `dev.jsonl` | only for checking our scorer against the official baseline numbers; not for development |

Only 100 dev PDFs exist; the 900 dev references on CodaBench were an error (confirmed by
the course, SPEC.md, section 13). Nothing has to be predicted for the other 800 ids.

## 1. Once

```bash
python3 -m pytest                          # tests on invented data, expect: all passed
python3 src/data.py make-dev-references    # writes data/processed/dev_with_pdf.jsonl
```

## 2. Predict, then score

Pattern: **predict** with `baselines.py` (later: the pipelines), then **score** the newest run.

```bash
# most-frequent baseline on the with_pdf set
python3 src/baselines.py --kind most_frequent --split dev --dev-set with_pdf
LATEST=$(ls -td runs/*_baseline_most_frequent_dev_with_pdf | head -1)
python3 src/evaluation.py score --references data/processed/dev_with_pdf.jsonl \
    --predictions "$LATEST/predictions.jsonl" --task all --categorical-only \
    --output "$LATEST/scores.json"
```

For the random baseline replace `most_frequent` by `random` (twice).

For the `full` set use `--dev-set full`, `*_dev_full` and
`--references data/raw/references/dev.jsonl`.

## 3. Convert PDFs and chunk them (M1)

```bash
python3 src/pdf_to_text.py --split dev --limit 3   # quick check: 3 documents
python3 src/pdf_to_text.py --split dev             # all dev documents with a PDF
python3 src/document_statistics.py --split dev     # page / token / chunk statistics
```

`pdf_to_text.py` stores every document as Docling JSON in `data/processed/docling/<split>/`
and skips documents that are already there, so `Ctrl + C` is safe: run it again to continue.
Failed documents are retried on every run. Every attempt is logged in
`data/processed/docling/<split>/conversion_log.jsonl`.

`document_statistics.py` writes `runs/<timestamp>_m1_statistics_<split>/` with
`summary.json` (report this back), `documents.jsonl` and `chunking_log.jsonl`
(chunking level and every split per document).
It prints the runtime of every document and stores each result as soon as it is done
(`documents/<id>.json` in the run folder). After `Ctrl + C`, continue the same run with
the command it printed at the start, e.g.
`python3 src/document_statistics.py --split dev --resume runs/<timestamp>_m1_statistics_dev`.

## 4. LLM client (M2)

Needs at least one key in `.env` (copy `.env.example`): `SAIA_API_KEY_1=`,
`SAIA_API_KEY_2=`, … numbered from 1 without gaps (1 to 4 keys). The old single
`SAIA_API_KEY=` still works if no numbered key is set. SAIA's limits (400 per hour,
800 per day) apply per Academic Cloud account, not per key: several keys of one account
share one quota. Only keys of n different accounts allow n × 800 requests a day. The
hourly quota refills gradually, not at the full hour.
Answers are cached in `data/processed/llm_cache/`, so the same prompt is never sent
twice, whichever key answered it.

The client sends each request with the key that has the most requests left for the
day. A key that reaches its daily limit is skipped; the run stops only when all keys
are used up ("Daily limit reached on all 2 usable API key(s)…") — run the same command
the next day to continue. A key the server rejects (HTTP 401/403) is logged as
"key 2 was rejected…" and not used again in that run: check that key in `.env`.
The summary at the end lists requests and remaining quota per key ("key 1", "key 2");
keys themselves are never shown.

```bash
python3 src/llm_client.py list-models                     # models the API offers
python3 src/llm_client.py check-keys                      # 1 request per key: OK / error, limits, quota
python3 src/llm_client.py ask --prompt-file tests/fixtures/llm_smoke_prompt.txt            # 1 request
python3 src/llm_client.py ask --prompt-file tests/fixtures/llm_smoke_prompt.txt            # again: 0 requests (cache)
python3 src/llm_client.py ask --prompt-file tests/fixtures/llm_smoke_prompt.txt --dry-run  # counts, sends nothing
```

`--model full_text` uses the full-text model instead of the chunk model.

`check-keys` sends "Answer with OK." to the chunk model once with every key, bypassing
the cache, and prints per key `OK` or the error (e.g. `HTTP 403`), the limits and the
remaining quota. Use it after adding a key, or to see how much quota is left. It costs
one request per key. `list-models` shows the quota as "not reported" because
`/v1/models` responses carry no `x-ratelimit-*` headers.

## 5. Pipeline F: full text (M3a)

Needs the converted dev documents (section 3) and the label files in `data/labels/`.

```bash
python3 src/run_pipeline.py full_text --split dev --dry-run    # counts requests, sends nothing
python3 src/run_pipeline.py full_text --split dev --limit 3    # 3 documents
python3 src/run_pipeline.py full_text --split dev              # all dev documents with a PDF
python3 src/run_pipeline.py full_text --split dev --experiment config/experiments/no_verbatim_check.yaml
```

Each run writes `runs/<timestamp>_full_text_dev[_...]/` with `summary.json` (report this
back), `documents.jsonl`, `verbatim_check_log.jsonl`, `llm_calls.jsonl` (all prompts and
answers) and `predictions.jsonl` + `predictions.zip`. A dry run or a stopped run writes no
predictions. Requests run in parallel: `llm.max_concurrent_requests_per_key` per key
(each key starts with one request until its first answer). When all keys reach their
daily limit, or after `Ctrl + C`, the run stops cleanly; run the same command again
(the next day) and the cache answers everything that was already received. A second
`Ctrl + C` quits at once.

The run without the checker (`no_verbatim_check.yaml`) sends the same prompts, so after
the normal run it is answered completely from the cache (0 requests).

Score a run:

```bash
LATEST=$(ls -td runs/*_full_text_dev | head -1)
python3 src/evaluation.py score --references data/processed/dev_with_pdf.jsonl \
    --predictions "$LATEST/predictions.jsonl" --task all --categorical-only \
    --output "$LATEST/scores.json"
```

## 6. Parallel PDF conversion

`pdf_to_text.py` converts `pdf_conversion.max_workers` documents at the same time (one
process each, about 6 GB RAM each). `--workers N` overrides the setting, `--output-dir`
writes into another folder (e.g. to measure the speed-up without touching the real one).
If a worker process dies (often out of memory), the run stops with a message; converted
documents are kept, run again with fewer workers.

## 7. Pipeline B: parallel chunks (M3b)

Needs the converted dev documents (section 3). Same options as pipeline F.

```bash
python3 src/run_pipeline.py parallel --split dev --dry-run     # chunks per paper, requests; sends nothing
python3 src/run_pipeline.py parallel --split dev --limit 3     # 3 documents
python3 src/run_pipeline.py parallel --split dev               # all dev documents with a PDF
```

Every paper is cut into chunks of `chunking.max_chunk_tokens` (16,000) tokens; a long
paper gets larger chunks, so it has at most `chunking.max_chunks_per_document` (20)
chunks, never larger than `models.chunked_part_tokens` (240,000). One request per chunk
with the chunk prompt (`paths.prompt_extraction_chunked`) and `models.chunked`
(DeepSeek, the same model as F). A paper that fits into one chunk is the full-text case:
it sends exactly F's request (full text, prompt v2), so after a run of F its answer comes
from the cache (`parallel.one_chunk_as_full_text`). Run the dry run first: it prints the
chunks in total, how many papers are sent as F's request, the median and maximum chunks
per paper and the papers with enlarged chunks, before any request is spent. The requests
counted by the dry run do not include answers already in the cache.

The run folder `runs/<timestamp>_parallel_dev[_...]/` has the same files as pipeline F;
`documents.jsonl` also gives `chunk_count`, `chunk_tokens` (the chunk size used) and
`full_text_request` per paper, `summary.json` the chunk numbers under `chunks`. Check `failed` and score as in
section 5, with `runs/*_parallel_dev`.

Experiments:

```bash
# merging (free: answered from the cache of the normal run)
python3 src/run_pipeline.py parallel --split dev --experiment config/experiments/merge_min_chunks_2.yaml
python3 src/run_pipeline.py parallel --split dev --experiment config/experiments/merge_min_chunks_3.yaml
python3 src/run_pipeline.py parallel --split dev --experiment config/experiments/no_verbatim_check.yaml
# smaller chunks (new requests, about 615 on dev)
python3 src/run_pipeline.py parallel --split dev --experiment config/experiments/chunk_8k.yaml
# the Qwen run of 2026-11-18 (from the cache: chunk prompt for every paper)
python3 src/run_pipeline.py parallel --split dev --experiment config/experiments/chunked_qwen.yaml
```

An experiment file replaces only the settings it names, and a run takes one file, so the
merge experiments run with DeepSeek.

`merge.min_chunks_per_label: k` keeps a label only if it has a verified quote in at least
min(k, chunks of the paper) different chunks; the default 1 is the union.

## Reference numbers (C1 score)

| Baseline | with_pdf (ours) | full (ours) | full (official) |
|---|---|---|---|
| random | – | 0.1001 | 0.1018 |
| most frequent | 0.2055 | 0.2443 | 0.2443 |

Upload `predictions.zip` of a `full` run to CodaBench; it must show the same score.