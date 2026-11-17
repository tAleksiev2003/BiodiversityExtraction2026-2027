# taskc_chunkSys_baseline

Evidence-grounded baseline for **ANLP 2026 Task C1 and C2b**. It reads scientific papers (PDF or GROBID TEI), splits them into overlapping text chunks, asks an OpenAI-compatible chat model to extract biodiversity findings, and keeps a finding only when its quote can be located in the source chunk. The result is a schema-valid submission (`predictions.jsonl` inside `predictions.zip`) plus a per-paper audit trail.

The same extraction rules, quote checks, and JSON Schema are used for two architectures:

- **Parallel.** Each chunk is extracted independently. Findings are merged in document order. Removals are rejected, so one chunk cannot delete a finding produced by another chunk.
- **Sequential.** Chunks are processed in order. Each call sees the verified state so far and may add a finding or remove one when the current chunk quotes a contradiction.


## What the system extracts

Each paper becomes one JSON object keyed by `zotitem_ptr_id`. Every output field is an array. Categorical values are stored as the official German enum strings. The model may answer with those German strings or with the English aliases in `core_pipeline/schemas/english_glosses.json` (151 aliases). Python maps aliases back to German before validation.

| Field | Task | Value |
| --- | --- | --- |
| `habitat` | C1 | Habitat enum, for example `Wald` |
| `biodiversity_level` | C1 and C2b | Diversity level enum, for example `Artenvielfalt` |
| `ecosystem_type` | C1 | Object with required `category` and optional `term` |
| `taxa` | C1 and C2b | Object with required `species_group`, optional `scientific_name` and `german_name` |
| `biodiversity_variable` | C2b | Free-text German phrase for the measured variable, not a numeric result |
| `ecosystem_service` | C2b | Object with required `category` and optional `term`. The schema `details` property is dropped before validation |

The prompt tells the model to extract what **this study** investigates. Background examples, reference-list species, and claims that belong only to cited papers are out of scope. Taxa should use the smallest defensible level: individual species when few are studied, otherwise an explicitly studied group. Latin names are not translated. Uncertain names and uncertain ecosystem terms are omitted.

## Pipeline

```
PDF or TEI
    │
    ├─ PDF ── pdf_to_tei() ── GROBID /api/processFulltextDocument ── cached TEI
    └─ TEI ── read the XML as-is
            │
            ▼
      tei_sections()          title, abstract, body blocks in order
            │                 bibliography / back matter excluded
            ▼
      make_chunks()           overlapping character spans, word-boundary cuts
            │
            ▼
      run_paper()             parallel or sequential LLM calls
            │
            ├─ prompts.build_messages()
            ├─ ChatClient.complete()        JSON object {add, remove}, cached
            ├─ verified_changes()           schema + quote must match this chunk
            └─ merge_findings()             stable IDs, unique evidence spans
            │
            ▼
      submission_record()     drop evidence metadata, apply min-vote filter,
                              validate against the merged JSON Schema
            │
            ▼
      predictions.jsonl + predictions.zip
      <paper_id>.audit.json, run_config.json, run_metrics.json
```

A run is fail-fast. The output directory must be empty. If any paper raises, the pipeline writes `failure.json` (paper id, error, ids already completed) and does **not** publish `predictions.jsonl` or `predictions.zip`. A submission is written only after every requested paper succeeds.

### Document preparation

`pdf_to_tei` posts each PDF to GROBID and caches the TEI by SHA-256 of the file bytes plus the endpoint URL. A cached file is reused without calling GROBID again. The response is parsed as XML before it is stored.

`tei_sections` keeps, in document order:

1. The title from `titleStmt/title`, when present.
2. The abstract, when present.
3. Body blocks. A `head` that is a direct child of a `div` starts a new section. Paragraphs, figures, tables, formulas, and list items are collected under the current heading. Nested copies of those elements are skipped so a table cell is not extracted twice.

A TEI file with no `body`, or with no usable text beyond the title, raises an error.

`make_chunks` splits each section into spans of at most `--chunk-chars` characters (default 5000), with `--overlap` characters (default 300) shared with the next span. The cut prefers a space in the second half of the window. Overlap must be smaller than half the chunk size, and the chunk size must be at least 100 characters. Every chunk records a stable id (`s<section>-c<index>`), section heading, section index, character offsets, and text. Offsets are into the section string, so later quote matching can recover a location in the original section.

### Model call

Each chunk is sent as one chat completion with `temperature` 0. The system message contains the extraction rules. The user message is a JSON payload:

- `mode`: `parallel` or `sequential`
- `OUTPUT_SCHEMA`: merged schema without `zotitem_ptr_id` (the id is supplied by the runner)
- `ENUM_CATALOGUE`: English alias → official German enum
- `PREVIOUS_VERIFIED_STATE`: the current verified findings in sequential mode, otherwise `[]`
- `CURRENT_CHUNK`: id, section, offsets, and text

The model must return one JSON object:

```json
{
  "add": [
    {
      "field": "habitat",
      "value": "Forest",
      "evidence": [{"chunk_id": "s000-c0000", "quote": "exact words from the current chunk"}]
    }
  ],
  "remove": [
    {"finding_id": "existing ID", "quote": "contradictory current text"}
  ]
}
```

`remove` must be empty in parallel mode. In sequential mode a removal is accepted only when `finding_id` names an existing finding and the quote is verified in the current chunk. Silence in a later chunk does not delete earlier findings; the previous state is kept automatically.

### Evidence checks

A proposed addition is dropped (and recorded in the audit as `rejected`) unless all of the following hold:

- The object has exactly the keys `field`, `value`, and `evidence`.
- `field` is one of the six output fields.
- `evidence` is a non-empty list. Each item has exactly `quote` and `chunk_id`, and `chunk_id` is the chunk being processed.
- `value` matches the field schema. English aliases are translated to German. Unknown enums, extra keys, and empty strings are rejected.
- Every quote is at least 12 non-whitespace characters and is found in the chunk.

Quote search normalizes Unicode (NFKC) and collapses whitespace, then looks for an exact match. Character offsets into the original chunk text are retained. If `--quote-threshold` is below 1.0, a conservative fuzzy fallback compares the quote with nearby word windows using `difflib.SequenceMatcher` and accepts the best window only when its ratio meets the threshold. The legal range is `0.9` to `1.0`. The default `1.0` requires an exact normalized match.

Accepted findings receive a 20-character id: the SHA-256 prefix of the canonical field and value. Merging unions evidence spans that differ in `(section_index, start, end)`. Overlapping spans inside one section count as **one vote**, so a finding repeated across overlapping chunks is not inflated. `--min-votes` (default 1) drops findings whose non-overlapping evidence count is below the threshold. The published record contains values only; quotes and offsets stay in the audit and in `final_state`.

### Parallel and sequential execution

Parallel mode submits every chunk to a `ThreadPoolExecutor` (`--workers`, default 4). Results are merged in **document order**, not completion order, and the previous state sent to the model is always empty. Removals are rejected.

Sequential mode walks chunks one by one. The prompt includes the verified state. Verified removals are applied before new additions are merged. Removal ids that are not in the current state are recorded as `unknown_removal_ids` and ignored. Each sequential audit entry also stores `state_after`.

Both modes return the same pair: a validated submission record, and an audit object `{chunks, calls, final_state}`.

### Client, cache, and budgets

`ChatClient` posts to `{base_url}/chat/completions`. Optional `LLM_API_KEY` is sent as a Bearer token and is never written into the cache or `run_config.json`.

Before the request, the client estimates size as the UTF-8 length of each message plus 64 bytes. If that estimate plus `--max-output` exceeds `--context-budget`, it raises and does not call the network. The estimate is a byte proxy, not a model tokenizer count.

Requests use `response_format: {type: json_object}` unless `--no-json-mode` is set. For the model names `qwen3.6-35b-a3b` and `qwen3.8-27b`, the request also sets `chat_template_kwargs.enable_thinking` to false.

A response is accepted only when `finish_reason` is `stop` or absent, the content parses as a JSON object, and both `add` and `remove` are arrays. Extra top-level keys are dropped. Truncated completions (`finish_reason` other than `stop`/`null`) are rejected and are not cached.

Successful calls are cached under `{cache}/llm/<sha256>.json`. The key hashes the base URL and the full request body, so a change of model, prompt, chunk, or decoding settings misses the cache. Transient HTTP statuses 429, 500, 502, 503, and 504, transport errors, and parse errors are retried up to three times with sleeps of 1s, 2s, and 4s. The last failure is raised.

GROBID TEI is cached separately under `{cache}/tei/`.

## Outputs

After a successful `run`:

| File | Contents |
| --- | --- |
| `predictions.jsonl` | One submission object per paper, in the order of the id list |
| `predictions.zip` | A zip whose only member is `predictions.jsonl` |
| `<paper_id>.audit.json` | Chunks, raw model responses, cache/usage metadata, rejected items, verified removals, and the final evidence-bearing state |
| `run_config.json` | CLI arguments, Python version, platform, and SHA-256 of each schema file |
| `run_metrics.json` | Wall time, paper count, call count, cache hits, summed prompt and completion tokens for cache misses, and SHA-256 of each source file |

`evaluate` writes a metrics JSON file. It does not modify predictions.

## Requirements

- Python 3.11 or newer.
- Runtime packages: `httpx`, `jsonschema` (`requirements.txt` / `pyproject.toml`).
- Optional evaluation extra: `bert-score` and `scipy`, used only when `--bert-model` is passed.
- A running [GROBID](https://grobid.readthedocs.io/) service when `--input-format pdf` (default URL `http://localhost:8070`).
- An OpenAI-compatible chat endpoint (default `http://localhost:11434/v1`, for example Ollama). Any server that implements `/chat/completions` can be used via `--base-url` and `--model`.

GROBID and the language-model server are external processes. They are not installed by `requirements.txt`.

From the project root, with this directory on `sys.path` (the default when you launch Python here):

```bash
python -m pip install -r requirements.txt
# optional German BERTScore diagnostic
python -m pip install "bert-score>=0.3.13,<0.4" "scipy>=1.11"
```

`pyproject.toml` still declares the distribution name `anlp-taskc` and a console script `anlp-taskc` that imports `anlp_taskc`. The source package in this tree is `core_pipeline`, and `python -m core_pipeline` is the entry point that matches these files.

## Running the pipeline

Before a PDF run, check that GROBID is up. The default `--grobid-url` is `http://localhost:8070`:

```bash
curl -sS -o /dev/null -w "%{http_code}\n" http://localhost:8070/api/isalive
```

A running service prints `200`. If `--grobid-url` is different, use that host and port in the URL. TEI input does not call GROBID.

A remote chat endpoint that requires authentication reads the key from `LLM_API_KEY` and sends it as `Authorization: Bearer <key>`. A local server that does not check a key, such as Ollama, can be used without setting it. The value is not written into `run_config.json` or the response cache. Set it in the same shell before `run`:

```bash
export LLM_API_KEY="your-api-key"
```

From the project root:

```text
python -m core_pipeline run ...
python -m core_pipeline evaluate ...
python -m core_pipeline check-schemas
```

`run` and `evaluate` take the flags below. `check-schemas` takes none. A flag marked required must be passed; the others have the default shown.

### `python -m core_pipeline run`

Extract one split into `predictions.jsonl` and `predictions.zip`. The papers that enter the pipeline are the ids listed in `--ids`, not every file under `--input`. `--input` is only the directory where those files are stored. For each id the runner reads exactly `{input}/{id}.pdf` or `{input}/{id}.tei.xml`. Any other file in that directory is ignored. Ids must be unique and match `[A-Za-z0-9_-]+`. `--output` must be an empty directory. `--context-budget` must be greater than `--max-output`, and `--max-output` must be at least 1.

`--ids` is a text file with one paper id per line and no file extension. `data/ids_15.txt` is the 15-paper run:

```text
25ABQZIH
25RIYD2C
29Q84X7V
2FDHAHAR
2NJD9Y9T
2RGQT8TU
33MV2F7X
3439JHQD
38EBGVBF
3RSTTAEZ
3TWLQG7F
42ELXZ2H
49Z7RPDP
4FCS62W5
4JF8F3WL
```

With `--input data/pdfs` and `--input-format pdf`, those lines select `data/pdfs/25ABQZIH.pdf`, `data/pdfs/25RIYD2C.pdf`, and the other 13 files named in the list. A PDF that sits in `data/pdfs` but is absent from the text file is not read.

| Parameter | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--input` | yes |  | Directory that holds the papers. This directory is not itself the input list |
| `--ids` | yes |  | Text file of the paper ids that are actually processed, one id per line |
| `--output` | yes |  | Empty directory for this experiment |
| `--mode` | yes |  | `parallel` or `sequential` |
| `--model` | yes |  | Exact model name served by the chat endpoint |
| `--input-format` | no | `pdf` | `pdf` calls GROBID; `tei` reads `{id}.tei.xml` directly |
| `--base-url` | no | `http://localhost:11434/v1` | OpenAI-compatible chat endpoint |
| `--grobid-url` | no | `http://localhost:8070` | GROBID service. Used only when `--input-format` is `pdf` |
| `--cache` | no | `.cache/taskc` | Disk cache for TEI and LLM responses |
| `--workers` | no | `4` | Parallel chunk threads. Must be at least 1. With `sequential`, chunks are still processed one by one |
| `--chunk-chars` | no | `5000` | Maximum characters per chunk. Must be at least 100 |
| `--overlap` | no | `300` | Characters repeated between adjacent chunks. Must be at least 0 and less than half of `--chunk-chars` |
| `--quote-threshold` | no | `1.0` | Minimum normalized quote match, from 0.9 to 1. `1.0` requires an exact match |
| `--min-votes` | no | `1` | Minimum non-overlapping evidence spans kept in the submission. Must be at least 1 |
| `--context-budget` | no | `65000` | Byte budget for the prompt plus `--max-output`. A larger estimate is refused before the request |
| `--max-output` | no | `4096` | `max_tokens` sent to the endpoint |
| `--no-json-mode` | no | off | Omit `response_format: json_object`. By default JSON mode is on |

```bash
python -m core_pipeline run \
  --input data/pdfs \
  --ids data/ids_15.txt \
  --output runs/example_parallel \
  --mode parallel \
  --model your-model-name
```

TEI input skips GROBID:

```bash
python -m core_pipeline run \
  --input data/tei \
  --ids data/ids_15.txt \
  --input-format tei \
  --output runs/example_sequential \
  --mode sequential \
  --model your-model-name
```

### `python -m core_pipeline evaluate`

Reads gold and predictions as JSONL or as a JSON list. Paper ids must be unique. Every prediction id must exist in the gold file; gold rows for papers that were not predicted are ignored. An empty overlap is an error.

| Parameter | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--gold` | yes |  | Gold JSONL file or JSON list |
| `--predictions` | yes |  | Prediction JSONL file or JSON list, usually `predictions.jsonl` |
| `--output` | yes |  | Path of the metrics JSON file to write |
| `--bert-model` | no | none | Hugging Face model id for `biodiversity_variable`. Omit it and `C2b free text (BERTScore)` is `null`. Needs the optional `bert-score` and `scipy` packages |

```bash
python -m core_pipeline evaluate \
  --gold data/dev_references_15.jsonl \
  --predictions runs/example_parallel/predictions.jsonl \
  --output runs/example_parallel/metrics.json
```

For each categorical path, values are flattened to a set of non-empty strings and scored with pooled exact-match precision, recall, and F1 (counts summed across papers, then one F1 per field). Empty denominators score 0.

The summary keys are:

- **C1 Score(Macro F1)** averages F1 over `habitat`, `biodiversity_level`, `ecosystem_type.category`, `ecosystem_type.term`, `taxa.species_group`.
- **C2b biodiversity (Macro F1)** averages F1 over `taxa.scientific_name`, `taxa.german_name`, `taxa.species_group`, `biodiversity_level`, `ecosystem_service.category`, `ecosystem_service.term`.
- **All categorical values (Macro)** averages F1 over the nine exact-match fields above, counting `taxa.species_group` and `biodiversity_level` once each. `biodiversity_variable` is excluded.
- **All categorical values (Micro)** pools TP, FP, and FN across those same nine fields into one F1.
- **C2b free text (BERTScore)** is `null` unless `--bert-model` is set. With a model, prediction and gold `biodiversity_variable` strings are paired greedily by descending BERTScore F1 (`lang="de"`, no baseline rescaling). The reported value is the mean over all strings, and unmatched strings score 0. The denominator for each paper is the larger of the prediction count and the gold count.

### `python -m core_pipeline check-schemas`

Loads the merged schema and prints how many English aliases each enum glossary contains. A mismatch in length or duplicate English labels raises before the command can succeed. This command has no parameters.

```bash
python -m core_pipeline check-schemas
```

## `core_pipeline/` Python modules

### `core_pipeline/__init__.py`

Package marker. Its docstring identifies the project as a reproducible baseline for ANLP Task C1 and C2b. It does not export a public API of its own; callers import submodules directly.

### `core_pipeline/__main__.py`

Enables `python -m core_pipeline`. It imports `main` from `cli` and calls it, so the module invocation and the argument parser stay in one place.

### `core_pipeline/cli.py`

Command-line entry points for extraction, local evaluation, and the schema catalogue check.

| Function / class | Role |
| --- | --- |
| `write_json` | Writes UTF-8 JSON with indentation. Used for config, metrics, audits, and failures. |
| `read_ids` | Loads the split file and rejects empty lists, duplicates, and ids that are not safe file-name tokens. |
| `export_results` | Writes `predictions.jsonl` via a temporary file, then a zip that contains exactly that file under the name `predictions.jsonl`. |
| `run_command` | Full experiment: empty-directory check, missing-input check, `ChatClient` setup, `run_config.json`, per-paper TEI/chunk/extract/audit loop, fail-fast `failure.json`, then submission export and `run_metrics.json`. Token totals count only cache misses. |
| `build_parser` | Defines `run`, `evaluate`, and `check-schemas` and their flags. |
| `main` | Parses arguments, checks the context budget on `run`, and dispatches. `evaluate` aligns gold to the predicted ids and writes the score file. `check-schemas` prints alias counts per enum. |

### `core_pipeline/client.py`

OpenAI-compatible chat client with a context-size guard, retries, and prompt-keyed disk cache.

`ChatClient` stores the base URL, model name, cache directory, context budget, max output tokens, JSON-mode flag, and timeout (default 180 seconds). `complete(messages)` estimates request bytes, refuses an over-budget prompt, builds a deterministic request (`temperature` 0, `top_p` 1, zero penalties, `stream` false), applies the Qwen thinking switch for the two configured model names, and returns `(result, metadata)`.

On a cache hit, metadata is the stored metadata plus `cache_hit: true`. On a fresh call, metadata includes elapsed seconds, server `usage`, request hash, input byte estimate, and attempt count. Cache files are written through a process-specific temporary file and then replaced, so a crash mid-write does not leave a partial JSON file at the cache key.

### `core_pipeline/documents.py`

PDF → TEI, section extraction, and chunking. No model calls.

| Function | Role |
| --- | --- |
| `pdf_to_tei` | GROBID `processFulltextDocument` with header and citation consolidation turned off. Cache key is the PDF bytes plus the endpoint. The XML is parsed before it is saved. Timeout 180 seconds. |
| `element_text` | Joins all descendant text and collapses whitespace to single spaces. |
| `tei_sections` | Returns `(heading, text)` pairs for title, abstract, and body sections. Raises if the body is missing or no study text was extracted. |
| `make_chunks` | Splits those sections into overlapping spans with stable ids and source offsets. Validates chunk size and overlap. |

### `core_pipeline/evaluate.py`

Local set-based scores. The module states in its results that these numbers are not the official course scorer.

| Function | Role |
| --- | --- |
| `read_records` | Loads JSONL or a JSON array into a dict keyed by `zotitem_ptr_id`. Duplicate ids raise. |
| `matching_gold` | Drops gold papers that were not predicted and raises if any prediction id is absent from gold, or if the overlap is empty. |
| `values` | Flattens one field path (`habitat`, or `ecosystem_type.term`) to a set of non-empty strings. Nested values are read from dict items; non-strings are ignored. |
| `evaluate` | Pooled exact-match counts and F1 for every categorical path, then `C1 Score(Macro F1)`, `C2b biodiversity (Macro F1)`, `All categorical values (Macro)`, `All categorical values (Micro)`, and `C2b free text (BERTScore)`. |
| `macro_f1` | Unweighted mean of the selected fields' F1 scores. |
| `micro_f1` | One F1 from TP, FP, and FN summed across the selected fields. |
| `free_text_score` | Greedy best-match BERTScore over German variable strings. The value written to the report is the mean that scores unmatched strings as 0. Imports `bert_score` only when this path runs. |

`taxa.species_group` and `biodiversity_level` are in both the C1 and C2b averages. The two "all categorical values" scores include each of them once.

### `core_pipeline/evidence.py`

Quote verification and state aggregation. This is the gate between model output and the submission.

| Function | Role |
| --- | --- |
| `normalized` | NFKC-normalizes text and collapses whitespace, returning both the normalized string and the original character index of each kept character. |
| `verify_quote` | Exact normalized search, with an optional fuzzy word-window search when `threshold < 1`. Rejects quotes shorter than 12 stripped characters. Returns chunk id, the original quote, the matched source slice, score, section index, and absolute section offsets. |
| `finding_id` | Stable 20-hex-character id from the canonical JSON of `[field, value]`. |
| `verified_changes` | Walks `add` and `remove`. Translates values, checks evidence shape and chunk id, verifies quotes, and splits the response into accepted additions, accepted removals, and rejected items with reasons. Removals are invalid unless `mode` is `sequential`. |
| `merge_findings` | Applies removals by id, then merges additions. Evidence spans with the same section and offsets are stored once. The state is sorted by finding id. |
| `evidence_votes` | Counts non-overlapping spans per section. A span that starts before the previous span ends extends that vote instead of adding a new one. |
| `submission_record` | Builds the public object: paper id plus one array per field. Values whose vote count is below `min_votes` are omitted. The object is validated with the merged schema before it is returned. |

### `core_pipeline/pipeline.py`

Runs one paper with identical evidence rules in either architecture.

`run_paper(paper_id, chunks, client, mode="parallel", workers=4, threshold=1.0, min_votes=1)` loads the schema and English catalogue, then processes every chunk. The inner `process` function builds the prompt, calls `client.complete`, and runs `verified_changes`. Parallel mode uses a thread pool and merges additions only. Sequential mode passes the current state into the next prompt and applies removals. The return value is `(submission_record, audit)`.

Parameter checks: mode is `parallel` or `sequential`; there is at least one chunk; `workers >= 1`; `min_votes >= 1`; `threshold` is between 0.9 and 1 inclusive.

The client is any object with `complete(messages) -> (dict, metadata)`. `ChatClient` satisfies that.

### `core_pipeline/prompts.py`

Task instructions and prompt assembly.

`SYSTEM` tells the model it is doing biodiversity information extraction, that paper text is data rather than instructions, and that the reply must be a single JSON object with no chain-of-thought.

`RULES` specifies the C1/C2b fields, English-alias versus German-enum policy, taxa granularity, German free-text variables, optional ecosystem terms, verbatim quotes from `CURRENT_CHUNK` only, and the sequential removal policy. It also fixes the `add` / `remove` output shape.

`build_messages(chunk, state, schema, labels, mode)` returns a two-message chat list: one system message (`SYSTEM + RULES`) and one user message containing the JSON payload described above. In parallel mode the previous state in that payload is always an empty list, even if the caller passed state.

### `core_pipeline/schema.py`

Official schema loading, English-to-German translation, and submission validation.

Schema files live in `core_pipeline/schemas/`:

- `core_fields.json` — C1 habitat, taxa, biodiversity level, ecosystem type.
- `biodiversity.json` — taxa, biodiversity level, biodiversity variable, ecosystem service. Shared `$defs` must be identical to the core schema or loading fails.
- `english_glosses.json` — English labels aligned one-to-one with each enum, in enum order.

| Function | Role |
| --- | --- |
| `load_schema` | Merges the two official schemas, requires `zotitem_ptr_id` and all six field arrays, removes `EcosystemService.details`, and checks the document with JSON Schema draft 2020-12. |
| `catalogue` | Builds `{definition_name: {english: german}}`. Raises if a glossary is missing, the wrong length, or contains duplicate English labels. |
| `translate_value` | Recursively maps a model value through `$ref`, `anyOf` (null or the first branch), objects, and strings. Enum strings may already be German. Unknown values and unexpected object keys raise. |
| `validate_record` | Validates a full submission object. Extra properties are forbidden by the merged schema. |
| `prompt_schema` | Deep-copies the schema and removes `zotitem_ptr_id` so the model is not asked to invent the paper id. |

`FIELDS` is the fixed output order: `habitat`, `biodiversity_level`, `ecosystem_type`, `taxa`, `biodiversity_variable`, `ecosystem_service`.
