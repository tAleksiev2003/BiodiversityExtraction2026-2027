# taskc_chunkSys_baseline

Evidence-grounded baseline for **ANLP 2026 Task C1 and C2b**. It reads scientific papers (PDF or GROBID TEI), splits them into overlapping text chunks, asks an OpenAI-compatible chat model to extract biodiversity findings, and keeps a finding only when its quote can be located in the source chunk. The result is a schema-valid submission (`predictions.jsonl` inside `predictions.zip`) plus a per-paper audit trail.

The same extraction rules, quote checks, and JSON Schema are used for five modes:

- **Parallel.** Each chunk is extracted independently. Findings are merged in document order. Removals are rejected, so one chunk cannot delete a finding produced by another chunk.
- **Sequential.** Chunks are processed in order. Each call sees the verified state so far and may add a finding or remove one when the current chunk quotes a contradiction.
- **rag.** Each paper is ranked only against its own pages. The top-k pages of each field group are extracted with that group's prompt, then merged.
- **grouped-all.** The same four group prompts as `rag`, sent every nonempty page, with no embedding call.
- **rag-all-fields.** The same retrieval as `rag`. The four top-k sets are unioned and sent once in an all-field prompt.


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
      submission_record()     precision filter, min-vote filter,
                              validate against the merged JSON Schema
            │
            ▼
      predictions.jsonl + predictions.zip
      <paper_id>.audit.json, run_config.json, run_metrics.json
```

That diagram is `--chunk-by chars`, the default. `--chunk-by pages` replaces `tei_sections` and `make_chunks` with `tei_pages` and `make_page_chunks`, then still calls `run_paper`. `rag`, `grouped-all`, and `rag-all-fields` force page coordinates and call `run_grouped` instead of `run_paper`. `rag` and `rag-all-fields` embed the paper's own pages first.

A run is fail-fast. The output directory must be empty. If any paper raises, the pipeline writes `failure.json` (paper id, error, ids already completed) and does **not** publish `predictions.jsonl` or `predictions.zip`. A submission is written only after every requested paper succeeds.

### Document preparation

`pdf_to_tei` posts each PDF to GROBID and caches the TEI by SHA-256 of the file bytes, the endpoint URL, and a marker for the request flags. A cached file is reused without calling GROBID again. The response is parsed as XML before it is stored. The character path sends the original full-text request. `--chunk-by pages`, and the three page modes, also send `segmentSentences=1` and `teiCoordinates` for `p`, `head`, `title`, and `s`. That marker is part of the cache key, so the coordinate TEI is stored beside the original TEI and a later `chars` run still reuses the old cache.

`tei_sections` keeps, in document order:

1. The title from `titleStmt/title`, when present.
2. The abstract, when present.
3. Body blocks. A `head` that is a direct child of a `div` starts a new section. Paragraphs, figures, tables, formulas, and list items are collected under the current heading. Nested copies of those elements are skipped so a table cell is not extracted twice. A short section whose heading or text is website chrome (for example a donate button, cookie notice, or newsletter prompt) is dropped. Title and abstract are kept.

A TEI file with no `body`, or with no usable text beyond the title, raises an error.

`make_chunks` splits each section into spans of at most `--chunk-chars` characters (default 5000), with `--overlap` characters (default 300) shared with the next span. The cut prefers a space in the second half of the window. Overlap must be smaller than half the chunk size, and the chunk size must be at least 100 characters. Every chunk records a stable id (`s<section>-c<index>`), section heading, section index, character offsets, and text. Offsets are into the section string, so later quote matching can recover a location in the original section. `--chunk-chars` and `--overlap` apply only when `--chunk-by` is `chars`.

### Page chunks

`--chunk-by pages` does not replace the character path. `tei_pages` reads coordinates in the TEI body. It uses `<s>` when GROBID segmented sentences, and `<p>` otherwise. Headings are placed on their page as well. A coordinate that lists more than one page is kept on the first page. Empty pages stay in the list, from page 1 through the highest page number in the file, so a chunk boundary follows the PDF page numbers. The title and abstract are written at the start of page 1. A TEI body with no coordinates raises `No page coordinates found in TEI body`.

`make_page_chunks` joins `--pages-per-chunk` consecutive pages (default 10, at least 1) into one chunk. The chunk id looks like `p001-p010`, and the section label is `Pages 1-10`. Quote offsets are positions in that joined text.

Page chunks do not pass through `tei_sections`, so the chrome filter does not run. Their section label is a page range. The species-rank check looks for a title, abstract, or methods heading, finds none, and is skipped. The title and abstract strings are still inside the first chunk.

`--input-format tei` can use this mode when the TEI already has GROBID `coords`.

If the byte estimate plus `--max-output` exceeds `--context-budget`, the client raises before the request. The error includes the input byte estimate, `max_output`, `context_budget`, and message count.

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

Accepted findings receive a 20-character id: the SHA-256 prefix of the canonical field and value. Merging unions evidence spans that differ in `(section_index, start, end)`. Overlapping spans inside one section count as **one vote**, so a finding repeated across overlapping chunks is not inflated. After the merge, a precision filter drops recommendation-only evidence, requires a German name to occur in its quote, keeps a species-rank name only when that genus occurs in the title, abstract, or methods, collapses a long biodiversity variable to its bare measure, and clears an ecosystem-service term whose distinctive words are absent from the quote. `--min-votes` (default 2) then drops findings whose non-overlapping evidence count is below the threshold. In `parallel` and `sequential`, `biodiversity_level` requires `max(--min-votes, 3)`. In `rag`, `grouped-all`, and `rag-all-fields` that field uses `--min-votes` itself, because one pass over a page does not repeat a span the way overlapping chunks can. The published record contains values only. Quotes and offsets stay in the audit. `final_state` is the post-filter state, and `precision_filter` lists removed evidence and edited values. Details are in `improve_01.md`.

### Parallel and sequential execution

Parallel mode submits every chunk to a `ThreadPoolExecutor` (`--workers`, default 4). Results are merged in **document order**, not completion order, and the previous state sent to the model is always empty. Removals are rejected.

Sequential mode walks chunks one by one. The prompt includes the verified state. Verified removals are applied before new additions are merged. Removal ids that are not in the current state are recorded as `unknown_removal_ids` and ignored. Each sequential audit entry also stores `state_after`.

Both modes return the same pair: a validated submission record, and an audit object `{chunks, calls, precision_filter, final_state}`. The progress line is printed before `run_paper`, so the id on screen is the paper being extracted.

### Truncated completions

`run_paper` catches `IncompleteResponse` and calls `split_chunk`. The cut prefers a blank line near the middle of the middle half of the text, then a newline, then a space, and otherwise uses the midpoint. The two pieces do not overlap. Whitespace between them is not given to either piece. Each piece keeps the parent `section_index`. `start` and `end` stay in the parent coordinate system, so a quote offset is still a position in the original chunk text and overlapping spans still count as one vote. The piece id appends `a` or `b`: `p001-p010` becomes `p001-p010a` and `p001-p010b`. The model must echo that id in evidence.

Both pieces see the previous state the parent chunk would have seen. In parallel mode that state is empty. In sequential mode it is the verified state from before this original chunk. Their additions and removals are merged together after both pieces return. A piece that is still truncated is split again, at most 8 times. A piece shorter than 200 characters, or one that cannot be made shorter, is stored with empty `add` and `remove`, `metadata.truncated` set, and `metadata.error` set to the exception text. The paper continues. The log prints `split <id> -> <left>, <right>` or `skip <id>: ...`.

Audit `chunks` stays the original segmentation. Audit `calls` lists only the leaves: successful piece calls, and a skipped slice when one remains truncated. `run_metrics.json` counts those leaves in `truncated_chunks`. They are omitted from the fresh prompt and completion token sums and from `token_usage_available_for_all_fresh_calls`. They are not counted as cache hits. A slice that stays truncated contributes no findings from that text.

### Client, cache, and budgets

`ChatClient` posts to `{base_url}/chat/completions`. Optional `LLM_API_KEY` is sent as a Bearer token and is never written into the cache or `run_config.json`.

Before the request, the client estimates size as the UTF-8 length of each message plus 64 bytes. If that estimate plus `--max-output` exceeds `--context-budget`, it raises and does not call the network. The error names the input byte estimate, `max_output`, `context_budget`, and message count. The estimate is a byte proxy, not a model tokenizer count.

Requests use `response_format: {type: json_object}` unless `--no-json-mode` is set. For the model names `qwen3.6-35b-a3b` and `qwen3.8-27b`, the request also sets `chat_template_kwargs.enable_thinking` to false.

A response is accepted when `finish_reason` is `stop`, absent, or `length`, the content parses as a JSON object, and both `add` and `remove` are arrays. Extra top-level keys are dropped. Any other `finish_reason` raises `ValueError`. When `finish_reason` is `length` and the body is already that JSON object, `parallel` and `sequential` accept it and cache it. When the body is cut off, or `add` and `remove` are not arrays, the client raises `IncompleteResponse` with the `max_tokens` value. That exception leaves the retry loop immediately and the body is not cached.

`rag`, `grouped-all`, and `rag-all-fields` construct the client with `strict_length=True`. Any `finish_reason` of `length` raises `IncompleteResponse` before `json.loads`. The body is not cached and is not retried. Those calls use a separate cache, `{cache}/llm_grouped_strict/`. The key adds `strict-length-v1`, so a grouped call does not reuse a baseline entry that accepted a parseable length cutoff. `parallel` and `sequential` keep `{cache}/llm/`.

Successful calls are cached under that directory as `<sha256>.json`. The key hashes the base URL and the full request body, so a change of model, prompt, chunk, or decoding settings misses the cache. Transient HTTP statuses 429, 500, 502, 503, and 504, transport errors, and parse errors are retried up to three times with sleeps of 1s, 2s, and 4s. `IncompleteResponse` is not retried. The last failure is raised.

GROBID TEI is cached separately under `{cache}/tei/`. Embedding vectors, used by `rag` and `rag-all-fields`, are cached under `{cache}/embeddings/`.

### Page RAG

Retrieval is paper-local. A paper is ranked only against its own pages. The index does not include other papers, the web, or gold labels. `parallel` and `sequential` still call `run_paper`.

| `--mode` | Text sent to the model | Extraction |
| --- | --- | --- |
| `parallel` | Character or page chunks | One all-field call per chunk |
| `sequential` | The same chunks, with the verified state | One all-field call per chunk, in order |
| `rag` | Top-k pages per field group | Four prompts, then one merge |
| `grouped-all` | Every nonempty page, once per field group | The same four prompts as `rag`, with no embedding call |
| `rag-all-fields` | The union of the four top-k page sets | One all-field prompt. Split only when the budget or the output length requires it |

`main` sets `chunk_by` to `pages` for the three new modes, so a PDF asks GROBID for the coordinate TEI described above. `--workers`, `--pages-per-chunk`, `--chunk-chars`, and `--overlap` do not decide which pages these modes send. `--top-k` is ignored by `grouped-all`. Before the first paper, `run` rejects `--top-k` below 1, `--min-votes` below 1, `--workers` below 1, and a `--quote-threshold` outside `0.9`–`1.0`.

`page_units` drops a page whose text is empty or whitespace. A kept page is one source. Its id is `{paper_id}:p006` for page 6, its section is `Page 6`, and quote offsets are indexes into that page string. Empty page numbers are listed on the audit as `empty_pages`. `total_pages` counts them. A paper with no nonempty page raises `No nonempty pages`.

The four queries are fixed English strings in `retrieval.GROUPS`. They are copied into `run_config.json` and each audit. They are not built from gold labels.

| Group | Fields |
| --- | --- |
| `habitat_ecosystem` | `habitat`, `ecosystem_type` |
| `taxa` | `taxa` |
| `biodiversity` | `biodiversity_level`, `biodiversity_variable` |
| `ecosystem_service` | `ecosystem_service` |

`rag` and `rag-all-fields` embed each query and each page window. The default query prefix is `Instruct: Retrieve passages from a scientific paper that answer the question.` followed by `Query: `. The document prefix is empty. A page longer than `--embedding-max-bytes` (default 6000, including the prefix, at least 128) is split into non-overlapping UTF-8 windows that cover the page. The cut keeps Unicode characters intact. The page score is the maximum cosine similarity of its windows to the query. Both vectors are L2-normalized. Equal scores are ordered by page number. Each group keeps `min(k, nonempty pages)` pages. The default `k` is 4. There is no score cutoff, no neighbor-page expansion, and no second ranker.

`rag-all-fields` unions those four sets, drops duplicate page ids, and sorts by page number. It does not fill the union back up to `4 * k`. A page selected by two groups is still one source and still one vote.

`EmbeddingClient` posts to `{embedding_base_url}/embeddings`. `--embedding-base-url` defaults to `--base-url`. The model default is `qwen3-embedding-4b`. The bearer token is `EMBEDDING_API_KEY`, or `LLM_API_KEY` when that variable is unset. The key is not written into the cache or `run_config.json`. The cache key hashes `embedding-v1`, the endpoint, the model, the role (`query` or `document`), and the full prefixed text. HTTP 429 and 5xx, transport errors, and vector-validation errors are retried three times. `--embedding-batch-size` must be at least 1. The default batch size is 16.

Grouped prompts keep `SYSTEM` and the part of `RULES` before `Parallel mode:`. `CURRENT_CHUNK` becomes `SOURCES`, and `CURRENT_ID` becomes `SOURCE_ID`. The schema in the prompt contains only that group's fields. `PREVIOUS_VERIFIED_STATE` is always `[]`. `remove` must be empty. A quote has to be contiguous inside one supplied source. The added instruction says a retrieved page is a candidate, not proof that a label is present.

Selected pages are packed into several calls when one prompt would exceed the context budget. A single page that still does not fit is passed through `split_chunk`. A page that cannot be split raises `Group prompt/source cannot fit context budget`. On `IncompleteResponse`, more than one source is cut into two lists at the midpoint, and one source is cut with `split_chunk`. The depth limit is 8. Past that limit, or when the remaining source cannot be split, the paper fails. Unlike `run_paper`, the text is not stored as an empty addition. `truncated_chunks` counts the parent call that was then split, not a page that was skipped.

Page sections are `Page N`, so the species-rank title/abstract/methods check is skipped here too. Title and abstract text remain at the start of page 1. After the merge, `apply_precision_filters` runs once, then `--min-votes` is applied with the page-mode biodiversity-level rule above. The audit `vote_filter` lists votes, the required count, and whether each finding was kept. `final_state` is the filtered state before that vote drop.

## Outputs

After a successful `run`:

| File | Contents |
| --- | --- |
| `predictions.jsonl` | One submission object per paper, in the order of the id list |
| `predictions.zip` | A zip whose only member is `predictions.jsonl` |
| `<paper_id>.audit.json` | Chunks, raw model responses, cache/usage metadata, rejected items, verified removals, `precision_filter`, and the final evidence-bearing state. Page-mode audits also store `retrieval`, `extraction_batches`, `embedding_windows`, `embedding_calls`, `vote_filter`, `empty_pages`, `total_pages`, and `provenance_notes` |
| `run_config.json` | CLI arguments, `field_groups`, Python version, platform, SHA-256 of each `core_pipeline/*.py` file, and SHA-256 of each schema file |
| `run_metrics.json` | Wall time, paper count, call count, cache hits, `truncated_chunks`, fresh prompt and completion tokens for cache misses, `logical_prompt_tokens` and `logical_completion_tokens` over every chat call (truncated calls contribute 0), `embedding_http_calls`, `embedding_cache_hits`, `embedding_fresh_tokens`, and SHA-256 of each source file. Embedding counts are 0 when the mode does not embed |

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

A remote chat endpoint that requires authentication reads the key from `LLM_API_KEY` and sends it as `Authorization: Bearer <key>`. A local server that does not check a key, such as Ollama, can be used without setting it. The embeddings endpoint uses `EMBEDDING_API_KEY` when that variable is set, and otherwise the same `LLM_API_KEY`. Neither value is written into `run_config.json` or the response cache. Set the chat key in the same shell before `run`:

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

Extract one split into `predictions.jsonl` and `predictions.zip`. The papers that enter the pipeline are the ids listed in `--ids`, not every file under `--input`. `--input` is only the directory where those files are stored. For each id the runner reads exactly `{input}/{id}.pdf` or `{input}/{id}.tei.xml`. Any other file in that directory is ignored. Ids must be unique and match `[A-Za-z0-9_-]+`. `--output` must be an empty directory. `--context-budget` must be greater than `--max-output`, and `--max-output` must be at least 1. `--top-k`, `--min-votes`, and `--workers` must be at least 1, and `--quote-threshold` must be from 0.9 to 1.0. `--pages-per-chunk` must be at least 1 when `--chunk-by` is `pages`.

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
| `--mode` | yes |  | `parallel`, `sequential`, `rag`, `grouped-all`, or `rag-all-fields`. The last three force `--chunk-by pages` |
| `--model` | yes |  | Exact model name served by the chat endpoint |
| `--input-format` | no | `pdf` | `pdf` calls GROBID; `tei` reads `{id}.tei.xml` directly |
| `--base-url` | no | `http://localhost:11434/v1` | OpenAI-compatible chat endpoint. Also the default embeddings endpoint |
| `--grobid-url` | no | `http://localhost:8070` | GROBID service. Used only when `--input-format` is `pdf` |
| `--cache` | no | `.cache/taskc` | Disk cache for TEI (`tei/`), baseline chat (`llm/`), grouped chat (`llm_grouped_strict/`), and embeddings (`embeddings/`) |
| `--workers` | no | `4` | Parallel chunk threads. Must be at least 1. Ignored when choosing pages for the three page modes. With `sequential`, chunks are still processed one by one |
| `--chunk-by` | no | `chars` | `chars` splits section text. `pages` groups physical PDF pages. Forced to `pages` for `rag`, `grouped-all`, and `rag-all-fields` |
| `--chunk-chars` | no | `5000` | Maximum characters per chunk when `--chunk-by` is `chars`. Must be at least 100 |
| `--overlap` | no | `300` | Characters repeated between adjacent chunks when `--chunk-by` is `chars`. Must be at least 0 and less than half of `--chunk-chars` |
| `--pages-per-chunk` | no | `10` | Physical pages per chunk when `--chunk-by` is `pages`. Must be at least 1. Does not choose pages for the three page modes |
| `--top-k` | no | `4` | Pages kept per field group for `rag` and `rag-all-fields`. Must be at least 1. Ignored by `grouped-all` |
| `--embedding-model` | no | `qwen3-embedding-4b` | Embeddings model for `rag` and `rag-all-fields` |
| `--embedding-base-url` | no | `--base-url` | Embeddings endpoint. Defaults to the chat base URL |
| `--embedding-max-bytes` | no | `6000` | UTF-8 byte guard for one embedding input, including the prefix. Must be at least 128 |
| `--embedding-batch-size` | no | `16` | Texts per embeddings request. Must be at least 1 |
| `--query-prefix` | no | the instruction prefix in Page RAG | Text placed before each query. The default ends with `Query: ` |
| `--document-prefix` | no | empty | Text placed before each page window |
| `--quote-threshold` | no | `1.0` | Minimum normalized quote match, from 0.9 to 1. `1.0` requires an exact match |
| `--min-votes` | no | `2` | Minimum non-overlapping evidence spans. Must be at least 1. `parallel` and `sequential` still require at least 3 for `biodiversity_level`. `rag`, `grouped-all`, and `rag-all-fields` use this number for that field too |
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
| `run_command` | Full experiment: empty-directory check, missing-input check, `ChatClient` setup, optional `EmbeddingClient`, `run_config.json`, per-paper TEI then `run_paper` or `run_grouped`, fail-fast `failure.json`, then submission export and `run_metrics.json`. Fresh token totals skip cache hits and truncated calls. Logical totals include cache hits. |
| `build_parser` | Defines `run`, `evaluate`, and `check-schemas` and their flags, including page chunks and embeddings. |
| `main` | Parses arguments, checks budgets and retrieval settings on `run`, forces `chunk_by=pages` for the three page modes, and dispatches. `evaluate` aligns gold to the predicted ids and writes the score file. `check-schemas` prints alias counts per enum. |

### `core_pipeline/client.py`

OpenAI-compatible chat client with a context-size guard, retries, and prompt-keyed disk cache.

`IncompleteResponse` is the length-cutoff exception. It is not retried.

`ChatClient` stores the base URL, model name, cache directory, context budget, max output tokens, JSON-mode flag, `strict_length`, and timeout (default 180 seconds). `complete(messages)` estimates request bytes, refuses an over-budget prompt, builds a deterministic request (`temperature` 0, `top_p` 1, zero penalties, `stream` false), applies the Qwen thinking switch for the two configured model names, and returns `(result, metadata)`. With `strict_length`, a `length` finish raises `IncompleteResponse` before the body is parsed. The grouped cache key includes `strict-length-v1`.

On a cache hit, metadata is the stored metadata plus `cache_hit: true`. On a fresh call, metadata includes elapsed seconds, server `usage`, request hash, input byte estimate, and attempt count. Cache files are written through a process-specific temporary file and then replaced, so a crash mid-write does not leave a partial JSON file at the cache key.

### `core_pipeline/documents.py`

PDF → TEI, section extraction, and chunking. No model calls.

| Function | Role |
| --- | --- |
| `pdf_to_tei` | GROBID `processFulltextDocument` with header and citation consolidation turned off. With `coordinates=True`, also requests sentence segmentation and coordinates. Cache key is the PDF bytes, the endpoint, and those flags. The XML is parsed before it is saved. Timeout 180 seconds. |
| `element_text` | Joins all descendant text and collapses whitespace to single spaces. |
| `tei_sections` | Returns `(heading, text)` pairs for title, abstract, and body sections. Drops short website-chrome sections. Raises if the body is missing or no study text was extracted. |
| `tei_pages` | Reads coordinate-bearing body text onto physical pages, from page 1 through the highest page number. Prepends title and abstract to page 1. Raises when the body has no coordinates. |
| `make_chunks` | Splits sections into overlapping character spans with stable ids and source offsets. Validates chunk size and overlap. |
| `make_page_chunks` | Joins consecutive physical pages into chunks such as `p001-p010`. |
| `split_chunk` | Halves one chunk on a blank line, newline, or space near the middle. Child offsets stay in the parent. Returns an empty list when the text is shorter than 200 characters or cannot shrink. |

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
| `apply_precision_filters` | After the merge, drops recommendation-only evidence and tightens German names, species-rank names, biodiversity variables, and ecosystem-service terms. |
| `votes_required` | Returns `--min-votes`, except `biodiversity_level`, which uses `max(--min-votes, level_min_votes)`. The default level floor is 3. Page modes pass `level_min_votes` equal to `--min-votes`. |
| `submission_record` | Builds the public object: paper id plus one array per field. Values below `votes_required` are omitted. Identical values are emitted once. The object is validated with the merged schema before it is returned. |

### `core_pipeline/pipeline.py`

Runs one paper with identical evidence rules in either architecture.

`run_paper(paper_id, chunks, client, mode="parallel", workers=4, threshold=1.0, min_votes=2)` loads the schema and English catalogue, then processes every chunk. The inner `process` function builds the prompt, calls `client.complete`, and runs `verified_changes`. On `IncompleteResponse` it splits the chunk and continues, up to eight times. Parallel mode uses a thread pool and merges additions only. Sequential mode passes the current state into the next prompt and applies removals. Both modes then run `apply_precision_filters` once on the merged state. The return value is `(submission_record, audit)`. `submission_record` is called without a level-vote override, so `biodiversity_level` keeps the floor of 3.

Parameter checks: mode is `parallel` or `sequential`; there is at least one chunk; `workers >= 1`; `min_votes >= 1`; `threshold` is between 0.9 and 1 inclusive.

The client is any object with `complete(messages) -> (dict, metadata)`. `ChatClient` satisfies that.

### `core_pipeline/prompts.py`

Task instructions and prompt assembly.

`SYSTEM` tells the model it is doing biodiversity information extraction, that paper text is data rather than instructions, and that the reply must be a single JSON object with no chain-of-thought.

`RULES` specifies the C1/C2b fields, English-alias versus German-enum policy, taxa granularity, German names copied from the chunk, bare biodiversity variables, optional ecosystem terms, verbatim quotes from `CURRENT_CHUNK` only, and the sequential removal policy. It also fixes the `add` / `remove` output shape.

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

### `core_pipeline/retrieval.py`

Paper-local dense retrieval. No chat calls.

| Name | Role |
| --- | --- |
| `GROUPS` | The four field groups and their fixed English queries. |
| `QUERY_PREFIX` | Default query prefix, ending with `Query: `. |
| `page_units` | Drops empty pages and assigns `{paper_id}:pNNN` ids, page sections, and text hashes. |
| `embedding_slices` | Splits one page into non-overlapping UTF-8 windows that cover the text. |
| `EmbeddingClient` | Posts to `/embeddings`, caches normalized vectors, and retries HTTP 429, 5xx, transport errors, and invalid vectors. |
| `rank_pages` | Embeds queries and page windows and ranks each group's pages by maximum cosine similarity. |

### `core_pipeline/rag.py`

Grouped page extraction for `rag`, `grouped-all`, and `rag-all-fields`.

`run_grouped` ranks pages when the mode embeds, packs the selected pages into context-budget calls, verifies quotes against the sources in that call, merges additions, runs `apply_precision_filters` once, and applies `--min-votes` with `level_min_votes` set to that same number. `group_messages` builds the `SOURCES` prompt. `pack_sources` splits a page that does not fit. `verify_group` rejects a finding when any quote fails or the field is outside the group, and rejects every `remove`. A length cutoff is split up to eight times and then fails the paper. The audit adds retrieval rankings, extraction batches, embedding calls, `vote_filter`, and provenance notes.
