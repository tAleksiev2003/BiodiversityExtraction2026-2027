# improve_01

Changes applied to `taskc_chunkSys_baseline` after the 15-paper parallel run in `runs/pdfs_15_parallel_deepseek4.1` (model `deepseek-v4.1-flash-1017`). That run scored C1 macro F1 0.455 and C2b macro F1 0.232. Most errors were extra labels, not missing ones. These edits raise the evidence bar and stop a later chunk from being unable to retract a bad finding.

The 15-paper run was not repeated. Prompt edits take effect on the next extraction. The post-merge filter can be applied to the saved audits without a new model call; that rescore is reported at the end.

## Files

| File | Change |
| --- | --- |
| `core_pipeline/cli.py` | Default `--min-votes` is 2. Optional `--chunk-by pages` groups physical pages. Prints the paper before extraction. `run_metrics.json` counts truncated slices. |
| `core_pipeline/evidence.py` | Vote rule for `biodiversity_level`, and `apply_precision_filters`. |
| `core_pipeline/pipeline.py` | Runs the precision filter once after parallel or sequential merge. Splits a chunk whose completion hits `max_tokens`. |
| `core_pipeline/prompts.py` | Extraction rules for scope, names, variables, and service terms. |
| `core_pipeline/documents.py` | Drops short website-chrome sections from GROBID text. Adds page extraction beside the character chunks. `split_chunk` halves one chunk. |
| `core_pipeline/client.py` | Context-guard errors include the byte estimate, output limit, budget, and message count. A truncated completion raises `IncompleteResponse` and is not retried. |
| `README.md` | Documents the new default, the filter, and the chrome rule. |

Section 8 ports page RAG from `ANLP_TaskC_Page_RAG 2`. Its file table covers `retrieval.py`, `rag.py`, and the later edits to `cli.py` and `client.py`. Sections 1–7 still describe `parallel` and `sequential`.

`--quote-threshold` stays 1.0. In the 15-paper run, 3 of 934 proposed additions failed the quote check. The quotes were already verbatim. `--workers` stays 4. The scored run used 1 worker; more workers change wall time only, because parallel merge is already in document order.

## 1. Vote threshold

`--min-votes` defaults to 2. A vote is one non-overlapping evidence span in the paper. The same sentence in two overlapping chunks is still one vote. A finding mentioned in the abstract and again in the results is two votes.

`biodiversity_level` requires `max(--min-votes, 3)`. On the saved audit, raising only this field from 2 votes to 3 kept recall at 0.80 and lifted its F1 from 0.421 to 0.533, because repeated false `Artenvielfalt` labels dropped off. Scientific names often appear once, so the global threshold stays at 2. Passing `--min-votes 4` or higher also raises the biodiversity-level threshold to that number.

`submission_record` emits each field value once when several findings collapse to the same value.

## 2. Keep-or-drop pass after the merge

Parallel mode unions chunks. A later chunk cannot delete a finding. `apply_precision_filters` runs after the merge in both modes. Sequential prompts still see the unfiltered state while chunks are processed; `state_after` in each call is that intermediate state. The submitted record and `final_state` are filtered. Edits and deletions are listed in the audit under `precision_filter`.

This is a deterministic pass, not a second model call. It encodes the same decisions: drop recommendations, drop checklist species that the title, abstract, and methods do not name, and do not keep a label merely because some chunk quoted it.

### Recommendation evidence

A span is removed when its quote matches `zu untersuchen`, `sollte`, `sollten`, `empfehl…`, `further research`, `future research`, `future work`, `remains to be investigated/studied/determined`, or `to be investigated`. If every span of a finding matches, the finding is deleted. On `25ABQZIH`, insects, amphibians, and birds came from a sentence about impacts that still need to be studied. That evidence no longer counts.

### Species-rank names

A scientific name is species-rank when it is a binomial, a trinomial, or an explicit `sp.` / `spp.` name. A single higher taxon such as `Copepoda` or `Mammalia` is not species-rank.

When the paper has a title, abstract, or methods section, a species-rank name is kept only if its genus token occurs in that text. Otherwise `scientific_name` is set to null and the species group is kept. If the paper has none of those sections, this check is skipped.

This is the checklist and review rule. `4JF8F3WL` is a mammal red list: the saved extraction listed 16 species, and the gold label is the group. `2NJD9Y9T` is a meta-analysis: species such as `Ecklonia radiata` were taken from the papers under review. Those species-rank names are removed when the genus is absent from the title, abstract, and methods. The prompt now says the same thing: on a checklist, red list, or review, emit the studied group, and emit a species only when the methods name it as a focal taxon.

### German names

`german_name` must occur in a surviving quote. The comparison is by tokens, not by a raw substring, so `Forelle` does not silently match inside an unrelated word.

- If the quote contains the name, the surface form from the quote is stored.
- A single preceding modifier is included when it is a name adjective such as `Gemeine`, `Echtes`, or `Atlantischer`. `Kabeljau` in `Atlantischer Kabeljau` becomes `Atlantischer Kabeljau`. Neighboring list items are not glued on: `Fischotter Biber Wildkatze` stays `Biber` when that is the predicted name.
- A one-word shortening is expanded to a longer token that ends with it. `Forelle` in a quote that says `Seeforelle` becomes `Seeforelle`.
- If the quote does not contain the name, `german_name` is null. `Dinoflagellaten` is dropped when the quote only says `Dinophyceae`.

The official taxa description still asks the model to translate names. `RULES` in `prompts.py` overrides that sentence: copy a German name from the current chunk, and omit it when the chunk has none. Do not shorten and do not translate from memory.

### Biodiversity variables

`biodiversity_variable` is reduced to the bare measure.

- A phrase that names an index (`index`, `evenness`, `Simpson`, `Jaccard`, `Shannon`, `Pielou`, `turnover`, `nestedness`) is kept as printed.
- A hyphenated token whose last piece is a known measure becomes that piece: `Phytoplankton-Biomasse` becomes `Biomasse`.
- A phrase of three or more words whose first word is a known measure (`Abundanz`, `Biomasse`, `Artenzahl`, `Artenreichtum`, `Dichte`, `Bedeckung`, `Produktivität`, and the same list with `ss` spellings) becomes that first word.
- Any other phrase longer than four words is dropped.
- One- and two-word phrases that are not index names are kept.

The saved run had 161 predicted variables and 7 gold variables, with zero exact matches. Five of the seven gold strings, including `Abundanz` and `Biomasse`, already sat inside a longer prediction. BERTScore stays near 0.035 while the unmatched strings score 0 and the denominator is the larger count. The filter cuts the saved set from 161 strings to 39. BERTScore was not recomputed.

### Ecosystem-service terms

A service term is kept only when at least two of its distinctive words (letters only, length at least 6, stopwords removed) occur in the quotes. Otherwise `term` is set to null and the category remains. Cleared-term findings are not merged with each other, so several one-span mistakes cannot pool into one vote count.

The saved run predicted 26 service terms and matched none. The one gold term is on `49Z7RPDP`. Dropping every service object would also drop that paper's correct category, and the field would still score 0. The filter clears unsupported terms and leaves the category to the vote rule. On the rescore, service-term false positives fall from 26 to 0, the gold term is still missed, and service-category F1 moves from 0.125 to 0.200.

## 3. GROBID chrome

`tei_sections` drops a body section when its heading matches website chrome, or when the section is shorter than 400 characters and the text matches. The patterns are donate prompts (`jetzt spenden`), cookie notices, newsletters, impressum, privacy links, share links, and sign-in prompts. Title and abstract are not dropped. Cached TEI is still reused; the filter runs when sections are built, so a new GROBID call is not required for this rule.

`25RIYD2C` had a section whose heading was a donate button. That section is now skipped.

`3439JHQD` is not fixed by this rule. GROBID returned a single section, and the text does not contain `Wald`, `Boden`, `Stadt`, or `Gewässer`. Five of the six habitat misses in the 15-paper run are that paper. The missing pages are not in the TEI, so the pipeline cannot recover them. That PDF needs a new extraction or a hand-checked TEI.

## 4. What the next run does

Use a new empty output directory. The prompt text changed, so existing LLM cache entries miss and the endpoint is called again. GROBID TEI cache entries are still reused. Omit `--min-votes` to get the new default of 2.

```bash
python -m core_pipeline run \
  --input data/pdfs_dev_100 \
  --ids data/ids_15.txt \
  --output runs/pdfs_15_parallel_deepseek4.1_improve01 \
  --mode parallel \
  --model deepseek-v4.1-flash-1017
```

## 5. Rescore of the saved audit

The numbers below apply `apply_precision_filters` and the new vote thresholds to `final_state` in the existing 15 audit files. The model strings are unchanged. This is the local scorer in `core_pipeline/evaluate.py`, not the course scorer. Gold `scientific_name` values of `"-"` still count as required strings.

| Score | Saved run (`--min-votes 1`, no filter) | This filter on the saved audit |
| --- | ---: | ---: |
| C1 macro F1 | 0.455 | 0.524 |
| C2b macro F1 | 0.232 | 0.301 |
| All categorical, macro | 0.328 | 0.380 |
| All categorical, micro | 0.328 | 0.426 |

| Field | Saved F1 | Filtered F1 | Filtered TP / FP / FN |
| --- | ---: | ---: | --- |
| habitat | 0.653 | 0.667 | 15 / 8 / 7 |
| biodiversity level | 0.296 | 0.533 | 4 / 6 / 1 |
| ecosystem category | 0.522 | 0.545 | 6 / 10 / 0 |
| ecosystem term | 0.387 | 0.400 | 5 / 13 / 2 |
| species group | 0.417 | 0.474 | 9 / 17 / 3 |
| scientific name | 0.333 | 0.449 | 11 / 10 / 17 |
| German name | 0.219 | 0.151 | 4 / 17 / 28 |
| service category | 0.125 | 0.200 | 1 / 8 / 0 |
| service term | 0.000 | 0.000 | 0 / 0 / 1 |

German-name F1 falls on this rescore. The saved model translated and shortened names, and some of those translations happened to match gold even though the German string is not in the chunk. The filter removes them. A new extraction follows the updated prompt, but a German gold name that never appears in an English paper still cannot be copied from the quote.

Filter actions on the 15 audits:

| Action | Count |
| --- | ---: |
| German name absent from quote | 57 |
| Species name absent from title, abstract, and methods | 46 |
| Variable collapsed to the bare measure | 31 |
| Variable dropped, no bare measure | 23 |
| Service term cleared | 25 |
| German name copied or expanded from the quote | 14 |
| Finding dropped, recommendation-only evidence | 5 |
| Recommendation span removed, other evidence kept | 1 |

## 6. Page chunks

`--chunk-by` defaults to `chars`. That path is the original GROBID full-text request, the original TEI cache, `tei_sections`, and `make_chunks`. `--chunk-chars` (default 5000) and `--overlap` (default 300) apply only on this path. The chrome rule in section 3 still runs here.

`--chunk-by pages` is an added mode. It does not replace the character path. For a PDF it calls GROBID with `segmentSentences=1` and `teiCoordinates` for `p`, `head`, `title`, and `s`. The cache key includes those flags, so the coordinate TEI is stored beside the original TEI and a later `chars` run still reuses the old cache. `--pages-per-chunk` defaults to 10 and must be at least 1.

`tei_pages` reads coordinates in the TEI body. It uses `<s>` when GROBID segmented sentences, and `<p>` otherwise. Headings are placed on their page as well. A coordinate that lists more than one page is kept on the first page. Empty pages stay in the list, from page 1 through the highest page number in the file, so a chunk boundary follows the PDF page numbers. The title and abstract are written at the start of page 1. `make_page_chunks` joins that many consecutive pages into one chunk. The chunk id looks like `p001-p010`, and the section label is `Pages 1-10`. Quote offsets are positions in that joined text.

Page chunks do not pass through `tei_sections`, so the chrome filter does not run. Their section label is a page range. The species-rank check in section 2 looks for a title, abstract, or methods heading, finds none, and is skipped. The title and abstract strings are still inside the first chunk.

`--input-format tei` can use this mode when the TEI already has GROBID `coords`. A TEI without coordinates raises `No page coordinates found in TEI body`.

A context-guard failure now reports the input byte estimate, `max_output`, `context_budget`, and message count, together with the existing instruction to raise the budget or shorten the chunks.

```bash
python -m core_pipeline run \
  --input data/pdfs_dev_100 \
  --ids data/ids_15.txt \
  --output runs/pdfs_15_parallel_deepseek4.1_pages \
  --mode parallel \
  --model deepseek-v4.1-flash-1017 \
  --chunk-by pages \
  --pages-per-chunk 10
```

## 7. Truncated completions

The 99-id page run in `runs/pdfs_99_parallel_deepseek4.1_pages_a` stopped on `46RX4AEN`. That PDF is about 314 pages. With `--pages-per-chunk 10` and `--max-output 16384`, one chunk returned `finish_reason: length`. The client retried the same request three times, then aborted the batch. Papers 1–20 had audits on disk. `predictions.jsonl` is written only after every id succeeds, so that file was not produced. The output directory was no longer empty, so the same `--output` could not be reused.

`ChatClient.complete` now separates a length cutoff from a transport error. When `finish_reason` is `length` and the body is already a JSON object with `add` and `remove` arrays, the object is accepted and cached. When the body is cut off, or those two keys are not arrays, the client raises `IncompleteResponse` with the `max_tokens` value. That exception leaves the retry loop immediately. The truncated body is not cached. Any other `finish_reason` still raises `ValueError` and still uses the three attempts for HTTP 429 and 5xx, transport errors, and parse errors.

`run_paper` catches `IncompleteResponse` and calls `split_chunk`. The cut prefers a blank line near the middle, then a newline, then a space, and otherwise uses the midpoint. The two pieces do not overlap. Whitespace between them is not given to either piece. Each piece keeps the parent `section_index`. `start` and `end` stay in the parent coordinate system, so a quote offset is still a position in the original chunk text and the vote rule in section 1 still collapses overlapping spans. The piece id appends `a` or `b`: `p001-p010` becomes `p001-p010a` and `p001-p010b`. The model must echo that id in evidence, the same way it echoes any other chunk id.

Both pieces see the previous state the parent chunk would have seen. In parallel mode that state is empty. In sequential mode it is the verified state from before this original chunk. Their additions and removals are merged together after both pieces return. A piece that is still truncated is split again, at most 8 times. A piece shorter than 200 characters, or one that cannot be made shorter, is stored with empty `add` and `remove`, `metadata.truncated` set, and `metadata.error` set to the exception text. The paper continues. The log prints `split <id> -> <left>, <right>` or `skip <id>: ...`.

Audit `chunks` stays the original segmentation. Audit `calls` lists only the leaves: successful piece calls, and a skipped slice when one remains truncated. Parallel merge is still in document order.

`run_metrics.json` adds `truncated_chunks`. Those calls are omitted from the fresh prompt and completion token sums, and from `token_usage_available_for_all_fresh_calls`. They are not counted as cache hits. The progress line is printed before `run_paper`, so the id on screen is the paper being extracted.

Chunks that already completed keep the same request body, so their LLM cache entries still hit. GROBID page TEI is unchanged. A rerun needs a new empty output directory. `--max-output` and `--pages-per-chunk` are left as given. A slice that stays truncated contributes no findings from that text.

## 8. Page RAG

`ANLP_TaskC_Page_RAG 2` (package 0.2.0) adds three modes on top of sections 1–7. Retrieval is paper-local. A paper is ranked only against its own pages. The index does not include other papers, the web, or gold labels. `parallel` and `sequential` still call `run_paper`. This port was not scored on the course dev set.

### Files

| File | Change |
| --- | --- |
| `core_pipeline/retrieval.py` | Field-group queries, page units, UTF-8 embedding windows, the embeddings client, and per-page ranking. |
| `core_pipeline/rag.py` | Grouped prompts, context packing, multi-source quote checks, and the three new modes. |
| `core_pipeline/cli.py` | `--mode` accepts `rag`, `grouped-all`, and `rag-all-fields`. Embedding flags, coordinate TEI, and the extra metrics. |
| `core_pipeline/client.py` | `strict_length` rejects a `finish_reason: length` completion before it can be cached. |
| `core_pipeline/documents.py` | Docstrings on the page-text helper and on `split_chunk`'s child builder. Behavior is unchanged. |

`pipeline.py`, `evidence.py`, `prompts.py`, `schema.py`, and `evaluate.py` are unchanged. The new modes call those modules.

### Modes

| `--mode` | Text sent to the model | Extraction |
| --- | --- | --- |
| `parallel` | Existing chunks, character or page | One all-field call per chunk. Unchanged. |
| `sequential` | Existing chunks, with the verified state | Unchanged. |
| `rag` | Top-k pages per field group | Four separate prompts, then one merge. |
| `grouped-all` | Every nonempty page, once per field group | The same four prompts as `rag`, with no embedding call. |
| `rag-all-fields` | The union of the four top-k page sets | One all-field prompt. Split only when the budget or the output length requires it. |

`main` sets `chunk_by` to `pages` for the three new modes, so a PDF asks GROBID for sentence segmentation and coordinates on `p`, `head`, `title`, and `s`. The coordinate TEI cache from section 6 is reused. A TEI input must already contain coordinates, or `tei_pages` raises `No page coordinates found in TEI body`. `--workers`, `--pages-per-chunk`, `--chunk-chars`, and `--overlap` do not decide which pages these modes send. `--top-k` is ignored by `grouped-all`.

For every mode, `run` now rejects `--top-k` below 1, `--min-votes` below 1, `--workers` below 1, and a `--quote-threshold` outside `0.9`–`1.0` before the first paper. `run_config.json` always stores `field_groups` and a SHA-256 of each `core_pipeline/*.py` file, including on `parallel` and `sequential`.

### Pages and queries

`tei_pages` is the section 6 reader. `page_units` drops a page whose text is empty or whitespace. A kept page is one source:

- `id` is `{paper_id}:p006` for page 6.
- `section` is `Page 6`, `section_index` is 5, `page_number` is 6.
- `start` is 0 and `end` is the length of that page string. Quote offsets are indexes into this string.
- `text_sha256` is the SHA-256 of the page text.

Empty page numbers are still listed on the audit as `empty_pages`. `total_pages` counts them. A paper with no nonempty page raises `No nonempty pages`.

The four queries live in `retrieval.GROUPS` and are copied into `run_config.json` and each audit. They are fixed English strings, not built from gold labels.

| Group | Fields |
| --- | --- |
| `habitat_ecosystem` | `habitat`, `ecosystem_type` |
| `taxa` | `taxa` |
| `biodiversity` | `biodiversity_level`, `biodiversity_variable` |
| `ecosystem_service` | `ecosystem_service` |

`rag` and `rag-all-fields` embed each query and each page window. The default query prefix is `Instruct: Retrieve passages from a scientific paper that answer the question.` followed by `Query: `. The document prefix is empty. A page longer than the byte guard is split into non-overlapping UTF-8 windows that cover the page. The default guard is 6000 bytes, including the prefix. The cut keeps Unicode characters intact and does not look for a sentence boundary. The page score is the maximum cosine similarity of its windows to the query. Both vectors are L2-normalized. Equal scores are ordered by page number. Each group keeps `min(k, nonempty pages)` pages. The default `k` is 4. There is no score cutoff, no neighbor-page expansion, and no second ranker.

`rag-all-fields` unions those four sets, drops duplicate page ids, and sorts by page number. It does not fill the union back up to `4 * k`, and it does not cut the union with another top-k. A page selected by two groups is still one source and still one vote.

### Embeddings client

`EmbeddingClient` posts to `{embedding_base_url}/embeddings`. `--embedding-base-url` defaults to `--base-url`. The model default is `qwen3-embedding-4b`. The bearer token is `EMBEDDING_API_KEY`, or `LLM_API_KEY` when that variable is unset. The key is not written into the cache or `run_config.json`.

Vectors are cached under `{cache}/embeddings/`. The key hashes `embedding-v1`, the endpoint, the model, the role (`query` or `document`), and the full prefixed text. A cached vector is checked for a finite, nonzero norm and then normalized again. A response whose indexes do not match the batch, or whose vectors differ in dimension, is an error. HTTP 429 and 5xx, transport errors, and those validation errors are retried three times. `--embedding-max-bytes` must be at least 128. `--embedding-batch-size` must be at least 1. The default batch size is 16.

### Grouped extraction

`group_messages` keeps `SYSTEM` and the part of `RULES` before `Parallel mode:`. `CURRENT_CHUNK` becomes `SOURCES`, and `CURRENT_ID` becomes `SOURCE_ID`. The schema in the prompt contains only that group's fields. `PREVIOUS_VERIFIED_STATE` is always `[]`. `remove` must be empty. A quote has to be contiguous inside one supplied source. The added instruction says a retrieved page is a candidate, not proof that a label is present.

`pack_sources` uses the same byte estimate as the chat client: UTF-8 length of each message plus 64, plus `max_output`, compared with `context_budget`. Selected pages are not dropped to make a prompt fit. They are packed into several calls. A single page that still does not fit is passed through `split_chunk`. A page that cannot be split raises `Group prompt/source cannot fit context budget`. Child ids append `a` or `b`, and `start` / `end` stay in the parent page, as in section 7.

`verify_group` accepts a finding only when every evidence item is on a source in that call and passes `verified_changes`. One bad quote rejects the whole finding. A field outside the group is rejected. Every `remove` item is rejected with `Removals forbidden in grouped extraction`. A verified span is annotated with `paper_id` and `page_number`.

Successful batches are merged in call order. The model does not see findings from an earlier batch. After every group or the all-field union has been merged, `apply_precision_filters` runs once. Page sections are `Page N`, so the species-rank check in section 2 still finds no title, abstract, or methods heading and is skipped. Title and abstract text remain at the start of page 1. `submission_record` then applies `--min-votes` (default 2). In these three modes `biodiversity_level` uses that same number: `run_grouped` passes `level_min_votes=min_votes`, so the extra floor of 3 does not apply. `parallel` and `sequential` still call `submission_record` without that argument, and `votes_required` keeps `max(--min-votes, 3)` for them. The audit `vote_filter` lists votes, the required count, and whether each finding was kept. `required` for a level finding is 2 when `--min-votes` is left at 2. A level mentioned only once is still dropped. `final_state` is the filtered state before that vote drop. The submitted values are in `predictions.jsonl`.

### Strict length cutoff

The three new modes construct `ChatClient` with `strict_length=True` and cache under `{cache}/llm_grouped_strict`. The cache key adds `strict-length-v1`, so a grouped call does not reuse a baseline entry that accepted a parseable length cutoff. When `finish_reason` is `length`, the client raises `IncompleteResponse` before `json.loads`, does not cache the body, and does not retry.

`run_grouped` records that call with `metadata.truncated` and `metadata.cache_hit` false, then splits. More than one source is cut into two lists at the midpoint. One source is cut with `split_chunk`. The depth limit is 8. Past that limit, or when the remaining source cannot be split, the paper fails. The text is not stored as an empty addition the way a short truncated slice is in section 7. `parallel` and `sequential` still use `{cache}/llm` and still accept a complete `{add, remove}` object when `finish_reason` is `length`.

### Metrics

`run_metrics.json` adds these fields on every mode. Embedding counts are 0 when the mode does not embed.

| Field | Meaning |
| --- | --- |
| `logical_prompt_tokens` | Prompt tokens over every chat call, including cache hits. Truncated calls contribute 0. |
| `logical_completion_tokens` | Completion tokens on the same set. |
| `embedding_http_calls` | Successful embedding HTTP requests. Failed retries are not included. |
| `embedding_cache_hits` | Cached vectors read, not HTTP requests. |
| `embedding_fresh_tokens` | `total_tokens` summed over those successful embedding responses. |

`truncated_chunks` counts chat calls whose metadata has `truncated`. In the new modes that is a parent call which was then split, not a page that was skipped.

Each new-mode audit also stores `retrieval` (query, full ranking, selected ids; `initial_batches` only when that group is extracted by itself), `extraction_batches` (fields, selected ids, initial batch count), `embedding_windows`, `embedding_calls`, and `provenance_notes`. `rag-all-fields` has one extraction batch record, `all_fields`, and does not write `initial_batches` onto the four retrieval rows.

### Commands

`grouped-all` does not call the embeddings endpoint:

```bash
python -m core_pipeline run \
  --input data/pdfs_dev_100 \
  --ids data/ids_15.txt \
  --input-format pdf \
  --output runs/pdfs_15_grouped_all \
  --mode grouped-all \
  --model deepseek-v4.1-flash-1017 \
  --min-votes 2 --quote-threshold 1.0 \
  --context-budget 65000 --max-output 4096 \
  --cache .cache/taskc
```

`rag` retrieves four pages per group. `rag-all-fields` uses the same retrieval and one all-field prompt over the union. Both need `LLM_API_KEY` for chat. Set `EMBEDDING_API_KEY` only when the embeddings endpoint uses a different key.

```bash
python -m core_pipeline run \
  --input data/pdfs_dev_100 \
  --ids data/ids_15.txt \
  --input-format pdf \
  --output runs/pdfs_15_rag_k4 \
  --mode rag \
  --model deepseek-v4.1-flash-1017 \
  --embedding-model qwen3-embedding-4b \
  --top-k 4 \
  --min-votes 2 --quote-threshold 1.0 \
  --context-budget 65000 --max-output 4096 \
  --cache .cache/taskc
```

Use a new empty output directory. Grouped prompts miss the old `{cache}/llm` entries on purpose. Coordinate TEI and, on a repeated `rag` run, embedding vectors are reused when the endpoint, model, prefix, and page text are unchanged.
