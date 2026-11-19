# SPEC: Evidence-verified LLM extraction for ANLP Shared Task C (C1 + C2b)

> Instructions for Claude Code: read this whole file and `CLAUDE.md` before planning.
> Build in the milestone order of section 9 and stop after each milestone so the team
> can review it. If something is unclear or listed in section 11, ask; do not guess.

---

## 1. Task in two sentences

Input: a document about biodiversity (PDF, English or German; scientific papers, but
also books and reports, from about 5 to more than 400 pages).
Output: one JSON record per document whose fields take values from fixed German lists or
short German texts, written as one line in `predictions.jsonl`.

Sources: `docs/` (task description, HowTo) and the CodaBench page of the task (newer
than the PDF; its details are included in this file).

## 2. Scope

**Task C1 (core fields)**, schema `data/schemas/core_fields.json`:

| Output field | Allowed values |
|---|---|
| `habitat` | 6, list of strings |
| `biodiversity_level` | 7, list of strings |
| `ecosystem_type` | list of `{"category": 6 values (required), "term": 46 values or null}` |
| `taxa` | list of `{"species_group": 16 values (required), "scientific_name", "german_name"}` |

**Task C2b (biodiversity)**, schema `data/schemas/biodiversity.json`, additionally:

| Output field | Kind |
|---|---|
| `taxa.scientific_name`, `taxa.german_name` | open text, **scored exactly** (case-sensitive) |
| `biodiversity_variable` | free text (German), BERTScore, not used for the ranking |
| `ecosystem_service` | list of `{"category": 3 values (required), "term": 67 values or null}`; `details` is not scored, omit it |

**Out of scope:** fine-tuning, training own models, any use of test data for tuning.

## 3. Pipeline

```
PDF ──► Docling ──► chunks C1 … Cn
            │
            ├──────────────────────────────────────────┐
            ▼                                          ▼
   F) FULL TEXT                          B) PARALLEL
   whole document, one call              chunks C1 … Cn, independent calls,
   model: deepseek-v4-flash-0731         merged in Python
            │                            model: deepseek-v4-flash-0731
            │                            (one chunk = F's request)
            │                                          │
            ▼                                          ▼
   verbatim checker (after EVERY LLM call, section 3a)
            │
            ▼
   post-processing (F only, src/postprocessing.py, section 13)
            │
            ▼
   validate against schema ─► predictions.jsonl ─► zip
```

1. **PDF → text:** Docling converts every document into text with headings and page
   numbers. One converter for all document types.
2. **Chunking** (pipeline B), with Docling's HybridChunker:
   - **headings found:** split at headings; merge small neighbouring sections up to
     `chunking.max_chunk_tokens`; split longer sections at paragraph boundaries;
   - **no headings found** (fallback): fixed windows of `max_chunk_tokens` at paragraph
     boundaries with `chunking.chunk_overlap_tokens` overlap.
   Never truncate silently: log the chunking level and every split per document.
   **Chunk size per document** (`src/chunk_packing.py`, decided 2026-11-18, section 13):
   effective size = `max(max_chunk_tokens, ceil(document_tokens / max_chunks_per_document))`,
   at most `models.chunked_part_tokens`. The chunker's pieces (at most `max_chunk_tokens`)
   are packed together in document order up to that size, never split, so a document
   has at most `max_chunks_per_document` chunks. If whole pieces do not fit into that
   many chunks, the size is raised to the smallest size that fits (still at most the
   cap); only a document that does not fit even at the cap gets more chunks (logged).
3. **LLM calls** (prompt: `config/prompts/extraction.txt`, section 6):
   - **F) Full text:** the whole document in one call with `models.full_text`.
     A document that does not fit into the context window (M1: 2 of 100 dev documents
     exceed 1M tokens) is never truncated silently and keeps the same model: it is split
     into as few parts as possible, each at most `models.full_text_part_tokens` (start:
     900,000, leaving room for prompt and answer), with the chunker from M1. Every part
     is one call; labels are merged as in pipeline B (verified quotes from any part
     count). These documents are logged and reported separately
     ('98 documents in one call, 2 in n parts').
   - **B) Parallel** (after LLM×MapReduce, Zhou et al. 2025): every chunk is processed
     independently with `models.chunked` and its own prompt
     (`paths.prompt_extraction_chunked`); the requests of all documents run in one
     thread pool. The verbatim checker runs per chunk with that chunk's text only.
     **A document with exactly one chunk is the full-text case** (`parallel.one_chunk_as_full_text`):
     it sends exactly F's request (the full text of `pdf_to_text.full_text`, prompt
     `paths.prompt_extraction`, the same model), so the cache key is F's and the answer
     comes from F's cache. Only documents with two or more chunks use the chunk prompt.
     Merging in Python, per document: a label is kept if it has a verified quote in at
     least min(`merge.min_chunks_per_label`, number of chunks of the document) different
     chunks (default 1 = union; the min() keeps short documents from being emptied),
     then C5 as in F. A chunk that does not mention a label is not a vote against it.
     Merging needs no request, so merge experiments come from the cache.
4. **Verbatim checker:** section 3a.
5. **Write output:** `predictions.jsonl` with exactly one line per document, validated
   against both schemas (`jsonschema`), then a zip containing only that file.

**Labels are German.** The model chooses enum values directly from the German lists in
the schema files. There is no English → German label mapping.

**Note for the analysis:** since 2026-11-18 F and B use the same model (DeepSeek) and
one-chunk documents get the same request in both, so differences between F and B come
from the documents with several chunks: chunking, the chunk prompt and the merging. The
first B run used Qwen (`config/experiments/chunked_qwen.yaml`); there, model and
architecture were mixed (section 13).

## 3a. Verbatim checker

**Purpose:** a deterministic Python filter that runs right after every LLM call and
discards each extracted label whose value is not in the schema or whose quote cannot be
found verbatim in the text that was read.

**Input:** the parsed LLM answer, the exact text sent to the model, the schema values,
`data/labels/term_to_category.json`.
**Output:** the cleaned labels plus one log entry per checked label (`label`, `check`,
`passed`, `score`, `reason`) in the run directory.

**Checks, in this order (a label is dropped at the first failed check):**

| # | Check | Drop the label if … | Prompt rule |
|---|---|---|---|
| C1 | Allowed value | an enum value is not exactly in the schema list | R1 |
| C2 | Consistent pair | a term does not belong to its category (`term_to_category.json`) | R6 |
| C3 | Plausible quote | the quote is missing, has fewer than `min_quote_words` or more than `max_quote_words` words, or contains "..." / "…" | R2 |
| C4 | Verbatim quote | the normalised quote is not in the normalised text with a fuzzy score ≥ `quote_match_threshold` | R2 |
| C5 | Enough evidence | fewer than `min_verified_quotes` verified quotes remain | – |

**Normalisation before C4** (same for quote and text): Unicode NFKC; lowercase; join
words split by line-break hyphenation; collapse whitespace; unify quotation marks,
dashes and ligatures. Matching: `rapidfuzz.fuzz.partial_ratio`.

**Special cases:** completed taxa names (`name_source = "completed"`) need a verbatim
quote, but the name itself is not searched in the text. For `biodiversity_variable`
only the quote is checked, never the German value.

**Limitation (report it):** the checker proves that a quote exists in the text, not that
it supports the label or that the study examined it.

**Reported per run:** share of labels dropped per check and per field.

## 4. Data and label files

See `README.md` (section "Repository Structure") for all folders and `CLAUDE.md` for
access rules.

- `data/raw/`: course data, never read by Claude Code. Train: 908 documents with
  references; dev: 100 PDFs with their references (`data/processed/dev_with_pdf.jsonl`;
  the 900 dev records on CodaBench were an error, section 13); test: 500 PDFs.
- **Schema files are the single source of truth for allowed values.** Load all values
  from the `enum` lists under `$defs`. Never hard-code or retype label strings.
- Known pitfalls in the schemas:
  1. The `description` of `EcosystemTypeSimple` spells ecosystem terms differently from
     `EcosystemTypeTermEnum`. Only the enum spelling is scored; prompts show enum values.
  2. `EcosystemServiceTermEnum` contains a typo ("1.3.X.X Andere (Versorungsleistungen …)").
     Keep it exactly as written.
  3. The schemas do not link terms to categories. `data/labels/term_to_category.json`
     provides this mapping (built by `src/scripts/build_term_to_category.py`, checked by hand).
     The 8 service terms that never occur in train are added by the first number of their
     code (`src/scripts/map_service_terms_by_prefix.py`, run after it; section 13).
- `data/labels/variable_vocabulary.json`: values of `biodiversity_variable` in the
  training references with counts (built by `src/scripts/build_variable_vocabulary.py`).
  The prompt shows the `prompt.vocabulary_top_n` most frequent values as preferred terms.

## 5. LLM access

- OpenAI-compatible API of GWDG SAIA (`llm.base_url` in the config).
- API keys from `.env` (loaded with `python-dotenv`): 1 to n keys as `SAIA_API_KEY_1`,
  `SAIA_API_KEY_2`, … (consecutive numbers); if no numbered key exists, the single
  `SAIA_API_KEY` is key 1. A key is never printed, logged or written anywhere else; logs,
  the summary and the run files name only its number ("key 2").
- **Limits per account, not per key** (measured 2026-11-18): SAIA applies the limits below
  per Academic Cloud account. Several keys of one account share one quota: a new key of
  the same account showed the same remaining counts. Only keys of different accounts add
  quota. The hourly limit refills gradually, not at the full hour (measured: 2 → 16 → 53
  → 107 remaining within minutes).
- **Several keys** (`src/api_key_pool.py`): every key has its own rate limiter and its own
  server counts; for keys of one account the server counts are the shared ones and win
  over the local count after the first answer. Each request goes to the key with
  the most requests left for the day (server count if known, else local count); keys
  whose hour or day is used up are skipped. A key that reaches its daily limit is left
  out and the run continues with the others; a key rejected with 401/403 (invalid or
  blocked) is logged and dropped for the rest of the run. The summary shows requests and
  remaining quota per key.
- Rate limits (observed in the SAIA response headers, M2): 10 requests per 10 seconds,
  400 per hour and 800 per day. Therefore:
  - **disk cache** (`paths.cache`): key = hash(model, full prompt, temperature), not the
    API key, so answers are shared across keys; a cached answer is never requested again. This also makes `Ctrl + C` safe: a stopped run
    continues where it ended;
  - client-side rate limiter per key (`llm.requests_per_10_seconds`, `llm.requests_per_hour`,
    `llm.requests_per_day`): before every request it waits until all three sliding
    windows of the chosen key have room (after a report of 0 left for the hour it waits a
    full hour, although the server refills earlier; section 11). 10 requests per minute would be 600 per hour, more than allowed;
  - when **all** keys have reached their daily limit, the run stops at once with a clear
    message (naming how many keys were used) instead of waiting for hours. The cache keeps all answers, so running the same command the
    next day continues where it stopped. After every response the server's own counts
    (`x-ratelimit-remaining-requests-hour`, `-day`) of that key win over its local count:
    0 left for the day drops the key until tomorrow, 0 left for the hour skips it for one
    hour (if every key's hour is used up, the run waits for the first one to free up). So a restart on the same day
    is safe; if a header is missing, the limiter falls back to counting locally;
  - retries with exponential backoff on 429/5xx;
  - `--dry-run`: builds all prompts and reports the number of requests without calling the API;
  - log the number of requests, the limits the server reports (`x-ratelimit-limit-*`,
    to confirm the values in the config) and the remaining quota (`x-ratelimit-remaining-*`),
    per key.

## 6. Prompt

Templates: `paths.prompt_extraction` (pipeline F, currently `extraction_v3.txt`) and
`paths.prompt_extraction_chunked` (pipeline B, `extraction_chunk_v2.txt`: v3 plus rule R9,
"you see only one chunk"); versions in `config/prompts/README.md`, the old pair (v2,
chunk_v1) in `config/experiments/prompt_v2.yaml`. Placeholders (`{habitat_labels}`,
`{ecosystem_terms_by_category}`, `{variable_vocabulary}`, `{taxa_question}`, …) are filled
by code from the schema files and `data/labels/`; `{min_quote_words}` and
`{max_quote_words}` from `verbatim_check` (since v3, so changing them changes the v3
prompts and costs new requests). The template contains:

1. a role (ecologist annotating a literature database), after Yang et al. (2026);
2. the fields with their allowed German values and (since v3) the guiding question of
   the database for each field, the `description` of the property in the schema;
3. hard rules R1–R8 against hallucination (allowed values only, verbatim quote for every
   label, only what this study examined, empty is valid, no world knowledge except for
   completing taxa names, consistent pairs, JSON only);
4. guided reasoning steps S1–S5 (study system, organisms, measurements, services,
   self-check), written into the `reasoning` field before the labels;
5. the JSON output structure.

Prompts v1 and v2 still contain a section "ADDITION FOR PIPELINE A" with ledger rules;
the code never reads it (pipeline A was dropped, section 13). v3 has no such section.

Parse and validate the answer with `pydantic`. Invalid JSON: retry once with the parse
error appended, then log and treat the call as empty.

## 7. Evaluation

**Own `src/evaluation.py`**, implementing the metric exactly as documented on CodaBench
(the official script has not been found yet):

- per field and document: compare the **set** of predicted values with the set of
  reference values, exactly and case-sensitive;
- subfields are separate fields (`ecosystem_type.category`, `ecosystem_type.term`,
  `taxa.species_group`, `taxa.scientific_name`, `taxa.german_name`,
  `ecosystem_service.category`, `ecosystem_service.term`); a missing field counts as empty;
- sum TP, FP and FN over all documents; precision, recall and F1 per field;
- **C1 score** = mean F1 of `habitat`, `biodiversity_level`, `ecosystem_type.category`,
  `ecosystem_type.term`, `taxa.species_group`;
- **C2b score** = mean F1 of `taxa.scientific_name`, `taxa.german_name`,
  `taxa.species_group`, `biodiversity_level`, `ecosystem_service.category`,
  `ecosystem_service.term`;
- `biodiversity_variable`: best-match BERTScore (`bert-base-multilingual-cased`, layer 9,
  no rescaling); implement last, skip with `--categorical-only`.

Interface (same as the official script, so it can be swapped in):
```
python src/evaluation.py score --references dev.jsonl --predictions predictions.jsonl --task all [--categorical-only]
```

**Checks:** gold vs. gold = 1.0; empty predictions = 0.0; the baselines reproduce the
official dev values (random allowed value: **0.1018**, most frequent training value:
**0.2443**); one submission scores the same locally and on CodaBench. If the official
script is found, compare both; the official one wins.

The output writer must catch everything CodaBench rejects: missing or duplicate ids,
unknown ids, duplicate JSON keys, non-string values.

## 8. Logging and reproducibility

- All settings in `config/settings.yaml`; one YAML file per experiment in
  `config/experiments/` that overrides only what changes.
- Each run writes to `runs/<timestamp>_<experiment>/`: copied config, git commit hash,
  prompts and raw LLM answers, verbatim checker log, `predictions.jsonl`, scores per
  field, number of requests, runtime, energy estimate (CodeCarbon).
- Fixed seeds; `requirements.txt` with pinned versions.

## 9. Milestones (stop after each)

| # | Milestone | Done when |
|---|---|---|
| M0 | Package layout in `src/`, config loading, data loading, `src/evaluation.py` (categorical fields), both baselines, validated writer + zip, invented fixtures in `tests/fixtures/` | tests pass on fixtures; the team confirms 0.1018 / 0.2443 on dev; a baseline zip is uploaded to CodaBench |
| M1 | Docling conversion and chunking | the team reports page counts and token lengths (how many documents exceed 50 pages, 256K and 1M tokens), share of fallback chunking, chunk statistics |
| M2 | LLM client with `.env`, cache, rate limiter, retries, dry-run | one fixture document end to end; a second run makes 0 requests |
| M3a | Pipeline F + verbatim checker | dev scores with and without the checker |
| M3b | Pipeline B | dev scores with and without the checker |
| M4 | dropped (2026-11-18): was pipeline A (sequential), see section 13 | – |
| M5 | Experiment runner and results table (pipelines F and B) | one command reproduces the table below |

| Pipeline | Model | Verbatim checker | `min_verified_quotes` |
|---|---|---|---|
| baselines (random, most frequent) | – | – | – |
| F full text | deepseek-v4-flash-0731 | off / on | 1 |
| B parallel | deepseek-v4-flash-0731 (Qwen: `chunked_qwen.yaml`) | off / on | 1, 2 |

**Research questions:** Does reading chunks independently (B) match reading the full
text (F) with the same model? (The first B run with a small model, Qwen, is kept as an
experiment.) How much does the verbatim checker help?

## 10. Code quality rules

The team learns from this code, so it must be easy to read and to extend:

- Python 3.13, type hints, short docstrings that explain *why*.
- Small modules with one responsibility each, e.g. `config`, `data`, `pdf_to_text`,
  `chunking`, `chunk_packing`, `labels`, `prompts`, `llm_client`, `pipelines/{full_text,parallel}`,
  `verbatim_checker`, `output`, `evaluation`.
- No magic numbers: every threshold and size comes from `config/settings.yaml`.
- Clear names over short names; no clever one-liners.
- `pytest` tests at least for evaluation, verbatim checker, chunking and output writer.
- No features beyond this spec without asking.

## 11. Open questions

1. ~~**Dev data:** `dev.jsonl` has 900 records, but only 100 dev PDFs were downloaded.~~
   Closed 2026-11-18: the dev set is the 100 documents with a PDF (section 13).
2. **Official `evaluate.py`:** asked on ISIS; until then use our own `src/evaluation.py`.
3. **Taxa names:** how to reach the exact spelling of the database (copy from text,
   LLM completion, or both). Check the naming conventions in the training references and
   fill `{taxa_naming_conventions}` in the prompt.
4. **Docling runtime** on very long or scanned documents (OCR is slow); measure in M1.
5. **Start values** of `quote_match_threshold`, `min_quote_words`, `max_quote_words`,
   `max_chunk_tokens`: tune on a few dev documents.
6. ~~**Chunk size for pipeline B**~~ Closed 2026-11-18: 16,000 tokens, at most 20 chunks
   per document (4 since 2026-11-19), cap 240,000 (section 3, step 2, and section 13). Still to do: report
   results separately for short and long documents.
7. **Rate limiter and the server's hour:** the limiter assumes a 60-minute window, but
   SAIA refills the hour gradually (section 5). Should it probe the server (e.g. one
   request after a few minutes) instead of waiting a full hour?
8. **Order of requests in pipeline B:** start the largest chunks first, so a long
   request does not run alone at the end of a run?
9. **Cache the prepared chunks:** preparing (loading, counting, chunking) takes about
   2 minutes per run of B, also for merge experiments that need no request.

## 12. Non-functional requirements

Time budgets (measured as wall-clock time and logged in every run):
- Test phase: 500 PDFs, 7 days, 3 submissions. PDF conversion plus extraction of
  all 500 documents must finish in under 24 hours.
  Measured in M1: 100 dev PDFs took 4.4 hours to convert (about 2.6 minutes each),
  so 500 PDFs take about 22 hours sequentially. Conversion must run in parallel.
- Dev run (about 100 PDFs, already converted): extraction should finish in under
  1 hour, so we can run and score at least one experiment per day.

Limits:
- SAIA API, per Academic Cloud account, not per key (from the x-ratelimit-limit-*
  response headers and section 5): 10 requests per 10 seconds, 400 per hour, 800 per day.
  Total capacity = these values x number of different accounts whose keys are in .env.
- PDF conversion: each Docling worker needs about 6 GB RAM.

Rules:
- Every run logs wall-clock time per stage (conversion, extraction, checking),
  the number of LLM requests and the average time per request.
- A speed-up must not change the results (tested: parallel = sequential output).

## 13. Decisions

| Date | Decision | Evidence | Status |
|---|---|---|---|
| 2026-11-18 | Verbatim checker: `max_quote_words` 40 for C1, 60 for C2b (same LLM answers, filtered differently; C1 and C2b are submitted separately) | Pipeline F, dev with PDF (100 docs). C1: 40 → 0.5250, 60 → 0.5196, 80 → 0.5158, 120 → 0.5119. C2b: 40 → 0.2854, 60 → 0.3113, 80 → 0.3113, 120 → 0.3109 | provisional: confirm after the error analysis; differences may partly be noise (100 docs) |
| 2026-11-18 | Extraction prompt v2 is the default (`config/prompts/extraction_v2.txt`); prompt versions are never edited after a run (`config/prompts/README.md`) | Pipeline F, dev with PDF: C1 0.5250 → 0.5680, C2b 0.2854 → 0.3302 (40 words). Cause found in the reasoning of the 25 empty answers: non-study documents (reports, books, web pages) were left empty because of "only what THIS study examined" | adopted |
| 2026-11-18 | Verbatim checker: `max_quote_words` 60 for C1 and C2b (replaces the provisional 40/60 split above) | With prompt v2: C1 0.5680 (40) vs 0.5711 (60), C2b 0.3302 (40) vs 0.3315 (60); differences are small, one setting is simpler | adopted |
| 2026-11-18 | Before scoring a run, check that `failed` in summary.json is empty; a failed document gets an empty prediction and lowers the score | v2 run: 2 failed documents lowered C1 from 0.5680 to 0.5610 | adopted |
| 2026-11-18 | Pipeline A (sequential) dropped; the project compares F and B | team decision | adopted |
| 2026-11-18 | Dev set = the 100 documents with a PDF and their references (`data/processed/dev_with_pdf.jsonl`); nothing is filled for the other dev ids. The `full` set (900) only checks our scorer against the official baseline numbers | only 100 dev PDFs exist; the 900 dev references on CodaBench were an error (confirmed by the course) | adopted |
| 2026-11-18 | Pipeline B chunk size: `max_chunk_tokens` 16,000 (was 8,000), `max_chunks_per_document` 20, effective size `max(max_chunk_tokens, ceil(document_tokens / 20))`; pieces packed in order so a document has at most 20 chunks; cap `models.chunked_part_tokens` 240,000 (safety net for test documents; chosen for Qwen's 256K context, kept with DeepSeek so the chunks stay the same). 8K stays a later experiment (`chunk_8k.yaml`) | estimated dev run about 420 requests (8K: about 615); more context per chunk; with 16K about 37 % of the dev documents fit into one chunk, which makes B closer to F. Largest dev document 2,387,800 tokens (3WEEGFGW) → about 120K per chunk, so the cap does not apply on dev. The M1 statistics were made with 8,000 and are not rerun | adopted |
| 2026-11-18 | Pipeline B has its own prompt (`extraction_chunk_v1.txt`), so F keeps prompt v2 | team decision | adopted |
| 2026-11-18 | Pipeline B merges by vote (`merge.min_chunks_per_label`, default 1), tunable without new requests (`merge_min_chunks_2.yaml`, `merge_min_chunks_3.yaml`) | team decision | adopted |
| 2026-11-18 | B with Qwen: C1 0.4527 / C2b 0.2070 (k=1), 0.4295 / 0.1852 (k=2), 0.4189 / 0.1626 (k=3); union is best | dev, 100 papers | adopted |
| 2026-11-18 | Gap F vs B is mostly not the chunking: one-chunk papers F 0.678 vs B 0.416 (C1), multi-chunk papers 0.524 vs 0.465 | dev | finding |
| 2026-11-18 | One-chunk documents use prompt v2 and F's request; B uses DeepSeek like F, Qwen kept as experiment | to separate model and chunking | adopted |
| 2026-11-18 | SAIA limits apply per Academic Cloud account, not per API key; the hourly limit refills gradually | a new key of the same account showed the same remaining counts; hour remaining 2 → 16 → 53 → 107 within minutes | finding |
| 2026-11-19 | Pipeline B with DeepSeek and at most 4 chunks per paper (`config/experiments/max_4_chunks.yaml`): C1 0.5781, C2b 0.3178 (F: 0.5711, 0.3315) | dev, 100 papers; same model as F, so the chunking alone is compared: about equal; B better on biodiversity_level and species_group, F better on taxa names | finding |
| 2026-11-19 | Verbatim checker: `min_quote_words` 3 (was 5) | Almost all C3 drops in B were quotes under 5 words (taxon names, short terms). B: C1 0.5781 → 0.5872, C2b 0.3178 → 0.3311; F unchanged (0.5710 / 0.3310); 1 word lowers F's C1 to 0.5579. `max_quote_words` 40-120 makes no difference with prompt v2 | adopted |
| 2026-11-19 | DeepSeek on SAIA was unreachable for about an hour (even a tiny request hung, Qwen answered at once); runs stopped cleanly and continued from the cache | risk for the test week: process the test PDFs early | finding |
| 2026-11-19 | Prompt v3 shows the schema's guiding question under each field (`{<field>_question}`, filled from the `description` of the schema properties habitat, ecosystem_type, biodiversity_level, taxa, biodiversity_variable, ecosystem_service) | the prompt listed only the allowed values. Dev: the model sets biodiversity_level in 72 of 100 papers, the database in 41; the question asks where biodiversity is "gemessen" | adopted, not run yet |
| 2026-11-19 | Prompt v3, R7: always give a German name (from the text, translated from English, or added from own knowledge with `name_source` "completed"); scientific name if known, else null | train (908 references): 99.8 % of taxa entries have a German name, 89 % a scientific name; dev: 79 missing German names do not occur in the text, so they must be added or translated, as the schema's taxa question demands | adopted, not run yet |
| 2026-11-19 | Prompt v3, R7: smallest feasible level (few species, at most 10: one entry each; many species or a group: one group entry), with the base rate "about half only groups, about a third only species"; the cap of 10 taxa stays | train: of 585 papers with taxa, 51 % list only groups, 31 % only species, 18 % both; 95 % have at most 11 single species | adopted, not run yet |
| 2026-11-19 | Scientific names stay without author citation (R7 spelling rules unchanged) | train: only 3 of 2,281 scientific names carry an author citation | adopted |
| 2026-11-19 | Prompt v3, R2: quote length from `verbatim_check.min_quote_words` / `max_quote_words` (`{min_quote_words}`, `{max_quote_words}`) | prompt v2 said 5-40 words, the checker allows 3-60 | adopted, not run yet; checker experiments on quote length now change the v3 prompt and cost requests |
| 2026-11-19 | Defaults: `extraction_v3.txt` (F), `extraction_chunk_v2.txt` (B, v3 plus R9); v3 drops the unused section "ADDITION FOR PIPELINE A". `config/experiments/prompt_v2.yaml` runs v2 / chunk_v1 from the cache | findings above | adopted, not run yet |
| 2026-11-19 | Pipeline B default `chunking.max_chunks_per_document` 4 (was 20); `max_4_chunks.yaml` is kept and now equals the default | B with at most 4 chunks is the best B setting: C1 0.5872, C2b 0.3311 (min_quote_words 3) | adopted |
| 2026-11-19 | `term_to_category.json`: the 8 ecosystem_service terms without an entry get the category of their code's first number (1.x Versorgungsleistungen, 2.x Regulierungs- und Erhaltungsleistungen, 3.x Kulturelle Leistungen); the rule is read from the mapped terms, never typed (`src/scripts/map_service_terms_by_prefix.py`) | all 59 mapped terms follow the rule (also 4.x -> Versorgungsleistungen); check against all train and dev reference pairs: to be run by the team | adopted; reference check pending |
| 2026-11-19 | Prompt v3 (schema questions, taxa base rate, always German names) rejected; defaults back to v2 / chunk_v1; v3 kept as `config/experiments/prompt_v3.yaml` | F: C1 0.5710 → 0.5122, C2b 0.3310 → 0.3180; B: 0.5872 → 0.5505, 0.3311 → 0.2881. Taxa too coarse (e.g. 16 species → one group), 131 instead of 96 ecosystem terms ("betrachtet"), biodiversity_level not better. Lesson: one change per run | adopted |
| 2026-11-19 | Pipeline F is the main system; B stays in the code and reproducible (`config/experiments/prompt_v2.yaml`: v2 / chunk_v1, DeepSeek 4.0, max 4 chunks) but is no longer developed. Defaults: `extraction_v4.txt` (v2 plus the database labels of the 3 most similar training papers, TF-IDF on the first pages) and `models.full_text` `deepseek-v4.1-flash-1017`. B stops with a message on prompts with training examples. Experiment files of earlier runs name their prompt, model and checker values, so they still come from the cache | F with prompt v4 and DeepSeek 4.1: C1 0.6644, C2b 0.3733 (v4 with DeepSeek 4.0: 0.5836 / 0.3602; v2: 0.5710 / 0.3310; B best: 0.5872 / 0.3311). With the same model F and B scored about equal; F needs about 107 instead of 255 requests per dev run and is simpler | adopted (team decision) |
| 2026-11-19 | Verifier (second call per paper, keep/drop only, `config/prompts/verify_v1.txt`) not adopted | On F v4.1: C1 0.6644 → 0.6524, C2b 0.3733 → 0.3603. It dropped mostly wrong labels (e.g. biodiversity_level 21 wrong vs 9 right), but dropping is only worth it if wrong:right > (2 − F1) / F1, i.e. about 3.9 : 1 at F1 0.41. Low-F1 fields need a very precise verifier | rejected |
| 2026-11-19 | Post-processing rule in F (`postprocess.drop_services_without_neighbour_support`, default true, `src/postprocessing.py`): if a paper has training neighbours (`neighbours_<split>.json`) and none of them has an ecosystem_service entry in the train references, all ecosystem_service entries of the paper are dropped; a paper without neighbours keeps its services. Runs after the verbatim checker, only removes entries, needs no request; without it: `config/experiments/no_service_rule.yaml`. summary.json reports papers and entries dropped | Measured without new requests. Dev references: papers whose 3 neighbours have no services have services themselves in 1 of 64 cases, papers with at least one such neighbour in 7 of 35. Train (878 papers, 3 nearest other train papers, TF-IDF): 35 of 525 (6.7 %) vs 163 of 353 (46 %), so the pattern is a property of the database, not of the dev set. On F v4.1 (DeepSeek 4.1, prompt v4) it keeps all correct services (category TP 5, term TP 4) and removes about half of the false ones (FP 25 → 12, 29 → 15): C2b 0.3733 → 0.4130, C1 unchanged (services are not part of C1); better in 99.8 % of 2,000 bootstrap samples. The dev gain is probably optimistic (dev 1 of 64 vs train 6.7 %). Confirmed by dev runs from the cache (0 requests): with the rule C1 0.6644, C2b 0.4130 (`runs/20261119-154747_full_text_dev`); without it (`no_service_rule.yaml`) 0.6644 / 0.3733, exactly F v4.1 | adopted |
| 2026-11-19 | Field prompt for biodiversity_level (decomposed prompting, Khot et al. 2023): `field_prompts.biodiversity_level` (null = off; `config/experiments/level_prompt_v1.yaml`: `config/prompts/level_v1.txt`). After the main call of a paper, one more call per part with the same model, examples and text; its labels go through the same verbatim checker and merge and REPLACE the main call's levels. The main call is unchanged (from the cache). If a level call fails (API error or invalid JSON after the retry), the paper keeps the main call's levels; summary.json `field_prompts`: requests, papers replaced, failed, papers_with_fallback. F only | biodiversity_level is often wrong both ways: F v2 set it in 72 of 100 dev papers, the database in 41; in the upper-bound analysis its recall was 42 % (docs/EXPERIMENTS.md, section 9); one prompt with seven described levels and base rates instead of one line in the main prompt. `docs/` has no official definitions of the levels (only the German–English names in the task description and the schema question "Auf welche der folgenden Ebenen wird Biodiversität in der Studie gemessen?"). Result on F v4.1 + service rule (`runs/20261119-161424_full_text_dev_level_prompt_v1`): C1 0.6644 → 0.6732, C2b 0.4130 → 0.4202, all other fields unchanged; biodiversity_level F1 0.412 → 0.455 (TP/FP/FN 28/41/39 → 46/89/21; recall 0.42 → 0.69, precision 0.41 → 0.34). Better in 73 % of 2,000 bootstrap samples. Confidence thresholds do not help (cross-validated 0.6666 / 0.4148); its confidences are overconfident and barely separate right from wrong. Adopted because the direction was predicted beforehand (a label pays off if P(correct) > F1/2 ≈ 0.21, Lipton et al. 2014; the verifier, which dropped levels, lowered the field to 0.358) and train has a level in 58 % of papers, dev in 41 % (hypothesis, medium confidence: more recall helps more on a test set like train). Default `field_prompts.biodiversity_level: config/prompts/level_v1.txt`; off: `no_level_prompt.yaml`; experiment files of earlier runs set it to null. Cost: about twice the requests of F (dev about 214 instead of 107; test about 1,080 instead of 540) | adopted (not robust, bootstrap 73 %) |
| 2026-11-19 | Verbalized confidence (Tian et al. 2023) and thresholds per field (Lipton et al. 2014): prompt v5 (`extraction_v5.txt` = v4 plus `"confidence"` per label, rule R9, S5 (d); `config/experiments/prompt_v5.yaml`). The parser accepts `confidence` for all prompts (a missing or invalid value becomes null, never a reason to drop a label); merging keeps a label's highest confidence. Every F run writes `labels_with_confidence.jsonl` (one line per label: value, confidence, kept by the checker, replaced by a field prompt, dropped by post-processing). `postprocess.min_confidence` ({field: threshold}, empty = off; null confidence is kept) runs before the service rule. `src/scripts/sweep_confidence.py` prints F1 per threshold 0.0–0.9 per scored field, the threshold F1(0)/2 ("theory") and a 5-fold cross-validated choice (seed 7) with its out-of-fold F1, and C1/C2b for no threshold, theory and out of fold | the verifier showed that dropping labels only pays with a precise signal (SPEC.md, 2026-11-19 verifier row); a threshold on the model's own probability is a cheap signal and can be tuned without requests. Plan: run v5 once, sweep, then set thresholds only if the out-of-fold scores beat no threshold. Result on F v4.1 + service rule (`runs/20261119-162000_full_text_dev_prompt_v5`): C1 0.6644 → 0.6234, C2b 0.4130 → 0.4181 without threshold; theory 0.6214 / 0.4181; cross-validated, out of fold 0.6133 / 0.4515. The confidences separate right from wrong (e.g. habitat 77 of 82 right at 0.9–1.0, 6 of 8 at 0.4–0.6), but R9 also made the model more liberal and it lost correct labels (habitat TP 103 → 98, ecosystem_type.category F1 0.722 → 0.654); a threshold cannot bring them back. Taxa names from v5 with run 12: C2b 0.4308 but C1 0.6706 and two main calls per paper: not used. Lesson: R9 changed two things at once. `postprocess.min_confidence`, the label file and the sweep script stay | rejected |
| 2026-11-19 | Prompt v6 (`config/prompts/extraction_v6.txt` = v4 plus rule R9 "Broad overviews": strategies, red lists, status reports, assessments, textbooks and invasive-species overviews get every habitat they cover, almost never an ecosystem_type, taxa only if one group is the main subject; `config/experiments/prompt_v6.yaml`, everything else as the defaults). | Dev: 24 of 35 habitat FN of the defaults come from 7 broad overviews; train: all 23 papers with >= 4 habitats are such overviews, ecosystem_type in 1 of 23, taxa in 7 of 23. Result on the defaults (C1 0.6732, C2b 0.4202) (`runs/20261119-171235_full_text_dev_prompt_v6`, 107 new main calls, level calls from the cache): C1 0.6606, C2b 0.4338. Habitat TP/FP/FN 103/10/35 → 120/32/18 (F1 0.821 → 0.828); species_group F1 0.707 → 0.756, scientific_name 0.331 → 0.365, german_name 0.331 → 0.360; ecosystem_type.category 0.722 → 0.674, term 0.661 → 0.591. Bootstrap vs defaults (2,000 resamples, seed 1): C1 better in 21 %, C2b in 77 %. R9 found 4 of the 7 broad overviews in dev but also treated 5 other documents as overviews and gave them 4-6 habitats (21 FP); species_group improved because false groups in overview-like documents fell 16 -> 1; ecosystem types lost TP in normal studies. Cause: the rule was checked in the wrong direction: in train P(overview \| >= 4 habitats) is high (all 23), but P(>= 4 habitats \| overview) is low (books 0 of 29, techreports 9 of 54, 34 of them with one habitat); whether an overview gets all habitats is inconsistent in the database. Lesson: check a rule in the direction in which the prompt applies it (docs/EXPERIMENTS.md, section 13) | rejected |
| 2026-11-19 | Post-processing rule `postprocess.drop_ecosystem_type_for_land_only_habitats` (default true since 2026-11-19, `src/postprocessing.py`): if a paper's predicted habitats are non-empty and all in `postprocess.land_only_habitats` (Agrar- und Offenland, Boden, Urbane Räume; checked against the habitat schema), all its ecosystem_type labels are dropped. Runs after `min_confidence`, before the service rule; only removes labels, needs no request; summary.json `ecosystem_type_rule` counts papers and entries. Off: `config/experiments/no_eco_type_rule.yaml` (reproduces 0.6732 / 0.4202 from the cache); experiment files of earlier runs switch it off; `eco_type_rule.yaml` now equals the defaults; with v6: `prompt_v6_eco_type_rule.yaml` | Train: 217 of 282 papers whose habitats are only these have no ecosystem_type (77 %); share with an ecosystem_type per habitat: Agrar- und Offenland 23 %, Boden 10 %, Urbane Räume 20 %, vs Binnengewässer 73 %, Küsten 58 %. Dev, from the cache (0 requests), 13 entries dropped in 13 papers: on the defaults C1 0.6732 → 0.6818, C2b unchanged 0.4202 (`runs/20261119-173148_full_text_dev_eco_type_rule`; category TP/FP 35/16 → 31/7, term 36/20 → 33/10); on v6 C1 0.6606 → 0.6754, C2b 0.4338 (`runs/20261119-173302_full_text_dev_prompt_v6_eco_type_rule`). Bootstrap (2,000 resamples, seed 1): C1 better in 77 % of samples. Adopted because the rule was derived from train before dev was looked at, needs no request and only drops labels | adopted (not robust, bootstrap 77 %) |
