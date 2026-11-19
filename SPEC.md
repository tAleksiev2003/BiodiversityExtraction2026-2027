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

Templates: `paths.prompt_extraction` (pipeline F, currently `extraction_v2.txt`) and
`paths.prompt_extraction_chunked` (pipeline B, `extraction_chunk_v1.txt`: v2 plus rule R9,
"you see only one chunk"); versions in `config/prompts/README.md`. Placeholders (`{habitat_labels}`,
`{ecosystem_terms_by_category}`, `{variable_vocabulary}`, …) are filled by code from the
schema files and `data/labels/`. The template contains:

1. a role (ecologist annotating a literature database), after Yang et al. (2026);
2. the fields with their allowed German values;
3. hard rules R1–R8 against hallucination (allowed values only, verbatim quote for every
   label, only what this study examined, empty is valid, no world knowledge except for
   completing taxa names, consistent pairs, JSON only);
4. guided reasoning steps S1–S5 (study system, organisms, measurements, services,
   self-check), written into the `reasoning` field before the labels;
5. the JSON output structure.

Prompts v1 and v2 still contain a section "ADDITION FOR PIPELINE A" with ledger rules;
the code never reads it (pipeline A was dropped, section 13). It is removed in the next
prompt version (`config/prompts/README.md`).

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
   per document, cap 240,000 (section 3, step 2, and section 13). Still to do: report
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
