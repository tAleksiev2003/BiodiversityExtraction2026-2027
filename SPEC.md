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
   F) FULL TEXT                          chunks C1 … Cn
   whole document, one call                 │
   model: deepseek-v4-flash-0731            ├─────────────────────────┐
            │                               ▼                         ▼
            │                     A) SEQUENTIAL             B) PARALLEL
            │                     ledger is passed on       independent calls
            │                     model: qwen3-30b-a3b      model: qwen3-30b-a3b
            │                               │                         │
            ▼                               ▼                         ▼
   verbatim checker (after EVERY LLM call, section 3a)
            │
            ▼
   validate against schema ─► predictions.jsonl ─► zip
```

1. **PDF → text:** Docling converts every document into text with headings and page
   numbers. One converter for all document types.
2. **Chunking** (pipelines A and B), with Docling's HybridChunker:
   - **headings found:** split at headings; merge small neighbouring sections up to
     `chunking.max_chunk_tokens`; split longer sections at paragraph boundaries;
   - **no headings found** (fallback): fixed windows of `max_chunk_tokens` at paragraph
     boundaries with `chunking.chunk_overlap_tokens` overlap.
   Never truncate silently: log the chunking level and every split per document.
3. **LLM calls** (prompt: `config/prompts/extraction.txt`, section 6):
   - **F) Full text:** the whole document in one call with `models.full_text`.
     A document that does not fit into the context window (M1: 2 of 100 dev documents
     exceed 1M tokens) is never truncated silently and keeps the same model: it is split
     into as few parts as possible, each at most `models.full_text_part_tokens` (start:
     900,000, leaving room for prompt and answer), with the chunker from M1. Every part
     is one call; labels are merged as in pipeline B (verified quotes from any part
     count). These documents are logged and reported separately
     ('98 documents in one call, 2 in n parts').
   - **A) Sequential** (after Chain of Agents, Zhang et al. 2024): call *i* receives
     chunk *i* plus the current evidence ledger and returns the updated ledger. It may
     add, confirm or remove labels (rules L1–L4 in the prompt). The last ledger is the
     result; there is no manager call.
   - **B) Parallel** (after LLM×MapReduce, Zhou et al. 2025): every chunk is processed
     independently. Merging in Python: a label is kept if it has at least
     `verbatim_check.min_verified_quotes` verified quotes across all chunks. A chunk that
     does not mention a label is not a vote against it.
4. **Verbatim checker:** section 3a.
5. **Write output:** `predictions.jsonl` with exactly one line per document, validated
   against both schemas (`jsonschema`), then a zip containing only that file.

**Labels are German.** The model chooses enum values directly from the German lists in
the schema files. There is no English → German label mapping.

**Note for the analysis:** F uses another model than A and B. Differences between F and
A/B therefore come from the architecture **and** the model. Report this as a limitation.

## 3a. Verbatim checker

**Purpose:** a deterministic Python filter that runs right after every LLM call and
discards each extracted label whose value is not in the schema or whose quote cannot be
found verbatim in the text that was read. In pipeline A it runs before the ledger is
passed on, so hallucinated labels never travel to later chunks.

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
| C4 | Verbatim quote | the normalised quote is not in the normalised text with a fuzzy score ≥ `quote_match_threshold` | R2, L4 |
| C5 | Enough evidence | fewer than `min_verified_quotes` verified quotes remain | – |

**Normalisation before C4** (same for quote and text): Unicode NFKC; lowercase; join
words split by line-break hyphenation; collapse whitespace; unify quotation marks,
dashes and ligatures. Matching: `rapidfuzz.fuzz.partial_ratio`.

**Special cases:** completed taxa names (`name_source = "completed"`) need a verbatim
quote, but the name itself is not searched in the text. For `biodiversity_variable`
only the quote is checked, never the German value. Removals in pipeline A also need a
verbatim quote; otherwise they are ignored.

**Limitation (report it):** the checker proves that a quote exists in the text, not that
it supports the label or that the study examined it.

**Reported per run:** share of labels dropped per check and per field.

## 4. Data and label files

See `README.md` (section "Repository Structure") for all folders and `CLAUDE.md` for
access rules.

- `data/raw/`: course data, never read by Claude Code. Train: 908 documents with
  references; dev: 900 reference records (see section 11, question 1); test: 500 PDFs.
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
- **Several keys** (`src/api_key_pool.py`): the limits below apply per key, so every key
  has its own rate limiter and its own server counts. Each request goes to the key with
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
    windows of the chosen key have room. 10 requests per minute would be 600 per hour, more than allowed;
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

Template: `config/prompts/extraction.txt`. Placeholders (`{habitat_labels}`,
`{ecosystem_terms_by_category}`, `{variable_vocabulary}`, …) are filled by code from the
schema files and `data/labels/`. The template contains:

1. a role (ecologist annotating a literature database), after Yang et al. (2026);
2. the fields with their allowed German values;
3. hard rules R1–R8 against hallucination (allowed values only, verbatim quote for every
   label, only what this study examined, empty is valid, no world knowledge except for
   completing taxa names, consistent pairs, JSON only);
4. guided reasoning steps S1–S5 (study system, organisms, measurements, services,
   self-check), written into the `reasoning` field before the labels;
5. the JSON output structure; an extra block with ledger rules L1–L4 for pipeline A.

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
| M4 | Pipeline A | dev scores with and without the checker |
| M5 | Experiment runner and results table | one command reproduces the table below |

| Pipeline | Model | Verbatim checker | `min_verified_quotes` |
|---|---|---|---|
| baselines (random, most frequent) | – | – | – |
| F full text | deepseek-v4-flash-0731 | off / on | 1 |
| B parallel | qwen3-30b-a3b-instruct-2507 | off / on | 1, 2 |
| A sequential | qwen3-30b-a3b-instruct-2507 | off / on | 1, 2 |

**Research questions:** Can a small model that reads chunks (A, B) match a large model
that reads the full text (F)? Does passing information between chunks help (A) or is
independent processing more robust (B)? How much does the verbatim checker help?

## 10. Code quality rules

The team learns from this code, so it must be easy to read and to extend:

- Python 3.13, type hints, short docstrings that explain *why*.
- Small modules with one responsibility each, e.g. `config`, `data`, `pdf_to_text`,
  `chunking`, `labels`, `prompts`, `llm_client`, `pipelines/{full_text,sequential,parallel}`,
  `verbatim_checker`, `output`, `evaluation`.
- No magic numbers: every threshold and size comes from `config/settings.yaml`.
- Clear names over short names; no clever one-liners.
- `pytest` tests at least for evaluation, verbatim checker, chunking and output writer.
- No features beyond this spec without asking.

## 11. Open questions

1. **Dev data:** `dev.jsonl` has 900 records, but only 100 dev PDFs were downloaded.
   Asked on ISIS. Until answered, develop on the 100 documents that have a PDF.
2. **Official `evaluate.py`:** asked on ISIS; until then use our own `src/evaluation.py`.
3. **Taxa names:** how to reach the exact spelling of the database (copy from text,
   LLM completion, or both). Check the naming conventions in the training references and
   fill `{taxa_naming_conventions}` in the prompt.
4. **Docling runtime** on very long or scanned documents (OCR is slow); measure in M1.
5. **Start values** of `quote_match_threshold`, `min_quote_words`, `max_quote_words`,
   `max_chunk_tokens`: tune on a few dev documents.
6. **Chunk size for pipelines A and B** (decide before M3b). M1 result: 2,168 chunks for
   100 dev documents with 8,000-token chunks; with the SAIA limit of 800 requests per
   day (from the response headers in M2) one pipeline B run takes almost 3 days; one
   document alone has 837 chunks. Proposal: adaptive chunk size
   `max(max_chunk_tokens, document_tokens / max_chunks_per_document)`, e.g. at most 20
   chunks per document, so normal papers keep 8,000-token chunks and books stay
   affordable without dropping text. Report results separately for short and long
   documents.
