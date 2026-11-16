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
     A document that does not fit into the context window is never truncated silently:
     it is processed with pipeline B instead, logged, and reported separately.
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
- API key from `.env` (variable `SAIA_API_KEY`, loaded with `python-dotenv`); never
  printed, logged or written anywhere else.
- Rate limits (free tier): 10 requests/min, 400/day, 3,000/month. Therefore:
  - **disk cache** (`paths.cache`): key = hash(model, full prompt, temperature); a cached
    answer is never requested again. This also makes `Ctrl + C` safe: a stopped run
    continues where it ended;
  - client-side rate limiter and retries with exponential backoff on 429/5xx;
  - `--dry-run`: builds all prompts and reports the number of requests without calling the API;
  - log the number of requests and the remaining quota (`x-ratelimit-remaining-*`).

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
