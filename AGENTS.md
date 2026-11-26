# AGENTS.md – Engineering & Workflow Guidelines

This document defines the mandatory coding, architectural, and workflow standards for the **BiodiversityExtraction2026-2027** repository (TU Berlin, *Advanced Natural Language Processing*, Shared Task C1 + C2b).

---

## 1. Collaboration & Pacing Rules
- **Step-by-Step Workflow:** Never generate multiple unrequested models or jump ahead in the roadmap. Build, inspect, and verify **one single component or model at a time** in agreement with the user.
- **Branch Discipline:** Work happens on dedicated feature/experiment branches (e.g., `lennard_testing`). Never push experimental code directly to `main`.
- **Data Privacy & Git Hygiene:** Course dataset files (`data/gold/`, `data/raw_pdfs/`, `data/parsed_text/`), generated model outputs (`outputs/`), and personal checklists (`AUFGABENLISTE.md`) must remain excluded via `.gitignore` and must never be committed to this repository. Datasets live in the **separate Gitea-only data repo** [`uni/BiodiversityExtraction2026-2027-data`](http://100.69.68.23:3000/uni/BiodiversityExtraction2026-2027-data) (PDFs via Git-LFS; tailnet-only, **never push data to GitHub** — no LFS there and course data protection forbids it). To work locally: clone the data repo next to the code repo and symlink `ln -s ../BiodiversityExtraction2026-2027-data/data data`. `data/parsed_text/` and `data/processed/` hold the canonical preprocessing caches (Docling JSON/Markdown, Marker Markdown, GROBID TEI, first_pages, neighbours): produced once on university GPU servers and versioned in the data repo; never reconvert locally, reuse the cache. Server workflow: `scripts/deploy_pdfs_to_gruenau.sh` + `scripts/deploy_inputs_to_gruenau.sh` (push), `scripts/preprocess_all_on_gruenau.sh` via ssh (run), `scripts/sync_parsed_from_gruenau.sh` (pull, verify, `--cleanup`).
- **Server Workflow (lenlenovo) — always sync before runs:** Long benchmark runs execute on the home server `lenlenovo` (`~/Home/BiodiversityExtraction2026-2027`, fresh clone; origin = Gitea mirror on lenserver-hp, reachable via Tailscale). Code is developed **locally** and pushed to Gitea/GitHub, so before starting or evaluating any run on the server: `git fetch && git pull` (code) and `git -C ~/Home/BiodiversityExtraction2026-2027-data pull` + `git lfs pull` (data). Never edit code directly on the server — changes made there will be silently overwritten by the next pull.

---

## 2. Code Quality & Documentation Standards
- **Language:** All source code, function/variable names, type annotations, docstrings, and inline comments **must be written in clean, idiomatic English**.
- **Top-of-File Executive Summary & Extensive Comments:** Every model script (`src/models/.../*.py`) **must** begin with a structured English `EXECUTIVE SUMMARY` docstring block summarizing:
  1. *Concept & Goal* (what hypothesis/baseline the script tests),
  2. *How It Works* (feature extraction, modeling, handling of empty fields & Subtask C2b conventions), and
  3. *Results on Dev Set* (exact C1 Macro-F1, C1+C2b Macro-F1, Micro-F1, and key takeaways).
  In addition, all functions and non-trivial code blocks must be thoroughly commented in English so teammates and instructors can immediately understand every step.
- **Modularity:** Every model tested must be implemented as **one standalone, self-contained Python script** inside its corresponding category folder:
  - Base / Trivial & Classical Models: `src/models/base_models/<model_id>.py`
  - LLM & Advanced Architectures: `src/models/architectures/<model_id>.py`
- **Target Schema:** All models target **Task C1 (Core)** + **Subtask C2b (Biodiversity)**. Predicted values must strictly match the official German enum strings defined in `schemas/core_fields.json` and `schemas/biodiversity.json`.

---

## 3. Reproducibility & Standardized Model Outputs
Every model script must call `save_reproducible_run()` from `src/evaluation/evaluator.py` so that every execution produces a standardized, self-contained output directory under `outputs/<category>/<model_id>/` with cleanly prefixed filenames:

1. **`<model_id>_run_info.json`**
   - Logs UTC timestamp, Git branch & commit hash, random seed(s), total runtime (seconds), hardware/OS specs, model configuration/hyperparameters, and summary F1 scores.
2. **`<model_id>_metrics.json`**
   - Logs corpus-level `c1_macro_f1` (official leaderboard metric), `c2b_macro_f1`, `micro_f1`, and per-field `precision`, `recall`, `f1`, `tp`, `fp`, and `fn`.
   - For stochastic models, runs must be evaluated across **at least 3 fixed seeds** (e.g., `[42, 43, 44]`), logging both per-seed metrics and `mean ± std`.
3. **`<model_id>_report.txt`**
   - Human-readable ASCII evaluation table ready for inspection and presentation slides.
4. **`<model_id>_predictions.jsonl`**
   - Raw JSONL predictions on the 100 valid Dev papers (`data/gold/dev_100_ids.txt`) to support systematic error analysis.
5. **`<model_id>_per_paper.jsonl`**
   - Per-paper, per-field correctness record for every pipeline: one line per Dev paper with `correct: true/false` (exact-match vs. gold, free-text fields excluded or flagged) for each of the 9 scored fields, plus the pipeline phase/state (parser, prompt flags, model, cache source). This is the binding basis for cross-pipeline comparisons (which papers does pipeline A win/lose vs. pipeline B) — corpus-level metrics alone are not sufficient.
6. **`<model_id>_submission.zip`**
   - CodaBench-ready ZIP archive containing `predictions.jsonl` at its root.
7. **Central Leaderboard Update (`outputs/all_models_comparison.md`)**
   - Automatically updated after every run to maintain a single source of truth comparing all evaluated models by C1 Macro-F1, C1+C2b Macro-F1, Micro-F1, runtime, seed, and Git commit.

---

## 4. Documentation & Result Quality Standards
- **Simple English for outsiders:** All docs (`README.md`, `docs/*.md`) must use short sentences and plain words so non-natives and course outsiders can follow. Explain every model, tool, and NLP term on first use (e.g. PyMuPDF = fast PDF text extractor; GROBID = structure-aware academic parser; Marker/Docling = layout parsers; TF-IDF, kNN, RAG, OFAT = one-factor-at-a-time). Never assume the reader knows our pipeline.
- **Fixed set of maintained files:** Keep exactly these up to date and do not scatter results elsewhere — `README.md` (story + timeline), `docs/experiment_matrix.md` (every run with scores, regenerated by script), `docs/experiment_log.md` (lab notebook: grouped experiment tables with deltas vs. baseline), `docs/dev_100_paper_audit_table.md` (gold problems per paper), `docs/pipeline_overview_*.md` (architecture history).
- **README timeline:** `README.md` holds the chronological development story with dates, scores, and commit refs, ending with the justified current pipeline for the next milestone.
- **Reference results, do not copy them:** Docs link to `docs/experiment_matrix.md` and per-run artifacts instead of duplicating numbers. If a number appears twice, the matrix wins.
- **Model registry:** Every run records provider + exact model id (e.g. `saia / deepseek-v4.1-flash-1017`) in `run_info.json`. No run without a named model. Docs name real models (Qwen3.6-27B, Mistral Large), never internal routing nicknames alone.
- **Both providers for every OFAT test:** Each isolated optimization runs on DeepSeek 4.1 (via the SAIA academic cloud) *and* Qwen3.6-27B (via the HU-Berlin bridge), because gains demonstrably do not transfer between providers. A flag is adopted per provider, never globally.
- **OFAT optimization protocol:** One Factor At a Time — change exactly one thing per run against its immediate baseline on the fixed Dev-100 split, measure the C1 / C1+C2b / Micro delta, log the verdict the same day. Never bundle untested effects; never tune thresholds on Dev labels without reporting it as dev-optimistic. Optimizations are queued by expected score gain per [docs/error_analysis_fields.md](docs/error_analysis_fields.md), cheapest first.
- **Per-paper proof:** The `<model_id>_per_paper.jsonl` record (section 3.5) is mandatory for every pipeline — corpus averages alone never justify a claim about where a method works.
- **Collapsible docs:** Long tables and per-item details go in `<details>` blocks (GitHub renders them folded). The overview table stays visible; each row's explanation folds open underneath.
- **Per-model description:** Every model gets exactly 3 sentences (what it does, key mechanism or result, takeaway) plus a direct script link in the fixed phrasing `find the full script here: <link>`.
- **Compute context:** Every phase states where it runs — local CPU in seconds (base models, no provider needed) or provider + exact model id + API/GPU need (all LLM runs).
