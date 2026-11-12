# AGENTS.md – Engineering & Workflow Guidelines

This document defines the mandatory coding, architectural, and workflow standards for the **BiodiversityExtraction2026-2027** repository (TU Berlin, *Advanced Natural Language Processing*, Shared Task C1 + C2b).

---

## 1. Collaboration & Pacing Rules
- **Step-by-Step Workflow:** Never generate multiple unrequested models or jump ahead in the roadmap. Build, inspect, and verify **one single component or model at a time** in agreement with the user.
- **Branch Discipline:** Work happens on dedicated feature/experiment branches (e.g., `lennard_testing`). Never push experimental code directly to `main`.
- **Data Privacy & Git Hygiene:** Course dataset files (`data/gold/`, `data/raw_pdfs/`, `data/parsed_text/`), generated model outputs (`outputs/`), and personal checklists (`AUFGABENLISTE.md`) must remain excluded via `.gitignore` and must never be committed to this repository. Datasets live in the **separate Gitea-only data repo** [`uni/BiodiversityExtraction2026-2027-data`](http://100.69.68.23:3000/uni/BiodiversityExtraction2026-2027-data) (PDFs via Git-LFS; tailnet-only, **never push data to GitHub** — no LFS there and course data protection forbids it). To work locally: clone the data repo next to the code repo and symlink `ln -s ../BiodiversityExtraction2026-2027-data/data data`. `data/parsed_text/` is a regenerable cache and stays out of both repos.
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
5. **`<model_id>_submission.zip`**
   - CodaBench-ready ZIP archive containing `predictions.jsonl` at its root.
6. **Central Leaderboard Update (`outputs/all_models_comparison.md`)**
   - Automatically updated after every run to maintain a single source of truth comparing all evaluated models by C1 Macro-F1, C1+C2b Macro-F1, Micro-F1, runtime, seed, and Git commit.
