# Experiment Log — Architecture Axis & Prompt Optimization

Central, committed record of every model/prompt change and its measured result
(course requirement: document how predictions perform). Companion artifacts:
`outputs/<category>/<model_id>/` per run (run_info.json logs the full config incl.
all experiment flags; metrics.json the scores; predictions.jsonl the raw output)
and `outputs/all_models_comparison.md` (auto-updated leaderboard).

## Context

- Task: Shared Task C1 (5 core fields) + C2b (Biodiversity), 100-paper Dev benchmark.
- Providers: `hu-reasoning` (Qwen3.6-27B, HU bridge) — fast comparison base;
  `mistral-large-4-1t-a52b` (SAIA) — second provider.
- Compute: all benchmark runs on the `lenlenovo` home server since 2026-11-12.

## Architecture Axis (monolithic vs. chunking)

| Run | Model | C1 Macro-F1 | C1+C2b | Micro | Note |
|---|---|---|---|---|---|
| Architecture 1 (monolithic full-text prompt) | hu-reasoning | **0.5754** | 0.4149 | 0.4515 | current best overall |
| Architecture 1 (monolithic full-text prompt) | mistral-large (SAIA) | 0.5143 | 0.3751 | 0.3834 | — |
| Architecture 2 / Pipeline B v1 (4 chunks, quote gate, vote merge) | hu-reasoning | 0.4318 | 0.2885 | 0.3152 | term F1 = 0.000 (see E1) |
| Pipeline B v1 | mistral-large (SAIA) | pending | — | — | stuck at 45/100 on GWDG 504 outage |

Finding: chunking trades recall for the quote gate. `ecosystem_type.term` collapses
to 0.000 in B (free-text term quotes die in strict enum normalization) while A1
reaches 0.49 there — that single field explains most of B's gap to A1.

## Prompt-Optimization Experiments (Pipeline B on hu-reasoning)

| Variant | Change vs. v1 | C1 Macro-F1 | C1+C2b | Micro | Status |
|---|---|---|---|---|---|
| v1 | baseline | 0.4318 | 0.2885 | 0.3152 | done |
| v2 `--field-guidelines` | thorough per-field requirement descriptions (guideline-paper insight) | **0.4565** | **0.3339** | **0.3566** | done — FPs halved: taxa.species_group 144→76, biodiversity_level 49→23, habitat 43→23 |
| E1 `--term-enum` | full 46+67 term enums in prompt + relaxed term matching (substring/fuzzy 0.60) | pending | — | — | running since 2026-11-12 20:45 UTC |
| E2 `--clean-text` | strip URLs/DOIs/emails + truncate at references heading | pending | — | — | queued after E1 |
| E3 `--translate-hint` | English-internal reasoning, German quotes/labels | pending | — | — | queued after E2 |

Runner: `scripts/run_hu_experiments.sh` (sequential, checkpoint-resumable).
Each variant gets its own model_id suffix (e.g. `..._fg_term`) → separate
checkpoint, artifacts, and leaderboard row.

## Gold-Label Audit (2026-11-12, 50 Dev papers manually reviewed)

76% clean / 24% questionable. Key patterns → optimization backlog:
1. ~10-15% "breadth documents" (strategies, lists, books) carry all-habitat/all-level
   labels — effectively unwinnable; natural score ceiling, not a model error.
2. Systematic annotator convention: floodplain (Aue) studies → ecosystem_type
   II/"Fließende Gewässer" even for forest biota (v3 guideline candidate).
3. Gold format noise under exact string matching: author epithets
   ("Adonis aestivalis L."), "Fucus spp" w/o period, combined German names —
   prediction-side micro-normalization is a cheap lever.
Full report: `outputs/error_analysis/gold_audit_report.md` (+ per-batch files).

## Full-Corpus Base-Model Comparison (2026-11-12)

All 9 base models on Dev-100 AND Train-908 (in-sample caveat documented).
Dev columns reproduce the official leaderboard (sanity check passed).
kNN shows the largest overfitting gap (0.479 dev vs 0.759 train); ensemble stays
Dev-best (0.526). Full table: `src/models/base_models/full_corpus_comparison.py`
(EXECUTIVE SUMMARY) and `outputs/base_models/full_corpus_comparison/`.

## Operational Notes

- SAIA outage since 2026-11-12 ~20:00 UTC: 504 "Inference registry error";
  watchdog auto-retries every 10 min, then chains SAIA v1 → v2 automatically.
- All server runs: tmux (`bio_saia`, `bio_hu`, `bio_exp`) + cron watchdogs;
  VPN model-only routing; results artifacts under `outputs/architectures/`.
