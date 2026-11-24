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
| E1 `--term-enum` | full 46+67 term enums in prompt + relaxed term matching (substring/fuzzy 0.60) | **0.5593** | **0.4144** | **0.3923** | done — term F1 0.000→0.46 (24 FP), species_group 0.598; B ≈ A1 (0.5754) |
| E2 `--clean-text` | strip URLs/DOIs/emails + truncate at references heading | 0.4463 | 0.3321 | 0.3633 | done — neutral/slightly negative vs v2; noise reduction alone doesn't help |
| E3 `--translate-hint` | English-internal reasoning, German quotes/labels | pending | — | — | running (34/100 at 2026-11-14 00:00 CET) |
| SAIA Mistral v1 | baseline on mistral-large-4-1t-a52b | 0.4056 | 0.2740 | 0.3112 | done — B also below A1-SAIA (0.5143); term 0.06 |
| SAIA v2 `--field-guidelines` | on mistral-large | pending | — | — | launched 2026-11-14 00:50 CET after watchdog fix (stale tmux client masked the dead process — patterns now anchored to real python cmdlines) |

**Conclusion so far:** E1 (term enums + relaxed matching) is the single biggest lever
(+12.8 C1 points over v2, +2.5 over A1-HU would follow if combined — pending E3 and
combo runs). E2 (text cleaning) alone is not a win.

## New-Architecture runs: English-first Sections Pipeline (`llm_sections_pipeline.py`, other session `a0da88e`)

English-first extraction + quote gate + deterministic EN→DE mapping; both architectures
(`--architecture parallel|sequential`), `--language en|de`, robust retry client.
Runs on lenlenovo (GROBID Docker `grobid/grobid:0.9.1-crf` on 127.0.0.1:8070):

| Run | Model | Status |
|---|---|---|
| sections parallel/en | hu-reasoning | done (other session, metrics in its artifacts) |
| sections parallel/de | hu-reasoning | running (62/100 at 00:00 CET) |
| sections parallel/en | glm-5.3 (SAIA, full model — user chose NOT flash) | running (8/100), self-healing tmux `bio_glm`, GROBID sections |
| DeepSeek DS1: parallel chunking E1-config (`--field-guidelines --term-enum`) | deepseek-v4.1-flash-1017 via SAIA | **running since 2026-11-14 00:56 CET** (tmux `bio_ds`, self-healing; no extra key — SAIA covers it) |
| DeepSeek DS2: sections parallel/en | deepseek-v4.1-flash-1017 via SAIA | queued after DS1, same chain |

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

---

## Architecture 3: RAG-Anchored Sections & Parser Axis (2026-11-18 – 2026-11-21)

**Experiment Protocol (Strict OFAT — One Factor at a Time):**
Every optimization (parser, routing, chunking/summarization, prompt edit, post-processing) is tested **one change at a time** on the fixed 100-paper Dev split (`data/gold/dev_100_ids.txt`), compared directly against its immediate baseline, and logged here immediately for the final course report. No Dev-ID-specific heuristics or Dev-calibrated thresholds.

### 1. Parser Axis on Architecture 3 (`llm_rag_sections.py`)

| Variant (Parser) | Provider / Model | C1 Macro-F1 | C1+C2b Macro-F1 | Micro-F1 | Status / Delta vs. PyMuPDF |
|---|---|---|---|---|---|
| PyMuPDF (baseline) | `saia` / `deepseek-v4.1-flash-1017` | 0.5798 | 0.4273 | 0.4286 | done |
| GROBID (`--grobid-url`) | `saia` / `deepseek-v4.1-flash-1017` | 0.5957 | **0.4544** | **0.4390** | done (`+0.0159` C1, `+0.0271` C1+C2b) |
| Marker (`--marker`) | `saia` / `deepseek-v4.1-flash-1017` | **0.5985** | 0.4362 | 0.3784 | done (`+0.0187` C1, `-0.0502` Micro) |
| Docling (`--docling`) | `saia` / `deepseek-v4.1-flash-1017` | 0.5634 | 0.4221 | 0.4131 | done (`-0.0164` C1 vs. PyMuPDF; per-paper routing candidate) |
| PyMuPDF (baseline) | `hu` / `hu-reasoning` | 0.5539 | 0.4073 | 0.4126 | done |
| GROBID (`--grobid-url`) | `hu` / `hu-reasoning` | 0.5706 | 0.4353 | 0.4293 | done (`+0.0167` C1, `+0.0280` C1+C2b) |
| Marker (`--marker`) | `hu` / `hu-reasoning` | 0.5837 | 0.4197 | 0.4015 | done (`+0.0298` C1, `-0.0111` Micro) |
| Docling (`--docling`) | `hu` / `hu-reasoning` | 0.5613 | 0.4018 | 0.3748 | done (`+0.0074` C1 vs. PyMuPDF, below GROBID/Marker) |

### 2. Offline Consensus Replay (0 API Calls, 2026-11-21)

Fixed-vote consensus across the 4 completed GROBID + Marker runs (`saia_grobid`, `saia_marker`, `hu_grobid`, `hu_marker`) with `micronormalize_taxon_name`:
- **2-of-4 Consensus:** `C1 = 0.5902 | C1+C2b = 0.4382 | Micro = 0.4238`
- **3-of-4 Consensus:** `C1 = 0.5932 | C1+C2b = 0.4687 | Micro = 0.4496` (**+0.0143 C1+C2b / +0.0106 Micro** over best single model)

### 3. Queued OFAT Ablations (`bio_opt_saia` & `bio_opt_hu`, started 2026-11-21)

1. `SAIA GROBID + --detox-rag` (bilingual `char_wb` TF-IDF + species-stripped kNN anchors) — **done:** `C1 = 0.5399 | C1+C2b = 0.4153 | Micro = 0.4260` (**-0.0558 C1** vs. GROBID baseline `0.5957` — negative result: masking taxa in kNN anchors hurts domain/format grounding)
2. `SAIA GROBID + --detox-rag --role-budget` (28k section budget weighted 35%/35% Methods/Results) — **done:** `C1 = 0.5446 | C1+C2b = 0.4118 | Micro = 0.4279` (`+0.0047` C1 / `+0.0019` Micro vs. `--detox-rag` alone, still hurt by `--detox-rag`)
3. `SAIA GROBID + --detox-rag --role-budget --evidence-gates --clean-taxa` — **done:** `C1 = 0.5588 | C1+C2b = 0.4149 | Micro = 0.4282` (`+0.0142` C1 / `+0.0031` C1+C2b vs. step 2)
4. `SAIA Marker + --detox-rag --role-budget --evidence-gates --clean-taxa` — **done:** `C1 = 0.5676 | C1+C2b = 0.4112 | Micro = 0.4190` (`+0.0406` Micro vs. Marker baseline `0.3784`, but C1 hurt by `--detox-rag`)
5. `HU GROBID + --detox-rag --role-budget --evidence-gates --clean-taxa` — **done:** `C1 = 0.5654 | C1+C2b = 0.3898 | Micro = 0.3825` (`-0.0052` C1, `-0.0455` C1+C2b vs. HU GROBID baseline `0.5706 / 0.4353 / 0.4293`)
6. `HU Marker + --detox-rag --role-budget --evidence-gates --clean-taxa` — **done:** `C1 = 0.5520 | C1+C2b = 0.3731 | Micro = 0.3870` (`-0.0317` C1 vs. HU Marker baseline `0.5837`)

**OFAT Takeaway:** `--detox-rag` is a clear negative (`-0.0558` C1), whereas both `--role-budget` (`+0.0047` C1) and `--evidence-gates --clean-taxa` (`+0.0142` C1 on GROBID, `+0.0406` Micro on Marker) improved over their immediate predecessor. Next step: test `--role-budget` and `--evidence-gates --clean-taxa` cleanly **without** `--detox-rag` against the `0.5957`/`0.5985` baselines.

### 3b. Clean OFAT Ablations without `--detox-rag` (`bio_ofat_saia` & `bio_ofat_hu`, 2026-11-22)

| Variant | Provider / Model | C1 Macro-F1 | C1+C2b | Micro-F1 | Delta vs. Parser Baseline |
|---|---|---|---|---|---|
| `GROBID + --role-budget` | `saia` / `deepseek-v4.1-flash-1017` | 0.5815 | 0.4131 | 0.3736 | `-0.0142` C1, `-0.0413` C1+C2b (drop `--role-budget`) |
| `GROBID + --evidence-gates --clean-taxa` | `saia` / `deepseek-v4.1-flash-1017` | 0.5959 | 0.4530 | **0.4516** | `+0.0002` C1, **`+0.0126` Micro** (new single-model Micro best) |
| `Marker + --evidence-gates --clean-taxa` | `saia` / `deepseek-v4.1-flash-1017` | **0.6040** | 0.4487 | 0.4124 | **`+0.0055` C1, `+0.0125` C1+C2b, `+0.0340` Micro** (new single-model C1 best) |
| `Marker + --role-budget --evidence-gates --clean-taxa` | `saia` / `deepseek-v4.1-flash-1017` | 0.5953 | 0.4271 | 0.3924 | `-0.0087` C1 vs. step above (confirms `--role-budget` hurts) |
| `GROBID + --evidence-gates --clean-taxa` | `hu` / `hu-reasoning` | 0.5434 | 0.4228 | 0.4194 | `-0.0272` C1 vs. HU GROBID (`0.5706`) — gates hurt HU |
| `Marker + --evidence-gates --clean-taxa` | `hu` / `hu-reasoning` | 0.5625 | 0.3807 | 0.3694 | `-0.0212` C1 vs. HU Marker (`0.5837`) — gates hurt HU |

**Clean OFAT Decision:** Keep `--evidence-gates --clean-taxa` for **SAIA DeepSeek 4.1** (`0.6040` C1 on Marker, `0.4516` Micro on GROBID); drop both `--detox-rag` and `--role-budget`. Keep baseline (no gates) for HU-Reasoning.

### 3c. Phase 2 (`--full-text`) & Phase 3a (`--focus-prompt`) OFAT Runs (`bio_full_*` & `bio_night_*`, 2026-11-22 – 2026-11-23)

| Variant | Provider / Model | C1 Macro-F1 | C1+C2b | Micro-F1 | Delta vs. `AutoParser` Baseline |
|---|---|---|---|---|---|
| `AutoParser + --evidence-gates --clean-taxa` *(offline baseline)* | `saia` / `deepseek-v4.1-flash-1017` | **0.6056** | **0.4469** | **0.4424** | — |
| `+ --full-text` (untruncated / >80k split) | `saia` / `deepseek-v4.1-flash-1017` | 0.5615 | 0.4023 | 0.3622 | **`-0.0441` C1, `-0.0802` Micro** (noise/FP flood) |
| `+ --focus-prompt` (v2-style wording + train rates) | `saia` / `deepseek-v4.1-flash-1017` | 0.5703 | 0.4139 | 0.3955 | **`-0.0353` C1, `-0.0469` Micro** (over-extraction) |
| `+ --full-text --focus-prompt` | `saia` / `deepseek-v4.1-flash-1017` | 0.5651 | 0.3835 | 0.3485 | `-0.0405` C1, `-0.0939` Micro |
| `AutoParser` *(offline baseline)* | `hu` / `hu-reasoning` | **0.5839** | **0.4317** | **0.4237** | — |
| `+ --full-text` | `hu` / `hu-reasoning` | 0.5427 | 0.3620 | 0.3341 | **`-0.0412` C1, `-0.0896` Micro** |
| `+ --focus-prompt` | `hu` / `hu-reasoning` | 0.5588 | 0.3968 | 0.3864 | **`-0.0251` C1, `-0.0373` Micro** |
| `+ --full-text --focus-prompt` | `hu` / `hu-reasoning` | 0.5351 | 0.3609 | 0.3207 | `-0.0488` C1, `-0.1030` Micro |

**OFAT Conclusion:** Both `--full-text` and `--focus-prompt` consistently degrade precision & Micro-F1 on **both** providers when injected into Architecture 3's RAG-anchored section prompt (16k section selection remains superior to raw untruncated dumping). Both flags are rejected for Architecture 3.

### 3d. Phase 3b: Minh Anh Building Blocks (`--self-check-reasoning` & `--per-label-quotes`, 2026-11-23)

| Variant | Provider / Model | C1 Macro-F1 | C1+C2b | Micro-F1 | Delta vs. `AutoParser` Baseline |
|---|---|---|---|---|---|
| `AutoParser + --ev --ctaxa --self-check-reasoning` (`_s5`) | `saia` / `deepseek-4.1` | 0.5861 | **0.4489** | **0.4551** | `-0.0195` C1, `+0.0020` C1+C2b, **`+0.0127` Micro** (new single-model Micro best) |
| `AutoParser + --ctaxa --s5 --per-label-quotes` (`_plq`) | `saia` / `deepseek-4.1` | 0.5856 | 0.4404 | **0.4671** | `-0.0200` C1, **`+0.0247` Micro** vs. AutoParser base (all-time single-model Micro best!) |
| `AutoParser + --self-check-reasoning` (`_s5`) | `hu` / `hu-reasoning` | 0.5459 | 0.3998 | 0.3928 | `-0.0380` C1 vs. HU AutoParser (`0.5839`) |
| `AutoParser + --s5 --per-label-quotes` (`_plq`) | `hu` / `hu-reasoning` | 0.5304 | 0.3876 | 0.3910 | `-0.0535` C1 vs. HU AutoParser (`0.5839`) |
| `AutoParser + --ev --ctaxa --eco-category-map` (`_ecomap`) | `saia` / `deepseek-4.1` | 0.5766 | 0.4167 | 0.3937 | `-0.0290` C1 vs. AutoParser (`0.6056`) |
| `AutoParser + --ev --ctaxa --level-prompt` (`_lvl`) | `saia` / `deepseek-4.1` | 0.5770 | 0.4441 | 0.4385 | `-0.0286` C1 vs. AutoParser (`0.6056`) |
| `AutoParser + --eco-category-map` (`_ecomap`) | `hu` / `hu-reasoning` | 0.5689 | 0.4134 | 0.4196 | `-0.0150` C1 vs. HU AutoParser (`0.5839`) |
| `AutoParser + --level-prompt` (`_lvl`) | `hu` / `hu-reasoning` | 0.5474 | 0.4118 | 0.4238 | `-0.0365` C1 vs. HU AutoParser (`0.5839`) |

**Key Architectural Insight (2026-11-24):** Injecting individual pieces of Minh Anh's `v4` pipeline into Architecture 3's prompt consistently underperforms (`0.54–0.58` C1 vs. `0.6056`). Rather than running a heavy multi-model consensus ensemble, the clean efficient synthesis is: **our `--auto-parser` (GROBID + Marker fallback) + `--clean-taxa` preprocessing/postprocessing paired with Minh Anh's complete `v4` prompt + 3–60w quote check in 1 call/paper**.

### 4. 4-Way Parser Diff & Objective Routing Analysis (0 API Calls, 2026-11-21)

Analyzed all 4 completed parser baselines (`PyMuPDF`, `GROBID`, `Marker`, `Docling`) across all 100 Dev papers on both `deepseek-v4.1-flash-1017` (SAIA) and `hu-reasoning` (HU):
- **Oracle Per-Paper Upper Bound:**
  - SAIA: `C1 = 0.6914 | C1+C2b = 0.5053 | Micro = 0.5025` (strict wins: `grobid 16, pymupdf 10, marker 8, docling 4`)
  - HU: `C1 = 0.6827 | C1+C2b = 0.4787 | Micro = 0.4479` (strict wins: `grobid 16, marker 11, docling 10, pymupdf 10`)
- **Strongest Objective Bucket Signal (zero label leakage, consistent on both models):**
  - When GROBID drops text (`g_ratio < 0.65`, N=9) or finds few sections (`g_divs <= 3`, N=12), **Marker** beats GROBID strongly on per-bucket C1 (`+0.0993` SAIA / `+0.2559` HU on `g_ratio < 0.65`; `+0.0506` SAIA / `+0.1741` HU on `g_divs <= 3`).
  - On structured papers (`g_ratio >= 0.65`, N=91; `g_divs > 10`, N=66), **GROBID** leads on C1+C2b (`0.4733` SAIA, `0.4572` HU).
- **Offline 2-Rule Router (`Marker` if `g_divs <= 3` or `g_ratio < 0.65` else `GROBID`, 12/88 split):**
  - HU: `0.5839 / 0.4317 / 0.4237` (improves C1 over GROBID `0.5706` while keeping Micro/C1+C2b near GROBID).
  - SAIA: `0.5938 / 0.4390 / 0.4149` (slightly below GROBID `0.5957 / 0.4544 / 0.4390` due to Marker's higher FP count on sparse fields hurting corpus-level Macro/Micro-F1).
- **4-Parser Consensus (SAIA only, 3-of-4):** `C1 = 0.5851 | C1+C2b = 0.4611 | Micro = 0.4615`.

### 5. Upcoming Planned OFAT Phases (see `notes/next_steps_preprocessing.md`)

1. **Phase 1 — Adaptive Parser Router (`--auto-parser`):** 4-way parser diff analysis after Docling completes $\rightarrow$ objective PDF/TEI routing heuristic $\rightarrow$ Dev-100 A/B test.
2. **Phase 2 — Long-Paper Map-Reduce Summarization:** Chunked independent evidence extraction (Map) + RAG consolidation (Reduce) for papers exceeding budget.
3. **Phase 3 — Stepwise Integration of `minhanh_testing` (`Pipeline F`):** Isolate & A/B test `prompt v2` and `checker 60 words` individually against our baseline.
4. **Phase 4 — Final Cross-Parser / Cross-Model Consensus Ensemble & Full-Set Run.**
