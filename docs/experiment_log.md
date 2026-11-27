# Experiment Log — Architecture Axis & Prompt Optimization

Lab notebook with grouped experiment tables (variant × delta vs. baseline). Story: `README.md`.
Single-run numbers: `docs/experiment_matrix.md`.

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

### 3e. Cohesive `--v4-prompt` Matrix on SAIA & HU (`bio_v4_*`, 2026-11-24 – 2026-11-25)

| Variant | Provider / Model | C1 Macro-F1 | C1+C2b | Micro-F1 | Finding |
|---|---|---:|---:|---:|---|
| `AutoParser + --ctaxa --v4` (16k) | `saia` / `deepseek-4.1` | 0.5953 | **0.4744** | **0.5019** | **All-time best C1+C2b & first >0.50 Micro** (`c2b_macro_f1 = 0.3942`, beats Minh Anh's clean `v4` `0.3733`) |
| `AutoParser + --ctaxa --v4 --full-text` | `saia` / `deepseek-4.1` | **0.6026** | 0.4625 | 0.4874 | `+0.0073` C1 vs. 16k; beats `Docling + v4` by **`+0.0261` C1 / `+0.0347` Micro** |
| `AutoParser + --ctaxa --v4 --bilingual-knn` | `saia` / `deepseek-4.1` | 0.6010 | 0.4646 | 0.4920 | `+0.0057` C1 vs. English TF-IDF |
| `AutoParser + --ctaxa --v4 --full --biknn` | `saia` / `deepseek-4.1` | 0.6016 | 0.4634 | 0.4706 | stable `0.60+` C1 |
| `Docling + --ctaxa --v4 --full-text` | `saia` / `deepseek-4.1` | 0.5765 | 0.4421 | 0.4527 | **Proof:** `AutoParser` beats `Docling` by **`+2.61` C1** under identical `v4` prompt |
| `AutoParser + --ctaxa --v4` (16k) | `hu` / `hu-reasoning` | 0.5571 | 0.4176 | 0.4279 | baseline |
| `AutoParser + --ctaxa --v4 --full-text` | `hu` / `hu-reasoning` | 0.5773 | 0.4128 | 0.4182 | `+0.0202` C1 vs. 16k; beats `Docling + v4` by **`+0.0224` C1 / `+0.0549` Micro** |
| `AutoParser + --ctaxa --v4 --bilingual-knn` | `hu` / `hu-reasoning` | 0.5671 | **0.4214** | **0.4292** | `+0.0100` C1 vs. English TF-IDF |
| `AutoParser + --ctaxa --v4 --full --biknn` | `hu` / `hu-reasoning` | **0.5815** | 0.4188 | 0.4197 | best `v4` C1 on HU (`+0.0244` over 16k `v4`) |
| `Docling + --ctaxa --v4 --full-text` | `hu` / `hu-reasoning` | 0.5549 | 0.3816 | 0.3633 | **Proof:** `AutoParser` beats `Docling` by **`+2.24` C1** on HU too |

Full 44-run per-field table: `docs/experiment_matrix.md`.

### 3f. Catch-up on Complete Caches, Fair Docling Rerun & Minh Repro (2026-11-26 – 2026-11-27)

**Setup.** After the Grünau batch all Dev parser caches are complete on `lenlenovo`: Marker 100/100, Docling Light 100/100, Minh's full Docling JSON (OCR on) 100/100, GROBID 99/100 (V2BVMBPF has no text anywhere). Queue: `scripts/run_catchup_queue.sh` + `scripts/run_minj_after_catchup.sh` (tmux `bio_catchup` / `bio_minj`, logs in `outputs/catchup_queue/`). `--minh-json` reads her Docling JSON in reading order, starts a section at every `section_header`, adds `[page N]` markers and skips footers. Everything downstream (16k budget, `--clean-taxa`, `--v4-prompt`) stays identical to the AutoParser run, so the parser text is the only difference.

**Results (Dev-100, C1 / C1+C2b / Micro; Δ against the row's immediate baseline):**

| Variant | DeepSeek 4.1 (SAIA) | Qwen3.6-27B (HU) | Finding |
|---|---|---|---|
| AutoParser `--clean-taxa`, no v4 (new no-v4 reference) | 0.5883 / 0.4400 / 0.4426 | 0.5689 / 0.4047 / 0.4083 | isolated clean-taxa run (26 Nov) |
| `+ --v4-prompt` | 0.5953 / 0.4744 / 0.5019 (Δ +0.0070 / +0.0344 / +0.0593) | 0.5571 / 0.4176 / 0.4279 (Δ −0.0118 / +0.0129 / +0.0196) | v4 lifts C1+C2b and Micro on DeepSeek 4.1 only; no C1 gain on either model |
| `--clean-taxa --level-prompt` | 0.5841 / 0.4453 / 0.4402 (Δ −0.0042 / +0.0053 / −0.0024) | 0.5469 / 0.4084 / 0.4101 (Δ −0.0220 / +0.0037 / +0.0018) | **rejected**: a second call per paper, no C1 gain on either model (second test after 3d) |
| Minh's full Docling JSON `--clean-taxa --v4-prompt` (fair rerun, vs AutoParser `--clean-taxa --v4-prompt`) | 0.5883 / 0.4626 / 0.4804 (Δ −0.0070 / −0.0118 / −0.0215) | **0.5894** / 0.4233 / **0.4426** (Δ +0.0323 / +0.0057 / +0.0147) | Gap to AutoParser shrinks from 2.6 to 0.7 C1 points on DeepSeek 4.1; on Qwen3.6-27B full Docling is level or ahead (best C1 and Micro of any Qwen run) |
| `--raw-fulltext` instead of `--full-text` (both `--clean-taxa --v4-prompt`, 25 Nov) | **0.6102** / **0.4822** / 0.5006 (Δ +0.0076 / +0.0197 / +0.0132) | 0.5458 / 0.4029 / 0.4136 (Δ −0.0315 / −0.0099 / −0.0046) | Top single run on DeepSeek 4.1, but not production-ready: no token-limit fallback (1 paper empty at the 1,048,576-token limit, 3 papers empty on Qwen). **Rejected** on Qwen3.6-27B |

**Paired bootstrap over the 100 Dev papers** (4,000 resamples, seed 42, difference A − B with 95 % CI; measures paper sampling only, not run-to-run LLM noise — every run is a single run):

| Pair (A − B) | C1 | C1+C2b | Micro |
|---|---|---|---|
| v4 − no v4, DeepSeek 4.1 | +0.0062 [−0.0431, +0.0569] | +0.0333 [−0.0043, +0.0739] | **+0.0587 [+0.0116, +0.1063]** |
| v4 − no v4, Qwen3.6-27B | −0.0120 [−0.0556, +0.0338] | +0.0122 [−0.0225, +0.0491] | +0.0190 [−0.0249, +0.0647] |
| level prompt − none, DeepSeek 4.1 | −0.0042 [−0.0324, +0.0219] | +0.0050 [−0.0197, +0.0315] | −0.0024 [−0.0365, +0.0270] |
| level prompt − none, Qwen3.6-27B | −0.0214 [−0.0558, +0.0127] | +0.0042 [−0.0204, +0.0301] | +0.0020 [−0.0337, +0.0370] |
| full Docling − AutoParser, DeepSeek 4.1 | −0.0071 [−0.0382, +0.0257] | −0.0127 [−0.0498, +0.0254] | −0.0218 [−0.0726, +0.0303] |
| full Docling − AutoParser, Qwen3.6-27B | +0.0321 [−0.0050, +0.0696] | +0.0054 [−0.0297, +0.0392] | +0.0142 [−0.0215, +0.0499] |
| `--raw-fulltext` − `--full-text`, DeepSeek 4.1 | +0.0079 [−0.0322, +0.0484] | +0.0190 [−0.0142, +0.0512] | +0.0128 [−0.0352, +0.0576] |
| `--raw-fulltext` − `--full-text`, Qwen3.6-27B | −0.0318 [−0.0722, +0.0052] | −0.0103 [−0.0521, +0.0239] | −0.0049 [−0.0449, +0.0323] |

Only the v4 Micro gain on DeepSeek 4.1 has a CI that excludes zero. Within our own top variants (raw-fulltext vs. `--full-text` vs. 16k, `--bilingual-knn`, full Docling vs. AutoParser) no C1 difference is significant.

**The 26 Nov redos did not recompute.** The catch-up queue launched 8 redos (v4 family on AutoParser and `marker_ev_ctaxa`, both models). Each ended after 2–3 s with `Papers: 100 (resumed: 100)` and 0 papers processed: `llm_rag_sections.py` resumes from `<run_id>_checkpoint.jsonl`, and the queue never removed the finished checkpoints, so the runs reloaded their old predictions and re-saved the old scores (their `run_info.json` now also shows a 1–3 s runtime instead of the real one; the redos fix both). Consequence: the AutoParser v4 numbers from 24–25 Nov (`autoparser_ctaxa_v4`, `_full_ctaxa_v4`, both `_biknn`) and both Marker `ev_ctaxa` numbers are still from **before** the complete caches (same for the OFAT ablations in 3b–3d). Fix: `scripts/run_redo_queue.sh` archives each run directory to `outputs/redo_backup/`, deletes the completed checkpoint and requests all 100 papers again (a partial checkpoint from a crash is resumed, not deleted). Re-queued on 27 Nov 02:12 UTC in tmux `bio_redo_saia` / `bio_redo_hu`, 5 runs per model: v4 16k, `--full-text`, `--bilingual-knn`, `--full-text --bilingual-knn`, Marker `--ev --ctaxa`.

**Redo results (finished 27 Nov 04:31 UTC SAIA / 09:56 UTC HU, 10/10 done markers, 0 errors).** Δ against the pre-cache number of the identical config:

| Redo run | DeepSeek 4.1 (SAIA) Δ C1 / C2b / Micro | Qwen3.6-27B (HU) Δ C1 / C2b / Micro |
|---|---|---|
| `autoparser_ctaxa_v4` (16k) | 0.5847 (−1.1 / +0.4 / −0.3) | 0.5700 (+1.3 / −0.5 / ±0.0) |
| `autoparser_full_ctaxa_v4` | 0.5759 (−2.7 / −1.3 / −2.4) | 0.5602 (−1.7 / −0.8 / +0.8) |
| `autoparser_ctaxa_v4_biknn` | **0.6102** (+0.9 / +0.7 / −0.2) | 0.5604 (−0.7 / −0.5 / −0.3) |
| `autoparser_full_ctaxa_v4_biknn` | 0.5987 (−0.3 / −0.1 / +0.3) | 0.5562 (−2.5 / −1.7 / −1.4) |
| `marker_ev_ctaxa` | 0.5767 (−2.7 / −1.5 / −1.6) | 0.5484 (−1.4 / +1.6 / +2.8) |

*(values in points, e.g. −2.7 = −0.0267; absolute values: v4 16k `0.5847/0.4784/0.4994`, full `0.5759/0.4497/0.4632`, biknn `0.6102/0.4715/0.4901`, full+biknn `0.5987/0.4621/0.4733`, marker_ev `0.5767/0.4339/0.3967` on DeepSeek 4.1; HU: `0.5700/0.4122/0.4279`, `0.5602/0.4053/0.4257`, `0.5604/0.4166/0.4264`, `0.5562/0.4016/0.4059`, `0.5484/0.3966/0.3970`.)*

**Takeaways.** (1) **Run-to-run variance is the largest effect we have measured:** identical configs (only cache state and day changed) moved C1 by −2.7 to +0.9 points on DeepSeek 4.1 and −2.5 to +1.3 on Qwen3.6-27B. Single-run OFAT deltas below ~2 C1 points should not be interpreted as real effects — this retroactively weakens several small "wins/losses" logged in 3b–3e. A proper variance test with fresh requests stays on the roadmap. (2) The ranking changes materially: `--v4 --bilingual-knn` (16k) is now the best **confirmed** run at C1 0.6102 / 0.4715 / 0.4901, tying `--raw-fulltext` (0.6102) on C1 while keeping a split fallback; rawft keeps the best C1+C2b (0.4822) and Micro (0.5006) but stays unconfirmed (pre-cache + no token fallback). (3) `--full-text` loses its edge after the redo (0.5759) — its old +0.7 C1 over 16k was variance. (4) Still pre-cache (not redone): all parser baselines from 18–21 Nov, the OFAT ablations in 3b–3d, `--raw-fulltext`, and the `docling_full_ctaxa_v4` 1:1 proof pair — rerun the Docling side before quoting the "AutoParser beats Docling" gap again.

**Second redo queue (queued 27 Nov 13:09 UTC, tmux `bio_redo2_saia` / `bio_redo2_hu`, `scripts/run_redo2_queue.sh`):** the remaining claim-anchoring pre-cache pairs, run order per model: `docling_full_ctaxa_v4` (`--docling --full-text --clean-taxa --v4-prompt`) then `autoparser_ctaxa_v4_rawft` (`--auto-parser --clean-taxa --v4-prompt --raw-fulltext`). The rawft redo keeps its known token-limit handicap on purpose (1 empty paper on DeepSeek 4.1, 3 on Qwen3.6-27B) — it is a same-config variance check and confirmation of 0.6102, not a production fix; the split fallback for over-limit papers stays a separate code task. **Results pending.**

**Flag-pricing validation queue (queued 27 Nov 14:54 UTC, tmux `bio_validate_saia` / `bio_validate_hu`, `scripts/run_validate_queue.sh`, waits for redo2 on the same side):** `--clean-taxa` and `--evidence-gates` were only ever measured inside bundles. Two runs per provider price them against the confirmed recipe: (A) recipe **minus** `--clean-taxa` → run_id `autoparser_v4_biknn` (both providers, fresh dirs), (B) recipe **plus** `--evidence-gates` → `autoparser_ev_ctaxa_v4_biknn` (the gates are *not* part of the v4 recipe — v4 ships its own quote check — so B measures them as an addition). Baselines: DeepSeek 4.1 `0.6102 / 0.4715 / 0.4901`, Qwen3.6-27B `0.5604 / 0.4166 / 0.4264` (both redo-confirmed). Caveat from the redo variance finding: single-run deltas under ~2 C1 points stay inconclusive even so. **Results pending.**

**Budget + level-fold queue (queued 27 Nov, `scripts/run_validate2_queue.sh`, waits for the flag-pricing queue on the same side):** two new flags in `llm_rag_sections.py` (2026-11-27, backward compatible — defaults reproduce the old behavior and run_ids exactly): `--budget-chars N` (total prompt character budget across sections, default 16000) and `--level-fold` (injects a compact adaptation of Minh's level_v1 rules — enum definitions, base rates, main-subject rule, empty-list validity — into the v4 main prompt; no second call). Four runs, both providers, recipe + `--budget-chars 64000` (run_id suffix `_b64k`) and recipe + `--level-fold` (`_lvlf`). Rationale: Minh's repro feeds 3–4× more document content, and her field-level leads sit exactly where our budget cuts (eco_type +6 to +9 points, level 0.444 vs. 0.394); the second-call variant of the level prompt lost C1 twice, the fold is untested. **Results pending.**

**Overfitting note for the planned neighbour-based service rule (from Minh's pipeline F).** The rule ("drop a paper's ecosystem services when none of its 3 similar training papers has any") is derived from TRAIN label statistics only — legitimate supervised signal, not dev overfitting. What uses dev is (a) the OFAT adoption decision and (b) any threshold tuning. We keep the rule threshold-free (0 of 3 neighbours) to limit that surface and log it as dev-optimistic-adopted per the OFAT protocol. Minh herself marks her post rules as "chosen after seeing dev" in her README, with her eco-type rule at bootstrap 95 % robustness and her level prompt at only 73 % — the level prompt rule is therefore NOT copied despite her higher level score, only the fold of equivalent rules into our main prompt is tested.

**Neighbour service rule implemented + queued (27 Nov, `scripts/run_validate3_queue.sh`, tmux `bio_v3_*`, waits for validate2 on the same side):** new flag `--service-gate` (run_id suffix `_svcg`, commit 2026-11-27): after normalization, drop the paper's `ecosystem_service` entries when none of the k=3 retrieved training neighbours (the same few-shot neighbours already used by the prompt) has any service labels. Two runs per provider, recipe + `--service-gate`. **Results pending.**

**Housekeeping (27 Nov):** `.gitignore` now ignores all of `outputs/*` (previously only `base_models/` and `architectures/` were covered — run-log directories such as `outputs/catchup_queue/` were untracked-but-unprotected on the server clone).

**Minh repro (0 extra requests, scored 27 Nov).** Her pipeline F (`minhanh_testing@ac98eac`) ran on `lenlenovo` in `~/Home/minhanh_repro/` with our PDFs, gold files and schemas (25 Nov, 217 fresh SAIA requests; 98 papers in one call, 2 in parts; verbatim checker 3–60 words; level prompt; eco-type rule dropped 10 entries; service rule dropped 13 entries in 12 papers). Her dev selection is exactly our `dev_100_ids`. Scored with our evaluator on the identical 100 papers: **C1 0.6508 / C1+C2b 0.5389 / Micro 0.5032** (her own scorer gives the same C1; her "c2b" 0.4534 is a different definition). Against our best single run (`autoparser_ctaxa_v4_rawft`): C1 **+0.0403 [+0.0012, +0.0796]**, C1+C2b **+0.0562 [+0.0113, +0.1045]**, Micro +0.0027 [−0.0331, +0.0355].

| Field F1 | Minh's pipeline F | Our best (`autoparser_ctaxa_v4_rawft`) |
|---|---:|---:|
| habitat | 0.790 | 0.792 |
| biodiversity_level | 0.444 | 0.394 |
| ecosystem_type.category | 0.719 | 0.626 |
| ecosystem_type.term | 0.620 | 0.558 |
| taxa.species_group | 0.680 | 0.680 |
| taxa.scientific_name | 0.327 | 0.380 |
| taxa.german_name | 0.347 | 0.356 |
| ecosystem_service.category | 0.522 | 0.353 |
| ecosystem_service.term | 0.400 | 0.200 |

She wins on `ecosystem_service` and `ecosystem_type`; we are ahead only on taxa names. Two open points: (1) her README reports C1 0.6881 for the same defaults, the repro gives 0.6508 — unexplained, so ask for her original `predictions.jsonl` and score it before quoting either number; (2) her neighbour-based service rule (drop a paper's services when none of its similar training papers has any) was never tested in our pipeline — it is not implemented there, and our `ecosystem_service.term` precision is 0.125.

**Commands.**

```bash
# fair Docling rerun on Minh's JSON cache (26 Nov) and the real redos (27 Nov)
.venv/bin/python -u src/models/architectures/llm_rag_sections.py --provider saia --model deepseek-v4.1-flash-1017 --rpm 20 --minh-json --clean-taxa --v4-prompt
.venv/bin/python -u src/models/architectures/llm_rag_sections.py --provider hu --model hu-reasoning --rpm 15 --minh-json --clean-taxa --v4-prompt
ssh lenlenovo 'tmux new -d -s bio_redo_saia "bash /tmp/run_redo_queue.sh saia"'   # and ... hu
# offline scoring + paired bootstrap (read-only, 0 API calls), from the repo root on lenlenovo
.venv/bin/python scripts/paired_bootstrap.py \
  --pair llm_rag_sections_hu_hu-reasoning_minhjson_ctaxa_v4 llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4 \
  --external minh=$HOME/Home/minhanh_repro/runs/20261125-180748_full_text_dev \
  --pair minh llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_rawft
```

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

### 5. Roadmap status (2026-11-27, see `notes/next_steps_preprocessing.md`)

1. **Phase 1 — Adaptive Parser Router (`--auto-parser`): done** (offline 2-rule router + live `--auto-parser` flag, Dev-100 tested).
2. **Phase 2 — Long-Paper Map-Reduce Summarization:** open (chunked evidence extraction + RAG consolidation for over-budget papers).
3. **Phase 3 — Stepwise Integration of `minhanh_testing` (`Pipeline F`): done** (single blocks rejected, cohesive v4 adopted; Minh repro scored, see 3f).
4. **Phase 4 — Final Cross-Parser / Cross-Model Consensus Ensemble & Full-Set Run:** open (waits for the true redos of 27 Nov, see 3f).
5. **Future hypotheses (updated 2026-11-27):** neighbour-based service rule from Minh's pipeline F (never tested in ours; our `ecosystem_service.term` precision is 0.125);
   level decision folded into one prompt instead of a second call (the second call lost C1 on both models, 3d and 3f); cheap PyMuPDF pre-filter router
   over all five parsers (PyMuPDF/GROBID/Marker/Docling Light/full Docling — agreement vs. oracle
   after the fair rerun); translated German taxa names accepted
   with English source quotes as evidence (ger_name 0.25, 45 papers never right).
