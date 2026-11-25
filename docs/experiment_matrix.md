# Master Experiment Matrix — Architecture 3 (Dev-100)

Vollständige Übersicht aller abgeschlossenen 100-Paper-Runs auf `data/gold/dev_100_ids.txt`,
sortiert nach **C1 Macro-F1** (offizieller Ranking-Score), inklusive aller 9 Einzelfeld-F1-Werte
und direktem Verweis auf das jeweilige Run-Verzeichnis (`outputs/architectures/<run_id>/`).

## 1. Gesamttabelle (alle abgeschlossenen 100-Paper-Runs)

| # | Modell | Parser | Änderungen / Flags | C1 | C1+C2b | Micro | Hab | BioLvl | EcoCat | EcoTerm | SpGrp | SciName | GerName | SrvCat | SrvTerm | Run-Ordner (`outputs/architectures/`) |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | SAIA / DeepSeek-4.1 | Marker | `--evidence-gates --clean-taxa` | **0.6040** | 0.4487 | 0.4124 | 0.756 | 0.449 | 0.649 | 0.494 | 0.672 | 0.260 | 0.281 | 0.342 | 0.135 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_marker_ev_ctaxa` |
| 2 | SAIA / DeepSeek-4.1 | AutoParser | `--full-text --clean-taxa --v4-prompt` | **0.6026** | 0.4625 | 0.4874 | 0.833 | 0.418 | 0.597 | 0.508 | 0.658 | 0.344 | 0.353 | 0.293 | 0.160 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_full_ctaxa_v4` |
| 3 | SAIA / DeepSeek-4.1 | AutoParser | `--full-text --clean-taxa --v4-prompt --bilingual-knn` | **0.6016** | 0.4634 | 0.4706 | 0.811 | 0.429 | 0.618 | 0.528 | 0.622 | 0.355 | 0.283 | 0.333 | 0.191 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_full_ctaxa_v4_biknn` |
| 4 | SAIA / DeepSeek-4.1 | AutoParser | `--clean-taxa --v4-prompt --bilingual-knn` | **0.6010** | 0.4646 | 0.4920 | 0.789 | 0.412 | 0.606 | 0.516 | 0.682 | 0.347 | 0.358 | 0.286 | 0.186 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_biknn` |
| 5 | SAIA / DeepSeek-4.1 | Marker | `*(baseline)*` | **0.5985** | 0.4362 | 0.3784 | 0.760 | 0.402 | 0.667 | 0.491 | 0.672 | 0.221 | 0.253 | 0.342 | 0.118 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_marker` |
| 6 | SAIA / DeepSeek-4.1 | GROBID | `--evidence-gates --clean-taxa` | **0.5959** | 0.4530 | 0.4516 | 0.774 | 0.359 | 0.667 | 0.539 | 0.640 | 0.300 | 0.341 | 0.300 | 0.157 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_grobid_ev_ctaxa` |
| 7 | SAIA / DeepSeek-4.1 | GROBID | `*(baseline)*` | **0.5957** | 0.4544 | 0.4390 | 0.788 | 0.413 | 0.636 | 0.490 | 0.652 | 0.282 | 0.337 | 0.333 | 0.159 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_grobid` |
| 8 | SAIA / DeepSeek-4.1 | AutoParser | `--clean-taxa --v4-prompt` | **0.5953** | 0.4744 | 0.5019 | 0.792 | 0.415 | 0.600 | 0.512 | 0.657 | 0.397 | 0.357 | 0.343 | 0.195 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4` |
| 9 | SAIA / DeepSeek-4.1 | Marker | `--role-budget --evidence-gates --clean-taxa` | **0.5953** | 0.4271 | 0.3924 | 0.775 | 0.409 | 0.649 | 0.503 | 0.641 | 0.252 | 0.277 | 0.261 | 0.077 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_marker_role_ev_ctaxa` |
| 10 | SAIA / DeepSeek-4.1 | AutoParser | `--evidence-gates --clean-taxa --self-check-reasoning` | **0.5861** | 0.4489 | 0.4551 | 0.757 | 0.414 | 0.638 | 0.500 | 0.622 | 0.326 | 0.310 | 0.343 | 0.131 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ev_ctaxa_s5` |
| 11 | SAIA / DeepSeek-4.1 | AutoParser | `--clean-taxa --self-check-reasoning --per-label-quotes` | **0.5856** | 0.4404 | 0.4671 | 0.778 | 0.402 | 0.599 | 0.522 | 0.627 | 0.390 | 0.342 | 0.210 | 0.094 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_s5_plq` |
| 12 | HU / hu-reasoning | Marker | `*(baseline)*` | **0.5837** | 0.4197 | 0.4015 | 0.788 | 0.405 | 0.621 | 0.489 | 0.616 | 0.264 | 0.246 | 0.240 | 0.108 | `llm_rag_sections_hu_hu-reasoning_marker` |
| 13 | HU / hu-reasoning | AutoParser | `--full-text --clean-taxa --v4-prompt --bilingual-knn` | **0.5815** | 0.4188 | 0.4197 | 0.824 | 0.400 | 0.645 | 0.477 | 0.562 | 0.297 | 0.203 | 0.219 | 0.143 | `llm_rag_sections_hu_hu-reasoning_autoparser_full_ctaxa_v4_biknn` |
| 14 | SAIA / DeepSeek-4.1 | GROBID | `--role-budget` | **0.5815** | 0.4131 | 0.3736 | 0.777 | 0.380 | 0.633 | 0.467 | 0.649 | 0.218 | 0.241 | 0.245 | 0.105 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_grobid_role` |
| 15 | SAIA / DeepSeek-4.1 | PyMuPDF | `*(baseline)*` | **0.5798** | 0.4273 | 0.4286 | 0.783 | 0.391 | 0.615 | 0.469 | 0.641 | 0.254 | 0.331 | 0.263 | 0.098 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017` |
| 16 | HU / hu-reasoning | AutoParser | `--full-text --clean-taxa --v4-prompt` | **0.5773** | 0.4128 | 0.4182 | 0.800 | 0.372 | 0.639 | 0.504 | 0.571 | 0.270 | 0.210 | 0.209 | 0.140 | `llm_rag_sections_hu_hu-reasoning_autoparser_full_ctaxa_v4` |
| 17 | SAIA / DeepSeek-4.1 | AutoParser | `--evidence-gates --clean-taxa --level-prompt` | **0.5770** | 0.4441 | 0.4385 | 0.765 | 0.420 | 0.597 | 0.487 | 0.617 | 0.284 | 0.344 | 0.342 | 0.143 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ev_ctaxa_lvl` |
| 18 | SAIA / DeepSeek-4.1 | AutoParser | `--evidence-gates --clean-taxa --eco-category-map` | **0.5766** | 0.4167 | 0.3937 | 0.767 | 0.419 | 0.609 | 0.442 | 0.646 | 0.232 | 0.277 | 0.261 | 0.098 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ev_ctaxa_ecomap` |
| 19 | SAIA / DeepSeek-4.1 | Docling | `--full-text --clean-taxa --v4-prompt` | **0.5765** | 0.4421 | 0.4527 | 0.838 | 0.408 | 0.562 | 0.462 | 0.614 | 0.342 | 0.337 | 0.279 | 0.138 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_docling_full_ctaxa_v4` |
| 20 | HU / hu-reasoning | GROBID | `*(baseline)*` | **0.5706** | 0.4353 | 0.4293 | 0.773 | 0.390 | 0.614 | 0.452 | 0.624 | 0.294 | 0.292 | 0.316 | 0.163 | `llm_rag_sections_hu_hu-reasoning_grobid` |
| 21 | SAIA / DeepSeek-4.1 | AutoParser | `--focus-prompt --evidence-gates --clean-taxa` | **0.5703** | 0.4139 | 0.3955 | 0.787 | 0.409 | 0.618 | 0.449 | 0.588 | 0.255 | 0.251 | 0.267 | 0.101 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_focus_ev_ctaxa` |
| 22 | HU / hu-reasoning | AutoParser | `--eco-category-map` | **0.5689** | 0.4134 | 0.4196 | 0.752 | 0.413 | 0.589 | 0.473 | 0.617 | 0.302 | 0.300 | 0.179 | 0.095 | `llm_rag_sections_hu_hu-reasoning_autoparser_ecomap` |
| 23 | SAIA / DeepSeek-4.1 | Marker | `--detox-rag --role-budget --evidence-gates --clean-taxa` | **0.5676** | 0.4112 | 0.4190 | 0.787 | 0.409 | 0.615 | 0.446 | 0.580 | 0.272 | 0.239 | 0.242 | 0.109 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_marker_detox_role_ev_ctaxa` |
| 24 | HU / hu-reasoning | AutoParser | `--clean-taxa --v4-prompt --bilingual-knn` | **0.5671** | 0.4214 | 0.4292 | 0.752 | 0.380 | 0.632 | 0.471 | 0.600 | 0.288 | 0.243 | 0.233 | 0.192 | `llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn` |
| 25 | HU / hu-reasoning | GROBID | `--detox-rag --role-budget --evidence-gates --clean-taxa` | **0.5654** | 0.3898 | 0.3825 | 0.797 | 0.412 | 0.581 | 0.424 | 0.613 | 0.209 | 0.164 | 0.233 | 0.075 | `llm_rag_sections_hu_hu-reasoning_grobid_detox_role_ev_ctaxa` |
| 26 | SAIA / DeepSeek-4.1 | AutoParser | `--full-text --focus-prompt --evidence-gates --clean-taxa` | **0.5651** | 0.3835 | 0.3485 | 0.807 | 0.413 | 0.619 | 0.439 | 0.548 | 0.211 | 0.204 | 0.154 | 0.057 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_full_focus_ev_ctaxa` |
| 27 | SAIA / DeepSeek-4.1 | Docling | `*(baseline)*` | **0.5634** | 0.4221 | 0.4131 | 0.762 | 0.405 | 0.598 | 0.423 | 0.629 | 0.287 | 0.290 | 0.292 | 0.114 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_docling` |
| 28 | HU / hu-reasoning | Marker | `--evidence-gates --clean-taxa` | **0.5625** | 0.3807 | 0.3694 | 0.770 | 0.410 | 0.586 | 0.464 | 0.582 | 0.208 | 0.213 | 0.154 | 0.039 | `llm_rag_sections_hu_hu-reasoning_marker_ev_ctaxa` |
| 29 | SAIA / DeepSeek-4.1 | AutoParser | `--full-text --evidence-gates --clean-taxa` | **0.5615** | 0.4023 | 0.3622 | 0.769 | 0.398 | 0.603 | 0.451 | 0.586 | 0.228 | 0.252 | 0.255 | 0.078 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_full_ev_ctaxa` |
| 30 | HU / hu-reasoning | Docling | `*(baseline)*` | **0.5613** | 0.4018 | 0.3748 | 0.746 | 0.391 | 0.583 | 0.459 | 0.627 | 0.250 | 0.203 | 0.235 | 0.121 | `llm_rag_sections_hu_hu-reasoning_docling` |
| 31 | SAIA / DeepSeek-4.1 | GROBID | `--detox-rag --role-budget --evidence-gates --clean-taxa` | **0.5588** | 0.4149 | 0.4282 | 0.775 | 0.428 | 0.579 | 0.407 | 0.605 | 0.263 | 0.262 | 0.293 | 0.123 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_grobid_detox_role_ev_ctaxa` |
| 32 | HU / hu-reasoning | AutoParser | `--focus-prompt` | **0.5588** | 0.3968 | 0.3864 | 0.787 | 0.408 | 0.586 | 0.465 | 0.548 | 0.266 | 0.223 | 0.185 | 0.103 | `llm_rag_sections_hu_hu-reasoning_autoparser_focus` |
| 33 | HU / hu-reasoning | AutoParser | `--clean-taxa --v4-prompt` | **0.5571** | 0.4176 | 0.4279 | 0.762 | 0.397 | 0.600 | 0.459 | 0.567 | 0.260 | 0.262 | 0.241 | 0.208 | `llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4` |
| 34 | HU / hu-reasoning | Docling | `--full-text --clean-taxa --v4-prompt` | **0.5549** | 0.3816 | 0.3633 | 0.797 | 0.398 | 0.594 | 0.416 | 0.570 | 0.217 | 0.198 | 0.130 | 0.115 | `llm_rag_sections_hu_hu-reasoning_docling_full_ctaxa_v4` |
| 35 | HU / hu-reasoning | PyMuPDF | `*(baseline)*` | **0.5539** | 0.4073 | 0.4126 | 0.760 | 0.379 | 0.574 | 0.416 | 0.640 | 0.269 | 0.278 | 0.233 | 0.118 | `llm_rag_sections_hu_hu-reasoning` |
| 36 | HU / hu-reasoning | Marker | `--detox-rag --role-budget --evidence-gates --clean-taxa` | **0.5520** | 0.3731 | 0.3870 | 0.769 | 0.407 | 0.600 | 0.444 | 0.540 | 0.265 | 0.180 | 0.085 | 0.068 | `llm_rag_sections_hu_hu-reasoning_marker_detox_role_ev_ctaxa` |
| 37 | HU / hu-reasoning | AutoParser | `--level-prompt` | **0.5474** | 0.4118 | 0.4238 | 0.812 | 0.339 | 0.595 | 0.424 | 0.568 | 0.290 | 0.320 | 0.245 | 0.114 | `llm_rag_sections_hu_hu-reasoning_autoparser_lvl` |
| 38 | HU / hu-reasoning | AutoParser | `--self-check-reasoning` | **0.5459** | 0.3998 | 0.3928 | 0.773 | 0.438 | 0.536 | 0.398 | 0.586 | 0.259 | 0.251 | 0.241 | 0.116 | `llm_rag_sections_hu_hu-reasoning_autoparser_s5` |
| 39 | SAIA / DeepSeek-4.1 | GROBID | `--detox-rag --role-budget` | **0.5446** | 0.4118 | 0.4279 | 0.773 | 0.365 | 0.574 | 0.402 | 0.609 | 0.311 | 0.270 | 0.286 | 0.116 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_grobid_detox_role` |
| 40 | HU / hu-reasoning | GROBID | `--evidence-gates --clean-taxa` | **0.5434** | 0.4228 | 0.4194 | 0.783 | 0.315 | 0.561 | 0.440 | 0.619 | 0.292 | 0.289 | 0.318 | 0.189 | `llm_rag_sections_hu_hu-reasoning_grobid_ev_ctaxa` |
| 41 | HU / hu-reasoning | AutoParser | `--full-text` | **0.5427** | 0.3620 | 0.3341 | 0.787 | 0.387 | 0.576 | 0.414 | 0.549 | 0.194 | 0.194 | 0.133 | 0.024 | `llm_rag_sections_hu_hu-reasoning_autoparser_full` |
| 42 | SAIA / DeepSeek-4.1 | GROBID | `--detox-rag` | **0.5399** | 0.4153 | 0.4260 | 0.796 | 0.372 | 0.571 | 0.384 | 0.576 | 0.291 | 0.269 | 0.311 | 0.167 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_grobid_detox` |
| 43 | HU / hu-reasoning | AutoParser | `--full-text --focus-prompt` | **0.5351** | 0.3609 | 0.3207 | 0.796 | 0.376 | 0.583 | 0.407 | 0.513 | 0.201 | 0.162 | 0.149 | 0.061 | `llm_rag_sections_hu_hu-reasoning_autoparser_full_focus` |
| 44 | HU / hu-reasoning | AutoParser | `--self-check-reasoning --per-label-quotes` | **0.5304** | 0.3876 | 0.3910 | 0.730 | 0.234 | 0.617 | 0.517 | 0.554 | 0.277 | 0.265 | 0.187 | 0.109 | `llm_rag_sections_hu_hu-reasoning_autoparser_s5_plq` |

## 2. Bestwert je Metrik / Einzelfeld (wo funktioniert was am besten?)

| Metrik / Feld | Bester F1 | Gewinner-Run | Parser & Flags |
|---|---:|---|---|
| `C1 Macro-F1` | **0.6040** | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_marker_ev_ctaxa` | SAIA / DeepSeek-4.1, Marker, `--evidence-gates --clean-taxa` |
| `C1+C2b Macro-F1` | **0.4744** | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4` | SAIA / DeepSeek-4.1, AutoParser, `--clean-taxa --v4-prompt` |
| `Micro-F1` | **0.5019** | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4` | SAIA / DeepSeek-4.1, AutoParser, `--clean-taxa --v4-prompt` |
| `habitat` | **0.8377** | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_docling_full_ctaxa_v4` | SAIA / DeepSeek-4.1, Docling, `--full-text --clean-taxa --v4-prompt` |
| `biodiversity_level` | **0.4494** | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_marker_ev_ctaxa` | SAIA / DeepSeek-4.1, Marker, `--evidence-gates --clean-taxa` |
| `ecosystem_type.category` | **0.6667** | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_grobid_ev_ctaxa` | SAIA / DeepSeek-4.1, GROBID, `--evidence-gates --clean-taxa` |
| `ecosystem_type.term` | **0.5395** | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_grobid_ev_ctaxa` | SAIA / DeepSeek-4.1, GROBID, `--evidence-gates --clean-taxa` |
| `taxa.species_group` | **0.6822** | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_biknn` | SAIA / DeepSeek-4.1, AutoParser, `--clean-taxa --v4-prompt --bilingual-knn` |
| `taxa.scientific_name` | **0.3974** | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4` | SAIA / DeepSeek-4.1, AutoParser, `--clean-taxa --v4-prompt` |
| `taxa.german_name` | **0.3584** | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_biknn` | SAIA / DeepSeek-4.1, AutoParser, `--clean-taxa --v4-prompt --bilingual-knn` |
| `ecosystem_service.category` | **0.3429** | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4` | SAIA / DeepSeek-4.1, AutoParser, `--clean-taxa --v4-prompt` |
| `ecosystem_service.term` | **0.2083** | `llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4` | HU / hu-reasoning, AutoParser, `--clean-taxa --v4-prompt` |

## 3. Offline-Router & Konsens-Referenzwerte (0 API-Calls)

| Variante | C1 | C1+C2b | Micro | Zusammensetzung |
|---|---:|---:|---:|---|
| **SAIA 2-Regel-Router (ev+ctaxa)** | **0.6056** | 0.4469 | 0.4424 | Marker bei `g_divs<=3` oder `g_ratio<0.65`, sonst GROBID |
| **6-Run Konsens (3-of-6)** | 0.6039 | 0.4515 | **0.4590** | `saia_grobid_ev`, `saia_marker_ev`, `saia_plq`, `saia_s5`, `hu_grobid`, `hu_marker` |
| **4-Run Konsens (3-of-4, +TaxonNorm)** | 0.5932 | **0.4687** | 0.4496 | `saia_grobid`, `saia_marker`, `hu_grobid`, `hu_marker` |

## 4. Aktuell laufende / eingereihte `v4`-Läufe (SAIA & HU)

- `*_autoparser_ctaxa_v4` *(läuft)*
- `*_autoparser_full_ctaxa_v4` *(eingereiht)*
- `*_autoparser_ctaxa_v4_biknn` *(eingereiht)*
- `*_docling_full_ctaxa_v4` *(eingereiht)*

*(Nach Abschluss einfach `python3 scripts/generate_experiment_matrix.py` auf `lenlenovo` ausführen, um diese Tabelle automatisch zu aktualisieren.)*
