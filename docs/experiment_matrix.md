# Master Experiment Matrix (Dev-100)

Complete overview of all finished 100-paper runs on `data/gold/dev_100_ids.txt`,
sorted by **C1 Macro-F1** (official ranking score). Table 1 details every Architecture-3
run with all 9 per-field F1 scores. Table 2 lists every run across all architectures.

## 1. Architecture 3 in detail (all finished 100-paper runs)

| # | Model | Parser | Change / Flags | C1 | C1+C2b | Micro | Hab | BioLvl | EcoCat | EcoTerm | SpGrp | SciName | GerName | SrvCat | SrvTerm | Run dir (`outputs/architectures/`) |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | SAIA / DeepSeek-4.1 | AutoParser | `--clean-taxa --v4-prompt --bilingual-knn` | **0.6102** | 0.4715 | 0.4901 | 0.811 | 0.400 | 0.625 | 0.551 | 0.664 | 0.355 | 0.335 | 0.316 | 0.186 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_biknn` |
| 2 | SAIA / DeepSeek-4.1 | AutoParser | `--evidence-gates --clean-taxa --v4-prompt --bilingual-knn` | **0.6065** | 0.4704 | 0.4828 | 0.777 | 0.409 | 0.648 | 0.562 | 0.636 | 0.349 | 0.322 | 0.324 | 0.205 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ev_ctaxa_v4_biknn` |
| 3 | SAIA / DeepSeek-4.1 | AutoParser | `--full-text --clean-taxa --v4-prompt --bilingual-knn` | **0.5987** | 0.4621 | 0.4733 | 0.803 | 0.412 | 0.626 | 0.527 | 0.626 | 0.313 | 0.319 | 0.343 | 0.191 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_full_ctaxa_v4_biknn` |
| 4 | SAIA / DeepSeek-4.1 | Marker | `*(baseline)*` | **0.5985** | 0.4362 | 0.3784 | 0.760 | 0.402 | 0.667 | 0.491 | 0.672 | 0.221 | 0.253 | 0.342 | 0.118 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_marker` |
| 5 | SAIA / DeepSeek-4.1 | GROBID | `--evidence-gates --clean-taxa` | **0.5959** | 0.4530 | 0.4516 | 0.774 | 0.359 | 0.667 | 0.539 | 0.640 | 0.300 | 0.341 | 0.300 | 0.157 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_grobid_ev_ctaxa` |
| 6 | SAIA / DeepSeek-4.1 | GROBID | `*(baseline)*` | **0.5957** | 0.4544 | 0.4390 | 0.788 | 0.413 | 0.636 | 0.490 | 0.652 | 0.282 | 0.337 | 0.333 | 0.159 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_grobid` |
| 7 | SAIA / DeepSeek-4.1 | Marker | `--role-budget --evidence-gates --clean-taxa` | **0.5953** | 0.4271 | 0.3924 | 0.775 | 0.409 | 0.649 | 0.503 | 0.641 | 0.252 | 0.277 | 0.261 | 0.077 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_marker_role_ev_ctaxa` |
| 8 | SAIA / DeepSeek-4.1 | Docling Light | `--full-text --clean-taxa --v4-prompt` | **0.5945** | 0.4503 | 0.4609 | 0.814 | 0.408 | 0.591 | 0.504 | 0.656 | 0.316 | 0.356 | 0.279 | 0.129 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_docling_full_ctaxa_v4` |
| 9 | HU / hu-reasoning | AutoParser | `--clean-taxa --v4-prompt --bilingual-knn --eco-type-land-rule` | **0.5939** | 0.4308 | 0.4390 | 0.767 | 0.378 | 0.692 | 0.561 | 0.571 | 0.341 | 0.222 | 0.230 | 0.115 | `llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn_ectl` |
| 10 | SAIA / DeepSeek-4.1 | AutoParser | `--clean-taxa --v4-prompt --raw-fulltext` | **0.5917** | 0.4717 | 0.5014 | 0.820 | 0.385 | 0.591 | 0.531 | 0.631 | 0.398 | 0.378 | 0.333 | 0.178 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_rawft` |
| 11 | SAIA / DeepSeek-4.1 | AutoParser | `--clean-taxa --v4-prompt --bilingual-knn --level-fold` | **0.5908** | 0.4703 | 0.4902 | 0.770 | 0.391 | 0.610 | 0.533 | 0.651 | 0.380 | 0.350 | 0.343 | 0.205 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_biknn_lvlf` |
| 12 | HU / hu-reasoning | MinhDoclingJSON | `--clean-taxa --v4-prompt` | **0.5894** | 0.4233 | 0.4426 | 0.780 | 0.360 | 0.661 | 0.516 | 0.630 | 0.299 | 0.253 | 0.185 | 0.125 | `llm_rag_sections_hu_hu-reasoning_minhjson_ctaxa_v4` |
| 13 | SAIA / DeepSeek-4.1 | MinhDoclingJSON | `--clean-taxa --v4-prompt` | **0.5883** | 0.4626 | 0.4804 | 0.789 | 0.368 | 0.610 | 0.525 | 0.650 | 0.344 | 0.355 | 0.294 | 0.229 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_minhjson_ctaxa_v4` |
| 14 | SAIA / DeepSeek-4.1 | AutoParser | `--clean-taxa` | **0.5883** | 0.4400 | 0.4426 | 0.756 | 0.389 | 0.667 | 0.503 | 0.627 | 0.299 | 0.331 | 0.273 | 0.116 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa` |
| 15 | SAIA / DeepSeek-4.1 | AutoParser | `--clean-taxa --v4-prompt --bilingual-knn --budget-chars 64000` | **0.5881** | 0.4543 | 0.4670 | 0.783 | 0.385 | 0.615 | 0.543 | 0.615 | 0.325 | 0.304 | 0.324 | 0.195 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_biknn_b64k` |
| 16 | SAIA / DeepSeek-4.1 | AutoParser | `--clean-taxa --v4-prompt --bilingual-knn --service-gate` | **0.5874** | 0.4696 | 0.4761 | 0.773 | 0.391 | 0.607 | 0.528 | 0.638 | 0.349 | 0.305 | 0.400 | 0.235 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_biknn_svcg` |
| 17 | SAIA / DeepSeek-4.1 | AutoParser | `--evidence-gates --clean-taxa --self-check-reasoning` | **0.5861** | 0.4489 | 0.4551 | 0.757 | 0.414 | 0.638 | 0.500 | 0.622 | 0.326 | 0.310 | 0.343 | 0.131 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ev_ctaxa_s5` |
| 18 | SAIA / DeepSeek-4.1 | AutoParser | `--clean-taxa --self-check-reasoning --per-label-quotes` | **0.5856** | 0.4404 | 0.4671 | 0.778 | 0.402 | 0.599 | 0.522 | 0.627 | 0.390 | 0.342 | 0.210 | 0.094 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_s5_plq` |
| 19 | SAIA / DeepSeek-4.1 | AutoParser | `--clean-taxa --v4-prompt` | **0.5847** | 0.4784 | 0.4994 | 0.793 | 0.359 | 0.602 | 0.496 | 0.673 | 0.384 | 0.357 | 0.375 | 0.267 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4` |
| 20 | SAIA / DeepSeek-4.1 | AutoParser | `--clean-taxa --level-prompt` | **0.5841** | 0.4453 | 0.4402 | 0.778 | 0.394 | 0.625 | 0.497 | 0.627 | 0.291 | 0.333 | 0.318 | 0.145 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_lvl` |
| 21 | HU / hu-reasoning | Marker | `*(baseline)*` | **0.5837** | 0.4197 | 0.4015 | 0.788 | 0.405 | 0.621 | 0.489 | 0.616 | 0.264 | 0.246 | 0.240 | 0.108 | `llm_rag_sections_hu_hu-reasoning_marker` |
| 22 | SAIA / DeepSeek-4.1 | GROBID | `--role-budget` | **0.5815** | 0.4131 | 0.3736 | 0.777 | 0.380 | 0.633 | 0.467 | 0.649 | 0.218 | 0.241 | 0.245 | 0.105 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_grobid_role` |
| 23 | SAIA / DeepSeek-4.1 | PyMuPDF | `*(baseline)*` | **0.5798** | 0.4273 | 0.4286 | 0.783 | 0.391 | 0.615 | 0.469 | 0.641 | 0.254 | 0.331 | 0.263 | 0.098 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017` |
| 24 | SAIA / DeepSeek-4.1 | AutoParser | `--evidence-gates --clean-taxa --level-prompt` | **0.5770** | 0.4441 | 0.4385 | 0.765 | 0.420 | 0.597 | 0.487 | 0.617 | 0.284 | 0.344 | 0.342 | 0.143 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ev_ctaxa_lvl` |
| 25 | SAIA / DeepSeek-4.1 | Marker | `--evidence-gates --clean-taxa` | **0.5767** | 0.4339 | 0.3967 | 0.742 | 0.398 | 0.649 | 0.480 | 0.614 | 0.260 | 0.266 | 0.333 | 0.163 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_marker_ev_ctaxa` |
| 26 | SAIA / DeepSeek-4.1 | AutoParser | `--evidence-gates --clean-taxa --eco-category-map` | **0.5766** | 0.4167 | 0.3937 | 0.767 | 0.419 | 0.609 | 0.442 | 0.646 | 0.232 | 0.277 | 0.261 | 0.098 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ev_ctaxa_ecomap` |
| 27 | SAIA / DeepSeek-4.1 | AutoParser | `--full-text --clean-taxa --v4-prompt` | **0.5759** | 0.4497 | 0.4632 | 0.795 | 0.369 | 0.564 | 0.500 | 0.652 | 0.311 | 0.324 | 0.333 | 0.200 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_full_ctaxa_v4` |
| 28 | HU / hu-reasoning | AutoParser | `--clean-taxa --v4-prompt --bilingual-knn svs` | **0.5727** | 0.4872 | 0.4630 | 0.765 | 0.348 | 0.637 | 0.517 | 0.595 | 0.325 | 0.263 | 0.533 | 0.400 | `llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn_svs` |
| 29 | HU / hu-reasoning | AutoParser | `--clean-taxa --v4-prompt --bilingual-knn --service-gate` | **0.5715** | 0.4265 | 0.4278 | 0.754 | 0.365 | 0.661 | 0.487 | 0.591 | 0.254 | 0.230 | 0.286 | 0.210 | `llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn_svcg` |
| 30 | HU / hu-reasoning | AutoParser | `--clean-taxa --v4-prompt --bilingual-knn dtsc` | **0.5715** | 0.4222 | 0.4292 | 0.765 | 0.392 | 0.640 | 0.504 | 0.557 | 0.264 | 0.241 | 0.250 | 0.186 | `llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn_dtsc` |
| 31 | SAIA / DeepSeek-4.1 | AutoParser | `--v4-prompt --bilingual-knn` | **0.5713** | 0.4622 | 0.4793 | 0.766 | 0.424 | 0.549 | 0.480 | 0.638 | 0.352 | 0.359 | 0.364 | 0.229 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_v4_biknn` |
| 32 | HU / hu-reasoning | AutoParser | `--evidence-gates --clean-taxa --v4-prompt --bilingual-knn` | **0.5713** | 0.4157 | 0.4269 | 0.754 | 0.377 | 0.632 | 0.525 | 0.569 | 0.296 | 0.218 | 0.203 | 0.167 | `llm_rag_sections_hu_hu-reasoning_autoparser_ev_ctaxa_v4_biknn` |
| 33 | HU / hu-reasoning | GROBID | `*(baseline)*` | **0.5706** | 0.4353 | 0.4293 | 0.773 | 0.390 | 0.614 | 0.452 | 0.624 | 0.294 | 0.292 | 0.316 | 0.163 | `llm_rag_sections_hu_hu-reasoning_grobid` |
| 34 | SAIA / DeepSeek-4.1 | AutoParser | `--focus-prompt --evidence-gates --clean-taxa` | **0.5703** | 0.4139 | 0.3955 | 0.787 | 0.409 | 0.618 | 0.449 | 0.588 | 0.255 | 0.251 | 0.267 | 0.101 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_focus_ev_ctaxa` |
| 35 | HU / hu-reasoning | AutoParser | `--clean-taxa --v4-prompt` | **0.5700** | 0.4122 | 0.4279 | 0.749 | 0.364 | 0.655 | 0.504 | 0.578 | 0.274 | 0.250 | 0.210 | 0.125 | `llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4` |
| 36 | HU / hu-reasoning | AutoParser | `--eco-category-map` | **0.5689** | 0.4134 | 0.4196 | 0.752 | 0.413 | 0.589 | 0.473 | 0.617 | 0.302 | 0.300 | 0.179 | 0.095 | `llm_rag_sections_hu_hu-reasoning_autoparser_ecomap` |
| 37 | HU / hu-reasoning | AutoParser | `--clean-taxa` | **0.5689** | 0.4047 | 0.4083 | 0.805 | 0.389 | 0.595 | 0.474 | 0.583 | 0.266 | 0.288 | 0.167 | 0.078 | `llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa` |
| 38 | HU / hu-reasoning | AutoParser | `--v4-prompt --bilingual-knn` | **0.5681** | 0.4265 | 0.4393 | 0.770 | 0.369 | 0.607 | 0.504 | 0.591 | 0.312 | 0.262 | 0.250 | 0.174 | `llm_rag_sections_hu_hu-reasoning_autoparser_v4_biknn` |
| 39 | SAIA / DeepSeek-4.1 | Marker | `--detox-rag --role-budget --evidence-gates --clean-taxa` | **0.5676** | 0.4112 | 0.4190 | 0.787 | 0.409 | 0.615 | 0.446 | 0.580 | 0.272 | 0.239 | 0.242 | 0.109 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_marker_detox_role_ev_ctaxa` |
| 40 | HU / hu-reasoning | AutoParser | `--clean-taxa --v4-prompt --bilingual-knn --german-pairing-names` | **0.5675** | 0.4203 | 0.4307 | 0.765 | 0.377 | 0.622 | 0.512 | 0.561 | 0.290 | 0.282 | 0.219 | 0.154 | `llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn_pgn` |
| 41 | HU / hu-reasoning | GROBID | `--detox-rag --role-budget --evidence-gates --clean-taxa` | **0.5654** | 0.3898 | 0.3825 | 0.797 | 0.412 | 0.581 | 0.424 | 0.613 | 0.209 | 0.164 | 0.233 | 0.075 | `llm_rag_sections_hu_hu-reasoning_grobid_detox_role_ev_ctaxa` |
| 42 | SAIA / DeepSeek-4.1 | AutoParser | `--full-text --focus-prompt --evidence-gates --clean-taxa` | **0.5651** | 0.3835 | 0.3485 | 0.807 | 0.413 | 0.619 | 0.439 | 0.548 | 0.211 | 0.204 | 0.154 | 0.057 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_full_focus_ev_ctaxa` |
| 43 | SAIA / DeepSeek-4.1 | Docling Light | `*(baseline)*` | **0.5634** | 0.4221 | 0.4131 | 0.762 | 0.405 | 0.598 | 0.423 | 0.629 | 0.287 | 0.290 | 0.292 | 0.114 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_docling` |
| 44 | HU / hu-reasoning | AutoParser | `--clean-taxa --v4-prompt --bilingual-knn ggl` | **0.5617** | 0.4154 | 0.4302 | 0.743 | 0.412 | 0.598 | 0.484 | 0.571 | 0.283 | 0.263 | 0.214 | 0.170 | `llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn_ggl` |
| 45 | SAIA / DeepSeek-4.1 | AutoParser | `--full-text --evidence-gates --clean-taxa` | **0.5615** | 0.4023 | 0.3622 | 0.769 | 0.398 | 0.603 | 0.451 | 0.586 | 0.228 | 0.252 | 0.255 | 0.078 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_full_ev_ctaxa` |
| 46 | HU / hu-reasoning | Docling Light | `*(baseline)*` | **0.5613** | 0.4018 | 0.3748 | 0.746 | 0.391 | 0.583 | 0.459 | 0.627 | 0.250 | 0.203 | 0.235 | 0.121 | `llm_rag_sections_hu_hu-reasoning_docling` |
| 47 | HU / hu-reasoning | AutoParser | `--clean-taxa --v4-prompt --bilingual-knn` | **0.5604** | 0.4166 | 0.4264 | 0.733 | 0.382 | 0.609 | 0.517 | 0.562 | 0.279 | 0.266 | 0.246 | 0.157 | `llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn` |
| 48 | HU / hu-reasoning | AutoParser | `--clean-taxa --v4-prompt --bilingual-knn --budget-chars 64000` | **0.5604** | 0.4075 | 0.4176 | 0.782 | 0.359 | 0.602 | 0.469 | 0.591 | 0.276 | 0.215 | 0.207 | 0.167 | `llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn_b64k` |
| 49 | HU / hu-reasoning | AutoParser | `--full-text --clean-taxa --v4-prompt` | **0.5602** | 0.4053 | 0.4257 | 0.795 | 0.320 | 0.610 | 0.484 | 0.592 | 0.288 | 0.266 | 0.188 | 0.105 | `llm_rag_sections_hu_hu-reasoning_autoparser_full_ctaxa_v4` |
| 50 | SAIA / DeepSeek-4.1 | GROBID | `--detox-rag --role-budget --evidence-gates --clean-taxa` | **0.5588** | 0.4149 | 0.4282 | 0.775 | 0.428 | 0.579 | 0.407 | 0.605 | 0.263 | 0.262 | 0.293 | 0.123 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_grobid_detox_role_ev_ctaxa` |
| 51 | HU / hu-reasoning | AutoParser | `--focus-prompt` | **0.5588** | 0.3968 | 0.3864 | 0.787 | 0.408 | 0.586 | 0.465 | 0.548 | 0.266 | 0.223 | 0.185 | 0.103 | `llm_rag_sections_hu_hu-reasoning_autoparser_focus` |
| 52 | HU / hu-reasoning | Docling Light | `--full-text --clean-taxa --v4-prompt` | **0.5585** | 0.3751 | 0.3467 | 0.806 | 0.407 | 0.592 | 0.456 | 0.531 | 0.165 | 0.169 | 0.150 | 0.099 | `llm_rag_sections_hu_hu-reasoning_docling_full_ctaxa_v4` |
| 53 | HU / hu-reasoning | AutoParser | `--clean-taxa --v4-prompt --raw-fulltext` | **0.5584** | 0.3968 | 0.3991 | 0.790 | 0.387 | 0.588 | 0.464 | 0.563 | 0.239 | 0.200 | 0.237 | 0.103 | `llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_rawft` |
| 54 | HU / hu-reasoning | AutoParser | `--full-text --clean-taxa --v4-prompt --bilingual-knn` | **0.5562** | 0.4016 | 0.4059 | 0.761 | 0.347 | 0.610 | 0.492 | 0.571 | 0.240 | 0.212 | 0.214 | 0.167 | `llm_rag_sections_hu_hu-reasoning_autoparser_full_ctaxa_v4_biknn` |
| 55 | HU / hu-reasoning | PyMuPDF | `*(baseline)*` | **0.5539** | 0.4073 | 0.4126 | 0.760 | 0.379 | 0.574 | 0.416 | 0.640 | 0.269 | 0.278 | 0.233 | 0.118 | `llm_rag_sections_hu_hu-reasoning` |
| 56 | HU / hu-reasoning | AutoParser | `--clean-taxa --v4-prompt --bilingual-knn --level-fold` | **0.5521** | 0.4165 | 0.4247 | 0.740 | 0.385 | 0.579 | 0.471 | 0.587 | 0.280 | 0.219 | 0.261 | 0.229 | `llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn_lvlf` |
| 57 | HU / hu-reasoning | Marker | `--detox-rag --role-budget --evidence-gates --clean-taxa` | **0.5520** | 0.3731 | 0.3870 | 0.769 | 0.407 | 0.600 | 0.444 | 0.540 | 0.265 | 0.180 | 0.085 | 0.068 | `llm_rag_sections_hu_hu-reasoning_marker_detox_role_ev_ctaxa` |
| 58 | HU / hu-reasoning | Marker | `--evidence-gates --clean-taxa` | **0.5484** | 0.3966 | 0.3970 | 0.768 | 0.438 | 0.556 | 0.417 | 0.564 | 0.296 | 0.244 | 0.182 | 0.105 | `llm_rag_sections_hu_hu-reasoning_marker_ev_ctaxa` |
| 59 | HU / hu-reasoning | AutoParser | `--level-prompt` | **0.5474** | 0.4118 | 0.4238 | 0.812 | 0.339 | 0.595 | 0.424 | 0.568 | 0.290 | 0.320 | 0.245 | 0.114 | `llm_rag_sections_hu_hu-reasoning_autoparser_lvl` |
| 60 | HU / hu-reasoning | AutoParser | `--clean-taxa --level-prompt` | **0.5469** | 0.4084 | 0.4101 | 0.770 | 0.349 | 0.566 | 0.465 | 0.584 | 0.284 | 0.298 | 0.218 | 0.141 | `llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_lvl` |
| 61 | HU / hu-reasoning | AutoParser | `--self-check-reasoning` | **0.5459** | 0.3998 | 0.3928 | 0.773 | 0.438 | 0.536 | 0.398 | 0.586 | 0.259 | 0.251 | 0.241 | 0.116 | `llm_rag_sections_hu_hu-reasoning_autoparser_s5` |
| 62 | SAIA / DeepSeek-4.1 | GROBID | `--detox-rag --role-budget` | **0.5446** | 0.4118 | 0.4279 | 0.773 | 0.365 | 0.574 | 0.402 | 0.609 | 0.311 | 0.270 | 0.286 | 0.116 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_grobid_detox_role` |
| 63 | HU / hu-reasoning | GROBID | `--evidence-gates --clean-taxa` | **0.5434** | 0.4228 | 0.4194 | 0.783 | 0.315 | 0.561 | 0.440 | 0.619 | 0.292 | 0.289 | 0.318 | 0.189 | `llm_rag_sections_hu_hu-reasoning_grobid_ev_ctaxa` |
| 64 | HU / hu-reasoning | AutoParser | `--full-text` | **0.5427** | 0.3620 | 0.3341 | 0.787 | 0.387 | 0.576 | 0.414 | 0.549 | 0.194 | 0.194 | 0.133 | 0.024 | `llm_rag_sections_hu_hu-reasoning_autoparser_full` |
| 65 | SAIA / DeepSeek-4.1 | GROBID | `--detox-rag` | **0.5399** | 0.4153 | 0.4260 | 0.796 | 0.372 | 0.571 | 0.384 | 0.576 | 0.291 | 0.269 | 0.311 | 0.167 | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_grobid_detox` |
| 66 | HU / hu-reasoning | AutoParser | `--full-text --focus-prompt` | **0.5351** | 0.3609 | 0.3207 | 0.796 | 0.376 | 0.583 | 0.407 | 0.513 | 0.201 | 0.162 | 0.149 | 0.061 | `llm_rag_sections_hu_hu-reasoning_autoparser_full_focus` |
| 67 | HU / hu-reasoning | AutoParser | `--self-check-reasoning --per-label-quotes` | **0.5304** | 0.3876 | 0.3910 | 0.730 | 0.234 | 0.617 | 0.517 | 0.554 | 0.277 | 0.265 | 0.187 | 0.109 | `llm_rag_sections_hu_hu-reasoning_autoparser_s5_plq` |

## 2. Best score per metric / field (where does what work best?)

| Metric / Field | Best F1 | Winner run | Parser & Flags |
|---|---:|---|---|
| `C1 Macro-F1` | **0.6102** | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_biknn` | SAIA / DeepSeek-4.1, AutoParser, `--clean-taxa --v4-prompt --bilingual-knn` |
| `C1+C2b Macro-F1` | **0.4872** | `llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn_svs` | HU / hu-reasoning, AutoParser, `--clean-taxa --v4-prompt --bilingual-knn svs` |
| `Micro-F1` | **0.5014** | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_rawft` | SAIA / DeepSeek-4.1, AutoParser, `--clean-taxa --v4-prompt --raw-fulltext` |
| `habitat` | **0.8203** | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_rawft` | SAIA / DeepSeek-4.1, AutoParser, `--clean-taxa --v4-prompt --raw-fulltext` |
| `biodiversity_level` | **0.4379** | `llm_rag_sections_hu_hu-reasoning_autoparser_s5` | HU / hu-reasoning, AutoParser, `--self-check-reasoning` |
| `ecosystem_type.category` | **0.6923** | `llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn_ectl` | HU / hu-reasoning, AutoParser, `--clean-taxa --v4-prompt --bilingual-knn --eco-type-land-rule` |
| `ecosystem_type.term` | **0.5620** | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ev_ctaxa_v4_biknn` | SAIA / DeepSeek-4.1, AutoParser, `--evidence-gates --clean-taxa --v4-prompt --bilingual-knn` |
| `taxa.species_group` | **0.6727** | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4` | SAIA / DeepSeek-4.1, AutoParser, `--clean-taxa --v4-prompt` |
| `taxa.scientific_name` | **0.3978** | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_rawft` | SAIA / DeepSeek-4.1, AutoParser, `--clean-taxa --v4-prompt --raw-fulltext` |
| `taxa.german_name` | **0.3776** | `llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_rawft` | SAIA / DeepSeek-4.1, AutoParser, `--clean-taxa --v4-prompt --raw-fulltext` |
| `ecosystem_service.category` | **0.5333** | `llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn_svs` | HU / hu-reasoning, AutoParser, `--clean-taxa --v4-prompt --bilingual-knn svs` |
| `ecosystem_service.term` | **0.4000** | `llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn_svs` | HU / hu-reasoning, AutoParser, `--clean-taxa --v4-prompt --bilingual-knn svs` |

## 3. All runs across all architectures

| # | Model | Category | C1 | C1+C2b | Micro | Runtime | Seed | Run dir (`outputs/`) |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.6102** | 0.4715 | 0.4901 | 973s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_biknn` |
| 2 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.6065** | 0.4704 | 0.4828 | 1360s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ev_ctaxa_v4_biknn` |
| 3 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5987** | 0.4621 | 0.4733 | 1143s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_full_ctaxa_v4_biknn` |
| 4 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5985** | 0.4362 | 0.3784 | 5916s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_marker` |
| 5 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5959** | 0.4530 | 0.4516 | 2427s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_grobid_ev_ctaxa` |
| 6 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5957** | 0.4544 | 0.4390 | 2911s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_grobid` |
| 7 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5953** | 0.4271 | 0.3924 | 3798s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_marker_role_ev_ctaxa` |
| 8 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5945** | 0.4503 | 0.4609 | 38082s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_docling_full_ctaxa_v4` |
| 9 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5939** | 0.4308 | 0.4390 | 5432s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn_ectl` |
| 10 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5917** | 0.4717 | 0.5014 | 1829s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_rawft` |
| 11 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5908** | 0.4703 | 0.4902 | 1237s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_biknn_lvlf` |
| 12 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5894** | 0.4233 | 0.4426 | 5278s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_minhjson_ctaxa_v4` |
| 13 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5883** | 0.4400 | 0.4426 | 5553s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa` |
| 14 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5883** | 0.4626 | 0.4804 | 928s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_minhjson_ctaxa_v4` |
| 15 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5881** | 0.4543 | 0.4670 | 1465s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_biknn_b64k` |
| 16 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5874** | 0.4696 | 0.4761 | 1495s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_biknn_svcg` |
| 17 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5861** | 0.4489 | 0.4551 | 3120s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ev_ctaxa_s5` |
| 18 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5856** | 0.4404 | 0.4671 | 2906s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_s5_plq` |
| 19 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5847** | 0.4784 | 0.4994 | 948s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4` |
| 20 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5841** | 0.4453 | 0.4402 | 3076s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_lvl` |
| 21 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5837** | 0.4197 | 0.4015 | 4458s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_marker` |
| 22 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5815** | 0.4131 | 0.3736 | 3590s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_grobid_role` |
| 23 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5798** | 0.4273 | 0.4286 | 4337s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017` |
| 24 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5770** | 0.4441 | 0.4385 | 6762s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ev_ctaxa_lvl` |
| 25 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5767** | 0.4339 | 0.3967 | 4105s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_marker_ev_ctaxa` |
| 26 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5766** | 0.4167 | 0.3937 | 5240s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ev_ctaxa_ecomap` |
| 27 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5759** | 0.4497 | 0.4632 | 1135s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_full_ctaxa_v4` |
| 28 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5727** | 0.4872 | 0.4630 | 6370s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn_svs` |
| 29 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5715** | 0.4222 | 0.4292 | 5846s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn_dtsc` |
| 30 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5715** | 0.4265 | 0.4278 | 4838s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn_svcg` |
| 31 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5713** | 0.4157 | 0.4269 | 4451s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_ev_ctaxa_v4_biknn` |
| 32 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5713** | 0.4622 | 0.4793 | 1497s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_v4_biknn` |
| 33 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5706** | 0.4353 | 0.4293 | 5825s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_grobid` |
| 34 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5703** | 0.4139 | 0.3955 | 1962s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_focus_ev_ctaxa` |
| 35 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5700** | 0.4122 | 0.4279 | 8392s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4` |
| 36 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5689** | 0.4047 | 0.4083 | 4650s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa` |
| 37 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5689** | 0.4134 | 0.4196 | 5124s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_ecomap` |
| 38 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5681** | 0.4265 | 0.4393 | 4663s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_v4_biknn` |
| 39 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5676** | 0.4112 | 0.4190 | 2658s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_marker_detox_role_ev_ctaxa` |
| 40 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5675** | 0.4203 | 0.4307 | 5072s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn_pgn` |
| 41 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5654** | 0.3898 | 0.3825 | 6503s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_grobid_detox_role_ev_ctaxa` |
| 42 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5651** | 0.3835 | 0.3485 | 6413s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_full_focus_ev_ctaxa` |
| 43 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5634** | 0.4221 | 0.4131 | 2114s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_docling` |
| 44 | SAIA / DeepSeek-4.1 | Architecture 2 (parallel chunks) | **0.5622** | 0.4244 | 0.4163 | 4043s | 42 | `architectures/llm_parallel_chunking_saia_deepseek-v4.1-flash-1017_fg_term` |
| 45 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5617** | 0.4154 | 0.4302 | 5327s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn_ggl` |
| 46 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5615** | 0.4023 | 0.3622 | 4639s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_full_ev_ctaxa` |
| 47 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5613** | 0.4018 | 0.3748 | 4483s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_docling` |
| 48 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5604** | 0.4166 | 0.4264 | 4482s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn` |
| 49 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5604** | 0.4075 | 0.4176 | 4683s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn_b64k` |
| 50 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5602** | 0.4053 | 0.4257 | 5002s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_full_ctaxa_v4` |
| 51 | HU / hu-reasoning | Architecture 2 (parallel chunks) | **0.5593** | 0.4144 | 0.3923 | 12202s | 42 | `architectures/llm_parallel_chunking_hu_hu-reasoning_fg_term` |
| 52 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5588** | 0.3968 | 0.3864 | 5220s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_focus` |
| 53 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5588** | 0.4149 | 0.4282 | 4358s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_grobid_detox_role_ev_ctaxa` |
| 54 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5585** | 0.3751 | 0.3467 | 20294s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_docling_full_ctaxa_v4` |
| 55 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5584** | 0.3968 | 0.3991 | 5803s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_rawft` |
| 56 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5562** | 0.4016 | 0.4059 | 5234s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_full_ctaxa_v4_biknn` |
| 57 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5539** | 0.4073 | 0.4126 | 4496s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning` |
| 58 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5521** | 0.4165 | 0.4247 | 7333s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn_lvlf` |
| 59 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5520** | 0.3731 | 0.3870 | 4864s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_marker_detox_role_ev_ctaxa` |
| 60 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5484** | 0.3966 | 0.3970 | 4690s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_marker_ev_ctaxa` |
| 61 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5474** | 0.4118 | 0.4238 | 7646s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_lvl` |
| 62 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5469** | 0.4084 | 0.4101 | 7425s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_lvl` |
| 63 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5459** | 0.3998 | 0.3928 | 7796s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_s5` |
| 64 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5446** | 0.4118 | 0.4279 | 3460s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_grobid_detox_role` |
| 65 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5434** | 0.4228 | 0.4194 | 5073s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_grobid_ev_ctaxa` |
| 66 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5427** | 0.3620 | 0.3341 | 6247s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_full` |
| 67 | SAIA / DeepSeek-4.1 | Architecture 3 (RAG sections) | **0.5399** | 0.4153 | 0.4260 | 2775s | 42 | `architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_grobid_detox` |
| 68 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5351** | 0.3609 | 0.3207 | 6395s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_full_focus` |
| 69 | HU / hu-reasoning | Architecture 3 (RAG sections) | **0.5304** | 0.3876 | 0.3910 | 7675s | 42 | `architectures/llm_rag_sections_hu_hu-reasoning_autoparser_s5_plq` |
| 70 | HU / hu-reasoning | Architecture 2 (parallel chunks) | **0.4833** | 0.3456 | 0.3686 | 9589s | 42 | `architectures/llm_parallel_chunking_hu_hu-reasoning_fg_trans` |
| 71 | HU / hu-reasoning | Architecture 2 (parallel chunks) | **0.4706** | 0.3392 | 0.3919 | 18844s | 42 | `architectures/llm_parallel_chunking_saia_mistral-large-4-1t-a52b_fg` |
| 72 | HU / hu-reasoning | Architecture 2 (parallel chunks) | **0.4565** | 0.3339 | 0.3566 | 12792s | 42 | `architectures/llm_parallel_chunking_hu_hu-reasoning_fg` |
| 73 | llm_sections_parallel_hu_hu-reasoning_en | Sections experiment (parallel, EN) | **0.4476** | 0.3090 | 0.3059 | 11377s | 42 | `architectures/llm_sections_parallel_hu_hu-reasoning_en` |
| 74 | HU / hu-reasoning | Architecture 2 (parallel chunks) | **0.4463** | 0.3321 | 0.3633 | 12085s | 42 | `architectures/llm_parallel_chunking_hu_hu-reasoning_fg_clean` |
| 75 | llm_sections_parallel_saia_glm-5.3_en | Sections experiment (parallel, EN) | **0.4406** | 0.3051 | 0.3078 | 28911s | 42 | `architectures/llm_sections_parallel_saia_glm-5.3_en` |
| 76 | HU / hu-reasoning | Architecture 2 (parallel chunks) | **0.4318** | 0.2885 | 0.3152 | 5179s | 42 | `architectures/llm_parallel_chunking_hu_hu-reasoning` |
| 77 | llm_sections_parallel_saia_glm-5.3-flash_en | Sections experiment (parallel, EN) | **0.4285** | 0.2920 | 0.2989 | 46716s | 42 | `architectures/llm_sections_parallel_saia_glm-5.3-flash_en` |
| 78 | HU / hu-reasoning | Architecture 2 (parallel chunks) | **0.4056** | 0.2740 | 0.3112 | 7252s | 42 | `architectures/llm_parallel_chunking_saia_mistral-large-4-1t-a52b` |
| 79 | llm_sections_parallel_saia_deepseek-v4.1-flash-1017_en | Sections experiment (parallel, EN) | **0.3887** | 0.2685 | 0.2797 | 13080s | 42 | `architectures/llm_sections_parallel_saia_deepseek-v4.1-flash-1017_en` |
| 80 | llm_sections_parallel_hu_hu-reasoning_de | Sections experiment (parallel, DE) | **0.2498** | 0.2210 | 0.2191 | 13734s | 42 | `architectures/llm_sections_parallel_hu_hu-reasoning_de` |
| 81 | llm_sections_sequential_hu_hu-reasoning_en | Sections experiment (sequential, EN) | **0.1629** | 0.1035 | 0.1360 | 12525s | 42 | `architectures/llm_sections_sequential_hu_hu-reasoning_en` |
| 82 | llm_sections_parallel_saia_mistral-large-4-1t-a52b_en | Sections experiment (parallel, EN) | **0.1490** | 0.0929 | 0.1181 | 55705s | 42 | `architectures/llm_sections_parallel_saia_mistral-large-4-1t-a52b_en` |

## 4. Offline router & consensus reference values (0 API calls)

| Variant | C1 | C1+C2b | Micro | Setup |
|---|---:|---:|---:|---|
| **SAIA 2-rule router (ev+ctaxa)** | **0.6056** | 0.4469 | 0.4424 | Marker if `g_divs<=3` or `g_ratio<0.65`, else GROBID |
| **6-run consensus (3-of-6)** | 0.6039 | 0.4515 | **0.4590** | `saia_grobid_ev`, `saia_marker_ev`, `saia_plq`, `saia_s5`, `hu_grobid`, `hu_marker` |
| **4-run consensus (3-of-4, +TaxonNorm)** | 0.5932 | **0.4687** | 0.4496 | `saia_grobid`, `saia_marker`, `hu_grobid`, `hu_marker` |

*(Regenerate on `lenlenovo` after new runs: `python3 scripts/generate_experiment_matrix.py`.)*
