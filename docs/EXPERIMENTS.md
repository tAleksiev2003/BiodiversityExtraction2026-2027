# Experiments

Every experiment of ANLP 2026/27 Shared Task C (C1 + C2b) so far, in the order we ran them,
with its result and what we learned. All scores are on the **dev set: the 100 papers with a
PDF** (`data/processed/dev_with_pdf.jsonl`), scored with `src/evaluation.py`
(`--categorical-only`). Decisions and their reasons: `SPEC.md`, section 13.
Prompt versions: `config/prompts/README.md`.

**Current best (the defaults):** pipeline F, prompt v4 (training examples), DeepSeek 4.1,
the post-processing rule on services (section 10) and the field prompt level_v1 for
biodiversity_level (section 11): **C1 0.6732, C2b 0.4202**
(`runs/20261119-161424_full_text_dev_level_prompt_v1`). **Not robust:** better than without
the level prompt in only 73 % of bootstrap samples.

> Dev scores are slightly optimistic: prompt, checker settings and model were all chosen on
> these 100 papers. Expect a few hundredths less on the test set.

---

## 1. Overview

| # | Date | System | Change against the line before | C1 | C2b | Run folder (`runs/`) |
|---|---|---|---|---|---|---|
| 0 | 11-16 | most-frequent baseline | – | 0.2055 | – | `*_baseline_most_frequent_dev_with_pdf` |
| 1 | 11-18 | F, prompt v1, DeepSeek 4.0, checker 5–40 words | first LLM system | 0.5250 | 0.2854 | `20261118-124839_full_text_dev` |
| 2 | 11-18 | F, v1, no verbatim checker | ablation | 0.5218 | 0.3130 | `20261118-132423_full_text_dev_no_verbatim_check` |
| 3 | 11-18 | F, prompt v2, checker 5–60 words | "main subject" instead of "this study only", base rates | 0.5711 | 0.3315 | `20261118-141641_full_text_dev` |
| 4 | 11-18 | B, Qwen3-30B-A3B, up to 20 chunks | chunked pipeline | 0.4527 | 0.2070 | `20261118-204828_parallel_dev` |
| 5 | 11-19 | B, DeepSeek 4.0, at most 4 chunks | same model as F, fewer chunks | 0.5781 | 0.3178 | `20261119-102812_parallel_dev_max_4_chunks` |
| 6 | 11-19 | B as #5, checker 3–60 words | shorter quotes allowed | 0.5872 | 0.3311 | `20261119-105128_parallel_dev_max_4_chunks_min_quote_words_3` |
| 7 | 11-19 | F, prompt v3 | schema questions, taxa base rate, always German names | 0.5122 | 0.3180 | `20261119-120510_full_text_dev` |
| 8 | 11-19 | F, prompt v4, DeepSeek 4.0 | v2 + examples of the 3 most similar training papers | 0.5836 | 0.3602 | `20261119-134529_full_text_dev_prompt_v4` |
| 9 | 11-19 | F, prompt v4, DeepSeek 4.1 | newer model | 0.6644 | 0.3733 | `20261119-140809_full_text_dev_model_deepseek_v41` |
| 10 | 11-19 | #9 + verifier | second call per paper, keep/drop only | 0.6524 | 0.3603 | `20261119-151706_verify_dev` |
| 11 | 11-19 | #9 + service rule | drop services if no training neighbour has any (section 10) | 0.6644 | 0.4130 | `20261119-154747_full_text_dev` |
| 12 | 11-19 | **#11 + level prompt v1** | one extra call per part for biodiversity_level (section 11); not robust (bootstrap 73 %) | **0.6732** | **0.4202** | `20261119-161424_full_text_dev_level_prompt_v1` |
| 13 | 11-19 | #11 with prompt v5 | confidence per label (section 12); without threshold / cross-validated threshold | 0.6234 / 0.6133 | 0.4181 / 0.4515 | `20261119-162000_full_text_dev_prompt_v5` |


---

## 2. Pipeline F (full text): prompt versions

| Prompt | Change | C1 | C2b | Verdict |
|---|---|---|---|---|
| v1 | first prompt: "only what THIS study examined" | 0.5250 | 0.2854 | – |
| **v2** | "what THIS document is mainly about"; field base rates from train; habitat may be derived; at most 10 taxa | 0.5711 | 0.3315 | adopted |
| v3 | schema's guiding question per field; taxa base rate ("half only groups"); always a German name; quote length from settings | 0.5122 | 0.3180 | **rejected** |
| **v4** | v2 + labels of the 3 most similar training papers | 0.5836 | 0.3602 | adopted |

(v1 and v2 with checker 40 / 60 words, DeepSeek 4.0; v3 and v4 with checker 3–60 words.)

**What we learned**

- **v1 → v2 (+0.046 C1):** the reasoning of the 25 empty answers showed that the model left
  reports, policy papers, books and web pages empty, because they are "not a study". The
  database labels what a document is about.
- **v3 (−0.059 C1):** the schema's guiding questions are vaguer than the database's practice.
  "Welche Ökosystemtypen werden *betrachtet*?" raised ecosystem terms from 96 to 131; the
  taxa base rate made the model group species too coarsely (16 arable plants → one group,
  38 correct German names lost). Five changes at once made the causes hard to separate.
- **v4 (+0.013 C1, +0.029 C2b):** showing the database's labels for similar papers lowers the
  false labels (biodiversity_level FP 78 → 64, ecosystem_type.term 60 → 44) and raises
  habitat recall (0.66 → 0.73). Bootstrap (2,000 samples): v4 better than v2 in 71 % (C1) and
  91 % (C2b) of samples.

## 3. Models (pipeline F, prompt v4)

| Model | C1 | C2b | Note |
|---|---|---|---|
| deepseek-v4-flash-0731 | 0.5836 | 0.3602 | – |
| **deepseek-v4.1-flash-1017** | **0.6644** | **0.3733** | false ecosystem types halved (category FP 31 → 16, term 44 → 20); better than 4.0 in 100 % of bootstrap samples (C1), 65 % (C2b) |
| qwen3.5-397b-a17b | – | – | did not answer on SAIA (2 of 100 papers in 30 min), to be retried |
| glm-5.3-flash | – | – | did not answer on SAIA, to be retried |

## 4. Verbatim checker (quote length)

| Pipeline | Setting | C1 | C2b |
|---|---|---|---|
| F v1 | max 40 / 60 / 80 / 120 words | 0.5250 / 0.5196 / 0.5158 / 0.5119 | 0.2854 / 0.3113 / 0.3113 / 0.3109 |
| F v2 | max 40 / 60 / 80 / 120 words | 0.5680 / 0.5711 / 0.5711 / 0.5711 | 0.3302 / 0.3315 / 0.3315 / 0.3315 |
| F v2 | min 1 / 2 / 3 / 5 words | 0.5579 / 0.5691 / 0.5710 / 0.5711 | 0.3309 / 0.3305 / 0.3310 / 0.3315 |
| B (4 chunks) | max 40 / 60 / 80 / 120 words | 0.5787 / 0.5781 / 0.5781 / 0.5781 | 0.3178 (all) |
| B (4 chunks) | min 1 / 2 / 3 / 5 words | 0.5862 / 0.5872 / 0.5872 / 0.5781 | 0.3280 / 0.3302 / 0.3311 / 0.3178 |
| F v1 | checker off | 0.5218 | 0.3130 |

**Adopted:** 3 to 60 words. With prompt v2 the upper limit no longer matters. Almost all C3
drops in B were quotes under 5 words (taxon names, short terms); 1 word lets false labels
through in F. The checker itself hardly changes the score but guarantees that every label
has a verifiable quote; DeepSeek invents almost no quotes (C4 drops 0.6 %; Qwen3-30B 6.7 %).

## 5. Pipeline B (parallel chunks)

| Setting | C1 | C2b |
|---|---|---|
| Qwen3-30B-A3B, up to 20 chunks of 16K, union (k=1) | 0.4527 | 0.2070 |
| same, vote k=2 / k=3 | 0.4295 / 0.4189 | 0.1852 / 0.1626 |
| DeepSeek 4.0, at most 4 chunks, one-chunk papers as F's request | 0.5781 | 0.3178 |
| same, checker 3–60 words | **0.5872** | **0.3311** |
| same, chunk prompt v2 (from prompt v3) | 0.5505 | 0.2881 |

**What we learned**

- The large gap of the Qwen run was **not** the chunking: on papers that fit into one chunk
  (no chunking at all) F led by 0.26 C1, on multi-chunk papers only by 0.06. Causes: the
  weaker model and the chunk rule R9 ("you see only one chunk") on whole papers.
- With the **same model**, F and B are about equal (C1 0.571 vs 0.587, within noise). B is
  better on biodiversity_level (bootstrap 99.8 %), F on taxa names (97–99 %).
- Union of chunk answers creates many false taxa; voting (k=2, 3) loses more correct labels
  than it removes false ones.
- **Decision (11-19):** pipeline F is the main system (about 107 instead of 255 requests per
  dev run, simpler). B stays reproducible with `config/experiments/prompt_v2.yaml`.

## 6. Ensembles (no new requests)

| Rule | C1 | C2b |
|---|---|---|
| B + taxa from F (F v2, B 4 chunks) | 0.5800 | **0.3481** |
| labels both F v2 and B agree on | 0.5958 | – |
| union of F v2 and B | 0.5664 | – |
| vote over 4 runs (F v4.1, F v4, F v2, B v2): ≥1 / ≥2 / ≥3 / 4 | 0.5707 / 0.6113 / 0.6264 / 0.6052 | 0.3185 / 0.3731 / 0.3724 / 0.3767 |

**What we learned:** ensembles of similar systems (same model and prompt) gain little;
"agree" beat B in only 78 % of bootstrap samples. Adding weaker runs to F v4.1 makes every
vote worse than F v4.1 alone (0.6644): the other runs add 48 correct but 278 false labels.

## 7. Verifier (second LLM call, keep/drop only)

`config/prompts/verify_v1.txt`, on F v4.1, DeepSeek 4.1, 88 calls. Habitat not checked;
"if unsure: drop" for biodiversity_level and ecosystem_service, "if unsure: keep" for
ecosystem_type and taxa.

| Variant | C1 | C2b |
|---|---|---|
| F v4.1 (source) | 0.6644 | 0.3733 |
| verifier, all four fields | 0.6524 | 0.3603 |
| verifier, without taxa decisions | 0.6569 | 0.3625 |

| Field | dropped wrong | dropped right | F1 change |
|---|---|---|---|
| biodiversity_level | 21 | 9 | −0.053 |
| ecosystem_type.category / term | 3 / 3 | 1 / 1 | +0.010 / +0.006 |
| taxa.species_group | 5 | 7 | −0.023 |
| taxa.scientific_name / german_name | 10 / 10 | 0 / 2 | +0.010 / 0.000 |
| ecosystem_service.category / term | 16 / 16 | 3 / 2 | +0.010 / −0.021 |

**What we learned:** the verifier dropped mostly wrong labels and still lowered the score.
Dropping a wrong label raises F1 by about F1/D, dropping a right one lowers it by about
(2 − F1)/D (D = 2TP + FP + FN). Dropping only pays if wrong:right > (2 − F1)/F1, i.e. about
3.9 : 1 at F1 0.41 and 9 : 1 at F1 0.2. Weak fields need a very precise verifier.
**Not adopted.**

## 8. Retrieval of similar training papers (for prompt v4)

First 6,000 characters of every PDF (PyMuPDF, `src/first_pages.py`; about 30 train PDFs and
1 dev PDF without text layer). Quality check without an LLM: predict each dev paper by a vote
of its k nearest training papers (`src/scripts/evaluate_neighbours.py`).

| Method | k=1 | k=2 | k=3 | k=5 |
|---|---|---|---|---|
| embeddings (multilingual-e5-small), C1 / C2b | 0.4703 / 0.1749 | 0.4492 / 0.1673 | 0.4597 / 0.2074 | 0.4596 / 0.2000 |
| **TF-IDF**, C1 / C2b | 0.4793 / 0.2102 | 0.4829 / 0.2144 | **0.4645 / 0.2300** | 0.4790 / 0.2403 |

**Adopted:** TF-IDF, k=3. It reads the full 6,000 characters (e5 only about 512 tokens) and
matches exact taxon names. The neighbour vote alone is more than twice the baseline.

## 9. Analyses (no requests)

- **Error analysis of F v2:** 25 empty answers, all with labels in the reference (non-study
  documents); 70 % of missing scientific names occur verbatim in the text (level and
  convention errors, not missing information); biodiversity_level set in 72 of 100 papers,
  the database in 41.
- **Consistency of the training references (908):** species_group consistent for > 99 % of
  genera (exceptions: Collembola, Diplopoda); author citations in 3 of 2,281 scientific
  names (so: no authors); 99.8 % of taxa entries have a German name; 51 % of papers with taxa
  list only groups; "-" as scientific name in 98 papers; biodiversity_level and
  biodiversity_variable disagree in 23 % of papers. Some errors cannot be fixed by any
  system.
- **Upper bounds (perfect filter on the proposed labels):** F v4.1 alone C1 0.80; all runs
  together C1 0.89. The systems find most correct labels; the remaining gap is mostly false
  and some missing labels (biodiversity_level recall 42 %).
- **Ecosystem services:** all 67 terms mapped to a category by their CICES code prefix
  (1.x, 4.x Versorgungsleistungen, 2.x Regulierungs- und Erhaltungsleistungen, 3.x Kulturelle
  Leistungen); 514 train and 8 dev pairs, 0 exceptions.

## 10. Post-processing

Rules after the verbatim checker, before the predictions are written (`src/postprocessing.py`,
pipeline F only). They only remove labels and need no request.

**Service rule** (`postprocess.drop_services_without_neighbour_support`, default true; off:
`config/experiments/no_service_rule.yaml`): if a paper has training neighbours and none of
its 3 neighbours has an ecosystem_service entry, drop all its ecosystem_service entries. A
paper without neighbours (no first-pages text) keeps its services.

| Where | Neighbours without services: paper has services | At least one neighbour with services: paper has services |
|---|---|---|
| dev references (100) | 1 of 64 | 7 of 35 |
| train (878, 3 nearest other train papers, TF-IDF) | 35 of 525 (6.7 %) | 163 of 353 (46 %) |

| On F v4.1 | service category TP / FP | service term TP / FP | C1 | C2b |
|---|---|---|---|---|
| without rule (`no_service_rule.yaml`) | 5 / 25 | 4 / 29 | 0.6644 | 0.3733 |
| with rule (the default) | 5 / 12 | 4 / 15 | 0.6644 | **0.4130** |

Confirmed by two dev runs from the cache (0 API requests each): with the rule
(`runs/20261119-154747_full_text_dev`) C1 0.6644, C2b 0.4130; without it C1 0.6644, C2b 0.3733,
exactly F v4.1 (#9). So the +0.040 C2b comes only from the rule.

**What we learned:** the rule keeps all correct services and removes about half of the false
ones; better in 99.8 % of 2,000 bootstrap samples. The pattern holds on train, so it is a
property of the database, not of the dev set. The dev gain is probably optimistic (dev 1 of
64 vs train 6.7 %). **Adopted.**

## 11. Field prompt for biodiversity_level (decomposed prompting)

`config/prompts/level_v1.txt` (after Khot et al. 2023): after the main call, one more call per
part only for biodiversity_level, with a description and base rate per level and a confidence
per label. Its levels replace the main call's; a failed call keeps them. On F v4.1 + service
rule (#11):

| | C1 | C2b | level F1 | level TP / FP / FN | recall | precision |
|---|---|---|---|---|---|---|
| without (#11, `no_level_prompt.yaml`) | 0.6644 | 0.4130 | 0.412 | 28 / 41 / 39 | 0.42 | 0.41 |
| with level_v1 (#12) | **0.6732** | **0.4202** | **0.455** | 46 / 89 / 21 | 0.69 | 0.34 |

All other fields unchanged. Newly found: Habitatdiversität 5 TP (was 1), Genetische Diversität
4 TP (was 0). Better in 73 % of 2,000 bootstrap samples (over the 100 papers): **not robust**.
Confidence thresholds (`sweep_confidence.py`): no threshold 0.6732 / 0.4202, theory the same,
cross-validated 0.6666 / 0.4148, so no threshold.

**Finding: the level confidences are overconfident and barely separate right from wrong**
(cf. Xiong et al., ICLR 2024):

| confidence | 0.2–0.4 | 0.4–0.6 | 0.6–0.8 | 0.8–1.0 |
|---|---|---|---|---|
| right / kept labels | 3 / 11 | 3 / 12 | 5 / 25 | 35 / 87 |

**Decision: adopted as default, not robust (bootstrap 73 %).** Reasons: (1) the direction was
predicted beforehand: for a field with F1 about 0.41 a label pays off if P(right) > F1/2 ≈ 0.21
(Lipton et al. 2014), so more recall should help; the verifier, which dropped levels, had
lowered the field to 0.358. (2) Train has a level in 58 % of papers, dev in 41 %; on a test
set like train more recall should help more (hypothesis, medium confidence).
**Cost:** one extra call per part, about twice the requests of F (dev about 214 instead of 107;
test with 500 papers about 1,080 instead of 540).

## 12. Verbalized confidence (prompt v5)

`config/prompts/extraction_v5.txt` = v4 plus a `"confidence"` per label (rule R9, after Tian et
al. 2023); thresholds per field chosen with `src/scripts/sweep_confidence.py` (F1/2 "theory",
Lipton et al. 2014, and 5-fold cross-validation). On top of #11:

| v5 | C1 | C2b |
|---|---|---|
| no threshold | 0.6234 | 0.4181 |
| theory (F1 at 0 / 2) | 0.6214 | 0.4181 |
| cross-validated, out of fold | 0.6133 | 0.4515 |

| field F1 | v4 + service rule (#11) | v5, no threshold |
|---|---|---|
| habitat | 0.821 (TP 103) | 0.784 (TP 98) |
| biodiversity_level | 0.412 | 0.384 |
| ecosystem_type.category | 0.722 | 0.654 |
| ecosystem_type.term | 0.661 | 0.600 |
| taxa.species_group | 0.707 | 0.695 |
| taxa.scientific_name | 0.331 | 0.351 |
| taxa.german_name | 0.331 | 0.388 |

**Finding: these confidences do separate right from wrong** (unlike level_v1), right / kept
labels:

| confidence | 0.4–0.6 | 0.6–0.8 | 0.8–0.9 | 0.9–1.0 |
|---|---|---|---|---|
| habitat | 6 / 8 | 10 / 16 | 3 / 4 | 77 / 82 |
| biodiversity_level | 1 / 14 | 9 / 35 | 9 / 19 | 9 / 14 |
| ecosystem_type | 5 / 15 | 4 / 9 | 19 / 27 | 11 / 15 |
| taxa | 7 / 15 | 25 / 47 | 32 / 51 | 69 / 81 |

But R9 also made the model more liberal and it lost correct labels (habitat −5 TP); a
threshold removes wrong labels but cannot bring back lost ones. Taxa names from v5 combined
with #12: C2b 0.4308 but C1 0.6706, and two main calls per paper: not used.
**Decision: rejected.** Lesson: R9 changed two things at once (confidence and more liberal
answers). `postprocess.min_confidence`, `labels_with_confidence.jsonl` and the sweep script stay.

## 13. Infrastructure

| Finding | Consequence |
|---|---|
| SAIA limits (10 / 10 s, 400 / h, 800 / day) apply per **account**, not per API key | more keys of one account do not help; one key per team member |
| the hourly limit refills gradually (2 → 16 → 53 → 107 within minutes) | open: the limiter assumes a 60-minute window |
| DeepSeek was unreachable for about an hour (11-19); Qwen 3.5 and GLM did not answer | process the test PDFs early in the test week |
| 2 Docling workers were not faster than 1 (500 s vs 508 s on 5 papers) | default 1 worker; converting 500 test PDFs takes about 21 h |
| Qwen3-30B timed out on 240K-token chunks; DeepSeek handled 900K-token parts | – |

## 14. Open

- Retry qwen3.5-397b-a17b and glm-5.3-flash (`config/experiments/model_*.yaml`).
- Test phase: Docling conversion, first pages and neighbours for the 500 test papers.
