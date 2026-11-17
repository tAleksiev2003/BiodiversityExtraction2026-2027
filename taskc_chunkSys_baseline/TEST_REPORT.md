# Test report: 15-PDF parallel run

Run directory: `runs/pdfs_15_parallel_deepseek4.1/`  
Date: 17 November 2026  

## Why this test uses 15 PDFs

The development pool has 100 PDFs (`data/ids_100.txt`, input directory `data/pdfs_dev_100`). That list was not used for this scored run.

In parallel mode every chunk is one chat completion. Some PDFs in the 100-document pool are long enough to split into more than 200–300 chunks. One such paper spends a few hundred SAIA requests by itself, and a full pass over 100 PDFs would go past the SAIA daily request quota on `https://chat-ai.academiccloud.de/v1`. The scored set is therefore 15 PDFs of workable length (`data/ids_15.txt`).

On this configuration those 15 papers produced **178 chunks and 178 calls**. The longest paper has 26 chunks. None of them is in the 200–300 chunk range.

Earlier attempts on the 100-id list did not finish, which matches the quota constraint:

| Attempt | Stopped on | What happened | Papers completed |
| --- | --- | --- | --- |
| `runs/pdfs_100_parallel` | `2E9XWUUE` | Server disconnected | 3 |
| `runs/pdfs_100_parallel01` | `2E9XWUUE` | Connection refused | 3 |
| `runs/pdfs_100_parallel02` | `2P53UVJA` | HTTP 429 Too Many Requests | 7 |
| `runs/pdfs_100_parallel03` | `2P53UVJA` | HTTP 429 Too Many Requests | 7 |

## Setup

| Item | Value |
| --- | --- |
| Command | `run`, parallel mode, `--workers 1` |
| Model | `deepseek-v4.1-flash-1017` |
| Endpoint | `https://chat-ai.academiccloud.de/v1` (SAIA / Academic Cloud) |
| PDF conversion | GROBID at `http://localhost:8070` |
| Chunks | 5000 characters, overlap 300 |
| Evidence | exact normalized quote (`--quote-threshold 1.0`), `--min-votes 1` |
| Context / output budget | 65000 / 16384 |
| JSON mode | on |
| Gold | `data/dev_references_15.jsonl` (same 15 ids) |
| Scores | `metrics.json`, including `C2b free text (BERTScore)` from `bert-base-multilingual-cased` |

`workers` is 1, so HTTP calls were issued one at a time. The mode is still parallel in the pipeline sense: each chunk is extracted with an empty previous state, and removals are rejected.

The run completed all 15 papers and wrote `predictions.jsonl` and `predictions.zip`. There is no `failure.json` in this directory.

## Runtime

| Measure | Value |
| --- | --- |
| Wall time | 1284 s (21.4 min) |
| Papers | 15 |
| Chunks / calls | 178 / 178 |
| Cache hits | 54 |
| Fresh calls | 124 |
| Fresh prompt tokens | 1,345,879 |
| Fresh completion tokens | 482,257 |
| Token usage reported for every fresh call | yes |

The 54 cache hits are exactly the five papers that an earlier 100-id attempt had already finished: `25ABQZIH`, `25RIYD2C`, `29Q84X7V`, `2FDHAHAR`, `2NJD9Y9T`. Their predictions come from cached model JSON, not from new SAIA calls during this run. A cold run of the same 15 PDFs would send all 178 requests. Fresh-token totals above cover only the other ten papers.

Quote checking rejected 3 of 934 proposed additions:

- 2 quotes were not found in the current chunk.
- 1 enum was unknown: `Küsten und Küstengwässer` (a garbled habitat label; the catalogue value is `Küsten und Küstengewässer`).

The other 931 additions passed the verbatim-quote gate. Errors below are mostly extra findings, not invented quotations.

## Papers

Chunk counts are the GROBID section text after the 5000/300 split. “Kept” is the number of findings that remained after the vote filter and schema validation.

| ID | Year | Chunks | Fresh calls | Cache hits | Kept findings | Title |
| --- | --- | --- | --- | --- | --- | --- |
| 25ABQZIH | 2022 | 6 | 0 | 6 | 14 | Informationspapier des Greifswald Moor Centrum zu Photovoltaik-Anlagen auf Moorböden |
| 25RIYD2C |  | 2 | 0 | 2 | 0 | Grünes Band Deutschland: Chronik |
| 29Q84X7V | 2017 | 13 | 0 | 13 | 5 | What we can learn from the German implementation of greening |
| 2FDHAHAR | 2013 | 15 | 0 | 15 | 73 | Climate change and eutrophication induced shifts in northern summer plankton communities |
| 2NJD9Y9T | 2022 | 18 | 0 | 18 | 87 | Ocean acidification drives global reshuffling of ecological communities |
| 2RGQT8TU | 2022 | 17 | 17 | 0 | 15 | Safeguarding food security and reinforcing the resilience of food systems |
| 33MV2F7X | 2022 | 5 | 5 | 0 | 6 | Vorsorgender Bodenschutz in Deutschland |
| 3439JHQD | 2014 | 2 | 2 | 0 | 8 | Citizen Science in der Naturschutzarbeit |
| 38EBGVBF | 2012 | 26 | 26 | 0 | 16 | Art und Ausmaß der Inanspruchnahme landwirtschaftlicher Flächen |
| 3RSTTAEZ | 2020 | 15 | 15 | 0 | 31 | Less Nutrients but More Phytoplankton: Long-Term Ecosystem Dynamics of the Southern North Sea |
| 3TWLQG7F | 2015 | 21 | 21 | 0 | 64 | Establishment of blue mussel beds to enhance fish habitats |
| 42ELXZ2H | 2006 | 19 | 19 | 0 | 35 | Vom Kreislauf zum Durchfluss: Österreichs Agrarmodernisierung |
| 49Z7RPDP | 2019 | 4 | 4 | 0 | 15 | Vielfalt und Leistungen nutzen — Regenwürmer in bayerischen Ackerböden |
| 4FCS62W5 | 2009 | 8 | 8 | 0 | 21 | Elevated CO2 stimulates marsh elevation gain, counterbalancing sea-level rise |
| 4JF8F3WL | 2020 | 7 | 7 | 0 | 40 | Säugetiere (Mammalia) |

Chunk counts run from 2 to 26 (mean 11.9). Predicted value counts across the 15 papers: habitat 27, biodiversity level 22, ecosystem type 43, taxa 148, biodiversity variable 161, ecosystem service 29.

## Scores

Pooled exact match over the 15 papers. Macro F1 is the unweighted mean of the field F1 scores. Micro F1 pools TP, FP, and FN first, then computes one F1. An empty gold-and-prediction pair contributes nothing to that field; a field with no true positives and some false positives scores 0.

| Output key | Score |
| --- | ---: |
| C1 Score(Macro F1) | 0.455 |
| C2b biodiversity (Macro F1) | 0.232 |
| All categorical values (Macro) | 0.328 |
| All categorical values (Micro) | 0.328 |
| C2b free text (BERTScore) | 0.035 |

All categorical values use the nine exact-match fields, with `taxa.species_group` and `biodiversity_level` counted once. At full precision the macro is 0.3280 and the micro is 0.3282. `C2b free text (BERTScore)` is 0.035 because 154 of 161 predicted variable strings have no gold partner and score 0; the arithmetic is in the section below.

### Task C1

**C1 Score(Macro F1): 0.455**

| Field | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| habitat | 16 | 11 | 6 | 0.593 | 0.727 | 0.653 |
| biodiversity_level | 4 | 18 | 1 | 0.182 | 0.800 | 0.296 |
| ecosystem_type.category | 6 | 11 | 0 | 0.353 | 1.000 | 0.522 |
| ecosystem_type.term | 6 | 18 | 1 | 0.250 | 0.857 | 0.387 |
| taxa.species_group | 10 | 26 | 2 | 0.278 | 0.833 | 0.417 |

### Task C2b, exact match

**C2b biodiversity (Macro F1): 0.232**

| Field | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| taxa.scientific_name | 17 | 57 | 11 | 0.230 | 0.607 | 0.333 |
| taxa.german_name | 14 | 82 | 18 | 0.146 | 0.438 | 0.219 |
| taxa.species_group | 10 | 26 | 2 | 0.278 | 0.833 | 0.417 |
| biodiversity_level | 4 | 18 | 1 | 0.182 | 0.800 | 0.296 |
| ecosystem_service.category | 1 | 14 | 0 | 0.067 | 1.000 | 0.125 |
| ecosystem_service.term | 0 | 26 | 1 | 0.000 | 0.000 | 0.000 |

`taxa.species_group` and `biodiversity_level` are in both the C1 and C2b averages. The all-categorical scores include each field once.

### C2b free text (BERTScore)

**C2b free text (BERTScore) = 0.035 because almost every predicted string is unmatched and counts as 0.**

Gold has 7 `biodiversity_variable` strings, on `2FDHAHAR` (2), `2NJD9Y9T` (3), `3RSTTAEZ` (1), and `3TWLQG7F` (1). The model predicted 161 strings. On every paper the prediction count is at least the gold count, so the denominator is 161. Greedy one-to-one alignment pairs all 7 gold strings. Those pairs have a mean BERTScore F1 of 0.805, and the other 154 predictions score 0:

`7 × 0.805 / 161 = 0.035`

Exact string overlap is still empty: none of the 7 gold phrases is copied verbatim. The 0.805 says the paired phrases are semantically close. The 0.035 says those 7 pairs are diluted by 154 extra measurements. BERTScore uses `bert-base-multilingual-cased`, `lang=de`, and no baseline rescaling. This remains a local diagnostic. Confirm the official aggregation before quoting it as a task score.

## What the errors look like

Recall is high and precision is low on almost every categorical field. The model keeps findings that the quote check can support, including mentions gold does not mark as what the study investigates.

- **Habitat** is the strongest field (F1 0.653). `4JF8F3WL` matches all five gold habitats. `3439JHQD` misses all five (`Agrar- und Offenland`, `Binnengewässer und Auen`, `Boden`, `Urbane Räume`, `Wald`) and predicts none. `25RIYD2C` is a two-chunk paper with an empty prediction; gold has `Agrar- und Offenland`. Several other papers add a second habitat (`Boden`, `Küsten und Küstengewässer`) beside a correct one.
- **Ecosystem type category** has perfect recall (6/6) and 11 extra labels. Terms are similar: 6 true positives, 18 false positives, 1 miss (`Benthal der Ostsee` on `3TWLQG7F`).
- **Biodiversity level** is usually an over-prediction. Gold is empty on 11 of 15 papers, and the model still emits `Artenvielfalt` or a second level. The only false negative is `Artenvielfalt` on `2FDHAHAR`, where `Funktionelle Diversität` was predicted instead of, and in addition to, the gold levels.
- **Taxa** are extracted far above the gold set. Scientific names (F1 0.333) are better than German names (F1 0.219). German-name misses are often a different rank or wording for the same organism group (`Cryptophyceae` versus `Cryptomonaden`, `Gemeine Miesmuschel` versus a scientific-name-only item). `42ELXZ2H` and `4JF8F3WL` add large name lists where gold has no names, or only a species group.
- **Ecosystem service** is the weakest block. Gold has a service annotation on `49Z7RPDP` only. Category matches that one paper and adds 14 extra categories elsewhere (F1 0.125). Term has no true positive: 26 predicted terms, and the single gold term (`2.2.6.2 Regulierung von Temperatur und Feuchtigkeit...` on `49Z7RPDP`) is missed (F1 0.000).
- **Biodiversity variables** are long German phrases. They survive the quote check and then fail exact match against the short gold inventory. BERTScore says a few of them are near a gold phrase; most are additional measurements.

Many gold records are sparse. Six papers (`25RIYD2C`, `29Q84X7V`, `2RGQT8TU`, `33MV2F7X`, `3439JHQD`, `42ELXZ2H`) have only a habitat label filled in for the scored fields, so any ecosystem type, taxon, variable, or service the model returns is a false positive even when the sentence is about the paper. That sparsity is part of why the C2b biodiversity macro F1 of 0.232 sits well below the C1 score of 0.455.

## Limits of this report

- Five papers were served from the LLM cache. The finding text is still the model’s, but this run did not re-request those 54 calls.
- The 15 PDFs were chosen for length so the run would stay inside the SAIA daily request cap. They are not a random sample of the 100, and they exclude the long documents that dominate request cost.
- `workers` was 1. Parallel mode here changes merging rules, not throughput.
