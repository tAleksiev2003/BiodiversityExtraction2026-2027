# How to run

Run every command from the repository root. Each prediction command creates a new
folder `runs/<timestamp>_<name>/` with `predictions.jsonl` and `predictions.zip`.

## Two dev sets

| Set | Papers | Used for |
|---|---|---|
| `with_pdf` | ~100 with a PDF | developing and comparing our systems (default) |
| `full` | all 900 in `dev.jsonl` | checking our scorer against the official numbers, CodaBench uploads |

## 1. Once

```bash
python3 -m pytest                          # tests on invented data, expect: all passed
python3 src/data.py make-dev-references    # writes data/processed/dev_with_pdf.jsonl
```

## 2. Predict, then score

Pattern: **predict** with `baselines.py` (later: the pipelines), then **score** the newest run.

```bash
# most-frequent baseline on the with_pdf set
python3 src/baselines.py --kind most_frequent --split dev --dev-set with_pdf
LATEST=$(ls -td runs/*_baseline_most_frequent_dev_with_pdf | head -1)
python3 src/evaluation.py score --references data/processed/dev_with_pdf.jsonl \
    --predictions "$LATEST/predictions.jsonl" --task all --categorical-only \
    --output "$LATEST/scores.json"
```

For the random baseline replace `most_frequent` by `random` (twice).

For the `full` set use `--dev-set full`, `*_dev_full` and
`--references data/raw/references/dev.jsonl`.

## 3. Convert PDFs and chunk them (M1)

```bash
python3 src/pdf_to_text.py --split dev --limit 3   # quick check: 3 documents
python3 src/pdf_to_text.py --split dev             # all dev documents with a PDF
python3 src/document_statistics.py --split dev     # page / token / chunk statistics
```

`pdf_to_text.py` stores every document as Docling JSON in `data/processed/docling/<split>/`
and skips documents that are already there, so `Ctrl + C` is safe: run it again to continue.
Failed documents are retried on every run. Every attempt is logged in
`data/processed/docling/<split>/conversion_log.jsonl`.

`document_statistics.py` writes `runs/<timestamp>_m1_statistics_<split>/` with
`summary.json` (report this back), `documents.jsonl` and `chunking_log.jsonl`
(chunking level and every split per document).
It prints the runtime of every document and stores each result as soon as it is done
(`documents/<id>.json` in the run folder). After `Ctrl + C`, continue the same run with
the command it printed at the start, e.g.
`python3 src/document_statistics.py --split dev --resume runs/<timestamp>_m1_statistics_dev`.

## Reference numbers (C1 score)

| Baseline | with_pdf (ours) | full (ours) | full (official) |
|---|---|---|---|
| random | – | 0.1001 | 0.1018 |
| most frequent | 0.2055 | 0.2443 | 0.2443 |

Upload `predictions.zip` of a `full` run to CodaBench; it must show the same score.