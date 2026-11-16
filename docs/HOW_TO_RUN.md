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

## Reference numbers (C1 score)

| Baseline | with_pdf (ours) | full (ours) | full (official) |
|---|---|---|---|
| random | – | 0.1001 | 0.1018 |
| most frequent | 0.2055 | 0.2443 | 0.2443 |

Upload `predictions.zip` of a `full` run to CodaBench; it must show the same score.