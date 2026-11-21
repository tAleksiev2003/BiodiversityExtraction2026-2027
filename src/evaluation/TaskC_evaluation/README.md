# Task C evaluation

Standalone evaluator for biodiversity information extraction. Requires Python 3.10+.
Categorical scoring uses only the Python standard library; free-text scoring additionally
requires `python -m pip install -r requirements-bertscore.txt`.

## Usage

```sh
python evaluate.py score --references references.jsonl --predictions predictions.jsonl \
  --task c1 --output results.json
python evaluate.py score --references references.jsonl --predictions predictions.jsonl \
  --task c2b --output results.json
```

`--task` accepts `c1`, `c2a`, `c2b`, `c2c`, or `all` (default). Every selection includes
C1. Add `--categorical-only` to omit free-text metrics without loading a model. Without
`--output`, the detailed JSON report is printed to stdout. Errors exit with status 1.

Both inputs are UTF-8 JSONL files with one object per paper, linked by the string
`zotitem_ptr_id`. Prediction IDs must match reference IDs exactly; ordering is irrelevant.
Duplicate IDs, duplicate JSON keys, missing/unknown IDs, and malformed scored values are
errors. Omitted fields, `null`, empty containers and blank strings count as empty. Values
must be strings or lists of strings; nested fields use objects or lists of objects.
Unscored fields are ignored. Example prediction:

```json
{"zotitem_ptr_id":"N82RZ7WL","habitat":["Küsten und Küstengewässer"],"taxa":[{"species_group":"Andere Pflanzen"}]}
```

Driver text accepts `description` or the dataset's `details` key in either input.
Both map to the reported `.description` field. If both keys occur in one driver object,
their normalized value sets must agree; otherwise evaluation fails.

## Scoring

| Group | Evaluated fields                                                                                                                                                          |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| C1    | `habitat`, `biodiversity_level`, `ecosystem_type.category`, `ecosystem_type.term`, `taxa.species_group`                                                                   |
| C2a   | `natural_region`, `climate`, `landuse`, `soil.name`, `soil.depth`                                                                                                         |
| C2b   | `taxa.scientific_name`, `taxa.german_name`, `taxa.species_group`, `biodiversity_variable`\*, `biodiversity_level`, `ecosystem_service.category`, `ecosystem_service.term` |
| C2c   | `direct_driver.category`, `direct_driver.description`*, `indirect_driver.category`, `indirect_driver.description`*                                                        |

For each field, values are deduplicated per paper. Nested subfields are evaluated
independently: category/term or category/description associations are not scored.
Nonblank strings are retained exactly, including case and whitespace; use German labels
as specified in the task schemas. Scientific and German taxa names use exact matching too.
Unknown labels count as false positives; the evaluator does not validate enum membership.

For exact-match fields, TP/FP/FN are pooled across papers. Precision is `TP/(TP+FP)`, recall
is `TP/(TP+FN)`, and F1 is their harmonic mean. Undefined divisions return 0, including
fields empty on both sides. **The ranking score, `c1_score`, is the mean F1 of the five C1
fields.** Group macro F1 averages exact-match fields; micro F1 pools their counts. Aggregate
`c2` covers selected C2 groups, and `all` covers C1 plus selected C2 groups. Shared fields
are counted once in each aggregate. Free text never contributes to categorical F1.

### Free text (\*)

For a paper and field, let `P` and `R` be the predicted and reference sets. Compute
pairwise BERTScore F1 `s(p,r)` and sum both directions of best matches:

```text
S = sum(p in P) max(r in R) s(p,r) + sum(r in R) max(p in P) s(p,r)
field BERTScore = sum_over_papers(S) / sum_over_papers(|P| + |R|)
```

Matches may be reused. If either set is empty, `S=0`; one-sided empty sets still contribute
to the denominator. A zero denominator yields 0. Group BERTScore pools these sums and
denominators across its text fields. Raw pair scores are not clipped or rescaled.

Settings are fixed in the script: `google-bert/bert-base-multilingual-cased`, revision
`3f076fdb1ab68d5b2880cb87a0886f315b8146f8`, layer 9, slow cased tokenizer, no IDF or baseline
rescaling, CPU, one thread, seed 0, deterministic algorithms, batches of 32. The tokenizer
truncates texts to the model's 512-token limit including special tokens. Use the pinned
requirements and the same runtime image for reproducible results; small numerical
differences across platforms remain possible. The report records settings and loaded
package versions. The first nonempty text comparison downloads the model; predictions
and references are processed locally. To preload the cache, run:

```sh
python evaluate.py prepare-model
# After preloading, require cached files only:
HF_HUB_OFFLINE=1 python evaluate.py score --references references.jsonl \
  --predictions predictions.jsonl --task c2b --output results.json
```

`HF_HOME` selects the model cache directory. No model is needed for C1 or C2a.

## CodaBench

Submit a ZIP containing exactly `predictions.jsonl` at its root. For a local simulation,
place gold data at `input/ref/references.jsonl` and predictions at
`input/res/predictions.jsonl`, then run:

```sh
python evaluate.py codabench input output
```

This uses the same evaluator, writes flat leaderboard values to `output/scores.json`,
and full per-field counts/settings to `output/detailed_scores.json`. Per-field scores
also appear in stdout. `metadata.yaml` supplies the hosted command. Install dependencies
and preload the model in the scoring environment; add `--task` to the metadata command
if the competition evaluates a specific C2 group. Leaderboard keys include `c1_score`,
`c2_macro_f1`, `all_micro_f1`, group keys such as `c2b_macro_f1` and `c2b_bertscore`, and
field keys such as `ecosystem_type_term_f1`. Text keys are omitted with `--categorical-only`.
