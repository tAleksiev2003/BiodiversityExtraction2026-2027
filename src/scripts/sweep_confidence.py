"""Choose `postprocess.min_confidence` thresholds from an F run, without new requests.

Reads labels_with_confidence.jsonl of a run (src/label_file.py) and rebuilds its
predictions for every threshold. The candidates are the labels the run would write
without thresholds: kept by the verbatim checker, not replaced by a field prompt and not
dropped by the service rule or the ecosystem_type rule (the service rule does not depend
on confidence; the ecosystem_type rule depends on the predicted habitats, which a habitat
threshold could change, so with both a habitat threshold and that rule on the sweep is an
approximation). Labels the run's own thresholds dropped are candidates again. A label without a confidence is kept
at every threshold, as in src/postprocessing.py.

A threshold applies to all labels of a field (the unit of the setting); a field with
subfields (taxa, ecosystem_type, ecosystem_service) is scored per subfield, as in
src/evaluation.py, and its threshold is chosen by the mean F1 of its scored subfields.

Printed per scored field: F1 at thresholds 0.0, 0.1, ..., 0.9. Per field:
- theory: F1 at threshold 0 divided by 2, rounded to the grid. For a calibrated
  classifier the F1-optimal threshold is half the best F1 (Lipton et al. 2014); F1 at 0
  stands in for the best F1;
- cross-validated: 5 folds of papers (seed `CV_SEED`); in each fold the threshold that is
  best on the other folds is applied to the held-out papers. Out-of-fold F1 estimates
  how well choosing a threshold on dev generalises; the suggested value is the median of
  the 5 fold choices.
At the end: C1 and C2b without thresholds, with the theory thresholds and out of fold.

Prints only numbers and schema labels (field names), never text from papers.

Usage (run by the team; it reads course data):
    python3 src/scripts/sweep_confidence.py --run runs/<run folder> \
        --references data/processed/dev_with_pdf.jsonl
"""
import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from src/
from data import ID_FIELD, index_by_id, load_references, read_jsonl  # noqa: E402
from evaluation import C1_FIELDS, C2B_FIELDS, evaluate, score_field  # noqa: E402
from label_file import LABEL_FILE_NAME  # noqa: E402
from output import PREDICTIONS_FILENAME  # noqa: E402
from postprocessing import RULE_ECOSYSTEM_TYPE, RULE_SERVICES  # noqa: E402

THRESHOLDS = [round(step / 10, 1) for step in range(10)]   # 0.0, 0.1, ..., 0.9
FOLDS = 5
CV_SEED = 7
SCORED_FIELDS = list(dict.fromkeys(C1_FIELDS + C2B_FIELDS))

Candidate = tuple[str, Any, float | None]   # field, entry, confidence
Records = dict[str, dict[str, Any]]


def parent_of(scored_field: str) -> str:
    return scored_field.partition(".")[0]


def parent_fields() -> list[str]:
    """Fields a threshold can be set for, in score order."""
    return list(dict.fromkeys(parent_of(name) for name in SCORED_FIELDS))


def is_candidate(line: dict[str, Any]) -> bool:
    return (line["kept_by_checker"] and not line["replaced_by_field_prompt"]
            and line["postprocessing_rule"] not in (RULE_SERVICES, RULE_ECOSYSTEM_TYPE))


def load_candidates(run_dir: Path) -> dict[str, list[Candidate]]:
    """Candidate labels per document id."""
    path = run_dir / LABEL_FILE_NAME
    if not path.exists():
        sys.exit(f"{path} missing: the run was made before labels_with_confidence.jsonl existed")
    candidates: dict[str, list[Candidate]] = defaultdict(list)
    for line in read_jsonl(path):
        if is_candidate(line):
            candidates[line["document_id"]].append((line["field"], line["value"], line["confidence"]))
    return candidates


def build_records(candidates: dict[str, list[Candidate]], document_ids: list[str],
                  thresholds: dict[str, float]) -> Records:
    """Prediction records with every label at or above its field's threshold (or without confidence)."""
    records: Records = {}
    for document_id in document_ids:
        record: dict[str, Any] = {ID_FIELD: document_id}
        for field, entry, confidence in candidates.get(document_id, []):
            threshold = thresholds.get(field, 0.0)
            if confidence is None or confidence >= threshold:
                record.setdefault(field, []).append(entry)
        records[document_id] = record
    return records


def field_f1(references: Records, records: Records, scored_fields: list[str]) -> float:
    """Mean F1 of the given scored fields, summed over the papers in `references`."""
    return statistics.mean(score_field(references, records, name).f1 for name in scored_fields)


def subfields_of(parent: str) -> list[str]:
    return [name for name in SCORED_FIELDS if parent_of(name) == parent]


def on_grid(value: float) -> float:
    return min(THRESHOLDS, key=lambda threshold: abs(threshold - value))


def best_threshold(candidates: dict[str, list[Candidate]], references: Records, parent: str) -> float:
    """Threshold with the highest mean F1 of the field's subfields; ties: the lowest."""
    document_ids = list(references)
    scores = [(field_f1(references, build_records(candidates, document_ids, {parent: threshold}),
                        subfields_of(parent)), -threshold) for threshold in THRESHOLDS]
    return -max(scores)[1]


def folds_of(document_ids: list[str]) -> list[list[str]]:
    shuffled = sorted(document_ids)
    random.Random(CV_SEED).shuffle(shuffled)
    return [shuffled[index::FOLDS] for index in range(FOLDS)]


def cross_validate(candidates: dict[str, list[Candidate]], references: Records,
                   parent: str) -> tuple[list[float], Records]:
    """Fold choices and the out-of-fold records (only this field's threshold applied)."""
    choices: list[float] = []
    out_of_fold: Records = {}
    for held_out in folds_of(list(references)):
        held_out_set = set(held_out)
        training = {document_id: reference for document_id, reference in references.items()
                    if document_id not in held_out_set}
        threshold = best_threshold(candidates, training, parent)
        choices.append(threshold)
        out_of_fold.update(build_records(candidates, held_out, {parent: threshold}))
    return choices, out_of_fold


def task_scores(references: Records, records: Records) -> dict[str, float]:
    return evaluate(references, records, "all")["task_scores"]


def main() -> None:
    parser = argparse.ArgumentParser(description="F1 per confidence threshold of an F run.")
    parser.add_argument("--run", type=Path, required=True, help="run folder with labels_with_confidence.jsonl")
    parser.add_argument("--references", type=Path, required=True)
    args = parser.parse_args()

    candidates = load_candidates(args.run)
    predicted_ids = list(index_by_id(read_jsonl(args.run / PREDICTIONS_FILENAME), PREDICTIONS_FILENAME))
    all_references = load_references(args.references)
    references = {document_id: all_references[document_id] for document_id in predicted_ids}
    document_ids = list(references)
    labels = [candidate for document_id in document_ids for candidate in candidates.get(document_id, [])]
    without_confidence = sum(1 for _, _, confidence in labels if confidence is None)
    print(f"papers: {len(document_ids)}, candidate labels: {len(labels)}, without confidence: {without_confidence}")

    print(f"\nF1 per threshold\n{'field':30s}" + "".join(f"{threshold:>7.1f}" for threshold in THRESHOLDS))
    for scored_field in SCORED_FIELDS:
        f1_values = [score_field(references, build_records(candidates, document_ids,
                                                           {parent_of(scored_field): threshold}),
                                 scored_field).f1 for threshold in THRESHOLDS]
        print(f"{scored_field:30s}" + "".join(f"{value:7.4f}" for value in f1_values))

    theory: dict[str, float] = {}
    cross_validated: dict[str, float] = {}
    out_of_fold_by_field: dict[str, Records] = {}
    print(f"\n{'field':22s} {'F1 at 0':>8s} {'theory':>7s} {'fold choices':>28s} {'chosen':>7s} {'OOF F1':>7s}")
    for parent in parent_fields():
        f1_at_zero = field_f1(references, build_records(candidates, document_ids, {}), subfields_of(parent))
        theory[parent] = on_grid(f1_at_zero / 2)
        choices, out_of_fold_by_field[parent] = cross_validate(candidates, references, parent)
        cross_validated[parent] = on_grid(statistics.median(choices))
        out_of_fold_f1 = field_f1(references, out_of_fold_by_field[parent], subfields_of(parent))
        print(f"{parent:22s} {f1_at_zero:8.4f} {theory[parent]:7.1f} "
              f"{', '.join(f'{choice:.1f}' for choice in choices):>28s} {cross_validated[parent]:7.1f} "
              f"{out_of_fold_f1:7.4f}")

    # Out of fold for all fields at once: each field's labels come from its own out-of-fold records.
    combined: Records = {document_id: {ID_FIELD: document_id} for document_id in document_ids}
    for parent, records in out_of_fold_by_field.items():
        for document_id, record in records.items():
            if parent in record:
                combined[document_id][parent] = record[parent]
    print(f"\n{'setting':38s} {'C1':>7s} {'C2b':>7s}")
    for name, records in (
        ("no threshold", build_records(candidates, document_ids, {})),
        ("theory (F1 at 0 / 2)", build_records(candidates, document_ids, theory)),
        (f"cross-validated, out of fold ({FOLDS} folds)", combined),
    ):
        scores = task_scores(references, records)
        print(f"{name:38s} {scores['c1']:7.4f} {scores['c2b']:7.4f}")

    suggested = {parent: threshold for parent, threshold in cross_validated.items() if threshold > 0}
    print(f"\nsuggested postprocess.min_confidence (cross-validated): {json.dumps(suggested)}")


if __name__ == "__main__":
    main()
