"""Score predictions against references, as documented on CodaBench (SPEC.md, section 7).

For each field and document the *set* of predicted values is compared with the set of
reference values (exact, case-sensitive). TP, FP and FN are summed over all documents,
then precision, recall and F1 are computed per field. Subfields such as
`ecosystem_type.term` are separate fields. A missing field counts as empty.

Usage (same interface as the official script, so it can be swapped in):
    python src/evaluation.py score --references dev.jsonl --predictions predictions.jsonl \
        --task all [--categorical-only] [--output scores.json]
"""
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from data import load_references, read_jsonl, index_by_id
from settings import load_settings

# Field names (not label values) that make up the two scores, from the CodaBench page.
C1_FIELDS = [
    "habitat",
    "biodiversity_level",
    "ecosystem_type.category",
    "ecosystem_type.term",
    "taxa.species_group",
]
C2B_FIELDS = [
    "taxa.scientific_name",
    "taxa.german_name",
    "taxa.species_group",
    "biodiversity_level",
    "ecosystem_service.category",
    "ecosystem_service.term",
]
FREE_TEXT_FIELDS = ["biodiversity_variable"]  # BERTScore; not used for the ranking
TASK_FIELDS = {"c1": C1_FIELDS, "c2b": C2B_FIELDS}


@dataclass
class FieldScore:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        relevant = self.true_positives + self.false_negatives
        return self.true_positives / relevant if relevant else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0


def extract_values(record: dict[str, Any] | None, field: str) -> set[str]:
    """Collect the set of values of a field such as `habitat` or `taxa.species_group`.

    None and empty strings count as "no value", e.g. a taxon without a German name or an
    ecosystem type with `term: null`.
    """
    if record is None:
        return set()
    parent, _, subfield = field.partition(".")
    items = record.get(parent) or []
    if not isinstance(items, list):
        items = [items]
    if subfield:
        raw_values = [item.get(subfield) for item in items if isinstance(item, dict)]
    else:
        raw_values = items
    return {value for value in raw_values if isinstance(value, str) and value}


def score_field(
    references: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    field: str,
) -> FieldScore:
    """Sum TP, FP and FN of one field over all reference documents."""
    score = FieldScore()
    for document_id, reference in references.items():
        gold = extract_values(reference, field)
        predicted = extract_values(predictions.get(document_id), field)
        score.true_positives += len(gold & predicted)
        score.false_positives += len(predicted - gold)
        score.false_negatives += len(gold - predicted)
    return score


def check_prediction_ids(references: dict[str, Any], predictions: dict[str, Any]) -> None:
    """Predictions for unknown documents are a format error, not just a false positive."""
    unknown_ids = set(predictions) - set(references)
    if unknown_ids:
        raise ValueError(f"{len(unknown_ids)} predicted ids are not in the references: {sorted(unknown_ids)}")


def evaluate(
    references: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    task: str,
) -> dict[str, Any]:
    """Return per-field scores and the task scores (mean F1 over the task's fields)."""
    check_prediction_ids(references, predictions)
    tasks = list(TASK_FIELDS) if task == "all" else [task]
    fields = list(dict.fromkeys(field for name in tasks for field in TASK_FIELDS[name]))

    field_scores = {field: score_field(references, predictions, field) for field in fields}
    task_scores = {
        name: sum(field_scores[field].f1 for field in TASK_FIELDS[name]) / len(TASK_FIELDS[name])
        for name in tasks
    }
    return {
        "documents": len(references),
        "fields": {
            field: {**asdict(score), "precision": score.precision, "recall": score.recall, "f1": score.f1}
            for field, score in field_scores.items()
        },
        "task_scores": task_scores,
    }


def print_report(result: dict[str, Any]) -> None:
    print(f"documents: {result['documents']}")
    print(f"{'field':30s} {'P':>7s} {'R':>7s} {'F1':>7s} {'TP':>6s} {'FP':>6s} {'FN':>6s}")
    for field, values in result["fields"].items():
        print(
            f"{field:30s} {values['precision']:7.4f} {values['recall']:7.4f} {values['f1']:7.4f} "
            f"{values['true_positives']:6d} {values['false_positives']:6d} {values['false_negatives']:6d}"
        )
    for name, value in result["task_scores"].items():
        print(f"{name.upper()} score: {value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score predictions.jsonl against references.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--references", type=Path, required=True)
    score_parser.add_argument("--predictions", type=Path, required=True)
    score_parser.add_argument("--task", choices=["all", *TASK_FIELDS], default=None)
    score_parser.add_argument("--categorical-only", action="store_true")
    score_parser.add_argument("--output", type=Path, default=None, help="optional scores.json")
    args = parser.parse_args()

    settings = load_settings()
    task = args.task or settings["evaluation"]["task"]
    categorical_only = args.categorical_only or settings["evaluation"]["categorical_only"]

    references = load_references(args.references)
    predictions = index_by_id(read_jsonl(args.predictions), str(args.predictions))
    result = evaluate(references, predictions, task)
    print_report(result)

    if not categorical_only:
        # SPEC.md, section 7: BERTScore for free text is implemented last.
        print(f"skipped (BERTScore not implemented yet): {', '.join(FREE_TEXT_FIELDS)}")

    if args.output is not None:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
