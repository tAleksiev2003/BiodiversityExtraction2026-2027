"""How good were the decisions of a verify run? Numbers only, no requests (src/run_verify.py).

Usage (run by the team, it reads course data):
    python3 src/scripts/analyze_verification.py --references data/processed/dev_with_pdf.jsonl --run runs/<verify run>

Prints:
1. per candidate field: candidates, kept, dropped (from verify_log.jsonl);
2. per scored subfield (e.g. taxa.species_group): of the kept and of the dropped
   candidates, how many values were right / wrong according to the references.
   "right" = the value is in the reference set of that document and subfield; a
   candidate without a value in that subfield (e.g. term null) is not counted.
   Counted per candidate: dropping an entry whose value another kept entry also has
   does not change the prediction, so the scores in 3. are the final word;
3. F1 per scored field and the C1 / C2b scores of the source run and of the verify run.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import index_by_id, load_references, read_jsonl  # noqa: E402
from evaluation import C1_FIELDS, C2B_FIELDS, evaluate, extract_values  # noqa: E402
from output import PREDICTIONS_FILENAME  # noqa: E402
from pipelines.verify import SOURCES_NAME, VERIFY_LOG_NAME  # noqa: E402
from verification import DROP, KEEP  # noqa: E402

SCORED_FIELDS = list(dict.fromkeys(C1_FIELDS + C2B_FIELDS))
RIGHT = "right"
WRONG = "wrong"


def scored_values(entry: Any, field: str) -> dict[str, str]:
    """{scored field: value} of one candidate, e.g. {"taxa.species_group": "...", ...}."""
    values = {}
    for scored_field in SCORED_FIELDS:
        parent, _, subfield = scored_field.partition(".")
        if parent != field:
            continue
        value = entry.get(subfield) if subfield else entry
        if isinstance(value, str) and value:
            values[scored_field] = value
    return values


def count_decisions(log: list[dict[str, Any]], references: dict[str, dict[str, Any]]) -> tuple[
        dict[str, Counter], dict[str, Counter]]:
    """Per candidate field: decision counts; per scored field: (decision, right/wrong) counts."""
    per_field: dict[str, Counter] = defaultdict(Counter)
    per_scored_field: dict[str, Counter] = defaultdict(Counter)
    for entry in log:
        decision = entry["decision"]
        per_field[entry["field"]]["candidates"] += 1
        per_field[entry["field"]][decision] += 1
        reference = references.get(entry["document_id"])
        for scored_field, value in scored_values(entry["value"], entry["field"]).items():
            correct = RIGHT if value in extract_values(reference, scored_field) else WRONG
            per_scored_field[scored_field][(decision, correct)] += 1
    return per_field, per_scored_field


def print_decisions(per_field: dict[str, Counter], per_scored_field: dict[str, Counter]) -> None:
    print(f"{'field':22s} {'candidates':>10s} {'kept':>6s} {'dropped':>8s}")
    for field, counts in per_field.items():
        print(f"{field:22s} {counts['candidates']:10d} {counts[KEEP]:6d} {counts[DROP]:8d}")
    print(f"\n{'scored field':30s} {'kept right':>10s} {'kept wrong':>10s} {'dropped right':>13s} {'dropped wrong':>13s}")
    for scored_field in SCORED_FIELDS:
        if scored_field not in per_scored_field:
            continue
        counts = per_scored_field[scored_field]
        print(f"{scored_field:30s} {counts[(KEEP, RIGHT)]:10d} {counts[(KEEP, WRONG)]:10d} "
              f"{counts[(DROP, RIGHT)]:13d} {counts[(DROP, WRONG)]:13d}")


def print_scores(source_result: dict[str, Any], verify_result: dict[str, Any]) -> None:
    print(f"\n{'field':30s} {'F1 source':>9s} {'F1 verify':>9s} {'change':>8s}")
    for field, source_values in source_result["fields"].items():
        before, after = source_values["f1"], verify_result["fields"][field]["f1"]
        print(f"{field:30s} {before:9.4f} {after:9.4f} {after - before:+8.4f}")
    for task, before in source_result["task_scores"].items():
        after = verify_result["task_scores"][task]
        print(f"{task.upper()} score: source {before:.4f}, verify {after:.4f} ({after - before:+.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse the decisions of a verify run.")
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="run folder of src/run_verify.py")
    args = parser.parse_args()

    references = load_references(args.references)
    sources = json.loads((args.run / SOURCES_NAME).read_text(encoding="utf-8"))
    log = read_jsonl(args.run / VERIFY_LOG_NAME)
    print(f"verify run: {args.run}\nsource run: {sources['source_run']}\n"
          f"decisions applied for: {', '.join(sources['applied_fields'])}\n")
    print_decisions(*count_decisions(log, references))

    verify_predictions_path = args.run / PREDICTIONS_FILENAME
    if not verify_predictions_path.exists():
        sys.exit("\nNo predictions.jsonl in the verify run (dry run or stopped): no scores.")
    source_predictions_path = Path(sources["source_run"]) / PREDICTIONS_FILENAME
    source_predictions = index_by_id(read_jsonl(source_predictions_path), str(source_predictions_path))
    verify_predictions = index_by_id(read_jsonl(verify_predictions_path), str(verify_predictions_path))
    print_scores(evaluate(references, source_predictions, "all"), evaluate(references, verify_predictions, "all"))


if __name__ == "__main__":
    main()
