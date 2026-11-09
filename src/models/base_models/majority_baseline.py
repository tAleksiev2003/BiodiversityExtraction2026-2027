#!/usr/bin/env python3
"""
========================================================================================
EXECUTIVE SUMMARY — Base Model 1: Majority Class (Constant) Baseline
========================================================================================
- Concept:
    Deterministic, zero-feature trivial baseline required by the Shared Task C rules
    (ANLP2026_shared_tasks_task_descriptions_v1.pdf, p. 10) and NLP workflow guidelines
    (HowTo_Approach_an_NLP_Task.pdf, p. 2).
- Mechanism:
    1. Scans all 908 gold training records (`data/gold/train.jsonl`) and counts empirical
       value occurrences for each C1 (Core) and C2b (Biodiversity) schema field.
    2. Selects the single most frequent value (`top-1 mode`) per field:
       - `habitat`: "Agrar- und Offenland" (301/908)
       - `biodiversity_level`: "Artenvielfalt" (349/908)
       - `ecosystem_type.category`: "II Biotoptypengruppen der Binnengewässer" (167/908)
       - `ecosystem_type.term`: "Fließende Gewässer" (92/908)
       - `taxa.species_group`: "Gefäßpflanzen" (193/908)
       - `taxa.scientific_name`: "-" (98/908, collective term marker)
       - `taxa.german_name`: "Fische" (61/908)
       - `ecosystem_service.category`: "Regulierungs- und Erhaltungsleistungen" (137/908)
       - `ecosystem_service.term`: "2.2.6.1 Regulierung der chemischen..." (38/908)
    3. Emits this exact constant record for every paper in the 100-paper Dev set without
       reading any paper text or PDF content.
- Key Results (100 Dev Papers):
    - Official C1 Leaderboard Macro-F1 : 0.2055
    - C1 + C2b Overall Macro-F1        : 0.1429
    - C1 + C2b Micro-F1                : 0.1403
- Key Takeaway / Limitation:
    Because the model predicts a value on 100% of papers, it suffers massive False Positives
    on sparse fields like `ecosystem_service` (which is empty in 78.1% of training papers),
    scoring 93 FP vs. only 7 TP on `ecosystem_service.category`.
========================================================================================
"""

from collections import Counter
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluator import (
    ALL_EVAL_FIELDS,
    evaluate_predictions,
    extract_field_set,
    load_jsonl_by_id,
    print_report,
    save_reproducible_run,
)


def fit_majority_model(train_records: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Counts empirical value frequencies per field across the training set and
    selects the top-1 most frequent value (along with its count and empty rate).

    Args:
        train_records: Dictionary of training records keyed by `zotitem_ptr_id`.

    Returns:
        Dictionary mapping each evaluated field name to its `majority_value`,
        `train_count`, and `empty_rate`.
    """
    counters = {f: Counter() for f in ALL_EVAL_FIELDS}
    empty_counts = {f: 0 for f in ALL_EVAL_FIELDS}

    # Iterate through all 908 training papers to accumulate label frequencies
    for rec in train_records.values():
        for f in ALL_EVAL_FIELDS:
            vals = extract_field_set(rec, f)
            if not vals:
                empty_counts[f] += 1
            for v in vals:
                counters[f][v] += 1

    # Extract the single most frequent label per field
    learned = {}
    for f in ALL_EVAL_FIELDS:
        top_val, top_count = counters[f].most_common(1)[0]
        learned[f] = {
            "majority_value": top_val,
            "train_count": top_count,
            "empty_rate": round(empty_counts[f] / len(train_records), 4),
        }
    return learned


def predict_single_paper(zot_id: str, learned: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Constructs a schema-compliant JSON prediction for a single paper using the
    constant majority values learned from the training set.
    """
    return {
        "zotitem_ptr_id": zot_id,
        "habitat": [learned["habitat"]["majority_value"]],
        "biodiversity_level": [learned["biodiversity_level"]["majority_value"]],
        "ecosystem_type": [
            {
                "category": learned["ecosystem_type.category"]["majority_value"],
                "term": learned["ecosystem_type.term"]["majority_value"],
            }
        ],
        "taxa": [
            {
                "species_group": learned["taxa.species_group"]["majority_value"],
                "scientific_name": learned["taxa.scientific_name"]["majority_value"],
                "german_name": learned["taxa.german_name"]["majority_value"],
            }
        ],
        "ecosystem_service": [
            {
                "category": learned["ecosystem_service.category"]["majority_value"],
                "term": learned["ecosystem_service.term"]["majority_value"],
            }
        ],
    }


def main() -> None:
    start_time = time.perf_counter()

    train_path = REPO_ROOT / "data" / "gold" / "train.jsonl"
    dev_path = REPO_ROOT / "data" / "gold" / "dev.jsonl"
    dev_ids_path = REPO_ROOT / "data" / "gold" / "dev_100_ids.txt"

    # Load gold annotations and filter Dev evaluation to the 100 valid Dev PDF IDs
    train_records = load_jsonl_by_id(train_path)
    dev_records = load_jsonl_by_id(dev_path)
    dev_100_ids = sorted(
        {line.strip() for line in dev_ids_path.read_text().splitlines() if line.strip()}
    )

    # Step 1: Learn top-1 majority values from the 908 training records
    learned = fit_majority_model(train_records)

    # Step 2: Generate constant predictions for all 100 Dev papers
    predictions = {zid: predict_single_paper(zid, learned) for zid in dev_100_ids}
    runtime_seconds = time.perf_counter() - start_time

    # Step 3: Evaluate predictions against gold Dev annotations
    metrics = evaluate_predictions(dev_records, predictions, filter_ids=set(dev_100_ids))

    # Step 4: Save all reproducible run artifacts & update central comparison table
    model_id = "majority_baseline"
    model_title = "Base Model 1: Majority Class (Constant) Baseline"
    out_dir = save_reproducible_run(
        model_id=model_id,
        model_title=model_title,
        category="base_models",
        predictions=predictions,
        metrics=metrics,
        runtime_seconds=runtime_seconds,
        seed=None,
        config={
            "strategy": "top_1_frequent_value_per_field",
            "num_train_papers": len(train_records),
            "num_dev_papers": len(dev_100_ids),
        },
        extra_artifacts={f"{model_id}_learned_values.json": learned},
    )

    print_report(model_title, metrics)
    print(f"Reproducible run artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
