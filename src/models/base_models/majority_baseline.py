#!/usr/bin/env python3
"""
Base Model 1: Majority Class (Constant) Baseline.

Learns the single most frequent value per field from `data/gold/train.jsonl`
and predicts it deterministically for every paper in the Dev set (`data/gold/dev_100_ids.txt`).

All standardized artifacts are saved to `outputs/base_models/majority_baseline/`.
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
    """
    counters = {f: Counter() for f in ALL_EVAL_FIELDS}
    empty_counts = {f: 0 for f in ALL_EVAL_FIELDS}

    for rec in train_records.values():
        for f in ALL_EVAL_FIELDS:
            vals = extract_field_set(rec, f)
            if not vals:
                empty_counts[f] += 1
            for v in vals:
                counters[f][v] += 1

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
    """Constructs a schema-compliant JSON prediction using the learned majority values."""
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

    train_records = load_jsonl_by_id(train_path)
    dev_records = load_jsonl_by_id(dev_path)
    dev_100_ids = sorted(
        {line.strip() for line in dev_ids_path.read_text().splitlines() if line.strip()}
    )

    # 1. Fit majority values on training data
    learned = fit_majority_model(train_records)

    # 2. Generate constant predictions for all 100 Dev papers
    predictions = {zid: predict_single_paper(zid, learned) for zid in dev_100_ids}
    runtime_seconds = time.perf_counter() - start_time

    # 3. Evaluate predictions against gold Dev labels
    metrics = evaluate_predictions(dev_records, predictions, filter_ids=set(dev_100_ids))

    # 4. Save all reproducible artifacts & update central comparison table
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
