#!/usr/bin/env python3
"""
Base Model 2: Random Baseline (Evaluated across 3 fixed seeds: 42, 43, 44).

Uniformly samples one allowed enum value per field from the official JSON schemas
(`schemas/core_fields.json` and `schemas/biodiversity.json`) and one empirical
scientific/German species name from `data/gold/train.jsonl`.

Following the guidelines in `HowTo_Approach_an_NLP_Task.pdf`, this stochastic baseline
is evaluated across three fixed random seeds ([42, 43, 44]) and logs both individual
seed runs and aggregated Mean +/- Std statistics to `outputs/base_models/random_baseline/`.
"""

import json
from pathlib import Path
import random
import statistics
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

SEEDS: list[int] = [42, 43, 44]


def load_allowed_enums(schemas_dir: Path) -> dict[str, list[str]]:
    """Loads the official allowed German enum lists from the C1 and C2b JSON schemas."""
    core = json.loads((schemas_dir / "core_fields.json").read_text(encoding="utf-8"))
    bio = json.loads((schemas_dir / "biodiversity.json").read_text(encoding="utf-8"))
    return {
        "habitat": core["$defs"]["HabitatEnum"]["enum"],
        "biodiversity_level": core["$defs"]["BiodiversityLevelEnum"]["enum"],
        "ecosystem_type.category": core["$defs"]["EcosystemTypeCategoryEnum"]["enum"],
        "ecosystem_type.term": core["$defs"]["EcosystemTypeTermEnum"]["enum"],
        "taxa.species_group": core["$defs"]["SpeciesGroupEnum"]["enum"],
        "ecosystem_service.category": bio["$defs"]["EcosystemServiceCategoryEnum"]["enum"],
        "ecosystem_service.term": bio["$defs"]["EcosystemServiceTermEnum"]["enum"],
    }


def predict_single_paper_random(
    zot_id: str,
    enums: dict[str, list[str]],
    train_sci_names: list[str],
    train_ger_names: list[str],
    rng: random.Random,
) -> dict[str, Any]:
    """Constructs a random schema-compliant JSON prediction using the provided seeded RNG."""
    return {
        "zotitem_ptr_id": zot_id,
        "habitat": [rng.choice(enums["habitat"])],
        "biodiversity_level": [rng.choice(enums["biodiversity_level"])],
        "ecosystem_type": [
            {
                "category": rng.choice(enums["ecosystem_type.category"]),
                "term": rng.choice(enums["ecosystem_type.term"]),
            }
        ],
        "taxa": [
            {
                "species_group": rng.choice(enums["taxa.species_group"]),
                "scientific_name": rng.choice(train_sci_names),
                "german_name": rng.choice(train_ger_names),
            }
        ],
        "ecosystem_service": [
            {
                "category": rng.choice(enums["ecosystem_service.category"]),
                "term": rng.choice(enums["ecosystem_service.term"]),
            }
        ],
    }


def main() -> None:
    start_time = time.perf_counter()

    train_path = REPO_ROOT / "data" / "gold" / "train.jsonl"
    dev_path = REPO_ROOT / "data" / "gold" / "dev.jsonl"
    dev_ids_path = REPO_ROOT / "data" / "gold" / "dev_100_ids.txt"
    schemas_dir = REPO_ROOT / "schemas"

    train_records = load_jsonl_by_id(train_path)
    dev_records = load_jsonl_by_id(dev_path)
    dev_100_ids = sorted(
        {line.strip() for line in dev_ids_path.read_text().splitlines() if line.strip()}
    )

    enums = load_allowed_enums(schemas_dir)
    train_sci_names = sorted(
        {v for r in train_records.values() for v in extract_field_set(r, "taxa.scientific_name")}
    )
    train_ger_names = sorted(
        {v for r in train_records.values() for v in extract_field_set(r, "taxa.german_name")}
    )

    seed_metrics: dict[int, dict[str, Any]] = {}
    primary_predictions: dict[str, dict[str, Any]] = {}

    for seed in SEEDS:
        rng = random.Random(seed)
        preds = {
            zid: predict_single_paper_random(zid, enums, train_sci_names, train_ger_names, rng)
            for zid in dev_100_ids
        }
        seed_metrics[seed] = evaluate_predictions(dev_records, preds, filter_ids=set(dev_100_ids))
        if seed == SEEDS[0]:
            primary_predictions = preds

    runtime_seconds = time.perf_counter() - start_time

    # Compute aggregated Mean and Standard Deviation across the 3 seeds
    c1_vals = [seed_metrics[s]["c1_macro_f1"] for s in SEEDS]
    c2b_vals = [seed_metrics[s]["c2b_macro_f1"] for s in SEEDS]
    micro_vals = [seed_metrics[s]["micro_f1"] for s in SEEDS]

    multi_seed_summary = {
        "seeds": SEEDS,
        "c1_macro_f1_mean": round(statistics.mean(c1_vals), 4),
        "c1_macro_f1_std": round(statistics.stdev(c1_vals), 4),
        "c2b_macro_f1_mean": round(statistics.mean(c2b_vals), 4),
        "c2b_macro_f1_std": round(statistics.stdev(c2b_vals), 4),
        "micro_f1_mean": round(statistics.mean(micro_vals), 4),
        "micro_f1_std": round(statistics.stdev(micro_vals), 4),
        "per_field_f1_mean_std": {
            f: {
                "mean": round(statistics.mean(seed_metrics[s]["fields"][f]["f1"] for s in SEEDS), 4),
                "std": round(statistics.stdev(seed_metrics[s]["fields"][f]["f1"] for s in SEEDS), 4),
            }
            for f in ALL_EVAL_FIELDS
        },
        "per_seed_results": {str(s): seed_metrics[s] for s in SEEDS},
    }

    # Use the aggregated mean scores as the top-level metrics while retaining Seed 42 details
    aggregated_metrics = dict(seed_metrics[SEEDS[0]])
    aggregated_metrics["c1_macro_f1"] = multi_seed_summary["c1_macro_f1_mean"]
    aggregated_metrics["c1_macro_f1_std"] = multi_seed_summary["c1_macro_f1_std"]
    aggregated_metrics["c2b_macro_f1"] = multi_seed_summary["c2b_macro_f1_mean"]
    aggregated_metrics["c2b_macro_f1_std"] = multi_seed_summary["c2b_macro_f1_std"]
    aggregated_metrics["micro_f1"] = multi_seed_summary["micro_f1_mean"]
    aggregated_metrics["micro_f1_std"] = multi_seed_summary["micro_f1_std"]

    model_id = "random_baseline"
    model_title = f"Base Model 2: Random Baseline (Seeds {SEEDS}, Mean across 3 runs)"
    out_dir = save_reproducible_run(
        model_id=model_id,
        model_title=model_title,
        category="base_models",
        predictions=primary_predictions,
        metrics=aggregated_metrics,
        runtime_seconds=runtime_seconds,
        seed=SEEDS,
        config={
            "strategy": "uniform_random_enum_choice",
            "seeds": SEEDS,
            "primary_predictions_seed": SEEDS[0],
            "num_dev_papers": len(dev_100_ids),
        },
        extra_artifacts={f"{model_id}_multi_seed_stats.json": multi_seed_summary},
    )

    print_report(model_title, aggregated_metrics)
    print(
        f"3-Seed Summary ({SEEDS}):\n"
        f"  C1 Leaderboard Macro-F1 : {multi_seed_summary['c1_macro_f1_mean']:.4f} ± {multi_seed_summary['c1_macro_f1_std']:.4f}\n"
        f"  C1+C2b Macro-F1         : {multi_seed_summary['c2b_macro_f1_mean']:.4f} ± {multi_seed_summary['c2b_macro_f1_std']:.4f}\n"
        f"  C1+C2b Micro-F1         : {multi_seed_summary['micro_f1_mean']:.4f} ± {multi_seed_summary['micro_f1_std']:.4f}\n"
        f"Reproducible run artifacts saved to: {out_dir}"
    )


if __name__ == "__main__":
    main()
