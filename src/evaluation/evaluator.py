#!/usr/bin/env python3
"""
Official-style Local Evaluator & Reproducible Run Logger for ANLP 2026 Shared Task C.

Implements the exact evaluation rules from ANLP2026_shared_tasks_task_descriptions_v1.pdf (pp. 9-10)
and the reproducibility logging guidelines from HowTo_Approach_an_NLP_Task.pdf (pp. 2-3):
- Set-based exact match per paper and field (summing TP, FP, FN across all evaluated papers).
- Computes Precision, Recall, F1 per field, C1 Leaderboard Macro-F1, and C1+C2b Macro/Micro-F1.
- Logs every run with git commit hash, seed, configuration, hardware info, runtime,
  formatted report, raw JSONL predictions, CodaBench ZIP archive, and updates a central
  Markdown comparison table (`outputs/all_models_comparison.md`).
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
from typing import Any
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[2]

# Task C1 (Core) fields used for official leaderboard ranking
C1_FIELDS: list[str] = [
    "habitat",
    "biodiversity_level",
    "ecosystem_type.category",
    "ecosystem_type.term",
    "taxa.species_group",
]

# Task C2b (Biodiversity) additional categorical/string fields
C2B_EXTRA_FIELDS: list[str] = [
    "taxa.scientific_name",
    "taxa.german_name",
    "ecosystem_service.category",
    "ecosystem_service.term",
]

ALL_EVAL_FIELDS: list[str] = C1_FIELDS + C2B_EXTRA_FIELDS


def extract_field_set(record: dict[str, Any], field_path: str) -> set[str]:
    """
    Extracts the set of unique non-empty string values for a given field or nested subfield.

    Args:
        record: A single paper's JSON dictionary (gold or predicted).
        field_path: Either a top-level key (e.g. 'habitat') or a dotted path
                    for nested list-of-dicts (e.g. 'taxa.species_group').

    Returns:
        A set of stripped, non-empty string values present in the record.
    """
    if "." not in field_path:
        val = record.get(field_path)
        if not val:
            return set()
        if isinstance(val, list):
            return {str(v).strip() for v in val if v is not None and str(v).strip()}
        return {str(val).strip()} if str(val).strip() else set()

    parent, child = field_path.split(".", 1)
    items = record.get(parent)
    if not items or not isinstance(items, list):
        return set()
    return {
        str(item.get(child)).strip()
        for item in items
        if isinstance(item, dict)
        and item.get(child) is not None
        and str(item.get(child)).strip()
    }


def evaluate_predictions(
    gold_records: dict[str, dict[str, Any]],
    pred_records: dict[str, dict[str, Any]],
    filter_ids: set[str] | None = None,
) -> dict[str, Any]:
    """
    Computes corpus-level TP, FP, FN, Precision, Recall, and F1 per field,
    along with C1 Macro-F1, C1+C2b Macro-F1, and Micro-F1.
    """
    eval_ids = sorted(filter_ids if filter_ids is not None else gold_records.keys())
    counts = {f: {"tp": 0, "fp": 0, "fn": 0} for f in ALL_EVAL_FIELDS}

    for zid in eval_ids:
        gold = gold_records.get(zid, {})
        pred = pred_records.get(zid, {})
        for f in ALL_EVAL_FIELDS:
            g_set = extract_field_set(gold, f)
            p_set = extract_field_set(pred, f)
            counts[f]["tp"] += len(g_set & p_set)
            counts[f]["fp"] += len(p_set - g_set)
            counts[f]["fn"] += len(g_set - p_set)

    field_metrics: dict[str, dict[str, float | int]] = {}
    for f, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        field_metrics[f] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    c1_macro_f1 = sum(float(field_metrics[f]["f1"]) for f in C1_FIELDS) / len(C1_FIELDS)
    c2b_macro_f1 = sum(float(field_metrics[f]["f1"]) for f in ALL_EVAL_FIELDS) / len(ALL_EVAL_FIELDS)

    total_tp = sum(counts[f]["tp"] for f in ALL_EVAL_FIELDS)
    total_fp = sum(counts[f]["fp"] for f in ALL_EVAL_FIELDS)
    total_fn = sum(counts[f]["fn"] for f in ALL_EVAL_FIELDS)
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    return {
        "num_papers": len(eval_ids),
        "c1_macro_f1": round(c1_macro_f1, 4),
        "c2b_macro_f1": round(c2b_macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "fields": field_metrics,
    }


def format_report(title: str, results: dict[str, Any]) -> str:
    """Formats the evaluation dictionary into a clean ASCII table."""
    lines = [
        f"{'=' * 74}",
        f" {title} (N = {results['num_papers']} papers)",
        f"{'=' * 74}",
        f"{'Field':<28} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'TP':>5} {'FP':>5} {'FN':>5}",
        f"{'-' * 74}",
    ]
    for f in C1_FIELDS:
        m = results["fields"][f]
        lines.append(
            f"{f:<28} | {m['precision']:7.4f} | {m['recall']:7.4f} | {m['f1']:7.4f} | {m['tp']:5d} {m['fp']:5d} {m['fn']:5d}"
        )
    lines.extend(
        [
            f"{'-' * 74}",
            f"{'>> C1 LEADERBOARD MACRO-F1':<28} | {'':>7} | {'':>7} | {results['c1_macro_f1']:7.4f} |",
            f"{'-' * 74}",
        ]
    )
    for f in C2B_EXTRA_FIELDS:
        m = results["fields"][f]
        lines.append(
            f"{f:<28} | {m['precision']:7.4f} | {m['recall']:7.4f} | {m['f1']:7.4f} | {m['tp']:5d} {m['fp']:5d} {m['fn']:5d}"
        )
    lines.extend(
        [
            f"{'-' * 74}",
            f"{'>> C1+C2b MACRO-F1':<28} | {'':>7} | {'':>7} | {results['c2b_macro_f1']:7.4f} |",
            f"{'>> C1+C2b MICRO-F1':<28} | {'':>7} | {'':>7} | {results['micro_f1']:7.4f} |",
            f"{'=' * 74}",
        ]
    )
    return "\n".join(lines) + "\n"


def print_report(title: str, results: dict[str, Any]) -> None:
    """Prints the formatted evaluation table to stdout."""
    print("\n" + format_report(title, results))


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    """Loads a JSONL file into a dictionary keyed by `zotitem_ptr_id`."""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                data[obj["zotitem_ptr_id"]] = obj
    return data


def save_submission_zip(records: dict[str, dict[str, Any]], out_zip_path: Path) -> None:
    """Packages predictions as `predictions.jsonl` inside a ZIP archive for CodaBench."""
    out_zip_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_content = "\n".join(json.dumps(rec, ensure_ascii=False) for rec in records.values()) + "\n"
    with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("predictions.jsonl", jsonl_content.encode("utf-8"))


def get_git_commit_info() -> dict[str, str]:
    """Retrieves the current Git branch and short commit hash for reproducibility."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        commit = "uncommitted"
    try:
        branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        branch = "unknown"
    return {"branch": branch, "commit": commit}


def save_reproducible_run(
    model_id: str,
    model_title: str,
    category: str,
    predictions: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
    runtime_seconds: float,
    seed: int | list[int] | None = None,
    config: dict[str, Any] | None = None,
    extra_artifacts: dict[str, Any] | None = None,
) -> Path:
    """
    Saves all standardized, reproducible artifacts for a model run and updates the
    central Markdown comparison table (`outputs/all_models_comparison.md`).

    Generated files in `outputs/<category>/<model_id>/`:
      1. `<model_id>_run_info.json`     (metadata, git commit, seed, runtime, hardware, config)
      2. `<model_id>_metrics.json`      (full Precision/Recall/F1 metrics)
      3. `<model_id>_report.txt`        (formatted ASCII evaluation table)
      4. `<model_id>_predictions.jsonl` (raw predictions per paper)
      5. `<model_id>_submission.zip`    (CodaBench-ready zip containing predictions.jsonl)
    """
    out_dir = REPO_ROOT / "outputs" / category / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    git_info = get_git_commit_info()
    run_info = {
        "model_id": model_id,
        "model_title": model_title,
        "category": category,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": git_info["branch"],
        "git_commit": git_info["commit"],
        "seed": seed,
        "runtime_seconds": round(runtime_seconds, 4),
        "hardware": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "config": config or {},
        "summary_scores": {
            "c1_macro_f1": metrics["c1_macro_f1"],
            "c2b_macro_f1": metrics["c2b_macro_f1"],
            "micro_f1": metrics["micro_f1"],
            "num_evaluated_papers": metrics["num_papers"],
        },
    }

    # 1. Save run metadata
    (out_dir / f"{model_id}_run_info.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # 2. Save detailed metrics JSON
    (out_dir / f"{model_id}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # 3. Save human-readable ASCII report
    report_str = format_report(model_title, metrics)
    (out_dir / f"{model_id}_report.txt").write_text(report_str, encoding="utf-8")

    # 4. Save raw JSONL predictions
    (out_dir / f"{model_id}_predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in predictions.values()) + "\n",
        encoding="utf-8",
    )

    # 5. Save CodaBench submission ZIP
    save_submission_zip(predictions, out_dir / f"{model_id}_submission.zip")

    # 6. Save optional model-specific artifacts (e.g., learned weights/counts)
    if extra_artifacts:
        for filename, content in extra_artifacts.items():
            (out_dir / filename).write_text(
                json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )

    # 7. Update central comparison leaderboard (`outputs/all_models_comparison.md`)
    _update_central_comparison_table(run_info)

    return out_dir


def _update_central_comparison_table(run_info: dict[str, Any]) -> None:
    """Rebuilds `outputs/all_models_comparison.md` from all saved `*_run_info.json` files."""
    outputs_root = REPO_ROOT / "outputs"
    info_files = sorted(outputs_root.glob("*/*/*_run_info.json"))

    rows = []
    for path in info_files:
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
            rows.append(info)
        except Exception:
            continue

    lines = [
        "# Model Run Comparison (ANLP Shared Task C1 + C2b)",
        "",
        "| Model ID | Category | Seed | N (Dev) | C1 Macro-F1 (Leaderboard) | C1+C2b Macro-F1 | Micro-F1 | Runtime (s) | Git Commit |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        sc = r.get("summary_scores", {})
        seed_str = str(r.get("seed")) if r.get("seed") is not None else "deterministic"
        lines.append(
            f"| `{r['model_id']}` | `{r['category']}` | `{seed_str}` | {sc.get('num_evaluated_papers', 100)} "
            f"| **{sc.get('c1_macro_f1', 0.0):.4f}** | {sc.get('c2b_macro_f1', 0.0):.4f} "
            f"| {sc.get('micro_f1', 0.0):.4f} | {r.get('runtime_seconds', 0.0):.3f}s | `{r.get('git_commit', '-')}` |"
        )
    lines.append("")
    (outputs_root / "all_models_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task C predictions against gold JSONL.")
    parser.add_argument(
        "--gold",
        type=str,
        default=str(REPO_ROOT / "data" / "gold" / "dev.jsonl"),
        help="Path to gold dev.jsonl",
    )
    parser.add_argument(
        "--pred",
        type=str,
        default=None,
        help="Path to predictions.jsonl (omit to run gold vs. gold sanity check)",
    )
    parser.add_argument(
        "--ids-file",
        type=str,
        default=str(REPO_ROOT / "data" / "gold" / "dev_100_ids.txt"),
        help="File with valid Dev PDF IDs (default: data/gold/dev_100_ids.txt)",
    )
    args = parser.parse_args()

    gold = load_jsonl_by_id(Path(args.gold))
    pred = load_jsonl_by_id(Path(args.pred)) if args.pred else gold
    filter_ids = (
        {l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()}
        if args.ids_file
        else None
    )

    title = f"Evaluation: {Path(args.pred).name}" if args.pred else "Sanity Check: Gold vs. Gold"
    res = evaluate_predictions(gold, pred, filter_ids=filter_ids)
    print_report(title, res)


if __name__ == "__main__":
    main()
