#!/usr/bin/env python3
"""Standalone Task C evaluation and CodaBench scoring entry point (Python 3.10+)."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from importlib.metadata import version
from itertools import islice, product
import json
import math
from pathlib import Path
import sys
from typing import Any

GROUPS = {
    "c1": (
        "habitat",
        "biodiversity_level",
        "ecosystem_type.category",
        "ecosystem_type.term",
        "taxa.species_group",
    ),
    "c2a": ("natural_region", "climate", "landuse", "soil.name", "soil.depth"),
    "c2b": (
        "taxa.scientific_name",
        "taxa.german_name",
        "taxa.species_group",
        "biodiversity_variable",
        "biodiversity_level",
        "ecosystem_service.category",
        "ecosystem_service.term",
    ),
    "c2c": (
        "direct_driver.category",
        "direct_driver.description",
        "indirect_driver.category",
        "indirect_driver.description",
    ),
}
TEXT_FIELDS = frozenset(
    ("biodiversity_variable", "direct_driver.description", "indirect_driver.description")
)
TEXT_ALIASES = {
    "direct_driver.description": "details",
    "indirect_driver.description": "details",
}
MODEL_ID = "google-bert/bert-base-multilingual-cased"
MODEL_REVISION = "3f076fdb1ab68d5b2880cb87a0886f315b8146f8"
BATCH_SIZE = 32
BERT_SETTINGS = {
    "model": MODEL_ID,
    "revision": MODEL_REVISION,
    "num_layers": 9,
    "idf": False,
    "rescale_with_baseline": False,
    "use_fast_tokenizer": False,
    "device": "cpu",
    "threads": 1,
    "seed": 0,
    "deterministic_algorithms": True,
    "batch_size": BATCH_SIZE,
    "aggregation": "sum_bidirectional_best_pair_f1 / (prediction_values + reference_values)",
}
PairScorer = Callable[[list[str], list[str]], Sequence[float]]


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """Load nonempty JSONL, rejecting malformed records, ambiguous keys and IDs."""
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(
                    line, object_pairs_hook=_unique_keys, parse_constant=_invalid_constant
                )
                if not isinstance(record, dict):
                    raise ValueError("Each record must be a JSON object")
                record_id = record.get("zotitem_ptr_id")
                if not isinstance(record_id, str) or not record_id.strip():
                    raise ValueError("zotitem_ptr_id must be a nonblank string")
                if record_id in records:
                    raise ValueError(f"Duplicate zotitem_ptr_id: {record_id!r}")
                records[record_id] = record
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def field_values(record: dict[str, Any], field: str) -> set[str]:
    """Collect a dotted field through lists; omit nulls/blanks and deduplicate."""
    result: set[str] = set()

    def visit(value: Any, parts: list[str], target: set[str]) -> None:
        if value is None or (isinstance(value, dict) and not value):
            return
        if isinstance(value, list):
            for item in value:
                visit(item, parts, target)
        elif parts:
            if not isinstance(value, dict):
                raise ValueError(f"{field}: expected an object or list of objects")
            alias = TEXT_ALIASES.get(field)
            if len(parts) == 1 and alias is not None and alias in value:
                aliased: set[str] = set()
                visit(value[alias], [], aliased)
                if parts[0] in value:
                    canonical: set[str] = set()
                    visit(value[parts[0]], [], canonical)
                    if canonical != aliased:
                        raise ValueError(f"{field}: conflicting description/details values")
                target.update(aliased)
            else:
                visit(value.get(parts[0]), parts[1:], target)
        elif isinstance(value, str):
            if value.strip():
                target.add(value)
        else:
            raise ValueError(f"{field}: values must be strings, lists of strings, or null")

    visit(record, field.split("."), result)
    return result


def categorical_scores(tp: int, fp: int, fn: int) -> dict[str, Any]:
    """Compute pooled precision/recall/F1 with zero for undefined divisions."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1, support=tp + fn)


def download_model() -> str:
    """Cache just the pinned model weights, configuration and tokenizer files."""
    from huggingface_hub import snapshot_download

    return snapshot_download(  # nosec B615: MODEL_REVISION is an immutable commit above.
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
        ],
    )


class BertPairs:
    """Lazily load the fixed BERTScore runtime; categorical scoring needs no extras."""

    def __init__(self) -> None:
        self.scorer: Any = None

    def __call__(self, predictions: list[str], references: list[str]) -> list[float]:
        if self.scorer is None:
            try:
                from bert_score import BERTScorer
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    "Free-text scoring requires requirements-bertscore.txt; "
                    "install it or use --categorical-only."
                ) from exc
            torch.manual_seed(0)
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            self.scorer = BERTScorer(
                model_type=download_model(),
                num_layers=9,
                batch_size=BATCH_SIZE,
                nthreads=1,
                all_layers=False,
                idf=False,
                device="cpu",
                rescale_with_baseline=False,
                use_fast_tokenizer=False,
            )
        _, _, f1 = self.scorer.score(predictions, references, batch_size=BATCH_SIZE)
        return f1.tolist()


def best_match_sum(predictions: set[str], references: set[str], pair_scorer: PairScorer) -> float:
    """Sum row and column maxima of the pairwise BERTScore F1 matrix."""
    if not predictions or not references:
        return 0.0
    # Sort for stable batching. Compare only within the same paper and field.
    candidates, gold = sorted(predictions), sorted(references)
    row_max = [-math.inf] * len(candidates)
    col_max = [-math.inf] * len(gold)
    pairs = product(range(len(candidates)), range(len(gold)))
    while batch := list(islice(pairs, BATCH_SIZE)):
        scores = pair_scorer([candidates[i] for i, _ in batch], [gold[j] for _, j in batch])
        if len(scores) != len(batch) or any(not math.isfinite(s) for s in scores):
            raise ValueError("BERTScore returned invalid pair scores")
        for (i, j), score in zip(batch, scores):
            row_max[i] = max(row_max[i], score)
            col_max[j] = max(col_max[j], score)
    return math.fsum(row_max) + math.fsum(col_max)


def evaluate(
    references: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    task: str = "all",
    categorical_only: bool = False,
    pair_scorer: PairScorer | None = None,
) -> dict[str, Any]:
    """Score C1 plus selected C2 groups; enforce an exact match of paper IDs."""
    if task not in (*GROUPS, "all"):
        raise ValueError(f"Unknown task: {task}")
    if not references:
        raise ValueError("No reference records")
    missing, unknown = (
        references.keys() - predictions.keys(),
        predictions.keys() - references.keys(),
    )
    if missing or unknown:
        raise ValueError(f"Paper IDs differ: missing={sorted(missing)}, unknown={sorted(unknown)}")
    selected = list(GROUPS) if task == "all" else list(dict.fromkeys(("c1", task)))
    fields = tuple(dict.fromkeys(field for group in selected for field in GROUPS[group]))
    active = [field for field in fields if not categorical_only or field not in TEXT_FIELDS]
    # Validate all scored values before starting any model downloads/inference.
    prepared = []
    for record_id in sorted(references):
        for field in active:
            try:
                predicted = field_values(predictions[record_id], field)
                gold = field_values(references[record_id], field)
            except ValueError as exc:
                raise ValueError(f"Paper {record_id!r}: {exc}") from exc
            prepared.append((field, predicted, gold))
    counts = {field: [0, 0, 0] for field in active}
    scorer = pair_scorer if pair_scorer is not None else BertPairs()
    for field, predicted, gold in prepared:
        if field in TEXT_FIELDS:
            counts[field][0] += best_match_sum(predicted, gold, scorer)
            counts[field][1] += len(predicted)
            counts[field][2] += len(gold)
        else:
            counts[field][0] += len(predicted & gold)
            counts[field][1] += len(predicted - gold)
            counts[field][2] += len(gold - predicted)
    per_field = {}
    for field, (first, second, third) in counts.items():
        if field in TEXT_FIELDS:
            per_field[field] = {
                "bertscore": first / (second + third) if second + third else 0.0,
                "best_match_sum": first,
                "prediction_values": second,
                "reference_values": third,
            }
        else:
            per_field[field] = categorical_scores(first, second, third)

    def summarize(names: Sequence[str]) -> dict[str, Any]:
        categories = [per_field[f] for f in names if f not in TEXT_FIELDS]
        texts = [per_field[f] for f in names if f in TEXT_FIELDS and f in per_field]
        summary = {
            "macro_f1": sum(s["f1"] for s in categories) / len(categories),
            "micro": categorical_scores(
                *(sum(s[k] for s in categories) for k in ("tp", "fp", "fn"))
            ),
            "categorical_fields": len(categories),
        }
        if texts:
            total = sum(s["prediction_values"] + s["reference_values"] for s in texts)
            summary["bertscore"] = (
                sum(s["best_match_sum"] for s in texts) / total if total else 0.0
            )
        return summary

    summaries = {group: summarize(GROUPS[group]) for group in selected}
    c2_fields = tuple(dict.fromkeys(f for g in selected if g != "c1" for f in GROUPS[g]))
    if c2_fields:
        summaries["c2"] = summarize(c2_fields)
    summaries["all"] = summarize(fields)
    scores = {"c1_score": summaries["c1"]["macro_f1"]}
    for group, summary in summaries.items():
        scores[f"{group}_macro_f1"] = summary["macro_f1"]
        scores[f"{group}_micro_f1"] = summary["micro"]["f1"]
        if "bertscore" in summary:
            scores[f"{group}_bertscore"] = summary["bertscore"]
    for field, values in per_field.items():
        for key in ("precision", "recall", "f1", "bertscore"):
            if key in values:
                scores[f"{field.replace('.', '_')}_{key}"] = values[key]
    settings = dict(BERT_SETTINGS)
    settings["enabled"] = not categorical_only and any(f in TEXT_FIELDS for f in fields)
    if isinstance(scorer, BertPairs) and scorer.scorer is not None:
        settings["packages"] = {
            name: version(name)
            for name in ("bert-score", "transformers", "torch", "huggingface-hub", "numpy")
        }
    return {
        "format_version": 1,
        "task": task,
        "papers": len(references),
        "bertscore_settings": settings,
        "fields": per_field,
        "summaries": summaries,
        "scores": scores,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    local = commands.add_parser("score", help="Evaluate reference and prediction JSONL files")
    local.add_argument("--references", type=Path, required=True)
    local.add_argument("--predictions", type=Path, required=True)
    local.add_argument("--output", type=Path, help="Detailed JSON report (default: stdout)")
    bench = commands.add_parser("codabench", help="Read INPUT/ref and INPUT/res; write OUTPUT")
    bench.add_argument("input_dir", type=Path)
    bench.add_argument("output_dir", type=Path)
    commands.add_parser(
        "prepare-model", help="Download the pinned model into the Hugging Face cache"
    )
    for command in (local, bench):
        command.add_argument(
            "--task", choices=(*GROUPS, "all"), default="all", help="C1 plus selected C2 group(s)"
        )
        command.add_argument(
            "--categorical-only",
            action="store_true",
            help="Omit free-text metrics; no extra dependencies",
        )
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-model":
            print(download_model())
            return 0
        if args.command == "codabench":
            refs = args.input_dir / "ref" / "references.jsonl"
            submission = args.input_dir / "res"
            if sorted(p.name for p in submission.iterdir()) != ["predictions.jsonl"]:
                raise ValueError("Submission must contain exactly one file: predictions.jsonl")
            preds = submission / "predictions.jsonl"
        else:
            refs, preds = args.references, args.predictions
        result = evaluate(load_jsonl(refs), load_jsonl(preds), args.task, args.categorical_only)
        if args.command == "codabench":
            write_json(args.output_dir / "detailed_scores.json", result)
            write_json(args.output_dir / "scores.json", result["scores"])
            print(json.dumps(result["scores"], ensure_ascii=False, indent=2, allow_nan=False))
        elif args.output:
            write_json(args.output, result)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
