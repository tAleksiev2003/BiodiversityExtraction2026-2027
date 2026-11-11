#!/usr/bin/env python3
"""
========================================================================================
ERROR ANALYSIS — Categorized Failure Cases on the 100 Dev Papers (Phase 5 Preparation)
========================================================================================
- Purpose:
    Systematic, per-field categorization of the errors made by our best models
    (default: the Approval-Voting Ensemble and TF-IDF + Linear SVM). The output is a
    Markdown report (`outputs/error_analysis/error_analysis_report.md`) that
    (a) provides presentation-ready failure statistics,
    (b) ranks error categories by frequency, and
    (c) derives concrete actions for the LLM architectures (prompt conventions and
        post-processing rules).
- What is analyzed per field:
    1. Confusion pairs: which gold value was replaced by which predicted value
       (over-prediction FPs vs. missed gold values FNs are shown separately).
    2. Empty-field errors: gold empty vs. predicted non-empty (over-firing) and
       gold non-empty vs. predicted empty (missed paper) — the dominant skill on this
       dataset given the 39-78% empty rates.
    3. Field-specific deep dives:
       - ecosystem_type: term errors split by "category correct vs. wrong".
       - taxa: gold species collapsed to '-' collective entries vs. predicted names
         that merely miss exact-match (fuzzy ratio >= 0.85 => spelling/inflection
         variant rather than wrong taxon).
- Usage:
    python3 src/evaluation/error_analysis.py [--models ensemble_voting_baseline
    tfidf_svm_baseline ...] [--examples 15]
"""

import argparse
from collections import Counter
import difflib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluator import (
    ALL_EVAL_FIELDS,
    extract_field_set,
    load_jsonl_by_id,
)

# Fields with free-text values (fuzzy variant analysis instead of enum confusions)
FREE_TEXT_FIELDS = {"taxa.scientific_name", "taxa.german_name"}


def load_predictions(model_id: str, category: str = "base_models") -> dict[str, dict[str, Any]]:
    """Loads a model's saved Dev predictions keyed by `zotitem_ptr_id`."""
    path = REPO_ROOT / "outputs" / category / model_id / f"{model_id}_predictions.jsonl"
    preds = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            preds[obj["zotitem_ptr_id"]] = obj
    return preds


def analyze_model(
    gold: dict[str, dict[str, Any]],
    pred: dict[str, dict[str, Any]],
    dev_ids: list[str],
    n_examples: int,
) -> dict[str, Any]:
    """Computes per-field error statistics and collects concrete example cases."""
    stats: dict[str, Any] = {}

    for field in ALL_EVAL_FIELDS:
        confusions: Counter = Counter()  # (gold_val, pred_val) co-occurring error pairs
        over_fire = 0  # gold empty, predicted something
        missed_all = 0  # gold non-empty, predicted empty
        partial = 0  # overlap but not identical
        perfect = 0
        examples: list[dict[str, Any]] = []

        # ecosystem_type deep dive counters
        term_err_cat_ok = 0
        term_err_cat_wrong = 0
        # taxa free-text deep dive counters
        fuzzy_variant = 0  # predicted name ~>= 0.85 similar to a missed gold name
        collapsed_to_dash = 0  # gold lists specific species, we predicted '-'
        gold_specific_pred_dash_papers = 0

        for zid in dev_ids:
            g = extract_field_set(gold[zid], field)
            p = extract_field_set(pred[zid], field)
            title = str(gold[zid].get("bibtex_title") or "")[:90]

            if not g and not p:
                continue  # true negative: both empty (the easy 40-80% majority)
            if g == p:
                perfect += 1
                continue

            if not g and p:
                over_fire += 1
                confusions.update({(None, v): 1 for v in p})
                if len(examples) < 50:
                    examples.append(
                        {"id": zid, "title": title, "type": "OVER-FIRE (gold leer)",
                         "gold": sorted(g), "pred": sorted(p)}
                    )
            elif g and not p:
                missed_all += 1
                confusions.update({(v, None): 1 for v in g})
                if len(examples) < 50:
                    examples.append(
                        {"id": zid, "title": title, "type": "MISSED (pred leer)",
                         "gold": sorted(g), "pred": sorted(p)}
                    )
            else:
                partial += 1
                fp, fn = p - g, g - p
                confusions.update({(gv, pv): 1 for gv in fn for pv in fp})
                if len(examples) < 50:
                    examples.append(
                        {"id": zid, "title": title, "type": "PARTIAL",
                         "gold": sorted(g), "pred": sorted(p), "fp": sorted(fp), "fn": sorted(fn)}
                    )

            # ---- field-specific deep dives ----
            if field == "ecosystem_type.term" and (p - g):
                g_cats = extract_field_set(gold[zid], "ecosystem_type.category")
                p_cats = extract_field_set(pred[zid], "ecosystem_type.category")
                if g_cats & p_cats:
                    term_err_cat_ok += len(p - g)
                else:
                    term_err_cat_wrong += len(p - g)

            if field in FREE_TEXT_FIELDS and (p - g):
                for pv in p - g:
                    if g and difflib.SequenceMatcher(None, pv.lower(), " ".join(g).lower()).ratio() >= 0.6:
                        fuzzy_variant += 1
            if field == "taxa.scientific_name":
                if g and g != {"-"} and (p == {"-"} or not p):
                    collapsed_to_dash += 1
                    gold_specific_pred_dash_papers += 1

        stats[field] = {
            "perfect": perfect,
            "partial": partial,
            "over_fire": over_fire,
            "missed_all": missed_all,
            "top_confusions": confusions.most_common(12),
            "examples": examples[:n_examples],
            "term_err_cat_ok": term_err_cat_ok,
            "term_err_cat_wrong": term_err_cat_wrong,
            "fuzzy_variant": fuzzy_variant,
            "collapsed_to_dash": collapsed_to_dash,
        }
    return stats


def render_report(model_id: str, stats: dict[str, Any]) -> str:
    """Renders one model's error statistics as a Markdown section."""
    lines = [f"## Model: `{model_id}`", ""]
    lines.append(
        "| Field | Perfect | Partial | Over-fire (gold leer) | Missed (pred leer) | "
        "Top-Konfusion (gold → pred) |"
    )
    lines.append("| :--- | ---: | ---: | ---: | ---: | :--- |")
    for field, s in stats.items():
        top = "; ".join(
            f"{g or '∅'} → {p or '∅'} ({c}×)" for (g, p), c in s["top_confusions"][:3]
        )
        lines.append(
            f"| `{field}` | {s['perfect']} | {s['partial']} | {s['over_fire']} | "
            f"{s['missed_all']} | {top} |"
        )
    lines.append("")

    for field, s in stats.items():
        if field == "ecosystem_type.term" and (s["term_err_cat_ok"] or s["term_err_cat_wrong"]):
            lines.append(
                f"- **ecosystem_type.term**: {s['term_err_cat_ok']} falsche Terms bei "
                f"richtiger Kategorie vs. {s['term_err_cat_wrong']} bei falscher Kategorie "
                f"(Kategorie-First-Strategie würde ~"
                f"{s['term_err_cat_ok'] / max(1, s['term_err_cat_ok'] + s['term_err_cat_wrong']) * 100:.0f}% "
                f"dieser Fehler adressieren)."
            )
        if field in FREE_TEXT_FIELDS and s["fuzzy_variant"]:
            lines.append(
                f"- **{field}**: {s['fuzzy_variant']} der FPs sind near-miss-Varianten "
                f"(Ähnlichkeit ≥ 0.6 zum Gold-String) — Normalisierung (Kleinschreibung, "
                f"Diakritika, Unterart-Kürzung) könnte diese in TPs überführen."
            )
        if field == "taxa.scientific_name" and s["collapsed_to_dash"]:
            lines.append(
                f"- **taxa.scientific_name**: {s['collapsed_to_dash']} Papers kollabieren "
                f"zu '-' / leer, obwohl spezifische Arten im Gold stehen — kollektive "
                f"'-'-Vorhersage ist zu aggressiv."
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Categorized error analysis on Dev papers.")
    parser.add_argument(
        "--models", nargs="+",
        default=["ensemble_voting_baseline", "tfidf_svm_baseline"],
        help="Model IDs under outputs/<category>/ (default: ensemble + SVM)",
    )
    parser.add_argument("--examples", type=int, default=15, help="Example cases per field")
    args = parser.parse_args()

    gold = load_jsonl_by_id(REPO_ROOT / "data" / "gold" / "dev.jsonl")
    dev_ids = [
        l.strip()
        for l in (REPO_ROOT / "data" / "gold" / "dev_100_ids.txt").read_text().splitlines()
        if l.strip()
    ]

    out_dir = REPO_ROOT / "outputs" / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = [
        "# Error Analysis — Dev Set (100 Papers)",
        "",
        f"Generated for: {', '.join(f'`{m}`' for m in args.models)}",
        "",
        "Fehlerkategorien pro Feld: **Partial** (teilweise richtig), **Over-fire** "
        "(Gold leer, Modell feuert), **Missed** (Gold vorhanden, Modell leer). "
        "Beispiel-Fälle und Detail-Analysen folgen pro Modell.",
        "",
    ]
    for model_id in args.models:
        print(f"Analyzing {model_id} ...")
        preds = load_predictions(model_id)
        stats = analyze_model(gold, preds, dev_ids, args.examples)
        report.append(render_report(model_id, stats))

        # Save raw example cases for slide preparation
        with open(out_dir / f"{model_id}_error_cases.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    (out_dir / "error_analysis_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Report written to {out_dir / 'error_analysis_report.md'}")


if __name__ == "__main__":
    main()
