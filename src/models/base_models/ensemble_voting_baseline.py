#!/usr/bin/env python3
"""
========================================================================================
EXECUTIVE SUMMARY — Base Model 9: Approval-Voting Ensemble of the 4 Supervised Baselines
========================================================================================
- Concept & Goal:
    Tests whether ensembling beats the best single classical model (TF-IDF + Linear SVM,
    C1 Macro-F1 = 0.4897). The four supervised base models — TF-IDF + Linear SVM,
    BoW + Complement Naive Bayes, TF-IDF + Logistic Regression, and k-NN Paper Retrieval —
    make partially uncorrelated errors (e.g., k-NN is strongest on the 41-class
    ecosystem_type.term, SVM on ecosystem_service.category). An ensemble that only keeps
    values on which several models agree should trade a little recall for substantially
    higher precision — exactly what the error profiles demand (e.g., SVM habitat:
    74 FP vs. 43 FN).
- How It Works:
    1. Zero Retraining: loads the already-saved Dev predictions
       (`outputs/base_models/<voter>/<voter>_predictions.jsonl`) of the four supervised
       baselines — all evaluated on the same 100 valid Dev papers.
    2. Approval Voting (threshold k of M = 4 voters): for every categorical eval field
       (habitat, biodiversity_level, ecosystem_type.category/term,
       ecosystem_service.category/term), a value enters the final prediction iff at
       least k voters predicted it. Set-valued fields are the natural domain of approval
       voting — no tie-breaking is ever needed. The threshold was selected by sensitivity
       analysis on the Dev set (k=1: 0.4232, k=2: 0.4822, k=3: 0.5260, k=4: 0.5065 C1
       Macro-F1): union (k=1) floods false positives, unanimity (k=4) loses recall on
       disagreement-heavy fields; strict majority k=3 wins. The Dev set is the designated
       development set of the shared task — the official CodaBench test set remains
       untouched by this choice.
    3. Coherent Object Reconstruction: taxa are voted as exact
       (species_group, scientific_name, german_name) triples (threshold k), preserving
       annotation coherence — in particular the '-' convention for collective groups
       (collective_term is re-derived as scientific_name == '-'). For ecosystem_type /
       ecosystem_service, category and term sets are voted independently (matching how
       the evaluator scores them) and re-zipped into objects; surplus terms are attached
       to the first surviving category, surplus categories receive term = null.
    4. Empty-Field Behaviour: values predicted by fewer than k models are dropped, so
       sparse fields (ecosystem_service, 78.1% empty in Train) converge towards empty
       predictions — usually the correct answer, at the price of recall on the few
       papers that genuinely study an ecosystem service.
- Results on Dev Set (100 valid papers, deterministic, k = 3 of 4):
    * C1 Leaderboard Macro-F1 : 0.5260  (best model overall; +0.0363 over SVM's 0.4897)
    * C1 + C2b Macro-F1       : 0.3783  (vs. 0.3675 SVM / 0.3657 LogReg)
    * C1 + C2b Micro-F1       : 0.3966  (best)
    * Per-Field Highlights    : habitat 0.6526 and ecosystem_type.category 0.6508 are
                                the best values of ANY model so far; ecosystem_type.term
                                0.4423, taxa.species_group 0.4741,
                                biodiversity_level 0.4103.
    * Takeaway                : First model above the 0.50 C1 barrier. Strict-majority
                                voting buys precision exactly where the single models
                                bled false positives (habitat: 0.6327 prec / 0.6526 F1).
                                The price: sparse-field recall (ecosystem_service.category
                                drops to 0.3077 vs. 0.4211 for SVM) and open-vocabulary
                                C2b species names, where exact-triple agreement is too
                                strict — a directed argument for the LLM architectures
                                and for hybrid per-field ensembling.
========================================================================================
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluator import (
    evaluate_predictions,
    load_jsonl_by_id,
    print_report,
    save_reproducible_run,
)

# The four supervised voters whose saved Dev predictions are combined.
VOTER_IDS: list[str] = [
    "tfidf_svm_baseline",
    "bow_naive_bayes_baseline",
    "tfidf_logreg_baseline",
    "knn_retrieval_baseline",
]

# Approval threshold: strict majority, selected via Dev sensitivity analysis
# (k=1: 0.4232 | k=2: 0.4822 | k=3: 0.5260 | k=4: 0.5065 C1 Macro-F1).
VOTE_THRESHOLD: int = 3

# Plain categorical fields voted directly on their value sets.
SIMPLE_VOTED_FIELDS: list[str] = [
    "habitat",
    "biodiversity_level",
    "ecosystem_type.category",
    "ecosystem_type.term",
    "ecosystem_service.category",
    "ecosystem_service.term",
    "biodiversity_variable",
]


def load_voter_predictions() -> dict[str, dict[str, dict[str, Any]]]:
    """Loads each voter's saved Dev predictions keyed by `zotitem_ptr_id`."""
    voters: dict[str, dict[str, dict[str, Any]]] = {}
    for vid in VOTER_IDS:
        path = REPO_ROOT / "outputs" / "base_models" / vid / f"{vid}_predictions.jsonl"
        preds: dict[str, dict[str, Any]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                preds[obj["zotitem_ptr_id"]] = obj
        voters[vid] = preds
    return voters


def vote_value_sets(
    records: list[dict[str, Any]], field_path: str, threshold: int
) -> list[str]:
    """
    Approval voting for a (possibly nested) categorical field: returns the values that
    appear in at least `threshold` of the voter records for this paper.
    """
    from src.evaluation.evaluator import extract_field_set

    counter: Counter[str] = Counter()
    for rec in records:
        counter.update(extract_field_set(rec, field_path))
    # Sort alphabetically for a deterministic, order-stable output list
    return sorted(v for v, c in counter.items() if c >= threshold)


def vote_taxa_triples(records: list[dict[str, Any]], threshold: int) -> list[dict[str, Any]]:
    """
    Approval voting on coherent taxa triples: a (species_group, scientific_name,
    german_name) triple survives iff at least `threshold` voters predicted exactly it.
    Keeping triples intact (instead of voting each subfield separately) preserves the
    annotation coherence between group, Latin binomial, and German name — including the
    '-' convention for collective groups.
    """
    counter: Counter[tuple] = Counter()
    for rec in records:
        triples = {
            (
                str(t.get("species_group") or "").strip(),
                str(t.get("scientific_name") or "").strip(),
                str(t.get("german_name") or "").strip(),
            )
            for t in (rec.get("taxa") or [])
            if isinstance(t, dict)
        }
        counter.update(triples)

    out: list[dict[str, Any]] = []
    for (group, sci, ger), _count in sorted(counter.items()):
        if not group or _count < threshold:
            continue
        out.append(
            {
                "species_group": group,
                "scientific_name": sci or None,
                "german_name": ger or None,
                # '-' as scientific name marks a collective taxonomic group
                "collective_term": sci == "-",
            }
        )
    return out


def vote_object_field(
    records: list[dict[str, Any]], parent: str, threshold: int
) -> list[dict[str, Any]]:
    """
    Approval voting for ecosystem_type / ecosystem_service: category and term value sets
    are voted independently (mirroring how the evaluator scores `.category` and `.term`
    separately), then re-zipped into objects. Surplus terms attach to the first surviving
    category; surplus categories receive term = null. Metric-neutral by construction
    because evaluation reads the sets, not the pairing.
    """
    categories = vote_value_sets(records, f"{parent}.category", threshold)
    terms = vote_value_sets(records, f"{parent}.term", threshold)
    objects: list[dict[str, Any]] = []
    for i, cat in enumerate(categories):
        objects.append({"category": cat, "term": terms[i] if i < len(terms) else None})
    # Terms without a matching category slot are appended under the first category
    for term in terms[len(categories):]:
        objects.append({"category": categories[0] if categories else None, "term": term})
    return objects


def ensemble_single_paper(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Combines the four voters' predictions for one paper into an ensemble prediction."""
    pred: dict[str, Any] = {}
    pred["habitat"] = vote_value_sets(records, "habitat", VOTE_THRESHOLD)
    pred["biodiversity_level"] = vote_value_sets(
        records, "biodiversity_level", VOTE_THRESHOLD
    )
    pred["biodiversity_variable"] = vote_value_sets(
        records, "biodiversity_variable", VOTE_THRESHOLD
    )
    pred["ecosystem_type"] = vote_object_field(records, "ecosystem_type", VOTE_THRESHOLD)
    pred["ecosystem_service"] = vote_object_field(
        records, "ecosystem_service", VOTE_THRESHOLD
    )
    pred["taxa"] = vote_taxa_triples(records, VOTE_THRESHOLD)
    return pred


def main() -> None:
    global VOTE_THRESHOLD  # overridable via --threshold for sensitivity analysis
    parser = argparse.ArgumentParser(
        description="Approval-voting ensemble over the saved predictions of the 4 "
        "supervised base models (no retraining, deterministic)."
    )
    parser.add_argument(
        "--threshold", type=int, default=VOTE_THRESHOLD,
        help="Approval threshold (default: 3 = strict majority of 4 voters)",
    )
    args = parser.parse_args()
    VOTE_THRESHOLD = args.threshold

    start_time = time.time()

    # --- load gold labels, valid Dev IDs, and all voter predictions ---
    gold = load_jsonl_by_id(REPO_ROOT / "data" / "gold" / "dev.jsonl")
    dev_ids = [
        line.strip()
        for line in (REPO_ROOT / "data" / "gold" / "dev_100_ids.txt").read_text().splitlines()
        if line.strip()
    ]
    voters = load_voter_predictions()
    for vid, preds in voters.items():
        missing = [z for z in dev_ids if z not in preds]
        if missing:
            raise SystemExit(f"ERROR: voter '{vid}' is missing {len(missing)} Dev papers")

    # --- combine per paper ---
    predictions: dict[str, dict[str, Any]] = {}
    for zot_id in dev_ids:
        pred = ensemble_single_paper([v[zot_id] for v in voters.values()])
        pred["zotitem_ptr_id"] = zot_id
        predictions[zot_id] = pred

    # --- evaluate and save standardized artifacts ---
    metrics = evaluate_predictions(gold, predictions, filter_ids=set(dev_ids))
    print_report(
        f"Base Model 9: Approval-Voting Ensemble (k={VOTE_THRESHOLD}/4, deterministic)",
        metrics,
    )

    save_reproducible_run(
        model_id="ensemble_voting_baseline",
        model_title="Base Model 9: Approval-Voting Ensemble (4 Supervised Baselines)",
        category="base_models",
        predictions=predictions,
        metrics=metrics,
        runtime_seconds=time.time() - start_time,
        seed=42,  # fixed convention; the procedure itself is fully deterministic
        config={
            "voters": VOTER_IDS,
            "vote_threshold": VOTE_THRESHOLD,
            "taxa_voting": "exact triple (species_group, scientific_name, german_name)",
            "note": "no retraining — combines saved Dev predictions of the voters",
        },
    )


if __name__ == "__main__":
    main()
