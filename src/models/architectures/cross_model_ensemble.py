#!/usr/bin/env python3
"""
========================================================================================
EXECUTIVE SUMMARY: Cross-Architecture & Cross-Model Voting Ensemble
========================================================================================
- Purpose:
    Combines complementary predictions from our strongest heterogeneous systems without
    requiring any additional LLM API calls:
      1. Architecture 1 (Monolithic HU-Reasoning):       C1 = 0.5754 (strongest habitat/eco_cat)
      2. Architecture 2 (DeepSeek-v4.1-Flash E1):        C1 = 0.5622, C1+C2b = 0.4244 (strongest term/taxa)
      3. Architecture 2 (HU-Reasoning E1):               C1 = 0.5593 (high quote precision)
      4. Classical ML Ensemble (9 Base Models):          C1 = 0.5260 (strong recall anchor)

- Mechanism:
    Per-field weighted voting across all 4 heterogeneous systems:
      - Simple list fields (`habitat`, `biodiversity_level`): items passing weighted vote
        threshold (with majority fallback if empty for dense fields).
      - Nested structured fields (`ecosystem_type`, `taxa`, `ecosystem_service`):
        merged by normalized canonical key, keeping entries with sufficient consensus.
========================================================================================
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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
from src.models.base_models.majority_baseline import fit_majority_model

MODEL_ID = "cross_architecture_ensemble"
CATEGORY = "architectures"
SEED = 42


def _load_sources() -> list[tuple[str, float, dict[str, dict[str, Any]]]]:
    """Loads the 4 heterogeneous prediction sets and their ensemble weights."""
    candidates = [
        (
            "monolithic_hu",
            1.2,
            REPO_ROOT / "outputs/architectures/llm_zeroshot_guidelines_hu_hu-reasoning/llm_zeroshot_guidelines_hu_hu-reasoning_predictions.jsonl",
        ),
        (
            "chunking_deepseek_e1",
            1.1,
            REPO_ROOT / "outputs/architectures/llm_parallel_chunking_saia_deepseek-v4.1-flash-1017_fg_term/llm_parallel_chunking_saia_deepseek-v4.1-flash-1017_fg_term_predictions.jsonl",
        ),
        (
            "chunking_hu_e1",
            1.0,
            REPO_ROOT / "outputs/architectures/llm_parallel_chunking_hu_hu-reasoning_fg_term/llm_parallel_chunking_hu_hu-reasoning_fg_term_predictions.jsonl",
        ),
        (
            "classical_ml_ensemble",
            0.7,
            REPO_ROOT / "outputs/base_models/ensemble_voting_baseline/ensemble_voting_baseline_predictions.jsonl",
        ),
    ]
    sources = []
    for name, weight, path in candidates:
        if path.exists():
            sources.append((name, weight, load_jsonl_by_id(path)))
        else:
            print(f"[WARNING] Source missing, skipping: {path}")
    return sources


def merge_paper_predictions(
    zot_id: str,
    sources: list[tuple[str, float, dict[str, dict[str, Any]]]],
    majority_learned: dict[str, dict[str, Any]],
    threshold: float = 1.7,
    sparse_threshold: float = 2.0,
) -> dict[str, Any]:
    """Merges predictions for a single paper across all sources via weighted voting."""
    hab_scores: dict[str, float] = defaultdict(float)
    bio_scores: dict[str, float] = defaultdict(float)
    eco_type_scores: dict[tuple[str, str], float] = defaultdict(float)
    taxa_scores: dict[tuple[str, str, str, bool], float] = defaultdict(float)
    service_scores: dict[tuple[str, str], float] = defaultdict(float)

    for _, weight, preds in sources:
        rec = preds.get(zot_id)
        if not rec:
            continue

        for h in rec.get("habitat") or []:
            if h:
                hab_scores[str(h).strip()] += weight

        for b in rec.get("biodiversity_level") or []:
            if b:
                bio_scores[str(b).strip()] += weight

        for et in rec.get("ecosystem_type") or []:
            if isinstance(et, dict) and et.get("category") and et.get("term"):
                eco_type_scores[(str(et["category"]).strip(), str(et["term"]).strip())] += weight

        for t in rec.get("taxa") or []:
            if isinstance(t, dict) and t.get("species_group"):
                grp = str(t["species_group"]).strip()
                sci = str(t.get("scientific_name") or "-").strip() or "-"
                ger = str(t.get("german_name") or grp).strip() or grp
                coll = bool(t.get("collective_term", sci == "-"))
                taxa_scores[(grp, sci, ger, coll)] += weight

        for es in rec.get("ecosystem_service") or []:
            if isinstance(es, dict) and es.get("category"):
                cat = str(es["category"]).strip()
                term = str(es.get("term") or "").strip()
                service_scores[(cat, term)] += weight

    # 1. Habitat (dense field -> threshold or top-1 fallback)
    habitats = [
        h for h, s in sorted(hab_scores.items(), key=lambda kv: -kv[1]) if s >= threshold
    ]
    if not habitats and hab_scores:
        habitats = [max(hab_scores.items(), key=lambda kv: kv[1])[0]]
    if not habitats:
        habitats = [majority_learned["habitat"]["majority_value"]]

    # 2. Biodiversity level
    bio_levels = [
        b for b, s in sorted(bio_scores.items(), key=lambda kv: -kv[1]) if s >= threshold
    ]

    # 3. Ecosystem type (require consensus; fallback to top-1 if available)
    eco_types = [
        {"category": cat, "term": term}
        for (cat, term), s in sorted(eco_type_scores.items(), key=lambda kv: -kv[1])
        if s >= threshold
    ]
    if not eco_types and eco_type_scores:
        best_cat, best_term = max(eco_type_scores.items(), key=lambda kv: kv[1])[0]
        eco_types = [{"category": best_cat, "term": best_term}]

    # 4. Taxa (group-level consensus + specific species if backed)
    taxa_list = []
    seen_taxa: set[tuple[str, str]] = set()
    for (grp, sci, ger, coll), s in sorted(taxa_scores.items(), key=lambda kv: -kv[1]):
        if s >= threshold and (grp, sci) not in seen_taxa:
            seen_taxa.add((grp, sci))
            taxa_list.append(
                {
                    "species_group": grp,
                    "scientific_name": sci,
                    "german_name": ger,
                    "collective_term": coll,
                }
            )
    if not taxa_list and taxa_scores:
        (grp, sci, ger, coll) = max(taxa_scores.items(), key=lambda kv: kv[1])[0]
        taxa_list.append(
            {
                "species_group": grp,
                "scientific_name": sci,
                "german_name": ger,
                "collective_term": coll,
            }
        )

    # 5. Ecosystem service (very sparse -> higher threshold)
    eco_services = [
        {"category": cat, "term": term} if term else {"category": cat}
        for (cat, term), s in sorted(service_scores.items(), key=lambda kv: -kv[1])
        if s >= sparse_threshold
    ][:2]

    return {
        "zotitem_ptr_id": zot_id,
        "habitat": habitats,
        "biodiversity_level": bio_levels,
        "ecosystem_type": eco_types[:3],
        "taxa": taxa_list[:6],
        "ecosystem_service": eco_services,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-Architecture Voting Ensemble")
    parser.add_argument("--threshold", type=float, default=1.7, help="Consensus vote threshold")
    parser.add_argument("--sparse-threshold", type=float, default=2.0, help="Threshold for sparse fields")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    gold = load_jsonl_by_id(REPO_ROOT / "data/gold/dev.jsonl")
    train = load_jsonl_by_id(REPO_ROOT / "data/gold/train.jsonl")
    dev_ids = [l.strip() for l in (REPO_ROOT / "data/gold/dev_100_ids.txt").read_text().splitlines() if l.strip()]

    majority_learned = fit_majority_model(train)
    sources = _load_sources()
    if not sources:
        raise SystemExit("No prediction sources found.")

    predictions = {
        zid: merge_paper_predictions(
            zid, sources, majority_learned, args.threshold, args.sparse_threshold
        )
        for zid in dev_ids
    }

    metrics = evaluate_predictions(gold, predictions, filter_ids=set(dev_ids))
    print_report("Cross-Architecture Voting Ensemble", metrics)

    if not args.no_save:
        save_reproducible_run(
            model_id=MODEL_ID,
            model_title="Cross-Architecture Voting Ensemble (Monolithic HU + DeepSeek E1 + HU E1 + Classical ML)",
            category=CATEGORY,
            predictions=predictions,
            metrics=metrics,
            runtime_seconds=time.time() - t0,
            seed=SEED,
            config={
                "sources": [(n, w) for n, w, _ in sources],
                "threshold": args.threshold,
                "sparse_threshold": args.sparse_threshold,
            },
        )


if __name__ == "__main__":
    main()
