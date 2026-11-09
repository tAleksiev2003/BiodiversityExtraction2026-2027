#!/usr/bin/env python3
"""
========================================================================================
EXECUTIVE SUMMARY — Base Model 7: k-Nearest-Neighbors (k-NN) Paper Retrieval Baseline
========================================================================================
- Concept & Goal:
    Implements a non-parametric, memory-based retrieval baseline ("Instance-Based Learning").
    Instead of fitting parametric weights per class, the model indexes all 908 training
    papers (`data/gold/train.jsonl`) in a TF-IDF vector space. For each Dev paper, it
    retrieves the `k = 5` most similar training papers via cosine similarity and transfers
    their annotations via similarity-weighted majority voting.
    This directly tests how much label structure can be recovered purely by finding
    topically similar ecological studies—serving as the classical precursor to Retrieval-
    Augmented Generation (RAG) and dynamic Few-Shot In-Context Learning (ICL) for LLMs.
- How It Works:
    1. Vector Space Indexing: Encodes the 908 training papers and 100 Dev papers using
       L2-normalized Word (1-2 gram) and Character-WB (4-5 gram) TF-IDF vectors.
    2. Cosine Top-k Retrieval: Computes the exact cosine similarity matrix `(100 x 908)`
       and selects the `k = 5` nearest training neighbors for each Dev paper.
    3. Similarity-Weighted Label Voting & Sparsity Preservation: Each label held by a
       neighbor receives a vote proportional to that neighbor's cosine similarity.
       Because empty fields in neighbors cast zero votes, sparse fields (`ecosystem_service`,
       78.1% empty) naturally remain empty unless multiple close neighbors share that label.
    4. Subtask C2b Taxa Resolution: Combines exact species lexicon matches found in the
       Dev paper text with `taxa.species_group` labels voted by the `k` nearest neighbors
       (using `scientific_name = "-"` for broad taxonomic groups).
- Results on Dev Set (100 valid papers, deterministic):
    * C1 Leaderboard Macro-F1 : 0.4793
    * C1 + C2b Macro-F1       : 0.3419
    * C1 + C2b Micro-F1       : 0.3610
    * Per-Field Highlights    : habitat (0.6268), ecosystem_type.category (0.6066),
                                ecosystem_type.term (0.4330 — best across all 8 baselines!),
                                taxa.scientific_name (0.2500).
    * Runtime                 : ~9.3s
========================================================================================
"""

from collections import defaultdict
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluator import (
    evaluate_predictions,
    extract_field_set,
    load_jsonl_by_id,
    print_report,
    save_reproducible_run,
)
from src.models.base_models.dictionary_baseline import (
    build_train_taxa_lexicon,
    extract_and_cache_paper_text,
)

# Number of nearest training papers retrieved per Dev paper
TOP_K_NEIGHBORS: int = 5

# Categorical fields predicted via similarity-weighted k-NN voting
VOTED_FIELDS: list[str] = [
    "habitat",
    "biodiversity_level",
    "ecosystem_type.category",
    "ecosystem_type.term",
    "taxa.species_group",
    "ecosystem_service.category",
    "ecosystem_service.term",
]

# Minimum fraction of similarity-weighted neighbor votes required to emit a label
FIELD_VOTE_THRESHOLDS: dict[str, float] = {
    "habitat": 0.30,
    "biodiversity_level": 0.32,
    "ecosystem_type.category": 0.32,
    "ecosystem_type.term": 0.28,
    "taxa.species_group": 0.30,
    "ecosystem_service.category": 0.45,
    "ecosystem_service.term": 0.45,
}

# Dense fields that always emit at least their top-1 highest-voted neighbor class
ALWAYS_EMIT_TOP1_FIELDS: set[str] = {
    "habitat",
    "taxa.species_group",
}


def build_retrieval_vectorizer() -> FeatureUnion:
    """
    Constructs a combined Word (1-2 gram) and Character-WB (4-5 gram) TF-IDF vectorizer
    for computing document-to-document cosine similarity between scientific papers.
    """
    return FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.90,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(4, 5),
                    min_df=3,
                    max_df=0.90,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def prepare_corpus_texts(
    records: dict[str, dict[str, Any]],
    ids: list[str],
    pdf_dir: Path,
    cache_dir: Path,
    use_pdfs: bool = False,
) -> list[str]:
    """
    Prepares lowercased text documents for each paper ID.
    Uses `bibtex_title` for the 908 training papers and `bibtex_title` (3x) + cached first
    2 PDF pages for the 100 Dev papers.
    """
    texts: list[str] = []
    for zid in ids:
        rec = records[zid]
        title = str(rec.get("bibtex_title") or "").strip()
        if use_pdfs:
            full_text, _ = extract_and_cache_paper_text(
                zid, rec, pdf_dir, cache_dir, max_pages=2
            )
            combined = f"{title}\n{title}\n{title}\n{full_text}"
        else:
            combined = title.lower()
        texts.append(combined)
    return texts


def vote_field_labels_from_neighbors(
    neighbor_ids: list[str],
    neighbor_weights: np.ndarray,
    train_records: dict[str, dict[str, Any]],
    field: str,
) -> list[str]:
    """
    Aggregates gold labels from the `k` nearest training neighbors using cosine-similarity
    weights and returns the labels exceeding the field's vote threshold (capped at top 2).
    """
    total_weight = float(np.sum(neighbor_weights)) or 1.0
    label_scores: dict[str, float] = defaultdict(float)

    for nid, weight in zip(neighbor_ids, neighbor_weights):
        gold_labels = extract_field_set(train_records[nid], field)
        for lbl in gold_labels:
            label_scores[lbl] += float(weight) / total_weight

    if not label_scores:
        return []

    threshold = FIELD_VOTE_THRESHOLDS[field]
    ranked = sorted(label_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    selected = [lbl for lbl, score in ranked if score >= threshold][:2]

    # Dense fields always emit at least the single strongest neighbor label
    if not selected and field in ALWAYS_EMIT_TOP1_FIELDS:
        selected = [ranked[0][0]]

    return selected


def predict_knn_retrieval(
    X_train_texts: list[str],
    train_records: dict[str, dict[str, Any]],
    train_ids: list[str],
    X_dev_texts: list[str],
    dev_ids: list[str],
    compiled_lexicon: list[tuple[re.Pattern[str], str, str, str]],
    k: int = TOP_K_NEIGHBORS,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """
    Retrieves the top-`k` most similar training papers for each Dev paper and constructs
    schema-valid predictions via similarity-weighted neighbor voting.
    Also returns a diagnostic neighbor log mapping each Dev ID to its retrieved neighbors.
    """
    vectorizer = build_retrieval_vectorizer()
    X_train = vectorizer.fit_transform(X_train_texts)
    X_dev = vectorizer.transform(X_dev_texts)

    # Compute (N_dev x N_train) cosine similarity matrix
    sim_matrix = cosine_similarity(X_dev, X_train)

    output_records: dict[str, dict[str, Any]] = {}
    retrieval_diagnostics: dict[str, list[dict[str, Any]]] = {}

    for i, zid in enumerate(dev_ids):
        sims = sim_matrix[i]
        # Sort training indices descending by cosine similarity
        top_indices = np.argsort(sims)[::-1][:k]
        neighbor_ids = [train_ids[idx] for idx in top_indices]
        # Add small epsilon to prevent zero-division if all similarities are 0
        neighbor_weights = np.maximum(sims[top_indices], 1e-6)

        retrieval_diagnostics[zid] = [
            {
                "train_id": nid,
                "cosine_similarity": round(float(sims[idx]), 4),
                "train_title": str(train_records[nid].get("bibtex_title") or "")[:100],
            }
            for idx, nid in zip(top_indices, neighbor_ids)
        ]

        voted: dict[str, list[str]] = {
            field: vote_field_labels_from_neighbors(
                neighbor_ids, neighbor_weights, train_records, field
            )
            for field in VOTED_FIELDS
        }

        dev_text = X_dev_texts[i]

        # 1. Habitat & Biodiversity Level
        habitats = voted["habitat"]
        bio_levels = voted["biodiversity_level"]

        # 2. Ecosystem Type
        eco_cats = voted["ecosystem_type.category"]
        eco_terms = voted["ecosystem_type.term"]
        eco_types = []
        for idx, cat in enumerate(eco_cats):
            term = eco_terms[idx] if idx < len(eco_terms) else (eco_terms[0] if eco_terms else None)
            eco_types.append({"category": cat, "term": term})

        # 3. Taxa (exact lexicon matches in Dev text + neighbor-voted species_group)
        taxa_entries = []
        seen_groups = set()
        for pattern, sci, ger, grp in compiled_lexicon:
            if pattern.search(dev_text):
                taxa_entries.append(
                    {
                        "species_group": grp,
                        "scientific_name": sci,
                        "german_name": ger,
                    }
                )
                seen_groups.add(grp)
                if len(taxa_entries) >= 3:
                    break

        for grp in voted["taxa.species_group"]:
            if grp not in seen_groups:
                taxa_entries.append(
                    {
                        "species_group": grp,
                        "scientific_name": "-",
                        "german_name": grp,
                    }
                )
                seen_groups.add(grp)

        # 4. Ecosystem Service
        srv_cats = voted["ecosystem_service.category"]
        srv_terms = voted["ecosystem_service.term"]
        eco_services = []
        for idx, cat in enumerate(srv_cats):
            term = srv_terms[idx] if idx < len(srv_terms) else (srv_terms[0] if srv_terms else None)
            eco_services.append({"category": cat, "term": term})

        output_records[zid] = {
            "zotitem_ptr_id": zid,
            "habitat": habitats,
            "biodiversity_level": bio_levels,
            "ecosystem_type": eco_types,
            "taxa": taxa_entries,
            "ecosystem_service": eco_services,
        }

    return output_records, retrieval_diagnostics


def main() -> None:
    start_time = time.perf_counter()

    train_path = REPO_ROOT / "data" / "gold" / "train.jsonl"
    dev_path = REPO_ROOT / "data" / "gold" / "dev.jsonl"
    dev_ids_path = REPO_ROOT / "data" / "gold" / "dev_100_ids.txt"
    pdf_dir = REPO_ROOT / "data" / "raw_pdfs" / "dev"
    cache_dir = REPO_ROOT / "data" / "parsed_text" / "dev"

    train_records = load_jsonl_by_id(train_path)
    dev_records = load_jsonl_by_id(dev_path)
    train_ids = sorted(train_records.keys())
    dev_100_ids = sorted(
        {line.strip() for line in dev_ids_path.read_text().splitlines() if line.strip()}
    )

    taxa_lexicon = build_train_taxa_lexicon(train_records)
    compiled_lexicon = [
        (re.compile(r"\b" + re.escape(term_key) + r"\b"), sci, ger, grp)
        for term_key, (sci, ger, grp) in taxa_lexicon.items()
    ]

    X_train_texts = prepare_corpus_texts(
        train_records, train_ids, pdf_dir, cache_dir, use_pdfs=False
    )
    X_dev_texts = prepare_corpus_texts(
        dev_records, dev_100_ids, pdf_dir, cache_dir, use_pdfs=True
    )

    predictions, retrieval_diagnostics = predict_knn_retrieval(
        X_train_texts=X_train_texts,
        train_records=train_records,
        train_ids=train_ids,
        X_dev_texts=X_dev_texts,
        dev_ids=dev_100_ids,
        compiled_lexicon=compiled_lexicon,
        k=TOP_K_NEIGHBORS,
    )

    runtime_seconds = time.perf_counter() - start_time
    metrics = evaluate_predictions(dev_records, predictions, filter_ids=set(dev_100_ids))

    model_id = "knn_retrieval_baseline"
    model_title = f"Base Model 7: k-NN Cosine Similarity Retrieval (k={TOP_K_NEIGHBORS})"
    out_dir = save_reproducible_run(
        model_id=model_id,
        model_title=model_title,
        category="base_models",
        predictions=predictions,
        metrics=metrics,
        runtime_seconds=runtime_seconds,
        seed=None,
        config={
            "strategy": "tfidf_cosine_similarity_knn_voting",
            "top_k_neighbors": TOP_K_NEIGHBORS,
            "features": "Word TF-IDF (1-2 grams) + Char-WB TF-IDF (4-5 grams)",
            "field_vote_thresholds": FIELD_VOTE_THRESHOLDS,
            "always_emit_top1_fields": sorted(ALWAYS_EMIT_TOP1_FIELDS),
            "num_train_papers": len(train_ids),
            "num_dev_papers": len(dev_100_ids),
        },
        extra_artifacts={
            f"{model_id}_neighbors_sample.json": dict(list(retrieval_diagnostics.items())[:15])
        },
    )

    print_report(model_title, metrics)
    print(f"Reproducible run artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
