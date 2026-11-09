#!/usr/bin/env python3
"""
========================================================================================
EXECUTIVE SUMMARY — Base Model 4: TF-IDF + Multi-Label Logistic Regression Baseline
========================================================================================
- Concept:
    Classical supervised machine learning baseline ("Simple Learned Baseline" from
    HowTo_Approach_an_NLP_Task.pdf & Lecture 1 N-Gram representations).
    Transforms paper text into L2-normalized TF-IDF word & character N-gram vectors and
    trains an independent One-vs-Rest (OvR) Logistic Regression classifier per schema field
    on the 908 training papers (`train.jsonl`).
- Input Features:
    Combined word N-grams (1-2) and character N-grams (3-5) extracted from the paper title
    (and cached PDF abstract/intro pages where available). Character N-grams capture
    morphological roots across English and German ecological terminology (e.g., "forest" /
    "forst", "benthic" / "benthal").
- Handling Empty Fields & Hierarchy:
    1. Probability Thresholding: Instead of forcing a prediction on every paper, a label is
       only emitted if `P(label | text) >= field_threshold`. Sparse fields like
       `ecosystem_service` (78.1% empty) and `ecosystem_type` (50.2% empty) use stricter
       thresholds, naturally predicting `[]` when confidence is low.
    2. Fallback for Dense Fields: For dense fields (`habitat`, only 5.7% empty, and
       `taxa.species_group`), if no class exceeds the threshold, the highest-scoring class
       (`argmax`) is selected.
    3. Taxa Lexicon Hybrid (Subtask C2b): Combines learned `taxa.species_group` predictions
       with exact species lexicon matching (`scientific_name` & `german_name`) and applies
       the `"-"` collective term convention when only a broad group is predicted.
- Reproducibility:
    Evaluated across 3 fixed random seeds ([42, 43, 44]) for the linear solver, logging all
    standardized artifacts to `outputs/base_models/tfidf_logreg_baseline/`.
========================================================================================
"""

from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MultiLabelBinarizer

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
from src.models.base_models.dictionary_baseline import (
    build_train_taxa_lexicon,
    extract_and_cache_paper_text,
)

SEEDS: list[int] = [42, 43, 44]

# Categorical fields modeled via Multi-Label Logistic Regression
LEARNED_FIELDS: list[str] = [
    "habitat",
    "biodiversity_level",
    "ecosystem_type.category",
    "ecosystem_type.term",
    "taxa.species_group",
    "ecosystem_service.category",
    "ecosystem_service.term",
]

# Field-specific probability thresholds reflecting training set sparsity:
# Dense fields use lower thresholds (and argmax fallback); sparse fields require high confidence.
FIELD_THRESHOLDS: dict[str, float] = {
    "habitat": 0.28,
    "biodiversity_level": 0.32,
    "ecosystem_type.category": 0.34,
    "ecosystem_type.term": 0.26,
    "taxa.species_group": 0.28,
    "ecosystem_service.category": 0.45,
    "ecosystem_service.term": 0.40,
}

# Dense fields that should always emit at least their top-1 highest-probability class
ALWAYS_EMIT_TOP1_FIELDS: set[str] = {
    "habitat",
    "taxa.species_group",
}


def build_feature_extractor() -> FeatureUnion:
    """
    Constructs a combined Word (1-2 gram) and Character-WB (3-5 gram) TF-IDF vectorizer.
    Character n-grams help bridge English paper text and German scientific terminology.
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
    Builds the input text string for each paper ID.
    For training papers without local PDFs, uses `bibtex_title`.
    For Dev papers, combines `bibtex_title` (weighted 3x for topical salience) with
    the first 2 pages of cached PDF text.
    """
    texts: list[str] = []
    for zid in ids:
        rec = records[zid]
        title = str(rec.get("bibtex_title") or "").strip()
        if use_pdfs:
            full_text, _ = extract_and_cache_paper_text(
                zid, rec, pdf_dir, cache_dir, max_pages=2
            )
            # Repeat title 3x so title keywords retain strong TF-IDF weight alongside PDF body
            combined = f"{title}\n{title}\n{title}\n{full_text}"
        else:
            combined = title.lower()
        texts.append(combined)
    return texts


def train_and_predict_for_seed(
    seed: int,
    X_train_texts: list[str],
    train_records: dict[str, dict[str, Any]],
    train_ids: list[str],
    X_dev_texts: list[str],
    dev_ids: list[str],
    taxa_lexicon: dict[str, tuple[str, str, str]],
) -> dict[str, dict[str, Any]]:
    """
    Fits the TF-IDF vectorizer and per-field One-vs-Rest Logistic Regression classifiers
    for a given random seed, then returns schema-compliant predictions for all Dev papers.
    """
    vectorizer = build_feature_extractor()
    X_train = vectorizer.fit_transform(X_train_texts)
    X_dev = vectorizer.transform(X_dev_texts)

    field_predictions: dict[str, list[list[str]]] = {}

    for field in LEARNED_FIELDS:
        # Extract gold label sets per training paper
        y_train_sets = [sorted(extract_field_set(train_records[zid], field)) for zid in train_ids]
        mlb = MultiLabelBinarizer()
        Y_train = mlb.fit_transform(y_train_sets)

        # Train balanced One-vs-Rest Logistic Regression to handle severe class imbalance
        clf = OneVsRestClassifier(
            LogisticRegression(
                C=2.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
                solver="liblinear",
            )
        )
        clf.fit(X_train, Y_train)

        # Predict calibrated class probabilities on the Dev set
        probs = clf.predict_proba(X_dev)
        threshold = FIELD_THRESHOLDS[field]

        preds_for_field: list[list[str]] = []
        for i in range(len(dev_ids)):
            row_probs = probs[i]
            selected_indices = np.where(row_probs >= threshold)[0].tolist()

            # Cap maximum predicted labels per field at top-2 by probability to preserve precision
            selected_indices.sort(key=lambda idx: row_probs[idx], reverse=True)
            selected_indices = selected_indices[:2]

            # Fallback for dense fields (`habitat`, `taxa.species_group`): always emit top-1
            if not selected_indices and field in ALWAYS_EMIT_TOP1_FIELDS:
                best_idx = int(np.argmax(row_probs))
                selected_indices = [best_idx]

            labels = [str(mlb.classes_[idx]) for idx in selected_indices]
            preds_for_field.append(labels)

        field_predictions[field] = preds_for_field

    # Assemble structured JSON records per Dev paper
    output_records: dict[str, dict[str, Any]] = {}
    for i, zid in enumerate(dev_ids):
        dev_text = X_dev_texts[i]

        # 1. Habitat & Biodiversity Level
        habitats = field_predictions["habitat"][i]
        bio_levels = field_predictions["biodiversity_level"][i]

        # 2. Ecosystem Type (pair predicted categories with predicted terms)
        eco_cats = field_predictions["ecosystem_type.category"][i]
        eco_terms = field_predictions["ecosystem_type.term"][i]
        eco_types = []
        for idx, cat in enumerate(eco_cats):
            term = eco_terms[idx] if idx < len(eco_terms) else (eco_terms[0] if eco_terms else None)
            eco_types.append({"category": cat, "term": term})

        # 3. Taxa (combine exact lexicon species matches with learned species_group predictions)
        taxa_entries = []
        seen_groups = set()
        for term_key, (sci, ger, grp) in taxa_lexicon.items():
            if re.search(r"\b" + re.escape(term_key) + r"\b", dev_text):
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

        # Add learned species groups using the "-" collective term convention
        for grp in field_predictions["taxa.species_group"][i]:
            if grp not in seen_groups:
                taxa_entries.append(
                    {
                        "species_group": grp,
                        "scientific_name": "-",
                        "german_name": grp,
                    }
                )
                seen_groups.add(grp)

        # 4. Ecosystem Service (only emit if category exceeded the strict threshold)
        srv_cats = field_predictions["ecosystem_service.category"][i]
        srv_terms = field_predictions["ecosystem_service.term"][i]
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

    return output_records


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

    # Prepare text corpora (908 Train titles vs. 100 Dev titles + cached PDF first 2 pages)
    X_train_texts = prepare_corpus_texts(
        train_records, train_ids, pdf_dir, cache_dir, use_pdfs=False
    )
    X_dev_texts = prepare_corpus_texts(
        dev_records, dev_100_ids, pdf_dir, cache_dir, use_pdfs=True
    )

    seed_metrics: dict[int, dict[str, Any]] = {}
    primary_predictions: dict[str, dict[str, Any]] = {}

    for seed in SEEDS:
        preds = train_and_predict_for_seed(
            seed=seed,
            X_train_texts=X_train_texts,
            train_records=train_records,
            train_ids=train_ids,
            X_dev_texts=X_dev_texts,
            dev_ids=dev_100_ids,
            taxa_lexicon=taxa_lexicon,
        )
        seed_metrics[seed] = evaluate_predictions(
            dev_records, preds, filter_ids=set(dev_100_ids)
        )
        if seed == SEEDS[0]:
            primary_predictions = preds

    runtime_seconds = time.perf_counter() - start_time

    # Aggregate 3-seed statistics
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
        "field_thresholds": FIELD_THRESHOLDS,
    }

    aggregated_metrics = dict(seed_metrics[SEEDS[0]])
    aggregated_metrics["c1_macro_f1"] = multi_seed_summary["c1_macro_f1_mean"]
    aggregated_metrics["c1_macro_f1_std"] = multi_seed_summary["c1_macro_f1_std"]
    aggregated_metrics["c2b_macro_f1"] = multi_seed_summary["c2b_macro_f1_mean"]
    aggregated_metrics["c2b_macro_f1_std"] = multi_seed_summary["c2b_macro_f1_std"]
    aggregated_metrics["micro_f1"] = multi_seed_summary["micro_f1_mean"]
    aggregated_metrics["micro_f1_std"] = multi_seed_summary["micro_f1_std"]

    model_id = "tfidf_logreg_baseline"
    model_title = f"Base Model 4: TF-IDF + Logistic Regression (Seeds {SEEDS})"
    out_dir = save_reproducible_run(
        model_id=model_id,
        model_title=model_title,
        category="base_models",
        predictions=primary_predictions,
        metrics=aggregated_metrics,
        runtime_seconds=runtime_seconds,
        seed=SEEDS,
        config={
            "classifier": "OneVsRestClassifier(LogisticRegression(C=2.0, class_weight='balanced'))",
            "features": "Word TF-IDF (1-2 grams) + Char-WB TF-IDF (4-5 grams)",
            "field_thresholds": FIELD_THRESHOLDS,
            "always_emit_top1_fields": sorted(ALWAYS_EMIT_TOP1_FIELDS),
            "num_train_papers": len(train_ids),
            "num_dev_papers": len(dev_100_ids),
        },
        extra_artifacts={f"{model_id}_multi_seed_stats.json": multi_seed_summary},
    )

    print_report(model_title, aggregated_metrics)
    print(f"Reproducible run artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
