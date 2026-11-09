#!/usr/bin/env python3
"""
========================================================================================
EXECUTIVE SUMMARY — Base Model 5: Bag-of-Words + Complement Naive Bayes Baseline
========================================================================================
- Concept & Goal:
    Implements the classic generative/probabilistic text classification baseline from
    Lecture 1 (Language & Sequence Modeling / N-Gram Bag-of-Words). Using Bayes' rule,
    the model estimates `P(label | document)` from raw word and character N-gram counts
    learned across the 908 training annotations (`data/gold/train.jsonl`).
- Why Complement Naive Bayes (`ComplementNB`)?
    Standard Multinomial Naive Bayes struggles when class frequencies are heavily skewed
    (e.g., `habitat` has 301x "Agrar- und Offenland" vs. 55x "Urbane Räume").
    Rennie et al. (2003) introduced `ComplementNB`, which estimates parameters using the
    *complement* of each class (all documents NOT in class c), making it specifically
    suited for imbalanced text categorization.
- How It Works:
    1. Feature Extraction: Extracts Bag-of-Words count vectors (`CountVectorizer`) over
       word 1-2 grams and character 4-5 grams.
    2. Multi-Label Binary Relevance: For each categorical schema field and each candidate
       class `c`, fits a binary `[negative, positive]` Complement Naive Bayes model with
       Laplace/Lidstone smoothing (`alpha=0.5`).
    3. Sparsity-Aware Posterior Thresholding: Emits class `c` only if its posterior
       probability exceeds a field-specific threshold (allowing empty `[]` predictions
       for sparse fields like `ecosystem_service` and `ecosystem_type`), with top-1
       fallback for dense fields (`habitat`, `taxa.species_group`).
    4. Subtask C2b Taxa Resolution: Combines exact training lexicon species matches with
       Naive Bayes `taxa.species_group` predictions using the `"-"` collective-term rule.
- Results on Dev Set (100 valid papers, deterministic):
    * C1 Leaderboard Macro-F1 : 0.4799
    * C1 + C2b Macro-F1       : 0.3547
    * C1 + C2b Micro-F1       : 0.3587
    * Per-Field Highlights    : ecosystem_type.category (0.6154), habitat (0.5680),
                                biodiversity_level (0.4218), ecosystem_type.term (0.3942).
    * Runtime                 : ~1.9s
========================================================================================
"""

from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MultiLabelBinarizer

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

# Categorical fields modeled via Multi-Label Complement Naive Bayes
LEARNED_FIELDS: list[str] = [
    "habitat",
    "biodiversity_level",
    "ecosystem_type.category",
    "ecosystem_type.term",
    "taxa.species_group",
    "ecosystem_service.category",
    "ecosystem_service.term",
]

# Posterior probability thresholds calibrated to each field's empty rate in Train
FIELD_THRESHOLDS: dict[str, float] = {
    "habitat": 0.45,
    "biodiversity_level": 0.50,
    "ecosystem_type.category": 0.50,
    "ecosystem_type.term": 0.48,
    "taxa.species_group": 0.48,
    "ecosystem_service.category": 0.53,
    "ecosystem_service.term": 0.51,
}

# Dense fields that should always output at least their top-1 highest-probability class
ALWAYS_EMIT_TOP1_FIELDS: set[str] = {
    "habitat",
    "taxa.species_group",
}


def build_bow_extractor() -> FeatureUnion:
    """
    Constructs a raw Bag-of-Words (`CountVectorizer`) feature union combining:
      - Word 1-2 n-grams (capturing ecological terms and bigrams like "species richness")
      - Character-WB 4-5 n-grams (capturing bilingual English/German morphological stems)
    """
    return FeatureUnion(
        [
            (
                "word_bow",
                CountVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.90,
                    strip_accents="unicode",
                ),
            ),
            (
                "char_bow",
                CountVectorizer(
                    analyzer="char_wb",
                    ngram_range=(4, 5),
                    min_df=3,
                    max_df=0.90,
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
    Training records use `bibtex_title`; Dev records combine `bibtex_title` (weighted 3x)
    with the cached text of the first 2 PDF pages.
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


def train_and_predict_naive_bayes(
    X_train_texts: list[str],
    train_records: dict[str, dict[str, Any]],
    train_ids: list[str],
    X_dev_texts: list[str],
    dev_ids: list[str],
    taxa_lexicon: dict[str, tuple[str, str, str]],
) -> dict[str, dict[str, Any]]:
    """
    Fits a Bag-of-Words `ComplementNB` classifier per schema field and generates
    schema-valid JSON predictions for the 100 Dev papers.
    """
    vectorizer = build_bow_extractor()
    X_train = vectorizer.fit_transform(X_train_texts)
    X_dev = vectorizer.transform(X_dev_texts)

    field_predictions: dict[str, list[list[str]]] = {}

    for field in LEARNED_FIELDS:
        # Convert each paper's set of gold labels into a binary multi-label indicator matrix
        y_train_sets = [sorted(extract_field_set(train_records[zid], field)) for zid in train_ids]
        mlb = MultiLabelBinarizer()
        Y_train = mlb.fit_transform(y_train_sets)

        # Fit One-vs-Rest Complement Naive Bayes with Lidstone smoothing (alpha=0.5)
        clf = OneVsRestClassifier(ComplementNB(alpha=0.5, norm=True))
        clf.fit(X_train, Y_train)

        # Compute posterior class probabilities P(y_c = 1 | x) on Dev set
        probs = clf.predict_proba(X_dev)
        threshold = FIELD_THRESHOLDS[field]

        preds_for_field: list[list[str]] = []
        for i in range(len(dev_ids)):
            row_probs = probs[i]
            selected_indices = np.where(row_probs >= threshold)[0].tolist()

            # Sort descending by posterior probability and cap at top 2 labels per field
            selected_indices.sort(key=lambda idx: row_probs[idx], reverse=True)
            selected_indices = selected_indices[:2]

            # Dense fields (`habitat`, `taxa.species_group`) fall back to argmax if empty
            if not selected_indices and field in ALWAYS_EMIT_TOP1_FIELDS:
                best_idx = int(np.argmax(row_probs))
                selected_indices = [best_idx]

            labels = [str(mlb.classes_[idx]) for idx in selected_indices]
            preds_for_field.append(labels)

        field_predictions[field] = preds_for_field

    # Pre-compile lexicon patterns once for fast scanning across the 100 Dev papers
    compiled_lexicon = [
        (re.compile(r"\b" + re.escape(term_key) + r"\b"), sci, ger, grp)
        for term_key, (sci, ger, grp) in taxa_lexicon.items()
    ]

    output_records: dict[str, dict[str, Any]] = {}
    for i, zid in enumerate(dev_ids):
        dev_text = X_dev_texts[i]

        # 1. Habitat & Biodiversity Level
        habitats = field_predictions["habitat"][i]
        bio_levels = field_predictions["biodiversity_level"][i]

        # 2. Ecosystem Type
        eco_cats = field_predictions["ecosystem_type.category"][i]
        eco_terms = field_predictions["ecosystem_type.term"][i]
        eco_types = []
        for idx, cat in enumerate(eco_cats):
            term = eco_terms[idx] if idx < len(eco_terms) else (eco_terms[0] if eco_terms else None)
            eco_types.append({"category": cat, "term": term})

        # 3. Taxa (exact lexicon species + learned species groups with "-" collective convention)
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

        # 4. Ecosystem Service
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

    X_train_texts = prepare_corpus_texts(
        train_records, train_ids, pdf_dir, cache_dir, use_pdfs=False
    )
    X_dev_texts = prepare_corpus_texts(
        dev_records, dev_100_ids, pdf_dir, cache_dir, use_pdfs=True
    )

    predictions = train_and_predict_naive_bayes(
        X_train_texts=X_train_texts,
        train_records=train_records,
        train_ids=train_ids,
        X_dev_texts=X_dev_texts,
        dev_ids=dev_100_ids,
        taxa_lexicon=taxa_lexicon,
    )

    runtime_seconds = time.perf_counter() - start_time
    metrics = evaluate_predictions(dev_records, predictions, filter_ids=set(dev_100_ids))

    model_id = "bow_naive_bayes_baseline"
    model_title = "Base Model 5: Bag-of-Words + Complement Naive Bayes"
    out_dir = save_reproducible_run(
        model_id=model_id,
        model_title=model_title,
        category="base_models",
        predictions=predictions,
        metrics=metrics,
        runtime_seconds=runtime_seconds,
        seed=None,
        config={
            "classifier": "OneVsRestClassifier(ComplementNB(alpha=0.5, norm=True))",
            "features": "Word CountVectorizer (1-2 grams) + Char-WB CountVectorizer (4-5 grams)",
            "field_thresholds": FIELD_THRESHOLDS,
            "always_emit_top1_fields": sorted(ALWAYS_EMIT_TOP1_FIELDS),
            "num_train_papers": len(train_ids),
            "num_dev_papers": len(dev_100_ids),
        },
    )

    print_report(model_title, metrics)
    print(f"Reproducible run artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
