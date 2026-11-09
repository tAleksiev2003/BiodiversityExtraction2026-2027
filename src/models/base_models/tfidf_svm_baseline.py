#!/usr/bin/env python3
"""
========================================================================================
EXECUTIVE SUMMARY — Base Model 6: TF-IDF + Linear Support Vector Machine (`LinearSVC`)
========================================================================================
- Concept & Goal:
    Implements the classic maximum-margin linear classifier for high-dimensional sparse
    text vectors: TF-IDF N-grams paired with a One-vs-Rest Linear Support Vector Machine
    (`LinearSVC`). In classical NLP literature (Joachims, 1998), Linear SVMs are considered
    the gold-standard non-neural text classifier because margin maximization generalizes
    exceptionally well across tens of thousands of correlated word/character N-gram features.
- How It Works:
    1. Feature Extraction: Uses the exact same combined Word (1-2 gram) + Character-WB
       (4-5 gram) sublinear TF-IDF representation as `tfidf_logreg_baseline.py` for a
       strictly fair comparison between Logistic Regression (log-loss) and SVM (hinge-loss).
    2. Maximum-Margin Training: Trains a One-vs-Rest `LinearSVC(C=1.0, class_weight='balanced')`
       across 3 fixed seeds (`[42, 43, 44]`) on the 908 training papers (`train.jsonl`).
    3. Sigmoid Margin Calibration & Sparsity Thresholding: Converts raw signed hyperplane
       distances `d(x) = w^T x + b` into `[0, 1]` confidence scores via the logistic sigmoid
       `1 / (1 + exp(-d(x)))`. Field-specific thresholds allow the model to predict empty `[]`
       on sparse fields (`ecosystem_service`, `ecosystem_type`) while using a top-1 fallback
       on dense fields (`habitat`, `taxa.species_group`).
    4. Subtask C2b Taxa Resolution: Combines pre-compiled training lexicon species matches
       with SVM-predicted `taxa.species_group` labels (using `scientific_name = "-"` for
       collective groups).
- Results on Dev Set (100 valid papers, Seeds `[42, 43, 44]`):
    * C1 Leaderboard Macro-F1 : 0.4897 ± 0.0000
    * C1 + C2b Macro-F1       : 0.3675 ± 0.0000
    * C1 + C2b Micro-F1       : 0.3753 ± 0.0000
    * Per-Field Highlights    : habitat (0.6189), ecosystem_type.category (0.5526),
                                taxa.species_group (0.4459), ecosystem_type.term (0.4262),
                                ecosystem_service.category (0.4211).
    * Runtime                 : ~9.8s
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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

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

# Fixed random seeds for multi-run evaluation of the dual coordinate descent solver
SEEDS: list[int] = [42, 43, 44]

# Categorical fields modeled via Multi-Label LinearSVC
LEARNED_FIELDS: list[str] = [
    "habitat",
    "biodiversity_level",
    "ecosystem_type.category",
    "ecosystem_type.term",
    "taxa.species_group",
    "ecosystem_service.category",
    "ecosystem_service.term",
]

# Sigmoid-transformed SVM margin thresholds (`0.50` corresponds to hyperplane boundary `d(x) = 0`)
FIELD_THRESHOLDS: dict[str, float] = {
    "habitat": 0.38,
    "biodiversity_level": 0.42,
    "ecosystem_type.category": 0.42,
    "ecosystem_type.term": 0.38,
    "taxa.species_group": 0.38,
    "ecosystem_service.category": 0.50,
    "ecosystem_service.term": 0.48,
}

# Dense fields that always emit at least their top-1 highest-margin class
ALWAYS_EMIT_TOP1_FIELDS: set[str] = {
    "habitat",
    "taxa.species_group",
}


def build_feature_extractor() -> FeatureUnion:
    """
    Constructs the combined Word (1-2 gram) and Character-WB (4-5 gram) TF-IDF vectorizer.
    Identical to `tfidf_logreg_baseline.py` to isolate the effect of the Linear SVM classifier.
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


def sigmoid(margins: np.ndarray) -> np.ndarray:
    """Maps signed SVM hyperplane distances `(-inf, +inf)` into `[0, 1]` via logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(margins, -20.0, 20.0)))


def train_and_predict_for_seed(
    seed: int,
    X_train: Any,
    X_dev: Any,
    X_dev_texts: list[str],
    train_records: dict[str, dict[str, Any]],
    train_ids: list[str],
    dev_ids: list[str],
    compiled_lexicon: list[tuple[re.Pattern[str], str, str, str]],
) -> dict[str, dict[str, Any]]:
    """
    Trains per-field One-vs-Rest `LinearSVC` classifiers for a given seed and outputs
    schema-compliant predictions for the 100 Dev papers.
    """
    field_predictions: dict[str, list[list[str]]] = {}

    for field in LEARNED_FIELDS:
        y_train_sets = [sorted(extract_field_set(train_records[zid], field)) for zid in train_ids]
        mlb = MultiLabelBinarizer()
        Y_train = mlb.fit_transform(y_train_sets)

        # Fit balanced Linear Support Vector Machine (L2 regularization, squared hinge loss)
        clf = OneVsRestClassifier(
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
            )
        )
        clf.fit(X_train, Y_train)

        # Convert signed hyperplane margins to [0, 1] confidence scores
        raw_margins = clf.decision_function(X_dev)
        probs = sigmoid(raw_margins)
        threshold = FIELD_THRESHOLDS[field]

        preds_for_field: list[list[str]] = []
        for i in range(len(dev_ids)):
            row_probs = probs[i]
            selected_indices = np.where(row_probs >= threshold)[0].tolist()

            # Sort descending by margin confidence and cap at top 2 labels per field
            selected_indices.sort(key=lambda idx: row_probs[idx], reverse=True)
            selected_indices = selected_indices[:2]

            # Fallback for dense fields (`habitat`, `taxa.species_group`)
            if not selected_indices and field in ALWAYS_EMIT_TOP1_FIELDS:
                best_idx = int(np.argmax(row_probs))
                selected_indices = [best_idx]

            labels = [str(mlb.classes_[idx]) for idx in selected_indices]
            preds_for_field.append(labels)

        field_predictions[field] = preds_for_field

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

        # 3. Taxa (exact lexicon species + SVM-predicted species_group with "-" convention)
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

    # Extract TF-IDF feature matrices once across seeds
    vectorizer = build_feature_extractor()
    X_train = vectorizer.fit_transform(X_train_texts)
    X_dev = vectorizer.transform(X_dev_texts)

    seed_metrics: dict[int, dict[str, Any]] = {}
    primary_predictions: dict[str, dict[str, Any]] = {}

    for seed in SEEDS:
        preds = train_and_predict_for_seed(
            seed=seed,
            X_train=X_train,
            X_dev=X_dev,
            X_dev_texts=X_dev_texts,
            train_records=train_records,
            train_ids=train_ids,
            dev_ids=dev_100_ids,
            compiled_lexicon=compiled_lexicon,
        )
        seed_metrics[seed] = evaluate_predictions(
            dev_records, preds, filter_ids=set(dev_100_ids)
        )
        if seed == SEEDS[0]:
            primary_predictions = preds

    runtime_seconds = time.perf_counter() - start_time

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
        "field_thresholds": FIELD_THRESHOLDS,
    }

    aggregated_metrics = dict(seed_metrics[SEEDS[0]])
    aggregated_metrics["c1_macro_f1"] = multi_seed_summary["c1_macro_f1_mean"]
    aggregated_metrics["c1_macro_f1_std"] = multi_seed_summary["c1_macro_f1_std"]
    aggregated_metrics["c2b_macro_f1"] = multi_seed_summary["c2b_macro_f1_mean"]
    aggregated_metrics["c2b_macro_f1_std"] = multi_seed_summary["c2b_macro_f1_std"]
    aggregated_metrics["micro_f1"] = multi_seed_summary["micro_f1_mean"]
    aggregated_metrics["micro_f1_std"] = multi_seed_summary["micro_f1_std"]

    model_id = "tfidf_svm_baseline"
    model_title = f"Base Model 6: TF-IDF + Linear SVM (Seeds {SEEDS})"
    out_dir = save_reproducible_run(
        model_id=model_id,
        model_title=model_title,
        category="base_models",
        predictions=primary_predictions,
        metrics=aggregated_metrics,
        runtime_seconds=runtime_seconds,
        seed=SEEDS,
        config={
            "classifier": "OneVsRestClassifier(LinearSVC(C=1.0, class_weight='balanced'))",
            "features": "Word TF-IDF (1-2 grams) + Char-WB TF-IDF (4-5 grams)",
            "margin_calibration": "logistic_sigmoid_on_decision_function",
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
