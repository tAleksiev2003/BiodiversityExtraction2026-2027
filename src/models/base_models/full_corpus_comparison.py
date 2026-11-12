#!/usr/bin/env python3
"""
========================================================================================
EXECUTIVE SUMMARY — Full-Corpus Base-Model Comparison (Dev-100 vs Train-908)
========================================================================================
- Concept & Goal:
    Cross-model analysis requested for the server benchmarking phase: run ALL base
    models (1-9) sequentially over the FULL corpus — the official Dev-100 split AND
    the complete Train-908 split (now possible because all 908 train PDFs are
    available on the server) — to provide a single comparison table.
    METHODOLOGICAL CAVEAT (printed on every artifact): the supervised base models
    (4-8) are FITTED on the 908 training papers, so their Train-split scores are
    IN-SAMPLE (memorization), not generalization. The Train column therefore
    measures training-set fit; only the Dev column is an honest generalization
    estimate. Baselines 1-3 (majority/random/dictionary-lexicon) are structurally
    identical on both splits except for the dictionary's lexicon overlap.

- How It Works:
    1. Reuses each verified base-model script's OWN fit/predict functions via
       imports (no copies, no edits to the standalone scripts): every model's
       public function takes generic eval arguments, so we call it once with the
       Dev-100 papers (dev PDFs, as in the official runs) and once with the
       Train-908 papers (train PDFs, freshly extracted + cached under
       data/parsed_text/train/).
    2. Fitting always uses the same representation as the official runs
       (metadata-only train texts, use_pdfs=False); only the EVALUATION texts
       differ per split (PDF-based, mirroring how each main() evaluates Dev).
    3. Seed protocol per model follows its own script (3 seeds where stochastic;
       mean +/- std is reported, primary seed = first seed).
    4. The approval-voting ensemble (Model 9) is computed from the four
       supervised voters' IN-MEMORY predictions per split (threshold k=3), so it
       does not depend on previously saved artifacts.
    5. Crash-safe checkpointing: results are written per model into
       outputs/base_models/full_corpus_comparison/results_<model>.json; a rerun
       skips finished models. `--limit N` truncates both splits for smoke tests.
    This script deliberately does NOT call save_reproducible_run(): train-split
    numbers must not pollute the official per-model leaderboard artifacts.

- Results on Full Corpus (filled after the server run):
    * PENDING — execute with:
        python3 src/models/base_models/full_corpus_comparison.py
      (then update this block with the comparison table)
========================================================================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluator import (
    evaluate_predictions,
    extract_field_set,
    load_jsonl_by_id,
)

# --- per-model imports: each model's own verified functions, nothing reimplemented ---
from src.models.base_models.majority_baseline import (
    fit_majority_model,
    predict_single_paper as majority_predict,
)
from src.models.base_models.random_baseline import (
    SEEDS as RANDOM_SEEDS,
    load_allowed_enums,
    predict_single_paper_random,
)
from src.models.base_models.dictionary_baseline import (
    build_train_taxa_lexicon,
    extract_and_cache_paper_text,
    predict_single_paper_dictionary,
)
from src.models.base_models.bow_naive_bayes_baseline import (
    prepare_corpus_texts as bow_prepare_texts,
    train_and_predict_naive_bayes,
)
from src.models.base_models.tfidf_logreg_baseline import (
    SEEDS as LOGREG_SEEDS,
    prepare_corpus_texts as logreg_prepare_texts,
    train_and_predict_for_seed as logreg_predict_seed,
)
from src.models.base_models.tfidf_svm_baseline import (
    SEEDS as SVM_SEEDS,
    build_feature_extractor as svm_build_features,
    prepare_corpus_texts as svm_prepare_texts,
    train_and_predict_for_seed as svm_predict_seed,
)
from src.models.base_models.knn_retrieval_baseline import (
    TOP_K_NEIGHBORS,
    prepare_corpus_texts as knn_prepare_texts,
    predict_knn_retrieval,
)
from src.models.base_models.crf_baseline import (
    SEEDS as CRF_SEEDS,
    fit_linear_chain_crf,
    predict_single_paper_crf,
)
from src.models.base_models.ensemble_voting_baseline import (
    VOTE_THRESHOLD,
    ensemble_single_paper,
)

MODEL_ID = "full_corpus_comparison"
OUT_DIR = REPO_ROOT / "outputs" / "base_models" / MODEL_ID

# The four supervised voters of the approval ensemble (Model 9), in VOTER_IDS order.
VOTER_KEYS = ["tfidf_svm", "bow_naive_bayes", "tfidf_logreg", "knn_retrieval"]


# --------------------------------------------------------------------------------------
# Split plumbing: one split = ids + gold records + PDF dir + text cache dir
# --------------------------------------------------------------------------------------

def load_splits(limit: int | None) -> dict[str, dict[str, Any]]:
    """Builds the two evaluation splits (dev-100 official, train-908 full corpus)."""
    train_records = load_jsonl_by_id(REPO_ROOT / "data" / "gold" / "train.jsonl")
    dev_records = load_jsonl_by_id(REPO_ROOT / "data" / "gold" / "dev.jsonl")
    dev_ids = sorted(
        {ln.strip() for ln in (REPO_ROOT / "data" / "gold" / "dev_100_ids.txt").read_text().splitlines() if ln.strip()}
    )
    train_ids = sorted(train_records.keys())
    if limit:  # smoke-test mode: truncate only the EVALUATION id lists; the
        # supervised fits always see the full 908 papers (as in the official mains)
        dev_ids = dev_ids[:limit]
        train_ids = train_ids[:limit]
    return {
        "dev": {
            "ids": dev_ids,
            "gold": dev_records,
            "pdf_dir": REPO_ROOT / "data" / "raw_pdfs" / "dev",
            "cache_dir": REPO_ROOT / "data" / "parsed_text" / "dev",
            "in_sample": False,
        },
        "train": {
            "ids": train_ids,
            "gold": train_records,
            "pdf_dir": REPO_ROOT / "data" / "raw_pdfs" / "train",
            "cache_dir": REPO_ROOT / "data" / "parsed_text" / "train",
            "in_sample": True,  # supervised models were fitted on these papers
        },
    }


def summarize_seeds(seed_metrics: dict[int, dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    """Mean +/- std of the headline metrics across seeds (single-run models pass one seed)."""
    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    def std(xs: list[float]) -> float:
        return (sum((x - mean(xs)) ** 2 for x in xs) / len(xs)) ** 0.5 if len(xs) > 1 else 0.0

    c1 = [seed_metrics[s]["c1_macro_f1"] for s in seeds]
    c2b = [seed_metrics[s]["c2b_macro_f1"] for s in seeds]
    micro = [seed_metrics[s]["micro_f1"] for s in seeds]
    return {
        "c1_macro_f1_mean": round(mean(c1), 4),
        "c1_macro_f1_std": round(std(c1), 4),
        "c2b_macro_f1_mean": round(mean(c2b), 4),
        "micro_f1_mean": round(mean(micro), 4),
        "per_seed": {str(s): {k: v for k, v in seed_metrics[s].items() if k.endswith(("f1", "micro_f1"))} for s in seeds},
    }


# --------------------------------------------------------------------------------------
# Per-model runners: each mirrors its script's main() wiring exactly, evaluated on
# both splits. All return {split: {seed: predictions}} plus per-split metric summaries.
# --------------------------------------------------------------------------------------

def run_majority(splits: dict, shared: dict) -> dict[str, Any]:
    """Model 1 — training-set majority values, split-independent fit."""
    learned = fit_majority_model(shared["train_records"])
    out: dict[str, Any] = {"results": {}, "predictions": {}}
    for split, cfg in splits.items():
        preds = {zid: majority_predict(zid, learned) for zid in cfg["ids"]}
        out["predictions"][split] = preds
        out["results"][split] = summarize_seeds({42: evaluate_predictions(cfg["gold"], preds, filter_ids=set(cfg["ids"]))}, [42])
    return out


def run_random(splits: dict, shared: dict) -> dict[str, Any]:
    """Model 2 — stratified random sampling from allowed enums + train name pools."""
    enums = load_allowed_enums(REPO_ROOT / "schemas")
    sci = sorted({v for r in shared["train_records"].values() for v in extract_field_set(r, "taxa.scientific_name")})
    ger = sorted({v for r in shared["train_records"].values() for v in extract_field_set(r, "taxa.german_name")})
    out: dict[str, Any] = {"results": {}, "predictions": {}}
    for split, cfg in splits.items():
        seed_preds: dict[int, dict] = {}
        for seed in RANDOM_SEEDS:
            rng = random.Random(seed)
            seed_preds[seed] = {
                zid: predict_single_paper_random(zid, enums, sci, ger, rng) for zid in cfg["ids"]
            }
        out["predictions"][split] = seed_preds[RANDOM_SEEDS[0]]
        out["results"][split] = summarize_seeds(
            {s: evaluate_predictions(cfg["gold"], seed_preds[s], filter_ids=set(cfg["ids"])) for s in RANDOM_SEEDS},
            RANDOM_SEEDS,
        )
    return out


def run_dictionary(splits: dict, shared: dict) -> dict[str, Any]:
    """Model 3 — lexicon lookup over each paper's first-2-pages text (PDF-backed)."""
    lexicon = build_train_taxa_lexicon(shared["train_records"])
    learned = fit_majority_model(shared["train_records"])
    out: dict[str, Any] = {"results": {}, "predictions": {}}
    for split, cfg in splits.items():
        preds, pdfs_used = {}, 0
        for zid in cfg["ids"]:
            text, used_pdf = extract_and_cache_paper_text(zid, cfg["gold"][zid], cfg["pdf_dir"], cfg["cache_dir"], max_pages=2)
            pdfs_used += int(used_pdf)
            preds[zid] = predict_single_paper_dictionary(zid, text, lexicon, learned)
        out["predictions"][split] = preds
        res = summarize_seeds({42: evaluate_predictions(cfg["gold"], preds, filter_ids=set(cfg["ids"]))}, [42])
        res["pdfs_used"] = f"{pdfs_used}/{len(cfg['ids'])}"
        out["results"][split] = res
    return out


def run_bow_naive_bayes(splits: dict, shared: dict) -> dict[str, Any]:
    """Model 5 — Complement NB over bag-of-words; fit on metadata texts (official setup)."""
    tr = shared["train_records"]
    train_ids = shared["train_ids"]
    X_train_texts = bow_prepare_texts(tr, train_ids, splits["dev"]["pdf_dir"], splits["dev"]["cache_dir"], use_pdfs=False)
    lexicon = build_train_taxa_lexicon(tr)
    out: dict[str, Any] = {"results": {}, "predictions": {}}
    for split, cfg in splits.items():
        X_eval = bow_prepare_texts(cfg["gold"], cfg["ids"], cfg["pdf_dir"], cfg["cache_dir"], use_pdfs=True)
        preds = train_and_predict_naive_bayes(
            X_train_texts=X_train_texts, train_records=tr, train_ids=train_ids,
            X_dev_texts=X_eval, dev_ids=cfg["ids"], taxa_lexicon=lexicon,
        )
        out["predictions"][split] = preds
        out["results"][split] = summarize_seeds({42: evaluate_predictions(cfg["gold"], preds, filter_ids=set(cfg["ids"]))}, [42])
    return out


def run_tfidf_logreg(splits: dict, shared: dict) -> dict[str, Any]:
    """Model 6? (tfidf_logreg) — TF-IDF + per-field One-vs-Rest LogReg, 3 seeds."""
    tr = shared["train_records"]
    train_ids = shared["train_ids"]
    X_train_texts = logreg_prepare_texts(tr, train_ids, splits["dev"]["pdf_dir"], splits["dev"]["cache_dir"], use_pdfs=False)
    lexicon = build_train_taxa_lexicon(tr)
    out: dict[str, Any] = {"results": {}, "predictions": {}}
    for split, cfg in splits.items():
        X_eval = logreg_prepare_texts(cfg["gold"], cfg["ids"], cfg["pdf_dir"], cfg["cache_dir"], use_pdfs=True)
        seed_preds = {}
        for seed in LOGREG_SEEDS:
            seed_preds[seed] = logreg_predict_seed(
                seed=seed, X_train_texts=X_train_texts, train_records=tr, train_ids=train_ids,
                X_dev_texts=X_eval, dev_ids=cfg["ids"], taxa_lexicon=lexicon,
            )
        out["predictions"][split] = seed_preds[LOGREG_SEEDS[0]]
        out["results"][split] = summarize_seeds(
            {s: evaluate_predictions(cfg["gold"], seed_preds[s], filter_ids=set(cfg["ids"])) for s in LOGREG_SEEDS},
            LOGREG_SEEDS,
        )
    return out


def run_tfidf_svm(splits: dict, shared: dict) -> dict[str, Any]:
    """Model 6 — TF-IDF + linear SVM, 3 seeds; vectorizer fit once on metadata texts."""
    tr = shared["train_records"]
    train_ids = shared["train_ids"]
    X_train_texts = svm_prepare_texts(tr, train_ids, splits["dev"]["pdf_dir"], splits["dev"]["cache_dir"], use_pdfs=False)
    lexicon = build_train_taxa_lexicon(tr)
    compiled = [(re.compile(r"\b" + re.escape(k) + r"\b"), sci, ger, grp) for k, (sci, ger, grp) in lexicon.items()]
    vectorizer = svm_build_features()
    X_train = vectorizer.fit_transform(X_train_texts)
    out: dict[str, Any] = {"results": {}, "predictions": {}}
    for split, cfg in splits.items():
        X_eval_texts = svm_prepare_texts(cfg["gold"], cfg["ids"], cfg["pdf_dir"], cfg["cache_dir"], use_pdfs=True)
        X_eval = vectorizer.transform(X_eval_texts)
        seed_preds = {}
        for seed in SVM_SEEDS:
            seed_preds[seed] = svm_predict_seed(
                seed=seed, X_train=X_train, X_dev=X_eval, X_dev_texts=X_eval_texts,
                train_records=tr, train_ids=train_ids, dev_ids=cfg["ids"], compiled_lexicon=compiled,
            )
        out["predictions"][split] = seed_preds[SVM_SEEDS[0]]
        out["results"][split] = summarize_seeds(
            {s: evaluate_predictions(cfg["gold"], seed_preds[s], filter_ids=set(cfg["ids"])) for s in SVM_SEEDS},
            SVM_SEEDS,
        )
    return out


def run_knn(splits: dict, shared: dict) -> dict[str, Any]:
    """Model 7 — k-NN label retrieval (k=5) over TF-IDF char+word space."""
    tr = shared["train_records"]
    train_ids = shared["train_ids"]
    X_train_texts = knn_prepare_texts(tr, train_ids, splits["dev"]["pdf_dir"], splits["dev"]["cache_dir"], use_pdfs=False)
    lexicon = build_train_taxa_lexicon(tr)
    compiled = [(re.compile(r"\b" + re.escape(k) + r"\b"), sci, ger, grp) for k, (sci, ger, grp) in lexicon.items()]
    out: dict[str, Any] = {"results": {}, "predictions": {}}
    for split, cfg in splits.items():
        X_eval = knn_prepare_texts(cfg["gold"], cfg["ids"], cfg["pdf_dir"], cfg["cache_dir"], use_pdfs=True)
        preds, _diag = predict_knn_retrieval(
            X_train_texts=X_train_texts, train_records=tr, train_ids=train_ids,
            X_dev_texts=X_eval, dev_ids=cfg["ids"], compiled_lexicon=compiled, k=TOP_K_NEIGHBORS,
        )
        out["predictions"][split] = preds
        out["results"][split] = summarize_seeds({42: evaluate_predictions(cfg["gold"], preds, filter_ids=set(cfg["ids"]))}, [42])
    return out


def run_crf(splits: dict, shared: dict) -> dict[str, Any]:
    """Model 8 — linear-chain CRF over weak BIO tags, 3 seeds; eval texts = title + 2 PDF pages."""
    tr = shared["train_records"]
    train_ids = shared["train_ids"]
    learned = fit_majority_model(tr)
    lexicon = build_train_taxa_lexicon(tr)
    compiled = [(re.compile(r"\b" + re.escape(k) + r"\b"), sci, ger, grp) for k, (sci, ger, grp) in lexicon.items()]
    out: dict[str, Any] = {"results": {}, "predictions": {}}
    # Raw eval texts per split (title + first-2-pages PDF text), cached per split dir.
    raw_texts: dict[str, dict[str, str]] = {}
    for split, cfg in splits.items():
        raw_texts[split] = {}
        for zid in cfg["ids"]:
            rec = cfg["gold"][zid]
            text, _ = extract_and_cache_paper_text(zid, rec, cfg["pdf_dir"], cfg["cache_dir"], max_pages=2)
            raw_texts[split][zid] = f"{str(rec.get('bibtex_title') or '')}\n{text}"
    # The CRF fit is split-independent: fit once per seed, then predict BOTH splits.
    seed_preds_by_split: dict[str, dict[int, dict]] = {s: {} for s in splits}
    for seed in CRF_SEEDS:
        vec, emission_clf, classes, trans = fit_linear_chain_crf(tr, train_ids, seed=seed)
        for split, cfg in splits.items():
            seed_preds_by_split[split][seed] = {
                zid: predict_single_paper_crf(zid, raw_texts[split][zid], vec, emission_clf, classes, trans, compiled, learned)
                for zid in cfg["ids"]
            }
    for split, cfg in splits.items():
        out["predictions"][split] = seed_preds_by_split[split][CRF_SEEDS[0]]
        out["results"][split] = summarize_seeds(
            {s: evaluate_predictions(cfg["gold"], seed_preds_by_split[split][s], filter_ids=set(cfg["ids"])) for s in CRF_SEEDS},
            CRF_SEEDS,
        )
    return out


def run_ensemble(splits: dict, shared: dict, voter_outputs: dict[str, dict]) -> dict[str, Any]:
    """Model 9 — approval voting (k=3) over the four supervised voters' in-memory preds."""
    out: dict[str, Any] = {"results": {}, "predictions": {}}
    for split, cfg in splits.items():
        voter_preds = [voter_outputs[k]["predictions"][split] for k in VOTER_KEYS]
        preds = {}
        for zid in cfg["ids"]:
            pred = ensemble_single_paper([v[zid] for v in voter_preds])
            pred["zotitem_ptr_id"] = zid
            preds[zid] = pred
        out["predictions"][split] = preds
        out["results"][split] = summarize_seeds({42: evaluate_predictions(cfg["gold"], preds, filter_ids=set(cfg["ids"]))}, [42])
    return out


# --------------------------------------------------------------------------------------
# Orchestration: sequential execution, per-model checkpoint files, final comparison table
# --------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all 9 base models over the full corpus (Dev-100 + Train-908)."
    )
    parser.add_argument("--limit", type=int, default=None, help="Smoke test: first N papers per split")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated model keys to (re)run")
    args = parser.parse_args()

    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = load_splits(args.limit)
    # Fits always use the FULL training corpus (all keys of train.jsonl); only the
    # evaluation id lists may be limited by --limit.
    shared = {"train_records": splits["train"]["gold"], "train_ids": sorted(splits["train"]["gold"].keys())}

    # Model registry in official numbering order; ensemble runs last from voter outputs.
    registry: dict[str, Callable] = {
        "majority": run_majority,
        "random": run_random,
        "dictionary": run_dictionary,
        "bow_naive_bayes": run_bow_naive_bayes,
        "tfidf_logreg": run_tfidf_logreg,
        "tfidf_svm": run_tfidf_svm,
        "knn_retrieval": run_knn,
        "crf": run_crf,
    }
    only = set(args.only.split(",")) if args.only else None

    voter_outputs: dict[str, dict] = {}
    for key, runner in registry.items():
        ckpt = OUT_DIR / f"results_{key}.json"
        if ckpt.exists() and not (only and key in only):
            print(f"[skip] {key} (checkpoint exists)", flush=True)
            data = json.loads(ckpt.read_text())
        else:
            t0 = time.time()
            print(f"[run ] {key} ...", flush=True)
            data = runner(splits, shared)
            data["runtime_seconds"] = round(time.time() - t0, 1)
            data["n_ids"] = {s: len(c["ids"]) for s, c in splits.items()}
            data["limit"] = args.limit
            # predictions stored separately (per split, primary seed) to keep JSON light
            for split in splits:
                pfile = OUT_DIR / f"predictions_{key}_{split}.jsonl"
                with pfile.open("w", encoding="utf-8") as fh:
                    for zid, pred in data["predictions"][split].items():
                        fh.write(json.dumps(pred, ensure_ascii=False) + "\n")
            data.pop("predictions")
            ckpt.write_text(json.dumps(data, ensure_ascii=False, indent=1))
            print(
                f"[done] {key} in {data['runtime_seconds']}s | "
                f"dev C1={data['results']['dev']['c1_macro_f1_mean']:.4f} "
                f"train C1={data['results']['train']['c1_macro_f1_mean']:.4f}",
                flush=True,
            )
        if key in VOTER_KEYS:
            # reload primary-seed predictions from the jsonl files for the ensemble
            voter_outputs[key] = {"predictions": {}}
            for split in splits:
                preds: dict[str, dict] = {}
                for line in (OUT_DIR / f"predictions_{key}_{split}.jsonl").read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        obj = json.loads(line)
                        preds[obj["zotitem_ptr_id"]] = obj
                voter_outputs[key]["predictions"][split] = preds

    # Ensemble last (needs all four voters on both splits)
    key = "ensemble_voting"
    ckpt = OUT_DIR / f"results_{key}.json"
    if ckpt.exists() and not (only and key in only):
        print(f"[skip] {key} (checkpoint exists)", flush=True)
    else:
        t0 = time.time()
        print(f"[run ] {key} ...", flush=True)
        data = run_ensemble(splits, shared, voter_outputs)
        data["runtime_seconds"] = round(time.time() - t0, 1)
        data["n_ids"] = {s: len(c["ids"]) for s, c in splits.items()}
        data["limit"] = args.limit
        for split in splits:
            with (OUT_DIR / f"predictions_{key}_{split}.jsonl").open("w", encoding="utf-8") as fh:
                for zid, pred in data["predictions"][split].items():
                    fh.write(json.dumps(pred, ensure_ascii=False) + "\n")
        data.pop("predictions")
        ckpt.write_text(json.dumps(data, ensure_ascii=False, indent=1))
        print(
            f"[done] {key} in {data['runtime_seconds']}s | "
            f"dev C1={data['results']['dev']['c1_macro_f1_mean']:.4f} "
            f"train C1={data['results']['train']['c1_macro_f1_mean']:.4f}",
            flush=True,
        )

    # ---- final comparison artifacts ----
    titles = {
        "majority": "M1 Majority Baseline",
        "random": "M2 Random Baseline",
        "dictionary": "M3 Dictionary Lookup",
        "bow_naive_bayes": "M5 BoW + Complement NB",
        "tfidf_logreg": "M4 TF-IDF + LogReg",
        "tfidf_svm": "M6 TF-IDF + Linear SVM",
        "knn_retrieval": "M7 k-NN Retrieval",
        "crf": "M8 Linear-Chain CRF",
        "ensemble_voting": "M9 Approval-Voting Ensemble",
    }
    rows = []
    for key, title in titles.items():
        f = OUT_DIR / f"results_{key}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        rows.append({
            "model": title,
            "dev_c1": d["results"]["dev"]["c1_macro_f1_mean"],
            "dev_c1_std": d["results"]["dev"]["c1_macro_f1_std"],
            "train_c1": d["results"]["train"]["c1_macro_f1_mean"],
            "dev_c2b": d["results"]["dev"]["c2b_macro_f1_mean"],
            "train_c2b": d["results"]["train"]["c2b_macro_f1_mean"],
            "dev_micro": d["results"]["dev"]["micro_f1_mean"],
            "train_micro": d["results"]["train"]["micro_f1_mean"],
            "runtime": d.get("runtime_seconds"),
        })

    lines = [
        "=" * 96,
        "FULL-CORPUS BASE-MODEL COMPARISON (Dev-100 vs Train-908)",
        f"generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} | limit: {args.limit or 'full'}",
        "=" * 96,
        "CAVEAT: supervised models (M4-M8, and M9 built on them) are FITTED on Train-908;",
        "their Train column is IN-SAMPLE (training-set fit), not generalization.",
        "-" * 96,
        f"{'Model':<32} | {'Dev C1':>8} | {'Train C1':>9} | {'D-T':>7} | {'Dev C2b':>8} | {'Tr C2b':>7} | {'Runtime s':>9}",
        "-" * 96,
    ]
    for r in rows:
        lines.append(
            f"{r['model']:<32} | {r['dev_c1']:>8.4f} | {r['train_c1']:>9.4f} | "
            f"{r['train_c1'] - r['dev_c1']:>7.4f} | {r['dev_c2b']:>8.4f} | {r['train_c2b']:>7.4f} | {r['runtime'] or 0:>9.1f}"
        )
    lines += ["=" * 96]

    report = "\n".join(lines)
    (OUT_DIR / f"{MODEL_ID}_report.txt").write_text(report, encoding="utf-8")
    (OUT_DIR / f"{MODEL_ID}_metrics.json").write_text(
        json.dumps({"rows": rows, "n_ids": {s: len(c['ids']) for s, c in splits.items()}, "limit": args.limit}, indent=1),
        encoding="utf-8",
    )
    print("\n" + report)
    print(f"\nDone in {time.time() - started:.1f}s | artifacts: {OUT_DIR}")


if __name__ == "__main__":
    main()
