#!/usr/bin/env python3
"""
========================================================================================
EXECUTIVE SUMMARY: Paired bootstrap between finished Dev-100 runs (offline, 0 API calls)
========================================================================================
- Concept & Goal:
    Answers "is the score difference between run A and run B real, or just noise from which
    100 papers are in the Dev set?". Both runs are resampled over the same papers, so the
    comparison is paired (each replicate uses one paper sample for A and B).
- How It Works:
    Loads `<run_dir>/*_predictions.jsonl` for every run, rebuilds the per-paper tp/fp/fn
    counts exactly like `src/evaluation/evaluator.py`, and recomputes C1 Macro-F1,
    C1+C2b Macro-F1 and Micro-F1 for B bootstrap samples (default 4000, seed 42, the same
    samples for every run). Prints the point scores with a replication check against
    `evaluate_predictions`, then per requested pair the mean difference A - B with a 95 %
    CI and P(A > B). Limit: it measures paper sampling only, not run-to-run LLM noise,
    because every run is a single run. Writes nothing.
- Results on Dev Set: meta-script, no runs of its own. Logged results: docs/experiment_log.md, 3f.

Usage (on the benchmark server, from the repo root):
    .venv/bin/python scripts/paired_bootstrap.py \
        --pair llm_rag_sections_hu_hu-reasoning_minhjson_ctaxa_v4 \
               llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4 \
        --external minh=$HOME/Home/minhanh_repro/runs/20261125-180748_full_text_dev \
        --pair minh llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_rawft
========================================================================================
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

def _find_repo_root() -> Path:
    """The script lives in <repo>/scripts/; when piped over stdin, walk up from the cwd."""
    script = globals().get("__file__")
    if script and Path(script).exists():
        return Path(script).resolve().parents[1]
    root = Path.cwd()
    while not (root / "src/evaluation/evaluator.py").exists() and root != root.parent:
        root = root.parent
    return root


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluator import (  # noqa: E402
    ALL_EVAL_FIELDS,
    C1_FIELDS,
    evaluate_predictions,
    extract_field_set,
    load_jsonl_by_id,
)

METRIC_NAMES = ("C1", "C1+C2b", "Micro")


def load_run(run_dir: Path) -> dict[str, dict]:
    """Loads the single `*_predictions.jsonl` of a run directory, keyed by paper id."""
    hits = glob.glob(str(run_dir / "*_predictions.jsonl"))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one predictions file in {run_dir}, found {len(hits)}")
    return load_jsonl_by_id(Path(hits[0]))


def per_paper_counts(gold: dict, preds: dict, ids: list[str]) -> np.ndarray:
    """(papers, fields, 3) tp/fp/fn counts, accumulated exactly like evaluate_predictions."""
    counts = np.zeros((len(ids), len(ALL_EVAL_FIELDS), 3), dtype=np.int64)
    for i, zid in enumerate(ids):
        g, p = gold.get(zid, {}), preds.get(zid, {})
        for j, field in enumerate(ALL_EVAL_FIELDS):
            g_set, p_set = extract_field_set(g, field), extract_field_set(p, field)
            counts[i, j] = (len(g_set & p_set), len(p_set - g_set), len(g_set - p_set))
    return counts


def score_matrix(counts: np.ndarray, sample_idx: np.ndarray) -> np.ndarray:
    """(papers, fields, 3) counts + sample index matrix (B, papers) -> (B, 3) metrics."""
    agg = counts[sample_idx].sum(axis=1).astype(float)
    tp, fp, fn = agg[..., 0], agg[..., 1], agg[..., 2]
    f1 = np.round(np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12), 0.0), 4)
    c1 = f1[:, : len(C1_FIELDS)].mean(axis=1)
    c1_c2b = f1.mean(axis=1)
    ttp, tfp, tfn = tp.sum(axis=1), fp.sum(axis=1), fn.sum(axis=1)
    micro = np.where(ttp > 0, 2 * ttp / np.maximum(2 * ttp + tfp + tfn, 1e-12), 0.0)
    return np.stack([c1, c1_c2b, micro], axis=1)


def bootstrap(counts: np.ndarray, n_boot: int, seed: int, chunk: int = 500) -> np.ndarray:
    """(n_boot, 3) replicate metrics; the same seed gives the same samples for every run."""
    rng = np.random.default_rng(seed)
    n = counts.shape[0]
    blocks = []
    for start in range(0, n_boot, chunk):
        idx = rng.integers(0, n, size=(min(chunk, n_boot - start), n))
        blocks.append(score_matrix(counts, idx))
    return np.vstack(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage")[0])
    parser.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"),
                        help="run directory names under outputs/architectures/ (repeatable)")
    parser.add_argument("--external", action="append", default=[], metavar="NAME=PATH",
                        help="extra predictions outside outputs/: a predictions.jsonl file or a run "
                             "directory containing one, e.g. minh=/path/to/run_dir")
    parser.add_argument("--boot", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.pair:
        parser.error("give at least one --pair A B")

    gold = load_jsonl_by_id(REPO_ROOT / "data/gold/dev.jsonl")
    ids = [line.strip() for line in (REPO_ROOT / "data/gold/dev_100_ids.txt").read_text().splitlines() if line.strip()]
    arch_root = REPO_ROOT / "outputs/architectures"

    preds: dict[str, dict] = {}
    for spec in args.external:
        name, _, path = spec.partition("=")
        target = Path(path).expanduser()
        if target.is_dir():  # a run directory: take its predictions.jsonl
            target = target / "predictions.jsonl"
        preds[name] = load_jsonl_by_id(target)
    for a, b in args.pair:
        for name in (a, b):
            if name not in preds:
                preds[name] = load_run(arch_root / name)

    # Point scores plus a replication check against the official evaluator.
    identity = np.arange(len(ids))[None, :]
    counts, booted = {}, {}
    print("=== Point scores on dev_100 (max diff vs evaluate_predictions must be ~0) ===")
    for name, run_preds in sorted(preds.items()):
        counts[name] = per_paper_counts(gold, run_preds, ids)
        mine = score_matrix(counts[name], identity)[0]
        ref = evaluate_predictions(gold, run_preds, filter_ids=set(ids))
        ref_v = np.array([ref["c1_macro_f1"], ref["c2b_macro_f1"], ref["micro_f1"]])
        print("%-62s %s   (max diff %.5f)" % (
            name[:62], " | ".join("%s %.4f" % (m, x) for m, x in zip(METRIC_NAMES, mine)),
            float(np.abs(mine - ref_v).max())))

    print()
    print("=== Paired bootstrap over Dev papers (B=%d, seed %d): A - B, 95%% CI, P(A>B) ===" % (args.boot, args.seed))
    for a, b in args.pair:
        for name in (a, b):
            if name not in booted:
                booted[name] = bootstrap(counts[name], args.boot, args.seed)
        diff = booted[a] - booted[b]
        parts = []
        for k, metric in enumerate(METRIC_NAMES):
            lo, hi = np.percentile(diff[:, k], [2.5, 97.5])
            parts.append("%s %+.4f [%+.4f, %+.4f] P=%.2f" % (
                metric, diff[:, k].mean(), lo, hi, (diff[:, k] > 0).mean()))
        print("%s  -  %s\n    %s" % (a, b, " | ".join(parts)))


if __name__ == "__main__":
    main()
