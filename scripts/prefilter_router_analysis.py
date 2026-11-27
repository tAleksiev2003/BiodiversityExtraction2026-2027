"""Offline cheap-prefilter router analysis (0 API calls).

Questions answered:
1. Do cheap, label-free signals (PyMuPDF + PDF metadata + GROBID shape) reliably
   predict which parser produces the best prediction for a paper?
2. Do candidate pre-filter rules beat the current 2-rule router (marker when
   g_divs<=3 or g_ratio<0.65, else grobid), cross-validated over 5 folds?
3. Paired bootstrap CIs of each candidate router vs. the current router.

Caveats printed in the header: parser runs differ in prompt flags and cache
state (pymupdf/grobid/marker/docling baselines are pre-cache plain runs,
marker_ev/grobid_ev are ev+ctaxa, autoparser/minhjson are ctaxa+v4) — winner
labels therefore mix parser and prompt effects.
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from src.evaluation.evaluator import (  # noqa: E402
    C1_FIELDS,
    evaluate_predictions,
    extract_field_set,
    load_jsonl_by_id,
)

SEED = 42
IDS = [l.strip() for l in open("data/gold/dev_100_ids.txt") if l.strip()]
gold = load_jsonl_by_id(Path("data/gold/dev.jsonl"))
signals = json.load(open("/tmp/pdf_signals.json"))
SAIA = "llm_rag_sections_saia_deepseek-v4.1-flash-1017_"
HU = "llm_rag_sections_hu_hu-reasoning_"
RUNSETS = {
    "saia": {
        "pymupdf": SAIA,
        "grobid_ev": SAIA + "grobid_ev_ctaxa",
        "marker_ev": SAIA + "marker_ev_ctaxa",
        "docling": SAIA + "docling",
        "autoparser_v4": SAIA + "autoparser_ctaxa_v4",
        "minhjson_v4": SAIA + "minhjson_ctaxa_v4",
    },
    "hu": {
        "pymupdf": HU,
        "grobid_ev": HU + "grobid_ev_ctaxa",
        "marker_ev": HU + "marker_ev_ctaxa",
        "docling": HU + "docling",
        "autoparser_v4": HU + "autoparser_ctaxa_v4",
        "minhjson_v4": HU + "minhjson_ctaxa_v4",
    },
}


def grobid_signals(zid):
    p = Path(f"data/parsed_text/grobid_dev/{zid}.tei.xml")
    if not p.exists():
        return None
    tei = p.read_text(errors="ignore")
    body = tei.split("<body", 1)[-1]
    divs = body.count("<div")
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)).strip()
    return divs, len(text)


sig = {}
for zid in IDS:
    s = dict(signals[zid])
    g = grobid_signals(zid)
    s["g_divs"], s["g_chars"] = g if g else (None, None)
    s["g_ratio"] = (s["g_chars"] / s["chars"]) if (g and s["chars"]) else None
    sig[zid] = s


def per_paper_c1(pred_records):
    """(100,) per-paper macro F1 over the 5 C1 fields (papers without a prediction -> np.nan)."""
    out = np.full(len(IDS), np.nan)
    for i, zid in enumerate(IDS):
        p = pred_records.get(zid)
        if not p:
            continue
        f1s = []
        for f in C1_FIELDS:
            g_set, p_set = extract_field_set(gold.get(zid, {}), f), extract_field_set(p, f)
            tp = len(g_set & p_set)
            fp, fn = len(p_set - g_set), len(g_set - p_set)
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1) if tp > 0 else 0.0)
        out[i] = float(np.mean(f1s))
    return out


scores, preds = {}, {}
for provider, runset in RUNSETS.items():
    scores[provider], preds[provider] = {}, {}
    for parser, run_dir in runset.items():
        hits = glob.glob(f"outputs/architectures/{run_dir}/*_predictions.jsonl")
        if not hits:
            continue
        preds[provider][parser] = load_jsonl_by_id(Path(hits[0]))
        n = len(preds[provider][parser])
        if n < 100:
            print(f"[skip] {provider}/{parser}: only {n} predictions")
            continue
        scores[provider][parser] = per_paper_c1(preds[provider][parser])

for provider in RUNSETS:
    print(f"\n=== {provider}: per-parser mean per-paper C1 (n=100 each) ===")
    for parser, arr in sorted(scores[provider].items(), key=lambda kv: -np.nanmean(kv[1])):
        print("  %-14s %.4f" % (parser, float(np.nanmean(arr))))

# Winner analysis: strict best with margin 0.02 per paper.
MARGIN = 0.02
print("\n=== strict per-paper winners (margin > %.2f per-paper C1) ===" % MARGIN)
winners = {}
for provider in RUNSETS:
    table = np.vstack([scores[provider][p] for p in sorted(scores[provider])])  # (P, 100)
    names = sorted(scores[provider])
    win_idx = []
    for i in range(len(IDS)):
        col = table[:, i]
        if np.isnan(col).any():
            col = col[~np.isnan(col)]
        order = np.argsort(-col)
        win_idx.append(order[0] if (col[order[0]] - col[order[1]]) > MARGIN else -1)
    winners[provider] = win_idx
    vals, counts = np.unique(win_idx, return_counts=True)
    dist = {("tie" if v == -1 else names[v]): int(c) for v, c in zip(vals, counts)}
    print("  %s: %s" % (provider, dist))

# Signal separation: mean signal per winner group (strict winners only).
def winner_mask(provider, parser):
    names = sorted(scores[provider])
    return np.array([names[w] == parser if w >= 0 else False for w in winners[provider]])


print("\n=== signal means for strict winner groups (top signals only) ===")
for provider in ("saia", "hu"):
    for parser in ("grobid_ev", "marker_ev", "minhjson_v4", "autoparser_v4", "docling"):
        if parser not in scores[provider]:
            continue
        m = winner_mask(provider, parser)
        if m.sum() < 5:
            continue
        parts = []
        for key in ("chars", "chars_per_page", "pages", "headings", "images", "umlaut_ratio", "g_ratio", "g_divs"):
            vals = np.array([sig[z][key] if sig[z][key] is not None else np.nan for z in IDS], dtype=float)
            parts.append("%s %.0f vs %.0f" % (key, np.nanmean(vals[m]), np.nanmean(vals[~m])))
        print("  %-4s %-13s n=%2d | %s" % (provider, parser, m.sum(), " | ".join(parts)))

# ---- candidate routers (label-free, cheap signals only) --------------------
def r0(zid, s):  # current router
    return "marker_ev" if (s["g_divs"] is not None and (s["g_divs"] <= 3 or (s["g_ratio"] or 1) < 0.65)) else "grobid_ev"


def r1_scan(zid, s):
    if s["text_layer_ratio"] is not None and s["text_layer_ratio"] < 0.3:
        return "marker_ev"
    if s["chars"] < 500:
        return "marker_ev"
    return r0(zid, s)


def r2_shortclean(zid, s, chars_cut=10000, head_cut=2):
    if s["text_layer_ratio"] is not None and s["text_layer_ratio"] < 0.3:
        return "marker_ev"
    if s["chars"] < 500:
        return "marker_ev"
    if s["chars"] < chars_cut and s["headings"] >= head_cut and s["umlaut_ratio"] < 0.02:
        return "pymupdf"
    return r0(zid, s)


def r3_longdoc(zid, s, pages_cut=40):
    if s["pages"] is not None and s["pages"] >= pages_cut:
        return "minhjson_v4"
    return r2_shortclean(zid, s)


def routed_corpus(provider, rule):
    """Corpus metrics when each paper takes its routed parser's predictions."""
    routed = {}
    for zid in IDS:
        parser = rule(zid, sig[zid])
        if parser not in preds[provider]:
            parser = "autoparser_v4"
        rec = preds[provider][parser].get(zid) or preds[provider]["autoparser_v4"].get(zid) or {}
        routed[zid] = rec
    m = evaluate_predictions(gold, routed, filter_ids=set(IDS))
    return m["c1_macro_f1"], m["c2b_macro_f1"], m["micro_f1"]


def boot_vs_r0(provider, rule, B=2000, seed=SEED):
    """Paired bootstrap of corpus C1 difference (rule - r0) by resampling papers."""
    rng = np.random.default_rng(seed)
    base = {p: scores[provider][p] for p in scores[provider]}
    chosen = {p: [rule(z, sig[z]) for z in IDS] for p in [provider]}
    names = sorted(base)
    arrs = {p: np.vstack([base[p] for p in names]) for p in [provider]}
    arr = arrs[provider]
    pick = {parser: np.array([1.0 if c == parser else np.nan for c in chosen[provider]]) for parser in names}

    def c1_at(idx, rule_pick):
        f1s = []
        for j, f in enumerate(C1_FIELDS):
            tp = fp = fn = 0.0
            for i in idx:
                col = arr[:, i]
                parr = rule_pick[:, i]
                val = np.nanmean(col[parr == 1]) if (parr == 1).any() else np.nan
                # approximate from per-paper F1: not exact set pooling, proxy only
            f1s.append(np.nanmean(col[pick_of(idx, rule_pick)]))
        return float(np.mean(f1s))

    # corpus-level via evaluate_predictions on resampled id sets (exact, slower but B small ok)
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, len(IDS), len(IDS))
        ids_b = [IDS[i] for i in idx]
        r_a = {}
        r_b = {}
        for i in idx:
            z = IDS[i]
            r_a[z] = preds[provider][r0(z, sig[z])].get(z) or {}
            p = rule(z, sig[z])
            r_b[z] = preds[provider].get(p, preds[provider]["autoparser_v4"]).get(z) or {}
        ma = evaluate_predictions(gold, r_a, filter_ids=set(ids_b))
        mb = evaluate_predictions(gold, r_b, filter_ids=set(ids_b))
        diffs.append(mb["c1_macro_f1"] - ma["c1_macro_f1"])
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return diffs.mean(), lo, hi


RULES = {
    "r0 current (marker if grobid weak)": lambda z, s: r0(z, s),
    "r1 = r0 + scan->marker": lambda z, s: r1_scan(z, s),
    "r2 = r1 + short-clean->pymupdf": lambda z, s: r2_shortclean(z, s),
    "r3 = r2 + long(>=40p)->minhjson": lambda z, s: r3_longdoc(z, s),
}
for provider in ("saia", "hu"):
    print(f"\n=== {provider}: candidate routers (corpus metrics, 100 papers) ===")
    for name, rule in RULES.items():
        c1, c2b, micro = routed_corpus(provider, rule)
        mean, lo, hi = boot_vs_r0(provider, rule, B=600)
        print("  %-34s C1 %.4f | C1+C2b %.4f | Micro %.4f | dC1 vs r0 %+.4f [%+.4f, %+.4f]" % (
            name, c1, c2b, micro, mean, lo, hi))

# 5-fold CV: pick r2/r3 thresholds on 4 folds, evaluate on the held-out fold.
print("\n=== 5-fold CV of the r3 rule family (thresholds picked per fold-train) ===")
rng = np.random.default_rng(SEED)
folds = rng.permutation(len(IDS)).reshape(5, -1)


def r3_param(zid, s, chars_cut, head_cut, pages_cut):
    if s["pages"] is not None and s["pages"] >= pages_cut:
        return "minhjson_v4"
    return r2_shortclean(zid, s, chars_cut, head_cut)


for provider in ("saia", "hu"):
    grid = [(c, h, p) for c in (6000, 10000, 15000) for h in (1, 2) for p in (25, 40)]
    hold_scores = []
    for k in range(5):
        hold = list(folds[k])
        train_idx = [i for kk in range(5) if kk != k for i in folds[kk]]
        best, best_score = None, -1
        for c, h, p in grid:
            routed = {}
            for i in train_idx:
                z = IDS[i]
                parser = r3_param(z, sig[z], c, h, p)
                routed[z] = preds[provider].get(parser, preds[provider]["autoparser_v4"]).get(z) or {}
            m = evaluate_predictions(gold, routed, filter_ids={IDS[i] for i in train_idx})
            if m["c1_macro_f1"] > best_score:
                best, best_score = (c, h, p), m["c1_macro_f1"]
        routed = {}
        for i in hold:
            z = IDS[i]
            parser = r3_param(z, sig[z], *best)
            routed[z] = preds[provider].get(parser, preds[provider]["autoparser_v4"]).get(z) or {}
        m = evaluate_predictions(gold, routed, filter_ids={IDS[i] for i in hold})
        hold_scores.append(m["c1_macro_f1"])
        print("  %s fold %d: picked chars<%d heads>=%d pages>=%d -> holdout C1 %.4f" % (
            provider, k, best[0], best[1], best[2], m["c1_macro_f1"]))
    # r0 baseline on the same folds for comparison
    r0_hold = []
    for k in range(5):
        routed = {IDS[i]: preds[provider][r0(IDS[i], sig[IDS[i]])].get(IDS[i]) or {} for i in folds[k]}
        r0_hold.append(evaluate_predictions(gold, routed, filter_ids={IDS[i] for i in folds[k]})["c1_macro_f1"])
    print("  %s CV mean: r3 %.4f vs r0 %.4f" % (provider, float(np.mean(hold_scores)), float(np.mean(r0_hold))))
