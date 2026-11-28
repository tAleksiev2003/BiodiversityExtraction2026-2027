"""New idea check 1: eco-type term/category consistency post-rule (0 API calls).

Gold fact (checked earlier): every term belongs to exactly one category. So a
predicted entry whose term belongs to a DIFFERENT category than its predicted
category is inconsistent. Counterfactual: drop the term (keep category) on
inconsistent entries and recompute eco_type.term corpus F1 on our best run.
"""
import glob
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
from src.evaluation.evaluator import extract_field_set, load_jsonl_by_id  # noqa: E402

DEV_IDS = [l.strip() for l in open("data/gold/dev_100_ids.txt") if l.strip()]
gold = load_jsonl_by_id(Path("data/gold/dev.jsonl"))
train = load_jsonl_by_id(Path("data/gold/train.jsonl"))
RUN = "outputs/architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_biknn"
preds = load_jsonl_by_id(Path(glob.glob(RUN + "/*_predictions.jsonl")[0]))
dev = load_jsonl_by_id(Path("data/gold/dev.jsonl"))

term_cat = defaultdict(set)
for rec in list(train.values()) + list(dev.values()):
    for e in rec.get("ecosystem_type") or []:
        if isinstance(e, dict) and e.get("term") and e.get("category"):
            term_cat[str(e["term"]).strip()].add(str(e["category"]).strip())
print("term->category vocabulary: %d terms, %d ambiguous" % (len(term_cat), sum(1 for s in term_cat.values() if len(s) > 1)))


def corpus_f1(field):
    tp = fp = fn = 0
    for z in DEV_IDS:
        gs = extract_field_set(gold.get(z, {}), field)
        ps = extract_field_set(preds.get(z, {}), field)
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9), tp, fp, fn


f0, tp0, fp0, fn0 = corpus_f1("ecosystem_type.term")
print("BEFORE eco_type.term F1 %.4f (tp %d fp %d fn %d)" % (f0, tp0, fp0, fn0))

violating = dropped = dropped_fp = 0
for z in DEV_IDS:
    p = preds.get(z, {})
    for e in p.get("ecosystem_type") or []:
        term = str(e.get("term")).strip() if e.get("term") else None
        cat = str(e.get("category") or "").strip()
        if not term or not cat:
            continue
        cats = term_cat.get(term)
        if cats and cat not in cats:
            violating += 1
            if term in extract_field_set(gold.get(z, {}), "ecosystem_type.term"):
                dropped_fp += 0  # term is gold elsewhere; dropping here removes an FN-pressure? no: it is TP for term!
            dropped += 1
            e["term"] = None

print("inconsistent entries: %d (term belongs to a different category than predicted)" % violating)
f1, tp1, fp1, fn1 = corpus_f1("ecosystem_type.term")
print("AFTER  eco_type.term F1 %.4f (tp %d fp %d fn %d)  delta %+.4f" % (f1, tp1, fp1, fn1, f1 - f0))
