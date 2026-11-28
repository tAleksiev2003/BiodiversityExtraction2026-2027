"""Precision-leak check: do we emit German taxa names on papers with NO German gold?

Gold never stores latin-only taxa entries, so a paper without gold german_name
has no gold taxa at all (or only group fields) — every german_name we emit there
is a pure FP. Reports per run: how many labels/papers leak, and the corpus-level
german_name P/R/F1 counterfactual if those predictions were emptied.
"""
import glob
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.evaluation.evaluator import extract_field_set, load_jsonl_by_id  # noqa: E402

dev = load_jsonl_by_id(Path("data/gold/dev.jsonl"))
DEV_IDS = [l.strip() for l in open("data/gold/dev_100_ids.txt") if l.strip()]
RUNS = {
    "saia best (ctaxa_v4_biknn)": "llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_biknn",
    "hu best (minhjson_ctaxa_v4)": "llm_rag_sections_hu_hu-reasoning_minhjson_ctaxa_v4",
}

no_ger = [z for z in DEV_IDS if not extract_field_set(dev[z], "taxa.german_name")]
no_taxa_at_all = [z for z in no_ger if not extract_field_set(dev[z], "taxa.species_group")]
print(f"papers without gold german_name: {len(no_ger)} | of those without any gold taxa group: {len(no_taxa_at_all)}")

for label, run in RUNS.items():
    hits = glob.glob(f"outputs/architectures/{run}/*_predictions.jsonl")
    if not hits:
        print(f"\n{label}: run missing")
        continue
    preds = load_jsonl_by_id(Path(hits[0]))

    def corpus_f1(empty_no_ger=False):
        tp = fp = fn = 0
        for z in DEV_IDS:
            g = extract_field_set(dev[z], "taxa.german_name")
            p = set() if (empty_no_ger and z in no_ger) else extract_field_set(preds.get(z, {}), "taxa.german_name")
            tp += len(g & p)
            fp += len(p - g)
            fn += len(g - p)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)

    leaked_labels = sum(len(extract_field_set(preds.get(z, {}), "taxa.german_name")) for z in no_ger)
    leaked_papers = sum(1 for z in no_ger if extract_field_set(preds.get(z, {}), "taxa.german_name"))
    p0, r0, f0 = corpus_f1(False)
    p1, r1, f1 = corpus_f1(True)
    print(f"\n=== {label} ===")
    print(f"  german labels emitted on no-gold papers: {leaked_labels} (in {leaked_papers} papers) -> pure FPs")
    print(f"  corpus ger_name F1 now:    P {p0:.3f} R {r0:.3f} F1 {f0:.3f}")
    print(f"  counterfactual (emptied):  P {p1:.3f} R {r1:.3f} F1 {f1:.3f}   (gain {f1 - f0:+.3f})")
