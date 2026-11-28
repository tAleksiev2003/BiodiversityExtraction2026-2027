"""What exactly breaks on taxa.german_name in English-language papers?

1. DB convention: are german names stored PAIRED with scientific names per taxon entry?
2. How many dev papers carry gold german names, and how many of those names exist
   LITERALLY in the document (German-language docs) vs. are translations (English papers)?
3. Our best run on translation-required papers: german recall/precision vs scientific recall.
4. Headroom: papers where we got the Latin name right but missed the paired German name;
   how many would a TRAIN-derived Latin->German glossary already fix?
"""
import glob
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, ".")
from src.evaluation.evaluator import extract_field_set, load_jsonl_by_id  # noqa: E402

DEV_IDS = [l.strip() for l in open("data/gold/dev_100_ids.txt") if l.strip()]
train = load_jsonl_by_id(Path("data/gold/train.jsonl"))
dev = load_jsonl_by_id(Path("data/gold/dev.jsonl"))
RUN = "outputs/architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_biknn"
preds = load_jsonl_by_id(Path(glob.glob(RUN + "/*_predictions.jsonl")[0]))


def route_text(zid):
    for path in (Path(f"data/parsed_text/marker_dev/{zid}.md"), Path(f"data/parsed_text/grobid_dev/{zid}.tei.xml")):
        if path.exists():
            raw = path.read_text(errors="ignore")
            if path.suffix == ".xml":
                raw = re.sub(r"<[^>]+>", " ", raw)
            return re.sub(r"\s+", " ", raw).lower()
    return ""


# --- 1) pairing convention in gold ------------------------------------------
entries = pair_both = ger_only = sci_only = 0
glossary = defaultdict(set)
for rec in list(train.values()) + list(dev.values()):
    for tax in rec.get("taxa") or []:
        if not isinstance(tax, dict):
            continue
        entries += 1
        sci, ger = tax.get("scientific_name"), tax.get("german_name")
        if sci and ger:
            pair_both += 1
            glossary[str(sci).strip()].add(str(ger).strip())
        elif ger:
            ger_only += 1
        elif sci:
            sci_only += 1
ambiguous = sum(1 for sci, ger_set in glossary.items() if len(ger_set) > 1)
print("=== 1) gold taxa entries: %d | paired sci+ger: %d | german-only: %d | sci-only: %d" % (
    entries, pair_both, ger_only, sci_only))
print("   train-derived glossary: %d latin names -> german, %d ambiguous (multiple german forms)" % (
    len(glossary), ambiguous))

# --- 2) dev papers with german gold: literal vs translation ------------------
papers_ger = [z for z in DEV_IDS if extract_field_set(dev[z], "taxa.german_name")]
literal_docs = translation_docs = 0
for z in papers_ger:
    doc = route_text(z)
    ger_names = extract_field_set(dev[z], "taxa.german_name")
    if any(name.lower()[:18] in doc for name in ger_names):
        literal_docs += 1
    else:
        translation_docs += 1
print("\n=== 2) dev papers with gold german names: %d | name literal in doc: %d | translation-needed: %d" % (
    len(papers_ger), literal_docs, translation_docs))

# --- 3) our performance on translation-needed papers -------------------------
tot_ger_tp = tot_ger_gold = tot_ger_pred = tot_sci_gold = tot_sci_tp = 0
sci_right_ger_miss_papers = 0
glossary_fixable = glossary_covered = 0
examples = []
for z in papers_ger:
    doc = route_text(z)
    ger_names = extract_field_set(dev[z], "taxa.german_name")
    if any(name.lower()[:18] in doc for name in ger_names):
        continue  # literal German doc, different problem
    p = preds.get(z, {})
    pred_ger = extract_field_set(p, "taxa.german_name")
    pred_sci = extract_field_set(p, "taxa.scientific_name")
    gold_sci = extract_field_set(dev[z], "taxa.scientific_name")
    tot_ger_gold += len(ger_names)
    tot_ger_pred += len(pred_ger)
    tot_ger_tp += len(ger_names & pred_ger)
    tot_sci_gold += len(gold_sci)
    tot_sci_tp += len(gold_sci & pred_sci)
    missed = ger_names - pred_ger
    if (gold_sci & pred_sci) and missed:
        sci_right_ger_miss_papers += 1
        for name in missed:
            # which latin does this german belong to (gold entry pairing)?
            latin = None
            for tax in dev[z].get("taxa") or []:
                if isinstance(tax, dict) and str(tax.get("german_name") or "").strip() == name:
                    latin = str(tax.get("scientific_name") or "").strip() or None
            if latin:
                glossary_covered += 1
                if glossary.get(latin) and name in glossary[latin] and latin in pred_sci:
                    glossary_fixable += 1
        if len(examples) < 6:
            examples.append((z, sorted(missed)[:3], sorted(gold_sci & pred_sci)[:3]))

prec = tot_ger_tp / max(tot_ger_pred, 1)
rec = tot_ger_tp / max(tot_ger_gold, 1)
f1 = 2 * prec * rec / max(prec + rec, 1e-9)
print("\n=== 3) on translation-needed papers ===")
print("   german: gold %d | pred %d | tp %d -> P %.2f R %.2f F1 %.2f" % (
    tot_ger_gold, tot_ger_pred, tot_ger_tp, prec, rec, f1))
print("   scientific (same papers): gold %d | tp %d -> recall %.2f" % (
    tot_sci_gold, tot_sci_tp, tot_sci_tp / max(tot_sci_gold, 1)))
print("   papers with correct latin but missing german: %d of %d" % (sci_right_ger_miss_papers, len(papers_ger)))
print("   of the missing german names, gold pairs them to a latin we predicted right: %d; glossary would fix %d" % (
    glossary_covered, glossary_fixable))
for z, missed, hit in examples:
    print("   %s: missed %s | latin hit %s" % (z, missed, hit))
