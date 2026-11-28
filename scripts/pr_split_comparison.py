import glob, sys
from pathlib import Path
sys.path.insert(0, ".")
from src.evaluation.evaluator import extract_field_set, load_jsonl_by_id
gold = load_jsonl_by_id(Path("data/gold/dev.jsonl"))
ids = [l.strip() for l in open("data/gold/dev_100_ids.txt") if l.strip()]
ours = load_jsonl_by_id(Path(sorted(glob.glob("outputs/architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_biknn/*_predictions.jsonl"))[0]))
minh = load_jsonl_by_id(Path("/home/lennard/Home/minhanh_repro/runs/20261125-180748_full_text_dev/predictions.jsonl"))
fields = ["biodiversity_level", "ecosystem_type.category", "ecosystem_type.term",
          "ecosystem_service.category", "ecosystem_service.term", "taxa.scientific_name"]
print("%-26s %-16s %-16s" % ("field", "OURS  P / R", "MINH  P / R"))
for f in fields:
    row = []
    for preds in (ours, minh):
        tp = fp = fn = 0
        for z in ids:
            gs = extract_field_set(gold.get(z, {}), f)
            ps = extract_field_set(preds.get(z, {}), f)
            tp += len(gs & ps); fp += len(ps - gs); fn += len(gs - ps)
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        row.append("%.2f / %.2f" % (p, r))
    print("%-26s %-16s %-16s" % (f, row[0], row[1]))
gold_lvl = sum(1 for z in ids if extract_field_set(gold.get(z, {}), "biodiversity_level"))
gold_srv = sum(1 for z in ids if extract_field_set(gold.get(z, {}), "ecosystem_service"))
for name, preds in ("OURS", ours), ("MINH", minh):
    emit_lvl = sum(1 for z in ids if extract_field_set(preds.get(z, {}), "biodiversity_level"))
    emit_srv = sum(1 for z in ids if extract_field_set(preds.get(z, {}), "ecosystem_service"))
    print("%s: emits level on %d/100 papers, services on %d/100 (gold: level %d, services %d)" % (name, emit_lvl, emit_srv, gold_lvl, gold_srv))
