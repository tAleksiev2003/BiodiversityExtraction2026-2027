"""Does Docling (OCR) help on scan papers? Empirical check on V2BVMBPF + text sizes.

Per-paper C1 F1 for the single dev scan paper across every parser run we have,
plus the extracted text volume per parser cache:
- PyMuPDF: no text layer expected -> ~0 chars
- GROBID: blind on scans -> ~0
- Docling Light (--docling): do_ocr=False -> expected ~0
- Marker: its own OCR -> proven ~27 KB
- Full Docling OCR (Minh JSON, --minh-json): do_ocr=True -> ?
"""
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.evaluation.evaluator import C1_FIELDS, extract_field_set, load_jsonl_by_id  # noqa: E402

ZID = "V2BVMBPF"
gold = load_jsonl_by_id(Path("data/gold/dev.jsonl"))
DEV_IDS = {l.strip() for l in open("data/gold/dev_100_ids.txt") if l.strip()}
g = gold.get(ZID, {})


def per_paper_c1(pred):
    f1s = []
    for f in C1_FIELDS:
        gs, ps = extract_field_set(g, f), extract_field_set(pred, f)
        tp, fp, fn = len(gs & ps), len(ps - gs), len(gs - ps)
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1) if tp > 0 else 0.0)
    return sum(f1s) / len(f1s)


RUNS = {
    "pymupdf baseline": "llm_rag_sections_saia_deepseek-v4.1-flash-1017",
    "grobid baseline": "llm_rag_sections_saia_deepseek-v4.1-flash-1017_grobid",
    "marker baseline": "llm_rag_sections_saia_deepseek-v4.1-flash-1017_marker",
    "marker ev+ctaxa": "llm_rag_sections_saia_deepseek-v4.1-flash-1017_marker_ev_ctaxa",
    "docling light baseline": "llm_rag_sections_saia_deepseek-v4.1-flash-1017_docling",
    "autoparser v4 (routes->marker)": "llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_biknn",
    "minhjson v4 (full OCR)": "llm_rag_sections_saia_deepseek-v4.1-flash-1017_minhjson_ctaxa_v4",
}

print("=== per-paper C1 F1 for", ZID, "===\n")
for label, run in RUNS.items():
    hits = glob.glob(f"outputs/architectures/{run}/*_predictions.jsonl")
    if not hits:
        print("  %-34s (run missing)" % label)
        continue
    preds = load_jsonl_by_id(Path(hits[0]))
    p = preds.get(ZID)
    if not p:
        print("  %-34s (paper missing)" % label)
        continue
    hab = extract_field_set(p, "habitat")
    sgrp = extract_field_set(p, "taxa.species_group")
    print("  %-34s C1 %.3f | habitat %s | sp.grp %s" % (label, per_paper_c1(p), sorted(hab) or "-", sorted(sgrp) or "-"))

print("\n=== gold (from audit: known hard paper) ===")
for f in C1_FIELDS:
    print("  %-26s %s" % (f, sorted(extract_field_set(g, f)) or "-"))

print("\n=== extracted text volume per parser cache ===")
caches = {
    "pymupdf (page texts)": glob.glob(f"data/parsed_text/dev/{ZID}_*.txt"),
    "grobid TEI": glob.glob(f"data/parsed_text/grobid_dev/{ZID}.tei.xml"),
    "marker md (own OCR)": glob.glob(f"data/parsed_text/marker_dev/{ZID}.md"),
    "docling light md (no OCR)": glob.glob(f"data/parsed_text/docling_dev/{ZID}.md"),
    "full docling OCR json": glob.glob(f"data/processed/docling/dev/{ZID}.json"),
}
for label, hits in caches.items():
    if not hits:
        print("  %-28s MISSING" % label)
        continue
    p = Path(hits[0])
    raw = p.read_text(errors="ignore")
    if label.startswith("full docling"):
        texts = re.findall(r'"text"\s*:\s*"([^"]+)"', raw)
        chars = sum(len(t) for t in texts)
    else:
        chars = len(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw)).strip())
    print("  %-28s %6d chars" % (label, chars))
