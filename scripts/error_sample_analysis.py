"""Exemplar error analysis: random wrong papers from our confirmed best run.

Stratified random sample (seed 42) of papers with field errors on the four weak
fields (biodiversity_level, ecosystem_service, taxa names, ecosystem_type),
from the confirmed best run (AutoParser + clean-taxa + v4 + bilingual-knn,
DeepSeek 4.1, redo 27 Nov). For each paper: title, router decision, doc size,
gold vs prediction per failing field, plus automated evidence probes:
- FN (gold missing in prediction): keyword probes searched in the prompt-window
  (first 16000 chars of the routed parser text) vs the full document ->
  distinguishes "budget cut the evidence" from "evidence not in text at all".
- FP (prediction not in gold): literal substring probe of the predicted value
  in the document -> hallucination vs paraphrase.
Also reports which papers carry a flag in docs/dev_100_paper_audit_table.md
(known gold-label problems).
"""
import glob
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.evaluation.evaluator import extract_field_set, load_jsonl_by_id  # noqa: E402

RUN = "outputs/architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_biknn"
IDS = [l.strip() for l in open("data/gold/dev_100_ids.txt") if l.strip()]
gold = load_jsonl_by_id(Path("data/gold/dev.jsonl"))
preds = load_jsonl_by_id(Path(glob.glob(RUN + "/*_predictions.jsonl")[0]))
audit = Path("docs/dev_100_paper_audit_table.md").read_text() if Path("docs/dev_100_paper_audit_table.md").exists() else ""

LEVEL_PROBES = {
    "Artenvielfalt": ["species richness", "diversity index", "shannon", "simpson", "species list", "biodiversity"],
    "Zusammensetzung der Artengemeinschaft": ["community composition", "assemblage", "abundance", "community structur"],
    "Habitatdiversität": ["habitat diversity", "habitat heterogeneity", "biotope", "habitat type", "mosaic"],
    "Funktionelle Diversität": ["functional diversity", "functional trait", "functional group", "multifunctional"],
    "Genetische Diversität": ["genetic diversity", "genotype", "microsatellite", "snp", "population genetic"],
    "Strukturelle Diversität": ["deadwood", "vegetation structure", "stand structure", "layer", "vertical structure"],
    "Phylogenetische Diversität": ["phylogenetic"],
}
SERVICE_PROBES = {
    "Versorgungsleistungen": ["provision", "crop", "yield", "timber", "food", "fodder", "bioenergy", "raw material"],
    "Regulierungs- und Erhaltungsleistungen": ["carbon", "climate regulat", "water quality", "flood", "pollination", "water purification", "habitat for species", "soil formation"],
    "Kulturelle Leistungen": ["recreation", "tourism", "aesthetic", "cultural ecosystem", "heritage", "well-being"],
}


def routed_parser(zid):
    """Approximates the AutoParser decision (marker when GROBID is weak)."""
    tei = Path(f"data/parsed_text/grobid_dev/{zid}.tei.xml")
    if not tei.exists():
        return "marker", Path(f"data/parsed_text/marker_dev/{zid}.md")
    body = tei.read_text(errors="ignore").split("<body", 1)[-1]
    divs = body.count("<div")
    txt = re.sub(r"<[^>]+>", " ", body)
    txt = re.sub(r"\s+", " ", txt).strip()
    ratio = len(txt) / max(1, 200000)  # rough; only the weak/non-weak decision matters
    pym = Path(f"data/parsed_text/marker_dev/{zid}.md")
    if divs <= 3 or len(txt) < 0.5 * 200000 * 0.5:  # g_divs<=3 or g_ratio<0.65 with unknown base
        if pym.exists():
            return "marker", pym
    return "grobid", tei


def route_text(zid):
    parser, path = routed_parser(zid)
    if not path.exists():
        return parser, "", ""
    raw = path.read_text(errors="ignore")
    if parser == "grobid":
        body = raw.split("<body", 1)[-1]
        raw = re.sub(r"<[^>]+>", " ", body)
        raw = re.sub(r"\s+", " ", raw).strip()
    return parser, raw[:16000].lower(), raw.lower()


def probe_hit(probes, window, full):
    in_budget = any(p in window for p in probes)
    in_doc = any(p in full for p in probes)
    return "BUDGET" if in_budget else ("DOC" if in_doc else "ABSENT")


errors = {}
for zid in IDS:
    g, p = gold.get(zid, {}), preds.get(zid, {})
    e = {}
    lvl_g, lvl_p = extract_field_set(g, "biodiversity_level"), extract_field_set(p, "biodiversity_level")
    if lvl_g - lvl_p or lvl_p - lvl_g:
        e["level"] = (lvl_g, lvl_p)
    srv_g, srv_p = set(), set()
    for sub in ("category", "term"):
        srv_g |= extract_field_set(g, f"ecosystem_service.{sub}")
        srv_p |= extract_field_set(p, f"ecosystem_service.{sub}")
    if srv_g - srv_p or srv_p - srv_g:
        e["service"] = (srv_g, srv_p)
    tax_g, tax_p = set(), set()
    for sub in ("scientific_name", "german_name"):
        tax_g |= extract_field_set(g, f"taxa.{sub}")
        tax_p |= extract_field_set(p, f"taxa.{sub}")
    if tax_g - tax_p or tax_p - tax_g:
        e["taxa"] = (tax_g, tax_p)
    eco_g, eco_p = set(), set()
    for sub in ("category", "term"):
        eco_g |= extract_field_set(g, f"ecosystem_type.{sub}")
        eco_p |= extract_field_set(p, f"ecosystem_type.{sub}")
    if eco_g - eco_p or eco_p - eco_g:
        e["ecotype"] = (eco_g, eco_p)
    if e:
        errors[zid] = e

rng = random.Random(42)
plan = [("level", 3), ("service", 3), ("taxa", 2), ("ecotype", 2)]
picked = []
for field, n in plan:
    pool = [z for z in errors if field in errors[z] and z not in picked]
    picked += rng.sample(pool, min(n, len(pool)))

print(f"papers with >=1 error on the weak fields: {len(errors)}/100\n")
for zid in picked:
    g, p = gold.get(zid, {}), preds.get(zid, {})
    parser, window, full = route_text(zid)
    flags = "AUDIT-FLAG" if zid in audit else "-"
    print("=" * 78)
    print(f"{zid} | route={parser} | flags={flags} | {str(g.get('bibtex_title'))[:90]}")
    e = errors[zid]
    if "level" in e:
        lg, lp = e["level"]
        print(f"  LEVEL   gold={sorted(lg)}")
        print(f"          pred={sorted(lp) or '(empty)'}")
        for v in sorted(lg - lp):
            print(f"    [FN] {v}: {probe_hit(LEVEL_PROBES.get(v, [v.lower()[:20]]), window, full)}")
        for v in sorted(lp - lg):
            print(f"    [FP] {v}: literal={'yes' if v.lower()[:18] in full else 'no'}")
    if "service" in e:
        sg, sp = e["service"]
        print(f"  SERVICE gold={sorted(sg)}")
        print(f"          pred={sorted(sp) or '(empty)'}")
        for v in sorted(sg - sp):
            probes = SERVICE_PROBES.get(v, [v.lower()[:20]])
            print(f"    [FN] {v[:60]}: {probe_hit(probes, window, full)}")
        for v in sorted(sp - sg):
            print(f"    [FP] {v[:60]}: literal={'yes' if v.lower()[:18] in full else 'no'}")
    if "taxa" in e:
        tg, tp = e["taxa"]
        print(f"  TAXA    gold={sorted(tg)[:6]}")
        print(f"          pred={sorted(tp)[:6] or '(empty)'}")
        for v in sorted(tg - tp)[:4]:
            probe = v.lower()[: min(len(v), 22)]
            print(f"    [FN] {v[:40]}: {probe_hit([probe], window, full)}")
        for v in sorted(tp - tg)[:4]:
            print(f"    [FP] {v[:40]}: literal={'yes' if v.lower()[:18] in full else 'no'}")
    if "ecotype" in e:
        cg, cp = e["ecotype"]
        print(f"  ECOTYPE gold={sorted(cg)[:4]}")
        print(f"          pred={sorted(cp)[:4] or '(empty)'}")
        for v in sorted(cg - cp)[:3]:
            print(f"    [FN] {v[:60]}: {probe_hit([v.lower()[:20]], window, full)}")
        for v in sorted(cp - cg)[:3]:
            print(f"    [FP] {v[:60]}: literal={'yes' if v.lower()[:18] in full else 'no'}")
