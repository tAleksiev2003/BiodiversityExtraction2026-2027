#!/usr/bin/env python3
"""
========================================================================================
EXECUTIVE SUMMARY: Experiment Matrix Generator (Architecture 3 — Dev-100)
========================================================================================
- Purpose:
    Scans `outputs/architectures/llm_rag_sections_*/*_metrics.json` for all completed
    100-paper Dev runs and writes a clean, report-ready Markdown master table to
    `docs/experiment_matrix.md`, including overall metrics, all 9 per-field F1 scores,
    decoded parser/flag columns, and direct paths to each run's artifact directory.
========================================================================================
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCH_DIR = REPO_ROOT / "outputs/architectures"
OUT_MD = REPO_ROOT / "docs/experiment_matrix.md"


def parse_run_id(run_id: str) -> tuple[str, str, str]:
    """Splits a run_id into (provider_model, parser, flags_str)."""
    if "_saia_deepseek-v4.1-flash-1017" in run_id:
        model = "SAIA / DeepSeek-4.1"
        rest = run_id.split("_saia_deepseek-v4.1-flash-1017", 1)[1].lstrip("_")
    elif "_hu_hu-reasoning" in run_id:
        model = "HU / hu-reasoning"
        rest = run_id.split("_hu_hu-reasoning", 1)[1].lstrip("_")
    else:
        model = "Other"
        rest = run_id

    tokens = [t for t in rest.split("_") if t]
    parser = "PyMuPDF"
    if tokens and tokens[0] in ("grobid", "marker", "docling", "autoparser"):
        p_map = {"grobid": "GROBID", "marker": "Marker", "docling": "Docling", "autoparser": "AutoParser"}
        parser = p_map[tokens[0]]
        tokens = tokens[1:]

    flag_map = {
        "full": "--full-text",
        "focus": "--focus-prompt",
        "detox": "--detox-rag",
        "role": "--role-budget",
        "ev": "--evidence-gates",
        "ctaxa": "--clean-taxa",
        "s5": "--self-check-reasoning",
        "plq": "--per-label-quotes",
        "ecomap": "--eco-category-map",
        "lvl": "--level-prompt",
        "v4": "--v4-prompt",
        "biknn": "--bilingual-knn",
    }
    flags = " ".join(flag_map.get(t, t) for t in tokens) if tokens else "*(baseline)*"
    return model, parser, flags


def build_markdown(rows: list[dict]) -> str:
    rows_sorted = sorted(rows, key=lambda r: (-r["c1"], -r["c1_c2b"]))
    lines = [
        "# Master Experiment Matrix — Architecture 3 (Dev-100)",
        "",
        "Vollständige Übersicht aller abgeschlossenen 100-Paper-Runs auf `data/gold/dev_100_ids.txt`,",
        "sortiert nach **C1 Macro-F1** (offizieller Ranking-Score), inklusive aller 9 Einzelfeld-F1-Werte",
        "und direktem Verweis auf das jeweilige Run-Verzeichnis (`outputs/architectures/<run_id>/`).",
        "",
        "## 1. Gesamttabelle (alle abgeschlossenen 100-Paper-Runs)",
        "",
        "| # | Modell | Parser | Änderungen / Flags | C1 | C1+C2b | Micro | Hab | BioLvl | EcoCat | EcoTerm | SpGrp | SciName | GerName | SrvCat | SrvTerm | Run-Ordner (`outputs/architectures/`) |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, r in enumerate(rows_sorted, 1):
        model, parser, flags = parse_run_id(r["run_id"])
        lines.append(
            f"| {idx} | {model} | {parser} | `{flags}` | **{r['c1']:.4f}** | {r['c1_c2b']:.4f} | {r['micro']:.4f} | "
            f"{r['hab']:.3f} | {r['lvl']:.3f} | {r['ecat']:.3f} | {r['eterm']:.3f} | {r['sgrp']:.3f} | "
            f"{r['tsci']:.3f} | {r['tger']:.3f} | {r['scat']:.3f} | {r['sterm']:.3f} | `{r['run_id']}` |"
        )

    # Best-per-field summary
    field_keys = [
        ("c1", "C1 Macro-F1"),
        ("c1_c2b", "C1+C2b Macro-F1"),
        ("micro", "Micro-F1"),
        ("hab", "habitat"),
        ("lvl", "biodiversity_level"),
        ("ecat", "ecosystem_type.category"),
        ("eterm", "ecosystem_type.term"),
        ("sgrp", "taxa.species_group"),
        ("tsci", "taxa.scientific_name"),
        ("tger", "taxa.german_name"),
        ("scat", "ecosystem_service.category"),
        ("sterm", "ecosystem_service.term"),
    ]
    lines += [
        "",
        "## 2. Bestwert je Metrik / Einzelfeld (wo funktioniert was am besten?)",
        "",
        "| Metrik / Feld | Bester F1 | Gewinner-Run | Parser & Flags |",
        "|---|---:|---|---|",
    ]
    for k, label in field_keys:
        best = max(rows, key=lambda r: r[k])
        model, parser, flags = parse_run_id(best["run_id"])
        lines.append(f"| `{label}` | **{best[k]:.4f}** | `{best['run_id']}` | {model}, {parser}, `{flags}` |")

    lines += [
        "",
        "## 3. Offline-Router & Konsens-Referenzwerte (0 API-Calls)",
        "",
        "| Variante | C1 | C1+C2b | Micro | Zusammensetzung |",
        "|---|---:|---:|---:|---|",
        "| **SAIA 2-Regel-Router (ev+ctaxa)** | **0.6056** | 0.4469 | 0.4424 | Marker bei `g_divs<=3` oder `g_ratio<0.65`, sonst GROBID |",
        "| **6-Run Konsens (3-of-6)** | 0.6039 | 0.4515 | **0.4590** | `saia_grobid_ev`, `saia_marker_ev`, `saia_plq`, `saia_s5`, `hu_grobid`, `hu_marker` |",
        "| **4-Run Konsens (3-of-4, +TaxonNorm)** | 0.5932 | **0.4687** | 0.4496 | `saia_grobid`, `saia_marker`, `hu_grobid`, `hu_marker` |",
        "",
        "## 4. Aktuell laufende / eingereihte `v4`-Läufe (SAIA & HU)",
        "",
        "- `*_autoparser_ctaxa_v4` *(läuft)*",
        "- `*_autoparser_full_ctaxa_v4` *(eingereiht)*",
        "- `*_autoparser_ctaxa_v4_biknn` *(eingereiht)*",
        "- `*_docling_full_ctaxa_v4` *(eingereiht)*",
        "",
        "*(Nach Abschluss einfach `python3 scripts/generate_experiment_matrix.py` auf `lenlenovo` ausführen, um diese Tabelle automatisch zu aktualisieren.)*",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = []
    for d in sorted(ARCH_DIR.glob("llm_rag_sections_*")):
        mf = d / f"{d.name}_metrics.json"
        if not mf.exists():
            continue
        m = json.loads(mf.read_text(encoding="utf-8"))
        if m.get("num_papers", 0) < 100:
            continue
        f = m["fields"]
        rows.append({
            "run_id": d.name,
            "c1": m["c1_macro_f1"],
            "c1_c2b": m["c2b_macro_f1"],
            "micro": m["micro_f1"],
            "hab": f["habitat"]["f1"],
            "lvl": f["biodiversity_level"]["f1"],
            "ecat": f["ecosystem_type.category"]["f1"],
            "eterm": f["ecosystem_type.term"]["f1"],
            "sgrp": f["taxa.species_group"]["f1"],
            "tsci": f["taxa.scientific_name"]["f1"],
            "tger": f["taxa.german_name"]["f1"],
            "scat": f["ecosystem_service.category"]["f1"],
            "sterm": f["ecosystem_service.term"]["f1"],
        })
    OUT_MD.write_text(build_markdown(rows), encoding="utf-8")
    print(f"Wrote {len(rows)} completed runs to {OUT_MD}")


if __name__ == "__main__":
    main()
