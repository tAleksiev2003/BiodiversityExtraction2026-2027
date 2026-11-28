#!/usr/bin/env python3
"""
========================================================================================
EXECUTIVE SUMMARY: Master Experiment Matrix Generator (all architectures, Dev-100)
========================================================================================
- Purpose:
    Scans every `outputs/<category>/*/*_metrics.json` for completed 100-paper Dev runs
    and writes a clean, report-ready Markdown master table to `docs/experiment_matrix.md`.
    Table 1 keeps the full Architecture-3 detail (all 9 per-field F1 scores, decoded
    parser/flag columns, artifact paths). Table 2 lists EVERY run across all
    architectures (classical baselines, monolithic, chunking, sections experiments,
    ensembles) with C1, C1+C2b, Micro, runtime, seed, and run directory.
- Results on Dev Set: meta-script, no runs of its own.
========================================================================================
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_MD = REPO_ROOT / "docs/experiment_matrix.md"

ARCH3_PREFIX = "llm_rag_sections"

FLAG_MAP = {
    "full": "--full-text",
    "rawft": "--raw-fulltext",
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
    "lvlf": "--level-fold",
    "b64k": "--budget-chars 64000",
    "svcg": "--service-gate",
    "gpn": "--german-plural-norm",
    "pgn": "--german-pairing-names",
    "ectl": "--eco-type-land-rule",
}


def parse_arch3(run_id: str) -> tuple[str, str, str]:
    """Splits an Arch-3 run_id into (model, parser, flags_str)."""
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
    if tokens and tokens[0] in ("grobid", "marker", "docling", "autoparser", "minhjson"):
        parser = {"grobid": "GROBID", "marker": "Marker",
                  "docling": "Docling Light", "autoparser": "AutoParser",
                  "minhjson": "MinhDoclingJSON"}[tokens[0]]
        tokens = tokens[1:]
    flags = " ".join(FLAG_MAP.get(t, t) for t in tokens) if tokens else "*(baseline)*"
    return model, parser, flags


def categorize(run_id: str, category_dir: str) -> tuple[str, str]:
    """Returns (category, model label) for any run_id."""
    if run_id.startswith("llm_rag_sections"):
        model, _, _ = parse_arch3(run_id)
        return "Architecture 3 (RAG sections)", model
    if category_dir == "base_models" or run_id.endswith("_baseline"):
        pretty = run_id.replace("_baseline", "").replace("_", " ")
        return "Classical baseline", pretty
    if run_id.startswith("llm_zeroshot_guidelines"):
        model = "HU / hu-reasoning" if "hu" in run_id else "SAIA / Mistral-Large"
        return "Architecture 1 (monolithic)", model
    if run_id.startswith("llm_parallel_chunking"):
        model = "SAIA / DeepSeek-4.1" if "deepseek" in run_id else (
            "HU / hu-reasoning" if "hu" in run_id else "SAIA / Mistral-Large")
        return "Architecture 2 (parallel chunks)", model
    if run_id.startswith(("llm_parallel_chunks", "llm_sequential_chunks")):
        return "Architecture 2 (2nd gen chunks)", "HU / hu-reasoning"
    if run_id.startswith("llm_sections_"):
        arch = "parallel" if "parallel" in run_id else "sequential"
        lang = "EN" if run_id.endswith(("_en",)) or "_en" in run_id else "DE"
        return f"Sections experiment ({arch}, {lang})", run_id
    if "ensemble" in run_id:
        return "Ensemble", run_id
    return "Other", run_id


def fmt_secs(v) -> str:
    try:
        return f"{float(v):.0f}s"
    except (TypeError, ValueError):
        return "—"


def fmt_seed(v) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return str(v) if v is not None else "—"


def collect() -> list[dict]:
    rows = []
    for cat_dir in ("architectures", "base_models"):
        root = REPO_ROOT / "outputs" / cat_dir
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            mf = d / f"{d.name}_metrics.json"
            if not mf.exists():
                continue
            try:
                m = json.loads(mf.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if m.get("num_papers", 0) < 100:
                continue
            f = m.get("fields", {})
            row = {
                "run_id": d.name,
                "category_dir": cat_dir,
                "c1": m.get("c1_macro_f1"),
                "c1_c2b": m.get("c2b_macro_f1"),
                "micro": m.get("micro_f1"),
            }
            if row["c1"] is None:
                continue
            for key, fname in [("hab", "habitat"), ("lvl", "biodiversity_level"),
                               ("ecat", "ecosystem_type.category"),
                               ("eterm", "ecosystem_type.term"),
                               ("sgrp", "taxa.species_group"),
                               ("tsci", "taxa.scientific_name"),
                               ("tger", "taxa.german_name"),
                               ("scat", "ecosystem_service.category"),
                               ("sterm", "ecosystem_service.term")]:
                row[key] = (f.get(fname, {}) or {}).get("f1")
            ri = d / f"{d.name}_run_info.json"
            info = {}
            if ri.exists():
                try:
                    info = json.loads(ri.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    info = {}
            row["runtime"] = fmt_secs(info.get("runtime_seconds"))
            row["seed"] = fmt_seed(info.get("seed"))
            row["commit"] = str(info.get("git_commit", "—"))[:7]
            rows.append(row)
    return rows


def build_markdown(rows: list[dict]) -> str:
    arch3 = [r for r in rows if r["run_id"].startswith(ARCH3_PREFIX)]
    arch3_sorted = sorted(arch3, key=lambda r: (-r["c1"], -(r["c1_c2b"] or 0)))
    lines = [
        "# Master Experiment Matrix (Dev-100)",
        "",
        "Complete overview of all finished 100-paper runs on `data/gold/dev_100_ids.txt`,",
        "sorted by **C1 Macro-F1** (official ranking score). Table 1 details every Architecture-3",
        "run with all 9 per-field F1 scores. Table 2 lists every run across all architectures.",
        "",
        "## 1. Architecture 3 in detail (all finished 100-paper runs)",
        "",
        "| # | Model | Parser | Change / Flags | C1 | C1+C2b | Micro | Hab | BioLvl | EcoCat | EcoTerm | SpGrp | SciName | GerName | SrvCat | SrvTerm | Run dir (`outputs/architectures/`) |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, r in enumerate(arch3_sorted, 1):
        model, parser, flags = parse_arch3(r["run_id"])

        def v(k):
            return f"{r[k]:.3f}" if r[k] is not None else "—"

        lines.append(
            f"| {idx} | {model} | {parser} | `{flags}` | **{r['c1']:.4f}** | {r['c1_c2b']:.4f} | {r['micro']:.4f} | "
            f"{v('hab')} | {v('lvl')} | {v('ecat')} | {v('eterm')} | {v('sgrp')} | "
            f"{v('tsci')} | {v('tger')} | {v('scat')} | {v('sterm')} | `{r['run_id']}` |"
        )

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
        "## 2. Best score per metric / field (where does what work best?)",
        "",
        "| Metric / Field | Best F1 | Winner run | Parser & Flags |",
        "|---|---:|---|---|",
    ]
    for k, label in field_keys:
        cand = [r for r in arch3 if r[k] is not None]
        if not cand:
            continue
        best = max(cand, key=lambda r: r[k])
        model, parser, flags = parse_arch3(best["run_id"])
        lines.append(f"| `{label}` | **{best[k]:.4f}** | `{best['run_id']}` | {model}, {parser}, `{flags}` |")

    lines += [
        "",
        "## 3. All runs across all architectures",
        "",
        "| # | Model | Category | C1 | C1+C2b | Micro | Runtime | Seed | Run dir (`outputs/`) |",
        "|---:|---|---|---:|---:|---:|---:|---|---|",
    ]
    for idx, r in enumerate(sorted(rows, key=lambda r: -r["c1"]), 1):
        cat, model = categorize(r["run_id"], r["category_dir"])
        c2b = f"{r['c1_c2b']:.4f}" if r["c1_c2b"] is not None else "—"
        mic = f"{r['micro']:.4f}" if r["micro"] is not None else "—"
        lines.append(
            f"| {idx} | {model} | {cat} | **{r['c1']:.4f}** | {c2b} | {mic} | "
            f"{r['runtime']} | {r['seed']} | `{r['category_dir']}/{r['run_id']}` |"
        )

    lines += [
        "",
        "## 4. Offline router & consensus reference values (0 API calls)",
        "",
        "| Variant | C1 | C1+C2b | Micro | Setup |",
        "|---|---:|---:|---:|---|",
        "| **SAIA 2-rule router (ev+ctaxa)** | **0.6056** | 0.4469 | 0.4424 | Marker if `g_divs<=3` or `g_ratio<0.65`, else GROBID |",
        "| **6-run consensus (3-of-6)** | 0.6039 | 0.4515 | **0.4590** | `saia_grobid_ev`, `saia_marker_ev`, `saia_plq`, `saia_s5`, `hu_grobid`, `hu_marker` |",
        "| **4-run consensus (3-of-4, +TaxonNorm)** | 0.5932 | **0.4687** | 0.4496 | `saia_grobid`, `saia_marker`, `hu_grobid`, `hu_marker` |",
        "",
        "*(Regenerate on `lenlenovo` after new runs: `python3 scripts/generate_experiment_matrix.py`.)*",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = collect()
    OUT_MD.write_text(build_markdown(rows), encoding="utf-8")
    print(f"Wrote {len(rows)} finished runs to {OUT_MD}")


if __name__ == "__main__":
    main()
