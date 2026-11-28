#!/usr/bin/env python3
"""
========================================================================================
EXECUTIVE SUMMARY: Architecture 3 — RAG-Anchored Section Routing Pipeline
========================================================================================
- Purpose:
    A clever, token-efficient hybrid extraction pipeline that avoids dumping noisy
    full-text PDFs while solving the cross-section context loss of blind chunking.

- Pipeline Stages:
    1. Structural Section Routing (0 LLM tokens):
       Uses `extract_semantic_sections` (PyMuPDF) to keep only high-signal sections
       (Abstract, Study Area & Methods, Results & Findings), discarding References,
       Acknowledgments, and noisy boilerplate (~50-60% token reduction).
    2. Dynamic kNN Few-Shot Anchoring (TF-IDF over 908 Train Papers):
       Retrieves the Top-2 most similar training papers for each Dev document and
       injects their Gold JSON labels as formatting and convention anchors (teaches
       Auen->Fließende Gewässer, collective Latin terms, and sparse-field restraint).
    3. Single Focused Extraction Call:
       Runs 1 precision-focused call with full field guidelines + 46/67 term enums
       using `RobustOpenAICompatClient` (with exponential backoff & checkpointing).
    4. Deterministic Schema Normalization & Micro-Cleanup:
       Normalizes enums, enforces collective-term '-' rules, and packages CodaBench ZIP.

- Expected Runtime:
    100 API calls total (~5-10 min with `deepseek-v4.1-flash-1017` on SAIA, ~45 min
    with `hu-reasoning` on HU Bridge).
========================================================================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluator import (
    evaluate_predictions,
    load_jsonl_by_id,
    print_report,
    save_reproducible_run,
)
from src.models.architectures.llm_parallel_chunking import (
    extract_grobid_chunks,
    micronormalize_taxon_name,
    verify_verbatim_quote,
)
from src.models.architectures.llm_sections_pipeline import (
    RobustOpenAICompatClient,
    extract_semantic_sections,
)
from src.models.architectures.llm_zeroshot_guidelines import (
    DryRunMockClient,
    PROVIDERS,
    SYSTEM_PROMPT,
    extract_json_block,
    format_field_spec,
    load_field_spec,
    normalize_prediction,
)
from src.models.base_models.majority_baseline import fit_majority_model

MODEL_BASE_ID = "llm_rag_sections"
CATEGORY = "architectures"
SEED = 42


def _load_corpus_text(zot_id: str, rec: dict[str, Any], text_dir: Path) -> str:
    """Loads pre-parsed text if available, otherwise falls back to title + abstract."""
    txt_file = text_dir / f"{zot_id}.txt"
    if txt_file.exists():
        return txt_file.read_text(encoding="utf-8", errors="ignore")[:12000]
    return f"{rec.get('bibtex_title') or ''} {rec.get('bibtex_abstractNote') or ''}".strip()


class TrainKNNRetriever:
    """Lightweight TF-IDF retriever over the 908 training papers for dynamic few-shot."""

    def __init__(
        self,
        train_records: dict[str, dict[str, Any]],
        train_text_dir: Path,
        detox_rag: bool = False,
        bilingual_knn: bool = False,
    ):
        self.train_ids = sorted(train_records.keys())
        self.train_records = train_records
        self.detox_rag = detox_rag
        docs = [_load_corpus_text(zid, train_records[zid], train_text_dir) for zid in self.train_ids]
        if detox_rag or bilingual_knn:
            # Bilingual char n-grams match both German/English text and Latin taxon stems
            self.vectorizer = TfidfVectorizer(
                max_features=20000, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True
            )
        else:
            self.vectorizer = TfidfVectorizer(max_features=15000, stop_words="english", sublinear_tf=True)
        self.train_matrix = self.vectorizer.fit_transform(docs)

    def retrieve_top_k(self, query_text: str, k: int = 2, exclude_id: str | None = None) -> list[dict[str, Any]]:
        q_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(q_vec, self.train_matrix).ravel()
        if exclude_id and exclude_id in self.train_ids:
            # Required when scoring train-split papers: never retrieve the paper itself.
            sims[self.train_ids.index(exclude_id)] = -1e9
        top_idx = np.argsort(-sims)[:k]
        examples = []
        for idx in top_idx:
            zid = self.train_ids[int(idx)]
            rec = self.train_records[zid]
            taxa_val = rec.get("taxa", [])
            if self.detox_rag and isinstance(taxa_val, list):
                # Strip specific species names from anchors to prevent copying neighbor taxa
                taxa_val = [
                    {"species_group": t.get("species_group"), "collective_term": t.get("collective_term")}
                    for t in taxa_val
                    if isinstance(t, dict) and t.get("species_group")
                ]
            gold_compact = {
                "habitat": rec.get("habitat", []),
                "biodiversity_level": rec.get("biodiversity_level", []),
                "ecosystem_type": rec.get("ecosystem_type", []),
                "taxa": taxa_val,
                "ecosystem_service": rec.get("ecosystem_service", []),
            }
            examples.append(
                {
                    "title": str(rec.get("bibtex_title") or "")[:180],
                    "bibtex_type": str(rec.get("bibtex_type") or "unknown"),
                    "gold_json": gold_compact,
                }
            )
        return examples


def _section_role_weight(name: str) -> float:
    """Assigns generic role weights (Methods/Results prioritized over Intro/Background)."""
    low = name.lower()
    if any(k in low for k in ("method", "study area", "material", "untersuchungsgebiet", "erhebung")):
        return 0.35
    if any(k in low for k in ("result", "ergebnis", "finding", "befund")):
        return 0.35
    if any(k in low for k in ("abstract", "title")):
        return 0.15
    if any(k in low for k in ("discussion", "diskussion", "conclusion", "fazit")):
        return 0.15
    if any(k in low for k in ("intro", "einleitung", "background", "hintergrund")):
        return 0.10
    return 0.20


def build_rag_user_prompt(
    rec: dict[str, Any],
    sections: list[dict[str, Any]],
    few_shot_examples: list[dict[str, Any]],
    field_spec: dict[str, Any],
    max_total_chars: int = 16000,
    role_budget: bool = False,
    evidence_gates: bool = False,
    full_text: bool = False,
    focus_prompt: bool = False,
    train_rates_note: str = "",
    self_check_reasoning: bool = False,
    per_label_quotes: bool = False,
    eco_category_map_str: str = "",
) -> str:
    """Builds a compact, RAG-anchored user prompt with filtered sections and few-shot gold."""
    title = str(rec.get("bibtex_title") or "").strip()

    # 1. Condense few-shot anchors
    examples_block = []
    for i, ex in enumerate(few_shot_examples, 1):
        examples_block.append(
            f"Example {i} (Title: {ex['title']}):\n{json.dumps(ex['gold_json'], ensure_ascii=False)}"
        )
    examples_str = "\n\n".join(examples_block)

    # 2. Pack semantic sections (untruncated when full_text=True)
    sec_blocks = []
    if full_text:
        for sec in sections:
            sec_blocks.append(f"### SECTION: {sec['section_name']}\n{sec['text']}")
    elif role_budget:
        budget_total = 28000
        weights = [_section_role_weight(s["section_name"]) for s in sections]
        w_sum = sum(weights) or 1.0
        for sec, w in zip(sections, weights):
            sec_cap = max(2500, int(budget_total * (w / w_sum)))
            sec_blocks.append(f"### SECTION: {sec['section_name']}\n{sec['text'][:sec_cap]}")
    else:
        budget_per_sec = max(2500, max_total_chars // max(1, len(sections)))
        for sec in sections:
            sec_blocks.append(f"### SECTION: {sec['section_name']}\n{sec['text'][:budget_per_sec]}")
    doc_text = "\n\n".join(sec_blocks)

    # 3. Term enums reminder (high-leverage E1 finding)
    eco_terms = field_spec["ecosystem_type"]["subfields"]["term"]["values"]
    srv_terms = field_spec["ecosystem_service"]["subfields"]["term"]["values"]
    eco_block = (
        f"ALLOWED ECOSYSTEM_TYPE TERMS BY CATEGORY (pair each term ONLY with its matching category):\n{eco_category_map_str}"
        if eco_category_map_str
        else f"ALLOWED ECOSYSTEM_TYPE.TERM VALUES:\n{json.dumps(eco_terms, ensure_ascii=False)}"
    )

    evidence_note = ""
    if evidence_gates and not per_label_quotes:
        evidence_note = (
            "\n\nEVIDENCE REQUIREMENT FOR SPARSE FIELDS:\n"
            "If you output `ecosystem_service` or `biodiversity_level`, also include an optional "
            "`_evidence` object with short verbatim quotes from the document proving them:\n"
            '`"_evidence": {"ecosystem_service": "<verbatim quote>", "biodiversity_level": ["<verbatim quote>"]}`\n'
            "If no verbatim evidence exists in the text, leave those fields as [].\n"
        )

    if focus_prompt:
        task_instruction = (
            "Extract what THIS document is mainly about (applies to ALL document types: empirical studies, "
            "reports, guidelines, policy papers, and books — never leave fields empty just because it is not a study; "
            "ignore mere background citations). Output at most 10 focal taxa."
            f"{train_rates_note}"
        )
    else:
        task_instruction = "Extract the main studied/focused attributes of THIS document (ignore background citations)."

    reasoning_note = ""
    if self_check_reasoning:
        reasoning_note = (
            "\n\nTAXA SPELLING & REASONING SELF-CHECK (R7 + S1-S5):\n"
            "- Scientific names WITHOUT author names/years (e.g. 'Fagus sylvatica', NOT 'Fagus sylvatica L.').\n"
            "- If only genus is known, write 'Genus spp.' (with period). At most 10 focal taxa; if >10, give higher group(s).\n"
            "- German names capitalized with normal German spelling (e.g. 'Rot-Buche').\n"
            "- Before final fields, include a compact `\"reasoning\"` string (max 120 words): "
            "'S1 (habitat/eco): ... S2 (taxa): ... S3 (bio_level): ... S4 (services): ... "
            "S5 (self-check: verify allowed list, main subject, exact text support; drop failures).' "
        )

    output_note = " Return a single valid JSON object matching the schema."
    if per_label_quotes:
        output_note = (
            "\n\nOUTPUT FORMAT (PER-LABEL VERBATIM QUOTES):\n"
            "Every single label MUST carry a verbatim `\"quote\"` (3 to 60 words copied character-by-character from TARGET DOCUMENT above).\n"
            "Return ONE JSON object only:\n"
            '{"reasoning": "S1: ... S2: ... S3: ... S4: ... S5: ...", "labels": ['
            '{"field": "habitat", "value": "<allowed>", "quote": "<3-60 words>"}, '
            '{"field": "ecosystem_type", "category": "<allowed>", "term": "<allowed or null>", "quote": "<3-60 words>"}, '
            '{"field": "biodiversity_level", "value": "<allowed>", "quote": "<3-60 words>"}, '
            '{"field": "taxa", "species_group": "<allowed>", "scientific_name": "<name or null>", "german_name": "<name or null>", "collective_term": false, "quote": "<3-60 words>"}, '
            '{"field": "ecosystem_service", "category": "<allowed>", "term": "<allowed or null>", "quote": "<3-60 words>"}'
            "]}"
        )

    return (
        f"SIMILAR ANNOTATED EXAMPLES (follow their exact label conventions & sparsity; NEVER copy taxa from examples):\n"
        f"{examples_str}\n\n"
        f"{eco_block}\n\n"
        f"ALLOWED ECOSYSTEM_SERVICE.TERM VALUES:\n{json.dumps(srv_terms, ensure_ascii=False)}\n\n"
        f"TARGET DOCUMENT\n"
        f"Title: {title}\n\n"
        f"{doc_text}\n\n"
        f"{task_instruction}"
        f"{reasoning_note}{evidence_note}{output_note}"
    )


_DOCLING_CONVERTER: Any = None


def _get_docling_converter() -> Any:
    """Lazily initializes one Docling DocumentConverter (no OCR, fast CPU)."""
    global _DOCLING_CONVERTER
    if _DOCLING_CONVERTER is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True, document_timeout=120.0)
        _DOCLING_CONVERTER = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
    return _DOCLING_CONVERTER


def _split_markdown_to_sections(
    md_text: str,
    zot_id: str,
    rec: dict[str, Any],
    pdf_dir: Path,
) -> list[dict[str, Any]]:
    """Splits Markdown by headings, strips References/Appendices, and groups into <= 4 blocks."""
    if not md_text.strip():
        return extract_semantic_sections(zot_id, rec, pdf_dir)

    stop_re = re.compile(
        r"^(references|bibliography|literatur(?:verzeichnis)?|acknowledg?ments?|danksagung|appendix|anhang)\b",
        re.IGNORECASE,
    )
    raw_sections: list[tuple[str, str]] = []
    cur_head = "Abstract & Introduction"
    cur_lines: list[str] = []
    for line in md_text.splitlines():
        m = re.match(r"^#{1,3}\s+(.*)$", line.strip())
        if m:
            body = "\n".join(cur_lines).strip()
            if len(body) > 100:
                raw_sections.append((cur_head, body))
            cur_head = m.group(1).strip()
            cur_lines = []
            if stop_re.match(cur_head):
                break
        else:
            cur_lines.append(line)
    else:
        body = "\n".join(cur_lines).strip()
        if len(body) > 100:
            raw_sections.append((cur_head, body))

    if not raw_sections:
        return extract_semantic_sections(zot_id, rec, pdf_dir)

    if len(raw_sections) <= 4:
        return [{"section_name": h[:60], "text": t} for h, t in raw_sections]

    step = max(1, (len(raw_sections) + 3) // 4)
    grouped: list[dict[str, Any]] = []
    for i in range(0, len(raw_sections), step):
        chunk = raw_sections[i : i + step]
        names = " / ".join(h[:25] for h, _ in chunk)
        text = "\n\n".join(f"## {h}\n{t}" for h, t in chunk)
        grouped.append({"section_name": names[:60], "text": text})
    return grouped[:4]


def extract_markdown_cached_sections(
    zot_id: str,
    rec: dict[str, Any],
    pdf_dir: Path,
    cache_subdir: str,
    convert_fn: Any = None,
) -> list[dict[str, Any]]:
    """Loads cached Markdown from `data/parsed_text/<cache_subdir>/` or runs `convert_fn`."""
    cache_dir = pdf_dir.parent.parent / "parsed_text" / cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{zot_id}.md"
    pdf_path = pdf_dir / f"{zot_id}.pdf"

    md_text = ""
    if cache_file.exists():
        md_text = cache_file.read_text(encoding="utf-8", errors="ignore")
    elif convert_fn is not None and pdf_path.exists():
        try:
            md_text = convert_fn(pdf_path)
            if md_text.strip():
                cache_file.write_text(md_text, encoding="utf-8")
        except Exception as exc:
            print(f"[{cache_subdir} fallback for {zot_id}: {exc}]", flush=True)

    return _split_markdown_to_sections(md_text, zot_id, rec, pdf_dir)


def extract_docling_sections(
    zot_id: str,
    rec: dict[str, Any],
    pdf_dir: Path,
) -> list[dict[str, Any]]:
    """Uses cached Docling Markdown (`docling_dev/` or `docling_train/`) or runs Docling on the fly."""
    cache_subdir = "docling_train" if pdf_dir.name == "train" else "docling_dev"
    return extract_markdown_cached_sections(
        zot_id,
        rec,
        pdf_dir,
        cache_subdir,
        lambda p: _get_docling_converter().convert(str(p)).document.export_to_markdown(),
    )


def extract_minh_json_sections(
    zot_id: str,
    rec: dict[str, Any],
    pdf_dir: Path,
) -> list[dict[str, Any]]:
    """Builds sections from Minh's full Docling JSON (`data/processed/docling/dev/`).

    Walks the document body in reading order, starts a new section on every
    `section_header`, and prefixes text with `[page N]` markers when the page
    changes (same markers Minh's own pipeline uses for quote tracing).
    Footers are skipped; over-long section lists are grouped into <= 4 blocks.
    Falls back to the AutoParser router when the JSON is missing or unreadable.
    """
    try:
        doc = json.loads(
            (REPO_ROOT / "data" / "processed" / "docling" / "dev" / f"{zot_id}.json").read_text(
                encoding="utf-8"
            )
        )
        texts = {t.get("self_ref"): t for t in doc.get("texts", []) if isinstance(t, dict)}
        raw_sections: list[tuple[str, list[str]]] = []
        cur_head, cur_lines, cur_page = "Front matter", [], None
        for child in doc.get("body", {}).get("children", []):
            ref = child.get("$ref") if isinstance(child, dict) else None
            item = texts.get(ref) if ref else None
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            if label == "page_footer":
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            if label == "section_header":
                body = " ".join(cur_lines).strip()
                if len(body) > 100:
                    raw_sections.append((cur_head, [body]))
                cur_head = text[:80]
                cur_lines = []
                continue
            prov = item.get("prov") or []
            page = prov[0].get("page_no") if prov and isinstance(prov[0], dict) else None
            if page != cur_page and page is not None:
                cur_lines.append(f"[page {page}]")
                cur_page = page
            cur_lines.append(text)
        tail = " ".join(cur_lines).strip()
        if len(tail) > 100:
            raw_sections.append((cur_head, [tail]))
        if len(raw_sections) <= 4:
            return [{"section_name": h[:60], "text": t[0]} for h, t in raw_sections]
        step = max(1, (len(raw_sections) + 3) // 4)
        grouped: list[dict[str, Any]] = []
        for i in range(0, len(raw_sections), step):
            chunk = raw_sections[i : i + step]
            names = " / ".join(h[:25] for h, _ in chunk)
            text = "\n\n".join(f"## {h}\n{t[0]}" for h, t in chunk)
            grouped.append({"section_name": names[:60], "text": text})
        return grouped[:4]
    except Exception as exc:
        print(f"[minh-json fallback for {zot_id}: {exc}]", flush=True)
        return extract_auto_routed_sections(
            zot_id, rec, pdf_dir, "http://127.0.0.1:8070"
        )


def extract_marker_sections(
    zot_id: str,
    rec: dict[str, Any],
    pdf_dir: Path,
) -> list[dict[str, Any]]:
    """Uses cached Marker Markdown (`marker_dev/` or `marker_train/`), falling back to PyMuPDF if missing."""
    cache_subdir = "marker_train" if pdf_dir.name == "train" else "marker_dev"
    return extract_markdown_cached_sections(zot_id, rec, pdf_dir, cache_subdir, None)


def build_german_vocab(train_records: dict[str, dict[str, Any]]) -> set[str]:
    """Collects every german_name the database ever used (train gold only)."""
    vocab = set()
    for rec in train_records.values():
        for tax in rec.get("taxa") or []:
            if isinstance(tax, dict) and tax.get("german_name"):
                vocab.add(str(tax["german_name"]).strip())
    return vocab


def apply_german_plural_norm(norm: dict[str, Any], vocab: set[str]) -> None:
    """
    Maps a predicted German taxa name that is NOT in the train vocabulary to its
    unique vocabulary variant (e.g. 'Seehund' -> 'Seehunde'). Smallest safe
    adjustment: only fires when exactly ONE candidate exists in the vocabulary,
    never touches names already in it (so per-name conventions like the
    singular-only 'Kegelrobbe' are preserved). Train-derived, no dev fitting.
    """
    for tax in norm.get("taxa") or []:
        name = tax.get("german_name")
        if not name or name in vocab:
            continue
        stems = {name}
        for suf in ("e", "n", "en"):
            if name.endswith(suf) and len(name) > len(suf) + 3:
                stems.add(name[: -len(suf)])
        candidates = set()
        umlauted = {
            stem[:i] + {"a": "ä", "o": "ö", "u": "ü"}[ch] + stem[i + 1:]
            for stem in stems
            for i, ch in enumerate(stem)
            if ch in "aou"
        }  # single-vowel umlaut only ("Baum"->"Bäum"+e, never "Bäüm")
        for stem in stems | umlauted:
            for variant in (stem + "e", stem + "n", stem + "en", stem + "s", stem):
                if variant != name and variant in vocab:
                    candidates.add(variant)
        if len(candidates) == 1:
            tax["german_name"] = candidates.pop()


def apply_clean_taxa(raw_pred: dict[str, Any], norm: dict[str, Any]) -> None:
    """
    Removes fabricated '-' / species_group fallbacks when the LLM left fields null/empty,
    and canonicalizes author epithets & spp. formatting without inventing labels.
    """
    raw_taxa = raw_pred.get("taxa") or []
    cleaned = []
    for idx, t in enumerate(norm.get("taxa") or []):
        raw_t = raw_taxa[idx] if idx < len(raw_taxa) and isinstance(raw_taxa[idx], dict) else {}
        sci = t.get("scientific_name")
        ger = t.get("german_name")
        # Drop fabricated '-' unless model explicitly emitted '-'
        if sci == "-" and str(raw_t.get("scientific_name") or "").strip() != "-":
            sci = None
        elif sci and sci != "-":
            sci = micronormalize_taxon_name(sci)
        # Drop fabricated species_group echo in german_name unless model explicitly emitted it
        if ger == t.get("species_group") and not str(raw_t.get("german_name") or "").strip():
            ger = None
        cleaned.append(
            {
                "species_group": t.get("species_group"),
                "scientific_name": sci,
                "german_name": ger,
                "collective_term": t.get("collective_term", False),
            }
        )
    norm["taxa"] = cleaned


def apply_evidence_gates(
    raw_pred: dict[str, Any],
    norm: dict[str, Any],
    source_text: str,
) -> None:
    """Verifies optional `_evidence` quotes for sparse fields and drops ungrounded entries."""
    ev = raw_pred.get("_evidence")
    if not isinstance(ev, dict):
        return
    if norm.get("ecosystem_service"):
        srv_q = str(ev.get("ecosystem_service") or "").strip()
        if not verify_verbatim_quote(srv_q, source_text):
            norm["ecosystem_service"] = []
    if norm.get("biodiversity_level"):
        lvl_quotes = ev.get("biodiversity_level") or []
        if isinstance(lvl_quotes, str):
            lvl_quotes = [lvl_quotes]
        if not any(verify_verbatim_quote(str(q), source_text) for q in lvl_quotes):
            norm["biodiversity_level"] = []


def extract_auto_routed_sections(
    zot_id: str,
    rec: dict[str, Any],
    pdf_dir: Path,
    grobid_url: str,
) -> list[dict[str, Any]]:
    """Routes to Marker when GROBID TEI structure is sparse or drops text; else uses GROBID."""
    py_path = REPO_ROOT / "data/parsed_text/dev_full" / f"{zot_id}.txt"
    if not py_path.exists():
        py_path = REPO_ROOT / "data/parsed_text/dev" / f"{zot_id}.txt"
    if py_path.exists():
        py_len = len(py_path.read_text(errors="ignore"))
    else:
        py_secs = extract_semantic_sections(zot_id, rec, pdf_dir)
        py_len = sum(len(s["text"]) for s in py_secs)

    grobid_secs = extract_grobid_chunks(zot_id, rec, pdf_dir, grobid_url)
    g_files = list((REPO_ROOT / "data/parsed_text/grobid_dev").glob(f"{zot_id}*.xml"))
    if g_files:
        g_xml = g_files[0].read_text(errors="ignore")
        g_divs = len(re.findall(r"<div[\s>]", g_xml))
        g_len = len(re.sub(r"<[^>]+>", " ", g_xml))
    else:
        g_divs = len(grobid_secs)
        g_len = sum(len(s["text"]) for s in grobid_secs)

    if g_divs <= 3 or (g_len / max(1, py_len)) < 0.65:
        marker_secs = extract_marker_sections(zot_id, rec, pdf_dir)
        if marker_secs:
            return marker_secs
    return grobid_secs


def extract_raw_auto_text(
    zot_id: str,
    rec: dict[str, Any],
    pdf_dir: Path,
    grobid_url: str,
) -> list[dict[str, Any]]:
    """Returns the unfiltered full AutoParser document text in one section (no 4-section filtering)."""
    py_path = REPO_ROOT / "data/parsed_text/dev_full" / f"{zot_id}.txt"
    if not py_path.exists():
        py_path = REPO_ROOT / "data/parsed_text/dev" / f"{zot_id}.txt"
    py_txt = py_path.read_text(errors="ignore") if py_path.exists() else ""
    py_len = len(py_txt)

    g_files = list((REPO_ROOT / "data/parsed_text/grobid_dev").glob(f"{zot_id}*.xml"))
    if not g_files:
        extract_grobid_chunks(zot_id, rec, pdf_dir, grobid_url)
        g_files = list((REPO_ROOT / "data/parsed_text/grobid_dev").glob(f"{zot_id}*.xml"))
    g_xml = g_files[0].read_text(errors="ignore") if g_files else ""
    g_divs = len(re.findall(r"<div[\s>]", g_xml))
    g_body_m = re.search(r"<body[^>]*>(.*?)</body>", g_xml, flags=re.S)
    g_body_txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", g_body_m.group(1) if g_body_m else g_xml)).strip()

    use_marker = g_divs <= 3 or (len(g_body_txt) / max(1, py_len)) < 0.65
    if use_marker:
        m_path = REPO_ROOT / "data/parsed_text/marker_dev" / f"{zot_id}.md"
        if m_path.exists():
            m_txt = m_path.read_text(errors="ignore").strip()
            if m_txt:
                return [{"section_name": "Full Document (Marker)", "text": m_txt}]
    if g_body_txt:
        return [{"section_name": "Full Document (GROBID)", "text": g_body_txt}]
    return [{"section_name": "Full Document (PyMuPDF)", "text": py_txt}]


def strict_normalize_prediction(
    parsed: dict[str, Any],
    field_spec: dict[str, Any],
) -> dict[str, Any]:
    """Exact-enum normalization without fuzzy substring guessing (matches Minh Anh's verifier)."""
    hab_set = set(field_spec["habitat"]["values"])
    lvl_set = set(field_spec["biodiversity_level"]["values"])
    ecat_set = set(field_spec["ecosystem_type"]["subfields"]["category"]["values"])
    eterm_set = set(field_spec["ecosystem_type"]["subfields"]["term"]["values"])
    grp_set = set(field_spec["taxa"]["subfields"]["species_group"]["values"])
    scat_set = set(field_spec["ecosystem_service"]["subfields"]["category"]["values"])
    sterm_set = set(field_spec["ecosystem_service"]["subfields"]["term"]["values"])

    hab = list(dict.fromkeys(v for v in (parsed.get("habitat") or []) if v in hab_set))
    lvl = list(dict.fromkeys(v for v in (parsed.get("biodiversity_level") or []) if v in lvl_set))

    eco = []
    seen_eco = set()
    for e in parsed.get("ecosystem_type") or []:
        if not isinstance(e, dict):
            continue
        c = e.get("category") if e.get("category") in ecat_set else None
        t = e.get("term") if e.get("term") in eterm_set else None
        if (c or t) and (c, t) not in seen_eco:
            seen_eco.add((c, t))
            eco.append({"category": c, "term": t})

    srv = []
    seen_srv = set()
    for e in parsed.get("ecosystem_service") or []:
        if not isinstance(e, dict):
            continue
        c = e.get("category") if e.get("category") in scat_set else None
        t = e.get("term") if e.get("term") in sterm_set else None
        if (c or t) and (c, t) not in seen_srv:
            seen_srv.add((c, t))
            srv.append({"category": c, "term": t})

    taxa = []
    seen_t = set()
    for t in parsed.get("taxa") or []:
        if not isinstance(t, dict):
            continue
        sg = t.get("species_group")
        if sg not in grp_set:
            continue
        sci = str(t.get("scientific_name")).strip() if t.get("scientific_name") else None
        if sci and sci != "-":
            sci = micronormalize_taxon_name(sci)
        ger = str(t.get("german_name")).strip() if t.get("german_name") else None
        k = (sg, sci, ger, bool(t.get("collective_term", False)))
        if k not in seen_t:
            seen_t.add(k)
            taxa.append({"species_group": sg, "scientific_name": sci, "german_name": ger, "collective_term": k[3]})

    return {
        "habitat": hab,
        "ecosystem_type": eco,
        "biodiversity_level": lvl,
        "taxa": taxa,
        "ecosystem_service": srv,
    }


def _split_sections_by_chars(
    sections: list[dict[str, Any]],
    max_chars: int = 80000,
) -> list[list[dict[str, Any]]]:
    """Splits sections into minimal contiguous parts without truncating long papers."""
    parts: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    cur_len = 0
    for sec in sections:
        text = sec["text"]
        if len(text) > max_chars:
            if cur:
                parts.append(cur)
                cur, cur_len = [], 0
            for start in range(0, len(text), max_chars):
                parts.append([{"section_name": sec["section_name"], "text": text[start : start + max_chars]}])
            continue
        if cur and cur_len + len(text) > max_chars:
            parts.append(cur)
            cur, cur_len = [], 0
        cur.append(sec)
        cur_len += len(text)
    if cur:
        parts.append(cur)
    return parts or [sections]


def _union_merge_predictions(preds: list[dict[str, Any]]) -> dict[str, Any]:
    """Merges verified predictions across minimal full-text parts via deduplicated union."""
    if len(preds) == 1:
        return preds[0]
    merged: dict[str, Any] = {}
    for f in ("habitat", "biodiversity_level"):
        seen: list[str] = []
        for p in preds:
            for v in p.get(f) or []:
                if v not in seen:
                    seen.append(v)
        merged[f] = seen
    for f in ("ecosystem_type", "ecosystem_service"):
        seen_pairs: set[tuple[Any, Any]] = set()
        items: list[dict[str, Any]] = []
        for p in preds:
            for item in p.get(f) or []:
                k = (item.get("category"), item.get("term"))
                if k not in seen_pairs:
                    seen_pairs.add(k)
                    items.append(item)
        merged[f] = items
    seen_taxa: set[tuple[Any, Any, Any, bool]] = set()
    taxa_items: list[dict[str, Any]] = []
    for p in preds:
        for t in p.get("taxa") or []:
            k = (t.get("species_group"), t.get("scientific_name"), t.get("german_name"), bool(t.get("collective_term", False)))
            if k not in seen_taxa:
                seen_taxa.add(k)
                taxa_items.append(t)
    merged["taxa"] = taxa_items
    return merged


def parse_and_verify_flat_labels(
    raw_pred: dict[str, Any],
    source_text: str,
    verify_quotes: bool = True,
) -> dict[str, Any]:
    """Converts Minh Anh's flat `labels` output to schema dict, dropping invalid 3-60 word quotes."""
    if not isinstance(raw_pred.get("labels"), list):
        return raw_pred
    out: dict[str, Any] = {
        "habitat": [],
        "ecosystem_type": [],
        "biodiversity_level": [],
        "taxa": [],
        "ecosystem_service": [],
    }
    for item in raw_pred["labels"]:
        if not isinstance(item, dict):
            continue
        if verify_quotes:
            q = str(item.get("quote") or "").strip()
            wc = len(q.split())
            if wc < 3 or wc > 60 or not verify_verbatim_quote(q, source_text):
                continue
        fld = item.get("field")
        if fld in ("habitat", "biodiversity_level"):
            val = item.get("value")
            if isinstance(val, str) and val.strip():
                out[fld].append(val.strip())
        elif fld in ("ecosystem_type", "ecosystem_service"):
            out[fld].append({"category": item.get("category"), "term": item.get("term")})
        elif fld == "taxa":
            out["taxa"].append({
                "species_group": item.get("species_group"),
                "scientific_name": item.get("scientific_name"),
                "german_name": item.get("german_name"),
                "collective_term": bool(item.get("collective_term", False)),
            })
    return out


LEVEL_V1_SYSTEM = (
    "You are an ecologist who annotates scientific studies for a German literature database on biodiversity. "
    "Decide ONLY one field: biodiversity_level, the levels of biodiversity that THIS document examines. "
    "Prove every answer with an exact verbatim quote from the text."
)


def run_level_prompt_call(
    client: Any,
    rec: dict[str, Any],
    part_text: str,
    few_shots: list[dict[str, Any]],
    allowed_levels: list[str],
    min_confidence: float = 0.5,
) -> list[str] | None:
    """Focused second call (level_v1 style) for biodiversity_level with quote verification."""
    ex_lines = []
    for idx, ex in enumerate(few_shots, 1):
        lvls = ex["gold_json"].get("biodiversity_level") or []
        ex_lines.append(f"Example {idx} ({ex['title']}): {json.dumps(lvls, ensure_ascii=False)}")
    user_p = (
        "Decide which biodiversity_level values THIS document examines as a main subject (ignore background citations). "
        "About 40% of documents have no level ([]). Allowed values (copy character by character):\n"
        "- Artenvielfalt (species richness, diversity indices, species lists)\n"
        "- Zusammensetzung der Artengemeinschaft (community composition, assemblages, abundance comparisons)\n"
        "- Habitatdiversität (variety of habitats/biotopes, habitat heterogeneity/mosaics)\n"
        "- Funktionelle Diversität (functional traits/groups, ecosystem functions)\n"
        "- Genetische Diversität (genetic variation, populations, molecular markers)\n"
        "- Strukturelle Diversität (stand/vegetation structure, layers, deadwood)\n"
        "- Phylogenetische Diversität (phylogenetic diversity/trees)\n\n"
        "Similar training examples (biodiversity_level):\n" + "\n".join(ex_lines) + "\n\n"
        f"Title: {str(rec.get('bibtex_title') or '').strip()}\n"
        f"Text:\n{part_text[:16000]}\n\n"
        'Return ONE JSON object only: {"reasoning": "S1: ... S2: ...", "labels": '
        '[{"field": "biodiversity_level", "value": "<allowed>", "confidence": 0.5, "quote": "<3-60 words verbatim>"}]}'
    )
    raw = client.chat(LEVEL_V1_SYSTEM, user_p)
    parsed = extract_json_block(raw)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("labels"), list):
        return None
    allowed_set = set(allowed_levels)
    verified: list[str] = []
    for item in parsed["labels"]:
        if not isinstance(item, dict):
            continue
        val = str(item.get("value") or "").strip()
        if val not in allowed_set:
            continue
        conf = float(item.get("confidence", 1.0) or 0.0)
        q = str(item.get("quote") or "").strip()
        wc = len(q.split())
        if conf >= min_confidence and 3 <= wc <= 60 and verify_verbatim_quote(q, part_text):
            if val not in verified:
                verified.append(val)
    return verified


V4_SYSTEM_PROMPT = (
    "You are an ecologist who annotates scientific studies for a German literature database on biodiversity. "
    "You extract what THIS document is mainly about, and you prove every answer with an exact quote from the text you are given."
)


def format_v4_examples(few_shot_examples: list[dict[str, Any]], max_taxa: int = 5) -> str:
    """Formats kNN examples cleanly in Minh Anh's compact v4 style (with bibtex_type)."""
    blocks = []
    for idx, ex in enumerate(few_shot_examples, 1):
        g = ex["gold_json"]
        btype = ex.get("bibtex_type") or "unknown"
        lines = [f"Example {idx}: {json.dumps(ex['title'], ensure_ascii=False)} (bibtex_type: {btype})"]
        for fld in ("habitat", "biodiversity_level"):
            vals = g.get(fld) or []
            lines.append(f"- {fld}: {'; '.join(json.dumps(v, ensure_ascii=False) for v in vals) if vals else '(empty)'}")
        for fld in ("ecosystem_type", "ecosystem_service"):
            vals = g.get(fld) or []
            if vals:
                parts = [f"{json.dumps(e.get('category'), ensure_ascii=False)} / {json.dumps(e.get('term'), ensure_ascii=False)}" for e in vals]
                lines.append(f"- {fld} (category / term): {'; '.join(parts)}")
            else:
                lines.append(f"- {fld} (category / term): (empty)")
        taxa = g.get("taxa") or []
        if taxa:
            shown = taxa[:max_taxa]
            tparts = [
                " / ".join(
                    json.dumps(t.get(k), ensure_ascii=False) if t.get(k) else "null"
                    for k in ("scientific_name", "german_name", "species_group")
                )
                for t in shown
            ]
            more = f" (and {len(taxa) - len(shown)} more)" if len(taxa) > len(shown) else ""
            lines.append(f"- taxa (scientific_name / german_name / species_group): {'; '.join(tparts)}{more}")
        else:
            lines.append("- taxa (scientific_name / german_name / species_group): (empty)")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_v4_user_prompt(
    rec: dict[str, Any],
    sections: list[dict[str, Any]],
    few_shot_examples: list[dict[str, Any]],
    field_spec: dict[str, Any],
    eco_category_map_str: str,
    max_total_chars: int = 16000,
    full_text: bool = False,
    level_fold: bool = False,
    pair_german: bool = False,
) -> str:
    """Builds Minh Anh's cohesive prompt v4 over our AutoParser sections.

    `level_fold` injects a dedicated biodiversity_level decision block (adapted from
    Minh's level_v1 field prompt) into the main prompt instead of using a second call.
    """
    if full_text:
        doc_text = "\n\n".join(f"### SECTION: {s['section_name']}\n{s['text']}" for s in sections)
    else:
        budget_per_sec = max(2500, max_total_chars // max(1, len(sections)))
        doc_text = "\n\n".join(f"### SECTION: {s['section_name']}\n{s['text'][:budget_per_sec]}" for s in sections)

    hab_vals = json.dumps(field_spec["habitat"]["values"], ensure_ascii=False)
    lvl_vals = json.dumps(field_spec["biodiversity_level"]["values"], ensure_ascii=False)
    grp_vals = json.dumps(field_spec["taxa"]["subfields"]["species_group"]["values"], ensure_ascii=False)
    srv_cats = json.dumps(field_spec["ecosystem_service"]["subfields"]["category"]["values"], ensure_ascii=False)
    srv_terms = json.dumps(field_spec["ecosystem_service"]["subfields"]["term"]["values"], ensure_ascii=False)
    examples_str = format_v4_examples(few_shot_examples)
    title = str(rec.get("bibtex_title") or "").strip()

    # R7, optionally extended by the DB pairing convention (93% of gold taxa
    # entries carry sci+german; 0 are latin-only): a scientific name always
    # comes with its common German name, translated if the paper is not German.
    r7 = (
        "R7 Taxa names: scientific names WITHOUT author/year ('Fagus sylvatica', not 'Fagus sylvatica L.'); "
        "genus only -> 'Genus spp.'; German names capitalized ('Rot-Buche'); at most 10 taxa entries (if more, give higher group)."
    )
    if pair_german:
        r7 += (
            " Every entry with a scientific_name MUST also carry german_name: give the common German name,"
            " translating it if the document is not German (e.g. 'Gadus morhua' -> 'Atlantischer Kabeljau');"
            " German-only collective entries are also valid (e.g. 'Wildbestäuber')."
        )
    r7 += "\n\n"

    return (
        f"## Task\nRead the text below (Title: {title}) and fill the record described in \"Fields\". "
        "Work through the reasoning steps first, then give the record.\n\n"
        f"## Fields\n"
        f"1. habitat (choose 0..n): {hab_vals}\n"
        f"2. ecosystem_type (choose 0..n pairs of category and term; term must belong to its category):\n{eco_category_map_str}\n"
        f"3. biodiversity_level (choose 0..n): {lvl_vals}\n"
        f"4. taxa (0..n entries): species_group (choose 1): {grp_vals}; plus scientific_name and german_name (rule R7).\n"
        f"5. ecosystem_service (choose 0..n pairs; term is optional):\n"
        f"   category: {srv_cats}\n"
        f"   term: {srv_terms}\n\n"
        "## Hard rules\n"
        "R1 Allowed values only: copy values character by character.\n"
        "R2 No label without evidence: every label needs a verbatim \"quote\" (3 to 60 words from the text below, no \"...\").\n"
        "R3 Main subject only: label what THIS document is mainly about (studies, reviews, reports, policy papers, books). Ignore single examples, background, or citations of other work.\n"
        "R4 Empirical rates: habitat ~94% (usually 1), taxa ~73%, biodiversity_level ~58%, ecosystem_type ~50% (usually <=1 pair), ecosystem_service ~22%. Empty is valid except usually habitat.\n"
        "R5 No world knowledge about locations/methods not in text (except R7 taxa names, and habitat may be derived from described environment).\n"
        "R6 Consistency: ecosystem_type term must match its category.\n"
        f"{r7}"
        "R8 Answer with ONE JSON object only.\n\n"
        "## Reasoning steps (write into \"reasoning\", max 120 words)\n"
        "S1 Study system -> habitat, ecosystem_type\n"
        "S2 Organisms -> taxa\n"
        "S3 Measurements -> biodiversity_level\n"
        "S4 Services -> ecosystem_service\n"
        "S5 Self-check: verify allowed list, exact quote, main subject (R3); drop failures.\n\n"
        "## Output structure\n"
        '{"reasoning": "S1: ... S2: ... S3: ... S4: ... S5: ...", "labels": ['
        '{"field": "habitat", "value": "<allowed>", "quote": "<verbatim quote>"}, '
        '{"field": "ecosystem_type", "category": "<allowed>", "term": "<allowed or null>", "quote": "<verbatim quote>"}, '
        '{"field": "biodiversity_level", "value": "<allowed>", "quote": "<verbatim quote>"}, '
        '{"field": "taxa", "species_group": "<allowed>", "scientific_name": "<name or null>", "german_name": "<name or null>", "collective_term": false, "quote": "<verbatim quote>"}, '
        '{"field": "ecosystem_service", "category": "<allowed>", "term": "<allowed or null>", "quote": "<verbatim quote>"}'
        "]}\n\n"
        "## How the database labelled similar documents (training examples)\n"
        f"{examples_str}\n\n"
        + (LEVEL_FOLD_BLOCK + "\n\n" if level_fold else "")
        + f"## Text\n<<<\n{doc_text}\n>>>"
    )


# Dedicated biodiversity_level decision rules folded into the main prompt
# (--level-fold). Compact adaptation of Minh's level_v1 field prompt: her enum
# definitions, base rates and main-subject rules, without the confidence field
# (our output schema has none) and without a second LLM call.
LEVEL_FOLD_BLOCK = """## biodiversity_level decision guide (apply to field 3)
Levels are labelled only when THIS document examines biodiversity at that level: it
measures, compares, models or reviews it as a main subject. A level that appears only
as background, as a goal in the introduction, or in citations of other work is NOT labelled.
- Artenvielfalt (~two thirds of labelled documents): species richness, species numbers,
  diversity indices (Shannon, Simpson), species lists/inventories, species gains and losses.
- Zusammensetzung der Artengemeinschaft (~one third): community composition, abundance or
  biomass of several species compared between sites/treatments/years, community shifts,
  microbial communities from sequencing.
- Habitatdiversität (~one in six): several habitat/biotope types compared, habitat
  heterogeneity, state or loss of habitats, habitat mosaics.
- Funktionelle Diversität (~one in six): functional groups, functional traits, multifunctionality,
  effects of diversity on ecosystem functions.
- Genetische Diversität (~one in fifteen): genetic variation, genotypes, molecular markers,
  population genetics.
- Strukturelle Diversität (~one in fifteen): stand or vegetation structure, layers, deadwood,
  mixed vs. pure stands, structural habitat heterogeneity.
- Phylogenetische Diversität (rare): phylogenetic diversity/relatedness, phylogenetic trees.
About four in ten documents get NO level at all (e.g. physiology, growth or behaviour studies
without comparing species; soil/water/climate processes; policy or services without biodiversity
data) - an empty list is a valid answer. Check every level on its own; typical pairs are
Artenvielfalt + Zusammensetzung der Artengemeinschaft, + Habitatdiversität, + Funktionelle Diversität.
Label "possible but doubtful" cases only when a concrete measurement or comparison is described."""


def main() -> None:
    parser = argparse.ArgumentParser(description="Architecture 3: RAG-Anchored Section Routing Pipeline")
    parser.add_argument("--provider", choices=list(PROVIDERS.keys()), default="saia")
    parser.add_argument("--model", default="deepseek-v4.1-flash-1017")
    parser.add_argument("--top-k", type=int, default=2, help="Number of kNN training anchors")
    parser.add_argument("--grobid-url", default=None)
    parser.add_argument("--docling", action="store_true", help="Use IBM Docling Markdown parser")
    parser.add_argument("--minh-json", action="store_true", help="Use Minh's full Docling JSON cache (OCR + page markers)")
    parser.add_argument("--marker", action="store_true", help="Use Marker PDF Markdown cache")
    parser.add_argument("--auto-parser", action="store_true", help="Route between GROBID and Marker using objective TEI/text signals")
    parser.add_argument("--full-text", action="store_true", help="Do not truncate sections; split only if >80k chars")
    parser.add_argument("--focus-prompt", action="store_true", help="Use document-focus wording + train base-rates (v2 insight)")
    parser.add_argument("--detox-rag", action="store_true", help="Char n-gram retrieval + strip taxa from kNN anchors")
    parser.add_argument("--role-budget", action="store_true", help="28k role-weighted section budget (Methods/Results prioritized)")
    parser.add_argument("--budget-chars", type=int, default=16000, help="Total character budget distributed across sections in the prompt (default 16000)")
    parser.add_argument("--evidence-gates", action="store_true", help="Require verbatim quote evidence on sparse fields")
    parser.add_argument("--clean-taxa", action="store_true", help="Do not fabricate '-' or species_group fallbacks in taxa")
    parser.add_argument("--self-check-reasoning", action="store_true", help="Structured S1-S5 self-check reasoning + R7 taxa spelling rules")
    parser.add_argument("--per-label-quotes", action="store_true", help="Flat labels JSON with 3-60 word verbatim quote check on every label")
    parser.add_argument("--eco-category-map", action="store_true", help="Group allowed ecosystem_type terms under each category")
    parser.add_argument("--level-prompt", action="store_true", help="Focused second call (level_v1) for biodiversity_level")
    parser.add_argument("--level-fold", action="store_true", help="Fold biodiversity_level decision rules into the main prompt (no second call)")
    parser.add_argument("--service-gate", action="store_true", help="Drop ecosystem_service labels when none of the retrieved training neighbours has any")
    parser.add_argument("--german-plural-norm", action="store_true", help="Map predicted German taxa names to their unique train-vocabulary variant (e.g. Seehund -> Seehunde)")
    parser.add_argument("--german-pairing-names", action="store_true", help="Prompt rule: every scientific_name entry must also carry its German name (DB pairing convention)")
    parser.add_argument("--split", choices=["dev", "train"], default="dev", help="Evaluate the dev-100 split (default) or train papers (router verification; needs --knn-exclude-self)")
    parser.add_argument("--ids-file", default=None, help="Optional file with paper ids to run (one per line); defaults to all ids of the split")
    parser.add_argument("--knn-exclude-self", action="store_true", help="Exclude the paper itself from kNN few-shot retrieval (mandatory for train-split scoring)")
    parser.add_argument("--v4-prompt", action="store_true", help="Use Minh Anh's cohesive prompt v4 + compact k=3 examples + 3-60w quote check")
    parser.add_argument("--bilingual-knn", action="store_true", help="Use bilingual char_wb 3-5 TF-IDF retrieval without stripping taxa")
    parser.add_argument("--raw-fulltext", action="store_true", help="Unfiltered full AutoParser text in 1 call + strict exact-enum normalization")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--rpm", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    if args.split == "train":
        gold = load_jsonl_by_id(REPO_ROOT / "data/gold/train.jsonl")
        if args.ids_file:
            dev_ids = [l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()]
        else:
            dev_ids = sorted(gold.keys())
    else:
        gold = load_jsonl_by_id(REPO_ROOT / "data/gold/dev.jsonl")
        dev_ids = [l.strip() for l in (REPO_ROOT / "data/gold/dev_100_ids.txt").read_text().splitlines() if l.strip()]
    if args.limit:
        dev_ids = dev_ids[: args.limit]

    pdf_dir = REPO_ROOT / ("data/raw_pdfs/train" if args.split == "train" else "data/raw_pdfs/dev")
    train_text_dir = REPO_ROOT / "data/parsed_text/train"

    print("Building TF-IDF kNN index over 908 training papers...")
    train_records = load_jsonl_by_id(REPO_ROOT / "data/gold/train.jsonl")
    german_vocab = build_german_vocab(train_records)
    retriever = TrainKNNRetriever(
        train_records, train_text_dir, detox_rag=args.detox_rag, bilingual_knn=args.bilingual_knn
    )
    majority_learned = fit_majority_model(train_records)

    train_rates_note = ""
    if args.focus_prompt and train_records:
        n_tr = len(train_records)
        rates = {
            f: round(100 * sum(1 for r in train_records.values() if r.get(f)) / n_tr)
            for f in ("habitat", "biodiversity_level", "ecosystem_type", "taxa", "ecosystem_service")
        }
        train_rates_note = (
            f" Empirical database non-empty rates: habitat={rates['habitat']}%, "
            f"biodiversity_level={rates['biodiversity_level']}%, ecosystem_type={rates['ecosystem_type']}%, "
            f"taxa={rates['taxa']}%, ecosystem_service={rates['ecosystem_service']}%."
        )

    # Load official field specifications & build system prompt
    field_spec = load_field_spec()
    system_prompt = V4_SYSTEM_PROMPT if args.v4_prompt else (SYSTEM_PROMPT + "\n\nFIELD SCHEMA & GUIDELINES:\n" + format_field_spec(field_spec))

    eco_category_map_str = ""
    if args.eco_category_map or args.v4_prompt:
        cat_terms: dict[str, set[str]] = {c: set() for c in field_spec["ecosystem_type"]["subfields"]["category"]["values"]}
        for r in train_records.values():
            for et in r.get("ecosystem_type") or []:
                c, t = et.get("category"), et.get("term")
                if c in cat_terms and t:
                    cat_terms[c].add(t)
        eco_category_map_str = "\n".join(f"- {c}: {json.dumps(sorted(ts), ensure_ascii=False)}" for c, ts in cat_terms.items())

    if args.dry_run:
        client: Any = DryRunMockClient(train_records)
        actual_model = "dryrun-mock"
    else:
        actual_model = args.model
        client = RobustOpenAICompatClient(args.provider, actual_model, args.temperature, args.rpm)

    model_slug = actual_model.replace("/", "_").replace(":", "_")
    if args.auto_parser:
        parser_suffix, parser_label = "_autoparser", " + AutoParser"
    elif args.marker:
        parser_suffix, parser_label = "_marker", " + Marker"
    elif args.docling:
        parser_suffix, parser_label = "_docling", " + Docling"
    elif args.minh_json:
        parser_suffix, parser_label = "_minhjson", " + MinhDoclingJSON"
    elif args.grobid_url:
        parser_suffix, parser_label = "_grobid", " + GROBID"
    else:
        parser_suffix, parser_label = "", ""

    opt_suffix = "".join([
        "_full" if args.full_text else "",
        "_focus" if args.focus_prompt else "",
        "_detox" if args.detox_rag else "",
        "_role" if args.role_budget else "",
        "_ev" if args.evidence_gates else "",
        "_ctaxa" if args.clean_taxa else "",
        "_s5" if args.self_check_reasoning else "",
        "_plq" if args.per_label_quotes else "",
        "_ecomap" if args.eco_category_map else "",
        "_lvl" if args.level_prompt else "",
        "_v4" if args.v4_prompt else "",
        "_biknn" if args.bilingual_knn else "",
        "_rawft" if args.raw_fulltext else "",
        "_lvlf" if args.level_fold else "",
        "_svcg" if args.service_gate else "",
        "_gpn" if args.german_plural_norm else "",
        "_pgn" if args.german_pairing_names else "",
        "_tr" if args.split == "train" else "",
        (f"_b{args.budget_chars // 1000}k" if args.budget_chars != 16000 else ""),
    ])
    run_model_id = f"{MODEL_BASE_ID}_{args.provider}_{model_slug}{parser_suffix}{opt_suffix}"
    run_title = f"Architecture 3 (RAG-Anchored Sections{parser_label}{opt_suffix}): {actual_model}"

    ckpt_dir = REPO_ROOT / "outputs" / CATEGORY / run_model_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / f"{run_model_id}_checkpoint.jsonl"

    predictions: dict[str, dict[str, Any]] = {}
    if ckpt_file.exists():
        for line in ckpt_file.read_text().splitlines():
            if line.strip():
                try:
                    obj = json.loads(line)
                    predictions[obj["zotitem_ptr_id"]] = obj
                except Exception:
                    pass

    print(f"=== Starting {run_title} | Papers: {len(dev_ids)} (resumed: {len(predictions)}) ===")

    for i, zot_id in enumerate(dev_ids, 1):
        if zot_id in predictions:
            continue

        rec = gold[zot_id]
        if args.raw_fulltext:
            sections = extract_raw_auto_text(
                zot_id, rec, pdf_dir, args.grobid_url or "http://127.0.0.1:8070"
            )
        elif args.auto_parser:
            sections = extract_auto_routed_sections(
                zot_id, rec, pdf_dir, args.grobid_url or "http://127.0.0.1:8070"
            )
        elif args.marker:
            sections = extract_marker_sections(zot_id, rec, pdf_dir)
        elif args.docling:
            sections = extract_docling_sections(zot_id, rec, pdf_dir)
        elif args.minh_json:
            sections = extract_minh_json_sections(zot_id, rec, pdf_dir)
        elif args.grobid_url:
            sections = extract_grobid_chunks(zot_id, rec, pdf_dir, args.grobid_url)
        else:
            sections = extract_semantic_sections(zot_id, rec, pdf_dir)
        combined_clean_text = " ".join(s["text"] for s in sections)
        effective_k = 3 if (args.v4_prompt and args.top_k == 2) else args.top_k
        few_shots = retriever.retrieve_top_k(
            combined_clean_text, k=effective_k, exclude_id=zot_id if args.knn_exclude_self else None
        )

        if isinstance(client, DryRunMockClient):
            client.mock_context = rec

        sec_parts = _split_sections_by_chars(sections, max_chars=80000) if (args.full_text and not args.raw_fulltext) else [sections]
        part_norms: list[dict[str, Any]] = []
        for part_secs in sec_parts:
            part_text = " ".join(s["text"] for s in part_secs)
            if args.v4_prompt:
                user_prompt = build_v4_user_prompt(
                    rec, part_secs, few_shots, field_spec, eco_category_map_str,
                    max_total_chars=args.budget_chars,
                    full_text=(args.full_text or args.raw_fulltext),
                    level_fold=args.level_fold,
                    pair_german=args.german_pairing_names,
                )
            else:
                user_prompt = build_rag_user_prompt(
                    rec,
                    part_secs,
                    few_shots,
                    field_spec,
                    max_total_chars=args.budget_chars,
                    role_budget=args.role_budget,
                    evidence_gates=args.evidence_gates,
                    full_text=(args.full_text or args.raw_fulltext),
                    focus_prompt=args.focus_prompt,
                    train_rates_note=train_rates_note,
                    self_check_reasoning=args.self_check_reasoning,
                    per_label_quotes=args.per_label_quotes,
                    eco_category_map_str=eco_category_map_str,
                )
            raw_out = client.chat(system_prompt, user_prompt)
            parsed = extract_json_block(raw_out) or {}
            if args.per_label_quotes or args.v4_prompt:
                parsed = parse_and_verify_flat_labels(
                    parsed, part_text, verify_quotes=not isinstance(client, DryRunMockClient)
                )
            if args.raw_fulltext:
                part_norm = strict_normalize_prediction(parsed, field_spec)
            else:
                part_norm = normalize_prediction(parsed, field_spec)
                if args.clean_taxa:
                    apply_clean_taxa(parsed, part_norm)
            if args.german_plural_norm:
                apply_german_plural_norm(part_norm, german_vocab)
            if args.evidence_gates and not args.per_label_quotes and not isinstance(client, DryRunMockClient):
                apply_evidence_gates(parsed, part_norm, part_text)
            if args.level_prompt and not isinstance(client, DryRunMockClient):
                lvl_override = run_level_prompt_call(
                    client,
                    rec,
                    part_text,
                    few_shots,
                    field_spec["biodiversity_level"]["values"],
                )
                if lvl_override is not None:
                    part_norm["biodiversity_level"] = lvl_override
            part_norms.append(part_norm)
        norm = _union_merge_predictions(part_norms)
        norm["zotitem_ptr_id"] = zot_id

        # Dense fallback for habitat if completely empty
        if not norm.get("habitat"):
            norm["habitat"] = [majority_learned["habitat"]["majority_value"]]

        # Neighbour service gate: ecosystem services are rare (~22% of train papers).
        # Drop this paper's service labels when none of the k retrieved training
        # neighbours has any — train-statistics rule, threshold-free (0 of k).
        if args.service_gate and norm.get("ecosystem_service"):
            neighbour_has_service = any(
                (fs.get("gold_json") or {}).get("ecosystem_service") for fs in few_shots
            )
            if not neighbour_has_service:
                print(
                    f"    service-gate: dropped {len(norm['ecosystem_service'])} service "
                    "entries (none of the neighbours has services)",
                    flush=True,
                )
                norm["ecosystem_service"] = []

        predictions[zot_id] = norm
        with ckpt_file.open("a", encoding="utf-8") as f_out:
            f_out.write(json.dumps(norm, ensure_ascii=False) + "\n")

        print(
            f"[{i:>3}/{len(dev_ids)}] {zot_id} | Secs: {len(sections)} | "
            f"Hab: {len(norm.get('habitat', []))} Taxa: {len(norm.get('taxa', []))}",
            flush=True,
        )

    metrics = evaluate_predictions(gold, predictions, filter_ids=set(dev_ids))
    print_report(f"{run_title} {'[DRY-RUN]' if args.dry_run else ''}", metrics)

    if not args.no_save and not args.dry_run:
        save_reproducible_run(
            model_id=run_model_id,
            model_title=run_title,
            category=CATEGORY,
            predictions=predictions,
            metrics=metrics,
            runtime_seconds=time.time() - t0,
            seed=SEED,
            config={
                "provider": args.provider,
                "model": actual_model,
                "top_k": args.top_k,
                "temperature": args.temperature,
                "split": args.split,
                "knn_exclude_self": args.knn_exclude_self,
                "german_plural_norm": args.german_plural_norm,
                "german_pairing_names": args.german_pairing_names,
            },
        )


if __name__ == "__main__":
    main()
