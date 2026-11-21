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
    ):
        self.train_ids = sorted(train_records.keys())
        self.train_records = train_records
        self.detox_rag = detox_rag
        docs = [_load_corpus_text(zid, train_records[zid], train_text_dir) for zid in self.train_ids]
        if detox_rag:
            # Bilingual char n-grams match both German/English text and Latin taxon stems
            self.vectorizer = TfidfVectorizer(
                max_features=20000, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True
            )
        else:
            self.vectorizer = TfidfVectorizer(max_features=15000, stop_words="english", sublinear_tf=True)
        self.train_matrix = self.vectorizer.fit_transform(docs)

    def retrieve_top_k(self, query_text: str, k: int = 2) -> list[dict[str, Any]]:
        q_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(q_vec, self.train_matrix).ravel()
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

    # 2. Pack high-signal semantic sections within budget
    sec_blocks = []
    if role_budget:
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

    evidence_note = ""
    if evidence_gates:
        evidence_note = (
            "\n\nEVIDENCE REQUIREMENT FOR SPARSE FIELDS:\n"
            "If you output `ecosystem_service` or `biodiversity_level`, also include an optional "
            "`_evidence` object with short verbatim quotes from the document proving them:\n"
            '`"_evidence": {"ecosystem_service": "<verbatim quote>", "biodiversity_level": ["<verbatim quote>"]}`\n'
            "If no verbatim evidence exists in the text, leave those fields as [].\n"
        )

    return (
        f"SIMILAR ANNOTATED EXAMPLES (follow their exact label conventions & sparsity; NEVER copy taxa from examples):\n"
        f"{examples_str}\n\n"
        f"ALLOWED ECOSYSTEM_TYPE.TERM VALUES:\n{json.dumps(eco_terms, ensure_ascii=False)}\n\n"
        f"ALLOWED ECOSYSTEM_SERVICE.TERM VALUES:\n{json.dumps(srv_terms, ensure_ascii=False)}\n\n"
        f"TARGET DOCUMENT\n"
        f"Title: {title}\n\n"
        f"{doc_text}\n\n"
        f"Extract the main studied/focused attributes of THIS document (ignore background citations)."
        f"{evidence_note} Return a single valid JSON object matching the schema."
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
    """Uses cached Docling Markdown (`docling_dev/`) or runs Docling on the fly."""
    return extract_markdown_cached_sections(
        zot_id,
        rec,
        pdf_dir,
        "docling_dev",
        lambda p: _get_docling_converter().convert(str(p)).document.export_to_markdown(),
    )


def extract_marker_sections(
    zot_id: str,
    rec: dict[str, Any],
    pdf_dir: Path,
) -> list[dict[str, Any]]:
    """Uses cached Marker Markdown (`marker_dev/`), falling back to PyMuPDF if missing."""
    return extract_markdown_cached_sections(zot_id, rec, pdf_dir, "marker_dev", None)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Architecture 3: RAG-Anchored Section Routing Pipeline")
    parser.add_argument("--provider", choices=list(PROVIDERS.keys()), default="saia")
    parser.add_argument("--model", default="deepseek-v4.1-flash-1017")
    parser.add_argument("--top-k", type=int, default=2, help="Number of kNN training anchors")
    parser.add_argument("--grobid-url", default=None)
    parser.add_argument("--docling", action="store_true", help="Use IBM Docling Markdown parser")
    parser.add_argument("--marker", action="store_true", help="Use Marker PDF Markdown cache")
    parser.add_argument("--detox-rag", action="store_true", help="Char n-gram retrieval + strip taxa from kNN anchors")
    parser.add_argument("--role-budget", action="store_true", help="28k role-weighted section budget (Methods/Results prioritized)")
    parser.add_argument("--evidence-gates", action="store_true", help="Require verbatim quote evidence on sparse fields")
    parser.add_argument("--clean-taxa", action="store_true", help="Do not fabricate '-' or species_group fallbacks in taxa")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--rpm", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    gold = load_jsonl_by_id(REPO_ROOT / "data/gold/dev.jsonl")
    train_records = load_jsonl_by_id(REPO_ROOT / "data/gold/train.jsonl")
    dev_ids = [l.strip() for l in (REPO_ROOT / "data/gold/dev_100_ids.txt").read_text().splitlines() if l.strip()]
    if args.limit:
        dev_ids = dev_ids[: args.limit]

    pdf_dir = REPO_ROOT / "data/raw_pdfs/dev"
    train_text_dir = REPO_ROOT / "data/parsed_text/train"

    print("Building TF-IDF kNN index over 908 training papers...")
    retriever = TrainKNNRetriever(train_records, train_text_dir, detox_rag=args.detox_rag)
    majority_learned = fit_majority_model(train_records)

    # Load official field specifications & build system prompt
    field_spec = load_field_spec()
    system_prompt = SYSTEM_PROMPT + "\n\nFIELD SCHEMA & GUIDELINES:\n" + format_field_spec(field_spec)

    if args.dry_run:
        client: Any = DryRunMockClient(train_records)
        actual_model = "dryrun-mock"
    else:
        actual_model = args.model
        client = RobustOpenAICompatClient(args.provider, actual_model, args.temperature, args.rpm)

    model_slug = actual_model.replace("/", "_").replace(":", "_")
    if args.marker:
        parser_suffix, parser_label = "_marker", " + Marker"
    elif args.docling:
        parser_suffix, parser_label = "_docling", " + Docling"
    elif args.grobid_url:
        parser_suffix, parser_label = "_grobid", " + GROBID"
    else:
        parser_suffix, parser_label = "", ""

    opt_suffix = "".join([
        "_detox" if args.detox_rag else "",
        "_role" if args.role_budget else "",
        "_ev" if args.evidence_gates else "",
        "_ctaxa" if args.clean_taxa else "",
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
        if args.marker:
            sections = extract_marker_sections(zot_id, rec, pdf_dir)
        elif args.docling:
            sections = extract_docling_sections(zot_id, rec, pdf_dir)
        elif args.grobid_url:
            sections = extract_grobid_chunks(zot_id, rec, pdf_dir, args.grobid_url)
        else:
            sections = extract_semantic_sections(zot_id, rec, pdf_dir)
        combined_clean_text = " ".join(s["text"] for s in sections)
        few_shots = retriever.retrieve_top_k(combined_clean_text, k=args.top_k)

        user_prompt = build_rag_user_prompt(
            rec,
            sections,
            few_shots,
            field_spec,
            role_budget=args.role_budget,
            evidence_gates=args.evidence_gates,
        )
        if isinstance(client, DryRunMockClient):
            client.mock_context = rec

        raw_out = client.chat(system_prompt, user_prompt)
        parsed = extract_json_block(raw_out) or {}
        norm = normalize_prediction(parsed, field_spec)
        if args.clean_taxa:
            apply_clean_taxa(parsed, norm)
        if args.evidence_gates and not isinstance(client, DryRunMockClient):
            apply_evidence_gates(parsed, norm, combined_clean_text)
        norm["zotitem_ptr_id"] = zot_id

        # Dense fallback for habitat if completely empty
        if not norm.get("habitat"):
            norm["habitat"] = [majority_learned["habitat"]["majority_value"]]

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
            },
        )


if __name__ == "__main__":
    main()
