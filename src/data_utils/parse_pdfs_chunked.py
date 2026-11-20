#!/usr/bin/env python3
"""
========================================================================================
EXECUTIVE SUMMARY: Chunk-Wise Full-PDF Markdown Converter (Marker + Docling)
========================================================================================
- Concept & Goal:
    Converts arbitrarily long PDFs (including 100-300+ page reports and books) into
    complete, cached Markdown files (`data/parsed_text/<parser>_<split>/<id>.md`) without
    ever hitting out-of-memory (OOM) or per-document timeout failures.

- How It Works:
    1. Inspects PDF page count via `pypdfium2` and splits processing into 15-page chunks
       (`page_range` 0-14, 15-29, ... for Marker; 1-15, 16-30, ... for Docling).
    2. Runs `gc.collect()` between chunks so peak RAM remains bounded (~1.5-3 GB).
    3. Concatenates all chunk Markdowns into one complete `.md` file and marks completion
       with a `.full_ok` sentinel so interrupted runs resume cleanly.

- Usage (Production / Test Set):
    python3 src/data_utils/parse_pdfs_chunked.py --parser marker --split test
    python3 src/data_utils/parse_pdfs_chunked.py --parser docling --split dev
========================================================================================
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import time

REPO_ROOT = Path(__file__).resolve().parents[2]


def get_num_pages(pdf_path: Path) -> int:
    """Returns total page count of a PDF using pypdfium2."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    n = len(doc)
    doc.close()
    return n


def run_marker(pdfs: list[Path], out_dir: Path, chunk_pages: int = 15) -> None:
    """Converts all PDFs in `pdfs` to full Markdown via Marker in `chunk_pages` slices."""
    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    out_dir.mkdir(parents=True, exist_ok=True)
    models = create_model_dict()
    for i, p in enumerate(pdfs, 1):
        target = out_dir / f"{p.stem}.md"
        done_flag = out_dir / f"{p.stem}.full_ok"
        if done_flag.exists() and target.exists() and target.stat().st_size > 50:
            continue
        t0 = time.time()
        try:
            n_pages = get_num_pages(p)
            parts: list[str] = []
            for start in range(0, n_pages, chunk_pages):
                end = min(start + chunk_pages - 1, n_pages - 1)
                cfg = ConfigParser({"page_range": f"{start}-{end}"}).generate_config_dict()
                conv = PdfConverter(config=cfg, artifact_dict=models)
                rendered = conv(str(p))
                text, _, _ = text_from_rendered(rendered)
                parts.append(text)
                del conv, rendered
                gc.collect()
            full_text = "\n\n".join(parts)
            target.write_text(full_text, encoding="utf-8")
            done_flag.touch()
            print(f"[Marker {i:3d}/{len(pdfs)}] {p.stem} ({n_pages}p) -> {len(full_text)} chars ({time.time()-t0:.1f}s)", flush=True)
        except Exception as exc:
            print(f"[Marker {i:3d}/{len(pdfs)}] {p.stem} ERROR: {exc}", flush=True)
            gc.collect()


def run_docling(pdfs: list[Path], out_dir: Path, chunk_pages: int = 15) -> None:
    """Converts all PDFs in `pdfs` to full Markdown via Docling in `chunk_pages` slices."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    out_dir.mkdir(parents=True, exist_ok=True)
    opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True, document_timeout=180.0)
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    for i, p in enumerate(pdfs, 1):
        target = out_dir / f"{p.stem}.md"
        done_flag = out_dir / f"{p.stem}.full_ok"
        if done_flag.exists() and target.exists() and target.stat().st_size > 50:
            continue
        t0 = time.time()
        try:
            n_pages = get_num_pages(p)
            parts: list[str] = []
            for start in range(1, n_pages + 1, chunk_pages):
                end = min(start + chunk_pages - 1, n_pages)
                res = converter.convert(str(p), page_range=(start, end))
                parts.append(res.document.export_to_markdown())
                del res
                gc.collect()
            full_md = "\n\n".join(parts)
            target.write_text(full_md, encoding="utf-8")
            done_flag.touch()
            print(f"[Docling {i:3d}/{len(pdfs)}] {p.stem} ({n_pages}p) -> {len(full_md)} chars ({time.time()-t0:.1f}s)", flush=True)
        except Exception as exc:
            print(f"[Docling {i:3d}/{len(pdfs)}] {p.stem} ERROR: {exc}", flush=True)
            gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk-wise full-PDF converter (Marker / Docling)")
    parser.add_argument("--parser", choices=["marker", "docling", "both"], default="both")
    parser.add_argument("--split", choices=["dev", "train", "test"], default="dev")
    parser.add_argument("--chunk-pages", type=int, default=15)
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    pdf_dir = REPO_ROOT / "data" / "raw_pdfs" / args.split
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if args.parser in ("marker", "both"):
        run_marker(pdfs, REPO_ROOT / "data" / "parsed_text" / f"marker_{args.split}", args.chunk_pages)
    if args.parser in ("docling", "both"):
        run_docling(pdfs, REPO_ROOT / "data" / "parsed_text" / f"docling_{args.split}", args.chunk_pages)


if __name__ == "__main__":
    main()
