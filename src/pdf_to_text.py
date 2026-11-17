"""Convert PDFs into Docling documents (text with headings and page numbers).

One converter for all document types (SPEC.md, section 3, step 1). Docling is slow,
especially with OCR on scanned pages, so every converted document is stored as Docling
JSON under `paths.docling/<split>/<id>.json` and never converted twice. A stopped
conversion (Ctrl + C) continues with the next missing document.

Usage (run by the team, it reads course data):
    python3 src/pdf_to_text.py --split dev               # the dev documents with a PDF
    python3 src/pdf_to_text.py --split dev --limit 3     # quick check on 3 documents
    python3 src/pdf_to_text.py --split test

Every attempt appends one line to `<split>/conversion_log.jsonl`
(id, status, pages, seconds, errors), so the runtime per document can be measured
(SPEC.md, section 11, question 4).
"""
import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument

from data import dev_ids_with_pdf, list_pdf_ids
from page_text import page_markdown
from settings import load_settings

SPLITS = ("train", "dev", "test")
CONVERSION_LOG_NAME = "conversion_log.jsonl"
DOCUMENT_SUFFIX = ".json"
# Marker written before the text of every page in the full text, so quotes and labels
# can later be traced back to a page.
PAGE_MARKER = "[page {page_number}]"
# A partial success still contains most of the text; only a failure is discarded.
USABLE_STATUSES = {ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS}


@dataclass
class ConversionRecord:
    """One line of the conversion log."""

    document_id: str
    status: str
    pages: int | None
    seconds: float
    errors: list[str] = field(default_factory=list)


def build_converter(settings: dict[str, Any]) -> DocumentConverter:
    """Create one Docling converter from the `pdf_conversion` settings."""
    conversion_settings = settings["pdf_conversion"]
    pipeline_options = PdfPipelineOptions(
        do_ocr=conversion_settings["do_ocr"],
        do_table_structure=conversion_settings["do_table_structure"],
        document_timeout=conversion_settings["document_timeout_seconds"],
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def converted_dir(settings: dict[str, Any], split: str) -> Path:
    return settings["paths"]["docling"] / split


def converted_path(settings: dict[str, Any], split: str, document_id: str) -> Path:
    return converted_dir(settings, split) / f"{document_id}{DOCUMENT_SUFFIX}"


def convert_pdf(
    converter: DocumentConverter, pdf_path: Path
) -> tuple[DoclingDocument | None, ConversionRecord]:
    """Convert one PDF; return the document (None on failure) and a log record.

    Errors are caught and logged instead of raised, so one broken PDF does not stop
    the conversion of all other documents.
    """
    start = time.perf_counter()
    try:
        result = converter.convert(pdf_path, raises_on_error=False)
    except Exception as error:  # Docling raises different types for broken files
        seconds = time.perf_counter() - start
        record = ConversionRecord(pdf_path.stem, ConversionStatus.FAILURE.value, None, seconds, [repr(error)])
        return None, record
    seconds = time.perf_counter() - start
    errors = [error.error_message for error in result.errors]
    if result.status not in USABLE_STATUSES:
        return None, ConversionRecord(pdf_path.stem, result.status.value, None, seconds, errors)
    document = result.document
    record = ConversionRecord(pdf_path.stem, result.status.value, document.num_pages(), seconds, errors)
    return document, record


def save_document(document: DoclingDocument, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save_as_json(path)


def load_document(path: Path) -> DoclingDocument:
    return DoclingDocument.load_from_json(path)


def full_text(document: DoclingDocument) -> str:
    """The whole document as Markdown, with a page marker before every page.

    Pipeline F sends this text in one call, and the M1 statistics count its tokens.
    Page headers and footers are left out by Docling (they are not body text).
    The page texts are the same as `export_to_markdown(page_no=...)`, but computed in one
    go (see page_text.py; page by page took about an hour for a 2000-page book).
    """
    page_texts = [
        PAGE_MARKER.format(page_number=page_number) + "\n" + markdown
        for page_number, markdown in page_markdown(document).items()
    ]
    return "\n\n".join(page_texts)


def append_log(record: ConversionRecord, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_conversion_log(log_path: Path) -> dict[str, dict[str, Any]]:
    """Latest log entry per document id (a retried document has several entries)."""
    latest: dict[str, dict[str, Any]] = {}
    if not log_path.exists():
        return latest
    with log_path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                entry = json.loads(line)
                latest[entry["document_id"]] = entry
    return latest


def document_ids_for_split(settings: dict[str, Any], split: str) -> list[str]:
    """Dev: the fixed development subset with a PDF (as in the baselines); else all PDFs."""
    if split == "dev":
        return dev_ids_with_pdf(settings)
    return list_pdf_ids(settings["paths"][f"pdfs_{split}"])


def convert_split(settings: dict[str, Any], split: str, limit: int | None) -> None:
    """Convert all missing documents of one split and log every attempt."""
    document_ids = document_ids_for_split(settings, split)
    if limit is not None:
        document_ids = document_ids[:limit]
    pdf_dir = settings["paths"][f"pdfs_{split}"]
    log_path = converted_dir(settings, split) / CONVERSION_LOG_NAME
    missing_ids = [doc_id for doc_id in document_ids if not converted_path(settings, split, doc_id).exists()]
    print(f"{split}: {len(document_ids)} documents, {len(document_ids) - len(missing_ids)} already converted")

    converter = build_converter(settings)
    for position, document_id in enumerate(missing_ids, start=1):
        document, record = convert_pdf(converter, pdf_dir / f"{document_id}.pdf")
        if document is not None:
            save_document(document, converted_path(settings, split, document_id))
        append_log(record, log_path)
        print(f"[{position}/{len(missing_ids)}] {document_id}: {record.status}, "
              f"{record.pages} pages, {record.seconds:.1f} s")
    print(f"Log: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PDFs with Docling.")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--limit", type=int, default=None, help="convert only the first N documents")
    parser.add_argument("--experiment", type=Path, default=None)
    args = parser.parse_args()
    convert_split(load_settings(args.experiment), args.split, args.limit)


if __name__ == "__main__":
    main()
