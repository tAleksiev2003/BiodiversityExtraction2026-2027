"""Read the first pages of every PDF as plain text, for finding similar papers.

Prompt v4 shows the labels of the most similar TRAIN papers (src/retrieval.py). Finding
them needs a short text per paper for all splits, also the 908 train papers that were
never converted with Docling. Title, abstract and introduction say what a paper is
about, so the first `retrieval.first_pages_chars` characters are enough. PyMuPDF reads
the text layer directly: seconds per PDF instead of minutes with Docling, but no OCR,
so a scanned PDF without a text layer gives no text (logged as `no_text`).

Usage (run by the team, it reads course data):
    python3 src/first_pages.py --split train
    python3 src/first_pages.py --split dev --limit 3

Writes `paths.first_pages/<split>/<id>.txt` and skips documents that are already there,
so a stopped run continues. Every attempt appends a line to
`<split>/first_pages_log.jsonl`; documents that failed are tried again on the next run.
"""
import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pymupdf

from pdf_to_text import SPLITS, document_ids_for_split
from settings import load_settings

TEXT_SUFFIX = ".txt"
LOG_NAME = "first_pages_log.jsonl"
STATUS_DONE = "done"
STATUS_NO_TEXT = "no_text"   # e.g. a scanned PDF without a text layer
STATUS_FAILED = "failed"     # PyMuPDF could not open or read the file


@dataclass
class FirstPagesRecord:
    """One line of the log."""

    document_id: str
    status: str
    pages_read: int
    characters: int
    seconds: float
    error: str | None = None


def read_first_pages(pdf_path: Path, max_characters: int) -> tuple[str, int]:
    """Text of the first pages, at most `max_characters`; also returns how many pages were read.

    Pages are read only until the limit is reached, so a 400-page book costs as little
    as a short paper.
    """
    page_texts: list[str] = []
    characters = 0
    with pymupdf.open(pdf_path) as document:
        for page in document:
            page_text = page.get_text().strip()
            page_texts.append(page_text)
            characters += len(page_text)
            if characters >= max_characters:
                break
    text = "\n\n".join(page_texts)
    return text[:max_characters].strip(), len(page_texts)


def save_text(text: str, path: Path) -> None:
    """Write under a temporary name, then rename, so a stopped run leaves no half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, path)


def extract_one(pdf_path: Path, text_path: Path, max_characters: int) -> FirstPagesRecord:
    """Read and save one document. Errors are logged, not raised, so one broken PDF does
    not stop the others."""
    started = time.perf_counter()
    document_id = pdf_path.stem
    try:
        text, pages_read = read_first_pages(pdf_path, max_characters)
    except Exception as error:  # PyMuPDF raises different types for broken files
        return FirstPagesRecord(document_id, STATUS_FAILED, 0, 0,
                                time.perf_counter() - started, repr(error))
    seconds = time.perf_counter() - started
    if not text:
        return FirstPagesRecord(document_id, STATUS_NO_TEXT, pages_read, 0, seconds)
    save_text(text, text_path)
    return FirstPagesRecord(document_id, STATUS_DONE, pages_read, len(text), seconds)


def append_log(record: FirstPagesRecord, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def first_pages_dir(settings: dict[str, Any], split: str) -> Path:
    return settings["paths"]["first_pages"] / split


def extract_documents(pdf_dir: Path, document_ids: list[str], output_dir: Path,
                      max_characters: int) -> dict[str, int]:
    """Extract every document that has no text file yet; returns counts per status."""
    counts = {STATUS_DONE: 0, STATUS_NO_TEXT: 0, STATUS_FAILED: 0, "skipped": 0}
    log_path = output_dir / LOG_NAME
    for number, document_id in enumerate(document_ids, start=1):
        text_path = output_dir / f"{document_id}{TEXT_SUFFIX}"
        if text_path.exists():
            counts["skipped"] += 1
            continue
        record = extract_one(pdf_dir / f"{document_id}.pdf", text_path, max_characters)
        append_log(record, log_path)
        counts[record.status] += 1
        if record.status != STATUS_DONE:
            print(f"[{number}/{len(document_ids)}] {document_id}: {record.status}"
                  + (f" ({record.error})" if record.error else ""), flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Read the first pages of every PDF of a split.")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--limit", type=int, default=None, help="only the first N documents")
    parser.add_argument("--experiment", type=Path, default=None)
    args = parser.parse_args()

    settings = load_settings(args.experiment)
    document_ids = document_ids_for_split(settings, args.split)
    if args.limit is not None:
        document_ids = document_ids[:args.limit]
    output_dir = first_pages_dir(settings, args.split)
    started = time.perf_counter()
    counts = extract_documents(settings["paths"][f"pdfs_{args.split}"], document_ids, output_dir,
                               settings["retrieval"]["first_pages_chars"])
    print(f"{args.split}: {len(document_ids)} documents: {counts['done']} read, "
          f"{counts['skipped']} already there, {counts['no_text']} without text, "
          f"{counts['failed']} failed ({time.perf_counter() - started:.0f} s)")
    print(f"Written to: {output_dir} (log: {LOG_NAME})")


if __name__ == "__main__":
    main()
