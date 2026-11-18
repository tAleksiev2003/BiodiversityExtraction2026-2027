"""Convert PDFs into Docling documents (text with headings and page numbers).

One converter for all document types (SPEC.md, section 3, step 1). Docling is slow,
especially with OCR on scanned pages, so every converted document is stored as Docling
JSON under `paths.docling/<split>/<id>.json` and never converted twice. A stopped
conversion (Ctrl + C) continues with the next missing document.

Usage (run by the team, it reads course data):
    python3 src/pdf_to_text.py --split dev               # the dev documents with a PDF
    python3 src/pdf_to_text.py --split dev --limit 3     # quick check on 3 documents
    python3 src/pdf_to_text.py --split test
    python3 src/pdf_to_text.py --split dev --limit 5 --workers 1 --output-dir data/processed/docling_speedtest_1

Every attempt appends one line to `<split>/conversion_log.jsonl`
(id, status, pages, seconds, errors), so the runtime per document can be measured
(SPEC.md, section 11, question 4).

Parallel conversion: one document takes minutes, mostly in Docling's models on one CPU
core at a time, so `pdf_conversion.max_workers` documents are converted at the same time,
each in its own process with its own converter (Docling's models are not shared between
threads; each process needs about 6 GB RAM). Results, skipping converted documents, the
log and Ctrl + C behave as with one worker: the main process writes the log, and Ctrl + C
stops all workers. The documents being converted at that moment are not saved (a
document file is written atomically, so no half-written file is left) and are converted
again by the next run.
"""
import argparse
import json
import multiprocessing
import os
import signal
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
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
    """Write under a temporary name, then rename: a stopped run never leaves a broken file
    that would be skipped as "already converted"."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    document.save_as_json(temporary_path)
    os.replace(temporary_path, path)


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


# One conversion task: the PDF and where to save its Docling JSON.
ConversionTask = tuple[Path, Path]


def convert_and_save(converter: DocumentConverter, task: ConversionTask) -> ConversionRecord:
    """Convert one PDF and save the document if usable; the same in both modes."""
    pdf_path, target_path = task
    document, record = convert_pdf(converter, pdf_path)
    if document is not None:
        save_document(document, target_path)
    return record


def convert_one_by_one(settings: dict[str, Any], tasks: list[ConversionTask]) -> Iterator[ConversionRecord]:
    """One converter in this process, documents one after the other."""
    converter = build_converter(settings)
    for task in tasks:
        yield convert_and_save(converter, task)


# Each worker process builds its converter once and keeps it for all its documents.
_worker_converter: DocumentConverter | None = None


def _start_worker(settings: dict[str, Any]) -> None:
    """Runs once in every worker process.

    Ctrl + C is ignored in the workers: the terminal sends it to every process, and the
    main process alone decides to stop them (otherwise every worker prints a traceback).
    """
    global _worker_converter
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _worker_converter = build_converter(settings)


def _convert_in_worker(task: ConversionTask) -> ConversionRecord:
    assert _worker_converter is not None, "worker started without _start_worker"
    return convert_and_save(_worker_converter, task)


class WorkerDiedError(RuntimeError):
    """A conversion process ended unexpectedly (often: out of memory)."""


def convert_in_parallel(
    settings: dict[str, Any], tasks: list[ConversionTask], workers: int
) -> Iterator[ConversionRecord]:
    """`workers` processes; records come back in the order the documents finish.

    ProcessPoolExecutor rather than multiprocessing.Pool: if a worker dies (e.g. out of
    memory), Pool waits forever, the executor raises BrokenProcessPool. "spawn" starts
    fresh processes (the default on macOS): forking a process that has already loaded
    PyTorch can hang on Linux.
    """
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_start_worker, initargs=(settings,))
    finished = False
    try:
        futures = [executor.submit(_convert_in_worker, task) for task in tasks]
        for future in as_completed(futures):
            try:
                yield future.result()
            except BrokenProcessPool as error:
                raise WorkerDiedError(
                    "A conversion process ended unexpectedly (often: out of memory; each worker "
                    "needs about 6 GB). Documents converted so far are saved; run the command "
                    "again, with fewer --workers if it happens again."
                ) from error
        finished = True
    finally:
        if finished:
            executor.shutdown(wait=True)
        else:
            _stop_workers_now(executor)


def _stop_workers_now(executor: ProcessPoolExecutor) -> None:
    """After Ctrl + C or an error: end the workers without waiting for their documents.

    Python 3.13 has no public way to do this (3.14 adds terminate_workers), so the
    executor's own list of processes is used.
    """
    processes = list((executor._processes or {}).values())
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        process.terminate()
    for process in processes:
        process.join()


def convert_documents(
    settings: dict[str, Any], split: str, document_ids: list[str], pdf_dir: Path, workers: int
) -> None:
    """Convert the documents that are not converted yet, log every attempt."""
    log_path = converted_dir(settings, split) / CONVERSION_LOG_NAME
    missing_ids = [doc_id for doc_id in document_ids if not converted_path(settings, split, doc_id).exists()]
    print(f"{split}: {len(document_ids)} documents, {len(document_ids) - len(missing_ids)} already converted")
    tasks = [(pdf_dir / f"{doc_id}.pdf", converted_path(settings, split, doc_id)) for doc_id in missing_ids]
    workers = max(1, min(workers, len(tasks)))
    print(f"Converting {len(tasks)} documents with {workers} worker(s)")

    started = time.perf_counter()
    records = convert_one_by_one(settings, tasks) if workers == 1 else convert_in_parallel(settings, tasks, workers)
    for position, record in enumerate(records, start=1):
        append_log(record, log_path)
        print(f"[{position}/{len(tasks)}] {record.document_id}: {record.status}, "
              f"{record.pages} pages, {record.seconds:.1f} s", flush=True)
    print(f"Wall-clock time: {time.perf_counter() - started:.1f} s for {len(tasks)} documents "
          f"with {workers} worker(s)")
    print(f"Log: {log_path}")


def convert_split(settings: dict[str, Any], split: str, limit: int | None, workers: int) -> None:
    """Convert all missing documents of one split."""
    document_ids = document_ids_for_split(settings, split)
    if limit is not None:
        document_ids = document_ids[:limit]
    convert_documents(settings, split, document_ids, settings["paths"][f"pdfs_{split}"], workers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PDFs with Docling.")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--limit", type=int, default=None, help="convert only the first N documents")
    parser.add_argument("--workers", type=int, default=None,
                        help="documents converted at the same time (default: pdf_conversion.max_workers)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="folder instead of paths.docling, e.g. to measure the speed-up separately")
    parser.add_argument("--experiment", type=Path, default=None)
    args = parser.parse_args()
    settings = load_settings(args.experiment)
    if args.output_dir is not None:
        settings["paths"]["docling"] = args.output_dir.resolve()
    workers = args.workers if args.workers is not None else settings["pdf_conversion"]["max_workers"]
    convert_split(settings, args.split, args.limit, workers)


if __name__ == "__main__":
    main()
