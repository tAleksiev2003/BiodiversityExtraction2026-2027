"""M1 statistics: page counts, token lengths and chunking of the converted documents.

These numbers decide how the pipelines are run (SPEC.md, section 9, M1): documents
longer than the context window of the full-text model cannot use pipeline F, and the
number of chunks is the number of LLM requests per document in pipeline B.

Usage (run by the team after src/pdf_to_text.py, it reads course data):
    python3 src/document_statistics.py --split dev
    python3 src/document_statistics.py --split dev --resume runs/<timestamp>_m1_statistics_dev

Very long documents take minutes each. The result of every document is stored as soon as
it is computed (`documents/<id>.json` in the run folder), so a stopped run (Ctrl + C)
continues with `--resume <run folder>` and skips the documents that are already done.

Writes to runs/<timestamp>_m1_statistics_<split>/ when all documents are done:
- documents.jsonl: one line per document (numbers only, no text);
- chunking_log.jsonl: chunking level and every split per document;
- summary.json: the overall numbers that the team reports back.
"""
import argparse
import json
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml
from chunking import LEVEL_FALLBACK, ChunkingConfig, chunk_document, load_tokenizer
from data import write_jsonl
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from pdf_to_text import (
    CONVERSION_LOG_NAME,
    SPLITS,
    converted_dir,
    converted_path,
    document_ids_for_split,
    full_text,
    load_document,
    read_conversion_log,
)
from runs import create_run_dir
from settings import load_settings

ROUND_DIGITS = 2
PER_DOCUMENT_DIR = "documents"
CONFIG_NAME = "config.yaml"  # written by runs.create_run_dir
# Settings that change the per-document results; a resumed run must use the same values.
RESULT_SETTINGS = ("chunking",)


def describe(values: list[float]) -> dict[str, float] | None:
    """Min, median, mean and max of a list; None if the list is empty."""
    if not values:
        return None
    return {
        "min": round(min(values), ROUND_DIGITS),
        "median": round(median(values), ROUND_DIGITS),
        "mean": round(mean(values), ROUND_DIGITS),
        "max": round(max(values), ROUND_DIGITS),
    }


def document_record(
    document_id: str,
    conversion_entry: dict[str, Any] | None,
    document_path: Path,
    tokenizer: BaseTokenizer,
    config: ChunkingConfig,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Statistics of one document and its chunking log entry (None if not converted)."""
    record: dict[str, Any] = {
        "document_id": document_id,
        "conversion_status": conversion_entry["status"] if conversion_entry else "not converted",
        "conversion_seconds": conversion_entry["seconds"] if conversion_entry else None,
    }
    if not document_path.exists():
        return record, None
    document = load_document(document_path)
    result = chunk_document(document, tokenizer, config)
    chunking_entry = result.log_entry(document_id)
    record.update({
        "pages": document.num_pages(),
        "full_text_tokens": tokenizer.count_tokens(full_text(document)),
        "chunking_level": result.level,
        "heading_count": result.heading_count,
        "chunk_count": len(result.chunks),
        "chunk_token_counts": [chunk.token_count for chunk in result.chunks],
        "split_count": len(result.splits),
        "borders_without_overlap": result.borders_without_overlap,
    })
    return record, chunking_entry


def summarise(records: list[dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any]:
    """Overall M1 numbers from the per-document records (thresholds from the settings)."""
    long_pages = settings["statistics"]["long_document_pages"]
    chunked_context = settings["models"]["chunked_context_tokens"]
    full_text_context = settings["models"]["full_text_context_tokens"]
    converted = [record for record in records if "pages" in record]
    fallback = [record for record in converted if record["chunking_level"] == LEVEL_FALLBACK]
    pages = [record["pages"] for record in converted]
    tokens = [record["full_text_tokens"] for record in converted]
    seconds = [record["conversion_seconds"] for record in converted if record["conversion_seconds"] is not None]
    seconds_per_page = [
        record["conversion_seconds"] / record["pages"]
        for record in converted
        if record["conversion_seconds"] is not None and record["pages"]
    ]
    chunk_counts = [record["chunk_count"] for record in converted]
    all_chunk_tokens = [count for record in converted for count in record["chunk_token_counts"]]
    return {
        "documents_selected": len(records),
        "documents_converted": len(converted),
        "documents_not_converted": sorted(
            f"{record['document_id']} ({record['conversion_status']})"
            for record in records if "pages" not in record
        ),
        "pages": describe(pages),
        "pages_total": sum(pages),
        f"documents_over_{long_pages}_pages": sum(1 for count in pages if count > long_pages),
        "full_text_tokens": describe(tokens),
        f"documents_over_{chunked_context}_tokens": sum(1 for count in tokens if count > chunked_context),
        f"documents_over_{full_text_context}_tokens": sum(1 for count in tokens if count > full_text_context),
        "chunking_headings": len(converted) - len(fallback),
        "chunking_fallback": len(fallback),
        "fallback_share": round(len(fallback) / len(converted), ROUND_DIGITS) if converted else None,
        "chunks_total": sum(chunk_counts),
        "chunks_per_document": describe(chunk_counts),
        "chunk_tokens": describe(all_chunk_tokens),
        "documents_without_chunks": sum(1 for count in chunk_counts if count == 0),
        "documents_with_splits": sum(1 for record in converted if record["split_count"] > 0),
        "splits_total": sum(record["split_count"] for record in converted),
        "borders_without_overlap_total": sum(record["borders_without_overlap"] for record in converted),
        "conversion_seconds": describe(seconds),
        "conversion_seconds_total": round(sum(seconds), ROUND_DIGITS),
        "conversion_seconds_per_page": describe(seconds_per_page),
        "settings": {
            "tokenizer": settings["chunking"]["tokenizer"],
            "max_chunk_tokens": settings["chunking"]["max_chunk_tokens"],
            "chunk_overlap_tokens": settings["chunking"]["chunk_overlap_tokens"],
            "min_headings": settings["chunking"]["min_headings"],
            "do_ocr": settings["pdf_conversion"]["do_ocr"],
        },
    }


def result_path(run_dir: Path, document_id: str) -> Path:
    return run_dir / PER_DOCUMENT_DIR / f"{document_id}.json"


def save_result(run_dir: Path, document_id: str, record: dict[str, Any],
                chunking_entry: dict[str, Any] | None) -> None:
    """Store one document's result; written to a temporary file first, so Ctrl + C
    during writing cannot leave a half-written file that would count as done."""
    path = result_path(run_dir, document_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    content = {"record": record, "chunking_entry": chunking_entry}
    temporary_path.write_text(json.dumps(content, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def load_result(run_dir: Path, document_id: str) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
    """A stored result, or None if this document is not done yet."""
    path = result_path(run_dir, document_id)
    if not path.exists():
        return None
    content = json.loads(path.read_text(encoding="utf-8"))
    return content["record"], content["chunking_entry"]


def check_resumable(run_dir: Path, settings: dict[str, Any], split: str) -> None:
    """Stop if the run folder belongs to another split or used other chunking settings,
    because mixing results from different settings would give wrong statistics."""
    if not run_dir.name.endswith(f"_m1_statistics_{split}"):
        raise SystemExit(f"{run_dir} is not an M1 statistics run of split {split!r}")
    stored = yaml.safe_load((run_dir / CONFIG_NAME).read_text(encoding="utf-8"))
    for name in RESULT_SETTINGS:
        if stored.get(name) != settings.get(name):
            raise SystemExit(f"{run_dir} was run with other {name!r} settings: {stored.get(name)}")


def collect_statistics(settings: dict[str, Any], split: str, limit: int | None,
                       resume_dir: Path | None = None) -> Path:
    """Chunk every converted document of a split and write the M1 statistics."""
    document_ids = document_ids_for_split(settings, split)
    if limit is not None:
        document_ids = document_ids[:limit]
    if resume_dir is None:
        run_dir = create_run_dir(settings, f"m1_statistics_{split}")
    else:
        check_resumable(resume_dir, settings, split)
        run_dir = resume_dir
    done_count = sum(1 for document_id in document_ids if result_path(run_dir, document_id).exists())
    print(f"{split}: {len(document_ids)} documents, {done_count} already done")
    print(f"Run folder: {run_dir}")
    print(f"If stopped, continue with: python3 src/document_statistics.py --split {split} --resume {run_dir}")

    conversion_log = read_conversion_log(converted_dir(settings, split) / CONVERSION_LOG_NAME)
    tokenizer = load_tokenizer(settings)
    config = ChunkingConfig.from_settings(settings)

    records = []
    chunking_log = []
    for position, document_id in enumerate(document_ids, start=1):
        stored = load_result(run_dir, document_id)
        if stored is not None:
            record, chunking_entry = stored
        else:
            start = time.perf_counter()
            record, chunking_entry = document_record(
                document_id, conversion_log.get(document_id),
                converted_path(settings, split, document_id), tokenizer, config,
            )
            seconds = time.perf_counter() - start
            save_result(run_dir, document_id, record, chunking_entry)
            print(f"[{position}/{len(document_ids)}] {document_id}: "
                  f"{record.get('pages', '-')} pages, {record.get('chunk_count', '-')} chunks, "
                  f"{record.get('chunking_level', record['conversion_status'])}, {seconds:.1f} s")
        records.append(record)
        if chunking_entry is not None:
            chunking_log.append(chunking_entry)

    summary = {"split": split, **summarise(records, settings)}
    write_jsonl(records, run_dir / "documents.jsonl")
    write_jsonl(chunking_log, run_dir / "chunking_log.jsonl")
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Written: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="M1 statistics of converted documents.")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--limit", type=int, default=None, help="only the first N documents")
    parser.add_argument("--experiment", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None, metavar="RUN_DIR",
                        help="continue a stopped run in this folder (skips documents already done)")
    args = parser.parse_args()
    collect_statistics(load_settings(args.experiment), args.split, args.limit, args.resume)


if __name__ == "__main__":
    main()
