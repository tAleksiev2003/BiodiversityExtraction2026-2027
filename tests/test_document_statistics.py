"""Tests for the M1 summary in src/document_statistics.py on invented per-document records."""
from typing import Any

from chunking import LEVEL_FALLBACK, LEVEL_HEADINGS
from document_statistics import describe, summarise


def invented_record(document_id: str, pages: int, tokens: int, level: str, chunk_tokens: list[int],
                    split_count: int = 0) -> dict[str, Any]:
    return {
        "document_id": document_id, "conversion_status": "success", "conversion_seconds": 2.0 * pages,
        "pages": pages, "full_text_tokens": tokens, "chunking_level": level, "heading_count": 0,
        "chunk_count": len(chunk_tokens), "chunk_token_counts": chunk_tokens,
        "split_count": split_count, "borders_without_overlap": 0,
    }


def test_describe_empty_and_values():
    assert describe([]) is None
    assert describe([1, 2, 6]) == {"min": 1, "median": 2, "mean": 3, "max": 6}


def test_summary_counts_against_thresholds_from_settings(settings: dict[str, Any]):
    long_pages = settings["statistics"]["long_document_pages"]
    chunked_context = settings["models"]["chunked_context_tokens"]
    full_text_context = settings["models"]["full_text_context_tokens"]
    records = [
        invented_record("DOC00001", 10, 5_000, LEVEL_HEADINGS, [3000, 2000]),
        invented_record("DOC00002", long_pages + 1, chunked_context + 1, LEVEL_FALLBACK, [100], split_count=2),
        invented_record("DOC00003", long_pages + 100, full_text_context + 1, LEVEL_HEADINGS, [7000]),
        {"document_id": "DOC00004", "conversion_status": "failure", "conversion_seconds": 1.0},
    ]
    summary = summarise(records, settings)
    assert summary["documents_selected"] == 4
    assert summary["documents_converted"] == 3
    assert summary["documents_not_converted"] == ["DOC00004 (failure)"]
    assert summary[f"documents_over_{long_pages}_pages"] == 2
    assert summary[f"documents_over_{chunked_context}_tokens"] == 2
    assert summary[f"documents_over_{full_text_context}_tokens"] == 1
    assert summary["chunking_fallback"] == 1
    assert summary["fallback_share"] == round(1 / 3, 2)
    assert summary["chunks_total"] == 4
    assert summary["chunk_tokens"]["max"] == 7000
    assert summary["documents_with_splits"] == 1
    assert summary["conversion_seconds_per_page"]["median"] == 2.0
