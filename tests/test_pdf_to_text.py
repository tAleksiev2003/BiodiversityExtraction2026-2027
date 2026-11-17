"""Tests for src/pdf_to_text.py with real Docling on the two invented fixture PDFs.

These tests are slower (Docling loads its layout models, and the first run downloads
them and the tokenizer from HuggingFace). The PDFs are made by
tests/fixtures/make_fixture_pdfs.py.
"""
from pathlib import Path
from typing import Any

import pytest
from docling_core.types.doc import DoclingDocument

from chunking import LEVEL_FALLBACK, LEVEL_HEADINGS, ChunkingConfig, chunk_document, load_tokenizer
from pdf_to_text import (
    ConversionRecord,
    append_log,
    build_converter,
    convert_pdf,
    full_text,
    load_document,
    read_conversion_log,
    save_document,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def converted(settings: dict[str, Any]) -> dict[str, tuple[DoclingDocument | None, ConversionRecord]]:
    converter = build_converter(settings)
    return {name: convert_pdf(converter, FIXTURES / f"{name}.pdf") for name in ("with_headings", "without_headings")}


def test_pdf_with_headings_is_converted_with_pages(converted):
    document, record = converted["with_headings"]
    assert record.status == "success"
    assert record.pages == 2
    assert record.seconds > 0
    text = full_text(document)
    assert "[page 1]" in text and "[page 2]" in text
    assert text.index("[page 2]") < text.index("Carabus fictus")  # the results are on page 2


def test_chunking_levels_on_real_pdfs(converted, settings):
    tokenizer = load_tokenizer(settings)
    chunking_config = ChunkingConfig.from_settings(settings)
    with_headings = chunk_document(converted["with_headings"][0], tokenizer, chunking_config)
    without_headings = chunk_document(converted["without_headings"][0], tokenizer, chunking_config)
    assert with_headings.level == LEVEL_HEADINGS
    assert with_headings.heading_count >= 4
    assert without_headings.level == LEVEL_FALLBACK
    for result in (with_headings, without_headings):
        assert len(result.chunks) == 1  # tiny documents fit into one chunk
        assert result.chunks[0].token_count <= chunking_config.max_chunk_tokens
        assert (result.chunks[0].first_page, result.chunks[0].last_page) == (1, 2)
    assert "Discussion" in with_headings.chunks[0].text
    assert "Musterdorf" in without_headings.chunks[0].text


def test_saved_document_loads_with_same_text(converted, tmp_path):
    document, _record = converted["with_headings"]
    path = tmp_path / "dev" / "DOC00001.json"
    save_document(document, path)
    assert full_text(load_document(path)) == full_text(document)


def test_broken_pdf_is_logged_not_raised(settings, tmp_path):
    broken_pdf = tmp_path / "DOC99999.pdf"
    broken_pdf.write_bytes(b"this is not a pdf")
    document, record = convert_pdf(build_converter(settings), broken_pdf)
    assert document is None
    assert record.document_id == "DOC99999"
    assert record.status == "failure"
    assert record.pages is None


def test_conversion_log_keeps_latest_attempt(tmp_path):
    log_path = tmp_path / "conversion_log.jsonl"
    append_log(ConversionRecord("DOC00001", "failure", None, 1.0, ["timeout"]), log_path)
    append_log(ConversionRecord("DOC00001", "success", 3, 2.0), log_path)
    append_log(ConversionRecord("DOC00002", "success", 5, 4.0), log_path)
    latest = read_conversion_log(log_path)
    assert latest["DOC00001"]["status"] == "success"
    assert latest["DOC00001"]["pages"] == 3
    assert set(latest) == {"DOC00001", "DOC00002"}
