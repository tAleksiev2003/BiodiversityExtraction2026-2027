"""Tests for src/pdf_to_text.py with real Docling on the two invented fixture PDFs.

These tests are slower (Docling loads its layout models, and the first run downloads
them and the tokenizer from HuggingFace). The PDFs are made by
tests/fixtures/make_fixture_pdfs.py.
"""
import copy
from pathlib import Path
from typing import Any

import pytest
from docling_core.types.doc import DoclingDocument

from chunking import LEVEL_FALLBACK, LEVEL_HEADINGS, ChunkingConfig, chunk_document, load_tokenizer
from pdf_to_text import (
    CONVERSION_LOG_NAME,
    ConversionRecord,
    append_log,
    build_converter,
    WorkerDiedError,
    convert_documents,
    convert_in_parallel,
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


def test_parallel_conversion_gives_the_same_results_as_one_by_one(settings, tmp_path, capsys):
    """Two worker processes, one per fixture PDF, against the one-by-one conversion."""
    document_ids = ["with_headings", "without_headings"]
    outputs = {}
    for workers in (1, 2):
        run_settings = copy.deepcopy(settings)
        run_settings["paths"]["docling"] = tmp_path / f"workers_{workers}"
        convert_documents(run_settings, "dev", document_ids, FIXTURES, workers)
        assert f"with {workers} worker(s)" in capsys.readouterr().out
        folder = run_settings["paths"]["docling"] / "dev"
        log = read_conversion_log(folder / CONVERSION_LOG_NAME)
        outputs[workers] = {
            document_id: (full_text(load_document(folder / f"{document_id}.json")),
                          log[document_id]["status"], log[document_id]["pages"])
            for document_id in document_ids
        }
        assert not list(folder.glob("*.tmp"))  # documents are saved atomically
    assert outputs[1] == outputs[2]
    assert all(status == "success" for _text, status, _pages in outputs[2].values())


def test_converted_documents_are_skipped(settings, tmp_path, capsys):
    run_settings = copy.deepcopy(settings)
    run_settings["paths"]["docling"] = tmp_path / "docling"
    convert_documents(run_settings, "dev", ["with_headings"], FIXTURES, 2)
    capsys.readouterr()
    convert_documents(run_settings, "dev", ["with_headings"], FIXTURES, 2)
    output = capsys.readouterr().out
    assert "1 already converted" in output
    assert "Converting 0 documents" in output
    log_lines = (tmp_path / "docling" / "dev" / CONVERSION_LOG_NAME).read_text().splitlines()
    assert len(log_lines) == 1  # the second run converted nothing


def test_a_dying_worker_raises_instead_of_hanging(tmp_path):
    """Workers that cannot start (here: settings without `pdf_conversion`) must not hang the run."""
    broken_settings = {"paths": {"docling": tmp_path / "docling"}}
    tasks = [(FIXTURES / "with_headings.pdf", tmp_path / "with_headings.json")]
    with pytest.raises(WorkerDiedError):
        list(convert_in_parallel(broken_settings, tasks, 2))
