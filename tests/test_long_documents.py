"""The speed-ups for long documents must not change any result.

Compares the fast code with the slow original on an invented document with many pages
(tests/many_pages_document.py):
- page texts and full text: page_text.py vs. `export_to_markdown(page_no=...)` page by page;
- chunks: `_merge_neighbours` vs. `_merge_neighbours_by_retokenizing`, with tokenizers
  whose counts behave differently where two texts are joined.
And checks that document_statistics.py continues a stopped run without losing results.
"""
import json
import re
from pathlib import Path
from typing import Any, Callable

import pytest
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.types.doc import DoclingDocument

import chunking
import document_statistics
import runs
from chunking import PARAGRAPH_SEPARATOR, ChunkingConfig, chunk_document, load_tokenizer
from many_pages_document import build_many_pages_document
from page_text import page_markdown
from pdf_to_text import PAGE_MARKER, full_text, save_document


@pytest.fixture(scope="module")
def long_document() -> DoclingDocument:
    return build_many_pages_document(page_count=120)


def full_text_page_by_page(document: DoclingDocument) -> str:
    """full_text() as it was before the speed-up (one Docling export per page)."""
    return "\n\n".join(
        PAGE_MARKER.format(page_number=page) + "\n" + document.export_to_markdown(page_no=page)
        for page in sorted(document.pages)
    )


# ---------- page texts and full text ----------

def test_page_texts_equal_docling_export_per_page(long_document):
    fast = page_markdown(long_document)
    assert list(fast) == sorted(long_document.pages)
    for page, text in fast.items():
        assert text == long_document.export_to_markdown(page_no=page), f"page {page} differs"


def test_full_text_is_unchanged(long_document):
    assert full_text(long_document) == full_text_page_by_page(long_document)


def test_full_text_is_unchanged_after_saving_and_loading(long_document, tmp_path):
    path = tmp_path / "DOC00001.json"
    save_document(long_document, path)
    loaded = DoclingDocument.load_from_json(path)
    assert full_text(loaded) == full_text_page_by_page(loaded)


def test_the_fixture_contains_the_tricky_cases(long_document):
    """Guards the test above: it is only meaningful if these cases really occur."""
    pages = page_markdown(long_document)
    assert pages[13] == ""  # empty page
    assert "5. p7lnext0w0" in pages[8]  # numbered list continued on the next page
    assert "p9tabcapw0" in pages[10] and "p9tabr0c0" in pages[9]  # caption on another page
    assert "p17backw0" in pages[16]  # reading order goes back one page
    assert not any("noprov" in text or "Running header" in text for text in pages.values())


# ---------- chunks ----------

class JunctionTokenizer(BaseTokenizer):
    """Like a real BPE tokenizer, punctuation and a following blank line are one token,
    so a joined text has fewer tokens than its parts added up."""

    max_tokens: int

    def count_tokens(self, text: str) -> int:
        return len(re.findall(r"\S*[.,]\n\n|\n\n|\S+", text))

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self) -> Callable[[str], int]:
        return self.count_tokens


class GrowingJunctionTokenizer(JunctionTokenizer):
    """A tokenizer that breaks the rule the fast merge relies on: joining adds tokens."""

    def count_tokens(self, text: str) -> int:
        if not text.strip():
            return 0  # the separator alone is free, but costs 3 tokens inside a text
        return len(text.split()) + 3 * text.count(PARAGRAPH_SEPARATOR)


def chunks_with_original_merge(document: DoclingDocument, tokenizer: BaseTokenizer, config: ChunkingConfig,
                               monkeypatch: pytest.MonkeyPatch) -> list[chunking.Chunk]:
    with monkeypatch.context() as patch:
        patch.setattr(chunking, "_merge_neighbours", chunking._merge_neighbours_by_retokenizing)
        return chunk_document(document, tokenizer, config).chunks


@pytest.mark.parametrize("tokenizer_class", [JunctionTokenizer, GrowingJunctionTokenizer])
@pytest.mark.parametrize("max_tokens", [60, 150, 400, 1000])
def test_chunks_are_unchanged(long_document, monkeypatch, tokenizer_class, max_tokens):
    tokenizer = tokenizer_class(max_tokens=max_tokens)
    config = ChunkingConfig(max_chunk_tokens=max_tokens, chunk_overlap_tokens=20, min_headings=1)
    expected = chunks_with_original_merge(long_document, tokenizer, config, monkeypatch)
    assert chunk_document(long_document, tokenizer, config).chunks == expected


def test_chunks_are_unchanged_with_word_tokenizer(long_document, monkeypatch, word_tokenizer):
    tokenizer = word_tokenizer(300)
    config = ChunkingConfig(max_chunk_tokens=300, chunk_overlap_tokens=20, min_headings=1)
    expected = chunks_with_original_merge(long_document, tokenizer, config, monkeypatch)
    assert chunk_document(long_document, tokenizer, config).chunks == expected


def test_merge_with_tiny_pieces_uses_exact_counts_at_the_limit():
    """Many small pieces ending in "." : the sum is too high at every join, so only the
    exact count lets them be merged, as in the original."""
    tokenizer = JunctionTokenizer(max_tokens=20)
    pieces = [chunking._Piece(text=f"word{index} end.", token_count=2, headings=(), pages=(index,))
              for index in range(30)]
    fast = chunking._merge_neighbours(pieces, tokenizer, 20)
    assert fast == chunking._merge_neighbours_by_retokenizing(pieces, tokenizer, 20)
    assert len(fast) < len(pieces) / 2  # pieces were really merged


def test_merge_falls_back_to_original_when_joining_adds_tokens(monkeypatch):
    tokenizer = GrowingJunctionTokenizer(max_tokens=10)
    pieces = [chunking._Piece(text=f"a{index} b{index}", token_count=2, headings=(), pages=(index,))
              for index in range(12)]
    fallback_calls = []
    original = chunking._merge_neighbours_by_retokenizing

    def spy(*args):
        fallback_calls.append(args)
        return original(*args)

    monkeypatch.setattr(chunking, "_merge_neighbours_by_retokenizing", spy)
    merged = chunking._merge_neighbours(pieces, tokenizer, 10)
    assert fallback_calls  # the sum said "fits", the joined text did not
    assert merged == original(pieces, tokenizer, 10)
    assert all(piece.token_count <= 10 for piece in merged)


def test_chunks_are_unchanged_with_real_tokenizer(long_document, monkeypatch, settings):
    tokenizer = load_tokenizer(settings)
    config = ChunkingConfig.from_settings(settings)
    small = ChunkingConfig(max_chunk_tokens=500, chunk_overlap_tokens=50, min_headings=config.min_headings)
    for chunking_config in (config, small):
        expected = chunks_with_original_merge(long_document, tokenizer, chunking_config, monkeypatch)
        assert chunk_document(long_document, tokenizer, chunking_config).chunks == expected


# ---------- resumable statistics ----------

@pytest.fixture
def statistics_setup(settings, tmp_path, monkeypatch, word_tokenizer) -> dict[str, Any]:
    """Settings that point to a temporary folder with three invented converted documents."""
    monkeypatch.setattr(runs, "TIMESTAMP_FORMAT", "%Y%m%d-%H%M%S-%f")  # several runs per second
    test_settings = {**settings, "paths": {**settings["paths"], "docling": tmp_path / "docling",
                                           "runs": tmp_path / "runs"}}
    document_ids = ["DOC00001", "DOC00002", "DOC00003"]
    for index, document_id in enumerate(document_ids):
        document = build_many_pages_document(page_count=10 + 5 * index)
        save_document(document, test_settings["paths"]["docling"] / "dev" / f"{document_id}.json")
    monkeypatch.setattr(document_statistics, "document_ids_for_split", lambda _settings, _split: document_ids)
    monkeypatch.setattr(document_statistics, "load_tokenizer", lambda _settings: word_tokenizer(8000))
    return test_settings


def read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_stopped_statistics_run_continues_without_recomputing(statistics_setup, monkeypatch, capsys):
    complete_dir = document_statistics.collect_statistics(statistics_setup, "dev", None)

    original_record = document_statistics.document_record
    computed: list[str] = []

    def stop_at_second_document(document_id, *args):
        if document_id == "DOC00002":
            raise KeyboardInterrupt
        computed.append(document_id)
        return original_record(document_id, *args)

    monkeypatch.setattr(document_statistics, "document_record", stop_at_second_document)
    with pytest.raises(KeyboardInterrupt):
        document_statistics.collect_statistics(statistics_setup, "dev", None)
    stopped_dir = max((statistics_setup["paths"]["runs"]).iterdir())
    assert stopped_dir != complete_dir
    assert document_statistics.result_path(stopped_dir, "DOC00001").exists()
    assert not (stopped_dir / "summary.json").exists()

    def count_calls(document_id, *args):
        computed.append(document_id)
        return original_record(document_id, *args)

    monkeypatch.setattr(document_statistics, "document_record", count_calls)
    computed.clear()
    capsys.readouterr()
    document_statistics.collect_statistics(statistics_setup, "dev", None, resume_dir=stopped_dir)
    assert computed == ["DOC00002", "DOC00003"]  # DOC00001 was not computed again
    assert " s\n" in capsys.readouterr().out  # runtime per document is printed

    for name in ("documents.jsonl", "chunking_log.jsonl"):
        assert read_lines(stopped_dir / name) == read_lines(complete_dir / name)
    assert (json.loads((stopped_dir / "summary.json").read_text(encoding="utf-8"))
            == json.loads((complete_dir / "summary.json").read_text(encoding="utf-8")))


def test_resume_refuses_other_chunking_settings(statistics_setup):
    run_dir = document_statistics.collect_statistics(statistics_setup, "dev", 1)
    other_chunking = {**statistics_setup["chunking"],
                      "max_chunk_tokens": statistics_setup["chunking"]["max_chunk_tokens"] + 1}
    with pytest.raises(SystemExit, match="chunking"):
        document_statistics.collect_statistics({**statistics_setup, "chunking": other_chunking}, "dev", None,
                                               resume_dir=run_dir)
    with pytest.raises(SystemExit, match="split"):
        document_statistics.collect_statistics(statistics_setup, "test", None, resume_dir=run_dir)
