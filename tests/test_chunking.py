"""Tests for src/chunking.py on small invented Docling documents.

Documents are built in code; every word is unique (e.g. "s1p2w3" = section 1,
paragraph 2, word 3), so we can check that no word is lost.
Sizes are counted in words (WordTokenizer from conftest.py).
"""
from typing import Callable

from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
)

from chunking import LEVEL_FALLBACK, LEVEL_HEADINGS, ChunkingConfig, chunk_document

PAGE_SIZE = Size(width=600, height=800)


def words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}w{index}" for index in range(count))


def provenance(page: int, text: str) -> ProvenanceItem:
    return ProvenanceItem(page_no=page, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, len(text)))


def build_document(
    sections: list[tuple[str | None, list[str]]], title: str | None = None, pages: list[int] | None = None
) -> DoclingDocument:
    """Sections as (heading or None, paragraphs). `pages` gives the page of each paragraph."""
    document = DoclingDocument(name="invented")
    for page_number in range(1, max(pages or [1]) + 1):
        document.add_page(page_no=page_number, size=PAGE_SIZE)
    if title:
        document.add_title(text=title, prov=provenance(1, title))
    paragraph_number = 0
    for heading, paragraphs in sections:
        if heading:
            document.add_heading(text=heading, prov=provenance(1, heading))
        for text in paragraphs:
            page = pages[paragraph_number] if pages else 1
            document.add_text(label=DocItemLabel.TEXT, text=text, prov=provenance(page, text))
            paragraph_number += 1
    return document


def all_words(texts: list[str]) -> set[str]:
    return {word for text in texts for word in text.split()}


def config(max_tokens: int, overlap: int = 0, min_headings: int = 1) -> ChunkingConfig:
    return ChunkingConfig(max_chunk_tokens=max_tokens, chunk_overlap_tokens=overlap, min_headings=min_headings)


# ---------- headings ----------

def test_small_sections_are_merged_into_one_chunk(word_tokenizer: Callable):
    sections = [(f"Heading{index}", [words(f"s{index}p0", 10)]) for index in range(3)]
    result = chunk_document(build_document(sections), word_tokenizer(1000), config(1000))
    assert result.level == LEVEL_HEADINGS
    assert result.heading_count == 3
    assert len(result.chunks) == 1
    assert result.splits == []
    for index in range(3):
        assert f"Heading{index}" in result.chunks[0].text


def test_long_section_is_split_between_paragraphs_and_logged(word_tokenizer: Callable):
    paragraphs = [words(f"s0p{index}", 30) for index in range(3)]
    result = chunk_document(build_document([("Methods", paragraphs)]), word_tokenizer(50), config(50))
    assert len(result.chunks) == 3
    assert all(chunk.token_count <= 50 for chunk in result.chunks)
    assert all(chunk.headings == ("Methods",) for chunk in result.chunks)
    section_splits = [split for split in result.splits if split["kind"] == "section"]
    assert section_splits == [{
        "kind": "section", "where": "Methods", "parts": 3,
        "reason": "section longer than max_chunk_tokens, split between paragraphs",
    }]
    assert all_words(paragraphs) <= all_words([chunk.text for chunk in result.chunks])


def test_paragraph_longer_than_limit_is_split_inside_and_logged(word_tokenizer: Callable):
    long_paragraph = words("s0p0", 120)
    result = chunk_document(build_document([("Results", [long_paragraph])]), word_tokenizer(50), config(50))
    assert all(chunk.token_count <= 50 for chunk in result.chunks)
    assert any(split["kind"] == "paragraph" and split["where"] == "Results" for split in result.splits)
    assert all_words([long_paragraph]) <= all_words([chunk.text for chunk in result.chunks])


def test_chunks_keep_first_and_last_page(word_tokenizer: Callable):
    document = build_document([("Intro", [words("a", 5), words("b", 5)])], pages=[1, 2])
    result = chunk_document(document, word_tokenizer(1000), config(1000))
    assert (result.chunks[0].first_page, result.chunks[0].last_page) == (1, 2)


# ---------- fallback ----------

def test_title_alone_is_not_a_heading(word_tokenizer: Callable):
    document = build_document([(None, [words("p0", 10)])], title="An invented title")
    result = chunk_document(document, word_tokenizer(1000), config(1000))
    assert result.level == LEVEL_FALLBACK
    assert result.heading_count == 0


def test_min_headings_setting_decides_the_level(word_tokenizer: Callable):
    document = build_document([("Only heading", [words("p0", 10)])])
    result = chunk_document(document, word_tokenizer(1000), config(1000, min_headings=2))
    assert result.level == LEVEL_FALLBACK


def test_fallback_windows_respect_limit_and_overlap(word_tokenizer: Callable):
    paragraphs = [words(f"p{index}", 20) for index in range(10)]
    result = chunk_document(build_document([(None, paragraphs)]), word_tokenizer(70), config(70, overlap=25))
    assert result.level == LEVEL_FALLBACK
    assert len(result.chunks) > 1
    assert all(chunk.token_count <= 70 for chunk in result.chunks)
    for previous, following in zip(result.chunks, result.chunks[1:]):
        last_paragraph_of_previous = previous.text.split("\n\n")[-1]
        assert following.text.startswith(last_paragraph_of_previous)  # 20 words <= 25 overlap
    assert result.borders_without_overlap == 0
    assert all_words(paragraphs) == all_words([chunk.text for chunk in result.chunks])


def test_fallback_counts_borders_without_overlap(word_tokenizer: Callable):
    paragraphs = [words(f"p{index}", 30) for index in range(4)]
    result = chunk_document(build_document([(None, paragraphs)]), word_tokenizer(70), config(70, overlap=10))
    assert len(result.chunks) == 2
    assert result.borders_without_overlap == 1  # a 30-word paragraph does not fit into 10


def test_fallback_splits_long_paragraph_and_logs_it(word_tokenizer: Callable):
    long_paragraph = words("p0", 200)
    result = chunk_document(build_document([(None, [long_paragraph])]), word_tokenizer(70), config(70, overlap=20))
    assert all(chunk.token_count <= 70 for chunk in result.chunks)
    assert [split["kind"] for split in result.splits] == ["paragraph"]
    assert result.splits[0]["parts"] >= 4  # pieces of at most 70 - 20 = 50 words
    assert all_words([long_paragraph]) == all_words([chunk.text for chunk in result.chunks])


def test_empty_document_gives_no_chunks(word_tokenizer: Callable):
    result = chunk_document(build_document([]), word_tokenizer(100), config(100))
    assert result.chunks == []
    assert result.log_entry("DOC00001")["chunk_count"] == 0


# ---------- settings and log ----------

def test_config_is_read_from_settings(settings: dict):
    chunking_config = ChunkingConfig.from_settings(settings)
    assert chunking_config.max_chunk_tokens == settings["chunking"]["max_chunk_tokens"]
    assert chunking_config.chunk_overlap_tokens == settings["chunking"]["chunk_overlap_tokens"]
    assert chunking_config.min_headings == settings["chunking"]["min_headings"]


def test_log_entry_contains_numbers_but_no_body_text(word_tokenizer: Callable):
    paragraphs = [words(f"s0p{index}", 30) for index in range(3)]
    result = chunk_document(build_document([("Methods", paragraphs)]), word_tokenizer(50), config(50))
    entry = result.log_entry("DOC00001")
    assert entry["level"] == LEVEL_HEADINGS
    assert entry["chunk_count"] == 3
    assert entry["chunk_tokens_max"] <= 50
    assert entry["split_count"] == len(result.splits)
    assert "s0p0w0" not in str(entry)
