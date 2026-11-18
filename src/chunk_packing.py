"""Chunk size per document for pipeline B: at most `max_chunks_per_document` chunks.

Why: every chunk is one LLM request. The M1 chunker splits at headings, so a book can
give hundreds of chunks (837 for one dev document), far more requests than the daily
limit allows. Long documents therefore get larger chunks (SPEC.md, section 13):

    effective size = max(max_chunk_tokens, ceil(document_tokens / max_chunks_per_document))

never above `models.chunked_part_tokens` (the model's context minus room for prompt and
answer). Typical papers keep `max_chunk_tokens`; only long documents get larger chunks.

How: the M1 chunker cuts the document into pieces of at most `max_chunk_tokens`; then
neighbouring pieces are packed together, in document order, up to the effective size.
A piece is never split. Packing whole pieces in order cannot always fill every chunk to
the limit, so it can need a few more than `max_chunks_per_document` chunks; then the
packing size is raised to the smallest size that gives at most that many chunks (still
never above `chunked_part_tokens`). Only if even the cap is not enough does a document
get more chunks; it is reported as capped.

Token counts of packed chunks are not re-tokenized (slow for long documents): the
counts of the pieces plus one separator each are added up. Joining texts does not add
tokens (see chunking._merge_neighbours), so this is an upper bound.
"""
import math
from dataclasses import dataclass
from typing import Any

from chunking import PARAGRAPH_SEPARATOR, Chunk


@dataclass(frozen=True)
class PackingConfig:
    max_chunk_tokens: int         # chunk size of normal documents (and of the chunker's pieces)
    max_chunks_per_document: int
    max_part_tokens: int          # cap: no chunk may be larger (model context)

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> "PackingConfig":
        return cls(
            max_chunk_tokens=settings["chunking"]["max_chunk_tokens"],
            max_chunks_per_document=settings["chunking"]["max_chunks_per_document"],
            max_part_tokens=settings["models"]["chunked_part_tokens"],
        )


@dataclass(frozen=True)
class PackedDocument:
    chunks: list[Chunk]
    chunk_tokens: int    # the packing size actually used for this document
    enlarged: bool       # chunk_tokens > max_chunk_tokens (a long document)
    capped: bool         # the cap was reached and the document has more chunks than allowed


def effective_chunk_tokens(document_tokens: int, config: PackingConfig) -> int:
    """The formula of SPEC.md, section 13, limited by the cap."""
    per_chunk = math.ceil(document_tokens / config.max_chunks_per_document)
    return min(max(config.max_chunk_tokens, per_chunk), config.max_part_tokens)


def _group_sizes(token_counts: list[int], capacity: int, separator_tokens: int) -> list[int]:
    """Greedy packing in order: how many pieces go into each chunk.

    Greedy is optimal here: for pieces in a fixed order, filling each chunk as far as
    possible gives the fewest chunks for a given capacity. A piece larger than the
    capacity gets a chunk of its own (it is never split).
    """
    sizes: list[int] = []
    current_tokens = 0
    for count in token_counts:
        if sizes and current_tokens + separator_tokens + count <= capacity:
            sizes[-1] += 1
            current_tokens += separator_tokens + count
        else:
            sizes.append(1)
            current_tokens = count
    return sizes


def _smallest_capacity(token_counts: list[int], lowest: int, highest: int,
                       max_chunks: int, separator_tokens: int) -> int | None:
    """The smallest capacity in [lowest, highest] that gives at most `max_chunks` chunks.

    The number of chunks only falls as the capacity grows, so a binary search finds it.
    None if even `highest` gives too many chunks.
    """
    if len(_group_sizes(token_counts, highest, separator_tokens)) > max_chunks:
        return None
    while lowest < highest:
        middle = (lowest + highest) // 2
        if len(_group_sizes(token_counts, middle, separator_tokens)) <= max_chunks:
            highest = middle
        else:
            lowest = middle + 1
    return lowest


def _join(pieces: list[Chunk], index: int, separator_tokens: int) -> Chunk:
    if len(pieces) == 1:
        return Chunk(index, pieces[0].text, pieces[0].token_count, pieces[0].headings,
                     pieces[0].first_page, pieces[0].last_page)
    pages = [page for piece in pieces for page in (piece.first_page, piece.last_page) if page is not None]
    return Chunk(
        index=index,
        text=PARAGRAPH_SEPARATOR.join(piece.text for piece in pieces),
        token_count=sum(piece.token_count for piece in pieces) + separator_tokens * (len(pieces) - 1),
        headings=pieces[0].headings,
        first_page=min(pages) if pages else None,
        last_page=max(pages) if pages else None,
    )


def pack_chunks(pieces: list[Chunk], document_tokens: int, config: PackingConfig,
                separator_tokens: int) -> PackedDocument:
    """Pack the chunker's pieces of one document into at most max_chunks_per_document chunks."""
    effective = effective_chunk_tokens(document_tokens, config)
    token_counts = [piece.token_count for piece in pieces]
    capacity = _smallest_capacity(token_counts, effective, config.max_part_tokens,
                                  config.max_chunks_per_document, separator_tokens)
    capped = capacity is None
    if capacity is None:
        capacity = config.max_part_tokens
    chunks = []
    start = 0
    for index, size in enumerate(_group_sizes(token_counts, capacity, separator_tokens)):
        chunks.append(_join(pieces[start:start + size], index, separator_tokens))
        start += size
    return PackedDocument(chunks, capacity, capacity > config.max_chunk_tokens, capped)
