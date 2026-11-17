"""Split a Docling document into chunks for pipelines A and B (SPEC.md, section 3, step 2).

Two levels:
- headings found: Docling's HybridChunker splits at headings and splits sections that
  are longer than `max_chunk_tokens` (first at paragraph boundaries, inside a paragraph
  only if one paragraph alone is too long). Small neighbouring sections are then merged
  up to `max_chunk_tokens`, because many tiny chunks would waste LLM requests.
- no headings found (fallback, e.g. scanned books): fixed windows of up to
  `max_chunk_tokens` at paragraph boundaries, with about `chunk_overlap_tokens` of
  overlap, so a sentence near a window border is seen in full at least once.

Nothing is truncated silently: every place where text had to be cut is listed in
`ChunkingResult.splits`, and the result reports the level that was used.
"""
from dataclasses import dataclass, field
from statistics import median
from typing import Any

import semchunk
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DocItemLabel, DoclingDocument, TableItem, TextItem

LEVEL_HEADINGS = "headings"
LEVEL_FALLBACK = "fallback"
# Only section headings count; a document title alone gives no structure to split at.
HEADING_LABELS = {DocItemLabel.SECTION_HEADER}
PARAGRAPH_SEPARATOR = "\n\n"
NO_HEADING = "(text before the first heading)"


@dataclass(frozen=True)
class ChunkingConfig:
    max_chunk_tokens: int
    chunk_overlap_tokens: int
    min_headings: int

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> "ChunkingConfig":
        chunking = settings["chunking"]
        return cls(
            max_chunk_tokens=chunking["max_chunk_tokens"],
            chunk_overlap_tokens=chunking["chunk_overlap_tokens"],
            min_headings=chunking["min_headings"],
        )


@dataclass(frozen=True)
class Chunk:
    """One piece of text that is sent to the LLM in one call."""

    index: int
    text: str
    token_count: int
    headings: tuple[str, ...]  # heading path of the first section in the chunk
    first_page: int | None
    last_page: int | None


@dataclass
class ChunkingResult:
    level: str  # LEVEL_HEADINGS or LEVEL_FALLBACK
    heading_count: int
    chunks: list[Chunk]
    # One entry per cut: where it happened, into how many parts, and why.
    splits: list[dict[str, Any]] = field(default_factory=list)
    # Fallback only: window borders without overlap because the last paragraph was
    # longer than chunk_overlap_tokens.
    borders_without_overlap: int = 0

    def log_entry(self, document_id: str) -> dict[str, Any]:
        """Summary for the chunking log (numbers and headings only, no body text)."""
        token_counts = [chunk.token_count for chunk in self.chunks]
        return {
            "document_id": document_id,
            "level": self.level,
            "heading_count": self.heading_count,
            "chunk_count": len(self.chunks),
            "chunk_tokens_min": min(token_counts, default=0),
            "chunk_tokens_median": median(token_counts) if token_counts else 0,
            "chunk_tokens_max": max(token_counts, default=0),
            "split_count": len(self.splits),
            "splits": self.splits,
            "borders_without_overlap": self.borders_without_overlap,
        }


@dataclass(frozen=True)
class _Piece:
    """Intermediate text block before neighbours are merged into chunks."""

    text: str
    token_count: int
    headings: tuple[str, ...]
    pages: tuple[int, ...]


def load_tokenizer(settings: dict[str, Any]) -> BaseTokenizer:
    """The tokenizer of the chunked model, so chunk sizes match what the model sees."""
    chunking = settings["chunking"]
    return HuggingFaceTokenizer.from_pretrained(chunking["tokenizer"], max_tokens=chunking["max_chunk_tokens"])


def count_headings(document: DoclingDocument) -> int:
    return sum(1 for item, _level in document.iterate_items() if item.label in HEADING_LABELS)


def chunk_document(document: DoclingDocument, tokenizer: BaseTokenizer, config: ChunkingConfig) -> ChunkingResult:
    """Choose the chunking level from the number of headings and chunk the document."""
    heading_count = count_headings(document)
    if heading_count >= config.min_headings:
        return _chunk_by_headings(document, tokenizer, config, heading_count)
    return _chunk_by_windows(document, tokenizer, config, heading_count)


# ---------- level 1: headings ----------

def _item_pages(items: list[Any]) -> tuple[int, ...]:
    return tuple(sorted({prov.page_no for item in items for prov in getattr(item, "prov", [])}))


def _chunk_by_headings(
    document: DoclingDocument, tokenizer: BaseTokenizer, config: ChunkingConfig, heading_count: int
) -> ChunkingResult:
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
    pieces = []
    item_refs = []
    for docling_chunk in chunker.chunk(document):
        text = chunker.contextualize(docling_chunk)  # heading path + text
        pieces.append(_Piece(
            text=text,
            token_count=tokenizer.count_tokens(text),
            headings=tuple(docling_chunk.meta.headings or ()),
            pages=_item_pages(docling_chunk.meta.doc_items),
        ))
        item_refs.append(tuple(item.self_ref for item in docling_chunk.meta.doc_items))
    splits = _find_section_splits(pieces) + _find_item_splits(pieces, item_refs)
    merged = _merge_neighbours(pieces, tokenizer, config.max_chunk_tokens)
    return ChunkingResult(LEVEL_HEADINGS, heading_count, _to_chunks(merged), splits)


def _heading_name(headings: tuple[str, ...]) -> str:
    return " > ".join(headings) if headings else NO_HEADING


def _find_section_splits(pieces: list[_Piece]) -> list[dict[str, Any]]:
    """Sections that HybridChunker had to cut into several pieces.

    HybridChunker already merges pieces with the same headings up to the token limit, so
    neighbouring pieces with equal headings mean the section was too long.
    """
    splits = []
    run_start = 0
    for position in range(1, len(pieces) + 1):
        run_ended = position == len(pieces) or pieces[position].headings != pieces[run_start].headings
        if not run_ended:
            continue
        part_count = position - run_start
        if part_count > 1:
            splits.append({
                "kind": "section",
                "where": _heading_name(pieces[run_start].headings),
                "parts": part_count,
                "reason": "section longer than max_chunk_tokens, split between paragraphs",
            })
        run_start = position
    return splits


def _find_item_splits(pieces: list[_Piece], item_refs: list[tuple[str, ...]]) -> list[dict[str, Any]]:
    """Single paragraphs or tables that were longer than the limit and cut inside.

    An item that was cut appears in the metadata of more than one piece.
    """
    positions_per_item: dict[str, list[int]] = {}
    for position, refs in enumerate(item_refs):
        for ref in refs:
            positions_per_item.setdefault(ref, []).append(position)
    splits = []
    for positions in positions_per_item.values():
        if len(positions) > 1:
            first_piece = pieces[positions[0]]
            splits.append({
                "kind": "paragraph",
                "where": _heading_name(first_piece.headings),
                "pages": list(first_piece.pages),
                "parts": len(positions),
                "reason": "one paragraph or table longer than max_chunk_tokens, split inside it",
            })
    return splits


def _merge_neighbours(pieces: list[_Piece], tokenizer: BaseTokenizer, max_tokens: int) -> list[_Piece]:
    """Greedily join neighbouring pieces while the joined text stays within max_tokens.

    Gives the same chunks as `_merge_neighbours_by_retokenizing`, without tokenizing the
    growing text at every step (slow for long documents). The known token counts plus the
    separator are added up instead. Joining two texts does not add tokens; it can only
    save one where they meet (".\\n\\n" is a single token in Qwen). So if the sum fits, the
    joined text fits too. Only if the sum is over the limit is the joined text tokenized
    to decide. If a joined chunk is still over the limit, that rule did not hold for this
    tokenizer, and the slow original is used for the whole document.
    """
    separator_tokens = tokenizer.count_tokens(PARAGRAPH_SEPARATOR)
    groups: list[list[_Piece]] = []
    group_tokens: list[int] = []  # token count of each joined group, or a bit more
    for piece in pieces:
        if groups:
            estimate = group_tokens[-1] + separator_tokens + piece.token_count
            if estimate > max_tokens:
                estimate = tokenizer.count_tokens(_join_texts(groups[-1] + [piece]))
            if estimate <= max_tokens:
                groups[-1].append(piece)
                group_tokens[-1] = estimate
                continue
        groups.append([piece])
        group_tokens.append(piece.token_count)

    merged = [_join_pieces(group, tokenizer) for group in groups]
    if any(len(group) > 1 and joined.token_count > max_tokens for group, joined in zip(groups, merged)):
        return _merge_neighbours_by_retokenizing(pieces, tokenizer, max_tokens)
    return merged


def _join_texts(pieces: list[_Piece]) -> str:
    return PARAGRAPH_SEPARATOR.join(piece.text for piece in pieces)


def _join_pieces(pieces: list[_Piece], tokenizer: BaseTokenizer) -> _Piece:
    """One piece from neighbouring pieces; the token count is exact (tokenized once)."""
    if len(pieces) == 1:
        return pieces[0]
    text = _join_texts(pieces)
    return _Piece(
        text=text,
        token_count=tokenizer.count_tokens(text),
        headings=pieces[0].headings,
        pages=tuple(sorted({page for piece in pieces for page in piece.pages})),
    )


def _merge_neighbours_by_retokenizing(
    pieces: list[_Piece], tokenizer: BaseTokenizer, max_tokens: int
) -> list[_Piece]:
    """The original merge: tokenizes the joined text at every step.

    Kept as the safe fallback of `_merge_neighbours` and as the reference in the tests.
    """
    merged: list[_Piece] = []
    for piece in pieces:
        if merged:
            candidate_text = merged[-1].text + PARAGRAPH_SEPARATOR + piece.text
            candidate_tokens = tokenizer.count_tokens(candidate_text)
            if candidate_tokens <= max_tokens:
                previous = merged[-1]
                merged[-1] = _Piece(
                    text=candidate_text,
                    token_count=candidate_tokens,
                    headings=previous.headings,
                    pages=tuple(sorted(set(previous.pages) | set(piece.pages))),
                )
                continue
        merged.append(piece)
    return merged


def _to_chunks(pieces: list[_Piece]) -> list[Chunk]:
    return [
        Chunk(
            index=index,
            text=piece.text,
            token_count=piece.token_count,
            headings=piece.headings,
            first_page=piece.pages[0] if piece.pages else None,
            last_page=piece.pages[-1] if piece.pages else None,
        )
        for index, piece in enumerate(pieces)
    ]


# ---------- level 2: fallback windows ----------

def _paragraph_text(item: Any, document: DoclingDocument) -> str:
    if isinstance(item, TableItem):
        return item.export_to_markdown(doc=document)
    if isinstance(item, TextItem):
        return item.text
    return ""  # pictures and groups carry no text of their own


def _paragraph_pieces(
    document: DoclingDocument, tokenizer: BaseTokenizer, max_paragraph_tokens: int
) -> tuple[list[_Piece], list[dict[str, Any]]]:
    """All paragraphs in reading order; paragraphs that are too long are cut and logged.

    A paragraph may use at most `max_chunk_tokens - chunk_overlap_tokens`, so it always
    fits into a window together with the overlap from the previous window.
    """
    splitter = semchunk.chunkerify(tokenizer.get_tokenizer(), chunk_size=max_paragraph_tokens)
    pieces = []
    splits = []
    for item, _level in document.iterate_items():
        text = _paragraph_text(item, document).strip()
        if not text:
            continue
        pages = _item_pages([item])
        token_count = tokenizer.count_tokens(text)
        if token_count <= max_paragraph_tokens:
            pieces.append(_Piece(text, token_count, (), pages))
            continue
        parts = splitter(text)
        splits.append({
            "kind": "paragraph",
            "where": NO_HEADING,
            "pages": list(pages),
            "parts": len(parts),
            "reason": "one paragraph longer than max_chunk_tokens - chunk_overlap_tokens, split inside it",
        })
        pieces.extend(_Piece(part, tokenizer.count_tokens(part), (), pages) for part in parts)
    return pieces, splits


def _window_tokens(paragraphs: list[_Piece], separator_tokens: int) -> int:
    """Token count of paragraphs joined by PARAGRAPH_SEPARATOR, without re-tokenizing."""
    if not paragraphs:
        return 0
    return sum(paragraph.token_count for paragraph in paragraphs) + separator_tokens * (len(paragraphs) - 1)


def _overlap_tail(window: list[_Piece], overlap_tokens: int, separator_tokens: int) -> list[_Piece]:
    """The last whole paragraphs of a window that fit into the overlap budget."""
    tail: list[_Piece] = []
    for paragraph in reversed(window):
        if _window_tokens([paragraph] + tail, separator_tokens) > overlap_tokens:
            break
        tail.insert(0, paragraph)
    return tail


def _chunk_by_windows(
    document: DoclingDocument, tokenizer: BaseTokenizer, config: ChunkingConfig, heading_count: int
) -> ChunkingResult:
    separator_tokens = tokenizer.count_tokens(PARAGRAPH_SEPARATOR)
    max_paragraph_tokens = config.max_chunk_tokens - config.chunk_overlap_tokens
    paragraphs, splits = _paragraph_pieces(document, tokenizer, max_paragraph_tokens)

    windows: list[list[_Piece]] = []
    current: list[_Piece] = []
    borders_without_overlap = 0
    for paragraph in paragraphs:
        if current and _window_tokens(current + [paragraph], separator_tokens) > config.max_chunk_tokens:
            windows.append(current)
            current = _overlap_tail(current, config.chunk_overlap_tokens, separator_tokens)
            if not current and config.chunk_overlap_tokens > 0:
                borders_without_overlap += 1
        current.append(paragraph)
    if current:
        windows.append(current)

    pieces = []
    for window in windows:
        text = PARAGRAPH_SEPARATOR.join(paragraph.text for paragraph in window)
        pages = tuple(sorted({page for paragraph in window for page in paragraph.pages}))
        pieces.append(_Piece(text, tokenizer.count_tokens(text), (), pages))
    return ChunkingResult(LEVEL_FALLBACK, heading_count, _to_chunks(pieces), splits, borders_without_overlap)
