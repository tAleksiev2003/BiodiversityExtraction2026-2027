"""An invented Docling document with many pages and the structures that make page export tricky.

Used by tests/test_page_text.py and tests/test_chunking.py to check that the fast code
gives exactly the same text and chunks as the slow original. All text is invented; every
word is unique, so a misplaced word would change the result.

What is in it, besides plain paragraphs and headings:
- lists (numbered, nested) that continue on the next page;
- tables and pictures whose caption is on another page, footnotes;
- page headers (furniture layer, not exported), an item without page;
- an empty page and a paragraph that comes after a later page in reading order;
- a chapter group and an inline group.
"""
from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    GroupLabel,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)

PAGE_SIZE = Size(width=600, height=800)
EMPTY_PAGE_EVERY = 13  # pages 13, 26, ... have no content at all


def _provenance(page: int, text: str = "") -> ProvenanceItem:
    return ProvenanceItem(page_no=page, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, len(text)))


def _words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}w{index}" for index in range(count))


def _table(prefix: str) -> TableData:
    cells = [
        TableCell(text=f"{prefix}r{row}c{col}", start_row_offset_idx=row, end_row_offset_idx=row + 1,
                  start_col_offset_idx=col, end_col_offset_idx=col + 1, column_header=row == 0)
        for row in range(3) for col in range(2)
    ]
    return TableData(num_rows=3, num_cols=2, table_cells=cells)


def build_many_pages_document(page_count: int = 120, paragraphs_per_page: int = 3,
                              words_per_paragraph: int = 40) -> DoclingDocument:
    document = DoclingDocument(name="invented_many_pages")
    for page in range(1, page_count + 1):
        document.add_page(page_no=page, size=PAGE_SIZE)
    document.add_title(text="Invented long report", prov=_provenance(1))
    document.add_text(label=DocItemLabel.TEXT, text="Paragraph without page " + _words("noprov", 5))

    chapter = None
    for page in range(1, page_count + 1):
        if page % EMPTY_PAGE_EVERY == 0:
            continue
        document.add_text(label=DocItemLabel.PAGE_HEADER, text=f"Running header {page}",
                          prov=_provenance(page), content_layer=ContentLayer.FURNITURE)
        if page % 10 == 1:
            chapter = document.add_group(label=GroupLabel.CHAPTER, name=f"chapter{page}")
            document.add_heading(text=f"Section {page}", prov=_provenance(page), parent=chapter)
        for index in range(paragraphs_per_page):
            text = _words(f"p{page}t{index}", words_per_paragraph) + "."  # "." + blank line = one token
            document.add_text(label=DocItemLabel.TEXT, text=text, prov=_provenance(page, text), parent=chapter)
        _add_special_items(document, page, page_count, chapter)
    return document


def _add_special_items(document: DoclingDocument, page: int, page_count: int, chapter) -> None:
    """Every few pages one of the structures that crosses page borders."""
    next_page = min(page + 1, page_count)
    if page % 7 == 0:  # a numbered list with a nested list, continued on the next page
        items = document.add_list_group(parent=chapter)
        for index in range(3):
            document.add_list_item(text=_words(f"p{page}li{index}", 6), enumerated=True,
                                   prov=_provenance(page), parent=items)
        nested = document.add_list_group(parent=items)
        document.add_list_item(text=_words(f"p{page}nested", 4), prov=_provenance(page), parent=nested)
        for index in range(2):
            document.add_list_item(text=_words(f"p{page}lnext{index}", 6), enumerated=True,
                                   prov=_provenance(next_page), parent=items)
    if page % 9 == 0:  # a table whose caption and footnote are printed on the next page
        # Top level (no chapter), so only the caption links the table to the next page.
        caption = document.add_text(label=DocItemLabel.CAPTION, text=_words(f"p{page}tabcap", 5),
                                    prov=_provenance(next_page))
        table = document.add_table(data=_table(f"p{page}tab"), caption=caption, prov=_provenance(page))
        footnote = document.add_text(label=DocItemLabel.FOOTNOTE, text=_words(f"p{page}foot", 4),
                                     prov=_provenance(next_page))
        table.footnotes.append(footnote.get_ref())
    if page % 11 == 0:  # a picture with its caption on the same page
        caption = document.add_text(label=DocItemLabel.CAPTION, text=_words(f"p{page}piccap", 5),
                                    prov=_provenance(page), parent=chapter)
        document.add_picture(caption=caption, prov=_provenance(page), parent=chapter)
    if page % 8 == 0:  # an inline group (e.g. a sentence with a formula)
        inline = document.add_inline_group(parent=chapter)
        document.add_text(label=DocItemLabel.TEXT, text=_words(f"p{page}inl", 3), prov=_provenance(page),
                          parent=inline)
        document.add_text(label=DocItemLabel.FORMULA, text="x = y", prov=_provenance(page), parent=inline)
    if page % 17 == 0 and page > 1:  # reading order goes back one page
        document.add_text(label=DocItemLabel.TEXT, text=_words(f"p{page}back", 5),
                          prov=_provenance(page - 1), parent=chapter)
