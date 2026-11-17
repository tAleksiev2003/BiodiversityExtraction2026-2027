"""Markdown text of every page of a Docling document, computed in one go.

Why this module exists: `DoclingDocument.export_to_markdown(page_no=p)` walks through the
whole document for every page and rebuilds the set of items to leave out each time. For
all pages one after the other, the time grows with pages x items (about an hour for a
2000-page book).

Here the same Docling Markdown serializer with the same settings writes each page, so the
text is identical to `export_to_markdown(page_no=p)` (tests/test_page_text.py checks this).
Only two things are faster:
- each page only walks the top-level parts of the document that have content on that page
  (all other parts would produce empty text for this page anyway);
- which items belong to which page is looked up once for the whole document.
"""
import sys
from dataclasses import dataclass
from typing import Any

from docling_core.transforms.serializer.base import SerializationResult
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
from docling_core.types.doc import (
    ContentLayer,
    DocItem,
    DoclingDocument,
    FloatingItem,
    NodeItem,
    RichTableCell,
)
from docling_core.types.doc.document import DEFAULT_CONTENT_LAYERS, DOCUMENT_TOKENS_EXPORT_LABELS
from pydantic import PrivateAttr

ALL_LAYERS = set(ContentLayer)
# Marks a top-level part that may produce text on every page (see _part_pages).
ON_EVERY_PAGE = None


def markdown_params(page: int) -> MarkdownParams:
    """The parameters that `export_to_markdown(page_no=page)` uses with its default arguments."""
    return MarkdownParams(labels=DOCUMENT_TOKENS_EXPORT_LABELS, layers=DEFAULT_CONTENT_LAYERS, pages={page})


def page_markdown(document: DoclingDocument) -> dict[int, str]:
    """Markdown text per page number, identical to `export_to_markdown(page_no=...)`."""
    serializer = _PageSerializer(doc=document, params=markdown_params(min(document.pages, default=1)))
    serializer.build_index()
    return {page: serializer.serialize_page(page) for page in sorted(document.pages)}


@dataclass(frozen=True)
class _PageExclusions:
    """Items that `export_to_markdown(page_no=page)` leaves out, without building a set.

    Same rule as Docling's `DocSerializer.get_excluded_refs` for one page: an item is left
    out if its label or content layer is not exported, if it has no page, or if its first
    page is another page. Rich table cells are left out together with their table.
    """

    page: int
    always: set[str]
    first_page: dict[str, int]
    table_pages_of_cell: dict[str, list[int]]

    def __contains__(self, ref: object) -> bool:
        if ref in self.always:
            return True
        if self.first_page.get(ref, self.page) != self.page:  # type: ignore[arg-type]
            return True
        return any(table_page != self.page for table_page in self.table_pages_of_cell.get(ref, ()))  # type: ignore[arg-type]


class _PageSerializer(MarkdownDocSerializer):
    """Docling's Markdown serializer, reused for all pages of one document."""

    _always_excluded: set[str] = PrivateAttr(default_factory=set)
    _first_page: dict[str, int] = PrivateAttr(default_factory=dict)
    _table_pages_of_cell: dict[str, list[int]] = PrivateAttr(default_factory=dict)
    _exclusions_per_page: dict[int, _PageExclusions] = PrivateAttr(default_factory=dict)
    _top_level_parts: list[NodeItem] = PrivateAttr(default_factory=list)
    _part_positions_per_page: dict[int, list[int]] = PrivateAttr(default_factory=dict)
    _part_positions_on_every_page: list[int] = PrivateAttr(default_factory=list)

    def build_index(self) -> None:
        """One pass over the document: the page of every item and of every top-level part."""
        self._index_exclusions()
        self._index_top_level_parts()

    def serialize_page(self, page: int) -> str:
        self.params = markdown_params(page)
        return self.serialize().text

    # ---------- which items are left out on a page ----------

    def _index_exclusions(self) -> None:
        labels, layers = self.params.labels, self.params.layers
        for item, _level in self.doc.iterate_items(with_groups=True, traverse_pictures=True,
                                                   included_content_layers=layers):
            if not isinstance(item, DocItem):
                continue
            if item.label not in labels or item.content_layer not in layers or not item.prov:
                self._always_excluded.add(item.self_ref)
            else:
                self._first_page[item.self_ref] = item.prov[0].page_no
        for table in self.doc.tables:
            cell_refs = self._rich_cell_refs(table)
            if table.self_ref in self._always_excluded or table.content_layer not in layers:
                self._always_excluded.add(table.self_ref)
                self._always_excluded.update(cell_refs)
            elif table.self_ref in self._first_page:
                for ref in cell_refs:
                    self._table_pages_of_cell.setdefault(ref, []).append(self._first_page[table.self_ref])

    def _rich_cell_refs(self, table: Any) -> set[str]:
        return {
            node.self_ref
            for cell in table.data.table_cells if isinstance(cell, RichTableCell)
            for node, _level in self.doc.iterate_items(root=cell.ref.resolve(self.doc), with_groups=True,
                                                       traverse_pictures=True, included_content_layers=ALL_LAYERS)
        }

    def get_excluded_refs(self, **kwargs: Any) -> Any:
        params = self.params.merge_with_patch(patch=kwargs)
        if not self._is_page_export(params):
            return super().get_excluded_refs(**kwargs)
        (page,) = params.pages
        if page not in self._exclusions_per_page:
            self._exclusions_per_page[page] = _PageExclusions(
                page, self._always_excluded, self._first_page, self._table_pages_of_cell
            )
        return self._exclusions_per_page[page]

    @staticmethod
    def _is_page_export(params: Any) -> bool:
        """True for the parameters of markdown_params(); anything else uses Docling's own code."""
        return (
            params.pages is not None and len(params.pages) == 1
            and params.labels == DOCUMENT_TOKENS_EXPORT_LABELS and params.layers == DEFAULT_CONTENT_LAYERS
            and params.start_idx == 0 and params.stop_idx == sys.maxsize
        )

    # ---------- which top-level parts have content on a page ----------

    def _index_top_level_parts(self) -> None:
        for child_ref in self.doc.body.children:
            part = child_ref.resolve(self.doc)
            if not isinstance(part, NodeItem):
                continue
            position = len(self._top_level_parts)
            self._top_level_parts.append(part)
            pages = self._part_pages(part)
            if pages is ON_EVERY_PAGE:
                self._part_positions_on_every_page.append(position)
                continue
            for page in pages:
                self._part_positions_per_page.setdefault(page, []).append(position)

    def _part_pages(self, part: NodeItem) -> set[int] | None:
        """All pages a top-level part can write text on (ON_EVERY_PAGE if unsure).

        Generous on purpose: every page of every item below the part, including captions
        and footnotes that are stored elsewhere. A group with metadata is not a page item,
        so Docling would write its metadata on every page.
        """
        pages: set[int] = set()
        to_visit = [part]
        seen: set[str] = set()
        while to_visit:
            root = to_visit.pop()
            for node, _level in self.doc.iterate_items(root=root, with_groups=True, traverse_pictures=True,
                                                       included_content_layers=ALL_LAYERS):
                if node.self_ref in seen:
                    continue
                seen.add(node.self_ref)
                if not isinstance(node, DocItem) and node.meta:
                    return ON_EVERY_PAGE
                if isinstance(node, DocItem):
                    pages.update(prov.page_no for prov in node.prov)
                if isinstance(node, FloatingItem):
                    for ref in [*node.captions, *node.footnotes, *node.references]:
                        to_visit.append(ref.resolve(self.doc))
        return pages

    def _parts_for_page(self, page: int) -> list[NodeItem]:
        """The top-level parts with content on `page`, in document order."""
        positions = set(self._part_positions_per_page.get(page, [])) | set(self._part_positions_on_every_page)
        return [self._top_level_parts[position] for position in sorted(positions)]

    def get_parts(
        self,
        item: NodeItem | None = None,
        *,
        traverse_pictures: bool = False,
        list_level: int = 0,
        is_inline_scope: bool = False,
        visited: set[str] | None = None,
        **kwargs: Any,
    ) -> list[SerializationResult]:
        """Like Docling's walk over the whole body, but only through the parts of one page.

        Same order, same `visited` bookkeeping and the same `level` as in Docling's walk
        (one deeper than inside the part, because the body is the root).
        """
        params = self.params.merge_with_patch(patch=kwargs)
        if item is not None or not self._is_page_export(params) or self.requires_page_break():
            return super().get_parts(item=item, traverse_pictures=traverse_pictures, list_level=list_level,
                                     is_inline_scope=is_inline_scope, visited=visited, **kwargs)
        (page,) = params.pages
        visited = visited if visited is not None else set()
        add_content = getattr(params, "add_content", True)
        visited.add(self.doc.body.self_ref)  # Docling's walk starts at the body, which writes nothing here
        parts = []
        for top_level_part in self._parts_for_page(page):
            for node, level in self.doc.iterate_items(root=top_level_part, with_groups=True,
                                                      traverse_pictures=params.traverse_pictures,
                                                      included_content_layers=params.layers):
                if node.self_ref in visited or self._skip_in_walk(node=node, root=None):
                    continue
                visited.add(node.self_ref)
                part = self.serialize(item=node, list_level=list_level, is_inline_scope=is_inline_scope,
                                      visited=visited, **(dict(level=level + 1) | kwargs))
                if part.text or not add_content:
                    parts.append(part)
        return parts
