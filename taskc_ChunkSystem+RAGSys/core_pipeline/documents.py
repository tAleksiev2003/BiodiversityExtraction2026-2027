"""Convert PDFs to TEI and create bounded chunks.

The default path keeps section order and splits by characters. An optional
page path uses GROBID coordinates and groups a fixed number of physical pages.
"""
import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import httpx

NS = {"t": "http://www.tei-c.org/ns/1.0"}
# Website chrome that GROBID sometimes keeps as a body section.
_CHROME = re.compile(
    r"jetzt spenden|cookie|newsletter|impressum|datenschutz|"
    r"share (on|this)|navigation|subscribe|sign in|log in",
    re.IGNORECASE)


def pdf_to_tei(pdf, endpoint, cache, coordinates=False):
    """Call GROBID once per PDF content and reuse its cached TEI result.

    coordinates=False is the original full-text request. coordinates=True also
    asks for sentence segmentation and page coordinates, and caches that TEI
    separately so the original GROBID result is left untouched.
    """
    data = Path(pdf).read_bytes()
    form = {"consolidateHeader": "0", "consolidateCitations": "0"}
    marker = b""
    if coordinates:
        form["segmentSentences"] = "1"
        form["teiCoordinates"] = ["p", "head", "title", "s"]
        marker = b"|teiCoordinates=p,head,title,s|segmentSentences=1|v3"
    digest = hashlib.sha256(data + endpoint.encode() + marker).hexdigest()
    target = Path(cache) / (digest + ".tei.xml")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target.read_text(encoding="utf-8") if coordinates else target.read_text()
    with httpx.Client(timeout=180) as client:
        response = client.post(endpoint.rstrip("/") + "/api/processFulltextDocument",
                               files={"input": (Path(pdf).name, data, "application/pdf")},
                               data=form)
        response.raise_for_status()
    ET.fromstring(response.text)
    target.write_text(response.text, encoding="utf-8")
    return response.text


def element_text(node):
    """Collapse XML text whitespace without duplicating inline elements."""
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip()


def _keep_section(heading, text):
    """Drop short navigation blocks such as a donate button. Keep title and abstract."""
    if heading in {"Title", "Abstract"}:
        return True
    if _CHROME.search(heading or ""):
        return False
    return not (len(text) < 400 and _CHROME.search(text or ""))


def tei_sections(xml):
    """Extract title, abstract, and body blocks in document order, excluding references."""
    root = ET.fromstring(xml)
    sections = []
    title = root.find(".//t:titleStmt/t:title", NS)
    if title is not None and element_text(title):
        sections.append(("Title", element_text(title)))
    abstract = root.find(".//t:abstract", NS)
    if abstract is not None and element_text(abstract):
        sections.append(("Abstract", element_text(abstract)))
    body = root.find(".//t:body", NS)
    if body is None:
        raise ValueError("TEI has no body; inspect GROBID extraction")
    heading = "Body"
    blocks = []
    parents = {child: parent for parent in body.iter() for child in parent}
    for node in body.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "head" and parents.get(node) is not None and parents[node].tag.endswith("div"):
            if blocks:
                text = "\n".join(blocks)
                if _keep_section(heading, text):
                    sections.append((heading, text))
                blocks = []
            heading = element_text(node) or "Body"
        elif tag in {"p", "figure", "table", "formula", "item"}:
            ancestor = parents.get(node)
            nested = False
            while ancestor is not None and ancestor is not body:
                if ancestor.tag.rsplit("}", 1)[-1] in {"p", "figure", "table", "formula", "item"}:
                    nested = True
                    break
                ancestor = parents.get(ancestor)
            if not nested and element_text(node):
                blocks.append(element_text(node))
    if blocks:
        text = "\n".join(blocks)
        if _keep_section(heading, text):
            sections.append((heading, text))
    if not any(h != "Title" and t for h, t in sections):
        raise ValueError("No usable study text extracted")
    return sections


def make_chunks(sections, max_chars=5000, overlap=300):
    """Split sections into overlapping text spans with stable source offsets."""
    if max_chars < 100 or not 0 <= overlap < max_chars // 2:
        raise ValueError("Require max_chars >= 100 and overlap < half the chunk size")
    chunks = []
    for section_index, (heading, text) in enumerate(sections):
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start + max_chars // 2, end)
                if boundary > start:
                    end = boundary
            chunks.append({"id": f"s{section_index:03d}-c{len(chunks):04d}",
                           "section": heading, "section_index": section_index,
                           "start": start, "end": end, "text": text[start:end]})
            if end == len(text):
                break
            start = end - overlap
    return chunks


def _page_numbers(element):
    """Return physical page numbers from a GROBID coords attribute."""
    pages = []
    for item in element.attrib.get("coords", "").split(";"):
        parts = item.split(",")
        if len(parts) != 5:
            continue
        try:
            page = int(float(parts[0]))
        except ValueError:
            continue
        if page not in pages:
            pages.append(page)
    return pages


def tei_pages(xml):
    """Extract text into physical PDF pages using GROBID coordinates."""
    root = ET.fromstring(xml)
    body = root.find(".//t:body", NS)
    if body is None:
        raise ValueError("TEI has no body; inspect GROBID extraction")
    page_text = {}

    def add_element(element):
        """Append coordinate-bearing text to its first source page."""
        text = element_text(element)
        pages = _page_numbers(element)
        if not text or not pages:
            return
        # A span that crosses a page boundary stays on its first physical page.
        page_text.setdefault(pages[0], []).append(text)

    sentences = body.findall(".//t:s", NS)
    if sentences:
        for sentence in sentences:
            add_element(sentence)
    else:
        for paragraph in body.findall(".//t:p", NS):
            add_element(paragraph)
    for heading in body.findall(".//t:head", NS):
        add_element(heading)
    if not page_text:
        raise ValueError("No page coordinates found in TEI body")

    all_pages = []
    for element in root.iter():
        all_pages.extend(_page_numbers(element))
    if not all_pages:
        raise ValueError("No physical page numbers found in TEI")

    pages = [
        (str(number), "\n".join(page_text.get(number, [])))
        for number in range(1, max(all_pages) + 1)
    ]
    prefix = []
    title = root.find(".//t:titleStmt/t:title", NS)
    abstract = root.find(".//t:abstract", NS)
    if title is not None and element_text(title):
        prefix.append("Title: " + element_text(title))
    if abstract is not None and element_text(abstract):
        prefix.append("Abstract: " + element_text(abstract))
    if prefix:
        page_number, first_text = pages[0]
        pages[0] = (page_number, "\n".join(prefix + ([first_text] if first_text else [])))
    return pages


def make_page_chunks(pages, pages_per_chunk=10):
    """Combine consecutive physical pages into fixed-size chunks."""
    if pages_per_chunk < 1:
        raise ValueError("pages_per_chunk must be at least 1")
    chunks = []
    for start in range(0, len(pages), pages_per_chunk):
        group = pages[start:start + pages_per_chunk]
        text = "\n\n".join(page_text for _, page_text in group)
        chunks.append({
            "id": f"p{start + 1:03d}-p{start + len(group):03d}",
            "section": f"Pages {group[0][0]}-{group[-1][0]}",
            "section_index": len(chunks),
            "start": 0,
            "end": len(text),
            "page_start": group[0][0],
            "page_end": group[-1][0],
            "text": text,
        })
    return chunks


def split_chunk(chunk):
    """Split one chunk into two shorter non-overlapping pieces.

    Offsets stay in the parent coordinate system, so evidence from the two
    pieces still lines up. Returns an empty list when the text cannot shrink.
    """
    text = chunk.get("text") or ""
    if len(text) < 200:
        return []
    window_start = max(1, len(text) // 4)
    window_end = min(len(text) - 1, (3 * len(text)) // 4)
    boundary = -1
    for separator in ("\n\n", "\n", " "):
        found = text.rfind(separator, window_start, window_end)
        if found > 0:
            boundary = found
            break
    if boundary < 0:
        boundary = len(text) // 2
    right = boundary
    while right < len(text) and text[right].isspace():
        right += 1
    if boundary < 1 or right >= len(text):
        return []

    def piece(suffix, start, end):
        """Create a child chunk retaining the parent offset coordinate system."""
        piece_text = text[start:end]
        return {**chunk, "id": chunk["id"] + suffix, "text": piece_text,
                "start": chunk["start"] + start, "end": chunk["start"] + end}

    return [piece("a", 0, boundary), piece("b", right, len(text))]
