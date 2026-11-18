"""Convert PDFs to TEI, preserve page boundaries, and create page-based chunks."""
import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

NS = {"t": "http://www.tei-c.org/ns/1.0"}


def pdf_to_tei(pdf, endpoint, cache):
    """Call GROBID once per PDF content and reuse its cached TEI result."""
    data = Path(pdf).read_bytes()
    digest =hashlib.sha256(
    data + endpoint.encode() + b"|teiCoordinates=p,head,title,s|segmentSentences=1|v3"
).hexdigest()
    target = Path(cache) / (digest + ".tei.xml")
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        return target.read_text(encoding="utf-8")

    with httpx.Client(timeout=180) as client:
        response = client.post(
            endpoint.rstrip("/") + "/api/processFulltextDocument",
            files={"input": (Path(pdf).name, data, "application/pdf")},
            data={
    "consolidateHeader": "0",
    "consolidateCitations": "0",
    "segmentSentences": "1",
    "teiCoordinates": ["p", "head", "title", "s"],
}
        )
        response.raise_for_status()

    ET.fromstring(response.text)
    target.write_text(response.text, encoding="utf-8")
    return response.text


def element_text(node):
    """Collapse XML text whitespace."""
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip()



def tei_pages(xml):
    """Extract text into physical PDF pages using GROBID coordinates."""
    root = ET.fromstring(xml)

    body = root.find(".//t:body", NS)
    if body is None:
        raise ValueError("TEI has no body; inspect GROBID extraction")

    page_text = {}

    def get_pages(element):
        """Return all physical page numbers represented by coords."""
        coords = element.attrib.get("coords", "")
        result = []

        for item in coords.split(";"):
            parts = item.split(",")
            if len(parts) == 5:
                try:
                    page = int(float(parts[0]))
                    if page not in result:
                        result.append(page)
                except ValueError:
                    continue

        return result

    def add_element(element):
        text = element_text(element)
        if not text:
            return

        pages = get_pages(element)
        if not pages:
            return

        # Assign the element to its first physical page.
        page_text.setdefault(pages[0], []).append(text)

    # Prefer sentence-level coordinates.
    sentences = body.findall(".//t:s", NS)

    if sentences:
        for sentence in sentences:
            add_element(sentence)
    else:
        # Fallback to paragraphs if sentence segmentation is unavailable.
        for paragraph in body.findall(".//t:p", NS):
            add_element(paragraph)

    # Include headings on their physical pages.
    for heading in body.findall(".//t:head", NS):
        add_element(heading)

    if not page_text:
        raise ValueError("No page coordinates found in TEI body")

    # Determine the physical page range from all coordinate records.
    all_pages = []

    for element in root.iter():
        all_pages.extend(get_pages(element))

    if not all_pages:
        raise ValueError("No physical page numbers found in TEI")

    first_page = 1
    last_page = max(all_pages)

    # Keep missing pages as empty strings so chunk boundaries remain physical.
    pages = [
        (str(number), "\n".join(page_text.get(number, [])))
        for number in range(first_page, last_page + 1)
    ]

    # Put title and abstract on the first physical page.
    title = root.find(".//t:titleStmt/t:title", NS)
    abstract = root.find(".//t:abstract", NS)

    prefix = []

    if title is not None and element_text(title):
        prefix.append("Title: " + element_text(title))

    if abstract is not None and element_text(abstract):
        prefix.append("Abstract: " + element_text(abstract))

    if prefix:
        page_number, first_text = pages[0]
        pages[0] = (
            page_number,
            "\n".join(prefix + ([first_text] if first_text else []))
        )

    return pages



def make_chunks(pages, pages_per_chunk=10):
    """Combine consecutive pages into chunks of a fixed page count."""
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
            "text": text
        })

    return chunks