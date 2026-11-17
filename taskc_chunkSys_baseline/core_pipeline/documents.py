"""Convert PDFs to TEI, preserve section order, and create bounded chunks."""
import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import httpx

NS = {"t": "http://www.tei-c.org/ns/1.0"}


def pdf_to_tei(pdf, endpoint, cache):
    """Call GROBID once per PDF content and reuse its cached TEI result."""
    data = Path(pdf).read_bytes()
    digest = hashlib.sha256(data + endpoint.encode()).hexdigest()
    target = Path(cache) / (digest + ".tei.xml")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target.read_text()
    with httpx.Client(timeout=180) as client:
        response = client.post(endpoint.rstrip("/") + "/api/processFulltextDocument",
                               files={"input": (Path(pdf).name, data, "application/pdf")},
                               data={"consolidateHeader": "0", "consolidateCitations": "0"})
        response.raise_for_status()
    ET.fromstring(response.text)
    target.write_text(response.text, encoding="utf-8")
    return response.text


def element_text(node):
    """Collapse XML text whitespace without duplicating inline elements."""
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip()


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
                sections.append((heading, "\n".join(blocks)))
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
        sections.append((heading, "\n".join(blocks)))
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
