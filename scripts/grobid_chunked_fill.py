#!/usr/bin/env python3
"""
GROBID gap filler: converts oversized PDFs that the GROBID service rejects as a
whole by splitting them into page ranges, converting each range separately, and
merging the TEI bodies back into one document per paper.

Usage (on the machine that runs GROBID, e.g. lenlenovo):
    .venv/bin/python /tmp/grobid_chunked_fill.py --ids 5SIYLM9W 6G2THNDX [...]
    .venv/bin/python /tmp/grobid_chunked_fill.py --all-missing --chunk-pages 50

Reads PDFs from data/raw_pdfs/dev/, writes data/parsed_text/grobid_dev/<id>.tei.xml
(skips existing valid files unless --force). Pure cache filler: never touches code.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import os as _os

# Repo root: two levels up from scripts/, or explicit override when run from /tmp.
try:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    assert (REPO_ROOT / "data" / "gold").exists()
except (IndexError, AssertionError):
    REPO_ROOT = Path(_os.environ.get("GROBID_REPO_ROOT", ".")).resolve()
NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def fetch_tei(pdf_bytes: bytes, zot_id: str, grobid_url: str, timeout: int) -> str | None:
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"input\"; "
        f'filename="{zot_id}.pdf"\r\nContent-Type: application/pdf\r\n\r\n'.encode()
        + pdf_bytes
        + f"\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        f"{grobid_url}/api/processFulltextDocument",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        return text if "<TEI" in text else None
    except Exception as exc:  # noqa: BLE001 - logged, conversion continues
        print(f"    chunk failed: {exc}", flush=True)
        return None


def merge_tei_bodies(tei_docs: list[str]) -> str | None:
    """Merges TEI docs by appending all body children into the first document."""
    roots = []
    for t in tei_docs:
        try:
            roots.append(ET.fromstring(t))
        except ET.ParseError:
            return None
    if not roots or not roots[0].tag.endswith("}TEI"):
        return None
    base_body = roots[0].find("tei:text/tei:body", NS)
    if base_body is None:
        return None
    for root in roots[1:]:
        body = root.find("tei:text/tei:body", NS)
        if body is not None:
            for child in list(body):
                base_body.append(child)
    ET.register_namespace("", "http://www.tei-c.org/ns/1.0")
    return ET.tostring(roots[0], encoding="unicode")


def convert_paper(pdf_path: Path, out_path: Path, grobid_url: str,
                   chunk_pages: int, timeout: int) -> bool:
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(pdf_path))
    n_pages = len(reader.pages)
    print(f"[{pdf_path.stem}] {n_pages}p, chunks of {chunk_pages}", flush=True)
    tei_parts: list[str] = []
    start = 0
    while start < n_pages:
        end = min(start + chunk_pages, n_pages)
        writer = PdfWriter()
        for p in range(start, end):
            writer.add_page(reader.pages[p])
        buf = io.BytesIO()
        writer.write(buf)
        tei = fetch_tei(buf.getvalue(), f"{pdf_path.stem}-p{start + 1}-{end}",
                        grobid_url, timeout)
        if tei is None and (end - start) > 20:
            # Halve the range once before giving up on it.
            mid = start + (end - start) // 2
            for s, e in ((start, mid), (mid, end)):
                w2 = PdfWriter()
                for p in range(s, e):
                    w2.add_page(reader.pages[p])
                b2 = io.BytesIO()
                w2.write(b2)
                t2 = fetch_tei(b2.getvalue(), f"{pdf_path.stem}-p{s + 1}-{e}",
                               grobid_url, timeout)
                if t2 is None:
                    print(f"  pages {s + 1}-{e}: FAILED, skipped", flush=True)
                else:
                    tei_parts.append(t2)
        elif tei is None:
            print(f"  pages {start + 1}-{end}: FAILED, skipped", flush=True)
        else:
            tei_parts.append(tei)
        start = end
    if not tei_parts:
        print(f"[{pdf_path.stem}] no usable chunks", flush=True)
        return False
    merged = merge_tei_bodies(tei_parts)
    if not merged or len(merged) < 2000:
        print(f"[{pdf_path.stem}] merged TEI too small ({len(merged or '')}), discarded", flush=True)
        return False
    out_path.write_text(merged, encoding="utf-8")
    print(f"[{pdf_path.stem}] OK: {len(tei_parts)} chunks -> {len(merged)} chars", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill GROBID gaps via chunked conversion.")
    parser.add_argument("--ids", nargs="*", default=[],
                        help="zotitem ids to convert")
    parser.add_argument("--all-missing", action="store_true",
                        help="convert all dev_100 ids lacking a valid TEI")
    parser.add_argument("--chunk-pages", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--grobid-url", default="http://127.0.0.1:8070")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    pdf_dir = REPO_ROOT / "data" / "raw_pdfs" / "dev"
    out_dir = REPO_ROOT / "data" / "parsed_text" / "grobid_dev"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all_missing:
        ids = [l.strip() for l in open(REPO_ROOT / "data" / "gold" / "dev_100_ids.txt") if l.strip()]
    else:
        ids = args.ids
    ok, fail = 0, []
    for zot_id in ids:
        out = out_dir / f"{zot_id}.tei.xml"
        if out.exists() and out.stat().st_size > 2000 and not args.force:
            print(f"[{zot_id}] exists, skipped", flush=True)
            ok += 1
            continue
        pdf = pdf_dir / f"{zot_id}.pdf"
        if not pdf.exists():
            print(f"[{zot_id}] no PDF", flush=True)
            fail.append(zot_id)
            continue
        t0 = time.time()
        if convert_paper(pdf, out, args.grobid_url, args.chunk_pages, args.timeout):
            ok += 1
        else:
            fail.append(zot_id)
        print(f"[{zot_id}] done in {time.time() - t0:.0f}s", flush=True)
    print(f"=== {ok} ok, {len(fail)} failed: {fail} ===", flush=True)


if __name__ == "__main__":
    main()
