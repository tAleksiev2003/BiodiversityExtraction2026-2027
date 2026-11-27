"""Marker (CPU) conversion for the train router-verification sample on lenlenovo.

Converts the paper ids listed in scripts/train_router_sample_ids.txt from
data/raw_pdfs/train/ into data/parsed_text/marker_train/<id>.md. Resume-safe:
existing files > 50 bytes are skipped. Includes the marker 2.2.0
SectionHeaderProcessor monkeypatch (pages with children=None crash marker).

Memory: runs on CPU only; the HP OOM happened with marker+docling in one
process — here marker alone with a thread cap stays around 2-3 GB.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HOME"] = os.path.expanduser("~/marker_hf_cache")
os.environ["TORCH_HOME"] = os.path.expanduser("~/marker_hf_cache/torch")

import gc
import time
import traceback
from pathlib import Path

import torch

torch.set_num_threads(2)  # lenlenovo has 4 cores; leave headroom for GROBID batch

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.processors.sectionheader import SectionHeaderProcessor

# marker 2.2.0 bug: pages without detected blocks keep children=None, which
# crashes SectionHeaderProcessor with "'NoneType' object is not iterable".
_orig_sh_call = SectionHeaderProcessor.__call__


def _safe_sh_call(self, document):
    for page in document.pages:
        if page.children is None:
            page.children = []
    return _orig_sh_call(self, document)


SectionHeaderProcessor.__call__ = _safe_sh_call

REPO = Path.home() / "Home/BiodiversityExtraction2026-2027"
IDS_FILE = REPO / "scripts/train_router_sample_ids.txt"
PDF_DIR = REPO / "data/raw_pdfs/train"
OUT_DIR = REPO / "data/parsed_text/marker_train"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ids = [line.strip() for line in IDS_FILE.read_text().splitlines() if line.strip()]
print(f"converting {len(ids)} train sample papers with Marker (CPU, 2 threads)", flush=True)

converter = PdfConverter(artifact_dict=create_model_dict())
done = failed = 0
for i, zid in enumerate(ids, 1):
    target = OUT_DIR / f"{zid}.md"
    if target.exists() and target.stat().st_size > 50:
        done += 1
        continue
    pdf = PDF_DIR / f"{zid}.pdf"
    if not pdf.exists():
        print(f"[Marker {i:3d}/{len(ids)}] {zid} NO PDF", flush=True)
        failed += 1
        continue
    t0 = time.time()
    try:
        rendered = converter(str(pdf))
        text, _, _ = text_from_rendered(rendered)
        target.write_text(text, encoding="utf-8")
        done += 1
        print(f"[Marker {i:3d}/{len(ids)}] {zid} -> {len(text)} chars ({time.time() - t0:.1f}s)", flush=True)
    except Exception:
        failed += 1
        print(f"[Marker {i:3d}/{len(ids)}] {zid} ERROR:\n{traceback.format_exc()}", flush=True)
    gc.collect()

print(f"MARKER TRAIN SAMPLE DONE: ok={done} failed={failed}", flush=True)
