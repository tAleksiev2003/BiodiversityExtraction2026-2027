#!/bin/bash
# Canonical preprocessing batch for ALL Dev-100 papers. Runs ON a Gruenau server:
#   ssh -l bekerlen gruenau9.informatik.hu-berlin.de 'bash -s' < scripts/preprocess_all_on_gruenau.sh
# Re-runnable: every step skips finished documents, Ctrl+C is safe.
#
# Produces under /vol/tmp/bekerlen/BiodiversityExtraction/out/:
#   minh_json/dev/<id>.json        Minh pipeline F input (Docling JSON, do_ocr=true)
#   our_docling_md/<id>.md         our RAG cache (Docling markdown, do_ocr=false)
#   our_marker_md/<id>.md          our RAG cache (Marker markdown, gaps filled)
#   minh_first_pages/dev/<id>.txt  Minh retrieval input (PyMuPDF, cheap)
# plus conversion logs and a MANIFEST with counts. Pull back with
# scripts/sync_parsed_from_gruenau.sh, then delete the scratch dir.
#
# Needs on the server: inputs/ (deploy_inputs_to_gruenau.sh) and
# pdfs/dev/ (deploy_pdfs_to_gruenau.sh). GPU is used when available,
# otherwise everything falls back to CPU automatically.
set -euo pipefail

WORK="${REMOTE_ROOT:-/vol/tmp/bekerlen/BiodiversityExtraction}"
CODE="$WORK/code"
MINH="$WORK/minh_repro"
VENV="$WORK/venv"
OUT="$WORK/out"
PDFS="$WORK/pdfs/dev"
INPUTS="$WORK/inputs"

echo "=== [0/6] inputs ==="
[ -d "$PDFS" ] || { echo "!! missing $PDFS (run deploy_pdfs_to_gruenau.sh)" >&2; exit 1; }
[ -f "$INPUTS/dev_100_ids.txt" ] || { echo "!! missing $INPUTS (run deploy_inputs_to_gruenau.sh)" >&2; exit 1; }
N_IDS=$(grep -c . "$INPUTS/dev_100_ids.txt")
echo "dev ids: $N_IDS, pdfs: $(ls "$PDFS"/*.pdf 2>/dev/null | wc -l)"

echo "=== [1/6] code ==="
if [ ! -d "$CODE/.git" ]; then
  git clone --branch lennard_testing \
    https://github.com/tAleksiev2003/BiodiversityExtraction2026-2027.git "$CODE"
fi
git -C "$CODE" fetch origin --prune
git -C "$CODE" checkout -q lennard_testing
git -C "$CODE" pull -q --ff-only origin lennard_testing || true
git -C "$CODE" worktree remove -f "$MINH" 2>/dev/null || true
git -C "$CODE" worktree add -f "$MINH" origin/minhanh_testing

echo "=== [2/6] env ==="
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" -q install --upgrade pip
"$VENV/bin/pip" -q install \
  "docling==2.151.1" "marker-pdf==2.2.0" pymupdf pypdfium2 pyyaml numpy scikit-learn

echo "=== [3/6] device ==="
if "$VENV/bin/python" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "GPU available -> GPU run"
else
  echo "no usable GPU -> CPU fallback"
  export CUDA_VISIBLE_DEVICES=""
fi

echo "=== [4/6] Minh Docling JSON (do_ocr=true) ==="
mkdir -p "$MINH/data/raw/pdfs" "$MINH/data/raw/references"
ln -sfn "$PDFS" "$MINH/data/raw/pdfs/dev"
ln -sfn "$INPUTS/train.jsonl" "$MINH/data/raw/references/train.jsonl"
ln -sfn "$INPUTS/dev.jsonl" "$MINH/data/raw/references/dev.jsonl"
cd "$MINH"
"$VENV/bin/python" src/pdf_to_text.py --split dev
mkdir -p "$OUT/minh_json"
cp -r "$MINH/data/processed/docling/dev/." "$OUT/minh_json/" 2>/dev/null || true

echo "=== [5/6] our markdown caches + Minh first_pages ==="
cd "$CODE"
OUT_DIR="$OUT" DEV_PDFS_DIR="$PDFS" "$VENV/bin/python" - <<'EOF'
# Convert all dev PDFs with our chunked converter into OUT dirs (skips done).
import os, sys
sys.path.insert(0, os.getcwd())
from pathlib import Path
from src.data_utils.parse_pdfs_chunked import run_marker, run_docling
pdfs = sorted(Path(os.environ["DEV_PDFS_DIR"]).glob("*.pdf"))
out = Path(os.environ["OUT_DIR"])
run_marker(pdfs, out / "our_marker_md", 15)
run_docling(pdfs, out / "our_docling_md", 15)
EOF
cd "$MINH"
mkdir -p "$MINH/data/processed/first_pages"
"$VENV/bin/python" src/first_pages.py --split dev || echo "!! first_pages failed, see log"
mkdir -p "$OUT/minh_first_pages"
cp -r "$MINH/data/processed/first_pages/dev/." "$OUT/minh_first_pages/" 2>/dev/null || true

echo "=== [6/6] manifest ==="
{
  echo "date_utc=$(date -u +%FT%TZ)"
  echo "host=$(hostname)"
  echo "dev_ids=$N_IDS"
  for d in minh_json our_docling_md our_marker_md minh_first_pages; do
    echo "$d=$(ls "$OUT/$d" 2>/dev/null | wc -l | tr -d ' ')"
  done
  du -sh "$OUT"/*
} | tee "$OUT/MANIFEST.txt"
echo "=== done, pull back with scripts/sync_parsed_from_gruenau.sh ==="
