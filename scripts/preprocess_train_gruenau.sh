#!/bin/bash
# Full preprocessing batch for the 908 TRAIN PDFs on a Gruenau GPU server.
# Twin of scripts/preprocess_all_on_gruenau.sh (Dev-100), parameterized for train.
#   ssh -l bekerlen gruenau9.informatik.hu-berlin.de 'bash -s' < scripts/preprocess_train_gruenau.sh
#
# Needs on the server FIRST (NOT included here):
#   1. Train PDFs (~2.6 GB) under $WORK/pdfs/train/  (separate authorised upload)
#   2. inputs/train.jsonl (+ dev.jsonl) under $WORK/inputs/ (deploy_inputs_to_gruenau.sh)
# Produces under /vol/tmp/bekerlen/BiodiversityExtraction/out_train/:
#   minh_json/<id>.json, our_docling_md/<id>.md, our_marker_md/<id>.md,
#   minh_first_pages/{train,dev}/<id>.txt
# Pull back with rsync (see bottom), verify counts, then delete server scratch.
# RUNTIME WARNING: ~908 docs x Docling-OCR = several days even on A100.
# Run ONLY after explicit go; Dev-100 batch first.
set -u

WORK="${REMOTE_ROOT:-/vol/tmp/bekerlen/BiodiversityExtraction}"
CODE="$WORK/code"
MINH="$WORK/minh_repro"
VENV="$WORK/venv"
OUT="$WORK/out_train"
PDFS="$WORK/pdfs/train"
INPUTS="$WORK/inputs"

echo "=== [0/6] inputs (train) ==="
[ -d "$PDFS" ] || { echo "!! missing $PDFS (upload train PDFs first)" >&2; exit 1; }
[ -f "$INPUTS/train.jsonl" ] || { echo "!! missing $INPUTS/train.jsonl" >&2; exit 1; }
N_PDFS=$(ls "$PDFS"/*.pdf 2>/dev/null | wc -l)
echo "train pdfs: $N_PDFS (expect 908)"

echo "=== [1/6] code (reuse dev checkout) ==="
[ -d "$CODE/.git" ] || { echo "!! missing $CODE (run the dev batch setup first)" >&2; exit 1; }
git -C "$CODE" fetch origin --prune 2>/dev/null || true
if [ ! -d "$MINH/.git" ]; then
  git -C "$CODE" worktree add -f "$MINH" origin/minhanh_testing 2>/dev/null || true
fi

echo "=== [2/6] env (reuse dev venv) ==="
[ -x "$VENV/bin/python" ] || { echo "!! missing $VENV (run the dev batch first)" >&2; exit 1; }

echo "=== [3/6] device ==="
if "$VENV/bin/python" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "GPU available -> GPU run"
else
  echo "no usable GPU -> CPU fallback"
  export CUDA_VISIBLE_DEVICES=""
fi

echo "=== [4/6] Minh Docling JSON (do_ocr=true, train) ==="
mkdir -p "$MINH/data/raw/pdfs" "$MINH/data/raw/references"
ln -sfn "$PDFS" "$MINH/data/raw/pdfs/train"
ln -sfn "$INPUTS/train.jsonl" "$MINH/data/raw/references/train.jsonl"
ln -sfn "$INPUTS/dev.jsonl" "$MINH/data/raw/references/dev.jsonl"
cd "$MINH"
"$VENV/bin/python" src/pdf_to_text.py --split train
mkdir -p "$OUT/minh_json"
cp -r "$MINH/data/processed/docling/train/." "$OUT/minh_json/" 2>/dev/null || true

echo "=== [5/6] our markdown caches (train) ==="
cd "$CODE"
OUT_DIR="$OUT" DEV_PDFS_DIR="$PDFS" "$VENV/bin/python" - <<'EOF'
import os, sys
sys.path.insert(0, os.getcwd())
from pathlib import Path
from src.data_utils.parse_pdfs_chunked import run_marker, run_docling
pdfs = sorted(Path(os.environ["DEV_PDFS_DIR"]).glob("*.pdf"))
out = Path(os.environ["OUT_DIR"])
run_marker(pdfs, out / "our_marker_md", 15)
run_docling(pdfs, out / "our_docling_md", 15)
EOF

echo "=== [6/6] Minh first_pages (train) ==="
cd "$MINH"
"$VENV/bin/python" src/first_pages.py --split train || echo "!! first_pages failed, see log"
mkdir -p "$OUT/minh_first_pages"
cp -r "$MINH/data/processed/first_pages/train/." "$OUT/minh_first_pages/" 2>/dev/null || true

{
  echo "date_utc=$(date -u +%FT%TZ)"
  echo "host=$(hostname)"
  echo "train_pdfs=$N_PDFS"
  for d in minh_json our_docling_md our_marker_md minh_first_pages; do
    echo "$d=$(ls "$OUT/$d" 2>/dev/null | wc -l | tr -d ' ')"
  done
  du -sh "$OUT"/*
} | tee "$OUT/MANIFEST.txt"
echo "=== done: pull back with rsync $OUT/ then verify, then cleanup ==="
