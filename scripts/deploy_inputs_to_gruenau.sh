#!/bin/bash
# Push small preprocessing inputs (gold references, id list) to Gruenau.
#   GRUENAU_HOST env var overrides the target (default gruenau9).
# Usage: ./scripts/deploy_inputs_to_gruenau.sh   (from repo root)
#
# The 100 Dev PDFs travel separately via deploy_pdfs_to_gruenau.sh.
# This script only copies kilobytes of JSONL so Minh's pipeline finds its
# references and our scripts find the id list. HU-internal servers only.
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-bekerlen}"
REMOTE="${REMOTE_USER}@${GRUENAU_HOST:-gruenau9.informatik.hu-berlin.de}"
REMOTE_ROOT="${REMOTE_ROOT:-/vol/tmp/bekerlen/BiodiversityExtraction}"

cd "$(dirname "$0")/.."

echo "=== creating remote input dir ==="
ssh -o ConnectTimeout=15 "$REMOTE" "mkdir -p '$REMOTE_ROOT/inputs'"

echo "=== rsync references + id list ==="
rsync -avh --progress \
  data/gold/train.jsonl \
  data/gold/dev.jsonl \
  data/gold/dev_100_ids.txt \
  "$REMOTE:$REMOTE_ROOT/inputs/"

echo "=== remote verification ==="
ssh -o ConnectTimeout=15 "$REMOTE" "ls -lh '$REMOTE_ROOT/inputs/'"
echo "=== done ==="
