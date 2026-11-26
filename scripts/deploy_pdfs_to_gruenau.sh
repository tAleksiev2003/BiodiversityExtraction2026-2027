#!/bin/bash
# Copy the 100 Dev PDFs to a Gruenau GPU server for central preprocessing.
#   GRUENAU_HOST env var overrides the target (default gruenau9, the active server).
# Usage: ./scripts/deploy_pdfs_to_gruenau.sh   (from repo root)
#
# Copies exactly the papers listed in data/gold/dev_100_ids.txt from the local
# data/raw_pdfs/dev/ into <REMOTE_ROOT>/pdfs/dev/ via rsync. Re-runnable:
# already transferred PDFs are skipped (--ignore-existing, --partial resumes).
# HU-internal servers only -- course PDFs never leave the university network.
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-bekerlen}"
REMOTE="${REMOTE_USER}@${GRUENAU_HOST:-gruenau9.informatik.hu-berlin.de}"
REMOTE_ROOT="${REMOTE_ROOT:-/vol/tmp/bekerlen/BiodiversityExtraction}"

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
IDS_FILE="$REPO_ROOT/data/gold/dev_100_ids.txt"
SRC_DIR="$REPO_ROOT/data/raw_pdfs/dev"
DST="$REMOTE:$REMOTE_ROOT/pdfs/dev/"

[ -f "$IDS_FILE" ] || { echo "!! IDS file missing: $IDS_FILE" >&2; exit 1; }
[ -d "$SRC_DIR" ] || { echo "!! PDF dir missing: $SRC_DIR" >&2; exit 1; }

N_IDS=$(grep -c . "$IDS_FILE")
echo "=== $N_IDS Dev PDFs -> $DST ==="

# Build the rsync file list (one '<id>.pdf' per line), skipping blanks.
LIST_FILE=$(mktemp)
trap 'rm -f "$LIST_FILE"' EXIT
sed -e 's/[[:space:]]//g' -e '/^$/d' -e 's/$/.pdf/' "$IDS_FILE" > "$LIST_FILE"

# Sanity check: every listed PDF exists locally.
MISSING=0
while IFS= read -r f; do
  [ -f "$SRC_DIR/$f" ] || { echo "!! missing locally: $f"; MISSING=1; }
done < "$LIST_FILE"
[ "$MISSING" -eq 0 ] || { echo "!! abort: PDFs missing locally" >&2; exit 1; }

echo "=== creating remote dir ==="
ssh -o ConnectTimeout=15 "$REMOTE" "mkdir -p '$REMOTE_ROOT/pdfs/dev'"

echo "=== rsync PDFs ==="
rsync -avh --progress --partial --ignore-existing \
  --files-from="$LIST_FILE" "$SRC_DIR/" "$DST"

echo "=== remote verification ==="
ssh -o ConnectTimeout=15 "$REMOTE" \
  "echo -n 'remote count: '; ls '$REMOTE_ROOT/pdfs/dev/'*.pdf 2>/dev/null | wc -l; du -sh '$REMOTE_ROOT/pdfs/dev'"

echo "=== done ($N_IDS expected) ==="
