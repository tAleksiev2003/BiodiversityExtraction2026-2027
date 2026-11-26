#!/bin/bash
# Pull canonical preprocessing outputs back from Gruenau into the local tree.
# Usage:
#   ./scripts/sync_parsed_from_gruenau.sh              # pull + verify only
#   ./scripts/sync_parsed_from_gruenau.sh --cleanup    # + delete server scratch
#
# Mapping (server out/ -> local data/):
#   minh_json/*.json          -> data/processed/docling/dev/
#   minh_first_pages/*.txt    -> data/processed/first_pages/dev/
#   our_docling_md/*.md       -> data/parsed_text/docling_dev/
#   our_marker_md/*.md        -> data/parsed_text/marker_dev/
# Verifies 100/100 files per set against data/gold/dev_100_ids.txt.
# For durable storage in the data repo, mirror data/processed + new caches
# with: cp -r data/processed ../BiodiversityExtraction2026-2027-data/data/
# and commit there (Gitea only, never GitHub).
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-bekerlen}"
REMOTE="${REMOTE_USER}@${GRUENAU_HOST:-gruenau9.informatik.hu-berlin.de}"
REMOTE_ROOT="${REMOTE_ROOT:-/vol/tmp/bekerlen/BiodiversityExtraction}"
CLEANUP=0
[ "${1:-}" = "--cleanup" ] && CLEANUP=1

cd "$(dirname "$0")/.."
IDS_FILE="data/gold/dev_100_ids.txt"
N_IDS=$(grep -c . "$IDS_FILE")

pull() { # $1 = remote subdir, $2 = local dir, $3 = glob
  echo "=== pulling $1 -> $2 ==="
  mkdir -p "$2"
  rsync -avh --progress --partial \
    --exclude='*.full_ok' "$REMOTE:$REMOTE_ROOT/out/$1/" "$2/"
}

pull minh_json          data/processed/docling/dev      '*.json'
pull minh_first_pages   data/processed/first_pages/dev  '*.txt'
pull our_docling_md     data/parsed_text/docling_dev    '*.md'
pull our_marker_md      data/parsed_text/marker_dev     '*.md'
rsync -avh "$REMOTE:$REMOTE_ROOT/out/MANIFEST.txt" data/processed/gruenau_manifest.txt 2>/dev/null || true

echo "=== verification (expect $N_IDS each) ==="
FAIL=0
check() { # $1 = dir, $2 = ext
  n=$(ls "$1"/*."$2" 2>/dev/null | wc -l | tr -d ' ')
  echo "$1: $n/$N_IDS .$2 files"
  [ "$n" -eq "$N_IDS" ] || FAIL=1
}
check data/processed/docling/dev json
check data/parsed_text/docling_dev md
check data/parsed_text/marker_dev md
missing_fp=$(comm -23 <(sort data/gold/dev_100_ids.txt) <(ls data/processed/first_pages/dev/*.txt 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/\.txt$//' | sort) | tr '\n' ' ')
echo "data/processed/first_pages/dev: missing=[${missing_fp:-none}] (V2BVMBPF has no text layer by design)"
case "$missing_fp" in ""|"V2BVMBPF ") ;; *) FAIL=1 ;; esac
echo "--- sizes ---"
du -sh data/processed/docling data/processed/first_pages \
  data/parsed_text/docling_dev data/parsed_text/marker_dev 2>/dev/null || true

if [ "$FAIL" -ne 0 ]; then
  echo "!! INCOMPLETE: re-run preprocessing or sync again, skip --cleanup" >&2
  exit 1
fi
echo "=== all 4 sets complete ==="

if [ "$CLEANUP" -eq 1 ]; then
  echo "=== deleting server scratch $REMOTE_ROOT ==="
  ssh -o ConnectTimeout=15 "$REMOTE" "rm -rf '$REMOTE_ROOT'"
  echo "=== server scratch deleted ==="
else
  echo "=== server scratch kept (re-run with --cleanup to delete) ==="
fi
