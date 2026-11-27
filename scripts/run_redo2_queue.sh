#!/bin/bash
# Second redo queue for lenlenovo: the Docling Light v4 pair and the rawft pair,
# both still on pre-cache checkpoints after the 27 Nov v4-family redos.
# Same safety logic as run_redo_queue.sh: archive a completed checkpoint to
# outputs/redo_backup/, delete it so all 100 papers rerun, then run with the
# exact flags that reproduce the original run_id. rawft keeps its known
# token-limit handicap on purpose (1 empty paper on SAIA, 3 on HU) - this redo
# is a same-config variance check, the split-fallback stays a code task.
#
# Start: ssh lenlenovo 'tmux new -d -s bio_redo2_saia "bash /tmp/run_redo2_queue.sh saia"'
#        ssh lenlenovo 'tmux new -d -s bio_redo2_hu   "bash /tmp/run_redo2_queue.sh hu"'
# Logs: outputs/catchup_queue/queue.log and outputs/catchup_queue/redo2_<tag>.log
set -u
PROJ="${PROJ:-$HOME/Home/BiodiversityExtraction2026-2027}"
cd "$PROJ" || exit 1
[ -f .env ] && set -a && source .env && set +a
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

SIDE="${1:-all}"
LOG_DIR="$PROJ/outputs/catchup_queue"
BACKUP_DIR="$PROJ/outputs/redo_backup"
mkdir -p "$LOG_DIR" "$BACKUP_DIR"
QLOG="$LOG_DIR/queue.log"
log() { echo "$(date -u +%FT%TZ) [redo2] $*" | tee -a "$QLOG"; }

if ! curl -s -m 10 -f -o /dev/null http://127.0.0.1:8070/api/isalive; then
  log "!! GROBID is not reachable on 127.0.0.1:8070, aborting"
  exit 1
fi
if [ "$SIDE" = "hu" ] && ! curl -s -m 10 -f -o /dev/null http://127.0.0.1:4000/v1/models; then
  log "!! HU bridge is not reachable on 127.0.0.1:4000, aborting"
  exit 1
fi

# redo_one <tag> <run_dir_name> <provider> <model> <rpm> <flags...>
redo_one() {
  local tag="$1" run_dir="$2" provider="$3" model="$4" rpm="$5"; shift 5
  if [ -f "$LOG_DIR/done_redo2_${tag}" ]; then
    log "SKIP $tag (done marker present)"
    return 0
  fi
  local d="$PROJ/outputs/architectures/$run_dir"
  local ck n
  ck=$(ls "$d"/*_checkpoint.jsonl 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    n=$(grep -c . "$ck")
    if [ "$n" -ge 100 ]; then
      tar -czf "$BACKUP_DIR/${run_dir}_pre-redo2_$(date -u +%Y%m%dT%H%M%SZ).tar.gz" \
        -C "$PROJ/outputs/architectures" "$run_dir"
      rm -f "$ck"
      log "archived $run_dir ($n old predictions), checkpoint removed"
    else
      log "$tag: partial redo checkpoint ($n papers), resuming"
    fi
  fi
  log ">>> START $tag ($provider/$model $*)"
  # shellcheck disable=SC2086
  $PY -u src/models/architectures/llm_rag_sections.py \
    --provider "$provider" --model "$model" --rpm "$rpm" "$@" \
    >> "$LOG_DIR/redo2_${tag}.log" 2>&1
  local rc=$?
  log "<<< END $tag (exit $rc)"
  [ $rc -ne 0 ] && return $rc
  if ! grep -q -E "^\[ *1/100\]" "$LOG_DIR/redo2_${tag}.log"; then
    log "!! $tag processed no paper (checkpoint was not cleared?) - not marked done"
    return 1
  fi
  touch "$LOG_DIR/done_redo2_${tag}"
  return 0
}

case "$SIDE" in
  saia) P=saia; M="deepseek-v4.1-flash-1017"; RPM=20; PRE="llm_rag_sections_saia_deepseek-v4.1-flash-1017_" ;;
  hu)   P=hu;   M="hu-reasoning";             RPM=15; PRE="llm_rag_sections_hu_hu-reasoning_" ;;
  *) log "!! unknown side '$SIDE' (use saia or hu), aborting"; exit 1 ;;
esac

# 1) Docling Light v4 pair (anchors the AutoParser-vs-Docling comparison)
redo_one "${P}_docling_full_ctaxa_v4" "${PRE}docling_full_ctaxa_v4" "$P" "$M" "$RPM" \
  --docling --full-text --clean-taxa --v4-prompt || true
# 2) rawft pair (same-config variance check; token-limit handicap expected)
redo_one "${P}_rawft" "${PRE}autoparser_ctaxa_v4_rawft" "$P" "$M" "$RPM" \
  --auto-parser --clean-taxa --v4-prompt --raw-fulltext || true

log "=== REDO2 $SIDE DONE ==="
