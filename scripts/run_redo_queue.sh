#!/bin/bash
# Redo queue for lenlenovo: reruns the v4 family and Marker on the complete parser caches.
#
# Why this script exists: llm_rag_sections.py resumes from <run_id>_checkpoint.jsonl.
# On 2026-11-26 every "redo" in run_catchup_queue.sh found 100 finished papers in its
# checkpoint, processed 0 papers and exited after 2-3 s (log line "resumed: 100"), so
# its scores were still the predictions from before the complete caches. This queue
# archives each run directory, deletes the completed checkpoint, and only then starts
# the run, so all 100 papers are requested again. A partial checkpoint (crash during a
# redo) is kept and resumed. Old artifacts stay recoverable in outputs/redo_backup/.
#
# Start (two sessions, at most one HU run at a time):
#   ssh lenlenovo 'tmux new -d -s bio_redo_saia "bash /tmp/run_redo_queue.sh saia"'
#   ssh lenlenovo 'tmux new -d -s bio_redo_hu   "bash /tmp/run_redo_queue.sh hu"'
# Logs: outputs/catchup_queue/queue.log and outputs/catchup_queue/redo_<tag>.log
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
log() { echo "$(date -u +%FT%TZ) [redo] $*" | tee -a "$QLOG"; }

# AutoParser needs GROBID for every paper; the HU side needs the local bridge.
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
  if [ -f "$LOG_DIR/done_redo_${tag}" ]; then
    log "SKIP $tag (done marker present)"
    return 0
  fi
  local d="$PROJ/outputs/architectures/$run_dir"
  local ck n
  ck=$(ls "$d"/*_checkpoint.jsonl 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    n=$(grep -c . "$ck")
    if [ "$n" -ge 100 ]; then
      # Completed pre-cache run: archive it and drop the checkpoint so all papers rerun.
      tar -czf "$BACKUP_DIR/${run_dir}_pre-redo_$(date -u +%Y%m%dT%H%M%SZ).tar.gz" \
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
    >> "$LOG_DIR/redo_${tag}.log" 2>&1
  local rc=$?
  log "<<< END $tag (exit $rc)"
  [ $rc -ne 0 ] && return $rc
  if ! grep -q -E "^\[ *1/100\]" "$LOG_DIR/redo_${tag}.log"; then
    log "!! $tag processed no paper (checkpoint was not cleared?) - not marked done"
    return 1
  fi
  touch "$LOG_DIR/done_redo_${tag}"
  return 0
}

SAIA_MODEL="deepseek-v4.1-flash-1017"
HU_MODEL="hu-reasoning"

case "$SIDE" in
  saia)
    P=saia; M="$SAIA_MODEL"; RPM=20; PRE="llm_rag_sections_saia_deepseek-v4.1-flash-1017_"
    ;;
  hu)
    P=hu; M="$HU_MODEL"; RPM=15; PRE="llm_rag_sections_hu_hu-reasoning_"
    ;;
  *)
    log "!! unknown side '$SIDE' (use saia or hu), aborting"; exit 1
    ;;
esac

redo_one "${P}_ctaxa_v4"           "${PRE}autoparser_ctaxa_v4"           "$P" "$M" "$RPM" --auto-parser --clean-taxa --v4-prompt || true
redo_one "${P}_full_ctaxa_v4"      "${PRE}autoparser_full_ctaxa_v4"      "$P" "$M" "$RPM" --auto-parser --clean-taxa --v4-prompt --full-text || true
redo_one "${P}_ctaxa_v4_biknn"     "${PRE}autoparser_ctaxa_v4_biknn"     "$P" "$M" "$RPM" --auto-parser --clean-taxa --v4-prompt --bilingual-knn || true
redo_one "${P}_full_ctaxa_v4_biknn" "${PRE}autoparser_full_ctaxa_v4_biknn" "$P" "$M" "$RPM" --auto-parser --clean-taxa --v4-prompt --full-text --bilingual-knn || true
redo_one "${P}_marker_ev_ctaxa"    "${PRE}marker_ev_ctaxa"               "$P" "$M" "$RPM" --marker --evidence-gates --clean-taxa || true

log "=== REDO $SIDE DONE ==="
