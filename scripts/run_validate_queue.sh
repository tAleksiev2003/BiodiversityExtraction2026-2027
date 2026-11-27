#!/bin/bash
# Validation queue for lenlenovo: prices --clean-taxa and --evidence-gates on the
# confirmed best recipe (AutoParser + clean-taxa + v4 + bilingual-knn, 16k).
#   A) recipe minus clean-taxa  -> run_id ..._autoparser_v4_biknn
#   B) recipe plus evidence-gates -> run_id ..._autoparser_ev_ctaxa_v4_biknn
# Runs on BOTH providers (OFAT rule: per-provider adoption). Waits for the redo2
# queue of the same side to finish before starting, so only one run per side is
# ever active. No completed checkpoints exist for these run_ids (fresh dirs);
# the checkpoint guard stays in for safety.
#
# Baselines to compare against afterwards:
#   saia recipe (ctaxa_v4_biknn):  0.6102 / 0.4715 / 0.4901  (redo 27 Nov)
#   hu   recipe (ctaxa_v4_biknn):  0.5604 / 0.4166 / 0.4264  (redo 27 Nov)
#
# Start: ssh lenlenovo 'tmux new -d -s bio_validate_saia "bash /tmp/run_validate_queue.sh saia"'
#        ssh lenlenovo 'tmux new -d -s bio_validate_hu   "bash /tmp/run_validate_queue.sh hu"'
# Logs: outputs/catchup_queue/queue.log and outputs/catchup_queue/validate_<tag>.log
set -u
PROJ="${PROJ:-$HOME/Home/BiodiversityExtraction2026-2027}"
cd "$PROJ" || exit 1
[ -f .env ] && set -a && source .env && set +a
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

SIDE="${1:-all}"
LOG_DIR="$PROJ/outputs/catchup_queue"
QLOG="$LOG_DIR/queue.log"
mkdir -p "$LOG_DIR"
log() { echo "$(date -u +%FT%TZ) [validate] $*" | tee -a "$QLOG"; }

if ! curl -s -m 10 -f -o /dev/null http://127.0.0.1:8070/api/isalive; then
  log "!! GROBID is not reachable on 127.0.0.1:8070, aborting"
  exit 1
fi
if [ "$SIDE" = "hu" ] && ! curl -s -m 10 -f -o /dev/null http://127.0.0.1:4000/v1/models; then
  log "!! HU bridge is not reachable on 127.0.0.1:4000, aborting"
  exit 1
fi

# Wait for the redo2 queue of this side (its DONE line) before starting.
while ! grep -q "\[redo2\] === REDO2 $SIDE DONE" "$QLOG" 2>/dev/null; do
  log "waiting for redo2 $SIDE to finish, sleeping 20m"
  sleep 1200
done
log "redo2 $SIDE finished, starting validation runs"

# run_one <tag> <run_dir_name> <provider> <model> <rpm> <flags...>
run_one() {
  local tag="$1" run_dir="$2" provider="$3" model="$4" rpm="$5"; shift 5
  if [ -f "$LOG_DIR/done_validate_${tag}" ]; then
    log "SKIP $tag (done marker present)"
    return 0
  fi
  local d="$PROJ/outputs/architectures/$run_dir"
  local ck n
  ck=$(ls "$d"/*_checkpoint.jsonl 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    n=$(grep -c . "$ck")
    if [ "$n" -ge 100 ]; then
      tar -czf "$PROJ/outputs/redo_backup/${run_dir}_pre-validate_$(date -u +%Y%m%dT%H%M%SZ).tar.gz" \
        -C "$PROJ/outputs/architectures" "$run_dir"
      rm -f "$ck"
      log "archived $run_dir ($n old predictions), checkpoint removed"
    else
      log "$tag: partial checkpoint ($n papers), resuming"
    fi
  fi
  log ">>> START $tag ($provider/$model $*)"
  # shellcheck disable=SC2086
  $PY -u src/models/architectures/llm_rag_sections.py \
    --provider "$provider" --model "$model" --rpm "$rpm" "$@" \
    >> "$LOG_DIR/validate_${tag}.log" 2>&1
  local rc=$?
  log "<<< END $tag (exit $rc)"
  [ $rc -ne 0 ] && return $rc
  if ! grep -q -E "^\[ *1/100\]" "$LOG_DIR/validate_${tag}.log"; then
    log "!! $tag processed no paper - not marked done"
    return 1
  fi
  touch "$LOG_DIR/done_validate_${tag}"
  return 0
}

case "$SIDE" in
  saia) P=saia; M="deepseek-v4.1-flash-1017"; RPM=20; PRE="llm_rag_sections_saia_deepseek-v4.1-flash-1017_" ;;
  hu)   P=hu;   M="hu-reasoning";             RPM=15; PRE="llm_rag_sections_hu_hu-reasoning_" ;;
  *) log "!! unknown side '$SIDE' (use saia or hu), aborting"; exit 1 ;;
esac

# A) recipe minus clean-taxa: prices the taxa-cleanup lever
run_one "${P}_v4_biknn_noctaxa" "${PRE}autoparser_v4_biknn" "$P" "$M" "$RPM" \
  --auto-parser --v4-prompt --bilingual-knn || true
# B) recipe plus evidence-gates: prices the gates as an addition to v4
run_one "${P}_ev_ctaxa_v4_biknn" "${PRE}autoparser_ev_ctaxa_v4_biknn" "$P" "$M" "$RPM" \
  --auto-parser --clean-taxa --v4-prompt --bilingual-knn --evidence-gates || true

log "=== VALIDATE $SIDE DONE ==="
