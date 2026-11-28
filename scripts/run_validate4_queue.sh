#!/bin/bash
# Fourth validation queue: the DB pairing convention as a prompt rule
# (--german-pairing-names, run_id suffix _pgn), on the confirmed recipe, both
# providers. Measures the german-name trade-off from the mechanism analysis:
# fixes ~19 missed pairings vs. possibly amplifying the precision leak on the
# 37 papers without gold taxa (24-32 leaked german labels, +2.1-2.3 ger_name F1
# if emptied). Requires commit >= b493cf9+.
#
# Per-side wait conditions (keeps one run per side at all times):
#   saia: waits for the train router verification (TRAINVERIF DONE)
#   hu:   waits for the service-gate queue (VALIDATE3 hu DONE)
#
# Baselines: recipe (27 Nov redos) saia 0.6102/0.4715/0.4901, hu (minhjson) 0.5894.
# Compare: ger_name precision/recall AND corpus C1/C2b (must not drop).
#
# Start: ssh lenlenovo 'tmux new -d -s bio_v4p_saia "bash /tmp/run_validate4_queue.sh saia"'
#        ssh lenlenovo 'tmux new -d -s bio_v4p_hu   "bash /tmp/run_validate4_queue.sh hu"'
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
log() { echo "$(date -u +%FT%TZ) [validate4] $*" | tee -a "$QLOG"; }

if ! curl -s -m 10 -f -o /dev/null http://127.0.0.1:8070/api/isalive; then
  log "!! GROBID is not reachable on 127.0.0.1:8070, aborting"
  exit 1
fi
if [ "$SIDE" = "hu" ] && ! curl -s -m 10 -f -o /dev/null http://127.0.0.1:4000/v1/models; then
  log "!! HU bridge is not reachable on 127.0.0.1:4000, aborting"
  exit 1
fi

if [ "$SIDE" = "saia" ]; then
  while ! grep -q "\[trainverif\] === TRAINVERIF DONE" "$QLOG" 2>/dev/null; do
    log "waiting for trainverif saia to finish, sleeping 20m"
    sleep 1200
  done
else
  while ! grep -q "\[validate3\] === VALIDATE3 $SIDE DONE" "$QLOG" 2>/dev/null; do
    log "waiting for validate3 $SIDE to finish, sleeping 20m"
    sleep 1200
  done
fi
log "prerequisites met, starting german-pairing runs"

run_one() {
  local tag="$1" run_dir="$2" provider="$3" model="$4" rpm="$5"; shift 5
  if [ -f "$LOG_DIR/done_validate4_${tag}" ]; then
    log "SKIP $tag (done marker present)"
    return 0
  fi
  local d="$PROJ/outputs/architectures/$run_dir"
  local ck n
  ck=$(ls "$d"/*_checkpoint.jsonl 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    n=$(grep -c . "$ck")
    if [ "$n" -ge 100 ]; then
      tar -czf "$PROJ/outputs/redo_backup/${run_dir}_pre-validate4_$(date -u +%Y%m%dT%H%M%SZ).tar.gz" \
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
    >> "$LOG_DIR/validate4_${tag}.log" 2>&1
  local rc=$?
  log "<<< END $tag (exit $rc)"
  [ $rc -ne 0 ] && return $rc
  if ! grep -q -E "^\[ *1/100\]" "$LOG_DIR/validate4_${tag}.log"; then
    log "!! $tag processed no paper - not marked done"
    return 1
  fi
  touch "$LOG_DIR/done_validate4_${tag}"
  return 0
}

case "$SIDE" in
  saia) P=saia; M="deepseek-v4.1-flash-1017"; RPM=20; PRE="llm_rag_sections_saia_deepseek-v4.1-flash-1017_" ;;
  hu)   P=hu;   M="hu-reasoning";             RPM=15; PRE="llm_rag_sections_hu_hu-reasoning_" ;;
  *) log "!! unknown side '$SIDE' (use saia or hu), aborting"; exit 1 ;;
esac

run_one "${P}_ctaxa_v4_biknn_pgn" "${PRE}autoparser_ctaxa_v4_biknn_pgn" "$P" "$M" "$RPM" \
  --auto-parser --clean-taxa --v4-prompt --bilingual-knn --german-pairing-names || true

log "=== VALIDATE4 $SIDE DONE ==="
