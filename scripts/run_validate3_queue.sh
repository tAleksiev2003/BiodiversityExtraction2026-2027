#!/bin/bash
# Third validation queue for lenlenovo: the neighbour-based service gate from
# Minh's pipeline F, on the confirmed recipe, both providers.
#   E) recipe + --service-gate -> run_id ..._autoparser_ctaxa_v4_biknn_svcg
# Requires the --service-gate flag (commit 2026-11-27). Waits for the
# budget/level-fold queue of the same side (its DONE line) before starting.
#
# Baselines: recipe redos of 27 Nov
#   saia 0.6102 / 0.4715 / 0.4901   hu 0.5604 / 0.4166 / 0.4264
# What to compare: C1+C2b and Micro (the gate targets ecosystem_service
# precision); C1 must not drop.
#
# Start: ssh lenlenovo 'tmux new -d -s bio_v3_saia "bash /tmp/run_validate3_queue.sh saia"'
#        ssh lenlenovo 'tmux new -d -s bio_v3_hu   "bash /tmp/run_validate3_queue.sh hu"'
# Logs: outputs/catchup_queue/queue.log and outputs/catchup_queue/validate3_<tag>.log
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
log() { echo "$(date -u +%FT%TZ) [validate3] $*" | tee -a "$QLOG"; }

if ! curl -s -m 10 -f -o /dev/null http://127.0.0.1:8070/api/isalive; then
  log "!! GROBID is not reachable on 127.0.0.1:8070, aborting"
  exit 1
fi
if [ "$SIDE" = "hu" ] && ! curl -s -m 10 -f -o /dev/null http://127.0.0.1:4000/v1/models; then
  log "!! HU bridge is not reachable on 127.0.0.1:4000, aborting"
  exit 1
fi

# Wait for the budget/level-fold queue of this side before starting.
while ! grep -q "\[validate2\] === VALIDATE2 $SIDE DONE" "$QLOG" 2>/dev/null; do
  log "waiting for validate2 $SIDE to finish, sleeping 20m"
  sleep 1200
done
log "validate2 $SIDE finished, starting service-gate runs"

# run_one <tag> <run_dir_name> <provider> <model> <rpm> <flags...>
run_one() {
  local tag="$1" run_dir="$2" provider="$3" model="$4" rpm="$5"; shift 5
  if [ -f "$LOG_DIR/done_validate3_${tag}" ]; then
    log "SKIP $tag (done marker present)"
    return 0
  fi
  local d="$PROJ/outputs/architectures/$run_dir"
  local ck n
  ck=$(ls "$d"/*_checkpoint.jsonl 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    n=$(grep -c . "$ck")
    if [ "$n" -ge 100 ]; then
      tar -czf "$PROJ/outputs/redo_backup/${run_dir}_pre-validate3_$(date -u +%Y%m%dT%H%M%SZ).tar.gz" \
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
    >> "$LOG_DIR/validate3_${tag}.log" 2>&1
  local rc=$?
  log "<<< END $tag (exit $rc)"
  [ $rc -ne 0 ] && return $rc
  if ! grep -q -E "^\[ *1/100\]" "$LOG_DIR/validate3_${tag}.log"; then
    log "!! $tag processed no paper - not marked done"
    return 1
  fi
  touch "$LOG_DIR/done_validate3_${tag}"
  return 0
}

case "$SIDE" in
  saia) P=saia; M="deepseek-v4.1-flash-1017"; RPM=20; PRE="llm_rag_sections_saia_deepseek-v4.1-flash-1017_" ;;
  hu)   P=hu;   M="hu-reasoning";             RPM=15; PRE="llm_rag_sections_hu_hu-reasoning_" ;;
  *) log "!! unknown side '$SIDE' (use saia or hu), aborting"; exit 1 ;;
esac

run_one "${P}_ctaxa_v4_biknn_svcg" "${PRE}autoparser_ctaxa_v4_biknn_svcg" "$P" "$M" "$RPM" \
  --auto-parser --clean-taxa --v4-prompt --bilingual-knn --service-gate || true

log "=== VALIDATE3 $SIDE DONE ==="
