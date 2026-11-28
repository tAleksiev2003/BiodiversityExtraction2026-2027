#!/bin/bash
# Sixth validation queue: OUR OWN service-precision prompt rule
# (--service-sharpen, run_id suffix _svs), on the confirmed recipe, both providers.
# From our P/R analysis: we emit services on 29/100 papers where gold has 8
# (precision 0.20 category / 0.11 term); the R9 prompt rule restricts services
# to studies that examine/measure them themselves. Our own wording — Minh fixed
# the same problem with a post-rule (service-gate, validate3) instead.
# Waits for validate5 of the same side (its DONE line) before starting.
#
# Baselines: recipe 27/28 Nov (saia 0.6102/0.4715/0.4901, hu 0.5604/0.4166/0.4264).
# Compare: ecosystem_service precision must rise; recall and C1/C2b must not drop.
#
# Start: ssh lenlenovo 'tmux new -d -s bio_v6_saia "bash /tmp/run_validate6_queue.sh saia"'
#        ssh lenlenovo 'tmux new -d -s bio_v6_hu   "bash /tmp/run_validate6_queue.sh hu"'
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
log() { echo "$(date -u +%FT%TZ) [validate6] $*" | tee -a "$QLOG"; }

if ! curl -s -m 10 -f -o /dev/null http://127.0.0.1:8070/api/isalive; then
  log "!! GROBID is not reachable on 127.0.0.1:8070, aborting"
  exit 1
fi
if [ "$SIDE" = "hu" ] && ! curl -s -m 10 -f -o /dev/null http://127.0.0.1:4000/v1/models; then
  log "!! HU bridge is not reachable on 127.0.0.1:4000, aborting"
  exit 1
fi

while ! grep -q "\[validate5\] === VALIDATE5 $SIDE DONE" "$QLOG" 2>/dev/null; do
  log "waiting for validate5 $SIDE to finish, sleeping 20m"
  sleep 1200
done
log "validate5 $SIDE finished, starting service-sharpen runs"

run_one() {
  local tag="$1" run_dir="$2" provider="$3" model="$4" rpm="$5"; shift 5
  if [ -f "$LOG_DIR/done_validate6_${tag}" ]; then
    log "SKIP $tag (done marker present)"
    return 0
  fi
  local d="$PROJ/outputs/architectures/$run_dir"
  local ck n
  ck=$(ls "$d"/*_checkpoint.jsonl 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    n=$(grep -c . "$ck")
    if [ "$n" -ge 100 ]; then
      tar -czf "$PROJ/outputs/redo_backup/${run_dir}_pre-validate6_$(date -u +%Y%m%dT%H%M%SZ).tar.gz" \
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
    >> "$LOG_DIR/validate6_${tag}.log" 2>&1
  local rc=$?
  log "<<< END $tag (exit $rc)"
  [ $rc -ne 0 ] && return $rc
  if ! grep -q -E "^\[ *1/100\]" "$LOG_DIR/validate6_${tag}.log"; then
    log "!! $tag processed no paper - not marked done"
    return 1
  fi
  touch "$LOG_DIR/done_validate6_${tag}"
  return 0
}

case "$SIDE" in
  saia) P=saia; M="deepseek-v4.1-flash-1017"; RPM=20; PRE="llm_rag_sections_saia_deepseek-v4.1-flash-1017_" ;;
  hu)   P=hu;   M="hu-reasoning";             RPM=15; PRE="llm_rag_sections_hu_hu-reasoning_" ;;
  *) log "!! unknown side '$SIDE' (use saia or hu), aborting"; exit 1 ;;
esac

run_one "${P}_ctaxa_v4_biknn_svs" "${PRE}autoparser_ctaxa_v4_biknn_svs" "$P" "$M" "$RPM" \
  --auto-parser --clean-taxa --v4-prompt --bilingual-knn --service-sharpen || true

log "=== VALIDATE6 $SIDE DONE ==="
