#!/bin/bash
# HU-only parallel batch runner: runs ALL remaining HU validation stages with up
# to 3 concurrent runs (bridge raised to max_parallel_requests=3, 30 RPM by user
# decision 2026-11-28). Self-regulating: before starting a job it counts the
# running HU pipeline processes (including stages started elsewhere, e.g. the
# level-fold run in the validate2 session), so the total never exceeds 3.
#
# Stages (tag|run_id|flags|done marker):
#   svcg  service-gate          (validate3)
#   pgn   german pairing rule   (validate4)
#   ectl  eco-type land rule    (validate5)
#   svs   service-sharpen       (validate6)
#   ggl   german glossary       (validate7, new)
#   fldr  field retrieval       (validate7, new)
#   dtsc  doc-type scaling      (validate7, new)
#
# Replaces the sequential HU waiters (bio_v3_hu..bio_v6_hu, kill those first).
# Start: ssh lenlenovo 'tmux new -d -s bio_hupar "bash /tmp/run_hu_parallel_batch.sh"'
# Logs: outputs/catchup_queue/queue.log and outputs/catchup_queue/hupar_<tag>.log
set -u
PROJ="${PROJ:-$HOME/Home/BiodiversityExtraction2026-2027}"
cd "$PROJ" || exit 1
[ -f .env ] && set -a && source .env && set +a
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

LOG_DIR="$PROJ/outputs/catchup_queue"
QLOG="$LOG_DIR/queue.log"
mkdir -p "$LOG_DIR"
MAX_TOTAL=3
MODEL="hu-reasoning"
RPM=15
PRE="llm_rag_sections_hu_hu-reasoning_"
log() { echo "$(date -u +%FT%TZ) [hupar] $*" | tee -a "$QLOG"; }

if ! curl -s -m 10 -f -o /dev/null http://127.0.0.1:8070/api/isalive; then
  log "!! GROBID is not reachable on 127.0.0.1:8070, aborting"
  exit 1
fi
if ! curl -s -m 10 -f -o /dev/null http://127.0.0.1:4000/v1/models; then
  log "!! HU bridge is not reachable on 127.0.0.1:4000, aborting"
  exit 1
fi

JOBS=(
  "svcg|${PRE}autoparser_ctaxa_v4_biknn_svcg|--auto-parser --clean-taxa --v4-prompt --bilingual-knn --service-gate|done_validate3_hu_ctaxa_v4_biknn_svcg"
  "pgn|${PRE}autoparser_ctaxa_v4_biknn_pgn|--auto-parser --clean-taxa --v4-prompt --bilingual-knn --german-pairing-names|done_validate4_hu_ctaxa_v4_biknn_pgn"
  "ectl|${PRE}autoparser_ctaxa_v4_biknn_ectl|--auto-parser --clean-taxa --v4-prompt --bilingual-knn --eco-type-land-rule|done_validate5_hu_ctaxa_v4_biknn_ectl"
  "svs|${PRE}autoparser_ctaxa_v4_biknn_svs|--auto-parser --clean-taxa --v4-prompt --bilingual-knn --service-sharpen|done_validate6_hu_ctaxa_v4_biknn_svs"
  "ggl|${PRE}autoparser_ctaxa_v4_biknn_ggl|--auto-parser --clean-taxa --v4-prompt --bilingual-knn --german-glossary|done_validate7_hu_ctaxa_v4_biknn_ggl"
  "fldr|${PRE}autoparser_ctaxa_v4_biknn_fldr|--auto-parser --clean-taxa --v4-prompt --bilingual-knn --field-retrieval|done_validate7_hu_ctaxa_v4_biknn_fldr"
  "dtsc|${PRE}autoparser_ctaxa_v4_biknn_dtsc|--auto-parser --clean-taxa --v4-prompt --bilingual-knn --doc-type-scale|done_validate7_hu_ctaxa_v4_biknn_dtsc"
)

hu_running() { pgrep -fc "provider hu --model" 2>/dev/null || echo 0; }

start_job() {
  local spec="$1" tag run_id flags
  IFS="|" read -r tag run_id flags _ <<< "$spec"
  local d="$PROJ/outputs/architectures/$run_id"
  local ck
  ck=$(ls "$d"/*_checkpoint.jsonl 2>/dev/null | head -1)
  if [ -n "$ck" ] && [ "$(grep -c . "$ck")" -ge 100 ]; then
    tar -czf "$PROJ/outputs/redo_backup/${run_id}_pre-hupar_$(date -u +%Y%m%dT%H%M%SZ).tar.gz" \
      -C "$PROJ/outputs/architectures" "$run_id"
    rm -f "$ck"
    log "archived $run_id (old predictions), checkpoint removed"
  fi
  log ">>> START $tag ($flags)"
  # shellcheck disable=SC2086
  $PY -u src/models/architectures/llm_rag_sections.py \
    --provider hu --model "$MODEL" --rpm "$RPM" $flags \
    >> "$LOG_DIR/hupar_${tag}.log" 2>&1 &
  echo $!
}

declare -A PIDS
all_done() {
  local e j marker
  for e in "${JOBS[@]}"; do
    IFS="|" read -r j _ _ marker <<< "$e"
    [ -f "$LOG_DIR/$marker" ] || return 1
  done
  return 0
}

log "HU parallel batch starting (max $MAX_TOTAL concurrent incl. external HU runs)"

while ! all_done; do
  # reconcile finished background jobs -> done markers
  for e in "${JOBS[@]}"; do
    IFS="|" read -r j _ _ marker <<< "$e"
    [ -f "$LOG_DIR/$marker" ] && continue
    pid="${PIDS[$j]:-}"
    [ -n "$pid" ] || continue
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null
      rc=$?
      if grep -q -E "^\[ *1/100\]" "$LOG_DIR/hupar_${j}.log" 2>/dev/null; then
        touch "$LOG_DIR/$marker"
        log "<<< END $j (exit $rc) -> done marker set"
      else
        log "!! $j finished but processed no paper (exit $rc) - NOT marked done, will retry"
      fi
      unset "PIDS[$j]"
    fi
  done

  # start pending jobs while capacity allows
  for e in "${JOBS[@]}"; do
    IFS="|" read -r j run_id flags marker <<< "$e"
    [ -f "$LOG_DIR/$marker" ] && continue
    [ -n "${PIDS[$j]:-}" ] && continue
    [ "$(hu_running)" -ge "$MAX_TOTAL" ] && break
    PIDS[$j]=$(start_job "$e")
    sleep 5
  done

  sleep 60
done
log "=== HU PARALLEL BATCH ALL DONE ==="
