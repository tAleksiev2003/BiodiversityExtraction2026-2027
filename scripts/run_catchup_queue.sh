#!/bin/bash
# Catch-up + OFAT queue for lenlenovo (runs inside tmux bio_catchup).
# Runs the queued Dev-100 configs sequentially on SAIA and HU, checkpoint-resumable.
# The LLM client waits out 429 quota resets itself; this script only chains runs.
# Raw-fulltext redos are NOT queued: they need the token-limit fallback first.
#
# Start:  ssh lenlenovo 'tmux new -d -s bio_catchup "bash /tmp/run_catchup_queue.sh"'
# Logs:   ~/Home/BiodiversityExtraction2026-2027/outputs/catchup_queue/queue.log
set -u
PROJ="${PROJ:-$HOME/Home/BiodiversityExtraction2026-2027}"
cd "$PROJ" || exit 1
[ -f .env ] && set -a && source .env && set +a
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

LOG_DIR="$PROJ/outputs/catchup_queue"
mkdir -p "$LOG_DIR"
QLOG="$LOG_DIR/queue.log"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$QLOG"; }

# --- preflight: wait for complete caches (pushed after Gruenau pull) ---
wait_for_caches() {
  while true; do
    n_marker=$(ls data/parsed_text/marker_dev/*.md 2>/dev/null | wc -l)
    n_grobid=$(ls data/parsed_text/grobid_dev/*.tei.xml 2>/dev/null | wc -l)
    if [ "$n_marker" -ge 100 ] && [ "$n_grobid" -ge 91 ]; then
      log "caches ready (marker=$n_marker grobid=$n_grobid)"
      return 0
    fi
    log "waiting for caches (marker=$n_marker/100 grobid=$n_grobid/91+)..."
    sleep 1800
  done
}

# run_one <tag> <provider> <model> <rpm> <extra flags...>
# Redos always execute: old metrics never count (that's the point of the redos).
# Completion is tracked with done_<tag> markers so restarts resume after the last run.
run_one() {
  local tag="$1" provider="$2" model="$3" rpm="$4"; shift 4
  if [ -f "$LOG_DIR/done_${tag}" ]; then
    log "SKIP $tag (done marker present)"
    return 0
  fi
  log ">>> START $tag ($provider/$model $*)"
  # shellcheck disable=SC2086
  $PY -u src/models/architectures/llm_rag_sections.py \
    --provider "$provider" --model "$model" --rpm "$rpm" $* \
    >> "$LOG_DIR/${tag}.log" 2>&1
  local rc=$?
  log "<<< END $tag (exit $rc)"
  [ $rc -eq 0 ] && touch "$LOG_DIR/done_${tag}"
  return $rc
}

SAIA_MODEL="deepseek-v4.1-flash-1017"
HU_MODEL="hu-reasoning"

wait_for_caches

# --- 1..4: v4-family redos on complete caches ---
run_one autoparser_ctaxa_v4      saia "$SAIA_MODEL" 20 --auto-parser --clean-taxa --v4-prompt || true
run_one autoparser_full_ctaxa_v4 saia "$SAIA_MODEL" 20 --auto-parser --clean-taxa --v4-prompt --full-text || true
run_one autoparser_ctaxa_v4_biknn saia "$SAIA_MODEL" 20 --auto-parser --clean-taxa --v4-prompt --bilingual-knn || true
run_one autoparser_ctaxa_v4_biknn_hu hu "$HU_MODEL" 15 --auto-parser --clean-taxa --v4-prompt --bilingual-knn || true
run_one marker_ev_ctaxa          saia "$SAIA_MODEL" 20 --marker --evidence-gates --clean-taxa || true
run_one marker_ev_ctaxa_hu        hu "$HU_MODEL" 15 --marker --evidence --clean-taxa || true
# --- 5: isolated clean-taxa (both providers) ---
run_one autoparser_ctaxa         saia "$SAIA_MODEL" 20 --auto-parser --clean-taxa || true
run_one autoparser_ctaxa_hu      hu   "$HU_MODEL"   15 --auto-parser --clean-taxa || true
# --- 6: HU v4 redos ---
run_one autoparser_ctaxa_v4_hu      hu "$HU_MODEL" 15 --auto-parser --clean-taxa --v4-prompt || true
run_one autoparser_full_ctaxa_v4_hu hu "$HU_MODEL" 15 --auto-parser --clean-taxa --v4-prompt --full-text || true
# --- 7: level-prompt test (both providers) ---
run_one autoparser_ctaxa_lvl      saia "$SAIA_MODEL" 20 --auto-parser --clean-taxa --level-prompt || true
run_one autoparser_ctaxa_lvl_hu   hu "$HU_MODEL" 15 --auto-parser --clean-taxa --level-prompt || true

log "=== QUEUE DONE ==="
