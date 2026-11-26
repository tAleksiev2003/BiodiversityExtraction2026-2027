#!/bin/bash
# Follow-up after bio_catchup: fair full-Docling rerun on Minh's JSON cache (do_ocr=true).
# Waits for all 12 catchup done markers, then runs minj + clean-taxa + v4 on both
# providers sequentially. The LLM client waits out 429 quota resets itself; this
# script only chains runs. Raw-fulltext is NOT queued: needs the token-limit fallback.
#
# Start:  ssh lenlenovo 'tmux new -d -s bio_minj "bash ~/Home/BiodiversityExtraction2026-2027/scripts/run_minj_after_catchup.sh"'
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

# --- wait for the catchup queue (12 run_one calls) ---
while [ "$(ls "$LOG_DIR"/done_* 2>/dev/null | wc -l)" -lt 12 ]; do
  log "minj wait: $(ls "$LOG_DIR"/done_* 2>/dev/null | wc -l)/12 catchup done, sleeping 30m"
  sleep 1800
done
log "catchup complete, starting minj reruns"

# --- preflight: Minh JSON cache must be complete ---
n_minj=$(ls data/processed/docling/dev/*.json 2>/dev/null | wc -l)
if [ "$n_minj" -lt 100 ]; then
  log "!! minj cache incomplete ($n_minj/100), aborting"
  exit 1
fi

# run_one <tag> <provider> <model> <rpm> <extra flags...>
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

run_one minj_ctaxa_v4      saia "$SAIA_MODEL" 20 --minh-json --clean-taxa --v4-prompt || true
run_one minj_ctaxa_v4_hu   hu   "$HU_MODEL"   15 --minh-json --clean-taxa --v4-prompt || true

log "=== MINJ DONE ==="
