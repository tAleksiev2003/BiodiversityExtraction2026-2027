#!/bin/bash
# Stacked confirmation runs (HU/Qwen): the adopted mechanisms COMBINED, since
# each was measured only individually on the recipe. Two variants, sequential
# (keeps total HU concurrency <= 3 together with the hupar runner):
#   stack1: recipe + service-gate + service-sharpen + eco-type-land-rule
#   stack2: stack1 + doc-type-scale
# Baselines: recipe 0.5604/0.4166/0.4264; individual gains +1.1 (svcg),
# +1.2 (svs), +3.4 (ectl). Expectation if effects add: ~0.59-0.60 C1;
# interactions may also cancel — that is exactly what this measures.
# Start: ssh lenlenovo 'tmux new -d -s bio_hustack "bash /tmp/run_hu_stacked.sh"'
set -u
PROJ="${PROJ:-$HOME/Home/BiodiversityExtraction2026-2027}"
cd "$PROJ" || exit 1
[ -f .env ] && set -a && source .env && set +a
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
LOG_DIR="$PROJ/outputs/catchup_queue"
log() { echo "$(date -u +%FT%TZ) [hustack] $*" | tee -a "$LOG_DIR/queue.log"; }

run_one() {
  local tag="$1" run_id="$2"; shift 2
  [ -f "$LOG_DIR/done_stacked_${tag}" ] && { log "SKIP $tag"; return 0; }
  local d="$PROJ/outputs/architectures/$run_id"
  local ck
  ck=$(ls "$d"/*_checkpoint.jsonl 2>/dev/null | head -1)
  if [ -n "$ck" ] && [ "$(grep -c . "$ck")" -ge 100 ]; then
    tar -czf "$PROJ/outputs/redo_backup/${run_id}_pre-stacked_$(date -u +%Y%m%dT%H%M%SZ).tar.gz" \
      -C "$PROJ/outputs/architectures" "$run_id"
    rm -f "$ck"
    log "archived $run_id, checkpoint removed"
  fi
  log ">>> START $tag"
  # shellcheck disable=SC2086
  $PY -u src/models/architectures/llm_rag_sections.py \
    --provider hu --model hu-reasoning --rpm 15 \
    --auto-parser --clean-taxa --v4-prompt --bilingual-knn "$@" \
    >> "$LOG_DIR/stacked_${tag}.log" 2>&1
  local rc=$?
  log "<<< END $tag (exit $rc)"
  [ $rc -eq 0 ] && touch "$LOG_DIR/done_stacked_${tag}"
}

run_one stack1 "llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn_svcg_svs_ectl" \
  --service-gate --service-sharpen --eco-type-land-rule || true
run_one stack2 llm_rag_sections_hu_hu-reasoning_autoparser_ctaxa_v4_biknn_svcg_svs_ectl_dtsc \
  --service-gate --service-sharpen --eco-type-land-rule --doc-type-scale || true

log "=== HU STACKED DONE ==="
