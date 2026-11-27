#!/bin/bash
# Train-split router verification, stage 1 (SAIA): run the pipeline on a fixed
# train sample with three parser routes, then compare per-paper C1 on TRAIN
# labels (labels never used to pick the rules). Requires commit >= 2026-11-27
# (--split/--ids-file/--knn-exclude-self). Waits for the validate3 queue.
# Sample: scripts/train_router_sample_ids.txt (generated from the signal audit:
# 20 long (pages>=25), 10 medium (10-24p), 10 short/clean, 6 scan).
#
# Start: ssh lenlenovo 'tmux new -d -s bio_trainverif "bash /tmp/run_train_router_verification.sh"'
# Logs: outputs/catchup_queue/queue.log and outputs/catchup_queue/trainverif_*.log
set -u
PROJ="${PROJ:-$HOME/Home/BiodiversityExtraction2026-2027}"
cd "$PROJ" || exit 1
[ -f .env ] && set -a && source .env && set +a
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

LOG_DIR="$PROJ/outputs/catchup_queue"
QLOG="$LOG_DIR/queue.log"
IDS_FILE="$PROJ/scripts/train_router_sample_ids.txt"
mkdir -p "$LOG_DIR"
log() { echo "$(date -u +%FT%TZ) [trainverif] $*" | tee -a "$QLOG"; }

[ -f "$IDS_FILE" ] || { log "!! $IDS_FILE missing, aborting"; exit 1; }
N=$(grep -c . "$IDS_FILE")
log "train router verification on $N train papers (sample ids file present)"

# Wait for the last queue stage of the chain.
while ! grep -q "\[validate3\] === VALIDATE3 saia DONE" "$QLOG" 2>/dev/null; do
  log "waiting for validate3 saia to finish, sleeping 20m"
  sleep 1200
done

run_one() {
  local tag="$1"; shift
  if [ -f "$LOG_DIR/done_trainverif_${tag}" ]; then
    log "SKIP $tag (done marker present)"
    return 0
  fi
  log ">>> START $tag ($*)"
  # shellcheck disable=SC2086
  $PY -u src/models/architectures/llm_rag_sections.py \
    --provider saia --model deepseek-v4.1-flash-1017 --rpm 20 \
    --split train --ids-file "$IDS_FILE" --knn-exclude-self --no-save "$@" \
    >> "$LOG_DIR/trainverif_${tag}.log" 2>&1
  local rc=$?
  log "<<< END $tag (exit $rc)"
  [ $rc -eq 0 ] && touch "$LOG_DIR/done_trainverif_${tag}"
  return 0
}

# Three parser routes, identical prompt flags otherwise (baseline prompt keeps
# marker/grobid comparable with their dev baselines; v4 variants optional later).
run_one trainverif_grobid --grobid-url http://127.0.0.1:8070
run_one trainverif_marker --marker
run_one trainverif_docling --docling

log "=== TRAINVERIF DONE ==="
