#!/bin/bash
# Two-phase watchdog for the SAIA GLM sections runs (llm_sections_pipeline.py).
#   Phase 1: keep glm-5.3-flash (parallel/en) alive until 100 papers.
#   Phase 2: once flash completes, launch glm-5.3 (full) resume until 100.
#   glm-5.3 full returns 503 (max capacity) while SAIA is saturated; the
#   runner exits and this watchdog retries next tick (every 10 min via cron).
# Checkpoints auto-resume: restarted runs skip papers already on disk.
# Deployment: copy to ~/Home/BiodiversityExtraction2026-2027/watchdog_glm.sh
# and add: */10 * * * * $HOME/Home/BiodiversityExtraction2026-2027/watchdog_glm.sh
set -u
PROJ="$HOME/Home/BiodiversityExtraction2026-2027"
D_FLASH="$PROJ/outputs/architectures/llm_sections_parallel_saia_glm-5.3-flash_en"
C_FLASH="$D_FLASH/llm_sections_parallel_saia_glm-5.3-flash_en_checkpoint.jsonl"
D_FULL="$PROJ/outputs/architectures/llm_sections_parallel_saia_glm-5.3_en"
C_FULL="$D_FULL/llm_sections_parallel_saia_glm-5.3_en_checkpoint.jsonl"
cd "$PROJ" || exit 1
n() { grep -c . "$1" 2>/dev/null || echo 0; }
flash_alive() { pgrep -f "llm_sections_pipeline.*glm-5.3-flash" > /dev/null; }
full_alive()  { pgrep -f "llm_sections_pipeline.*--model glm-5.3 " > /dev/null || pgrep -f "llm_sections_pipeline.*--model glm-5.3$" > /dev/null; }

# Phase 2 done: nothing to do.
[ "$(n "$C_FULL")" -ge 100 ] && exit 0

if [ "$(n "$C_FLASH")" -lt 100 ]; then
  # Phase 1: keep flash alive.
  flash_alive && exit 0
  mkdir -p "$D_FLASH"
  tmux kill-session -t bio_glm 2>/dev/null
  tmux new-session -d -s bio_glm "source .env && .venv/bin/python src/models/architectures/llm_sections_pipeline.py --provider saia --model glm-5.3-flash --architecture parallel --language en --rpm 10 >> $D_FLASH/run.log 2>&1"
  echo "$(date -u +%FT%TZ) restart flash (papers=$(n "$C_FLASH"))" >> "$D_FLASH/watchdog.log"
  exit 0
fi

# Phase 2: flash done -> launch full (resume from its own checkpoint).
full_alive && exit 0
mkdir -p "$D_FULL"
tmux kill-session -t bio_glm_full 2>/dev/null
tmux new-session -d -s bio_glm_full "source .env && .venv/bin/python src/models/architectures/llm_sections_pipeline.py --provider saia --model glm-5.3 --architecture parallel --language en --rpm 10 >> $D_FULL/run.log 2>&1"
echo "$(date -u +%FT%TZ) launch full (papers=$(n "$C_FULL"))" >> "$D_FULL/watchdog.log"
