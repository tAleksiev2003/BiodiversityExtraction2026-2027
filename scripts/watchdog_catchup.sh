#!/bin/bash
# Watchdog for the catch-up queue on lenlenovo (bio_catchup tmux session).
# Install: */30 * * * * $HOME/Home/BiodiversityExtraction2026-2027/scripts/watchdog_catchup.sh
# Only restarts a dead queue while runs are still incomplete (metrics-gated, idempotent).
set -u
PROJ="$HOME/Home/BiodiversityExtraction2026-2027"
SCRIPT="$PROJ/scripts/run_catchup_queue.sh"
[ -f "$SCRIPT" ] || SCRIPT="/tmp/run_catchup_queue.sh"
cd "$PROJ" || exit 1

tmux has-session -t bio_catchup 2>/dev/null && exit 0

# 10 queued runs; restart only while done markers are missing.
done_n=$(ls "$PROJ/outputs/catchup_queue"/done_* 2>/dev/null | wc -l)
[ "$done_n" -ge 10 ] && exit 0

echo "$(date -u +%FT%TZ) watchdog: queue dead ($done_n metrics), restarting" \
  >> "$PROJ/outputs/catchup_queue/watchdog.log"
tmux kill-session -t bio_catchup 2>/dev/null
tmux new-session -d -s bio_catchup "bash $SCRIPT >> $PROJ/outputs/catchup_queue/queue.log 2>&1"
