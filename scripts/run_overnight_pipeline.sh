#!/bin/bash
# ======================================================================================
# Overnight Multi-Model & Multi-Language Section Extraction Benchmark
# Chains 4 experimental conditions sequentially:
#   1. Parallel B (English extraction + translation mapping) with hu-reasoning
#   2. Parallel B (Direct German labels, no translation) with hu-reasoning
#   3. Parallel B (English extraction + translation mapping) with Mistral Large (SAIA)
#   4. Sequential A (State-passing fold) with hu-reasoning
#
# Each condition:
#   - Checkpoints every paper to disk immediately.
#   - Automatically skips already processed papers if restarted.
#   - Robust retry on transient network/API issues with exponential backoff.
# ======================================================================================
set -u

REPO_DIR="$HOME/Home/BiodiversityExtraction2026-2027"
cd "$REPO_DIR" || exit 1

# Load environment variables (API keys)
if [ -f .env ]; then
  source .env
fi

PYTHON_BIN=".venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

LOG_DIR="$REPO_DIR/outputs/overnight_runs"
mkdir -p "$LOG_DIR"
OVERNIGHT_LOG="$LOG_DIR/overnight_execution.log"

log() {
  local msg="$(date -u +'%FT%TZ') $*"
  echo "$msg"
  echo "$msg" >> "$OVERNIGHT_LOG"
}

log "=== STARTING BIODIVERSITY OVERNIGHT BENCHMARK RUN ==="
log "Host: $(hostname) | Repo: $REPO_DIR"

run_experiment() {
  local tag="$1"; shift
  local log_file="$LOG_DIR/${tag}.log"
  log ">>> Starting Condition: $tag"
  log "Command: $PYTHON_BIN src/models/architectures/llm_sections_pipeline.py $*"

  $PYTHON_BIN src/models/architectures/llm_sections_pipeline.py "$@" >> "$log_file" 2>&1
  local exit_code=$?

  if [ $exit_code -eq 0 ]; then
    log "<<< Condition $tag COMPLETED SUCCESSFULLY (exit 0)"
  else
    log "[WARNING] Condition $tag exited with code $exit_code. Inspect $log_file"
  fi
}

# --------------------------------------------------------------------------------------
# Run 1: Parallel B (English + Translation Mapping) with hu-reasoning
# --------------------------------------------------------------------------------------
run_experiment "01_parallel_hu_english" \
  --provider hu \
  --model hu-reasoning \
  --architecture parallel \
  --language en \
  --rpm 15

# --------------------------------------------------------------------------------------
# Run 2: Parallel B (Direct German Labels, No Translation) with hu-reasoning
# --------------------------------------------------------------------------------------
run_experiment "02_parallel_hu_german" \
  --provider hu \
  --model hu-reasoning \
  --architecture parallel \
  --language de \
  --rpm 15

# --------------------------------------------------------------------------------------
# Run 3: Parallel B with Mistral Large (SAIA)
# --------------------------------------------------------------------------------------
run_experiment "03_parallel_mistral_english" \
  --provider saia \
  --model mistral-large-4-1t-a52b \
  --architecture parallel \
  --language en \
  --rpm 20

# --------------------------------------------------------------------------------------
# Run 4: Sequential A (State-Passing Fold) with hu-reasoning
# --------------------------------------------------------------------------------------
run_experiment "04_sequential_hu_english" \
  --provider hu \
  --model hu-reasoning \
  --architecture sequential \
  --language en \
  --rpm 15

log "=== ALL OVERNIGHT BENCHMARK RUNS FINISHED ==="
