#!/bin/bash
# DeepSeek comparison runs (provider comparison on the two leading architectures).
# WAITS for DEEPSEEK_API_KEY: add the key to ~/Home/BiodiversityExtraction2026-2027/.env
# (export DEEPSEEK_API_KEY=sk-...) and this chain starts by itself; each run
# auto-resumes from its checkpoint and retries on crash until 100 papers are done.
set -u
PROJ="$HOME/Home/BiodiversityExtraction2026-2027"
cd "$PROJ" || exit 1

CKPT_B="$PROJ/outputs/architectures/llm_parallel_chunking_deepseek_deepseek-chat_fg_term/llm_parallel_chunking_deepseek_deepseek-chat_fg_term_checkpoint_api_deepseek_deepseek-chat_fg_term.jsonl"
CKPT_S="$PROJ/outputs/architectures/llm_sections_parallel_deepseek_deepseek-chat_en/llm_sections_parallel_deepseek_deepseek-chat_en_checkpoint.jsonl"

# --- wait for the key (checks .env fresh each tick) ---
while true; do
  set -a; source .env 2>/dev/null; set +a
  [ -n "${DEEPSEEK_API_KEY:-}" ] && break
  echo "$(date -u +%FT%TZ) waiting for DEEPSEEK_API_KEY in $PROJ/.env ..." >> "$PROJ/outputs/deepseek_chain.log"
  sleep 300
done

run_to_100() {
  local name="$1" ckpt="$2"; shift 2
  while [ "$(grep -c . "$ckpt" 2>/dev/null || echo 0)" -lt 100 ]; do
    echo "$(date -u +%FT%TZ) [run ] $name (papers=$(grep -c . "$ckpt" 2>/dev/null || echo 0))" >> "$PROJ/outputs/deepseek_chain.log"
    "$@"
    echo "$(date -u +%FT%TZ) [exit] $name rc=$?" >> "$PROJ/outputs/deepseek_chain.log"
    sleep 30
  done
  echo "$(date -u +%FT%TZ) [done] $name" >> "$PROJ/outputs/deepseek_chain.log"
}

mkdir -p "$(dirname "$CKPT_B")" "$(dirname "$CKPT_S")"

# DS1: Pipeline B with the E1 winning config (field guidelines + term enum) —
# directly comparable to HU E1 (C1 0.5593) and SAIA v2-fg.
run_to_100 "DS1_parallel_chunking_fg_term" "$CKPT_B" \
  .venv/bin/python src/models/architectures/llm_parallel_chunking.py \
  --provider deepseek --model deepseek-chat --rpm 30 --field-guidelines --term-enum

# DS2: English-first sections pipeline (same config family as the running glm-5.3
# and the other session's HU en/de runs).
run_to_100 "DS2_sections_parallel_en" "$CKPT_S" \
  .venv/bin/python src/models/architectures/llm_sections_pipeline.py \
  --provider deepseek --model deepseek-chat --architecture parallel --language en --rpm 30

echo "$(date -u +%FT%TZ) ALLE DEEPSEEK-LÄUFE FERTIG" >> "$PROJ/outputs/deepseek_chain.log"
