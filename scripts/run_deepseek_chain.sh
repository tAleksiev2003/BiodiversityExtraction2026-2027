#!/bin/bash
# DeepSeek comparison runs — DeepSeek via SAIA (deepseek-v4.1-flash-1017, covered by
# the existing SAIA_API_KEY; no separate DeepSeek key needed).
# DS1: Pipeline B with the E1 winning config (field guidelines + term enum) —
#      directly comparable to HU E1 (C1 0.5593) and SAIA mistral v2-fg.
# DS2: English-first sections pipeline (same config family as the glm-5.3 run and
#      the other session's HU en/de runs).
# Each run auto-resumes from its checkpoint and retries on crash until 100 papers.
set -u
PROJ="$HOME/Home/BiodiversityExtraction2026-2027"
cd "$PROJ" || exit 1
source .env 2>/dev/null

MODEL="deepseek-v4.1-flash-1017"
LOG="$PROJ/outputs/deepseek_chain.log"
CKPT_B="$PROJ/outputs/architectures/llm_parallel_chunking_saia_${MODEL}_fg_term/llm_parallel_chunking_saia_${MODEL}_fg_term_checkpoint_api_saia_${MODEL}_fg_term.jsonl"
CKPT_S="$PROJ/outputs/architectures/llm_sections_parallel_saia_${MODEL/_en/}_en/llm_sections_parallel_saia_${MODEL}_en_checkpoint.jsonl"

run_to_100() {
  local name="$1" ckpt="$2"; shift 2
  while [ "$(grep -c . "$ckpt" 2>/dev/null || echo 0)" -lt 100 ]; do
    echo "$(date -u +%FT%TZ) [run ] $name (papers=$(grep -c . "$ckpt" 2>/dev/null || echo 0))" >> "$LOG"
    "$@"
    echo "$(date -u +%FT%TZ) [exit] $name rc=$? — retry in 60s" >> "$LOG"
    sleep 60
  done
  echo "$(date -u +%FT%TZ) [done] $name" >> "$LOG"
}

mkdir -p "$(dirname "$CKPT_B")" "$(dirname "$CKPT_S")"

run_to_100 "DS1_parallel_chunking_fg_term" "$CKPT_B" \
  .venv/bin/python src/models/architectures/llm_parallel_chunking.py \
  --provider saia --model "$MODEL" --rpm 15 --field-guidelines --term-enum

run_to_100 "DS2_sections_parallel_en" "$CKPT_S" \
  .venv/bin/python src/models/architectures/llm_sections_pipeline.py \
  --provider saia --model "$MODEL" --architecture parallel --language en --rpm 15

echo "$(date -u +%FT%TZ) ALLE DEEPSEEK-LÄUFE FERTIG" >> "$LOG"
