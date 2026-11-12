#!/bin/bash
# Sequential HU experiment chain: E1 term-enum, E2 clean-text, E3 translate-hint.
# Each run auto-resumes from its own checkpoint; a crash moves on to the next.
set -u
cd "$HOME/Home/BiodiversityExtraction2026-2027"
source .env 2>/dev/null
run() {
  local name="$1"; shift
  local ckpt="outputs/architectures/$1"; shift || true
  local dir
  dir=$(dirname "$ckpt")
  mkdir -p "$dir"
  if [ "$(grep -c . "$ckpt" 2>/dev/null || echo 0)" -ge 100 ]; then
    echo "$(date -u +%FT%TZ) [skip] $name (fertig)"; return 0
  fi
  echo "$(date -u +%FT%TZ) [run ] $name"
  .venv/bin/python src/models/architectures/llm_parallel_chunking.py --provider hu --model hu-reasoning --rpm 15 "$@" >> "$dir/run.log" 2>&1
  echo "$(date -u +%FT%TZ) [done] $name (exit $?)"
}
run E1_fg_term  llm_parallel_chunking_hu_hu-reasoning_fg_term/llm_parallel_chunking_hu_hu-reasoning_fg_term_checkpoint_api_hu_hu-reasoning_fg_term.jsonl  --field-guidelines --term-enum
run E2_fg_clean llm_parallel_chunking_hu_hu-reasoning_fg_clean/llm_parallel_chunking_hu_hu-reasoning_fg_clean_checkpoint_api_hu_hu-reasoning_fg_clean.jsonl --field-guidelines --clean-text
run E3_fg_trans llm_parallel_chunking_hu_hu-reasoning_fg_trans/llm_parallel_chunking_hu_hu-reasoning_fg_trans_checkpoint_api_hu_hu-reasoning_fg_trans.jsonl --field-guidelines --translate-hint
echo "$(date -u +%FT%TZ) ALLE EXPERIMENTE FERTIG"
