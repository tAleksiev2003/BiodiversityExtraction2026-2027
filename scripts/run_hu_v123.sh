#!/bin/bash
# HU Hebel 1-3 chain (all on top of the E1 winning config):
#   R1: --field-guidelines --v3-guidelines --term-enum
#   R2: R1 + --latin-collective            (Hebel 2)
#   R3: R1 + --strict-votes                (Hebel 3)
# Sequential (single HU bridge), checkpoint-resumable; crashes move to next.
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
run R1_v3_term       llm_parallel_chunking_hu_hu-reasoning_fg_v3_term/llm_parallel_chunking_hu_hu-reasoning_fg_v3_term_checkpoint_api_hu_hu-reasoning_fg_v3_term.jsonl        --field-guidelines --v3-guidelines --term-enum
run R2_v3_term_latin llm_parallel_chunking_hu_hu-reasoning_fg_v3_term_latin/llm_parallel_chunking_hu_hu-reasoning_fg_v3_term_latin_checkpoint_api_hu_hu-reasoning_fg_v3_term_latin.jsonl --field-guidelines --v3-guidelines --term-enum --latin-collective
run R3_v3_term_svote llm_parallel_chunking_hu_hu-reasoning_fg_v3_term_svote/llm_parallel_chunking_hu_hu-reasoning_fg_v3_term_svote_checkpoint_api_hu_hu-reasoning_fg_v3_term_svote.jsonl --field-guidelines --v3-guidelines --term-enum --strict-votes
echo "$(date -u +%FT%TZ) ALLE HU-HEBEL-LAEUFE FERTIG"
