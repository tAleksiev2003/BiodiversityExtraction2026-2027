# Prompt versions

Never edit a version that has been run: copy it to the next number and change the copy.
The default version is set in `config/settings.yaml` (`paths.prompt_extraction`); older
versions can be run with an experiment file, e.g. `config/experiments/prompt_v1.yaml`.
Answers are cached per prompt, so re-running an old version costs no requests.

| Version | Date | Change | Dev with PDF (pipeline F, max_quote_words 40 / 60) |
|---|---|---|---|
| v1 | 2026-11-18 | first prompt (M3a) | C1 0.5250 / 0.5196, C2b 0.2854 / 0.3113 |
| v2 | 2026-11-18 | main subject instead of "this study only" (R3, system line, S2, S5); field base rates from train (R4); habitat may be derived (R5); at most 10 taxa (R7) | C1 0.5680 / 0.5711, C2b 0.3302 / 0.3315 |
| v3 | 2026-11-19 | the schema's guiding question under each field; R2 quote length from `verbatim_check` (`{min_quote_words}` to `{max_quote_words}`, was "5 to 40"); R7: smallest feasible level with base rate (half only groups, a third only species), always a German name (from text, translated or completed), scientific name or null; section "ADDITION FOR PIPELINE A" removed (SPEC.md, section 13) | C1 0.5122, C2b 0.3180 (checker 3-60 words): worse than v2 (0.5710, 0.3310). Taxa too coarse (groups instead of species), more ecosystem terms ("betrachtet"), biodiversity_level not better |
| chunk_v1 | 2026-11-18 | pipeline B only (`paths.prompt_extraction_chunked`): copy of v2 without the section "ADDITION FOR PIPELINE A", plus rule R9 (one chunk only: label what THIS chunk shows, empty is normal also for habitat, R4 rates are per document) | pipeline B with Qwen, k=1: C1 0.4527, C2b 0.2070. Since 2026-11-18 papers with one chunk use v2 instead |

| chunk_v2 | 2026-11-19 | pipeline B only: v3 plus rule R9 from chunk_v1 (one chunk only) | B (max 4 chunks): C1 0.5505, C2b 0.2881: worse than chunk_v1 (0.5872, 0.3311) |

Default: v2 (F) and chunk_v1 (B); v3 and chunk_v2 were tried on 2026-11-19 and were worse.
They run from the cache with `config/experiments/prompt_v3.yaml`. Since v3 the quote length in R2
comes from `verbatim_check`, so changing `min_quote_words` or `max_quote_words` changes the
v3 prompt and needs new requests.
