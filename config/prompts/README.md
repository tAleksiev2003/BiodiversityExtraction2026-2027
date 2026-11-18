# Prompt versions

Never edit a version that has been run: copy it to the next number and change the copy.
The default version is set in `config/settings.yaml` (`paths.prompt_extraction`); older
versions can be run with an experiment file, e.g. `config/experiments/prompt_v1.yaml`.
Answers are cached per prompt, so re-running an old version costs no requests.

| Version | Date | Change | Dev with PDF (pipeline F, max_quote_words 40 / 60) |
|---|---|---|---|
| v1 | 2026-11-18 | first prompt (M3a) | C1 0.5250 / 0.5196, C2b 0.2854 / 0.3113 |
| v2 | 2026-11-18 | main subject instead of "this study only" (R3, system line, S2, S5); field base rates from train (R4); habitat may be derived (R5); at most 10 taxa (R7) | C1 0.5680 / 0.5711, C2b 0.3302 / 0.3315 |
