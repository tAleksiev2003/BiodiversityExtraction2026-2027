# Instructions for Claude Code

This is a student shared task (ANLP 2026/27, Task C1 + C2b). The full plan is in
`SPEC.md`. Read it completely before planning anything.

## Working rules

1. Build only the milestone you are asked for (SPEC.md, section 9) and stop afterwards.
2. If something is unclear or listed under "Open decisions" (SPEC.md, section 11), ask.
   Do not guess.
3. Follow the code quality rules (SPEC.md, section 10). The team learns from this code:
   clear names, small modules, type hints, short docstrings that explain *why*.
4. Read every setting from `config/settings.yaml`. Read allowed label values only from
   the schema files in `data/schemas/`. Never retype label strings.

## Data protection (strict)

- **Never read, list, open or print anything in `data/raw/`, `data/processed/`, `runs/`
  or `.env`.** They contain copyright-protected course data, results derived from it,
  or the API key. This is also blocked in `.claude/settings.json`.
- **Never run the pipeline, the helper scripts or the evaluation on real data.** The
  team runs these commands and reports scores, statistics or error messages back.
- For development and tests, use only `tests/fixtures/` with data you invent yourself.

## Where things are

| Path | Content | You may read it |
|---|---|---|
| `SPEC.md` | the plan | yes |
| `README.md` | overview for humans | yes |
| `docs/` | task description, HowTo | yes |
| `config/` | settings, prompts, experiment configs | yes |
| `data/schemas/` | JSON schemas (allowed values) | yes |
| `data/labels/` | generated label files | yes |
| `src/` | code | yes |
| `tests/` | tests and invented fixtures | yes |
| `data/raw/`, `data/processed/`, `runs/`, `.env` | course data, derived data, key | **no** |
