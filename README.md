# ANLP 2026/27 – Shared Task C (C1 + C2b)

## 1. What is it?

A pipeline that reads scientific papers and books about biodiversity (PDF, English or
German) and fills a fixed German record for each of them: habitats, ecosystem types,
species groups, biodiversity levels, measured variables and ecosystem services.

We compare three ways of feeding long documents to a language model:

| Pipeline | Idea | Model |
|---|---|---|
| **F – full text** | the whole document in one call | `deepseek-v4-flash-0731` (1M context) |
| **A – sequential** | chunks one after another; an evidence ledger is passed on and can be corrected | `qwen3-30b-a3b-instruct-2507` |
| **B – parallel** | chunks independently; results are merged in Python | `qwen3-30b-a3b-instruct-2507` |

Every label must come with a verbatim quote. A Python verbatim checker drops every label
whose value is not in the schema or whose quote is not in the text.

The full plan is in `SPEC.md`. Rules for Claude Code are in `CLAUDE.md`.

## 2. Quick Start

1. **Install** (Python 3.13):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Add the API key:** copy `.env.example` to `.env` and enter your SAIA key.
   `.env` is never committed.
3. **Add the course data** (see section 5, "Repository Structure"):
   - PDFs into `data/raw/pdfs/train/`, `data/raw/pdfs/dev/`
   - reference files into `data/raw/references/`
   - schema files into `data/schemas/` as `core_fields.json` and `biodiversity.json`
   - task description and HowTo into `docs/`
4. **Generate the label files** (runs locally, data stays on your machine):
   ```bash
   python3 src/scripts/build_term_to_category.py \
       --references data/raw/references/train.jsonl \
       --schema-dir data/schemas \
       --output data/labels/term_to_category.json

   python3 src/scripts/build_variable_vocabulary.py \
       --references data/raw/references/train.jsonl \
       --output data/labels/variable_vocabulary.json
   ```
   Check the reports printed by both scripts by hand.
5. **Run the tests** (invented data only): `python -m pytest`
6. **Baselines and evaluation (M0).** There are two dev sets:
   - `full`: all ids in `dev.jsonl`. Use it to check our `evaluation.py` against the
     official values (random 0.1018, most frequent 0.2443) and for CodaBench uploads,
     because CodaBench rejects missing ids.
   - `with_pdf`: only the dev documents with a PDF. This is our main development set, and
     all systems are compared on it.
   ```bash
   # full dev set
   python src/baselines.py --kind random --split dev --dev-set full
   python src/baselines.py --kind most_frequent --split dev --dev-set full
   python src/evaluation.py score --references data/raw/references/dev.jsonl \
       --predictions runs/<run>/predictions.jsonl --task all --categorical-only

   # dev set with PDF
   python src/data.py make-dev-references          # writes data/processed/dev_with_pdf.jsonl
   python src/baselines.py --kind most_frequent --split dev --dev-set with_pdf
   python src/evaluation.py score --references data/processed/dev_with_pdf.jsonl \
       --predictions runs/<run>/predictions.jsonl --task all --categorical-only
   ```
   Each baseline writes `runs/<timestamp>_<name>/` containing `predictions.jsonl`,
   `predictions.zip` (upload this one), `config.yaml` and `commit.txt`.
7. **Run a pipeline:** the commands are added in milestones M1–M5 (see `SPEC.md`).

## 3. Quick Shutdown

- **Stop a running pipeline:** press `Ctrl + C` in the terminal.
- **Nothing is lost:** every LLM answer is stored in a disk cache as soon as it arrives.
- **Continue later:** start the same command again. Cached answers are reused, so only
  the missing requests are sent and no quota is wasted.
- **Leave the virtual environment:** `deactivate`

## 4. Input and Output

**Input:** one PDF per document, named after its id, e.g. `N82RZ7WL.pdf`.

**Output:** `predictions.jsonl`, one line per document, zipped for CodaBench:

```json
{"zotitem_ptr_id": "N82RZ7WL",
 "habitat": ["Küsten und Küstengewässer"],
 "ecosystem_type": [{"category": "I Biotoptypengruppen der Meere und Küsten", "term": "Benthal der Ostsee"}],
 "biodiversity_level": ["Artenvielfalt"],
 "taxa": [{"species_group": "Andere Pflanzen", "scientific_name": "Fucus spp", "german_name": "Blasentang"}],
 "biodiversity_variable": ["Biomasse"],
 "ecosystem_service": [{"category": "Regulierungs- und Erhaltungsleistungen", "term": "<exact term from the schema>"}]}
```

(Shown on several lines for reading; in the file it is one line. The C1 part and the
taxa names come from the reference record in the task description; `biodiversity_variable`
and `ecosystem_service` are format examples only.)

| Kind of field | Fields | Scored by |
|---|---|---|
| fixed list (enum) | `habitat`, `biodiversity_level`, `ecosystem_type.*`, `taxa.species_group`, `ecosystem_service.*` | exact match |
| open text, exact | `taxa.scientific_name`, `taxa.german_name` | exact match |
| free text | `biodiversity_variable` | BERTScore, not used for the ranking |

## 5. Repository Structure

Listed in working order (GitHub and Finder sort alphabetically).

| Folder / file | Content | Created by | Claude Code may read |
|---|---|---|---|
| `docs/` | task description, HowTo | you | yes |
| `data/raw/` | course PDFs and reference files | you | **no** |
| `data/schemas/` | JSON schemas with all allowed values | you (copied, never retyped) | yes |
| `data/labels/` | term → category mapping, vocabulary of measured variables | scripts in `src/scripts/` | yes |
| `config/settings.yaml` | everything you may change | you | yes |
| `config/prompts/` | prompt templates | you | yes |
| `config/experiments/` | one YAML file per experiment | you / Claude Code | yes |
| `src/` | pipeline code | Claude Code | yes |
| `src/scripts/` | helper scripts that build `data/labels/` | given | yes |
| `data/processed/` | Docling output and chunks | pipeline | **no** |
| `runs/` | one folder per run: logs, predictions, scores | pipeline | **no** |
| `tests/` | tests and invented test data | Claude Code | yes |
| `.env` | API key | you | **no** |

## 6. What to modify if necessary

| You want to … | Change … |
|---|---|
| use another model | `config/settings.yaml` → `models` |
| change chunk size or overlap | `config/settings.yaml` → `chunking` |
| make the verbatim checker stricter or looser | `config/settings.yaml` → `verbatim_check` |
| change how many vocabulary terms the prompt shows | `config/settings.yaml` → `prompt.vocabulary_top_n` |
| work on a smaller dev subset | `config/settings.yaml` → `data.dev_subset_size` |
| change the instructions to the model | `config/prompts/extraction.txt` |
| add an experiment | new file in `config/experiments/` |
| change paths | `config/settings.yaml` → `paths` |

Do not change allowed label values by hand: they are read from `data/schemas/`.
