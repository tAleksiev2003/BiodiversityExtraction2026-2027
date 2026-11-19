"""Combine two runs: every field from a base run, the taxa from another run.

Why: on dev, pipeline F names the taxa better (it sees the whole paper and so the main
species), while pipeline B is better on biodiversity_level (a chunk makes the model read the
methods closely). A bootstrap over the 100 dev papers (2,000 samples) showed these two
differences as robust (F better on the taxa names in 97-99 % of the samples, B better on
biodiversity_level in 99.8 %); all other fields differed only by chance. So the rule has
one exception and no per-field tuning (SPEC.md, section 13).

The taxa entries are taken as a whole, because scientific_name, german_name and
species_group belong together in one entry.

Output: a normal run folder runs/<timestamp>_<name>/ with predictions.jsonl and
predictions.zip (checked like every submission) and sources.json (which runs and which rule).

Usage:
    python3 src/scripts/combine_predictions.py \
        --base runs/20261119-115739_parallel_dev_prompt_v2 \
        --taxa-from runs/20261119-115620_full_text_dev_prompt_v2 \
        --name ensemble_B_taxa_from_F_dev
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from src/
from data import ID_FIELD, index_by_id, read_jsonl  # noqa: E402
from output import PREDICTIONS_FILENAME, write_predictions  # noqa: E402
from runs import create_run_dir  # noqa: E402
from settings import load_settings  # noqa: E402

TAXA_FIELD = "taxa"


def load_predictions(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / PREDICTIONS_FILENAME
    return index_by_id(read_jsonl(path), str(path))


def combine(base: dict[str, dict[str, Any]], taxa_from: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per id: the base record with the taxa of the other run, sorted by id."""
    if set(base) != set(taxa_from):
        missing = sorted(set(base) ^ set(taxa_from))
        raise SystemExit(f"the two runs predict different documents, e.g. {missing[:5]}")
    records = []
    for document_id in sorted(base):
        record = dict(base[document_id])
        record[TAXA_FIELD] = taxa_from[document_id].get(TAXA_FIELD) or []
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", type=Path, required=True, help="run folder: every field except taxa")
    parser.add_argument("--taxa-from", type=Path, required=True, help="run folder: the taxa")
    parser.add_argument("--name", default="ensemble_B_taxa_from_F", help="suffix of the new run folder")
    args = parser.parse_args()

    settings = load_settings()
    base, taxa_from = load_predictions(args.base), load_predictions(args.taxa_from)
    records = combine(base, taxa_from)

    run_dir = create_run_dir(settings, args.name)
    paths = settings["paths"]
    zip_path = write_predictions(records, sorted(base), run_dir,
                                 [paths["schema_core"], paths["schema_biodiversity"]])
    sources = {
        "rule": "every field from base, the taxa entries from taxa_from",
        "base": str(args.base),
        "taxa_from": str(args.taxa_from),
        "documents": len(records),
    }
    (run_dir / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    print(f"{len(records)} documents combined ({ID_FIELD} from both runs match)")
    print(f"Written: {zip_path}")


if __name__ == "__main__":
    main()
