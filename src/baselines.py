"""The two baselines that every valid submission must beat (task description, Task C).

- random: one random allowed value for every enum field and subfield;
- most_frequent: the most frequent training value for every scored field and subfield,
  chosen independently per subfield (category and term are not forced to match).

Usage (run by the team, it reads course data):
    python src/baselines.py --kind random --split dev --dev-set full
    python src/baselines.py --kind most_frequent --split dev --dev-set with_pdf
    python src/baselines.py --kind most_frequent --split test

`--dev-set full` predicts for all ids in the dev references (needed for CodaBench, which
rejects missing ids); `--dev-set with_pdf` predicts only for the dev documents with a PDF,
our main development set.
"""
import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any

from data import ID_FIELD, dev_ids_with_pdf, list_pdf_ids, load_references
from evaluation import C1_FIELDS, C2B_FIELDS, FREE_TEXT_FIELDS, extract_values
from labels import FieldSpec, load_field_specs
from output import write_predictions
from runs import create_run_dir
from settings import load_settings

SCORED_FIELDS = list(dict.fromkeys(C1_FIELDS + C2B_FIELDS + FREE_TEXT_FIELDS))


def scored_field_specs(settings: dict[str, Any]) -> list[FieldSpec]:
    """Field specs from both schemas, limited to fields that are scored."""
    paths = settings["paths"]
    specs = load_field_specs([paths["schema_core"], paths["schema_biodiversity"]])
    return [specs[name] for name in SCORED_FIELDS]


def build_record(document_id: str, values: dict[str, str], specs: list[FieldSpec]) -> dict[str, Any]:
    """Turn {field name: one value} into a record with lists and one object per parent.

    Example: {"ecosystem_type.category": c, "ecosystem_type.term": t}
    becomes {"ecosystem_type": [{"category": c, "term": t}]}.
    """
    record: dict[str, Any] = {ID_FIELD: document_id}
    for spec in specs:
        if spec.name not in values:
            continue
        if spec.subfield is None:
            record[spec.parent] = [values[spec.name]]
        else:
            record.setdefault(spec.parent, [{}])[0][spec.subfield] = values[spec.name]
    return record


def random_baseline(ids: list[str], specs: list[FieldSpec], seed: int) -> list[dict[str, Any]]:
    """One random allowed value per enum field; open-text fields have no allowed values."""
    rng = random.Random(seed)
    enum_specs = [spec for spec in specs if spec.is_enum]
    records = []
    for document_id in ids:
        values = {spec.name: rng.choice(spec.allowed_values) for spec in enum_specs}
        records.append(build_record(document_id, values, specs))
    return records


def most_frequent_values(
    train_references: dict[str, dict[str, Any]], specs: list[FieldSpec]
) -> dict[str, str]:
    """Most frequent training value per field, counted once per document.

    Values of enum fields that are not in the schema are ignored, because the writer
    would reject them.
    """
    values: dict[str, str] = {}
    for spec in specs:
        counts: Counter = Counter()
        for reference in train_references.values():
            for value in extract_values(reference, spec.name):
                if not spec.is_enum or value in spec.allowed_values:
                    counts[value] += 1
        if counts:
            values[spec.name] = counts.most_common(1)[0][0]
    return values


def most_frequent_baseline(
    ids: list[str], specs: list[FieldSpec], train_references: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    values = most_frequent_values(train_references, specs)
    return [build_record(document_id, values, specs) for document_id in ids]


def target_ids(settings: dict[str, Any], split: str, dev_set: str) -> list[str]:
    """Ids that must be predicted for the chosen split and dev set."""
    paths = settings["paths"]
    if split == "test":
        return list_pdf_ids(paths["pdfs_test"])
    if dev_set == "full":
        return sorted(load_references(paths["references_dev"]))
    return dev_ids_with_pdf(settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a baseline predictions.jsonl and zip.")
    parser.add_argument("--kind", choices=["random", "most_frequent"], required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument(
        "--dev-set", choices=["full", "with_pdf"], default="with_pdf",
        help="only for --split dev: all dev reference ids, or only those with a PDF",
    )
    parser.add_argument("--experiment", type=Path, default=None)
    args = parser.parse_args()

    settings = load_settings(args.experiment)
    specs = scored_field_specs(settings)
    ids = target_ids(settings, args.split, args.dev_set)

    if args.kind == "random":
        records = random_baseline(ids, specs, settings["baselines"]["random_seed"])
    else:
        train_references = load_references(settings["paths"]["references_train"])
        records = most_frequent_baseline(ids, specs, train_references)

    run_name = f"baseline_{args.kind}_{args.split}"
    if args.split == "dev":
        run_name += f"_{args.dev_set}"
    run_dir = create_run_dir(settings, run_name)
    paths = settings["paths"]
    zip_path = write_predictions(records, ids, run_dir, [paths["schema_core"], paths["schema_biodiversity"]])
    print(f"documents: {len(records)}")
    print(f"Written: {zip_path}")


if __name__ == "__main__":
    main()
