"""Read reference files and find which documents exist.

Usage (run by the team, it reads course data):
    python src/data.py make-dev-references

This writes the dev references limited to the documents that have a PDF
(`paths.references_dev_with_pdf`). That file is our main development set: only these
documents can be processed by the LLM pipelines.
"""
import argparse
import json
import random
from pathlib import Path
from typing import Any

from settings import load_settings

ID_FIELD = "zotitem_ptr_id"  # key that links a record to its PDF (task description)
PDF_SUFFIX = ".pdf"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """json.loads silently keeps the last of two equal keys; CodaBench rejects them."""
    record: dict[str, Any] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate JSON key {key!r}")
        record[key] = value
    return record


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSON object per line; empty lines are skipped, duplicate keys are errors."""
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line, object_pairs_hook=_reject_duplicate_keys))
            except ValueError as error:
                raise ValueError(f"{path}, line {line_number}: {error}") from error
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write one compact JSON object per line, keeping German umlauts readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def index_by_id(records: list[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    """Map id -> record and fail on missing or duplicate ids."""
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        document_id = record.get(ID_FIELD)
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(f"{source}: record without a valid {ID_FIELD}")
        if document_id in by_id:
            raise ValueError(f"{source}: duplicate id {document_id}")
        by_id[document_id] = record
    return by_id


def load_references(path: Path) -> dict[str, dict[str, Any]]:
    return index_by_id(read_jsonl(path), str(path))


def list_pdf_ids(pdf_dir: Path) -> list[str]:
    """Ids of all PDFs in a folder; every PDF is named after its id."""
    return sorted(
        pdf_path.stem
        for pdf_path in pdf_dir.iterdir()
        if pdf_path.suffix.lower() == PDF_SUFFIX
    )


def select_dev_subset(ids: list[str], size: int | None, seed: int) -> list[str]:
    """Pick a fixed random subset, so every run uses the same documents."""
    sorted_ids = sorted(ids)
    if size is None or size >= len(sorted_ids):
        return sorted_ids
    return sorted(random.Random(seed).sample(sorted_ids, size))


def dev_ids_with_pdf(settings: dict[str, Any]) -> list[str]:
    """Dev ids that have both a reference record and a PDF (main development set)."""
    paths = settings["paths"]
    reference_ids = set(load_references(paths["references_dev"]))
    pdf_ids = set(list_pdf_ids(paths["pdfs_dev"]))
    pdfs_without_reference = pdf_ids - reference_ids
    if pdfs_without_reference:
        print(f"warning: {len(pdfs_without_reference)} dev PDFs have no reference record")
    ids = sorted(reference_ids & pdf_ids)
    return select_dev_subset(ids, settings["data"]["dev_subset_size"], settings["data"]["dev_subset_seed"])


def make_dev_references(settings: dict[str, Any]) -> Path:
    """Write the dev references limited to the documents with a PDF."""
    paths = settings["paths"]
    references = load_references(paths["references_dev"])
    selected_ids = dev_ids_with_pdf(settings)
    output_path = paths["references_dev_with_pdf"]
    write_jsonl([references[document_id] for document_id in selected_ids], output_path)
    print(f"dev references: {len(references)}, with PDF (selected): {len(selected_ids)}")
    print(f"Written: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Data helpers.")
    parser.add_argument("command", choices=["make-dev-references"])
    parser.add_argument("--experiment", type=Path, default=None)
    args = parser.parse_args()
    settings = load_settings(args.experiment)
    if args.command == "make-dev-references":
        make_dev_references(settings)


if __name__ == "__main__":
    main()
