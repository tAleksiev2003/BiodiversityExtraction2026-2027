"""Validate predictions and write predictions.jsonl plus the CodaBench zip.

CodaBench rejects a submission for missing, duplicate or unknown ids, duplicate JSON keys
and values of the wrong type, and a rejected upload costs time. So everything is checked
here before a file is written, and a wrong record stops the write with its id.
"""
import zipfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from data import ID_FIELD, read_jsonl, write_jsonl
from labels import load_schema

PREDICTIONS_FILENAME = "predictions.jsonl"  # name required by CodaBench
ZIP_FILENAME = "predictions.zip"
# Subfields allowed by the schema but not scored; leaving them out keeps the file small.
UNSCORED_SUBFIELDS = {"ecosystem_service": ["details"]}


class OutputError(ValueError):
    """A prediction that CodaBench would reject."""


def build_validators(schema_paths: list[Path]) -> list[tuple[Draft202012Validator, set[str]]]:
    """One validator per schema, together with the top-level fields it knows."""
    validators = []
    for schema_path in schema_paths:
        schema = load_schema(schema_path)
        validators.append((Draft202012Validator(schema), set(schema["properties"])))
    return validators


def drop_unscored_subfields(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy without subfields that are not scored (e.g. ecosystem_service.details)."""
    cleaned = dict(record)
    for field, subfields in UNSCORED_SUBFIELDS.items():
        if isinstance(cleaned.get(field), list):
            cleaned[field] = [
                {key: value for key, value in item.items() if key not in subfields}
                if isinstance(item, dict) else item
                for item in cleaned[field]
            ]
    return cleaned


def validate_record(
    record: dict[str, Any], validators: list[tuple[Draft202012Validator, set[str]]]
) -> None:
    """Check one record against all schemas.

    Both schemas forbid unknown top-level keys, but our record combines C1 and C2b fields
    plus the id. Therefore each schema only sees the fields it defines (its "projection"),
    and keys that no schema defines are reported separately.
    """
    document_id = record.get(ID_FIELD)
    if not isinstance(document_id, str) or not document_id:
        raise OutputError(f"record without a valid {ID_FIELD}: {record!r:.200}")

    known_fields = set().union(*(fields for _, fields in validators))
    unknown_fields = set(record) - known_fields - {ID_FIELD}
    if unknown_fields:
        raise OutputError(f"{document_id}: unknown fields {sorted(unknown_fields)}")

    for validator, fields in validators:
        projection = {key: value for key, value in record.items() if key in fields}
        errors = sorted(validator.iter_errors(projection), key=lambda error: list(error.path))
        if errors:
            first = errors[0]
            location = "/".join(str(part) for part in first.path)
            raise OutputError(f"{document_id}: {location}: {first.message}")


def check_ids(records: list[dict[str, Any]], expected_ids: list[str]) -> None:
    """Every expected id exactly once and nothing else."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        document_id = record[ID_FIELD]
        if document_id in seen:
            duplicates.add(document_id)
        seen.add(document_id)

    problems = []
    missing = set(expected_ids) - seen
    unknown = seen - set(expected_ids)
    if missing:
        problems.append(f"{len(missing)} missing ids: {sorted(missing)}")
    if duplicates:
        problems.append(f"{len(duplicates)} duplicate ids: {sorted(duplicates)}")
    if unknown:
        problems.append(f"{len(unknown)} unknown ids: {sorted(unknown)}")
    if problems:
        raise OutputError("; ".join(problems))


def write_predictions(
    records: list[dict[str, Any]],
    expected_ids: list[str],
    output_dir: Path,
    schema_paths: list[Path],
) -> Path:
    """Validate all records, write predictions.jsonl and zip it. Returns the zip path."""
    validators = build_validators(schema_paths)
    cleaned_records = [drop_unscored_subfields(record) for record in records]
    for record in cleaned_records:
        validate_record(record, validators)
    check_ids(cleaned_records, expected_ids)

    predictions_path = output_dir / PREDICTIONS_FILENAME
    write_jsonl(cleaned_records, predictions_path)
    # Read the file back with the strict reader to be sure it parses line by line.
    if len(read_jsonl(predictions_path)) != len(cleaned_records):
        raise OutputError(f"{predictions_path}: line count differs from the number of records")

    zip_path = output_dir / ZIP_FILENAME
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        # arcname without folders: CodaBench expects the file at the root of the zip.
        archive.write(predictions_path, arcname=PREDICTIONS_FILENAME)
    return zip_path
