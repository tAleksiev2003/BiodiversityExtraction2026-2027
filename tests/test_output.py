"""Tests for src/output.py: everything CodaBench would reject must be caught."""
import json
import zipfile

import pytest

from data import ID_FIELD
from output import (
    PREDICTIONS_FILENAME, OutputError, build_validators, check_ids,
    drop_unscored_subfields, validate_record, write_predictions,
)


@pytest.fixture
def validators(schema_paths):
    return build_validators(schema_paths)


def test_valid_records_pass(clean_predictions, validators):
    for record in clean_predictions:
        validate_record(record, validators)


@pytest.mark.parametrize("bad_record, message", [
    ({ID_FIELD: "D1", "habitat": ["not an allowed value"]}, "habitat"),
    ({ID_FIELD: "D1", "habitat": [42]}, "habitat"),
    ({ID_FIELD: "D1", "habitat": "a string instead of a list"}, "habitat"),
    ({ID_FIELD: "D1", "unknown_field": []}, "unknown fields"),
    ({"habitat": []}, ID_FIELD),
    ({ID_FIELD: 123}, ID_FIELD),
])
def test_invalid_records_are_rejected(bad_record, message, validators):
    with pytest.raises(OutputError, match=message):
        validate_record(bad_record, validators)


def test_extra_subfield_is_rejected(validators, enum):
    record = {ID_FIELD: "D1", "taxa": [{"species_group": enum("taxa.species_group"), "invented": "x"}]}
    with pytest.raises(OutputError):
        validate_record(record, validators)


def test_details_is_dropped(enum):
    record = {ID_FIELD: "D1", "ecosystem_service": [
        {"category": enum("ecosystem_service.category"), "details": "free text"}]}
    cleaned = drop_unscored_subfields(record)
    assert "details" not in cleaned["ecosystem_service"][0]
    assert "details" in record["ecosystem_service"][0]  # the input is not changed


@pytest.mark.parametrize("ids_in_records, expected_ids, message", [
    (["A"], ["A", "B"], "missing"),
    (["A", "A", "B"], ["A", "B"], "duplicate"),
    (["A", "B", "C"], ["A", "B"], "unknown"),
])
def test_id_problems_are_rejected(ids_in_records, expected_ids, message):
    with pytest.raises(OutputError, match=message):
        check_ids([{ID_FIELD: document_id} for document_id in ids_in_records], expected_ids)


def test_write_predictions_writes_one_line_per_document_and_zip(tmp_path, clean_predictions, schema_paths):
    expected_ids = [record[ID_FIELD] for record in clean_predictions]
    zip_path = write_predictions(clean_predictions, expected_ids, tmp_path, schema_paths)

    lines = (tmp_path / PREDICTIONS_FILENAME).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)[ID_FIELD] for line in lines] == expected_ids
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == [PREDICTIONS_FILENAME]


def test_write_predictions_writes_nothing_on_error(tmp_path, clean_predictions, schema_paths):
    with pytest.raises(OutputError):
        write_predictions(clean_predictions, ["OTHER"], tmp_path, schema_paths)
    assert not (tmp_path / PREDICTIONS_FILENAME).exists()
