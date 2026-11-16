"""Tests for src/baselines.py on invented records."""
from baselines import SCORED_FIELDS, most_frequent_baseline, most_frequent_values, random_baseline
from data import ID_FIELD
from output import build_validators, validate_record

IDS = ["DOC00001", "DOC00002"]


def scored(specs):
    return [specs[name] for name in SCORED_FIELDS]


def test_random_baseline_is_valid_and_reproducible(specs, schema_paths):
    first = random_baseline(IDS, scored(specs), seed=7)
    second = random_baseline(IDS, scored(specs), seed=7)
    assert first == second
    validators = build_validators(schema_paths)
    for record in first:
        validate_record(record, validators)
    assert [record[ID_FIELD] for record in first] == IDS


def test_random_baseline_fills_every_enum_field(specs):
    record = random_baseline(IDS, scored(specs), seed=7)[0]
    for spec in scored(specs):
        if spec.is_enum:
            values = record[spec.parent] if spec.subfield is None else [record[spec.parent][0][spec.subfield]]
            assert values[0] in spec.allowed_values


def test_most_frequent_picks_value_seen_in_most_documents(invented_references, specs, enum):
    values = most_frequent_values(invented_references, scored(specs))
    # habitat 0 occurs in two documents, habitat 1 in one.
    assert values["habitat"] == enum("habitat", 0)
    assert values["taxa.scientific_name"] == "Inventus fictus"
    # Subfields are chosen independently.
    assert values["ecosystem_type.category"] in (enum("ecosystem_type.category", 0), enum("ecosystem_type.category", 1))


def test_most_frequent_ignores_values_outside_the_schema(specs):
    references = {"D1": {ID_FIELD: "D1", "habitat": ["not an allowed value"]}}
    assert "habitat" not in most_frequent_values(references, scored(specs))


def test_most_frequent_baseline_is_valid(invented_references, specs, schema_paths):
    records = most_frequent_baseline(IDS, scored(specs), invented_references)
    validators = build_validators(schema_paths)
    for record in records:
        validate_record(record, validators)
