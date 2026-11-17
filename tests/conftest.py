"""Shared test fixtures.

All documents here are invented. Label values are taken from the schema enums by
position (e.g. the first habitat), so no label string is typed into the tests.
"""
import copy
from pathlib import Path
from typing import Any, Callable

import pytest
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer

from data import ID_FIELD
from labels import FieldSpec, load_field_specs
from settings import load_settings


@pytest.fixture(scope="session")
def settings() -> dict[str, Any]:
    return load_settings()


@pytest.fixture(scope="session")
def schema_paths(settings: dict[str, Any]) -> list[Path]:
    return [settings["paths"]["schema_core"], settings["paths"]["schema_biodiversity"]]


@pytest.fixture(scope="session")
def specs(schema_paths: list[Path]) -> dict[str, FieldSpec]:
    return load_field_specs(schema_paths)


@pytest.fixture(scope="session")
def enum(specs: dict[str, FieldSpec]) -> Callable[[str, int], str]:
    """enum("habitat", 1) -> the second allowed habitat value."""
    def pick(field: str, index: int = 0) -> str:
        return specs[field].allowed_values[index]
    return pick


@pytest.fixture
def invented_references(enum: Callable[[str, int], str]) -> dict[str, dict[str, Any]]:
    """Three invented reference records covering list fields, subfields and empty fields."""
    records = [
        {
            ID_FIELD: "DOC00001",
            "habitat": [enum("habitat", 0), enum("habitat", 1)],
            "biodiversity_level": [enum("biodiversity_level", 0)],
            "ecosystem_type": [{"category": enum("ecosystem_type.category", 0),
                                "term": enum("ecosystem_type.term", 0), "description": ""}],
            "taxa": [{"species_group": enum("taxa.species_group", 0),
                      "scientific_name": "Inventus fictus", "german_name": "Erfundene Art",
                      "collective_term": False}],
            "biodiversity_variable": ["Erfundene Variable"],
            "ecosystem_service": [{"category": enum("ecosystem_service.category", 0),
                                   "term": enum("ecosystem_service.term", 0)}],
        },
        {
            ID_FIELD: "DOC00002",
            "habitat": [enum("habitat", 0)],
            "ecosystem_type": [{"category": enum("ecosystem_type.category", 1), "term": None}],
            "taxa": [{"species_group": enum("taxa.species_group", 1),
                      "scientific_name": None, "german_name": None}],
        },
        {ID_FIELD: "DOC00003"},  # a study that covers none of the fields
    ]
    return {record[ID_FIELD]: record for record in records}


@pytest.fixture
def clean_predictions(invented_references: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The references turned into valid predictions (extra reference-only keys removed)."""
    predictions = []
    for record in copy.deepcopy(list(invented_references.values())):
        for entry in record.get("ecosystem_type", []):
            entry.pop("description", None)
        for entry in record.get("taxa", []):
            entry.pop("collective_term", None)
        predictions.append(record)
    return predictions


class WordTokenizer(BaseTokenizer):
    """Counts whitespace-separated words, so chunk sizes in tests are easy to predict.

    The real tokenizer (a HuggingFace download) is only used in test_pdf_to_text.py.
    """

    max_tokens: int

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self) -> Callable[[str], int]:
        return self.count_tokens  # semchunk accepts a token-counting function


@pytest.fixture
def word_tokenizer() -> Callable[[int], WordTokenizer]:
    """word_tokenizer(50) -> a tokenizer with max_tokens = 50."""
    return lambda max_tokens: WordTokenizer(max_tokens=max_tokens)
