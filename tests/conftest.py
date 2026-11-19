"""Shared test fixtures.

All documents here are invented. Label values are taken from the schema enums by
position (e.g. the first habitat), so no label string is typed into the tests.
"""
import copy
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer

from data import ID_FIELD, write_jsonl
from few_shot_examples import FewShotExamples
from labels import FieldSpec, load_field_specs
from retrieval import neighbours_path
from settings import REPO_ROOT, deep_merge, load_settings


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


INVENTED_VARIABLE = "Erfundene Variable"


@pytest.fixture
def invented_label_files(tmp_path: Path, specs: dict[str, FieldSpec]) -> dict[str, Path]:
    """Invented stand-ins for the generated files in data/labels/ (derived from course data).

    term_to_category: ecosystem_type terms go round-robin to the categories; every
    ecosystem_service term goes to the first category, except the last term, which is
    left out on purpose (a term without an entry, dropped by check C2).
    """
    def round_robin(field: str) -> dict[str, str]:
        categories = specs[f"{field}.category"].allowed_values
        terms = specs[f"{field}.term"].allowed_values
        return {term: categories[index % len(categories)] for index, term in enumerate(terms)}

    service_terms = specs["ecosystem_service.term"].allowed_values
    service_category = specs["ecosystem_service.category"].allowed_values[0]
    term_to_category = {
        "ecosystem_type": round_robin("ecosystem_type"),
        "ecosystem_service": {term: service_category for term in service_terms[:-1]},
    }
    paths = {
        "term_to_category": tmp_path / "labels" / "term_to_category.json",
        "variable_vocabulary": tmp_path / "labels" / "variable_vocabulary.json",
    }
    paths["term_to_category"].parent.mkdir(parents=True, exist_ok=True)
    paths["term_to_category"].write_text(json.dumps(term_to_category, ensure_ascii=False), encoding="utf-8")
    paths["variable_vocabulary"].write_text(
        json.dumps([{"value": INVENTED_VARIABLE, "count": 3}], ensure_ascii=False), encoding="utf-8")
    return paths


INVENTED_TRAIN_IDS = ("TRAIN0001", "TRAIN0002", "TRAIN0003", "TRAIN0004")
INVENTED_DEV_IDS = ("DOC00001", "DOC00002", "DOC00003", "DOC00004")


def write_invented_neighbours(settings: dict[str, Any], tmp_path: Path, enum: Callable[[str, int], str]) -> None:
    """Invented train references and neighbours_dev.json, as src/retrieval.py writes it, so
    prompt v4 (the default since 2026-11-19) can be filled. Every dev id gets the first
    `retrieval.k` train papers; method and k match the settings."""
    few_shot, retrieval = settings["few_shot"], settings["retrieval"]
    references = [{ID_FIELD: train_id,
                   few_shot["title_field"]: f"Invented training paper {train_id}",
                   few_shot["type_field"]: "article",
                   "habitat": [enum("habitat", index)],
                   "taxa": [{"species_group": enum("taxa.species_group", index),
                             "scientific_name": None, "german_name": f"Erfundene Gruppe {index}"}]}
                  for index, train_id in enumerate(INVENTED_TRAIN_IDS)]
    settings["paths"]["references_train"] = tmp_path / "references" / "train.jsonl"
    settings["paths"]["processed"] = tmp_path / "processed"
    write_jsonl(references, settings["paths"]["references_train"])
    neighbours = [{"id": train_id, "similarity": 0.5} for train_id in INVENTED_TRAIN_IDS[:retrieval["k"]]]
    content = {"split": "dev", "method": retrieval["method"], "k": retrieval["k"], "without_text": [],
               "neighbours": {document_id: neighbours for document_id in INVENTED_DEV_IDS}}
    path = neighbours_path(settings, "dev")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content), encoding="utf-8")


@pytest.fixture
def invented_settings(
    settings: dict[str, Any], tmp_path: Path, invented_label_files: dict[str, Path],
    enum: Callable[[str, int], str],
) -> dict[str, Any]:
    """The real settings, but every path that holds data points into a temporary folder
    (including invented training examples for prompt v4)."""
    copied = copy.deepcopy(settings)
    copied["paths"].update(invented_label_files)
    copied["paths"]["cache"] = tmp_path / "cache"
    copied["paths"]["docling"] = tmp_path / "docling"
    copied["paths"]["runs"] = tmp_path / "runs"
    # The level prompt is on by default (2026-11-19). The invented fake APIs answer only the
    # main prompt, so the tests of the main calls run without it; test_field_prompts.py
    # switches it on and checks the default.
    copied["field_prompts"] = {field: None for field in copied["field_prompts"]}
    write_invented_neighbours(copied, tmp_path, enum)
    return copied


@pytest.fixture
def invented_examples(invented_settings: dict[str, Any]) -> FewShotExamples:
    """The training examples of the invented dev documents (what run_pipeline.py loads)."""
    return FewShotExamples.from_settings(invented_settings, "dev")


@pytest.fixture
def apply_experiment() -> Callable[[dict[str, Any], str], dict[str, Any]]:
    """apply_experiment(settings, "prompt_v2.yaml") -> a copy of the settings with that
    experiment file merged on top, as load_settings does, keeping the invented data paths."""
    def apply(settings: dict[str, Any], experiment_name: str) -> dict[str, Any]:
        path = REPO_ROOT / "config" / "experiments" / experiment_name
        override = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        experiment_paths = override.pop("paths", {})
        merged = deep_merge(copy.deepcopy(settings), override)
        merged["paths"] = {**settings["paths"],
                           **{name: (REPO_ROOT / value).resolve() for name, value in experiment_paths.items()}}
        return merged
    return apply
