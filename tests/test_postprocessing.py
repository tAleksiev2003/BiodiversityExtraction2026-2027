"""The post-processing rule on ecosystem services (src/postprocessing.py), invented data only.

Train references and neighbours are built in code; label values are taken from the schema
enums by position. The end-to-end checks (cache, summary.json) are in
test_full_text_pipeline.py.
"""
import copy
from typing import Any, Callable

import pytest

from data import ID_FIELD
from few_shot_examples import FewShotExamples
from postprocessing import (
    NOT_APPLIED_NO_EXAMPLES, NOT_APPLIED_SWITCHED_OFF, drop_services_without_neighbour_support)
from prompts import ECOSYSTEM_SERVICE_FIELD
from verbatim_checker import CheckedLabel

WITHOUT_SERVICES = ("TRAIN_A", "TRAIN_B", "TRAIN_C")
WITH_SERVICES = "TRAIN_S"
PAPER_NONE_SUPPORT = "PAPER001"     # all neighbours without services
PAPER_ONE_SUPPORT = "PAPER002"      # one neighbour with services
PAPER_NO_NEIGHBOURS = "PAPER003"    # no first-pages text, so no neighbours


@pytest.fixture
def examples(settings: dict[str, Any], enum: Callable[[str, int], str]) -> FewShotExamples:
    def reference(train_id: str, with_services: bool) -> dict[str, Any]:
        services = [{"category": enum("ecosystem_service.category", 0), "term": None}] if with_services else []
        return {ID_FIELD: train_id, "habitat": [enum("habitat", 0)], ECOSYSTEM_SERVICE_FIELD: services}

    references = {train_id: reference(train_id, False) for train_id in WITHOUT_SERVICES}
    references[WITH_SERVICES] = reference(WITH_SERVICES, True)
    neighbour_ids = {
        PAPER_NONE_SUPPORT: list(WITHOUT_SERVICES),
        PAPER_ONE_SUPPORT: [WITHOUT_SERVICES[0], WITH_SERVICES, WITHOUT_SERVICES[1]],
    }
    return FewShotExamples(neighbour_ids, references, settings["few_shot"])


@pytest.fixture
def labels(enum: Callable[[str, int], str]) -> dict[str, list[CheckedLabel]]:
    """Every paper has one habitat, one taxon and two services."""
    def paper_labels() -> list[CheckedLabel]:
        taxon = {"species_group": enum("taxa.species_group", 0), "scientific_name": "Inventus fictus",
                 "german_name": "Erfundene Art"}
        services = [{"category": enum("ecosystem_service.category", 0), "term": enum("ecosystem_service.term", index)}
                    for index in (0, 1)]
        return ([CheckedLabel("habitat", (enum("habitat", 0),), enum("habitat", 0), "quote")]
                + [CheckedLabel("taxa", ("Inventus fictus",), taxon, "quote")]
                + [CheckedLabel(ECOSYSTEM_SERVICE_FIELD, (entry["category"], entry["term"]), entry, "quote")
                   for entry in services])
    return {paper: paper_labels() for paper in (PAPER_NONE_SUPPORT, PAPER_ONE_SUPPORT, PAPER_NO_NEIGHBOURS)}


def fields_of(labels: list[CheckedLabel]) -> list[str]:
    return [label.field for label in labels]


def test_services_dropped_when_no_neighbour_has_any(settings, examples, labels):
    kept, result = drop_services_without_neighbour_support(labels, examples, settings)
    assert ECOSYSTEM_SERVICE_FIELD not in fields_of(kept[PAPER_NONE_SUPPORT])
    assert result.applied
    assert result.as_dict()["papers_dropped"] == 1
    assert result.as_dict()["ecosystem_service_entries_dropped"] == 2
    assert result.dropped_entries == {PAPER_NONE_SUPPORT: 2}


def test_services_kept_when_one_neighbour_has_one(settings, examples, labels):
    kept, _ = drop_services_without_neighbour_support(labels, examples, settings)
    assert kept[PAPER_ONE_SUPPORT] == labels[PAPER_ONE_SUPPORT]


def test_services_kept_for_a_paper_without_neighbours(settings, examples, labels):
    kept, _ = drop_services_without_neighbour_support(labels, examples, settings)
    assert kept[PAPER_NO_NEIGHBOURS] == labels[PAPER_NO_NEIGHBOURS]


def test_other_fields_untouched(settings, examples, labels):
    original = copy.deepcopy(labels)
    kept, _ = drop_services_without_neighbour_support(labels, examples, settings)
    others = [label for label in original[PAPER_NONE_SUPPORT] if label.field != ECOSYSTEM_SERVICE_FIELD]
    assert kept[PAPER_NONE_SUPPORT] == others
    assert labels == original  # the input is not changed


def test_setting_false_changes_nothing(settings, examples, labels):
    switched_off = copy.deepcopy(settings)
    switched_off["postprocess"]["drop_services_without_neighbour_support"] = False
    kept, result = drop_services_without_neighbour_support(labels, examples, switched_off)
    assert kept == labels
    assert not result.applied and result.reason == NOT_APPLIED_SWITCHED_OFF
    assert result.as_dict()["ecosystem_service_entries_dropped"] == 0


def test_prompt_without_examples_changes_nothing(settings, labels):
    kept, result = drop_services_without_neighbour_support(labels, None, settings)
    assert kept == labels
    assert not result.applied and result.reason == NOT_APPLIED_NO_EXAMPLES


def test_experiment_file_switches_the_rule_off(settings, apply_experiment):
    assert settings["postprocess"]["drop_services_without_neighbour_support"] is True
    experiment = apply_experiment(settings, "no_service_rule.yaml")
    assert experiment["postprocess"]["drop_services_without_neighbour_support"] is False
