"""The post-processing rules on ecosystem services and ecosystem_type (src/postprocessing.py), invented data only.

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
    NOT_APPLIED_ECOSYSTEM_TYPE_OFF, NOT_APPLIED_NO_EXAMPLES, NOT_APPLIED_SWITCHED_OFF, RULE_ECOSYSTEM_TYPE,
    RULE_MIN_CONFIDENCE, drop_ecosystem_type_for_land_only_habitats, drop_services_without_neighbour_support,
    land_only_habitats, postprocess)
from prompts import ECOSYSTEM_SERVICE_FIELD, ECOSYSTEM_TYPE_FIELD
from settings import REPO_ROOT, load_settings
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


# --- the ecosystem_type rule for papers with land-only habitats --------------------------------

PAPER_LAND_ONLY = "PAPER101"     # only land habitats -> ecosystem_type dropped
PAPER_MIXED = "PAPER102"         # one land habitat, one other -> kept
PAPER_NO_HABITAT = "PAPER103"    # no habitat -> kept
PAPER_OTHER_ONLY = "PAPER104"    # only a habitat outside the land set -> kept


def land_and_other_habitats(settings: dict[str, Any], specs) -> tuple[list[str], list[str]]:
    """The configured land habitats and the remaining schema habitats, without typing a name."""
    land = list(settings["postprocess"]["land_only_habitats"])
    other = [habitat for habitat in specs["habitat"].allowed_values if habitat not in land]
    return land, other


def habitat_label(value: str) -> CheckedLabel:
    return CheckedLabel("habitat", (value,), value, "quote")


@pytest.fixture
def rule_on(settings: dict[str, Any]) -> dict[str, Any]:
    switched_on = copy.deepcopy(settings)
    switched_on["postprocess"]["drop_ecosystem_type_for_land_only_habitats"] = True
    return switched_on


@pytest.fixture
def habitat_papers(settings, specs, enum) -> dict[str, list[CheckedLabel]]:
    """Every paper has two ecosystem_type labels and one taxon; the habitats differ."""
    land, other = land_and_other_habitats(settings, specs)
    ecosystem_types = [{"category": enum("ecosystem_type.category", 0), "term": None},
                       {"category": enum("ecosystem_type.category", 1), "term": None}]
    taxon = {"species_group": enum("taxa.species_group", 0), "scientific_name": None, "german_name": "Erfundene Art"}

    def paper(habitats: list[str]) -> list[CheckedLabel]:
        return ([habitat_label(habitat) for habitat in habitats]
                + [CheckedLabel(ECOSYSTEM_TYPE_FIELD, (entry["category"], entry["term"]), entry, "quote")
                   for entry in ecosystem_types]
                + [CheckedLabel("taxa", ("Erfundene Art",), taxon, "quote")])

    return {PAPER_LAND_ONLY: paper(land), PAPER_MIXED: paper([land[0], other[0]]),
            PAPER_NO_HABITAT: paper([]), PAPER_OTHER_ONLY: paper([other[0]])}


def test_land_habitats_are_schema_values(settings, specs):
    land, other = land_and_other_habitats(settings, specs)
    assert land and other  # the rule would be meaningless otherwise
    assert set(land) <= set(specs["habitat"].allowed_values)
    assert land_only_habitats(settings) == frozenset(land)


def test_ecosystem_type_dropped_for_land_only_habitats(rule_on, habitat_papers):
    kept, result = drop_ecosystem_type_for_land_only_habitats(habitat_papers, rule_on)
    assert ECOSYSTEM_TYPE_FIELD not in fields_of(kept[PAPER_LAND_ONLY])
    assert fields_of(kept[PAPER_LAND_ONLY]) == [field for field in fields_of(habitat_papers[PAPER_LAND_ONLY])
                                                if field != ECOSYSTEM_TYPE_FIELD]
    assert result.applied
    assert result.dropped_entries == {PAPER_LAND_ONLY: 2}
    assert result.as_dict()["papers_dropped"] == 1
    assert result.as_dict()["ecosystem_type_entries_dropped"] == 2


@pytest.mark.parametrize("paper", [PAPER_MIXED, PAPER_NO_HABITAT, PAPER_OTHER_ONLY])
def test_ecosystem_type_kept_otherwise(rule_on, habitat_papers, paper):
    kept, _ = drop_ecosystem_type_for_land_only_habitats(habitat_papers, rule_on)
    assert kept[paper] == habitat_papers[paper]


def test_ecosystem_type_rule_does_not_change_its_input(rule_on, habitat_papers):
    original = copy.deepcopy(habitat_papers)
    drop_ecosystem_type_for_land_only_habitats(habitat_papers, rule_on)
    assert habitat_papers == original


def test_ecosystem_type_rule_is_on_by_default(settings, habitat_papers):
    # Default since 2026-11-19 (SPEC.md, section 13); not robust (bootstrap 77 %).
    assert settings["postprocess"]["drop_ecosystem_type_for_land_only_habitats"] is True
    kept, result = drop_ecosystem_type_for_land_only_habitats(habitat_papers, settings)
    assert result.applied and result.dropped_entries == {PAPER_LAND_ONLY: 2}
    assert ECOSYSTEM_TYPE_FIELD not in fields_of(kept[PAPER_LAND_ONLY])


def test_ecosystem_type_rule_switched_off_changes_nothing(settings, habitat_papers):
    switched_off = copy.deepcopy(settings)
    switched_off["postprocess"]["drop_ecosystem_type_for_land_only_habitats"] = False
    kept, result = drop_ecosystem_type_for_land_only_habitats(habitat_papers, switched_off)
    assert kept == habitat_papers
    assert not result.applied and result.reason == NOT_APPLIED_ECOSYSTEM_TYPE_OFF


def test_run_without_the_setting_keeps_the_rule_off(settings, habitat_papers):
    # The copied config.yaml of runs made before the rule existed has no such entry
    # (src/pipelines/verify.py rebuilds those runs), so a missing entry means off.
    old_run = copy.deepcopy(settings)
    del old_run["postprocess"]["drop_ecosystem_type_for_land_only_habitats"]
    kept, result = drop_ecosystem_type_for_land_only_habitats(habitat_papers, old_run)
    assert kept == habitat_papers and not result.applied


def test_unknown_land_habitat_is_an_error(rule_on, habitat_papers):
    rule_on["postprocess"]["land_only_habitats"] = ["Kein Habitat"]
    with pytest.raises(ValueError):
        drop_ecosystem_type_for_land_only_habitats(habitat_papers, rule_on)


def test_ecosystem_type_rule_runs_after_min_confidence_and_is_in_the_summary(rule_on, habitat_papers, specs):
    # A low-confidence habitat outside the land set is dropped first, so the paper counts as land-only.
    _, other = land_and_other_habitats(rule_on, specs)
    paper = habitat_papers[PAPER_MIXED]
    paper[1] = CheckedLabel("habitat", (other[0],), other[0], "quote", confidence=0.1)
    rule_on["postprocess"]["min_confidence"] = {"habitat": 0.5}
    result = postprocess({PAPER_MIXED: paper}, None, rule_on)
    assert ECOSYSTEM_TYPE_FIELD not in fields_of(result.labels_by_document[PAPER_MIXED])
    dropped = [label for label in paper if label.field == ECOSYSTEM_TYPE_FIELD]
    assert all(result.rule_that_dropped(PAPER_MIXED, label) == RULE_ECOSYSTEM_TYPE for label in dropped)
    assert result.rule_that_dropped(PAPER_MIXED, paper[1]) == RULE_MIN_CONFIDENCE
    assert result.summary()[RULE_ECOSYSTEM_TYPE]["ecosystem_type_entries_dropped"] == 2


@pytest.mark.parametrize("experiment, prompt_name", [("eco_type_rule.yaml", "extraction_v4.txt"),
                                                     ("prompt_v6_eco_type_rule.yaml", "extraction_v6.txt")])
def test_experiment_files_switch_the_ecosystem_type_rule_on(settings, apply_experiment, experiment, prompt_name):
    merged = apply_experiment(settings, experiment)
    assert merged["postprocess"]["drop_ecosystem_type_for_land_only_habitats"] is True
    assert merged["paths"]["prompt_extraction"].name == prompt_name
    assert merged["postprocess"]["drop_services_without_neighbour_support"] is True
    assert merged["field_prompts"] == settings["field_prompts"]


# --- experiment files and the new default (2026-11-19) ----------------------------------------

EXPERIMENTS_DIR = REPO_ROOT / "config" / "experiments"
RULE_KEY = "drop_ecosystem_type_for_land_only_habitats"
# Files whose purpose is this rule, or that do not run pipeline F's post-processing with their
# own settings (src/run_verify.py uses the copied settings of its source run).
WITH_THE_RULE = {"eco_type_rule.yaml", "prompt_v6_eco_type_rule.yaml"}
NOT_PINNED = WITH_THE_RULE | {"no_eco_type_rule.yaml", "verify_without_taxa.yaml"}
EARLIER_EXPERIMENTS = sorted(path.name for path in EXPERIMENTS_DIR.glob("*.yaml") if path.name not in NOT_PINNED)


def without_rule(settings: dict[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(settings)
    changed["postprocess"][RULE_KEY] = False
    return changed


def test_no_eco_type_rule_equals_the_defaults_without_the_rule(settings):
    assert load_settings(EXPERIMENTS_DIR / "no_eco_type_rule.yaml") == without_rule(settings)


def test_eco_type_rule_experiment_equals_the_defaults(settings):
    assert load_settings(EXPERIMENTS_DIR / "eco_type_rule.yaml") == settings


def test_prompt_v6_eco_type_rule_is_prompt_v6_with_the_rule(settings):
    v6 = load_settings(EXPERIMENTS_DIR / "prompt_v6.yaml")
    v6_with_rule = load_settings(EXPERIMENTS_DIR / "prompt_v6_eco_type_rule.yaml")
    assert v6["postprocess"][RULE_KEY] is False and v6_with_rule["postprocess"][RULE_KEY] is True
    assert v6_with_rule["paths"]["prompt_extraction"].name == "extraction_v6.txt"
    v6["postprocess"][RULE_KEY] = True
    assert v6 == v6_with_rule


@pytest.mark.parametrize("experiment", EARLIER_EXPERIMENTS)
def test_earlier_experiments_keep_the_eco_type_rule_off(experiment):
    # Made before the rule became the default: switched off, as when they were run.
    assert load_settings(EXPERIMENTS_DIR / experiment)["postprocess"][RULE_KEY] is False


def test_no_service_rule_switches_only_the_service_rule_off(settings):
    pinned = load_settings(EXPERIMENTS_DIR / "no_service_rule.yaml")
    assert pinned["postprocess"]["drop_services_without_neighbour_support"] is False
    assert pinned["postprocess"]["land_only_habitats"] == settings["postprocess"]["land_only_habitats"]
