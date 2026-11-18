"""Tests for src/verbatim_checker.py (SPEC.md, section 3a) on an invented study text.

Label values come from the schema enums by position; term -> category pairs come from
the invented term_to_category.json in conftest.py (ecosystem_type: term i belongs to
category i mod 6; ecosystem_service: the last term has no entry).
"""
import dataclasses
from typing import Any, Callable

import pytest

from data import ID_FIELD
from llm_answer import ExtractedLabel
from output import build_validators, validate_record
from verbatim_checker import (
    CHECK_ALLOWED_VALUE,
    CHECK_CONSISTENT_PAIR,
    CHECK_ENOUGH_EVIDENCE,
    CHECK_PLAUSIBLE_QUOTE,
    CHECK_VERBATIM_QUOTE,
    CheckerConfig,
    LabelRules,
    build_record,
    check_answer,
    drop_statistics,
    merge_labels,
    normalise,
)

STUDY_TEXT = (
    "[page 1]\n## Methods\n"
    "We sampled ground beetles with pitfall traps along twelve hedgerows in Musterdorf.\n"
    "The traps were emptied every two weeks between April and September.\n\n"
    "[page 2]\n## Results\n"
    "Species richness of carabid beetles was highest in old hedge-\nrows with a wide herb layer.\n"
    "The “fictional” ﬁeld margins—sown in 2019—held fewer individuals."
)
SAMPLING_QUOTE = "We sampled ground beetles with pitfall traps along twelve hedgerows"
RICHNESS_QUOTE = "Species richness of carabid beetles was highest in old hedgerows"
MARGIN_QUOTE = 'The "fictional" field margins - sown in 2019 - held fewer individuals.'
TRAPS_QUOTE = "The traps were emptied every two weeks between April and September."


@pytest.fixture
def rules(invented_settings: dict[str, Any]) -> LabelRules:
    return LabelRules.from_settings(invented_settings)


@pytest.fixture
def config(invented_settings: dict[str, Any]) -> CheckerConfig:
    return CheckerConfig.from_settings(invented_settings)


def label(field: str, quote: str | None = SAMPLING_QUOTE, **values: Any) -> ExtractedLabel:
    return ExtractedLabel(field=field, quote=quote, **values)


def check(labels: list[ExtractedLabel], rules: LabelRules, config: CheckerConfig, text: str = STUDY_TEXT):
    return check_answer(labels, text, rules, config, "DOC00001", "1/1")


def only_entry(log):
    assert len(log) == 1
    return log[0]


# --- normalisation ---------------------------------------------------------------------

def test_normalise_unifies_case_quotes_dashes_ligatures_hyphenation_and_spaces() -> None:
    assert normalise("Old hedge-\nrows") == "old hedgerows"
    assert normalise("ﬁeld  margins\n\tsown") == "field margins sown"
    assert normalise("„Wald“ – ‚Feld‘") == normalise('"Wald" - \'Feld\'')
    assert normalise("Arten­vielfalt") == "artenvielfalt"


# --- C1 to C4 ---------------------------------------------------------------------------

def test_label_with_verbatim_quote_passes(rules, config, enum) -> None:
    kept, log = check([label("habitat", value=enum("habitat", 0))], rules, config)
    assert [item.entry for item in kept] == [enum("habitat", 0)]
    entry = only_entry(log)
    assert (entry.check, entry.passed, entry.score) == (CHECK_VERBATIM_QUOTE, True, 100.0)


@pytest.mark.parametrize("quote", [RICHNESS_QUOTE, MARGIN_QUOTE])
def test_quote_matches_despite_hyphenation_typography_and_ligatures(rules, config, enum, quote) -> None:
    kept, _ = check([label("biodiversity_level", quote, value=enum("biodiversity_level", 0))], rules, config)
    assert len(kept) == 1


def test_c1_value_not_in_schema(rules, config, enum) -> None:
    almost_allowed = enum("habitat", 0) + "x"
    kept, log = check([label("habitat", value=almost_allowed)], rules, config)
    assert kept == []
    assert (only_entry(log).check, only_entry(log).passed) == (CHECK_ALLOWED_VALUE, False)


@pytest.mark.parametrize("bad_label", [
    {"field": "invented_field", "value": "x"},
    {"field": "habitat"},                                   # no value
    {"field": "taxa", "scientific_name": "Carabus fictus"},  # species_group is required
    {"field": "ecosystem_type", "term": None},               # category is required
])
def test_c1_unknown_field_or_missing_value(rules, config, bad_label) -> None:
    kept, log = check([ExtractedLabel(quote=SAMPLING_QUOTE, **bad_label)], rules, config)
    assert kept == []
    assert only_entry(log).check == CHECK_ALLOWED_VALUE


def test_c1_comes_before_c3(rules, config) -> None:
    _, log = check([label("habitat", quote="too short", value="not allowed")], rules, config)
    assert only_entry(log).check == CHECK_ALLOWED_VALUE


def test_c2_term_of_another_category_is_dropped(rules, config, enum) -> None:
    wrong_pair = label("ecosystem_type", category=enum("ecosystem_type.category", 1),
                       term=enum("ecosystem_type.term", 0))
    right_pair = label("ecosystem_type", category=enum("ecosystem_type.category", 0),
                       term=enum("ecosystem_type.term", 0))
    category_only = label("ecosystem_type", category=enum("ecosystem_type.category", 2), term=None)
    kept, log = check([wrong_pair, right_pair, category_only], rules, config)
    assert [item.entry for item in kept] == [
        {"category": enum("ecosystem_type.category", 0), "term": enum("ecosystem_type.term", 0)},
        {"category": enum("ecosystem_type.category", 2), "term": None},
    ]
    assert (log[0].check, log[0].passed) == (CHECK_CONSISTENT_PAIR, False)


def test_c2_service_term_without_mapping_is_dropped_and_logged(rules, config, specs, caplog) -> None:
    unmapped_term = specs["ecosystem_service.term"].allowed_values[-1]
    service = label("ecosystem_service", category=specs["ecosystem_service.category"].allowed_values[0],
                    term=unmapped_term)
    with caplog.at_level("INFO"):
        kept, log = check([service], rules, config)
    assert kept == []
    entry = only_entry(log)
    assert entry.check == CHECK_CONSISTENT_PAIR
    assert "no entry in term_to_category.json" in entry.reason
    assert unmapped_term in caplog.text


@pytest.mark.parametrize("quote, reason", [
    (None, "no quote"),
    ("   ", "no quote"),
    ("ground beetles with traps", "fewer than"),            # 4 words < min_quote_words (5)
    (" ".join(["word"] * 41), "more than"),                 # > max_quote_words (40)
    ("We sampled ground beetles ... twelve hedgerows", "ellipsis"),
    ("We sampled ground beetles … twelve hedgerows", "ellipsis"),
])
def test_c3_implausible_quotes(rules, config, enum, quote, reason) -> None:
    kept, log = check([label("habitat", quote, value=enum("habitat", 0))], rules, config)
    assert kept == []
    assert only_entry(log).check == CHECK_PLAUSIBLE_QUOTE
    assert reason in only_entry(log).reason


def test_c4_invented_quote_is_dropped_with_its_score(rules, config, enum) -> None:
    invented = "We counted wild bees on flowering meadows near the river"
    kept, log = check([label("habitat", invented, value=enum("habitat", 0))], rules, config)
    assert kept == []
    entry = only_entry(log)
    assert entry.check == CHECK_VERBATIM_QUOTE
    assert entry.score is not None and entry.score < config.quote_match_threshold


def test_c4_small_copy_error_is_tolerated(rules, config, enum) -> None:
    one_typo = SAMPLING_QUOTE.replace("pitfall", "pitfal")
    kept, log = check([label("habitat", one_typo, value=enum("habitat", 0))], rules, config)
    assert len(kept) == 1
    assert config.quote_match_threshold <= only_entry(log).score < 100


def test_c4_searches_only_the_text_sent_to_the_model(rules, config, enum) -> None:
    # A quote from another part of the document does not count for this call.
    other_part = "[page 3]\n## Discussion\nOur results agree with earlier hedgerow studies in Musterland."
    kept, _ = check([label("habitat", SAMPLING_QUOTE, value=enum("habitat", 0))], rules, config, text=other_part)
    assert kept == []


def test_completed_taxa_names_and_german_variables_need_only_the_quote(rules, config, enum) -> None:
    taxon = label("taxa", RICHNESS_QUOTE, species_group=enum("taxa.species_group", 0),
                  scientific_name="Carabidae", german_name="Laufkäfer", name_source="completed")
    variable = label("biodiversity_variable", RICHNESS_QUOTE, value=" Artenzahl der Laufkäfer ")
    kept, _ = check([taxon, variable], rules, config)
    assert [item.entry for item in kept] == [
        {"scientific_name": "Carabidae", "german_name": "Laufkäfer",
         "species_group": enum("taxa.species_group", 0)},
        "Artenzahl der Laufkäfer",
    ]


# --- C5 and merging ---------------------------------------------------------------------

def test_c5_counts_different_quotes_over_all_calls(rules, config, enum) -> None:
    strict = dataclasses.replace(config, min_verified_quotes=2)
    habitat = enum("habitat", 0)
    level = enum("biodiversity_level", 0)
    first_call, _ = check([label("habitat", SAMPLING_QUOTE, value=habitat),
                           label("biodiversity_level", RICHNESS_QUOTE, value=level)], rules, strict)
    second_call, _ = check([label("habitat", TRAPS_QUOTE, value=habitat),
                            label("biodiversity_level", RICHNESS_QUOTE, value=level)], rules, strict)
    kept, log = merge_labels(first_call + second_call, strict, "DOC00001")
    assert [item.entry for item in kept] == [habitat]  # the same quote twice counts once
    assert [entry.check for entry in log] == [CHECK_ENOUGH_EVIDENCE, CHECK_ENOUGH_EVIDENCE]
    assert all(entry.field == "biodiversity_level" and not entry.passed for entry in log)


def test_merge_keeps_one_label_per_value_in_first_order(rules, config, enum) -> None:
    labels, _ = check([label("habitat", TRAPS_QUOTE, value=enum("habitat", 1)),
                       label("habitat", SAMPLING_QUOTE, value=enum("habitat", 0)),
                       label("habitat", RICHNESS_QUOTE, value=enum("habitat", 1))], rules, config)
    kept, log = merge_labels(labels, config, "DOC00001")
    assert [item.entry for item in kept] == [enum("habitat", 1), enum("habitat", 0)]
    assert log == []


# --- checker switched off (ablation) -------------------------------------------------------

def test_disabled_checker_keeps_only_c1(rules, config, enum) -> None:
    off = dataclasses.replace(config, enabled=False, min_verified_quotes=2)
    wrong_pair = label("ecosystem_type", None, category=enum("ecosystem_type.category", 1),
                       term=enum("ecosystem_type.term", 0))
    labels = [label("habitat", None, value=enum("habitat", 0)),           # no quote: kept
              label("habitat", "invented", value="not allowed"),          # C1: dropped
              wrong_pair]                                                 # C2 skipped: kept
    kept, log = check(labels, rules, off)
    merged, merge_log = merge_labels(kept, off, "DOC00001")
    assert len(merged) == 2
    assert merge_log == []  # no C5
    assert [(entry.check, entry.passed) for entry in log] == [
        (CHECK_ALLOWED_VALUE, True), (CHECK_ALLOWED_VALUE, False), (CHECK_ALLOWED_VALUE, True)]


# --- record and statistics ------------------------------------------------------------------

def test_record_is_valid_for_codabench(rules, config, enum, schema_paths) -> None:
    labels, _ = check([
        label("habitat", value=enum("habitat", 0)),
        label("taxa", species_group=enum("taxa.species_group", 1), scientific_name=None, german_name=None),
        label("ecosystem_service", category=enum("ecosystem_service.category", 0),
              term=enum("ecosystem_service.term", 0)),
        label("biodiversity_variable", value="Abundanz"),
    ], rules, config)
    kept, _ = merge_labels(labels, config, "DOC00001")
    record = build_record(ID_FIELD, "DOC00001", kept, rules)
    validate_record(record, build_validators(schema_paths))
    assert record["ecosystem_type"] == []  # every field is present, empty if nothing was kept
    assert record["taxa"] == [{"scientific_name": None, "german_name": None,
                               "species_group": enum("taxa.species_group", 1)}]


def test_drop_statistics_per_check_and_field(rules, config, enum) -> None:
    _, log = check([label("habitat", value=enum("habitat", 0)),
                    label("habitat", "too short", value=enum("habitat", 1)),
                    label("habitat", value="not allowed"),
                    label("biodiversity_level", value=enum("biodiversity_level", 0))], rules, config)
    statistics = drop_statistics([entry.as_dict() for entry in log])
    habitat = statistics["per_field"]["habitat"]
    assert habitat["checked"] == 3
    assert habitat["kept"] == 1
    assert habitat["dropped"][CHECK_ALLOWED_VALUE] == 1
    assert habitat["dropped_share"][CHECK_PLAUSIBLE_QUOTE] == pytest.approx(1 / 3)
    assert statistics["all_fields"]["checked"] == 4
    assert statistics["all_fields"]["dropped_share"][CHECK_ALLOWED_VALUE] == pytest.approx(1 / 4)
