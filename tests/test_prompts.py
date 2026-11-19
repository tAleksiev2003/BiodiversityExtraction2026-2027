"""Prompt v3 / chunk v2 (rejected, run with prompt_v3.yaml) and the ecosystem_service prefix rule (invented data only)."""
import copy
import json
import re
import sys
from pathlib import Path

import pytest

from llm_cache import cache_key
from prompts import ExtractionPrompt
from settings import REPO_ROOT, load_settings

sys.path.insert(0, str(REPO_ROOT / "src" / "scripts"))
from map_service_terms_by_prefix import (  # noqa: E402
    check_pairs, code_prefix, derive_prefix_rule, fill_by_prefix)

PROMPT_V2_EXPERIMENT = REPO_ROOT / "config" / "experiments" / "prompt_v2.yaml"
PROMPT_V3_EXPERIMENT = REPO_ROOT / "config" / "experiments" / "prompt_v3.yaml"
QUESTION_LINE = "Question of the database: "
# Cache keys of the v2 prompts, computed with the code BEFORE prompt v3 existed (2026-11-19),
# for model "m", temperature 0, the invented label files of conftest.py and the call below.
# If they change, a run with prompt_v2.yaml no longer comes from the cache.
V2_CACHE_KEYS = {
    "prompt_extraction": "ec551026be7d023c6ecd9122daaacb77ac1bb8c7bcab71b0deffcf6e06a595b9",
    "prompt_extraction_chunked": "12698f19504aa04e1bcd01d5b8c67b5bded928e5d14d55f816cf5122fb96ac99",
}
PROMPT_KEYS = tuple(V2_CACHE_KEYS)


def filled(settings, prompt_key: str, text: str = "Invented text.") -> str:
    prompt = ExtractionPrompt.from_settings(settings, settings["paths"][prompt_key])
    system, user = prompt.messages("DOC00001", "1/1", text)
    return system["content"] + "\n" + user["content"]


def with_v3_prompts(settings) -> dict:
    """The given settings with the prompt pair of prompt_v3.yaml (v3 is not the default)."""
    v3 = load_settings(PROMPT_V3_EXPERIMENT)["paths"]
    changed = copy.deepcopy(settings)
    for key in PROMPT_KEYS:
        changed["paths"][key] = v3[key]
    return changed


@pytest.fixture
def v3_settings(invented_settings):
    return with_v3_prompts(invented_settings)


def schema_questions(settings) -> dict[str, str]:
    """Read the descriptions straight from the schema JSON, independent of labels.py."""
    questions = {}
    for key in ("schema_core", "schema_biodiversity"):
        schema = json.loads(settings["paths"][key].read_text(encoding="utf-8"))
        for field, node in schema["properties"].items():
            questions[field] = node["description"].strip()
    return questions


# --- prompt v3 and chunk v2 ---------------------------------------------------------------

def test_defaults_are_v2_and_chunk_v1_and_v3_is_an_experiment(settings):
    # v3 was worse than v2 on dev (2026-11-19), so the defaults went back to v2 / chunk_v1.
    assert settings["paths"]["prompt_extraction"].name == "extraction_v2.txt"
    assert settings["paths"]["prompt_extraction_chunked"].name == "extraction_chunk_v1.txt"
    v3 = load_settings(PROMPT_V3_EXPERIMENT)["paths"]
    assert v3["prompt_extraction"].name == "extraction_v3.txt"
    assert v3["prompt_extraction_chunked"].name == "extraction_chunk_v2.txt"


@pytest.mark.parametrize("prompt_key", PROMPT_KEYS)
def test_every_schema_question_is_in_the_v3_prompts(v3_settings, prompt_key):
    content = filled(v3_settings, prompt_key)
    questions = schema_questions(v3_settings)
    assert len(questions) == 6
    for field, question in questions.items():
        assert f"{QUESTION_LINE}{question}\n" in content, field
    assert content.count(QUESTION_LINE) == len(questions)


@pytest.mark.parametrize("prompt_key", PROMPT_KEYS)
def test_quote_length_in_the_v3_prompts_comes_from_the_settings(v3_settings, prompt_key):
    checker = v3_settings["verbatim_check"]
    content = filled(v3_settings, prompt_key)
    assert f"{checker['min_quote_words']} to {checker['max_quote_words']} words" in content

    changed = copy.deepcopy(v3_settings)
    changed["verbatim_check"].update(min_quote_words=7, max_quote_words=33)
    assert "7 to 33 words" in filled(changed, prompt_key)


@pytest.mark.parametrize("prompt_key", PROMPT_KEYS)
def test_v3_prompts_are_filled_and_have_no_pipeline_a_section(v3_settings, prompt_key):
    text = "An invented text with {braces}."
    content = filled(v3_settings, prompt_key, text)
    assert re.search(r"\{[a-z_]+\}", content.replace(text, "")) is None  # every placeholder filled
    assert "PIPELINE A" not in content.upper() and "LEDGER" not in content.upper()


def test_chunk_v2_is_v3_plus_rule_r9(settings):
    paths = with_v3_prompts(settings)["paths"]
    v3 = paths["prompt_extraction"].read_text(encoding="utf-8")
    chunk_v2 = paths["prompt_extraction_chunked"].read_text(encoding="utf-8")
    r9 = paths["prompt_extraction_chunked"].with_name("extraction_chunk_v1.txt").read_text(
        encoding="utf-8")
    r9 = r9[r9.index("R9  "):r9.index("\n## Reasoning steps")]
    assert chunk_v2.replace(r9, "", 1) == v3


# --- prompt_v2.yaml reproduces the old requests --------------------------------------------

def test_prompt_v2_experiment_gives_exactly_the_old_prompts(invented_settings, invented_label_files):
    old = load_settings(PROMPT_V2_EXPERIMENT)
    assert old["paths"]["prompt_extraction"].name == "extraction_v2.txt"
    assert old["paths"]["prompt_extraction_chunked"].name == "extraction_chunk_v1.txt"
    old["paths"].update(invented_label_files)
    for prompt_key, expected_key in V2_CACHE_KEYS.items():
        prompt = ExtractionPrompt.from_settings(old, old["paths"][prompt_key])
        messages = prompt.messages("DOC00001", "1/1", "Invented text.")
        assert cache_key("m", messages, 0) == expected_key, prompt_key
        assert QUESTION_LINE not in messages[1]["content"]


# --- ecosystem_service prefix rule ---------------------------------------------------------

def test_code_prefix():
    assert code_prefix("2.2.1.5 Invented") == "2"
    assert code_prefix("1.3.X.X Invented (with dots. inside)") == "1"


@pytest.fixture
def service_values(specs):
    return specs["ecosystem_service.term"].allowed_values, specs["ecosystem_service.category"].allowed_values


def test_prefix_rule_fills_missing_terms(service_values):
    terms, categories = service_values
    category_of_prefix = {"1": categories[0], "2": categories[1], "3": categories[2], "4": categories[0]}
    complete = {term: category_of_prefix[code_prefix(term)] for term in terms}
    left_out = [terms[0], terms[-1]]
    mapped = {term: category for term, category in complete.items() if term not in left_out}

    rule = derive_prefix_rule(mapped)
    assert rule == category_of_prefix
    assert fill_by_prefix(terms, mapped, rule) == {term: complete[term] for term in left_out}


def test_prefix_rule_refuses_a_prefix_with_two_categories(service_values):
    terms, categories = service_values
    ones = [term for term in terms if code_prefix(term) == "1"]
    with pytest.raises(ValueError):
        derive_prefix_rule({ones[0]: categories[0], ones[1]: categories[1]})


def test_check_pairs_reports_breaking_reference_pairs(service_values):
    terms, categories = service_values
    one, two = (next(term for term in terms if code_prefix(term) == prefix) for prefix in "12")
    rule = {"1": categories[0], "2": categories[1]}
    pairs = [(categories[0], one), (categories[0], one), (categories[1], two), (categories[2], two)]
    follows, breaks = check_pairs(pairs, rule)
    assert follows == {"1": 2, "2": 1}
    assert breaks == {(two, categories[2]): 1}
