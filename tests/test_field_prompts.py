"""Field prompts in pipeline F (decomposed prompting): end to end with invented documents.

Reuses the invented documents and the fake API of test_full_text_pipeline.py. The fake
answers the real level prompt (config/prompts/level_v1.txt) with invented levels:
DOC00001 and DOC00002 get new levels; DOC00003's level call fails, so it keeps the level
of its main call. No test reaches the real API or reads course data.
"""
import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from data import ID_FIELD, read_jsonl
from label_file import LABEL_FILE_NAME, SOURCE_FIELD_PROMPT, SOURCE_MAIN
from output import PREDICTIONS_FILENAME
from pipelines.full_text import field_prompt_paths
from prompts import ExtractionPrompt, PER_CALL_PLACEHOLDERS
from settings import REPO_ROOT
from test_full_text_pipeline import (  # noqa: F401  (pipeline_settings is a fixture)
    BIRDS, EARTHWORMS, RICHNESS, SAMPLING, InventedLlm, pipeline_settings, run)

LEVEL_PROMPT = REPO_ROOT / "config" / "prompts" / "level_v1.txt"
LEVEL_FIELD = "biodiversity_level"
LEVEL_SYSTEM_MARKER = "In this task you decide only one field"
LEVEL_PATTERN = re.compile(r"part (\d+)/(\d+) of document (\w+)")

FAIL_WITH_API_ERROR = "api_error"
FAIL_WITH_INVALID_JSON = "invalid_json"


class LevelLlm(InventedLlm):
    """Main answers as in test_full_text_pipeline, plus one verified level in the main
    answers of DOC00001 and DOC00003; answers the level prompt per document."""

    def __init__(self, enum, unmapped_service_term: str, level_failure: dict[str, str] | None = None) -> None:
        super().__init__(enum, unmapped_service_term)
        self.level_failure = level_failure if level_failure is not None else {"DOC00003": FAIL_WITH_API_ERROR}
        self.level_prompts: list[list[dict[str, str]]] = []

    def answer(self, messages: list[dict[str, str]]) -> str:
        content = super().answer(messages)
        document_id = re.search(r"chunk \d+/\d+ of document (\w+)", messages[1]["content"]).group(1)
        main_levels = {"DOC00001": (3, SAMPLING), "DOC00003": (4, BIRDS)}
        if document_id not in main_levels or not content.startswith("{"):
            return content
        index, quote = main_levels[document_id]
        answer = json.loads(content)
        answer["labels"].append({"field": LEVEL_FIELD, "value": self.enum(LEVEL_FIELD, index), "quote": quote})
        return json.dumps(answer)

    def level_answer(self, document_id: str) -> str:
        if self.level_failure.get(document_id) == FAIL_WITH_INVALID_JSON:
            return "no json here"
        labels: list[dict[str, Any]] = []
        if document_id == "DOC00001":
            labels = [{"field": LEVEL_FIELD, "value": self.enum(LEVEL_FIELD, 1), "confidence": 0.8, "quote": RICHNESS},
                      {"field": "habitat", "value": self.enum("habitat", 5), "quote": SAMPLING}]  # ignored
        elif document_id == "DOC00002":
            # Every part claims it; only the part with the quote verifies it.
            labels = [{"field": LEVEL_FIELD, "value": self.enum(LEVEL_FIELD, 2), "confidence": "high",
                       "quote": EARTHWORMS}]
        return json.dumps({"reasoning": "S1: invented", "labels": labels})

    def handle(self, request: httpx.Request) -> httpx.Response:
        messages = json.loads(request.content)["messages"]
        if LEVEL_SYSTEM_MARKER not in messages[0]["content"]:
            return super().handle(request)
        with self._lock:
            self.level_prompts.append(messages)
        document_id = LEVEL_PATTERN.search(messages[1]["content"]).group(3)
        if self.level_failure.get(document_id) == FAIL_WITH_API_ERROR:
            return httpx.Response(400, json={"error": {"message": "invented error"}})
        body = {"id": "x", "object": "chat.completion", "created": 0, "model": "invented",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": self.level_answer(document_id)}}]}
        return httpx.Response(200, json=body)


@pytest.fixture(autouse=True)
def no_waiting(pipeline_settings: dict[str, Any]) -> None:
    """The field calls push a run over 10 requests; the 10-second window of the client-side
    rate limiter would make every test wait 10 s. It is tested in test_llm_client.py."""
    pipeline_settings["llm"]["requests_per_10_seconds"] = 1000


@pytest.fixture
def level_settings(pipeline_settings: dict[str, Any]) -> dict[str, Any]:
    pipeline_settings["field_prompts"] = {LEVEL_FIELD: str(LEVEL_PROMPT)}
    return pipeline_settings


@pytest.fixture
def level_llm(enum, specs) -> LevelLlm:
    return LevelLlm(enum, specs["ecosystem_service.term"].allowed_values[-1])


def predictions(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {record[ID_FIELD]: record for record in read_jsonl(run_dir / PREDICTIONS_FILENAME)}


def without_field(prompts: dict[str, Any], field: str) -> dict[str, Any]:
    return {key: value for key, value in prompts.items() if key != field}


# --- the prompt file -------------------------------------------------------------------------

def test_level_prompt_names_every_level_and_fills_all_placeholders(invented_settings, specs, invented_examples):
    """level_v1.txt types the level names into its text: each must be the schema's spelling."""
    template = LEVEL_PROMPT.read_text(encoding="utf-8")
    for level in specs[LEVEL_FIELD].allowed_values:
        assert level in template, level
    prompt = ExtractionPrompt.from_settings(invented_settings, LEVEL_PROMPT)
    assert prompt.uses_examples
    _, user = prompt.messages("DOC00001", "1/1", "An invented text.", invented_examples.text_for("DOC00001"))
    for name in PER_CALL_PLACEHOLDERS:
        assert "{" + name + "}" not in user["content"]
    assert "part 1/1 of document DOC00001" in user["content"]
    assert '{"field": "biodiversity_level"' in user["content"]  # escaped braces of the output structure


# --- replacing, fallback, cache ---------------------------------------------------------------

def test_field_prompt_replaces_the_level_labels(level_settings, level_llm, tmp_path, word_tokenizer, enum):
    summary, run_dir = run(level_settings, level_llm, tmp_path, word_tokenizer)
    by_id = predictions(run_dir)
    assert by_id["DOC00001"][LEVEL_FIELD] == [enum(LEVEL_FIELD, 1)]   # not the main call's level 3
    assert by_id["DOC00001"]["habitat"] == [enum("habitat", 0)]       # the field call's habitat is ignored
    assert by_id["DOC00002"][LEVEL_FIELD] == [enum(LEVEL_FIELD, 2)]   # merged over its parts
    numbers = summary["field_prompts"][LEVEL_FIELD]
    assert numbers["prompt"] == LEVEL_PROMPT.name
    assert numbers["papers_replaced"] == 2
    part_count = summary["documents_in_parts"]["DOC00002"]
    assert numbers["requests"] == 1 + part_count == len(level_llm.level_prompts) - 1  # DOC00003's call failed


def test_failed_field_call_keeps_the_main_labels(level_settings, level_llm, tmp_path, word_tokenizer, enum):
    summary, run_dir = run(level_settings, level_llm, tmp_path, word_tokenizer)
    assert predictions(run_dir)["DOC00003"][LEVEL_FIELD] == [enum(LEVEL_FIELD, 4)]
    numbers = summary["field_prompts"][LEVEL_FIELD]
    assert list(numbers["failed"]) == ["DOC00003"] and numbers["papers_with_fallback"] == 1
    assert summary["failed"] == {}   # the paper itself did not fail


def test_invalid_json_in_the_field_call_keeps_the_main_labels(level_settings, enum, specs, tmp_path, word_tokenizer):
    llm = LevelLlm(enum, specs["ecosystem_service.term"].allowed_values[-1],
                   level_failure={"DOC00001": FAIL_WITH_INVALID_JSON})
    summary, run_dir = run(level_settings, llm, tmp_path, word_tokenizer)
    assert predictions(run_dir)["DOC00001"][LEVEL_FIELD] == [enum(LEVEL_FIELD, 3)]
    assert list(summary["field_prompts"][LEVEL_FIELD]["failed"]) == ["DOC00001"]


def test_main_call_is_unchanged_and_comes_from_the_cache(pipeline_settings, level_llm, tmp_path, word_tokenizer):
    summary_main, run_main = run(pipeline_settings, level_llm, tmp_path, word_tokenizer, run_name="main_only")
    main_prompts = list(level_llm.prompts)
    assert summary_main["field_prompts"] == {} and level_llm.level_prompts == []

    pipeline_settings["field_prompts"] = {LEVEL_FIELD: str(LEVEL_PROMPT)}
    summary, run_dir = run(pipeline_settings, level_llm, tmp_path, word_tokenizer, run_name="with_level")
    new_main_prompts = [messages for messages in level_llm.prompts[len(main_prompts):]
                        if LEVEL_SYSTEM_MARKER not in messages[0]["content"]]
    assert new_main_prompts == []   # every main call came from the cache
    assert summary["requests"]["api_requests"] == len(level_llm.level_prompts)
    before, after = predictions(run_main), predictions(run_dir)
    for document_id in before:
        assert without_field(after[document_id], LEVEL_FIELD) == without_field(before[document_id], LEVEL_FIELD)


def test_field_prompt_off_sends_no_field_calls(pipeline_settings, level_llm, tmp_path, word_tokenizer):
    pipeline_settings["field_prompts"] = {LEVEL_FIELD: None}
    summary, _ = run(pipeline_settings, level_llm, tmp_path, word_tokenizer)
    assert level_llm.level_prompts == [] and summary["field_prompts"] == {}


# --- label file and min_confidence in a run ---------------------------------------------------

def test_label_file_records_confidence_source_and_fate(level_settings, level_llm, tmp_path, word_tokenizer, enum):
    _, run_dir = run(level_settings, level_llm, tmp_path, word_tokenizer)
    lines = read_jsonl(run_dir / LABEL_FILE_NAME)
    first = [line for line in lines if line["document_id"] == "DOC00001"]
    level = next(line for line in first if line["field"] == LEVEL_FIELD and line["kept_by_checker"]
                 and not line["replaced_by_field_prompt"])
    assert level["value"] == enum(LEVEL_FIELD, 1) and level["confidence"] == 0.8
    assert level["source"] == SOURCE_FIELD_PROMPT and level["dropped_by_postprocessing"] is False
    replaced = [line for line in first if line["replaced_by_field_prompt"]]
    assert [(line["value"], line["source"]) for line in replaced] == [(enum(LEVEL_FIELD, 3), SOURCE_MAIN)]
    dropped = {line["field"]: line["dropped_by_check"] for line in first if not line["kept_by_checker"]}
    assert dropped == {LEVEL_FIELD: "C4", "ecosystem_service": "C2"}
    second_level = [line for line in lines if line["document_id"] == "DOC00002" and line["kept_by_checker"]
                    and line["field"] == LEVEL_FIELD]
    assert [line["confidence"] for line in second_level] == [None]   # "high" is not a number
    assert all("quote" not in json.dumps(line) for line in lines)    # no text from the papers


def test_min_confidence_in_a_run_drops_and_counts(level_settings, level_llm, tmp_path, word_tokenizer, enum):
    level_settings["postprocess"]["min_confidence"] = {LEVEL_FIELD: 0.9}
    summary, run_dir = run(level_settings, level_llm, tmp_path, word_tokenizer)
    by_id = predictions(run_dir)
    assert by_id["DOC00001"][LEVEL_FIELD] == []                      # 0.8 < 0.9
    assert by_id["DOC00002"][LEVEL_FIELD] == [enum(LEVEL_FIELD, 2)]  # no confidence: kept
    numbers = summary["min_confidence"]
    assert numbers["labels_dropped"] == 1 and numbers["labels_dropped_per_field"] == {LEVEL_FIELD: 1}
    lines = read_jsonl(run_dir / LABEL_FILE_NAME)
    assert [line["postprocessing_rule"] for line in lines if line["dropped_by_postprocessing"]] == ["min_confidence"]


# --- default and experiment files ---------------------------------------------------------------

def test_level_prompt_is_the_default_and_no_level_prompt_switches_it_off(settings, apply_experiment):
    """Default since 2026-11-19 (SPEC.md, section 13); no_level_prompt.yaml reproduces F v4.1 +
    service rule from the cache, level_prompt_v1.yaml stays as it was."""
    assert field_prompt_paths(settings) == {LEVEL_FIELD: LEVEL_PROMPT}
    assert field_prompt_paths(apply_experiment(settings, "no_level_prompt.yaml")) == {}
    assert field_prompt_paths(apply_experiment(settings, "level_prompt_v1.yaml")) == {LEVEL_FIELD: LEVEL_PROMPT}
