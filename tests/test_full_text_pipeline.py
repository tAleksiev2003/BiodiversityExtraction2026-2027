"""Pipeline F end to end with invented documents and mocked LLM answers.

The documents are built in code and saved as Docling JSON, like src/pdf_to_text.py does.
A fake API answers every prompt with an invented JSON answer that depends on the
document and the part named in the prompt. Tokens are counted as words (WordTokenizer),
so a small `full_text_part_tokens` makes one document long enough to be split.
No test reaches the real API or reads course data.
"""
import copy
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import openai
import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel, DoclingDocument, ProvenanceItem, Size

from data import ID_FIELD, read_jsonl
from llm_answer import STATUS_DRY_RUN, STATUS_INVALID, STATUS_OK, STATUS_OK_AFTER_RETRY
from llm_client import LLMClient
from output import PREDICTIONS_FILENAME
from pdf_to_text import save_document
from pipelines.full_text import CHECK_LOG_NAME, SUMMARY_NAME, run_full_text
from prompts import ExtractionPrompt
from verbatim_checker import CHECK_CONSISTENT_PAIR, CHECK_VERBATIM_QUOTE

INVENTED_MODEL_ANSWER_DELAYS = {"DOC00001": 0.3, "DOC00002": 0.1, "DOC00003": 0.0}
PART_TOKENS = 80  # words; DOC00002 has more and is split into parts

SAMPLING = "We sampled ground beetles with pitfall traps along twelve hedgerows in Musterdorf."
RICHNESS = "Species richness of carabid beetles was highest in old hedgerows with a wide herb layer."
EARTHWORMS = "Soil samples from the arable fields showed a high density of earthworms in spring."
BIRDS = "Breeding birds were mapped on six transects in the invented Musterwald forest reserve."
PROMPT_PATTERN = re.compile(r"chunk (\d+)/(\d+) of document (\w+)")


def filler(section: int) -> str:
    """A paragraph of 40 unique invented words."""
    return " ".join(f"s{section}w{index}" for index in range(40))


def build_document(sections: list[tuple[str, list[str]]]) -> DoclingDocument:
    document = DoclingDocument(name="invented")
    document.add_page(page_no=1, size=Size(width=600, height=800))
    for heading, paragraphs in sections:
        document.add_heading(text=heading, prov=ProvenanceItem(
            page_no=1, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, len(heading))))
        for text in paragraphs:
            document.add_text(label=DocItemLabel.TEXT, text=text, prov=ProvenanceItem(
                page_no=1, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, len(text))))
    return document


INVENTED_DOCUMENTS = {
    "DOC00001": [("Methods", [SAMPLING]), ("Results", [RICHNESS])],
    "DOC00002": [("Introduction", [filler(1)]), ("Study site", [filler(2)]),
                 ("Methods", [filler(3)]), ("Results", [EARTHWORMS, filler(4)])],
    "DOC00003": [("Methods", [BIRDS])],
}
NOT_CONVERTED = "DOC00004"
ALL_IDS = ["DOC00003", "DOC00001", NOT_CONVERTED, "DOC00002"]  # not sorted on purpose


class InventedLlm:
    """Answers like a model would, per document and part; thread-safe; counts requests."""

    def __init__(self, enum: Callable[[str, int], str], unmapped_service_term: str,
                 remaining_day: int | None = None) -> None:
        self.enum = enum
        self.unmapped_service_term = unmapped_service_term
        self.remaining_day = remaining_day
        self.prompts: list[list[dict[str, str]]] = []
        self._lock = threading.Lock()

    def answer(self, messages: list[dict[str, str]]) -> str:
        part, _part_count, document_id = PROMPT_PATTERN.search(messages[1]["content"]).groups()
        enum = self.enum
        if document_id == "DOC00001":
            labels = [
                {"field": "habitat", "value": enum("habitat", 0), "quote": SAMPLING},
                {"field": "biodiversity_level", "value": enum("biodiversity_level", 0),
                 "quote": "Wild bees were counted on flowering meadows near the river"},  # invented: C4
                {"field": "ecosystem_service", "category": enum("ecosystem_service.category", 0),
                 "term": self.unmapped_service_term, "quote": SAMPLING},                 # no mapping: C2
                {"field": "taxa", "species_group": enum("taxa.species_group", 0),
                 "scientific_name": "Carabidae", "german_name": "Laufkäfer",
                 "name_source": "completed", "quote": RICHNESS},
            ]
        elif document_id == "DOC00002":
            # Every part claims the same label with the same quote; only the part that
            # contains the quote can verify it.
            labels = [{"field": "habitat", "value": enum("habitat", 1), "quote": EARTHWORMS}]
        else:
            if len(messages) == 2:
                return "Here is the record: {not json"  # the retry gets the parse error
            labels = [{"field": "habitat", "value": enum("habitat", 2), "quote": BIRDS}]
        return json.dumps({"reasoning": f"S1: invented notes for part {part}", "labels": labels})

    def handle(self, request: httpx.Request) -> httpx.Response:
        messages = json.loads(request.content)["messages"]
        with self._lock:
            self.prompts.append(messages)
        document_id = PROMPT_PATTERN.search(messages[1]["content"]).group(3)
        time.sleep(INVENTED_MODEL_ANSWER_DELAYS[document_id])  # finish in another order than sent
        headers = {}
        if self.remaining_day is not None:
            headers["x-ratelimit-remaining-requests-day"] = str(self.remaining_day)
        body = {"id": "x", "object": "chat.completion", "created": 0, "model": "invented",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": self.answer(messages)}}]}
        return httpx.Response(200, json=body, headers=headers)

    def openai_client(self) -> openai.OpenAI:
        return openai.OpenAI(api_key="invented-test-key", base_url="https://llm.invalid/v1", max_retries=0,
                             http_client=httpx.Client(transport=httpx.MockTransport(self.handle)))


@pytest.fixture
def pipeline_settings(invented_settings: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(invented_settings)
    copied["models"]["full_text_part_tokens"] = PART_TOKENS
    for document_id, sections in INVENTED_DOCUMENTS.items():
        save_document(build_document(sections), copied["paths"]["docling"] / "dev" / f"{document_id}.json")
    return copied


@pytest.fixture
def fake_llm(enum, specs) -> InventedLlm:
    return InventedLlm(enum, specs["ecosystem_service.term"].allowed_values[-1])


def run(settings: dict[str, Any], llm: InventedLlm | None, tmp_path: Path, word_tokenizer,
        dry_run: bool = False, run_name: str = "run") -> tuple[dict[str, Any], Path]:
    run_dir = tmp_path / run_name
    run_dir.mkdir()
    client = LLMClient(settings, dry_run=dry_run, call_log_path=run_dir / "llm_calls.jsonl",
                       openai_clients=[llm.openai_client()] if llm is not None else None)
    summary = run_full_text(settings, ALL_IDS, client, run_dir, settings["paths"]["docling"] / "dev",
                            word_tokenizer(PART_TOKENS))
    return summary, run_dir


def test_full_run_writes_sorted_checked_predictions(pipeline_settings, fake_llm, tmp_path, word_tokenizer, enum):
    summary, run_dir = run(pipeline_settings, fake_llm, tmp_path, word_tokenizer)

    predictions = read_jsonl(run_dir / PREDICTIONS_FILENAME)
    assert [record[ID_FIELD] for record in predictions] == sorted(ALL_IDS)  # not the finishing order
    by_id = {record[ID_FIELD]: record for record in predictions}

    first = by_id["DOC00001"]
    assert first["habitat"] == [enum("habitat", 0)]
    assert first["biodiversity_level"] == []   # invented quote: C4
    assert first["ecosystem_service"] == []    # term without a category: C2
    assert first["taxa"] == [{"scientific_name": "Carabidae", "german_name": "Laufkäfer",
                              "species_group": enum("taxa.species_group", 0)}]

    assert by_id["DOC00002"]["habitat"] == [enum("habitat", 1)]  # verified in one of its parts
    assert by_id["DOC00003"]["habitat"] == [enum("habitat", 2)]  # after the JSON retry
    assert all(values == [] for key, values in by_id[NOT_CONVERTED].items() if key != ID_FIELD)

    assert summary["documents_in_one_call"] == 2
    assert summary["documents_in_parts"]["DOC00002"] >= 2
    assert summary["not_converted"] == [NOT_CONVERTED]
    assert summary["answer_statuses"][STATUS_OK_AFTER_RETRY] == 1
    part_count = summary["documents_in_parts"]["DOC00002"]
    calls = 2 + part_count + 1  # DOC00003 is asked twice
    assert summary["requests"]["api_requests"] == calls == len(fake_llm.prompts)
    assert summary["wall_clock_seconds"] > 0
    assert summary["requests"]["average_request_seconds"] is not None
    assert (run_dir / SUMMARY_NAME).exists()


def test_check_log_names_the_check_of_every_dropped_label(pipeline_settings, fake_llm, tmp_path,
                                                          word_tokenizer, enum):
    _, run_dir = run(pipeline_settings, fake_llm, tmp_path, word_tokenizer)
    log = read_jsonl(run_dir / CHECK_LOG_NAME)
    first = [entry for entry in log if entry["document_id"] == "DOC00001"]
    dropped = {entry["field"]: entry["check"] for entry in first if not entry["passed"]}
    assert dropped == {"biodiversity_level": CHECK_VERBATIM_QUOTE, "ecosystem_service": CHECK_CONSISTENT_PAIR}
    # DOC00002: the quote is found only in the part that contains it.
    second = [entry for entry in log if entry["document_id"] == "DOC00002"]
    assert sum(entry["passed"] for entry in second) == 1
    assert {entry["check"] for entry in second if not entry["passed"]} == {CHECK_VERBATIM_QUOTE}


def test_long_document_parts_are_complete_and_within_the_limit(pipeline_settings, fake_llm, tmp_path,
                                                               word_tokenizer):
    run(pipeline_settings, fake_llm, tmp_path, word_tokenizer)
    part_prompts = [messages[1]["content"] for messages in fake_llm.prompts
                    if "of document DOC00002" in messages[1]["content"]]
    part_texts = [prompt.split("<<<\n", 1)[1].rsplit("\n>>>", 1)[0] for prompt in part_prompts]
    assert all(len(text.split()) <= PART_TOKENS for text in part_texts)
    words_sent = {word for text in part_texts for word in text.split()}
    for section in range(1, 5):  # no word of the document was left out
        assert set(filler(section).split()) <= words_sent


def test_second_run_is_answered_from_the_cache(pipeline_settings, fake_llm, tmp_path, word_tokenizer):
    run(pipeline_settings, fake_llm, tmp_path, word_tokenizer, run_name="first")
    requests_before = len(fake_llm.prompts)
    summary, run_dir = run(pipeline_settings, fake_llm, tmp_path, word_tokenizer, run_name="second")
    assert len(fake_llm.prompts) == requests_before
    assert summary["requests"]["api_requests"] == 0
    assert (run_dir / PREDICTIONS_FILENAME).exists()


def test_dry_run_counts_requests_and_never_retries(pipeline_settings, tmp_path, word_tokenizer):
    summary, run_dir = run(pipeline_settings, None, tmp_path, word_tokenizer, dry_run=True)
    statuses = summary["answer_statuses"]
    assert set(statuses) == {STATUS_DRY_RUN}
    assert STATUS_INVALID not in statuses
    calls = 2 + summary["documents_in_parts"]["DOC00002"]
    assert summary["requests"]["dry_run_requests"] == calls
    assert summary["requests"]["api_requests"] == 0
    assert not (run_dir / PREDICTIONS_FILENAME).exists()  # an all-empty file would only mislead


def test_daily_limit_stops_cleanly_without_predictions(pipeline_settings, enum, specs, tmp_path, word_tokenizer):
    llm = InventedLlm(enum, specs["ecosystem_service.term"].allowed_values[-1], remaining_day=0)
    summary, run_dir = run(pipeline_settings, llm, tmp_path, word_tokenizer)
    assert summary["stopped"] is not None and "Daily limit" in summary["stopped"]
    assert len(llm.prompts) == 1  # the first answer said 0 left: nothing else was sent
    assert not (run_dir / PREDICTIONS_FILENAME).exists()
    assert (run_dir / SUMMARY_NAME).exists()


def test_checker_switched_off_keeps_unverified_labels(pipeline_settings, fake_llm, tmp_path, word_tokenizer, enum):
    pipeline_settings["verbatim_check"]["enabled"] = False
    _, run_dir = run(pipeline_settings, fake_llm, tmp_path, word_tokenizer)
    by_id = {record[ID_FIELD]: record for record in read_jsonl(run_dir / PREDICTIONS_FILENAME)}
    assert by_id["DOC00001"]["biodiversity_level"] == [enum("biodiversity_level", 0)]
    assert len(by_id["DOC00001"]["ecosystem_service"]) == 1  # C2 is skipped too


# --- prompt -------------------------------------------------------------------------------

def test_prompt_shows_every_allowed_value_and_keeps_the_text_unchanged(invented_settings, specs) -> None:
    prompt = ExtractionPrompt.from_settings(invented_settings)
    text = "An invented text with {braces} and a {chunk_text} placeholder look-alike."
    system, user = prompt.messages("DOC00001", "1/1", text)
    assert system["role"] == "system" and user["role"] == "user"
    content = user["content"]
    for name in ("habitat", "biodiversity_level", "taxa.species_group", "ecosystem_type.category",
                 "ecosystem_type.term", "ecosystem_service.category", "ecosystem_service.term"):
        for value in specs[name].allowed_values:
            assert json.dumps(value, ensure_ascii=False) in content, (name, value)
    assert text in content
    assert "chunk 1/1 of document DOC00001" in content
    assert '{"field": "habitat"' in content  # the escaped braces of the output structure
    assert "Erfundene Variable" in content     # vocabulary from the invented label file
    assert "LEDGER" not in content.upper().split("## TEXT")[0]  # the unused pipeline A section is not sent
