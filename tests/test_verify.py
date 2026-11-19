"""Verification step (src/run_verify.py) with invented documents and mocked LLM answers.

First a source F run is made with run_full_text and a fake API, exactly like a real F
run (its answers go to the temporary cache). Then the verify run builds on that run
folder: it rebuilds F's labels from the cache and asks the fake API, which answers each
verify prompt by a policy (keep, drop, no decision, invalid JSON) per candidate.
No test reaches the real API or reads course data.
"""
import copy
import json
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable

import httpx
import openai
import pytest
import yaml

from data import ID_FIELD, read_jsonl, write_jsonl
from few_shot_examples import FewShotExamples
from llm_answer import STATUS_OK_AFTER_RETRY
from llm_client import LLMClient
from output import PREDICTIONS_FILENAME
from pdf_to_text import save_document
from pipelines.full_text import CALL_LOG_NAME, SUMMARY_NAME, run_full_text
from pipelines.verify import (
    SOURCES_NAME, STATUS_SOURCE_NOT_REBUILT, VERIFY_LOG_NAME, SourceRun, run_verify)
from runs import create_run_dir
from settings import REPO_ROOT
from test_full_text_pipeline import build_document, filler
from verification import (
    DROP, KEEP, NEVER_CHECKED_FIELD, NOTE_MISSING_DECISION, NOTE_NOT_APPLIED, Candidate, VerifySettings)
from verify_prompt import VerifyPrompt

PART_TOKENS = 80  # words; DOC00002 has more and is split into parts
EXTRACTION_MODEL = "invented-extraction-model"
VERIFY_MODEL = "invented-verify-model"

SAMPLING = "We sampled ground beetles with pitfall traps along twelve hedgerows in Musterdorf."
POLLINATION = "Pollination of the invented crop was measured by counting visits of wild bees."
RICHNESS = "Species richness of carabid beetles was highest in old hedgerows with a wide herb layer."
FIRST_DESCRIBED = "Hedgerow beetles were first described in the invented Musterdorf survey."
EARTHWORMS = "Soil samples from the arable fields showed a high density of earthworms in spring."
BIRDS = "Breeding birds were mapped on six transects in the invented Musterwald forest reserve."

INVENTED_DOCUMENTS = {
    "DOC00001": [("Methods", [SAMPLING, POLLINATION]), ("Results", [RICHNESS])],
    "DOC00002": [("Introduction", [FIRST_DESCRIBED, filler(1)]), ("Study site", [filler(2)]),
                 ("Methods", [filler(3)]), ("Results", [EARTHWORMS, filler(4)])],
    "DOC00003": [("Methods", [BIRDS])],
}
NOT_CONVERTED = "DOC00004"
ALL_IDS = ["DOC00001", "DOC00002", "DOC00003", NOT_CONVERTED]

EXTRACTION_PATTERN = re.compile(r"chunk (\d+)/(\d+) of document (\w+)")
CANDIDATE_PATTERN = re.compile(r"^- (C\d+) \| field: (\w+) \| (.*)$", re.MULTILINE)
MISSING = "missing"   # policy: leave the candidate out of the answer
INVALID = "maybe"     # policy: a decision that is neither keep nor drop

Policy = Callable[[str, str], str]  # (field, candidate line) -> keep | drop | MISSING | INVALID


def keep_all(field: str, line: str) -> str:
    return KEEP


def drop_all(field: str, line: str) -> str:
    return DROP


class InventedLlm:
    """Answers extraction prompts like F's model and verify prompts by a policy; thread-safe."""

    def __init__(self, enum: Callable[[str, int], str], policy: Policy = keep_all,
                 invalid_first_verify_answer: bool = False) -> None:
        self.enum = enum
        self.policy = policy
        self.invalid_first_verify_answer = invalid_first_verify_answer
        self.extraction_prompts: list[list[dict[str, str]]] = []
        self.verify_prompts: list[list[dict[str, str]]] = []
        self._lock = threading.Lock()

    def extraction_answer(self, messages: list[dict[str, str]]) -> str:
        document_id = EXTRACTION_PATTERN.search(messages[1]["content"]).group(3)
        enum = self.enum
        labels: list[dict[str, Any]] = []
        if document_id == "DOC00001":
            labels = [
                {"field": "habitat", "value": enum("habitat", 0), "quote": SAMPLING},
                {"field": "biodiversity_level", "value": enum("biodiversity_level", 0), "quote": RICHNESS},
                {"field": "ecosystem_type", "category": enum("ecosystem_type.category", 0),
                 "term": enum("ecosystem_type.term", 0), "quote": SAMPLING},
                {"field": "taxa", "species_group": enum("taxa.species_group", 0),
                 "scientific_name": "Carabidae", "german_name": "Laufkäfer", "quote": RICHNESS},
                {"field": "taxa", "species_group": enum("taxa.species_group", 1),
                 "scientific_name": None, "german_name": "Erfundener Heckenkäfer", "quote": SAMPLING},
                {"field": "ecosystem_service", "category": enum("ecosystem_service.category", 0),
                 "term": enum("ecosystem_service.term", 0), "quote": POLLINATION},
                {"field": "biodiversity_variable", "value": "Erfundene Variable", "quote": RICHNESS},
            ]
        elif document_id == "DOC00002":
            # Every part claims both labels; each quote is only in one part.
            labels = [
                {"field": "taxa", "species_group": enum("taxa.species_group", 2),
                 "scientific_name": None, "german_name": "Erfundene Gruppe", "quote": FIRST_DESCRIBED},
                {"field": "biodiversity_level", "value": enum("biodiversity_level", 1), "quote": EARTHWORMS},
            ]
        elif document_id == "DOC00003":
            labels = [{"field": "habitat", "value": enum("habitat", 2), "quote": BIRDS}]
        return json.dumps({"reasoning": "invented", "labels": labels})

    def verify_answer(self, messages: list[dict[str, str]]) -> str:
        if self.invalid_first_verify_answer and len(messages) == 2:
            return "Decisions: {not json"
        decisions = []
        for candidate_id, field, line in CANDIDATE_PATTERN.findall(messages[1]["content"]):
            decision = self.policy(field, line)
            if decision != MISSING:
                decisions.append({"id": candidate_id, "decision": decision, "reason": "invented"})
        return json.dumps({"decisions": decisions})

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        messages = body["messages"]
        with self._lock:
            if body["model"] == VERIFY_MODEL:
                self.verify_prompts.append(messages)
            else:
                self.extraction_prompts.append(messages)
        content = self.verify_answer(messages) if body["model"] == VERIFY_MODEL else self.extraction_answer(messages)
        response = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": content}}]}
        return httpx.Response(200, json=response)

    def openai_client(self) -> openai.OpenAI:
        return openai.OpenAI(api_key="invented-test-key", base_url="https://llm.invalid/v1", max_retries=0,
                             http_client=httpx.Client(transport=httpx.MockTransport(self.handle)))


@pytest.fixture
def verify_settings(invented_settings: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(invented_settings)
    copied["models"]["full_text_part_tokens"] = PART_TOKENS
    copied["models"]["full_text"] = EXTRACTION_MODEL
    copied["verify"]["model"] = VERIFY_MODEL
    for document_id, sections in INVENTED_DOCUMENTS.items():
        save_document(build_document(sections), copied["paths"]["docling"] / "dev" / f"{document_id}.json")
    return copied


@pytest.fixture
def source_run(verify_settings, enum, word_tokenizer) -> Path:
    """A finished F run folder (config.yaml, summary.json, predictions.jsonl, cache)."""
    llm = InventedLlm(enum)
    run_dir = create_run_dir(verify_settings, "full_text_dev")
    client = LLMClient(verify_settings, call_log_path=run_dir / CALL_LOG_NAME, openai_clients=[llm.openai_client()])
    run_full_text(verify_settings, ALL_IDS, client, run_dir, verify_settings["paths"]["docling"] / "dev",
                  word_tokenizer(PART_TOKENS), FewShotExamples.from_settings(verify_settings, "dev"))
    return run_dir


def verify(settings: dict[str, Any], source_dir: Path, llm: InventedLlm, word_tokenizer,
           dry_run: bool = False, name: str = "verify_dev") -> tuple[dict[str, Any], Path]:
    source = SourceRun.load(source_dir)
    run_dir = create_run_dir(settings, name)
    client = LLMClient(settings, dry_run=dry_run, call_log_path=run_dir / CALL_LOG_NAME,
                       openai_clients=[llm.openai_client()])
    examples = FewShotExamples.from_settings(settings, "dev")
    summary = run_verify(settings, source, client, run_dir, settings["paths"]["docling"] / "dev",
                         word_tokenizer(PART_TOKENS), examples, examples)
    return summary, run_dir


def by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {record[ID_FIELD]: record for record in read_jsonl(path)}


def test_habitat_never_becomes_a_candidate(verify_settings, source_run, enum, word_tokenizer):
    llm = InventedLlm(enum)
    _, run_dir = verify(verify_settings, source_run, llm, word_tokenizer)
    log = read_jsonl(run_dir / VERIFY_LOG_NAME)
    assert log and NEVER_CHECKED_FIELD not in {entry["field"] for entry in log}
    assert not any(f"field: {NEVER_CHECKED_FIELD}" in messages[1]["content"] for messages in llm.verify_prompts)
    assert "biodiversity_variable" not in {entry["field"] for entry in log}  # not a candidate field either
    with pytest.raises(ValueError, match=NEVER_CHECKED_FIELD):
        broken = copy.deepcopy(verify_settings)
        broken["verify"]["candidate_fields"] = [NEVER_CHECKED_FIELD, *broken["verify"]["candidate_fields"]]
        VerifySettings.from_settings(broken)


def test_a_dropped_taxa_entry_disappears_completely(verify_settings, source_run, enum, word_tokenizer):
    def drop_carabidae(field: str, line: str) -> str:
        return DROP if field == "taxa" and "Carabidae" in line else KEEP

    _, run_dir = verify(verify_settings, source_run, InventedLlm(enum, drop_carabidae), word_tokenizer)
    source = by_id(source_run / PREDICTIONS_FILENAME)["DOC00001"]
    verified = by_id(run_dir / PREDICTIONS_FILENAME)["DOC00001"]
    assert len(source["taxa"]) == 2
    assert verified["taxa"] == [entry for entry in source["taxa"] if entry["scientific_name"] != "Carabidae"]
    assert "Laufkäfer" not in json.dumps(verified, ensure_ascii=False)  # german_name gone with the entry
    assert enum("taxa.species_group", 0) not in {entry["species_group"] for entry in verified["taxa"]}


def test_unchecked_fields_equal_the_source_run(verify_settings, source_run, enum, word_tokenizer):
    summary, run_dir = verify(verify_settings, source_run, InventedLlm(enum, drop_all), word_tokenizer)
    source = by_id(source_run / PREDICTIONS_FILENAME)
    verified = by_id(run_dir / PREDICTIONS_FILENAME)
    assert list(verified) == sorted(ALL_IDS)
    for document_id in ALL_IDS:
        for field in ("habitat", "biodiversity_variable"):
            assert verified[document_id][field] == source[document_id][field]
    assert source["DOC00001"]["habitat"] and source["DOC00001"]["biodiversity_variable"]
    for field in verify_settings["verify"]["fields"]:
        assert verified["DOC00001"][field] == []  # everything checked was dropped
    assert verified[NOT_CONVERTED] == source[NOT_CONVERTED]
    assert verified["DOC00003"] == source["DOC00003"]  # habitat only: nothing to check
    assert summary["failed"] == {} and summary["not_converted"] == [NOT_CONVERTED]
    sources = json.loads((run_dir / SOURCES_NAME).read_text(encoding="utf-8"))
    assert sources["source_run"] == str(source_run) and sources["verify_model"] == VERIFY_MODEL


def test_a_missing_or_invalid_decision_keeps_the_candidate(verify_settings, source_run, enum, word_tokenizer):
    def unclear(field: str, line: str) -> str:
        return {"biodiversity_level": MISSING, "ecosystem_type": INVALID}.get(field, DROP)

    summary, run_dir = verify(verify_settings, source_run, InventedLlm(enum, unclear), word_tokenizer)
    source = by_id(source_run / PREDICTIONS_FILENAME)["DOC00001"]
    verified = by_id(run_dir / PREDICTIONS_FILENAME)["DOC00001"]
    assert verified["biodiversity_level"] == source["biodiversity_level"] != []
    assert verified["ecosystem_type"] == source["ecosystem_type"] != []
    assert verified["ecosystem_service"] == []
    log = read_jsonl(run_dir / VERIFY_LOG_NAME)
    unclear_entries = [entry for entry in log if entry["field"] in ("biodiversity_level", "ecosystem_type")]
    assert unclear_entries and all(entry["decision"] == KEEP and entry["model_decision"] is None
                                   and entry["note"] == NOTE_MISSING_DECISION for entry in unclear_entries)
    counts = summary["decisions_per_field"]
    assert counts["biodiversity_level"]["kept_without_valid_decision"] == counts["biodiversity_level"]["candidates"]


def test_invalid_json_gets_one_retry(verify_settings, source_run, enum, word_tokenizer):
    llm = InventedLlm(enum, drop_all, invalid_first_verify_answer=True)
    summary, run_dir = verify(verify_settings, source_run, llm, word_tokenizer)
    assert summary["answer_statuses"] == {STATUS_OK_AFTER_RETRY: summary["calls"]}
    assert by_id(run_dir / PREDICTIONS_FILENAME)["DOC00001"]["taxa"] == []


def test_dry_run_counts_one_request_per_paper_or_part(verify_settings, source_run, enum, word_tokenizer):
    llm = InventedLlm(enum)
    summary, run_dir = verify(verify_settings, source_run, llm, word_tokenizer, dry_run=True)
    # DOC00001: one call; DOC00002: two parts with candidates; DOC00003: habitat only, no call.
    assert summary["requests"]["dry_run_requests"] == 3 == summary["calls"]
    assert summary["requests"]["api_requests"] == 0 and llm.verify_prompts == [] == llm.extraction_prompts
    assert summary["failed"] == {}  # the source run was rebuilt from the cache
    assert not (run_dir / PREDICTIONS_FILENAME).exists()


def test_each_part_gets_only_the_candidates_whose_quote_it_holds(verify_settings, source_run, enum,
                                                                 word_tokenizer):
    llm = InventedLlm(enum)
    verify(verify_settings, source_run, llm, word_tokenizer)
    part_prompts = [messages[1]["content"] for messages in llm.verify_prompts
                    if FIRST_DESCRIBED in messages[1]["content"] or EARTHWORMS in messages[1]["content"]]
    assert len(part_prompts) == 2
    for prompt in part_prompts:
        text = prompt.split("<<<\n", 1)[1]
        for _, _, line in CANDIDATE_PATTERN.findall(prompt):
            quote = json.loads(line.rsplit("| quote: ", 1)[1])
            assert quote in text


def test_verify_fields_switches_a_field_off_from_the_cache(verify_settings, source_run, enum, word_tokenizer,
                                                           apply_experiment):
    verify(verify_settings, source_run, InventedLlm(enum, drop_all), word_tokenizer, name="verify_dev")
    without_taxa = apply_experiment(verify_settings, "verify_without_taxa.yaml")
    assert "taxa" not in without_taxa["verify"]["fields"]
    llm = InventedLlm(enum, keep_all)  # would keep everything: must not be asked
    summary, run_dir = verify(without_taxa, source_run, llm, word_tokenizer, name="verify_dev_without_taxa")
    assert summary["requests"]["api_requests"] == 0 and llm.verify_prompts == []  # same prompts: cache
    source = by_id(source_run / PREDICTIONS_FILENAME)
    verified = by_id(run_dir / PREDICTIONS_FILENAME)
    assert verified["DOC00001"]["taxa"] == source["DOC00001"]["taxa"] != []
    assert verified["DOC00002"]["taxa"] == source["DOC00002"]["taxa"] != []
    assert verified["DOC00001"]["biodiversity_level"] == []  # still applied
    taxa_entries = [entry for entry in read_jsonl(run_dir / VERIFY_LOG_NAME) if entry["field"] == "taxa"]
    assert all(entry["decision"] == KEEP and entry["model_decision"] == DROP
               and entry["note"] == NOTE_NOT_APPLIED for entry in taxa_entries)
    assert summary["decisions_per_field"]["taxa"]["drop_not_applied"] == len(taxa_entries) == 3


def test_filled_prompt_contains_every_candidate_id(verify_settings, enum):
    candidates = [
        Candidate("C1", "biodiversity_level", enum("biodiversity_level", 0), RICHNESS, 0),
        Candidate("C2", "taxa", {"scientific_name": None, "german_name": "Erfundene Art",
                                 "species_group": enum("taxa.species_group", 0)}, SAMPLING, 0),
        Candidate("C3", "ecosystem_service", {"category": enum("ecosystem_service.category", 0), "term": None},
                  POLLINATION, 0),
    ]
    prompt = VerifyPrompt.from_file(verify_settings["paths"]["prompt_verify"])
    document_text = "An invented text with {braces} and {candidates} inside."
    user_text = prompt.messages("invented examples", candidates, document_text)[1]["content"]
    found = [candidate_id for candidate_id, _, _ in CANDIDATE_PATTERN.findall(user_text)]
    assert found == ["C1", "C2", "C3"]
    assert document_text in user_text  # sent unchanged
    assert '{"decisions"' in user_text  # the {{ }} of the template became single braces
    assert "{examples}" not in user_text and "{document_text}" not in user_text


def test_the_run_prompts_name_every_logged_candidate(verify_settings, source_run, enum, word_tokenizer):
    llm = InventedLlm(enum)
    _, run_dir = verify(verify_settings, source_run, llm, word_tokenizer)
    sent = sorted(candidate_id for messages in llm.verify_prompts
                  for candidate_id, _, _ in CANDIDATE_PATTERN.findall(messages[1]["content"]))
    logged = sorted(entry["candidate_id"] for entry in read_jsonl(run_dir / VERIFY_LOG_NAME))
    # DOC00001: 4 (its service is dropped by the service rule of the source run: the invented
    # neighbours have no services, src/postprocessing.py), DOC00002: 2
    assert sent == logged and len(logged) == 4 + 2


def test_a_source_that_cannot_be_rebuilt_keeps_its_prediction(verify_settings, source_run, enum, word_tokenizer):
    predictions_path = source_run / PREDICTIONS_FILENAME
    records = read_jsonl(predictions_path)
    for record in records:
        if record[ID_FIELD] == "DOC00001":
            record["biodiversity_level"] = []  # no longer what the cache gives
    write_jsonl(records, predictions_path)
    summary, run_dir = verify(verify_settings, source_run, InventedLlm(enum, drop_all), word_tokenizer)
    assert summary["failed"]["DOC00001"].startswith(STATUS_SOURCE_NOT_REBUILT)
    assert by_id(run_dir / PREDICTIONS_FILENAME)["DOC00001"] == by_id(predictions_path)["DOC00001"]
    assert json.loads((run_dir / SUMMARY_NAME).read_text(encoding="utf-8"))["failed"] == summary["failed"]


def test_experiment_file_exists_and_only_changes_verify_fields():
    path = REPO_ROOT / "config" / "experiments" / "verify_without_taxa.yaml"
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert list(content) == ["verify"] and list(content["verify"]) == ["fields"]


def test_analysis_counts_dropped_labels_as_right_or_wrong(verify_settings, source_run, enum, word_tokenizer,
                                                          tmp_path):
    _, run_dir = verify(verify_settings, source_run, InventedLlm(enum, drop_all), word_tokenizer)
    # Invented references: DOC00001's taxa are right, everything else it predicted is wrong.
    references = {document_id: {ID_FIELD: document_id} for document_id in ALL_IDS}
    references["DOC00001"]["taxa"] = by_id(source_run / PREDICTIONS_FILENAME)["DOC00001"]["taxa"]
    references_path = tmp_path / "references.jsonl"
    write_jsonl(list(references.values()), references_path)
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "scripts" / "analyze_verification.py"),
         "--references", str(references_path), "--run", str(run_dir)],
        capture_output=True, text=True, check=True)
    _, decisions, right_wrong, scores = result.stdout.split("\n\n")  # the sections of the report

    def rows(section: str) -> dict[str, list[str]]:
        return {line.split()[0]: line.split()[1:] for line in section.splitlines()[1:]}

    assert rows(decisions)["taxa"] == ["3", "0", "3"]                         # candidates, kept, dropped
    assert rows(decisions)["biodiversity_level"] == ["2", "0", "2"]
    # kept right, kept wrong, dropped right, dropped wrong; DOC00002's group is not in the references
    assert rows(right_wrong)["taxa.species_group"] == ["0", "0", "2", "1"]
    assert rows(right_wrong)["biodiversity_level"] == ["0", "0", "0", "2"]
    assert "C1 score: source" in result.stdout
