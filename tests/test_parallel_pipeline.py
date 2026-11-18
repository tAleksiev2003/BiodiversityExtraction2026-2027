"""Pipeline B with invented documents, a fake API and words as tokens (WordTokenizer).

Covers: the effective chunk size and the limit of chunks per document (chunk_packing.py),
the vote across chunks, the checker with the text of one chunk only, the chunk prompt,
one-chunk documents sent as pipeline F's request, the dry run, and requests of different
documents in flight at the same time.
No test reaches the real API or reads course data.
"""
import copy
import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import openai
import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel, DoclingDocument, ProvenanceItem, Size

from chunk_packing import PackingConfig, effective_chunk_tokens, pack_chunks
from chunking import PARAGRAPH_SEPARATOR, Chunk, ChunkingConfig, chunk_document
from data import ID_FIELD, read_jsonl
from llm_client import LLMClient
from output import PREDICTIONS_FILENAME
from pdf_to_text import save_document
from pipelines.full_text import CALL_LOG_NAME, CHECK_LOG_NAME, DOCUMENT_LOG_NAME, SUMMARY_NAME, run_full_text
from pipelines.parallel import CHECK_CHUNK_VOTE, merge_document, run_parallel, vote_across_chunks, MergeConfig
from prompts import ExtractionPrompt
from verbatim_checker import CHECK_VERBATIM_QUOTE, CheckedLabel, CheckerConfig

CHUNK_TOKENS = 60  # words: every section of DOC00001 is one chunk
SAMPLING = "We sampled ground beetles with pitfall traps along twelve hedgerows in Musterdorf."
RICHNESS = "Species richness of carabid beetles was highest in old hedgerows with a wide herb layer."
DENSITY = "The density of carabid beetles was highest in old hedgerows close to the arable fields."
BIRDS = "Breeding birds were mapped on six transects in the invented Musterwald forest reserve."
PROMPT_PATTERN = re.compile(r"chunk (\d+)/(\d+) of document (\w+)")
CHUNK_RULE = "R9  One chunk only."  # only in the chunk prompt (extraction_chunk_v1.txt)


def words(prefix: str, count: int) -> str:
    """Unique invented words, so no quote is found by accident."""
    return " ".join(f"{prefix}w{index}" for index in range(count))


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
    # Three sections of about 45 words: three chunks. SAMPLING is only in chunk 1,
    # RICHNESS only in chunk 2, DENSITY only in chunk 3.
    "DOC00001": [("Methods", [SAMPLING, words("s1", 30)]), ("Results", [RICHNESS, words("s2", 30)]),
                 ("Discussion", [DENSITY, words("s3", 30)])],
    "DOC00002": [("Methods", [BIRDS])],  # one chunk
}
NOT_CONVERTED = "DOC00004"
ALL_IDS = ["DOC00002", NOT_CONVERTED, "DOC00001"]  # not sorted on purpose


# --- effective chunk size and packing ------------------------------------------------------

def packing(max_chunk_tokens: int, max_chunks: int, cap: int) -> PackingConfig:
    return PackingConfig(max_chunk_tokens=max_chunk_tokens, max_chunks_per_document=max_chunks,
                         max_part_tokens=cap)


def pieces(token_counts: list[int]) -> list[Chunk]:
    """Chunks whose text has exactly the given number of (unique) words."""
    return [Chunk(index, words(f"p{index}", count), count, (f"Section {index}",), index + 1, index + 1)
            for index, count in enumerate(token_counts)]


def test_effective_chunk_size_follows_the_formula_and_the_cap():
    config = packing(max_chunk_tokens=100, max_chunks=20, cap=1000)
    assert effective_chunk_tokens(1500, config) == 100         # short: max_chunk_tokens
    assert effective_chunk_tokens(2001, config) == 101         # ceil(2001 / 20)
    assert effective_chunk_tokens(10_000, config) == 500
    assert effective_chunk_tokens(50_000, config) == 1000      # the cap


def test_many_small_pieces_are_packed_into_at_most_the_allowed_number_of_chunks():
    # Like a book split at many headings: 300 pieces, far more than 20 chunks.
    small = pieces([7, 13, 5, 21] * 75)
    total = sum(piece.token_count for piece in small)
    config = packing(max_chunk_tokens=100, max_chunks=20, cap=10_000)
    packed = pack_chunks(small, total, config, separator_tokens=1)
    assert len(packed.chunks) <= 20
    assert len(packed.chunks) >= math.ceil(total / packed.chunk_tokens)
    assert packed.enlarged and not packed.capped
    assert all(chunk.token_count <= packed.chunk_tokens for chunk in packed.chunks)
    # In document order, nothing lost, no piece split.
    assert PARAGRAPH_SEPARATOR.join(chunk.text for chunk in packed.chunks) == \
        PARAGRAPH_SEPARATOR.join(piece.text for piece in small)
    assert [chunk.index for chunk in packed.chunks] == list(range(len(packed.chunks)))


def test_size_is_raised_a_little_when_whole_pieces_do_not_fit_into_the_allowed_number():
    # 3 pieces of 60, at most 2 chunks: the formula gives 100, but then every piece needs
    # its own chunk. The smallest size for 2 chunks is 60 + 1 + 60 = 121.
    packed = pack_chunks(pieces([60, 60, 60]), 180, packing(100, 2, 1000), separator_tokens=1)
    assert len(packed.chunks) == 2
    assert packed.chunk_tokens == 121
    assert [chunk.token_count for chunk in packed.chunks] == [121, 60]


def test_cap_allows_more_chunks_than_the_limit():
    packed = pack_chunks(pieces([100] * 10), 1000, packing(100, 2, 300), separator_tokens=0)
    assert packed.capped and packed.chunk_tokens == 300
    assert [chunk.token_count for chunk in packed.chunks] == [300, 300, 300, 100]


def test_short_document_keeps_the_chunker_pieces():
    # As the chunker leaves them: no two neighbours fit into max_chunk_tokens together.
    small = pieces([60, 50, 70])
    packed = pack_chunks(small, 180, packing(100, 20, 1000), separator_tokens=0)
    assert not packed.enlarged
    assert [chunk.text for chunk in packed.chunks] == [piece.text for piece in small]


def test_document_with_many_headings_gets_at_most_the_allowed_number_of_chunks(word_tokenizer):
    """The M1 case: splitting at headings gives many small chunks; packing joins them."""
    document = build_document([(f"Heading {section}", [words(f"h{section}", 25)]) for section in range(80)])
    tokenizer = word_tokenizer(60)
    result = chunk_document(document, tokenizer, ChunkingConfig(60, 0, 1))
    assert len(result.chunks) > 20
    total = sum(chunk.token_count for chunk in result.chunks)
    packed = pack_chunks(result.chunks, total, packing(60, 20, 10_000),
                         tokenizer.count_tokens(PARAGRAPH_SEPARATOR))
    assert len(packed.chunks) <= 20
    assert len(packed.chunks) >= math.ceil(total / packed.chunk_tokens)
    sent = {word for chunk in packed.chunks for word in chunk.text.split()}
    assert all(set(words(f"h{section}", 25).split()) <= sent for section in range(80))


# --- vote across chunks ----------------------------------------------------------------------

def label(value: str, quote: str) -> CheckedLabel:
    return CheckedLabel("habitat", ("habitat", value), value, quote)


# Label "a" in chunks 1, 2, 3; "b" in chunks 1 and 3; "c" in chunk 2 only.
THREE_CHUNKS = [[label("a", "q1"), label("b", "q2")], [label("a", "q3"), label("c", "q4")],
                [label("b", "q5"), label("a", "q6")]]


@pytest.mark.parametrize("min_chunks, kept", [(1, {"a", "b", "c"}), (2, {"a", "b"}), (3, {"a"})])
def test_vote_keeps_labels_found_in_enough_chunks(min_chunks, kept):
    labels, log = vote_across_chunks(THREE_CHUNKS, min_chunks, "DOC00001")
    assert {label.entry for label in labels} == kept
    assert {entry.label["entry"] for entry in log} == {"a", "b", "c"} - kept
    assert all(entry.check == CHECK_CHUNK_VOTE and not entry.passed for entry in log)


def test_vote_needs_at_most_all_chunks_of_a_short_document():
    two_chunks = [[label("a", "q1"), label("b", "q2")], [label("a", "q3")]]
    labels, _ = vote_across_chunks(two_chunks, 3, "DOC00001")  # 3 needed, but only 2 chunks
    assert {label.entry for label in labels} == {"a"}
    one_chunk = [[label("a", "q1"), label("b", "q2")]]
    labels, _ = vote_across_chunks(one_chunk, 3, "DOC00002")
    assert {label.entry for label in labels} == {"a", "b"}


def test_merge_gives_one_label_per_key_after_the_vote(settings):
    merged, _ = merge_document(THREE_CHUNKS, MergeConfig(2), CheckerConfig.from_settings(settings), "DOC00001")
    assert [label.entry for label in merged] == ["a", "b"]


# --- chunk prompt ------------------------------------------------------------------------------

def test_chunk_prompt_is_filled_and_has_no_pipeline_a_section(invented_settings, specs):
    paths = invented_settings["paths"]
    prompt = ExtractionPrompt.from_settings(invented_settings, paths["prompt_extraction_chunked"])
    text = "An invented chunk with {braces}."
    system, user = prompt.messages("DOC00001", "2/3", text)
    content = system["content"] + user["content"]
    for name in ("habitat", "biodiversity_level", "taxa.species_group", "ecosystem_type.term",
                 "ecosystem_service.category", "ecosystem_service.term"):
        for value in specs[name].allowed_values:
            assert json.dumps(value, ensure_ascii=False) in content, (name, value)
    assert "chunk 2/3 of document DOC00001" in content and text in content
    assert re.search(r"\{[a-z_]+\}", content.replace(text, "")) is None  # every placeholder filled
    assert "PIPELINE A" not in content.upper() and "LEDGER" not in content.upper()
    assert paths["prompt_extraction_chunked"] != paths["prompt_extraction"]  # F keeps its prompt


# --- the run -----------------------------------------------------------------------------------

def chunk_text(messages: list[dict[str, str]]) -> str:
    return messages[1]["content"].split("<<<\n", 1)[1].rsplit("\n>>>", 1)[0]


class InventedLlm:
    """Answers every chunk of a document with the same labels; thread-safe.

    DOC00001: habitat 0 quoting SAMPLING (chunk 1), habitat 1 quoting RICHNESS (chunk 2)
    and DENSITY (chunk 3). So with the chunk text alone, habitat 0 is verified in one
    chunk and habitat 1 in two. `fail_document`: every request of it gets HTTP 400.
    """

    def __init__(self, enum: Callable[[str, int], str], delay: float = 0.0, remaining_day: int | None = None,
                 fail_document: str | None = None) -> None:
        self.enum = enum
        self.delay = delay
        self.remaining_day = remaining_day
        self.fail_document = fail_document
        self.prompts: list[list[dict[str, str]]] = []
        self.in_flight: list[str] = []          # document of every request in flight
        self.documents_in_flight_together = 0   # most different documents in flight at once
        self.max_in_flight = 0
        self._lock = threading.Lock()

    def answer(self, document_id: str) -> str:
        enum = self.enum
        if document_id == "DOC00001":
            labels = [{"field": "habitat", "value": enum("habitat", 0), "quote": SAMPLING},
                      {"field": "habitat", "value": enum("habitat", 1), "quote": RICHNESS},
                      {"field": "habitat", "value": enum("habitat", 1), "quote": DENSITY}]
        else:
            labels = [{"field": "habitat", "value": enum("habitat", 2), "quote": BIRDS}]
        return json.dumps({"reasoning": "S1: invented notes", "labels": labels})

    def handle(self, request: httpx.Request) -> httpx.Response:
        messages = json.loads(request.content)["messages"]
        document_id = PROMPT_PATTERN.search(messages[1]["content"]).group(3)
        with self._lock:
            self.prompts.append(messages)
            self.in_flight.append(document_id)
            self.max_in_flight = max(self.max_in_flight, len(self.in_flight))
            self.documents_in_flight_together = max(self.documents_in_flight_together, len(set(self.in_flight)))
        try:
            time.sleep(self.delay)
            if document_id == self.fail_document:
                return httpx.Response(400, json={"error": {"message": "invented: prompt too long"}})
            headers = {}
            if self.remaining_day is not None:
                headers["x-ratelimit-remaining-requests-day"] = str(self.remaining_day)
            body = {"id": "x", "object": "chat.completion", "created": 0, "model": "invented",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": self.answer(document_id)}}]}
            return httpx.Response(200, json=body, headers=headers)
        finally:
            with self._lock:
                self.in_flight.remove(document_id)

    def openai_client(self) -> openai.OpenAI:
        return openai.OpenAI(api_key="invented-test-key", base_url="https://llm.invalid/v1", max_retries=0,
                             http_client=httpx.Client(transport=httpx.MockTransport(self.handle)))


@pytest.fixture
def pipeline_settings(invented_settings: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(invented_settings)
    copied["chunking"]["max_chunk_tokens"] = CHUNK_TOKENS
    for document_id, sections in INVENTED_DOCUMENTS.items():
        save_document(build_document(sections), copied["paths"]["docling"] / "dev" / f"{document_id}.json")
    return copied


def run(settings: dict[str, Any], llm: InventedLlm | None, tmp_path: Path, word_tokenizer,
        dry_run: bool = False, run_name: str = "run", document_ids: list[str] = ALL_IDS
        ) -> tuple[dict[str, Any], Path]:
    run_dir = tmp_path / run_name
    run_dir.mkdir()
    client = LLMClient(settings, dry_run=dry_run, call_log_path=run_dir / CALL_LOG_NAME,
                       openai_clients=[llm.openai_client()] if llm is not None else None)
    summary = run_parallel(settings, document_ids, client, run_dir, settings["paths"]["docling"] / "dev",
                           word_tokenizer(CHUNK_TOKENS))
    return summary, run_dir


def predictions_by_id(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {record[ID_FIELD]: record for record in read_jsonl(run_dir / PREDICTIONS_FILENAME)}


def test_full_run_checks_each_chunk_alone_and_writes_sorted_predictions(pipeline_settings, enum, tmp_path,
                                                                        word_tokenizer):
    llm = InventedLlm(enum)
    summary, run_dir = run(pipeline_settings, llm, tmp_path, word_tokenizer)

    assert [record[ID_FIELD] for record in read_jsonl(run_dir / PREDICTIONS_FILENAME)] == sorted(ALL_IDS)
    by_id = predictions_by_id(run_dir)
    assert by_id["DOC00001"]["habitat"] == [enum("habitat", 0), enum("habitat", 1)]  # k = 1: union
    assert by_id["DOC00002"]["habitat"] == [enum("habitat", 2)]
    assert all(values == [] for key, values in by_id[NOT_CONVERTED].items() if key != ID_FIELD)

    assert summary["chunks"]["chunks_total"] == 4 == len(llm.prompts)  # 3 + 1, one request per chunk
    assert summary["not_converted"] == [NOT_CONVERTED] and summary["failed"] == {}

    # The checker only searched the text of the chunk that was answered.
    log = [entry for entry in read_jsonl(run_dir / CHECK_LOG_NAME) if entry["document_id"] == "DOC00001"]
    sampling = {entry["call"]: entry for entry in log if entry["label"].get("quote") == SAMPLING}
    assert sampling["1/3"]["passed"]
    assert not sampling["2/3"]["passed"] and sampling["2/3"]["check"] == CHECK_VERBATIM_QUOTE
    assert not sampling["3/3"]["passed"]

    documents = {entry["document_id"]: entry for entry in read_jsonl(run_dir / DOCUMENT_LOG_NAME)}
    assert documents["DOC00001"]["chunk_count"] == 3
    assert documents["DOC00001"]["chunk_tokens"] == CHUNK_TOKENS


def test_merge_experiment_comes_from_the_cache_and_votes(pipeline_settings, enum, tmp_path, word_tokenizer):
    llm = InventedLlm(enum)
    run(pipeline_settings, llm, tmp_path, word_tokenizer, run_name="first")
    requests_before = len(llm.prompts)

    pipeline_settings["merge"]["min_chunks_per_label"] = 2
    summary, run_dir = run(pipeline_settings, llm, tmp_path, word_tokenizer, run_name="second")
    assert len(llm.prompts) == requests_before and summary["requests"]["api_requests"] == 0
    by_id = predictions_by_id(run_dir)
    assert by_id["DOC00001"]["habitat"] == [enum("habitat", 1)]  # habitat 0 was verified in one chunk only
    assert by_id["DOC00002"]["habitat"] == [enum("habitat", 2)]  # one chunk: min(2, 1) = 1 is enough
    assert summary["verbatim_check"]["all_fields"]["dropped"][CHECK_CHUNK_VOTE] == 1


def test_dry_run_counts_one_request_per_chunk(pipeline_settings, tmp_path, word_tokenizer, capsys):
    summary, run_dir = run(pipeline_settings, None, tmp_path, word_tokenizer, dry_run=True)
    assert summary["requests"]["dry_run_requests"] == summary["chunks"]["chunks_total"] == 4
    assert summary["requests"]["api_requests"] == 0
    assert summary["chunks"]["chunks_per_document_max"] == 3
    assert summary["chunks"]["chunks_per_document_median"] == 2  # documents with 3 and 1 chunks
    assert not (run_dir / PREDICTIONS_FILENAME).exists()
    printed = capsys.readouterr().out
    assert "Chunks: 4 in total" in printed and "median 2" in printed and "max 3" in printed


def test_dry_run_lists_documents_with_enlarged_chunks(pipeline_settings, tmp_path, word_tokenizer, capsys):
    pipeline_settings["chunking"]["max_chunks_per_document"] = 2  # DOC00001 must fit into 2 chunks
    summary, _ = run(pipeline_settings, None, tmp_path, word_tokenizer, dry_run=True)
    assert summary["chunks"]["chunks_per_document_max"] == 2
    assert list(summary["chunks"]["enlarged"]) == ["DOC00001"]
    assert summary["chunks"]["enlarged"]["DOC00001"] > CHUNK_TOKENS
    assert "DOC00001" in capsys.readouterr().out


def test_requests_of_different_documents_run_in_parallel_within_the_limit(pipeline_settings, enum, tmp_path,
                                                                          word_tokenizer):
    pipeline_settings["llm"]["max_concurrent_requests_per_key"] = 3
    for copy_number in range(3):  # more documents with several chunks
        save_document(build_document(INVENTED_DOCUMENTS["DOC00001"]),
                      pipeline_settings["paths"]["docling"] / "dev" / f"DOC1000{copy_number}.json")
    llm = InventedLlm(enum, delay=0.2)
    document_ids = ["DOC10000", "DOC10001", "DOC10002", "DOC00002"]
    summary, _ = run(pipeline_settings, llm, tmp_path, word_tokenizer, document_ids=document_ids)
    assert len(llm.prompts) == 3 * 3 + 1
    assert llm.max_in_flight <= 3                     # the limit per key holds
    assert llm.documents_in_flight_together >= 2      # chunks of different documents at the same time
    assert summary["failed"] == {}


def test_failed_request_gives_an_empty_prediction_for_that_document_only(pipeline_settings, enum, tmp_path,
                                                                         word_tokenizer):
    llm = InventedLlm(enum, fail_document="DOC00002")
    summary, run_dir = run(pipeline_settings, llm, tmp_path, word_tokenizer)
    assert list(summary["failed"]) == ["DOC00002"]
    by_id = predictions_by_id(run_dir)
    assert by_id["DOC00002"]["habitat"] == []
    assert by_id["DOC00001"]["habitat"] == [enum("habitat", 0), enum("habitat", 1)]


def test_daily_limit_stops_cleanly_without_predictions(pipeline_settings, enum, tmp_path, word_tokenizer):
    llm = InventedLlm(enum, remaining_day=0)
    summary, run_dir = run(pipeline_settings, llm, tmp_path, word_tokenizer)
    assert summary["stopped"] is not None and "Daily limit" in summary["stopped"]
    assert len(llm.prompts) == 1  # the first answer said 0 left: nothing else was sent
    assert not (run_dir / PREDICTIONS_FILENAME).exists()
    assert (run_dir / SUMMARY_NAME).exists()


# --- one-chunk documents are the full-text case -----------------------------------------------

def run_pipeline_f(settings: dict[str, Any], llm: InventedLlm, tmp_path: Path, word_tokenizer,
                   document_ids: list[str]) -> Path:
    run_dir = tmp_path / "pipeline_f"
    run_dir.mkdir()
    client = LLMClient(settings, call_log_path=run_dir / CALL_LOG_NAME, openai_clients=[llm.openai_client()])
    run_full_text(settings, document_ids, client, run_dir, settings["paths"]["docling"] / "dev",
                  word_tokenizer(settings["models"]["full_text_part_tokens"]))
    return run_dir


def logged_requests(run_dir: Path) -> list[tuple[str, str, list[dict[str, str]]]]:
    """(cache key, model, messages) of every call in the run's call log."""
    return [(entry["cache_key"], entry["model"], entry["messages"]) for entry in read_jsonl(run_dir / CALL_LOG_NAME)]


def all_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(message["content"] for message in messages)


def test_one_chunk_document_sends_exactly_pipeline_f_request(pipeline_settings, enum, tmp_path, word_tokenizer):
    pipeline_settings["models"]["chunked"] = pipeline_settings["models"]["full_text"]
    llm = InventedLlm(enum)
    f_run_dir = run_pipeline_f(pipeline_settings, llm, tmp_path, word_tokenizer, ["DOC00002"])
    summary, b_run_dir = run(pipeline_settings, llm, tmp_path, word_tokenizer, document_ids=["DOC00002"])

    assert logged_requests(b_run_dir) == logged_requests(f_run_dir)  # same model, messages, cache key
    assert CHUNK_RULE not in all_text(logged_requests(b_run_dir)[0][2])
    assert summary["requests"]["api_requests"] == 0 and len(llm.prompts) == 1  # answered from F's cache
    assert summary["chunks"]["documents_sent_as_full_text"] == 1
    assert predictions_by_id(b_run_dir)["DOC00002"]["habitat"] == [enum("habitat", 2)]


def test_documents_with_several_chunks_keep_the_chunk_prompt(pipeline_settings, enum, tmp_path, word_tokenizer):
    llm = InventedLlm(enum)
    run(pipeline_settings, llm, tmp_path, word_tokenizer)
    prompts = {(match.group(3), match.group(1)): all_text(messages)
               for messages in llm.prompts
               for match in [PROMPT_PATTERN.search(messages[1]["content"])]}
    assert all(CHUNK_RULE in prompts[("DOC00001", number)] for number in ("1", "2", "3"))
    assert CHUNK_RULE not in prompts[("DOC00002", "1")]


def test_switch_off_sends_one_chunk_documents_with_the_chunk_prompt(pipeline_settings, enum, tmp_path,
                                                                     word_tokenizer):
    """As in the Qwen run of 2026-11-18 (config/experiments/chunked_qwen.yaml)."""
    pipeline_settings["parallel"]["one_chunk_as_full_text"] = False
    llm = InventedLlm(enum)
    summary, _ = run(pipeline_settings, llm, tmp_path, word_tokenizer, document_ids=["DOC00002"])
    assert CHUNK_RULE in all_text(llm.prompts[0])
    assert summary["chunks"]["documents_sent_as_full_text"] == 0
