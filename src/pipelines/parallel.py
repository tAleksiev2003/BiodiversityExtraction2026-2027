"""Pipeline B: chunks processed independently, merged in Python (SPEC.md, section 3, step 3B).

1. Prepare every document: load the converted document (Docling JSON), count the tokens
   of its full text, cut it with the M1 chunker and pack the pieces into at most
   `chunking.max_chunks_per_document` chunks (src/chunk_packing.py).
2. One request per chunk with `models.chunked` and the chunk prompt
   (`paths.prompt_extraction_chunked`). All (document, chunk) requests go into one
   thread pool, so the requests of different documents run at the same time, within
   the limits of the LLM client.
   A document that fits into one chunk is the full-text case (`parallel.one_chunk_as_full_text`,
   SPEC.md, section 13): it sends exactly pipeline F's request (the full text of
   pdf_to_text.full_text, prompt `paths.prompt_extraction`). With the same model the
   cache key is the same, so the answer comes from F's cache, and the difference
   between F and B comes only from the documents that need several chunks.
3. The verbatim checker (C1 to C4) right after every answer, with the text of that
   chunk only: a quote from another chunk of the same document does not count.
4. Merge per document, in Python, when all its chunks are answered:
   - vote: a label is kept if it has a verified quote in at least
     min(merge.min_chunks_per_label, number of chunks of the document) different chunks
     (1 = union; the min() keeps short documents from being emptied). A chunk that does
     not mention a label is not a vote against it;
   - then C5 as in pipeline F (verbatim_check.min_verified_quotes).
   Merging needs no request, so runs with another merge setting come from the cache.

A document with a failed request (an API error that retries did not fix) gets an empty
prediction and is listed under "failed"; the next run asks only the missing chunks,
because the answers already received are cached.

With the checker switched off (ablation), the vote counts the chunks that name a label
(only C1 is checked), because there are no verified quotes.

Since 2026-11-19 pipeline F is the main system and B is no longer developed (SPEC.md,
section 13). B cannot send prompts with training examples (prompt v4, the default), so
it runs with config/experiments/prompt_v2.yaml (or a file built on it); with the
defaults it stops with a message that names that file (`unsupported_settings`).
"""
import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import median
from typing import Any

import openai
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer

from chunk_packing import PackingConfig, pack_chunks
from chunking import PARAGRAPH_SEPARATOR, Chunk, ChunkingConfig, chunk_document
from data import ID_FIELD
from llm_answer import read_retry_template, request_answer
from llm_client import LLMClient, describe_error
from output import write_predictions
from parallel_runner import run_in_parallel
from pdf_to_text import DOCUMENT_SUFFIX, full_text, load_document
from pipelines.full_text import (
    CHECK_LOG_NAME,
    DOCUMENT_LOG_NAME,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_NOT_CONVERTED,
    STOP_ERRORS,
    SUMMARY_NAME,
    append_jsonl,
)
from prompts import ExtractionPrompt
from verbatim_checker import (
    ALL_CHECKS,
    CALL_CHECKS,
    CheckedLabel,
    CheckerConfig,
    CheckLogEntry,
    LabelRules,
    build_record,
    check_answer,
    drop_statistics,
    merge_labels,
)

logger = logging.getLogger(__name__)

CHECK_CHUNK_VOTE = "vote"  # log entry of a label dropped by the vote across chunks
REPORTED_CHECKS = (*ALL_CHECKS, CHECK_CHUNK_VOTE)
MERGED_CALL = "all"        # `call` of the log entries written when a document is merged
REQUEST_ID_SEPARATOR = "#"  # request ids look like "DOC00001#3" (document, chunk number)


# --- merging -------------------------------------------------------------------------------

@dataclass(frozen=True)
class MergeConfig:
    min_chunks_per_label: int

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> "MergeConfig":
        return cls(min_chunks_per_label=settings["merge"]["min_chunks_per_label"])


def vote_across_chunks(
    labels_per_chunk: list[list[CheckedLabel]], min_chunks: int, document_id: str
) -> tuple[list[CheckedLabel], list[CheckLogEntry]]:
    """Keep the labels found in at least min(min_chunks, number of chunks) different chunks.

    The labels come back in their original order (chunk by chunk); one log entry per
    dropped label key.
    """
    needed = min(min_chunks, len(labels_per_chunk))
    chunks_per_key: dict[tuple[str | None, ...], set[int]] = defaultdict(set)
    first_label: dict[tuple[str | None, ...], CheckedLabel] = {}
    for chunk_number, labels in enumerate(labels_per_chunk):
        for label in labels:
            chunks_per_key[label.key].add(chunk_number)
            first_label.setdefault(label.key, label)
    kept = [label for labels in labels_per_chunk for label in labels
            if len(chunks_per_key[label.key]) >= needed]
    log = [
        CheckLogEntry(document_id, MERGED_CALL, label.field, {"entry": label.entry}, CHECK_CHUNK_VOTE, False, None,
                      f"found in {len(chunks_per_key[key])} of {len(labels_per_chunk)} chunks, {needed} needed")
        for key, label in first_label.items() if len(chunks_per_key[key]) < needed
    ]
    return kept, log


def merge_document(
    labels_per_chunk: list[list[CheckedLabel]], merge_config: MergeConfig,
    checker_config: CheckerConfig, document_id: str,
) -> tuple[list[CheckedLabel], list[CheckLogEntry]]:
    """The vote across chunks, then C5 and one label per key (as in pipeline F)."""
    voted, vote_log = vote_across_chunks(labels_per_chunk, merge_config.min_chunks_per_label, document_id)
    merged, evidence_log = merge_labels(voted, checker_config, document_id)
    return merged, vote_log + evidence_log


# --- documents and requests ---------------------------------------------------------------

@dataclass
class PreparedDocument:
    """One document cut into the chunks that are sent, before any request."""

    document_id: str
    status: str                      # STATUS_DONE or STATUS_NOT_CONVERTED
    token_count: int | None = None   # full text, as in pipeline F and the M1 statistics
    chunking_level: str | None = None
    chunks: list[Chunk] = field(default_factory=list)
    full_text_request: bool = False  # one chunk, sent as pipeline F's request (its text is the full text)
    chunk_tokens: int | None = None  # effective chunk size used for this document
    enlarged: bool = False
    capped: bool = False
    seconds: float = 0.0


@dataclass(frozen=True)
class ChunkRequest:
    document_id: str
    number: int   # 1-based
    count: int    # chunks of the document
    text: str
    full_text_request: bool = False  # F's prompt instead of the chunk prompt

    @property
    def request_id(self) -> str:
        return f"{self.document_id}{REQUEST_ID_SEPARATOR}{self.number}"

    @property
    def call(self) -> str:
        """As in the prompt and the check log: "3/12" = chunk 3 of 12."""
        return f"{self.number}/{self.count}"


@dataclass
class ChunkAnswer:
    labels: list[CheckedLabel] = field(default_factory=list)
    check_log: list[CheckLogEntry] = field(default_factory=list)
    answer_status: str | None = None
    requests_sent: int = 0
    error: str | None = None
    seconds: float = 0.0


@dataclass
class DocumentResult:
    """Everything the run needs from one document, after all its chunks are answered."""

    document_id: str
    status: str
    token_count: int | None = None
    chunk_count: int = 0
    chunk_tokens: int | None = None
    chunking_level: str | None = None
    full_text_request: bool = False
    labels: list[CheckedLabel] = field(default_factory=list)
    check_log: list[CheckLogEntry] = field(default_factory=list)
    answer_statuses: list[str] = field(default_factory=list)
    requests_sent: int = 0
    error: str | None = None
    seconds: float = 0.0  # preparation plus the time of its requests (which overlap with other documents)

    def log_entry(self) -> dict[str, Any]:
        """One line of documents.jsonl (no text, no labels)."""
        return {
            "document_id": self.document_id,
            "status": self.status,
            "token_count": self.token_count,
            "chunk_count": self.chunk_count,
            "chunk_tokens": self.chunk_tokens,
            "chunking_level": self.chunking_level,
            "full_text_request": self.full_text_request,
            "labels_kept": len(self.labels),
            "answer_statuses": self.answer_statuses,
            "requests_sent": self.requests_sent,
            "error": self.error,
            "seconds": round(self.seconds, 1),
        }


PROMPT_V2_EXPERIMENT = "config/experiments/prompt_v2.yaml"


def unsupported_settings(settings: dict[str, Any]) -> str | None:
    """Why B cannot run with these settings, or None if it can.

    B never fills `{examples}`: its one-chunk papers send F's prompt, which is v4 by
    default, and examples for chunks were never built (B is no longer developed).
    """
    paths = settings["paths"]
    for template_path in (paths["prompt_extraction"], paths["prompt_extraction_chunked"]):
        if ExtractionPrompt.from_settings(settings, template_path).uses_examples:
            return (f"Pipeline B does not support prompts with training examples ({template_path.name}). "
                    "It is no longer developed (SPEC.md, section 13, 2026-11-19); run it with its last "
                    f"setting: --experiment {PROMPT_V2_EXPERIMENT} (or an experiment file built on it).")
    return None


class ParallelPipeline:
    """Prepares documents (main thread) and answers chunks (safe to call from several threads)."""

    def __init__(self, settings: dict[str, Any], client: LLMClient, document_dir: Path,
                 tokenizer: BaseTokenizer) -> None:
        """`tokenizer` must have `chunking.max_chunk_tokens` as its max_tokens, because the
        chunker cuts at that size (see chunking.load_tokenizer)."""
        reason = unsupported_settings(settings)
        if reason is not None:
            raise ValueError(reason)
        self.client = client
        self.document_dir = document_dir
        self.model: str = settings["models"]["chunked"]
        self.chunk_prompt = ExtractionPrompt.from_settings(settings, settings["paths"]["prompt_extraction_chunked"])
        self.full_text_prompt = ExtractionPrompt.from_settings(settings)  # pipeline F's prompt
        self.one_chunk_as_full_text: bool = settings["parallel"]["one_chunk_as_full_text"]
        self.retry_template = read_retry_template(settings["paths"]["prompt_retry_invalid_json"])
        self.rules = LabelRules.from_settings(settings)
        self.checker_config = CheckerConfig.from_settings(settings)
        self.merge_config = MergeConfig.from_settings(settings)
        self.chunking_config = ChunkingConfig.from_settings(settings)
        self.packing_config = PackingConfig.from_settings(settings)
        self._tokenizer = tokenizer
        self._separator_tokens = tokenizer.count_tokens(PARAGRAPH_SEPARATOR)

    def prepare(self, document_id: str) -> PreparedDocument:
        """Chunks of one document. Runs before any request, so the dry run shows the real chunks."""
        started = time.perf_counter()
        path = self.document_dir / f"{document_id}{DOCUMENT_SUFFIX}"
        if not path.exists():
            logger.warning("%s: not converted (%s missing), empty prediction", document_id, path)
            return PreparedDocument(document_id, STATUS_NOT_CONVERTED)
        document = load_document(path)
        text = full_text(document)
        token_count = self._tokenizer.count_tokens(text)
        pieces = chunk_document(document, self._tokenizer, self.chunking_config)
        packed = pack_chunks(pieces.chunks, token_count, self.packing_config, self._separator_tokens)
        if packed.capped:
            logger.warning("%s: %d chunks of up to %d tokens (chunked_part_tokens), more than %d",
                           document_id, len(packed.chunks), packed.chunk_tokens,
                           self.packing_config.max_chunks_per_document)
        chunks = packed.chunks
        full_text_request = self.one_chunk_as_full_text and len(chunks) == 1
        if full_text_request:
            # The chunker's text differs slightly from F's (e.g. no page markers); send F's
            # text, so the request and its cache key are exactly F's.
            chunks = [replace(chunks[0], text=text, token_count=token_count)]
        return PreparedDocument(document_id, STATUS_DONE, token_count, pieces.level, chunks,
                                full_text_request, packed.chunk_tokens, packed.enlarged, packed.capped,
                                time.perf_counter() - started)

    def answer_chunk(self, request: ChunkRequest) -> ChunkAnswer:
        """One request and the checker with the text of this chunk only."""
        started = time.perf_counter()
        answer = ChunkAnswer()
        prompt = self.full_text_prompt if request.full_text_request else self.chunk_prompt
        messages = prompt.messages(request.document_id, request.call, request.text)
        try:
            outcome = request_answer(self.client, self.model, messages, self.retry_template)
        except openai.APIError as error:
            # Errors that retries did not fix: the document gets an empty prediction.
            logger.error("%s chunk %s: request failed (%s)", request.document_id, request.call,
                         describe_error(error))
            answer.error = describe_error(error)
            answer.seconds = time.perf_counter() - started
            return answer
        answer.answer_status = outcome.status
        answer.requests_sent = outcome.requests_sent
        answer.labels, answer.check_log = check_answer(outcome.answer.labels, request.text, self.rules,
                                                       self.checker_config, request.document_id, request.call)
        answer.seconds = time.perf_counter() - started
        return answer

    def finish_document(self, prepared: PreparedDocument, answers: list[ChunkAnswer]) -> DocumentResult:
        """Merge the answers of all chunks (in chunk order) into the document's labels."""
        result = DocumentResult(
            prepared.document_id, prepared.status, prepared.token_count, len(prepared.chunks),
            prepared.chunk_tokens, prepared.chunking_level, prepared.full_text_request,
            answer_statuses=[answer.answer_status for answer in answers if answer.answer_status is not None],
            requests_sent=sum(answer.requests_sent for answer in answers),
            seconds=prepared.seconds + sum(answer.seconds for answer in answers),
        )
        for answer in answers:
            result.check_log.extend(answer.check_log)
        errors = [answer.error for answer in answers if answer.error is not None]
        if errors:
            result.status = STATUS_FAILED
            result.error = errors[0]
            return result
        if prepared.status != STATUS_DONE:
            return result
        result.labels, merge_log = merge_document([answer.labels for answer in answers], self.merge_config,
                                                  self.checker_config, prepared.document_id)
        result.check_log.extend(merge_log)
        return result


def chunk_requests(prepared: PreparedDocument) -> list[ChunkRequest]:
    count = len(prepared.chunks)
    return [ChunkRequest(prepared.document_id, number, count, chunk.text, prepared.full_text_request)
            for number, chunk in enumerate(prepared.chunks, start=1)]


# --- statistics and printing ----------------------------------------------------------------

def chunk_statistics(prepared: list[PreparedDocument], packing_config: PackingConfig) -> dict[str, Any]:
    """Numbers to check the request estimate before a real run (printed by the dry run too)."""
    converted = [document for document in prepared if document.status == STATUS_DONE]
    counts = [len(document.chunks) for document in converted]
    return {
        "max_chunk_tokens": packing_config.max_chunk_tokens,
        "max_chunks_per_document": packing_config.max_chunks_per_document,
        "chunked_part_tokens": packing_config.max_part_tokens,
        "chunks_total": sum(counts),
        "chunks_per_document_median": median(counts) if counts else 0,
        "chunks_per_document_max": max(counts, default=0),
        "documents_in_one_chunk": sum(1 for count in counts if count == 1),
        "documents_sent_as_full_text": sum(1 for document in converted if document.full_text_request),
        "enlarged": {document.document_id: document.chunk_tokens for document in converted if document.enlarged},
        "capped": {document.document_id: len(document.chunks) for document in converted if document.capped},
    }


def print_chunk_statistics(statistics: dict[str, Any], document_count: int) -> None:
    print(f"\nChunks: {statistics['chunks_total']} in total (= requests without retries) for "
          f"{document_count} documents; per document median {statistics['chunks_per_document_median']}, "
          f"max {statistics['chunks_per_document_max']}; {statistics['documents_in_one_chunk']} documents "
          f"in one chunk, {statistics['documents_sent_as_full_text']} of them sent as pipeline F's "
          f"request (full text, prompt_extraction)")
    enlarged = statistics["enlarged"]
    print(f"Chunk size enlarged (above {statistics['max_chunk_tokens']} tokens): {len(enlarged)} documents"
          + "".join(f"\n  {document_id}: {tokens} tokens" for document_id, tokens in enlarged.items()))
    if statistics["capped"]:
        print(f"At the cap of {statistics['chunked_part_tokens']} tokens, more than "
              f"{statistics['max_chunks_per_document']} chunks: {statistics['capped']}")


def describe_document(result: DocumentResult) -> str:
    """Progress line text for one finished document."""
    if result.status != STATUS_DONE:
        return f"{result.status}{': ' + result.error if result.error else ''}"
    checked = sum(1 for entry in result.check_log if entry.check in CALL_CHECKS)
    return (f"{result.chunk_count} chunks of up to {result.chunk_tokens} tokens, {result.token_count} tokens, "
            f"{checked} labels checked, {len(result.labels)} kept, {result.seconds:.0f} s")


def build_summary(
    results: dict[str, DocumentResult],
    document_ids: list[str],
    statistics: dict[str, Any],
    check_entries: list[dict[str, Any]],
    client: LLMClient,
    settings: dict[str, Any],
    times: dict[str, float],
    parallel_requests: int,
    stop_reason: str | None,
) -> dict[str, Any]:
    """Numbers for summary.json; everything the team reports back."""
    request_stats = client.stats.as_dict()
    api_requests = request_stats["api_requests"]
    return {
        "pipeline": "parallel",
        "model": settings["models"]["chunked"],
        "dry_run": client.dry_run,
        "verbatim_check_settings": settings["verbatim_check"],
        "merge_settings": settings["merge"],
        "parallel_settings": settings["parallel"],
        "documents": len(document_ids),
        "documents_finished": len(results),
        "stopped": stop_reason,
        "chunks": statistics,
        "not_converted": sorted(result.document_id for result in results.values()
                                if result.status == STATUS_NOT_CONVERTED),
        "failed": {result.document_id: result.error for result in results.values()
                   if result.status == STATUS_FAILED},
        "answer_statuses": dict(Counter(status for result in results.values()
                                        for status in result.answer_statuses)),
        "parallel_requests": parallel_requests,
        "preparation_seconds": round(times["preparation"], 1),
        "extraction_seconds": round(times["extraction"], 1),
        "wall_clock_seconds": round(times["preparation"] + times["extraction"], 1),
        "wall_clock_seconds_per_request": round(times["extraction"] / api_requests, 2) if api_requests else None,
        "requests": request_stats,
        "verbatim_check": drop_statistics(check_entries, REPORTED_CHECKS),
    }


def print_summary(summary: dict[str, Any], client: LLMClient) -> None:
    if summary["not_converted"]:
        print(f"Not converted (empty prediction): {len(summary['not_converted'])}")
    if summary["failed"]:
        print(f"Failed (empty prediction): {summary['failed']}")
    print(f"Answers: {summary['answer_statuses']}")
    overall = summary["verbatim_check"]["all_fields"]
    shares = ", ".join(f"{check} {share:.1%}" for check, share in overall["dropped_share"].items())
    print(f"Verbatim checker and vote: {overall['checked']} labels checked, dropped per check: {shares}")
    print(client.summary())
    print(f"Wall-clock time: {summary['wall_clock_seconds']:.0f} s (preparing chunks "
          f"{summary['preparation_seconds']:.0f} s, requests {summary['extraction_seconds']:.0f} s with up to "
          f"{summary['parallel_requests']} parallel requests"
          + (f", {summary['wall_clock_seconds_per_request']} s per request"
             if summary["wall_clock_seconds_per_request"] is not None else "") + ")")


# --- the run ---------------------------------------------------------------------------------

def run_parallel(
    settings: dict[str, Any],
    document_ids: list[str],
    client: LLMClient,
    run_dir: Path,
    document_dir: Path,
    tokenizer: BaseTokenizer,
) -> dict[str, Any]:
    """Process all documents, write the run files and (unless stopped or dry run) the predictions.

    Run files as in pipeline F: documents.jsonl and verbatim_check_log.jsonl (written as
    documents finish, so a stopped run keeps them), summary.json, predictions.jsonl + zip.
    """
    pipeline = ParallelPipeline(settings, client, document_dir, tokenizer)
    print(f"{len(document_ids)} documents, model {pipeline.model}, merge.min_chunks_per_label "
          f"{pipeline.merge_config.min_chunks_per_label}, verbatim checker "
          f"{'on' if pipeline.checker_config.enabled else 'off'}")

    started = time.perf_counter()
    prepared: dict[str, PreparedDocument] = {}
    for position, document_id in enumerate(document_ids, start=1):
        document = pipeline.prepare(document_id)
        prepared[document_id] = document
        if document.status == STATUS_DONE:
            as_full_text = ", sent as F's request" if document.full_text_request else ""
            print(f"[{position}/{len(document_ids)}] chunked {document_id}: {len(document.chunks)} chunks of "
                  f"up to {document.chunk_tokens} tokens ({document.token_count} tokens, "
                  f"{document.chunking_level}{as_full_text})", flush=True)
    preparation_seconds = time.perf_counter() - started
    statistics = chunk_statistics(list(prepared.values()), pipeline.packing_config)
    print_chunk_statistics(statistics, len(document_ids))

    requests = {request.request_id: request
                for document in prepared.values() for request in chunk_requests(document)}
    answers: dict[str, dict[int, ChunkAnswer]] = defaultdict(dict)
    results: dict[str, DocumentResult] = {}
    check_entries: list[dict[str, Any]] = []
    check_log_path = run_dir / CHECK_LOG_NAME
    document_log_path = run_dir / DOCUMENT_LOG_NAME

    def finish(document_id: str) -> None:
        document = prepared[document_id]
        chunk_answers = [answers[document_id][number] for number in range(1, len(document.chunks) + 1)]
        result = pipeline.finish_document(document, chunk_answers)
        results[document_id] = result
        entries = [entry.as_dict() for entry in result.check_log]
        check_entries.extend(entries)
        append_jsonl(check_log_path, entries)
        append_jsonl(document_log_path, [result.log_entry()])
        print(f"[{len(results)}/{len(document_ids)}] {document_id}: {describe_document(result)}", flush=True)

    # Documents without requests (not converted, or no text at all) are finished at once.
    for document_id, document in prepared.items():
        if not document.chunks:
            finish(document_id)

    def on_answer(request_id: str, answer: ChunkAnswer) -> None:
        """Called in the main thread as answers arrive; merges a document when it is complete."""
        request = requests[request_id]
        answers[request.document_id][request.number] = answer
        if len(answers[request.document_id]) == request.count:
            finish(request.document_id)

    parallel_requests = client.max_parallel_requests()
    print(f"\n{len(requests)} chunk requests, up to {parallel_requests} at the same time", flush=True)
    started = time.perf_counter()
    run = run_in_parallel(list(requests), lambda request_id: pipeline.answer_chunk(requests[request_id]),
                          parallel_requests, STOP_ERRORS, client.request_stop, on_answer)
    extraction_seconds = time.perf_counter() - started

    times = {"preparation": preparation_seconds, "extraction": extraction_seconds}
    summary = build_summary(results, document_ids, statistics, check_entries, client, settings, times,
                            parallel_requests, run.stop_reason)
    (run_dir / SUMMARY_NAME).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(summary, client)

    if run.stop_reason is not None or client.dry_run:
        return summary
    records = [build_record(ID_FIELD, document_id, results[document_id].labels, pipeline.rules)
               for document_id in sorted(document_ids)]
    paths = settings["paths"]
    zip_path = write_predictions(records, document_ids, run_dir,
                                 [paths["schema_core"], paths["schema_biodiversity"]])
    print(f"Written: {zip_path}")
    return summary
