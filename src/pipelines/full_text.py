"""Pipeline F: the whole document in one LLM call (SPEC.md, section 3, step 3F).

Per document:
1. load the converted document (Docling JSON from src/pdf_to_text.py) and build its full
   text with page markers;
2. if the text has more than `models.full_text_part_tokens` tokens, split it into as few
   parts as possible with the chunker from M1 (never truncated silently; such documents
   are reported separately);
3. one call per part with `models.full_text` and the prompt of section 6 (prompt v4 also
   gets the labels of the document's most similar training papers, few_shot_examples.py);
4. the verbatim checker (C1 to C4) right after every call, with the text of that part;
5. merge the parts as in pipeline B (C5: verified quotes from any part count).

Documents are processed in parallel (src/parallel_runner.py), as many as the LLM client
may send requests at the same time. The output is sorted by id, so it does not depend
on the order in which documents finish.

Counting tokens: the tokenizer of `chunking.tokenizer` (Qwen) is used, as in the M1
statistics. DeepSeek counts a little differently; `full_text_part_tokens` leaves room.
"""
import json
import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openai
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.types.doc import DoclingDocument

from api_key_pool import MissingApiKeyError, NoUsableApiKeyError
from chunking import ChunkingConfig, chunk_document
from data import ID_FIELD
from few_shot_examples import FewShotExamples
from llm_answer import AnswerOutcome, ExtractionAnswer, read_retry_template, request_answer
from llm_client import LLMClient, describe_error
from output import write_predictions
from parallel_runner import run_in_parallel
from pdf_to_text import DOCUMENT_SUFFIX, full_text, load_document
from prompts import ExtractionPrompt
from rate_limiter import DailyLimitReachedError
from verbatim_checker import (
    CHECK_ENOUGH_EVIDENCE,
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

STATUS_DONE = "done"
STATUS_NOT_CONVERTED = "not_converted"   # no Docling JSON: empty prediction
STATUS_FAILED = "failed"                 # API error after all retries: empty prediction

CHECK_LOG_NAME = "verbatim_check_log.jsonl"
DOCUMENT_LOG_NAME = "documents.jsonl"
SUMMARY_NAME = "summary.json"
CALL_LOG_NAME = "llm_calls.jsonl"

# Errors after which no further request can succeed today: stop the run cleanly.
STOP_ERRORS = (DailyLimitReachedError, NoUsableApiKeyError, MissingApiKeyError)


def part_call(number: int, part_count: int) -> str:
    """The call name of one part, e.g. "2/3"; it is part of the prompt (and the cache key)."""
    return f"{number}/{part_count}"


@dataclass
class DocumentResult:
    """Everything the run needs from one document; built in a worker thread."""

    document_id: str
    status: str
    token_count: int | None = None
    part_count: int = 0
    labels: list[CheckedLabel] = field(default_factory=list)
    check_log: list[CheckLogEntry] = field(default_factory=list)
    answer_statuses: list[str] = field(default_factory=list)
    requests_sent: int = 0
    error: str | None = None
    seconds: float = 0.0

    def log_entry(self) -> dict[str, Any]:
        """One line of documents.jsonl (no text, no labels)."""
        return {
            "document_id": self.document_id,
            "status": self.status,
            "token_count": self.token_count,
            "part_count": self.part_count,
            "labels_kept": len(self.labels),
            "answer_statuses": self.answer_statuses,
            "requests_sent": self.requests_sent,
            "error": self.error,
            "seconds": round(self.seconds, 1),
        }


class FullTextPipeline:
    """Processes one document at a time; safe to call from several threads."""

    def __init__(
        self,
        settings: dict[str, Any],
        client: LLMClient,
        document_dir: Path,
        tokenizer: BaseTokenizer,
        examples: FewShotExamples | None = None,
    ) -> None:
        """`tokenizer` must have `models.full_text_part_tokens` as its max_tokens, because
        the chunker splits at that size (see chunking.load_tokenizer). `examples` is
        needed if and only if the prompt has an {examples} section (v4)."""
        self.client = client
        self.document_dir = document_dir
        self.model: str = settings["models"]["full_text"]
        self.part_tokens: int = settings["models"]["full_text_part_tokens"]
        self.prompt = ExtractionPrompt.from_settings(settings)
        if self.prompt.uses_examples and examples is None:
            raise ValueError("the prompt has an {examples} section, but no training examples were given")
        self.examples = examples if self.prompt.uses_examples else None
        self.retry_template = read_retry_template(settings["paths"]["prompt_retry_invalid_json"])
        self.rules = LabelRules.from_settings(settings)
        self.checker_config = CheckerConfig.from_settings(settings)
        chunking = settings["chunking"]
        self.part_config = ChunkingConfig(
            max_chunk_tokens=self.part_tokens,
            chunk_overlap_tokens=chunking["chunk_overlap_tokens"],
            min_headings=chunking["min_headings"],
        )
        self._tokenizer = tokenizer
        # HuggingFace tokenizers can fail when used by several threads at once.
        self._tokenizer_lock = threading.Lock()

    def split_into_parts(self, document: DoclingDocument) -> tuple[list[str], int]:
        """The full text as one part, or chunker parts if it is too long. Returns the token count too."""
        text = full_text(document)
        with self._tokenizer_lock:
            token_count = self._tokenizer.count_tokens(text)
            if token_count <= self.part_tokens:
                return [text], token_count
            result = chunk_document(document, self._tokenizer, self.part_config)
        logger.info("Document with %d tokens split into %d parts (%s chunking, %d splits)",
                    token_count, len(result.chunks), result.level, len(result.splits))
        return [chunk.text for chunk in result.chunks], token_count

    def check_part(
        self, document_id: str, call: str, part_text: str, examples: str | None
    ) -> tuple[list[CheckedLabel], list[CheckLogEntry], AnswerOutcome[ExtractionAnswer]]:
        """One call for one part plus checks C1 to C4. Also used by src/verification.py to
        rebuild the labels of an earlier run from the cache."""
        messages = self.prompt.messages(document_id, call, part_text, examples)
        outcome = request_answer(self.client, self.model, messages, self.retry_template)
        checked, check_log = check_answer(outcome.answer.labels, part_text, self.rules,
                                          self.checker_config, document_id, call)
        return checked, check_log, outcome

    def process_document(self, document_id: str) -> DocumentResult:
        """Calls, checks and merging for one document."""
        started = time.perf_counter()
        path = self.document_dir / f"{document_id}{DOCUMENT_SUFFIX}"
        if not path.exists():
            logger.warning("%s: not converted (%s missing), empty prediction", document_id, path)
            return DocumentResult(document_id, STATUS_NOT_CONVERTED)
        parts, token_count = self.split_into_parts(load_document(path))
        result = DocumentResult(document_id, STATUS_DONE, token_count, len(parts))
        examples = self.examples.text_for(document_id) if self.examples is not None else None
        verified: list[CheckedLabel] = []
        try:
            for number, part_text in enumerate(parts, start=1):
                call = part_call(number, len(parts))
                checked, check_log, outcome = self.check_part(document_id, call, part_text, examples)
                result.answer_statuses.append(outcome.status)
                result.requests_sent += outcome.requests_sent
                verified.extend(checked)
                result.check_log.extend(check_log)
        except openai.APIError as error:
            # Errors that retries did not fix (e.g. prompt too long): this document gets an
            # empty prediction, the others go on; the next run tries it again.
            logger.error("%s: request failed (%s), empty prediction", document_id, describe_error(error))
            result.status = STATUS_FAILED
            result.error = describe_error(error)
            result.labels = []
            result.seconds = time.perf_counter() - started
            return result
        result.labels, evidence_log = merge_labels(verified, self.checker_config, document_id)
        result.check_log.extend(evidence_log)
        result.seconds = time.perf_counter() - started
        return result


def append_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as file:
        for entry in entries:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def describe_document(result: DocumentResult) -> str:
    """Progress line text for one finished document."""
    if result.status != STATUS_DONE:
        return f"{result.status}{': ' + result.error if result.error else ''}"
    calls = "one call" if result.part_count == 1 else f"{result.part_count} parts"
    checked = sum(1 for entry in result.check_log if entry.check != CHECK_ENOUGH_EVIDENCE)
    return (f"{calls}, {result.token_count} tokens, {checked} labels checked, {len(result.labels)} kept, "
            f"answers: {', '.join(result.answer_statuses)}, {result.seconds:.0f} s")


def build_summary(
    results: dict[str, DocumentResult],
    document_ids: list[str],
    check_entries: list[dict[str, Any]],
    client: LLMClient,
    settings: dict[str, Any],
    wall_clock_seconds: float,
    parallel_requests: int,
    stop_reason: str | None,
) -> dict[str, Any]:
    """Numbers for summary.json; everything the team reports back."""
    done = [result for result in results.values() if result.status == STATUS_DONE]
    request_stats = client.stats.as_dict()
    api_requests = request_stats["api_requests"]
    return {
        "model": settings["models"]["full_text"],
        "dry_run": client.dry_run,
        "verbatim_check_settings": settings["verbatim_check"],
        "documents": len(document_ids),
        "documents_finished": len(results),
        "stopped": stop_reason,
        "documents_in_one_call": sum(1 for result in done if result.part_count == 1),
        "documents_in_parts": {result.document_id: result.part_count
                               for result in done if result.part_count > 1},
        "not_converted": sorted(result.document_id for result in results.values()
                                if result.status == STATUS_NOT_CONVERTED),
        "failed": {result.document_id: result.error for result in results.values()
                   if result.status == STATUS_FAILED},
        "answer_statuses": dict(Counter(status for result in results.values()
                                        for status in result.answer_statuses)),
        "parallel_requests": parallel_requests,
        "wall_clock_seconds": round(wall_clock_seconds, 1),
        "wall_clock_seconds_per_request": round(wall_clock_seconds / api_requests, 2) if api_requests else None,
        "requests": request_stats,
        "verbatim_check": drop_statistics(check_entries),
    }


def print_summary(summary: dict[str, Any], client: LLMClient) -> None:
    in_parts = summary["documents_in_parts"]
    parts_text = ", ".join(f"{document_id}: {parts} parts" for document_id, parts in in_parts.items())
    print(f"\n{summary['documents_in_one_call']} documents in one call, {len(in_parts)} in parts"
          + (f" ({parts_text})" if parts_text else ""))
    if summary["not_converted"]:
        print(f"Not converted (empty prediction): {len(summary['not_converted'])}")
    if summary["failed"]:
        print(f"Failed (empty prediction): {summary['failed']}")
    print(f"Answers: {summary['answer_statuses']}")
    overall = summary["verbatim_check"]["all_fields"]
    shares = ", ".join(f"{check} {share:.1%}" for check, share in overall["dropped_share"].items())
    print(f"Verbatim checker: {overall['checked']} labels checked, dropped per check: {shares}")
    print(client.summary())
    print(f"Wall-clock time: {summary['wall_clock_seconds']:.0f} s with up to "
          f"{summary['parallel_requests']} parallel requests"
          + (f", {summary['wall_clock_seconds_per_request']} s per request"
             if summary["wall_clock_seconds_per_request"] is not None else ""))


def run_full_text(
    settings: dict[str, Any],
    document_ids: list[str],
    client: LLMClient,
    run_dir: Path,
    document_dir: Path,
    tokenizer: BaseTokenizer,
    examples: FewShotExamples | None = None,
) -> dict[str, Any]:
    """Process all documents, write the run files and (unless stopped or dry run) the predictions.

    `examples`: the training examples of the split, needed for prompt v4.

    Run files: documents.jsonl and verbatim_check_log.jsonl (written as documents
    finish, so a stopped run keeps them), summary.json, predictions.jsonl + zip.
    """
    pipeline = FullTextPipeline(settings, client, document_dir, tokenizer, examples)
    parallel_requests = client.max_parallel_requests()
    print(f"{len(document_ids)} documents, model {pipeline.model}, up to {parallel_requests} "
          f"parallel requests, verbatim checker {'on' if pipeline.checker_config.enabled else 'off'}")
    check_log_path = run_dir / CHECK_LOG_NAME
    document_log_path = run_dir / DOCUMENT_LOG_NAME
    check_entries: list[dict[str, Any]] = []
    finished = 0

    def on_result(document_id: str, result: DocumentResult) -> None:
        nonlocal finished
        finished += 1
        entries = [entry.as_dict() for entry in result.check_log]
        check_entries.extend(entries)
        append_jsonl(check_log_path, entries)
        append_jsonl(document_log_path, [result.log_entry()])
        print(f"[{finished}/{len(document_ids)}] {document_id}: {describe_document(result)}", flush=True)

    started = time.perf_counter()
    run = run_in_parallel(document_ids, pipeline.process_document, parallel_requests,
                          STOP_ERRORS, client.request_stop, on_result)
    wall_clock_seconds = time.perf_counter() - started

    summary = build_summary(run.results, document_ids, check_entries, client, settings,
                            wall_clock_seconds, parallel_requests, run.stop_reason)
    (run_dir / SUMMARY_NAME).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(summary, client)

    if run.stop_reason is not None or client.dry_run:
        return summary
    records = [build_record(ID_FIELD, document_id, run.results[document_id].labels, pipeline.rules)
               for document_id in sorted(document_ids)]
    paths = settings["paths"]
    zip_path = write_predictions(records, document_ids, run_dir,
                                 [paths["schema_core"], paths["schema_biodiversity"]])
    print(f"Written: {zip_path}")
    return summary
