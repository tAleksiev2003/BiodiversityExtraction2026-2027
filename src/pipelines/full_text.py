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
5. field prompts (`field_prompts`, decomposed prompting after Khot et al. 2023): for a
   field with its own prompt, one more call per part with the same model, examples and
   text; its labels go through the same checker and REPLACE the main call's labels of
   that field. The main call is unchanged, so it still comes from the cache. If a field
   call fails (API error, or invalid JSON after the retry), the paper keeps the main
   call's labels of that field (counted in summary.json);
6. merge the parts as in pipeline B (C5: verified quotes from any part count).

After all documents: the post-processing rules of src/postprocessing.py (they only drop
labels), then the predictions and labels_with_confidence.jsonl (src/label_file.py) are written.

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
from label_file import LABEL_FILE_NAME, document_lines, field_prompt_call
from llm_answer import STATUS_INVALID, AnswerOutcome, ExtractionAnswer, read_retry_template, request_answer
from llm_client import LLMClient, describe_error
from output import write_predictions
from parallel_runner import run_in_parallel
from pdf_to_text import DOCUMENT_SUFFIX, full_text, load_document
from postprocessing import PostprocessResult, postprocess
from prompts import ExtractionPrompt
from rate_limiter import DailyLimitReachedError
from settings import REPO_ROOT
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


def field_prompt_paths(settings: dict[str, Any]) -> dict[str, Path]:
    """{field: template path} of the field prompts that are switched on (`field_prompts`,
    null = off). Relative paths are read from the repository root, like `paths`."""
    paths = {}
    for field_name, value in (settings.get("field_prompts") or {}).items():
        if value is None:
            continue
        path = Path(value)
        paths[field_name] = path if path.is_absolute() else (REPO_ROOT / path).resolve()
    return paths


class FieldPromptFailedError(RuntimeError):
    """A field call could not be answered: the paper keeps the main call's labels of that field."""


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
    # Field prompts (decomposed prompting); all empty when `field_prompts` is off.
    merged_before_postprocessing: list[CheckedLabel] = field(default_factory=list)
    replaced_labels: list[CheckedLabel] = field(default_factory=list)   # main-call labels a field prompt replaced
    field_prompt_fields: set[str] = field(default_factory=set)          # fields answered by their field prompt
    field_prompt_failures: dict[str, str] = field(default_factory=dict) # field -> why the main labels were kept
    field_requests_sent: dict[str, int] = field(default_factory=dict)

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
            "field_prompts_replaced": sorted(self.field_prompt_fields),
            "field_prompt_failures": self.field_prompt_failures,
            "field_prompt_requests_sent": self.field_requests_sent,
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
        self.field_prompts: dict[str, ExtractionPrompt] = {}
        for field_name, path in field_prompt_paths(settings).items():
            if field_name not in self.rules.fields():
                raise ValueError(f"field_prompts: {field_name!r} is not an output field")
            field_prompt = ExtractionPrompt.from_settings(settings, path)
            if field_prompt.uses_examples and examples is None:
                raise ValueError(f"the field prompt {path.name} has an {{examples}} section, "
                                 "but no training examples were given")
            self.field_prompts[field_name] = field_prompt
        self.field_prompt_names = {field_name: path.name for field_name, path in field_prompt_paths(settings).items()}
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

    def ask_field_prompt(
        self, field_name: str, document_id: str, parts: list[str], examples: str | None, result: DocumentResult
    ) -> tuple[list[CheckedLabel], list[CheckLogEntry]]:
        """One call per part with the field prompt, checks C1 to C4. Only labels of
        `field_name` are used; anything else the model adds is ignored. Raises
        FieldPromptFailedError if a call fails, so no part's answer is used half."""
        prompt = self.field_prompts[field_name]
        checked_labels: list[CheckedLabel] = []
        check_log: list[CheckLogEntry] = []
        for number, part_text in enumerate(parts, start=1):
            call = part_call(number, len(parts))
            try:
                outcome = request_answer(self.client, self.model, prompt.messages(document_id, call, part_text, examples),
                                         self.retry_template)
            except openai.APIError as error:
                raise FieldPromptFailedError(describe_error(error)) from error
            result.field_requests_sent[field_name] = result.field_requests_sent.get(field_name, 0) + outcome.requests_sent
            if outcome.status == STATUS_INVALID:
                raise FieldPromptFailedError(f"invalid JSON after the retry in part {call}")
            labels = [label for label in outcome.answer.labels if label.field == field_name]
            checked, log = check_answer(labels, part_text, self.rules, self.checker_config, document_id,
                                        field_prompt_call(field_name, call))
            checked_labels.extend(checked)
            check_log.extend(log)
        return checked_labels, check_log

    def apply_field_prompts(self, document_id: str, parts: list[str], examples: str | None,
                            verified: list[CheckedLabel], result: DocumentResult) -> list[CheckedLabel]:
        """The verified labels with every field that has a field prompt replaced by its answer."""
        for field_name in self.field_prompts:
            try:
                field_labels, field_log = self.ask_field_prompt(field_name, document_id, parts, examples, result)
            except FieldPromptFailedError as error:
                logger.warning("%s: field prompt for %s failed (%s), main call's labels kept",
                               document_id, field_name, error)
                result.field_prompt_failures[field_name] = str(error)
                continue
            main_labels = [label for label in verified if label.field == field_name]
            result.replaced_labels.extend(merge_labels(main_labels, self.checker_config, document_id)[0])
            verified = [label for label in verified if label.field != field_name] + field_labels
            result.check_log.extend(field_log)
            result.field_prompt_fields.add(field_name)
        return verified

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
        verified = self.apply_field_prompts(document_id, parts, examples, verified, result)
        result.labels, evidence_log = merge_labels(verified, self.checker_config, document_id)
        result.merged_before_postprocessing = list(result.labels)
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
    postprocessed: PostprocessResult,
    field_prompt_names: dict[str, str],
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
        "field_prompts": field_prompt_summary(results, field_prompt_names),
        **postprocessed.summary(),   # min_confidence, ecosystem_type_rule, service_rule
    }


def field_prompt_summary(results: dict[str, DocumentResult], field_prompt_names: dict[str, str]) -> dict[str, Any]:
    """Per field with a field prompt: requests sent, papers whose field call failed (they
    keep the main call's labels of that field) and papers whose labels were replaced."""
    summary = {}
    for field_name, prompt_name in field_prompt_names.items():
        failed = {result.document_id: result.field_prompt_failures[field_name]
                  for result in results.values() if field_name in result.field_prompt_failures}
        summary[field_name] = {
            "prompt": prompt_name,
            "requests": sum(result.field_requests_sent.get(field_name, 0) for result in results.values()),
            "papers_replaced": sum(1 for result in results.values() if field_name in result.field_prompt_fields),
            "failed": dict(sorted(failed.items())),
            "papers_with_fallback": len(failed),
        }
    return summary


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
    for field_name, numbers in summary["field_prompts"].items():
        print(f"Field prompt {numbers['prompt']} ({field_name}): {numbers['papers_replaced']} papers replaced, "
              f"{numbers['papers_with_fallback']} kept the main call's labels, {numbers['requests']} requests")
    confidence = summary["min_confidence"]
    if confidence["applied"]:
        print(f"min_confidence {confidence['thresholds']}: {confidence['labels_dropped']} labels dropped "
              f"in {confidence['papers_with_drops']} papers")
    ecosystem_type_rule = summary["ecosystem_type_rule"]
    if ecosystem_type_rule["applied"]:
        print(f"Ecosystem_type rule: {ecosystem_type_rule['ecosystem_type_entries_dropped']} ecosystem_type entries "
              f"dropped in {ecosystem_type_rule['papers_dropped']} papers (only habitats "
              f"{ecosystem_type_rule['land_only_habitats']}; terms {ecosystem_type_rule['terms_dropped']})")
    else:
        print(f"Ecosystem_type rule: {ecosystem_type_rule['reason']}")
    rule = summary["service_rule"]
    if rule["applied"]:
        print(f"Service rule: {rule['ecosystem_service_entries_dropped']} ecosystem_service entries "
              f"dropped in {rule['papers_dropped']} papers (no neighbour has services)")
    else:
        print(f"Service rule: {rule['reason']}")
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

    `examples`: the training examples of the split, needed for prompt v4; the service
    rule (src/postprocessing.py) reuses their neighbours and train references.

    Run files: documents.jsonl and verbatim_check_log.jsonl (written as documents
    finish, so a stopped run keeps them), labels_with_confidence.jsonl, summary.json,
    predictions.jsonl + zip.
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

    labels_by_document = {document_id: result.labels for document_id, result in run.results.items()}
    postprocessed = postprocess(labels_by_document, examples, settings)
    label_lines = [line for document_id in sorted(run.results)
                   for line in document_lines(document_id, run.results[document_id].check_log,
                                              run.results[document_id].merged_before_postprocessing,
                                              run.results[document_id].replaced_labels,
                                              run.results[document_id].field_prompt_fields, postprocessed)]
    append_jsonl(run_dir / LABEL_FILE_NAME, label_lines)

    summary = build_summary(run.results, document_ids, check_entries, client, settings,
                            wall_clock_seconds, parallel_requests, run.stop_reason, postprocessed,
                            pipeline.field_prompt_names)
    (run_dir / SUMMARY_NAME).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(summary, client)

    if run.stop_reason is not None or client.dry_run:
        return summary
    records = [build_record(ID_FIELD, document_id, postprocessed.labels_by_document[document_id], pipeline.rules)
               for document_id in sorted(document_ids)]
    paths = settings["paths"]
    zip_path = write_predictions(records, document_ids, run_dir,
                                 [paths["schema_core"], paths["schema_biodiversity"]])
    print(f"Written: {zip_path}")
    return summary
