"""Verification run: one more call per paper on top of an existing F run (src/verification.py).

Per document:
1. rebuild what the source F run did, with F's own code and the source run's settings
   (its config.yaml): the same parts, the same prompts, the answers from the cache (a
   client in dry-run mode, so nothing is ever sent for the source run), the verbatim
   checker and merging. The rebuilt record must equal the source run's prediction;
   otherwise the document is reported as failed and keeps its source prediction;
2. the labels F kept, except habitat, become candidates C1, C2, ...;
3. one call per part that holds candidates (one per paper for papers in one call), with
   `verify.model`, the prompt `paths.prompt_verify`, the training examples of prompt v4,
   the candidates whose quote is in that part, and the text F sent for that part;
4. the decisions remove the dropped entries from the source prediction; habitat and all
   other fields stay as they are.

Failed documents (API error, source not rebuilt) keep their source prediction and are
listed under "failed" in summary.json.
"""
import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openai
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer

from data import ID_FIELD, index_by_id, read_jsonl
from few_shot_examples import FewShotExamples
from llm_answer import STATUS_DRY_RUN, read_retry_template, request_parsed
from llm_client import LLMClient, describe_error
from output import PREDICTIONS_FILENAME, drop_unscored_subfields, write_predictions
from parallel_runner import run_in_parallel
from pdf_to_text import DOCUMENT_SUFFIX, load_document
from pipelines.full_text import STOP_ERRORS, SUMMARY_NAME, FullTextPipeline, append_jsonl, part_call
from runs import load_run_settings
from verbatim_checker import CheckedLabel, build_record, merge_labels
from verification import (
    DROP,
    EMPTY_VERIFY_ANSWER,
    NOTE_MISSING_DECISION,
    NOTE_NOT_APPLIED,
    Candidate,
    CandidateDecision,
    VerifySettings,
    apply_decisions,
    collect_candidates,
    decide,
    parse_verify_answer,
)
from verify_prompt import VerifyPrompt

logger = logging.getLogger(__name__)

VERIFY_LOG_NAME = "verify_log.jsonl"
SOURCES_NAME = "sources.json"
# A key only pipeline F writes into summary.json: tells an F run from a B run.
FULL_TEXT_SUMMARY_KEY = "documents_in_one_call"

STATUS_DONE = "done"
STATUS_NOT_CONVERTED = "not_converted"        # no Docling JSON: source prediction unchanged
STATUS_FAILED = "failed"                      # API error: source prediction unchanged
STATUS_SOURCE_NOT_REBUILT = "source_not_rebuilt"  # cache miss or different labels: unchanged


class SourceNotRebuiltError(RuntimeError):
    """The source run's labels of one document could not be rebuilt from the cache."""


@dataclass(frozen=True)
class SourceRun:
    """An existing F run: its folder, settings and predictions."""

    run_dir: Path
    settings: dict[str, Any]
    predictions: dict[str, dict[str, Any]]
    commit: str

    @classmethod
    def load(cls, run_dir: Path) -> "SourceRun":
        """Fails early for a folder that is not a finished pipeline F run."""
        summary_path = run_dir / SUMMARY_NAME
        predictions_path = run_dir / PREDICTIONS_FILENAME
        if not summary_path.exists() or not predictions_path.exists():
            raise FileNotFoundError(f"{run_dir} has no {SUMMARY_NAME} or {PREDICTIONS_FILENAME}: "
                                    "a stopped run or a dry run cannot be verified")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if FULL_TEXT_SUMMARY_KEY not in summary:
            raise ValueError(f"{run_dir} is not a pipeline F run")
        if summary.get("failed"):
            logger.warning("The source run has failed documents (empty predictions): %s", summary["failed"])
        commit_path = run_dir / "commit.txt"
        commit = commit_path.read_text(encoding="utf-8").strip() if commit_path.exists() else "unknown"
        predictions = index_by_id(read_jsonl(predictions_path), str(predictions_path))
        return cls(run_dir, load_run_settings(run_dir), predictions, commit)

    @property
    def document_ids(self) -> list[str]:
        return sorted(self.predictions)


@dataclass
class VerifyResult:
    """Everything the run needs from one document; built in a worker thread."""

    document_id: str
    status: str
    record: dict[str, Any]
    part_count: int = 0
    candidate_count: int = 0
    decisions: list[CandidateDecision] = field(default_factory=list)
    log: list[dict[str, Any]] = field(default_factory=list)
    answer_statuses: list[str] = field(default_factory=list)
    requests_sent: int = 0
    error: str | None = None
    seconds: float = 0.0


class VerifyPipeline:
    """Processes one document at a time; safe to call from several threads."""

    def __init__(
        self,
        settings: dict[str, Any],
        source: SourceRun,
        source_pipeline: FullTextPipeline,
        client: LLMClient,
        examples: FewShotExamples,
        source_examples: FewShotExamples | None,
        document_dir: Path,
    ) -> None:
        """`source_pipeline`: F built from the source run's settings with a dry-run client,
        so it only reads the cache. `examples`: the training examples for the verify
        prompt; `source_examples`: those of the source prompt (None if it has none)."""
        self.verify = VerifySettings.from_settings(settings)
        self.source = source
        self.source_pipeline = source_pipeline
        self.client = client
        self.examples = examples
        self.source_examples = source_examples
        self.document_dir = document_dir
        self.prompt = VerifyPrompt.from_file(settings["paths"]["prompt_verify"])
        self.retry_template = read_retry_template(settings["paths"]["prompt_retry_invalid_json"])

    def rebuild_candidates(self, document_id: str, parts: list[str]) -> list[Candidate]:
        """The candidates of one document, from the source run's cached answers."""
        examples = self.source_examples.text_for(document_id) if self.source_examples is not None else None
        labels_per_part: list[list[CheckedLabel]] = []
        for number, part_text in enumerate(parts, start=1):
            call = part_call(number, len(parts))
            checked, _, outcome = self.source_pipeline.check_part(document_id, call, part_text, examples)
            if outcome.status == STATUS_DRY_RUN:
                raise SourceNotRebuiltError(f"the source answer of part {call} is not in the cache")
            labels_per_part.append(checked)
        all_labels = [label for part_labels in labels_per_part for label in part_labels]
        kept, _ = merge_labels(all_labels, self.source_pipeline.checker_config, document_id)
        rebuilt = drop_unscored_subfields(build_record(ID_FIELD, document_id, kept, self.source_pipeline.rules))
        if rebuilt != self.source.predictions[document_id]:
            raise SourceNotRebuiltError("the labels rebuilt from the cache differ from the source predictions")
        return collect_candidates(labels_per_part, kept, self.verify.candidate_fields)

    def process_document(self, document_id: str) -> VerifyResult:
        started = time.perf_counter()
        source_record = self.source.predictions[document_id]
        result = VerifyResult(document_id, STATUS_DONE, source_record)
        path = self.document_dir / f"{document_id}{DOCUMENT_SUFFIX}"
        if not path.exists():
            logger.warning("%s: not converted (%s missing), source prediction unchanged", document_id, path)
            result.status = STATUS_NOT_CONVERTED
            return result
        parts, _ = self.source_pipeline.split_into_parts(load_document(path))
        result.part_count = len(parts)
        try:
            candidates = self.rebuild_candidates(document_id, parts)
        except SourceNotRebuiltError as error:
            logger.error("%s: %s; source prediction unchanged", document_id, error)
            result.status, result.error = STATUS_SOURCE_NOT_REBUILT, str(error)
            return result
        result.candidate_count = len(candidates)
        examples = self.examples.text_for(document_id)
        decisions: list[CandidateDecision] = []
        log: list[dict[str, Any]] = []
        try:
            for index, part_text in enumerate(parts):
                part_candidates = [candidate for candidate in candidates if candidate.part_index == index]
                if not part_candidates:
                    continue  # nothing to check: no request
                call = part_call(index + 1, len(parts))
                messages = self.prompt.messages(examples, part_candidates, part_text)
                outcome = request_parsed(self.client, self.verify.model, messages, self.retry_template,
                                         parse_verify_answer, EMPTY_VERIFY_ANSWER)
                result.answer_statuses.append(outcome.status)
                result.requests_sent += outcome.requests_sent
                part_decisions = decide(part_candidates, outcome.answer, self.verify.applied_fields, document_id)
                decisions.extend(part_decisions)
                log.extend(decision.log_entry(document_id, call) for decision in part_decisions)
        except openai.APIError as error:
            logger.error("%s: request failed (%s), source prediction unchanged", document_id, describe_error(error))
            result.status, result.error = STATUS_FAILED, describe_error(error)
            result.seconds = time.perf_counter() - started
            return result
        result.decisions, result.log = decisions, log
        result.record = apply_decisions(source_record, decisions)
        result.seconds = time.perf_counter() - started
        return result


def decision_counts(results: dict[str, VerifyResult], candidate_fields: tuple[str, ...]) -> dict[str, Any]:
    """Per field: candidates, kept, dropped, and why some were kept without a "drop"."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for result in results.values():
        for decision in result.decisions:
            field_counts = counts[decision.candidate.field]
            field_counts["candidates"] += 1
            field_counts["dropped" if decision.decision == DROP else "kept"] += 1
            if decision.note == NOTE_MISSING_DECISION:
                field_counts["kept_without_valid_decision"] += 1
            if decision.note == NOTE_NOT_APPLIED and decision.model_decision == DROP:
                field_counts["drop_not_applied"] += 1
    names = ("candidates", "kept", "dropped", "kept_without_valid_decision", "drop_not_applied")
    return {field: {name: counts[field][name] for name in names} for field in candidate_fields}


def describe_result(result: VerifyResult) -> str:
    """Progress line text for one finished document."""
    if result.status != STATUS_DONE:
        return f"{result.status}{': ' + result.error if result.error else ''}"
    dropped = sum(1 for decision in result.decisions if decision.decision == DROP)
    return (f"{result.part_count} part(s), {result.candidate_count} candidates, {dropped} dropped, "
            f"answers: {', '.join(result.answer_statuses) or 'none'}, {result.seconds:.0f} s")


def write_sources(run_dir: Path, source: SourceRun, settings: dict[str, Any]) -> None:
    """sources.json: what this run builds on and what it used."""
    verify = VerifySettings.from_settings(settings)
    sources = {
        "source_run": str(source.run_dir),
        "source_commit": source.commit,
        "source_prompt": str(source.settings["paths"]["prompt_extraction"]),
        "source_model": source.settings["models"]["full_text"],
        "verify_prompt": str(settings["paths"]["prompt_verify"]),
        "verify_model": verify.model,
        "temperature": settings["llm"]["temperature"],
        "candidate_fields": list(verify.candidate_fields),
        "applied_fields": list(verify.applied_fields),
    }
    (run_dir / SOURCES_NAME).write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")


def run_verify(
    settings: dict[str, Any],
    source: SourceRun,
    client: LLMClient,
    run_dir: Path,
    document_dir: Path,
    tokenizer: BaseTokenizer,
    examples: FewShotExamples,
    source_examples: FewShotExamples | None,
) -> dict[str, Any]:
    """Verify all documents of the source run and write the run files.

    `tokenizer` must have the source run's `models.full_text_part_tokens` as max_tokens,
    so the parts are the source run's parts. Run files: sources.json, verify_log.jsonl
    (written as documents finish), summary.json, and unless stopped or a dry run
    predictions.jsonl + zip.
    """
    write_sources(run_dir, source, settings)
    source_client = LLMClient(source.settings, dry_run=True)  # cache only: never sends
    source_pipeline = FullTextPipeline(source.settings, source_client, document_dir, tokenizer, source_examples)
    pipeline = VerifyPipeline(settings, source, source_pipeline, client, examples, source_examples, document_dir)
    document_ids = source.document_ids
    parallel_requests = client.max_parallel_requests()
    print(f"{len(document_ids)} documents from {source.run_dir.name}, model {pipeline.verify.model}, "
          f"up to {parallel_requests} parallel requests, decisions applied for: "
          f"{', '.join(pipeline.verify.applied_fields)}")
    log_path = run_dir / VERIFY_LOG_NAME
    finished = 0

    def on_result(document_id: str, result: VerifyResult) -> None:
        nonlocal finished
        finished += 1
        append_jsonl(log_path, result.log)
        print(f"[{finished}/{len(document_ids)}] {document_id}: {describe_result(result)}", flush=True)

    started = time.perf_counter()
    run = run_in_parallel(document_ids, pipeline.process_document, parallel_requests,
                          STOP_ERRORS, client.request_stop, on_result)
    wall_clock_seconds = time.perf_counter() - started
    results = run.results
    request_stats = client.stats.as_dict()
    summary = {
        "pipeline": "verify",
        "source_run": str(source.run_dir),
        "model": pipeline.verify.model,
        "dry_run": client.dry_run,
        "documents": len(document_ids),
        "documents_finished": len(results),
        "stopped": run.stop_reason,
        "failed": {result.document_id: f"{result.status}: {result.error}" for result in results.values()
                   if result.status in (STATUS_FAILED, STATUS_SOURCE_NOT_REBUILT)},
        "not_converted": sorted(result.document_id for result in results.values()
                                if result.status == STATUS_NOT_CONVERTED),
        "calls": sum(len(result.answer_statuses) for result in results.values()),
        "answer_statuses": dict(Counter(status for result in results.values()
                                        for status in result.answer_statuses)),
        "decisions_per_field": decision_counts(results, pipeline.verify.candidate_fields),
        "parallel_requests": parallel_requests,
        "wall_clock_seconds": round(wall_clock_seconds, 1),
        "requests": request_stats,
    }
    (run_dir / SUMMARY_NAME).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(summary, client)

    if run.stop_reason is not None or client.dry_run:
        return summary
    records = [results[document_id].record for document_id in document_ids]
    paths = settings["paths"]
    zip_path = write_predictions(records, document_ids, run_dir, [paths["schema_core"], paths["schema_biodiversity"]])
    print(f"Written: {zip_path}")
    return summary


def print_summary(summary: dict[str, Any], client: LLMClient) -> None:
    if summary["failed"]:
        print(f"\nFailed (source prediction unchanged): {summary['failed']}")
    print(f"\nVerify calls: {summary['calls']}, answers: {summary['answer_statuses']}")
    print(f"{'field':20s} {'candidates':>10s} {'kept':>6s} {'dropped':>8s} {'no decision':>12s} {'drop not applied':>17s}")
    for field_name, counts in summary["decisions_per_field"].items():
        print(f"{field_name:20s} {counts['candidates']:10d} {counts['kept']:6d} {counts['dropped']:8d} "
              f"{counts['kept_without_valid_decision']:12d} {counts['drop_not_applied']:17d}")
    print(client.summary())
    print(f"Wall-clock time: {summary['wall_clock_seconds']:.0f} s")
