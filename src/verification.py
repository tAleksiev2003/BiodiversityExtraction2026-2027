"""Verification step for pipeline F: keep or drop the labels of an existing F run.

Why (SPEC.md, section 13): F with prompt v4 and DeepSeek 4.1 proposes labels the database
does not give, most of all biodiversity_level, ecosystem_service and (less) ecosystem_type
and taxa. The verbatim checker only proves that a quote exists, not that the label fits
the document. A second call shows the model the labels F kept, with their quotes, and asks
for each one: would the database give this label to this document? The model never adds
or changes a label; it only answers "keep" or "drop".

This module holds the logic without any I/O: which labels become candidates, how the
answer is read, and how the decisions change a prediction record. The run itself (cache,
parallel requests, run folder) is in src/pipelines/verify.py.

Rules:
- habitat is never a candidate (103 of 113 habitat labels were right on dev);
- a candidate is one label as F kept it: a taxa candidate is the whole entry
  (species_group / scientific_name / german_name), ecosystem_type and ecosystem_service
  candidates are the category/term pair;
- `verify.candidate_fields`: the fields sent to the model (prompt verify_v1 has a rule
  for each); `verify.fields`: the fields whose decisions are applied. A field that is
  sent but not applied keeps all its labels; the prompts stay the same, so such a
  variant comes from the cache;
- a candidate without a valid decision is kept (and logged): an unclear answer must not
  delete labels.
"""
import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from llm_answer import parse_json_model
from verbatim_checker import CheckedLabel, LabelEntry

logger = logging.getLogger(__name__)

# Field name (not a label value): never checked, see the module docstring.
NEVER_CHECKED_FIELD = "habitat"
CANDIDATE_ID_PREFIX = "C"
KEEP = "keep"
DROP = "drop"
DECISIONS = (KEEP, DROP)

# Why a candidate was kept although the model did not say "drop" (verify_log.jsonl, "note").
NOTE_MISSING_DECISION = "no valid decision: kept"
NOTE_NOT_APPLIED = "field not in verify.fields: kept"


@dataclass(frozen=True)
class VerifySettings:
    """The `verify` section of settings.yaml, checked once at the start of a run."""

    model: str
    candidate_fields: tuple[str, ...]
    applied_fields: tuple[str, ...]

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> "VerifySettings":
        verify = settings["verify"]
        candidate_fields = tuple(verify["candidate_fields"])
        applied_fields = tuple(verify["fields"])
        if NEVER_CHECKED_FIELD in candidate_fields:
            raise ValueError(f"verify.candidate_fields must not contain {NEVER_CHECKED_FIELD}: it is never checked")
        not_sent = set(applied_fields) - set(candidate_fields)
        if not_sent:
            raise ValueError(f"verify.fields {sorted(not_sent)} are not in verify.candidate_fields, "
                             "so the model never sees them")
        return cls(verify["model"], candidate_fields, applied_fields)


@dataclass(frozen=True)
class Candidate:
    """One label F kept, as the model sees it."""

    candidate_id: str      # "C1", "C2", ... unique per document
    field: str
    entry: LabelEntry      # exactly as in the prediction record
    quote: str | None      # the quote that passed the verbatim checker
    part_index: int        # 0-based: the part of the document whose text holds the quote


def collect_candidates(
    labels_per_part: list[list[CheckedLabel]],
    kept_labels: list[CheckedLabel],
    candidate_fields: tuple[str, ...],
) -> list[Candidate]:
    """The candidates of one document, in the order of the prediction record.

    `labels_per_part`: the labels that passed C1 to C4 in each part; `kept_labels`: the
    labels after merging (C5), i.e. what the source run wrote. A label is shown with the
    first part that verified it, together with that part's quote, so the model gets the
    text in which the quote occurs.
    """
    candidates: list[Candidate] = []
    for label in kept_labels:
        if label.field not in candidate_fields:
            continue
        part_index, quote = first_verified_part(label, labels_per_part)
        candidate_id = f"{CANDIDATE_ID_PREFIX}{len(candidates) + 1}"
        candidates.append(Candidate(candidate_id, label.field, label.entry, quote, part_index))
    return candidates


def first_verified_part(label: CheckedLabel, labels_per_part: list[list[CheckedLabel]]) -> tuple[int, str | None]:
    """Index and quote of the first part with a verified label of the same key."""
    for part_index, part_labels in enumerate(labels_per_part):
        for part_label in part_labels:
            if part_label.key == label.key:
                return part_index, part_label.quote
    raise ValueError(f"label {label.key} was kept but not verified in any part")


# --- the answer ---

class VerifyDecision(BaseModel):
    """One decision as the model wrote it. Everything is optional here: a single broken
    decision is handled per candidate (kept), not by asking again for the whole answer."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    decision: str | None = None
    reason: str | None = None


class VerifyAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    decisions: list[VerifyDecision] = []


EMPTY_VERIFY_ANSWER = VerifyAnswer()


def parse_verify_answer(content: str) -> VerifyAnswer:
    """Raises AnswerParseError for invalid JSON or a wrong structure (then F's one retry)."""
    return parse_json_model(content, VerifyAnswer)


@dataclass(frozen=True)
class CandidateDecision:
    """What happened to one candidate; one line of verify_log.jsonl."""

    candidate: Candidate
    model_decision: str | None   # "keep", "drop" or None (missing or invalid)
    reason: str
    decision: str                # what was applied: "keep" or "drop"
    note: str                    # why it was kept against or without the model's word

    def log_entry(self, document_id: str, call: str) -> dict[str, Any]:
        return {
            "document_id": document_id,
            "call": call,
            "candidate_id": self.candidate.candidate_id,
            "field": self.candidate.field,
            "value": self.candidate.entry,
            "quote": self.candidate.quote,
            "decision": self.decision,
            "model_decision": self.model_decision,
            "reason": self.reason,
            "note": self.note,
        }


def model_decisions(answer: VerifyAnswer) -> dict[str, tuple[str, str]]:
    """{candidate id: (decision, reason)} for every valid decision; the first one wins
    if an id is answered twice."""
    decisions: dict[str, tuple[str, str]] = {}
    for item in answer.decisions:
        if item.id is None or item.decision is None:
            continue
        decision = item.decision.strip().lower()
        candidate_id = item.id.strip()
        if decision not in DECISIONS:
            continue
        if candidate_id in decisions:
            logger.info("Candidate %s answered twice; the first decision counts", candidate_id)
            continue
        decisions[candidate_id] = (decision, item.reason or "")
    return decisions


def decide(
    candidates: list[Candidate], answer: VerifyAnswer, applied_fields: tuple[str, ...], document_id: str
) -> list[CandidateDecision]:
    """One decision per candidate: the model's, unless it is missing (kept) or its field
    is not applied (kept)."""
    given = model_decisions(answer)
    results = []
    for candidate in candidates:
        if candidate.candidate_id not in given:
            logger.info("%s: no valid decision for %s (%s), kept", document_id,
                        candidate.candidate_id, candidate.field)
            results.append(CandidateDecision(candidate, None, "", KEEP, NOTE_MISSING_DECISION))
            continue
        model_decision, reason = given[candidate.candidate_id]
        if candidate.field not in applied_fields:
            results.append(CandidateDecision(candidate, model_decision, reason, KEEP, NOTE_NOT_APPLIED))
            continue
        results.append(CandidateDecision(candidate, model_decision, reason, model_decision, ""))
    return results


def apply_decisions(source_record: dict[str, Any], decisions: list[CandidateDecision]) -> dict[str, Any]:
    """A copy of the source record without the dropped entries; everything else unchanged.

    A dropped taxa entry disappears as a whole; entries are unique per field (merging
    keeps one label per key), so removing by equality removes exactly that entry.
    """
    record = {key: list(value) if isinstance(value, list) else value for key, value in source_record.items()}
    for result in decisions:
        if result.decision != DROP:
            continue
        field = result.candidate.field
        record[field] = [entry for entry in record[field] if entry != result.candidate.entry]
    return record
