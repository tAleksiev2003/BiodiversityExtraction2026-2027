"""Rules that run after the verbatim checker and before the predictions are written.

Both rules only remove labels; they never add or change one. They need no request, so
their effect can be compared from the cache. `postprocess()` runs them in this order:

1. `postprocess.min_confidence` ({field: threshold}, empty = off): drop a label whose
   confidence (prompt v5, field prompts) is below its field's threshold. A label without
   a confidence is kept: older prompts give none, and a missing value says nothing about
   the label. Thresholds come from src/scripts/sweep_confidence.py (Lipton et al. 2014).
2. `postprocess.drop_services_without_neighbour_support` (SPEC.md, section 13,
   2026-11-19): if none of a paper's training neighbours (neighbours_<split>.json, the
   examples of prompt v4) has an ecosystem_service entry in the train references, all
   ecosystem_service labels of the paper are dropped. Why: in train, papers whose 3
   nearest train papers have no services have services themselves in only 6.7 % of cases
   (46 % otherwise), and the model proposes many false services. A paper without
   neighbours (no first-pages text) keeps its services, because there is no evidence
   either way. Without it: config/experiments/no_service_rule.yaml.
"""
import logging
from dataclasses import dataclass, field
from typing import Any

from few_shot_examples import FewShotExamples
from prompts import ECOSYSTEM_SERVICE_FIELD
from verbatim_checker import CheckedLabel

logger = logging.getLogger(__name__)

RULE_MIN_CONFIDENCE = "min_confidence"
RULE_SERVICES = "service_rule"

# Why the service rule did nothing in a run (written to summary.json).
NOT_APPLIED_SWITCHED_OFF = "switched off (postprocess.drop_services_without_neighbour_support: false)"
NOT_APPLIED_NO_EXAMPLES = "not applied: the prompt has no training examples, so no neighbours were loaded"

LabelsByDocument = dict[str, list[CheckedLabel]]


def postprocess_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Runs made before a rule existed have no such entry in their copied config.yaml; for
    them the rule is off, so src/pipelines/verify.py rebuilds them unchanged."""
    return settings.get("postprocess", {})


def service_rule_enabled(settings: dict[str, Any]) -> bool:
    return bool(postprocess_settings(settings).get("drop_services_without_neighbour_support", False))


def min_confidence_thresholds(settings: dict[str, Any]) -> dict[str, float]:
    return dict(postprocess_settings(settings).get("min_confidence") or {})


# --- rule 1: min_confidence ---

@dataclass
class ConfidenceRuleResult:
    """What the confidence thresholds did in one run; the numbers go into summary.json."""

    thresholds: dict[str, float]
    dropped_per_field: dict[str, int] = field(default_factory=dict)
    papers: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied": bool(self.thresholds),
            "thresholds": self.thresholds,
            "papers_with_drops": len(self.papers),
            "labels_dropped": sum(self.dropped_per_field.values()),
            "labels_dropped_per_field": dict(sorted(self.dropped_per_field.items())),
        }


def below_threshold(label: CheckedLabel, thresholds: dict[str, float]) -> bool:
    """True if the label's field has a threshold and its confidence is known and lower."""
    threshold = thresholds.get(label.field)
    return threshold is not None and label.confidence is not None and label.confidence < threshold


def drop_below_min_confidence(
    labels_by_document: LabelsByDocument, settings: dict[str, Any]
) -> tuple[LabelsByDocument, ConfidenceRuleResult]:
    """Return the labels per document without those below their field's threshold."""
    thresholds = min_confidence_thresholds(settings)
    result = ConfidenceRuleResult(thresholds)
    if not thresholds:
        return labels_by_document, result
    kept_by_document: LabelsByDocument = {}
    for document_id, labels in labels_by_document.items():
        kept = []
        for label in labels:
            if below_threshold(label, thresholds):
                result.dropped_per_field[label.field] = result.dropped_per_field.get(label.field, 0) + 1
                result.papers.add(document_id)
            else:
                kept.append(label)
        kept_by_document[document_id] = kept
    return kept_by_document, result


# --- rule 2: services without neighbour support ---

@dataclass
class ServiceRuleResult:
    """What the service rule did in one run; the numbers go into summary.json."""

    applied: bool
    reason: str | None = None   # why it was not applied
    dropped_entries: dict[str, int] = field(default_factory=dict)  # document id -> services dropped

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "reason": self.reason,
            "papers_dropped": len(self.dropped_entries),
            "ecosystem_service_entries_dropped": sum(self.dropped_entries.values()),
            "dropped_per_paper": dict(sorted(self.dropped_entries.items())),
        }


def neighbours_have_services(examples: FewShotExamples, document_id: str) -> bool | None:
    """True if at least one neighbour has an ecosystem_service entry, False if none has,
    None if the paper has no neighbours (then the rule does not apply)."""
    neighbours = examples.neighbour_references(document_id)
    if not neighbours:
        return None
    return any(reference.get(ECOSYSTEM_SERVICE_FIELD) for reference in neighbours)


def drop_services_without_neighbour_support(
    labels_by_document: LabelsByDocument,
    examples: FewShotExamples | None,
    settings: dict[str, Any],
) -> tuple[LabelsByDocument, ServiceRuleResult]:
    """Return the labels per document after the rule, plus what it dropped.

    The input is not changed; documents the rule does not touch keep their list as it is.
    """
    if not service_rule_enabled(settings):
        return labels_by_document, ServiceRuleResult(applied=False, reason=NOT_APPLIED_SWITCHED_OFF)
    if examples is None:
        logger.warning("Service rule %s", NOT_APPLIED_NO_EXAMPLES)
        return labels_by_document, ServiceRuleResult(applied=False, reason=NOT_APPLIED_NO_EXAMPLES)

    result = ServiceRuleResult(applied=True)
    kept_by_document: LabelsByDocument = {}
    for document_id, labels in labels_by_document.items():
        services = [label for label in labels if label.field == ECOSYSTEM_SERVICE_FIELD]
        if services and neighbours_have_services(examples, document_id) is False:
            kept_by_document[document_id] = [label for label in labels
                                             if label.field != ECOSYSTEM_SERVICE_FIELD]
            result.dropped_entries[document_id] = len(services)
        else:
            kept_by_document[document_id] = labels
    return kept_by_document, result


# --- both rules ---

@dataclass
class PostprocessResult:
    labels_by_document: LabelsByDocument
    dropped_by_rule: dict[str, dict[int, str]]  # document id -> {id(label): rule}, for the label file
    confidence_rule: ConfidenceRuleResult
    service_rule: ServiceRuleResult

    def rule_that_dropped(self, document_id: str, label: CheckedLabel) -> str | None:
        return self.dropped_by_rule.get(document_id, {}).get(id(label))

    def summary(self) -> dict[str, Any]:
        return {RULE_MIN_CONFIDENCE: self.confidence_rule.as_dict(), RULE_SERVICES: self.service_rule.as_dict()}


def postprocess(
    labels_by_document: LabelsByDocument, examples: FewShotExamples | None, settings: dict[str, Any]
) -> PostprocessResult:
    """Both rules in order; remembers which rule dropped which label (by identity)."""
    dropped_by_rule: dict[str, dict[int, str]] = {}

    def remember(before: LabelsByDocument, after: LabelsByDocument, rule: str) -> None:
        for document_id, labels in before.items():
            kept_ids = {id(label) for label in after[document_id]}
            for label in labels:
                if id(label) not in kept_ids:
                    dropped_by_rule.setdefault(document_id, {})[id(label)] = rule

    after_confidence, confidence_rule = drop_below_min_confidence(labels_by_document, settings)
    remember(labels_by_document, after_confidence, RULE_MIN_CONFIDENCE)
    after_services, service_rule = drop_services_without_neighbour_support(after_confidence, examples, settings)
    remember(after_confidence, after_services, RULE_SERVICES)
    return PostprocessResult(after_services, dropped_by_rule, confidence_rule, service_rule)
