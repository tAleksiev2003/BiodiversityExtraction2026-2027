"""Rules that run after the verbatim checker and before the predictions are written.

All rules only remove labels; they never add or change one. They need no request, so
their effect can be compared from the cache. `postprocess()` runs them in this order:

1. `postprocess.min_confidence` ({field: threshold}, empty = off): drop a label whose
   confidence (prompt v5, field prompts) is below its field's threshold. A label without
   a confidence is kept: older prompts give none, and a missing value says nothing about
   the label. Thresholds come from src/scripts/sweep_confidence.py (Lipton et al. 2014).
2. `postprocess.drop_ecosystem_type_for_land_only_habitats` (SPEC.md, section 13,
   2026-11-19): if all predicted habitats of a paper are in `postprocess.land_only_habitats`
   (farmland, soil, urban areas) and there is at least one, its ecosystem_type labels whose
   term is in `postprocess.ecosystem_type_terms_dropped_for_land_only_habitats` are dropped
   (`null` in that list = a label without term). Why: the idea came from the dev error
   analysis (most ecosystem_type false positives were "Äcker und Ackerbrachen" in papers
   without any ecosystem_type); the rates come from train. Such papers get specific
   semi-natural biotope types (Trockenrasen 21, Grün- und Freiflächen 7, Niedermoore 5, ...)
   but "Äcker und Ackerbrachen" in only 23 of 282 (8 %), stable across random halves of
   train (6-11 %) and before/after 2015 (8 % / 8 %). A paper without a predicted habitat
   keeps its ecosystem_type, because there is no evidence either way.
   The first version (superseded) dropped every ecosystem_type label of such a paper; it
   runs when the term setting is `all` or missing (copied config.yaml of runs made before
   the setting existed), so those runs are rebuilt unchanged.
3. `postprocess.drop_services_without_neighbour_support` (SPEC.md, section 13,
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
from labels import load_field_specs
from prompts import ECOSYSTEM_SERVICE_FIELD, ECOSYSTEM_TYPE_FIELD
from verbatim_checker import CheckedLabel

logger = logging.getLogger(__name__)

RULE_MIN_CONFIDENCE = "min_confidence"
RULE_ECOSYSTEM_TYPE = "ecosystem_type_rule"
RULE_SERVICES = "service_rule"
HABITAT_FIELD = "habitat"

# Why the service rule did nothing in a run (written to summary.json).
NOT_APPLIED_SWITCHED_OFF = "switched off (postprocess.drop_services_without_neighbour_support: false)"
NOT_APPLIED_NO_EXAMPLES = "not applied: the prompt has no training examples, so no neighbours were loaded"
NOT_APPLIED_ECOSYSTEM_TYPE_OFF = "switched off (postprocess.drop_ecosystem_type_for_land_only_habitats: false)"
# postprocess.ecosystem_type_terms_dropped_for_land_only_habitats: this value (or no entry)
# drops every ecosystem_type label, as the first, superseded version of the rule did.
ALL_ECOSYSTEM_TYPE_TERMS = "all"

LabelsByDocument = dict[str, list[CheckedLabel]]


def postprocess_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Runs made before a rule existed have no such entry in their copied config.yaml; for
    them the rule is off, so src/pipelines/verify.py rebuilds them unchanged."""
    return settings.get("postprocess", {})


def service_rule_enabled(settings: dict[str, Any]) -> bool:
    return bool(postprocess_settings(settings).get("drop_services_without_neighbour_support", False))


def min_confidence_thresholds(settings: dict[str, Any]) -> dict[str, float]:
    return dict(postprocess_settings(settings).get("min_confidence") or {})


def ecosystem_type_rule_enabled(settings: dict[str, Any]) -> bool:
    return bool(postprocess_settings(settings).get("drop_ecosystem_type_for_land_only_habitats", False))


def land_only_habitats(settings: dict[str, Any]) -> frozenset[str]:
    """The habitats of `postprocess.land_only_habitats`, each checked against the schema.

    The names are typed in settings.yaml, so a typo would silently switch the rule off for
    that habitat; checking them against the schema enum turns such a typo into an error.
    """
    configured = list(postprocess_settings(settings).get("land_only_habitats") or [])
    if not configured:
        raise ValueError("postprocess.land_only_habitats is empty, but the ecosystem_type rule is on")
    allowed = load_field_specs([settings["paths"]["schema_core"]])[HABITAT_FIELD].allowed_values
    unknown = [habitat for habitat in configured if habitat not in allowed]
    if unknown:
        raise ValueError(f"postprocess.land_only_habitats: not in the habitat schema: {unknown}")
    return frozenset(configured)


def ecosystem_type_terms_dropped(settings: dict[str, Any]) -> frozenset[str | None] | None:
    """The terms of `postprocess.ecosystem_type_terms_dropped_for_land_only_habitats`, each
    checked against the schema (None in the list = a label without term), or None for
    "every term" (`all` or no entry: the first version of the rule)."""
    configured = postprocess_settings(settings).get("ecosystem_type_terms_dropped_for_land_only_habitats",
                                                    ALL_ECOSYSTEM_TYPE_TERMS)
    if configured == ALL_ECOSYSTEM_TYPE_TERMS:
        return None
    terms = list(configured or [])
    if not terms:
        raise ValueError("postprocess.ecosystem_type_terms_dropped_for_land_only_habitats is empty, "
                         "but the ecosystem_type rule is on")
    allowed = load_field_specs([settings["paths"]["schema_core"]])[f"{ECOSYSTEM_TYPE_FIELD}.term"].allowed_values
    unknown = [term for term in terms if term is not None and term not in allowed]
    if unknown:
        raise ValueError("postprocess.ecosystem_type_terms_dropped_for_land_only_habitats: "
                         f"not in the ecosystem_type term schema: {unknown}")
    return frozenset(terms)


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


# --- rule 2: ecosystem_type for papers with land-only habitats ---

@dataclass
class EcosystemTypeRuleResult:
    """What the ecosystem_type rule did in one run; the numbers go into summary.json."""

    applied: bool
    reason: str | None = None   # why it was not applied
    habitats: list[str] = field(default_factory=list)
    terms: list[str | None] | None = None   # terms dropped; None = every term (first version)
    dropped_entries: dict[str, int] = field(default_factory=dict)  # document id -> ecosystem_type labels dropped

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "reason": self.reason,
            "land_only_habitats": self.habitats,
            "terms_dropped": ALL_ECOSYSTEM_TYPE_TERMS if self.terms is None else self.terms,
            "papers_dropped": len(self.dropped_entries),
            "ecosystem_type_entries_dropped": sum(self.dropped_entries.values()),
            "dropped_per_paper": dict(sorted(self.dropped_entries.items())),
        }


def has_only_land_habitats(labels: list[CheckedLabel], land_habitats: frozenset[str]) -> bool:
    """True if the paper has at least one habitat label and every one is in `land_habitats`."""
    habitats = [label.entry for label in labels if label.field == HABITAT_FIELD]
    return bool(habitats) and all(habitat in land_habitats for habitat in habitats)


def is_dropped_term(label: CheckedLabel, terms: frozenset[str | None] | None) -> bool:
    """True for an ecosystem_type label whose term the rule drops (every term if `terms` is None)."""
    if label.field != ECOSYSTEM_TYPE_FIELD:
        return False
    return terms is None or label.entry.get("term") in terms


def drop_ecosystem_type_for_land_only_habitats(
    labels_by_document: LabelsByDocument, settings: dict[str, Any]
) -> tuple[LabelsByDocument, EcosystemTypeRuleResult]:
    """Return the labels per document after the rule, plus what it dropped.

    The input is not changed; documents the rule does not touch keep their list as it is.
    """
    if not ecosystem_type_rule_enabled(settings):
        return labels_by_document, EcosystemTypeRuleResult(applied=False, reason=NOT_APPLIED_ECOSYSTEM_TYPE_OFF)

    land_habitats = land_only_habitats(settings)
    terms = ecosystem_type_terms_dropped(settings)
    result = EcosystemTypeRuleResult(applied=True, habitats=sorted(land_habitats),
                                     terms=None if terms is None else sorted(terms, key=lambda term: term or ""))
    kept_by_document: LabelsByDocument = {}
    for document_id, labels in labels_by_document.items():
        dropped = [label for label in labels if is_dropped_term(label, terms)]
        if dropped and has_only_land_habitats(labels, land_habitats):
            kept_by_document[document_id] = [label for label in labels if not is_dropped_term(label, terms)]
            result.dropped_entries[document_id] = len(dropped)
        else:
            kept_by_document[document_id] = labels
    return kept_by_document, result


# --- rule 3: services without neighbour support ---

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


# --- all rules ---

@dataclass
class PostprocessResult:
    labels_by_document: LabelsByDocument
    dropped_by_rule: dict[str, dict[int, str]]  # document id -> {id(label): rule}, for the label file
    confidence_rule: ConfidenceRuleResult
    ecosystem_type_rule: EcosystemTypeRuleResult
    service_rule: ServiceRuleResult

    def rule_that_dropped(self, document_id: str, label: CheckedLabel) -> str | None:
        return self.dropped_by_rule.get(document_id, {}).get(id(label))

    def summary(self) -> dict[str, Any]:
        return {RULE_MIN_CONFIDENCE: self.confidence_rule.as_dict(),
                RULE_ECOSYSTEM_TYPE: self.ecosystem_type_rule.as_dict(),
                RULE_SERVICES: self.service_rule.as_dict()}


def postprocess(
    labels_by_document: LabelsByDocument, examples: FewShotExamples | None, settings: dict[str, Any]
) -> PostprocessResult:
    """All rules in order; remembers which rule dropped which label (by identity)."""
    dropped_by_rule: dict[str, dict[int, str]] = {}

    def remember(before: LabelsByDocument, after: LabelsByDocument, rule: str) -> None:
        for document_id, labels in before.items():
            kept_ids = {id(label) for label in after[document_id]}
            for label in labels:
                if id(label) not in kept_ids:
                    dropped_by_rule.setdefault(document_id, {})[id(label)] = rule

    after_confidence, confidence_rule = drop_below_min_confidence(labels_by_document, settings)
    remember(labels_by_document, after_confidence, RULE_MIN_CONFIDENCE)
    after_ecosystem_type, ecosystem_type_rule = drop_ecosystem_type_for_land_only_habitats(after_confidence, settings)
    remember(after_confidence, after_ecosystem_type, RULE_ECOSYSTEM_TYPE)
    after_services, service_rule = drop_services_without_neighbour_support(after_ecosystem_type, examples, settings)
    remember(after_ecosystem_type, after_services, RULE_SERVICES)
    return PostprocessResult(after_services, dropped_by_rule, confidence_rule, ecosystem_type_rule, service_rule)
