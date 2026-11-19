"""Rules that run after the verbatim checker and before the predictions are written.

Rule `postprocess.drop_services_without_neighbour_support` (SPEC.md, section 13, 2026-11-19):
if none of a paper's training neighbours (neighbours_<split>.json, the examples of prompt v4)
has an ecosystem_service entry in the train references, all ecosystem_service labels of the
paper are dropped. Why: in train, papers whose 3 nearest train papers have no services have
services themselves in only 6.7 % of cases (46 % otherwise), and the model proposes many
false services. A paper without neighbours (no first-pages text) keeps its services,
because there is no evidence either way.

The rule only removes labels; it never adds or changes one. It needs no request, so its
effect can be compared from the cache (config/experiments/no_service_rule.yaml).
"""
import logging
from dataclasses import dataclass, field
from typing import Any

from few_shot_examples import FewShotExamples
from prompts import ECOSYSTEM_SERVICE_FIELD
from verbatim_checker import CheckedLabel

logger = logging.getLogger(__name__)

# Why the rule did nothing in a run (written to summary.json).
NOT_APPLIED_SWITCHED_OFF = "switched off (postprocess.drop_services_without_neighbour_support: false)"
NOT_APPLIED_NO_EXAMPLES = "not applied: the prompt has no training examples, so no neighbours were loaded"


def service_rule_enabled(settings: dict[str, Any]) -> bool:
    """Runs made before the rule existed have no `postprocess` section in their copied
    config.yaml; for them the rule is off, so src/pipelines/verify.py rebuilds them unchanged."""
    return bool(settings.get("postprocess", {}).get("drop_services_without_neighbour_support", False))


@dataclass
class ServiceRuleResult:
    """What the rule did in one run; the numbers go into summary.json."""

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
    labels_by_document: dict[str, list[CheckedLabel]],
    examples: FewShotExamples | None,
    settings: dict[str, Any],
) -> tuple[dict[str, list[CheckedLabel]], ServiceRuleResult]:
    """Return the labels per document after the rule, plus what it dropped.

    The input is not changed; documents the rule does not touch keep their list as it is.
    """
    if not service_rule_enabled(settings):
        return labels_by_document, ServiceRuleResult(applied=False, reason=NOT_APPLIED_SWITCHED_OFF)
    if examples is None:
        logger.warning("Service rule %s", NOT_APPLIED_NO_EXAMPLES)
        return labels_by_document, ServiceRuleResult(applied=False, reason=NOT_APPLIED_NO_EXAMPLES)

    result = ServiceRuleResult(applied=True)
    kept_by_document: dict[str, list[CheckedLabel]] = {}
    for document_id, labels in labels_by_document.items():
        services = [label for label in labels if label.field == ECOSYSTEM_SERVICE_FIELD]
        if services and neighbours_have_services(examples, document_id) is False:
            kept_by_document[document_id] = [label for label in labels
                                             if label.field != ECOSYSTEM_SERVICE_FIELD]
            result.dropped_entries[document_id] = len(services)
        else:
            kept_by_document[document_id] = labels
    return kept_by_document, result
