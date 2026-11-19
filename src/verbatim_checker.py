"""Verbatim checker: drop every label without a verifiable quote (SPEC.md, section 3a).

A deterministic filter that runs right after every LLM call. It gets the labels of the
answer and the exact text that was sent to the model (not the whole prompt, so a quote
copied from the label lists of the prompt does not count). Checks, in this order; a
label is dropped at the first check it fails:

    C1  allowed value     every enum value is exactly in the schema list; required
                          subfields (e.g. taxa.species_group) are present
    C2  consistent pair   the term belongs to its category (term_to_category.json); a
                          term without an entry there is dropped too
    C3  plausible quote   min_quote_words..max_quote_words words, no "..." or "…"
    C4  verbatim quote    normalised quote in normalised text, rapidfuzz partial_ratio
                          >= quote_match_threshold
    C5  enough evidence   per document, at least min_verified_quotes different verified
                          quotes for the same label (counted over all calls, `merge_labels`)

Only quotes are searched, never the label values: neither the German value of
biodiversity_variable nor a taxon name (completed names are not in the text anyway).

`verbatim_check.enabled: false` (ablation) keeps only C1, because the output writer and
CodaBench reject values outside the schema; C2 to C5 are skipped.

Limitation (report it): the checker proves that a quote exists in the text, not that it
supports the label or that the study examined it.
"""
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from typing import Any

from rapidfuzz import fuzz

from labels import FieldSpec, load_field_specs, load_term_to_category
from llm_answer import ExtractedLabel
from output import UNSCORED_SUBFIELDS

logger = logging.getLogger(__name__)

CHECK_ALLOWED_VALUE = "C1"
CHECK_CONSISTENT_PAIR = "C2"
CHECK_PLAUSIBLE_QUOTE = "C3"
CHECK_VERBATIM_QUOTE = "C4"
CHECK_ENOUGH_EVIDENCE = "C5"
ALL_CHECKS = (CHECK_ALLOWED_VALUE, CHECK_CONSISTENT_PAIR, CHECK_PLAUSIBLE_QUOTE,
              CHECK_VERBATIM_QUOTE, CHECK_ENOUGH_EVIDENCE)
# Checks run on every label of one call; the others (C5, pipeline B's chunk vote) run
# later per document, on labels that were already counted once.
CALL_CHECKS = (CHECK_ALLOWED_VALUE, CHECK_CONSISTENT_PAIR, CHECK_PLAUSIBLE_QUOTE, CHECK_VERBATIM_QUOTE)

# Subfield names (not label values) of the fields that pair a term with a category.
CATEGORY_SUBFIELD = "category"
TERM_SUBFIELD = "term"
ELLIPSES = ("...", "…")

# --- normalisation (the same for quote and text) ---

SOFT_HYPHEN = "­"
UNIFIED_CHARACTERS = str.maketrans({
    **{quote: "'" for quote in "‘’‚‛′`´‹›"},
    **{quote: '"' for quote in "“”„‟«»″"},
    **{dash: "-" for dash in "‐‑‒–—―−"},
})
# "Biodiver-\nsität" -> "Biodiversität": a word split by hyphenation at a line break.
LINE_BREAK_HYPHENATION = re.compile(r"(\w)-[ \t]*\r?\n\s*(\w)")
WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Unicode NFKC (also ligatures like "ﬁ"), unified quotes and dashes, hyphenation
    at line breaks joined, lower case, whitespace collapsed."""
    text = unicodedata.normalize("NFKC", text).replace(SOFT_HYPHEN, "")
    text = text.translate(UNIFIED_CHARACTERS)
    text = LINE_BREAK_HYPHENATION.sub(r"\1\2", text)
    return WHITESPACE.sub(" ", text.lower()).strip()


# --- settings and allowed values ---

@dataclass(frozen=True)
class CheckerConfig:
    enabled: bool
    quote_match_threshold: float
    min_quote_words: int
    max_quote_words: int
    min_verified_quotes: int

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> "CheckerConfig":
        check = settings["verbatim_check"]
        return cls(
            enabled=check["enabled"],
            quote_match_threshold=check["quote_match_threshold"],
            min_quote_words=check["min_quote_words"],
            max_quote_words=check["max_quote_words"],
            min_verified_quotes=check["min_verified_quotes"],
        )


@dataclass(frozen=True)
class LabelRules:
    """Allowed values (schema files) and term -> category pairs (data/labels/)."""

    specs: dict[str, FieldSpec]
    term_to_category: dict[str, dict[str, str]]  # {field: {term: category}}

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> "LabelRules":
        paths = settings["paths"]
        return cls(
            specs=load_field_specs([paths["schema_core"], paths["schema_biodiversity"]]),
            term_to_category=load_term_to_category(paths["term_to_category"]),
        )

    def fields(self) -> list[str]:
        """Top-level output fields in schema order (habitat, taxa, ...)."""
        return list(dict.fromkeys(spec.parent for spec in self.specs.values()))

    def subfields(self, field: str) -> list[FieldSpec]:
        """Scored subfields of an object field; empty for a field of plain values."""
        unscored = UNSCORED_SUBFIELDS.get(field, [])
        return [
            spec for spec in self.specs.values()
            if spec.parent == field and spec.subfield is not None and spec.subfield not in unscored
        ]


# --- results ---

LabelEntry = str | dict[str, str | None]  # what goes into the prediction record


@dataclass(frozen=True)
class CheckedLabel:
    """A label that passed C1 to C4 in one call (or only C1 with the checker off)."""

    field: str
    key: tuple[str | None, ...]   # identifies the label across calls (merging, C5)
    entry: LabelEntry
    normalised_quote: str | None
    quote: str | None = None      # as the model wrote it; shown to the verifier (src/verification.py)
    confidence: float | None = None  # the model's probability (0..1) if the prompt asks for one


@dataclass(frozen=True)
class CheckLogEntry:
    """One line of verbatim_check_log.jsonl in the run folder."""

    document_id: str
    call: str           # e.g. "1/1" or "2/3" (part 2 of 3)
    field: str
    label: dict[str, Any]
    check: str          # the check that dropped the label, or the last one it passed
    passed: bool
    score: float | None  # C4: partial_ratio score
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class _Dropped(Exception):
    """A label failed `check` (internal: ends checking that label)."""

    def __init__(self, check: str, reason: str, score: float | None = None) -> None:
        super().__init__(reason)
        self.check = check
        self.reason = reason
        self.score = score


# --- C1 to C4: one call ---

def _clean_open_text(value: str | None) -> str | None:
    """Open text (taxa names, variables): surrounding spaces removed, empty = no value."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def check_allowed_value(label: ExtractedLabel, rules: LabelRules) -> tuple[LabelEntry, tuple[str | None, ...]]:
    """C1. Returns the record entry and the label's key, or raises _Dropped."""
    if label.field not in rules.fields():
        raise _Dropped(CHECK_ALLOWED_VALUE, f"unknown field {label.field!r}")
    subfields = rules.subfields(label.field)
    if not subfields:  # a field of plain values, e.g. habitat
        spec = rules.specs[label.field]
        value = label.value if spec.is_enum else _clean_open_text(label.value)
        if not value:
            raise _Dropped(CHECK_ALLOWED_VALUE, "no value")
        if spec.is_enum and value not in spec.allowed_values:
            raise _Dropped(CHECK_ALLOWED_VALUE, "value not in the schema list")
        return value, (label.field, value)

    entry: dict[str, str | None] = {}
    for spec in subfields:
        value = getattr(label, spec.subfield)
        if not spec.is_enum:
            value = _clean_open_text(value)
        if value is None:
            if spec.required:
                raise _Dropped(CHECK_ALLOWED_VALUE, f"{spec.subfield} missing")
        elif spec.is_enum and value not in spec.allowed_values:
            raise _Dropped(CHECK_ALLOWED_VALUE, f"{spec.subfield} not in the schema list")
        entry[spec.subfield] = value
    return entry, (label.field, *entry.values())


def check_consistent_pair(field: str, entry: LabelEntry, rules: LabelRules) -> None:
    """C2. Only fields with a term -> category mapping; a missing term is allowed."""
    mapping = rules.term_to_category.get(field)
    if mapping is None or not isinstance(entry, dict) or entry.get(TERM_SUBFIELD) is None:
        return
    term = entry[TERM_SUBFIELD]
    if term not in mapping:
        logger.info("Dropped %s term without an entry in term_to_category.json: %s", field, term)
        raise _Dropped(CHECK_CONSISTENT_PAIR, "term has no entry in term_to_category.json")
    if mapping[term] != entry[CATEGORY_SUBFIELD]:
        raise _Dropped(CHECK_CONSISTENT_PAIR, f"term belongs to category {mapping[term]!r}")


def check_plausible_quote(quote: str | None, config: CheckerConfig) -> str:
    """C3. Returns the quote."""
    if not quote or not quote.strip():
        raise _Dropped(CHECK_PLAUSIBLE_QUOTE, "no quote")
    if any(ellipsis in quote for ellipsis in ELLIPSES):
        raise _Dropped(CHECK_PLAUSIBLE_QUOTE, "quote contains an ellipsis")
    word_count = len(quote.split())
    if word_count < config.min_quote_words:
        raise _Dropped(CHECK_PLAUSIBLE_QUOTE, f"quote has {word_count} words, fewer than {config.min_quote_words}")
    if word_count > config.max_quote_words:
        raise _Dropped(CHECK_PLAUSIBLE_QUOTE, f"quote has {word_count} words, more than {config.max_quote_words}")
    return quote


def check_verbatim_quote(quote: str, normalised_text: str, config: CheckerConfig) -> tuple[str, float]:
    """C4. Returns the normalised quote and its score."""
    normalised_quote = normalise(quote)
    score = round(fuzz.partial_ratio(normalised_quote, normalised_text), 1)
    if score < config.quote_match_threshold:
        raise _Dropped(CHECK_VERBATIM_QUOTE, f"quote not found in the text (score {score})", score)
    return normalised_quote, score


def check_answer(
    labels: list[ExtractedLabel],
    source_text: str,
    rules: LabelRules,
    config: CheckerConfig,
    document_id: str,
    call: str,
) -> tuple[list[CheckedLabel], list[CheckLogEntry]]:
    """Run C1 to C4 on every label of one answer; one log entry per label."""
    normalised_text = normalise(source_text) if config.enabled else ""
    kept: list[CheckedLabel] = []
    log: list[CheckLogEntry] = []
    for label in labels:
        label_dict = label.model_dump(exclude_none=True)
        try:
            entry, key = check_allowed_value(label, rules)
            if not config.enabled:
                kept.append(CheckedLabel(label.field, key, entry, None, confidence=label.confidence))
                log.append(CheckLogEntry(document_id, call, label.field, label_dict, CHECK_ALLOWED_VALUE,
                                         True, None, "checker disabled: only C1"))
                continue
            check_consistent_pair(label.field, entry, rules)
            quote = check_plausible_quote(label.quote, config)
            normalised_quote, score = check_verbatim_quote(quote, normalised_text, config)
        except _Dropped as dropped:
            log.append(CheckLogEntry(document_id, call, label.field, label_dict, dropped.check,
                                     False, dropped.score, dropped.reason))
            continue
        kept.append(CheckedLabel(label.field, key, entry, normalised_quote, quote, label.confidence))
        log.append(CheckLogEntry(document_id, call, label.field, label_dict, CHECK_VERBATIM_QUOTE,
                                 True, score, "quote found in the text"))
    return kept, log


# --- C5: all calls of one document ---

def merge_labels(
    labels: list[CheckedLabel], config: CheckerConfig, document_id: str
) -> tuple[list[CheckedLabel], list[CheckLogEntry]]:
    """One label per key, in order of first appearance; C5 drops labels with too few quotes.

    Quotes are counted per document over all calls (parts), and the same quote given
    twice counts once. A call that does not mention a label is not a vote against it.
    The kept label gets the highest confidence of its group (None if no call gave one):
    like a quote, the strongest part decides.
    """
    groups: dict[tuple[str | None, ...], list[CheckedLabel]] = defaultdict(list)
    for label in labels:
        groups[label.key].append(label)
    kept: list[CheckedLabel] = []
    log: list[CheckLogEntry] = []
    for group in groups.values():
        if config.enabled:
            quote_count = len({label.normalised_quote for label in group})
            if quote_count < config.min_verified_quotes:
                reason = f"{quote_count} verified quote(s), {config.min_verified_quotes} needed"
                for label in group:
                    log.append(CheckLogEntry(document_id, "all", label.field, {"entry": label.entry},
                                             CHECK_ENOUGH_EVIDENCE, False, None, reason))
                continue
        confidences = [label.confidence for label in group if label.confidence is not None]
        kept.append(replace(group[0], confidence=max(confidences) if confidences else None))
    return kept, log


def build_record(id_field: str, document_id: str, labels: list[CheckedLabel], rules: LabelRules) -> dict[str, Any]:
    """The prediction record: every field as a list (empty if no label was kept)."""
    record: dict[str, Any] = {id_field: document_id}
    for field in rules.fields():
        record[field] = [label.entry for label in labels if label.field == field]
    return record


# --- statistics for the run summary ---

def drop_statistics(entries: list[dict[str, Any]], checks: tuple[str, ...] = ALL_CHECKS) -> dict[str, Any]:
    """Share of labels dropped per check and per field (SPEC.md, section 3a).

    `entries` are the log entries as dicts. Every label checked in a call has exactly one
    entry with a check from C1 to C4; other entries (C5, pipeline B's chunk vote) are
    labels dropped later, per document. `checks`: the checks to report.
    """
    checked_per_field: Counter = Counter()
    dropped: dict[str, Counter] = defaultdict(Counter)  # field -> check -> count
    for entry in entries:
        if entry["check"] in CALL_CHECKS:
            checked_per_field[entry["field"]] += 1
        if not entry["passed"]:
            dropped[entry["field"]][entry["check"]] += 1

    def summary(checked: int, dropped_counts: Counter) -> dict[str, Any]:
        total_dropped = sum(dropped_counts.values())
        return {
            "checked": checked,
            "kept": checked - total_dropped,
            "dropped": {check: dropped_counts.get(check, 0) for check in checks},
            "dropped_share": {
                check: (dropped_counts.get(check, 0) / checked if checked else 0.0) for check in checks
            },
        }

    all_dropped: Counter = Counter()
    for counts in dropped.values():
        all_dropped.update(counts)
    fields = sorted(set(checked_per_field) | set(dropped))
    return {
        "all_fields": summary(sum(checked_per_field.values()), all_dropped),
        "per_field": {field: summary(checked_per_field[field], dropped[field]) for field in fields},
    }
