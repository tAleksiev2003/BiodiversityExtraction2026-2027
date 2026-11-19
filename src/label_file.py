"""labels_with_confidence.jsonl: every label the model gave, with its confidence and fate.

One line per label, written by pipeline F into the run folder. src/scripts/sweep_confidence.py
reads it to choose `postprocess.min_confidence` thresholds without new requests, so the
file keeps the labels the thresholds of the run dropped, too.

- Labels the verbatim checker dropped (C1 to C5): one line per call that gave them, with
  the check (`dropped_by_check`) and the raw values the model wrote.
- Labels the checker kept: one line per document and label (after merging the parts), with
  `replaced_by_field_prompt` (main-call labels of a field that a field prompt answered)
  and the post-processing rule that dropped it, if any.

No quotes and no reasoning are written: only values, numbers and flags.
"""
from typing import Any

from postprocessing import PostprocessResult
from verbatim_checker import CHECK_ENOUGH_EVIDENCE, CheckedLabel, CheckLogEntry

LABEL_FILE_NAME = "labels_with_confidence.jsonl"
SOURCE_MAIN = "main"
SOURCE_FIELD_PROMPT = "field_prompt"
FIELD_PROMPT_CALL_SUFFIX = " prompt "   # call names of field prompts: "biodiversity_level prompt 1/1"
NOT_VALUES = ("field", "quote", "confidence")  # keys of a raw label that are not its value


def field_prompt_call(field: str, call: str) -> str:
    """The call name of a field prompt in the logs; the prompt itself still says `call`."""
    return f"{field}{FIELD_PROMPT_CALL_SUFFIX}{call}"


def source_of_call(call: str) -> str:
    return SOURCE_FIELD_PROMPT if FIELD_PROMPT_CALL_SUFFIX in call else SOURCE_MAIN


def raw_value(label: dict[str, Any]) -> Any:
    """The value(s) of a raw label as the model wrote them: "value" for plain fields,
    else the subfields (category/term, taxa names ...)."""
    values = {key: value for key, value in label.items() if key not in NOT_VALUES}
    return values["value"] if set(values) == {"value"} else values


def checker_drop_line(entry: CheckLogEntry) -> dict[str, Any]:
    """A label dropped by C1 to C4 (raw label) or by C5 (label {"entry": ...})."""
    if entry.check == CHECK_ENOUGH_EVIDENCE:
        value, confidence = entry.label.get("entry"), None
    else:
        value, confidence = raw_value(entry.label), entry.label.get("confidence")
    return {
        "document_id": entry.document_id,
        "call": entry.call,
        "source": source_of_call(entry.call),
        "field": entry.field,
        "value": value,
        "confidence": confidence,
        "kept_by_checker": False,
        "dropped_by_check": entry.check,
        "replaced_by_field_prompt": False,
        "dropped_by_postprocessing": False,
        "postprocessing_rule": None,
    }


def kept_label_line(document_id: str, label: CheckedLabel, source: str, replaced: bool,
                    rule: str | None) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "call": None,
        "source": source,
        "field": label.field,
        "value": label.entry,
        "confidence": label.confidence,
        "kept_by_checker": True,
        "dropped_by_check": None,
        "replaced_by_field_prompt": replaced,
        "dropped_by_postprocessing": rule is not None,
        "postprocessing_rule": rule,
    }


def document_lines(
    document_id: str,
    check_log: list[CheckLogEntry],
    merged_labels: list[CheckedLabel],
    replaced_labels: list[CheckedLabel],
    field_prompt_fields: set[str],
    postprocessed: PostprocessResult,
) -> list[dict[str, Any]]:
    """All lines of one document. `merged_labels`: after the checker and merging, before
    post-processing; `replaced_labels`: merged main-call labels a field prompt replaced."""
    lines = [checker_drop_line(entry) for entry in check_log if not entry.passed]
    for label in replaced_labels:
        lines.append(kept_label_line(document_id, label, SOURCE_MAIN, True, None))
    for label in merged_labels:
        source = SOURCE_FIELD_PROMPT if label.field in field_prompt_fields else SOURCE_MAIN
        lines.append(kept_label_line(document_id, label, source, False,
                                     postprocessed.rule_that_dropped(document_id, label)))
    return lines
