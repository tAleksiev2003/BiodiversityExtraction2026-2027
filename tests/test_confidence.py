"""Verbalized confidence: parsing, merging, the min_confidence rule and the sweep script.

Invented data only; label values are taken from the schema enums by position.
"""
import copy
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

from data import ID_FIELD, read_jsonl, write_jsonl
from evaluation import evaluate
from label_file import LABEL_FILE_NAME, SOURCE_MAIN, kept_label_line
from llm_answer import parse_answer, parse_confidence
from output import PREDICTIONS_FILENAME
from postprocessing import RULE_MIN_CONFIDENCE, RULE_SERVICES, drop_below_min_confidence, postprocess
from prompts import ECOSYSTEM_SERVICE_FIELD
from settings import REPO_ROOT
from verbatim_checker import CheckedLabel, CheckerConfig, merge_labels

sys.path.insert(0, str(REPO_ROOT / "src" / "scripts"))
import sweep_confidence  # noqa: E402

LEVEL = "biodiversity_level"


# --- parsing -----------------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    (0.7, 0.7), (1, 1.0), (0, 0.0),
    ("0.8", None), ("high", None),          # strings are invalid
    (1.5, None), (-0.1, None),              # out of range
    (True, None), (None, None), ([0.5], None),
])
def test_parse_confidence(raw: Any, expected: float | None) -> None:
    assert parse_confidence(raw) == expected


def test_invalid_or_missing_confidence_never_drops_a_label(enum) -> None:
    answer = parse_answer(
        '{"reasoning": "", "labels": ['
        f'{{"field": "habitat", "value": "{enum("habitat", 0)}", "confidence": 0.6, "quote": "q"}},'
        f'{{"field": "habitat", "value": "{enum("habitat", 1)}", "confidence": "sehr hoch", "quote": "q"}},'
        f'{{"field": "habitat", "value": "{enum("habitat", 2)}", "confidence": 7, "quote": "q"}},'
        f'{{"field": "habitat", "value": "{enum("habitat", 3)}", "quote": "q"}}]}}')
    assert [label.confidence for label in answer.labels] == [0.6, None, None, None]


def test_merge_keeps_the_highest_confidence_of_a_label(enum) -> None:
    config = CheckerConfig(enabled=True, quote_match_threshold=90, min_quote_words=3, max_quote_words=60,
                           min_verified_quotes=1)
    value = enum(LEVEL, 0)
    parts = [CheckedLabel(LEVEL, (LEVEL, value), value, "quote one", confidence=0.3),
             CheckedLabel(LEVEL, (LEVEL, value), value, "quote two", confidence=0.7),
             CheckedLabel(LEVEL, (LEVEL, value), value, "quote three")]
    kept, _ = merge_labels(parts, config, "DOC00001")
    assert len(kept) == 1 and kept[0].confidence == 0.7
    unknown = CheckedLabel("habitat", ("habitat", "x"), "x", "q")
    assert merge_labels([unknown], config, "DOC00001")[0][0].confidence is None


# --- min_confidence ------------------------------------------------------------------------------

def level_label(enum: Callable[[str, int], str], index: int, confidence: float | None) -> CheckedLabel:
    value = enum(LEVEL, index)
    return CheckedLabel(LEVEL, (LEVEL, value), value, "quote", confidence=confidence)


def with_thresholds(settings: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    copied = copy.deepcopy(settings)
    copied["postprocess"]["min_confidence"] = thresholds
    return copied


def test_min_confidence_drops_below_and_keeps_unknown(settings, enum) -> None:
    habitat = CheckedLabel("habitat", ("habitat", enum("habitat", 0)), enum("habitat", 0), "q", confidence=0.1)
    labels = {"DOC1": [level_label(enum, 0, 0.3), level_label(enum, 1, 0.5), level_label(enum, 2, None), habitat]}
    kept, result = drop_below_min_confidence(labels, with_thresholds(settings, {LEVEL: 0.5}))
    assert [label.entry for label in kept["DOC1"]] == [enum(LEVEL, 1), enum(LEVEL, 2), enum("habitat", 0)]
    numbers = result.as_dict()
    assert numbers["applied"] and numbers["labels_dropped"] == 1 and numbers["papers_with_drops"] == 1
    assert numbers["labels_dropped_per_field"] == {LEVEL: 1}


def test_min_confidence_empty_changes_nothing(settings, enum) -> None:
    labels = {"DOC1": [level_label(enum, 0, 0.0)]}
    assert settings["postprocess"]["min_confidence"] == {}
    kept, result = drop_below_min_confidence(labels, settings)
    assert kept == labels and not result.as_dict()["applied"]


def test_min_confidence_runs_before_the_service_rule(settings, enum, invented_examples) -> None:
    """The invented neighbours have no services: the service rule would drop both services;
    the one below its threshold is counted by min_confidence."""
    def service(index: int, confidence: float) -> CheckedLabel:
        entry = {"category": enum("ecosystem_service.category", 0), "term": enum("ecosystem_service.term", index)}
        return CheckedLabel(ECOSYSTEM_SERVICE_FIELD, tuple(entry.values()), entry, "q", confidence=confidence)
    low, high = service(0, 0.1), service(1, 0.9)
    result = postprocess({"DOC00001": [low, high]}, invented_examples,
                         with_thresholds(settings, {ECOSYSTEM_SERVICE_FIELD: 0.5}))
    assert result.labels_by_document["DOC00001"] == []
    assert result.rule_that_dropped("DOC00001", low) == RULE_MIN_CONFIDENCE
    assert result.rule_that_dropped("DOC00001", high) == RULE_SERVICES
    assert result.summary()["service_rule"]["ecosystem_service_entries_dropped"] == 1


# --- sweep script on a tiny invented run ---------------------------------------------------------

SWEEP_PAPERS = [f"DOC{number:05d}" for number in range(1, 11)]


@pytest.fixture
def invented_run(tmp_path: Path, enum) -> tuple[Path, Path]:
    """10 papers. Each has the right habitat (confidence 0.9) and a wrong one (0.3); every
    second paper also a right level without confidence. Extra lines: a wrong level the run's
    own min_confidence dropped (a candidate again), a service the service rule dropped, a
    label the checker dropped and one a field prompt replaced (none of these three is a
    candidate)."""
    right, wrong = enum("habitat", 0), enum("habitat", 1)
    references, predictions, lines = [], [], []
    for number, document_id in enumerate(SWEEP_PAPERS):
        reference = {ID_FIELD: document_id, "habitat": [right], LEVEL: []}
        lines.append(line(document_id, "habitat", right, 0.9))
        lines.append(line(document_id, "habitat", wrong, 0.3))
        prediction = {ID_FIELD: document_id, "habitat": [right, wrong], LEVEL: []}
        if number % 2 == 0:
            reference[LEVEL] = [enum(LEVEL, 0)]
            lines.append(line(document_id, LEVEL, enum(LEVEL, 0), None))
            prediction[LEVEL] = [enum(LEVEL, 0)]
        references.append(reference)
        predictions.append(prediction)
    lines.append(line(SWEEP_PAPERS[0], LEVEL, enum(LEVEL, 1), 0.2, rule=RULE_MIN_CONFIDENCE))
    service = {"category": enum("ecosystem_service.category", 0), "term": None}
    lines.append(line(SWEEP_PAPERS[0], ECOSYSTEM_SERVICE_FIELD, service, 0.9, rule=RULE_SERVICES))
    lines.append({**line(SWEEP_PAPERS[1], LEVEL, enum(LEVEL, 2), 0.9), "kept_by_checker": False})
    lines.append({**line(SWEEP_PAPERS[1], LEVEL, enum(LEVEL, 3), 0.9), "replaced_by_field_prompt": True})
    run_dir = tmp_path / "run"
    write_jsonl(lines, run_dir / LABEL_FILE_NAME)
    write_jsonl(predictions, run_dir / PREDICTIONS_FILENAME)
    references_path = tmp_path / "references.jsonl"
    write_jsonl(references, references_path)
    return run_dir, references_path


def line(document_id: str, field: str, value: Any, confidence: float | None, rule: str | None = None) -> dict[str, Any]:
    label = CheckedLabel(field, (field,), value, "q", confidence=confidence)
    return kept_label_line(document_id, label, SOURCE_MAIN, False, rule)


def test_sweep_candidates_skip_checker_drops_replaced_labels_and_service_rule(invented_run) -> None:
    run_dir, _ = invented_run
    candidates = sweep_confidence.load_candidates(run_dir)
    first = candidates[SWEEP_PAPERS[0]]
    assert sorted(field for field, _, _ in first) == ["biodiversity_level", "biodiversity_level", "habitat", "habitat"]
    assert len(candidates[SWEEP_PAPERS[1]]) == 2   # the checker drop and the replaced level are not candidates


def test_sweep_thresholds(invented_run) -> None:
    run_dir, references_path = invented_run
    candidates = sweep_confidence.load_candidates(run_dir)
    references = {entry[ID_FIELD]: entry for entry in read_jsonl(references_path)}
    # Habitat: 10 right (0.9), 10 wrong (0.3): best from 0.4 on; ties go to the lowest.
    assert sweep_confidence.best_threshold(candidates, references, "habitat") == 0.4
    choices, out_of_fold = sweep_confidence.cross_validate(candidates, references, "habitat")
    assert choices == [0.4] * sweep_confidence.FOLDS
    assert all(record["habitat"] == [references[document_id]["habitat"][0]]
               for document_id, record in out_of_fold.items())
    # Theory: F1 at 0 = 2/3 for habitat, half of it on the grid = 0.3.
    records = sweep_confidence.build_records(candidates, list(references), {})
    f1_at_zero = sweep_confidence.field_f1(references, records, ["habitat"])
    assert f1_at_zero == pytest.approx(2 / 3)
    assert sweep_confidence.on_grid(f1_at_zero / 2) == 0.3
    # A label without confidence survives every threshold.
    strict = sweep_confidence.build_records(candidates, list(references), {LEVEL: 0.9})
    assert strict[SWEEP_PAPERS[0]][LEVEL] == [references[SWEEP_PAPERS[0]][LEVEL][0]]


def test_sweep_script_prints_the_scores_of_the_candidates(invented_run, capsys, monkeypatch) -> None:
    run_dir, references_path = invented_run
    monkeypatch.setattr(sys, "argv", ["sweep_confidence.py", "--run", str(run_dir),
                                      "--references", str(references_path)])
    sweep_confidence.main()
    output = capsys.readouterr().out
    # "no threshold" = the run's candidates; here: the run's predictions plus the wrong level
    # its own min_confidence dropped.
    references = {entry[ID_FIELD]: entry for entry in read_jsonl(references_path)}
    candidates = sweep_confidence.load_candidates(run_dir)
    expected = evaluate(references, sweep_confidence.build_records(candidates, SWEEP_PAPERS, {}), "all")
    no_threshold = next(text for text in output.splitlines() if text.startswith("no threshold"))
    assert no_threshold.split()[-2:] == [f"{expected['task_scores']['c1']:.4f}",
                                         f"{expected['task_scores']['c2b']:.4f}"]
    assert 'suggested postprocess.min_confidence (cross-validated): {"habitat": 0.4' in output
    assert "papers: 10, candidate labels: 26, without confidence: 5" in output
