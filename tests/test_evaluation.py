"""Tests for src/evaluation.py on invented records."""
import pytest

from data import ID_FIELD
from evaluation import C1_FIELDS, C2B_FIELDS, evaluate, extract_values, score_field


def test_gold_against_gold_scores_one(invented_references):
    result = evaluate(invented_references, invented_references, "all")
    assert result["task_scores"] == {"c1": 1.0, "c2b": 1.0}


def test_empty_predictions_score_zero(invented_references):
    result = evaluate(invented_references, {}, "all")
    assert result["task_scores"] == {"c1": 0.0, "c2b": 0.0}


def test_hand_computed_counts(invented_references, enum):
    # DOC00001: one of two habitats right, one wrong extra.  DOC00002: nothing predicted.
    predictions = {
        "DOC00001": {ID_FIELD: "DOC00001", "habitat": [enum("habitat", 0), enum("habitat", 2)]},
    }
    score = score_field(invented_references, predictions, "habitat")
    assert (score.true_positives, score.false_positives, score.false_negatives) == (1, 1, 2)
    assert score.precision == pytest.approx(0.5)
    assert score.recall == pytest.approx(1 / 3)
    assert score.f1 == pytest.approx(0.4)


def test_subfields_are_scored_separately(invented_references, enum):
    # Right category, wrong term: category is a TP, term is an FP plus an FN.
    predictions = {
        "DOC00001": {ID_FIELD: "DOC00001", "ecosystem_type": [
            {"category": enum("ecosystem_type.category", 0), "term": enum("ecosystem_type.term", 5)}]},
    }
    category = score_field(invented_references, predictions, "ecosystem_type.category")
    term = score_field(invented_references, predictions, "ecosystem_type.term")
    assert category.true_positives == 1
    assert (term.true_positives, term.false_positives, term.false_negatives) == (0, 1, 1)


def test_null_and_missing_values_count_as_empty(invented_references):
    assert extract_values(invented_references["DOC00002"], "ecosystem_type.term") == set()
    assert extract_values(invented_references["DOC00002"], "taxa.german_name") == set()
    assert extract_values(invented_references["DOC00003"], "habitat") == set()
    assert extract_values(None, "habitat") == set()


def test_comparison_is_case_sensitive(invented_references):
    gold_name = "Inventus fictus"
    predictions = {"DOC00001": {ID_FIELD: "DOC00001",
                                "taxa": [{"species_group": "x", "scientific_name": gold_name.lower()}]}}
    score = score_field(invented_references, predictions, "taxa.scientific_name")
    assert score.true_positives == 0


def test_unknown_prediction_id_is_an_error(invented_references):
    with pytest.raises(ValueError, match="not in the references"):
        evaluate(invented_references, {"UNKNOWN": {ID_FIELD: "UNKNOWN"}}, "c1")


def test_task_score_is_mean_f1_of_its_fields(invented_references, enum):
    # Only habitat is predicted, and perfectly: C1 = (1 + 0 + 0 + 0 + 0) / 5.
    predictions = {
        doc_id: {ID_FIELD: doc_id, "habitat": record.get("habitat", [])}
        for doc_id, record in invented_references.items()
    }
    result = evaluate(invented_references, predictions, "c1")
    assert result["task_scores"]["c1"] == pytest.approx(1 / len(C1_FIELDS))
    assert "c2b" not in result["task_scores"]
    assert set(result["fields"]) == set(C1_FIELDS)
    assert len(C2B_FIELDS) == 6
