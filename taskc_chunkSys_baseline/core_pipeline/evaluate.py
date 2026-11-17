"""Local set-based diagnostics; the course scorer remains authoritative."""
import json
from pathlib import Path

CORE = ["habitat", "biodiversity_level", "ecosystem_type.category", "ecosystem_type.term", "taxa.species_group"]
C2B = ["taxa.scientific_name", "taxa.german_name", "taxa.species_group", "biodiversity_level",
       "ecosystem_service.category", "ecosystem_service.term"]
# Exact-match fields once each. biodiversity_variable is free text and stays out of these averages.
CATEGORICAL = list(dict.fromkeys(CORE + C2B))
C1_KEY = "C1 Score(Macro F1)"
C2B_KEY = "C2b biodiversity (Macro F1)"
ALL_MACRO_KEY = "All categorical values (Macro)"
ALL_MICRO_KEY = "All categorical values (Micro)"
BERT_KEY = "C2b free text (BERTScore)"


def read_records(path):
    """Read JSONL or a JSON list and reject duplicate paper identifiers."""
    text = Path(path).read_text(encoding="utf-8").strip()
    records = json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    result = {}
    for row in records:
        key = row["zotitem_ptr_id"]
        if key in result:
            raise ValueError(f"Duplicate ID: {key}")
        result[key] = row
    return result


def matching_gold(gold, predicted):
    """Keep gold rows for the predicted papers and reject unknown prediction IDs."""
    missing = sorted(set(predicted) - set(gold))
    if missing:
        raise ValueError(f"Predictions missing from gold: {missing}")
    aligned = {key: gold[key] for key in gold if key in predicted}
    if not aligned:
        raise ValueError("Cannot evaluate an empty dataset")
    return aligned


def values(record, field):
    """Flatten one field path into a set of non-empty scalar values."""
    root, _, child = field.partition(".")
    data = record.get(root, [])
    data = data if isinstance(data, list) else [data]
    if child:
        data = [item.get(child) for item in data if isinstance(item, dict)]
    return {item for item in data if isinstance(item, str) and item.strip()}


def macro_f1(scores, fields):
    """Unweighted mean of per-field F1. Each field counts once."""
    return sum(scores[f]["f1"] for f in fields) / len(fields)


def micro_f1(scores, fields):
    """One F1 from TP, FP, and FN pooled across fields. Each field counts once."""
    tp = sum(scores[f]["tp"] for f in fields)
    fp = sum(scores[f]["fp"] for f in fields)
    fn = sum(scores[f]["fn"] for f in fields)
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0


def evaluate(gold, predicted, bert_model=None):
    """Compute pooled exact-match field F1 and optional best-match German BERTScore."""
    gold = matching_gold(gold, predicted)
    scores = {}
    for field in CATEGORICAL:
        tp = fp = fn = 0
        for key in gold:
            actual, expected = values(predicted[key], field), values(gold[key], field)
            tp += len(actual & expected)
            fp += len(actual - expected)
            fn += len(expected - actual)
        scores[field] = {"tp": tp, "fp": fp, "fn": fn,
                         "precision": tp / (tp + fp) if tp + fp else 0,
                         "recall": tp / (tp + fn) if tp + fn else 0,
                         "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0}
    result = {"fields": scores,
              C1_KEY: macro_f1(scores, CORE),
              C2B_KEY: macro_f1(scores, C2B),
              ALL_MACRO_KEY: macro_f1(scores, CATEGORICAL),
              ALL_MICRO_KEY: micro_f1(scores, CATEGORICAL),
              BERT_KEY: None,
              "note": "Local diagnostic, not the official course scorer; empty denominators score zero. "
                      "All categorical values count each exact-match field once. "
                      "C2b free text (BERTScore) is null without a BERT model; otherwise it is the "
                      "best-match BERTScore F1 averaged over all biodiversity_variable strings, "
                      "with unmatched strings as zero."}
    if bert_model:
        result[BERT_KEY] = free_text_score(gold, predicted, bert_model)["unmatched_zero_mean"]
    return result


def free_text_score(gold, predicted, model):
    """Greedily align variable strings by descending BERTScore. The reported score uses unmatched items as zero."""
    from bert_score import BERTScorer
    scorer = BERTScorer(model_type=model, lang="de", rescale_with_baseline=False)
    total, denominator, matched_count = 0.0, 0, 0
    for key in gold:
        actual = sorted(values(predicted[key], "biodiversity_variable"))
        expected = sorted(values(gold[key], "biodiversity_variable"))
        denominator += max(len(actual), len(expected))
        if not actual or not expected:
            continue
        pairs = [(i, j) for i in range(len(actual)) for j in range(len(expected))]
        _, _, f1 = scorer.score([actual[i] for i, j in pairs], [expected[j] for i, j in pairs])
        used_a, used_e = set(), set()
        ranked = sorted(zip(f1.tolist(), pairs), reverse=True)
        for score, (i, j) in ranked:
            if i not in used_a and j not in used_e:
                used_a.add(i)
                used_e.add(j)
                total += score
                matched_count += 1
    return {"matched_mean": total / matched_count if matched_count else 0,
            "unmatched_zero_mean": total / denominator if denominator else 0,
            "model": model, "note": "Diagnostic only; verify official aggregation before reporting."}
