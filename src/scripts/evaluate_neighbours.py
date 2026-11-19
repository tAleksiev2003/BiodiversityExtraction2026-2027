"""How good are the neighbours? Score their gold labels as a prediction, without any request.

Prompt v4 shows the labels of the most similar training papers. Before spending requests,
this script checks whether the neighbours carry useful information: every dev paper is
predicted by a vote of its k neighbours' gold labels (a label wins if at least
`retrieval.vote_min_share` of the neighbours have it), and the prediction is scored with
src/evaluation.py. This is done for both methods (sentence embeddings, TF-IDF) and every k
in `retrieval.evaluate_k`.

Compare the table with the most frequent baseline on the same 100 papers (src/baselines.py
--dev-set with_pdf) and with F, prompt v2 (C1 0.571, C2b 0.331). Neighbours that score
clearly above the baseline label similar papers in a similar way; if they do not, the
examples mostly show noise.

The vote is per scored field: `taxa.species_group` and `taxa.scientific_name` are voted on
separately, as the evaluation compares them separately. Dev papers without a first-pages
text get an empty prediction.

Usage (run by the team, it reads course data; prints only scores):
    python3 src/scripts/evaluate_neighbours.py
"""
import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from src/
from data import ID_FIELD, load_references  # noqa: E402
from evaluation import C1_FIELDS, C2B_FIELDS, evaluate, extract_values  # noqa: E402
from retrieval import METHODS, find_neighbours  # noqa: E402
from settings import load_settings  # noqa: E402

SPLIT = "dev"
SCORED_FIELDS = list(dict.fromkeys(C1_FIELDS + C2B_FIELDS))


def votes_needed(neighbour_count: int, min_share: float) -> int:
    """Smallest number of neighbours that is at least `min_share` of them (0.5 of 3 -> 2)."""
    return max(1, math.ceil(neighbour_count * min_share))


def vote_record(document_id: str, neighbour_references: list[dict[str, Any]],
                min_share: float) -> dict[str, Any]:
    """A record whose scored values are those that enough neighbours share.

    Only for scoring in memory: a subfield value becomes its own entry (e.g.
    {"species_group": ...}), which the evaluation reads exactly like a full entry, because
    it compares every subfield as a separate set.
    """
    record: dict[str, Any] = {ID_FIELD: document_id}
    needed = votes_needed(len(neighbour_references), min_share)
    for field in SCORED_FIELDS:
        counts = Counter(value for reference in neighbour_references
                         for value in extract_values(reference, field))
        winners = sorted(value for value, count in counts.items() if count >= needed)
        parent, _, subfield = field.partition(".")
        entries = [{subfield: value} for value in winners] if subfield else winners
        record.setdefault(parent, []).extend(entries)
    return record


def vote_predictions(neighbours: dict[str, list[str]], train_references: dict[str, dict[str, Any]],
                     k: int, min_share: float) -> dict[str, dict[str, Any]]:
    return {document_id: vote_record(document_id, [train_references[i] for i in ids[:k]], min_share)
            for document_id, ids in neighbours.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the vote of each dev paper's train neighbours.")
    parser.add_argument("--experiment", type=Path, default=None)
    args = parser.parse_args()

    settings = load_settings(args.experiment)
    retrieval = settings["retrieval"]
    k_values = retrieval["evaluate_k"]
    dev_references = load_references(settings["paths"]["references_dev_with_pdf"])
    train_references = load_references(settings["paths"]["references_train"])

    rows = []
    for method in METHODS:
        ranked, without_text = find_neighbours(settings, SPLIT, method, max(k_values))
        neighbours = {document_id: [train_id for train_id, _ in pairs] for document_id, pairs in ranked.items()}
        for k in k_values:
            predictions = vote_predictions(neighbours, train_references, k, retrieval["vote_min_share"])
            scores = evaluate(dev_references, predictions, "all")["task_scores"]
            rows.append((method, k, scores["c1"], scores["c2b"]))

    print(f"\nDev papers: {len(dev_references)}, without first-pages text (empty prediction): "
          f"{len(without_text)}; vote: a label needs at least {retrieval['vote_min_share']:.0%} "
          f"of the k neighbours")
    print(f"{'method':10s} {'k':>3s} {'C1':>7s} {'C2b':>7s}")
    for method, k, c1, c2b in rows:
        print(f"{method:10s} {k:3d} {c1:7.4f} {c2b:7.4f}")


if __name__ == "__main__":
    main()
