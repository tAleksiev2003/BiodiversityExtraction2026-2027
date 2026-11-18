"""Break the errors of a prediction run down into causes, printing only counts.

The score table says *how many* false positives (FP) and false negatives (FN) a field
has. This script says *why*, so the team can decide what to change before spending
requests on a new prompt:

1. Empty predictions: documents where the model gave no label at all. Is the
   reference empty there too, or did the model miss something?
2. Per field: is an FP an extra value in a document whose reference is empty for
   that field ("over-extraction"), or a wrong value next to correct ones ("wrong
   choice")? Is an FN in a document where nothing was predicted ("missed entirely")?
3. Taxa names: is an FP only a spelling variant of a reference name, the same genus
   (e.g. "Quercus spp." vs "Quercus robur"), or a different taxon?
4. Taxa per document: does the model name many more taxa than the database?

Output contains only counts, schema label values and document ids, never text or
taxon names from the papers, so it can be shared within the team.

Usage:
    python3 src/scripts/analyze_errors.py \
        --references data/processed/dev_with_pdf.jsonl \
        --predictions runs/20261118-124839_full_text_dev/predictions.jsonl
"""
import argparse
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from src/
from data import index_by_id, load_references, read_jsonl  # noqa: E402
from evaluation import C1_FIELDS, C2B_FIELDS, extract_values  # noqa: E402

SCORED_FIELDS = list(dict.fromkeys(C1_FIELDS + C2B_FIELDS))
TAXON_NAME_FIELDS = ["taxa.scientific_name", "taxa.german_name"]
SPELLING_SIMILARITY = 90   # rapidfuzz ratio (0-100) above which two names count as spelling variants
TOP_LABELS = 5             # most frequent wrong labels shown per categorical field
TOP_DOCUMENTS = 5          # documents with the most taxa FPs shown by id


def share(part: int, total: int) -> str:
    return f"{part:5d}  ({100 * part / total:5.1f} %)" if total else f"{part:5d}"


def normalise_taxon(name: str) -> str:
    """Remove differences that do not change the taxon: case, author and year,
    'spp'/'sp' with or without dot, hyphens and repeated spaces."""
    name = name.lower()
    name = re.sub(r"\(.*?\)", " ", name)                 # (Linnaeus, 1758)
    name = re.sub(r"\b1[6-9]\d\d\b|\b20\d\d\b", " ", name)  # years
    name = re.sub(r"\bl\.\s*$", " ", name)                # trailing "L."
    name = re.sub(r"\bspp?\.?\s*$", " spp", name)         # sp / sp. / spp / spp. -> spp
    name = name.replace("-", " ")
    return re.sub(r"\s+", " ", name).strip()


def genus(name: str) -> str:
    return normalise_taxon(name).split(" ", 1)[0]


def classify_taxon_error(value: str, other_side: set[str]) -> str:
    """Why a predicted (or reference) name has no exact partner on the other side."""
    if not other_side:
        return "other side empty"
    if any(normalise_taxon(value) == normalise_taxon(other) for other in other_side):
        return "spelling variant (same after normalising)"
    if any(fuzz.ratio(normalise_taxon(value), normalise_taxon(other)) >= SPELLING_SIMILARITY
           for other in other_side):
        return "spelling variant (very similar)"
    if any(genus(value) == genus(other) for other in other_side):
        return "same genus, different rank or species"
    return "different taxon"


def report_empty_predictions(references: dict, predictions: dict) -> None:
    empty_ids = [doc_id for doc_id in references
                 if not any(extract_values(predictions.get(doc_id), f) for f in SCORED_FIELDS)]
    reference_also_empty = [doc_id for doc_id in empty_ids
                            if not any(extract_values(references[doc_id], f) for f in SCORED_FIELDS)]
    print(f"=== 1. Documents without any predicted label: {len(empty_ids)} of {len(references)} ===")
    print(f"   reference empty too (nothing missed):     {share(len(reference_also_empty), len(empty_ids))}")
    print(f"   reference has values (should not be empty): "
          f"{share(len(empty_ids) - len(reference_also_empty), len(empty_ids))}")
    print("   reference values missed in these documents, per field (share of the field's FN):")
    for field in SCORED_FIELDS:
        missed = sum(len(extract_values(references[doc_id], field)) for doc_id in empty_ids)
        total_fn = sum(len(extract_values(references[d], field) - extract_values(predictions.get(d), field))
                       for d in references)
        print(f"      {field:30s} {share(missed, total_fn)}")


def report_error_types(references: dict, predictions: dict) -> None:
    print("\n=== 2. FP and FN per field, by situation ===")
    print(f"   {'field':30s} {'FP total':>8s} {'ref empty':>10s} {'FN total':>9s} {'pred empty':>11s}")
    for field in SCORED_FIELDS:
        fp = fp_ref_empty = fn = fn_pred_empty = 0
        for doc_id, reference in references.items():
            gold = extract_values(reference, field)
            predicted = extract_values(predictions.get(doc_id), field)
            fp += len(predicted - gold)
            fn += len(gold - predicted)
            if not gold:
                fp_ref_empty += len(predicted)
            if not predicted:
                fn_pred_empty += len(gold)
        print(f"   {field:30s} {fp:8d} {fp_ref_empty:10d} {fn:9d} {fn_pred_empty:11d}")
    print("   ref empty  = FP in documents whose reference has no value for this field (over-extraction)")
    print("   pred empty = FN in documents where nothing was predicted for this field (missed entirely)")


def report_wrong_labels(references: dict, predictions: dict) -> None:
    """Schema labels only (public), never taxon names."""
    print(f"\n=== 2b. Most frequent wrong labels (top {TOP_LABELS}, schema values) ===")
    for field in SCORED_FIELDS:
        if field in TAXON_NAME_FIELDS:
            continue
        fp_labels: Counter = Counter()
        fn_labels: Counter = Counter()
        for doc_id, reference in references.items():
            gold = extract_values(reference, field)
            predicted = extract_values(predictions.get(doc_id), field)
            fp_labels.update(predicted - gold)
            fn_labels.update(gold - predicted)
        print(f"   {field}")
        print(f"      FP: {fp_labels.most_common(TOP_LABELS)}")
        print(f"      FN: {fn_labels.most_common(TOP_LABELS)}")


def report_taxon_names(references: dict, predictions: dict) -> None:
    print("\n=== 3. Taxa names: why FP and FN ===")
    for field in TAXON_NAME_FIELDS:
        fp_causes: Counter = Counter()
        fn_causes: Counter = Counter()
        for doc_id, reference in references.items():
            gold = extract_values(reference, field)
            predicted = extract_values(predictions.get(doc_id), field)
            fp_causes.update(classify_taxon_error(value, gold - predicted) for value in predicted - gold)
            fn_causes.update(classify_taxon_error(value, predicted - gold) for value in gold - predicted)
        for title, causes in (("FP (predicted, not in reference)", fp_causes),
                              ("FN (in reference, not predicted)", fn_causes)):
            total = sum(causes.values())
            print(f"   {field} – {title}: {total}")
            for cause, count in causes.most_common():
                print(f"      {cause:45s} {share(count, total)}")


def report_taxa_per_document(references: dict, predictions: dict) -> None:
    field = "taxa.scientific_name"
    gold_counts = {d: len(extract_values(r, field)) for d, r in references.items()}
    pred_counts = {d: len(extract_values(predictions.get(d), field)) for d in references}
    fp_per_doc = {d: len(extract_values(predictions.get(d), field) - extract_values(references[d], field))
                  for d in references}
    total_fp = sum(fp_per_doc.values())
    top = sorted(fp_per_doc.items(), key=lambda item: item[1], reverse=True)[:TOP_DOCUMENTS]

    print(f"\n=== 4. Scientific names per document ===")
    for name, counts in (("reference", gold_counts), ("prediction", pred_counts)):
        values = list(counts.values())
        print(f"   {name:10s} total {sum(values):5d}, median {statistics.median(values):4.0f}, "
              f"max {max(values):4d}, documents with none {sum(v == 0 for v in values):3d}")
    print(f"   documents with more predicted than reference names: "
          f"{sum(pred_counts[d] > gold_counts[d] for d in references)}")
    print(f"   top {TOP_DOCUMENTS} documents by FP: {top}, "
          f"together {share(sum(count for _, count in top), total_fp).strip()} of all FP")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()

    references = load_references(args.references)
    predictions = index_by_id(read_jsonl(args.predictions), str(args.predictions))

    report_empty_predictions(references, predictions)
    report_error_types(references, predictions)
    report_wrong_labels(references, predictions)
    report_taxon_names(references, predictions)
    report_taxa_per_document(references, predictions)


if __name__ == "__main__":
    main()
