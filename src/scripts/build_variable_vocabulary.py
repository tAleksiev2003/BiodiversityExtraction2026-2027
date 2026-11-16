"""Build the vocabulary of measured variables (biodiversity_variable) from the references.

biodiversity_variable is a free-text field. To make the model use the same wording as
the database, the prompt shows the most frequent values from the training references
as preferred terms. This script collects those values and their counts.

Usage:
    python src/scripts/build_variable_vocabulary.py \
        --references data/raw/references/train.jsonl \
        --output data/labels/variable_vocabulary.json

The output is a list sorted by frequency, e.g.
    [{"value": "Biomasse", "count": 57}, {"value": "Artenreichtum", "count": 41}, ...]

The script also prints values that differ only in upper/lower case or surrounding
spaces, because these are probably spelling variants that a human should check.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

FIELD = "biodiversity_variable"


def count_values(references_path: Path) -> Counter:
    """Count how often each biodiversity_variable value occurs in the references."""
    counts: Counter = Counter()
    with references_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            for value in record.get(FIELD) or []:
                if isinstance(value, str) and value.strip():
                    counts[value] += 1
    return counts


def find_spelling_variants(counts: Counter) -> dict[str, list[str]]:
    """Group values that are equal after lowercasing and stripping spaces."""
    groups: dict[str, list[str]] = defaultdict(list)
    for value in counts:
        groups[value.strip().lower()].append(value)
    return {key: values for key, values in groups.items() if len(values) > 1}


def print_report(counts: Counter, variants: dict[str, list[str]]) -> None:
    """Print a short overview and the variants a human should check."""
    print(f"distinct values: {len(counts)}")
    print(f"total mentions:  {sum(counts.values())}")
    print("\n10 most frequent values:")
    for value, count in counts.most_common(10):
        print(f"   {count:4d}  {value}")
    print(f"\npossible spelling variants (check by hand): {len(variants)}")
    for values in variants.values():
        print(f"   - {values}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    counts = count_values(args.references)
    print_report(counts, find_spelling_variants(counts))

    vocabulary = [{"value": value, "count": count} for value, count in counts.most_common()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(vocabulary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWritten: {args.output}")


if __name__ == "__main__":
    main()
