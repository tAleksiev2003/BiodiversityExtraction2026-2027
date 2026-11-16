"""Build the term -> category mapping for ecosystem_type and ecosystem_service.

The schema files list categories and terms as two separate enums but do not say
which term belongs to which category. The training references, however, contain
(category, term) pairs. This script collects those pairs and writes one mapping
file that the pipeline can use in prompts and for validation.

Usage:
    python build_term_to_category.py \
        --references data/raw/references/train.jsonl \
        --schema-dir data/schemas \
        --output data/labels/term_to_category.json

The script prints a report with three lists that need a human check:
    1. terms from the schema that never occur in the references (fill in by hand),
    2. terms that occur with more than one category (decide by hand),
    3. terms in the references that are not in the schema (spelling differences).
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

# Which reference field belongs to which schema file and enum names.
FIELDS = {
    "ecosystem_type": {
        "schema_file": "core_fields.json",
        "term_enum": "EcosystemTypeTermEnum",
    },
    "ecosystem_service": {
        "schema_file": "biodiversity.json",
        "term_enum": "EcosystemServiceTermEnum",
    },
}


def load_enum(schema_path: Path, enum_name: str) -> list[str]:
    """Return the allowed values of one enum, exactly as written in the schema."""
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return schema["$defs"][enum_name]["enum"]


def count_pairs(references_path: Path, field: str) -> dict[str, Counter]:
    """Count how often each term appears together with each category."""
    categories_per_term: dict[str, Counter] = defaultdict(Counter)
    with references_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            for entry in record.get(field) or []:
                term = entry.get("term")
                category = entry.get("category")
                if term and category:  # skip pairs without a term
                    categories_per_term[term][category] += 1
    return categories_per_term


def build_mapping(
    categories_per_term: dict[str, Counter], allowed_terms: list[str]
) -> tuple[dict[str, str], dict]:
    """Choose one category per term and collect everything a human must check."""
    mapping = {}
    conflicts = {}
    for term, category_counts in sorted(categories_per_term.items()):
        most_common_category, _ = category_counts.most_common(1)[0]
        mapping[term] = most_common_category
        if len(category_counts) > 1:
            conflicts[term] = dict(category_counts)

    report = {
        "missing_terms": sorted(set(allowed_terms) - set(categories_per_term)),
        "conflicting_terms": conflicts,
        "unknown_terms": sorted(set(categories_per_term) - set(allowed_terms)),
    }
    # Only terms that are valid schema values belong into the mapping file.
    mapping = {term: cat for term, cat in mapping.items() if term in allowed_terms}
    return mapping, report


def print_report(field: str, mapping: dict[str, str], report: dict) -> None:
    """Print what was found and what still needs a human decision."""
    print(f"\n=== {field} ===")
    print(f"terms mapped from references: {len(mapping)}")
    print(f"1. never seen in references (fill in by hand): {len(report['missing_terms'])}")
    for term in report["missing_terms"]:
        print(f"   - {term}")
    print(f"2. seen with several categories (decide by hand): {len(report['conflicting_terms'])}")
    for term, counts in report["conflicting_terms"].items():
        print(f"   - {term}: {counts}")
    print(f"3. not in the schema (spelling differences?): {len(report['unknown_terms'])}")
    for term in report["unknown_terms"]:
        print(f"   - {term}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--schema-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {}
    for field, config in FIELDS.items():
        allowed_terms = load_enum(args.schema_dir / config["schema_file"], config["term_enum"])
        categories_per_term = count_pairs(args.references, field)
        mapping, report = build_mapping(categories_per_term, allowed_terms)
        print_report(field, mapping, report)
        result[field] = mapping

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWritten: {args.output}")


if __name__ == "__main__":
    main()
