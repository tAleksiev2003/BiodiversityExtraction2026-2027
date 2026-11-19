"""Give every ecosystem_service term a category by the first number of its code.

The service terms follow CICES: the code of each term starts with its section, e.g.
"1.1.2.3 ..." belongs to the same category as every other "1.x" term. Eight schema terms
never occur in the training references, so build_term_to_category.py cannot map them,
and check C2 of the verbatim checker drops them. This script fills them in.

The rule (first number -> category) is not typed in here: it is read from the terms that
build_term_to_category.py already mapped, and it is only used if all mapped terms with
that first number agree. Before writing, the rule is checked against every
(category, term) pair in the reference files; any pair that breaks it stops the script.

Run it after build_term_to_category.py (which rewrites the file without these terms):

    python3 src/scripts/map_service_terms_by_prefix.py            # check train + dev, then write
    python3 src/scripts/map_service_terms_by_prefix.py --dry-run  # check and print only

The output contains only schema values and counts, no text from the papers.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import from src/
from data import read_jsonl  # noqa: E402
from labels import load_field_specs, load_term_to_category  # noqa: E402
from settings import load_settings  # noqa: E402

SERVICE_FIELD = "ecosystem_service"
REFERENCE_FILES = ("references_train", "references_dev_with_pdf")  # keys in settings.yaml paths


def code_prefix(term: str) -> str:
    """'2.2.1.5 Feuerschutz' -> '2': the CICES section of a term."""
    code = term.split(" ", 1)[0]
    return code.split(".", 1)[0]


def derive_prefix_rule(mapped: dict[str, str]) -> dict[str, str]:
    """{first number: category} from already mapped terms; a prefix with two categories is an error."""
    categories_per_prefix: dict[str, set[str]] = defaultdict(set)
    for term, category in mapped.items():
        categories_per_prefix[code_prefix(term)].add(category)
    conflicts = {prefix: sorted(cats) for prefix, cats in categories_per_prefix.items() if len(cats) > 1}
    if conflicts:
        raise ValueError(f"Mapped terms disagree within a prefix, no rule possible: {conflicts}")
    return {prefix: next(iter(cats)) for prefix, cats in sorted(categories_per_prefix.items())}


def check_pairs(pairs: Iterable[tuple[str, str]], rule: dict[str, str]) -> tuple[Counter, Counter]:
    """Count (category, term) pairs that follow the rule and those that do not.

    Returns (follow counts per prefix, breaking (term, category) pairs with counts).
    A term whose prefix has no rule counts as breaking it.
    """
    follows: Counter = Counter()
    breaks: Counter = Counter()
    for category, term in pairs:
        prefix = code_prefix(term)
        if rule.get(prefix) == category:
            follows[prefix] += 1
        else:
            breaks[(term, category)] += 1
    return follows, breaks


def reference_pairs(records: Iterable[dict[str, Any]]) -> Iterable[tuple[str, str]]:
    """Every (category, term) pair with a term in the ecosystem_service field of the references."""
    for record in records:
        for entry in record.get(SERVICE_FIELD) or []:
            if entry.get("term") and entry.get("category"):
                yield entry["category"], entry["term"]


def fill_by_prefix(schema_terms: tuple[str, ...], mapped: dict[str, str],
                   rule: dict[str, str]) -> dict[str, str]:
    """{term: category} for the schema terms that have no entry yet (schema order)."""
    added = {}
    for term in schema_terms:
        if term not in mapped:
            if code_prefix(term) not in rule:
                raise ValueError(f"No rule for the prefix of {term!r}")
            added[term] = rule[code_prefix(term)]
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="check and print, write nothing")
    args = parser.parse_args()

    settings = load_settings()
    paths = settings["paths"]
    specs = load_field_specs([paths["schema_core"], paths["schema_biodiversity"]])
    schema_terms = specs[f"{SERVICE_FIELD}.term"].allowed_values
    term_to_category = load_term_to_category(paths["term_to_category"])
    mapped = term_to_category[SERVICE_FIELD]

    rule = derive_prefix_rule(mapped)
    print(f"Rule from {len(mapped)} mapped terms (first number of the code -> category):")
    for prefix, category in rule.items():
        print(f"   {prefix}.x -> {category}")

    all_follow = True
    for name in REFERENCE_FILES:
        follows, breaks = check_pairs(reference_pairs(read_jsonl(paths[name])), rule)
        total = sum(follows.values()) + sum(breaks.values())
        print(f"\nCheck {name}: {total} (category, term) pairs, {sum(follows.values())} follow the rule, "
              f"{sum(breaks.values())} break it")
        for prefix, count in sorted(follows.items()):
            print(f"   {prefix}.x: {count} follow")
        for (term, category), count in breaks.most_common():
            print(f"   BREAKS ({count}x): {term!r} with {category!r}")
        all_follow = all_follow and not breaks

    added = fill_by_prefix(schema_terms, mapped, rule)
    print(f"\nTerms without an entry, mapped by prefix: {len(added)}")
    for term, category in added.items():
        print(f"   {term} -> {category}")

    if not all_follow:
        sys.exit("Some reference pairs break the prefix rule: nothing written. Check them by hand.")
    if args.dry_run:
        print("\nDry run: nothing written.")
        return
    term_to_category[SERVICE_FIELD] = {**mapped, **added}
    paths["term_to_category"].write_text(
        json.dumps(term_to_category, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWritten: {paths['term_to_category']}")


if __name__ == "__main__":
    main()
