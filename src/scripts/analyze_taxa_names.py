"""Count how taxa names are written in the training references.

taxa.scientific_name and taxa.german_name are scored by exact match, so the model must
follow the spelling habits of the database (e.g. "Fucus spp" without a dot). This
script counts those habits, so the team can write the {taxa_naming_conventions} rules
for the extraction prompt.

It prints only counts and shares. With --examples N it also prints up to N example
names per pattern; those are course data: look at them yourself, do not share them.

Usage:
    python src/scripts/analyze_taxa_names.py --references data/raw/references/train.jsonl
    python src/scripts/analyze_taxa_names.py --references data/raw/references/train.jsonl --examples 3
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# pattern name -> regular expression, checked against every scientific name
SCIENTIFIC_PATTERNS = {
    "ends with ' spp'": r" spp$",
    "ends with ' spp.'": r" spp\.$",
    "ends with ' sp'": r" sp$",
    "ends with ' sp.'": r" sp\.$",
    "contains 'ssp.' or 'subsp.'": r"\b(ssp|subsp)\.",
    "contains 'agg.', 's.l.' or 's.str.'": r"\b(agg\.|s\.\s?l\.|s\.\s?str\.)",
    "contains 'cf.' or 'aff.'": r"\b(cf|aff)\.",
    "contains a year (author citation)": r"\b1[6-9]\d\d\b|\b20\d\d\b",
    "contains ' L.' (Linnaeus)": r" L\.(\s|$)",
    "contains parentheses": r"[()]",
    "contains '&' or ','": r"[&,]",
    "contains hybrid sign 'x' or '×'": r"(^| )[x×]( |$)",
    "starts with a lowercase letter": r"^[a-zäöü]",
}
GERMAN_PATTERNS = {
    "starts with an uppercase letter": r"^[A-ZÄÖÜ]",
    "starts with a lowercase letter": r"^[a-zäöü]",
    "contains a hyphen": r"-",
    "contains a space (several words)": r" ",
    "contains a comma or slash": r"[,/]",
    "contains parentheses": r"[()]",
}


def read_taxa(references_path: Path) -> list[dict]:
    taxa = []
    with references_path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                taxa.extend(json.loads(line).get("taxa") or [])
    return taxa


def present(value) -> bool:
    return isinstance(value, str) and value.strip() != ""


def count_patterns(names: list[str], patterns: dict[str, str]) -> tuple[Counter, dict[str, list[str]]]:
    counts: Counter = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for name in names:
        for label, pattern in patterns.items():
            if re.search(pattern, name):
                counts[label] += 1
                examples[label].append(name)
    return counts, examples


def word_count_shares(names: list[str]) -> Counter:
    return Counter(min(len(name.split()), 4) for name in names)  # 4 = "4 or more"


def share(part: int, total: int) -> str:
    return f"{part:6d}  ({100 * part / total:5.1f} %)" if total else f"{part:6d}"


def print_patterns(title: str, names: list[str], patterns: dict[str, str], max_examples: int) -> None:
    counts, examples = count_patterns(names, patterns)
    print(f"\n=== {title}: {len(names)} names ===")
    for label in patterns:
        print(f"   {label:40s} {share(counts[label], len(names))}")
        for example in examples[label][:max_examples]:
            print(f"      e.g. {example}")
    print("   number of words:")
    words = word_count_shares(names)
    for number in sorted(words):
        label = "4 or more" if number == 4 else str(number)
        print(f"      {label:37s} {share(words[number], len(names))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=0, help="example names per pattern (course data!)")
    args = parser.parse_args()

    taxa = read_taxa(args.references)
    scientific = [t["scientific_name"].strip() for t in taxa if present(t.get("scientific_name"))]
    german = [t["german_name"].strip() for t in taxa if present(t.get("german_name"))]

    print(f"taxa entries: {len(taxa)}")
    print(f"   with scientific name:   {share(len(scientific), len(taxa))}")
    print(f"   with German name:       {share(len(german), len(taxa))}")
    both = sum(present(t.get("scientific_name")) and present(t.get("german_name")) for t in taxa)
    neither = sum(not present(t.get("scientific_name")) and not present(t.get("german_name")) for t in taxa)
    print(f"   with both names:        {share(both, len(taxa))}")
    print(f"   with no name at all:    {share(neither, len(taxa))}")

    collective = [t for t in taxa if t.get("collective_term") is True]
    print(f"\ncollective terms (collective_term = true): {share(len(collective), len(taxa))}")
    print(f"   of these with scientific name: {share(sum(present(t.get('scientific_name')) for t in collective), len(collective))}")
    print(f"   of these with German name:     {share(sum(present(t.get('german_name')) for t in collective), len(collective))}")

    print_patterns("scientific names", scientific, SCIENTIFIC_PATTERNS, args.examples)
    print_patterns("German names", german, GERMAN_PATTERNS, args.examples)


if __name__ == "__main__":
    main()
