"""Build the ecosystem_type term -> category mapping from the schema description.

The description of EcosystemTypeSimple in core_fields.json is the official and complete
source: for every category it lists the allowed ecosystem types. It spells some names
differently from EcosystemTypeTermEnum (the values that are scored), e.g.
"Wälder (subalpin)" vs. "Subalpine Wälder". This script

1. splits the description into one list of names per category,
2. matches every name to exactly one term of the enum (word overlap, one-to-one),
3. checks the result against the pairs seen in the training references
   (the ecosystem_type part of an existing term_to_category.json built by
   build_term_to_category.py), and
4. replaces the ecosystem_type part of that file. ecosystem_service is left as it is.

Every match that is not exact is printed, so a human can check it.

Usage:
    python src/scripts/build_ecosystem_type_mapping.py \
        --schema data/schemas/core_fields.json \
        --mapping data/labels/term_to_category.json
"""
import argparse
import json
import re
from pathlib import Path

# Phrase in the description that introduces the list of a category -> Roman numeral
# at the start of the category name in EcosystemTypeCategoryEnum.
CATEGORY_PHRASES = [
    ("Meere oder Küsten", "I"),
    ("Binnengewässer", "II"),
    ("terrestrisches oder semiterrestrisches", "III"),
    ("technische Biotope", "IV"),
    ("Alpen", "V"),
    ("Weitere", "VI"),
]
LIST_START = "möglich:"


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def split_description(description: str) -> dict[str, list[str]]:
    """Numeral -> list of names, read from the 'möglich: a; b; c.' parts in order."""
    parts = description.split(LIST_START)
    intros, lists = parts[:-1], parts[1:]
    if len(lists) != len(CATEGORY_PHRASES):
        raise SystemExit(f"expected {len(CATEGORY_PHRASES)} lists in the description, found {len(lists)}")
    names_per_numeral = {}
    for (phrase, numeral), intro, raw_list in zip(CATEGORY_PHRASES, intros, lists):
        if phrase not in normalise(intro):
            raise SystemExit(f"list {numeral}: expected the phrase '{phrase}' before it")
        list_text = re.split(r"\b(?:Wenn|Passt)\b", raw_list)[0]  # stop at the next sentence
        names = [normalise(name).rstrip(".") for name in list_text.split(";")]
        names_per_numeral[numeral] = [name for name in names if name]
    return names_per_numeral


def similarity(name: str, term: str) -> float:
    """Share of words that match, in both directions; a word matches if one is a prefix
    of the other ('subalpin' ~ 'subalpinen'). 1.0 = same words."""
    name_words, term_words = words(name), words(term)

    def covered(source: list[str], target: list[str]) -> float:
        hits = sum(any(a.startswith(b) or b.startswith(a) for b in target) for a in source)
        return hits / len(source) if source else 0.0

    return (covered(name_words, term_words) + covered(term_words, name_words)) / 2


def match_names_to_terms(names: list[str], terms: list[str]) -> dict[str, str]:
    """One-to-one matching, best pairs first (exact equal names win automatically)."""
    candidates = sorted(
        ((1.0 + 1.0 if name == term else similarity(name, term), name, term)
         for name in names for term in terms),
        reverse=True,
    )
    name_to_term: dict[str, str] = {}
    used_terms: set[str] = set()
    for score, name, term in candidates:
        if score <= 0 or name in name_to_term or term in used_terms:
            continue
        name_to_term[name] = term
        used_terms.add(term)
    return name_to_term


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    args = parser.parse_args()

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    defs = schema["$defs"]
    terms = defs["EcosystemTypeTermEnum"]["enum"]
    categories = {cat.split(" ", 1)[0]: cat for cat in defs["EcosystemTypeCategoryEnum"]["enum"]}
    names_per_numeral = split_description(defs["EcosystemTypeSimple"]["description"])

    all_names = [name for names in names_per_numeral.values() for name in names]
    name_to_term = match_names_to_terms(all_names, terms)

    from_schema: dict[str, str] = {}
    print("=== matches that are not exact (check by hand) ===")
    for numeral, names in names_per_numeral.items():
        for name in names:
            term = name_to_term.get(name)
            if term is None:
                raise SystemExit(f"no enum term found for '{name}'")
            if term != name:
                print(f"   {numeral:>3}  {name}  ->  {term}")
            from_schema[term] = categories[numeral]

    missing = sorted(set(terms) - set(from_schema))
    print(f"\nterms in the enum: {len(terms)}, mapped from the description: {len(from_schema)}")
    if missing:
        raise SystemExit(f"enum terms not found in the description: {missing}")

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    from_train = mapping.get("ecosystem_type", {})
    conflicts = {t: (c, from_schema[t]) for t, c in from_train.items() if from_schema.get(t) != c}
    print(f"check against training references: {len(from_train) - len(conflicts)} of "
          f"{len(from_train)} pairs agree, {len(conflicts)} conflicts")
    for term, (train_cat, schema_cat) in conflicts.items():
        print(f"   CONFLICT {term}: train says '{train_cat}', description says '{schema_cat}'")
    if conflicts:
        raise SystemExit("not written: resolve the conflicts first")

    mapping["ecosystem_type"] = dict(sorted(from_schema.items()))
    args.mapping.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWritten: {args.mapping} (ecosystem_type: {len(from_schema)} terms; "
          f"ecosystem_service unchanged: {len(mapping.get('ecosystem_service', {}))} terms)")


if __name__ == "__main__":
    main()
