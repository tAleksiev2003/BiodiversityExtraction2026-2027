"""Plural-convention check for taxa.german_name over ALL train + dev gold labels.

Questions:
1. Does the database always use plural forms for German taxa names, or mixed?
   -> for every singular/plural candidate pair in the train vocabulary, how often
      is each side used.
2. How much would a train-vocabulary plural normalisation fix on dev for our
   best run? -> predicted singular names whose suffixed/umlauted form exists in
   the train vocab AND matches that paper's gold.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
from src.evaluation.evaluator import extract_field_set, load_jsonl_by_id  # noqa: E402

train = load_jsonl_by_id(Path("data/gold/train.jsonl"))
dev = load_jsonl_by_id(Path("data/gold/dev.jsonl"))
DEV_IDS = {l.strip() for l in open("data/gold/dev_100_ids.txt") if l.strip()}
RUN = "outputs/architectures/llm_rag_sections_saia_deepseek-v4.1-flash-1017_autoparser_ctaxa_v4_biknn"
preds = load_jsonl_by_id(Path(sorted(glob.glob(RUN + "/*_predictions.jsonl"))[0])) if (glob := __import__("glob")) else None


def name_counts(records, only_ids=None):
    counts = Counter()
    for zid, rec in records.items():
        if only_ids is not None and zid not in only_ids:
            continue
        for value in extract_field_set(rec, "taxa.german_name"):
            counts[value] += 1
    return counts


train_names = name_counts(train)
dev_names = name_counts(dev, DEV_IDS)

# --- 1) singular/plural candidate pairs in the train vocabulary -------------
SUFFIXES = ("e", "n", "en", "s")
UMLAUT = str.maketrans("aou", "äöü")


def singular_candidates(plural):
    """Generate plausible singular forms of a vocab plural (suffix strip + un-umlaut)."""
    out = set()
    for suf in SUFFIXES:
        if plural.endswith(suf) and len(plural) > len(suf) + 3:
            stem = plural[: -len(suf)]
            out.add(stem)
            out.add(stem.translate({ord(c): chr(ord(c) - 1) for c in ()}))  # placeholder
            # un-umlaut every possible vowel (ä->a, ö->o, ü->u), one at a time
            for i, ch in enumerate(stem):
                if ch in "äöü":
                    plain = stem[:i] + {"ä": "a", "ö": "o", "ü": "u"}[ch] + stem[i + 1:]
                    out.add(plain)
    return out


pairs = []
for plural in train_names:
    for singular in singular_candidates(plural):
        if singular in train_names:
            pairs.append((singular, plural, train_names[singular], train_names[plural]))
            break
pairs.sort(key=lambda t: -(t[2] + t[3]))

print("=== 1) singular/plural pairs in TRAIN gold vocabulary (top 20 by usage) ===")
print("   singular -> plural | papers with singular | papers with plural")
plural_only = both = singular_only = 0
for singular, plural, n_s, n_p in pairs[:20]:
    kind = "BOTH" if (n_s >= 2 and n_p >= 2) else ("plural-only" if n_p > n_s else "singular-only")
    print("   %-24s -> %-24s %4d | %4d   [%s]" % (singular, plural, n_s, n_p, kind))
for singular, plural, n_s, n_p in pairs:
    if n_s >= 2 and n_p >= 2:
        both += 1
    elif n_p > n_s:
        plural_only += 1
    else:
        singular_only += 1
print("   summary: plural-dominant %d | used-as-both %d | singular-dominant %d (pairs=%d)" % (
    plural_only, both, singular_only, len(pairs)))

print()
print("=== specific lemmas (train / dev usage) ===")
for word in ("Seehund", "Seehunde", "Fisch", "Fische", "Vogel", "Vögel", "Wal", "Wale",
             "Insekt", "Insekten", "Robbe", "Robben", "Kegelrobbe", "Kegelrobben",
             "Pflanze", "Pflanzen", "Baum", "Bäume"):
    print("   %-14s train %3d | dev %2d" % (word, train_names.get(word, 0), dev_names.get(word, 0)))

# --- 2) how much would plural normalisation fix on dev ----------------------
VOCAB_PLURALS = {p for p in train_names}
single_e = {"e", "n", "en"}


def normalise(name):
    """Predicted singular -> train-vocab plural, if unique and unambiguous."""
    if name in VOCAB_PLURALS:
        return None  # already a vocab form, nothing to do
    cands = []
    stems = {name}
    for suf in single_e:
        if name.endswith(suf) and len(name) > len(suf) + 3:
            stems.add(name[: -len(suf)])
    for stem in stems:
        for variant in (
            stem + "e", stem + "n", stem + "en", stem + "s",
            stem.translate(UMLAUT) + "e", stem.translate(UMLAUT),
        ):
            if variant in VOCAB_PLURALS and variant != name:
                cands.append(variant)
    cands = sorted(set(cands))
    return cands[0] if len(cands) == 1 else None


fixed_papers = fixed_labels = 0
examples = []
for zid in DEV_IDS:
    g, p = dev.get(zid, {}), preds.get(zid, {})
    gold_set = extract_field_set(g, "taxa.german_name")
    if not gold_set:
        continue
    pred_set = extract_field_set(p, "taxa.german_name")
    fixed_here = []
    for name in pred_set - gold_set:
        target = normalise(name)
        if target and target in gold_set:
            fixed_here.append((name, target))
    if fixed_here:
        fixed_papers += 1
        fixed_labels += len(fixed_here)
        examples.append((zid, fixed_here[:3]))

print()
print("=== 2) plural normalisation on dev (best run) ===")
print("   papers fixable: %d | labels flipped FP->TP: %d" % (fixed_papers, fixed_labels))
for zid, fixes in examples:
    print("   %s: %s" % (zid, ", ".join("%s -> %s" % f for f in fixes)))
