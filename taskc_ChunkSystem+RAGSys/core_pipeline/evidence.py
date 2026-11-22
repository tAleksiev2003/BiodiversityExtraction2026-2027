"""Verify quoted text and aggregate findings without overlap-inflated votes."""
import hashlib
import json
import re
import unicodedata
from difflib import SequenceMatcher
from .schema import FIELDS, translate_value, validate_record


def normalized(text):
    """Normalize Unicode and whitespace while retaining original character offsets."""
    output, positions = [], []
    for index, char in enumerate(text):
        for expanded in unicodedata.normalize("NFKC", char):
            if expanded.isspace():
                if output and output[-1] != " ":
                    output.append(" ")
                    positions.append(index)
            else:
                output.append(expanded)
                positions.append(index)
    while output and output[-1] == " ":
        output.pop()
        positions.pop()
    return "".join(output), positions


def verify_quote(quote, chunk, threshold=1.0):
    """Locate a quote using normalized exact matching or optional conservative fuzziness."""
    if not isinstance(quote, str) or len(quote.strip()) < 12:
        return None
    needle, _ = normalized(quote)
    haystack, positions = normalized(chunk["text"])
    start = haystack.find(needle)
    score = 1.0
    end = start + len(needle)
    if start < 0 and threshold < 1:
        words = list(re.finditer(r"\S+", haystack))
        n = len(needle.split())
        best = (0.0, -1, -1)
        for i in range(len(words)):
            for size in {max(1, n - 1), n, n + 1}:
                if i + size > len(words):
                    continue
                a, b = words[i].start(), words[i + size - 1].end()
                ratio = SequenceMatcher(None, needle, haystack[a:b], autojunk=False).ratio()
                if ratio > best[0]:
                    best = (ratio, a, b)
        score, start, end = best
    if start < 0 or score < threshold:
        return None
    raw_start, raw_end = positions[start], positions[end - 1] + 1
    return {"chunk_id": chunk["id"], "quote": quote,
            "matched_text": chunk["text"][raw_start:raw_end], "match_score": score,
            "section_index": chunk["section_index"],
            "start": chunk["start"] + raw_start, "end": chunk["start"] + raw_end}


def finding_id(field, value):
    """Generate a stable identifier from the canonical field and value."""
    content = json.dumps([field, value], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()[:20]


def verified_changes(response, chunk, schema, labels, mode, threshold):
    """Reject malformed or unsupported findings and return verified additions/removals."""
    additions, removals, rejected = [], [], []
    for item in response["add"]:
        try:
            if not isinstance(item, dict) or set(item) != {"field", "value", "evidence"}:
                raise ValueError("Invalid finding structure")
            field = item["field"]
            if field not in FIELDS or not isinstance(item["evidence"], list) or not item["evidence"]:
                raise ValueError("Unknown field or missing evidence")
            value = translate_value(item["value"], schema["properties"][field]["items"], schema, labels)
            evidence = []
            for ev in item["evidence"]:
                if not isinstance(ev, dict) or set(ev) != {"quote", "chunk_id"} or ev["chunk_id"] != chunk["id"]:
                    raise ValueError("Invalid evidence or wrong chunk ID")
                match = verify_quote(ev["quote"], chunk, threshold)
                if match is None:
                    raise ValueError("Quote not found")
                evidence.append(match)
            additions.append({"id": finding_id(field, value), "field": field,
                              "value": value, "evidence": evidence})
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append({"item": item, "reason": str(exc)})
    for item in response["remove"]:
        if mode != "sequential" or not isinstance(item, dict) or set(item) != {"finding_id", "quote"}:
            rejected.append({"item": item, "reason": "Invalid removal"})
            continue
        evidence = verify_quote(item["quote"], chunk, threshold)
        if evidence and isinstance(item["finding_id"], str):
            removals.append({**item, "evidence": evidence})
        else:
            rejected.append({"item": item, "reason": "Unverified removal"})
    return additions, removals, rejected


def merge_findings(state, additions, removals=()):
    """Apply explicit removals and merge unique evidence into a canonical state."""
    indexed = {item["id"]: {**item, "evidence": list(item["evidence"])} for item in state}
    for item in removals:
        indexed.pop(item["finding_id"], None)
    for item in additions:
        existing = indexed.setdefault(item["id"], {**item, "evidence": []})
        keys = {(e["section_index"], e["start"], e["end"]) for e in existing["evidence"]}
        for ev in item["evidence"]:
            key = (ev["section_index"], ev["start"], ev["end"])
            if key not in keys:
                existing["evidence"].append(ev)
                keys.add(key)
    return sorted(indexed.values(), key=lambda item: item["id"])


def evidence_votes(evidence):
    """Count non-overlapping source spans rather than repeated chunk appearances."""
    votes = 0
    section, end = None, -1
    for ev in sorted(evidence, key=lambda e: (e["section_index"], e["start"], e["end"])):
        if section != ev["section_index"] or ev["start"] >= end:
            votes += 1
            section, end = ev["section_index"], ev["end"]
        else:
            end = max(end, ev["end"])
    return votes


# Biodiversity level is usually a repeated framing word. One extra vote is required.
BIODIVERSITY_LEVEL_MIN_VOTES = 3
_RECOMMEND = re.compile(
    r"zu untersuchen|\bsollte\b|\bsollten\b|\bempfehl\w*|"
    r"further research|future research|future work|"
    r"remains to be (investigated|studied|determined)|to be investigated",
    re.IGNORECASE)
_INDEX = re.compile(r"\b(index|evenness|simpson|jaccard|shannon|pielou|turnover|nestedness)\b", re.IGNORECASE)
_MEASURE = {"abundanz", "biomasse", "artenzahl", "artenreichtum", "diversität", "diversitaet",
            "evenness", "dichte", "bedeckung", "produktivität", "produktivitaet"}
_NAME_STOP = {"der", "die", "das", "den", "dem", "des", "ein", "eine", "und", "oder", "im", "in",
              "am", "an", "von", "vom", "für", "fur", "the", "and", "of", "for", "with", "auf",
              "dieser", "diese", "dieses", "jeder", "jede", "jenes", "jener", "jene"}
# Preceding token that can belong to a German common name. Neighboring list items do not qualify.
_NAME_ADJECTIVE = {"gemeine", "gemeiner", "gemeines", "echte", "echter", "echtes",
                   "kleine", "kleiner", "kleines", "große", "großer", "großes",
                   "gewöhnliche", "gewöhnlicher", "gewöhnliches", "rote", "roter", "rotes",
                   "schwarze", "schwarzer", "braune", "brauner", "braunes",
                   "nordische", "nordischer", "europäische", "europäischer",
                   "atlantische", "atlantischer"}
_SERVICE_STOP = _NAME_STOP | {"einschließlich", "einschliesslich", "sowie", "durch", "zur", "zum",
                              "ihre", "ihrer", "ihren", "seiner", "seine", "einer", "einem", "eines",
                              "including", "other", "from", "that", "this", "welche", "welcher"}


def votes_required(field, min_votes, level_min_votes=BIODIVERSITY_LEVEL_MIN_VOTES):
    """Return the evidence-span threshold for one field.

    Multi-chunk runs keep the biodiversity-level floor of 3. A single-pass caller
    passes level_min_votes equal to min_votes so that field uses the same bar.
    """
    if field == "biodiversity_level":
        return max(min_votes, level_min_votes)
    return min_votes


def submission_record(paper_id, state, schema, min_votes=2,
                      level_min_votes=BIODIVERSITY_LEVEL_MIN_VOTES):
    """Remove evidence metadata and create a validated C1/C2b submission object."""
    result = {"zotitem_ptr_id": paper_id, **{field: [] for field in FIELDS}}
    seen = set()
    for item in state:
        if evidence_votes(item["evidence"]) < votes_required(item["field"], min_votes, level_min_votes):
            continue
        key = json.dumps([item["field"], item["value"]], sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        result[item["field"]].append(item["value"])
    validate_record(result, schema)
    return result


def _section_role(heading):
    """Classify a section heading as title, abstract, methods, or other body text."""
    name = (heading or "").casefold()
    if name in {"title", "titel"}:
        return "title"
    if any(token in name for token in ("abstract", "zusammenfassung", "summary")):
        return "abstract"
    if any(token in name for token in ("method", "methode", "material", "sampling", "probenahme",
                                       "datenerhebung", "study area", "untersuchungsgebiet")):
        return "methods"
    return "other"


def _study_scope_text(chunks):
    """Join title, abstract, and methods. Return None when none of those sections exist."""
    parts = [chunk["text"] for chunk in chunks if _section_role(chunk.get("section")) in {"title", "abstract", "methods"}]
    return "\n".join(parts) if parts else None


def _tokens(text):
    """Return alphabetic tokens, keeping hyphenated compounds together."""
    return re.findall(r"[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß-]*", text or "")


def _is_species_rank(name):
    """True for a binomial, trinomial, or an explicit sp./spp. name."""
    parts = name.split()
    if re.search(r"\bspp?\.?$", name, re.IGNORECASE):
        return True
    return len(parts) >= 2 and parts[0][:1].isupper() and parts[1][:1].islower()


def _name_adjective(token):
    """True for a capitalized modifier such as Gemeine or Atlantischer, not for a neighboring taxon."""
    fold = token.casefold()
    if fold in _NAME_STOP or len(token) <= 2:
        return False
    return fold in _NAME_ADJECTIVE or fold.endswith(("ischer", "ische", "isches", "iger", "iges"))


def german_name_from_quotes(name, quotes):
    """Copy the German name from a quote. A shortening is expanded only within one noun phrase."""
    target = [part.casefold() for part in _tokens(name)]
    if not target:
        return None
    best = None
    for quote in quotes:
        tokens = _tokens(quote)
        folds = [token.casefold() for token in tokens]
        width = len(target)
        for index in range(len(folds)):
            surface = None
            if folds[index:index + width] == target:
                surface = " ".join(tokens[index:index + width])
                if index > 0 and _name_adjective(tokens[index - 1]):
                    surface = tokens[index - 1] + " " + surface
            elif width == 1 and folds[index].endswith(target[0]) and len(folds[index]) > len(target[0]) + 1:
                surface = tokens[index]
            if surface and (best is None or len(surface) > len(best)):
                best = surface
    return best


def collapse_variable(text):
    """Reduce a taxon-qualified variable to its measure. Keep index names as printed."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text or _INDEX.search(text):
        return text or None
    tokens = text.split()
    if len(tokens) == 1 and "-" in tokens[0]:
        tail = tokens[0].rsplit("-", 1)[-1].casefold()
        if tail in _MEASURE:
            return tokens[0].rsplit("-", 1)[-1]
    if len(tokens) <= 2:
        return text
    head = tokens[0].casefold().strip(".,;")
    if head in _MEASURE:
        return tokens[0].strip(".,;")
    if len(tokens) > 4:
        return None
    return text


def service_term_supported(term, quotes):
    """True when at least two distinctive words of a CICES term occur in the quotes."""
    words = [word for word in re.findall(r"[A-Za-zÄÖÜäöüß]{6,}", term)
             if word.casefold() not in _SERVICE_STOP]
    if not words:
        return False
    blob = " ".join(quotes).casefold()
    hits = sum(1 for word in words if word.casefold() in blob)
    return hits >= (2 if len(words) >= 2 else 1)


def _refine_value(field, value, evidence, scope):
    """Apply name, variable, and service-term constraints.

    A return value of None drops the finding. The note list records edits that
    keep the finding, plus the reason when it is dropped.
    """
    quotes = [item.get("quote") or "" for item in evidence]
    if field == "taxa" and isinstance(value, dict):
        updated = dict(value)
        notes = []
        scientific = updated.get("scientific_name")
        if isinstance(scientific, str) and _is_species_rank(scientific.strip()) and scope is not None:
            genus = re.sub(r"\bspp?\.?$", "", scientific.strip(), flags=re.IGNORECASE).split()[0]
            if not re.search(rf"\b{re.escape(genus)}\b", scope, re.IGNORECASE):
                updated["scientific_name"] = None
                notes.append("species name absent from title, abstract, and methods")
        german = updated.get("german_name")
        if isinstance(german, str) and german.strip() and german.strip() != "-":
            anchored = german_name_from_quotes(german.strip(), quotes)
            if anchored != german.strip():
                updated["german_name"] = anchored
                notes.append("german name copied from quote" if anchored else "german name absent from quote")
        if not updated.get("species_group"):
            return None, ["taxa missing species_group"]
        return updated, notes
    if field == "biodiversity_variable" and isinstance(value, str):
        collapsed = collapse_variable(value)
        if collapsed is None:
            return None, ["biodiversity variable has no bare measure"]
        if collapsed != value.strip():
            return collapsed, ["collapsed biodiversity variable to the bare measure"]
        return collapsed, []
    if field == "ecosystem_service" and isinstance(value, dict):
        updated = dict(value)
        term = updated.get("term")
        if isinstance(term, str) and term.strip() and not service_term_supported(term, quotes):
            updated["term"] = None
            return updated, ["cleared unsupported service term"]
        return updated, []
    return value, []


def apply_precision_filters(state, chunks):
    """Drop recommendation-only evidence and tighten names, variables, and service terms.

    Service findings whose term was cleared keep their own evidence. They are not
    merged with other category-only services, so several one-span mistakes cannot
    pool into one vote count. Other refined values are merged by field and value.
    """
    scope = _study_scope_text(chunks)
    mergeable, separate, dropped = [], [], []
    for item in state:
        evidence = [span for span in item["evidence"] if not _RECOMMEND.search(span.get("quote") or "")]
        if not evidence:
            dropped.append({"id": item["id"], "field": item["field"], "reason": "recommendation-only evidence"})
            continue
        if len(evidence) < len(item["evidence"]):
            dropped.append({"id": item["id"], "field": item["field"], "reason": "removed recommendation evidence"})
        value, notes = _refine_value(item["field"], item["value"], evidence, scope)
        if value is None:
            dropped.append({"id": item["id"], "field": item["field"], "reason": notes[0]})
            continue
        refined = {"id": finding_id(item["field"], value), "field": item["field"],
                   "value": value, "evidence": evidence}
        if "cleared unsupported service term" in notes:
            separate.append(refined)
        else:
            mergeable.append(refined)
        for reason in notes:
            dropped.append({"id": item["id"], "field": item["field"], "reason": reason})
    return merge_findings([], mergeable) + separate, dropped
