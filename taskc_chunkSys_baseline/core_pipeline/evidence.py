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


def submission_record(paper_id, state, schema, min_votes=1):
    """Remove evidence metadata and create a validated C1/C2b submission object."""
    result = {"zotitem_ptr_id": paper_id, **{field: [] for field in FIELDS}}
    for item in state:
        if evidence_votes(item["evidence"]) >= min_votes:
            result[item["field"]].append(item["value"])
    validate_record(result, schema)
    return result
