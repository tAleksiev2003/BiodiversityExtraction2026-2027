"""Grouped page-RAG and a matching all-page extraction control."""
import json
from .client import IncompleteResponse
from .documents import split_chunk
from .prompts import SYSTEM, RULES
from .schema import FIELDS, load_schema, catalogue, prompt_schema
from .evidence import (verified_changes, merge_findings, apply_precision_filters,
                       submission_record, evidence_votes, votes_required)
from .retrieval import GROUPS, page_units, rank_pages


def group_messages(sources, group, schema, labels):
    """Build identical multi-source prompts for retrieved and all-page contexts."""
    spec = prompt_schema(schema)
    spec["properties"] = {k: v for k, v in spec["properties"].items() if k in group["fields"]}
    spec["required"] = list(group["fields"])
    rules = RULES.split("Parallel mode:")[0].replace("CURRENT_CHUNK", "SOURCES").replace("CURRENT_ID", "SOURCE_ID")
    rules += "\nExtract ONLY ALLOWED_FIELDS. remove MUST be []. Each evidence chunk_id must identify " \
             "one supplied source. Quotes must be contiguous within that source, never across sources. " \
             "Retrieved sources are candidates, not proof of a positive label. Return empty add when " \
             "unsupported. Provide distinct supporting quotes when available; do not repeat a quote."
    payload = {"mode": "all-fields" if group["fields"] == list(FIELDS) else "grouped", "ALLOWED_FIELDS": group["fields"], "OUTPUT_SCHEMA": spec,
               "ENUM_CATALOGUE": labels, "PREVIOUS_VERIFIED_STATE": [], "SOURCES": sources}
    return [{"role": "system", "content": SYSTEM + rules},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def message_bytes(messages):
    """Use the same byte-based input guard as the baseline chat client."""
    return sum(len(m["content"].encode()) + 64 for m in messages)


def pack_sources(sources, group, schema, labels, client):
    """Pack all selected text into bounded calls, splitting long pages without omission."""
    batches, current = [], []
    queue = list(sources)
    while queue:
        source = queue.pop(0)
        proposed = current + [source]
        fits = message_bytes(group_messages(proposed, group, schema, labels)) + client.max_output <= client.context_budget
        if fits:
            current = proposed
        elif current:
            batches.append(current)
            current = []
            queue.insert(0, source)
        else:
            parts = split_chunk(source)
            if not parts:
                raise ValueError("Group prompt/source cannot fit context budget; raise budget or shorten schema")
            queue[0:0] = parts
    if current:
        batches.append(current)
    return batches


def verify_group(response, sources, fields, schema, labels, threshold):
    """Accept an item only when every citation is valid in a supplied source."""
    by_id = {s["id"]: s for s in sources}
    additions, rejected = [], []
    for item in response["add"]:
        try:
            if not isinstance(item, dict) or set(item) != {"field", "value", "evidence"}:
                raise ValueError("Invalid finding structure")
            if item["field"] not in fields or not isinstance(item["evidence"], list) or not item["evidence"]:
                raise ValueError("Out-of-group field or missing evidence")
            partial = []
            for ev in item["evidence"]:
                if not isinstance(ev, dict) or set(ev) != {"chunk_id", "quote"}:
                    raise ValueError("Invalid evidence structure")
                source = by_id.get(ev["chunk_id"])
                if source is None:
                    raise ValueError("Evidence source was not supplied to this call")
                found, _, errors = verified_changes(
                    {"add": [{**item, "evidence": [ev]}], "remove": []},
                    source, schema, labels, "parallel", threshold)
                if errors:
                    raise ValueError(errors[0]["reason"])
                for finding in found:
                    for span in finding["evidence"]:
                        span.update({"paper_id": source["paper_id"], "page_number": source["page_number"]})
                partial = merge_findings(partial, found)
            additions = merge_findings(additions, partial)
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append({"item": item, "reason": str(exc)})
    rejected.extend({"item": item, "reason": "Removals forbidden in grouped extraction"}
                    for item in response["remove"])
    return additions, rejected


def run_grouped(paper_id, pages, client, embedder=None, mode="rag", top_k=4,
                threshold=1.0, min_votes=2):
    """Select pages and extract separate field groups or their all-field union."""
    if mode not in {"rag", "grouped-all", "rag-all-fields"} or top_k < 1 or min_votes < 1 or not 0.9 <= threshold <= 1:
        raise ValueError("Invalid grouped extraction settings")
    units = page_units(paper_id, pages)
    if not units:
        raise ValueError("No nonempty pages")
    if mode in {"rag", "rag-all-fields"} and embedder is None:
        raise ValueError("RAG requires an embedding client")
    schema = load_schema()
    labels = catalogue(schema)
    before = len(embedder.calls) if embedder else 0
    rankings, windows = rank_pages(units, embedder) if mode in {"rag", "rag-all-fields"} else ({}, [])
    index = {u["id"]: u for u in units}
    state, calls, retrieval = [], [], []
    extraction_groups = {**GROUPS, "all_fields": {"fields": list(FIELDS)}}
    extraction_batches = []

    def extract(sources, group_name, depth=0):
        """Extract a batch and recursively recover incomplete outputs without silent skips."""
        nonlocal state
        group = extraction_groups[group_name]
        messages = group_messages(sources, group, schema, labels)
        try:
            response, metadata = client.complete(messages)
        except IncompleteResponse as exc:
            calls.append({"group": group_name, "sources": sources,
                          "metadata": {"cache_hit": False, "truncated": True, "usage": {}},
                          "error": str(exc), "rejected": []})
            if depth >= 8:
                raise ValueError("Completion remains truncated after eight splits") from exc
            if len(sources) > 1:
                middle = len(sources) // 2
                children = [sources[:middle], sources[middle:]]
            else:
                parts = split_chunk(sources[0])
                if not parts:
                    raise ValueError("Unsplittable source completion is truncated") from exc
                children = [[part] for part in parts]
            for child in children:
                extract(child, group_name, depth + 1)
            return
        additions, rejected = verify_group(response, sources, group["fields"], schema, labels, threshold)
        state = merge_findings(state, additions)
        calls.append({"group": group_name, "sources": sources, "response": response,
                      "metadata": metadata, "rejected": rejected})

    selected_by_group = {}
    for name, group in GROUPS.items():
        selected = [index[x["unit_id"]] for x in rankings[name][:top_k]] if mode in {"rag", "rag-all-fields"} else list(units)
        selected.sort(key=lambda u: u["page_number"])
        selected_by_group[name] = selected
        retrieval.append({"group": name, "query": group["query"], "fields": group["fields"],
                          "ranking": rankings.get(name, []),
                          "selected_ids": [u["id"] for u in selected]})
    if mode == "rag-all-fields":
        selected_ids = {u["id"] for selected in selected_by_group.values() for u in selected}
        jobs = [("all_fields", [u for u in units if u["id"] in selected_ids])]
    else:
        jobs = list(selected_by_group.items())
    for name, selected in jobs:
        selected.sort(key=lambda u: u["page_number"])
        batches = pack_sources(selected, extraction_groups[name], schema, labels, client)
        extraction_batches.append({"group": name, "fields": extraction_groups[name]["fields"],
                                   "selected_ids": [u["id"] for u in selected], "initial_batches": len(batches)})
        if mode != "rag-all-fields":
            next(r for r in retrieval if r["group"] == name)["initial_batches"] = len(batches)
        for batch in batches:
            extract(batch, name)
    state, dropped = apply_precision_filters(state, units)
    # One pass over each page cannot repeat a span the way overlapping chunks can.
    level_min_votes = min_votes
    vote_audit = [{"finding_id": f["id"], "field": f["field"],
                   "votes": evidence_votes(f["evidence"]),
                   "required": votes_required(f["field"], min_votes, level_min_votes),
                   "kept": evidence_votes(f["evidence"]) >= votes_required(f["field"], min_votes, level_min_votes)}
                  for f in state]
    record = submission_record(paper_id, state, schema, min_votes, level_min_votes)
    return record, {"mode": mode, "chunks": units, "calls": calls, "retrieval": retrieval,
                    "extraction_batches": extraction_batches,
                    "embedding_windows": windows, "embedding_calls": embedder.calls[before:] if embedder else [],
                    "total_pages": len(pages), "empty_pages": [int(n) for n, t in pages if not t.strip()],
                    "precision_filter": dropped, "vote_filter": vote_audit, "final_state": state,
                    "provenance_notes": ["Page 1 contains synthetically prepended title/abstract.",
                        "Cross-page TEI elements retain the baseline first-page assignment.",
                        "Offsets are Python character slices into canonical page text; section_index is page-1.",
                        "Page labels preserve the baseline behavior that skips section-based species scope checks."]}
