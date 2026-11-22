"""Execute the two architectures using identical extraction and evidence rules."""
from concurrent.futures import ThreadPoolExecutor
from .client import IncompleteResponse
from .documents import split_chunk
from .schema import load_schema, catalogue, prompt_schema
from .prompts import build_messages
from .evidence import verified_changes, merge_findings, submission_record, apply_precision_filters


def run_paper(paper_id, chunks, client, mode="parallel", workers=4, threshold=1.0, min_votes=2):
    """Process all chunks and return a submission record plus a complete audit trail."""
    if mode not in {"parallel", "sequential"}:
        raise ValueError("mode must be parallel or sequential")
    if not chunks or workers < 1 or min_votes < 1 or not 0.9 <= threshold <= 1:
        raise ValueError("Invalid processing parameters or empty chunks")
    schema = load_schema()
    labels = catalogue(schema)
    state, audit = [], []

    def process(chunk, previous, depth=0):
        """Extract one chunk. Split it when the completion is cut off by max_tokens."""
        messages = build_messages(chunk, previous, prompt_schema(schema), labels, mode)
        try:
            response, metadata = client.complete(messages)
        except IncompleteResponse as exc:
            parts = split_chunk(chunk) if depth < 8 else []
            if not parts:
                print(f"  skip {chunk['id']}: {exc}", flush=True)
                entry = {"chunk_id": chunk["id"], "response": {"add": [], "remove": []},
                         "metadata": {"cache_hit": False, "truncated": True, "elapsed_s": 0,
                                      "usage": {}, "error": str(exc)},
                         "rejected": [], "verified_removals": []}
                return [], [], [entry]
            print(f"  split {chunk['id']} -> {parts[0]['id']}, {parts[1]['id']}", flush=True)
            additions, removals, entries = [], [], []
            for part in parts:
                part_add, part_remove, part_entries = process(part, previous, depth + 1)
                additions.extend(part_add)
                removals.extend(part_remove)
                entries.extend(part_entries)
            return additions, removals, entries
        additions, removals, rejected = verified_changes(response, chunk, schema, labels, mode, threshold)
        entry = {"chunk_id": chunk["id"], "response": response, "metadata": metadata,
                 "rejected": rejected, "verified_removals": removals}
        return additions, removals, [entry]

    if mode == "parallel":
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process, chunk, []) for chunk in chunks]
            for future in futures:  # Preserve document order regardless of completion order.
                additions, removals, entries = future.result()
                state = merge_findings(state, additions)
                audit.extend(entries)
    else:
        for chunk in chunks:
            additions, removals, entries = process(chunk, state)
            known_ids = {item["id"] for item in state}
            state = merge_findings(state, additions, removals)
            for entry in entries:
                entry["unknown_removal_ids"] = [
                    r["finding_id"] for r in entry["verified_removals"]
                    if r["finding_id"] not in known_ids]
                entry["state_after"] = state
            audit.extend(entries)
    state, dropped = apply_precision_filters(state, chunks)
    return submission_record(paper_id, state, schema, min_votes), {
        "chunks": chunks, "calls": audit, "precision_filter": dropped, "final_state": state}
