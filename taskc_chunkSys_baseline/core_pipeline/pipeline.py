"""Execute the two architectures using identical extraction and evidence rules."""
from concurrent.futures import ThreadPoolExecutor
from .schema import load_schema, catalogue, prompt_schema
from .prompts import build_messages
from .evidence import verified_changes, merge_findings, submission_record


def run_paper(paper_id, chunks, client, mode="parallel", workers=4, threshold=1.0, min_votes=1):
    """Process all chunks and return a submission record plus a complete audit trail."""
    if mode not in {"parallel", "sequential"}:
        raise ValueError("mode must be parallel or sequential")
    if not chunks or workers < 1 or min_votes < 1 or not 0.9 <= threshold <= 1:
        raise ValueError("Invalid processing parameters or empty chunks")
    schema = load_schema()
    labels = catalogue(schema)
    state, audit = [], []

    def process(chunk, previous):
        """Extract one chunk and verify every proposed state change."""
        messages = build_messages(chunk, previous, prompt_schema(schema), labels, mode)
        response, metadata = client.complete(messages)
        additions, removals, rejected = verified_changes(response, chunk, schema, labels, mode, threshold)
        return additions, removals, {"chunk_id": chunk["id"], "response": response,
                                     "metadata": metadata, "rejected": rejected,
                                     "verified_removals": removals}

    if mode == "parallel":
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process, chunk, []) for chunk in chunks]
            for future in futures:  # Preserve document order regardless of completion order.
                additions, removals, entry = future.result()
                state = merge_findings(state, additions)
                audit.append(entry)
    else:
        for chunk in chunks:
            additions, removals, entry = process(chunk, state)
            known_ids = {item["id"] for item in state}
            entry["unknown_removal_ids"] = [r["finding_id"] for r in removals if r["finding_id"] not in known_ids]
            state = merge_findings(state, additions, removals)
            entry["state_after"] = state
            audit.append(entry)
    return submission_record(paper_id, state, schema, min_votes), {"chunks": chunks, "calls": audit, "final_state": state}
