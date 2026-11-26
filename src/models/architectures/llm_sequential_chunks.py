#!/usr/bin/env python3
"""
========================================================================================
EXECUTIVE SUMMARY — Architecture 2, Variant A: Sequential Chunk Pipeline (State Fold)
========================================================================================
- Concept & Goal:
    Second architecture-level model, implementing variant A of the team's agreed
    chunk pipeline (Google-Doc diagram, 11.11.2026): the paper is split into section
    chunks; chunks are read ONE AFTER ANOTHER by an LLM, and the JSON label state is
    passed from call i to call i+1, which may add labels (quote from the current
    section), keep confirmed ones, or remove contradicted ones. Research question:
    does carrying context across sections beat independent calls (Model 3)?
- How It Works:
    1. Chunking: pypdf full text (cached from Model 1) is split into sections by a
       heading heuristic (GROBID substitute; Shaukat et al.: sections beat fixed
       token windows). References/acknowledgements are dropped.
    2. Sequential fold: each LLM call receives the current state (labels + verbatim
       quotes + source sections) plus ONE section, and returns the updated full
       state (English labels; every new label must quote the current section).
    3. Python verification: after each fold, quotes of labels newly attributed to
       the current section are fuzzy-verified against that section's text; a label
       that is never verifiable in any section it passed through is dropped at the
       end (hallucination filter).
    4. Final step: deterministic EN->DE label mapping onto the official German enum
       strings, dedup, collective-term conventions, taxa/variable caps; then
       evaluation + `save_reproducible_run()` artifacts.
    Provider: HU-Berlin bridge (llm3, Qwen3.6-27B thinking, sampling pinned
    server-side). Calls are strictly sequential (one in flight), RPM-throttled.
    Offline Self-Test (`--dry-run`): a deterministic mock replaces the API and every
    pipeline stage (chunking, fold, quote verification, mapping, evaluation) runs
    for real, without spending a single request.
- Results on Dev Set (100 valid papers):
    * Dry-Run Self-Test  : pipeline mechanics verified (deterministic mock)
    * Real Run (hu llm3) : PENDING -
        python3 src/models/architectures/llm_sequential_chunks.py --provider hu
      (then update this block with the measured scores)
========================================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluator import (
    evaluate_predictions,
    load_jsonl_by_id,
    print_report,
    save_reproducible_run,
)
from src.models.architectures._llm_chunks_common import (
    CHUNK_SYSTEM_PROMPT,
    ENUMS,
    OpenAICompatClient,
    ParallelMockClient,
    SequentialMockClient,
    assemble_prediction,
    build_chunk_prompt,
    canon,
    extract_fulltext,
    extract_json_block,
    parse_chunk_items,
    verify_quote,
)

MODEL_ID = "llm_sequential_chunks"
MODEL_TITLE = "Architecture 2, Variant A: Sequential Chunk Pipeline (State Fold, HU llm3)"
CATEGORY = "architectures"
SEED = 42
MAX_ATTEMPTS = 2

# A state entry maps field -> list of {..label fields.., 'quote', 'section', 'verified'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential chunk pipeline (Approach A).")
    parser.add_argument("--provider", choices=["hu", "saia", "gemini"], default="hu")
    parser.add_argument("--model", default=None)
    parser.add_argument("--rpm", type=int, default=15)
    parser.add_argument("--max-section-chars", type=int, default=6500)
    parser.add_argument("--max-total-chars", type=int, default=100_000)
    parser.add_argument("--limit", type=int, default=None, help="Only first N papers")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def fold_paper(
    client: Any,
    sections: list[dict[str, str]],
    stats: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    """
    Runs the sequential fold over one paper's sections. Returns the final verified
    state (per-field item lists with quote/provenance). Labels are kept in the state
    verbatim (English); quote verification is accumulated per fold step.
    """
    # State entries: same shape as parse_chunk_items output + provenance fields.
    state: dict[str, list[dict[str, Any]]] = {
        "habitat": [], "biodiversity_level": [], "biodiversity_variable": [],
        "ecosystem_type": [], "taxa": [], "ecosystem_service": [],
    }
    state_keys: dict[str, set[str]] = {f: set() for f in state}

    def entry_key(field: str, it: dict[str, Any]) -> str:
        if field in ("habitat", "biodiversity_level"):
            return canon(it["label"])
        if field == "biodiversity_variable":
            return canon(it["value"])
        if field in ("ecosystem_type", "ecosystem_service"):
            return canon(f"{it['category']}|{it.get('term') or ''}")
        return canon(f"{it['species_group']}|{it.get('scientific_name') or ''}|{it.get('german_name') or ''}")

    for idx, section in enumerate(sections, 1):
        # The state block shows labels (with quotes) the model carries forward.
        state_block = json.dumps(
            {f: [it for it in items] for f, items in state.items()},
            ensure_ascii=False,
        ) if any(state.values()) else "(empty - no labels collected yet)"

        if isinstance(client, SequentialMockClient):
            client._current_chunk = section  # mock needs the chunk for its quote

        parsed = None
        for attempt in range(MAX_ATTEMPTS):
            raw = client.chat(CHUNK_SYSTEM_PROMPT, build_chunk_prompt(section, state_block))
            parsed = extract_json_block(raw)
            if parsed is not None:
                break
            stats["repair_retries"] += 1
        stats["n_calls"] += 1
        if parsed is None:
            stats["parse_failures"] += 1
            continue  # keep previous state; next chunk gets another chance

        new_items = parse_chunk_items(parsed)

        # Merge model output into state. An output entry either:
        #  - matches an existing state entry (by label key) -> carried/confirmed;
        #    keep it, update quote provenance if the quote verifies HERE
        #  - is new -> added; verified only if its quote matches THIS section
        for field, items in new_items.items():
            merged: list[dict[str, Any]] = []
            merged_keys = set()
            for it in items:
                key = entry_key(field, it)
                prev = next((e for e in state[field] if entry_key(field, e) == key), None)
                here = verify_quote(it["quote"], section["text"])
                if prev is not None:
                    e = {**it, "section": idx, "verified": prev["verified"] or here}
                    # carried entries may keep their earlier quote if not re-quoted
                    if not here and prev["verified"]:
                        e["quote"] = prev["quote"]
                    merged.append(e)
                    merged_keys.add(key)
                else:
                    merged.append({**it, "section": idx, "verified": here})
                    merged_keys.add(key)
                    if here:
                        stats["quotes_verified"] += 1
            # keep state entries the model did not repeat (unless clearly dropped):
            # the prompt says the model returns the UPDATED COMPLETE state, so an
            # entry missing from the output was deliberately removed.
            state[field] = merged
            state_keys[field] = merged_keys

    # Hallucination filter: drop anything never verified in ANY section.
    final: dict[str, list[dict[str, Any]]] = {}
    for field, items in state.items():
        kept = [it for it in items if it.get("verified")]
        stats["labels_dropped_unverified"] += len(items) - len(kept)
        final[field] = kept
    return final


def main() -> None:
    args = parse_args()
    start = time.time()

    gold = load_jsonl_by_id(REPO_ROOT / "data" / "gold" / "dev.jsonl")
    dev_ids = [
        line.strip()
        for line in (REPO_ROOT / "data" / "gold" / "dev_100_ids.txt").read_text().splitlines()
        if line.strip()
    ][: args.limit or None]

    if args.dry_run:
        client: Any = SequentialMockClient()
    else:
        from src.models.architectures._llm_chunks_common import PROVIDERS

        model = args.model or PROVIDERS[args.provider]["default_model"]
        client = OpenAICompatClient(args.provider, model, temperature=0.0, rpm=args.rpm)

    # Checkpoint name includes the provider: switching providers (e.g. hu -> saia)
    # must not silently reuse another provider's predictions as "already done".
    mode_tag = "dryrun" if args.dry_run else f"api_{args.provider}"
    ckpt_file = (
        REPO_ROOT / "outputs" / CATEGORY / MODEL_ID
        / f"{MODEL_ID}_checkpoint_{mode_tag}.jsonl"
    )
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, dict[str, Any]] = {}
    if ckpt_file.exists():
        for line in ckpt_file.read_text().splitlines():
            try:
                obj = json.loads(line)
                predictions[obj["zotitem_ptr_id"]] = obj
            except json.JSONDecodeError:
                continue

    stats = {"n_calls": 0, "repair_retries": 0, "parse_failures": 0,
             "quotes_verified": 0, "labels_dropped_unverified": 0}
    section_counts: list[int] = []

    for i, zot_id in enumerate(dev_ids, 1):
        if zot_id in predictions:
            continue
        rec = gold[zot_id]
        full_text = extract_fulltext(zot_id, rec)
        # Budget: header + sections until the char budget is exhausted.
        sections = []
        used = 0
        for sec in _split(full_text, args.max_section_chars):
            if used + len(sec["text"]) > args.max_total_chars:
                break
            sections.append(sec)
            used += len(sec["text"])
        if not sections:
            sections = [{"title": "Full text (short)", "text": full_text[:args.max_total_chars]}]
        section_counts.append(len(sections))

        final_state = fold_paper(client, sections, stats)
        pred = assemble_prediction(final_state)
        pred["zotitem_ptr_id"] = zot_id
        predictions[zot_id] = pred
        with open(ckpt_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
        print(
            f"[{i:>3}/{len(dev_ids)}] {zot_id} sections={len(sections)} "
            f"habitat={len(pred['habitat'])} taxa={len(pred['taxa'])} "
            f"ecosystem_type={len(pred['ecosystem_type'])}",
            flush=True,
        )

    metrics = evaluate_predictions(gold, predictions, filter_ids=set(dev_ids))
    print_report(MODEL_TITLE + (" [DRY-RUN]" if args.dry_run else ""), metrics)
    runtime = time.time() - start
    print(
        f"\nsections/paper avg={sum(section_counts) / max(len(section_counts), 1):.1f} | "
        f"calls={stats['n_calls']} | parse_failures={stats['parse_failures']} | "
        f"quotes_verified={stats['quotes_verified']} | "
        f"dropped_unverified={stats['labels_dropped_unverified']}"
    )
    if args.no_save:
        print("--no-save set: skipping run artifacts + leaderboard update.")
    else:
        save_reproducible_run(
            model_id=MODEL_ID,
            model_title=MODEL_TITLE + (" [DRY-RUN MOCK]" if args.dry_run else ""),
            category=CATEGORY,
            predictions=predictions,
            metrics=metrics,
            runtime_seconds=runtime,
            seed=SEED,
            config={
                "mode": "dry_run_mock" if args.dry_run else "api",
                "approach": "A_sequential_state_fold",
                "provider": args.provider,
                "model": getattr(client, "model", "sequential-mock"),
                "max_section_chars": args.max_section_chars,
                "max_total_chars": args.max_total_chars,
                "n_papers": len(dev_ids),
                "stats": stats,
                "avg_sections_per_paper": round(
                    sum(section_counts) / max(len(section_counts), 1), 1
                ),
            },
        )


def _split(full_text: str, max_section_chars: int) -> list[dict[str, str]]:
    """Local wrapper so the import graph stays explicit in the run log."""
    from src.models.architectures._llm_chunks_common import split_into_sections

    return split_into_sections(full_text, max_section_chars)


if __name__ == "__main__":
    main()
