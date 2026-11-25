#!/usr/bin/env python3
"""
========================================================================================
EXECUTIVE SUMMARY — Architecture Model 3: Parallel Chunk Pipeline (Approach B)
========================================================================================
- Concept & Goal:
    Third architecture-level model, implementing variant B of the team's agreed
    chunk pipeline (Google-Doc diagram, 11.11.2026): every section chunk is processed
    by an INDEPENDENT LLM call (no state passing, so no call can infect the others
    with early errors), and the per-chunk label sets are merged in Python by voting:
    a label counts if it has at least one fuzzy-verified verbatim quote. An empty
    section is NOT a vote against a label — information is often simply not in that
    section. Research question: does independence beat carrying state (Model 2)?
- How It Works:
    1. Chunking: identical to Model 2 (heading-heuristic sections over the cached
       pypdf full text; references dropped, section size capped).
    2. Independent calls: each chunk gets the same English-label prompt WITHOUT any
       state; every label must carry a verbatim quote from that chunk. Quotes are
       verified immediately against the chunk text; unverified labels are dropped
       before merging (hallucination filter at the source).
    3. Vote merge (pure Python, no manager LLM call): labels are grouped by
       canonical key per field; vote count = number of chunks that produced a
       verified copy. All labels with >= 1 vote survive; for object fields the
       most-voted category/term combination wins; taxa and biodiversity_variable
       entries are deduplicated canonically and capped.
    4. Final step: deterministic EN->DE mapping onto the official German enums,
       collective-term conventions, evaluation, `save_reproducible_run()` artifacts.
    Provider: HU-Berlin bridge (llm3, Qwen3.6-27B thinking, sampling pinned
    server-side). "Parallel" is logical independence — requests are still issued
    sequentially through the RPM-throttled bridge (HU allows max 1 in flight).
    Offline Self-Test (`--dry-run`): deterministic mock, full pipeline otherwise.
- Results on Dev Set (100 valid papers):
    * Dry-Run Self-Test  : pipeline mechanics verified (deterministic mock)
    * Real Run (hu llm3) : PENDING -
        python3 src/models/architectures/llm_parallel_chunks.py --provider hu
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
    PROVIDERS,
    OpenAICompatClient,
    ParallelMockClient,
    assemble_prediction,
    build_chunk_prompt,
    canon,
    extract_fulltext,
    extract_json_block,
    parse_chunk_items,
    split_into_sections,
    verify_quote,
)

MODEL_ID = "llm_parallel_chunks"
MODEL_TITLE = "Architecture 3: Parallel Chunk Pipeline (Vote Merge, HU llm3)"
CATEGORY = "architectures"
SEED = 42
MAX_ATTEMPTS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel chunk pipeline (Approach B).")
    parser.add_argument("--provider", choices=list(PROVIDERS), default="hu")
    parser.add_argument("--model", default=None)
    parser.add_argument("--rpm", type=int, default=15)
    parser.add_argument("--max-section-chars", type=int, default=6500)
    parser.add_argument("--max-total-chars", type=int, default=100_000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def process_chunk(
    client: Any, section: dict[str, str], stats: dict[str, int]
) -> dict[str, list[dict[str, Any]]]:
    """One independent chunk call; returns quote-verified items only."""
    if isinstance(client, ParallelMockClient):
        client._current_chunk = section  # mock builds its quote from the chunk

    parsed = None
    for attempt in range(MAX_ATTEMPTS):
        raw = client.chat(CHUNK_SYSTEM_PROMPT, build_chunk_prompt(section))
        parsed = extract_json_block(raw)
        if parsed is not None:
            break
        stats["repair_retries"] += 1
    stats["n_calls"] += 1
    if parsed is None:
        stats["parse_failures"] += 1
        return {f: [] for f in (
            "habitat", "biodiversity_level", "biodiversity_variable",
            "ecosystem_type", "taxa", "ecosystem_service")}

    items = parse_chunk_items(parsed)
    verified: dict[str, list[dict[str, Any]]] = {}
    for field, entries in items.items():
        kept = []
        for it in entries:
            if verify_quote(it["quote"], section["text"]):
                stats["quotes_verified"] += 1
                kept.append(it)
            else:
                stats["labels_dropped_unverified"] += 1
        verified[field] = kept
    return verified


def _merge_key(field: str, it: dict[str, Any]) -> str:
    """Canonical key under which per-chunk votes are summed up."""
    if field in ("habitat", "biodiversity_level"):
        return canon(it["label"])
    if field == "biodiversity_variable":
        return canon(it["value"])
    if field in ("ecosystem_type", "ecosystem_service"):
        return canon(f"{it['category']}|{it.get('term') or ''}")
    return canon(f"{it['species_group']}|{it.get('scientific_name') or ''}|{it.get('german_name') or ''}")


def merge_chunk_results(
    chunk_results: list[dict[str, list[dict[str, Any]]]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Vote merge: group verified items per field by canonical key; votes = number of
    chunks that produced the key; every key with >= 1 vote survives (an empty chunk
    is not a vote against). The strongest variant of each key is kept.
    """
    merged: dict[str, list[dict[str, Any]]] = {}
    for field in ("habitat", "biodiversity_level", "biodiversity_variable",
                  "ecosystem_type", "taxa", "ecosystem_service"):
        by_key: dict[str, list[dict[str, Any]]] = {}
        for result in chunk_results:
            for it in result.get(field, []):
                by_key.setdefault(_merge_key(field, it), []).append(it)
        # For each key keep one representative (first seen; entries within a key
        # only differ in quote) and record its vote count for downstream ordering.
        items = []
        for key, variants in by_key.items():
            rep = dict(variants[0])
            rep["votes"] = len(variants)
            items.append(rep)
        items.sort(key=lambda it: -it.get("votes", 1))
        merged[field] = items
    return merged


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
        client: Any = ParallelMockClient()
    else:
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
        sections, used = [], 0
        for sec in split_into_sections(full_text, args.max_section_chars):
            if used + len(sec["text"]) > args.max_total_chars:
                break
            sections.append(sec)
            used += len(sec["text"])
        if not sections:
            sections = [{"title": "Full text (short)", "text": full_text[:args.max_total_chars]}]
        section_counts.append(len(sections))

        chunk_results = [process_chunk(client, sec, stats) for sec in sections]
        merged = merge_chunk_results(chunk_results)
        pred = assemble_prediction(merged)
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
                "approach": "B_parallel_vote_merge",
                "provider": args.provider,
                "model": getattr(client, "model", "parallel-mock"),
                "max_section_chars": args.max_section_chars,
                "max_total_chars": args.max_total_chars,
                "n_papers": len(dev_ids),
                "stats": stats,
                "avg_sections_per_paper": round(
                    sum(section_counts) / max(len(section_counts), 1), 1
                ),
            },
        )


if __name__ == "__main__":
    main()
