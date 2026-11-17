"""Command-line entry points for extraction and local evaluation."""
import argparse
import hashlib
import json
import platform
import re
import sys
import time
import zipfile
from pathlib import Path
from .client import ChatClient
from .documents import pdf_to_tei, tei_sections, make_chunks
from .pipeline import run_paper
from .evaluate import read_records, evaluate, matching_gold
from .schema import load_schema, catalogue, DATA


def write_json(path, value):
    """Write readable UTF-8 JSON to disk."""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_ids(path):
    """Read an explicit split ID list and reject duplicates or unsafe file names."""
    ids = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", key) for key in ids):
        raise ValueError("IDs must be unique non-empty alphanumeric names with optional _ or -")
    return ids


def export_results(output, records):
    """Write final JSONL and a ZIP containing exactly predictions.jsonl."""
    target = output / "predictions.jsonl"
    temporary = target.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    temporary.replace(target)
    with zipfile.ZipFile(output / "predictions.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(target, "predictions.jsonl")


def run_command(args):
    """Run one split with fail-fast errors and publish submissions only on full success."""
    ids = read_ids(args.ids)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite prior experiments or leave stale submissions after a failed rerun.
    if any(output.iterdir()):
        raise ValueError("Output directory must be empty; use a new experiment directory")
    source = Path(args.input)
    suffix = ".pdf" if args.input_format == "pdf" else ".tei.xml"
    paths = [source / (key + suffix) for key in ids]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing split inputs: {missing}")
    started = time.monotonic()
    client = ChatClient(args.base_url, args.model, Path(args.cache) / "llm",
                        args.context_budget, args.max_output, not args.no_json_mode)
    config = vars(args).copy()
    config.update({"python": sys.version, "platform": platform.platform(),
                   "schema_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in DATA.glob("*.json")}})
    write_json(output / "run_config.json", config)
    records, usage, source_hashes = [], [], {}
    try:
        for paper_id, path in zip(ids, paths):
            source_hashes[paper_id] = hashlib.sha256(path.read_bytes()).hexdigest()
            xml = pdf_to_tei(path, args.grobid_url, Path(args.cache) / "tei") if args.input_format == "pdf" else path.read_text()
            chunks = make_chunks(tei_sections(xml), args.chunk_chars, args.overlap)
            record, audit = run_paper(paper_id, chunks, client, args.mode, args.workers,
                                      args.quote_threshold, args.min_votes)
            write_json(output / (paper_id + ".audit.json"), audit)
            records.append(record)
            usage.extend(call["metadata"] for call in audit["calls"])
            print(f"[{len(records)}/{len(ids)}] {paper_id}: {len(chunks)} chunks", flush=True)
    except Exception as exc:
        write_json(output / "failure.json", {"paper_id": paper_id, "error": str(exc),
                                            "completed_ids": [r["zotitem_ptr_id"] for r in records]})
        raise
    export_results(output, records)
    fresh_usage = [u for u in usage if not u["cache_hit"]]
    write_json(output / "run_metrics.json", {
        "wall_seconds": time.monotonic() - started, "papers": len(records),
        "calls": len(usage), "cache_hits": len(usage) - len(fresh_usage),
        "fresh_prompt_tokens": sum(u.get("usage", {}).get("prompt_tokens", 0) for u in fresh_usage),
        "fresh_completion_tokens": sum(u.get("usage", {}).get("completion_tokens", 0) for u in fresh_usage),
        "token_usage_available_for_all_fresh_calls": all("prompt_tokens" in u.get("usage", {}) for u in fresh_usage),
        "source_sha256": source_hashes})


def build_parser():
    """Define reproducible CLI options for extraction and evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--input", required=True)
    run.add_argument("--ids", required=True, help="One paper ID per line; no file extension")
    run.add_argument("--input-format", choices=["pdf", "tei"], default="pdf")
    run.add_argument("--output", required=True)
    run.add_argument("--mode", choices=["sequential", "parallel"], required=True)
    run.add_argument("--model", required=True, help="Exact model name served by your endpoint")
    run.add_argument("--base-url", default="http://localhost:11434/v1")
    run.add_argument("--grobid-url", default="http://localhost:8070")
    run.add_argument("--cache", default=".cache/taskc")
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--chunk-chars", type=int, default=5000)
    run.add_argument("--overlap", type=int, default=300)
    run.add_argument("--quote-threshold", type=float, default=1.0)
    run.add_argument("--min-votes", type=int, default=1)
    run.add_argument("--context-budget", type=int, default=65000)
    run.add_argument("--max-output", type=int, default=4096)
    run.add_argument("--no-json-mode", action="store_true")
    ev = commands.add_parser("evaluate")
    ev.add_argument("--gold", required=True)
    ev.add_argument("--predictions", required=True)
    ev.add_argument("--bert-model", default=None)
    ev.add_argument("--output", required=True)
    commands.add_parser("check-schemas")
    return parser


def main():
    """Dispatch a subcommand and return a nonzero status on failure."""
    args = build_parser().parse_args()
    if args.command == "run":
        if args.max_output < 1 or args.context_budget <= args.max_output:
            raise ValueError("Invalid context/output budget")
        run_command(args)
    elif args.command == "evaluate":
        predicted = read_records(args.predictions)
        gold = matching_gold(read_records(args.gold), predicted)
        print(len(gold), flush=True)
        write_json(args.output, evaluate(gold, predicted, args.bert_model))
    else:
        print(json.dumps({key: len(values) for key, values in catalogue(load_schema()).items()}, indent=2))
