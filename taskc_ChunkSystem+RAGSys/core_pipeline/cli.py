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
from .documents import pdf_to_tei, tei_sections, tei_pages, make_chunks, make_page_chunks
from .pipeline import run_paper
from .rag import run_grouped
from .retrieval import EmbeddingClient, QUERY_PREFIX, GROUPS
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
    client = ChatClient(args.base_url, args.model, Path(args.cache) / ("llm_grouped_strict" if args.mode in {"rag", "grouped-all", "rag-all-fields"} else "llm"),
                        args.context_budget, args.max_output, not args.no_json_mode,
                        strict_length=args.mode in {"rag", "grouped-all", "rag-all-fields"})
    embedder = None
    if args.mode in {"rag", "rag-all-fields"}:
        embedder = EmbeddingClient(args.embedding_base_url or args.base_url, args.embedding_model,
            Path(args.cache) / "embeddings", args.query_prefix, args.document_prefix,
            args.embedding_max_bytes, args.embedding_batch_size)
    config = vars(args).copy()
    config.update({"field_groups": GROUPS,
                   "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in Path(__file__).parent.glob("*.py")},
                   "python": sys.version, "platform": platform.platform(),
                   "schema_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in DATA.glob("*.json")}})
    write_json(output / "run_config.json", config)
    records, usage, source_hashes = [], [], {}
    embedding_usage = []
    try:
        for paper_id, path in zip(ids, paths):
            source_hashes[paper_id] = hashlib.sha256(path.read_bytes()).hexdigest()
            if args.input_format == "pdf":
                xml = pdf_to_tei(path, args.grobid_url, Path(args.cache) / "tei",
                                 coordinates=args.chunk_by == "pages" or args.mode in {"rag", "grouped-all", "rag-all-fields"})
            else:
                xml = path.read_text()
            if args.mode in {"rag", "grouped-all", "rag-all-fields"}:
                pages = tei_pages(xml)
                print(f"[{len(records) + 1}/{len(ids)}] {paper_id}: {len(pages)} pages ({args.mode})", flush=True)
                record, audit = run_grouped(paper_id, pages, client, embedder, args.mode,
                                           args.top_k, args.quote_threshold, args.min_votes)
                embedding_usage.extend(audit["embedding_calls"])
            else:
                if args.chunk_by == "pages":
                    chunks = make_page_chunks(tei_pages(xml), args.pages_per_chunk)
                else:
                    chunks = make_chunks(tei_sections(xml), args.chunk_chars, args.overlap)
                print(f"[{len(records) + 1}/{len(ids)}] {paper_id}: {len(chunks)} chunks", flush=True)
                record, audit = run_paper(paper_id, chunks, client, args.mode, args.workers,
                                          args.quote_threshold, args.min_votes)
            write_json(output / (paper_id + ".audit.json"), audit)
            records.append(record)
            usage.extend(call["metadata"] for call in audit["calls"])
    except Exception as exc:
        write_json(output / "failure.json", {"paper_id": paper_id, "error": str(exc),
                                            "completed_ids": [r["zotitem_ptr_id"] for r in records]})
        raise
    export_results(output, records)
    fresh_usage = [u for u in usage if not u["cache_hit"] and not u.get("truncated")]
    write_json(output / "run_metrics.json", {
        "wall_seconds": time.monotonic() - started, "papers": len(records),
        "calls": len(usage), "cache_hits": sum(1 for u in usage if u["cache_hit"]),
        "truncated_chunks": sum(1 for u in usage if u.get("truncated")),
        "fresh_prompt_tokens": sum(u.get("usage", {}).get("prompt_tokens", 0) for u in fresh_usage),
        "fresh_completion_tokens": sum(u.get("usage", {}).get("completion_tokens", 0) for u in fresh_usage),
        "token_usage_available_for_all_fresh_calls": all("prompt_tokens" in u.get("usage", {}) for u in fresh_usage),
        "logical_prompt_tokens": sum(u.get("usage", {}).get("prompt_tokens", 0) for u in usage),
        "logical_completion_tokens": sum(u.get("usage", {}).get("completion_tokens", 0) for u in usage),
        "embedding_http_calls": sum(not u["cache_hit"] for u in embedding_usage),
        "embedding_cache_hits": sum(u["items"] for u in embedding_usage if u["cache_hit"]),
        "embedding_fresh_tokens": sum(u.get("usage", {}).get("total_tokens", 0)
                                      for u in embedding_usage if not u["cache_hit"]),
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
    run.add_argument("--mode", choices=["sequential", "parallel", "rag", "grouped-all", "rag-all-fields"], required=True)
    run.add_argument("--model", required=True, help="Exact model name served by your endpoint")
    run.add_argument("--base-url", default="http://localhost:11434/v1")
    run.add_argument("--grobid-url", default="http://localhost:8070")
    run.add_argument("--cache", default=".cache/taskc")
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--chunk-by", choices=["chars", "pages"], default="chars",
                     help="chars keeps section text split by characters; pages groups physical PDF pages")
    run.add_argument("--chunk-chars", type=int, default=5000,
                     help="Maximum characters per chunk when --chunk-by is chars")
    run.add_argument("--overlap", type=int, default=300,
                     help="Overlapping characters when --chunk-by is chars")
    run.add_argument("--pages-per-chunk", type=int, default=10,
                     help="Physical pages per chunk when --chunk-by is pages")
    run.add_argument("--quote-threshold", type=float, default=1.0)
    run.add_argument("--min-votes", type=int, default=2)
    run.add_argument("--context-budget", type=int, default=65000)
    run.add_argument("--max-output", type=int, default=4096)
    run.add_argument("--no-json-mode", action="store_true")
    run.add_argument("--embedding-model", default="qwen3-embedding-4b")
    run.add_argument("--embedding-base-url", default=None, help="Defaults to the chat base URL")
    run.add_argument("--embedding-max-bytes", type=int, default=6000)
    run.add_argument("--embedding-batch-size", type=int, default=16)
    run.add_argument("--query-prefix", default=QUERY_PREFIX)
    run.add_argument("--document-prefix", default="")
    run.add_argument("--top-k", type=int, default=4)
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
        if args.chunk_by == "pages" and args.pages_per_chunk < 1:
            raise ValueError("pages_per_chunk must be at least 1")
        if args.top_k < 1 or args.min_votes < 1 or args.workers < 1 or not 0.9 <= args.quote_threshold <= 1:
            raise ValueError("Invalid retrieval/extraction settings")
        if args.mode in {"rag", "grouped-all", "rag-all-fields"}:
            args.chunk_by = "pages"
        run_command(args)
    elif args.command == "evaluate":
        predicted = read_records(args.predictions)
        gold = matching_gold(read_records(args.gold), predicted)
        print(len(gold), flush=True)
        write_json(args.output, evaluate(gold, predicted, args.bert_model))
    else:
        print(json.dumps({key: len(values) for key, values in catalogue(load_schema()).items()}, indent=2))
