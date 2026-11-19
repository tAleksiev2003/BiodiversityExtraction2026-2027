"""Run an LLM pipeline on one split and write a run folder (SPEC.md, sections 3 and 8).

Usage (run by the team: it reads course data and, without --dry-run, sends requests):
    python3 src/run_pipeline.py full_text --split dev --dry-run     # count the requests
    python3 src/run_pipeline.py full_text --split dev --limit 3     # 3 documents
    python3 src/run_pipeline.py full_text --split dev
    python3 src/run_pipeline.py full_text --split dev --experiment config/experiments/no_verbatim_check.yaml
    python3 src/run_pipeline.py parallel --split dev --experiment config/experiments/prompt_v2.yaml --dry-run   # pipeline B

Needs the converted documents (src/pdf_to_text.py); prompt v4 (the default) also needs
data/processed/neighbours_<split>.json (src/first_pages.py, src/retrieval.py). Writes runs/<timestamp>_<name>/ with
config.yaml, commit.txt, llm_calls.jsonl (prompts and raw answers), documents.jsonl,
verbatim_check_log.jsonl, summary.json and, unless stopped or a dry run,
predictions.jsonl + predictions.zip. A stopped run (daily limit, Ctrl + C) continues
from the cache when the same command is run again.
"""
import argparse
import logging
import sys
from pathlib import Path

from chunking import load_tokenizer
from few_shot_examples import FewShotExamples
from llm_client import LLMClient
from pdf_to_text import converted_dir, document_ids_for_split
from pipelines.full_text import CALL_LOG_NAME, run_full_text
from pipelines.parallel import run_parallel, unsupported_settings
from prompts import ExtractionPrompt
from runs import create_run_dir
from settings import load_settings

PIPELINES = ("full_text", "parallel")
SPLITS = ("dev", "test")


def run_name(pipeline: str, split: str, experiment: Path | None, limit: int | None, dry_run: bool) -> str:
    """Folder name that tells runs apart, e.g. full_text_dev_no_verbatim_check_dry_run."""
    name = f"{pipeline}_{split}"
    if experiment is not None:
        name += f"_{experiment.stem}"
    if limit is not None:
        name += f"_first{limit}"
    if dry_run:
        name += "_dry_run"
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an LLM pipeline on one split.")
    parser.add_argument("pipeline", choices=PIPELINES)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--limit", type=int, default=None, help="only the first N documents")
    parser.add_argument("--dry-run", action="store_true",
                        help="build all prompts and count the requests, send nothing")
    parser.add_argument("--experiment", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per HTTP call is too much
    settings = load_settings(args.experiment)
    if args.pipeline == "parallel":
        reason = unsupported_settings(settings)
        if reason is not None:
            sys.exit(reason)  # before a run folder is created
    uses_examples = ExtractionPrompt.from_settings(settings).uses_examples
    examples = FewShotExamples.from_settings(settings, args.split) if uses_examples else None
    document_ids = document_ids_for_split(settings, args.split)
    if args.limit is not None:
        document_ids = document_ids[:args.limit]

    run_dir = create_run_dir(settings, run_name(args.pipeline, args.split, args.experiment,
                                                args.limit, args.dry_run))
    print(f"Run folder: {run_dir}")
    client = LLMClient(settings, dry_run=args.dry_run, call_log_path=run_dir / CALL_LOG_NAME)
    document_dir = converted_dir(settings, args.split)
    if args.pipeline == "full_text":
        tokenizer = load_tokenizer(settings, max_tokens=settings["models"]["full_text_part_tokens"])
        summary = run_full_text(settings, document_ids, client, run_dir, document_dir, tokenizer, examples)
    else:
        tokenizer = load_tokenizer(settings)  # cuts at chunking.max_chunk_tokens, then packing
        summary = run_parallel(settings, document_ids, client, run_dir, document_dir, tokenizer)
    if summary["stopped"] is not None:
        sys.exit(f"STOPPED: {summary['stopped']}\nNo predictions written. Run the same "
                 "command again to continue; answered prompts come from the cache.")


if __name__ == "__main__":
    main()
