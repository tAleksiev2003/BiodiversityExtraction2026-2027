"""Verify the labels of an existing pipeline F run (src/verification.py, SPEC.md, section 13).

One more call per paper (per part for papers F split into parts) that only keeps or
drops the labels the source run kept; habitat is never checked. The extraction is not
repeated: the source run's answers come from the cache.

Usage (run by the team: it reads course data and, without --dry-run, sends requests):
    python3 src/run_verify.py --split dev --source-run runs/<F run> --dry-run   # count the requests
    python3 src/run_verify.py --split dev --source-run runs/<F run>
    python3 src/run_verify.py --split dev --source-run runs/<F run> --experiment config/experiments/verify_without_taxa.yaml

Writes runs/<timestamp>_verify_<split>[_<experiment>]/ with config.yaml, commit.txt,
sources.json, llm_calls.jsonl (verify prompts and raw answers), verify_log.jsonl (one
line per candidate), summary.json and, unless stopped or a dry run, predictions.jsonl +
predictions.zip. A stopped run continues from the cache when run again.
"""
import argparse
import logging
import sys
from pathlib import Path

from chunking import load_tokenizer
from few_shot_examples import FewShotExamples
from llm_client import LLMClient
from pdf_to_text import converted_dir
from pipelines.full_text import CALL_LOG_NAME
from pipelines.verify import SourceRun, run_verify
from prompts import ExtractionPrompt
from run_pipeline import SPLITS, run_name
from runs import create_run_dir
from settings import load_settings
from verification import VerifySettings

PIPELINE_NAME = "verify"


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep or drop the labels of an existing F run.")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--source-run", type=Path, required=True, help="run folder of a pipeline F run")
    parser.add_argument("--dry-run", action="store_true",
                        help="build all prompts and count the requests, send nothing")
    parser.add_argument("--experiment", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = load_settings(args.experiment)
    VerifySettings.from_settings(settings)  # fails before a run folder is created
    source = SourceRun.load(args.source_run)
    # The verify prompt always shows the training examples; the source prompt only if it is v4.
    examples = FewShotExamples.from_settings(settings, args.split)
    source_uses_examples = ExtractionPrompt.from_settings(source.settings).uses_examples
    source_examples = FewShotExamples.from_settings(source.settings, args.split) if source_uses_examples else None

    run_dir = create_run_dir(settings, run_name(PIPELINE_NAME, args.split, args.experiment, None, args.dry_run))
    print(f"Run folder: {run_dir}")
    client = LLMClient(settings, dry_run=args.dry_run, call_log_path=run_dir / CALL_LOG_NAME)
    # The source run's part size, so every paper is split exactly as F split it.
    tokenizer = load_tokenizer(source.settings, max_tokens=source.settings["models"]["full_text_part_tokens"])
    summary = run_verify(settings, source, client, run_dir, converted_dir(settings, args.split),
                         tokenizer, examples, source_examples)
    if summary["stopped"] is not None:
        sys.exit(f"STOPPED: {summary['stopped']}\nNo predictions written. Run the same "
                 "command again to continue; answered prompts come from the cache.")


if __name__ == "__main__":
    main()
