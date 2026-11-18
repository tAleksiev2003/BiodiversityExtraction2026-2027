"""Read the model's JSON answer and retry once if it is not valid (SPEC.md, section 6).

The answer is parsed and validated with pydantic. Only the shape is checked here (an
object with a list of labels, each with a "field" and string values). Whether a value
is allowed and whether its quote is in the text is the job of the verbatim checker.

Invalid answer: the same prompt is sent once more with the model's answer and the parse
error appended (config/prompts/retry_invalid_json.txt). If that fails too, the call is
logged and treated as empty. A dry-run answer is always empty and never retried: there
is nothing to parse yet.
"""
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from llm_cache import Message
from llm_client import SOURCE_API, SOURCE_DRY_RUN, LLMClient

logger = logging.getLogger(__name__)

# Some models wrap JSON in a Markdown code fence despite rule R8; the fence is not part of
# the answer, so it is removed before parsing.
CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)

STATUS_OK = "ok"
STATUS_OK_AFTER_RETRY = "ok_after_retry"
STATUS_INVALID = "invalid_json"   # also invalid after the retry: treated as empty
STATUS_DRY_RUN = "dry_run"        # no answer yet: treated as empty


class ExtractedLabel(BaseModel):
    """One label as the model wrote it; which attributes are used depends on `field`."""

    model_config = ConfigDict(extra="ignore")  # e.g. "details" or notes the model adds

    field: str
    value: str | None = None             # habitat, biodiversity_level, biodiversity_variable
    category: str | None = None          # ecosystem_type, ecosystem_service
    term: str | None = None              # ecosystem_type, ecosystem_service
    species_group: str | None = None     # taxa
    scientific_name: str | None = None   # taxa
    german_name: str | None = None       # taxa
    name_source: str | None = None       # taxa: "text" or "completed" (rule R7)
    quote: str | None = None


class ExtractionAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reasoning: str = ""
    labels: list[ExtractedLabel] = []


EMPTY_ANSWER = ExtractionAnswer()


class AnswerParseError(ValueError):
    """The answer is not a JSON object of the required shape."""


def parse_answer(content: str) -> ExtractionAnswer:
    """Parse and validate one answer; raises AnswerParseError with a short reason."""
    fenced = CODE_FENCE.match(content)
    text = fenced.group(1) if fenced else content
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as error:
        raise AnswerParseError(f"invalid JSON: {error}") from error
    try:
        return ExtractionAnswer.model_validate(data)
    except ValidationError as error:
        # One line per problem, without the (possibly long) input values.
        problems = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc']) or 'answer'}: {problem['msg']}"
            for problem in error.errors(include_input=False)
        )
        raise AnswerParseError(f"wrong structure: {problems}") from error


@dataclass(frozen=True)
class AnswerOutcome:
    """The parsed answer of one call and how it was obtained."""

    answer: ExtractionAnswer
    status: str          # STATUS_* above
    requests_sent: int   # 0 (cache, dry run), 1 or 2 (with retry): for the run summary
    error: str | None = None


def read_retry_template(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def request_answer(
    client: LLMClient, model: str, messages: list[Message], retry_template: str
) -> AnswerOutcome:
    """Ask the model; on an invalid answer ask once more with the error appended."""
    result = client.complete(model, messages)
    requests_sent = int(result.source == SOURCE_API)
    if result.source == SOURCE_DRY_RUN:
        return AnswerOutcome(EMPTY_ANSWER, STATUS_DRY_RUN, requests_sent)
    try:
        return AnswerOutcome(parse_answer(result.content), STATUS_OK, requests_sent)
    except AnswerParseError as first_error:
        retry_messages: list[Message] = [
            *messages,
            {"role": "assistant", "content": result.content},
            {"role": "user", "content": retry_template.format(error=first_error)},
        ]
    retry_result = client.complete(model, retry_messages)
    requests_sent += int(retry_result.source == SOURCE_API)
    if retry_result.source == SOURCE_DRY_RUN:
        return AnswerOutcome(EMPTY_ANSWER, STATUS_DRY_RUN, requests_sent)
    try:
        return AnswerOutcome(parse_answer(retry_result.content), STATUS_OK_AFTER_RETRY, requests_sent)
    except AnswerParseError as second_error:
        logger.warning("Answer still invalid after one retry, treated as empty: %s", second_error)
        return AnswerOutcome(EMPTY_ANSWER, STATUS_INVALID, requests_sent, str(second_error))
