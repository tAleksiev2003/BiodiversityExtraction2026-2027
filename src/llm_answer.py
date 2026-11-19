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
from typing import Any, Callable, Generic, TypeVar

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


Answer = TypeVar("Answer", bound=BaseModel)


def parse_json_model(content: str, model_class: type[Answer]) -> Answer:
    """Parse one answer into `model_class`; raises AnswerParseError with a short reason.

    Shared by the extraction answer and the verification answer (src/verification.py),
    so both strip code fences and report problems the same way.
    """
    fenced = CODE_FENCE.match(content)
    text = fenced.group(1) if fenced else content
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as error:
        raise AnswerParseError(f"invalid JSON: {error}") from error
    try:
        return model_class.model_validate(data)
    except ValidationError as error:
        # One line per problem, without the (possibly long) input values.
        problems = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc']) or 'answer'}: {problem['msg']}"
            for problem in error.errors(include_input=False)
        )
        raise AnswerParseError(f"wrong structure: {problems}") from error


def parse_answer(content: str) -> ExtractionAnswer:
    """Parse and validate one extraction answer."""
    return parse_json_model(content, ExtractionAnswer)


@dataclass(frozen=True)
class AnswerOutcome(Generic[Answer]):
    """The parsed answer of one call and how it was obtained."""

    answer: Answer
    status: str          # STATUS_* above
    requests_sent: int   # 0 (cache, dry run), 1 or 2 (with retry): for the run summary
    error: str | None = None


def read_retry_template(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def request_parsed(
    client: LLMClient,
    model: str,
    messages: list[Message],
    retry_template: str,
    parse: Callable[[str], Answer],
    empty: Answer,
) -> AnswerOutcome[Answer]:
    """Ask the model; on an invalid answer ask once more with the error appended.

    `parse` raises AnswerParseError for an invalid answer; `empty` stands in for an
    answer that is still invalid after the retry, or for a dry run.
    """
    result = client.complete(model, messages)
    requests_sent = int(result.source == SOURCE_API)
    if result.source == SOURCE_DRY_RUN:
        return AnswerOutcome(empty, STATUS_DRY_RUN, requests_sent)
    try:
        return AnswerOutcome(parse(result.content), STATUS_OK, requests_sent)
    except AnswerParseError as first_error:
        retry_messages: list[Message] = [
            *messages,
            {"role": "assistant", "content": result.content},
            {"role": "user", "content": retry_template.format(error=first_error)},
        ]
    retry_result = client.complete(model, retry_messages)
    requests_sent += int(retry_result.source == SOURCE_API)
    if retry_result.source == SOURCE_DRY_RUN:
        return AnswerOutcome(empty, STATUS_DRY_RUN, requests_sent)
    try:
        return AnswerOutcome(parse(retry_result.content), STATUS_OK_AFTER_RETRY, requests_sent)
    except AnswerParseError as second_error:
        logger.warning("Answer still invalid after one retry, treated as empty: %s", second_error)
        return AnswerOutcome(empty, STATUS_INVALID, requests_sent, str(second_error))


def request_answer(
    client: LLMClient, model: str, messages: list[Message], retry_template: str
) -> AnswerOutcome[ExtractionAnswer]:
    """An extraction answer (pipelines F and B), with the one retry of request_parsed."""
    return request_parsed(client, model, messages, retry_template, parse_answer, EMPTY_ANSWER)
