"""LLM client for the OpenAI-compatible SAIA API (SPEC.md, section 5).

Every LLM call of the pipelines goes through `LLMClient.complete`. SAIA allows only
10 requests per 10 seconds, 400 per hour and 800 per day, so each call passes these
steps in order:
1. disk cache: an answered prompt is never requested again (src/llm_cache.py);
2. dry run: count the requests a real run would send, without sending them;
3. rate limiter: wait until all windows have room; stop at the daily limit
   (src/rate_limiter.py);
4. retries with exponential backoff on 429, 5xx, timeouts and connection errors;
5. logging of the number of requests, the limits the server reports
   (x-ratelimit-limit-*) and the remaining quota (x-ratelimit-remaining-*).

The API key is read from .env (variable `llm.api_key_env`) only when the first real
request is sent, so a dry run or a fully cached run works without a key. The key is
never printed, logged or written anywhere.

Usage (run by the team; the commands that send requests need the key in .env):
    python3 src/llm_client.py ask --prompt-file tests/fixtures/llm_smoke_prompt.txt
    python3 src/llm_client.py ask --prompt-file tests/fixtures/llm_smoke_prompt.txt --dry-run
    python3 src/llm_client.py list-models
"""
import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import openai
from dotenv import load_dotenv

from llm_cache import Message, ResponseCache, cache_key
from rate_limiter import DailyLimitReachedError, RateLimiter
from settings import REPO_ROOT, load_settings

logger = logging.getLogger(__name__)

ENV_FILE = REPO_ROOT / ".env"

# Where an answer came from: only SOURCE_API costs a request.
SOURCE_API = "api"
SOURCE_CACHE = "cache"
SOURCE_DRY_RUN = "dry_run"

QUOTA_HEADER_PREFIX = "x-ratelimit-remaining"
LIMIT_HEADER_PREFIX = "x-ratelimit-limit"
# Header names seen in a real SAIA response (M2); the server's counts win over local counting.
REMAINING_HOUR_HEADER = "x-ratelimit-remaining-requests-hour"
REMAINING_DAY_HEADER = "x-ratelimit-remaining-requests-day"
RETRY_AFTER_HEADER = "retry-after"
TOO_MANY_REQUESTS = 429
FIRST_SERVER_ERROR = 500
BACKOFF_FACTOR = 2  # each retry waits twice as long as the one before

MODEL_ROLES = ("chunked", "full_text", "full_text_fallback")  # keys under `models` in settings.yaml


class MissingApiKeyError(RuntimeError):
    """The key variable is neither in the environment nor in .env."""


def load_api_key(env_name: str, env_file: Path = ENV_FILE) -> str:
    """Return the key from the environment, after python-dotenv has loaded `env_file`.

    An environment variable that is already set wins over .env (python-dotenv default).
    The error message names only the variable, never a value.
    """
    load_dotenv(env_file)
    api_key = os.environ.get(env_name, "").strip()
    if not api_key:
        raise MissingApiKeyError(
            f"{env_name} is not set. Copy .env.example to .env and enter your SAIA key."
        )
    return api_key


def create_openai_client(settings: dict[str, Any]) -> openai.OpenAI:
    """OpenAI SDK client pointed at SAIA.

    The SDK's own retries are switched off (max_retries=0): LLMClient retries itself, so
    every attempt is rate limited, counted and logged.
    """
    llm = settings["llm"]
    return openai.OpenAI(
        api_key=load_api_key(llm["api_key_env"]),
        base_url=llm["base_url"],
        timeout=llm["timeout_seconds"],
        max_retries=0,
    )


@dataclass(frozen=True)
class ChatResult:
    """One answer of the model.

    In a dry run `content` is empty and `source` is SOURCE_DRY_RUN; pipelines treat
    it like an empty answer.
    """

    content: str
    source: str
    cache_key: str


@dataclass
class RequestStats:
    """Counters for one client; the pipelines write them into the run folder."""

    api_requests: int = 0       # every request sent to the server, retries included
    retries: int = 0
    cache_hits: int = 0
    dry_run_requests: int = 0   # different prompts a real run would still send
    remaining_quota: dict[str, str] = field(default_factory=dict)  # last x-ratelimit-remaining-* headers
    # Last x-ratelimit-limit-* headers: the limits the server really applies, so we can
    # check the values in settings.yaml instead of inferring them.
    rate_limits: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def headers_with_prefix(headers: Mapping[str, str], prefix: str) -> dict[str, str]:
    """All headers whose lower-case name starts with `prefix`, names in lower case."""
    return {
        name.lower(): value
        for name, value in headers.items()
        if name.lower().startswith(prefix)
    }


def is_retryable(error: Exception) -> bool:
    """Retry on rate limits, server errors, timeouts and lost connections.

    Other errors (wrong model name, invalid key, prompt too long) fail the same way on
    every attempt, so retrying them would only waste quota.
    """
    if isinstance(error, openai.APIConnectionError):  # includes openai.APITimeoutError
        return True
    if isinstance(error, openai.APIStatusError):
        return error.status_code == TOO_MANY_REQUESTS or error.status_code >= FIRST_SERVER_ERROR
    return False


def parse_retry_after(headers: Mapping[str, str]) -> float | None:
    """Seconds from a Retry-After header; None if missing or given as a date."""
    value = headers.get(RETRY_AFTER_HEADER)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_remaining(headers: Mapping[str, str], name: str) -> int | None:
    """Integer value of a remaining-requests header; None if missing or not a number."""
    value = headers.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def describe_error(error: Exception) -> str:
    if isinstance(error, openai.APIStatusError):
        return f"HTTP {error.status_code}"
    return type(error).__name__


class LLMClient:
    """Cached, rate-limited and retrying chat completions (see the module docstring)."""

    def __init__(
        self,
        settings: dict[str, Any],
        dry_run: bool = False,
        call_log_path: Path | None = None,
        openai_client: openai.OpenAI | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """`call_log_path`: JSONL file in the run folder that receives every prompt and
        answer (SPEC.md, section 8). `openai_client`, `sleep` and `clock` are replaced in
        tests, so the tests never reach the real API and never really wait.
        """
        llm = settings["llm"]
        self.temperature: float = llm["temperature"]
        self.max_retries: int = llm["max_retries"]
        self.retry_initial_wait_seconds: float = llm["retry_initial_wait_seconds"]
        self.retry_max_wait_seconds: float = llm["retry_max_wait_seconds"]
        self.requests_per_day: int = llm["requests_per_day"]
        self.dry_run = dry_run
        self.call_log_path = call_log_path
        self.cache = ResponseCache(settings["paths"]["cache"])
        self.rate_limiter = RateLimiter.from_settings(llm, clock=clock, sleep=sleep)
        self.stats = RequestStats()
        self._settings = settings
        self._openai_client = openai_client
        self._sleep = sleep
        self._dry_run_keys: set[str] = set()

    @property
    def openai_client(self) -> openai.OpenAI:
        """Created on first use, so a dry run or a fully cached run needs no API key."""
        if self._openai_client is None:
            self._openai_client = create_openai_client(self._settings)
        return self._openai_client

    def complete(self, model: str, messages: list[Message]) -> ChatResult:
        """Answer from the cache if possible; otherwise send the request (unless dry run)."""
        key = cache_key(model, messages, self.temperature)
        cached_content = self.cache.get(key)
        if cached_content is not None:
            self.stats.cache_hits += 1
            result = ChatResult(cached_content, SOURCE_CACHE, key)
        elif self.dry_run:
            # A set, because a real run would answer a repeated prompt from the cache.
            self._dry_run_keys.add(key)
            self.stats.dry_run_requests = len(self._dry_run_keys)
            result = ChatResult("", SOURCE_DRY_RUN, key)
        else:
            content = self._request_with_retries(model, messages)
            self.cache.put(key, model, self.temperature, content)
            result = ChatResult(content, SOURCE_API, key)
        self._log_call(model, messages, result)
        return result

    def list_models(self) -> list[str]:
        """Model names the API offers (one request)."""
        self.rate_limiter.wait_for_slot()
        self.stats.api_requests += 1
        return sorted(model.id for model in self.openai_client.models.list())

    def _request_with_retries(self, model: str, messages: list[Message]) -> str:
        attempt = 0
        while True:
            try:
                return self._send_once(model, messages)
            except (openai.APIStatusError, openai.APIConnectionError) as error:
                if not is_retryable(error) or attempt >= self.max_retries:
                    raise
                wait_seconds = self._backoff_seconds(attempt, error)
                self.stats.retries += 1
                logger.warning(
                    "Request failed (%s), retry %d of %d in %.0f s",
                    describe_error(error), attempt + 1, self.max_retries, wait_seconds,
                )
                self._sleep(wait_seconds)
                attempt += 1

    def _send_once(self, model: str, messages: list[Message]) -> str:
        """One request; records the rate limit headers of successful and failed answers."""
        self.rate_limiter.wait_for_slot()
        self.stats.api_requests += 1
        try:
            raw_response = self.openai_client.chat.completions.with_raw_response.create(
                model=model, messages=messages, temperature=self.temperature
            )
        except openai.APIStatusError as error:
            self._record_quota(error.response.headers)
            raise
        self._record_quota(raw_response.headers)
        completion = raw_response.parse()
        logger.info(
            "API request %d (%s), limits: %s, remaining quota: %s",
            self.stats.api_requests, model,
            self.stats.rate_limits or "not reported",
            self.stats.remaining_quota or "not reported",
        )
        return completion.choices[0].message.content or ""

    def _record_quota(self, headers: Mapping[str, str]) -> None:
        remaining = headers_with_prefix(headers, QUOTA_HEADER_PREFIX)
        if remaining:
            self.stats.remaining_quota = remaining
        limits = headers_with_prefix(headers, LIMIT_HEADER_PREFIX)
        if limits:
            self.stats.rate_limits = limits
        self.rate_limiter.update_server_remaining(
            parse_remaining(headers, REMAINING_HOUR_HEADER),
            parse_remaining(headers, REMAINING_DAY_HEADER),
        )

    def _backoff_seconds(self, attempt: int, error: Exception) -> float:
        """Exponential backoff; a longer Retry-After from the server wins. Both capped."""
        wait_seconds = self.retry_initial_wait_seconds * BACKOFF_FACTOR ** attempt
        if isinstance(error, openai.APIStatusError):
            retry_after = parse_retry_after(error.response.headers)
            if retry_after is not None:
                wait_seconds = max(wait_seconds, retry_after)
        return min(wait_seconds, self.retry_max_wait_seconds)

    def _log_call(self, model: str, messages: list[Message], result: ChatResult) -> None:
        if self.call_log_path is None:
            return
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "cache_key": result.cache_key,
            "source": result.source,
            "messages": messages,
            "content": result.content,
            "rate_limits": self.stats.rate_limits,
            "remaining_quota": self.stats.remaining_quota,
        }
        self.call_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.call_log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def summary(self) -> str:
        """Short report of the requests, printed at the end of a run."""
        stats = self.stats
        lines = [
            f"Requests sent to the API: {stats.api_requests} (retries: {stats.retries})",
            f"Answers from the cache:   {stats.cache_hits}",
        ]
        if self.dry_run:
            lines.append(f"Dry run: a real run would send {stats.dry_run_requests} request(s)")
            if stats.dry_run_requests > self.requests_per_day:
                lines.append(
                    f"WARNING: more than the daily limit of {self.requests_per_day}; the run "
                    "stops at the limit and continues from the cache on the next day"
                )
        lines.append(f"Limits reported by the server: {stats.rate_limits or 'not reported'}")
        lines.append(f"Remaining quota: {stats.remaining_quota or 'not reported'}")
        return "\n".join(lines)


def read_prompt_file(path: Path) -> list[Message]:
    """Send a whole text file as one user message.

    Enough for the smoke test; the real prompts (system + user) are built by the
    pipelines from config/prompts/extraction.txt (milestone M3).
    """
    return [{"role": "user", "content": path.read_text(encoding="utf-8").strip()}]


def run_ask(settings: dict[str, Any], prompt_file: Path, model_role: str, dry_run: bool) -> None:
    client = LLMClient(settings, dry_run=dry_run)
    model = settings["models"][model_role]
    result = client.complete(model, read_prompt_file(prompt_file))
    print(f"Model:  {model}")
    print(f"Source: {result.source}")
    if result.source != SOURCE_DRY_RUN:
        print(f"Answer: {result.content}")
    print(client.summary())


def run_list_models(settings: dict[str, Any]) -> None:
    client = LLMClient(settings)
    configured = {settings["models"][role] for role in MODEL_ROLES}
    for model in client.list_models():
        marker = "  <- in settings.yaml" if model in configured else ""
        print(f"{model}{marker}")
    print(client.summary())


def main() -> None:
    parser = argparse.ArgumentParser(description="Talk to the SAIA API (cached, rate limited).")
    parser.add_argument("--experiment", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    ask_parser = commands.add_parser("ask", help="send the text of one file as a prompt")
    ask_parser.add_argument("--prompt-file", type=Path, required=True)
    ask_parser.add_argument("--model", choices=MODEL_ROLES, default="chunked",
                            help="which model of `models` in settings.yaml (default: chunked)")
    ask_parser.add_argument("--dry-run", action="store_true",
                            help="count the request without sending it")
    commands.add_parser("list-models", help="list the models the API offers")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per HTTP call is too much
    settings = load_settings(args.experiment)
    try:
        if args.command == "ask":
            run_ask(settings, args.prompt_file, args.model, args.dry_run)
        else:
            run_list_models(settings)
    except MissingApiKeyError as error:
        sys.exit(f"ERROR: {error}")
    except DailyLimitReachedError as error:
        sys.exit(f"STOPPED: {error}")


if __name__ == "__main__":
    main()
