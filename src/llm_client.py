"""LLM client for the OpenAI-compatible SAIA API (SPEC.md, section 5).

Every LLM call of the pipelines goes through `LLMClient.complete`. SAIA allows only
10 requests per 10 seconds, 400 per hour and 800 per day, so each call passes these
steps in order:
1. disk cache: an answered prompt is never requested again (src/llm_cache.py);
2. dry run: count the requests a real run would send, without sending them;
3. key choice: with several API keys, the key with the most requests left today
   (src/api_key_pool.py); every key has its own limiter (SAIA counts per account, section 5);
4. rate limiter of that key: wait until all windows have room (src/rate_limiter.py);
   stop only when every key has reached its daily limit;
5. retries with exponential backoff on 429, 5xx, timeouts and connection errors;
   a key the server rejects (401/403) is dropped and the request goes to another key;
6. logging of the number of requests, the limits the server reports
   (x-ratelimit-limit-*) and the remaining quota (x-ratelimit-remaining-*) per key.

One answer can take minutes, so the pipelines call `complete` from several threads at
once (`max_parallel_requests()` = `llm.max_concurrent_requests_per_key` x number of
keys). The cache, the key pool and every rate limiter are thread-safe, so the limits of
each key still hold exactly; the counters and the call log are protected by a lock.
`request_stop()` (after Ctrl + C) wakes every thread that waits for a rate limit or a
retry and makes it raise RunStoppedError; requests already sent finish and are cached.

The API keys are read from .env (`<llm.api_key_env>_1`, `_2`, ... or `<llm.api_key_env>`)
only when the first real request is sent, so a dry run or a fully cached run works
without a key. A key is never printed, logged or written anywhere; logs only show its
number ("key 2").

Usage (run by the team; the commands that send requests need the key in .env):
    python3 src/llm_client.py ask --prompt-file tests/fixtures/llm_smoke_prompt.txt
    python3 src/llm_client.py ask --prompt-file tests/fixtures/llm_smoke_prompt.txt --dry-run
    python3 src/llm_client.py list-models
    python3 src/llm_client.py check-keys     # one request per key: OK or error, limits, quota
"""
import argparse
import json
import logging
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import openai

from api_key_pool import (
    ApiKeyPool,
    ApiKeySlot,
    KeyStats,
    MissingApiKeyError,
    NoUsableApiKeyError,
    load_api_keys,
)
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

# check-keys: the smallest useful chat request. A chat request, not /v1/models, because
# only chat responses carry the x-ratelimit-* headers.
CHECK_KEYS_PROMPT = "Answer with OK."
CHECK_KEYS_MODEL_ROLE = "chunked"
KEY_OK = "OK"
MODELS_ENDPOINT_NOTE = (
    "Note: /v1/models responses carry no x-ratelimit-* headers, so list-models shows the "
    "limits and the quota as 'not reported'; check-keys sends a chat request instead."
)


# The server refuses this key (invalid or blocked): another attempt with it cannot succeed.
KEY_REJECTED_ERRORS = (openai.AuthenticationError, openai.PermissionDeniedError)  # 401, 403


class RunStoppedError(RuntimeError):
    """The run was stopped (Ctrl + C); raised instead of waiting or sending a new request."""


def create_openai_client(settings: dict[str, Any], api_key: str) -> openai.OpenAI:
    """OpenAI SDK client pointed at SAIA, one per API key.

    The SDK's own retries are switched off (max_retries=0): LLMClient retries itself, so
    every attempt is rate limited, counted and logged.
    """
    llm = settings["llm"]
    return openai.OpenAI(
        api_key=api_key,
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


@dataclass(frozen=True)
class KeyCheck:
    """Result of check-keys for one key; holds the key's number, never the key."""

    label: str
    status: str                       # KEY_OK, or the error, e.g. "HTTP 403"
    rate_limits: dict[str, str]
    remaining_quota: dict[str, str]


@dataclass
class RequestStats:
    """Counters for one client; the pipelines write them into the run folder."""

    api_requests: int = 0       # every request sent to the server, retries included
    retries: int = 0
    request_seconds_total: float = 0.0  # time from sending to the answer, summed over requests
    cache_hits: int = 0
    dry_run_requests: int = 0   # different prompts a real run would still send
    # Per key ("key 1", ...): requests, rejection, and the last x-ratelimit-* headers, i.e.
    # the limits the server really applies, so we can check the values in settings.yaml.
    per_key: dict[str, KeyStats] = field(default_factory=dict)

    def average_request_seconds(self) -> float | None:
        """Mean time one request took (None before the first request)."""
        if self.api_requests == 0:
            return None
        return self.request_seconds_total / self.api_requests

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["average_request_seconds"] = self.average_request_seconds()
        return result


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
        openai_clients: Sequence[openai.OpenAI] | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """`call_log_path`: JSONL file in the run folder that receives every prompt and
        answer (SPEC.md, section 8). `openai_clients` (one per key), `sleep` and `clock`
        are replaced in tests, so the tests never reach the real API and never really wait.
        Without `sleep`, waits end early when `request_stop()` is called.
        """
        llm = settings["llm"]
        self.temperature: float = llm["temperature"]
        self.max_retries: int = llm["max_retries"]
        self.retry_initial_wait_seconds: float = llm["retry_initial_wait_seconds"]
        self.retry_max_wait_seconds: float = llm["retry_max_wait_seconds"]
        self.requests_per_day: int = llm["requests_per_day"]
        self.max_concurrent_requests_per_key: int = llm["max_concurrent_requests_per_key"]
        self.dry_run = dry_run
        self.call_log_path = call_log_path
        self.cache = ResponseCache(settings["paths"]["cache"])
        self.stats = RequestStats()
        self._settings = settings
        self._openai_clients = openai_clients
        self._key_pool: ApiKeyPool | None = None
        self._stop_requested = threading.Event()
        self._sleep = sleep if sleep is not None else self._sleep_unless_stopped
        self._clock = clock
        self._dry_run_keys: set[str] = set()
        # One lock for the counters, the dry-run set, the call log and creating the pool.
        self._lock = threading.Lock()

    @property
    def key_pool(self) -> ApiKeyPool:
        """Created on first use, so a dry run or a fully cached run needs no API key."""
        with self._lock:
            if self._key_pool is None:
                clients = self._openai_clients
                if clients is None:
                    llm = self._settings["llm"]
                    api_keys = load_api_keys(llm["api_key_env"], ENV_FILE)
                    clients = [create_openai_client(self._settings, api_key) for api_key in api_keys]
                self._key_pool = ApiKeyPool(clients, self._make_rate_limiter,
                                            self.max_concurrent_requests_per_key)
                self.stats.per_key = {slot.label: slot.stats for slot in self._key_pool.slots}
                logger.info("Using %d API key(s)", len(self._key_pool.slots))
            return self._key_pool

    def max_parallel_requests(self) -> int:
        """How many requests the pipelines may send at the same time.

        `max_concurrent_requests_per_key` for every key. A dry run sends nothing and
        loads no key, and a run without a key in .env can only use the cache, so both
        count as one key.
        """
        if self.dry_run:
            return self.max_concurrent_requests_per_key
        try:
            key_count = len(self.key_pool.slots)
        except MissingApiKeyError:
            key_count = 1  # the first request that is not cached raises the error again
        return self.max_concurrent_requests_per_key * key_count

    def request_stop(self) -> None:
        """Stop the run: no new request is sent, and waiting threads wake up and raise RunStoppedError."""
        self._stop_requested.set()

    def _raise_if_stopped(self) -> None:
        if self._stop_requested.is_set():
            raise RunStoppedError("The run was stopped; answers received so far are in the cache.")

    def _sleep_unless_stopped(self, seconds: float) -> None:
        """Like time.sleep, but ends at once when request_stop() is called."""
        if self._stop_requested.wait(seconds):
            self._raise_if_stopped()

    def _make_rate_limiter(self) -> RateLimiter:
        """A fresh limiter per key. SAIA counts per account; the server counts in each
        answer correct a key's local count when several keys share one account."""
        return RateLimiter.from_settings(self._settings["llm"], clock=self._clock, sleep=self._sleep)

    def complete(self, model: str, messages: list[Message]) -> ChatResult:
        """Answer from the cache if possible; otherwise send the request (unless dry run)."""
        key = cache_key(model, messages, self.temperature)
        # The cache key does not depend on the API key: an answer is shared by all keys.
        cached_content = self.cache.get(key)
        used_slot: ApiKeySlot | None = None
        if cached_content is not None:
            with self._lock:
                self.stats.cache_hits += 1
            result = ChatResult(cached_content, SOURCE_CACHE, key)
        elif self.dry_run:
            with self._lock:
                # A set, because a real run would answer a repeated prompt from the cache.
                self._dry_run_keys.add(key)
                self.stats.dry_run_requests = len(self._dry_run_keys)
            result = ChatResult("", SOURCE_DRY_RUN, key)
        else:
            def send_chat(client: openai.OpenAI) -> Any:
                return client.chat.completions.with_raw_response.create(
                    model=model, messages=messages, temperature=self.temperature
                )
            used_slot, raw_response = self._request_with_retries(send_chat, model)
            completion = raw_response.parse()
            content = completion.choices[0].message.content or ""
            self.cache.put(key, model, self.temperature, content)
            result = ChatResult(content, SOURCE_API, key)
        self._log_call(model, messages, result, used_slot)
        return result

    def list_models(self) -> list[str]:
        """Model names the API offers (one request)."""
        def send_list(client: openai.OpenAI) -> Any:
            return client.models.with_raw_response.list()
        _, raw_response = self._request_with_retries(send_list, "list models")
        return sorted(model.id for model in raw_response.parse().data)

    def check_keys(self, model: str) -> list[KeyCheck]:
        """Send one tiny chat request with every key, bypassing the cache.

        Every key must really send a request, so the cache is neither read nor written,
        no key is chosen by the pool, and nothing is retried: a failure is the result.
        A key whose limiter says its day is used up is not sent and reported as such.
        """
        messages: list[Message] = [{"role": "user", "content": CHECK_KEYS_PROMPT}]

        def send_check(client: openai.OpenAI) -> Any:
            return client.chat.completions.with_raw_response.create(
                model=model, messages=messages, temperature=self.temperature
            )

        results = []
        for slot in self.key_pool.slots:
            try:
                ticket = slot.rate_limiter.reserve_slot()
                self._send_once(slot, ticket, send_check, f"check {slot.label}")
                status = KEY_OK
            except DailyLimitReachedError:
                status = "not sent: daily limit reached"
            except (openai.APIStatusError, openai.APIConnectionError) as error:
                status = describe_error(error)
                if isinstance(error, KEY_REJECTED_ERRORS):
                    self.key_pool.reject(slot, status)
            results.append(KeyCheck(slot.label, status, dict(slot.stats.rate_limits),
                                    dict(slot.stats.remaining_quota)))
        return results

    def _request_with_retries(
        self, send: Callable[[openai.OpenAI], Any], description: str
    ) -> tuple[ApiKeySlot, Any]:
        """Send with the best key; retry transient errors, move on from rejected keys.

        `send` makes one raw request with the given OpenAI client. Returns the key used
        and the raw response (headers + body).
        """
        attempt = 0
        while True:
            self._raise_if_stopped()
            # Chooses a key and reserves a place in its limiter (may wait); raises when
            # all keys are used up for the day or rejected.
            slot, ticket = self.key_pool.acquire()
            try:
                self._raise_if_stopped()  # the stop may have come while waiting for a key
                return slot, self._send_once(slot, ticket, send, description)
            except KEY_REJECTED_ERRORS as error:
                # Not a retry: this key is gone, the next loop picks another one at once.
                self.key_pool.reject(slot, describe_error(error))
                continue
            except (openai.APIStatusError, openai.APIConnectionError) as error:
                if not is_retryable(error) or attempt >= self.max_retries:
                    raise
                wait_seconds = self._backoff_seconds(attempt, error)
                with self._lock:
                    self.stats.retries += 1
                logger.warning(
                    "Request failed with %s (%s), retry %d of %d in %.0f s",
                    slot.label, describe_error(error), attempt + 1, self.max_retries, wait_seconds,
                )
            finally:
                # Before the backoff wait: a waiting retry does not keep the key busy.
                self.key_pool.release(slot)
            self._sleep(wait_seconds)
            attempt += 1

    def _send_once(
        self, slot: ApiKeySlot, ticket: int, send: Callable[[openai.OpenAI], Any], description: str
    ) -> Any:
        """One request with one key, whose limiter has already recorded it as `ticket`.

        Records the rate limit headers of every answer and the time the request took.
        """
        with self._lock:
            self.stats.api_requests += 1
            request_number = self.stats.api_requests
            slot.stats.requests += 1
        started = time.perf_counter()  # real time, also in tests: it is only reported
        try:
            raw_response = send(slot.openai_client)
        except openai.APIStatusError as error:
            self._record_quota(slot, ticket, error.response.headers, time.perf_counter() - started)
            raise
        except openai.APIConnectionError:
            self._add_request_seconds(time.perf_counter() - started)
            slot.rate_limiter.request_ended_without_answer(ticket)
            raise
        self._record_quota(slot, ticket, raw_response.headers, time.perf_counter() - started)
        logger.info(
            "API request %d (%s) with %s, limits: %s, remaining quota: %s",
            request_number, description, slot.label,
            slot.stats.rate_limits or "not reported",
            slot.stats.remaining_quota or "not reported",
        )
        return raw_response

    def _add_request_seconds(self, seconds: float) -> None:
        with self._lock:
            self.stats.request_seconds_total += seconds

    def _record_quota(
        self, slot: ApiKeySlot, ticket: int, headers: Mapping[str, str], seconds: float
    ) -> None:
        remaining = headers_with_prefix(headers, QUOTA_HEADER_PREFIX)
        limits = headers_with_prefix(headers, LIMIT_HEADER_PREFIX)
        with self._lock:
            self.stats.request_seconds_total += seconds
            if remaining:
                slot.stats.remaining_quota = remaining
            if limits:
                slot.stats.rate_limits = limits
        slot.rate_limiter.update_server_remaining(
            parse_remaining(headers, REMAINING_HOUR_HEADER),
            parse_remaining(headers, REMAINING_DAY_HEADER),
            ticket,
        )

    def _backoff_seconds(self, attempt: int, error: Exception) -> float:
        """Exponential backoff; a longer Retry-After from the server wins. Both capped."""
        wait_seconds = self.retry_initial_wait_seconds * BACKOFF_FACTOR ** attempt
        if isinstance(error, openai.APIStatusError):
            retry_after = parse_retry_after(error.response.headers)
            if retry_after is not None:
                wait_seconds = max(wait_seconds, retry_after)
        return min(wait_seconds, self.retry_max_wait_seconds)

    def _log_call(
        self, model: str, messages: list[Message], result: ChatResult, slot: ApiKeySlot | None
    ) -> None:
        """`slot` is the key that answered; None for cache hits and dry runs."""
        if self.call_log_path is None:
            return
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "cache_key": result.cache_key,
            "source": result.source,
            "api_key": slot.label if slot is not None else None,
            "messages": messages,
            "content": result.content,
            "rate_limits": slot.stats.rate_limits if slot is not None else {},
            "remaining_quota": slot.stats.remaining_quota if slot is not None else {},
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._lock:  # one whole line per call, never two lines mixed
            self.call_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.call_log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(line)

    def summary(self) -> str:
        """Short report of the requests, printed at the end of a run."""
        stats = self.stats
        lines = [
            f"Requests sent to the API: {stats.api_requests} (retries: {stats.retries})",
            f"Answers from the cache:   {stats.cache_hits}",
        ]
        average_seconds = stats.average_request_seconds()
        if average_seconds is not None:
            lines.append(f"Average time per request: {average_seconds:.1f} s")
        if self.dry_run:
            lines.append(f"Dry run: a real run would send {stats.dry_run_requests} request(s)")
            if stats.dry_run_requests > self.requests_per_day:
                lines.append(
                    f"WARNING: more than the daily limit of {self.requests_per_day} per key; "
                    "the run stops when all keys reach it and continues from the cache on "
                    "the next day"
                )
        if not stats.per_key:
            lines.append("API keys: none loaded (no request was sent)")
        for label, key_stats in stats.per_key.items():
            lines.append(f"{label}: {key_stats.requests} request(s)")
            if key_stats.rejected:
                lines.append(f"    rejected by the server ({key_stats.rejected}), dropped for this run")
            lines.append(f"    limits reported by the server: {key_stats.rate_limits or 'not reported'}")
            lines.append(f"    remaining quota: {key_stats.remaining_quota or 'not reported'}")
        return "\n".join(lines)


def read_prompt_file(path: Path) -> list[Message]:
    """Send a whole text file as one user message.

    Enough for the smoke test; the real prompts (system + user) are built by the
    pipelines from the file in paths.prompt_extraction (milestone M3).
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
    print(MODELS_ENDPOINT_NOTE)


def format_key_checks(checks: list[KeyCheck]) -> str:
    """One block per key: status, limits and remaining quota (key numbers only)."""
    lines = []
    for check in checks:
        lines.append(f"{check.label}: {check.status}")
        lines.append(f"    limits reported by the server: {check.rate_limits or 'not reported'}")
        lines.append(f"    remaining quota: {check.remaining_quota or 'not reported'}")
    return "\n".join(lines)


def run_check_keys(settings: dict[str, Any]) -> None:
    client = LLMClient(settings)
    model = settings["models"][CHECK_KEYS_MODEL_ROLE]
    checks = client.check_keys(model)
    print(f"Model: {model}  (one request per key, cache bypassed)")
    print(format_key_checks(checks))
    print(MODELS_ENDPOINT_NOTE)


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
    commands.add_parser("check-keys",
                        help="send one tiny request with every key; show status, limits, quota")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per HTTP call is too much
    settings = load_settings(args.experiment)
    try:
        if args.command == "ask":
            run_ask(settings, args.prompt_file, args.model, args.dry_run)
        elif args.command == "check-keys":
            run_check_keys(settings)
        else:
            run_list_models(settings)
    except (MissingApiKeyError, NoUsableApiKeyError) as error:
        sys.exit(f"ERROR: {error}")
    except DailyLimitReachedError as error:
        sys.exit(f"STOPPED: {error}")
    except KeyboardInterrupt:
        sys.exit("STOPPED by Ctrl + C. Answers received so far are in the cache.")


if __name__ == "__main__":
    main()
