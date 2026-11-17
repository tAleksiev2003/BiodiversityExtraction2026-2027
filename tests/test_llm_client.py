"""Tests for the LLM client, its cache and its rate limiter.

No test reaches the real API: every request is answered by `FakeApi` through an
`httpx.MockTransport`. `sleep` and the clock are replaced, so no test really waits.
"""
import copy
import json
from pathlib import Path
from typing import Any, Callable

import httpx
import openai
import pytest

from llm_cache import ResponseCache, cache_key
from llm_client import (
    SOURCE_API,
    SOURCE_CACHE,
    SOURCE_DRY_RUN,
    LLMClient,
    MissingApiKeyError,
    load_api_key,
    read_prompt_file,
)
from rate_limiter import (
    SECONDS_PER_10_SECONDS,
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    DailyLimitReachedError,
    RateLimiter,
)

FIXTURES = Path(__file__).parent / "fixtures"
SMOKE_PROMPT = FIXTURES / "llm_smoke_prompt.txt"
INVENTED_MODEL = "invented-model"
INVENTED_ANSWER = "The study counted ground beetles along hedgerows."
# Header names as SAIA sends them (M2); the values are invented.
QUOTA_HEADERS = {
    "x-ratelimit-remaining-requests-10s": "9",
    "x-ratelimit-remaining-requests-hour": "399",
    "x-ratelimit-remaining-requests-day": "799",
}
LIMIT_HEADERS = {"x-ratelimit-limit-requests-hour": "400", "x-ratelimit-limit-requests-day": "800"}


def completion_json(content: str) -> dict[str, Any]:
    """The smallest valid OpenAI chat completion body."""
    return {
        "id": "invented-id",
        "object": "chat.completion",
        "created": 0,
        "model": INVENTED_MODEL,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
    }


class FakeApi:
    """Answers requests from a list of planned responses and counts them.

    A planned response is an int (an HTTP error status), an exception (raised like a
    network failure) or a string (a successful answer with that content). `headers`
    replaces the rate limit headers of every response (e.g. to report 0 remaining).
    """

    def __init__(
        self,
        planned: list[int | str | Exception] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.planned = list(planned or [])
        self.headers = {**QUOTA_HEADERS, **LIMIT_HEADERS} if headers is None else headers
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": "model-b", "object": "model", "created": 0, "owned_by": "x"},
                {"id": "model-a", "object": "model", "created": 0, "owned_by": "x"},
            ]})
        response = self.planned.pop(0) if self.planned else INVENTED_ANSWER
        if isinstance(response, Exception):
            raise response
        if isinstance(response, int):
            return httpx.Response(response, json={"error": {"message": "invented error"}},
                                  headers={"retry-after": "1", **self.headers})
        return httpx.Response(200, json=completion_json(response), headers=self.headers)

    def openai_client(self) -> openai.OpenAI:
        return openai.OpenAI(
            api_key="invented-test-key",
            base_url="https://llm.invalid/v1",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(self.handle)),
        )


@pytest.fixture
def llm_settings(settings: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    """The real settings, but with the cache in a temporary folder."""
    copied = copy.deepcopy(settings)
    copied["paths"]["cache"] = tmp_path / "cache"
    return copied


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def make_client(llm_settings: dict[str, Any], sleeps: list[float]) -> Callable[..., LLMClient]:
    def build(api: FakeApi | None = None, **kwargs: Any) -> LLMClient:
        return LLMClient(
            llm_settings,
            openai_client=api.openai_client() if api is not None else None,
            sleep=sleeps.append,
            **kwargs,
        )
    return build


def user_message(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


# --- cache -------------------------------------------------------------------------

def test_cache_key_depends_on_model_prompt_and_temperature() -> None:
    base = cache_key("model", user_message("text"), 0)
    assert base == cache_key("model", user_message("text"), 0)
    assert base != cache_key("other-model", user_message("text"), 0)
    assert base != cache_key("model", user_message("other text"), 0)
    assert base != cache_key("model", user_message("text"), 0.5)


def test_cache_round_trip_leaves_no_temporary_file(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    assert cache.get("key") is None
    cache.put("key", "model", 0, "answer")
    assert cache.get("key") == "answer"
    assert [path.name for path in tmp_path.iterdir()] == ["key.json"]


# --- rate limiter --------------------------------------------------------------------

class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def make_limiter(
    clock: FakeClock, per_10_seconds: int = 10, per_hour: int = 400, per_day: int = 800
) -> RateLimiter:
    return RateLimiter(per_10_seconds, per_hour, per_day, clock=clock, sleep=clock.sleep)


def test_rate_limiter_waits_when_the_10_second_window_is_full() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, per_10_seconds=3)
    assert [limiter.wait_for_slot() for _ in range(3)] == [0.0, 0.0, 0.0]
    assert limiter.wait_for_slot() == SECONDS_PER_10_SECONDS
    assert clock.now == SECONDS_PER_10_SECONDS


def test_rate_limiter_does_not_wait_for_old_requests() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, per_10_seconds=2)
    limiter.wait_for_slot()
    clock.now = 5.0
    limiter.wait_for_slot()
    clock.now = 12.0  # the first request has left the window, the second has not
    assert limiter.wait_for_slot() == 0.0
    assert limiter.wait_for_slot() == pytest.approx(3.0)  # until the request at 5 s is 10 s old


def test_rate_limiter_waits_when_the_hour_is_full() -> None:
    # 10 per minute would be 600 per hour: the hourly window must slow the run down.
    clock = FakeClock()
    limiter = make_limiter(clock, per_10_seconds=10, per_hour=20)
    for _ in range(20):
        limiter.wait_for_slot()  # 10 requests at 0 s, 10 at 10 s
    assert clock.now == SECONDS_PER_10_SECONDS
    limiter.wait_for_slot()
    # The 10-second window would allow it at 20 s; the hour is full until the requests
    # sent at 0 s are one hour old.
    assert clock.now == pytest.approx(SECONDS_PER_HOUR)


def test_rate_limiter_respects_both_waiting_windows_at_once() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, per_10_seconds=2, per_hour=3)
    for _ in range(3):
        limiter.wait_for_slot()  # at 0, 0 and 10 s
    assert clock.now == SECONDS_PER_10_SECONDS
    limiter.wait_for_slot()
    # The 10-second window has room at 20 s, but the hour is full until 3600 s.
    assert clock.now == pytest.approx(SECONDS_PER_HOUR)


def test_rate_limiter_stops_at_the_daily_limit_without_waiting() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, per_day=2)
    limiter.wait_for_slot()
    limiter.wait_for_slot()
    time_before = clock.now
    with pytest.raises(DailyLimitReachedError, match="tomorrow"):
        limiter.wait_for_slot()
    assert clock.now == time_before  # stopped at once instead of sleeping for hours


def test_rate_limiter_allows_requests_again_after_a_day() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, per_day=1)
    limiter.wait_for_slot()
    clock.now = SECONDS_PER_DAY
    assert limiter.wait_for_slot() == 0.0


def test_client_stops_at_the_daily_limit_and_keeps_earlier_answers(
    llm_settings: dict[str, Any], sleeps: list[float]
) -> None:
    llm_settings["llm"]["requests_per_day"] = 1
    api = FakeApi(headers={})  # no server counts: the local daily count decides
    client = LLMClient(llm_settings, openai_client=api.openai_client(), sleep=sleeps.append)
    client.complete(INVENTED_MODEL, user_message("chunk 1"))
    with pytest.raises(DailyLimitReachedError):
        client.complete(INVENTED_MODEL, user_message("chunk 2"))
    assert len(api.requests) == 1  # the second request was never sent

    next_day = LLMClient(llm_settings, openai_client=FakeApi().openai_client(), sleep=sleeps.append)
    assert next_day.complete(INVENTED_MODEL, user_message("chunk 1")).source == SOURCE_CACHE


# --- server-reported remaining counts win over local counting ----------------------------

def test_server_reporting_0_for_the_day_stops_although_local_count_is_low() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock)  # a restart: the local count is empty
    limiter.update_server_remaining(remaining_hour=10, remaining_day=0)
    with pytest.raises(DailyLimitReachedError):
        limiter.wait_for_slot()
    assert clock.now == 0.0


def test_server_count_for_the_day_wins_over_a_full_local_count() -> None:
    # e.g. the server's day already restarted while our sliding 24 hours have not passed
    clock = FakeClock()
    limiter = make_limiter(clock, per_day=1)
    limiter.wait_for_slot()
    limiter.update_server_remaining(remaining_hour=10, remaining_day=5)
    assert limiter.wait_for_slot() == 0.0


def test_server_reporting_0_for_the_hour_waits_one_hour_from_the_report() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock)
    clock.now = 100.0
    limiter.update_server_remaining(remaining_hour=0, remaining_day=500)
    clock.now = 160.0
    assert limiter.wait_for_slot() == pytest.approx(SECONDS_PER_HOUR - 60.0)
    assert clock.now == pytest.approx(100.0 + SECONDS_PER_HOUR)


def test_missing_server_headers_fall_back_to_local_counting() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, per_day=1)
    limiter.update_server_remaining(remaining_hour=0, remaining_day=0)
    limiter.update_server_remaining(remaining_hour=None, remaining_day=None)  # newer response
    assert limiter.wait_for_slot() == 0.0  # local count: 0 of 1 used
    with pytest.raises(DailyLimitReachedError):
        limiter.wait_for_slot()  # local count: 1 of 1 used


def make_clocked_client(
    llm_settings: dict[str, Any], api: FakeApi, clock: FakeClock
) -> LLMClient:
    """A client whose limiter and retries use the fake clock, so waits move time forward."""
    return LLMClient(llm_settings, openai_client=api.openai_client(),
                     sleep=clock.sleep, clock=clock)


def test_client_stops_when_the_server_reports_0_left_for_the_day(
    llm_settings: dict[str, Any]
) -> None:
    api = FakeApi(headers={**QUOTA_HEADERS, "x-ratelimit-remaining-requests-day": "0"})
    client = make_clocked_client(llm_settings, api, FakeClock())
    assert client.complete(INVENTED_MODEL, user_message("chunk 1")).source == SOURCE_API
    with pytest.raises(DailyLimitReachedError):
        client.complete(INVENTED_MODEL, user_message("chunk 2"))
    assert len(api.requests) == 1  # stopped before the second request
    assert client.cache.get(cache_key(INVENTED_MODEL, user_message("chunk 1"),
                                      client.temperature)) == INVENTED_ANSWER


def test_client_waits_when_the_server_reports_0_left_for_the_hour(
    llm_settings: dict[str, Any]
) -> None:
    clock = FakeClock()
    api = FakeApi(headers={**QUOTA_HEADERS, "x-ratelimit-remaining-requests-hour": "0"})
    client = make_clocked_client(llm_settings, api, clock)
    client.complete(INVENTED_MODEL, user_message("chunk 1"))
    client.complete(INVENTED_MODEL, user_message("chunk 2"))
    assert len(api.requests) == 2
    assert clock.now == pytest.approx(SECONDS_PER_HOUR)


def test_client_reads_server_counts_from_a_429_answer(llm_settings: dict[str, Any]) -> None:
    # The retry after a 429 must not be sent when that 429 said the day is used up.
    api = FakeApi([429], headers={**QUOTA_HEADERS, "x-ratelimit-remaining-requests-day": "0"})
    client = make_clocked_client(llm_settings, api, FakeClock())
    with pytest.raises(DailyLimitReachedError):
        client.complete(INVENTED_MODEL, user_message("chunk 1"))
    assert len(api.requests) == 1


# --- end to end with the fixture prompt (SPEC.md, milestone M2) --------------------------

def test_second_run_makes_zero_requests(make_client: Callable[..., LLMClient]) -> None:
    messages = read_prompt_file(SMOKE_PROMPT)

    first_api = FakeApi()
    first_run = make_client(first_api)
    first_result = first_run.complete(INVENTED_MODEL, messages)
    assert first_result.source == SOURCE_API
    assert first_result.content == INVENTED_ANSWER
    assert len(first_api.requests) == 1
    assert first_run.stats.api_requests == 1

    second_api = FakeApi()
    second_run = make_client(second_api)  # a new process: only the disk cache is shared
    second_result = second_run.complete(INVENTED_MODEL, messages)
    assert second_result.source == SOURCE_CACHE
    assert second_result.content == INVENTED_ANSWER
    assert second_api.requests == []
    assert second_run.stats.api_requests == 0
    assert second_run.stats.cache_hits == 1


def test_request_body_contains_model_prompt_and_temperature(
    make_client: Callable[..., LLMClient], llm_settings: dict[str, Any]
) -> None:
    api = FakeApi()
    make_client(api).complete(INVENTED_MODEL, user_message("invented text"))
    body = json.loads(api.requests[0].content)
    assert body["model"] == INVENTED_MODEL
    assert body["messages"] == user_message("invented text")
    assert body["temperature"] == llm_settings["llm"]["temperature"]


# --- dry run -------------------------------------------------------------------------

def test_dry_run_sends_nothing_and_needs_no_key(
    make_client: Callable[..., LLMClient], llm_settings: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(llm_settings["llm"]["api_key_env"], raising=False)
    client = make_client(api=None, dry_run=True)  # no OpenAI client: any request would fail
    first = client.complete(INVENTED_MODEL, user_message("chunk 1"))
    client.complete(INVENTED_MODEL, user_message("chunk 2"))
    client.complete(INVENTED_MODEL, user_message("chunk 1"))  # a real run would use the cache
    assert first.source == SOURCE_DRY_RUN
    assert first.content == ""
    assert client.stats.dry_run_requests == 2
    assert client.stats.api_requests == 0
    assert "would send 2 request(s)" in client.summary()


def test_dry_run_uses_cached_answers(make_client: Callable[..., LLMClient]) -> None:
    make_client(FakeApi()).complete(INVENTED_MODEL, user_message("chunk 1"))
    dry_client = make_client(api=None, dry_run=True)
    assert dry_client.complete(INVENTED_MODEL, user_message("chunk 1")).source == SOURCE_CACHE
    assert dry_client.stats.dry_run_requests == 0


def test_dry_run_warns_above_the_daily_limit(
    llm_settings: dict[str, Any], sleeps: list[float]
) -> None:
    llm_settings["llm"]["requests_per_day"] = 1
    client = LLMClient(llm_settings, dry_run=True, sleep=sleeps.append)
    client.complete(INVENTED_MODEL, user_message("chunk 1"))
    client.complete(INVENTED_MODEL, user_message("chunk 2"))
    assert "WARNING" in client.summary()


# --- retries ---------------------------------------------------------------------------

def test_retries_on_rate_limit_with_exponential_backoff(
    make_client: Callable[..., LLMClient], llm_settings: dict[str, Any], sleeps: list[float]
) -> None:
    api = FakeApi([429, 503, INVENTED_ANSWER])
    client = make_client(api)
    result = client.complete(INVENTED_MODEL, user_message("text"))
    initial_wait = llm_settings["llm"]["retry_initial_wait_seconds"]
    assert result.content == INVENTED_ANSWER
    assert client.stats.api_requests == 3
    assert client.stats.retries == 2
    assert sleeps == [initial_wait, initial_wait * 2]


def test_longer_retry_after_from_server_wins(
    make_client: Callable[..., LLMClient], llm_settings: dict[str, Any], sleeps: list[float]
) -> None:
    llm_settings["llm"]["retry_initial_wait_seconds"] = 0.5  # shorter than Retry-After: 1
    make_client(FakeApi([429, INVENTED_ANSWER])).complete(INVENTED_MODEL, user_message("text"))
    assert sleeps == [1.0]


def test_retries_on_connection_errors(make_client: Callable[..., LLMClient]) -> None:
    api = FakeApi([httpx.ConnectError("invented network failure"), INVENTED_ANSWER])
    client = make_client(api)
    assert client.complete(INVENTED_MODEL, user_message("text")).content == INVENTED_ANSWER
    assert client.stats.retries == 1


def test_gives_up_after_max_retries_and_caches_nothing(
    make_client: Callable[..., LLMClient], llm_settings: dict[str, Any]
) -> None:
    max_retries = llm_settings["llm"]["max_retries"]
    api = FakeApi([500] * (max_retries + 1))
    client = make_client(api)
    with pytest.raises(openai.InternalServerError):
        client.complete(INVENTED_MODEL, user_message("text"))
    assert len(api.requests) == max_retries + 1
    assert not llm_settings["paths"]["cache"].exists()


def test_does_not_retry_client_errors(make_client: Callable[..., LLMClient]) -> None:
    api = FakeApi([400])
    with pytest.raises(openai.BadRequestError):
        make_client(api).complete(INVENTED_MODEL, user_message("text"))
    assert len(api.requests) == 1


# --- quota, call log, models -----------------------------------------------------------

def test_remaining_quota_and_limits_are_recorded(make_client: Callable[..., LLMClient]) -> None:
    client = make_client(FakeApi())
    client.complete(INVENTED_MODEL, user_message("text"))
    assert client.stats.remaining_quota == QUOTA_HEADERS
    assert client.stats.rate_limits == LIMIT_HEADERS
    assert "x-ratelimit-limit-requests-day" in client.summary()


def test_call_log_has_prompt_answer_and_source(
    make_client: Callable[..., LLMClient], tmp_path: Path
) -> None:
    log_path = tmp_path / "run" / "llm_calls.jsonl"
    client = make_client(FakeApi(), call_log_path=log_path)
    client.complete(INVENTED_MODEL, user_message("text"))
    client.complete(INVENTED_MODEL, user_message("text"))
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [entry["source"] for entry in entries] == [SOURCE_API, SOURCE_CACHE]
    assert entries[0]["messages"] == user_message("text")
    assert entries[0]["content"] == INVENTED_ANSWER
    assert entries[0]["rate_limits"] == LIMIT_HEADERS


def test_list_models_is_sorted(make_client: Callable[..., LLMClient]) -> None:
    client = make_client(FakeApi())
    assert client.list_models() == ["model-a", "model-b"]
    assert client.stats.api_requests == 1


# --- API key ---------------------------------------------------------------------------

INVENTED_KEY_VARIABLE = "INVENTED_TEST_ONLY_KEY"


def test_api_key_is_loaded_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(INVENTED_KEY_VARIABLE, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"{INVENTED_KEY_VARIABLE}=invented-secret\n", encoding="utf-8")
    assert load_api_key(INVENTED_KEY_VARIABLE, env_file) == "invented-secret"
    monkeypatch.delenv(INVENTED_KEY_VARIABLE)  # load_dotenv set it; do not leak into other tests


def test_missing_api_key_names_the_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(INVENTED_KEY_VARIABLE, raising=False)
    with pytest.raises(MissingApiKeyError, match=INVENTED_KEY_VARIABLE):
        load_api_key(INVENTED_KEY_VARIABLE, tmp_path / "missing.env")
