"""Tests for the LLM client, its cache, its rate limiter and its API key pool.

No test reaches the real API: every request is answered by `FakeApi` through an
`httpx.MockTransport`. `sleep` and the clock are replaced, so no test really waits.
"""
import copy
import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx
import openai
import pytest

from api_key_pool import MissingApiKeyError, NoUsableApiKeyError, load_api_keys
from llm_cache import ResponseCache, cache_key
from llm_client import (
    CHECK_KEYS_PROMPT,
    KEY_OK,
    MODELS_ENDPOINT_NOTE,
    SOURCE_API,
    SOURCE_CACHE,
    SOURCE_DRY_RUN,
    LLMClient,
    format_key_checks,
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

    def openai_client(self, api_key: str = "invented-test-key") -> openai.OpenAI:
        return openai.OpenAI(
            api_key=api_key,
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
            openai_clients=[api.openai_client()] if api is not None else None,
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
    client = LLMClient(llm_settings, openai_clients=[api.openai_client()], sleep=sleeps.append)
    client.complete(INVENTED_MODEL, user_message("chunk 1"))
    with pytest.raises(DailyLimitReachedError):
        client.complete(INVENTED_MODEL, user_message("chunk 2"))
    assert len(api.requests) == 1  # the second request was never sent

    next_day = LLMClient(llm_settings, openai_clients=[FakeApi().openai_client()], sleep=sleeps.append)
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
    return LLMClient(llm_settings, openai_clients=[api.openai_client()],
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
    base_name = llm_settings["llm"]["api_key_env"]
    monkeypatch.delenv(base_name, raising=False)
    monkeypatch.delenv(f"{base_name}_1", raising=False)
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
    assert client.stats.per_key["key 1"].remaining_quota == QUOTA_HEADERS
    assert client.stats.per_key["key 1"].rate_limits == LIMIT_HEADERS
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
    assert [entry["api_key"] for entry in entries] == ["key 1", None]


def test_list_models_is_sorted(make_client: Callable[..., LLMClient]) -> None:
    client = make_client(FakeApi())
    assert client.list_models() == ["model-a", "model-b"]
    assert client.stats.api_requests == 1


# --- loading API keys from .env ----------------------------------------------------------

INVENTED_BASE_NAME = "INVENTED_TEST_ONLY_KEY"


@pytest.fixture
def clean_key_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No invented key variables before the test; load_dotenv sets them, so delete them after."""
    names = [INVENTED_BASE_NAME] + [f"{INVENTED_BASE_NAME}_{number}" for number in range(1, 6)]
    for name in names:
        monkeypatch.delenv(name, raising=False)
    yield
    for name in names:
        monkeypatch.delenv(name, raising=False)


def write_env_file(tmp_path: Path, lines: list[str]) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_file


def test_numbered_keys_are_loaded_in_order(tmp_path: Path, clean_key_environment: None) -> None:
    env_file = write_env_file(tmp_path, [
        f"{INVENTED_BASE_NAME}_2=invented-two",
        f"{INVENTED_BASE_NAME}_1=invented-one",
        f"{INVENTED_BASE_NAME}=invented-single",  # ignored: numbered keys exist
    ])
    assert load_api_keys(INVENTED_BASE_NAME, env_file) == ["invented-one", "invented-two"]


def test_single_old_style_key_is_used_as_key_1(tmp_path: Path, clean_key_environment: None) -> None:
    env_file = write_env_file(tmp_path, [f"{INVENTED_BASE_NAME}=invented-single"])
    assert load_api_keys(INVENTED_BASE_NAME, env_file) == ["invented-single"]


def test_numbering_gap_stops_counting_and_warns_by_name_only(
    tmp_path: Path, clean_key_environment: None, caplog: pytest.LogCaptureFixture
) -> None:
    env_file = write_env_file(tmp_path, [
        f"{INVENTED_BASE_NAME}_1=invented-one",
        f"{INVENTED_BASE_NAME}_3=invented-three",
    ])
    with caplog.at_level(logging.WARNING):
        assert load_api_keys(INVENTED_BASE_NAME, env_file) == ["invented-one"]
    assert f"{INVENTED_BASE_NAME}_3" in caplog.text
    assert "invented-three" not in caplog.text


def test_missing_api_key_names_the_variables(tmp_path: Path, clean_key_environment: None) -> None:
    with pytest.raises(MissingApiKeyError, match=f"{INVENTED_BASE_NAME}_1"):
        load_api_keys(INVENTED_BASE_NAME, tmp_path / "missing.env")


# --- several API keys ----------------------------------------------------------------------

def quota_headers(remaining_day: int) -> dict[str, str]:
    """Invented rate limit headers that report `remaining_day` requests left for the day."""
    return {**QUOTA_HEADERS, **LIMIT_HEADERS,
            "x-ratelimit-remaining-requests-day": str(remaining_day)}


def make_multi_key_client(
    llm_settings: dict[str, Any], apis: list[FakeApi], clock: FakeClock, **kwargs: Any
) -> LLMClient:
    """One fake API per key, so each test can see which key received which request."""
    return LLMClient(llm_settings, openai_clients=[api.openai_client() for api in apis],
                     sleep=clock.sleep, clock=clock, **kwargs)


def test_requests_go_to_the_key_with_most_requests_left(llm_settings: dict[str, Any]) -> None:
    key_1 = FakeApi(headers=quota_headers(remaining_day=100))
    key_2 = FakeApi(headers=quota_headers(remaining_day=500))
    client = make_multi_key_client(llm_settings, [key_1, key_2], FakeClock())
    for chunk_number in range(4):
        client.complete(INVENTED_MODEL, user_message(f"chunk {chunk_number}"))
    # 1st: both unknown (local count 800 each), tie -> key 1, which reports 100 left.
    # 2nd: key 2 (local 800). After that key 2 reports 500 > 100, so it keeps winning.
    assert len(key_1.requests) == 1
    assert len(key_2.requests) == 3
    assert client.stats.per_key["key 2"].requests == 3


def test_requests_alternate_between_keys_without_server_counts(
    llm_settings: dict[str, Any]
) -> None:
    keys = [FakeApi(headers={}), FakeApi(headers={})]  # local counting only
    client = make_multi_key_client(llm_settings, keys, FakeClock())
    for chunk_number in range(4):
        client.complete(INVENTED_MODEL, user_message(f"chunk {chunk_number}"))
    assert [len(api.requests) for api in keys] == [2, 2]


def test_key_at_its_daily_limit_is_skipped(llm_settings: dict[str, Any]) -> None:
    key_1 = FakeApi(headers=quota_headers(remaining_day=0))
    key_2 = FakeApi(headers=quota_headers(remaining_day=500))
    client = make_multi_key_client(llm_settings, [key_1, key_2], FakeClock())
    for chunk_number in range(3):
        client.complete(INVENTED_MODEL, user_message(f"chunk {chunk_number}"))
    assert len(key_1.requests) == 1  # its answer said 0 left: never used again
    assert len(key_2.requests) == 2


def test_key_with_its_hour_used_up_is_skipped(llm_settings: dict[str, Any]) -> None:
    clock = FakeClock()
    key_1 = FakeApi(headers={**quota_headers(remaining_day=700),
                             "x-ratelimit-remaining-requests-hour": "0"})
    key_2 = FakeApi(headers=quota_headers(remaining_day=100))
    client = make_multi_key_client(llm_settings, [key_1, key_2], clock)
    for chunk_number in range(3):
        client.complete(INVENTED_MODEL, user_message(f"chunk {chunk_number}"))
    # key 1 has more left for the day, but its hour is used up: key 2 instead, no waiting.
    assert [len(key_1.requests), len(key_2.requests)] == [1, 2]
    assert clock.now < SECONDS_PER_HOUR


def test_all_keys_at_their_daily_limit_raise_and_name_the_number_of_keys(
    llm_settings: dict[str, Any]
) -> None:
    keys = [FakeApi(headers=quota_headers(remaining_day=0)) for _ in range(2)]
    client = make_multi_key_client(llm_settings, keys, FakeClock())
    client.complete(INVENTED_MODEL, user_message("chunk 1"))
    client.complete(INVENTED_MODEL, user_message("chunk 2"))
    with pytest.raises(DailyLimitReachedError, match="all 2 usable API key"):
        client.complete(INVENTED_MODEL, user_message("chunk 3"))
    assert [len(api.requests) for api in keys] == [1, 1]  # the third request was never sent


def test_rejected_key_is_dropped_and_the_request_goes_to_another_key(
    llm_settings: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    key_1 = FakeApi([403])
    key_2 = FakeApi()
    client = make_multi_key_client(llm_settings, [key_1, key_2], FakeClock())
    with caplog.at_level(logging.WARNING):
        first = client.complete(INVENTED_MODEL, user_message("chunk 1"))
        client.complete(INVENTED_MODEL, user_message("chunk 2"))
    assert first.content == INVENTED_ANSWER
    assert len(key_1.requests) == 1  # never used again after the 403
    assert len(key_2.requests) == 2
    assert client.stats.retries == 0  # moving to another key is not a retry
    assert "key 1 was rejected by the server (HTTP 403)" in caplog.text
    assert "rejected by the server (HTTP 403)" in client.summary()


def test_all_keys_rejected_raise_a_clear_error(llm_settings: dict[str, Any]) -> None:
    keys = [FakeApi([401]), FakeApi([403])]
    client = make_multi_key_client(llm_settings, keys, FakeClock())
    with pytest.raises(NoUsableApiKeyError, match="all 2 API key"):
        client.complete(INVENTED_MODEL, user_message("chunk 1"))


def test_cache_is_shared_across_keys_and_needs_no_key(
    llm_settings: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    first_run = make_multi_key_client(llm_settings, [FakeApi(), FakeApi()], FakeClock())
    first_run.complete(INVENTED_MODEL, user_message("chunk 1"))

    base_name = llm_settings["llm"]["api_key_env"]
    for name in (base_name, f"{base_name}_1"):
        monkeypatch.delenv(name, raising=False)
    # No OpenAI clients and no key in the environment: loading a key would fail.
    cached_run = LLMClient(llm_settings, sleep=FakeClock().sleep)
    assert cached_run.complete(INVENTED_MODEL, user_message("chunk 1")).source == SOURCE_CACHE
    assert cached_run.stats.per_key == {}  # the keys were never loaded


def test_no_key_string_appears_in_any_output(
    llm_settings: dict[str, Any], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    secrets = ["invented-secret-key-one", "invented-secret-key-two"]
    apis = [FakeApi([403]), FakeApi([429])]
    call_log_path = tmp_path / "run" / "llm_calls.jsonl"
    client = LLMClient(
        llm_settings,
        openai_clients=[api.openai_client(secret) for api, secret in zip(apis, secrets)],
        call_log_path=call_log_path,
        sleep=FakeClock().sleep,
    )
    with caplog.at_level(logging.DEBUG):  # also the debug logs of openai and httpx
        client.complete(INVENTED_MODEL, user_message("chunk 1"))  # 403, 429, then success
        client.list_models()
    # The secrets really were sent, so the test would notice if they leaked.
    assert apis[1].requests[0].headers["authorization"] == f"Bearer {secrets[1]}"
    outputs = [caplog.text, client.summary(), json.dumps(client.stats.as_dict()),
               call_log_path.read_text(encoding="utf-8")]
    for secret in secrets:
        for output in outputs:
            assert secret not in output


# --- check-keys ----------------------------------------------------------------------------

def test_check_keys_sends_one_request_per_key_and_reports_each(
    llm_settings: dict[str, Any]
) -> None:
    keys = [FakeApi(headers=quota_headers(remaining_day=700)), FakeApi([403]),
            FakeApi(headers={})]
    client = make_multi_key_client(llm_settings, keys, FakeClock())
    checks = client.check_keys(INVENTED_MODEL)

    assert [len(api.requests) for api in keys] == [1, 1, 1]
    assert json.loads(keys[0].requests[0].content)["messages"] == user_message(CHECK_KEYS_PROMPT)
    assert [check.label for check in checks] == ["key 1", "key 2", "key 3"]
    assert [check.status for check in checks] == [KEY_OK, "HTTP 403", KEY_OK]
    assert checks[0].remaining_quota["x-ratelimit-remaining-requests-day"] == "700"
    assert checks[0].rate_limits == LIMIT_HEADERS
    assert checks[2].remaining_quota == {}  # no headers in that response
    assert "not reported" in format_key_checks(checks)


def test_check_keys_bypasses_the_cache(llm_settings: dict[str, Any]) -> None:
    keys = [FakeApi(), FakeApi()]
    client = make_multi_key_client(llm_settings, keys, FakeClock())
    client.complete(INVENTED_MODEL, user_message(CHECK_KEYS_PROMPT))  # now cached
    client.check_keys(INVENTED_MODEL)
    client.check_keys(INVENTED_MODEL)
    assert sum(len(api.requests) for api in keys) == 1 + 2 * 2
    assert client.stats.cache_hits == 0


def test_check_keys_does_not_retry(llm_settings: dict[str, Any]) -> None:
    key = FakeApi([500])
    checks = make_multi_key_client(llm_settings, [key], FakeClock()).check_keys(INVENTED_MODEL)
    assert checks[0].status == "HTTP 500"
    assert len(key.requests) == 1


def test_check_keys_output_never_contains_a_key(
    llm_settings: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    secrets = ["invented-secret-key-one", "invented-secret-key-two"]
    apis = [FakeApi(), FakeApi([401])]
    client = LLMClient(
        llm_settings,
        openai_clients=[api.openai_client(secret) for api, secret in zip(apis, secrets)],
        sleep=FakeClock().sleep,
    )
    with caplog.at_level(logging.DEBUG):
        output = format_key_checks(client.check_keys(INVENTED_MODEL))
    for secret in secrets:
        assert secret not in output
        assert secret not in caplog.text
    assert "key 2: HTTP 401" in output


def test_models_endpoint_note_explains_missing_quota() -> None:
    assert "/v1/models" in MODELS_ENDPOINT_NOTE
    assert "not reported" in MODELS_ENDPOINT_NOTE
