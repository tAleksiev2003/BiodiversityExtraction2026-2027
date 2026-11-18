"""Parallel requests: they really overlap, and every per-key limit still holds exactly.

A fake slow API (httpx.MockTransport) answers each request after a short real delay, so
several threads have requests in flight at the same time. It records, per key, when each
request arrived and how many were in flight, and it can play a server that counts the
day itself (x-ratelimit-remaining-requests-day) and rejects requests over its limit.
No test reaches the real API.
"""
import copy
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest

import rate_limiter
from llm_cache import ResponseCache
from llm_client import LLMClient, RunStoppedError
from parallel_runner import STOPPED_BY_USER, run_in_parallel
from rate_limiter import DailyLimitReachedError

INVENTED_MODEL = "invented-model"
DELAY_SECONDS = 0.2


def completion_json(content: str) -> dict[str, Any]:
    return {
        "id": "invented-id", "object": "chat.completion", "created": 0, "model": INVENTED_MODEL,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
    }


class SlowFakeApi:
    """One key's server: answers after `delay` seconds and records what it saw.

    `server_day_limit`: the server counts requests itself, reports the rest in the
    x-ratelimit-remaining-requests-day header and answers 429 above the limit.
    `hour_remaining`: a fixed x-ratelimit-remaining-requests-hour value.
    """

    def __init__(self, delay: float = DELAY_SECONDS, server_day_limit: int | None = None,
                 hour_remaining: int | None = None) -> None:
        self.delay = delay
        self.server_day_limit = server_day_limit
        self.hour_remaining = hour_remaining
        self.arrival_times: list[float] = []
        self.rejected_over_limit = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = threading.Lock()

    def handle(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.arrival_times.append(time.monotonic())
            count = len(self.arrival_times)
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        headers = {}
        if self.hour_remaining is not None:
            headers["x-ratelimit-remaining-requests-hour"] = str(self.hour_remaining)
        try:
            time.sleep(self.delay)
            if self.server_day_limit is not None:
                remaining = self.server_day_limit - count
                headers["x-ratelimit-remaining-requests-day"] = str(max(remaining, 0))
                if remaining < 0:
                    with self._lock:
                        self.rejected_over_limit += 1
                    return httpx.Response(429, json={"error": {"message": "over limit"}}, headers=headers)
            return httpx.Response(200, json=completion_json("invented answer"), headers=headers)
        finally:
            with self._lock:
                self.in_flight -= 1

    def openai_client(self) -> openai.OpenAI:
        return openai.OpenAI(api_key="invented-test-key", base_url="https://llm.invalid/v1", max_retries=0,
                             http_client=httpx.Client(transport=httpx.MockTransport(self.handle)))


@pytest.fixture
def parallel_settings(invented_settings: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(invented_settings)
    copied["llm"]["max_concurrent_requests_per_key"] = 4
    copied["llm"]["max_retries"] = 0  # a 429 from the fake server must show up, not be retried
    return copied


def make_client(settings: dict[str, Any], apis: list[SlowFakeApi]) -> LLMClient:
    """A client with real sleeping (the threads must really wait for each other)."""
    return LLMClient(settings, openai_clients=[api.openai_client() for api in apis])


def ask_all(client: LLMClient, prompt_count: int):
    """Send `prompt_count` different prompts through the parallel runner, as the pipelines do."""
    prompts = [f"invented chunk {number:03d}" for number in range(prompt_count)]

    def ask(prompt: str) -> str:
        return client.complete(INVENTED_MODEL, [{"role": "user", "content": prompt}]).source

    return run_in_parallel(prompts, ask, client.max_parallel_requests(),
                           (DailyLimitReachedError,), client.request_stop)


def max_requests_in_any_window(arrival_times: list[float], window_seconds: float) -> int:
    times = sorted(arrival_times)
    return max((sum(1 for other in times if start <= other < start + window_seconds) for start in times),
               default=0)


def test_requests_run_in_parallel_on_all_keys(parallel_settings: dict[str, Any]) -> None:
    apis = [SlowFakeApi(), SlowFakeApi()]
    client = make_client(parallel_settings, apis)
    assert client.max_parallel_requests() == 4 * 2

    started = time.monotonic()
    run = ask_all(client, prompt_count=16)
    elapsed = time.monotonic() - started

    assert run.stop_reason is None
    assert len(run.results) == 16
    assert client.stats.api_requests == 16
    assert sum(api.max_in_flight for api in apis) == 8  # 4 per key at the same time
    assert all(api.max_in_flight <= 4 for api in apis)
    # One after the other: 16 x 0.2 s = 3.2 s; 8 at a time: about 2 x 0.2 s.
    assert elapsed < 16 * DELAY_SECONDS / 2
    assert client.stats.average_request_seconds() == pytest.approx(DELAY_SECONDS, abs=0.1)


def test_short_window_limit_holds_per_key_with_parallel_requests(
    parallel_settings: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The "10-second" window shrunk to 1 s so the test is fast; 2 requests per window.
    window_seconds = 1.0
    monkeypatch.setattr(rate_limiter, "SECONDS_PER_10_SECONDS", window_seconds)
    parallel_settings["llm"]["requests_per_10_seconds"] = 2
    apis = [SlowFakeApi(delay=0.01), SlowFakeApi(delay=0.01)]
    run = ask_all(make_client(parallel_settings, apis), prompt_count=8)

    assert len(run.results) == 8
    assert sum(len(api.arrival_times) for api in apis) == 8
    tolerance = 0.1  # the request reaches the server a little after the limiter let it go
    for api in apis:
        assert max_requests_in_any_window(api.arrival_times, window_seconds - tolerance) <= 2


def test_daily_limit_holds_exactly_with_requests_in_flight(parallel_settings: dict[str, Any]) -> None:
    # The local count would allow 800; the server knows only 5 per key are left today.
    # Its counts in answers lag behind the requests in flight, so the client must
    # subtract the requests it sent after the reported one.
    apis = [SlowFakeApi(server_day_limit=5), SlowFakeApi(server_day_limit=5)]
    client = make_client(parallel_settings, apis)
    run = ask_all(client, prompt_count=30)

    assert run.stop_reason is not None and "Daily limit reached on all 2" in run.stop_reason
    assert [api.rejected_over_limit for api in apis] == [0, 0]  # the server never said no
    assert [len(api.arrival_times) for api in apis] == [5, 5]   # and no request was wasted
    assert len(run.results) == 10
    assert client.stats.api_requests == 10


def test_answers_received_before_the_stop_stay_in_the_cache(parallel_settings: dict[str, Any]) -> None:
    # Only 3 left on the server, fewer than the 4 parallel requests per key: the key sends
    # one request alone until the server has told how many are left, so none is rejected.
    apis = [SlowFakeApi(server_day_limit=3)]
    first_run = ask_all(make_client(parallel_settings, apis), prompt_count=6)
    assert first_run.stop_reason is not None
    assert len(first_run.results) == 3
    assert apis[0].rejected_over_limit == 0

    next_day = SlowFakeApi()
    second_client = make_client(parallel_settings, [next_day])
    second_run = ask_all(second_client, prompt_count=6)
    assert second_run.stop_reason is None
    assert second_client.stats.cache_hits == 3
    assert len(next_day.arrival_times) == 3  # only the 3 prompts that were not answered


def test_request_stop_wakes_a_thread_waiting_for_the_hour(parallel_settings: dict[str, Any]) -> None:
    api = SlowFakeApi(delay=0.0, hour_remaining=0)  # every answer: nothing left this hour
    client = make_client(parallel_settings, [api])
    client.complete(INVENTED_MODEL, [{"role": "user", "content": "first"}])
    errors: list[BaseException] = []

    def second_request() -> None:
        try:
            client.complete(INVENTED_MODEL, [{"role": "user", "content": "second"}])
        except BaseException as error:  # noqa: BLE001 - collected for the assertion
            errors.append(error)

    thread = threading.Thread(target=second_request)
    thread.start()
    time.sleep(0.2)            # now it waits for the hour to free up
    assert thread.is_alive()
    client.request_stop()      # what Ctrl + C does
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], RunStoppedError)
    assert len(api.arrival_times) == 1


# --- parallel runner ---------------------------------------------------------------------

def test_runner_stops_starting_new_documents_after_a_stop_error() -> None:
    started: list[str] = []
    stops: list[bool] = []

    def process(document_id: str) -> str:
        started.append(document_id)
        if document_id == "DOC00002":
            raise DailyLimitReachedError("Daily limit reached on all 1 usable API key(s)")
        return document_id.lower()

    ids = [f"DOC0000{number}" for number in range(1, 6)]
    run = run_in_parallel(ids, process, 1, (DailyLimitReachedError,), lambda: stops.append(True))
    assert run.stop_reason == "Daily limit reached on all 1 usable API key(s)"
    assert started == ["DOC00001", "DOC00002"]
    assert run.results == {"DOC00001": "doc00001"}
    assert stops == [True]


def test_runner_re_raises_unexpected_errors() -> None:
    def process(document_id: str) -> str:
        raise KeyError("a bug")

    with pytest.raises(KeyError):
        run_in_parallel(["DOC00001", "DOC00002"], process, 2, (DailyLimitReachedError,), lambda: None)


def test_runner_handles_ctrl_c_in_the_main_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    import parallel_runner

    def interrupted(_futures):
        raise KeyboardInterrupt
        yield  # a generator, like as_completed

    stops: list[bool] = []
    monkeypatch.setattr(parallel_runner, "as_completed", interrupted)
    run = run_in_parallel(["DOC00001"], lambda document_id: document_id, 1,
                          (DailyLimitReachedError,), lambda: stops.append(True))
    assert run.stop_reason == STOPPED_BY_USER
    assert stops == [True]


# --- cache --------------------------------------------------------------------------------

def test_cache_is_safe_with_many_writing_threads(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")

    def write(thread_number: int) -> None:
        for number in range(20):
            cache.put(f"shared-{number}", INVENTED_MODEL, 0, f"answer {number}")
            cache.put(f"own-{thread_number}-{number}", INVENTED_MODEL, 0, f"answer {thread_number}")
            assert cache.get(f"shared-{number}") == f"answer {number}"

    threads = [threading.Thread(target=write, args=(number,)) for number in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    files = list((tmp_path / "cache").iterdir())
    assert len(files) == 20 + 8 * 20
    assert not [path for path in files if path.suffix == ".tmp"]
