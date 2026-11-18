"""Client-side rate limiter: wait instead of being rejected by the server.

SAIA limits requests in three sliding windows at the same time (SPEC.md, section 5):
`llm.requests_per_10_seconds`, `llm.requests_per_hour` and `llm.requests_per_day`.
Sending faster only produces 429 errors, and each rejected request may still count
against the quota. So before every request we wait until every window has room.

The daily window is different: when it is full, the next free slot can be hours away.
Instead of blocking that long, the limiter raises DailyLimitReachedError and the run
stops. Every answer received so far is in the disk cache, so running the same command
the next day continues where the run stopped.

Local counting only knows the requests of its own process. So after every response the
client passes the server's own remaining counts for the hour and the day
(`update_server_remaining`), and these win over the local count: a run restarted on the
same day stops at once if the server reports 0 left for the day, and waits if it reports
0 left for the hour. If a response lacks a header, that window falls back to local
counting. The 10-second window is always counted locally.

SAIA counts these limits per API key, so the client keeps one limiter per key
(src/api_key_pool.py) and stops only when every key has reached its daily limit.
"""
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping

SECONDS_PER_10_SECONDS = 10.0
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class Window:
    """At most `max_requests` requests in any `seconds` seconds."""

    seconds: float
    max_requests: int


class DailyLimitReachedError(RuntimeError):
    """The daily window is full; the run must stop and continue the next day."""


class RateLimiter:
    """Sliding windows: short windows make the caller wait, the daily window stops it."""

    def __init__(
        self,
        requests_per_10_seconds: int,
        requests_per_hour: int,
        requests_per_day: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        # clock and sleep can be replaced in tests, so the tests do not really wait.
        self.ten_second_window = Window(SECONDS_PER_10_SECONDS, requests_per_10_seconds)
        self.hour_window = Window(SECONDS_PER_HOUR, requests_per_hour)
        self.daily_window = Window(SECONDS_PER_DAY, requests_per_day)
        self._clock = clock
        self._sleep = sleep
        self._sent_at: deque[float] = deque()  # send times of the last 24 hours, oldest first
        # Remaining requests the server reported in the last response; None = not reported.
        self._server_remaining_hour: int | None = None
        self._server_remaining_day: int | None = None
        self._server_reported_at = 0.0

    @classmethod
    def from_settings(
        cls,
        llm_settings: Mapping[str, Any],
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> "RateLimiter":
        """Build the limiter from the `llm` section of settings.yaml."""
        return cls(
            llm_settings["requests_per_10_seconds"],
            llm_settings["requests_per_hour"],
            llm_settings["requests_per_day"],
            clock=clock,
            sleep=sleep,
        )

    def update_server_remaining(self, remaining_hour: int | None, remaining_day: int | None) -> None:
        """Record the server's remaining counts from the latest response (None = header missing)."""
        self._server_remaining_hour = remaining_hour
        self._server_remaining_day = remaining_day
        self._server_reported_at = self._clock()

    def _forget_old_requests(self, now: float) -> None:
        # The daily window is the longest one, so older requests matter to no window.
        while self._sent_at and now - self._sent_at[0] >= self.daily_window.seconds:
            self._sent_at.popleft()

    def _requests_in(self, window: Window, now: float) -> list[float]:
        return [sent for sent in self._sent_at if now - sent < window.seconds]

    def _seconds_until_room(self, window: Window, now: float) -> float:
        """0 if the window has room; otherwise the time until enough requests leave it."""
        in_window = self._requests_in(window, now)
        if len(in_window) < window.max_requests:
            return 0.0
        # After this request leaves the window, it holds max_requests - 1 requests.
        oldest_to_leave = in_window[len(in_window) - window.max_requests]
        return oldest_to_leave + window.seconds - now

    def _remaining_today(self, now: float) -> int:
        if self._server_remaining_day is not None:
            return self._server_remaining_day
        return self.daily_window.max_requests - len(self._requests_in(self.daily_window, now))

    def _seconds_until_hour_has_room(self, now: float) -> float:
        if self._server_remaining_hour is None:
            return self._seconds_until_room(self.hour_window, now)
        if self._server_remaining_hour > 0:
            return 0.0
        # We do not know when the server's oldest request leaves its window. One hour
        # after the report is safe for both sliding and fixed hour windows.
        return self._server_reported_at + self.hour_window.seconds - now

    # --- read-only queries, used to choose between several API keys (src/api_key_pool.py) ---

    def remaining_today(self) -> int:
        """Requests left for the day: the server's count if reported, else the local count."""
        now = self._clock()
        self._forget_old_requests(now)
        return self._remaining_today(now)

    def seconds_until_hour_has_room(self) -> float:
        """0 if the hour window has room, else how long until it has."""
        now = self._clock()
        self._forget_old_requests(now)
        return max(0.0, self._seconds_until_hour_has_room(now))

    def wait_for_slot(self) -> float:
        """Block until every window allows one more request, record it, return the seconds waited.

        Raises DailyLimitReachedError instead of waiting when the daily window is full.
        """
        now = self._clock()
        self._forget_old_requests(now)
        if self._remaining_today(now) <= 0:
            raise DailyLimitReachedError(
                f"Daily limit of {self.daily_window.max_requests} requests reached. "
                "All answers so far are in the cache: run the same command again tomorrow "
                "to continue where this run stopped."
            )
        waited = 0.0
        while True:
            # Waiting for one window never fills another, so we wait for the slowest one.
            wait_seconds = max(
                self._seconds_until_room(self.ten_second_window, now),
                self._seconds_until_hour_has_room(now),
            )
            if wait_seconds <= 0:
                break
            self._sleep(wait_seconds)
            waited += wait_seconds
            now = self._clock()
            self._forget_old_requests(now)
        self._sent_at.append(now)
        return waited
