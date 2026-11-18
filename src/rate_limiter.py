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

SAIA counts these limits per Academic Cloud account, not per API key (SPEC.md, section 5):
several keys of one account share one quota. The client still keeps one limiter per key
(src/api_key_pool.py) and stops only when every key has reached its daily limit. With
keys of different accounts the local counts are right; with keys of one account only the
server counts (which are the account's) are right, and they win as soon as a response
reports them. The server's hour refills gradually, not at the full hour, so waiting one
hour after a report of 0 left is safe but can be longer than needed.

Parallel requests (several threads share one limiter):
- checking the windows and recording a request happen under one lock, so two threads
  can never both take the last free place in a window; the waiting happens outside the
  lock, so other threads can still ask the limiter how much room it has;
- with requests in flight, a server count is already out of date when it arrives, and
  answers can arrive in another order than the requests were sent. Every recorded
  request therefore gets a ticket number. A count in the answer to ticket k surely
  includes every request that was answered before k was sent. It may miss the others:
  the requests still unanswered when k was sent, and all requests sent after k. The
  limiter subtracts all of them (`_ServerCount`). That may count a request twice, but
  never too few, so the limits hold exactly. Of several counts, the one that leaves the
  most requests is kept, since each one alone is a safe lower bound.
"""
import threading
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


@dataclass(frozen=True)
class _ServerCount:
    """One remaining count from the server, with what it may not include yet.

    `remaining`: the header value; `ticket`: the request whose answer carried it;
    `unanswered_when_sent`: requests still in flight when that request was sent.
    """

    remaining: int
    ticket: int
    unanswered_when_sent: int
    reported_at: float

    def remaining_now(self, tickets_issued: int) -> int:
        """Requests surely left: the count minus everything it may have missed."""
        sent_after = tickets_issued - self.ticket
        return self.remaining - self.unanswered_when_sent - sent_after


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
        self._lock = threading.Lock()
        self._sent_at: deque[float] = deque()  # send times of the last 24 hours, oldest first
        self._tickets_issued = 0  # requests recorded so far; the newest request's ticket
        # Ticket -> number of other requests in flight when it was sent; removed on its answer.
        self._unanswered: dict[int, int] = {}
        # The server's counts (None = not reported in the last answer: count locally).
        self._server_hour: _ServerCount | None = None
        self._server_day: _ServerCount | None = None
        self.has_answer = False  # True after the first answer (with or without headers)

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

    # --- answers ---

    def update_server_remaining(
        self, remaining_hour: int | None, remaining_day: int | None, ticket: int | None = None
    ) -> None:
        """Record the server's remaining counts from an answer (None = header missing).

        `ticket` is the number `reserve_slot` gave the request of this answer. Without a
        ticket (one request at a time) the answer belongs to the newest request and
        nothing was in flight.
        """
        with self._lock:
            if ticket is None:
                ticket, unanswered_when_sent = self._tickets_issued, 0
            else:
                unanswered_when_sent = self._unanswered.pop(ticket, 0)
            now = self._clock()
            self._server_hour = self._better_count(self._server_hour, remaining_hour, ticket,
                                                   unanswered_when_sent, now)
            self._server_day = self._better_count(self._server_day, remaining_day, ticket,
                                                  unanswered_when_sent, now)
            self.has_answer = True

    def request_ended_without_answer(self, ticket: int) -> None:
        """A request failed before an answer came (e.g. connection error)."""
        with self._lock:
            self._unanswered.pop(ticket, None)

    def _better_count(
        self, current: _ServerCount | None, remaining: int | None, ticket: int,
        unanswered_when_sent: int, now: float,
    ) -> _ServerCount | None:
        """The new count, unless the current one leaves more requests (both are safe)."""
        if remaining is None:
            return None  # this answer has no header: count locally again
        new = _ServerCount(remaining, ticket, unanswered_when_sent, now)
        if current is not None and current.remaining_now(self._tickets_issued) > new.remaining_now(self._tickets_issued):
            return current
        return new

    # --- windows (called with the lock held) ---

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
        if self._server_day is not None:
            return self._server_day.remaining_now(self._tickets_issued)
        return self.daily_window.max_requests - len(self._requests_in(self.daily_window, now))

    def _seconds_until_hour_has_room(self, now: float) -> float:
        if self._server_hour is None:
            return self._seconds_until_room(self.hour_window, now)
        if self._server_hour.remaining_now(self._tickets_issued) > 0:
            return 0.0
        # We do not know when the server's oldest request leaves its window. One hour
        # after the report is safe for both sliding and fixed hour windows.
        return self._server_hour.reported_at + self.hour_window.seconds - now

    def _seconds_until_slot(self, now: float) -> float:
        # Waiting for one window never fills another, so we wait for the slowest one.
        return max(
            self._seconds_until_room(self.ten_second_window, now),
            self._seconds_until_hour_has_room(now),
        )

    def _raise_if_day_used_up(self, now: float) -> None:
        if self._remaining_today(now) <= 0:
            raise DailyLimitReachedError(
                f"Daily limit of {self.daily_window.max_requests} requests reached. "
                "All answers so far are in the cache: run the same command again tomorrow "
                "to continue where this run stopped."
            )

    def _record_request(self, now: float) -> int:
        self._sent_at.append(now)
        self._tickets_issued += 1
        self._unanswered[self._tickets_issued] = len(self._unanswered)
        return self._tickets_issued

    # --- read-only queries, used to choose between several API keys (src/api_key_pool.py) ---

    def remaining_today(self) -> int:
        """Requests left for the day: the server's count if reported, else the local count."""
        with self._lock:
            now = self._clock()
            self._forget_old_requests(now)
            return self._remaining_today(now)

    def seconds_until_hour_has_room(self) -> float:
        """0 if the hour window has room, else how long until it has."""
        with self._lock:
            now = self._clock()
            self._forget_old_requests(now)
            return max(0.0, self._seconds_until_hour_has_room(now))

    def seconds_until_slot(self) -> float:
        """0 if a request could be sent right now, else how long until one can."""
        with self._lock:
            now = self._clock()
            self._forget_old_requests(now)
            return max(0.0, self._seconds_until_slot(now))

    # --- recording a request ---

    def try_reserve(self) -> int | None:
        """Record a request if every window has room right now; return its ticket.

        Returns None instead of waiting, so the key pool can choose a key and reserve
        its slot in one step. Raises DailyLimitReachedError if the day is used up.
        """
        with self._lock:
            now = self._clock()
            self._forget_old_requests(now)
            self._raise_if_day_used_up(now)
            if self._seconds_until_slot(now) > 0:
                return None
            return self._record_request(now)

    def reserve_slot(self) -> int:
        """Block until every window allows one more request, record it, return its ticket."""
        _waited, ticket = self._wait_and_record()
        return ticket

    def wait_for_slot(self) -> float:
        """Block until every window allows one more request, record it, return the seconds waited.

        Raises DailyLimitReachedError instead of waiting when the daily window is full.
        """
        waited, _ticket = self._wait_and_record()
        return waited

    def _wait_and_record(self) -> tuple[float, int]:
        """Check and record under the lock, sleep outside it (see the module docstring)."""
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                self._forget_old_requests(now)
                self._raise_if_day_used_up(now)
                wait_seconds = self._seconds_until_slot(now)
                if wait_seconds <= 0:
                    return waited, self._record_request(now)
            # Another thread may take the free place while we sleep; the loop checks again.
            self._sleep(wait_seconds)
            waited += wait_seconds
