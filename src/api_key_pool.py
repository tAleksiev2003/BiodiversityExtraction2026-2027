"""Several SAIA API keys, each with its own rate limiter (SPEC.md, section 5).

SAIA applies its limits (10 per 10 s, 400 per hour, 800 per day) per Academic Cloud
account, not per key: several keys of one account share one quota (measured 2026-11-18,
SPEC.md, section 5). Only keys of n different accounts give n times as many requests per
day. Every key has its own `RateLimiter` and its own server counts (x-ratelimit-remaining-*
headers); for keys of one account the server counts are the shared ones, so they keep
the pool from sending more than the account allows once each key has had an answer.

Keys come from .env as SAIA_API_KEY_1, SAIA_API_KEY_2, ... (consecutive numbers). If no
numbered key exists, the old single SAIA_API_KEY is used as key 1.

A key is never printed, logged or stored: logs, summaries and run files only name its
number ("key 2"). The key string lives only inside its OpenAI client.

Several threads send requests at the same time:
- choosing a key and reserving a place in its rate limiter happen in one locked step
  (`acquire`), otherwise two threads could both see the same key as the best one before
  either has used it;
- each key has at most `max_in_flight_per_key` requests in flight (sent or waiting for
  its limiter); a thread that finds every key busy waits until one is released
  (`release`, after the answer);
- until a key's first answer has arrived, it has only one request in flight: before
  that the server's count for the day is unknown, and a run restarted shortly before the
  daily limit would otherwise send several requests the server rejects.
"""
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import openai
from dotenv import load_dotenv

from rate_limiter import DailyLimitReachedError, RateLimiter

logger = logging.getLogger(__name__)


class MissingApiKeyError(RuntimeError):
    """Neither numbered keys nor the single key variable are set."""


class NoUsableApiKeyError(RuntimeError):
    """The server rejected every key (HTTP 401/403); waiting does not help."""


def load_api_keys(base_name: str, env_file: Path) -> list[str]:
    """Keys from `<base_name>_1`, `<base_name>_2`, ... or, if there are none, `<base_name>`.

    An environment variable that is already set wins over .env (python-dotenv default).
    Numbering must be consecutive: counting stops at the first missing number, and later
    numbered variables are reported (by name only) so a gap is not silently ignored.
    Error and warning messages name only variables, never values.
    """
    load_dotenv(env_file)
    keys: list[str] = []
    while True:
        value = os.environ.get(f"{base_name}_{len(keys) + 1}", "").strip()
        if not value:
            break
        keys.append(value)

    numbered_pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)$")
    skipped = sorted(
        name for name in os.environ
        if (match := numbered_pattern.match(name)) and int(match.group(1)) > len(keys)
        and os.environ[name].strip()
    )
    if skipped:
        logger.warning(
            "Ignored %s: key numbers must be consecutive from %s_1 (only %d found)",
            ", ".join(skipped), base_name, len(keys),
        )

    if not keys:
        single_key = os.environ.get(base_name, "").strip()
        if single_key:
            keys.append(single_key)
    if not keys:
        raise MissingApiKeyError(
            f"Neither {base_name}_1 nor {base_name} is set. Copy .env.example to .env "
            "and enter your SAIA key(s)."
        )
    return keys


@dataclass
class KeyStats:
    """Counters of one key; safe to log and to write into the run folder (no key string)."""

    requests: int = 0                # requests sent with this key, retries included
    rejected: str | None = None      # e.g. "HTTP 403" once the server refused the key
    remaining_quota: dict[str, str] = field(default_factory=dict)  # last x-ratelimit-remaining-*
    rate_limits: dict[str, str] = field(default_factory=dict)      # last x-ratelimit-limit-*


@dataclass
class ApiKeySlot:
    """One key: its client (which holds the key string), its limiter and its counters."""

    number: int
    openai_client: openai.OpenAI
    rate_limiter: RateLimiter
    stats: KeyStats = field(default_factory=KeyStats)
    in_flight: int = 0  # requests of this key between acquire() and release()

    @property
    def label(self) -> str:
        """The only way a key appears in logs and summaries."""
        return f"key {self.number}"

    @property
    def usable(self) -> bool:
        return self.stats.rejected is None


class ApiKeyPool:
    """Chooses the key for each request and drops keys the server rejects."""

    def __init__(
        self,
        openai_clients: Sequence[openai.OpenAI],
        make_rate_limiter: Callable[[], RateLimiter],
        max_in_flight_per_key: int,
    ) -> None:
        """One slot per client; key numbers start at 1 in the order of the clients."""
        self.slots = [
            ApiKeySlot(number, client, make_rate_limiter())
            for number, client in enumerate(openai_clients, start=1)
        ]
        self.max_in_flight_per_key = max_in_flight_per_key
        self._lock = threading.Lock()
        self._key_released = threading.Condition(self._lock)

    def acquire(self) -> tuple[ApiKeySlot, int]:
        """Choose a key, reserve one request in its limiter; return the key and the ticket.

        Call `release(slot)` when the request is done. If the chosen key has room now,
        choosing and reserving happen under one lock, so the next thread already sees
        this request. If it has to wait (every key's hour or 10 seconds are full), the
        waiting happens outside the lock. A key whose day runs out while we wait is left
        and another key is chosen. Raises DailyLimitReachedError when every key is used
        up for the day, NoUsableApiKeyError when every key was rejected.
        """
        while True:
            with self._key_released:
                slot = self.choose()
                while slot is None:  # every key has max_in_flight_per_key requests in flight
                    self._key_released.wait()
                    slot = self.choose()
                slot.in_flight += 1
                ticket = slot.rate_limiter.try_reserve()
            if ticket is not None:
                return slot, ticket
            try:
                return slot, slot.rate_limiter.reserve_slot()
            except DailyLimitReachedError:
                self.release(slot)  # choose() now skips this key
            except BaseException:
                self.release(slot)  # e.g. RunStoppedError while waiting
                raise

    def release(self, slot: ApiKeySlot) -> None:
        """The request of `acquire` is done: the key can take another one."""
        with self._key_released:
            slot.in_flight -= 1
            self._key_released.notify_all()

    def choose(self) -> ApiKeySlot | None:
        """The key to use next; None if every key that could be used is busy.

        Keys that were rejected or whose day is used up are skipped (raises if none is
        left). Among the keys that are not busy, the ones that can send right now win,
        and among those the key with the most requests left today. If none can send now
        (10 seconds or hour full), the key that has room first is returned and its
        limiter waits. Ties go to the lowest key number.
        """
        usable = [slot for slot in self.slots if slot.usable]
        if not usable:
            raise NoUsableApiKeyError(
                f"The server rejected all {len(self.slots)} API key(s) (HTTP 401/403). "
                "Check the keys in .env."
            )
        with_day_left = [slot for slot in usable if slot.rate_limiter.remaining_today() > 0]
        if not with_day_left:
            raise DailyLimitReachedError(self._daily_limit_message(len(usable)))
        not_busy = [slot for slot in with_day_left if slot.in_flight < self._in_flight_limit(slot)]
        if not not_busy:
            return None
        ready_now = [slot for slot in not_busy if slot.rate_limiter.seconds_until_slot() == 0]
        if ready_now:
            # max() keeps the first of equal values, i.e. the lowest key number.
            return max(ready_now, key=lambda slot: slot.rate_limiter.remaining_today())
        return min(not_busy, key=lambda slot: (slot.rate_limiter.seconds_until_slot(), slot.in_flight))

    def _in_flight_limit(self, slot: ApiKeySlot) -> int:
        """One request until the key's first answer (see the module docstring)."""
        return self.max_in_flight_per_key if slot.rate_limiter.has_answer else 1

    def reject(self, slot: ApiKeySlot, reason: str) -> None:
        """Stop using a key for the rest of this run (invalid or blocked key)."""
        with self._lock:
            slot.stats.rejected = reason
        logger.warning(
            "%s was rejected by the server (%s); not used for the rest of this run",
            slot.label, reason,
        )

    def _daily_limit_message(self, usable_count: int) -> str:
        rejected_count = len(self.slots) - usable_count
        rejected_note = f" ({rejected_count} more rejected by the server)" if rejected_count else ""
        return (
            f"Daily limit reached on all {usable_count} usable API key(s){rejected_note}. "
            "All answers so far are in the cache: run the same command again tomorrow "
            "to continue where this run stopped."
        )
