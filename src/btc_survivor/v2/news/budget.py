"""The REST budget of PLAN_V2 section 7.2, with rolling limits, pauses and failure backoff per host on an injected
clock.

Each host group, Finnhub, SEC across sec.gov and data.sec.gov, the Federal Reserve and BLS, keeps the times of its
recent calls, and a call goes only once every rolling window has room and the minimum spacing has passed, so
Finnhub never makes more than 30 calls in any 60 seconds. A 429 pauses every Finnhub channel, and a 403 or 429 pauses
every channel of an official host. Times are integer milliseconds, so a wait ends exactly when its window frees.
There are no retries here, a failed poll only comes due later.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from btc_survivor.v2.config import NewsConfig

FINNHUB, SEC, FED, BLS = "finnhub", "sec", "fed", "bls"
GROUPS = (FINNHUB, SEC, FED, BLS)
SECOND_MS, MINUTE_MS = 1000, 60_000
FORBIDDEN, TOO_MANY_REQUESTS = 403, 429
MAX_DOUBLINGS = 30               # far past any cap, and 2 to this power stays an exact float


@dataclass(frozen=True)
class Limit:
    """At most calls in any window_ms."""

    calls: int
    window_ms: int


@dataclass(frozen=True)
class HostRules:
    """The rolling limits, minimum spacing and pause of one host group, and the HTTP statuses that pause it."""

    limits: tuple[Limit, ...]
    min_spacing_ms: int
    pause_ms: int
    pause_on: frozenset[int]


def _ms(seconds: float) -> int:
    """Seconds from the config as whole milliseconds."""
    return round(seconds * 1000)


def rules_from_config(config: NewsConfig) -> dict[str, HostRules]:
    """The limits of section 7.2 from the news config. Finnhub pauses only on a 429, the official hosts on a 403 or
    a 429."""
    finnhub, edgar, fed, bls = config.finnhub, config.edgar, config.fed, config.bls
    blocked = frozenset({FORBIDDEN, TOO_MANY_REQUESTS})
    sec_limits = (Limit(edgar.max_requests_per_second, SECOND_MS), Limit(edgar.max_requests_per_minute, MINUTE_MS))
    return {
        FINNHUB: HostRules((Limit(finnhub.max_calls_per_minute, MINUTE_MS),), _ms(finnhub.min_call_spacing_seconds),
                           _ms(finnhub.rate_limit_pause_seconds), frozenset({TOO_MANY_REQUESTS})),
        SEC: HostRules(sec_limits, 0, _ms(edgar.block_pause_seconds), blocked),
        FED: HostRules((Limit(fed.max_requests_per_minute, MINUTE_MS),), 0, _ms(fed.block_pause_seconds), blocked),
        BLS: HostRules((Limit(bls.max_requests_per_minute, MINUTE_MS),), 0, _ms(bls.block_pause_seconds), blocked),
    }


def monotonic_ms() -> int:
    """The default clock, monotonic milliseconds that no wall clock change can move."""
    return time.monotonic_ns() // 1_000_000


def failure_delay_s(cadence_s: float, failures: int, cap_s: float) -> float:
    """When a channel polls again after failures consecutive failed polls, the cadence times 2 to the power of the
    failures capped at cap_s, never sooner than the cadence itself, and with no jitter."""
    if failures <= 0:
        return cadence_s
    return max(cadence_s, min(cadence_s * 2.0 ** min(failures, MAX_DOUBLINGS), cap_s))


class Budget:
    """Holds every host group to its rules. One instance serves every channel task on the event loop, and each check
    and its record run with no await between them."""

    def __init__(self, rules: dict[str, HostRules], clock_ms: Callable[[], int] = monotonic_ms,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self.rules = dict(rules)
        self._calls = {group: deque(maxlen=max([1, *(limit.calls for limit in rule.limits)]))
                       for group, rule in self.rules.items()}
        self._paused_until = dict.fromkeys(self.rules, 0)
        self._clock_ms = clock_ms
        self._sleep = sleep

    def _wait(self, group: str, now: int) -> int:
        """The wait at now, until the pause ends, the spacing passes and every window has room."""
        rule, calls = self.rules[group], self._calls[group]
        until = [self._paused_until[group]]
        if calls:
            until.append(calls[-1] + rule.min_spacing_ms)
        until += [calls[-limit.calls] + limit.window_ms for limit in rule.limits if len(calls) >= limit.calls]
        return max(0, max(until) - now)

    def wait_ms(self, group: str) -> int:
        """How long a call to group must wait from now, 0 when it may go at once."""
        return self._wait(group, self._clock_ms())

    def try_acquire(self, group: str) -> int:
        """Record a call to group and return 0 when it may go now, or return the wait and record nothing."""
        now = self._clock_ms()
        wait = self._wait(group, now)
        if wait == 0:
            self._calls[group].append(now)
        return wait

    async def acquire(self, group: str) -> int:
        """Wait until a call to group may go, record it and return the milliseconds waited, which the collector logs
        as budget_wait when above zero."""
        start = self._clock_ms()
        while (wait := self.try_acquire(group)) > 0:
            await self._sleep(wait / 1000)
        return self._clock_ms() - start

    def report(self, group: str, http_status: int | None) -> bool:
        """Pause the whole group after a status its rules pause on, and say whether it did."""
        rule = self.rules[group]
        if http_status not in rule.pause_on:
            return False
        self._paused_until[group] = max(self._paused_until[group], self._clock_ms() + rule.pause_ms)
        return True

    def paused_ms(self, group: str) -> int:
        """How much longer group stays paused, 0 when it is not paused."""
        return max(0, self._paused_until[group] - self._clock_ms())
