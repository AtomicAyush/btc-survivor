"""Feed health. A minute counts only when the feed was fully trustworthy for all of it.

The feed is Healthy, Recovering or Stale. It goes Stale when messages or good book updates stop for too long,
and Recovering after any reconnect, lost book or clock skew. It returns to Healthy at the first minute boundary
that closes a full minute in which everything held. Every exit from Healthy bumps the epoch, which labels use to
refuse to span a gap.
"""

from __future__ import annotations

import statistics
from collections import deque

from btc_survivor.clock import MS_PER_MINUTE

HEALTHY = "healthy"
RECOVERING = "recovering"
STALE = "stale"

SKEW_SAMPLES = 50
SKEW_MIN_SAMPLES = 10


class HealthMonitor:
    def __init__(self, stale_after_ms: int, max_skew_ms: int) -> None:
        self.stale_after_ms = stale_after_ms
        self.max_skew_ms = max_skew_ms
        self.state = STALE
        self.epoch = 0
        self.connected = False
        self.book_ok = False
        self.skew_ok = True
        self.last_msg_ms: int | None = None
        self.last_good_book_ms: int | None = None
        self.good_since: int | None = None
        self.events: list[tuple[int, str, str]] = []
        self._skews: deque[int] = deque(maxlen=SKEW_SAMPLES)
        self._last_sample: tuple[int, int] | None = None  # events from one message share it, so it counts once

    @property
    def is_healthy(self) -> bool:
        return self.state == HEALTHY

    def event(self, ms: int, kind: str, detail: str = "") -> None:
        self.events.append((ms, kind, detail))

    def drain_events(self) -> list[tuple[int, str, str]]:
        events, self.events = self.events, []
        return events

    def _set_state(self, ms: int, state: str, detail: str = "") -> None:
        if state == self.state:
            return
        if self.state == HEALTHY:
            self.epoch += 1
        self.state = state
        self.event(ms, state, detail)

    def _lose(self, ms: int, state: str, detail: str) -> None:
        """Something broke, so the clean stretch starts over."""
        self.good_since = None
        self._set_state(ms, state, detail)

    def _conditions_ok(self) -> bool:
        return self.connected and self.book_ok and self.skew_ok

    def _maybe_start_good(self, ms: int) -> None:
        if self.good_since is None and self._conditions_ok():
            self.good_since = ms
            if self.state == STALE:
                self._set_state(ms, RECOVERING, "fresh data")

    def on_connected(self, ms: int) -> None:
        """A new connection gets the full stale limit to deliver its snapshot."""
        self.connected = True
        self.book_ok = False
        self.last_msg_ms = self.last_good_book_ms = ms
        self._skews.clear()
        self.skew_ok = True
        self.event(ms, "connected")
        self._lose(ms, RECOVERING if self.state != STALE else STALE, "connected, waiting for a snapshot")

    def on_disconnected(self, ms: int, detail: str = "") -> None:
        self.connected = False
        self.book_ok = False
        self.event(ms, "disconnected", detail)
        self._lose(ms, RECOVERING if self.state != STALE else STALE, "disconnected")

    def on_message(self, ms: int, exch_ms: int | None) -> None:
        """Any message proves the socket is alive. Its exchange timestamp feeds the clock skew check."""
        self.last_msg_ms = ms
        if exch_ms is None or (ms, exch_ms) == self._last_sample:
            return
        self._last_sample = (ms, exch_ms)
        self._skews.append(ms - exch_ms)
        if len(self._skews) < SKEW_MIN_SAMPLES:
            return
        skew = statistics.median(self._skews)
        ok = abs(skew) <= self.max_skew_ms
        if ok != self.skew_ok:
            self.skew_ok = ok
            if ok:
                self.event(ms, "clock_skew", f"cleared, median {skew:.0f} ms")
                self._maybe_start_good(ms)
            else:
                self.event(ms, "clock_skew", f"median {skew:.0f} ms")
                self._lose(ms, RECOVERING if self.state != STALE else STALE, "clock skew")

    def on_snapshot(self, ms: int, valid: bool, detail: str = "", problem: str = "crossed") -> None:
        self.event(ms, "snapshot", detail)
        if valid:
            self.book_ok = True
            self.last_good_book_ms = ms
            self._maybe_start_good(ms)
        else:
            self.on_book_lost(ms, problem, "snapshot arrived unusable")

    def on_book_update(self, ms: int) -> None:
        if self.book_ok:
            self.last_good_book_ms = ms

    def on_book_lost(self, ms: int, kind: str, detail: str = "") -> None:
        """A gap or a crossed book. The book is thrown away and a resync follows."""
        self.book_ok = False
        self.event(ms, kind, detail)
        self._lose(ms, RECOVERING if self.state != STALE else STALE, kind)

    def check(self, now_ms: int) -> bool:
        """Go Stale when messages or good book updates have stopped for longer than the limit. True means stuck."""
        silent = self.last_msg_ms is None or now_ms - self.last_msg_ms > self.stale_after_ms
        no_book = self.last_good_book_ms is None or now_ms - self.last_good_book_ms > self.stale_after_ms
        if silent or no_book:
            if self.state != STALE:
                self._lose(now_ms, STALE, "no messages" if silent else "no good book")
            return True
        return False

    def judge_minute(self, boundary_ms: int) -> bool:
        """Whether the minute ending at boundary_ms counts. The full clean minute that completes a recovery counts."""
        self.check(boundary_ms)
        healthy = (self.state != STALE and self._conditions_ok() and self.good_since is not None
                   and self.good_since <= boundary_ms - MS_PER_MINUTE)
        if healthy:
            self._set_state(boundary_ms, HEALTHY, "a full clean minute")
        elif self.state == HEALTHY:
            self._lose(boundary_ms, RECOVERING, "minute not clean")
        return healthy
