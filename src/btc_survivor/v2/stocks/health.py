"""Price health of the stock feed without an order book, by PLAN_V2 section 4.5, the reconnect, resubscribe and backoff
decisions of section 4.6 as pure functions, and the news health of one minute by section 7.8.

Outside the regular session every symbol is closed and nothing is judged. Only the whole feed connected, disconnected
and error events are written then, so disconnects can be counted by reason around the clock, and never a state change, a
stale event or a reconnect request. In session a symbol is stale while any condition fails, recovering while every
condition holds, and healthy from the first boundary closing a full minute in which every condition held. The whole feed
conditions are a connected socket, a message within feed_stale_after_seconds counted from the later of the last message
and the open, and a median delay of the last 50 non replayed session prints within max_print_delay_seconds either way
once 10 samples exist. The symbol condition is an eligible print within symbol_stale_after_seconds counted from the
later of its last eligible print and the open. A silence starts at the instant its limit ran out, not at the call that
noticed it, so live and replay give the same states, epochs and events whatever the ticker does. Every exit from healthy
bumps the symbol's epoch and every close bumps every epoch.

The reconnect rules match the inline copy in scripts/v2_record_stocks.py. The one difference is the backoff reset in
session, which waits here for a healthy symbol, where the recorder has no price health and resets after 60 seconds.
"""

from __future__ import annotations

import statistics
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import NamedTuple

from btc_survivor.clock import MS_PER_MINUTE, MS_PER_SECOND, iso
from btc_survivor.types import FeedNotice
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import StockFeedConfig
from btc_survivor.v2.sessions import Session
from btc_survivor.v2.stocks.finnhub import PrintBatch
from btc_survivor.v2.stocks.prints import ELIGIBLE, PriceRules, age_ms, eligibility, in_session, is_replayed

CLOSED, STALE, RECOVERING, HEALTHY = "closed", "stale", "recovering", "healthy"
# Whole feed event kinds, written with no symbol. State changes are written per symbol with the new state as kind.
SESSION_OPEN, SESSION_CLOSE, PRINT_DELAY = "session_open", "session_close", "print_delay"
CONNECTED, DISCONNECTED, ERROR = "connected", "disconnected", "error"

DELAY_SAMPLES = 50
DELAY_MIN_SAMPLES = 10
# The delay rule run once on the first recorded session.
DELAY_FLOOR_SECONDS = 2
DELAY_CAP_SECONDS = 5
DELAY_STOP_MS = 5 * MS_PER_SECOND

# The reconnect rules of section 4.6, the same values as scripts/v2_record_stocks.py.
PRE_OPEN_CHECK_MS = 10 * MS_PER_MINUTE
PRE_OPEN_SILENCE_MS = 5 * MS_PER_MINUTE
RESUBSCRIBE_AFTER_MS = 120 * MS_PER_SECOND
RESUBSCRIBE_EVERY_MS = 5 * MS_PER_MINUTE

# Why a minute is not news healthy, for the reason column of news_minutes.
NO_HEARTBEAT, OLD_HEARTBEAT = "no_heartbeat", "heartbeat_old"
UNHEALTHY_HEARTBEAT, HASH_MISMATCH = "heartbeat_unhealthy", "hash_mismatch"


class HealthEvent(NamedTuple):
    """One feed_events row, with symbol None for a whole feed event."""

    ms: int
    kind: str
    symbol: str | None
    detail: str


@dataclass
class SymbolHealth:
    """One symbol's state, epoch, last eligible session print, freshness and the start of its clean stretch."""

    state: str = CLOSED
    epoch: int = 0
    last_eligible_ms: int | None = None
    fresh: bool = True
    good_since: int | None = None


class PriceHealth:
    """The price health of every symbol, fed every feed line in stamp order and asked to judge each session boundary
    before anything stamped at or after it, as the minute step does."""

    def __init__(self, symbols: Sequence[str], feed_stale_ms: int, symbol_stale_ms: int, max_delay_ms: int,
                 rules: PriceRules) -> None:
        self.symbols = {symbol: SymbolHealth() for symbol in symbols}
        self.feed_stale_ms = feed_stale_ms
        self.symbol_stale_ms = symbol_stale_ms
        self.max_delay_ms = max_delay_ms
        self.rules = rules
        self.session: Session | None = None
        self.entered_ms: int | None = None
        self.connected = False
        self.connected_ms: int | None = None
        self.last_msg_ms: int | None = None
        self.silent = False
        self.delay_ok = True
        self.printed: set[str] = set()
        self.events: list[HealthEvent] = []
        self._delays: deque[int] = deque(maxlen=DELAY_SAMPLES)
        self._now: int | None = None
        self._judged_ms = 0
        self._next: Session | None = None
        self._silent_detail = f"no message for {feed_stale_ms / MS_PER_SECOND:g} s"
        self._quiet_detail = f"no eligible print for {symbol_stale_ms / MS_PER_SECOND:g} s"

    @classmethod
    def from_config(cls, feed: StockFeedConfig) -> PriceHealth:
        return cls(feed.symbols, round(feed.feed_stale_after_seconds * MS_PER_SECOND),
                   round(feed.symbol_stale_after_seconds * MS_PER_SECOND),
                   round(feed.max_print_delay_seconds * MS_PER_SECOND), PriceRules.from_config(feed))

    @property
    def feed_ok(self) -> bool:
        """Whether every whole feed condition holds now."""
        return self.connected and not self.silent and self.delay_ok

    @property
    def any_healthy(self) -> bool:
        return any(s.state == HEALTHY for s in self.symbols.values())

    def drain_events(self) -> list[HealthEvent]:
        events, self.events = self.events, []
        return events

    # Feed lines

    def on_event(self, ev: PrintBatch | FeedNotice) -> None:
        """Apply one decoded event. Every event but the connection markers proves the socket alive."""
        if isinstance(ev, PrintBatch):
            self.on_batch(ev)
        elif ev.kind == CONNECTED:
            self.on_connected(ev.recv_ms)
        elif ev.kind == DISCONNECTED:
            self.on_disconnected(ev.recv_ms, ev.detail)
        else:
            self.on_message(ev.recv_ms)
            if ev.kind == ERROR:
                self.events.append(HealthEvent(ev.recv_ms, ERROR, None, ev.detail))

    def on_connected(self, ms: int) -> None:
        """A new connection restarts the silence clock and clears the delay samples and the symbols printed on it. One
        without a close before it, as a recording shows after a kill, first ends the old one."""
        self.advance(ms)
        if self.connected:
            self.on_disconnected(ms, "connected again without a close")
        self.connected = True
        self.connected_ms = self.last_msg_ms = ms
        self.silent = False
        self._delays.clear()
        self.printed = set()
        self.events.append(HealthEvent(ms, CONNECTED, None, ""))
        if self.session is None:
            self.delay_ok = True
            return
        if not self.delay_ok:
            self.delay_ok = True
            self.events.append(HealthEvent(ms, PRINT_DELAY, None, "cleared by a new connection"))
        self._settle_all(ms, CONNECTED)

    def on_disconnected(self, ms: int, detail: str = "") -> None:
        self.advance(ms)
        self.connected = False
        self.events.append(HealthEvent(ms, DISCONNECTED, None, detail))
        if self.session is not None:
            self._settle_all(ms, DISCONNECTED)

    def on_message(self, ms: int) -> None:
        """Any message line proves the socket alive, including one the adapter turns into nothing."""
        self.advance(ms)
        self.last_msg_ms = ms
        if self.silent:
            self.silent = False
            self._settle_all(ms, "messages again")

    def on_batch(self, batch: PrintBatch) -> None:
        """Take delay samples from the non replayed session prints of a batch and freshen each symbol with an eligible
        one, by the same rules as the candles."""
        ms = batch.recv_ms
        self.on_message(ms)
        self.printed.update(p.symbol for p in batch.prints)
        session = self.session
        if session is None:
            return
        sampled = False
        renewed: list[str] = []
        for p in batch.prints:
            s = self.symbols.get(p.symbol)
            if s is None or is_replayed(p.vendor_ms, self.connected_ms) or not in_session(session, ms, p.vendor_ms):
                continue
            self._delays.append(age_ms(ms, p.vendor_ms))
            sampled = True
            if eligibility(ms, p.vendor_ms, p.conditions, self.rules) == ELIGIBLE:
                s.last_eligible_ms = ms
                if not s.fresh:
                    s.fresh = True
                    renewed.append(p.symbol)
        if sampled and self._delay_changed(ms):
            self._settle_all(ms, PRINT_DELAY if not self.delay_ok else "print delay cleared")
        for symbol in renewed:
            self._settle(symbol, self.symbols[symbol], ms, "eligible print")

    def _delay_changed(self, ms: int) -> bool:
        """Judge the median delay of the samples, writing a print_delay event when the objection starts or ends."""
        if len(self._delays) < DELAY_MIN_SAMPLES:
            return False
        median = statistics.median(self._delays)
        ok = abs(median) <= self.max_delay_ms
        if ok == self.delay_ok:
            return False
        self.delay_ok = ok
        detail = f"median {median:.0f} ms" if not ok else f"cleared, median {median:.0f} ms"
        self.events.append(HealthEvent(ms, PRINT_DELAY, None, detail))
        return True

    # Time

    def advance(self, now_ms: int) -> None:
        """Apply every silence, close and open due at or before now_ms, in time order. A monitor first used inside a
        session enters it at that instant."""
        if self._now is not None and now_ms <= self._now:
            return
        since = now_ms if self._now is None else self._now
        self._now = now_ms
        while True:
            if self.session is not None:
                self._expire(min(now_ms, self.session.close_ms - 1))
                if now_ms < self.session.close_ms:
                    return
                self._close(self.session.close_ms)
            upcoming = self._upcoming(now_ms)
            if now_ms < upcoming.open_ms:
                return
            self._open(upcoming, max(upcoming.open_ms, since))

    def judge_minute(self, boundary_ms: int) -> dict[str, bool]:
        """Whether each symbol was healthy for the minute ending at boundary_ms, which needs a full minute in which
        every condition held. A boundary outside the session is never healthy."""
        if self._now is not None and self._now >= boundary_ms or boundary_ms <= self._judged_ms:
            raise ValueError(f"judge the minute ending {iso(boundary_ms)} once, before applying anything stamped at or "
                             f"after it")
        self._judged_ms = boundary_ms
        self.advance(boundary_ms - 1)
        counts = self.session is not None and self.session.has_boundary(boundary_ms)
        verdict = {}
        for symbol, s in self.symbols.items():
            healthy = counts and s.good_since is not None and s.good_since <= boundary_ms - MS_PER_MINUTE
            if healthy and s.state != HEALTHY:
                self._set(symbol, s, boundary_ms, HEALTHY, "a full clean minute")
            verdict[symbol] = healthy
        return verdict

    def _upcoming(self, now_ms: int) -> Session:
        """The session in progress at now_ms or the next to open, looked up again only once the cached one closed."""
        if self._next is None or self._next.close_ms <= now_ms:
            self._next = sessions.next_session(now_ms)
        return self._next

    def _open(self, session: Session, at: int) -> None:
        """Enter a session, where the silence clocks start and the delay samples clear, so a connected symbol moves to
        recovering with its clean stretch starting at the open."""
        self.session = session
        self.entered_ms = at
        self.silent = False
        self.delay_ok = True
        self._delays.clear()
        self.events.append(HealthEvent(at, SESSION_OPEN, None, "" if at == session.open_ms else "joined in session"))
        for symbol, s in self.symbols.items():
            s.fresh = True
            s.good_since = at if self.feed_ok else None
            self._set(symbol, s, at, RECOVERING if self.feed_ok else STALE, SESSION_OPEN)

    def _close(self, at: int) -> None:
        """Close every symbol, bumping every epoch once, whatever its state."""
        self.events.append(HealthEvent(at, SESSION_CLOSE, None, ""))
        for symbol, s in self.symbols.items():
            s.epoch += 1
            s.state = CLOSED
            s.good_since = None
            self.events.append(HealthEvent(at, CLOSED, symbol, SESSION_CLOSE))
        self.session = None

    def _expire(self, until: int) -> None:
        """Apply, in time order, the silence of the feed and of each symbol whose limit ran out at or before until."""
        start = self.entered_ms
        due: list[tuple[int, int]] = []
        if not self.silent:
            at = (start if self.last_msg_ms is None else max(self.last_msg_ms, start)) + self.feed_stale_ms
            if at <= until:
                due.append((at, -1))
        for index, s in enumerate(self.symbols.values()):
            if s.fresh:
                at = (start if s.last_eligible_ms is None else max(s.last_eligible_ms, start)) + self.symbol_stale_ms
                if at <= until:
                    due.append((at, index))
        if not due:
            return
        names = list(self.symbols)
        for at, index in sorted(due):
            if index < 0:
                self.silent = True
                self._settle_all(at, self._silent_detail)
            else:
                s = self.symbols[names[index]]
                s.fresh = False
                self._settle(names[index], s, at, self._quiet_detail)

    # States

    def _settle_all(self, at: int, detail: str) -> None:
        for symbol, s in self.symbols.items():
            self._settle(symbol, s, at, detail)

    def _settle(self, symbol: str, s: SymbolHealth, at: int, detail: str) -> None:
        """Start a clean stretch when every condition holds and none is running, or end it when one fails."""
        if s.state == CLOSED:
            return
        if self.feed_ok and s.fresh:
            if s.good_since is None:
                s.good_since = at
                if s.state == STALE:
                    self._set(symbol, s, at, RECOVERING, detail)
        else:
            s.good_since = None
            if s.state != STALE:
                self._set(symbol, s, at, STALE, detail)

    def _set(self, symbol: str, s: SymbolHealth, at: int, state: str, detail: str) -> None:
        """Change a symbol's state with an event, bumping its epoch when it leaves healthy."""
        if s.state == HEALTHY:
            s.epoch += 1
        s.state = state
        self.events.append(HealthEvent(at, state, symbol, detail))


# The delay rule


class DelayRule(NamedTuple):
    """The two delay settings from a recorded session, and whether its median delay stops the build for the owner."""

    max_print_delay_seconds: float
    late_print_seconds: float
    stop: bool


def delay_rule(median_ms: int, p95_ms: int) -> DelayRule:
    """The rule from a recorded session's median and p95 delays, never from prices, the larger of 2 and
    ceil(1.5 * p95) seconds capped at 5, twice that for late prints, and a stop when the median is more than 5 seconds
    either way, since live health objects to a median delay either way."""
    seconds = min(DELAY_CAP_SECONDS, max(DELAY_FLOOR_SECONDS, -(-3 * p95_ms // (2 * MS_PER_SECOND))))
    return DelayRule(float(seconds), float(2 * seconds), abs(median_ms) > DELAY_STOP_MS)


# Reconnects, resubscribes and the backoff


def silence_reason(now_ms: int, last_ms: int, stale_ms: int, checked_open_ms: int) -> tuple[str | None, int]:
    """Why the open connection should be replaced for silence, or None, with the open whose pre open check has run.
    In session after stale_ms without a message, counted from the later of last_ms and the open, and once per session,
    at the first check in the ten minutes before the open, after five silent minutes."""
    session = sessions.next_session(now_ms)
    if now_ms >= session.open_ms:
        if now_ms - max(last_ms, session.open_ms) >= stale_ms:
            return f"no message for {stale_ms / MS_PER_SECOND:g} s in session", checked_open_ms
    elif now_ms >= session.open_ms - PRE_OPEN_CHECK_MS and checked_open_ms != session.open_ms:
        if now_ms - last_ms >= PRE_OPEN_SILENCE_MS:
            return "silent for 5 minutes before the open", session.open_ms
        return None, session.open_ms
    return None, checked_open_ms


def quiet_symbols(now_ms: int, connected_ms: int, symbols: Sequence[str], printed: set[str],
                  resubscribed: dict[str, int]) -> list[str]:
    """Symbols to subscribe again, those with no print on this connection 120 seconds into the session and the
    connection while others print, each at most once every 5 minutes. One that printed and went quiet is halted."""
    session = sessions.session_at(now_ms)
    if session is None or not printed or now_ms - max(session.open_ms, connected_ms) < RESUBSCRIBE_AFTER_MS:
        return []
    return [s for s in symbols if s not in printed
            and (s not in resubscribed or now_ms - resubscribed[s] >= RESUBSCRIBE_EVERY_MS)]


def should_mark_healthy(now_ms: int, connected_ms: int | None, after_ms: int, any_healthy: bool) -> bool:
    """Whether to reset the reconnect backoff, outside the session once the open connection has lasted after_ms, and in
    it once any symbol is healthy, since a quiet symbol is no reason to back off."""
    if connected_ms is None:
        return False
    if sessions.session_at(now_ms) is None:
        return now_ms - connected_ms >= after_ms
    return any_healthy


# News health of one minute


class Heartbeat(NamedTuple):
    """One news_heartbeat line as the stock side reads it, whether it said healthy and whether it carried the hashes
    recorded at launch."""

    recv_ms: int
    healthy: bool
    hashes_match: bool


def news_minute(boundary_ms: int, previous_ms: int | None, lines: Sequence[Heartbeat], max_age_ms: int) -> str | None:
    """Why the minute ending at boundary_ms is not news healthy, or None when it is. previous_ms stamps the newest line
    before the minute and lines are those received in it in order, the newest of which must never be more than
    max_age_ms old during the minute, each saying healthy with the launch hashes."""
    stamps = [line.recv_ms for line in lines]
    if previous_ms is None and (not stamps or stamps[0] > boundary_ms - MS_PER_MINUTE):
        return NO_HEARTBEAT
    points = ([] if previous_ms is None else [previous_ms]) + stamps + [boundary_ms]
    if any(later - earlier > max_age_ms for earlier, later in pairwise(points)):
        return OLD_HEARTBEAT
    if not all(line.hashes_match for line in lines):
        return HASH_MISMATCH
    if not all(line.healthy for line in lines):
        return UNHEALTHY_HEARTBEAT
    return None
