"""Seeded synthetic raw stock recordings in the format scripts/v2_record_stocks.py writes, for tests and soak runs of
PLAN_V2. Never used live, and every price, size and condition code is made up.

A market of random walks trades each symbol in Poisson bursts from 04:00 to 20:00 New York time on session days,
thinly before the open and after the close, with odd lots, condition codes and a receive delay on every print. A
recorder process delivers the prints of each tick as one Finnhub trade message on its connection, with a ping after 3
quiet seconds, an assumption since Finnhub's ping cadence is unverified. It follows the recorder's own rules, so a
silence of 5 seconds in session closes the connection, a pause of more than 10 seconds closes it with the reason
"process paused for N s", reconnects back off from 5 to 60 seconds with seeded jitter, and connection ids restart at
1 at each start and skip any id a refused handshake used.
The market and the connection draw from separate generators, so faults never change the market apart from a halt.

Faults are the server dropping the connection, silences, pauses, kills with a later restart, halts, late, early and
replayed prints. Two control lines, PROCESS_START and PROCESS_KILL, mark a start and a kill for write_recording and
never reach a file. write_recording uses the recorder's own StockRecorder, so files start at each process start and
at UTC midnight, and a killed process leaves its file without the gzip trailer and with half a line at its end.
"""

from __future__ import annotations

import gzip
import json
import math
import zlib
from collections import Counter, deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from btc_survivor.clock import MS_PER_HOUR, MS_PER_MINUTE, MS_PER_SECOND
from btc_survivor.feed.base import CLOSE, MESSAGE, OPEN
from btc_survivor.recorder import recording_files
from btc_survivor.v2 import sessions
from btc_survivor.v2.sessions import Session
from btc_survivor.v2.stocks.prints import OTHER, POST, PRE, REGULAR, REPLAY_LIMIT_MS, phase_in
from btc_survivor.v2.stocks.runner import StockRecorder

Line = tuple[str, str, int, int]

SYMBOLS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "QQQ")
PROCESS_START, PROCESS_KILL = "process_start", "process_kill"
# Fault kinds.
DISCONNECT, SILENCE, PAUSE, KILL, HALT = "disconnect", "silence", "pause", "kill", "halt"
LATE_PRINTS, EARLY_PRINTS, REPLAYED_PRINTS = "late_prints", "early_prints", "replayed_prints"
FAULTS = (DISCONNECT, SILENCE, PAUSE, KILL, HALT, LATE_PRINTS, EARLY_PRINTS, REPLAYED_PRINTS)
WINDOWED = (SILENCE, HALT, LATE_PRINTS, EARLY_PRINTS)

# The recorder's rules, from scripts/v2_record_stocks.py and the committed [stock_feed] settings.
STALE_MS = 5 * MS_PER_SECOND
PAUSE_LIMIT_MS = 10 * MS_PER_SECOND
HEALTHY_MS = 60 * MS_PER_SECOND
BACKOFF_START_MS = 5 * MS_PER_SECOND
BACKOFF_MAX_MS = 60 * MS_PER_SECOND
DROPPED = "ConnectionClosedError: no close frame received or sent"
SILENT = "no message for 5 s in session"
SHUTDOWN = "shutdown"
PING = '{"type":"ping"}'

IDLE_PING_MS = 3 * MS_PER_SECOND
KEEP_RECENT_MS = 10 * MS_PER_MINUTE
ROUND_LOT = 100
# Condition code sets with their weights, None meaning a print with no c field at all.
CONDITIONS: tuple[tuple[str, ...] | None, ...] = (None, (), ("1",), ("12",), ("1", "12"), ("24",), ("37",))
CONDITION_WEIGHTS = (0.15, 0.05, 0.45, 0.15, 0.1, 0.05, 0.05)


@dataclass(frozen=True)
class Fault:
    """One fault from at_ms. A disconnect drops the connection, and refused handshakes come before the next open. A
    silence stops every message, a pause freezes the recorder process and a kill ends it, restarting it duration_ms
    later with refused handshakes first, unless that falls at or after the end. A halt stops one symbol trading. Late
    prints arrive shift_ms later than usual, early prints carry a vendor time shift_ms after their receipt, for one
    symbol or all, and replayed prints open the next connection with the prints of the duration_ms before it."""

    kind: str
    at_ms: int
    duration_ms: int = 0
    symbol: str | None = None
    shift_ms: int = 0
    refused: int = 0

    def __post_init__(self) -> None:
        if self.kind not in FAULTS:
            raise ValueError(f"unknown fault kind {self.kind!r}")
        if self.duration_ms < 0 or self.shift_ms < 0 or self.refused < 0:
            raise ValueError("a fault's duration, shift and refused handshakes are never negative")
        if self.kind == HALT and self.symbol is None:
            raise ValueError("a halt names its symbol")

    def active(self, t: int) -> bool:
        return self.at_ms <= t < self.at_ms + self.duration_ms


class _Link:
    """One recorder process's connection, with ids from 1 that skip refused handshakes, the reused FeedClient's
    backoff and the stamp of its last line, which the silence rule reads."""

    def __init__(self, rng: np.random.Generator, start_ms: int, refused: int = 0) -> None:
        self.rng = rng
        self.conn_id = self.attempt = 0
        self.open_ms: int | None = None
        self.last_ms = start_ms
        self.next_open: int | None = start_ms
        self._refuse(refused)

    def backoff(self) -> int:
        """The delay before the next attempt, doubling from 5 seconds to 60 with jitter between half and all of it."""
        base = min(BACKOFF_MAX_MS, BACKOFF_START_MS * 2 ** min(self.attempt, 30))
        self.attempt += 1
        return int(self.rng.uniform(base / 2, base))

    def _refuse(self, refused: int) -> None:
        for _ in range(refused):
            self.conn_id += 1
            self.next_open += self.backoff()

    def due(self, t: int) -> bool:
        return self.open_ms is None and self.next_open is not None and self.next_open <= t

    def open(self, t: int) -> Line:
        self.conn_id += 1
        self.open_ms = self.last_ms = t
        self.next_open = None
        return OPEN, "", t, self.conn_id

    def message(self, t: int, raw: str) -> Line:
        self.last_ms = t
        return MESSAGE, raw, t, self.conn_id

    def close(self, t: int, reason: str, refused: int = 0) -> Line:
        """End the connection at t and plan the next open after the backoff and any refused handshakes. A connection
        that stayed open 60 seconds had its backoff reset."""
        if t - self.open_ms >= HEALTHY_MS:
            self.attempt = 0
        line = (CLOSE, reason, t, self.conn_id)
        self.open_ms = None
        self.next_open = t + self.backoff()
        self._refuse(refused)
        return line


class SyntheticTape:
    """A seeded synthetic market and recorder. truth holds each session date's true volume per symbol from 04:00 to
    20:00 New York time, and closes the price of its last regular print, filled as lines are generated."""

    def __init__(self, seed: int, symbols: Sequence[str] = SYMBOLS, tick_ms: int = MS_PER_SECOND,
                 regular_rate: float = 0.5, extended_rate: float = 0.05, odd_lot_share: float = 0.3,
                 delay_ms: float = 150.0, step_sigma: float = 0.0005) -> None:
        streams = np.random.SeedSequence(seed).spawn(2)
        self._market_rng = np.random.default_rng(streams[0])
        self._link_rng = np.random.default_rng(streams[1])
        self.symbols = tuple(symbols)
        self.tick_ms = tick_ms
        self.rates = {REGULAR: regular_rate, PRE: extended_rate, POST: extended_rate}
        self.odd_lot_share = odd_lot_share
        self.delay_ms = delay_ms
        self.step_sigma = step_sigma
        self.prices = {s: float(p) for s, p in zip(self.symbols, self._market_rng.uniform(50, 500, len(symbols)))}
        self.truth: Counter[tuple[date, str]] = Counter()
        self.closes: dict[tuple[date, str], float] = {}
        self._last_vendor: dict[tuple[date, str], int] = {}
        self._recent: deque[tuple[int, dict]] = deque()
        self._hour: int | None = None
        self._session: Session | None = None

    def lines(self, start_ms: int, end_ms: int, faults: Iterable[Fault] = ()) -> Iterator[Line]:
        """Yield (kind, raw, recv_ms, conn_id) lines, with the two control lines, of one recorder started at start_ms
        and stopped by SIGTERM at end_ms, through the faults."""
        faults = sorted(faults, key=lambda f: f.at_ms)
        pending = [f for f in faults if f.kind not in WINDOWED]
        windows = [f for f in faults if f.kind in WINDOWED]
        link = _Link(self._link_rng, start_ms)
        replay_ms = 0
        paused_from: int | None = None
        paused_until = 0
        dead_until: int | None = None
        restart: Fault | None = None
        held: list[dict] = []
        yield PROCESS_START, "", start_ms, 0
        t = start_ms
        while t < end_ms:
            if dead_until is not None and t >= dead_until:
                dead_until, link = None, _Link(self._link_rng, t, restart.refused)
                yield PROCESS_START, "", t, 0
            if paused_from is not None and t >= paused_until:
                # The ticker closes after a pause of more than 10 seconds, or else the held messages arrive now.
                if link.open_ms is not None and t - paused_from > PAUSE_LIMIT_MS:
                    yield link.close(t, f"process paused for {(t - paused_from) // MS_PER_SECOND} s")
                elif link.open_ms is not None and held:
                    yield link.message(t, trade_message(held))
                elif link.next_open is not None:
                    link.next_open = max(link.next_open, t)
                paused_from, held = None, []
            while pending and pending[0].at_ms <= t:
                fault = pending.pop(0)
                if dead_until is not None:
                    continue
                if fault.kind == DISCONNECT and link.open_ms is not None and paused_from is None:
                    yield link.close(t, DROPPED, fault.refused)
                elif fault.kind == PAUSE and paused_from is None:
                    paused_from, paused_until = t, t + fault.duration_ms
                elif fault.kind == KILL:
                    yield PROCESS_KILL, "", t, link.conn_id
                    dead_until, restart, paused_from, held = min(t + fault.duration_ms, end_ms), fault, None, []
                elif fault.kind == REPLAYED_PRINTS:
                    replay_ms = fault.duration_ms
            items = self._market(t, windows)
            silent = any(f.kind == SILENCE and f.active(t) for f in windows)
            if dead_until is not None or paused_from is not None:
                if paused_from is not None and link.open_ms is not None and not silent:
                    held.extend(items)
                t += self.tick_ms
                continue
            if link.due(t):
                yield link.open(t)
                if replay_ms:
                    replayed = [item for vendor, item in self._recent if t - replay_ms <= vendor < t - REPLAY_LIMIT_MS]
                    if replayed:
                        yield link.message(t, trade_message(replayed))
                    replay_ms = 0
            if link.open_ms is not None and not silent:
                if items:
                    yield link.message(t, trade_message(items))
                elif t - link.last_ms >= IDLE_PING_MS:
                    yield link.message(t, PING)
            session = self._session_at(t)
            if link.open_ms is not None and session is not None and session.open_ms <= t < session.close_ms \
                    and t - max(link.last_ms, session.open_ms) >= STALE_MS:
                yield link.close(t, SILENT)
            t += self.tick_ms
        if dead_until is None and link.open_ms is not None:
            yield CLOSE, SHUTDOWN, end_ms, link.conn_id

    def _session_at(self, t: int) -> Session | None:
        """The session on the New York date of t, cached by the hour, since New York offsets are whole hours."""
        hour = t // MS_PER_HOUR
        if hour != self._hour:
            self._hour, self._session = hour, sessions.session_on(sessions.ny_date(t))
        return self._session

    def _market(self, t: int, windows: Sequence[Fault]) -> list[dict]:
        """The prints traded in the tick ending at t, in symbol order, adding each to the truth and the recent prints
        that a reconnect may replay. A halted symbol's prints are drawn and dropped, so the market stays the same."""
        session = self._session_at(t)
        rate = 0.0 if session is None else self.rates.get(phase_in(session, t), 0.0)
        if not rate:
            return []
        rng = self._market_rng
        counts = rng.poisson(rate * self.tick_ms / MS_PER_SECOND, len(self.symbols))
        n = int(counts.sum())
        if n == 0:
            return []
        ages = 1 + rng.exponential(self.delay_ms, n).astype(np.int64)
        odd = rng.random(n) < self.odd_lot_share
        odd_sizes = rng.integers(1, ROUND_LOT, n)
        lots = rng.integers(1, 6, n) * ROUND_LOT
        codes = rng.choice(len(CONDITIONS), n, p=CONDITION_WEIGHTS)
        steps = rng.normal(0.0, self.step_sigma, n)
        active = [f for f in windows if f.active(t)]
        items = []
        i = 0
        for symbol, count in zip(self.symbols, counts):
            for _ in range(count):
                self.prices[symbol] *= math.exp(steps[i])
                if not any(f.kind == HALT and f.symbol == symbol for f in active):
                    vendor = self._vendor_ms(t, int(ages[i]), symbol, active)
                    size = int(odd_sizes[i] if odd[i] else lots[i])
                    items.append(self._traded(session, symbol, round(self.prices[symbol], 2), size, vendor,
                                              CONDITIONS[codes[i]]))
                i += 1
        return items

    @staticmethod
    def _vendor_ms(t: int, age: int, symbol: str, active: Sequence[Fault]) -> int:
        """The vendor time of a print received at t, moved by any late or early fault on its symbol."""
        for f in active:
            if f.symbol in (None, symbol):
                if f.kind == EARLY_PRINTS:
                    return t + f.shift_ms
                if f.kind == LATE_PRINTS:
                    age += f.shift_ms
        return t - age

    def _traded(self, session: Session, symbol: str, price: float, size: int, vendor: int,
                conditions: tuple[str, ...] | None) -> dict:
        """One print in Finnhub's field names, counted in the truth by its vendor time as the tape check counts it."""
        item: dict = {} if conditions is None else {"c": list(conditions)}
        item.update(p=price, s=symbol, t=vendor, v=size)
        phase = phase_in(session, vendor)
        key = (session.day, symbol)
        if phase != OTHER:
            self.truth[key] += size
        if phase == REGULAR and vendor >= self._last_vendor.get(key, vendor):
            self._last_vendor[key], self.closes[key] = vendor, price
        self._recent.append((vendor, item))
        while self._recent and self._recent[0][0] < vendor - KEEP_RECENT_MS:
            self._recent.popleft()
        return item


def trade_message(items: Sequence[dict]) -> str:
    return json.dumps({"data": list(items), "type": "trade"}, separators=(",", ":"))


def recorded(lines: Iterable[Line]) -> list[Line]:
    """The lines a recording holds, without the control lines."""
    return [line for line in lines if line[0] not in (PROCESS_START, PROCESS_KILL)]


def write_recording(folder: Path, lines: Iterable[Line]) -> list[Path]:
    """Write lines through the recorder's StockRecorder, a new file at each process start and at UTC midnight, and
    return the files in name order. A killed process's file loses its gzip trailer and ends with half a line."""
    recorder: StockRecorder | None = None
    killed: list[tuple[Path, str]] = []
    last: Line | None = None
    for line in lines:
        kind = line[0]
        if kind in (PROCESS_START, PROCESS_KILL):
            if recorder is not None:
                recorder.close()
                if kind == PROCESS_KILL and recorder.path is not None:
                    was, raw, recv_ms, conn_id = last
                    text = json.dumps({"t": recv_ms, "k": was, "c": conn_id, "d": raw}, separators=(",", ":"))
                    killed.append((recorder.path, text[:len(text) // 2]))
            recorder = StockRecorder(folder) if kind == PROCESS_START else None
        else:
            recorder.write(*line)
            last = line
    if recorder is not None:
        recorder.close()
    for path, partial in killed:
        cut(path, partial)
    return recording_files([folder])


def cut(path: Path, partial: str = "") -> None:
    """Rewrite a closed recording file as a kill leaves it, flushed but without the gzip trailer, with a partial line
    after its last whole line."""
    compressor = zlib.compressobj(wbits=31)
    data = gzip.decompress(path.read_bytes()) + partial.encode("utf-8")
    path.write_bytes(compressor.compress(data) + compressor.flush(zlib.Z_SYNC_FLUSH))
