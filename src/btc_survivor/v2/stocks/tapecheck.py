"""The tape coverage report of PLAN_V2 section 4.9, from a raw recording and the owner's official figures alone.

Finnhub's free websocket carries a partial tape, a small share of consolidated volume, and the owner kept it on 12
October 2026. So this report measures how much of the tape the feed carries instead of judging whether it is
consolidated, and criterion 1.1 asks only that the coverage stays stable, the median ratio of the first clean session
at or after sessions 10, 20 and 40 within a factor of 2 of the gate session's, clean or not.

It reads only the raw recording that scripts/v2_record_stocks.py or the stock process writes, replaying finnhub.parse
and the print classification of prints.py, so it needs no run database. Per symbol it sums the volume of every print
whose vendor time falls from 04:00 to 20:00 New York time on the session date, split into pre market, regular and post
market, leaving out replayed prints, which repeat what an earlier connection may already have carried. Late and early
prints count, because they are real prints whose timing alone is doubtful. The ratio r of that full day volume to the
official consolidated volume is the symbol's coverage, and the median over the symbols is the session's.

Lost time. Disconnected time anywhere in the window and outage time in the regular session are left out of the covered
ratio. An outage is a whole feed silence on an open connection of at least the config's feed_stale_after_seconds, the
silence after which live health judges the whole feed stale, counted in full. Each phase's volume is scaled by the
phase's length over its covered length, which assumes an even rate within a phase, so a session with no lost time has
a covered ratio exactly equal to r, and a phase with no covered time leaves the covered ratio unknown.

A clean session is a full session, not an early close, with no disconnected or outage time from 04:00 to 20:00, as
section 4.9 sets out. Eligibility, behind the late share, the eligible prints and the last eligible print before 16:00
that meets the official close, follows the run config's late_print_seconds and price_excluded_conditions.

Connections follow the rules in the recorder's docstring. The files are read in name order as one stream. A
connection starts at its open marker, or at the first line when the stream starts inside one, and ends at its close
marker, or at its last line before the next open or the end of the stream, as a kill leaves it. A file started at UTC
midnight has no open marker and simply continues the open connection. Time from the end of one connection to the
next open is disconnected, a close with the reason "process paused for N s" adds the N seconds before it, and
connection ids are never used, because they restart at each start and skip refused handshakes. Reading stops at the
first line 10 minutes after the window, then follows that line's connection for 15 seconds more, so a pause close
the recorder writes just after waking from a long freeze still counts.

A recording whose last connection is still open at its end, with no line 10 minutes after the window, is refused as
incomplete, because a recorder still running on a quiet connection cannot be told apart from one killed before the
window ended. The check then writes no record, and runs again once the recorder has written a later line or stopped
with SIGTERM, whose shutdown close marks the end.

The official figures are a hand filled data/v2/checks/official_YYYY_MM_DD.toml named by the New York session date,
in the format of configs/v2/official_example.toml, and a file that git does not ignore is refused, so the figures and
the records beside them never enter the repository. consolidated_label is true when the page labels its volume as
consolidated, and each [symbols.SYMBOL] table holds that day's volume and official close, both above zero, for exactly
the config's symbols. Without the label every symbol also needs second_volume from a second consolidated display,
which must agree with volume within 5 percent, and volume is the figure used.

The record. Each check writes coverage_YYYY_MM_DD.json beside the official file, readable by the owner alone and
replaced whole by a rerun, with the session date, the clean flag and its reason, the early close flag, the median and
per symbol ratios with and without lost time, the disconnected and outage seconds and each phase's covered share.
read_records reads every record for evaluate, and within_factor is the comparison of criterion 1.1.

The delay rule of section 4.5, from health.delay_rule. Delays are receive stamp minus vendor time of the non replayed
prints whose vendor time falls in the regular session, with nearest rank percentiles over whole milliseconds.
max_print_delay_seconds becomes the larger of 2 and ceil(1.5 * p95) in seconds, capped at 5, with late_print_seconds
twice that, both timings of this system. The delay verdict passes when the median is within max_print_delay_seconds
either way, as price health judges it live, so a feed more than 5 seconds behind fails it, and so does a local clock
that runs ahead.

By default the command prints only the clean flag and its reason, counts, lost time, the delay verdict, the two
settings and timings of this system, and exits 0 for a clean session, 1 for any other session and 2 when it writes no
record. --detail adds the ratios and their medians, volumes, the close comparison, covered time per phase, delay
percentiles, the odd lot, late, early and replayed shares, the condition histogram, and the disconnected and outage
intervals.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import tomllib
from array import array
from bisect import bisect_left
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from btc_survivor.clock import MS_PER_MINUTE, MS_PER_SECOND, iso
from btc_survivor.feed.base import CLOSE, MESSAGE, OPEN, decode
from btc_survivor.recorder import read_recording, recording_files
from btc_survivor.v2 import output, sessions
from btc_survivor.v2.config import ConfigError, StockFeedConfig, load_stock_config
from btc_survivor.v2.sessions import CalendarError, Session
from btc_survivor.v2.stocks.finnhub import FinnhubAdapter, PrintBatch
from btc_survivor.v2.stocks.health import delay_rule
from btc_survivor.v2.stocks.prints import (EARLY, ELIGIBLE, LATE, OTHER, POST, POST_MARKET_MS, PRE, PRE_MARKET_MS,
                                           REGULAR, PriceRules, eligibility, in_session, is_replayed, phase_in)

PHASES = (PRE, REGULAR, POST)
# Criterion 1.1, a clean session's median ratio within this factor of the gate session's either way.
STABILITY_FACTOR = 2.0
RECORD_KIND = "coverage"
COVERAGE_NAME = re.compile(r"coverage_(\d{4})_(\d{2})_(\d{2})\.json")
SECOND_DISPLAY_TOLERANCE = 0.05
ROUND_LOT = 100
# Prints received this long after the window still count by their vendor time, and reading stops after it.
LATE_MARGIN_MS = 10 * MS_PER_MINUTE
# How long the connection of the first line past the margin is still followed, more than the recorder's 5 second
# close timeout and 1 second tick, so a pause close written just after a long freeze is read.
TRAIL_MS = 15 * MS_PER_SECOND
# The adapter needs a token shape, and replay never connects, so no secret is read.
REPLAY_TOKEN = "replay"
NO_CONDITION = "none"
NO_CLOSE = "no close marker"
END_OF_RECORDING = "end of the recording"
BEFORE_RECORDING = "before the recording"
OUTAGE = "outage"
PAUSED = re.compile(r"process paused for (\d+) s")
# The close a recorder or stock process writes when a connection stayed silent in session, from health.silence_reason.
SILENCE_CLOSE = re.compile(r"no message for \d+(?:\.\d+)? s in session")
OFFICIAL_NAME = re.compile(r"official_(\d{4})_(\d{2})_(\d{2})\.toml")
FILE_NAME = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{6})\.jsonl\.gz")
DEFAULT_CONFIG = Path("configs/v2/s1_prices.toml")

EARLY_CLOSE_REASON = "an early close is never a clean session"
DISCONNECTED_REASON = "disconnected time in the window"
OUTAGE_REASON = "outage time in the session"
CLEAN_REASON = "a full session connected throughout"

INCOMPLETE = ("the recording does not reach 10 minutes past the window and its last connection has no close marker, "
              "so the recorder may still be running or was killed, and the check needs a later line or the close "
              "that SIGTERM writes")
NOT_IGNORED = ("the official file must sit in the gitignored data folder, such as data/v2/checks, because its figures "
               "and the coverage records beside them never enter the repository")


class TapeCheckError(ValueError):
    """Raised for official figures, records or recordings the check cannot use, never naming a figure."""


@dataclass(frozen=True)
class Official:
    """The owner's official figures for one session date."""

    day: date
    volumes: dict[str, float]
    closes: dict[str, float]


def _figure(value: Any, where: str) -> float:
    """A hand copied figure, a finite number above zero, so a placeholder zero is refused."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise TapeCheckError(f"{where} must be a number above zero, copied from the official page")
    return float(value)


def parse_official(data: dict[str, Any], day: date, name: str = "the official figures") -> Official:
    """Validate the official figures read from TOML, refusing unknown keys, placeholders and a second display that
    disagrees by more than 5 percent."""
    unknown = sorted(set(data) - {"consolidated_label", "symbols"})
    if unknown:
        raise TapeCheckError(f"{name} has unknown keys {', '.join(unknown)}")
    label = data.get("consolidated_label")
    if not isinstance(label, bool):
        raise TapeCheckError(f"{name} needs consolidated_label = true or false")
    symbols = data.get("symbols")
    if not isinstance(symbols, dict) or not symbols:
        raise TapeCheckError(f"{name} needs one [symbols.SYMBOL] table per symbol")
    volumes, closes = {}, {}
    for symbol, figures in symbols.items():
        if not isinstance(figures, dict):
            raise TapeCheckError(f"{name} needs a [symbols.{symbol}] table")
        extra = sorted(set(figures) - {"volume", "close", "second_volume"})
        if extra:
            raise TapeCheckError(f"{name} has unknown keys {', '.join(extra)} for {symbol}")
        volumes[symbol] = _figure(figures.get("volume"), f"{name} {symbol} volume")
        closes[symbol] = _figure(figures.get("close"), f"{name} {symbol} close")
        if "second_volume" in figures or not label:
            second = _figure(figures.get("second_volume"), f"{name} {symbol} second_volume, needed without a "
                                                           f"consolidated label,")
            if abs(second - volumes[symbol]) > SECOND_DISPLAY_TOLERANCE * volumes[symbol]:
                raise TapeCheckError(f"{name} {symbol} volume and second_volume differ by more than 5 percent")
    return Official(day, volumes, closes)


def _named_day(pattern: re.Pattern[str], name: str) -> date | None:
    """The date in a file name of the pattern, or None for another name or an impossible date."""
    match = pattern.fullmatch(name)
    try:
        return date(*(int(part) for part in match.groups())) if match else None
    except ValueError:
        return None


def load_official(path: Path) -> Official:
    """Read official_YYYY_MM_DD.toml, taking the session date from its name."""
    path = Path(path)
    day = _named_day(OFFICIAL_NAME, path.name)
    if day is None:
        raise TapeCheckError(f"{path.name} is not named official_YYYY_MM_DD.toml with a real date")
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise TapeCheckError(f"{path.name} is not valid TOML: {exc}") from exc
    return parse_official(data, day, path.name)


def window(session: Session) -> tuple[int, int]:
    """04:00 to 20:00 New York time around a full session, or to 17:00 after an early close, in UTC milliseconds."""
    return session.open_ms - PRE_MARKET_MS, session.close_ms + POST_MARKET_MS


def phase_spans(session: Session) -> dict[str, tuple[int, int]]:
    """The pre market, regular and post market spans of the window, in UTC milliseconds."""
    start, end = window(session)
    return {PRE: (start, session.open_ms), REGULAR: (session.open_ms, session.close_ms), POST: (session.close_ms, end)}


def session_of(official: Official) -> Session:
    """The regular session on the official figures' date, refusing a weekend or a closure."""
    session = sessions.session_on(official.day)
    if session is None:
        raise TapeCheckError(f"{official.day.isoformat()} has no regular session")
    return session


def file_start_ms(path: Path) -> int | None:
    """The UTC start in a recording file's name, or None for a name the recorder did not give."""
    match = FILE_NAME.fullmatch(path.name)
    try:
        start = datetime.strptime(match.group(1), "%Y-%m-%d_%H%M%S") if match else None
    except ValueError:
        start = None
    return None if start is None else int(start.replace(tzinfo=timezone.utc).timestamp()) * MS_PER_SECOND


def session_files(files: Sequence[Path], start_ms: int) -> list[Path]:
    """The files from the last one started at or before start_ms on, because each file runs until the next starts."""
    first = 0
    for i, path in enumerate(files):
        started = file_start_ms(path)
        if started is not None and started <= start_ms:
            first = i
    return list(files[first:])


def recording_from(recording: Path, start_ms: int) -> list[Path]:
    """The recording files that cover start_ms onwards, refusing a path with none."""
    files = session_files([f for f in recording_files([recording]) if f.is_file()], start_ms)
    if not files:
        raise TapeCheckError(f"no recording files at {recording}")
    return files


@dataclass(frozen=True)
class Gap:
    """Lost time from start_ms to end_ms, with what ended the connection before it or the word outage."""

    start_ms: int
    end_ms: int
    cause: str

    @property
    def ms(self) -> int:
        return self.end_ms - self.start_ms


class Connections:
    """Connection spans and the gaps between them from one stream of recorded lines, by the recorder's rules."""

    def __init__(self) -> None:
        self.spans: list[tuple[int, int]] = []
        self.gaps: list[Gap] = []
        self.first_ms: int | None = None
        self._start: int | None = None
        self._last = 0
        self._ended: tuple[int, str] | None = None

    def line(self, kind: str, raw: str, recv_ms: int) -> None:
        """Follow one line, where an open ends any connection still open and any line outside one starts one."""
        if kind == OPEN:
            self._end(NO_CLOSE)
        if self._start is None:
            self._begin(recv_ms)
        self._last = recv_ms
        if kind == CLOSE:
            paused = PAUSED.fullmatch(raw)
            if paused:
                self.gaps.append(Gap(recv_ms - int(paused.group(1)) * MS_PER_SECOND, recv_ms, raw))
            self._end(raw)

    def _begin(self, ms: int) -> None:
        if self.first_ms is None:
            self.first_ms = ms
        if self._ended is not None:
            self.gaps.append(Gap(self._ended[0], ms, self._ended[1]))
        self._start, self._ended = ms, None

    def _end(self, cause: str) -> None:
        if self._start is not None:
            self.spans.append((self._start, self._last))
            self._start, self._ended = None, (self._last, cause)

    @property
    def open(self) -> bool:
        """Whether a connection is still open, with no close marker yet."""
        return self._start is not None

    def finish(self) -> None:
        """End the stream, so a connection still open ends at its last line."""
        self._end(END_OF_RECORDING)

    def disconnected(self, start_ms: int, end_ms: int) -> list[Gap]:
        """The gaps of a finished stream clipped to the window and merged, counting the time before the first line
        and after the end of the last connection, so a missing recording is disconnected too."""
        gaps = list(self.gaps)
        gaps.append(Gap(start_ms, end_ms if self.first_ms is None else self.first_ms, BEFORE_RECORDING))
        if self._ended is not None:
            gaps.append(Gap(self._ended[0], end_ms, self._ended[1]))
        clipped = sorted((Gap(max(g.start_ms, start_ms), min(g.end_ms, end_ms), g.cause) for g in gaps
                          if min(g.end_ms, end_ms) > max(g.start_ms, start_ms)), key=lambda g: g.start_ms)
        merged: list[Gap] = []
        for gap in clipped:
            if merged and gap.start_ms <= merged[-1].end_ms:
                merged[-1] = Gap(merged[-1].start_ms, max(merged[-1].end_ms, gap.end_ms), merged[-1].cause)
            else:
                merged.append(gap)
        return merged

    def connections_in(self, start_ms: int, end_ms: int) -> int:
        return sum(1 for a, b in self.spans if a < end_ms and b >= start_ms)


def connected_stretches(gaps: Sequence[Gap], start_ms: int, end_ms: int) -> list[tuple[int, int]]:
    """The stretches from start_ms to end_ms outside the gaps, which must be merged and in time order."""
    stretches, at = [], start_ms
    for gap in gaps:
        a, b = max(gap.start_ms, start_ms), min(gap.end_ms, end_ms)
        if b <= a:
            continue
        if a > at:
            stretches.append((at, a))
        at = max(at, b)
    if at < end_ms:
        stretches.append((at, end_ms))
    return stretches


def silences(stretches: Sequence[tuple[int, int]], times: Iterable[int]) -> Iterator[tuple[int, int]]:
    """Every silence inside the connected stretches, from a stretch's start or a time in it to the next time or the
    stretch's end, so no silence spans disconnected time, as health counts from the open and from each connect."""
    ordered = sorted(times)
    for a, b in stretches:
        previous = a
        for t in ordered[bisect_left(ordered, a):bisect_left(ordered, b)]:
            if t > previous:
                yield previous, t
            previous = t
        if b > previous:
            yield previous, b


def longest(found: Iterable[tuple[int, int]]) -> tuple[int, int] | None:
    """The longest silence, the earliest of equal ones, or None when there is none."""
    best: tuple[int, int] | None = None
    for a, b in found:
        if best is None or b - a > best[1] - best[0]:
            best = (a, b)
    return best


@dataclass
class SymbolTape:
    """One symbol's prints with vendor time in the window, replayed ones counted apart, and the distinct receive
    stamps of its eligible regular session prints. Shares are of the prints."""

    symbol: str
    volume: dict[str, float] = field(default_factory=lambda: {PRE: 0.0, REGULAR: 0.0, POST: 0.0})
    prints: int = 0
    odd_lots: int = 0
    late: int = 0
    early: int = 0
    replayed: int = 0
    delays: Counter[int] = field(default_factory=Counter)
    first_ms: int | None = None
    last_ms: int | None = None
    close_vendor_ms: int | None = None
    close_price: float | None = None
    eligible_ms: array = field(default_factory=lambda: array("q"))

    @property
    def full(self) -> float:
        return sum(self.volume.values())

    def add(self, recv_ms: int, vendor_ms: int, price: float, size: float, phase: str, kind: str) -> None:
        """Count one non replayed print, keeping the last eligible regular print by vendor time for the close."""
        self.volume[phase] += size
        self.prints += 1
        self.odd_lots += int(size < ROUND_LOT)
        self.late += int(kind == LATE)
        self.early += int(kind == EARLY)
        if self.first_ms is None:
            self.first_ms = recv_ms
        self.last_ms = recv_ms
        if phase == REGULAR:
            self.delays[recv_ms - vendor_ms] += 1
            if kind == ELIGIBLE and (self.close_vendor_ms is None or vendor_ms >= self.close_vendor_ms):
                self.close_vendor_ms, self.close_price = vendor_ms, price

    def renewed(self, recv_ms: int) -> None:
        """Keep the receive stamp of an eligible session print, which live health counts as the symbol's freshness."""
        if not self.eligible_ms or self.eligible_ms[-1] != recv_ms:
            self.eligible_ms.append(recv_ms)


@dataclass
class Tape:
    """What one session's recording shows over the span read, before any official figure. messages holds the
    receive stamp of every message line in the regular session and silence_closes counts the closes there for
    silence."""

    session: Session
    symbols: dict[str, SymbolTape]
    gaps: list[Gap]
    connections: int
    lines: int
    conditions: Counter[str]
    dropped: Counter[str]
    unknown: Counter[str]
    complete: bool
    messages: array
    silence_closes: int

    @property
    def disconnected_ms(self) -> int:
        return sum(gap.ms for gap in self.gaps)

    @property
    def delays(self) -> Counter[int]:
        total: Counter[int] = Counter()
        for tally in self.symbols.values():
            total.update(tally.delays)
        return total

    def stretches(self) -> list[tuple[int, int]]:
        """The connected stretches of the regular session."""
        return connected_stretches(self.gaps, self.session.open_ms, self.session.close_ms)

    def outages(self, feed_stale_ms: int) -> list[Gap]:
        """Whole feed silences of the regular session on an open connection lasting at least feed_stale_ms."""
        return [Gap(a, b, OUTAGE) for a, b in silences(self.stretches(), self.messages) if b - a >= feed_stale_ms]


def _trail(stream: Iterator[tuple[str, str, int, int]], links: Connections, until_ms: int) -> None:
    """Follow the open connection until its close, the next open or until_ms, so a close the recorder writes just after
    waking from a long freeze, holding the pause, is still applied."""
    for kind, raw, recv_ms, _ in stream:
        if kind == OPEN or recv_ms >= until_ms:
            return
        if kind in (MESSAGE, CLOSE):
            links.line(kind, raw, recv_ms)
            if kind == CLOSE:
                return


def measure(lines: Iterable[tuple[str, str, int, int]], session: Session, symbols: Sequence[str],
            rules: PriceRules, span: tuple[int, int] | None = None) -> Tape:
    """Replay recorded (kind, raw, recv_ms, conn_id) lines through the Finnhub parse for one session over span, the
    04:00 to 20:00 window by default, stopping at the first line 10 minutes after it ends and following that line's
    connection a little longer for a pause close, while lines of any other kind, such as a stock process's news
    lines, belong to no connection and are skipped. The tape is complete when such a line exists or the last
    connection closed."""
    start_ms, end_ms = span or window(session)
    adapter = FinnhubAdapter(symbols, REPLAY_TOKEN)
    tallies = {symbol: SymbolTape(symbol) for symbol in symbols}
    links = Connections()
    conditions: Counter[str] = Counter()
    messages = array("q")
    connected_ms: int | None = None
    count = silence_closes = 0
    reached = False
    stream = iter(lines)
    for kind, raw, recv_ms, conn_id in stream:
        if kind not in (OPEN, MESSAGE, CLOSE):
            continue
        links.line(kind, raw, recv_ms)
        if recv_ms >= end_ms + LATE_MARGIN_MS:
            reached = True
            if kind == MESSAGE:
                _trail(stream, links, recv_ms + TRAIL_MS)
            break
        count += 1
        if session.open_ms <= recv_ms < session.close_ms:
            if kind == MESSAGE:
                messages.append(recv_ms)
            elif kind == CLOSE and SILENCE_CLOSE.fullmatch(raw):
                silence_closes += 1
        for event in decode(adapter, kind, raw, recv_ms, conn_id):
            if not isinstance(event, PrintBatch):
                if event.kind == "connected":
                    connected_ms = event.recv_ms
                continue
            for p in event.prints:
                phase = phase_in(session, p.vendor_ms)
                if phase == OTHER:
                    continue
                if is_replayed(p.vendor_ms, connected_ms):
                    tallies[p.symbol].replayed += 1
                    continue
                rule = eligibility(event.recv_ms, p.vendor_ms, p.conditions, rules)
                tallies[p.symbol].add(event.recv_ms, p.vendor_ms, p.price, p.size, phase, rule)
                if rule == ELIGIBLE and in_session(session, event.recv_ms, p.vendor_ms):
                    tallies[p.symbol].renewed(event.recv_ms)
                conditions.update(p.conditions or (NO_CONDITION,))
    complete = reached or not links.open
    links.finish()
    return Tape(session, tallies, links.disconnected(start_ms, end_ms), links.connections_in(start_ms, end_ms),
                count, conditions, Counter(adapter.dropped), Counter(adapter.unknown), complete, messages,
                silence_closes)


def percentile(counts: Counter[int], percent: int) -> int | None:
    """The nearest rank percentile of whole millisecond samples held as counts, or None without samples."""
    total = sum(counts.values())
    if total == 0:
        return None
    rank = max(1, -(-total * percent // 100))
    seen = 0
    for value in sorted(counts):
        seen += counts[value]
        if seen >= rank:
            break
    return value


@dataclass(frozen=True)
class Coverage:
    """Each symbol's ratio r and covered ratio, each phase's covered share and the outages behind it."""

    ratios: dict[str, float]
    covered: dict[str, float] | None
    shares: dict[str, float]
    lost_ms: dict[str, int]
    outages: tuple[Gap, ...]

    @property
    def median(self) -> float:
        return statistics.median(self.ratios.values())

    @property
    def median_covered(self) -> float | None:
        return None if self.covered is None else statistics.median(self.covered.values())

    @property
    def outage_ms(self) -> int:
        return sum(gap.ms for gap in self.outages)


def _overlap(gap: Gap, start_ms: int, end_ms: int) -> int:
    return max(0, min(gap.end_ms, end_ms) - max(gap.start_ms, start_ms))


def coverage(tape: Tape, volumes: dict[str, float], feed_stale_ms: int) -> Coverage:
    """The ratios of a tape read over the whole window against the official volumes, and the covered ratios with
    disconnected and outage time left out, by scaling each phase's volume by its length over its covered length.
    The scale is computed first, so it is exactly 1 and the covered ratio exactly r for a phase with no lost time."""
    outages = tuple(tape.outages(feed_stale_ms))
    lost = [*tape.gaps, *outages]
    spans = phase_spans(tape.session)
    lost_ms = {phase: sum(_overlap(gap, a, b) for gap in lost) for phase, (a, b) in spans.items()}
    covered_ms = {phase: (b - a) - lost_ms[phase] for phase, (a, b) in spans.items()}
    shares = {phase: covered_ms[phase] / (b - a) for phase, (a, b) in spans.items()}
    ratios = {s: tape.symbols[s].full / volumes[s] for s in tape.symbols}
    covered = None
    if all(ms > 0 for ms in covered_ms.values()):
        scale = {phase: (b - a) / covered_ms[phase] for phase, (a, b) in spans.items()}
        covered = {s: sum(t.volume[p] * scale[p] for p in PHASES) / volumes[s] for s, t in tape.symbols.items()}
    return Coverage(ratios, covered, shares, lost_ms, outages)


def clean_reason(tape: Tape, outages: Sequence[Gap] = ()) -> str | None:
    """Why the session is not clean, or None for a full session with no disconnected time from 04:00 to 20:00 and no
    outage in the regular session."""
    if tape.session.early_close:
        return EARLY_CLOSE_REASON
    if tape.disconnected_ms:
        return DISCONNECTED_REASON
    if outages:
        return OUTAGE_REASON
    return None


def within_factor(baseline: float | None, median: float | None, factor: float = STABILITY_FACTOR) -> bool:
    """Whether median lies within factor of baseline either way, edges included, the comparison of criterion 1.1.
    A missing value or one not above zero never passes."""
    if baseline is None or median is None or not (math.isfinite(baseline) and math.isfinite(median)):
        return False
    if baseline <= 0 or median <= 0:
        return False
    return baseline / factor <= median <= baseline * factor


@dataclass(frozen=True)
class Result:
    """One check, its tape, the official figures, the coverage and its run time."""

    tape: Tape
    official: Official
    coverage: Coverage
    seconds: float

    @property
    def reason(self) -> str | None:
        return clean_reason(self.tape, self.coverage.outages)

    @property
    def clean(self) -> bool:
        return self.reason is None


def check(lines: Iterable[tuple[str, str, int, int]], official: Official, symbols: Sequence[str], rules: PriceRules,
          feed_stale_ms: int) -> Result:
    """Measure the recording over the window of the official figures' session and compute its coverage."""
    if set(official.volumes) != set(symbols):
        missing = sorted(set(symbols) - set(official.volumes))
        extra = sorted(set(official.volumes) - set(symbols))
        raise TapeCheckError(f"the official figures must cover exactly the config's symbols, missing "
                             f"{', '.join(missing) or 'none'} and extra {', '.join(extra) or 'none'}")
    started = time.monotonic()
    tape = measure(lines, session_of(official), symbols, rules)
    return Result(tape, official, coverage(tape, official.volumes, feed_stale_ms), time.monotonic() - started)


def stale_ms(feed: StockFeedConfig) -> int:
    """The config's feed_stale_after_seconds in milliseconds, rounded as live health rounds it."""
    return round(feed.feed_stale_after_seconds * MS_PER_SECOND)


# The record


def record_of(result: Result) -> dict[str, Any]:
    """The coverage record of one check, with ratios and seconds as plain JSON numbers and null for an unknown
    covered ratio."""
    tape, cov = result.tape, result.coverage
    return {
        "kind": RECORD_KIND, "session": tape.session.day.isoformat(), "clean": result.clean,
        "reason": result.reason or CLEAN_REASON, "early_close": tape.session.early_close,
        "median_ratio": cov.median, "median_ratio_covered": cov.median_covered,
        "ratios": dict(cov.ratios), "ratios_covered": None if cov.covered is None else dict(cov.covered),
        "disconnected_seconds": tape.disconnected_ms / MS_PER_SECOND,
        "outage_seconds": cov.outage_ms / MS_PER_SECOND, "covered_share": dict(cov.shares),
    }


def record_path(folder: Path, day: date) -> Path:
    return Path(folder) / f"coverage_{day:%Y_%m_%d}.json"


def write_record(folder: Path, record: dict[str, Any]) -> Path:
    """Write a record readable by the owner alone from its creation, through a hidden temporary file replaced into
    place, so a rerun replaces it whole and a crash never leaves half a record under the record's name."""
    path = record_path(folder, date.fromisoformat(record["session"]))
    temporary = path.with_name(f".{path.name}.tmp")
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.chmod(0o600)
    os.replace(temporary, path)
    return path


@dataclass(frozen=True)
class CoverageRecord:
    """One session's coverage record as evaluate reads it."""

    session: date
    clean: bool
    early_close: bool
    median_ratio: float
    median_ratio_covered: float | None
    ratios: dict[str, float]
    ratios_covered: dict[str, float] | None
    disconnected_seconds: float
    outage_seconds: float


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _ratios(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(isinstance(k, str) and _number(v) for k, v in value.items())


def parse_record(data: Any, day: date, name: str) -> CoverageRecord:
    """Validate one record read from JSON, whose session must be the date in its file name."""
    try:
        valid = (data["kind"] == RECORD_KIND and date.fromisoformat(data["session"]) == day
                 and isinstance(data["clean"], bool) and isinstance(data["early_close"], bool)
                 and _number(data["median_ratio"]) and _ratios(data["ratios"])
                 and (data["median_ratio_covered"] is None) == (data["ratios_covered"] is None)
                 and (data["ratios_covered"] is None or (_number(data["median_ratio_covered"])
                                                         and _ratios(data["ratios_covered"])))
                 and _number(data["disconnected_seconds"]) and _number(data["outage_seconds"]))
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise TapeCheckError(f"{name} is not a coverage record")
    return CoverageRecord(day, data["clean"], data["early_close"], float(data["median_ratio"]),
                          None if data["median_ratio_covered"] is None else float(data["median_ratio_covered"]),
                          {k: float(v) for k, v in data["ratios"].items()},
                          None if data["ratios_covered"] is None else
                          {k: float(v) for k, v in data["ratios_covered"].items()},
                          float(data["disconnected_seconds"]), float(data["outage_seconds"]))


def read_records(folder: Path) -> list[CoverageRecord]:
    """Every coverage record of the checks folder in session order, refusing a malformed one. The tape_*.json
    verdicts of the replaced consolidated check are not coverage records and are never read."""
    out = []
    for path in sorted(Path(folder).glob("coverage_*.json")):
        day = _named_day(COVERAGE_NAME, path.name)
        if day is None:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise TapeCheckError(f"{path.name} is not a coverage record") from exc
        out.append(parse_record(data, day, path.name))
    return sorted(out, key=lambda record: record.session)


def not_ignored(path: Path) -> bool:
    """Whether path lies in a git working tree that does not ignore it, such as a tracked config folder. Outside any
    working tree, or without git, nothing is refused."""
    path = Path(path).resolve()
    try:
        found = subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=path.parent, capture_output=True,
                               timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return found.returncode == 1


# Output


def _seconds(ms: int | None) -> str:
    return "" if ms is None else f"{ms / MS_PER_SECOND:.3f}"


def _share(part: int, whole: int) -> str:
    return f"{part / whole:.4f}" if whole else ""


def _ratio(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def report(result: Result, files: int) -> output.Report:
    """The default summary, the clean flag, counts, lost time and timings only, with per symbol ratios, volumes and
    closes and the medians as rows."""
    tape, cov = result.tape, result.coverage
    delays = tape.delays
    median = percentile(delays, 50)
    rule = None if not delays else delay_rule(median, percentile(delays, 95))
    delay_ok = rule is not None and not rule.stop and abs(median) <= rule.max_print_delay_seconds * MS_PER_SECOND
    if rule is None:
        settings: tuple[Any, Any] = (output.Word("not computed"), output.Word("not computed"))
    else:
        settings = (output.Seconds(rule.max_print_delay_seconds), output.Seconds(rule.late_print_seconds))
    rows = []
    for symbol, ratio in cov.ratios.items():
        tally, close = tape.symbols[symbol], result.official.closes[symbol]
        rows.append({
            "symbol": symbol, "ratio": _ratio(ratio), "covered_ratio": _ratio(None if cov.covered is None
                                                                             else cov.covered[symbol]),
            "volume": f"{tally.full:.0f}", "official": f"{result.official.volumes[symbol]:.0f}",
            "pre": f"{tally.volume[PRE]:.0f}", "regular": f"{tally.volume[REGULAR]:.0f}",
            "post": f"{tally.volume[POST]:.0f}",
            "last_eligible": "" if tally.close_price is None else f"{tally.close_price:.4f}",
            "official_close": f"{close:.4f}",
            "close_diff_pct": "" if tally.close_price is None else f"{100 * (tally.close_price / close - 1):+.3f}",
        })
    rows.append({"symbol": "median", "ratio": _ratio(cov.median), "covered_ratio": _ratio(cov.median_covered)})
    return output.Report("tape coverage", (
        ("clean session", result.clean),
        ("reason", output.Word(result.reason or CLEAN_REASON)),
        ("symbols", output.Count(len(cov.ratios))),
        ("symbols with prints", output.Count(sum(1 for tally in tape.symbols.values() if tally.prints))),
        ("covered ratio computed", cov.covered is not None),
        ("disconnected time", output.Seconds(tape.disconnected_ms / MS_PER_SECOND)),
        ("outage time", output.Seconds(cov.outage_ms / MS_PER_SECOND)),
        ("connections", output.Count(tape.connections)),
        ("silence closes", output.Count(tape.silence_closes)),
        ("delay verdict", delay_ok),
        ("max_print_delay_seconds", settings[0]),
        ("late_print_seconds", settings[1]),
        ("files", output.Count(files)),
        ("lines", output.Count(tape.lines)),
        ("run time", output.Seconds(result.seconds)),
    ), tuple(rows))


def _intervals(gaps: Iterable[Gap]) -> tuple[dict[str, str], ...]:
    return tuple({"from": iso(g.start_ms), "to": iso(g.end_ms), "seconds": _seconds(g.ms), "after": g.cause}
                 for g in gaps)


def detail_reports(result: Result) -> list[output.Report]:
    """The owner's further tables, print quality per symbol, covered time per phase, the condition histogram, the
    disconnected and outage intervals and the parse counts."""
    tape, cov = result.tape, result.coverage
    quality = []
    for tally in tape.symbols.values():
        seen, delays = tally.prints + tally.replayed, tally.delays
        quality.append({
            "symbol": tally.symbol, "odd_lot_share": _share(tally.odd_lots, tally.prints),
            "late_share": _share(tally.late, tally.prints), "early_share": _share(tally.early, tally.prints),
            "replayed_share": _share(tally.replayed, seen), "delay_p50_s": _seconds(percentile(delays, 50)),
            "delay_p95_s": _seconds(percentile(delays, 95)), "delay_max_s": _seconds(percentile(delays, 100)),
            "first_print": "" if tally.first_ms is None else iso(tally.first_ms),
            "last_print": "" if tally.last_ms is None else iso(tally.last_ms),
        })
    covered = tuple({"phase": phase, "seconds": _seconds(b - a), "lost_s": _seconds(cov.lost_ms[phase]),
                     "covered_share": f"{cov.shares[phase]:.6f}"}
                    for phase, (a, b) in phase_spans(tape.session).items())
    total = sum(tape.conditions.values())
    histogram = [{"condition": code, "prints": n, "share": _share(n, total)}
                 for code, n in sorted(tape.conditions.items(), key=lambda item: (-item[1], item[0]))]
    parse = [*({"what": f"prints dropped as {reason}", "count": n} for reason, n in sorted(tape.dropped.items())),
             *({"what": f"messages of type {name}", "count": n} for name, n in sorted(tape.unknown.items()))]
    return [output.Report("tape prints", (), tuple(quality)),
            output.Report("covered time", (), covered),
            output.Report("condition histogram", (), tuple(histogram)),
            output.Report("disconnected intervals", (), _intervals(tape.gaps)),
            output.Report("outage intervals", (), _intervals(cov.outages)),
            output.Report("parse counts", (), tuple(parse))]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The tape-check arguments, for this module's own parser or the btc_survivor_v2 subcommand."""
    parser.add_argument("recording", metavar="RECORDING", type=Path,
                        help="the raw recording folder, such as data/v2/raw_stocks, or one recording file")
    parser.add_argument("official", metavar="OFFICIAL", type=Path,
                        help="the hand filled data/v2/checks/official_YYYY_MM_DD.toml, whose folder also keeps "
                             "the coverage records")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"the stock run config giving the symbols, feed_stale_after_seconds, "
                             f"late_print_seconds and price_excluded_conditions (default {DEFAULT_CONFIG})")
    output.add_detail_flag(parser)


def run(args: argparse.Namespace) -> int:
    """Report one session's coverage and write its record, exiting 0 for a clean session and 1 for any other. An
    official file that git would track, or a recording that does not reach past the window, is refused before any
    record."""
    cfg = load_stock_config(args.config)
    official = load_official(args.official)
    if not_ignored(args.official):
        raise TapeCheckError(NOT_IGNORED)
    files = recording_from(args.recording, window(session_of(official))[0])
    feed = cfg.stock_feed
    result = check(read_recording(files), official, feed.symbols, PriceRules.from_config(feed), stale_ms(feed))
    if not result.tape.complete:
        raise TapeCheckError(INCOMPLETE)
    write_record(args.official.parent, record_of(result))
    output.emit(report(result, len(files)), args.detail)
    if args.detail:
        for extra in detail_reports(result):
            output.emit(extra, True)
    return 0 if result.clean else 1


DESCRIPTION = ("Print the tape coverage report of PLAN_V2 section 4.9 from a raw recording and the official figures "
               "alone, and write its record beside the official file. By default it prints only verdicts, counts and "
               "timings.")


def main(argv: list[str] | None = None) -> int:
    """Run tape-check, printing an expected failure as one error line with exit code 2."""
    parser = argparse.ArgumentParser(prog="btc_survivor_v2 tape-check", description=DESCRIPTION)
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (TapeCheckError, ConfigError, CalendarError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
