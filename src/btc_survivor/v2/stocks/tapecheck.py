"""The consolidated tape check of PLAN_V2 section 4.9, from a raw recording and the owner's official figures alone.

It reads only the raw recording that scripts/v2_record_stocks.py writes, replaying finnhub.parse and the print
classification of prints.py, so it needs no run database. Per symbol it sums the volume of every print whose vendor
time falls from 04:00 to 20:00 New York time on the session date, split into pre market, regular and post market,
leaving out replayed prints, which repeat what an earlier connection may already have carried. Late and early prints
count, because they are real prints whose timing alone is doubtful. Eligibility, behind the late share and the last
eligible print before 16:00 that meets the official close, follows the run config's late_print_seconds and
price_excluded_conditions. The ratio r of the full day volume to the official consolidated volume sets the verdict.

* Consolidated when the median r is from 0.85 to 1.15 and no symbol is below 0.70.
* Not consolidated when the median r is below 0.50, or when an inconclusive clean session follows another one.
* Inconclusive otherwise, where a median above 1.15 means the official figure was a single venue's.
* Not a clean session when the session is an early close or the recording shows any disconnected time from 04:00 to
  20:00 New York time, whatever the ratios say.

Each symbol falls in one band, under half below 0.50, under floor below 0.70, low below 0.85, in band up to 1.15 and
above band beyond it.

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

Earlier verdicts. Each check writes tape_YYYY_MM_DD.json beside the official file, holding only the session date, the
clean flag, the verdict from the ratios alone and the final verdict. The record of the latest earlier clean session
is the previous check, because an inconclusive verdict repeats the check on the next clean session, so when its ratio
verdict and this one are both inconclusive the verdict becomes not consolidated. A rerun for the same date replaces
its record and never reads it.

The delay rule of section 4.5, from health.delay_rule. Delays are receive stamp minus vendor time of the non replayed
prints whose vendor time falls in the regular session, with nearest rank percentiles over whole milliseconds.
max_print_delay_seconds becomes the larger of 2 and ceil(1.5 * p95) in seconds, capped at 5, with late_print_seconds
twice that, both timings of this system. The delay verdict passes when the median is within max_print_delay_seconds
either way, as price health judges it live, so a feed more than 5 seconds behind fails it, and so does a local clock
that runs ahead.

By default the command prints only the verdicts, the band counts, the clean flag, disconnected time, the delay
verdict, the two settings and counts and timings of this system. --detail adds per symbol ratios and volumes, delay
percentiles, the odd lot, late, early and replayed shares, the close comparison, the condition histogram and the
disconnected intervals.
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
from btc_survivor.v2.config import ConfigError, load_stock_config
from btc_survivor.v2.sessions import CalendarError, Session
from btc_survivor.v2.stocks.finnhub import FinnhubAdapter, PrintBatch
from btc_survivor.v2.stocks.health import delay_rule
from btc_survivor.v2.stocks.prints import (EARLY, ELIGIBLE, LATE, OTHER, POST, POST_MARKET_MS, PRE, PRE_MARKET_MS,
                                           REGULAR, PriceRules, eligibility, is_replayed, phase_in)

CONSOLIDATED, NOT_CONSOLIDATED, INCONCLUSIVE = "consolidated", "not consolidated", "inconclusive"
NOT_CLEAN = "not a clean session"
HALF, FLOOR, LOW_EDGE, HIGH_EDGE = 0.50, 0.70, 0.85, 1.15
UNDER_HALF, UNDER_FLOOR, LOW, IN_BAND, ABOVE_BAND = "under half", "under floor", "low", "in band", "above band"
BANDS = (UNDER_HALF, UNDER_FLOOR, LOW, IN_BAND, ABOVE_BAND)
SECOND_DISPLAY_TOLERANCE = 0.05
ROUND_LOT = 100
# Prints received this long after 20:00 still count by their vendor time, and reading stops after it.
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
PAUSED = re.compile(r"process paused for (\d+) s")
OFFICIAL_NAME = re.compile(r"official_(\d{4})_(\d{2})_(\d{2})\.toml")
RECORD_NAME = re.compile(r"tape_(\d{4})_(\d{2})_(\d{2})\.json")
FILE_NAME = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{6})\.jsonl\.gz")
DEFAULT_CONFIG = Path("configs/v2/s1_prices.toml")


INCOMPLETE = ("the recording does not reach 10 minutes past the window and its last connection has no close marker, "
              "so the recorder may still be running or was killed, and the check needs a later line or the close "
              "that SIGTERM writes")
NOT_IGNORED = ("the official file must sit in the gitignored data folder, such as data/v2/checks, because its figures "
               "and the tape records beside it never enter the repository")


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


def load_official(path: Path) -> Official:
    """Read official_YYYY_MM_DD.toml, taking the session date from its name."""
    path = Path(path)
    match = OFFICIAL_NAME.fullmatch(path.name)
    try:
        day = date(*(int(part) for part in match.groups())) if match else None
    except ValueError:
        day = None
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


@dataclass(frozen=True)
class Gap:
    """Disconnected time from start_ms to end_ms, with what ended the connection before it."""

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


@dataclass
class SymbolTape:
    """One symbol's prints with vendor time in the window, replayed ones counted apart. Shares are of the prints."""

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


@dataclass
class Tape:
    """What one session's recording shows, before any official figure."""

    session: Session
    symbols: dict[str, SymbolTape]
    gaps: list[Gap]
    connections: int
    lines: int
    conditions: Counter[str]
    dropped: Counter[str]
    unknown: Counter[str]
    complete: bool

    @property
    def disconnected_ms(self) -> int:
        return sum(gap.ms for gap in self.gaps)

    @property
    def delays(self) -> Counter[int]:
        total: Counter[int] = Counter()
        for tally in self.symbols.values():
            total.update(tally.delays)
        return total


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
            rules: PriceRules) -> Tape:
    """Replay recorded (kind, raw, recv_ms, conn_id) lines through the Finnhub parse for one session's window,
    stopping at the first line 10 minutes after it ends, which still shows whether the connection lasted, after
    following that connection a little longer for a pause close. Lines of any other kind, such as the news lines a
    stock process records, belong to no connection and are skipped. The tape is complete when such a line exists or
    the last connection closed."""
    start_ms, end_ms = window(session)
    adapter = FinnhubAdapter(symbols, REPLAY_TOKEN)
    tallies = {symbol: SymbolTape(symbol) for symbol in symbols}
    links = Connections()
    conditions: Counter[str] = Counter()
    connected_ms: int | None = None
    count = 0
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
                conditions.update(p.conditions or (NO_CONDITION,))
    complete = reached or not links.open
    links.finish()
    return Tape(session, tallies, links.disconnected(start_ms, end_ms), links.connections_in(start_ms, end_ms),
                count, conditions, Counter(adapter.dropped), Counter(adapter.unknown), complete)


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


def band(ratio: float) -> str:
    if ratio < HALF:
        return UNDER_HALF
    if ratio < FLOOR:
        return UNDER_FLOOR
    if ratio < LOW_EDGE:
        return LOW
    return IN_BAND if ratio <= HIGH_EDGE else ABOVE_BAND


def ratio_verdict(ratios: Sequence[float]) -> tuple[str, str]:
    """The verdict of section 4.9 from the ratios alone, with its reason."""
    median = statistics.median(ratios)
    if median < HALF:
        return NOT_CONSOLIDATED, "median ratio under half"
    if LOW_EDGE <= median <= HIGH_EDGE and min(ratios) >= FLOOR:
        return CONSOLIDATED, "median ratio in band and no symbol under floor"
    if median > HIGH_EDGE:
        return INCONCLUSIVE, "median ratio above band so the reference looks single venue"
    if median >= LOW_EDGE:
        return INCONCLUSIVE, "a symbol under floor"
    return INCONCLUSIVE, "median ratio between half and band"


def clean_reason(tape: Tape) -> str | None:
    """Why the session is not clean, or None for a full session with no disconnected time from 04:00 to 20:00."""
    if tape.session.early_close:
        return "an early close is never a clean session"
    if tape.disconnected_ms:
        return "disconnected time in the window"
    return None


def final_verdict(ratio: tuple[str, str], unclean: str | None, previous: str | None) -> tuple[str, str]:
    """The verdict and reason after the clean session rule and the previous clean session's ratio verdict."""
    if unclean is not None:
        return NOT_CLEAN, unclean
    if ratio[0] == INCONCLUSIVE and previous == INCONCLUSIVE:
        return NOT_CONSOLIDATED, "two inconclusive clean sessions in a row"
    return ratio


def record_path(folder: Path, day: date) -> Path:
    return Path(folder) / f"tape_{day:%Y_%m_%d}.json"


def previous_verdict(folder: Path, day: date) -> str | None:
    """The ratio verdict of the latest earlier clean session in the tape records of folder, or None."""
    latest: tuple[date, str] | None = None
    for path in sorted(Path(folder).glob("tape_*.json")):
        if not RECORD_NAME.fullmatch(path.name):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            session, clean, verdict = date.fromisoformat(record["session"]), record["clean"], record["ratio_verdict"]
        except (ValueError, KeyError, TypeError) as exc:
            raise TapeCheckError(f"{path.name} is not a tape check record") from exc
        if session < day and clean is True and (latest is None or session > latest[0]):
            latest = (session, verdict)
    return None if latest is None else latest[1]


def write_record(folder: Path, day: date, clean: bool, ratio: str, verdict: str) -> Path:
    """Write this check's record, verdicts only and readable by the owner alone from its creation, replacing a
    rerun's."""
    path = record_path(folder, day)
    record = {"session": day.isoformat(), "clean": clean, "ratio_verdict": ratio, "verdict": verdict}
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    path.chmod(0o600)
    return path


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


@dataclass(frozen=True)
class Result:
    """One check, its tape, ratios, the verdict from the ratios alone, the final verdict and its run time."""

    tape: Tape
    official: Official
    ratios: dict[str, float]
    ratio: tuple[str, str]
    verdict: tuple[str, str]
    seconds: float

    @property
    def clean(self) -> bool:
        return clean_reason(self.tape) is None


def check(lines: Iterable[tuple[str, str, int, int]], official: Official, symbols: Sequence[str], rules: PriceRules,
          previous: str | None = None) -> Result:
    """Measure the recording for the official figures' session and judge it, given the previous clean verdict."""
    if set(official.volumes) != set(symbols):
        missing = sorted(set(symbols) - set(official.volumes))
        extra = sorted(set(official.volumes) - set(symbols))
        raise TapeCheckError(f"the official figures must cover exactly the config's symbols, missing "
                             f"{', '.join(missing) or 'none'} and extra {', '.join(extra) or 'none'}")
    started = time.monotonic()
    tape = measure(lines, session_of(official), symbols, rules)
    ratios = {s: tape.symbols[s].full / official.volumes[s] for s in symbols}
    ratio = ratio_verdict(list(ratios.values()))
    verdict = final_verdict(ratio, clean_reason(tape), previous)
    return Result(tape, official, ratios, ratio, verdict, time.monotonic() - started)


def _seconds(ms: int | None) -> str:
    return "" if ms is None else f"{ms / MS_PER_SECOND:.3f}"


def _share(part: int, whole: int) -> str:
    return f"{part / whole:.4f}" if whole else ""


def report(result: Result, files: int) -> output.Report:
    """The default summary, verdicts, counts and timings only, with per symbol volumes and closes as rows."""
    tape = result.tape
    bands = Counter(band(r) for r in result.ratios.values())
    delays = tape.delays
    median = percentile(delays, 50)
    rule = None if not delays else delay_rule(median, percentile(delays, 95))
    delay_ok = rule is not None and not rule.stop and abs(median) <= rule.max_print_delay_seconds * MS_PER_SECOND
    if rule is None:
        settings: tuple[Any, Any] = (output.Word("not computed"), output.Word("not computed"))
    else:
        settings = (output.Seconds(rule.max_print_delay_seconds), output.Seconds(rule.late_print_seconds))
    rows = []
    for symbol, ratio in result.ratios.items():
        tally, close = tape.symbols[symbol], result.official.closes[symbol]
        rows.append({
            "symbol": symbol, "band": band(ratio), "ratio": f"{ratio:.4f}", "volume": f"{tally.full:.0f}",
            "official": f"{result.official.volumes[symbol]:.0f}", "pre": f"{tally.volume[PRE]:.0f}",
            "regular": f"{tally.volume[REGULAR]:.0f}", "post": f"{tally.volume[POST]:.0f}",
            "last_eligible": "" if tally.close_price is None else f"{tally.close_price:.4f}",
            "official_close": f"{close:.4f}",
            "close_diff_pct": "" if tally.close_price is None else f"{100 * (tally.close_price / close - 1):+.3f}",
        })
    return output.Report("tape check", (
        ("verdict", output.Word(result.verdict[0])),
        ("reason", output.Word(result.verdict[1])),
        ("ratio verdict", output.Word(result.ratio[0])),
        *((f"symbols {name}", output.Count(bands[name])) for name in BANDS),
        ("clean session", result.clean),
        ("disconnected time", output.Seconds(tape.disconnected_ms / MS_PER_SECOND)),
        ("connections", output.Count(tape.connections)),
        ("delay verdict", delay_ok),
        ("max_print_delay_seconds", settings[0]),
        ("late_print_seconds", settings[1]),
        ("files", output.Count(files)),
        ("lines", output.Count(tape.lines)),
        ("run time", output.Seconds(result.seconds)),
    ), tuple(rows))


def detail_reports(result: Result) -> list[output.Report]:
    """The owner's further tables, print quality per symbol, the condition histogram, the disconnected intervals and
    the parse counts."""
    tape = result.tape
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
    total = sum(tape.conditions.values())
    histogram = [{"condition": code, "prints": n, "share": _share(n, total)}
                 for code, n in sorted(tape.conditions.items(), key=lambda item: (-item[1], item[0]))]
    gaps = [{"from": iso(g.start_ms), "to": iso(g.end_ms), "seconds": _seconds(g.ms), "after": g.cause}
            for g in tape.gaps]
    parse = [*({"what": f"prints dropped as {reason}", "count": n} for reason, n in sorted(tape.dropped.items())),
             *({"what": f"messages of type {name}", "count": n} for name, n in sorted(tape.unknown.items()))]
    return [output.Report("tape prints", (), tuple(quality)),
            output.Report("condition histogram", (), tuple(histogram)),
            output.Report("disconnected intervals", (), tuple(gaps)),
            output.Report("parse counts", (), tuple(parse))]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The tape-check arguments, for this module's own parser or the btc_survivor_v2 subcommand."""
    parser.add_argument("recording", metavar="RECORDING", type=Path,
                        help="the raw recording folder, such as data/v2/raw_stocks, or one recording file")
    parser.add_argument("official", metavar="OFFICIAL", type=Path,
                        help="the hand filled data/v2/checks/official_YYYY_MM_DD.toml, whose folder also keeps "
                             "the tape check records")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"the stock run config giving the symbols, late_print_seconds and "
                             f"price_excluded_conditions (default {DEFAULT_CONFIG})")
    output.add_detail_flag(parser)


def run(args: argparse.Namespace) -> int:
    """Check one session, write its record and print the report, exiting 0 only when consolidated. An official file
    that git would track, or a recording that does not reach past the window, is refused before any record."""
    cfg = load_stock_config(args.config)
    official = load_official(args.official)
    if not_ignored(args.official):
        raise TapeCheckError(NOT_IGNORED)
    start_ms = window(session_of(official))[0]
    files = session_files([f for f in recording_files([args.recording]) if f.is_file()], start_ms)
    if not files:
        raise TapeCheckError(f"no recording files at {args.recording}")
    previous = previous_verdict(args.official.parent, official.day)
    result = check(read_recording(files), official, cfg.stock_feed.symbols, PriceRules.from_config(cfg.stock_feed),
                   previous)
    if not result.tape.complete:
        raise TapeCheckError(INCOMPLETE)
    write_record(args.official.parent, official.day, result.clean, result.ratio[0], result.verdict[0])
    output.emit(report(result, len(files)), args.detail)
    if args.detail:
        for extra in detail_reports(result):
            output.emit(extra, True)
    return 0 if result.verdict[0] == CONSOLIDATED else 1


DESCRIPTION = ("Print the tape verdict of PLAN_V2 section 4.9 from a raw recording and the official figures alone. By "
               "default it prints only verdicts, counts and timings.")


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
