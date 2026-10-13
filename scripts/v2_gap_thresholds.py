"""Print the stale thresholds of PLAN_V2 section 4.5 from one regular session of a raw stock recording, by the rule the
owner wrote on 12 October 2026 for Finnhub's partial tape. It never writes anything and never prints a price, a size
or a message body.

    symbol_stale_after_seconds = 2.5 times the longest per symbol gap in session, rounded up to a multiple of 30 s
    feed_stale_after_seconds   = 5 times the longest whole feed gap in session, rounded up to a multiple of 5 s

A symbol gap is a silence between the receive stamps of that symbol's eligible prints, counted as live health counts
a symbol's freshness, from the open or an eligible print to the next eligible print or the close, so it runs across
reconnects, since live health restarts only the whole feed clock at a connect. A print counts when it was received in
the regular session with its vendor time there too, not replayed, and eligible by the config's late_print_seconds and
price_excluded_conditions. The time the recording was disconnected or paused inside a symbol gap is left out of it,
because no print could be seen then, so a recorder that closed a silent connection and reconnected never splits a
symbol's silence in two. A whole feed gap is a silence between message lines of any kind, pings included, which
counts from the open and from each connect, as live health restarts the feed clock there, and never spans
disconnected time.

A recorder closes a connection silent for its own feed_stale_after_seconds, so the longest whole feed gap it records
is at most about that setting plus its one second tick. The silence closes count shows how often that happened, and
a pause of up to 10 seconds that the recorder survived shows as a silence it cannot tell apart from the feed's. The
report also gives the longest symbol gap inside one connection and the disconnects inside the longest symbol gap, so
the owner sees how much the reconnects mattered, and --detail gives each longest gap's times, to read a halt against
Nasdaq's halt list before the values go into the configs.

By default it prints both thresholds, the longest gaps behind them, whether each threshold fits this script's sanity
bounds, 300 s for a symbol and 60 s for the feed, past which the owner decides, and the config's cross check, with
counts and timings of this system. --detail adds each symbol's longest gap with its times. It exits 0 when both
thresholds fit, 1 when one does not and 2 when it cannot measure the session.

    python scripts/v2_gap_thresholds.py data/v2/raw_stocks 2026-10-12 [--config configs/v2/s1_prices.toml] [--detail]
"""

from __future__ import annotations

import argparse
import sys
import time
from bisect import bisect_right
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import accumulate, pairwise
from pathlib import Path

from btc_survivor.clock import MS_PER_SECOND, iso
from btc_survivor.recorder import read_recording
from btc_survivor.v2 import output, sessions
from btc_survivor.v2.config import ConfigError, load_stock_config
from btc_survivor.v2.sessions import CalendarError, Session
from btc_survivor.v2.stocks import tapecheck
from btc_survivor.v2.stocks.prints import PriceRules
from btc_survivor.v2.stocks.tapecheck import Gap, Tape, TapeCheckError

DEFAULT_CONFIG = Path("configs/v2/s1_prices.toml")
# The rule, as a multiple of the longest gap and a step in seconds to round up to.
SYMBOL_TIMES_TWO, SYMBOL_STEP_SECONDS = 5, 30
FEED_TIMES, FEED_STEP_SECONDS = 5, 5
# This script's sanity bounds, past which the owner decides before any value goes into a run config.
SYMBOL_MAX_SECONDS, FEED_MAX_SECONDS = 300, 60
WHOLE_FEED = "whole feed"

INCOMPLETE = ("the recording does not reach 10 minutes past the close and its last connection has no close marker, "
              "so the recorder may still be running or was killed, and the rule needs a later line or the close that "
              "SIGTERM writes")
NO_CONNECTED_TIME = "the recording holds no connected time in the regular session"


def symbol_threshold(gap_ms: int) -> int:
    """2.5 times the gap rounded up to a multiple of 30 s, in whole seconds, in integer arithmetic so that no float
    rounding moves a boundary."""
    step_ms = SYMBOL_STEP_SECONDS * MS_PER_SECOND
    return SYMBOL_STEP_SECONDS * -(-SYMBOL_TIMES_TWO * gap_ms // (2 * step_ms))


def feed_threshold(gap_ms: int) -> int:
    """5 times the gap rounded up to a multiple of 5 s, in whole seconds, in integer arithmetic."""
    step_ms = FEED_STEP_SECONDS * MS_PER_SECOND
    return FEED_STEP_SECONDS * -(-FEED_TIMES * gap_ms // step_ms)


@dataclass(frozen=True)
class Longest:
    """The longest silence of one symbol or of the whole feed, from and to, its length with disconnected time left
    out, how many stamps it was measured between and how many disconnects fell inside it."""

    name: str
    stamps: int
    gap: tuple[int, int]
    ms: int
    disconnects: int = 0


def connected_time(stretches: Sequence[tuple[int, int]]) -> Callable[[int], int]:
    """A function giving the connected time from the first stretch's start to a time, so the connected time between
    two times is the difference of its values. The stretches must be in time order and not overlap."""
    starts = [a for a, _ in stretches]
    before = list(accumulate((b - a for a, b in stretches), initial=0))

    def upto(t: int) -> int:
        i = bisect_right(starts, t) - 1
        if i < 0:
            return 0
        a, b = stretches[i]
        return before[i] + min(t, b) - a

    return upto


def symbol_gap(name: str, stretches: Sequence[tuple[int, int]], gaps: Sequence[Gap], stamps: Iterable[int],
               session: Session) -> Longest:
    """One symbol's longest silence as live health counts it, from the open or an eligible print to the next one or
    the close across reconnects, with the disconnected time inside it left out, the earliest of equal ones."""
    upto = connected_time(stretches)
    inside = sorted(t for t in stamps if session.open_ms < t < session.close_ms)
    points = [session.open_ms, *inside, session.close_ms]
    a, b = max(pairwise(points), key=lambda ab: upto(ab[1]) - upto(ab[0]))
    disconnects = sum(1 for g in gaps if g.start_ms < b and g.end_ms > a)
    return Longest(name, len(inside), (a, b), upto(b) - upto(a), disconnects)


def feed_gap(stretches: Sequence[tuple[int, int]], messages: Sequence[int]) -> Longest:
    """The longest silence of the whole feed inside one connection, as live health restarts its clock at a connect."""
    found = tapecheck.longest(tapecheck.silences(stretches, messages))
    return Longest(WHOLE_FEED, len(messages), found, found[1] - found[0])


def within_one_connection(stretches: Sequence[tuple[int, int]], tape: Tape) -> int:
    """The longest silence between one symbol's eligible prints inside one connection, the measure a reconnect cuts,
    reported beside the rule's measure."""
    found = [tapecheck.longest(tapecheck.silences(stretches, t.eligible_ms)) for t in tape.symbols.values()]
    return max((b - a for a, b in filter(None, found)), default=0)


@dataclass(frozen=True)
class Thresholds:
    """Both thresholds by the rule, the longest gaps behind them and what the session showed."""

    session: Session
    feed: Longest
    symbols: dict[str, Longest]
    within_ms: int
    disconnected_ms: int
    connections: int
    silence_closes: int
    lines: int

    @property
    def worst(self) -> Longest:
        """The symbol with the longest gap, the first in config order of equal ones."""
        return max(self.symbols.values(), key=lambda found: found.ms)

    @property
    def symbol_seconds(self) -> int:
        return symbol_threshold(self.worst.ms)

    @property
    def feed_seconds(self) -> int:
        return feed_threshold(self.feed.ms)

    @property
    def fits(self) -> bool:
        """Whether both thresholds fit the sanity bounds and the symbol one is at least the feed one, as the config's
        cross check needs."""
        return (self.symbol_seconds <= SYMBOL_MAX_SECONDS and self.feed_seconds <= FEED_MAX_SECONDS
                and self.symbol_seconds >= self.feed_seconds)


def thresholds(tape: Tape) -> Thresholds:
    """The rule applied to a tape read over its regular session, refusing one with no connected time there."""
    stretches = tape.stretches()
    if not stretches:
        raise TapeCheckError(NO_CONNECTED_TIME)
    symbols = {s: symbol_gap(s, stretches, tape.gaps, t.eligible_ms, tape.session) for s, t in tape.symbols.items()}
    disconnected = sum(gap.ms for gap in tape.gaps)
    return Thresholds(tape.session, feed_gap(stretches, tape.messages), symbols, within_one_connection(stretches, tape),
                      disconnected, tape.connections, tape.silence_closes, tape.lines)


def measure_session(recording: Path, day: date, config: Path) -> tuple[Thresholds, int]:
    """Read the recording over the regular session on day with the config's symbols and eligibility rules, refusing a
    day with no session and an incomplete recording, and return the thresholds and the number of files read."""
    session = sessions.session_on(day)
    if session is None:
        raise TapeCheckError(f"{day.isoformat()} has no regular session")
    feed = load_stock_config(config).stock_feed
    files = tapecheck.recording_from(recording, session.open_ms)
    tape = tapecheck.measure(read_recording(files), session, feed.symbols, PriceRules.from_config(feed),
                             span=(session.open_ms, session.close_ms))
    if not tape.complete:
        raise TapeCheckError(INCOMPLETE)
    return thresholds(tape), len(files)


def report(found: Thresholds, files: int, seconds: float) -> output.Report:
    """The thresholds, the longest gaps, the bound and cross check verdicts and counts, with each symbol's longest
    gap and its times as rows."""
    rows = [{"name": item.name, "stamps": item.stamps, "longest_gap_s": f"{item.ms / MS_PER_SECOND:.3f}",
             "from": iso(item.gap[0]), "to": iso(item.gap[1]),
             "rule_s": feed_threshold(item.ms) if item.name == WHOLE_FEED else symbol_threshold(item.ms),
             "disconnects": item.disconnects}
            for item in (*found.symbols.values(), found.feed)]
    return output.Report("gap thresholds", (
        ("symbol_stale_after_seconds", output.Seconds(found.symbol_seconds)),
        ("feed_stale_after_seconds", output.Seconds(found.feed_seconds)),
        ("longest symbol gap", output.Seconds(found.worst.ms / MS_PER_SECOND)),
        ("longest whole feed gap", output.Seconds(found.feed.ms / MS_PER_SECOND)),
        ("longest symbol gap inside one connection", output.Seconds(found.within_ms / MS_PER_SECOND)),
        ("disconnects inside the longest symbol gap", output.Count(found.worst.disconnects)),
        ("symbol threshold within the bound", found.symbol_seconds <= SYMBOL_MAX_SECONDS),
        ("feed threshold within the bound", found.feed_seconds <= FEED_MAX_SECONDS),
        ("symbol threshold at least the feed threshold", found.symbol_seconds >= found.feed_seconds),
        ("full session", not found.session.early_close),
        ("symbols without an eligible print", output.Count(sum(1 for s in found.symbols.values() if not s.stamps))),
        ("disconnected time left out", output.Seconds(found.disconnected_ms / MS_PER_SECOND)),
        ("connections", output.Count(found.connections)),
        ("silence closes", output.Count(found.silence_closes)),
        ("files", output.Count(files)),
        ("lines", output.Count(found.lines)),
        ("run time", output.Seconds(seconds)),
    ), tuple(rows))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print the stale thresholds of PLAN_V2 section 4.5 from one regular "
                                                 "session of a raw stock recording. By default it prints only "
                                                 "thresholds, verdicts, counts and timings.")
    parser.add_argument("recording", metavar="RECORDING", type=Path,
                        help="the raw recording folder, such as data/v2/raw_stocks, or one recording file")
    parser.add_argument("day", metavar="DATE", type=date.fromisoformat,
                        help="the New York session date, such as 2026-10-12")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"the stock run config giving the symbols, late_print_seconds and "
                             f"price_excluded_conditions (default {DEFAULT_CONFIG})")
    output.add_detail_flag(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Measure one session and print its thresholds, printing an expected failure as one error line with exit code
    2."""
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    try:
        found, files = measure_session(args.recording, args.day, args.config)
    except (TapeCheckError, ConfigError, CalendarError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output.emit(report(found, files, time.monotonic() - started), args.detail)
    return 0 if found.fits else 1


if __name__ == "__main__":
    sys.exit(main())
