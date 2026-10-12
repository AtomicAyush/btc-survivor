"""The polling channels and their cadences, PLAN_V2 section 7.2.

The schedule reads America/New_York through zoneinfo only to pick a cadence, and works in UTC milliseconds otherwise.
Every weekday counts as a session and filing day, holidays included, so the daily windows apply on every weekday and
never on a weekend. A fast window around a scheduled release in [calendar] overrides the daily cadence. A cadence of
None means the channel is dormant, which only fed_press_monetary is, outside its FOMC statement windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from btc_survivor.clock import MS_PER_MINUTE, MS_PER_SECOND
from btc_survivor.v2.config import NewsConfig
from btc_survivor.v2.sessions import self_test

FINNHUB, EDGAR, FED, BLS = "finnhub", "edgar", "fed", "bls"


@dataclass(frozen=True)
class Channel:
    """One URL pattern on its own schedule. source is as news_items stores it, symbol names the company of a
    per symbol channel, and feed names the Fed or BLS feed."""

    name: str
    source: str
    symbol: str | None = None
    feed: str | None = None


@dataclass(frozen=True)
class _Rule:
    """Weekday windows as (start, end, seconds), innermost first, the cadence outside them, and fast windows as UTC
    [start, end) in milliseconds with their cadence."""

    daily: tuple[tuple[time, time, int], ...]
    default: int | None
    events: tuple[tuple[int, int], ...] = ()
    event_seconds: int = 0


def _hhmm(text: str) -> time:
    return time.fromisoformat(text)


def _window(pair: tuple[str, str], seconds: int) -> tuple[time, time, int]:
    return _hhmm(pair[0]), _hhmm(pair[1]), seconds


class Schedule:
    """The enabled channels of a news config and the cadence each one polls at, at any instant."""

    def __init__(self, config: NewsConfig):
        self.zone = ZoneInfo(config.news.schedule_tz)
        self_test(self.zone)
        channels: list[Channel] = []
        rules: dict[str, _Rule] = {}

        def add(channel: Channel, rule: _Rule) -> None:
            channels.append(channel)
            rules[channel.name] = rule

        fh, ed, fed, bls, cal = config.finnhub, config.edgar, config.fed, config.bls, config.calendar
        if fh.enabled:
            finnhub = _Rule((_window(fh.regular_window_et, fh.regular_poll_seconds),
                             _window(fh.extended_window_et, fh.extended_poll_seconds)), fh.overnight_poll_seconds)
            for symbol in fh.company_symbols:
                add(Channel(f"finnhub_company_{symbol}", FINNHUB, symbol=symbol), finnhub)
            add(Channel("finnhub_general", FINNHUB), finnhub)
        if ed.enabled:
            add(Channel("edgar_atom_8k", EDGAR),
                _Rule((_window(ed.filing_window_et, ed.atom_poll_seconds),), ed.off_window_poll_seconds))
            for symbol in ed.ciks:
                add(Channel(f"edgar_sub_{symbol}", EDGAR, symbol=symbol),
                    _Rule((_window(ed.filing_window_et, ed.submissions_poll_seconds),), ed.off_window_poll_seconds))
        if fed.enabled:
            start, end = (_hhmm(t) for t in fed.fomc_fast_window_et)
            fomc = tuple((self._utc(day, start), self._utc(day, end))
                         for day in sorted({datetime.fromisoformat(t).date() for t in cal.fomc_statement}))
            fed_rules = {
                "press_all": _Rule((_window(fed.press_window_et, fed.press_poll_seconds),),
                                   fed.off_hours_poll_seconds),
                "press_monetary": _Rule((), None, fomc, fed.fomc_fast_poll_seconds),
                "speeches": _Rule((_window(fed.speeches_window_et, fed.speeches_poll_seconds),),
                                  fed.speeches_off_hours_poll_seconds),
            }
            for feed in fed.feeds:
                add(Channel(f"fed_{feed}", FED, feed=feed), fed_rules[feed])
        if bls.enabled:
            before = bls.release_fast_before_minutes * MS_PER_MINUTE
            after = bls.release_fast_after_minutes * MS_PER_MINUTE
            for feed in bls.feeds:
                releases = (self._local_ms(datetime.fromisoformat(t)) for t in getattr(cal, feed))
                add(Channel(f"bls_{feed}", BLS, feed=feed),
                    _Rule((_window(bls.window_et, bls.poll_seconds),), bls.off_hours_poll_seconds,
                          tuple((ms - before, ms + after) for ms in releases), bls.release_fast_poll_seconds))
        self.channels: tuple[Channel, ...] = tuple(channels)
        self._rules = rules

    def _local_ms(self, local: datetime) -> int:
        """A naive New York wall clock time as UTC milliseconds."""
        return int(local.replace(tzinfo=self.zone).timestamp()) * MS_PER_SECOND

    def _utc(self, day: date, at: time) -> int:
        return self._local_ms(datetime.combine(day, at))

    def _day(self, ms: int) -> date:
        return datetime.fromtimestamp(ms // MS_PER_SECOND, self.zone).date()

    def cadence(self, channel: str, at_ms: int) -> int | None:
        """The cadence in seconds in force at an instant, or None while the channel is dormant."""
        rule = self._rules[channel]
        if any(start <= at_ms < end for start, end in rule.events):
            return rule.event_seconds
        day = self._day(at_ms)
        if day.weekday() < 5:
            for start, end, seconds in rule.daily:
                if self._utc(day, start) <= at_ms < self._utc(day, end):
                    return seconds
        return rule.default

    def _next_change(self, rule: _Rule, after_ms: int) -> int | None:
        """The first window edge strictly after an instant, the only instants at which a cadence can change."""
        edges = [edge for window in rule.events for edge in window if edge > after_ms]
        day = self._day(after_ms)
        for offset in range(8 if rule.daily else 0):
            current = day + timedelta(days=offset)
            if current.weekday() >= 5:
                continue
            later = [ms for start, end, _ in rule.daily for ms in (self._utc(current, start), self._utc(current, end))
                     if ms > after_ms]
            if later:
                edges.append(min(later))
                break
        return min(edges, default=None)

    def next_due(self, channel: str, last_ms: int | None, now_ms: int) -> int | None:
        """The earliest instant at or after now_ms at which the time since the poll that started at last_ms reaches
        the cadence in force at that instant, so a cadence that gets faster polls at once. A channel never polled is
        due as soon as it is not dormant, and None means it stays dormant for good."""
        rule = self._rules[channel]
        at = now_ms
        while True:
            seconds = self.cadence(channel, at)
            change = self._next_change(rule, at)
            if seconds is not None:
                due = at if last_ms is None else max(at, last_ms + seconds * MS_PER_SECOND)
                if change is None or due < change:
                    return due
            elif change is None:
                return None
            at = change
