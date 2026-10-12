"""The US equity regular session calendar for 2026 to 2028 and the session minute index.

A session runs 09:30 to 16:00 New York time, or to 13:00 on an early close, on every weekday that is not
a full closure, converted to UTC through zoneinfo. The session minute index k counts regular session
minutes only, with k = 1 at the 09:31 boundary of Friday 2 January 2026, so a night, weekend or holiday
adds no minutes and the first minute of a session follows the last minute of the previous one directly.
"""

from __future__ import annotations

import bisect
import functools
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from btc_survivor.clock import MS_PER_MINUTE, MS_PER_SECOND, iso


class CalendarError(ValueError):
    """Raised for a date outside the table or a time zone database that gives a wrong open."""


FIRST_YEAR = 2026
LAST_YEAR = 2028
TABLE_END = date(LAST_YEAR, 12, 31)
TIME_ZONE = "America/New_York"
OPEN_TIME = time(9, 30)
CLOSE_TIME = time(16, 0)
EARLY_CLOSE_TIME = time(13, 0)
FULL_SESSION_MINUTES = 390
EARLY_SESSION_MINUTES = 210

# NYSE full closures and 13:00 ET early closes, hand copied from the NYSE hours page,
# https://www.nyse.com/markets/hours-calendars, on 11 October 2026 into PLAN_V2 section 4.4.
# Nasdaq is understood to follow the same dates. The table is append only, and
# tests/test_v2_sessions.py recomputes every date from the NYSE holiday rules.
CLOSED = frozenset({
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25),
    date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25),
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26), date(2027, 5, 31),
    date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25), date(2027, 12, 24),
    # 1 January 2028 is a Saturday and is not observed, because 31 December 2027 ends the year.
    date(2028, 1, 17), date(2028, 2, 21), date(2028, 4, 14), date(2028, 5, 29), date(2028, 6, 19),
    date(2028, 7, 4), date(2028, 9, 4), date(2028, 11, 23), date(2028, 12, 25),
})
EARLY_CLOSES = frozenset({
    date(2026, 11, 27), date(2026, 12, 24),
    date(2027, 11, 26),
    date(2028, 7, 3), date(2028, 11, 24),
})

# Opens written as UTC, independent of any time zone database, on both sides of the 2026 and 2027 clock changes.
KNOWN_OPENS = {
    date(2026, 10, 30): datetime(2026, 10, 30, 13, 30, tzinfo=timezone.utc),
    date(2026, 11, 2): datetime(2026, 11, 2, 14, 30, tzinfo=timezone.utc),
    date(2027, 3, 12): datetime(2027, 3, 12, 14, 30, tzinfo=timezone.utc),
    date(2027, 3, 15): datetime(2027, 3, 15, 13, 30, tzinfo=timezone.utc),
}


@dataclass(frozen=True)
class Session:
    """One regular session, its UTC bounds in milliseconds and the index k of its first minute."""

    day: date
    open_ms: int
    close_ms: int
    first_k: int

    @property
    def minutes(self) -> int:
        return (self.close_ms - self.open_ms) // MS_PER_MINUTE

    @property
    def early_close(self) -> bool:
        return self.minutes == EARLY_SESSION_MINUTES

    @property
    def last_k(self) -> int:
        return self.first_k + self.minutes - 1

    def has_boundary(self, boundary_ms: int) -> bool:
        """Whether a minute boundary closes a minute of this session, from 09:31 to the close."""
        return self.open_ms < boundary_ms <= self.close_ms and boundary_ms % MS_PER_MINUTE == 0

    def session_minute(self, boundary_ms: int) -> int:
        """The session minute a boundary closes, 1 at 09:31 and 390 at a 16:00 close."""
        if not self.has_boundary(boundary_ms):
            raise ValueError(f"{iso(boundary_ms)} is not a minute boundary of the {self.day.isoformat()} session")
        return (boundary_ms - self.open_ms) // MS_PER_MINUTE

    def boundary(self, session_minute: int) -> int:
        """The UTC boundary in milliseconds that closes a session minute."""
        if not 1 <= session_minute <= self.minutes:
            raise ValueError(f"session minute {session_minute} is outside 1 to {self.minutes}")
        return self.open_ms + session_minute * MS_PER_MINUTE

    def k(self, boundary_ms: int) -> int:
        """The session minute index of a boundary of this session."""
        return self.first_k + self.session_minute(boundary_ms) - 1

    def evaluation_minutes(self, horizon_minutes: int = 30) -> tuple[int, ...]:
        """Session minutes 30, 60 and so on whose horizon still ends within the session, 12 in a full session."""
        return tuple(range(horizon_minutes, self.minutes - horizon_minutes + 1, horizon_minutes))


def new_york() -> tzinfo:
    """The New York time zone from the time zone database, or a CalendarError when it cannot be read."""
    try:
        return ZoneInfo(TIME_ZONE)
    except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
        raise CalendarError(f"the time zone database has no usable {TIME_ZONE}: {exc}") from exc


def check_covered(day: date) -> None:
    """Refuse a date outside the table, so no session is ever guessed."""
    if not FIRST_YEAR <= day.year <= LAST_YEAR:
        raise CalendarError(f"{day.isoformat()} is outside the session table, {FIRST_YEAR} to {LAST_YEAR}")


def _utc_ms(day: date, at: time, tz: tzinfo) -> int:
    return int(datetime.combine(day, at, tzinfo=tz).timestamp()) * MS_PER_SECOND


@functools.cache
def _table() -> tuple[tuple[Session, ...], dict[date, Session]]:
    """Every session of the table in order with its running k, and the same sessions by date.

    The self test runs first, so no session or k can ever come from a time zone database with a wrong open.
    """
    tz = new_york()
    self_test(tz)
    sessions: list[Session] = []
    day = date(FIRST_YEAR, 1, 1)
    while day <= TABLE_END:
        if day.weekday() < 5 and day not in CLOSED:
            close = EARLY_CLOSE_TIME if day in EARLY_CLOSES else CLOSE_TIME
            first_k = sessions[-1].last_k + 1 if sessions else 1
            sessions.append(Session(day, _utc_ms(day, OPEN_TIME, tz), _utc_ms(day, close, tz), first_k))
        day += timedelta(days=1)
    return tuple(sessions), {session.day: session for session in sessions}


def all_sessions() -> tuple[Session, ...]:
    return _table()[0]


def ny_date(ms: int) -> date:
    """The New York calendar date of a UTC instant in milliseconds."""
    return datetime.fromtimestamp(ms // MS_PER_SECOND, new_york()).date()


def session_on(day: date) -> Session | None:
    """The session held on a New York date, or None on a weekend or full closure."""
    check_covered(day)
    return _table()[1].get(day)


def session_at(ms: int) -> Session | None:
    """The session whose regular hours hold the instant ms, the open included and the close excluded."""
    session = session_on(ny_date(ms))
    return session if session is not None and session.open_ms <= ms < session.close_ms else None


def session_of_boundary(boundary_ms: int) -> Session | None:
    """The session in which a minute boundary closes a minute, from 09:31 to the close."""
    session = session_on(ny_date(boundary_ms))
    return session if session is not None and session.has_boundary(boundary_ms) else None


def k_at(boundary_ms: int) -> int | None:
    """The session minute index k of a boundary, or None for a boundary outside every session."""
    session = session_of_boundary(boundary_ms)
    return None if session is None else session.k(boundary_ms)


def session_of_k(k: int) -> Session:
    """The session holding session minute index k, refusing a k outside the table."""
    sessions = all_sessions()
    if not 1 <= k <= sessions[-1].last_k:
        raise CalendarError(f"session minute index {k} is outside the session table, 1 to {sessions[-1].last_k}")
    return sessions[bisect.bisect_right(sessions, k, key=lambda s: s.first_k) - 1]


def boundary_of_k(k: int) -> int:
    """The UTC boundary in milliseconds of session minute index k."""
    session = session_of_k(k)
    return session.boundary(k - session.first_k + 1)


def next_session(ms: int) -> Session:
    """The session in progress at ms or else the next one to open, refusing to look past the table."""
    check_covered(ny_date(ms))
    sessions = all_sessions()
    index = bisect.bisect_right(sessions, ms, key=lambda s: s.close_ms)
    if index == len(sessions):
        raise CalendarError(f"no session follows {iso(ms)} in the session table, which ends {TABLE_END.isoformat()}")
    return sessions[index]


def self_test(tz: tzinfo | None = None) -> None:
    """Refuse to run when the time zone database gives a wrong open on any of the four known dates."""
    zone = new_york() if tz is None else tz
    for day, expected in KNOWN_OPENS.items():
        got = _utc_ms(day, OPEN_TIME, zone)
        want = int(expected.timestamp()) * MS_PER_SECOND
        if got != want:
            raise CalendarError(
                f"the time zone database puts the {day.isoformat()} open at {iso(got)}, expected {iso(want)}")


def calendar_warning(day: date, warn_days: int) -> str | None:
    """The detail of the daily calendar_warning event from warn_days before the table ends, else None."""
    check_covered(day)
    days_left = (TABLE_END - day).days
    if days_left > warn_days:
        return None
    return (f"the session table ends on {TABLE_END.isoformat()}, {days_left} days after {day.isoformat()}, "
            f"so sessions.py needs the closures of {LAST_YEAR + 1}")
