import io
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest

from btc_survivor.v2 import sessions
from btc_survivor.v2.sessions import CalendarError

UTC = timezone.utc
NY = ZoneInfo("America/New_York")
MON, THU, FRI, SUN = 0, 3, 4, 6
YEARS = range(sessions.FIRST_YEAR, sessions.LAST_YEAR + 1)


def ms(dt: datetime) -> int:
    return int(dt.timestamp()) * 1000


def ny_ms(day: date, hour: int, minute: int) -> int:
    return ms(datetime.combine(day, time(hour, minute), tzinfo=NY))


def utc_ms(*parts: int) -> int:
    return ms(datetime(*parts, tzinfo=UTC))


# The NYSE holiday rules, recomputed here so the hand copied table is checked against an independent source.

def easter(year: int) -> date:
    """Western Easter Sunday by the anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    g = (b - (b + 8) // 25 + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year, month + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def observed(day: date) -> date | None:
    """NYSE Rule 7.2 moves a Saturday holiday to the Friday before, unless that Friday ends a month or the
    year, and a Sunday holiday to the Monday after."""
    if day.weekday() == 5:
        friday = day - timedelta(days=1)
        return friday if friday.month == day.month else None
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def rule_closures(year: int) -> set[date]:
    holidays = [
        date(year, 1, 1),                    # New Year's Day
        nth_weekday(year, 1, MON, 3),        # Martin Luther King Jr. Day
        nth_weekday(year, 2, MON, 3),        # Washington's Birthday
        easter(year) - timedelta(days=2),    # Good Friday
        last_weekday(year, 5, MON),          # Memorial Day
        date(year, 6, 19),                   # Juneteenth
        date(year, 7, 4),                    # Independence Day
        nth_weekday(year, 9, MON, 1),        # Labor Day
        nth_weekday(year, 11, THU, 4),       # Thanksgiving Day
        date(year, 12, 25),                  # Christmas Day
    ]
    return {day for day in map(observed, holidays) if day is not None}


def rule_early_closes(year: int) -> set[date]:
    """NYSE closes at 13:00 the day after Thanksgiving, and on 3 July and 24 December when they are trading days."""
    closed = rule_closures(year)
    candidates = [date(year, 7, 3), nth_weekday(year, 11, THU, 4) + timedelta(days=1), date(year, 12, 24)]
    return {day for day in candidates if day.weekday() < 5 and day not in closed}


def test_easter_helper_on_known_years():
    assert easter(2024) == date(2024, 3, 31)
    assert easter(2025) == date(2025, 4, 20)
    assert [easter(y) for y in YEARS] == [date(2026, 4, 5), date(2027, 3, 28), date(2028, 4, 16)]


@pytest.mark.parametrize("year", YEARS)
def test_table_matches_the_nyse_rules(year):
    assert {d for d in sessions.CLOSED if d.year == year} == rule_closures(year)
    assert {d for d in sessions.EARLY_CLOSES if d.year == year} == rule_early_closes(year)


def test_table_covers_only_its_years_on_weekdays():
    assert not sessions.CLOSED & sessions.EARLY_CLOSES
    for day in sessions.CLOSED | sessions.EARLY_CLOSES:
        assert day.year in YEARS and day.weekday() < 5, day
    assert date(2028, 1, 1).weekday() == 5 and date(2027, 12, 31) not in sessions.CLOSED


@pytest.mark.parametrize("day", sorted(sessions.CLOSED), ids=str)
def test_every_closure_has_no_session(day):
    assert sessions.session_on(day) is None
    assert sessions.session_at(ny_ms(day, 12, 0)) is None
    assert sessions.k_at(ny_ms(day, 10, 0)) is None


@pytest.mark.parametrize("day", sorted(sessions.EARLY_CLOSES), ids=str)
def test_every_early_close_ends_at_13_new_york(day):
    session = sessions.session_on(day)
    assert session.early_close and session.minutes == 210
    assert session.open_ms == ny_ms(day, 9, 30)
    assert session.close_ms == ny_ms(day, 13, 0)


@pytest.mark.parametrize("day, close_ms", [
    (date(2026, 11, 27), utc_ms(2026, 11, 27, 18, 0)),
    (date(2026, 12, 24), utc_ms(2026, 12, 24, 18, 0)),
    (date(2027, 11, 26), utc_ms(2027, 11, 26, 18, 0)),
    (date(2028, 7, 3), utc_ms(2028, 7, 3, 17, 0)),
], ids=str)
def test_early_closes_in_utc(day, close_ms):
    assert sessions.session_on(day).close_ms == close_ms


@pytest.mark.parametrize("day", [date(2026, 10, 12), date(2027, 12, 31)], ids=str)
def test_open_on_columbus_day_and_the_friday_before_a_saturday_new_year(day):
    session = sessions.session_on(day)
    assert session is not None and session.minutes == 390 and not session.early_close


@pytest.mark.parametrize("day, open_ms", [
    (date(2026, 10, 30), utc_ms(2026, 10, 30, 13, 30)),
    (date(2026, 11, 2), utc_ms(2026, 11, 2, 14, 30)),
    (date(2027, 3, 12), utc_ms(2027, 3, 12, 14, 30)),
    (date(2027, 3, 15), utc_ms(2027, 3, 15, 13, 30)),
], ids=str)
def test_opens_across_daylight_saving(day, open_ms):
    session = sessions.session_on(day)
    assert session.open_ms == open_ms
    assert session.close_ms == open_ms + 390 * 60_000


@pytest.mark.parametrize("year", YEARS)
def test_each_year_has_251_sessions(year):
    assert sum(1 for s in sessions.all_sessions() if s.day.year == year) == 251


def test_k_is_one_at_0931_on_2_january_2026():
    first = sessions.all_sessions()[0]
    assert first.day == date(2026, 1, 2) and first.first_k == 1
    assert sessions.k_at(utc_ms(2026, 1, 2, 14, 31)) == 1
    assert sessions.k_at(ny_ms(date(2026, 1, 2), 16, 0)) == 390
    assert sessions.k_at(ny_ms(date(2026, 1, 5), 9, 31)) == 391
    assert sessions.k_at(ny_ms(date(2026, 1, 2), 9, 30)) is None   # the open closes no session minute
    assert sessions.k_at(ny_ms(date(2026, 1, 2), 16, 1)) is None
    assert sessions.k_at(ny_ms(date(2026, 1, 2), 9, 31) + 1) is None  # not a minute boundary


HOLIDAY_MONDAYS = sorted(d for d in sessions.CLOSED if d.weekday() == MON)


@pytest.mark.parametrize("monday", HOLIDAY_MONDAYS, ids=str)
def test_tuesday_after_a_holiday_monday_follows_the_friday_close(monday):
    friday, tuesday = monday - timedelta(days=3), monday + timedelta(days=1)
    assert not sessions.session_on(friday).early_close
    assert sessions.k_at(ny_ms(tuesday, 9, 31)) == sessions.k_at(ny_ms(friday, 16, 0)) + 1


def test_k_runs_on_without_gaps_across_every_session():
    all_ = sessions.all_sessions()
    for before, after in zip(all_, all_[1:]):
        assert after.first_k == before.last_k + 1
        assert after.open_ms > before.close_ms
    for k in range(1, all_[-1].last_k + 1, 97):
        boundary = sessions.boundary_of_k(k)
        assert sessions.k_at(boundary) == k
        assert sessions.session_of_k(k) is sessions.session_of_boundary(boundary)


def test_full_and_early_sessions_and_their_evaluation_minutes():
    full = sessions.session_on(date(2026, 10, 12))
    assert full.minutes == 390
    assert full.evaluation_minutes() == tuple(range(30, 361, 30)) and len(full.evaluation_minutes()) == 12
    assert full.boundary(30) == ny_ms(full.day, 10, 0)
    assert full.boundary(360) == ny_ms(full.day, 15, 30)
    early = sessions.session_on(date(2026, 11, 27))
    assert early.minutes == 210
    assert early.evaluation_minutes() == (30, 60, 90, 120, 150, 180)
    assert early.boundary(180) == ny_ms(early.day, 12, 30)
    assert early.boundary(early.evaluation_minutes()[-1]) + 30 * 60_000 == early.close_ms


def test_launch_window_minutes_match_the_plan():
    window = [s for s in sessions.all_sessions() if date(2026, 11, 2) <= s.day <= date(2026, 12, 31)]
    assert len(window) == 42
    assert sum(s.minutes for s in window) == 16_020


def test_session_minute_and_boundary_round_trip():
    session = sessions.session_on(date(2027, 3, 15))
    for minute in (1, 30, 390):
        assert session.session_minute(session.boundary(minute)) == minute
    with pytest.raises(ValueError):
        session.boundary(0)
    with pytest.raises(ValueError):
        session.boundary(391)
    with pytest.raises(ValueError):
        session.session_minute(session.open_ms)


def test_session_at_instants_and_new_york_dates():
    day = date(2026, 11, 2)
    session = sessions.session_on(day)
    assert sessions.session_at(session.open_ms) is session
    assert sessions.session_at(session.close_ms - 1) is session
    assert sessions.session_at(session.close_ms) is None
    assert sessions.session_at(session.open_ms - 1) is None
    assert sessions.session_at(ny_ms(date(2026, 10, 31), 12, 0)) is None   # a Saturday
    assert sessions.ny_date(utc_ms(2026, 11, 2, 4, 59)) == date(2026, 11, 1)
    assert sessions.ny_date(utc_ms(2026, 11, 2, 5, 0)) == day


def test_next_session():
    friday, tuesday = date(2026, 1, 16), date(2026, 1, 20)
    assert sessions.next_session(ny_ms(friday, 12, 0)).day == friday
    assert sessions.next_session(ny_ms(friday, 16, 0)).day == tuesday
    assert sessions.next_session(ny_ms(date(2026, 1, 19), 9, 20)).day == tuesday
    assert sessions.next_session(ny_ms(tuesday, 9, 20)).day == tuesday


def test_dates_past_the_table_raise():
    last = sessions.all_sessions()[-1]
    assert last.day == date(2028, 12, 29)
    with pytest.raises(CalendarError):
        sessions.session_on(date(2029, 1, 2))
    with pytest.raises(CalendarError):
        sessions.session_on(date(2025, 12, 31))
    with pytest.raises(CalendarError):
        sessions.session_at(utc_ms(2029, 1, 2, 15, 0))
    with pytest.raises(CalendarError):
        sessions.k_at(utc_ms(2029, 1, 2, 15, 0))
    with pytest.raises(CalendarError):
        sessions.next_session(last.close_ms)
    with pytest.raises(CalendarError):
        sessions.session_of_k(last.last_k + 1)
    with pytest.raises(CalendarError):
        sessions.boundary_of_k(0)
    with pytest.raises(CalendarError):
        sessions.calendar_warning(date(2029, 1, 1), 60)
    assert sessions.boundary_of_k(last.last_k) == last.close_ms


class OldUsEastern(tzinfo):
    """New York under the United States rules before 2007, standing in for an outdated time zone database."""

    def utcoffset(self, dt):
        return timedelta(hours=-5) + self.dst(dt)

    def dst(self, dt):
        start = datetime.combine(nth_weekday(dt.year, 4, SUN, 1), time(2))
        end = datetime.combine(last_weekday(dt.year, 10, SUN), time(2))
        return timedelta(hours=1) if start <= dt.replace(tzinfo=None) < end else timedelta(0)

    def tzname(self, dt):
        return "OLD"


def test_self_test_passes_on_the_real_database():
    sessions.self_test()


@pytest.mark.parametrize("tz, first_wrong", [
    (timezone(timedelta(hours=-5)), "2026-10-30"),
    (timezone(timedelta(hours=-4)), "2026-11-02"),
    (UTC, "2026-10-30"),
    (OldUsEastern(), "2026-10-30"),
], ids=["fixed_est", "fixed_edt", "utc", "rules_before_2007"])
def test_self_test_refuses_a_broken_time_zone(tz, first_wrong):
    with pytest.raises(CalendarError, match=first_wrong):
        sessions.self_test(tz)


def _missing(key):
    raise ZoneInfoNotFoundError(f"No time zone found with key {key}")


def _corrupt(key):
    return ZoneInfo.from_file(io.BytesIO(b"not a time zone file"), key=key)


@pytest.mark.parametrize("loader", [_missing, _corrupt], ids=["missing", "corrupt"])
def test_self_test_refuses_a_missing_or_corrupt_database(monkeypatch, loader):
    monkeypatch.setattr(sessions, "ZoneInfo", loader)
    with pytest.raises(CalendarError, match="America/New_York"):
        sessions.self_test()


@pytest.mark.parametrize("loader", [lambda key: timezone(timedelta(hours=-5)), lambda key: OldUsEastern(), _missing],
                         ids=["fixed_est", "rules_before_2007", "missing"])
def test_no_session_comes_from_a_broken_time_zone(monkeypatch, loader):
    monkeypatch.setattr(sessions, "ZoneInfo", loader)
    sessions._table.cache_clear()
    try:
        with pytest.raises(CalendarError):
            sessions.session_on(date(2026, 11, 2))
        with pytest.raises(CalendarError):
            sessions.session_of_k(1)
    finally:
        sessions._table.cache_clear()


def test_warning_from_60_days_before_the_table_ends():
    assert sessions.calendar_warning(date(2026, 10, 11), 60) is None
    assert sessions.calendar_warning(date(2028, 10, 31), 60) is None
    detail = sessions.calendar_warning(date(2028, 11, 1), 60)
    assert detail is not None and "2028-12-31" in detail and "60 days" in detail
    assert sessions.calendar_warning(date(2028, 12, 31), 60) is not None
    assert sessions.calendar_warning(date(2028, 11, 1), 59) is None
