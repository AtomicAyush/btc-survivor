"""Polling cadences on both sides of every window in EDT and EST, across the 1 November 2026 clock change, on
weekends and holidays, and in the fast windows around scheduled releases. Instants are written with fixed UTC
offsets, so no expected value comes from the time zone database the schedule reads."""

import dataclasses
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from btc_survivor.v2.config import CalendarConfig, load_news_config
from btc_survivor.v2.news import schedule as schedule_module
from btc_survivor.v2.news.schedule import Channel, Schedule
from btc_survivor.v2.sessions import CalendarError
from helpers import ROOT

EDT, EST, UTC = timezone(timedelta(hours=-4)), timezone(timedelta(hours=-5)), timezone.utc
SECOND = 1000
CONFIG = load_news_config(ROOT / "configs" / "v2" / "news.toml")
# A calendar for these tests only, kept apart from the committed one, with BLS releases in both offsets and the two
# 2026 FOMC statement days. tests/test_v2_news_calendar.py checks the committed calendar.
CALENDAR = CalendarConfig(
    fomc_statement=("2026-10-28T14:00", "2026-12-09T14:00"),
    cpi=("2026-10-14T08:30", "2026-11-10T08:30"),
    empsit=("2026-11-06T08:30",),
    ppi=("2026-10-15T08:30",),
    jolts=("2026-11-03T10:00",),
)
SCHEDULE = Schedule(dataclasses.replace(CONFIG, calendar=CALENDAR))

FINNHUB = [f"finnhub_company_{s}" for s in ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA")]
EDGAR_SUB = [f"edgar_sub_{s}" for s in ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA")]
BLS = ["bls_cpi", "bls_empsit", "bls_ppi", "bls_jolts"]
ALL = [*FINNHUB, "finnhub_general", "edgar_atom_8k", *EDGAR_SUB, "fed_press_all", "fed_press_monetary",
       "fed_speeches", *BLS]


def at(y, mo, d, h, mi, tz, s=0):
    return int(datetime(y, mo, d, h, mi, s, tzinfo=tz).timestamp()) * SECOND


def cadence(channel, ms):
    return SCHEDULE.cadence(channel, ms)


def polls(channel, start, end):
    """How many polls a channel makes from start to end when each one starts the moment it is due."""
    count, last, now = 0, None, start
    while (due := SCHEDULE.next_due(channel, last, now)) is not None and due < end:
        count, last, now = count + 1, due, due
    return count


def test_the_committed_config_has_the_planned_channels():
    schedule = Schedule(CONFIG)
    assert [c.name for c in schedule.channels] == ALL
    by_name = {c.name: c for c in schedule.channels}
    assert by_name["finnhub_company_AAPL"] == Channel("finnhub_company_AAPL", "finnhub", symbol="AAPL")
    assert by_name["finnhub_general"] == Channel("finnhub_general", "finnhub")
    assert by_name["edgar_atom_8k"] == Channel("edgar_atom_8k", "edgar")
    assert by_name["edgar_sub_TSLA"] == Channel("edgar_sub_TSLA", "edgar", symbol="TSLA")
    assert by_name["fed_press_monetary"] == Channel("fed_press_monetary", "fed", feed="press_monetary")
    assert by_name["bls_jolts"] == Channel("bls_jolts", "bls", feed="jolts")
    assert "finnhub_company_QQQ" not in by_name and "edgar_sub_QQQ" not in by_name


def test_disabled_sources_have_no_channels():
    off = dataclasses.replace(
        CONFIG, finnhub=dataclasses.replace(CONFIG.finnhub, enabled=False),
        edgar=dataclasses.replace(CONFIG.edgar, enabled=False), bls=dataclasses.replace(CONFIG.bls, enabled=False))
    assert [c.name for c in Schedule(off).channels] == ["fed_press_all", "fed_press_monetary", "fed_speeches"]
    none = dataclasses.replace(off, fed=dataclasses.replace(CONFIG.fed, enabled=False))
    assert Schedule(none).channels == ()


# Each daily edge as (channels, local time, cadence just before, cadence from the edge on).
EDGES = [
    (FINNHUB + ["finnhub_general"], (4, 0), 600, 120),
    (FINNHUB + ["finnhub_general"], (9, 30), 120, 60),
    (FINNHUB + ["finnhub_general"], (16, 0), 60, 120),
    (FINNHUB + ["finnhub_general"], (20, 0), 120, 600),
    (["edgar_atom_8k"], (6, 0), 900, 30),
    (["edgar_atom_8k"], (22, 0), 30, 900),
    (EDGAR_SUB, (6, 0), 900, 60),
    (EDGAR_SUB, (22, 0), 60, 900),
    (["fed_press_all"], (7, 0), 900, 60),
    (["fed_press_all"], (19, 0), 60, 900),
    (["fed_speeches"], (7, 0), 3600, 600),
    (["fed_speeches"], (21, 0), 600, 3600),
    (BLS, (7, 0), 7200, 1800),
    (BLS, (19, 0), 1800, 7200),
]
WEEKDAYS = [((2026, 10, 29), EDT), ((2026, 11, 5), EST), ((2026, 10, 30), EDT), ((2026, 11, 2), EST)]


def test_every_channel_but_the_fomc_one_has_daily_edges():
    assert {c for channels, *_ in EDGES for c in channels} == set(ALL) - {"fed_press_monetary"}


@pytest.mark.parametrize("day, tz", WEEKDAYS, ids=["thursday edt", "thursday est", "friday edt", "monday est"])
@pytest.mark.parametrize("channels, hm, before, after", EDGES, ids=lambda v: str(v) if isinstance(v, tuple) else None)
def test_both_sides_of_every_window(day, tz, channels, hm, before, after):
    edge = at(*day, *hm, tz)
    for channel in channels:
        assert cadence(channel, edge - 1) == before, channel
        assert cadence(channel, edge) == after, channel


def test_the_same_utc_instant_moves_across_1_november_2026():
    """04:00 New York is 08:00 UTC on Friday 30 October and 09:00 UTC on Monday 2 November."""
    assert cadence("finnhub_general", at(2026, 10, 30, 8, 0, UTC)) == 120
    assert cadence("finnhub_general", at(2026, 10, 30, 8, 0, UTC) - 1) == 600
    assert cadence("finnhub_general", at(2026, 11, 2, 8, 0, UTC)) == 600
    assert cadence("finnhub_general", at(2026, 11, 2, 9, 0, UTC)) == 120
    assert cadence("finnhub_general", at(2026, 10, 30, 13, 30, UTC)) == 60
    assert cadence("finnhub_general", at(2026, 11, 2, 13, 30, UTC)) == 120
    assert cadence("finnhub_general", at(2026, 11, 2, 14, 30, UTC)) == 60
    assert cadence("edgar_atom_8k", at(2026, 10, 30, 10, 0, UTC)) == 30
    assert cadence("edgar_atom_8k", at(2026, 11, 2, 10, 0, UTC)) == 900
    assert cadence("bls_cpi", at(2026, 10, 30, 23, 0, UTC)) == 7200
    assert cadence("bls_cpi", at(2026, 11, 2, 23, 0, UTC)) == 1800


WEEKEND = {**{c: 600 for c in FINNHUB + ["finnhub_general"]}, "edgar_atom_8k": 900, **{c: 900 for c in EDGAR_SUB},
           "fed_press_all": 900, "fed_press_monetary": None, "fed_speeches": 3600, **{c: 7200 for c in BLS}}


@pytest.mark.parametrize("instant", [
    at(2026, 10, 31, 0, 0, EDT), at(2026, 10, 31, 10, 0, EDT), at(2026, 10, 31, 14, 0, EDT),
    at(2026, 10, 31, 23, 59, EDT, 59) + 999,
    at(2026, 11, 1, 1, 30, EDT), at(2026, 11, 1, 1, 30, EST), at(2026, 11, 1, 12, 0, EST),
    at(2026, 11, 7, 9, 30, EST), at(2026, 11, 8, 20, 59, EST),
], ids=["sat midnight", "sat 10", "sat 14", "sat last ms", "sun 0130 edt", "sun 0130 est", "sun noon",
        "sat 0930 est", "sun 2059 est"])
def test_weekends_poll_at_the_off_cadences(instant):
    assert {c: cadence(c, instant) for c in ALL} == WEEKEND


def test_the_weekend_ends_at_midnight_new_york():
    monday = at(2026, 11, 2, 0, 0, EST)
    assert cadence("finnhub_general", monday - 1) == 600 and cadence("finnhub_general", monday) == 600
    assert cadence("finnhub_general", at(2026, 11, 2, 4, 0, EST)) == 120
    assert cadence("finnhub_general", at(2026, 10, 30, 20, 0, EDT) - 1) == 120


@pytest.mark.parametrize("day, tz", [((2026, 11, 26), EST), ((2026, 11, 11), EST), ((2026, 12, 25), EST)],
                         ids=["thanksgiving", "veterans day", "christmas"])
def test_holidays_count_as_weekdays(day, tz):
    assert cadence("finnhub_company_AAPL", at(*day, 10, 0, tz)) == 60
    assert cadence("edgar_atom_8k", at(*day, 10, 0, tz)) == 30
    assert cadence("fed_press_all", at(*day, 10, 0, tz)) == 60
    assert cadence("bls_empsit", at(*day, 10, 0, tz)) == 1800


@pytest.mark.parametrize("day, tz", [((2026, 10, 28), EDT), ((2026, 12, 9), EST)], ids=["edt", "est"])
def test_the_fomc_fast_window(day, tz):
    start, end = at(*day, 13, 58, tz), at(*day, 14, 20, tz)
    assert cadence("fed_press_monetary", start - 1) is None
    assert cadence("fed_press_monetary", start) == 15
    assert cadence("fed_press_monetary", end - 1) == 15
    assert cadence("fed_press_monetary", end) is None
    assert cadence("fed_press_all", at(*day, 14, 0, tz)) == 60
    assert polls("fed_press_monetary", at(*day, 0, 0, tz), at(*day, 0, 0, tz) + 86_400_000) == 88


def test_the_fomc_channel_is_dormant_on_other_days():
    for instant in (at(2026, 11, 4, 14, 0, EST), at(2026, 10, 27, 14, 0, EDT), at(2026, 10, 29, 14, 0, EDT)):
        assert cadence("fed_press_monetary", instant) is None


def test_every_committed_fomc_day_gets_its_window():
    schedule = Schedule(CONFIG)
    assert CONFIG.calendar.fomc_statement
    for text in CONFIG.calendar.fomc_statement:
        statement = int(datetime.fromisoformat(text).replace(tzinfo=ZoneInfo("America/New_York")).timestamp()) * SECOND
        assert schedule.cadence("fed_press_monetary", statement) == 15, text
        assert schedule.cadence("fed_press_monetary", statement - 2 * 60 * SECOND - 1) is None, text


@pytest.mark.parametrize("channel, release, tz", [
    ("bls_cpi", (2026, 10, 14, 8, 30), EDT), ("bls_cpi", (2026, 11, 10, 8, 30), EST),
    ("bls_empsit", (2026, 11, 6, 8, 30), EST), ("bls_ppi", (2026, 10, 15, 8, 30), EDT),
    ("bls_jolts", (2026, 11, 3, 10, 0), EST),
], ids=["cpi edt", "cpi est", "empsit est", "ppi edt", "jolts est"])
def test_bls_fast_windows_from_one_minute_before_to_15_after(channel, release, tz):
    moment = at(*release, tz)
    start, end = moment - 60 * SECOND, moment + 15 * 60 * SECOND
    assert cadence(channel, start - 1) == 1800
    assert cadence(channel, start) == 15
    assert cadence(channel, moment) == 15
    assert cadence(channel, end - 1) == 15
    assert cadence(channel, end) == 1800
    for other in BLS:
        if other != channel:
            assert cadence(other, moment) == 1800, other


def test_a_release_day_adds_about_64_polls():
    ordinary = polls("bls_cpi", at(2026, 10, 13, 0, 0, EDT), at(2026, 10, 14, 0, 0, EDT))
    release = polls("bls_cpi", at(2026, 10, 14, 0, 0, EDT), at(2026, 10, 15, 0, 0, EDT))
    assert ordinary == 30
    assert abs(release - ordinary - 64) <= 1


@pytest.mark.parametrize("day, tz", [((2026, 10, 29), EDT), ((2026, 11, 5), EST)], ids=["edt", "est"])
def test_weekday_requests_match_the_plan(day, tz):
    """PLAN_V2 section 7.2 gives about 5,060, 720, 1,950, 6,940, 770, 95 and 120 requests a weekday."""
    start = at(*day, 0, 0, tz)
    end = start + 86_400_000
    count = {c: polls(c, start, end) for c in ALL}
    assert sum(count[c] for c in FINNHUB) == 7 * 723
    assert count["finnhub_general"] == 723
    assert count["edgar_atom_8k"] == 1952
    assert sum(count[c] for c in EDGAR_SUB) == 7 * 992
    assert count["fed_press_all"] == 768
    assert count["fed_press_monetary"] == 0
    assert count["fed_speeches"] == 94
    assert sum(count[c] for c in BLS) == 120


def test_a_faster_cadence_polls_at_once():
    last = at(2026, 10, 14, 8, 10, EDT)
    assert SCHEDULE.next_due("bls_cpi", last, last + 5 * SECOND) == at(2026, 10, 14, 8, 29, EDT)
    last = at(2026, 11, 2, 5, 59, EST)
    assert SCHEDULE.next_due("edgar_atom_8k", last, last + SECOND) == at(2026, 11, 2, 6, 0, EST)
    last = at(2026, 10, 29, 9, 29, EDT, 30)
    assert SCHEDULE.next_due("finnhub_general", last, last + SECOND) == at(2026, 10, 29, 9, 30, EDT, 30)


def test_a_slower_cadence_counts_from_the_last_poll():
    last = at(2026, 10, 14, 8, 44, EDT, 50)
    assert SCHEDULE.next_due("bls_cpi", last, last + SECOND) == last + 1800 * SECOND
    last = at(2026, 10, 29, 19, 59, EDT)
    assert SCHEDULE.next_due("finnhub_general", last, last + SECOND) == last + 600 * SECOND
    last = at(2026, 10, 30, 21, 59, EDT, 50)
    assert SCHEDULE.next_due("edgar_atom_8k", last, last + SECOND) == last + 900 * SECOND


def test_ordinary_polls_follow_the_cadence():
    last = at(2026, 10, 29, 10, 0, EDT)
    assert SCHEDULE.next_due("finnhub_company_NVDA", last, last + 2 * SECOND) == last + 60 * SECOND
    assert SCHEDULE.next_due("finnhub_company_NVDA", last, last + 90 * SECOND) == last + 90 * SECOND


def test_next_due_counts_utc_time_across_the_clock_change():
    last = at(2026, 11, 1, 0, 30, EDT)
    due = SCHEDULE.next_due("bls_cpi", last, last + SECOND)
    assert due == last + 7200 * SECOND == at(2026, 11, 1, 1, 30, EST)
    last = at(2026, 10, 31, 23, 50, EDT)
    assert SCHEDULE.next_due("finnhub_general", last, last + SECOND) == last + 600 * SECOND


def test_a_channel_never_polled_is_due_at_once_unless_dormant():
    now = at(2026, 10, 31, 3, 7, EDT, 11)
    for channel in ALL:
        if channel != "fed_press_monetary":
            assert SCHEDULE.next_due(channel, None, now) == now, channel
    assert SCHEDULE.next_due("fed_press_monetary", None, now) == at(2026, 12, 9, 13, 58, EST)
    assert SCHEDULE.next_due("fed_press_monetary", None, at(2026, 10, 28, 14, 0, EDT)) == \
        at(2026, 10, 28, 14, 0, EDT)


def test_the_fomc_channel_sleeps_until_the_next_statement_day():
    last = at(2026, 10, 28, 14, 19, EDT, 45)
    assert SCHEDULE.next_due("fed_press_monetary", last, last + 2 * SECOND) == at(2026, 12, 9, 13, 58, EST)
    last = at(2026, 12, 9, 14, 19, EST, 45)
    assert SCHEDULE.next_due("fed_press_monetary", last, last + 2 * SECOND) is None
    last = at(2026, 12, 9, 14, 0, EST)
    assert SCHEDULE.next_due("fed_press_monetary", last, last + SECOND) == last + 15 * SECOND


def test_a_broken_time_zone_database_is_refused(monkeypatch):
    monkeypatch.setattr(schedule_module, "ZoneInfo", lambda name: timezone(timedelta(hours=-5)))
    with pytest.raises(CalendarError):
        Schedule(CONFIG)
