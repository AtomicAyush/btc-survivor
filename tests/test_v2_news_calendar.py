"""The committed [calendar] of configs/v2/news.toml covers the collector's start on 12 October 2026, a week before the
19 October launch, through 31 March 2027, as PLAN_V2 section 16 requires. Each series lists every release once, on a
weekday session at its usual time, with no gap its cadence cannot explain, and every BLS release gets its fast
window."""

from datetime import date, datetime
from itertools import pairwise
from zoneinfo import ZoneInfo

import pytest

from btc_survivor.v2 import sessions
from btc_survivor.v2.config import load_news_config
from btc_survivor.v2.news.schedule import Schedule
from helpers import ROOT

CONFIG = load_news_config(ROOT / "configs" / "v2" / "news.toml")
FIRST, LAST = date(2026, 10, 12), date(2027, 3, 31)
BLS = ("cpi", "empsit", "ppi", "jolts")
KINDS = ("fomc_statement", *BLS)
TIMES = {"fomc_statement": "14:00", "cpi": "08:30", "empsit": "08:30", "ppi": "08:30", "jolts": "10:00"}
# The shortest and longest number of days between two releases of one series. A monthly BLS release comes 20 to 42
# days after the one before, and an FOMC statement six to eight weeks after the last.
GAPS = {"fomc_statement": (28, 56), **{kind: (20, 42) for kind in BLS}}
# No BLS page read on 12 October 2026 listed a JOLTS release after 1 December 2026, so JOLTS coverage is checked only
# that far. Once BLS lists the later dates and they are added before launch, this entry goes.
COVERED_THROUGH = {"jolts": date(2026, 12, 1)}
# Weekdays on which markets open but BLS and the Fed publish nothing, Columbus Day and Veterans Day, as PLAN_V2
# section 7.7 notes.
QUIET_WEEKDAYS = {date(2026, 10, 12), date(2026, 11, 11)}
SCHEDULE = Schedule(CONFIG)


def releases(kind: str) -> list[datetime]:
    """The committed release times of one series as naive New York wall clock times."""
    return [datetime.fromisoformat(text) for text in getattr(CONFIG.calendar, kind)]


@pytest.mark.parametrize("kind", KINDS)
def test_every_release_falls_inside_the_window(kind):
    assert releases(kind)
    assert [r for r in releases(kind) if not FIRST <= r.date() <= LAST] == []


@pytest.mark.parametrize("kind", KINDS)
def test_each_series_covers_launch_through_31_march_2027(kind):
    """No stretch from the collector's start, through each release, to 31 March 2027 is longer than the series'
    cadence allows, so no release in between is missing."""
    end = COVERED_THROUGH.get(kind, LAST)
    days = [FIRST, *(r.date() for r in releases(kind) if r.date() <= end), end]
    gaps = {f"{a} to {b}": (b - a).days for a, b in pairwise(days)}
    assert max(gaps.values()) <= GAPS[kind][1], gaps


@pytest.mark.parametrize("kind", ("cpi", "empsit", "ppi"))
def test_cpi_employment_and_ppi_release_once_in_every_month_from_november(kind):
    months = [(r.year, r.month) for r in releases(kind)]
    for month in ((2026, 11), (2026, 12), (2027, 1), (2027, 2), (2027, 3)):
        assert months.count(month) == 1, month


@pytest.mark.parametrize("kind", KINDS)
def test_no_release_is_listed_twice(kind):
    days = [r.date() for r in releases(kind)]
    assert len(set(days)) == len(days)
    gaps = {f"{a} to {b}": (b - a).days for a, b in pairwise(days)}
    assert min(gaps.values()) >= GAPS[kind][0], gaps


def test_no_day_holds_two_bls_series():
    """The official schedules put no two of the four releases on one day in the window, so a repeat is a copying
    slip."""
    days = [r.date() for kind in BLS for r in releases(kind)]
    assert len(set(days)) == len(days)


@pytest.mark.parametrize("kind", KINDS)
def test_every_release_falls_on_a_weekday_session(kind):
    for release in releases(kind):
        day = release.date()
        assert day.weekday() < 5, release
        assert sessions.session_on(day) is not None, release
        assert day not in QUIET_WEEKDAYS, release


@pytest.mark.parametrize("kind", KINDS)
def test_every_release_keeps_its_usual_time(kind):
    assert {r.strftime("%H:%M") for r in releases(kind)} == {TIMES[kind]}


@pytest.mark.parametrize("kind", BLS)
def test_every_committed_bls_release_gets_its_fast_window(kind):
    """The FOMC windows are checked in tests/test_v2_news_schedule.py."""
    zone, minute = ZoneInfo(CONFIG.news.schedule_tz), 60_000
    for release in releases(kind):
        moment = int(release.replace(tzinfo=zone).timestamp()) * 1000
        start = moment - CONFIG.bls.release_fast_before_minutes * minute
        end = moment + CONFIG.bls.release_fast_after_minutes * minute
        assert SCHEDULE.cadence(f"bls_{kind}", start - 1) == CONFIG.bls.poll_seconds, release
        assert SCHEDULE.cadence(f"bls_{kind}", start) == CONFIG.bls.release_fast_poll_seconds, release
        assert SCHEDULE.cadence(f"bls_{kind}", end - 1) == CONFIG.bls.release_fast_poll_seconds, release
        assert SCHEDULE.cadence(f"bls_{kind}", end) == CONFIG.bls.poll_seconds, release
