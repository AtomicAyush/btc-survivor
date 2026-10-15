"""The print rules on synthetic stamps: classification by vendor time, price eligibility and replayed prints."""

import dataclasses
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import pytest

from btc_survivor.v2 import sessions
from btc_survivor.v2.config import load_stock_config
from btc_survivor.v2.stocks import prints
from btc_survivor.v2.stocks.prints import (EARLY, ELIGIBLE, EXCLUDED, LATE, OTHER, POST, PRE, REGULAR, PriceRules,
                                           classify, eligibility, in_session, is_replayed, phase_in)
from helpers import ROOT

NY = ZoneInfo("America/New_York")
TUESDAY = date(2026, 10, 13)
EARLY_CLOSE = date(2026, 11, 27)
RULES = PriceRules(late_ms=4_000, excluded=frozenset({"Z"}))


def ny_ms(day: date, hour: int, minute: int, second: int = 0, millis: int = 0) -> int:
    return int(datetime.combine(day, time(hour, minute, second), tzinfo=NY).timestamp()) * 1000 + millis


def utc_ms(*parts: int) -> int:
    return int(datetime(*parts, tzinfo=timezone.utc).timestamp()) * 1000


@pytest.mark.parametrize("at, expected", [
    ((3, 59, 59, 999), OTHER), ((4, 0), PRE), ((9, 29, 59, 999), PRE), ((9, 30), REGULAR),
    ((15, 59, 59, 999), REGULAR), ((16, 0), POST), ((19, 59, 59, 999), POST), ((20, 0), OTHER), ((23, 0), OTHER),
    ((0, 30), OTHER),
])
def test_a_full_day_classifies_by_vendor_time(at, expected):
    assert classify(ny_ms(TUESDAY, *at)) == expected


@pytest.mark.parametrize("at, expected", [
    ((4, 0), PRE), ((12, 59, 59, 999), REGULAR), ((13, 0), POST), ((16, 59, 59, 999), POST), ((17, 0), OTHER),
])
def test_post_market_ends_at_17_on_an_early_close(at, expected):
    assert sessions.session_on(EARLY_CLOSE).early_close
    assert classify(ny_ms(EARLY_CLOSE, *at)) == expected


def test_weekends_and_closures_are_other():
    assert classify(ny_ms(date(2026, 10, 17), 10, 0)) == OTHER     # a Saturday
    assert classify(ny_ms(date(2026, 11, 26), 10, 0)) == OTHER     # Thanksgiving
    assert classify(ny_ms(date(2026, 11, 26), 5, 0)) == OTHER


def test_the_extended_hours_follow_new_york_across_the_clock_change():
    assert classify(utc_ms(2026, 10, 30, 8, 0)) == PRE              # 04:00 EDT
    assert classify(utc_ms(2026, 10, 30, 7, 59, 59)) == OTHER
    assert classify(utc_ms(2026, 11, 2, 8, 59, 59)) == OTHER        # 03:59:59 EST
    assert classify(utc_ms(2026, 11, 2, 9, 0)) == PRE
    assert classify(utc_ms(2026, 11, 2, 14, 30)) == REGULAR
    assert classify(utc_ms(2026, 11, 3, 0, 59, 59)) == POST         # 19:59:59 EST
    assert classify(utc_ms(2026, 11, 3, 1, 0)) == OTHER


@pytest.mark.parametrize("vendor_ms", [0, -1, 10 ** 18, -10 ** 18, utc_ms(2030, 3, 4, 15, 0),
                                       utc_ms(2025, 12, 31, 15, 0)])
def test_a_vendor_time_outside_the_table_is_other_and_never_raises(vendor_ms):
    assert classify(vendor_ms) == OTHER


def test_phase_in_uses_the_session_bounds():
    session = sessions.session_on(TUESDAY)
    assert phase_in(session, session.open_ms) == REGULAR
    assert phase_in(session, session.close_ms) == POST
    assert phase_in(session, session.open_ms - prints.PRE_MARKET_MS) == PRE
    assert phase_in(session, session.close_ms + prints.POST_MARKET_MS) == OTHER


def test_a_candle_needs_receive_and_vendor_time_in_the_regular_hours():
    session = sessions.session_on(TUESDAY)
    o, c = session.open_ms, session.close_ms
    assert in_session(session, o, o)
    assert in_session(session, c - 1, c - 1)
    assert not in_session(session, o + 500, o - 1)                   # a pre market print received after the open
    assert not in_session(session, c, c - 300)                      # a session print received after the close
    assert not in_session(session, o - 1, o - 200)


@pytest.mark.parametrize("age, expected", [
    (0, ELIGIBLE), (4_000, ELIGIBLE), (4_001, LATE), (60_000, LATE), (-2_000, ELIGIBLE), (-2_001, EARLY),
    (-30_000, EARLY),
])
def test_eligibility_by_age_on_arrival(age, expected):
    vendor = ny_ms(TUESDAY, 10, 0)
    assert eligibility(vendor + age, vendor, (), RULES) == expected
    assert prints.age_ms(vendor + age, vendor) == age


def test_an_excluded_condition_and_the_order_of_reasons():
    vendor = ny_ms(TUESDAY, 10, 0)
    assert eligibility(vendor + 100, vendor, ("1", "Z"), RULES) == EXCLUDED
    assert eligibility(vendor + 100, vendor, ("1", "12"), RULES) == ELIGIBLE
    assert eligibility(vendor + 5_000, vendor, ("Z",), RULES) == LATE
    assert eligibility(vendor - 3_000, vendor, ("Z",), RULES) == EARLY
    assert eligibility(vendor + 100, vendor, ("Z",), PriceRules(4_000)) == ELIGIBLE


def test_price_rules_come_from_the_stock_feed_config():
    feed = load_stock_config(ROOT / "configs" / "v2" / "s1_prices.toml").stock_feed
    # Every code Finnhub's table does not mark as updating the last sale, set on 15 October 2026 (PLAN_V2 4.4).
    not_last_sale = {"3", "5", "9", "10", "11", "12", "15", "16", "17", "18", "20", "21", "22", "24", "25", "26", "27",
                     "30", "31", "32", "34", "36", "37", "38", "40"}
    assert PriceRules.from_config(feed) == PriceRules(4_000, frozenset(not_last_sale))
    # Regular trades, sweeps and the trade through exempt flag a sweep carries all set the price, by the owner's
    # decision of 15 October 2026.
    assert {"1", "8", "41"}.isdisjoint(not_last_sale)
    changed = dataclasses.replace(feed, late_print_seconds=7.5, price_excluded_conditions=("Z", "12"))
    assert PriceRules.from_config(changed) == PriceRules(7_500, frozenset({"Z", "12"}))


def test_replayed_prints_are_more_than_one_second_before_the_connected_stamp():
    connected = ny_ms(TUESDAY, 10, 0)
    assert not is_replayed(connected - 1_000, connected)
    assert is_replayed(connected - 1_001, connected)
    assert not is_replayed(connected + 5_000, connected)
    assert not is_replayed(0, None)
