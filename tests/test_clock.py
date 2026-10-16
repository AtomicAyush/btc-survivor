import pytest

from btc_survivor import clock
from btc_survivor.clock import LiveClock, iso, minute_floor, minute_index, minute_of_day, next_minute, parse_iso

T = 1_791_000_000_000  # a fixed instant in late 2026, in UTC milliseconds


def test_minute_helpers():
    boundary = minute_floor(T)
    assert boundary % 60_000 == 0
    assert minute_floor(boundary) == boundary
    assert next_minute(boundary) == boundary + 60_000
    assert next_minute(boundary + 1) == boundary + 60_000
    assert minute_index(boundary) * 60_000 == boundary
    assert minute_of_day(0) == 0
    assert minute_of_day(clock.MS_PER_DAY - 1) == 1439


def test_iso_format_and_round_trip():
    assert iso(0) == "1970-01-01T00:00:00.000Z"
    assert iso(1_500) == "1970-01-01T00:00:01.500Z"
    assert parse_iso(iso(T + 7)) == T + 7


def test_iso_text_order_matches_time_order():
    stamps = [T, T + 1, T + 999, T + 1000, T + 60_000]
    texts = [iso(s) for s in stamps]
    assert texts == sorted(texts)


def test_parse_exchange_timestamps_with_nanoseconds():
    assert iso(parse_iso("2026-10-11T02:53:49.416000123Z")) == "2026-10-11T02:53:49.416Z"
    assert iso(parse_iso("2026-10-11T02:53:49Z")) == "2026-10-11T02:53:49.000Z"
    assert iso(parse_iso("2026-10-11T02:53:49.5+00:00")) == "2026-10-11T02:53:49.500Z"


@pytest.mark.parametrize("text", ["2026-10-11 02:53:49Z", "2026-10-11T02:53:49+02:00", "yesterday"])
def test_parse_rejects_non_utc_or_malformed(text):
    with pytest.raises(ValueError):
        parse_iso(text)


def test_live_stamp_never_goes_backwards_and_respects_floor(monkeypatch):
    c = LiveClock()
    readings = iter([1000, 900, 950, 2000])
    monkeypatch.setattr(c, "now_ms", lambda: next(readings))
    assert c.stamp() == 1000
    assert c.stamp() == 1000          # wall clock stepped back
    assert c.stamp(floor_ms=1200) == 1200  # engine time is ahead of the wall clock
    assert c.stamp() == 2000


def test_monotonic_advances():
    assert LiveClock.monotonic() <= LiveClock.monotonic()
