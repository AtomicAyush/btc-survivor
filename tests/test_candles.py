import numpy as np
import pytest

from btc_survivor.feed.candles import CandleBuilder, MinuteSeries

M = 1_791_000_000_000  # a minute boundary
MIN = 60_000


def test_candle_ohlcv_and_count():
    b = CandleBuilder()
    for price, size in [(100.0, 1.0), (102.0, 0.5), (99.0, 0.25), (101.0, 2.0)]:
        b.add_trade(price, size)
    c = b.close(M, mid=100.5, valid=True)
    assert (c.open, c.high, c.low, c.close) == (100.0, 102.0, 99.0, 101.0)
    assert c.volume == 3.75 and c.trade_count == 4 and c.mid == 100.5 and c.valid and c.ts == M


def test_empty_minute_repeats_previous_close_with_zero_volume():
    b = CandleBuilder()
    first = b.close(M, mid=100.0, valid=True)
    assert first.open is None and first.volume == 0.0  # nothing traded yet at all
    b.add_trade(101.0, 1.0)
    b.close(M + MIN, mid=101.0, valid=True)
    empty = b.close(M + 2 * MIN, mid=101.2, valid=True)
    assert (empty.open, empty.high, empty.low, empty.close) == (101.0,) * 4
    assert empty.volume == 0.0 and empty.trade_count == 0 and empty.mid == 101.2


def test_series_window_is_chronological_and_marks_gaps_invalid():
    s = MinuteSeries(10)
    s.append(M, 100.0, 1.0, True)
    s.append(M + MIN, 101.0, 2.0, True)
    s.append(M + 4 * MIN, 104.0, 4.0, True)  # two minutes skipped
    mid, vol, valid = s.window(s.last_index, 5)
    assert valid.tolist() == [True, True, False, False, True]
    assert mid[[0, 1, 4]].tolist() == [100.0, 101.0, 104.0]
    assert np.isnan(mid[2]) and vol[2] == 0.0


def test_series_invalid_minute_keeps_no_values():
    s = MinuteSeries(5)
    s.append(M, 100.0, 1.0, False)
    mid, vol, valid = s.window(s.last_index, 1)
    assert not valid[0] and np.isnan(mid[0]) and vol[0] == 0.0


def test_series_forgets_minutes_older_than_capacity_and_handles_long_jumps():
    s = MinuteSeries(4)
    for i in range(6):
        s.append(M + i * MIN, 100.0 + i, 1.0, True)
    mid, _, valid = s.window(s.last_index, 4)
    assert mid.tolist() == [102.0, 103.0, 104.0, 105.0]
    s.append(M + 1000 * MIN, 200.0, 1.0, True)
    mid, _, valid = s.window(s.last_index, 4)
    assert valid.tolist() == [False, False, False, True]


def test_series_rejects_out_of_order_and_oversized_windows():
    s = MinuteSeries(4)
    s.append(M, 1.0, 1.0, True)
    with pytest.raises(ValueError):
        s.append(M, 1.0, 1.0, True)
    with pytest.raises(ValueError):
        s.window(s.last_index, 5)


def test_window_ending_after_the_last_minute_is_invalid_there():
    s = MinuteSeries(4)
    s.append(M, 1.0, 1.0, True)
    _, _, valid = s.window(s.last_index + 1, 2)
    assert valid.tolist() == [True, False]
