"""StockSeries on synthetic minutes: its capacity of 420, the buy and sell rings, and session starts by k."""

import dataclasses
from datetime import date

import numpy as np
import pytest

from btc_survivor.v2 import sessions
from btc_survivor.v2.sessions import CalendarError
from btc_survivor.v2.stocks.candles import StockCandle, StockCandles
from btc_survivor.v2.stocks.finnhub import Print, PrintBatch
from btc_survivor.v2.stocks.prints import PriceRules
from btc_survivor.v2.stocks.series import MIN_CAPACITY, StockSeries

DAY = sessions.session_on(date(2026, 10, 13))
NEXT = sessions.session_on(date(2026, 10, 14))
FRIDAY = sessions.session_on(date(2026, 10, 16))
MONDAY = sessions.session_on(date(2026, 10, 19))


def fill(series, first, last, valid=True):
    for k in range(first, last + 1):
        series.append(k, 100.0 + k, 10.0 + k, 1.0 * k, 0.5 * k, valid)


def test_the_capacity_is_420_and_never_less():
    assert MIN_CAPACITY == 420 and StockSeries().capacity == 420
    with pytest.raises(ValueError, match="at least 420"):
        StockSeries(419)
    assert StockSeries(500).capacity == 500


def test_the_ring_holds_the_last_420_minutes():
    series = StockSeries()
    fill(series, 1, 421)
    w = series.window(421, 420)
    assert w.valid.all() and w.k.tolist() == list(range(2, 422))
    assert w.ref_price[0] == 102.0 and w.buy_volume[-1] == 421.0 and w.sell_volume[-1] == 210.5
    assert not series.window(420, 420).valid[0]               # k = 1 has left the ring
    with pytest.raises(ValueError):
        series.window(421, 421)
    assert series.last_k == 421 and series.valid_total == 421


def test_buy_and_sell_rings_follow_validity():
    series = StockSeries()
    series.append(1, 100.0, 10.0, 6.0, 3.0, True)
    series.append(2, 101.0, 20.0, 8.0, 4.0, False)
    series.append(3, None, 30.0, 9.0, 5.0, True)              # valid without a reference price is not valid
    series.append(5, 103.0, 40.0, 7.0, 2.0, True)             # k = 4 skipped
    w = series.window(5, 5)
    assert w.valid.tolist() == [True, False, False, False, True]
    assert w.buy_volume.tolist() == [6.0, 0.0, 0.0, 0.0, 7.0]
    assert w.sell_volume.tolist() == [3.0, 0.0, 0.0, 0.0, 2.0]
    assert w.volume.tolist() == [10.0, 0.0, 0.0, 0.0, 40.0]
    assert np.isnan(w.ref_price[1:4]).all()


def test_a_stale_buy_or_sell_slot_is_masked_after_a_long_skip():
    series = StockSeries()
    series.append(1, 100.0, 10.0, 7.0, 7.0, True)
    series.append(422, 101.0, 10.0, 2.0, 1.0, True)           # k = 421 shares k = 1's slot and was skipped
    w = series.window(422, 2)
    assert w.valid.tolist() == [False, True]
    assert w.buy_volume.tolist() == [0.0, 2.0] and w.sell_volume.tolist() == [0.0, 1.0]


def test_minutes_append_in_order_from_k_1():
    series = StockSeries()
    with pytest.raises(ValueError, match="below 1"):
        series.append(0, 100.0, 1.0, 0.0, 0.0, True)
    series.append(5, 100.0, 1.0, 0.0, 0.0, True)
    with pytest.raises(ValueError):
        series.append(5, 100.0, 1.0, 0.0, 0.0, True)
    w = series.window(6, 2)
    assert w.valid.tolist() == [True, False]                  # newer than the last append


def test_session_starts_come_from_the_calendar():
    middle = DAY.first_k + 200
    assert StockSeries.session_start(middle) == DAY.first_k
    assert StockSeries.session_start(DAY.first_k) == DAY.first_k
    assert StockSeries.session_start(DAY.last_k) == DAY.first_k
    assert StockSeries.session_start(DAY.last_k + 1) == NEXT.first_k == DAY.last_k + 1
    assert StockSeries.session_start(MONDAY.first_k + 5) == MONDAY.first_k == FRIDAY.last_k + 1
    with pytest.raises(CalendarError):
        StockSeries.session_start(0)


def test_a_night_adds_no_minutes_so_a_window_runs_across_the_open():
    series = StockSeries()
    fill(series, FRIDAY.last_k - 2, MONDAY.first_k + 2)
    w = series.window(MONDAY.first_k + 2, 6)
    assert w.valid.all()
    assert (w.k >= series.session_start(MONDAY.first_k + 2)).tolist() == [False, False, False, True, True, True]


def test_append_candle_takes_k_and_the_judged_validity():
    series = StockSeries()
    candle = StockCandle("AAPL", DAY.boundary(1), DAY.first_k, 10.0, 11.0, 9.0, 10.5, 10.25, 30.0, 12.0, 8.0, 4, 3, 1)
    series.append_candle(candle)
    assert not series.window(DAY.first_k, 1).valid[0]
    series.append_candle(dataclasses.replace(candle, ts=DAY.boundary(2), k=DAY.first_k + 1).judged(True))
    w = series.window(DAY.first_k + 1, 1)
    assert w.valid[0] and w.ref_price[0] == 10.25 and w.volume[0] == 30.0
    assert (w.buy_volume[0], w.sell_volume[0]) == (12.0, 8.0)


def test_a_full_session_of_candles_fills_390_minutes_by_k():
    candles = StockCandles(("AAPL",), PriceRules(4_000))
    series = StockSeries()
    for minute in range(1, DAY.minutes + 1):
        recv = DAY.boundary(minute) - 30_000
        candles.add(PrintBatch(recv, 1, (Print("AAPL", 100.0 + minute, 1.0, recv - 50, ()),)))
        [candle] = candles.close(DAY.boundary(minute))
        series.append_candle(candle.judged(True))
    w = series.window(DAY.last_k, DAY.minutes)
    assert w.valid.all() and w.k[0] == DAY.first_k
    assert w.volume.sum() == 390.0 and w.buy_volume.sum() == 389.0 and w.sell_volume.sum() == 0.0
    assert w.ref_price[-1] == 488.0                           # the median of the last five, 486 to 490
