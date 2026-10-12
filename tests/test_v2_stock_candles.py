"""Receive time candles on synthetic prints: boundaries, classification, eligibility, replayed prints, the median
reference price, its reset at the open, empty minutes and the tick rule."""

from datetime import date

import pytest

from btc_survivor.v2 import sessions
from btc_survivor.v2.stocks.candles import OUTSIDE, REPLAYED, StockCandle, StockCandles
from btc_survivor.v2.stocks.finnhub import Print, PrintBatch
from btc_survivor.v2.stocks.prints import EARLY, PriceRules

MIN = 60_000
DAY = sessions.session_on(date(2026, 10, 13))
NEXT = sessions.session_on(date(2026, 10, 14))
OPEN = DAY.open_ms
B1, B2, B3, B4 = (OPEN + n * MIN for n in range(1, 5))
SYMBOLS = ("AAPL", "MSFT", "QQQ")
RULES = PriceRules(late_ms=4_000, excluded=frozenset({"Z"}))


@pytest.fixture
def candles():
    return StockCandles(SYMBOLS, RULES)


def put(candles, recv, price, size=1.0, symbol="AAPL", age=100, conditions=()):
    """Add one print received at recv whose vendor time is age milliseconds earlier."""
    candles.add(PrintBatch(recv, 1, (Print(symbol, price, size, recv - age, conditions),)))


def aapl(candles, boundary):
    return candles.close(boundary)[0]


def test_a_print_at_b_minus_1_ms_is_in_candle_b_and_one_at_b_in_the_next(candles):
    put(candles, B1 - 1, 10.0, 2.0)
    with pytest.raises(ValueError, match="close the minute"):
        put(candles, B1, 11.0, 3.0)
    first = aapl(candles, B1)
    assert (first.ts, first.k, first.volume, first.close) == (B1, DAY.first_k, 2.0, 10.0)
    put(candles, B1, 11.0, 3.0)
    second = aapl(candles, B2)
    assert (second.ts, second.k, second.volume, second.open, second.close) == (B2, DAY.first_k + 1, 3.0, 11.0, 11.0)


def test_close_gives_every_symbol_in_config_order(candles):
    put(candles, OPEN + 10, 10.0, symbol="QQQ", age=5)
    out = candles.close(B1)
    assert [c.symbol for c in out] == list(SYMBOLS)
    assert [c.prints for c in out] == [0, 0, 1]
    assert all(isinstance(c, StockCandle) and not c.valid for c in out)


def test_a_print_enters_a_candle_only_with_vendor_and_receive_time_in_the_session(candles):
    put(candles, OPEN - 5, 9.0, 4.0, age=50)                  # received in pre market
    put(candles, OPEN + 400, 8.0, 5.0, age=500)               # vendor time 09:29:59.900, pre market
    put(candles, OPEN + 500, 10.0, 1.0, age=400)              # vendor time 09:30:00.100
    candle = aapl(candles, B1)
    assert (candle.volume, candle.prints, candle.open, candle.ref_price) == (1.0, 1, 10.0, 10.0)
    assert candles.ignored[OUTSIDE] == 2


def test_the_last_minute_takes_prints_received_before_the_close_only(candles):
    close = DAY.close_ms
    put(candles, close - 1, 10.0, 2.0)
    last = aapl(candles, close)
    assert (last.k, last.volume) == (DAY.last_k, 2.0)
    put(candles, close, 11.0, 3.0, age=200)                   # vendor time in session, received after the close
    put(candles, close + 5_000, 12.0, 3.0)                    # post market
    assert candles.ignored[OUTSIDE] == 2
    with pytest.raises(ValueError):
        candles.close(close + MIN)


def test_a_late_print_counts_for_volume_only(candles):
    put(candles, OPEN + 10_000, 100.0, 1.0)
    put(candles, OPEN + 20_000, 200.0, 5.0, age=4_001)        # late
    put(candles, OPEN + 30_000, 101.0, 2.0, age=4_000)        # exactly at the limit, eligible
    put(candles, OPEN + 40_000, 50.0, 7.0, conditions=("1", "Z"))  # excluded condition
    candle = aapl(candles, B1)
    assert (candle.volume, candle.prints, candle.price_prints, candle.late_prints) == (15.0, 4, 2, 1)
    assert (candle.open, candle.high, candle.low, candle.close) == (100.0, 101.0, 100.0, 101.0)
    assert candle.ref_price == 100.5
    assert (candle.buy_volume, candle.sell_volume) == (2.0, 0.0)
    assert candles.last_eligible_ms("AAPL") == OPEN + 30_000
    put(candles, B1 + 1_000, 100.5, 4.0)                      # below 101, the previous eligible print, not 50 or 200
    assert aapl(candles, B2).sell_volume == 4.0


def test_a_print_more_than_2_seconds_early_is_refused(candles):
    put(candles, OPEN + 10_000, 100.0, 3.0, age=-2_001)
    put(candles, OPEN + 11_000, 101.0, 2.0, age=-2_000)
    candle = aapl(candles, B1)
    assert (candle.volume, candle.prints, candle.price_prints, candle.open) == (2.0, 1, 1, 101.0)
    assert candles.ignored[EARLY] == 1


def test_replayed_prints_are_ignored_for_price_and_volume(candles):
    connected = OPEN + 5_000
    candles.on_connected(connected)
    candles.add(PrintBatch(connected + 10, 2, (Print("AAPL", 90.0, 9.0, connected - 1_001, ()),
                                                Print("AAPL", 100.0, 1.0, connected - 1_000, ()))))
    candle = aapl(candles, B1)
    assert (candle.volume, candle.prints, candle.ref_price, candle.low) == (1.0, 1, 100.0, 100.0)
    assert candles.ignored[REPLAYED] == 1


def test_the_reference_price_is_the_median_of_up_to_5_eligible_prints(candles):
    put(candles, OPEN + 1_000, 10.0)
    assert aapl(candles, B1).ref_price == 10.0
    put(candles, B1 + 1_000, 30.0)
    assert aapl(candles, B2).ref_price == 20.0
    put(candles, B2 + 1_000, 20.0)
    assert aapl(candles, B3).ref_price == 20.0
    for i, price in enumerate([50.0, 40.0, 60.0]):
        put(candles, B3 + 1_000 * (i + 1), price)
    assert aapl(candles, B4).ref_price == 40.0                # the last five, 30, 20, 50, 40 and 60
    put(candles, B4 + 1_000, 1_000.0, age=9_000)              # a late print never enters the median
    assert aapl(candles, B4 + MIN).ref_price == 40.0


def test_the_buffer_and_the_signs_reset_at_the_open(candles):
    put(candles, DAY.close_ms - 2_000, 100.0)
    put(candles, DAY.close_ms - 1_000, 99.0)
    assert aapl(candles, DAY.close_ms).ref_price == 99.5
    quiet = aapl(candles, NEXT.open_ms + MIN)
    assert quiet.k == NEXT.first_k == DAY.last_k + 1
    assert quiet.ref_price is None and quiet.close == 99.0 and not quiet.judged(True).valid
    put(candles, NEXT.open_ms + MIN + 1_000, 105.0, 3.0)      # above yesterday's last print, yet unsigned
    candle = aapl(candles, NEXT.open_ms + 2 * MIN)
    assert (candle.ref_price, candle.buy_volume, candle.sell_volume) == (105.0, 0.0, 0.0)


def test_the_reset_happens_on_the_first_print_of_a_session_too(candles):
    put(candles, DAY.close_ms - 1_000, 100.0)
    aapl(candles, DAY.close_ms)
    put(candles, NEXT.open_ms + 10, 101.0, age=5)
    candle = aapl(candles, NEXT.open_ms + MIN)
    assert (candle.ref_price, candle.buy_volume) == (101.0, 0.0)


def test_an_empty_minute_repeats_the_previous_close_and_is_never_healthy(candles):
    nothing = aapl(candles, B1)
    assert (nothing.open, nothing.close, nothing.ref_price, nothing.volume) == (None, None, None, 0.0)
    put(candles, B1 + 1_000, 100.0, 2.0)
    put(candles, B1 + 2_000, 102.0, 1.0)
    assert aapl(candles, B2).judged(True).valid
    empty = aapl(candles, B3)
    assert (empty.open, empty.high, empty.low, empty.close) == (102.0,) * 4
    assert (empty.volume, empty.prints, empty.buy_volume, empty.sell_volume) == (0.0, 0, 0.0, 0.0)
    assert empty.ref_price == 101.0
    assert not empty.judged(True).valid
    put(candles, B3 + 1_000, 500.0, 6.0, age=9_000)          # only a late print
    late_only = aapl(candles, B4)
    assert (late_only.close, late_only.volume, late_only.price_prints) == (102.0, 6.0, 0)
    assert not late_only.judged(True).valid


def test_judged_keeps_an_unhealthy_minute_invalid(candles):
    put(candles, OPEN + 1_000, 100.0)
    candle = aapl(candles, B1)
    assert candle.judged(True).valid and not candle.judged(False).valid and not candle.valid


def test_the_tick_rule_signs_each_eligible_print_against_the_previous_one_of_its_symbol(candles):
    prices = [10.0, 11.0, 11.0, 10.0, 10.0, 12.0]
    for i, price in enumerate(prices):
        put(candles, OPEN + 1_000 * (i + 1), price, float(i + 1))
        put(candles, OPEN + 1_000 * (i + 1) + 500, 50.0 - i, 100.0, symbol="MSFT")
    aapl_candle, msft, _ = candles.close(B1)
    assert (aapl_candle.buy_volume, aapl_candle.sell_volume, aapl_candle.volume) == (11.0, 9.0, 21.0)
    assert (msft.buy_volume, msft.sell_volume) == (0.0, 500.0)
    put(candles, B1 + 1_000, 12.0, 7.0)                       # equal, so the previous sign, plus one
    assert aapl(candles, B2).buy_volume == 7.0


def test_one_batch_signs_its_prints_in_vendor_time_order(candles):
    recv = OPEN + 5_000
    candles.add(PrintBatch(recv, 1, (Print("AAPL", 10.0, 1.0, recv - 300, ()), Print("AAPL", 9.0, 2.0, recv - 200, ()),
                                     Print("AAPL", 9.5, 4.0, recv - 100, ()))))
    candle = aapl(candles, B1)
    assert (candle.buy_volume, candle.sell_volume, candle.open, candle.close) == (4.0, 2.0, 10.0, 9.5)


def test_boundaries_close_in_order_and_only_inside_a_session(candles):
    with pytest.raises(ValueError, match="closes no session minute"):
        candles.close(OPEN)
    with pytest.raises(ValueError, match="closes no session minute"):
        candles.close(B1 + 1)
    candles.close(B2)
    with pytest.raises(ValueError, match="not after"):
        candles.close(B2)
    with pytest.raises(ValueError, match="closed minute"):
        put(candles, B2 - 1, 10.0)


def test_a_skipped_boundary_is_allowed_when_no_print_waits(candles):
    put(candles, OPEN + 1_000, 10.0)
    candles.close(B1)
    put(candles, B3 + 1_000, 11.0)
    with pytest.raises(ValueError, match="prints wait"):
        candles.close(B3)
    candle = aapl(candles, B4)
    assert (candle.k, candle.close, candle.ref_price) == (DAY.first_k + 3, 11.0, 10.5)
