"""Stock costs by hand: the SEC fee and trading activity fee rounding, fees only on sales including short opens, the
round trips of 3.6, 4.6 and 6.6 bps, borrow across a weekend, the required edge in label standard deviations, rent,
the death line and the Roll diagnostic."""

import dataclasses
import math
from datetime import date

import numpy as np
import pytest

from btc_survivor.account import Account
from btc_survivor.clock import MS_PER_HOUR
from btc_survivor.types import LONG, SHORT, Leg
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import load_stock_config
from btc_survivor.v2.stocks import costs
from btc_survivor.v2.stocks.broker import StockBroker
from btc_survivor.v2.stocks.finnhub import Print
from helpers import ROOT

CFG = load_stock_config(ROOT / "configs" / "v2" / "s1_prices.toml")
COSTS = CFG.stock_costs
FRIDAY = sessions.session_on(date(2026, 10, 16))
MONDAY = sessions.session_on(date(2026, 10, 19))

# Section 5.1 planning prices and section 12 planning volatilities, round trips, dollars at 500 and required edges.
PLANNING = {
    "QQQ": (600, 0.20, 3.6, 0.18, 30, 11.2),
    "MSFT": (500, 0.22, 4.6, 0.23, 33, 12.9),
    "AAPL": (250, 0.25, 4.6, 0.23, 38, 14.1),
    "GOOGL": (250, 0.30, 4.6, 0.23, 45, 16.0),
    "AMZN": (220, 0.30, 4.6, 0.23, 45, 16.0),
    "META": (700, 0.35, 6.6, 0.33, 53, 19.9),
    "NVDA": (180, 0.45, 4.6, 0.23, 68, 21.6),
    "TSLA": (400, 0.60, 6.6, 0.33, 91, 29.3),
}


def thirty_minute_sd_bps(yearly_vol):
    """Three quarters of daily variance inside the session, spread evenly over its 13 half hours."""
    return yearly_vol * math.sqrt(0.75 / 252 / 13) * 1e4


def test_sec_fee_rounds_up_to_the_cent():
    assert costs.sec_fee(COSTS, 500.0) == 0.02          # 0.0103 dollars before rounding
    assert costs.sec_fee(COSTS, 1000.0) == 0.03         # 0.0206 dollars before rounding
    assert costs.sec_fee(COSTS, 100.0) == 0.01          # the one cent minimum
    assert costs.sec_fee(COSTS, 1_000_000.0) == 20.6
    raw = dataclasses.replace(COSTS, fee_round_up_to_cent=False)
    assert costs.sec_fee(raw, 500.0) == pytest.approx(0.0103)


def test_rounding_ignores_float_noise_but_not_a_real_fraction_of_a_cent():
    assert costs.round_fee(0.03) == 0.03
    assert costs.round_fee(0.1 + 0.2) == 0.3            # 0.30000000000000004 stays 30 cents
    assert costs.round_fee(0.0300001) == 0.04
    assert costs.round_fee(0.0) == 0.0


def test_trading_activity_fee_minimum_and_cap():
    assert costs.taf_fee(COSTS, 1.0) == 0.01            # 0.000195 dollars rounds up to the one cent minimum
    assert costs.taf_fee(COSTS, 0.004) == 0.01          # a fractional sale still pays a cent
    assert costs.taf_fee(COSTS, 100.0) == 0.02          # 0.0195
    assert costs.taf_fee(COSTS, 50_000.0) == 9.75
    assert costs.taf_fee(COSTS, 50_206.0) == 9.79       # 9.79017 capped at 9.79 per trade
    assert costs.taf_fee(COSTS, 1_000_000.0) == 9.79
    raw = dataclasses.replace(COSTS, fee_round_up_to_cent=False)
    assert costs.taf_fee(raw, 1.0) == pytest.approx(0.000195)
    assert costs.taf_fee(raw, 1_000_000.0) == 9.79


def test_a_sale_of_about_500_pays_two_cents_and_one_cent():
    fees = costs.leg_fees(COSTS, costs.SELL, 2.0, 250.0)
    assert (fees.sec, fees.taf, fees.total) == (0.02, 0.01, 0.03)
    assert costs.leg_fees(COSTS, costs.BUY, 2.0, 250.0) == costs.NO_FEES


def _execute(broker, acct, legs, price=250.0, ms=None):
    ms = FRIDAY.open_ms + 60 * 60_000 if ms is None else ms
    order = broker.make_order(1, "AAPL", legs, ms - 2_000, 250.0, "decision", np.random.default_rng(0))
    return broker.execute(order, Print("AAPL", price, 100.0, ms - 1_000, ()), max(ms, order.due_ms), acct)


def test_fees_fall_only_on_sales_including_short_opens():
    broker = StockBroker.from_config(CFG)
    acct = Account(10_000.0)
    buy = _execute(broker, acct, [Leg("open", LONG, 2.0)]).fills[0]                   # long open, a buy
    assert (buy.taker_side, buy.fee, buy.sec_fee, buy.taf_fee) == ("buy", 0.0, 0.0, 0.0)
    sell = _execute(broker, acct, [Leg("close", LONG)]).fills[0]                      # long close, a sale
    assert (sell.taker_side, sell.sec_fee, sell.taf_fee, sell.fee) == ("sell", 0.02, 0.01, 0.03)
    short = _execute(broker, acct, [Leg("open", SHORT, 2.0)]).fills[0]                # short open, a sale
    assert (short.taker_side, short.sec_fee, short.taf_fee, short.fee) == ("sell", 0.02, 0.01, 0.03)
    cover = _execute(broker, acct, [Leg("close", SHORT)]).fills[0]                    # short close, a buy
    assert (cover.taker_side, cover.fee) == ("buy", 0.0)
    assert acct.fees == pytest.approx(0.06)
    assert acct.invariant_errors(None) == []


@pytest.mark.parametrize("symbol", sorted(PLANNING))
def test_round_trips_of_3_6_4_6_and_6_6_bps_with_their_dollars(symbol):
    price, _, bps, dollars, _, _ = PLANNING[symbol]
    broker = StockBroker.from_config(CFG)
    qty = broker.open_qty(LONG, price)
    trip = costs.round_trip(COSTS, symbol, LONG, qty, price, 30)
    assert trip.fees_bps == pytest.approx(0.6, abs=1e-5)   # 0.03 dollars of fees on about 500
    assert trip.borrow_bps == 0.0
    assert round(trip.total_bps, 1) == bps
    assert round(trip.dollars, 2) == dollars
    assert trip.legs_bps == 2 * (COSTS.half_spread_bps[symbol] + COSTS.extra_slippage_bps)


def test_a_short_round_trip_adds_borrow_over_the_horizon_on_its_whole_share_notional():
    trip = costs.round_trip(COSTS, "TSLA", SHORT, 1.0, 400.0, 30)
    assert trip.notional == 400.0
    assert trip.borrow_bps == pytest.approx(100.0 / 8760 / 2)        # about 0.0057 bps for 30 minutes
    assert trip.fees_bps == pytest.approx(0.02 / 400 * 1e4)          # 0.01 of each fee on a 400 dollar sale
    assert costs.borrow_bps_per_hour(COSTS) == pytest.approx(0.0114155, abs=1e-7)
    with pytest.raises(ValueError):
        costs.round_trip(COSTS, "TSLA", SHORT, 0.0, 400.0, 30)


def test_borrow_accrues_across_a_weekend_in_calendar_time():
    """A whole share short carried from Friday 15:50 ET to its session flatten at Monday 09:45 ET pays borrow for
    every calendar hour between, not only for the session minutes."""
    assert FRIDAY.day.weekday() == 4 and MONDAY.day.weekday() == 0
    broker = StockBroker.from_config(CFG)
    acct = Account(10_000.0)
    opened = FRIDAY.boundary(380)                        # 15:50 ET
    _execute(broker, acct, [Leg("open", SHORT, 2.0)], ms=opened)
    entry_notional = acct.qty * acct.entry_price
    filled_at = acct.entry_ms
    flattened = MONDAY.boundary(15)                      # 09:45 ET
    order = broker.make_order(1, "AAPL", [Leg("close", SHORT)], flattened, 250.0, "session_flatten",
                              np.random.default_rng(1))
    ex = broker.execute(order, Print("AAPL", 250.0, 10.0, order.due_ms, ()), order.due_ms, acct)
    hours = (order.due_ms - filled_at) / MS_PER_HOUR
    assert 65.9 < hours < 66.0
    expected = 0.01 * entry_notional * hours / (365 * 24)
    assert ex.closed.funding == pytest.approx(expected, rel=1e-12)
    assert acct.funding == pytest.approx(expected, rel=1e-12)
    assert costs.borrow_cost(COSTS, entry_notional, filled_at, order.due_ms) == pytest.approx(expected, rel=1e-12)
    session_hours = 10 / 60 + 15 / 60
    assert expected > 100 * costs.borrow_cost(COSTS, entry_notional, 0, round(session_hours * MS_PER_HOUR))
    assert ex.closed.exit_reason == "session_flatten"
    assert acct.invariant_errors(None) == []


@pytest.mark.parametrize("symbol", sorted(PLANNING))
def test_required_edge_is_cost_plus_a_quarter_label_standard_deviation(symbol):
    price, vol, bps, _, sd, edge = PLANNING[symbol]
    label_sd = thirty_minute_sd_bps(vol)
    assert label_sd == pytest.approx(sd, abs=0.5)
    qty = StockBroker.from_config(CFG).open_qty(LONG, price)
    cost = costs.round_trip(COSTS, symbol, LONG, qty, price, 30).total_bps
    required = costs.required_edge_bps(cost, CFG.trading.margin_sigma, label_sd)
    assert required == pytest.approx(cost + 0.25 * label_sd)
    assert required == pytest.approx(edge, abs=0.1)


def test_required_edge_scales_with_the_label_standard_deviation():
    assert costs.required_edge_bps(4.6, 0.25, 0.0) == 4.6
    assert costs.required_edge_bps(4.6, 0.25, 40.0) == pytest.approx(14.6)
    assert costs.required_edge_bps(4.6, 0.25, 80.0) - costs.required_edge_bps(4.6, 0.25, 40.0) == pytest.approx(10.0)


def test_rent_per_usable_minute_and_per_session():
    assert costs.rent_per_minute(COSTS) == pytest.approx(0.0043081, abs=1e-7)
    assert costs.session_rent(COSTS) == pytest.approx(1.68, abs=0.005)
    assert costs.rent_per_minute(COSTS) * 210 == pytest.approx(0.90, abs=0.005)
    assert costs.rent_per_minute(COSTS) * 390 * 252 == pytest.approx(423.4, abs=0.1)


def test_death_line_is_a_round_trip_on_the_actual_notional_plus_a_session_of_rent():
    qqq = costs.death_line(COSTS, "QQQ", LONG, 500 / 600, 600.0, 30)
    assert qqq == pytest.approx(0.18 + costs.session_rent(COSTS), abs=0.005)            # about 1.86
    meta_short = costs.death_line(COSTS, "META", SHORT, 1.0, 700.0, 30)
    trip = costs.round_trip(COSTS, "META", SHORT, 1.0, 700.0, 30)
    assert meta_short == pytest.approx(trip.total_bps / 1e4 * 700.0 + costs.session_rent(COSTS))
    assert 1.85 < qqq < meta_short < 2.15


def test_fill_and_mark_prices_move_against_the_bot():
    assert costs.allowance_bps(COSTS, "QQQ") == 1.5
    assert costs.allowance_bps(COSTS, "AAPL") == 2.0
    assert costs.allowance_bps(COSTS, "TSLA") == 3.0
    assert costs.fill_price(100.0, costs.BUY, 2.0) == pytest.approx(100.02)
    assert costs.fill_price(100.0, costs.SELL, 2.0) == pytest.approx(99.98)
    assert costs.mark_price(100.0, LONG, 3.0) == pytest.approx(99.97)
    assert costs.mark_price(100.0, SHORT, 3.0) == pytest.approx(100.03)


def test_roll_diagnostic_recovers_a_planted_bounce_and_only_reports():
    rng = np.random.default_rng(7)
    mid = 250.0 * np.exp(np.cumsum(rng.normal(0, 0.2e-4, 20_000)))
    prices = mid * (1 + 1.5e-4 * rng.choice([-1.0, 1.0], len(mid)))
    estimate = costs.roll_half_spread_bps(prices)
    assert estimate == pytest.approx(1.5, rel=0.1)
    dp = 1e4 * np.diff(np.log(prices))
    assert estimate == pytest.approx(math.sqrt(-np.cov(dp[1:], dp[:-1])[0, 1]), rel=1e-9)
    acc = costs.RollAccumulator()
    for price in prices:
        acc.add(price)
    check = costs.spread_check(COSTS, "AAPL", acc)
    assert check == ("AAPL", 20_000, acc.half_spread_bps(), 1.0, True)
    assert not costs.spread_check(COSTS, "TSLA", acc).flagged
    assert COSTS.half_spread_bps["AAPL"] == 1.0                  # reported, never applied
    drift = 250.0 * np.exp(np.cumsum(1e-4 * np.sin(np.arange(200) / 10)))
    assert costs.roll_half_spread_bps(drift) is None             # changes that persist, no negative covariance
    assert costs.roll_half_spread_bps([250.0, 250.1, 250.0]) is None
    empty = costs.spread_check(COSTS, "AAPL", costs.RollAccumulator())
    assert (empty.prints, empty.estimate_bps, empty.flagged) == (0, None, False)
