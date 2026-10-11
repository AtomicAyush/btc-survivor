import numpy as np
import pytest

from btc_survivor.account import Account
from btc_survivor.broker import Broker, estimate_round_trip
from btc_survivor.config import load_config
from btc_survivor.feed.book import LocalBook
from btc_survivor.types import LONG, SHORT, Leg
from helpers import EXAMPLE_CONFIG

CFG = load_config(EXAMPLE_CONFIG)
COSTS = CFG.costs  # 90 bps taker, 1 bp slippage, 1000 ms plus up to 300 ms, 0.5 bps per hour, 2 bps short open
HOUR = 3_600_000


def book():
    b = LocalBook(20)
    b.apply_snapshot(bids=[(99.9, 1.0), (99.8, 2.0)], asks=[(100.1, 1.0), (100.2, 2.0)])
    return b


def run(broker, acct, legs, decided_ms=0, reason="decision", rng=None, at=None, b=None):
    """Make an order and settle it at its due time, or at the given time."""
    order = broker.make_order(1, legs, decided_ms, 100.0, reason, rng or np.random.default_rng(1))
    return order, broker.execute(order, b or book(), order.due_ms if at is None else at, acct)


def test_cost_estimate_long_hand_computed():
    est = estimate_round_trip(book(), 50.0, COSTS, LONG, 30)
    assert est.fee_bps == 90.0 and est.spread_bps == pytest.approx(20.0) and est.impact_bps == pytest.approx(0.0)
    assert est.total_bps == pytest.approx(2 * 90 + 20 + 0 + 2 * 1)


def test_cost_estimate_includes_impact_beyond_best_price():
    est = estimate_round_trip(book(), 150.0, COSTS, LONG, 30)  # 1.5 BTC, walks into the 100.2 level
    avg = (100.1 * 1.0 + 100.2 * 0.5) / 1.5
    impact = (avg - 100.1) / 100.0 * 1e4
    assert est.impact_bps == pytest.approx(impact)
    assert est.total_bps == pytest.approx(180 + 20 + 2 * impact + 2)


def test_cost_estimate_short_adds_open_fee_and_one_horizon_of_funding():
    est = estimate_round_trip(book(), 50.0, COSTS, SHORT, 30)
    assert est.short_bps == pytest.approx(2.0 + 0.5 * 0.5)
    assert est.total_bps == pytest.approx(202 + 2.25)


def test_cost_estimate_short_impact_and_horizon_hand_computed():
    est = estimate_round_trip(book(), 150.0, COSTS, SHORT, 15)  # 1.5 BTC sold into the bids
    avg = (99.9 * 1.0 + 99.8 * 0.5) / 1.5
    impact = (99.9 - avg) / 100.0 * 1e4
    assert est.impact_bps == pytest.approx(impact)
    assert est.short_bps == pytest.approx(2.0 + 0.5 * 15 / 60)
    assert est.total_bps == pytest.approx(180 + 20 + 2 * impact + 2 + 2.0 + 0.125)


def test_cost_estimate_is_none_without_depth_or_valid_book():
    assert estimate_round_trip(book(), 1000.0, COSTS, LONG, 30) is None
    assert estimate_round_trip(LocalBook(), 50.0, COSTS, LONG, 30) is None


def test_latency_is_base_plus_seeded_jitter():
    broker = Broker(COSTS, 500.0)
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    lat_a = [broker.make_order(1, [], 0, 100.0, "decision", rng_a).latency_ms for _ in range(2000)]
    lat_b = [broker.make_order(1, [], 0, 100.0, "decision", rng_b).latency_ms for _ in range(2000)]
    assert lat_a == lat_b
    assert min(lat_a) == 1000 and max(lat_a) == 1300


def test_order_cannot_fill_before_it_is_due():
    broker = Broker(COSTS, 50.0)
    acct = Account(10000.0)
    order = broker.make_order(1, [Leg("open", LONG, 0.5)], 0, 100.0, "decision", np.random.default_rng(1))
    with pytest.raises(ValueError, match="not due"):
        broker.execute(order, book(), order.due_ms - 1, acct)
    assert not acct.is_open


def test_fill_exactly_at_and_just_after_the_due_instant():
    for delay in (0, 1):
        broker = Broker(COSTS, 50.0)
        acct = Account(10000.0)
        order, ex = run(broker, acct, [Leg("open", LONG, 0.5)], at=None)
        order2 = broker.make_order(1, [Leg("close", LONG)], 60_000, 100.0, "decision", np.random.default_rng(2))
        ex2 = broker.execute(order2, book(), order2.due_ms + delay, acct)
        assert ex.fills[0].filled_ms == order.due_ms
        assert ex2.fills[0].filled_ms == order2.due_ms + delay
        assert ex2.fills[0].latency_ms == order2.due_ms - 60_000


def test_open_long_fill_price_fee_and_slippage_hand_computed():
    broker = Broker(COSTS, 50.0)
    acct = Account(10000.0)
    qty = broker.open_qty(100.0)
    assert qty == 0.5
    order, ex = run(broker, acct, [Leg("open", LONG, qty)])
    fill = ex.fills[0]
    price = 100.1 * 1.0001
    fee = 0.009 * 0.5 * price
    assert fill.status == "filled" and fill.taker_side == "buy"
    assert fill.book_avg_price == pytest.approx(100.1)
    assert fill.price == pytest.approx(price)
    assert fill.fee == pytest.approx(fee)
    assert fill.slippage_bps == pytest.approx((price - 100.0) / 100.0 * 1e4)
    assert fill.latency_ms == order.due_ms - order.decided_ms
    assert acct.side == LONG and acct.qty == 0.5 and acct.entry_price == pytest.approx(price)
    assert acct.cash == pytest.approx(10000.0 - fee)
    assert ex.opened_trade_id == 1 and broker.next_trade_id == 2


def test_close_long_settles_gross_and_fee_and_uses_held_quantity():
    broker = Broker(COSTS, 50.0)
    acct = Account(10000.0)
    rng = np.random.default_rng(1)
    run(broker, acct, [Leg("open", LONG, 0.5)], rng=rng)
    open_price, open_fee = acct.entry_price, acct.entry_fee
    _, ex = run(broker, acct, [Leg("close", LONG, qty=99.0)], decided_ms=60_000, rng=rng)
    close_price = 99.9 * 0.9999
    close_fee = 0.009 * 0.5 * close_price
    closed = ex.closed
    assert ex.fills[0].qty == 0.5  # the held quantity, not the leg's
    assert closed.exit_reason == "signal"
    assert closed.gross == pytest.approx(0.5 * (close_price - open_price))
    assert closed.net == pytest.approx(closed.gross - open_fee - close_fee)
    assert acct.cash == pytest.approx(10000.0 - open_fee - close_fee + closed.gross)
    assert acct.trades == 1 and acct.wins == 0 and not acct.is_open
    assert ex.fills[0].slippage_bps == pytest.approx((100.0 - close_price) / 100.0 * 1e4)


def test_short_pays_open_charge_and_hourly_funding():
    broker = Broker(COSTS, 100.0)
    acct = Account(10000.0)
    rng = np.random.default_rng(1)
    order, _ = run(broker, acct, [Leg("open", SHORT, 1.0)], rng=rng)
    opened = order.due_ms
    price = 99.9 * 0.9999
    charge = 2.0 / 1e4 * 1.0 * price
    assert acct.funding == pytest.approx(charge)
    paid = acct.accrue_funding(opened + 2 * HOUR, COSTS.short_funding_bps_per_hour)
    assert paid == pytest.approx(0.5 / 1e4 * 1.0 * price * 2)
    close_order = broker.make_order(1, [Leg("close", SHORT)], opened + 2 * HOUR, 100.0, "stop", rng)
    ex = broker.execute(close_order, book(), opened + 3 * HOUR, acct)
    assert ex.closed.exit_reason == "stop"
    assert ex.closed.funding == pytest.approx(charge + 0.5 / 1e4 * price * 3)
    assert acct.stop_loss_losses == pytest.approx(-ex.closed.gross)
    assert acct.invariant_errors(None) == []


def test_flip_closes_then_opens_and_both_legs_pay():
    broker = Broker(COSTS, 50.0)
    acct = Account(10000.0)
    rng = np.random.default_rng(1)
    run(broker, acct, [Leg("open", LONG, 0.5)], rng=rng)
    _, ex = run(broker, acct, [Leg("close", LONG), Leg("open", SHORT, 0.5)], decided_ms=60_000, rng=rng)
    assert [f.leg for f in ex.fills] == ["close", "open"]
    assert all(f.fee > 0 for f in ex.fills)
    assert ex.closed.exit_reason == "flip"
    assert acct.side == SHORT and ex.opened_trade_id == 2


def test_flip_legs_share_depth_so_the_second_leg_pays_deeper_levels():
    broker = Broker(COSTS, 100.0)
    acct = Account(10000.0)
    rng = np.random.default_rng(1)
    run(broker, acct, [Leg("open", LONG, 1.0)], rng=rng)  # buys the whole 100.1 ask level
    _, ex = run(broker, acct, [Leg("close", LONG), Leg("open", SHORT, 1.0)], decided_ms=60_000, rng=rng)
    close_fill, open_fill = ex.fills
    assert close_fill.book_avg_price == pytest.approx(99.9)  # the close takes the 1.0 at 99.9
    assert open_fill.book_avg_price == pytest.approx(99.8)  # the short open must go one level deeper
    assert open_fill.levels == 2


def test_order_too_large_for_visible_depth_is_rejected_and_logged():
    broker = Broker(COSTS, 1000.0)
    acct = Account(10000.0)
    _, ex = run(broker, acct, [Leg("open", LONG, 10.0)])
    assert ex.rejected == "insufficient_depth"
    assert ex.fills[0].status == "rejected" and ex.fills[0].price is None
    assert acct.cash == 10000.0 and not acct.is_open


def test_sell_too_large_for_visible_depth_is_rejected():
    broker = Broker(COSTS, 1000.0)
    acct = Account(10000.0)
    _, ex = run(broker, acct, [Leg("open", SHORT, 3.5)])  # only 3.0 visible on the bid side
    assert ex.rejected == "insufficient_depth" and ex.fills[0].taker_side == "sell"
    assert not acct.is_open


def test_flip_whose_open_leg_fails_keeps_the_close():
    broker = Broker(COSTS, 50.0)
    acct = Account(10000.0)
    rng = np.random.default_rng(1)
    run(broker, acct, [Leg("open", LONG, 0.5)], rng=rng)
    _, ex = run(broker, acct, [Leg("close", LONG), Leg("open", SHORT, 50.0)], decided_ms=60_000, rng=rng)
    assert [f.status for f in ex.fills] == ["filled", "rejected"]
    assert not acct.is_open and ex.closed is not None


def test_rejected_close_stops_the_rest_of_a_flip():
    broker = Broker(COSTS, 50.0)
    acct = Account(10000.0)
    _, ex = run(broker, acct, [Leg("close", LONG), Leg("open", SHORT, 0.5)])
    assert len(ex.fills) == 1 and ex.fills[0].status == "rejected"
    assert ex.rejected == "no_position" and not acct.is_open


def test_rejections_for_invalid_book_and_missing_position():
    broker = Broker(COSTS, 50.0)
    acct = Account(10000.0)
    _, ex = run(broker, acct, [Leg("close", LONG)], reason="stop")
    assert ex.rejected == "no_position"
    _, ex = run(broker, acct, [Leg("open", LONG, 0.5)], b=LocalBook())
    assert ex.rejected == "book_invalid"
    order = broker.make_order(1, [Leg("open", LONG, 0.5)], 0, 100.0, "decision", np.random.default_rng(1))
    ex = broker.reject(order, "feed_not_healthy", acct)
    assert ex.fills[0].status == "rejected" and ex.fills[0].reason == "feed_not_healthy"


def test_crossed_book_is_never_traded():
    b = book()
    b.apply_changes([("bid", 100.3, 5.0)])
    assert not b.valid
    assert estimate_round_trip(b, 50.0, COSTS, LONG, 30) is None
    broker = Broker(COSTS, 50.0)
    _, ex = run(broker, Account(10000.0), [Leg("open", LONG, 0.5)], b=b)
    assert ex.rejected == "book_invalid"
