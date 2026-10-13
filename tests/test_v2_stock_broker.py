"""Print fills, latency and its per order generator, rejections, flips, quantities, the short sale restriction and the
split guard, on hand written prints. A print fills an order only when it traded at or after the due time, so a late
report of a trade made inside the latency never fills, however the price moved."""

import dataclasses
from datetime import date

import numpy as np
import pytest

from btc_survivor.account import Account
from btc_survivor.types import FLAT, LONG, SHORT, Leg
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import load_stock_config
from btc_survivor.v2.stocks import broker as broker_module
from btc_survivor.v2.stocks import costs
from btc_survivor.v2.stocks.broker import (DISCONNECTED, NO_POSITION, SESSION_CLOSED, SPLIT_GUARD, SYMBOL_STALE,
                                           StockBroker, SymbolGuards, first_fill, fill_prints, floor_to, jitter_rng,
                                           rejection)
from btc_survivor.v2.stocks.finnhub import Print, PrintBatch
from btc_survivor.v2.stocks.prints import PriceRules
from helpers import ROOT

CFG = load_stock_config(ROOT / "configs" / "v2" / "s1_prices.toml")
RULES = PriceRules(late_ms=4_000, excluded=frozenset({"Z"}))
DAY = sessions.session_on(date(2026, 10, 13))
NEXT = sessions.session_on(date(2026, 10, 14))
AFTER = sessions.session_on(date(2026, 10, 15))
B = DAY.boundary(60)                                    # 10:30 ET


def broker():
    return StockBroker(CFG.stock_costs, CFG.trading)


def order(b, legs, symbol="AAPL", decided=B, ref=250.0, reason="decision", seed=0):
    return b.make_order(3, symbol, legs, decided, ref, reason, np.random.default_rng(seed))


def batch(recv, *prints):
    return PrintBatch(recv, 1, tuple(prints))


def p(price, vendor, symbol="AAPL", conditions=()):
    return Print(symbol, price, 10.0, vendor, conditions)


def test_latency_is_one_second_plus_seeded_jitter_up_to_300_ms():
    b = broker()
    rng = np.random.default_rng(5)
    dues = [b.make_order(1, "AAPL", [Leg("close", LONG)], B, 250.0, "stop", rng).latency_ms for _ in range(2000)]
    assert min(dues) == 1000 and max(dues) == 1300
    again = np.random.default_rng(5)
    assert dues[:20] == [b.make_order(1, "AAPL", [Leg("close", LONG)], B, 250.0, "stop", again).latency_ms
                         for _ in range(20)]


def test_each_order_draws_its_jitter_from_a_generator_of_its_bot_and_decision_time():
    b = broker()
    seed = CFG.run.master_seed

    def latency(bot_id, decided):
        return b.make_order(bot_id, "AAPL", [Leg("close", LONG)], decided, 250.0, "stop",
                            jitter_rng(seed, bot_id, decided)).latency_ms

    times = [B + 60_000 * i for i in range(2000)]
    latencies = [latency(7, t) for t in times]
    assert min(latencies) == 1000 and max(latencies) == 1300
    assert latencies[:50] == [latency(7, t) for t in times[:50]]          # the same order always gets one latency
    assert latencies[:50] != [latency(8, t) for t in times[:50]]          # another bot draws its own
    first = jitter_rng(seed, 7, B).integers(0, 301)
    assert first == np.random.default_rng(np.random.SeedSequence([seed, 4, 7, B])).integers(0, 301)


def test_fills_at_the_first_eligible_print_traded_and_received_at_or_after_due():
    b = broker()
    o = order(b, [Leg("open", LONG, 2.0)])
    due = o.due_ms
    early = batch(due - 1, p(250.5, due - 50))
    assert first_fill(o, early.recv_ms, fill_prints(early, DAY, RULES, None)) is None
    at_due = batch(due, p(251.0, due, "MSFT"), p(250.2, due), p(250.9, due))
    chosen = first_fill(o, at_due.recv_ms, fill_prints(at_due, DAY, RULES, None))
    assert chosen.price == 250.2                       # exactly at due fills, on the first print of the symbol
    acct = Account(10_000.0)
    ex = b.execute(o, chosen, at_due.recv_ms, acct)
    fill = ex.fills[0]
    assert fill.status == "filled" and fill.filled_ms == due and fill.print_after_due_ms == 0
    assert fill.print_price == 250.2
    assert fill.price == pytest.approx(250.2 * (1 + 2.0 / 1e4))
    assert fill.allowance_bps == 2.0 and fill.levels is None and fill.book_avg_price is None
    assert fill.slippage_bps == pytest.approx((fill.price - 250.0) / 250.0 * 1e4)
    assert acct.side == LONG and acct.entry_price == fill.price and ex.opened_trade_id == 1


def test_refuses_a_print_with_vendor_time_before_due():
    b = broker()
    o = order(b, [Leg("open", LONG, 2.0)])
    due = o.due_ms
    late_arrival = batch(due + 5, p(249.0, B - 1), p(249.2, B), p(249.4, due - 1), p(249.5, due))
    chosen = first_fill(o, late_arrival.recv_ms, fill_prints(late_arrival, DAY, RULES, None))
    assert chosen.vendor_ms == due and chosen.price == 249.5
    for vendor in (B - 1, B, due - 1):
        only_old = batch(due + 5, p(249.0, vendor))
        assert first_fill(o, only_old.recv_ms, fill_prints(only_old, DAY, RULES, None)) is None
        with pytest.raises(ValueError):
            b.execute(o, p(249.0, vendor), due + 5, Account(10_000.0))


def test_a_stale_print_from_inside_the_latency_never_fills_across_a_jump():
    # The order goes out at 10:30. AAPL trades at 250.00 400 ms later and jumps to 251.00 at due, and the feed brings
    # the old print 20 ms after due in one batch with the new ones.
    b = broker()
    o = order(b, [Leg("open", LONG, 2.0)])
    due = o.due_ms
    rep = batch(due + 20, p(250.0, B + 400), p(251.0, due), p(251.5, due + 10))
    chosen = first_fill(o, rep.recv_ms, fill_prints(rep, DAY, RULES, None))
    assert chosen.price == 251.0                       # exactly at due included, the stale print skipped
    stale = batch(due + 20, p(250.0, B + 400))
    assert fill_prints(stale, DAY, RULES, None) != ()   # eligible for candles and health, yet it fills no order
    assert first_fill(o, stale.recv_ms, fill_prints(stale, DAY, RULES, None)) is None
    with pytest.raises(ValueError, match="cannot fill"):
        b.execute(o, p(250.0, B + 400), due + 20, Account(10_000.0))
    later = batch(due + 900, p(251.2, due + 700))
    fill = b.execute(o, first_fill(o, later.recv_ms, fill_prints(later, DAY, RULES, None)), later.recv_ms,
                     Account(10_000.0)).fills[0]
    assert fill.print_price == 251.2 and fill.print_after_due_ms == 700 and fill.filled_ms == due + 900
    rejected = b.reject(o, SYMBOL_STALE, Account(10_000.0)).fills[0]
    assert rejected.print_after_due_ms is None


def test_only_eligible_regular_session_prints_fill():
    b = broker()
    o = order(b, [Leg("open", LONG, 2.0)])
    t = o.due_ms + 10
    stale = batch(t, p(250.0, t - 4_001))                      # late by more than late_print_seconds
    excluded = batch(t, p(250.0, t - 10, conditions=("Z",)))   # an excluded condition code
    early = batch(t, p(250.0, t + 2_001))                      # more than 2 seconds early
    for bad in (stale, excluded, early):
        assert fill_prints(bad, DAY, RULES, None) == ()
    replayed = batch(t, p(250.0, t - 1_500))
    assert fill_prints(replayed, DAY, RULES, None) != ()
    assert fill_prints(replayed, DAY, RULES, connected_ms=t) == ()   # before a reconnect, so a replay
    close = DAY.close_ms
    auction = batch(close + 50, p(250.0, close))               # the closing cross and post market prints
    assert fill_prints(auction, sessions.session_at(close + 50), RULES, None) == ()
    assert fill_prints(auction, DAY, RULES, None) == ()


def test_rejects_on_stale_symbol_disconnect_and_session_end():
    assert rejection(True, True, True) is None
    assert rejection(False, True, True) == SYMBOL_STALE
    assert rejection(False, False, True) == DISCONNECTED
    assert rejection(True, True, False) == SESSION_CLOSED
    assert rejection(False, False, False) == SESSION_CLOSED
    b = broker()
    acct = Account(10_000.0)
    o = order(b, [Leg("open", SHORT, 2.0)])
    for reason in (SYMBOL_STALE, DISCONNECTED, SESSION_CLOSED):
        ex = b.reject(o, reason, acct)
        fill = ex.fills[0]
        assert ex.rejected == reason and fill.status == "rejected" and fill.reason == reason
        assert fill.price is None and fill.filled_ms is None and fill.fee is None and fill.symbol == "AAPL"
    assert not acct.is_open and acct.fees == 0.0 and acct.cash == 10_000.0


def test_a_flip_fills_both_legs_at_one_print():
    b = broker()
    acct = Account(10_000.0)
    b.execute(order(b, [Leg("open", LONG, 2.0)]), p(250.0, B + 1_500), B + 2_000, acct)
    flip = order(b, [Leg("close", LONG), Leg("open", SHORT, b.open_qty(SHORT, 250.0))], decided=B + 60_000)
    ex = b.execute(flip, p(260.0, flip.due_ms), flip.due_ms, acct)
    close, open_ = ex.fills
    assert close.print_price == open_.print_price == 260.0
    assert close.price == pytest.approx(260.0 * (1 - 2e-4)) and open_.price == close.price
    assert ex.closed.exit_reason == "flip" and ex.closed.gross == pytest.approx(2.0 * (close.price - 250.05))
    assert acct.side == SHORT and acct.qty == 2.0
    assert (close.sec_fee, close.taf_fee, open_.sec_fee, open_.taf_fee) == (0.02, 0.01, 0.02, 0.01)
    assert acct.invariant_errors(costs.mark_price(260.0, SHORT, 2.0)) == []


def test_bots_on_one_symbol_share_a_print():
    b = broker()
    accounts = [Account(10_000.0) for _ in range(3)]
    orders = [order(b, [Leg("open", LONG, 2.0)], seed=s) for s in range(3)]
    last_due = max(o.due_ms for o in orders)
    prints = fill_prints(batch(last_due, p(251.0, last_due)), DAY, RULES, None)
    fills = [b.execute(o, first_fill(o, last_due, prints), last_due, a).fills[0] for o, a in zip(orders, accounts)]
    assert {f.print_price for f in fills} == {251.0}
    assert [f.trade_id for f in fills] == [1, 2, 3]


def test_a_leg_rejects_without_a_matching_position():
    b = broker()
    acct = Account(10_000.0)
    ex = b.execute(order(b, [Leg("close", SHORT)]), p(250.0, B + 1_500), B + 2_000, acct)
    assert ex.rejected == NO_POSITION and ex.closed is None and acct.side == FLAT


def test_longs_round_down_to_6_decimals_and_shorts_to_whole_shares_with_their_notional():
    b = broker()
    assert b.open_qty(LONG, 250.0) == 2.0
    assert b.open_qty(LONG, 3.0) == 166.666666                 # 166.6666667 rounded down
    assert b.open_qty(LONG, 220.0) == 2.272727
    assert b.open_qty(LONG, 0.7) == 714.285714
    assert floor_to(1.9999999999999998, 6) == 2.0               # float noise is not a lost share
    assert b.open_qty(SHORT, 400.0) == 1.0                      # 1.25 shares to the nearest whole share
    assert b.open_qty(SHORT, 180.0) == 3.0                      # 2.78
    assert b.open_qty(SHORT, 200.0) == 3.0                      # 2.5, halves round up
    assert b.open_qty(SHORT, 2_000.0) == 1.0                    # 0.25, the one share minimum
    acct = Account(10_000.0)
    ex = b.execute(order(b, [Leg("open", SHORT, 1.0)], symbol="TSLA", ref=400.0), p(400.0, B + 1_500, "TSLA"),
                   B + 2_000, acct)
    fill = ex.fills[0]
    assert fill.qty == 1.0 and fill.notional == pytest.approx(400.0 * (1 - 3e-4))
    fractional = StockBroker(CFG.stock_costs, dataclasses.replace(CFG.trading, short_whole_shares=False))
    assert fractional.open_qty(SHORT, 400.0) == 1.25


def guards():
    return SymbolGuards(("AAPL", "MSFT"), drop_pct=10.0, max_gap_pct=25.0, history_minutes=391)


def test_short_sale_restriction_for_the_rest_of_the_session_and_the_next():
    g = guards()
    g.on_price("AAPL", DAY, DAY.first_k + 100, 200.0)
    g.on_price("AAPL", DAY, DAY.last_k, 200.0)                  # the previous session's last healthy price
    g.on_price("AAPL", NEXT, NEXT.first_k + 5, 181.0)           # down 9.5 percent
    assert g.shorts_allowed("AAPL", NEXT.day)
    g.on_price("AAPL", NEXT, NEXT.first_k + 6, 180.0)           # down 10 percent
    assert not g.shorts_allowed("AAPL", NEXT.day)
    g.on_price("AAPL", NEXT, NEXT.first_k + 7, 199.0)           # a recovery keeps the restriction
    assert not g.shorts_allowed("AAPL", NEXT.day)
    g.on_price("AAPL", AFTER, AFTER.first_k, 199.0)
    assert not g.shorts_allowed("AAPL", AFTER.day)              # the next session too
    after_that = sessions.next_session(AFTER.close_ms)
    assert g.shorts_allowed("AAPL", after_that.day)
    assert g.shorts_allowed("MSFT", NEXT.day)                   # per symbol


def test_short_sale_restriction_never_triggers_without_a_healthy_previous_price():
    g = guards()
    g.on_price("AAPL", NEXT, NEXT.first_k, 100.0)               # no earlier session at all
    g.on_price("AAPL", NEXT, NEXT.first_k + 1, 50.0)
    assert g.shorts_allowed("AAPL", NEXT.day)
    g = guards()
    g.on_price("AAPL", DAY, DAY.last_k, 200.0)
    g.on_price("AAPL", AFTER, AFTER.first_k + 3, 150.0)         # NEXT had no healthy price
    assert g.state["AAPL"].prev_close is None
    assert g.shorts_allowed("AAPL", AFTER.day)
    assert g.features_allowed("AAPL", AFTER.first_k + 3)


def test_split_guard_holds_features_for_391_fresh_minutes_and_writes_an_event():
    g = guards()
    assert g.on_price("AAPL", DAY, DAY.last_k, 400.0) == []
    events = g.on_price("AAPL", NEXT, NEXT.first_k, 200.0)      # a two for one split
    assert [kind for kind, _ in events] == [SPLIT_GUARD]
    assert "-50.0 percent" in events[0][1]
    start = NEXT.first_k
    assert not g.features_allowed("AAPL", start)
    assert not g.features_allowed("AAPL", start + 389)
    assert g.features_allowed("AAPL", start + 390)              # 391 fresh session minutes, start to start + 390
    assert g.features_allowed("MSFT", start)
    assert g.on_price("AAPL", NEXT, NEXT.first_k + 1, 201.0) == []   # only the first price of a session is judged
    quiet = guards()
    quiet.on_price("AAPL", DAY, DAY.last_k, 400.0)
    assert quiet.on_price("AAPL", NEXT, NEXT.first_k, 490.0) == []   # a 22.5 percent move is not a split
    assert quiet.features_allowed("AAPL", NEXT.first_k)


def test_guards_come_from_the_config():
    g = SymbolGuards.from_config(CFG)
    assert set(g.state) == set(CFG.stock_feed.symbols)
    assert (g.drop, g.max_gap, g.history_minutes) == (0.10, 0.25, 391)


def test_the_broker_module_reads_no_book():
    source = broker_module.__file__
    with open(source, encoding="utf-8") as fh:
        text = fh.read()
    assert "LocalBook" not in text and "feed.book" not in text
