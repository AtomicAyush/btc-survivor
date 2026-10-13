"""The pooled stock learner on hand built minutes: 8 predictions from one pass, trading only the home symbol, the open
window, the session close at 15:55, the forced timer from the first Active minute, the stop at 3 label standard
deviations, the required edge, the short sale restriction and the usable minute rules."""

import dataclasses
from datetime import date

import numpy as np
import pytest

from btc_survivor.bot import ACTIVE, DEAD, WARMUP
from btc_survivor.model import MLP
from btc_survivor.types import FLAT, LONG, SHORT
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import load_stock_config
from btc_survivor.v2.stocks import bot as bot_module
from btc_survivor.v2.stocks import costs
from btc_survivor.v2.stocks.bot import (SESSION_CLOSE, SESSION_FLATTEN, SESSION_WINDOW, SHORT_SALE_RESTRICTION,
                                        StockBot, StockMinute, opens_allowed, session_close_due)
from btc_survivor.v2.stocks.broker import StockBroker
from btc_survivor.v2.stocks.finnhub import Print
from btc_survivor.v2.stocks.learning import PooledWindow, with_news_inputs
from helpers import ROOT

CFG = load_stock_config(ROOT / "configs" / "v2" / "s1_prices.toml")
SYMBOLS = CFG.stock_feed.symbols
N = len(SYMBOLS)
DAY = sessions.session_on(date(2026, 10, 13))
NEXT = sessions.session_on(date(2026, 10, 14))
EARLY = sessions.session_on(date(2026, 11, 27))
PRICES = np.array([250.0, 500.0, 180.0, 220.0, 250.0, 700.0, 400.0, 600.0])
SD = 40.0


class Window(PooledWindow):
    """The pooled window holding 48 seeded samples with zero news inputs, recording each batch request."""

    def __init__(self):
        super().__init__(SYMBOLS, CFG.model.replay_window_minutes, CFG.model.inputs)
        self.calls = []
        rng = np.random.default_rng(3)
        for i in range(48):
            x = np.concatenate([rng.normal(size=10), np.zeros(2)])
            self.add(DAY.first_k + i // N, SYMBOLS[i % N], x, float(rng.normal(0, 30)))

    def batch(self, rng, newest, size, scales):
        self.calls.append((newest, size))
        return super().batch(rng, newest, size, scales)


def minute(session, m, *, ref=None, sd=SD, healthy=True, news_ok=True, valid=True, shorts_ok=True, rows=None,
           matured=(), window=None):
    boundary = session.boundary(m)
    full = lambda value, dtype=float: np.array(value, dtype=dtype) if np.ndim(value) else np.full(N, value, dtype)
    return StockMinute(boundary, session.k(boundary), session, np.zeros((N, 12)) if rows is None else rows,
                       full(valid, bool), full(healthy, bool), news_ok, PRICES.copy() if ref is None else full(ref),
                       full(sd), full(shorts_ok, bool), tuple(matured), window)


def constant(bot, output):
    """Make every prediction exactly output label standard deviations."""
    bot.model.W2[:] = 0.0
    bot.model.b2[0] = output


def active(slot=0, output=1.0, cfg=CFG):
    bot = StockBot.newborn(cfg, slot + 1, slot, DAY.open_ms)
    bot.state = ACTIVE
    bot.minutes_since_open = 0
    constant(bot, output)
    return bot


def broker(cfg=CFG):
    return StockBroker.from_config(cfg)


def settle(bot, b, price=None):
    """Fill the bot's pending order on one print at its due time."""
    order = bot.pending_order
    price = order.ref_price if price is None else price
    ex = b.execute(order, Print(order.symbol, price, 10.0, order.due_ms, ()), order.due_ms, bot.account)
    bot.on_execution(ex)
    return ex


def phase(slot, session=DAY, after=30):
    """The first session minute at or after after where the slot decides."""
    m = after
    while not (session.k(session.boundary(m)) % 30 == slot % 30):
        m += 1
    return m


def test_eight_predictions_from_one_pass(monkeypatch):
    bot = active(slot=3)
    rng = np.random.default_rng(1)
    rows = rng.normal(size=(N, 12))
    rows[:, 10:] = 0.0
    calls = []
    real = bot_module.predict_rows
    monkeypatch.setattr(bot_module, "predict_rows", lambda model, X: calls.append(X.shape) or real(model, X))
    bot.model = with_news_inputs(MLP.random(10, 16, rng, 0.1))
    sd = np.linspace(20.0, 90.0, N)
    bot.step(minute(DAY, 47, rows=rows, sd=sd), broker())
    assert calls == [(N, 12)]
    expected = [bot.model.predict(rows[i]) * sd[i] for i in range(N)]
    assert bot.preds == pytest.approx(expected, rel=1e-12)
    valid = np.ones(N, bool)
    valid[[2, 5]] = False
    sd[7] = np.nan
    bot.step(minute(DAY, 48, rows=rows, sd=sd, valid=valid), broker())
    assert calls == [(N, 12), (N, 12)]
    assert np.isnan(bot.preds[[2, 5, 7]]).all() and not np.isnan(bot.preds[[0, 1, 3, 4, 6]]).any()


def test_newborn_draws_ten_inputs_then_appends_zero_news_columns_then_the_timer_delay():
    bot = StockBot.newborn(CFG, 5, 4, DAY.open_ms)
    rng = np.random.default_rng(np.random.SeedSequence([CFG.run.master_seed, 1, 5]))
    narrow = MLP.random(10, 16, rng, 0.1)
    assert np.array_equal(bot.model.W1[:, :10], narrow.W1) and not bot.model.W1[:, 10:].any()
    assert np.array_equal(bot.model.W2, narrow.W2) and bot.model.size == 225
    assert bot.head_start == int(rng.integers(0, 180))
    assert bot.rng.random() == rng.random()
    assert bot.state == WARMUP and bot.warmup_minutes == 360 and bot.force_minutes == 180
    again = StockBot.newborn(CFG, 5, 4, DAY.open_ms)
    assert np.array_equal(again.model.params, bot.model.params) and again.head_start == bot.head_start


def test_home_symbol_is_slot_mod_8_and_decisions_come_at_its_phase_minutes():
    assert [StockBot.newborn(CFG, s + 1, s, 0).symbol for s in (0, 8, 16, 7, 15, 23, 9)] == \
        ["AAPL", "AAPL", "AAPL", "QQQ", "QQQ", "QQQ", "MSFT"]
    bot = active(slot=9, output=0.01)
    per_session = [m for m in range(1, 391) if bot.is_phase_minute(DAY.k(DAY.boundary(m)))]
    assert len(per_session) == 13 and per_session[0] == 9
    assert len([m for m in per_session if opens_allowed(DAY, m, CFG.session)]) == 11
    b = broker()
    for m in range(31, 70):
        res = bot.step(minute(DAY, m), b)
        assert (res.decision is not None) == (m in (39, 69))


def test_trades_only_the_home_symbol():
    bot = active(slot=9, output=2.0)                     # home MSFT, every prediction 80 bps
    b = broker()
    m = phase(9)
    ref = PRICES.copy()
    res = bot.step(minute(DAY, m, ref=ref), b)
    row = res.decision
    assert row["symbol"] == "MSFT" and row["ref_mid"] == 500.0 and row["action"] == "open_long"
    assert [o.symbol for o in res.orders] == ["MSFT"] and res.orders[0].ref_price == 500.0
    assert res.orders[0].legs[0].qty == 1.0
    assert not np.isnan(bot.preds).any()                  # it still predicts every symbol
    settle(bot, b)
    assert bot.account.side == LONG


def test_home_symbol_halted_stops_rent_marks_and_decisions_but_not_learning():
    bot = active(slot=9)
    window = Window()
    healthy = np.ones(N, bool)
    healthy[1] = False                                     # MSFT halted
    m = phase(9)
    res = bot.step(minute(DAY, m, healthy=healthy, valid=healthy, matured=[(0, 100, 5.0)], window=window),
                   broker())
    assert res.decision is None and bot.account.rent == 0.0 and bot.alive_minutes == 0
    assert bot.trained_minutes == 1 and window.calls == [(1, 16)]
    assert np.isnan(bot.preds[1]) and not np.isnan(np.delete(bot.preds, 1)).any()


def test_a_news_outage_pauses_rent_training_predictions_and_decisions():
    bot = active(slot=9)
    window = Window()
    m = phase(9)
    res = bot.step(minute(DAY, m, news_ok=False, matured=[(0, 100, 5.0)], window=window), broker())
    assert res.decision is None and bot.account.rent == 0.0 and window.calls == [] and bot.trained_minutes == 0
    assert np.isnan(bot.preds).all()
    bot.step(minute(DAY, m + 1), broker())
    assert bot.account.rent == pytest.approx(costs.rent_per_minute(CFG.stock_costs))


def test_with_no_margin_the_required_edge_is_the_round_trip_and_gates_the_decision():
    # margin_sigma is 0, so a learner opens on its own whenever its prediction clears the round trip cost.
    assert CFG.trading.margin_sigma == 0.0
    b = broker()
    m = phase(0)
    trip = costs.round_trip(CFG.stock_costs, "AAPL", LONG, 2.0, 250.0, 30).total_bps
    assert trip == pytest.approx(4.6, abs=1e-6)
    below = active(output=(trip - 0.01) / SD)
    row = below.step(minute(DAY, m), b).decision
    assert row["required_edge_bps"] == pytest.approx(trip) and row["margin_bps"] == 0.0
    assert row["cost_estimate_bps"] == pytest.approx(trip) and row["gate_passed"] == 0 and row["action"] == "none"
    above = active(output=(trip + 0.01) / SD)
    row = above.step(minute(DAY, m), b).decision
    assert row["gate_passed"] == 1 and row["forced"] == 0 and row["action"] == "open_long"
    short = active(output=-(trip + 1.0) / SD)
    row = short.step(minute(DAY, m), b).decision
    short_trip = costs.round_trip(CFG.stock_costs, "AAPL", SHORT, 2.0, 250.0, 30).total_bps
    assert row["action"] == "open_short" and row["cost_estimate_bps"] == pytest.approx(short_trip)
    assert row["required_edge_bps"] == pytest.approx(short_trip)
    assert short.pending_order.legs[0].qty == 2.0


def test_a_margin_adds_margin_sigma_label_standard_deviations_to_the_required_edge():
    cfg = dataclasses.replace(CFG, trading=dataclasses.replace(CFG.trading, margin_sigma=0.25))
    m = phase(0)
    trip = costs.round_trip(cfg.stock_costs, "AAPL", LONG, 2.0, 250.0, 30).total_bps
    required = trip + 0.25 * SD
    row = active(output=(required - 0.01) / SD, cfg=cfg).step(minute(DAY, m), broker(cfg)).decision
    assert row["required_edge_bps"] == pytest.approx(required) and row["margin_bps"] == pytest.approx(10.0)
    assert row["gate_passed"] == 0 and row["action"] == "none"
    row = active(output=(required + 0.01) / SD, cfg=cfg).step(minute(DAY, m), broker(cfg)).decision
    assert row["gate_passed"] == 1 and row["action"] == "open_long"


def test_the_open_window_runs_from_10_00_to_15_25_or_12_25_on_an_early_close():
    rules = CFG.session
    assert [opens_allowed(DAY, m, rules) for m in (29, 30, 355, 356)] == [False, True, True, False]
    assert DAY.boundary(30) == DAY.open_ms + 30 * 60_000 and DAY.boundary(355) == DAY.close_ms - 35 * 60_000
    assert [opens_allowed(EARLY, m, rules) for m in (29, 30, 175, 176)] == [False, True, True, False]
    assert EARLY.minutes == 210


def test_outside_the_window_an_open_becomes_none_and_a_flip_a_close():
    b = broker()
    early = active(slot=25, output=2.0)                   # decides at session minute 25, 09:55 ET
    row = early.step(minute(DAY, 25), b).decision
    assert (row["action"], row["reason"], row["gate_passed"]) == ("none", SESSION_WINDOW, 1)
    assert early.pending_order is None
    late = active(slot=25, output=-2.0)                   # holds a long at 15:55, the prediction says short
    late.step(minute(DAY, 355), b)
    settle(late, b)
    assert late.account.side == SHORT
    constant(late, 2.0)
    row = late.step(minute(DAY, 385), b).decision
    assert (row["action"], row["reason"]) == ("close", SESSION_WINDOW)
    assert [leg.action for leg in late.pending_order.legs] == ["close"]
    edge = active(slot=25, output=2.0)
    assert edge.step(minute(EARLY, 175), b).decision["action"] == "open_long"
    edge_late = active(slot=25, output=2.0)
    assert edge_late.step(minute(EARLY, 205), b).decision["reason"] == SESSION_WINDOW


def test_forced_exploration_waits_for_the_next_allowed_phase_minute():
    b = broker()
    bot = active(slot=25, output=0.01)                    # far below the gate
    bot.minutes_since_open = 500
    row = bot.step(minute(DAY, 385), b).decision
    assert (row["action"], row["forced"], row["reason"]) == ("none", 0, SESSION_WINDOW)
    row = bot.step(minute(NEXT, 25), b).decision
    assert row["action"] == "none" and bot.minutes_since_open == 502
    row = bot.step(minute(NEXT, 55), b).decision
    assert (row["action"], row["forced"], row["gate_passed"]) == ("open_long", 1, 0)
    settle(bot, b)
    assert bot.minutes_since_open == 0


def test_the_session_close_goes_out_at_15_55_or_12_55():
    rules = CFG.session
    assert [session_close_due(DAY, m, rules) for m in (384, 385, 390)] == [False, True, True]
    assert DAY.boundary(385) == DAY.close_ms - 5 * 60_000
    assert [session_close_due(EARLY, m, rules) for m in (204, 205)] == [False, True]
    b = broker()
    bot = active(output=2.0)
    bot.step(minute(DAY, phase(0, after=330)), b)
    settle(bot, b)
    assert bot.account.is_open
    order = bot.send_close(DAY.boundary(385), 251.0, b, SESSION_CLOSE)
    assert order.reason == SESSION_CLOSE and bot.send_close(DAY.boundary(385), 251.0, b, SESSION_CLOSE) is None
    ex = settle(bot, b, 251.0)
    assert ex.closed.exit_reason == "session_close" and not bot.account.is_open and bot.stop_bps is None
    assert bot.send_close(DAY.boundary(386), 251.0, b, SESSION_CLOSE) is None
    carried = active(output=2.0)
    carried.step(minute(DAY, phase(0, after=330)), b)
    settle(carried, b)
    carried.send_close(NEXT.boundary(1), 260.0, b, SESSION_FLATTEN)
    assert settle(carried, b, 260.0).closed.exit_reason == "session_flatten"


def test_forced_timer_counts_usable_minutes_from_the_first_active_minute():
    cfg = dataclasses.replace(CFG, time=dataclasses.replace(CFG.time, warmup_hours=3 / 60))
    bot = StockBot.newborn(cfg, 7, 6, DAY.open_ms)
    constant(bot, 0.01)
    head = bot.head_start
    assert 0 <= head < 180
    b = broker(cfg)
    window = Window()
    for m in range(31, 61):                               # 30 usable minutes in warmup without a matured sample
        bot.step(minute(DAY, m), b)
    assert bot.state == WARMUP and bot.minutes_since_open == 0
    for m in range(61, 64):
        bot.step(minute(DAY, m, matured=[(0, 1, 1.0)], window=window), b)
    assert bot.state == ACTIVE and bot.trained_minutes == 3 and bot.minutes_since_open == head
    first_active = 63
    m = 64
    while bot.minutes_since_open < 180:
        news_ok = m % 7 != 0                              # minutes without news health do not count
        before = bot.minutes_since_open
        bot.step(minute(DAY, m, news_ok=news_ok), b)
        assert bot.minutes_since_open == before + (1 if news_ok else 0)
        m += 1
    usable = sum(1 for x in range(first_active + 1, m) if x % 7 != 0)
    assert usable == 180 - head
    while bot.pending_order is None:
        row = bot.step(minute(DAY, m), b).decision
        m += 1
    assert row["forced"] == 1 and row["action"] == "open_long"


def test_first_forced_trades_spread_over_the_three_hours_after_warmup():
    heads = [StockBot.newborn(CFG, i + 1, i, DAY.open_ms).head_start for i in range(24)]
    assert all(0 <= h < 180 for h in heads) and len(set(heads)) > 15
    # A timer started h minutes early reaches 180 minutes after 180 - h usable minutes, within 3 hours of warmup.
    assert all(0 < 180 - h <= 180 for h in heads)


def test_the_stop_fires_at_3_label_standard_deviations_fixed_at_entry():
    b = broker()
    bot = active(output=2.0)
    m = phase(0)
    bot.step(minute(DAY, m, sd=40.0), b)
    settle(bot, b, 250.0)
    entry = bot.account.entry_price
    assert bot.stop_bps == pytest.approx(120.0)
    allowance = costs.allowance_bps(CFG.stock_costs, "AAPL")
    ref_at = lambda adverse_bps: entry * (1 - adverse_bps / 1e4) / (1 - allowance / 1e4)
    ref = PRICES.copy()
    ref[0] = ref_at(119.0)
    res = bot.step(minute(DAY, m + 1, ref=ref, sd=200.0), b)   # a wider label scale now does not move the stop
    assert not res.stopped and bot.pending_order is None
    ref[0] = ref_at(121.0)
    res = bot.step(minute(DAY, m + 2, ref=ref, sd=10.0), b)    # nor does a narrower one
    assert res.stopped and res.orders[0].reason == "stop" and res.orders[0].legs[0].action == "close"
    assert settle(bot, b, ref[0]).closed.exit_reason == "stop"


def test_a_short_stop_and_the_short_sale_restriction():
    b = broker()
    blocked = active(output=-2.0)
    m = phase(0)
    shorts_ok = np.ones(N, bool)
    shorts_ok[0] = False
    row = blocked.step(minute(DAY, m, shorts_ok=shorts_ok), b).decision
    assert (row["action"], row["reason"]) == ("none", SHORT_SALE_RESTRICTION) and blocked.pending_order is None
    other = active(slot=1, output=-2.0)                   # MSFT is not restricted
    assert other.step(minute(DAY, phase(1), shorts_ok=shorts_ok), b).decision["action"] == "open_short"
    flipper = active(output=2.0)
    flipper.step(minute(DAY, m), b)
    settle(flipper, b)
    constant(flipper, -2.0)
    row = flipper.step(minute(DAY, m + 30, shorts_ok=shorts_ok), b).decision
    assert (row["action"], row["reason"]) == ("close", SHORT_SALE_RESTRICTION)
    short = active(output=-2.0)
    short.step(minute(DAY, m, sd=50.0), b)
    settle(short, b, 250.0)
    assert short.account.side == SHORT and short.stop_bps == pytest.approx(150.0)
    entry = short.account.entry_price
    ref = PRICES.copy()
    ref[0] = entry * (1 + 151.0 / 1e4) / (1 + 2.0 / 1e4)
    assert short.step(minute(DAY, m + 1, ref=ref), b).stopped


def test_death_below_a_round_trip_plus_a_session_of_rent():
    b = broker()
    bot = active(output=0.0)
    line = costs.death_line(CFG.stock_costs, "AAPL", LONG, 2.0, 250.0, 30)
    bot.account.charge_rent(10_000.0 - line - 0.01)      # the step's own rent leaves equity just above the line
    assert not bot.step(minute(DAY, 40), b).died
    bot.account.charge_rent(0.01)
    res = bot.step(minute(DAY, 41), b)
    assert res.died and bot.state == DEAD and bot.death_threshold == pytest.approx(line)
    assert bot.step(minute(DAY, 42), b).decision is None and np.isnan(bot.preds).all()


def test_learning_draws_the_newly_matured_samples_plus_the_window_and_scores_home_predictions():
    bot = active(slot=0)
    window = Window()
    m = phase(0)
    bot.step(minute(DAY, m), broker())
    pred = bot.preds[0]
    assert bot.pending_preds == {DAY.k(DAY.boundary(m)): pred}
    k = DAY.k(DAY.boundary(m))
    matured = [(0, k, 12.0), (3, k, -4.0), (7, k, 1.0)]
    before = bot.model.params.copy()
    bot.step(minute(DAY, m + 30, matured=matured, window=window), broker())
    assert window.calls == [(3, 16)] and bot.trained_minutes == 1
    assert not np.array_equal(before, bot.model.params)
    assert bot.skill.n == 1 and bot.skill.pred[0] == pred and bot.skill.label[0] == 12.0
    assert k not in bot.pending_preds


def test_no_order_while_one_is_pending_and_a_flat_position_after_a_rejected_open():
    b = broker()
    bot = active(output=2.0)
    m = phase(0)
    bot.step(minute(DAY, m), b)
    row = bot.step(minute(DAY, m + 30), b).decision
    assert row["reason"] == "an order is still pending"
    bot.on_execution(b.reject(bot.pending_order, "symbol_not_healthy", bot.account))
    assert bot.account.side == FLAT and bot.stop_bps is None and bot.pending_order is None


def test_a_batch_holding_a_label_with_no_scale_yet_trains_nothing():
    bot = active(slot=0)
    sd = np.full(N, SD)
    sd[4] = np.nan                                         # GOOGL samples sit in the window without a scale
    before = bot.model.params.copy()
    bot.step(minute(DAY, 40, sd=sd, matured=[(0, 1, 1.0)], window=Window()), broker())
    assert bot.trained_minutes == 0 and np.array_equal(before, bot.model.params)
    bot.step(minute(DAY, 41, matured=[(0, 1, 1.0)], window=Window()), broker())
    assert bot.trained_minutes == 1 and not np.array_equal(before, bot.model.params)
