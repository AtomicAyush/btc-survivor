"""StockTwin on hand built minutes: upkeep on its learner's home symbol only, marking at the reference price, the
version 1 coin flip rules, the session_close and session_flatten exits, and no LocalBook built or read."""

import inspect
from datetime import date

import numpy as np
import pytest

import btc_survivor.bot as v1_bot
import btc_survivor.broker as v1_broker
import btc_survivor.feed.book as book_module
from btc_survivor.bot import ACTIVE, DEAD
from btc_survivor.controls import TwinBot
from btc_survivor.types import FLAT, LONG, SHORT
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import load_stock_config
from btc_survivor.v2.stocks import bot as bot_module
from btc_survivor.v2.stocks import costs
from btc_survivor.v2.stocks import twins as twins_module
from btc_survivor.v2.stocks.bot import SESSION_CLOSE, SESSION_FLATTEN, BotStep, StockBot, StockMinute
from btc_survivor.v2.stocks.broker import StockBroker
from btc_survivor.v2.stocks.finnhub import Print
from btc_survivor.v2.stocks.twins import StockTwin
from helpers import ROOT

CFG = load_stock_config(ROOT / "configs" / "v2" / "s1_prices.toml")
N = len(CFG.stock_feed.symbols)
DAY = sessions.session_on(date(2026, 10, 13))
NEXT = sessions.session_on(date(2026, 10, 14))
PRICES = np.array([250.0, 500.0, 180.0, 220.0, 250.0, 700.0, 400.0, 600.0])


def minute(session, m, *, ref=None, sd=40.0, healthy=True, news_ok=True, shorts_ok=True, matured=(), window=None):
    boundary = session.boundary(m)
    full = lambda value, dtype=float: np.array(value, dtype=dtype) if np.ndim(value) else np.full(N, value, dtype)
    return StockMinute(boundary, session.k(boundary), session, np.zeros((N, 12)), np.ones(N, bool),
                       full(healthy, bool), news_ok, PRICES.copy() if ref is None else full(ref), full(sd),
                       full(shorts_ok, bool), tuple(matured), window)


def twin(slot=2, seed=None, cfg=CFG):
    if seed is None:
        return StockTwin.newborn(cfg, 100 + slot, slot, DAY.open_ms)
    return StockTwin(cfg, 100 + slot, slot, DAY.open_ms, np.random.default_rng(seed))


def learner_row(action, m=60, session=DAY, symbol="NVDA"):
    return {"bot_id": 3, "ts": session.boundary(m), "symbol": symbol, "position_before": "flat", "ref_mid": 180.0,
            "prediction_bps": 3.0, "required_edge_bps": 20.0, "cost_estimate_bps": 4.6, "margin_bps": 15.0,
            "gate_passed": 0, "forced": 1, "action": action, "reason": "learner reason"}


def settle(bot, b, price=None):
    order = bot.pending_order
    price = order.ref_price if price is None else price
    ex = b.execute(order, Print(order.symbol, price, 5.0, order.due_ms, ()), order.due_ms, bot.account)
    bot.on_execution(ex)
    return ex


def opened(t, b, action="open_long", m=60, ref=None):
    ctx = minute(DAY, m, ref=ref)
    row = t.mirror(learner_row(action, m), 3, ctx, b)
    settle(t, b)
    return row


def test_a_twin_is_active_at_birth_and_never_learns_predicts_or_decides():
    t = twin()
    assert t.state == ACTIVE and t.symbol == "NVDA" and t.home == 2

    class Window:
        def batch(self, rng, newest, size, scales):
            raise AssertionError("a twin never draws a training batch")

    res = t.step(minute(DAY, 2, matured=[(2, 1, 5.0)], window=Window()), StockBroker.from_config(CFG))
    assert res.decision is None and res.orders == [] and t.trained_minutes == 0
    assert np.isnan(t.preds).all()
    assert t.account.rent == pytest.approx(costs.rent_per_minute(CFG.stock_costs))


def test_upkeep_runs_on_the_home_symbol_only_and_marks_at_the_reference_price():
    b = StockBroker.from_config(CFG)
    t = twin(slot=2, seed=1)
    opened(t, b)
    side, entry, qty = t.account.side, t.account.entry_price, t.account.qty
    allowance = costs.allowance_bps(CFG.stock_costs, "NVDA")
    ref = PRICES.copy()
    ref[2] = 185.0
    ref[[0, 1, 3, 4, 5, 6, 7]] = np.nan                    # only the home symbol has a price this minute
    healthy = np.zeros(N, bool)
    healthy[2] = True
    rent = t.account.rent
    t.step(minute(DAY, 61, ref=ref, healthy=healthy), b)
    mark = costs.mark_price(185.0, side, allowance)
    assert t.equity == pytest.approx(t.account.cash + side * qty * (mark - entry))
    assert t.mark(185.0) == mark
    assert t.account.rent == pytest.approx(rent + costs.rent_per_minute(CFG.stock_costs))
    others = np.ones(N, bool)
    others[2] = False                                       # every other symbol healthy, the home symbol halted
    equity = t.equity
    t.step(minute(DAY, 62, healthy=others, ref=np.where(others, 999.0, np.nan)), b)
    assert t.equity == equity and t.account.rent == pytest.approx(rent + costs.rent_per_minute(CFG.stock_costs))
    assert t.account.invariant_errors(mark) == []


def test_version_1_coin_flip_rules_on_a_seeded_coin():
    b = StockBroker.from_config(CFG)
    for seed in range(40):
        coin = np.random.default_rng(seed)
        expected = LONG if coin.random() < 0.5 else SHORT
        t = twin(seed=seed)
        row = opened(t, b)
        assert t.account.side == expected
        assert row["action"] == ("open_long" if expected == LONG else "open_short")
        assert row["bot_id"] == t.bot_id and row["forced"] == 0 and "copying open_long" in row["reason"]
    sides = []
    for seed in range(400):
        t = twin(seed=seed)
        opened(t, b, "open_short")
        sides.append(t.account.side)
    assert 160 < sides.count(LONG) < 240                    # a fair coin, whatever the learner chose


def test_closes_flips_and_a_stale_position_closed_before_the_next_open():
    b = StockBroker.from_config(CFG)
    t = twin(seed=3)
    opened(t, b)
    row = t.mirror(learner_row("close", 90), 3, minute(DAY, 90), b)
    assert row["action"] == "close" and [leg.action for leg in t.pending_order.legs] == ["close"]
    settle(t, b)
    assert t.account.side == FLAT
    row = t.mirror(learner_row("close", 120), 3, minute(DAY, 120), b)
    assert row["action"] == "none" and "nothing to copy" in row["reason"] and t.pending_order is None
    opened(t, b, m=150)
    row = t.mirror(learner_row("open_long", 180), 3, minute(DAY, 180), b)    # a stale position left over
    assert row["action"] == "flip" and [leg.action for leg in t.pending_order.legs] == ["close", "open"]
    settle(t, b)
    assert t.mirror(learner_row("hold", 210), 3, minute(DAY, 210), b) is None
    assert t.mirror(learner_row("none", 210), 3, minute(DAY, 210), b) is None
    t.mirror(learner_row("flip", 240), 3, minute(DAY, 240), b)
    row = t.mirror(learner_row("close", 241), 3, minute(DAY, 241), b)
    assert row["action"] == "none" and "still pending" in row["reason"]


def test_a_short_the_twin_may_not_open_becomes_a_long():
    b = StockBroker.from_config(CFG)
    shorts_ok = np.ones(N, bool)
    shorts_ok[2] = False                                    # the short sale restriction on NVDA
    for seed in range(30):
        t = twin(seed=seed)
        t.mirror(learner_row("open_long"), 3, minute(DAY, 60, shorts_ok=shorts_ok), b)
        assert t.pending_order.legs[0].position_side == LONG
    assert any(twin(seed=s).rng.random() >= 0.5 for s in range(30))       # some coins did say short


def test_follow_copies_the_decision_and_exits_with_the_learner():
    b = StockBroker.from_config(CFG)
    learner = StockBot.newborn(CFG, 3, 2, DAY.open_ms)
    t = twin(seed=4)
    ctx = minute(DAY, 60)
    out = t.follow(learner, BotStep(decision=learner_row("open_long")), ctx, b)
    assert out.decision["action"] in ("open_long", "open_short") and "order" not in out.decision
    assert out.orders == [t.pending_order]
    settle(t, b)
    stopped = t.follow(learner, BotStep(stopped=True), minute(DAY, 61), b)
    assert stopped.orders and stopped.orders[0].reason == "copy_exit"
    assert settle(t, b).closed.exit_reason == "copy_exit"
    t2 = twin(seed=5)
    opened(t2, b)
    out = t2.follow(learner, BotStep(), minute(DAY, 62), b)      # the learner is flat with nothing pending
    assert [o.reason for o in out.orders] == ["copy_exit"]
    halted = np.ones(N, bool)
    halted[2] = False
    t3 = twin(seed=6)
    opened(t3, b)
    assert t3.follow(learner, BotStep(died=True), minute(DAY, 63, healthy=halted), b).orders == []


def test_session_close_and_session_flatten_exits():
    b = StockBroker.from_config(CFG)
    t = twin(seed=7)
    opened(t, b, m=330)
    order = t.send_close(DAY.boundary(385), 181.0, b, SESSION_CLOSE)
    assert order.reason == SESSION_CLOSE and order.symbol == "NVDA"
    assert settle(t, b).closed.exit_reason == "session_close"
    carried = twin(seed=8)
    opened(carried, b, m=330)
    carried.send_close(NEXT.boundary(1), 175.0, b, SESSION_FLATTEN)
    closed = settle(carried, b).closed
    assert closed.exit_reason == "session_flatten" and closed.exit_price < 175.0


def test_its_own_stop_at_3_label_sd_fixed_at_entry_and_its_own_death():
    b = StockBroker.from_config(CFG)
    seed = next(s for s in range(100) if np.random.default_rng(s).random() < 0.5)
    t = twin(seed=seed)                                     # a seed whose first coin says long
    t.mirror(learner_row("open_long"), 3, minute(DAY, 60, sd=50.0), b)
    settle(t, b)
    assert t.account.side == LONG and t.stop_bps == pytest.approx(150.0)
    allowance = costs.allowance_bps(CFG.stock_costs, "NVDA")
    ref = PRICES.copy()
    ref[2] = t.account.entry_price * (1 - 151.0 / 1e4) / (1 - allowance / 1e4)
    res = t.step(minute(DAY, 61, ref=ref, sd=500.0), b)
    assert res.stopped and res.orders[0].reason == "stop"
    poor = twin(seed=1)
    poor.account.charge_rent(9_999.0)
    res = poor.step(minute(DAY, 62), b)
    assert res.died and poor.state == DEAD
    assert poor.mirror(learner_row("open_long"), 3, minute(DAY, 63), b) is None
    assert poor.step(minute(DAY, 63), b).orders == []


def test_no_local_book_is_ever_built_or_read(monkeypatch):
    """Every LocalBook attribute and the version 1 book readers are tripwires through a whole twin life."""
    def trip(*args, **kwargs):
        raise AssertionError("a LocalBook was built or read")

    for name, value in list(vars(book_module.LocalBook).items()):
        if not name.startswith("__") or name == "__init__":
            monkeypatch.setattr(book_module.LocalBook, name, property(trip) if not callable(value) else trip)
    monkeypatch.setattr(v1_broker, "estimate_round_trip", trip)
    monkeypatch.setattr(v1_bot, "estimate_round_trip", trip)
    monkeypatch.setattr(v1_bot.Bot, "mark_price", trip)
    with pytest.raises(AssertionError):
        book_module.LocalBook(20)
    b = StockBroker.from_config(CFG)
    learner = StockBot.newborn(CFG, 3, 2, DAY.open_ms)
    t = twin(seed=9)
    for m in range(31, 391):
        ctx = minute(DAY, m, ref=PRICES * (1 + 0.001 * np.sin(m)))
        decision = learner_row("flip" if m % 60 == 0 else "open_long", m) if m % 30 == 0 and m <= 355 else None
        out = t.follow(learner, BotStep(decision=decision), ctx, b)
        if t.pending_order is not None:
            settle(t, b)
        assert out is not None
        if m == 385 and t.account.is_open:
            t.send_close(ctx.boundary_ms, float(ctx.ref[2]), b, SESSION_CLOSE)
            settle(t, b)
    assert t.account.trades > 5 and not t.account.is_open and t.account.invariant_errors(None) == []
    assert not issubclass(StockTwin, (v1_bot.Bot, TwinBot))
    for module in (twins_module, bot_module):
        source = inspect.getsource(module)
        assert "LocalBook" not in source.replace("a LocalBook", "") and "feed.book" not in source
