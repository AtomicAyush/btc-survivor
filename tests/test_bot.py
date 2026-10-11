import dataclasses
import itertools

import numpy as np
import pytest

from btc_survivor.bot import ACTIVE, DEAD, WARMUP, Bot, MinuteContext, decide
from btc_survivor.broker import Broker, Execution
from btc_survivor.config import load_config
from btc_survivor.feed.book import LocalBook
from btc_survivor.genes import Genes
from btc_survivor.labels import Matured, MaturedWindow
from btc_survivor.model import MLP
from btc_survivor.standardizer import LabelScale
from btc_survivor.types import FLAT, LONG, SHORT
from helpers import EXAMPLE_CONFIG

BASE = load_config(EXAMPLE_CONFIG)
M = 1_791_000_000_000  # a minute boundary, minute index divisible by 30
MIN = 60_000
HORIZON = BASE.time.horizon_minutes


def config(**trading):
    cfg = dataclasses.replace(BASE, time=dataclasses.replace(BASE.time, warmup_hours_random=0.05))  # 3 minutes
    if trading:
        cfg = dataclasses.replace(cfg, trading=dataclasses.replace(cfg.trading, **trading))
    return cfg


def book(mid=100.0):
    b = LocalBook(20)
    b.apply_snapshot(bids=[(mid - 0.01 - i * 0.01, 50.0) for i in range(20)],
                     asks=[(mid + 0.01 + i * 0.01, 50.0) for i in range(20)])
    return b


def ready_scale(std=10.0):
    scale = LabelScale(1000, 30)
    for i in range(30):
        scale.update(std if i % 2 else -std)
    return scale


def window():
    w = MaturedWindow(240, 11)
    for i in range(10):
        w.add(Matured(M - (40 - i) * MIN, np.zeros(11), 0.0))
    return w


def make_bot(cfg=None, output=0.0, slot=0, state=ACTIVE):
    """A bot whose network always outputs the given value, so prediction_bps is 10 times it."""
    cfg = cfg or config()
    model = MLP(11, cfg.model.hidden_units)
    model.b2[0] = output
    bot = Bot(cfg, bot_id=7, slot=slot, generation=0, born_ms=M, genes=Genes(8.0, 0.0), model=model,
              rng=np.random.default_rng(1))
    bot.state = state
    return bot


def ctx(t, b=None, z=True, matured=None, scale=None, w=None):
    return MinuteContext(boundary_ms=t, book=b or book(), z=np.zeros(11) if z else None, matured=matured,
                         window=w or window(), label_scale=scale or ready_scale())


def broker(cfg=None):
    cfg = cfg or config()
    return Broker(cfg.costs, cfg.trading.notional)


# The decision rule as a table: (position, prediction, required, allow_short, forced) -> (action, gate, forced used)
TABLE = [
    (FLAT, 200, 100, True, False, ("open_long", True, False)),
    (FLAT, -200, 100, True, False, ("open_short", True, False)),
    (LONG, 200, 100, True, False, ("hold", True, False)),
    (SHORT, -200, 100, True, False, ("hold", True, False)),
    (LONG, -200, 100, True, False, ("flip", True, False)),
    (SHORT, 200, 100, True, False, ("flip", True, False)),
    (LONG, 50, 100, True, False, ("close", False, False)),
    (SHORT, -50, 100, True, False, ("close", False, False)),
    (FLAT, 50, 100, True, False, ("none", False, False)),
    (FLAT, 100, 100, True, False, ("open_long", True, False)),
    (FLAT, -200, 100, False, False, ("none", False, False)),
    (LONG, -200, 100, False, False, ("close", False, False)),
    (FLAT, 50, 100, True, True, ("open_long", False, True)),
    (FLAT, -50, 100, True, True, ("open_short", False, True)),
    (FLAT, -50, 100, False, True, ("none", False, False)),
    (FLAT, 0, 100, True, True, ("none", False, False)),
    (LONG, 50, 100, True, True, ("close", False, False)),
    (FLAT, 200, 100, True, True, ("open_long", True, False)),
]


@pytest.mark.parametrize("position,pred,required,allow_short,forced,expected", TABLE)
def test_decision_rule_table(position, pred, required, allow_short, forced, expected):
    assert decide(position, pred, required, allow_short, forced) == expected


def test_decision_rule_skill_gate_blocks_but_forced_still_explores():
    assert decide(FLAT, 500, 100, True, False, skill_ok=False) == ("none", False, False)
    assert decide(LONG, 500, 100, True, False, skill_ok=False) == ("close", False, False)
    assert decide(FLAT, 500, 100, True, True, skill_ok=False) == ("open_long", False, True)


def test_every_combination_returns_a_known_action():
    for position, pred, short, forced in itertools.product((FLAT, LONG, SHORT), (-300, -1, 0, 1, 300),
                                                           (True, False), (True, False)):
        action, gate, used = decide(position, pred, 100, short, forced)
        assert action in {"open_long", "open_short", "hold", "flip", "close", "none"}
        assert not (gate and used)


def test_decides_only_at_its_phase_minute():
    bot = make_bot(slot=3)
    b = broker()
    phase_minutes = []
    for i in range(1, 61):
        if bot.step(ctx(M + i * MIN), b).decision is not None:
            phase_minutes.append(i)
    assert phase_minutes == [3, 33]


def test_no_decision_without_a_valid_feature_vector_or_label_scale():
    bot = make_bot()
    assert bot.step(ctx(M + HORIZON * MIN, z=False), broker()).decision is None
    assert bot.step(ctx(M + 2 * HORIZON * MIN, scale=LabelScale(1000, 30)), broker()).decision is None


def test_warmup_logs_its_phase_minutes_and_ends_after_enough_training_minutes():
    bot = make_bot(output=50.0, state=WARMUP)
    res = bot.step(ctx(M + HORIZON * MIN), broker())
    assert res.decision["action"] == "none" and res.decision["reason"].startswith("warmup")
    assert res.orders == []
    for i in range(3):
        bot.step(ctx(M + (HORIZON + 1 + i) * MIN, matured=Matured(M, np.zeros(11), 1.0)), broker())
        if i == 1:
            assert bot.state == WARMUP and bot.alive_minutes >= 3  # two training minutes are not enough
    assert bot.state == ACTIVE and bot.trained_minutes == 3


def test_no_training_without_a_newly_matured_sample():
    bot = make_bot(state=WARMUP)
    for i in range(1, 10):
        bot.step(ctx(M + i * MIN), broker())
    assert bot.trained_minutes == 0 and bot.state == WARMUP


def test_rent_is_charged_every_step():
    bot = make_bot()
    for i in range(1, 1441):
        bot.step(ctx(M + i * MIN), broker())
    assert bot.account.rent == pytest.approx(BASE.costs.daily_running_cost)


def test_passing_signal_opens_with_notional_sized_quantity():
    bot = make_bot(output=25.0)  # 250 bps against a required edge near 190
    res = bot.step(ctx(M + HORIZON * MIN), broker())
    d = res.decision
    assert d["action"] == "open_long" and d["gate_passed"] == 1 and d["forced"] == 0
    assert d["required_edge_bps"] == pytest.approx(d["cost_estimate_bps"] + 8.0)
    assert d["cost_estimate_bps"] == pytest.approx(180 + 2.0 + 2.0, abs=0.1)
    [order] = res.orders
    assert order.reason == "decision" and order.legs[0].qty == pytest.approx(500.0 / 100.0)
    assert bot.pending_order is order
    assert d["position_before"] == "flat" and d["ref_mid"] == pytest.approx(100.0)
    assert d["prediction_bps"] == pytest.approx(250.0) and d["margin_bps"] == 8.0 and d["ts"] == M + HORIZON * MIN


def test_failing_signal_while_flat_does_nothing_and_logs_why():
    bot = make_bot(output=1.0)
    d = bot.step(ctx(M + HORIZON * MIN), broker()).decision
    assert d["action"] == "none" and d["gate_passed"] == 0
    assert "below required edge" in d["reason"]


def test_flip_closes_then_opens_the_other_side():
    bot = make_bot(output=-25.0)
    bot.account.open(LONG, 5.0, 100.0, 0.0, M, trade_id=1)
    [order] = bot.step(ctx(M + HORIZON * MIN), broker()).orders
    assert [(l.action, l.position_side) for l in order.legs] == [("close", LONG), ("open", SHORT)]


def test_forced_exploration_counts_from_birth_and_only_when_flat():
    cfg = config()
    bot = make_bot(cfg, output=1.0)
    b = broker(cfg)
    forced_at = None
    for i in range(1, 13 * 60 + 1):
        res = bot.step(ctx(M + i * MIN), b)
        if res.decision and res.decision["forced"]:
            forced_at = i
            break
    assert forced_at is not None and 12 * 60 <= forced_at < 12 * 60 + HORIZON
    assert res.decision["action"] == "open_long" and res.decision["gate_passed"] == 0
    assert res.orders[0].forced
    bot.on_execution(Execution(opened_trade_id=1))
    assert bot.minutes_since_open == 0 and bot.pending_order is None


def test_holding_a_position_past_the_timer_follows_the_normal_rule():
    bot = make_bot(output=1.0)
    bot.account.open(LONG, 5.0, 100.0, 0.0, M, trade_id=1)
    bot.minutes_since_open = 10_000
    res = bot.step(ctx(M + HORIZON * MIN), broker())
    d = res.decision
    assert d["action"] == "close" and d["forced"] == 0 and d["position_before"] == "long"
    assert [(l.action, l.position_side) for l in res.orders[0].legs] == [("close", LONG)]


def test_stop_loss_closes_and_suppresses_that_minutes_decision():
    bot = make_bot(output=25.0)
    bot.account.open(LONG, 5.0, 100.0, 0.0, M, trade_id=1)
    res = bot.step(ctx(M + HORIZON * MIN, b=book(98.9)), broker())  # about 1.1 percent below entry
    assert res.stopped and res.orders[0].reason == "stop"
    assert res.decision["action"] == "none" and res.decision["reason"] == "stop loss fired this minute"


def test_small_adverse_move_does_not_stop():
    bot = make_bot(output=25.0)
    bot.account.open(LONG, 5.0, 100.0, 0.0, M, trade_id=1)
    res = bot.step(ctx(M + 1 * MIN, b=book(99.5)), broker())
    assert not res.stopped and res.orders == []


def custom_book(bids, asks):
    b = LocalBook(20)
    b.apply_snapshot(bids=bids, asks=asks)
    return b


def test_short_stop_uses_the_ask_side_walk():
    bot = make_bot()
    bot.account.open(SHORT, 5.0, 100.0, 0.0, M, trade_id=1)
    b = custom_book([(100.90, 50.0)], [(101.01, 50.0)])  # mid 100.955 is under 1 percent, the ask is over
    assert bot.step(ctx(M + 1 * MIN, b=b), broker()).stopped


def test_long_stop_uses_the_bid_side_walk_for_its_quantity():
    bot = make_bot()
    bot.account.open(LONG, 5.0, 100.0, 0.0, M, trade_id=1)
    b = custom_book([(99.5, 0.1), (98.9, 50.0)], [(100.01, 50.0)])  # best bid and mid are under 1 percent away
    res = bot.step(ctx(M + 1 * MIN, b=b), broker())
    walk_avg = (99.5 * 0.1 + 98.9 * 4.9) / 5.0
    assert res.stopped
    assert bot.equity == pytest.approx(bot.account.cash + 5.0 * (walk_avg - 100.0))


def test_stop_fires_at_exactly_the_threshold():
    bot = make_bot()
    bot.account.open(LONG, 5.0, 100.0, 0.0, M, trade_id=1)
    assert bot.step(ctx(M + 1 * MIN, b=custom_book([(99.0, 50.0)], [(99.02, 50.0)])), broker()).stopped


def test_a_pending_stop_is_not_sent_twice():
    bot = make_bot()
    bot.account.open(LONG, 5.0, 100.0, 0.0, M, trade_id=1)
    b = broker()
    assert bot.step(ctx(M + 1 * MIN, b=book(98.9)), b).stopped
    res = bot.step(ctx(M + 2 * MIN, b=book(98.9)), b)
    assert res.orders == [] and not res.stopped


def test_death_is_checked_before_the_stop():
    bot = make_bot()
    bot.account.open(LONG, 5.0, 100.0, 0.0, M, trade_id=1)
    bot.account.charge_rent(bot.account.cash - 12.0)
    res = bot.step(ctx(M + 1 * MIN, b=book(98.9)), broker())
    assert res.died and not res.stopped and res.orders[0].reason == "death"


def test_holding_a_position_without_features_keeps_it_and_still_checks_the_stop():
    bot = make_bot(output=-25.0)
    bot.account.open(LONG, 5.0, 100.0, 0.0, M, trade_id=1)
    res = bot.step(ctx(M + HORIZON * MIN, z=False), broker())
    assert res.decision is None and res.orders == [] and bot.account.side == LONG
    res = bot.step(ctx(M + 2 * HORIZON * MIN, z=False, b=book(98.9)), broker())
    assert res.stopped and res.orders[0].reason == "stop"


def test_dead_bot_does_nothing_and_pays_nothing():
    bot = make_bot(output=25.0, state=DEAD)
    res = bot.step(ctx(M + HORIZON * MIN), broker())
    assert res.decision is None and res.orders == [] and bot.account.rent == 0.0


def test_zero_warmup_starts_active():
    cfg = dataclasses.replace(BASE, time=dataclasses.replace(BASE.time, warmup_hours_random=0))
    assert Bot.newborn(cfg, 1, 0, M).state == ACTIVE


def test_no_learning_without_a_valid_feature_vector():
    bot = make_bot(state=WARMUP)
    bot.step(ctx(M + 1 * MIN, z=False, matured=Matured(M, np.zeros(11), 1.0)), broker())
    assert bot.trained_minutes == 0


@pytest.mark.parametrize("state", [WARMUP, ACTIVE])
def test_rent_accrues_in_warmup_and_active(state):
    bot = make_bot(state=state)
    for i in range(1, 61):
        bot.step(ctx(M + i * MIN), broker())
    assert bot.account.rent == pytest.approx(BASE.costs.daily_running_cost / 24)


def test_warmup_bot_can_die():
    bot = make_bot(state=WARMUP)
    bot.account.charge_rent(bot.account.cash - 5.0)
    assert bot.step(ctx(M + 1 * MIN), broker()).died


def test_short_funding_accrues_each_minute():
    bot = make_bot()
    bot.account.open(SHORT, 5.0, 100.0, 0.0, M, trade_id=1)
    for i in range(1, 61):
        bot.step(ctx(M + i * MIN), broker())
    assert bot.account.funding == pytest.approx(BASE.costs.short_funding_bps_per_hour / 1e4 * 500.0)


def test_rejected_open_keeps_the_forced_timer_running():
    bot = make_bot(output=1.0)
    bot.minutes_since_open = 10_000
    res = bot.step(ctx(M + HORIZON * MIN), broker())
    assert res.decision["forced"] == 1
    bot.on_execution(Execution(rejected="insufficient_depth"))
    assert bot.pending_order is None and bot.minutes_since_open > 10_000
    assert bot.step(ctx(M + 2 * HORIZON * MIN), broker()).decision["forced"] == 1
    bot.on_execution(Execution())  # a close or an empty settlement does not reset it either
    assert bot.minutes_since_open > 10_000


def test_prediction_survives_until_its_label_matures():
    bot = make_bot(output=2.0)
    b = broker()
    for i in range(1, HORIZON + 1):
        bot.step(ctx(M + i * MIN), b)
    bot.step(ctx(M + (HORIZON + 1) * MIN, matured=Matured(M + 1 * MIN, np.zeros(11), 5.0)), b)
    assert bot.skill.stats().n == 1


def test_prediction_is_scaled_by_the_label_scale():
    bot = make_bot(output=2.0)
    bot.step(ctx(M + 1 * MIN, scale=ready_scale(7.0)), broker())
    assert bot.pending_preds[M + 1 * MIN] == pytest.approx(14.0)


def test_skill_gate_opens_only_with_a_full_window_of_positive_correlation():
    cfg = dataclasses.replace(config(), model=dataclasses.replace(BASE.model, gate_on_skill=True))
    for pairs, expected in ((360, "open_long"), (359, "none")):
        bot = make_bot(cfg, output=25.0)
        for i in range(pairs):
            bot.skill.add(float(i % 7), float(i % 7) + 0.1)
        d = bot.step(ctx(M + HORIZON * MIN), broker(cfg)).decision
        assert d["action"] == expected
        assert d["gate_passed"] == (1 if pairs == 360 else 0)


def test_short_prediction_cost_includes_short_charges():
    long_row = make_bot(output=1.0).step(ctx(M + HORIZON * MIN), broker()).decision
    short_row = make_bot(output=-1.0).step(ctx(M + HORIZON * MIN), broker()).decision
    assert short_row["cost_estimate_bps"] - long_row["cost_estimate_bps"] == pytest.approx(2.0 + 0.5 * 0.5)


def test_warmup_rows_still_carry_the_real_gate_result_and_cost():
    bot = make_bot(output=25.0, state=WARMUP)
    d = bot.step(ctx(M + HORIZON * MIN), broker()).decision
    assert d["action"] == "none" and d["gate_passed"] == 1 and d["cost_estimate_bps"] > 180


def test_genes_are_clamped_to_their_bounds():
    assert Genes(0.0, 1.0).clamped(BASE.genes) == Genes(2.0, 0.01)
    bot = Bot(config(), 1, 0, 0, M, Genes(0.0, 0.0), MLP(11, 16), np.random.default_rng(1))
    bot.state = ACTIVE
    d = bot.step(ctx(M + HORIZON * MIN), broker()).decision
    assert d["margin_bps"] == 2.0 and d["required_edge_bps"] == pytest.approx(d["cost_estimate_bps"] + 2.0)


def test_pending_order_blocks_a_new_decision():
    bot = make_bot(output=25.0)
    b = broker()
    bot.step(ctx(M + HORIZON * MIN), b)
    d = bot.step(ctx(M + 2 * HORIZON * MIN), b).decision
    assert d["action"] == "none" and d["reason"] == "an order is still pending"


def test_death_when_equity_cannot_cover_a_round_trip_and_a_day_of_rent():
    bot = make_bot()
    bot.account.charge_rent(bot.account.cash - 5.0)  # leaves 5 dollars, below about 9.2 plus 1.16
    res = bot.step(ctx(M + 1 * MIN), broker())
    assert res.died and bot.state == DEAD and res.orders == []
    assert bot.death_threshold == pytest.approx(0.0184 * 500 + 1.16, abs=0.01)


def test_death_with_an_open_position_sends_a_close_or_adopts_a_pending_one():
    bot = make_bot()
    bot.account.open(LONG, 5.0, 100.0, 0.0, M, trade_id=1)
    bot.account.charge_rent(bot.account.cash - 5.0)
    res = bot.step(ctx(M + 1 * MIN), broker())
    assert res.died and res.orders[0].reason == "death"
    bot2 = make_bot()
    bot2.account.open(LONG, 5.0, 100.0, 0.0, M, trade_id=1)
    pending = broker().make_order(7, [], M, 100.0, "stop", np.random.default_rng(1))
    bot2.pending_order = pending
    bot2.account.charge_rent(bot2.account.cash - 5.0)
    res2 = bot2.step(ctx(M + 1 * MIN), broker())
    assert res2.died and res2.orders == [] and bot2.pending_order is pending


def test_prediction_is_scored_against_its_label_before_training():
    bot = make_bot(output=2.0)  # 20 bps
    b = broker()
    bot.step(ctx(M + 1 * MIN), b)
    assert bot.pending_preds[M + 1 * MIN] == pytest.approx(20.0)
    bot.step(ctx(M + (1 + HORIZON) * MIN, matured=Matured(M + 1 * MIN, np.zeros(11), -15.0)), b)
    stats = bot.skill.stats()
    assert stats.n == 1 and stats.mae == pytest.approx(35.0)
    assert M + 1 * MIN not in bot.pending_preds


def test_old_predictions_that_can_no_longer_mature_are_forgotten():
    bot = make_bot()
    for i in range(1, 3 * HORIZON):
        bot.step(ctx(M + i * MIN), broker())
    assert len(bot.pending_preds) <= HORIZON


def test_skill_gate_blocks_an_unproven_bot_when_enabled():
    cfg = dataclasses.replace(config(), model=dataclasses.replace(BASE.model, gate_on_skill=True))
    bot = make_bot(cfg, output=25.0)
    d = bot.step(ctx(M + HORIZON * MIN), broker(cfg)).decision
    assert d["action"] == "none" and d["reason"] == "skill gate closed"


def test_shorts_disabled_turns_a_negative_signal_into_flat():
    cfg = config(allow_short=False)
    bot = make_bot(cfg, output=-25.0)
    d = bot.step(ctx(M + HORIZON * MIN), broker(cfg)).decision
    assert d["action"] == "none" and "shorts are disabled" in d["reason"]


def test_book_too_thin_for_the_notional_is_logged():
    thin = LocalBook(20)
    thin.apply_snapshot(bids=[(99.99, 0.01)], asks=[(100.01, 0.01)])
    d = make_bot(output=25.0).step(ctx(M + HORIZON * MIN, b=thin), broker()).decision
    assert d["action"] == "none" and "cannot fill" in d["reason"]


def test_hourly_equity_is_sampled_every_sixty_minutes():
    bot = make_bot()
    for i in range(1, 60):
        bot.step(ctx(M + i * MIN), broker())
    assert len(bot.hourly_equity) == 1
    bot.step(ctx(M + 60 * MIN), broker())
    assert len(bot.hourly_equity) == 2
    for i in range(61, 121):
        bot.step(ctx(M + i * MIN), broker())
    assert len(bot.hourly_equity) == 3  # the start plus two hours


def test_newborn_bots_are_seeded_from_the_master_seed():
    a = Bot.newborn(BASE, 5, 0, M)
    b = Bot.newborn(BASE, 5, 0, M)
    c = Bot.newborn(BASE, 6, 0, M)
    assert np.array_equal(a.model.params, b.model.params) and not np.array_equal(a.model.params, c.model.params)
    assert a.state == WARMUP and a.warmup_minutes == 360 and a.genes == Genes(8.0, 0.001)
