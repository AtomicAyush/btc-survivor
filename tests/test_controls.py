import dataclasses

import numpy as np
import pytest

from btc_survivor.bot import ACTIVE, DEAD, MinuteContext
from btc_survivor.broker import Broker
from btc_survivor.config import load_config
from btc_survivor.controls import TwinBot
from btc_survivor.feed.book import LocalBook
from btc_survivor.labels import Matured, MaturedWindow
from btc_survivor.standardizer import LabelScale
from btc_survivor.types import FLAT, LONG, SHORT
from helpers import EXAMPLE_CONFIG

CFG = load_config(EXAMPLE_CONFIG)
M = 1_791_000_000_000
MIN = 60_000


def book(mid=100.0):
    b = LocalBook(20)
    b.apply_snapshot(bids=[(mid - 0.01 - i * 0.01, 50.0) for i in range(20)],
                     asks=[(mid + 0.01 + i * 0.01, 50.0) for i in range(20)])
    return b


def broker():
    return Broker(CFG.costs, CFG.trading.notional)


def learner_row(action, ts=M + 30 * MIN):
    return {"bot_id": 1, "ts": ts, "position_before": "flat", "ref_mid": 100.0, "prediction_bps": 3.0,
            "required_edge_bps": 192.0, "cost_estimate_bps": 184.0, "margin_bps": 8.0, "gate_passed": 0,
            "forced": 1, "action": action, "reason": "learner reason"}


def twin(seed=1):
    return TwinBot(CFG, 21, 0, M, np.random.default_rng(seed))


def ready_scale():
    scale = LabelScale(1000, 30)
    for i in range(30):
        scale.update(10.0 if i % 2 else -10.0)
    return scale


def test_twin_is_active_from_birth_and_never_decides_on_its_own():
    t = twin()
    assert t.state == ACTIVE and t.origin == "control"
    w = MaturedWindow(240, 11)
    w.add(Matured(M, np.zeros(11), 5.0))
    ctx = MinuteContext(M + 30 * MIN, book(), np.zeros(11), Matured(M, np.zeros(11), 5.0), w, ready_scale())
    res = t.step(ctx, broker())  # the plain bot step would predict and decide here
    assert res.decision is None and res.orders == [] and t.trained_minutes == 0
    assert t.account.rent == pytest.approx(CFG.costs.daily_running_cost / 1440)


def test_twin_dies_through_the_shared_upkeep_and_then_stays_inert():
    t = twin()
    t.account.open(LONG, 5.0, 100.0, 0.0, M, trade_id=1)
    t.account.charge_rent(t.account.cash - 3.0)
    ctx = MinuteContext(M + MIN, book(), None, None, MaturedWindow(240, 11), LabelScale(1000, 30))
    res = t.step(ctx, broker())
    assert res.died and t.state == DEAD and res.orders[0].reason == "death"
    rent = t.account.rent
    again = t.step(MinuteContext(M + 2 * MIN, book(), None, None, MaturedWindow(240, 11), LabelScale(1000, 30)), broker())
    assert again.orders == [] and t.account.rent == rent


def test_twin_closes_when_its_learner_is_stopped_out_or_dies():
    t = twin()
    assert t.copy_exit(M, book(), broker()) is None  # flat, nothing to close
    t.account.open(SHORT, 5.0, 100.0, 0.0, M, trade_id=1)
    order = t.copy_exit(M + MIN, book(), broker())
    assert order.reason == "copy_exit" and [(l.action, l.position_side) for l in order.legs] == [("close", SHORT)]
    assert t.copy_exit(M + 2 * MIN, book(), broker()) is None  # already closing


@pytest.mark.parametrize("action", ["hold", "none"])
def test_twin_ignores_holds_and_do_nothing(action):
    assert twin().mirror(learner_row(action), 1, book(), broker()) is None


@pytest.mark.parametrize("action", ["open_long", "open_short"])
def test_twin_opens_a_coin_flip_side_when_its_learner_opens(action):
    row = twin().mirror(learner_row(action), 1, book(), broker())
    [leg] = row["order"].legs
    assert leg.action == "open" and leg.qty == pytest.approx(5.0)
    assert row["action"] == ("open_long" if leg.position_side == LONG else "open_short")
    assert row["bot_id"] == 21 and row["ts"] == M + 30 * MIN and row["forced"] == 0
    assert row["prediction_bps"] == 3.0 and "twin of bot 1" in row["reason"]


def test_twin_closes_and_flips_with_its_learner():
    t = twin()
    t.account.open(SHORT, 5.0, 100.0, 0.0, M, trade_id=1)
    row = t.mirror(learner_row("close"), 1, book(), broker())
    assert row["action"] == "close" and [(l.action, l.position_side) for l in row["order"].legs] == [("close", SHORT)]
    t.pending_order = None
    row = t.mirror(learner_row("flip"), 1, book(), broker())
    assert row["action"] == "flip" and [l.action for l in row["order"].legs] == ["close", "open"]


def test_twin_with_nothing_to_copy_logs_why():
    row = twin().mirror(learner_row("close"), 1, book(), broker())
    assert row["action"] == "none" and "nothing to copy" in row["reason"]


def test_twin_holding_a_stale_position_closes_it_and_copies_the_open():
    t = twin()
    t.account.open(LONG, 5.0, 100.0, 0.0, M, trade_id=1)  # left over after its learner was stopped out
    row = t.mirror(learner_row("open_short"), 1, book(), broker())
    legs = row["order"].legs
    assert row["action"] == "flip"
    assert [(l.action, l.position_side) for l in legs][0] == ("close", LONG) and legs[1].action == "open"


def test_twin_flat_when_learner_flips_simply_opens():
    row = twin().mirror(learner_row("flip"), 1, book(), broker())
    assert row["action"] in ("open_long", "open_short") and len(row["order"].legs) == 1


def test_twin_with_a_pending_order_waits():
    t = twin()
    t.mirror(learner_row("open_long"), 1, book(), broker())
    row = t.mirror(learner_row("open_long", ts=M + 60 * MIN), 1, book(), broker())
    assert row["action"] == "none" and "pending" in row["reason"] and "order" not in row


def test_dead_twin_copies_nothing():
    t = twin()
    t.state = DEAD
    assert t.mirror(learner_row("open_long"), 1, book(), broker()) is None


def test_coin_is_fair_and_seeded():
    sides = []
    for seed in range(2000):
        row = twin(seed).mirror(learner_row("open_long"), 1, book(), broker())
        sides.append(row["order"].legs[0].position_side)
    share = sides.count(LONG) / len(sides)
    assert 0.46 < share < 0.54
    def sides(seed):
        t, out = twin(seed), []
        for i in range(20):
            row = t.mirror(learner_row("open_long", ts=M + i * MIN), 1, book(), broker())
            out.append(row["order"].legs[-1].position_side)
            t.pending_order = None
        return out
    assert sides(7) == sides(7) and sides(7) != sides(8)


def test_flip_open_leg_side_comes_from_the_coin():
    sides = []
    for seed in range(400):
        t = twin(seed)
        t.account.open(SHORT, 5.0, 100.0, 0.0, M, trade_id=1)
        sides.append(t.mirror(learner_row("flip"), 1, book(), broker())["order"].legs[1].position_side)
    assert 0.4 < sides.count(LONG) / len(sides) < 0.6


def test_shorts_disabled_twin_only_goes_long():
    cfg = dataclasses.replace(CFG, trading=dataclasses.replace(CFG.trading, allow_short=False))
    for seed in range(50):
        t = TwinBot(cfg, 21, 0, M, np.random.default_rng(seed))
        assert t.mirror(learner_row("open_long"), 1, book(), broker())["order"].legs[0].position_side == LONG


def test_twin_has_its_own_stop_loss():
    t = twin()
    t.account.open(LONG, 5.0, 100.0, 0.0, M, trade_id=1)
    ctx = MinuteContext(M + MIN, book(98.9), None, None, MaturedWindow(240, 11), LabelScale(1000, 30))
    res = t.step(ctx, broker())
    assert res.stopped and res.orders[0].reason == "stop" and t.account.side == LONG
    assert t.account.side != FLAT
