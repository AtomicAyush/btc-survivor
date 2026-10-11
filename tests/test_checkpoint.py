import numpy as np
import pytest

from btc_survivor import checkpoint
from btc_survivor.bot import Bot
from btc_survivor.config import load_config
from btc_survivor.controls import TwinBot
from btc_survivor.labels import Matured, MaturedWindow
from btc_survivor.standardizer import LabelScale
from btc_survivor.types import LONG, SHORT, Leg, Order
from helpers import EXAMPLE_CONFIG

CFG = load_config(EXAMPLE_CONFIG)
M = 1_791_000_000_000


def trained_bot():
    """A learner with Adam state, skill pairs, an open short, a pending order and used generator state."""
    bot = Bot.newborn(CFG, 7, 6, M)
    rng = np.random.default_rng(0)
    window, scale = MaturedWindow(240, 11), LabelScale(1000, 5)
    for i in range(40):
        window.add(Matured(M + i, rng.normal(size=11), float(rng.normal(0, 10))))
        scale.update(float(rng.normal(0, 10)))
    for i in range(25):
        X, y = window.batch(bot.rng, 8)
        _, grad = bot.model.loss_and_grad(X, y / scale.scale, 1.0)
        bot.opt.step(bot.model, grad, 0.001, 1e-4, 5.0)
        bot.skill.add(float(i), float(-i))
    bot.account.open(SHORT, 0.006, 80_000.0, 4.3, M, trade_id=3, open_charge=0.1)
    bot.account.charge_rent(1.5)
    bot.pending_order = Order(7, [Leg("close", SHORT), Leg("open", LONG, 0.006)], M, M + 1200, 80_000.0, "decision")
    bot.pending_preds = {M: 1.5, M + 60_000: -2.0}
    bot.hourly_equity.extend([9990.0, 9980.0])
    bot.alive_minutes, bot.trained_minutes, bot.minutes_since_open = 500, 300, 40
    bot.death_threshold = 12.5
    return bot


def round_trip(bot):
    arrays = checkpoint.Arrays()
    meta, blob = checkpoint.encode({"bots": [checkpoint.dump_bot(bot, arrays)]}, arrays)
    decoded, loaded = checkpoint.decode(meta, blob)
    return checkpoint.load_bot(CFG, decoded["bots"][0], loaded)


def test_learner_round_trip_is_bit_for_bit_and_its_generator_continues_identically():
    bot = trained_bot()
    back = round_trip(bot)
    assert type(back) is Bot and back.genes == bot.genes and back.origin == bot.origin
    for a, b in ((bot.model.params, back.model.params), (bot.opt.m, back.opt.m), (bot.opt.v, back.opt.v),
                 (bot.skill.pred, back.skill.pred), (bot.skill.label, back.skill.label)):
        assert a.tobytes() == b.tobytes()
    assert np.shares_memory(back.model.W1, back.model.params)  # the views still point at the restored weights
    assert back.opt.t == bot.opt.t and back.skill.n == bot.skill.n and back.skill.head == bot.skill.head
    assert back.account == bot.account and back.pending_order == bot.pending_order
    assert back.pending_preds == bot.pending_preds and list(back.hourly_equity) == list(bot.hourly_equity)
    assert back.hourly_equity.maxlen == bot.hourly_equity.maxlen
    for name in ("state", "slot", "generation", "born_ms", "alive_minutes", "trained_minutes",
                 "minutes_since_open", "warmup_minutes", "equity", "death_threshold"):
        assert getattr(back, name) == getattr(bot, name)
    assert np.array_equal(back.rng.random(5), bot.rng.random(5))


def test_twin_round_trip_stays_a_twin():
    twin = TwinBot.newborn(CFG, 30, 9, M)
    twin.rng.random(3)
    back = round_trip(twin)
    assert isinstance(back, TwinBot) and back.origin == "control" and back.slot == 9
    assert np.array_equal(back.rng.random(4), twin.rng.random(4))


def test_encoding_is_deterministic_and_keeps_shapes():
    arrays = checkpoint.Arrays()
    arrays.put("b", np.arange(6.0).reshape(2, 3))
    arrays.put("a", np.array([np.nan, -0.0, 1e-300]))
    one = checkpoint.encode({"x": 1, "y": [1.5, None]}, arrays)
    assert one == checkpoint.encode({"y": [1.5, None], "x": 1}, arrays)
    meta, out = checkpoint.decode(*one)
    assert meta["x"] == 1 and out["b"].shape == (2, 3) and out["a"].tobytes() == arrays.items["a"].tobytes()


@pytest.mark.parametrize("damage", ["blob", "json", "format", "layout"])
def test_damaged_checkpoints_raise_value_error(damage):
    arrays = checkpoint.Arrays()
    arrays.put("a", np.ones(4))
    meta, blob = checkpoint.encode({}, arrays)
    if damage == "blob":
        blob = blob[:5]
    elif damage == "json":
        meta = meta[:-3]
    elif damage == "format":
        meta = meta.replace(f'"format":{checkpoint.FORMAT_VERSION}', '"format":99')
    else:
        meta = meta.replace(",0,32]", ",0,64]")
    with pytest.raises(ValueError):
        checkpoint.decode(meta, blob)
