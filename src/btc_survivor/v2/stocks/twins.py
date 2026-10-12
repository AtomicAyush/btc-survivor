"""Coin flip twins for the stock learners by PLAN_V2 sections 3 and 10.2. StockTwin replaces the version 1 twin upkeep,
which walks a LocalBook, with the stock upkeep on its learner's home symbol, marking at the reference price.

A twin copies its learner's opens, closes and flips at the same moment with a fair coin from its own seeded generator
choosing the side of any open, closes any stale position before copying the next open, and closes when its learner is
stopped out, dies or is flat with nothing pending. It pays the same costs, borrow and rent, has its own stop at 3 label
standard deviations fixed at its own entry and its own death check, takes the session_close and session_flatten exits
like any bot, and never learns, predicts or decides on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from btc_survivor.bot import ACTIVE, DEAD, POSITION_NAMES
from btc_survivor.model import MLP
from btc_survivor.seeds import bot_rng
from btc_survivor.types import FLAT, LONG, SHORT, Leg
from btc_survivor.v2.config import StockConfig
from btc_survivor.v2.stocks.bot import OPENS, BotStep, StockBot, StockMinute
from btc_survivor.v2.stocks.broker import StockBroker, StockOrder


@dataclass
class TwinStep:
    """What a twin did in one minute after its learner, the decision row it copied and the orders it sent."""

    decision: dict | None = None
    orders: list[StockOrder] = field(default_factory=list)
    died: bool = False


class StockTwin(StockBot):
    def __init__(self, cfg: StockConfig, bot_id: int, slot: int, born_ms: int, rng: np.random.Generator) -> None:
        super().__init__(cfg, bot_id, slot, born_ms, MLP(1, 1), rng)
        self.state = ACTIVE

    @classmethod
    def newborn(cls, cfg: StockConfig, bot_id: int, slot: int, born_ms: int) -> StockTwin:
        return cls(cfg, bot_id, slot, born_ms, bot_rng(cfg.run.master_seed, bot_id))

    def step(self, ctx: StockMinute, broker: StockBroker) -> BotStep:
        """Only the shared upkeep on the home symbol. A twin never learns, predicts or decides on its own."""
        res = BotStep()
        if self.state != DEAD:
            self._upkeep(ctx, broker, res)
        return res

    def follow(self, learner: StockBot, learner_res: BotStep, ctx: StockMinute, broker: StockBroker) -> TwinStep:
        """The twin's whole minute after its learner's step, its upkeep, a copy of the learner's decision, and an exit
        when the learner was stopped out, died or is flat with nothing pending, so the pair stays in step."""
        own = self.step(ctx, broker)
        out = TwinStep(orders=list(own.orders), died=own.died)
        if learner_res.decision is not None:
            out.decision = self.mirror(learner_res.decision, learner.bot_id, ctx, broker)
            if out.decision is not None and out.decision.pop("order", None) is not None:
                out.orders.append(self.pending_order)
        learner_out = not learner.account.is_open and learner.pending_order is None
        if learner_res.stopped or learner_res.died or learner_out:
            exit_order = self.copy_exit(ctx, broker)
            if exit_order is not None:
                out.orders.append(exit_order)
        return out

    def mirror(self, learner_row: dict, learner_id: int, ctx: StockMinute, broker: StockBroker) -> dict | None:
        """Copy the learner's open, close or flip at the same moment, with a coin choosing the side of any open. A
        short the twin may not open, with shorts disabled or under the short sale restriction, becomes a long."""
        action = learner_row["action"]
        if self.state == DEAD or action not in (*OPENS, "close", "flip"):
            return None
        acct, h = self.account, self.home
        now, ref = learner_row["ts"], float(ctx.ref[h])
        row = {**learner_row, "bot_id": self.bot_id, "position_before": POSITION_NAMES[acct.side], "forced": 0}
        if self.pending_order is not None:
            return {**row, "action": "none", "reason": f"twin of bot {learner_id}, an order is still pending"}
        coin = LONG if self.rng.random() < 0.5 else SHORT
        if coin == SHORT and not (self.cfg.trading.allow_short and ctx.shorts_ok[h]):
            coin = LONG
        # A stale position left over from a learner stop or death is closed first, so every learner open is copied.
        legs = [Leg("close", acct.side)] if acct.side != FLAT else []
        if action != "close":
            legs.append(Leg("open", coin, broker.open_qty(coin, ref)))
        if not legs:
            return {**row, "action": "none", "reason": f"twin of bot {learner_id}, nothing to copy for {action}"}
        names = {LONG: "open_long", SHORT: "open_short"}
        twin_action = "close" if len(legs) == 1 and legs[0].action == "close" else \
            ("flip" if len(legs) == 2 else names[coin])
        if legs[-1].action == "open":
            self._stop_at_send = self.cfg.trading.stop_loss_sigma * float(ctx.label_sd[h])
        self.pending_order = broker.make_order(self.bot_id, self.symbol, legs, now, ref, "decision", self.rng)
        side_text = "long" if coin == LONG else "short"
        return {**row, "action": twin_action, "order": self.pending_order,
                "reason": f"twin of bot {learner_id} copying {action}, the coin chose {side_text}"}

    def copy_exit(self, ctx: StockMinute, broker: StockBroker) -> StockOrder | None:
        """Close at the same moment the learner's stop or death close is sent, or once the learner is out, while the
        home symbol is healthy."""
        if self.state == DEAD or not ctx.healthy[self.home]:
            return None
        return self.send_close(ctx.boundary_ms, float(ctx.ref[self.home]), broker, "copy_exit")
