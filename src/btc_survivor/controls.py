"""Coin flip twins. A twin copies its learner's trade timing but picks the direction with a fair coin.

A twin pays the same costs and rent and has its own stop loss and death check, and it also closes when its learner
is stopped out or dies, so the only difference from its learner is the direction. It never learns or parents.
"""

from __future__ import annotations

import numpy as np

from btc_survivor.bot import ACTIVE, DEAD, POSITION_NAMES, Bot, MinuteContext, StepResult
from btc_survivor.broker import Broker
from btc_survivor.config import Config
from btc_survivor.feed.book import LocalBook
from btc_survivor.genes import initial_genes
from btc_survivor.model import MLP
from btc_survivor.seeds import bot_rng
from btc_survivor.types import FLAT, LONG, SHORT, Leg, Order

OPENS = ("open_long", "open_short")


class TwinBot(Bot):
    def __init__(self, cfg: Config, bot_id: int, slot: int, born_ms: int, rng: np.random.Generator) -> None:
        super().__init__(cfg, bot_id, slot, 0, born_ms, initial_genes(cfg.genes), MLP(1, 1), rng, origin="control")
        self.state = ACTIVE

    @classmethod
    def newborn(cls, cfg: Config, bot_id: int, slot: int, born_ms: int) -> "TwinBot":
        return cls(cfg, bot_id, slot, born_ms, bot_rng(cfg.run.master_seed, bot_id))

    def step(self, ctx: MinuteContext, broker: Broker) -> StepResult:
        """Only the shared upkeep. A twin never learns, predicts or decides on its own."""
        res = StepResult()
        if self.state != DEAD:
            self._upkeep(ctx, broker, res)
        return res

    def mirror(self, learner_row: dict, learner_id: int, book: LocalBook, broker: Broker) -> dict | None:
        """Copy the learner's open, close or flip at the same moment, with a coin choosing the side of any open."""
        action = learner_row["action"]
        if self.state == DEAD or action not in (*OPENS, "close", "flip"):
            return None
        acct = self.account
        now = learner_row["ts"]
        row = {**learner_row, "bot_id": self.bot_id, "position_before": POSITION_NAMES[acct.side], "forced": 0}
        if self.pending_order is not None:
            return {**row, "action": "none", "reason": f"twin of bot {learner_id}, an order is still pending"}
        coin = LONG if self.rng.random() < 0.5 else SHORT
        if coin == SHORT and not self.cfg.trading.allow_short:
            coin = LONG
        # A stale position left over from a learner stop or death is closed first, so every learner open is copied.
        legs = [Leg("close", acct.side)] if acct.side != FLAT else []
        if action != "close":
            legs.append(Leg("open", coin, broker.open_qty(book.mid)))
        if not legs:
            return {**row, "action": "none", "reason": f"twin of bot {learner_id}, nothing to copy for {action}"}
        names = {("open", LONG): "open_long", ("open", SHORT): "open_short"}
        twin_action = "close" if len(legs) == 1 and legs[0].action == "close" else \
            ("flip" if len(legs) == 2 else names[("open", coin)])
        order = broker.make_order(self.bot_id, legs, now, book.mid, "decision", self.rng)
        self.pending_order = order
        side_text = "long" if coin == LONG else "short"
        return {**row, "action": twin_action, "order": order,
                "reason": f"twin of bot {learner_id} copying {action}, the coin chose {side_text}"}

    def copy_exit(self, now: int, book: LocalBook, broker: Broker) -> Order | None:
        """Close at the same moment the learner's stop or death close is sent, so the pair stays in step."""
        if self.state == DEAD or self.pending_order is not None or not self.account.is_open:
            return None
        self.pending_order = broker.make_order(self.bot_id, [Leg("close", self.account.side)], now, book.mid,
                                               "copy_exit", self.rng)
        return self.pending_order
