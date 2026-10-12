"""The pooled learners and coin flip twins of one stock run, by PLAN_V2 sections 10.1 to 10.3.

Every slot holds one learner and, with coin flip twins on, one twin, all born together from the master seed. Learners
take ids 1 to size in slot order and twins the next size ids, as version 1 numbers them, and each bot draws from its
own generator seeded by the master seed and its id. Each learner is drawn as MLP.random at the 10 price inputs with two
zero news columns appended, then its forced timer delay, so S1 and S2, which share the master seed, are born identical
in every slot. Nothing is ever replaced or mutated, so a death leaves its slot empty for the rest of the run.
"""

from __future__ import annotations

import numpy as np

from btc_survivor.bot import DEAD
from btc_survivor.v2.config import StockConfig
from btc_survivor.v2.stocks.bot import StockBot
from btc_survivor.v2.stocks.twins import StockTwin

LEARNER, TWIN = "random", "control"


class StockPopulation:
    """The learner and twin of every slot, None where a bot died or twins are off."""

    def __init__(self, learners: list[StockBot | None], twins: list[StockTwin | None]) -> None:
        if len(learners) != len(twins):
            raise ValueError("every slot needs a learner entry and a twin entry")
        self.learners = learners
        self.twins = twins

    @classmethod
    def fresh(cls, cfg: StockConfig, born_ms: int) -> StockPopulation:
        """Learners with ids 1 to size in slot order, then their twins with the next ids, all from the master seed."""
        size = cfg.population.size
        learners: list[StockBot | None] = [StockBot.newborn(cfg, slot + 1, slot, born_ms) for slot in range(size)]
        twins: list[StockTwin | None] = [None] * size
        if cfg.controls.coin_flip_twins:
            twins = [StockTwin.newborn(cfg, size + 1 + slot, slot, born_ms) for slot in range(size)]
        return cls(learners, twins)

    @property
    def size(self) -> int:
        return len(self.learners)

    def slot_bots(self) -> list[StockBot]:
        """Every bot still in a slot, learners in slot order first, then twins."""
        return [b for b in self.learners if b is not None] + [t for t in self.twins if t is not None]

    def living_learners(self) -> list[StockBot]:
        return [b for b in self.learners if b is not None and b.state != DEAD]

    def living_twins(self) -> list[StockTwin]:
        return [t for t in self.twins if t is not None and t.state != DEAD]

    def remove(self, bot: StockBot) -> None:
        """Empty the slot of a bot that died. No bot ever takes its place."""
        group = self.twins if isinstance(bot, StockTwin) else self.learners
        if group[bot.slot] is not bot:
            raise ValueError(f"bot {bot.bot_id} is not in slot {bot.slot}")
        group[bot.slot] = None


def birth_differences(a: StockPopulation, b: StockPopulation) -> list[str]:
    """Where two populations differ at birth, as S1 and S2 must not, comparing every slot's ids, home symbols, weights,
    optimizer state, generator state, forced timer delay and state, empty when they are born identical."""
    if a.size != b.size:
        return [f"sizes differ, {a.size} and {b.size}"]
    problems = []
    for x, y in zip(a.learners + a.twins, b.learners + b.twins):
        if x is None or y is None:
            if (x is None) != (y is None):
                problems.append("one slot is empty in only one population")
            continue
        name = f"bot {x.bot_id}"
        if (x.bot_id, x.slot, x.symbol, x.state, x.head_start) != (y.bot_id, y.slot, y.symbol, y.state, y.head_start):
            problems.append(f"{name} differs in id, slot, home symbol, state or forced timer delay")
        if x.model.params.shape != y.model.params.shape or not np.array_equal(x.model.params, y.model.params):
            problems.append(f"{name} has different weights")
        if not (np.array_equal(x.opt.m, y.opt.m) and np.array_equal(x.opt.v, y.opt.v) and x.opt.t == y.opt.t):
            problems.append(f"{name} has a different optimizer state")
        if x.rng.bit_generator.state != y.rng.bit_generator.state:
            problems.append(f"{name} has a different generator state")
    return problems
