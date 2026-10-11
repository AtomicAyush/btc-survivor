"""Keeps every slot filled. Dead learners are replaced by a random newcomer or a mutated copy of a fit survivor.

Bots only leave by dying, never for ranking. Coin flip twins sit beside learners in the same slots, are never
parents, and a dead twin is replaced by a fresh twin.
"""

from __future__ import annotations

import math

import numpy as np

from btc_survivor.bot import DEAD, Bot
from btc_survivor.config import Config
from btc_survivor.controls import TwinBot
from btc_survivor.features import N_FEATURES
from btc_survivor.genes import mutate
from btc_survivor.model import MLP
from btc_survivor.seeds import bot_rng, population_rng


def fitness(bot: Bot, floor: float) -> float:
    """Mean hourly equity change over its floored standard deviation, multiplied by it when the mean is negative.

    Using the mean keeps young and old bots comparable, and the negative branch stops volatile losers ranking higher.
    """
    samples = np.asarray(bot.hourly_equity, dtype=float)
    if len(samples) < 2:
        return -math.inf
    changes = np.diff(samples)
    mean = float(changes.mean())
    spread = max(float(changes.std()), floor)
    return mean / spread if mean >= 0 else mean * spread / floor


class Population:
    def __init__(self, cfg: Config, learners: list[Bot], twins: list[TwinBot | None], next_bot_id: int,
                 rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.learners = learners
        self.twins = twins
        self.next_bot_id = next_bot_id
        self.rng = rng

    @classmethod
    def fresh(cls, cfg: Config, born_ms: int) -> "Population":
        """Random learners with ids 1 to size in slot order, then their twins, all from the master seed."""
        size = cfg.population.size
        learners = [Bot.newborn(cfg, slot + 1, slot, born_ms) for slot in range(size)]
        twins: list[TwinBot | None] = [None] * size
        next_id = size + 1
        if cfg.controls.coin_flip_twins:
            twins = [TwinBot.newborn(cfg, size + 1 + slot, slot, born_ms) for slot in range(size)]
            next_id = 2 * size + 1
        return cls(cfg, learners, twins, next_id, population_rng(cfg.run.master_seed))

    def _new_id(self) -> int:
        bot_id = self.next_bot_id
        self.next_bot_id += 1
        return bot_id

    def eligible(self) -> list[Bot]:
        """Living learners old enough and experienced enough to be parents, in slot order."""
        pop = self.cfg.population
        return [b for b in self.learners if b.state != DEAD and b.alive_minutes >= pop.parent_min_age_hours * 60
                and b.account.trades >= pop.parent_min_trades]

    def select_parent(self) -> Bot | None:
        """Tournament among eligible bots. Ties on fitness break on rolling skill correlation."""
        eligible = self.eligible()
        if not eligible:
            return None
        k = min(self.cfg.population.tournament_size, len(eligible))
        picks = [eligible[i] for i in self.rng.choice(len(eligible), size=k, replace=False)]
        floor = self.cfg.population.fitness_std_floor_usd

        def key(bot: Bot) -> tuple[float, float]:
            corr = bot.skill.stats().corr
            return fitness(bot, floor), corr if corr is not None else -math.inf

        return max(picks, key=key)

    def replace(self, dead: Bot, born_ms: int) -> tuple[Bot, dict | None]:
        """Fill a dead learner's slot. Returns the new bot and, for a mutated child, its lineage row."""
        cfg = self.cfg
        newcomer = self.rng.random() < cfg.population.newcomer_fraction
        parent = None if newcomer else self.select_parent()
        bot_id = self._new_id()
        if parent is None:
            child = Bot.newborn(cfg, bot_id, dead.slot, born_ms)
            lineage = None
        else:
            child, lineage = self._mutated_child(parent, bot_id, dead.slot, born_ms)
        self.learners[dead.slot] = child
        return child, lineage

    def _mutated_child(self, parent: Bot, bot_id: int, slot: int, born_ms: int) -> tuple[Bot, dict]:
        cfg = self.cfg
        rng = bot_rng(cfg.run.master_seed, bot_id)
        sigma_w, sigma_g = cfg.population.mutation_sigma_weights, cfg.population.mutation_sigma_genes
        params = parent.model.params + rng.normal(0.0, sigma_w, parent.model.size)
        genes, margin_factor, lr_factor = mutate(parent.genes, cfg.genes, sigma_g, rng)
        model = MLP(N_FEATURES, cfg.model.hidden_units, params=params)
        child = Bot(cfg, bot_id, slot, parent.generation + 1, born_ms, genes, model, rng,
                    origin="mutated", parent_id=parent.bot_id)
        child.delay_forced_timer()
        lineage = {
            "bot_id": bot_id, "parent_id": parent.bot_id, "generation": child.generation, "born_ms": born_ms,
            "sigma_weights": sigma_w, "sigma_genes": sigma_g, "margin_factor": margin_factor,
            "lr_factor": lr_factor, "parent_margin_bps": parent.genes.margin_bps, "parent_lr": parent.genes.lr,
            "parent_fitness": fitness(parent, cfg.population.fitness_std_floor_usd),
            "parent_skill_corr": parent.skill.stats().corr,
        }
        return child, lineage

    def replace_twin(self, dead: TwinBot, born_ms: int) -> TwinBot:
        twin = TwinBot.newborn(self.cfg, self._new_id(), dead.slot, born_ms)
        self.twins[dead.slot] = twin
        return twin
