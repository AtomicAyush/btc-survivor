import dataclasses
import math
from collections import deque

import numpy as np
import pytest

from btc_survivor.bot import DEAD, WARMUP, Bot
from btc_survivor.config import load_config
from btc_survivor.controls import TwinBot
from btc_survivor.population import Population, fitness
from helpers import EXAMPLE_CONFIG

CFG = load_config(EXAMPLE_CONFIG)
M = 1_791_000_000_000
FLOOR = CFG.population.fitness_std_floor_usd


def with_population(**changes):
    return dataclasses.replace(CFG, population=dataclasses.replace(CFG.population, **changes))


def make_eligible(bot, trades=20, hours=24, equity=None):
    bot.alive_minutes = hours * 60
    bot.account.trades = trades
    if equity is not None:
        bot.hourly_equity = deque(equity, maxlen=CFG.population.fitness_window_hours + 1)


def test_fresh_population_fills_every_slot_with_seeded_learners_and_twins():
    pop = Population.fresh(CFG, M)
    assert [b.slot for b in pop.learners] == list(range(20)) and [b.bot_id for b in pop.learners] == list(range(1, 21))
    assert [t.bot_id for t in pop.twins] == list(range(21, 41)) and pop.next_bot_id == 41
    assert all(b.state == WARMUP and b.origin == "random" for b in pop.learners)
    assert all(isinstance(t, TwinBot) and t.origin == "control" for t in pop.twins)
    again = Population.fresh(CFG, M)
    assert all(np.array_equal(a.model.params, b.model.params) for a, b in zip(pop.learners, again.learners))


def test_runs_with_different_money_start_from_identical_bots():
    small = dataclasses.replace(CFG, trading=dataclasses.replace(CFG.trading, start_equity=1000.0))
    a, b = Population.fresh(CFG, M), Population.fresh(small, M)
    assert all(np.array_equal(x.model.params, y.model.params) for x, y in zip(a.learners, b.learners))
    assert b.learners[0].account.cash == 1000.0


def test_twins_can_be_switched_off():
    cfg = dataclasses.replace(CFG, controls=dataclasses.replace(CFG.controls, coin_flip_twins=False))
    pop = Population.fresh(cfg, M)
    assert pop.twins == [None] * 20 and pop.next_bot_id == 21


def test_fitness_uses_the_mean_hourly_change_over_its_spread():
    bot = Bot.newborn(CFG, 1, 0, M)
    bot.hourly_equity = deque([100.0, 104.0, 102.0, 106.0])
    changes = np.array([4.0, -2.0, 4.0])
    assert fitness(bot, FLOOR) == pytest.approx(changes.mean() / changes.std())


def test_fitness_does_not_favour_younger_bots_with_the_same_behaviour():
    young, old = Bot.newborn(CFG, 1, 0, M), Bot.newborn(CFG, 2, 1, M)
    young.hourly_equity = deque(10_000 - np.arange(25) * 1.0)
    old.hourly_equity = deque(10_000 - np.arange(169) * 1.0)
    assert fitness(young, FLOOR) == pytest.approx(fitness(old, FLOOR))


def test_volatility_counts_against_losing_bots_too():
    steady, volatile = Bot.newborn(CFG, 1, 0, M), Bot.newborn(CFG, 2, 1, M)
    steady.hourly_equity = deque(10_000 - np.arange(25) * 1.0)
    volatile.hourly_equity = deque(10_000 - np.arange(25) * 1.0 + np.tile([0.0, 20.0], 13)[:25])
    assert fitness(steady, FLOOR) > fitness(volatile, FLOOR)


def test_fitness_needs_two_samples_and_floors_the_spread():
    bot = Bot.newborn(CFG, 1, 0, M)
    assert fitness(bot, FLOOR) == -math.inf
    bot.hourly_equity = deque([100.0, 100.5, 101.0])
    assert fitness(bot, FLOOR) == pytest.approx(0.5 / FLOOR)


def test_eligibility_needs_age_and_closed_trades_and_life():
    pop = Population.fresh(CFG, M)
    a, b, c, d = pop.learners[:4]
    make_eligible(a)
    make_eligible(b, trades=19)
    make_eligible(c, hours=23)
    make_eligible(d)
    d.state = DEAD
    assert pop.eligible() == [a]


def test_no_eligible_parent_means_a_random_newcomer():
    pop = Population.fresh(CFG, M)
    dead = pop.learners[5]
    child, lineage = pop.replace(dead, M + 1)
    assert lineage is None and child.origin == "random" and child.slot == 5 and child.generation == 0
    assert child.bot_id == 41 and pop.learners[5] is child


def test_tournament_picks_the_fittest_of_its_sample():
    cfg = with_population(tournament_size=20)
    pop = Population.fresh(cfg, M)
    for i, bot in enumerate(pop.learners):
        make_eligible(bot, equity=[10_000.0 + i * h for h in range(25)])
    assert pop.select_parent() is pop.learners[-1]


def test_single_eligible_bot_is_the_parent_and_ties_break_on_skill():
    pop = Population.fresh(CFG, M)
    only = pop.learners[3]
    make_eligible(only, equity=[10_000.0] * 25)
    assert pop.select_parent() is only
    cfg = with_population(tournament_size=20)
    for better in (0, 1):  # whichever bot the generator samples first, skill must decide the tie
        pop = Population.fresh(cfg, M)
        pair = pop.learners[0], pop.learners[1]
        for bot in pair:
            make_eligible(bot, equity=[10_000.0] * 25)
            for i in range(40):
                bot.skill.add(float(i), float(i) if bot is pair[better] else float(i % 3))
        assert pop.select_parent() is pair[better]


def test_tournament_only_sees_its_sample():
    from btc_survivor.seeds import population_rng

    pop = Population.fresh(CFG, M)  # tournament size 3
    picks = population_rng(CFG.run.master_seed).choice(20, size=3, replace=False)
    best_overall = next(i for i in range(19, -1, -1) if i not in picks)
    for i, bot in enumerate(pop.learners):
        make_eligible(bot, equity=[10_000.0 + (100 if i == best_overall else i) * h for h in range(25)])
    winner = pop.select_parent()
    assert winner is pop.learners[max(picks)] and winner is not pop.learners[best_overall]


def test_newcomer_fraction_is_respected():
    pop = Population.fresh(CFG, M)
    parent = pop.learners[0]
    make_eligible(parent, equity=[10_000.0] * 25)
    origins = []
    for _ in range(4000):
        child, _ = pop.replace(pop.learners[1], M)
        origins.append(child.origin)
    share = origins.count("random") / len(origins)
    assert 0.08 < share < 0.12


def test_mutated_child_copies_the_parent_with_noise_and_starts_fresh():
    cfg = with_population(newcomer_fraction=0.0)
    pop = Population.fresh(cfg, M)
    parent = pop.learners[2]
    make_eligible(parent, equity=[10_000.0 + h for h in range(25)])
    parent.generation = 3
    parent.opt.t = 50
    child, lineage = pop.replace(pop.learners[7], M + 5)
    noise = child.model.params - parent.model.params
    assert child.origin == "mutated" and child.parent_id == parent.bot_id and child.generation == 4
    assert child.slot == 7 and child.bot_id == 41
    assert 0.04 < noise.std() < 0.06 and abs(noise.mean()) < 0.01
    assert child.opt.t == 0 and not child.opt.m.any()
    assert child.account.cash == cfg.trading.start_equity and child.account.trades == 0
    assert child.state == WARMUP and child.warmup_minutes == 60
    assert -cfg.trading.force_trade_after_hours * 60 < child.minutes_since_open < 0
    assert lineage["bot_id"] == 41 and lineage["parent_id"] == parent.bot_id and lineage["generation"] == 4
    assert lineage["born_ms"] == M + 5 and lineage["sigma_weights"] == 0.05 and lineage["sigma_genes"] == 0.20
    assert child.genes.margin_bps == pytest.approx(min(max(parent.genes.margin_bps * lineage["margin_factor"], 2.0), 60.0))
    assert child.genes.lr == pytest.approx(min(max(parent.genes.lr * lineage["lr_factor"], cfg.genes.lr_min),
                                               cfg.genes.lr_max))
    assert lineage["parent_fitness"] == pytest.approx(fitness(parent, FLOOR))
    assert lineage["parent_margin_bps"] == parent.genes.margin_bps and lineage["parent_lr"] == parent.genes.lr
    assert lineage["parent_skill_corr"] == parent.skill.stats().corr
    from btc_survivor.seeds import bot_rng
    assert np.allclose(noise, bot_rng(cfg.run.master_seed, 41).normal(0.0, 0.05, parent.model.size))


def test_twins_are_never_parents_and_are_replaced_by_fresh_twins():
    pop = Population.fresh(CFG, M)
    for twin in pop.twins:
        make_eligible(twin, equity=[10_000.0 + 100 * h for h in range(25)])
    assert pop.select_parent() is None
    old = pop.twins[4]
    old.account.charge_rent(old.account.cash - 3.0)
    old.state = DEAD
    new = pop.replace_twin(old, M + 9)
    assert isinstance(new, TwinBot) and new.slot == 4 and new.bot_id == 41 and pop.twins[4] is new
    assert new.account is not old.account and new.account.cash == CFG.trading.start_equity and new.account.trades == 0


def test_replacement_is_deterministic_for_a_seed():
    def run():
        cfg = with_population(newcomer_fraction=0.5)
        pop = Population.fresh(cfg, M)
        for bot in pop.learners[:5]:
            make_eligible(bot, equity=[10_000.0 + bot.bot_id * h for h in range(25)])
        return [(c.origin, c.parent_id) for c, _ in (pop.replace(pop.learners[10], M) for _ in range(30))]
    assert run() == run()
