"""Every random generator comes from the one master seed, through a fixed tag per purpose."""

from __future__ import annotations

import numpy as np

BOT = 1
POPULATION = 2
RECONNECT = 3


def bot_rng(master_seed: int, bot_id: int) -> np.random.Generator:
    """A bot's own generator for its initial weights, mutation noise, replay batches and latency jitter."""
    return np.random.default_rng(np.random.SeedSequence([master_seed, BOT, bot_id]))


def population_rng(master_seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([master_seed, POPULATION]))


def reconnect_rng(master_seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([master_seed, RECONNECT]))
