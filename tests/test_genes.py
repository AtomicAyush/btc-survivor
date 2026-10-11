import numpy as np
import pytest

from btc_survivor.config import load_config
from btc_survivor.genes import Genes, initial_genes, mutate
from helpers import EXAMPLE_CONFIG

CFG = load_config(EXAMPLE_CONFIG).genes


def test_initial_genes_come_from_the_config():
    assert initial_genes(CFG) == Genes(8.0, 0.001)


def test_mutated_genes_always_stay_within_bounds():
    rng = np.random.default_rng(0)
    parent = Genes(8.0, 0.001)
    for _ in range(100_000):
        child, _, _ = mutate(parent, CFG, sigma=3.0, rng=rng)  # an extreme sigma that often hits both bounds
        assert CFG.margin_bps_min <= child.margin_bps <= CFG.margin_bps_max
        assert CFG.lr_min <= child.lr <= CFG.lr_max


def test_mutation_factors_are_lognormal_with_the_configured_sigma():
    rng = np.random.default_rng(1)
    logs = np.array([np.log(mutate(Genes(8.0, 0.001), CFG, 0.2, rng)[1:]) for _ in range(20_000)])
    assert np.allclose(logs.mean(axis=0), 0.0, atol=0.01)
    assert np.allclose(logs.std(axis=0), 0.2, atol=0.01)


def test_child_is_parent_times_factor_before_clamping():
    child, margin_factor, lr_factor = mutate(Genes(8.0, 0.001), CFG, 0.2, np.random.default_rng(5))
    assert child.margin_bps == pytest.approx(min(max(8.0 * margin_factor, 2.0), 60.0))
    assert child.lr == pytest.approx(min(max(0.001 * lr_factor, 0.0001), 0.01))


def test_mutation_is_deterministic_for_a_seed():
    a = mutate(Genes(8.0, 0.001), CFG, 0.2, np.random.default_rng(9))
    b = mutate(Genes(8.0, 0.001), CFG, 0.2, np.random.default_rng(9))
    assert a == b
