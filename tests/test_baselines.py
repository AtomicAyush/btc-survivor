import numpy as np
import pytest

from btc_survivor.baselines import LinearBaseline


def test_recovers_a_linear_rule_from_noisy_samples():
    rng = np.random.default_rng(0)
    model = LinearBaseline(3, half_life_samples=1e9)
    for _ in range(20_000):
        z = rng.normal(size=3)
        model.update(z, 2.0 + 5.0 * z[0] - 3.0 * z[2] + rng.normal(0, 10))
    assert model.coef == pytest.approx([2.0, 5.0, 0.0, -3.0], abs=0.3)
    assert model.predict(np.array([1.0, 0.0, 1.0])) == pytest.approx(4.0, abs=0.6)


def test_early_fits_shrink_toward_zero_and_later_samples_forget_the_old_rule():
    model = LinearBaseline(1, half_life_samples=5000)
    model.update(np.array([0.0]), 7.0)
    assert 0 < model.predict(np.array([0.0])) < 0.01  # one sample against a prior worth a thousand
    rng = np.random.default_rng(1)
    for sign in (1.0, -1.0):
        for _ in range(40_000):
            z = rng.normal(size=1)
            model.update(z, sign * 4.0 * z[0])
    assert -4.0 < model.coef[1] < -3.3  # the old +4 is forgotten, with mild shrinkage from the prior


def test_untrained_baseline_predicts_zero():
    assert LinearBaseline(11, 100).predict(np.ones(11)) == 0.0
