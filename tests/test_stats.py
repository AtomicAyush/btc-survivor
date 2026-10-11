import math

import numpy as np
import pytest

from btc_survivor.cli.stats import block_test, diebold_mariano, newey_west_se, skill_test


def test_newey_west_without_lags_is_the_classical_standard_error_and_grows_with_autocorrelation():
    rng = np.random.default_rng(0)
    u = rng.normal(size=5000)
    assert newey_west_se(u, 0) == pytest.approx(u.std() / math.sqrt(len(u)))
    ar = np.zeros(5000)
    for t in range(1, 5000):
        ar[t] = 0.8 * ar[t - 1] + rng.normal()
    assert newey_west_se(ar, 30) > 2 * newey_west_se(ar, 0)


def test_skill_test_finds_a_real_correlation_and_rarely_passes_on_noise():
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    found = skill_test(x, 0.2 * x + rng.normal(size=2000), 12)
    assert found["passed"] and found["corr"] == pytest.approx(0.2, abs=0.05)
    passes = sum(skill_test(rng.normal(size=1000), rng.normal(size=1000), 12)["passed"] for _ in range(400))
    assert passes / 400 < 0.04  # nominal one sided 1 percent
    assert skill_test(np.zeros(10), rng.normal(size=10), 12)["corr"] is None


def test_diebold_mariano_favours_the_forecast_with_smaller_errors():
    rng = np.random.default_rng(2)
    label = rng.normal(size=3000)
    good, bad = label - (label * 0.5), label - rng.normal(size=3000)
    result = diebold_mariano(good, bad, 12)
    assert result["t"] > 5 and result["mse_ratio"] < 1


def test_t_quantile_matches_student_t_and_tends_to_the_normal():
    from btc_survivor.cli.stats import t_quantile_99

    assert t_quantile_99(54) == pytest.approx(2.397, abs=0.003)
    assert t_quantile_99(10) == pytest.approx(2.764, abs=0.02)
    assert t_quantile_99(100_000) == pytest.approx(2.326, abs=0.001)


def test_block_test_uses_one_mean_per_block():
    result = block_test({0: [1.0, 3.0], 1: [2.0], 2: [4.0, 4.0, 4.0]})
    means = np.array([2.0, 2.0, 4.0])
    assert result["blocks"] == 3 and result["pairs"] == 6 and result["mean"] == pytest.approx(means.mean())
    assert result["t"] == pytest.approx(means.mean() / (means.std(ddof=1) / math.sqrt(3)))
    assert result["threshold"] > 2.33  # the t quantile with 2 degrees of freedom, not the normal one
    assert not block_test({0: [5.0, 6.0]})["passed"]


def test_t_quantile_uses_the_table_at_few_degrees_of_freedom():
    from btc_survivor.cli.stats import t_quantile_99

    assert [t_quantile_99(df) for df in (1, 2, 3)] == pytest.approx([31.821, 6.965, 4.541], abs=0.01)
    assert t_quantile_99(9) > t_quantile_99(10) > t_quantile_99(11)
