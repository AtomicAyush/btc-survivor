import numpy as np
import pytest

from btc_survivor.standardizer import LabelScale, RunningStats, Standardizer, alpha_for_half_life


def test_running_stats_match_plain_mean_and_variance_while_young():
    rng = np.random.default_rng(1)
    data = rng.normal(5.0, 2.0, size=(100, 3))
    stats = RunningStats(3, half_life_samples=10_000)
    for row in data:
        stats.update(row)
    assert np.allclose(stats.mean, data.mean(axis=0))
    assert np.allclose(stats.var, data.var(axis=0))


def test_running_stats_become_exponential_after_one_over_alpha_samples():
    stats = RunningStats(1, half_life_samples=10)
    for _ in range(1000):
        stats.update(np.array([0.0]))
    for _ in range(10):
        stats.update(np.array([1.0]))
    assert stats.mean[0] == pytest.approx(0.5, abs=1e-9)  # ten samples at a half life of ten


def test_alpha_half_life():
    a = alpha_for_half_life(168 * 60)
    assert (1 - a) ** (168 * 60) == pytest.approx(0.5)


def test_zero_until_min_samples_then_scaled_and_clipped():
    s = Standardizer(2, 1000, min_samples=30, clip=5.0)
    rng = np.random.default_rng(0)
    for i in range(30):
        assert np.all(s.transform_then_update(rng.normal(size=2)) == 0)
    z = s.transform_then_update(np.array([0.0, 1e6]))
    assert z[1] == 5.0 and abs(z[0]) < 5


def test_thirty_standard_normal_samples_give_roughly_unit_z_scores():
    rng = np.random.default_rng(3)
    s = Standardizer(1, 168 * 60, min_samples=30, clip=5.0)
    for _ in range(30):
        s.transform_then_update(rng.normal(size=1))
    z = np.array([s.transform_then_update(rng.normal(size=1))[0] for _ in range(2000)])
    assert 0.8 < z.std() < 1.2 and abs(z.mean()) < 0.15


def test_current_sample_does_not_shape_its_own_z_score():
    s = Standardizer(1, 1000, min_samples=3, clip=5.0)
    for v in (1.0, 2.0, 3.0):
        s.transform_then_update(np.array([v]))
    mean, std = s.stats.mean.copy(), s.stats.std.copy()
    z = s.transform_then_update(np.array([10.0]))
    assert z[0] == pytest.approx(min(5.0, (10.0 - mean[0]) / std[0]))


def test_constant_feature_gives_zero_not_nan_and_a_jump_is_clipped():
    s = Standardizer(1, 1000, min_samples=5, clip=5.0)
    for _ in range(50):
        z = s.transform_then_update(np.array([0.0012]))
    assert z[0] == 0.0
    assert s.transform_then_update(np.array([0.0024]))[0] == 5.0


def test_label_scale_needs_min_samples():
    scale = LabelScale(1000, min_samples=3)
    for v in (10.0, -10.0):
        scale.update(v)
    assert not scale.ready
    scale.update(10.0)
    assert scale.ready and scale.scale == pytest.approx(np.std([10.0, -10.0, 10.0]))
