import numpy as np
import pytest

from btc_survivor.skill import RollingSkill


def test_stats_match_numpy_over_the_window():
    rng = np.random.default_rng(1)
    preds, labels = rng.normal(size=500), rng.normal(size=500)
    skill = RollingSkill(360)
    for p, y in zip(preds, labels):
        skill.add(p, y)
    p, y = preds[-360:], labels[-360:]
    s = skill.stats()
    assert s.n == 360 and skill.full
    assert s.corr == pytest.approx(np.corrcoef(p, y)[0, 1])
    assert s.dir_acc == pytest.approx(np.mean(np.sign(p) == np.sign(y)))
    assert s.mae == pytest.approx(np.mean(np.abs(p - y)))


def test_correlation_needs_enough_pairs_and_variance():
    skill = RollingSkill(360)
    assert skill.stats().n == 0 and skill.stats().corr is None
    for i in range(29):
        skill.add(float(i), float(i))
    assert skill.stats().corr is None
    skill.add(29.0, 29.0)
    assert skill.stats().corr == pytest.approx(1.0)
    flat = RollingSkill(360)
    for i in range(40):
        flat.add(0.0, float(i))
    assert flat.stats().corr is None and flat.stats().dir_acc is None


def test_direction_ignores_zero_pairs():
    skill = RollingSkill(10)
    for p, y in [(1, 1), (-1, 1), (0, 5), (2, 0)]:
        skill.add(p, y)
    assert skill.stats().dir_acc == 0.5
