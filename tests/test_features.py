import math

import numpy as np
import pytest

from btc_survivor.clock import minute_of_day
from btc_survivor.config import load_config
from btc_survivor.features import N_FEATURES, FeatureBuilder
from btc_survivor.feed.book import LocalBook
from btc_survivor.feed.candles import MinuteSeries
from helpers import EXAMPLE_CONFIG

CFG = load_config(EXAMPLE_CONFIG)
M = 1_791_000_000_000
MIN = 60_000


def builder():
    return FeatureBuilder(CFG.features, CFG.time.history_minutes_required)


def book():
    b = LocalBook(20)
    b.apply_snapshot(bids=[(99.99, 2.0), (99.98, 1.0)], asks=[(100.01, 1.0), (100.02, 0.5)])
    return b


def build(mids, volumes=None, valid=None, capacity=1000):
    s = MinuteSeries(capacity)
    for i, mid in enumerate(mids):
        ok = True if valid is None else bool(valid[i])
        s.append(M + i * MIN, mid, 1.0 if volumes is None else volumes[i], ok)
    return s


def test_not_valid_until_enough_healthy_history():
    mids = [100.0] * 360
    assert builder().compute(build(mids[:359]), M + 358 * MIN, book()) is None
    assert builder().compute(build(mids), M + 359 * MIN, book()) is not None


def test_hand_computed_values():
    mids = [100.0 * math.exp(1e-4 * i) for i in range(400)]  # one bp per minute, exactly
    series = build(mids, volumes=[1.0] * 385 + [3.0] * 15)
    t = M + 399 * MIN
    f = builder().compute(series, t, book())
    assert f.shape == (N_FEATURES,)
    assert f[:4] == pytest.approx([5.0, 15.0, 30.0, 60.0])
    assert f[4:6] == pytest.approx([0.0, 0.0], abs=1e-9)
    baseline = (360 - 15 + 3.0 * 15) / 360 * 15
    assert f[6] == pytest.approx(math.log((45.0 + 1e-8) / (baseline + 1e-8)))
    assert f[7] == pytest.approx((3.0 - 1.5) / 4.5)
    assert f[8] == pytest.approx(0.02 / 100.0 * 1e4)
    angle = 2 * math.pi * minute_of_day(t) / 1440
    assert f[9:] == pytest.approx([math.sin(angle), math.cos(angle)])


def test_volatility_is_the_std_of_one_minute_returns_in_bps():
    steps = [1e-4 if i % 2 else -1e-4 for i in range(400)]
    mids = list(100.0 * np.exp(np.cumsum(steps)))
    f = builder().compute(build(mids), M + 399 * MIN, book())
    assert f[4:6] == pytest.approx([1.0, 1.0])


def test_return_needs_a_valid_mid_at_exactly_its_start_minute():
    mids = [100.0] * 400
    valid = [True] * 400
    valid[399 - 5] = False
    assert builder().compute(build(mids, valid=valid), M + 399 * MIN, book()) is None


def test_volatility_ignores_returns_that_span_a_missing_minute():
    rng = np.random.default_rng(5)
    mids = list(100.0 * np.exp(np.cumsum(rng.normal(0, 1e-4, 400))))
    valid = [True] * 400
    for i in (380, 381):
        valid[i] = False
    full = builder().compute(build(mids, valid=valid), M + 399 * MIN, book())
    window = np.array(mids[-31:])
    ok = np.array(valid[-31:])
    pairs = ok[1:] & ok[:-1]
    expected = np.std(1e4 * np.log(window[1:][pairs] / window[:-1][pairs]))
    assert full[4] == pytest.approx(expected)


def test_coverage_rule_rejects_windows_with_too_many_missing_minutes():
    mids = [100.0] * 400
    valid = [True] * 400
    for i in range(399 - 100, 399 - 60):  # 40 of the last 120 minutes missing, under 80 percent coverage
        valid[i] = False
    assert builder().compute(build(mids, valid=valid), M + 399 * MIN, book()) is None


def test_volume_ratio_is_clipped():
    mids = [100.0] * 400
    f = builder().compute(build(mids, volumes=[1.0] * 385 + [0.0] * 15), M + 399 * MIN, book())
    assert f[6] == -3.0


def test_invalid_book_gives_no_vector():
    series = build([100.0] * 400)
    assert builder().compute(series, M + 399 * MIN, LocalBook(20)) is None


def test_features_never_use_future_data():
    """Features at t from the full series equal features from the series truncated at t."""
    rng = np.random.default_rng(11)
    n = 700
    mids = list(100.0 * np.exp(np.cumsum(rng.normal(0, 6e-4, n))))
    volumes = list(rng.uniform(0.1, 3.0, n))
    valid = list(rng.random(n) > 0.03)
    full = build(mids, volumes, valid)
    checked = 0
    for t in range(360, n):
        truncated = build(mids[:t + 1], volumes[:t + 1], valid[:t + 1])
        a = builder().compute(full, M + t * MIN, book())
        b = builder().compute(truncated, M + t * MIN, book())
        assert (a is None) == (b is None), t
        if a is not None:
            assert np.array_equal(a, b), t
            checked += 1
    assert checked > 100


def test_unhealthy_minutes_do_not_count_toward_the_history_requirement():
    mids = [100.0] * 370
    valid = [True] * 370
    for i in range(100, 110):
        valid[i] = False
    series = build(mids, valid=valid)
    assert builder().compute(series, M + 359 * MIN, book()) is None  # only 350 healthy minutes so far
    assert builder().compute(series, M + 369 * MIN, book()) is not None


def test_imbalance_uses_the_top_ten_levels_only():
    deep = LocalBook(20)
    bids = [(99.99 - i * 0.01, 1.0 + i) for i in range(10)] + [(99.80, 100.0), (99.79, 100.0)]
    asks = [(100.01 + i * 0.01, 2.0) for i in range(12)]
    deep.apply_snapshot(bids=bids, asks=asks)
    f = builder().compute(build([100.0] * 400), M + 399 * MIN, deep)
    top_bids, top_asks = sum(1.0 + i for i in range(10)), 20.0
    assert f[7] == pytest.approx((top_bids - top_asks) / (top_bids + top_asks))
    assert f[7] != pytest.approx((1.0 - 2.0) / 3.0)


def test_volume_coverage_rules_apply_to_both_windows():
    n = 800
    end = n - 1
    mids = [100.0] * n
    recent_gap = [True] * n
    for i in range(end - 4, end):
        recent_gap[i] = False  # 4 of the last 15 minutes missing
    assert builder().compute(build(mids, valid=recent_gap), M + end * MIN, book()) is None
    baseline_gap = [True] * n
    for i in range(end - 300, end - 200):
        baseline_gap[i] = False  # 100 of the last 360 minutes missing
    assert builder().compute(build(mids, valid=baseline_gap), M + end * MIN, book()) is None


def test_volume_terms_are_scaled_from_valid_minutes_only():
    n = 800
    end = n - 1
    valid = [True] * n
    for i in list(range(end - 300, end - 250)) + [end - 3, end - 2]:
        valid[i] = False
    f = builder().compute(build([100.0] * n, volumes=[1.0] * n, valid=valid), M + end * MIN, book())
    assert f[6] == pytest.approx(0.0, abs=1e-9)  # one unit per valid minute in both windows
