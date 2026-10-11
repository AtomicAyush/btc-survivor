import math

import numpy as np

from btc_survivor.labels import LabelQueue, Matured, MaturedWindow

M = 1_791_000_000_000
MIN = 60_000


def z(v):
    return np.full(11, float(v))


def test_label_matures_exactly_at_the_horizon_and_not_before():
    q = LabelQueue(horizon_minutes=30)
    q.push(M, 0, z(1), 100.0)
    for minute in range(1, 30):
        assert q.mature(M + minute * MIN, 0, 101.0) is None
    sample = q.mature(M + 30 * MIN, 0, 101.0)
    assert sample.minute_ms == M and sample.label_bps == 1e4 * math.log(101.0 / 100.0)
    assert np.array_equal(sample.z, z(1))


def test_label_never_crosses_a_feed_gap():
    q = LabelQueue(horizon_minutes=30)
    q.push(M, 0, z(1), 100.0)
    q.push(M + MIN, 0, z(2), 100.0)
    assert q.mature(M + 30 * MIN, 1, 101.0) is None  # the epoch moved since the record was made
    q.drop_other_epochs(1)
    assert not q.pending and q.discarded == 2


def test_skipped_minutes_never_mature_against_a_later_mid():
    q = LabelQueue(horizon_minutes=30)
    q.push(M, 0, z(1), 100.0)
    q.push(M + MIN, 0, z(2), 100.0)
    sample = q.mature(M + 31 * MIN, 0, 105.0)  # the boundary at M + 30 minutes was never processed
    assert sample.minute_ms == M + MIN and q.discarded == 1


def test_window_keeps_the_last_capacity_samples_and_batches_start_with_the_newest():
    w = MaturedWindow(capacity=4, n_features=11)
    for i in range(6):
        w.add(Matured(i, z(i), float(i)))
    assert w.count == 4 and sorted(w.label.tolist()) == [2.0, 3.0, 4.0, 5.0]
    X, y = w.batch(np.random.default_rng(1), 8)
    assert X.shape == (8, 11) and set(y.tolist()) <= {2.0, 3.0, 4.0, 5.0}


def test_batches_start_with_the_newest_and_only_draw_filled_rows():
    w = MaturedWindow(capacity=240, n_features=11)
    for i in range(1, 7):
        w.add(Matured(i, z(i), float(i)))
    for seed in range(20):
        for size in (8, 1):
            _, y = w.batch(np.random.default_rng(seed), size)
            assert y[0] == 6.0 and len(y) == size
            assert set(y.tolist()) <= {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
