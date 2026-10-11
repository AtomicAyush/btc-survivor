"""The synthetic learner check itself: its planted signal has the stated strength and the learner can find it."""

import sys

import numpy as np
import pytest

from btc_survivor.config import load_config
from helpers import EXAMPLE_CONFIG, ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from learner_check import forward_sums, run, synthetic  # noqa: E402

CFG = load_config(EXAMPLE_CONFIG)


@pytest.mark.parametrize("corr, persistence", [(0.1, 0.9), (0.3, 0.99)])
def test_planted_signal_has_the_requested_correlation(corr, persistence):
    _, labels, truth = synthetic(1, 200_000, 30, corr, persistence)
    assert np.corrcoef(truth, labels)[0, 1] == pytest.approx(corr, abs=0.02)


def test_labels_sum_only_the_moves_after_their_minute():
    moves = np.arange(100.0)
    labels = forward_sums(moves, 30, 50)
    assert [labels[t] for t in (0, 7, 49)] == [moves[t + 1:t + 31].sum() for t in (0, 7, 49)]


def test_default_learner_finds_a_strong_signal_and_stays_quiet_on_none():
    strong = run(CFG, 0, 4, 0.4, 0.99)
    assert strong.corr > 0.2 and strong.linear > 0.2
    null = run(CFG, 0, 4, 0.0, 0.99)
    assert abs(null.corr) < 0.1 and null.spread_bps < 5.0


def test_a_signal_the_features_cannot_carry_is_refused():
    with pytest.raises(ValueError, match="not reachable"):
        synthetic(1, 1000, 240, 0.2, 0.9)
