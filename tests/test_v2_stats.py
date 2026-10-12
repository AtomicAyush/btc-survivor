"""Pooled skill, pooled_dm and session blocks of PLAN_V2 sections 11.2 and 11.3, on hand examples and seeded
synthetic rows with a market factor."""

import math

import numpy as np
import pytest

from btc_survivor.cli.stats import block_test, newey_west_se, skill_test, t_quantile_99
from btc_survivor.v2 import stats
from btc_survivor.v2.stats import matrix_skill, pooled_dm, pooled_skill, session_block_test, symbol_slopes

SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "QQQ"]
MARKET_SHARE = np.array([0.45] * 7 + [0.90])  # the design of section 11.2


def rows_of(minutes, symbols, pred, label):
    """Flatten minute by symbol matrices into rows, leaving out NaN cells."""
    m, s, p, y = [], [], [], []
    for i, minute in enumerate(minutes):
        for j, symbol in enumerate(symbols):
            if not np.isnan(pred[i, j]):
                m.append(minute), s.append(symbol), p.append(pred[i, j]), y.append(label[i, j])
    return m, s, p, y


def market_draws(rng, runs, minutes, c, r=0.0):
    """Gaussian predictions with pairwise correlation c and labels with the market factor of section 11.2, with
    true correlation r between each symbol's prediction and label."""
    pred = math.sqrt(c) * rng.normal(size=(runs, minutes, 1)) + math.sqrt(1 - c) * rng.normal(size=(runs, minutes, 8))
    noise = (np.sqrt(MARKET_SHARE) * rng.normal(size=(runs, minutes, 1))
             + np.sqrt(1 - MARKET_SHARE) * rng.normal(size=(runs, minutes, 8)))
    return pred, r * pred + math.sqrt(1 - r * r) * noise


def test_balanced_pooled_correlation_is_the_mean_within_symbol_correlation():
    rng = np.random.default_rng(3)
    pred, label = market_draws(rng, 1, 120, 0.5, 0.3)
    pred, label = pred[0] * np.arange(1, 9) + 5.0, label[0] * 30.0 - 2.0  # scale and shift never matter
    result = pooled_skill(*rows_of(range(120), SYMBOLS, pred, label))
    within = np.mean([np.corrcoef(pred[:, j], label[:, j])[0, 1] for j in range(8)])
    assert result.corr == pytest.approx(within, abs=1e-12)
    assert result.rows == 960 and result.minutes == 120 and result.left_out == ()
    assert result.symbols == tuple(sorted(SYMBOLS))


def test_a_symbol_whose_prediction_never_varies_is_left_out_and_named():
    rng = np.random.default_rng(4)
    pred, label = market_draws(rng, 1, 60, 0.5, 0.3)
    pred, label = pred[0], label[0]
    pred[:, 2] = 0.25  # NVDA, a constant that rounding would turn into noise if compared inexactly
    result = pooled_skill(*rows_of(range(60), SYMBOLS, pred, label))
    others = [j for j in range(8) if j != 2]
    assert result.left_out == ("NVDA",) and "NVDA" not in result.symbols and result.rows == 420
    assert result.corr == pytest.approx(np.mean([np.corrcoef(pred[:, j], label[:, j])[0, 1] for j in others]))


def test_hand_computed_driscoll_kraay_error():
    # Each column already has mean 0 and population standard deviation 1, so the z scores are the values.
    minute = [1, 2, 3, 4] * 2
    symbol = ["AAPL"] * 4 + ["MSFT"] * 4
    pred = [1, -1, 1, -1] + [1, 1, -1, -1]
    label = [1, 1, -1, -1] + [1, 1, -1, -1]
    # Products are AAPL 1, -1, -1, 1 and MSFT 1, 1, 1, 1, so the minute means are 1, 0, 0, 1 with mean 0.5.
    # Deviations 0.5, -0.5, -0.5, 0.5 give gamma0 = 1 / 4 and gamma1 = -0.25 / 4 = -0.0625 over n = 4,
    # so with one lag s = 0.25 + 2 * (1 - 1 / 2) * (-0.0625) = 0.1875 and se = sqrt(0.1875 / 4).
    se = math.sqrt(0.1875 / 4)
    result = pooled_skill(minute, symbol, pred, label, lags=1)
    assert result.corr == pytest.approx(0.5) and result.se == pytest.approx(se)
    assert result.t == pytest.approx(0.5 / se) and result.t == pytest.approx(2.3094, abs=1e-4)
    assert not result.passed  # t is just under 2.33
    order = [5, 2, 7, 0, 3, 6, 1, 4]  # row order never matters
    shuffled = pooled_skill([minute[i] for i in order], [symbol[i] for i in order], [pred[i] for i in order],
                            [label[i] for i in order], lags=1)
    assert shuffled == result


def test_a_missing_row_drops_out_of_its_minute_mean():
    minute = [1, 2, 3, 4, 1, 2, 4]
    symbol = ["AAPL"] * 4 + ["MSFT"] * 3
    pred = [1.0, 2.0, 3.0, 4.0, 1.0, 3.0, 2.0]
    label = [2.0, 1.0, 4.0, 3.0, 1.0, 2.0, 3.0]
    za = (np.array(pred[:4]) - 2.5) / np.std(pred[:4]) * (np.array(label[:4]) - 2.5) / np.std(label[:4])
    zm = (np.array(pred[4:]) - 2.0) / np.std(pred[4:]) * (np.array(label[4:]) - 2.0) / np.std(label[4:])
    means = np.array([(za[0] + zm[0]) / 2, (za[1] + zm[1]) / 2, za[2], (za[3] + zm[2]) / 2])
    result = pooled_skill(minute, symbol, pred, label, lags=2)
    assert result.minutes == 4 and result.rows == 7
    assert result.corr == pytest.approx(means.mean()) and result.se == pytest.approx(newey_west_se(means, 2))


def test_eight_copies_of_one_symbol_give_its_own_t():
    rng = np.random.default_rng(5)
    pred = rng.normal(size=468)
    label = 0.15 * pred + rng.normal(size=468)
    single = skill_test(pred, label, 12)
    result = matrix_skill(np.tile(pred[:, None], 8), np.tile(label[:, None], 8))
    assert result.corr == pytest.approx(single["corr"]) and result.t == pytest.approx(single["t"])
    assert result.passed == single["passed"]


def test_a_fast_null_simulation_rejects_near_one_percent_and_pooling_without_clustering_does_not():
    rng = np.random.default_rng(6)
    pred, label = market_draws(rng, 1000, 468, 0.5)
    clustered, unclustered = 0, 0
    for p, y in zip(pred, label):
        clustered += matrix_skill(p, y).passed
        u = ((p - p.mean(0)) / p.std(0) * (y - y.mean(0)) / y.std(0)).ravel()
        unclustered += u.mean() / newey_west_se(u, 0) > 2.33
    assert 0.003 <= clustered / 1000 <= 0.02
    assert unclustered / 1000 > 0.05  # section 11.2 found 7.8 percent


def test_the_upper_bound_and_detectable_correlation():
    result = pooled_skill([1, 2, 3, 4] * 2, ["AAPL"] * 4 + ["MSFT"] * 4, [1, -1, 1, -1, 1, 1, -1, -1],
                          [1, 1, -1, -1, 1, 1, -1, -1], lags=1)
    assert result.upper_bound == pytest.approx(0.5 + 1.645 * result.se)
    assert result.detectable == pytest.approx((2.33 + 1.28) * result.se)
    wider = pooled_skill([1, 2, 3, 4] * 2, ["AAPL"] * 4 + ["MSFT"] * 4, [1, -1, 1, -1, 1, 1, -1, -1],
                         [1, 1, -1, -1, 1, 1, -1, -1], lags=1, one_sided_z=3.0, upper_bound_z=2.0)
    assert wider.upper_bound == pytest.approx(0.5 + 2.0 * result.se)
    assert wider.detectable == pytest.approx(4.28 * result.se)


def test_the_upper_bound_covers_a_planted_correlation_in_about_95_percent_of_runs():
    rng = np.random.default_rng(7)
    pred, label = market_draws(rng, 1000, 468, 0.5, 0.1)
    results = [matrix_skill(p, y) for p, y in zip(pred, label)]
    covered = np.mean([r.upper_bound >= 0.1 for r in results])
    assert 0.925 <= covered <= 0.975
    assert np.mean([r.corr for r in results]) == pytest.approx(0.1, abs=0.005)


def test_too_few_minutes_give_no_statistic_and_never_pass():
    result = pooled_skill([1, 2, 1, 2], ["AAPL", "AAPL", "MSFT", "MSFT"], [1, 2, 3, 4], [2, 1, 4, 3])
    assert result.minutes == 2 and result.corr is None and result.t is None and not result.passed
    assert result.upper_bound is None and result.detectable is None
    empty = pooled_skill([], [], [], [])
    assert empty.rows == 0 and empty.corr is None and not empty.passed


def test_duplicate_rows_and_values_that_are_not_finite_are_refused():
    with pytest.raises(ValueError, match="more than one row"):
        pooled_skill([1, 1, 2], ["AAPL", "AAPL", "AAPL"], [1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError, match="finite"):
        pooled_skill([1, 2, 3], ["AAPL"] * 3, [1, float("nan"), 3], [1, 2, 3])
    with pytest.raises(ValueError, match="equal length"):
        pooled_skill([1, 2], ["AAPL"], [1, 2], [1, 2])


def test_pooled_dm_by_hand():
    # AAPL labels 1, -1, 0 have variance 2/3 and MSFT labels 2, 0, -2 variance 8/3. Forecast a errs by 1 on AAPL's
    # second row and 2 on MSFT's first, forecast b is zero, so the scaled squared errors per minute are
    # a 0.75, 0.75, 0 and b 1.5, 0.75, 0.75, their differences 0.75, 0, 0.75.
    minute = [1, 2, 3, 1, 2, 3]
    symbol = ["AAPL"] * 3 + ["MSFT"] * 3
    label = [1, -1, 0, 2, 0, -2]
    pred_a = [1, 0, 0, 0, 0, -2]
    pred_b = [0] * 6
    result = pooled_dm(minute, symbol, pred_a, pred_b, label, lags=0)
    se = math.sqrt(((0.25 ** 2) + 0.5 ** 2 + 0.25 ** 2) / 3 / 3)
    assert result.mean == pytest.approx(0.5) and result.se == pytest.approx(se)
    assert result.t == pytest.approx(0.5 / se) and result.mse_ratio == pytest.approx(0.5)
    assert result.rows == 6 and result.minutes == 3 and result.symbols == ("AAPL", "MSFT")
    flipped = pooled_dm(minute, symbol, pred_b, pred_a, label, lags=0)
    assert flipped.t == pytest.approx(-result.t) and flipped.mse_ratio == pytest.approx(2.0)


def test_pooled_dm_uses_the_clustered_error_and_leaves_out_a_constant_label():
    rng = np.random.default_rng(8)
    pred, label = market_draws(rng, 1, 300, 0.5, 0.3)
    pred, label = pred[0], label[0]
    label[:, 7] = 1.0
    m, s, p, y = rows_of(range(300), SYMBOLS, pred, label)
    result = pooled_dm(m, s, p, [0.0] * len(p), y)
    keep = list(range(7))
    var = label[:, keep].var(axis=0)
    d = (((label[:, keep]) ** 2 - (label[:, keep] - pred[:, keep]) ** 2) / var).mean(axis=1)
    assert result.left_out == ("QQQ",) and result.mean == pytest.approx(d.mean())
    assert result.se == pytest.approx(newey_west_se(d, 12)) and result.t < 0  # a 0.3 correlation loses to zero
    assert stats.LAGS == 12


def test_session_blocks_by_hand():
    # Differences 1, 3, 0, 2, 4 in sessions 1, 1, 2, 3, 3 give block means 2, 0, 3.
    result = session_block_test([1, 1, 2, 3, 3], [2, 4, 1, 3, 5], [1, 1, 1, 1, 1])
    means = np.array([2.0, 0.0, 3.0])
    assert result["blocks"] == 3 and result["pairs"] == 5 and result["mean"] == pytest.approx(5 / 3)
    assert result["t"] == pytest.approx((5 / 3) / math.sqrt(7 / 3 / 3))
    assert result["t"] == pytest.approx(means.mean() / (means.std(ddof=1) / math.sqrt(3)))
    assert result["threshold"] == pytest.approx(t_quantile_99(2)) and not result["passed"]
    assert stats.session_blocks([1, 1, 2, 3, 3], [1, 3, 0, 2, 4]) == {1: [1.0, 3.0], 2: [0.0], 3: [2.0, 4.0]}
    assert result == block_test({1: [1.0, 3.0], 2: [0.0], 3: [2.0, 4.0]})
    with pytest.raises(ValueError):
        stats.session_blocks([1, 2], [1.0])


def test_symbol_slopes_by_hand():
    slopes = symbol_slopes(["AAPL"] * 3 + ["MSFT"] * 3, [1, 2, 3, 5, 5, 5], [2, 4, 6, 1, 2, 3])
    assert slopes == {"AAPL": pytest.approx(2.0), "MSFT": None}
