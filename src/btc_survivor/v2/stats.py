"""Pooled skill and squared error tests of PLAN_V2 sections 11.2 and 11.3, over the rows of eight symbols.

pooled_skill z scores prediction and label within each symbol, multiplies the two z scores of every row, averages
that product over the symbols present at each evaluation minute, and judges the mean of those per minute values with
the version 1 newey_west_se at 12 lags, one session. That is the Driscoll Kraay error in its simplest form, which
allows for the common market move at a minute and slow drift in the forecasts. A symbol whose prediction never varies
is left out and named. Beside t it gives the detectable correlation, (2.33 + 1.28) standard errors, and the one sided
95 percent upper confidence bound, the correlation plus 1.645 standard errors, which a failed test is reported with.

partial_skill is the incremental test H2 of section 11.5, run only after the confirmatory test passed. Within each
symbol it residualises the prediction and the label on an intercept and the stored price z scores by least squares
over the rows judged, and its correlation is the pooled correlation of the two residuals, judged with the same
clustered error and z values. Its small sample correction takes the clustered error from the products of the news
residual's z score and the label's own z score, each symbol's scaled by the label's standard deviation over its
residual's, so their mean is still the partial correlation. In 40,000 synthetic null runs at 468 rows per symbol with
10 exogenous persistent scores, at c of 0.2, 0.5 and 1.0, the uncorrected test rejected 1.4 to 1.6 percent with a mean
squared t 6 to 8 percent above the raw test's on the same draws, and the corrected test 1.1 to 1.4 percent, as the raw
test did, with the raw test's mean squared t. With scores built from the label's own past returns, as three of the
ten are, the corrected test alone rejected 1.4 to 1.6 percent in a review's 20,000 runs per case, so H2 by itself is
somewhat liberal. The claim rests on the fixed sequence instead, H1 and then H2, which rejected 0.8 to 0.9 percent of
those null runs and 0.2 to 0.8 percent under seven momentum confounds, so its familywise error stays within one sided
1 percent.

pooled_dm treats squared errors the same way, dividing each row's difference by its symbol's label variance, and
session_block_test compares paired trades through the version 1 block_test with one block per session. The lags and
z values default to the [evaluation] section of PLAN_V2 section 15, and evaluate passes the ones stored in meta.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np

from btc_survivor.cli.stats import POWER_90, block_test, newey_west_se

LAGS = 12  # evaluation minutes in one full session
ONE_SIDED_Z = 2.33  # one sided 1 percent
UPPER_BOUND_Z = 1.645  # one sided 95 percent
MIN_MINUTES = 3  # fewer per minute values give no statistic, as in the version 1 skill_test
# A residual whose square sum is at most this share of its column's centred square sum is the scores' own fit and
# rounding, so its symbol is left out rather than judged on rounding noise.
EXPLAINED = 1e-12


@dataclass(frozen=True)
class PooledSkill:
    """One predictor's pooled correlation with its clustered error, t, verdict, detectable correlation and bound."""

    rows: int
    minutes: int
    symbols: tuple[str, ...]  # the symbols used
    left_out: tuple[str, ...]  # symbols with rows whose prediction or label never varies
    corr: float | None
    se: float | None
    t: float | None
    passed: bool
    detectable: float | None
    upper_bound: float | None


@dataclass(frozen=True)
class PooledDM:
    """Squared errors of forecast a against forecast b in label variances, a positive t favouring a."""

    rows: int
    minutes: int
    symbols: tuple[str, ...]
    left_out: tuple[str, ...]  # symbols whose label never varies
    mse_ratio: float | None  # a's scaled squared error over b's, below 1 when a is better
    mean: float | None  # mean per minute difference, b's scaled squared error minus a's
    se: float | None
    t: float | None


def pivot(minute: Sequence[Hashable], symbol: Sequence[str], *columns: Sequence[float]
          ) -> tuple[np.ndarray, tuple[str, ...], list[np.ndarray]]:
    """One matrix per column with a row per minute and a column per symbol, both sorted, NaN where a symbol has no
    row. A second row for one minute and symbol and a value that is not finite are refused."""
    minute, symbol = np.asarray(minute), np.asarray(symbol, dtype=str)
    if minute.shape != symbol.shape or minute.ndim != 1:
        raise ValueError("minute and symbol must be one dimensional and of equal length")
    minutes, mi = np.unique(minute, return_inverse=True)
    symbols, si = np.unique(symbol, return_inverse=True)
    if np.unique(mi * max(len(symbols), 1) + si).size < len(minute):
        raise ValueError("there is more than one row for a minute and symbol")
    matrices = []
    for column in columns:
        values = np.asarray(column, dtype=float)
        if values.shape != minute.shape or not np.isfinite(values).all():
            raise ValueError("every column needs one finite value per row")
        matrix = np.full((len(minutes), len(symbols)), np.nan)
        matrix[mi, si] = values
        matrices.append(matrix)
    return minutes, tuple(str(s) for s in symbols), matrices


def varies(column: np.ndarray) -> bool:
    """Whether a column has two present values that differ, compared exactly so rounding never counts as variation."""
    present = column[~np.isnan(column)]
    return len(present) >= 2 and present.max() > present.min()


def zscores(matrix: np.ndarray) -> np.ndarray:
    """Each column minus its mean over its present rows, over its population standard deviation, NaN where absent."""
    return (matrix - np.nanmean(matrix, axis=0)) / np.nanstd(matrix, axis=0)


def minute_means(matrix: np.ndarray) -> np.ndarray:
    """The mean over the symbols present at each minute, for the minutes where at least one is present."""
    present = ~np.isnan(matrix)
    keep = present.any(axis=1)
    return np.nansum(matrix[keep], axis=1) / present[keep].sum(axis=1)


def clustered(values: np.ndarray, lags: int) -> tuple[float, float, float] | None:
    """The mean of the per minute values, its version 1 Newey West error and t, or None below three minutes."""
    if len(values) < MIN_MINUTES:
        return None
    mean = float(values.mean())
    se = newey_west_se(values, lags)
    return mean, se, mean / se if se > 0 else 0.0


def judged(rows: int, values: np.ndarray, used: tuple[str, ...], left_out: tuple[str, ...], lags: int,
           one_sided_z: float, upper_bound_z: float) -> PooledSkill:
    """The verdict of section 11.2 on per minute values whose mean is the correlation, with the bound and the
    detectable correlation from their clustered error."""
    result = clustered(values, lags)
    if result is None:
        return PooledSkill(rows, len(values), used, left_out, None, None, None, False, None, None)
    corr, se, t = result
    return PooledSkill(rows, len(values), used, left_out, corr, se, t, t > one_sided_z,
                       (one_sided_z + POWER_90) * se, corr + upper_bound_z * se)


def matrix_skill(pred: np.ndarray, label: np.ndarray, symbols: Sequence[str] | None = None, lags: int = LAGS,
                 one_sided_z: float = ONE_SIDED_Z, upper_bound_z: float = UPPER_BOUND_Z) -> PooledSkill:
    """Pooled skill from minute by symbol matrices with NaN for a missing row, the steps of section 11.2."""
    names = tuple(symbols) if symbols is not None else tuple(str(j) for j in range(pred.shape[1]))
    both = ~np.isnan(pred) & ~np.isnan(label)
    pred, label = np.where(both, pred, np.nan), np.where(both, label, np.nan)
    keep = [j for j in range(pred.shape[1]) if varies(pred[:, j]) and varies(label[:, j])]
    left_out = tuple(names[j] for j in range(pred.shape[1]) if j not in keep and both[:, j].any())
    values = minute_means(zscores(pred[:, keep]) * zscores(label[:, keep]))
    rows, used = int(both[:, keep].sum()), tuple(names[j] for j in keep)
    return judged(rows, values, used, left_out, lags, one_sided_z, upper_bound_z)


def pooled_skill(minute: Sequence[Hashable], symbol: Sequence[str], pred: Sequence[float], label: Sequence[float],
                 lags: int = LAGS, one_sided_z: float = ONE_SIDED_Z,
                 upper_bound_z: float = UPPER_BOUND_Z) -> PooledSkill:
    """One predictor's rows in the window, one per evaluation minute and symbol, judged by section 11.2."""
    _, symbols, (p, y) = pivot(minute, symbol, pred, label)
    return matrix_skill(p, y, symbols, lags, one_sided_z, upper_bound_z)


def residual(values: np.ndarray, controls: np.ndarray) -> np.ndarray | None:
    """values minus their least squares fit on an intercept and the controls, one row each, or None when nothing is
    left beyond the fit's rounding, so the controls explain the values."""
    design = np.column_stack([np.ones(len(values)), controls])
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    out = values - design @ beta
    centred = values - values.mean()
    return out if out @ out > EXPLAINED * (centred @ centred) else None


def matrix_partial_skill(pred: np.ndarray, label: np.ndarray, controls: np.ndarray,
                         symbols: Sequence[str] | None = None, lags: int = LAGS, one_sided_z: float = ONE_SIDED_Z,
                         upper_bound_z: float = UPPER_BOUND_Z) -> PooledSkill:
    """The incremental test from minute by symbol matrices and controls shaped minutes by symbols by scores, a row
    counting only with every control present. A symbol is left out when its prediction or label never varies, the
    scores explain either, or it has fewer than two rows beyond its fit's coefficients."""
    names = tuple(symbols) if symbols is not None else tuple(str(j) for j in range(pred.shape[1]))
    both = ~np.isnan(pred) & ~np.isnan(label) & np.isfinite(controls).all(axis=2)
    products = np.full(pred.shape, np.nan)
    keep = []
    for j in range(pred.shape[1]):
        rows = both[:, j]
        p, y, x = pred[rows, j], label[rows, j], controls[rows, j]
        if not (varies(p) and varies(y)) or rows.sum() <= x.shape[1] + 2:
            continue
        p_resid, y_resid = residual(p, x), residual(y, x)
        if p_resid is None or y_resid is None:
            continue
        scale = y.std() / y_resid.std()
        products[rows, j] = scale * (p_resid - p_resid.mean()) / p_resid.std() * (y - y.mean()) / y.std()
        keep.append(j)
    left_out = tuple(names[j] for j in range(pred.shape[1]) if j not in keep and both[:, j].any())
    values = minute_means(products[:, keep])
    rows, used = int(both[:, keep].sum()), tuple(names[j] for j in keep)
    return judged(rows, values, used, left_out, lags, one_sided_z, upper_bound_z)


def partial_skill(minute: Sequence[Hashable], symbol: Sequence[str], pred: Sequence[float], label: Sequence[float],
                  controls: Sequence[Sequence[float]], lags: int = LAGS, one_sided_z: float = ONE_SIDED_Z,
                  upper_bound_z: float = UPPER_BOUND_Z) -> PooledSkill:
    """The incremental test on rows, each with its price z scores in controls, every row with the same number."""
    if len(minute) == 0:
        return judged(0, np.array([]), (), (), lags, one_sided_z, upper_bound_z)
    scores = np.asarray(controls, dtype=float)
    if scores.ndim != 2 or scores.shape[0] != len(minute):
        raise ValueError("controls need one row of scores per row, every row with the same number")
    _, symbols, (p, y, *x) = pivot(minute, symbol, pred, label, *scores.T)
    return matrix_partial_skill(p, y, np.stack(x, axis=2), symbols, lags, one_sided_z, upper_bound_z)


def pooled_dm(minute: Sequence[Hashable], symbol: Sequence[str], pred_a: Sequence[float], pred_b: Sequence[float],
              label: Sequence[float], lags: int = LAGS) -> PooledDM:
    """Whether forecast a has smaller squared errors than forecast b on the same rows. Each row's difference is
    divided by its symbol's label variance over these rows, averaged per minute and judged with the clustered error."""
    _, symbols, (a, b, y) = pivot(minute, symbol, pred_a, pred_b, label)
    keep = [j for j in range(len(symbols)) if varies(y[:, j])]
    left_out = tuple(symbols[j] for j in range(len(symbols)) if j not in keep)
    y, variance = y[:, keep], np.nanvar(y[:, keep], axis=0)
    scaled_a = minute_means((y - a[:, keep]) ** 2 / variance)
    scaled_b = minute_means((y - b[:, keep]) ** 2 / variance)
    rows, used = int((~np.isnan(y)).sum()), tuple(symbols[j] for j in keep)
    result = clustered(scaled_b - scaled_a, lags)
    if result is None:
        return PooledDM(rows, len(scaled_a), used, left_out, None, None, None, None)
    mean, se, t = result
    ratio = float(scaled_a.mean() / scaled_b.mean()) if scaled_b.mean() > 0 else None
    return PooledDM(rows, len(scaled_a), used, left_out, ratio, mean, se, t)


def session_blocks(session: Sequence[int], values: Sequence[float]) -> dict[int, list[float]]:
    """The values grouped into one block per session, in order of first appearance."""
    blocks: dict[int, list[float]] = {}
    for key, value in zip(session, values, strict=True):
        blocks.setdefault(int(key), []).append(float(value))
    return blocks


def session_block_test(session: Sequence[int], first: Sequence[float], second: Sequence[float]) -> dict:
    """First minus second over paired trades, such as a learner's gross profit and its twin's from one decision,
    tested through the version 1 block_test with one block per session, so a positive t favours the first."""
    diffs = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    return block_test(session_blocks(session, diffs))


def symbol_slopes(symbol: Sequence[str], pred: Sequence[float], label: Sequence[float]) -> dict[str, float | None]:
    """The slope of label on prediction within each symbol, 1 when a bps forecast is calibrated, and None where the
    prediction never varies."""
    symbol, pred, label = np.asarray(symbol, dtype=str), np.asarray(pred, dtype=float), np.asarray(label, dtype=float)
    slopes: dict[str, float | None] = {}
    for name in sorted(set(symbol.tolist())):
        p, y = pred[symbol == name], label[symbol == name]
        slopes[name] = float(np.mean((p - p.mean()) * (y - y.mean())) / p.var()) if varies(p) else None
    return slopes
