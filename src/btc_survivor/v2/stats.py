"""Pooled skill and squared error tests of PLAN_V2 sections 11.2 and 11.3, over the rows of eight symbols.

pooled_skill z scores prediction and label within each symbol, multiplies the two z scores of every row, averages
that product over the symbols present at each evaluation minute, and judges the mean of those per minute values with
the version 1 newey_west_se at 12 lags, one session. That is the Driscoll Kraay error in its simplest form, which
allows for the common market move at a minute and slow drift in the forecasts. A symbol whose prediction never varies
is left out and named. Beside t it gives the detectable correlation, (2.33 + 1.28) standard errors, and the one sided
95 percent upper confidence bound, the correlation plus 1.645 standard errors, which a failed test is reported with.

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
    result = clustered(values, lags)
    if result is None:
        return PooledSkill(rows, len(values), used, left_out, None, None, None, False, None, None)
    corr, se, t = result
    return PooledSkill(rows, len(values), used, left_out, corr, se, t, t > one_sided_z,
                       (one_sided_z + POWER_90) * se, corr + upper_bound_z * se)


def pooled_skill(minute: Sequence[Hashable], symbol: Sequence[str], pred: Sequence[float], label: Sequence[float],
                 lags: int = LAGS, one_sided_z: float = ONE_SIDED_Z,
                 upper_bound_z: float = UPPER_BOUND_Z) -> PooledSkill:
    """One predictor's rows in the window, one per evaluation minute and symbol, judged by section 11.2."""
    _, symbols, (p, y) = pivot(minute, symbol, pred, label)
    return matrix_skill(p, y, symbols, lags, one_sided_z, upper_bound_z)


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
