"""Statistics for the success criteria in PLAN section 21, written out in numpy so every step can be checked."""

from __future__ import annotations

import math

import numpy as np

ONE_SIDED_1PCT = 2.33
POWER_90 = 1.28


def newey_west_se(u: np.ndarray, lags: int) -> float:
    """Standard error of the mean of u, allowing for correlation up to lags steps with Bartlett weights."""
    n = len(u)
    d = u - u.mean()
    s = d @ d / n
    for lag in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - lag / (lags + 1)) * (d[lag:] @ d[:-lag]) / n
    return math.sqrt(max(s, 0.0) / n)


def skill_test(pred: np.ndarray, label: np.ndarray, lags: int) -> dict:
    """Correlation of forecast and label as the mean product of their z scores, with a Newey West t statistic."""
    n = len(pred)
    if n < 3 or pred.std() == 0 or label.std() == 0:
        return {"n": n, "corr": None, "t": None, "passed": False, "detectable": None}
    u = (pred - pred.mean()) / pred.std() * (label - label.mean()) / label.std()
    corr = float(u.mean())
    se = newey_west_se(u, lags)
    t = corr / se if se > 0 else 0.0
    return {"n": n, "corr": corr, "t": t, "passed": t > ONE_SIDED_1PCT,
            "detectable": (ONE_SIDED_1PCT + POWER_90) / math.sqrt(n)}


def diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray, lags: int) -> dict:
    """Whether forecast a has smaller squared errors than forecast b. A positive t favours a."""
    d = errors_b**2 - errors_a**2
    if len(d) < 3:
        return {"n": len(d), "mse_ratio": None, "t": None}
    se = newey_west_se(d, lags)
    ratio = float(np.mean(errors_a**2) / np.mean(errors_b**2)) if np.mean(errors_b**2) > 0 else None
    return {"n": len(d), "mse_ratio": ratio, "t": float(d.mean() / se) if se > 0 else 0.0}


T99_SMALL = (31.821, 6.965, 4.541, 3.747, 3.365, 3.143, 2.998, 2.896, 2.821)  # degrees of freedom 1 to 9


def t_quantile_99(df: int) -> float:
    """The one sided 1 percent quantile of Student's t, from a table below 10 degrees of freedom and an expansion above."""
    if df < 10:
        return T99_SMALL[df - 1]
    z = 2.3263  # the exact normal quantile, since the expansion amplifies rounding
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df * df)


def block_test(blocks: dict[int, list[float]]) -> dict:
    """Mean of the block means and its t statistic, treating each block as one independent observation."""
    means = np.array([np.mean(v) for v in blocks.values() if v])
    k = len(means)
    if k < 2:
        return {"blocks": k, "pairs": sum(len(v) for v in blocks.values()), "mean": None, "t": None,
                "threshold": None, "passed": False}
    se = means.std(ddof=1) / math.sqrt(k)
    t = float(means.mean() / se) if se > 0 else 0.0
    threshold = t_quantile_99(k - 1)
    return {"blocks": k, "pairs": sum(len(v) for v in blocks.values()), "mean": float(means.mean()), "t": t,
            "threshold": threshold, "passed": t > threshold}
