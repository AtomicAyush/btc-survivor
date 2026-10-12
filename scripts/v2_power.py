"""Synthetic power of the confirmatory test of PLAN_V2 section 11.4. It never reads market data, a recording or a
database.

It simulates the news predictor of section 6.3 for the eight symbols as a mix of each symbol's own company input and
one macro input shared by all eight, with weights sqrt(1 - c) and sqrt(c), so c, the correlation between two symbols'
predictions at one minute, is the macro share of each prediction's variance. Each input is zero inflated. Impacts
arrive as Poisson events, a share of them neutral with strength zero and the rest with strength direction times
confidence over 5, and each input is sum(w * strength) / (sum(w) + 1) with w halving every 120 session minutes and
an impact dropped once older than 600. An impact that arrives outside the session counts from the next open at full
weight. Labels are the twelve 30 minute returns of each full session, with a market factor explaining 45 percent of
each stock's variance and 90 percent of QQQ's, Student t shocks, and a log volatility following an AR(1) process for
the factor and for each symbol, scaled to the 30 minute standard deviations of section 12. A true correlation r is
planted as label = sd * (r * prediction / its standard deviation + sqrt(1 - r**2) * shock).

Each run is judged by stats.matrix_skill, the estimator evaluate uses, over sessions 2 to N after one burn in
session. For each number of sessions N and each c it prints

* null rejection, the share of runs at r = 0 whose t passes 2.33, near 1 percent when the test is valid,
* detectable, (2.33 + 1.28) times the root mean square standard error at r = 0, the expected size of the figure
  evaluate prints from the realized error,
* power 90, the true correlation the test finds in 90 percent of runs, from each run's critical correlation, the
  smallest planted r whose t passes 2.33 on that run's draws,
* the power at a true correlation of 0.075 and of 0.11.

Every generator value is a planning value set by reasoning before launch and never fitted to recordings. Every figure
printed is synthetic, which section 13.3 allows in default output, so the rows always print.

    python scripts/v2_power.py [--runs 2000] [--sessions 20 40 60] [--c 0.2 0.5 0.8 1.0] [--seed 12345]
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field

import numpy as np

from btc_survivor.cli.stats import POWER_90
from btc_survivor.v2 import output
from btc_survivor.v2.stats import ONE_SIDED_Z, matrix_skill

SYMBOLS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "QQQ")
SD_BPS = np.array([38.0, 33.0, 68.0, 45.0, 45.0, 53.0, 91.0, 30.0])  # 30 minute standard deviations, section 12
MARKET_SHARE = np.array([0.45] * 7 + [0.90])  # variance explained by the market factor, as in section 11.2
BLOCK_MINUTES = 30
BLOCKS_PER_SESSION = 13  # 390 / 30, so blocks line up with every open and close
EVAL_PER_SESSION = 12  # session minutes 30 to 360
BURN_IN_SESSIONS = 1
CRITICAL_HIGH = 0.5  # the largest planted correlation the search tries
CRITICAL_STEPS = 12  # bisection steps, to within 0.00012


@dataclass(frozen=True)
class Generator:
    """Planning values of the synthetic inputs and labels, set by reasoning before launch."""

    impacts_per_session: float = 4.0  # new impacts per input per session day, of the 100 to 300 items of section 7.9
    session_share: float = 0.4  # arriving inside the session, about 6.5 of the 16 hours of busy polling
    neutral_share: float = 0.5  # direction 0, strength 0 with full weight in the denominator
    half_life_minutes: float = 120.0
    drop_after_minutes: float = 600.0  # a multiple of 30
    shock_df: float = 5.0  # Student t degrees of freedom
    vol_persistence: float = 0.95  # per 30 minute return, about one session's half life
    vol_log_sd: float = 0.35


@dataclass(frozen=True)
class Settings:
    """What one simulation covers, the defaults being the full settings run before launch."""

    runs: int = 2000
    sessions: tuple[int, ...] = (20, 40, 60)
    cs: tuple[float, ...] = (0.2, 0.5, 0.8, 1.0)
    power_at: tuple[float, ...] = (0.075, 0.11)
    seed: int = 12345
    chunk: int = 100  # runs drawn at once
    generator: Generator = field(default_factory=Generator)


@dataclass(frozen=True)
class Scenario:
    """The simulated result for one number of sessions and one c."""

    sessions: int
    c: float
    null_rejection: float
    detectable: float
    power_90: float  # inf when more than a tenth of runs never pass below CRITICAL_HIGH
    power: tuple[float, ...]  # at Settings.power_at


def impacts(rng: np.random.Generator, shape: tuple[int, int, int], gen: Generator
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded impacts on blocks shaped (runs, inputs, blocks), as flat block cells, minutes after the block's start
    and strengths. Those inside the session fall uniformly in their block, and the rest sit at the open, offset 0."""
    sessions = shape[2] // BLOCKS_PER_SESSION
    inside = rng.poisson(gen.impacts_per_session * gen.session_share / BLOCKS_PER_SESSION, shape)
    at_open = np.zeros(shape, dtype=np.int64)
    at_open[..., ::BLOCKS_PER_SESSION] = rng.poisson(gen.impacts_per_session * (1 - gen.session_share),
                                                     (*shape[:2], sessions))
    cells = np.arange(inside.size)
    cell = np.r_[np.repeat(cells, inside.ravel()), np.repeat(cells, at_open.ravel())]
    offset = np.r_[rng.uniform(0, BLOCK_MINUTES, inside.sum()), np.zeros(at_open.sum())]
    direction = np.where(rng.random(len(cell)) < gen.neutral_share, 0.0, rng.choice([-1.0, 1.0], len(cell)))
    return cell, offset, direction * rng.integers(1, 6, len(cell)) / 5


def inputs_at(cell: np.ndarray, offset: np.ndarray, strength: np.ndarray, shape: tuple[int, ...], gen: Generator
              ) -> np.ndarray:
    """The input at the start of every block from the impacts of earlier blocks, sum(w * strength) / (sum(w) + 1)
    with w = 2 ** (-age / half life), dropping impacts older than drop_after_minutes. An impact's weight is split
    into 2 ** (offset / half life) within its block and a decay per whole block after it."""
    if gen.drop_after_minutes % BLOCK_MINUTES:
        raise ValueError(f"drop_after_minutes must be a multiple of {BLOCK_MINUTES}")
    grow = 2.0 ** (offset / gen.half_life_minutes)
    size = math.prod(shape)
    sums = np.bincount(cell, strength * grow, size).reshape(shape)
    weights = np.bincount(cell, grow, size).reshape(shape)
    num, den = np.zeros(shape), np.zeros(shape)
    for lag in range(1, round(gen.drop_after_minutes / BLOCK_MINUTES) + 1):
        decay = 2.0 ** (-lag * BLOCK_MINUTES / gen.half_life_minutes)
        num[..., lag:] += decay * sums[..., :-lag]
        den[..., lag:] += decay * weights[..., :-lag]
    return num / (den + 1.0)


def eval_blocks(sessions: int) -> np.ndarray:
    """The blocks starting at the evaluation minutes 30 to 360 of each session."""
    return (np.arange(sessions)[:, None] * BLOCKS_PER_SESSION + np.arange(1, EVAL_PER_SESSION + 1)).ravel()


def input_values(rng: np.random.Generator, runs: int, inputs: int, sessions: int, gen: Generator) -> np.ndarray:
    """Zero inflated decayed inputs at every evaluation minute, shaped (runs, inputs, sessions * 12)."""
    shape = (runs, inputs, sessions * BLOCKS_PER_SESSION)
    return inputs_at(*impacts(rng, shape, gen), shape, gen)[..., eval_blocks(sessions)]


def shocks(rng: np.random.Generator, shape: tuple[int, int, int], gen: Generator) -> np.ndarray:
    """Unit variance Student t shocks along axis 1 times a volatility exp(x), with x an AR(1) process whose mean of
    -s**2 makes the volatility's mean square exactly 1."""
    s, phi, df = gen.vol_log_sd, gen.vol_persistence, gen.shock_df
    x = np.empty(shape)
    x[:, 0] = rng.normal(-s * s, s, (shape[0], shape[2]))
    eta = rng.normal(0.0, s * math.sqrt(1 - phi * phi), shape)
    for i in range(1, shape[1]):
        x[:, i] = -s * s + phi * (x[:, i - 1] + s * s) + eta[:, i]
    return np.exp(x) * rng.standard_t(df, shape) * math.sqrt((df - 2) / df)


def label_noise(rng: np.random.Generator, runs: int, sessions: int, gen: Generator) -> np.ndarray:
    """Unit variance 30 minute shocks of the eight symbols at every evaluation minute, shaped (runs, minutes, 8)."""
    minutes = sessions * EVAL_PER_SESSION
    market = shocks(rng, (runs, minutes, 1), gen)
    own = shocks(rng, (runs, minutes, len(SYMBOLS)), gen)
    return np.sqrt(MARKET_SHARE) * market + np.sqrt(1 - MARKET_SHARE) * own


def predictions(inputs: np.ndarray, c: float) -> np.ndarray:
    """Company inputs 0 to 7 mixed with the shared macro input 8 at macro share c, shaped (runs, minutes, 8)."""
    return (math.sqrt(1 - c) * inputs[:, :len(SYMBOLS)] + math.sqrt(c) * inputs[:, len(SYMBOLS):]).transpose(0, 2, 1)


def planted(pred: np.ndarray, noise: np.ndarray, r: float, scale: float) -> np.ndarray:
    """Labels in bps whose correlation with the prediction is r, where scale is the prediction's standard deviation."""
    return SD_BPS * (r * pred / scale + math.sqrt(1 - r * r) * noise)


def critical(pred: np.ndarray, noise: np.ndarray, scale: float, t0: float) -> float:
    """The smallest planted correlation whose t passes 2.33 on these draws, by bisection, 0 when the null run already
    passes and inf when even CRITICAL_HIGH does not."""
    if t0 > ONE_SIDED_Z:
        return 0.0
    if matrix_skill(pred, planted(pred, noise, CRITICAL_HIGH, scale)).t <= ONE_SIDED_Z:
        return math.inf
    low, high = 0.0, CRITICAL_HIGH
    for _ in range(CRITICAL_STEPS):
        mid = (low + high) / 2
        low, high = (low, mid) if matrix_skill(pred, planted(pred, noise, mid, scale)).t > ONE_SIDED_Z else (mid, high)
    return high


def judge_run(pred: np.ndarray, noise: np.ndarray, scale: float, power_at: tuple[float, ...]
              ) -> tuple[bool, float, float, list[bool]]:
    """One run's null verdict, null standard error, critical correlation and verdicts at power_at."""
    null = matrix_skill(pred, planted(pred, noise, 0.0, scale))
    passes = [matrix_skill(pred, planted(pred, noise, r, scale)).t > ONE_SIDED_Z for r in power_at]
    return null.t > ONE_SIDED_Z, null.se, critical(pred, noise, scale, null.t), passes


def simulate(settings: Settings) -> list[Scenario]:
    """Every scenario over the same seeded draws, the shorter windows being the first sessions of the longest."""
    rng = np.random.default_rng(settings.seed)
    longest = max(settings.sessions)
    keys = [(n, c) for n in settings.sessions for c in settings.cs]
    judged: dict[tuple[int, float], list] = {key: [] for key in keys}
    for start in range(0, settings.runs, settings.chunk):
        size = min(settings.chunk, settings.runs - start)
        inputs = input_values(rng, size, len(SYMBOLS) + 1, longest, settings.generator)
        noise = label_noise(rng, size, longest, settings.generator)
        scale = float(inputs.std())
        for c in settings.cs:
            pred = predictions(inputs, c)
            for n in settings.sessions:
                window = slice(BURN_IN_SESSIONS * EVAL_PER_SESSION, n * EVAL_PER_SESSION)
                judged[n, c] += [judge_run(pred[run, window], noise[run, window], scale, settings.power_at)
                                 for run in range(size)]
    scenarios = []
    for n, c in keys:
        rejected, se, crit, passes = zip(*judged[n, c])
        detectable = (ONE_SIDED_Z + POWER_90) * math.sqrt(np.mean(np.square(se)))
        scenarios.append(Scenario(n, c, float(np.mean(rejected)), detectable,
                                  float(np.quantile(crit, 0.9, method="higher")),
                                  tuple(float(x) for x in np.mean(passes, axis=0))))
    return scenarios


def null_bound(runs: int, level: float = 0.01) -> float:
    """The largest null rejection share consistent with the 1 percent level, three binomial errors above it."""
    return level + 3 * math.sqrt(level * (1 - level) / runs)


def report(settings: Settings, scenarios: list[Scenario], seconds: float) -> output.Report:
    """Counts, the null verdict and the run time, then one row per scenario, all synthetic."""
    rows = tuple({"sessions": s.sessions, "rows per symbol": (s.sessions - BURN_IN_SESSIONS) * EVAL_PER_SESSION,
                  "c": f"{s.c:.2f}", "null rejection": f"{s.null_rejection:.4f}", "detectable": f"{s.detectable:.3f}",
                  "power 90": f"{s.power_90:.3f}" if math.isfinite(s.power_90) else f"above {CRITICAL_HIGH}",
                  **{f"power at {r:g}": f"{p:.2f}" for r, p in zip(settings.power_at, s.power)}}
                 for s in scenarios)
    return output.Report("synthetic power", (
        ("runs", output.Count(settings.runs)),
        ("symbols", output.Count(len(SYMBOLS))),
        ("scenarios", output.Count(len(scenarios))),
        ("null rejection within the level", all(s.null_rejection <= null_bound(settings.runs) for s in scenarios)),
        ("run time", output.Seconds(seconds)),
    ), rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic power of the confirmatory news test. Never reads "
                                                 "market data. Every figure printed is synthetic.")
    parser.add_argument("--runs", type=int, default=Settings.runs)
    parser.add_argument("--sessions", type=int, nargs="+", default=list(Settings.sessions),
                        help="sessions to the cutoff, each at least 2, with session 1 as burn in")
    parser.add_argument("--c", type=float, nargs="+", default=list(Settings.cs),
                        help="correlation between two symbols' predictions, from 0 to 1")
    parser.add_argument("--seed", type=int, default=Settings.seed)
    args = parser.parse_args(argv)
    if args.runs < 1 or min(args.sessions) < 2 or not all(0 <= c <= 1 for c in args.c):
        parser.error("runs must be at least 1, sessions at least 2 and every c from 0 to 1")
    settings = Settings(runs=args.runs, sessions=tuple(sorted(set(args.sessions))), cs=tuple(sorted(set(args.c))),
                        seed=args.seed)
    started = time.monotonic()
    scenarios = simulate(settings)
    output.emit(report(settings, scenarios, time.monotonic() - started), detail=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
