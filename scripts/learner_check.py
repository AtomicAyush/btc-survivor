"""Synthetic capability check for the online learner. It never reads market data.

It plants a signal of known strength in synthetic features with overlapping labels, runs the project's own bot
learning code exactly as the engine does each minute, and reports how much of the signal the learner recovers and
how large its predictions are when there is nothing to predict. Learner settings are fixed from this check before
launch, never from market data, so it cannot tune anything to the market.

    python scripts/learner_check.py --config config.toml [--days 28] [--seeds 3] [--bots 6] [--lr 0.001] [--flip-days 2]
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from dataclasses import dataclass

import numpy as np

from btc_survivor.baselines import LinearBaseline
from btc_survivor.bot import Bot, MinuteContext
from btc_survivor.clock import MS_PER_MINUTE
from btc_survivor.config import Config, load_config
from btc_survivor.features import N_FEATURES
from btc_survivor.labels import Matured, MaturedWindow
from btc_survivor.standardizer import LabelScale

MINUTE_STD_BPS = 3.5  # one minute BTC return spread measured by the live recorder


@dataclass(frozen=True)
class Result:
    corr: float  # prequential correlation of the population mean prediction with the label
    ideal: float  # correlation of the true conditional mean with the label
    spread_bps: float  # standard deviation of one bot's predictions
    mse_vs_zero: float  # mean squared error of the population mean prediction over that of predicting zero
    slope: float  # regression slope of label on the population mean prediction, 1 when calibrated
    linear: float  # prequential correlation of the linear baseline with the label


def forward_sums(moves: np.ndarray, horizon: int, minutes: int) -> np.ndarray:
    """The label for minute t is the sum of the moves in minutes t plus 1 to t plus horizon, never earlier ones."""
    csum = np.r_[0.0, np.cumsum(moves)]
    return csum[horizon + 1:horizon + 1 + minutes] - csum[1:1 + minutes]


def synthetic(seed: int, minutes: int, horizon: int, corr: float, persistence: float, flip_days: float = 0.0):
    """AR(1) z features and overlapping labels, with feature 0 carrying a linear signal of the given correlation.

    With flip_days the signal changes sign that often, to check that the learner still adapts to a new regime.
    """
    rng = np.random.default_rng(seed)
    phis = np.r_[persistence, np.linspace(0.0, 0.99, N_FEATURES - 1)]
    z = np.zeros((minutes + horizon + 1, N_FEATURES))
    z[0] = rng.normal(size=N_FEATURES)
    shocks = rng.normal(size=z.shape) * np.sqrt(1 - phis**2)
    for t in range(1, len(z)):
        z[t] = phis * z[t - 1] + shocks[t]
    phi = persistence
    c = sum(phi**k for k in range(1, horizon + 1))
    v = horizon + 2 * sum((horizon - d) * phi**d for d in range(1, horizon))
    beta = corr * MINUTE_STD_BPS * math.sqrt(horizon) / math.sqrt(max(c * c - corr * corr * v, 1e-12))
    sign = np.ones(len(z))
    if flip_days:
        sign = np.where((np.arange(len(z)) // int(flip_days * 1440)) % 2 == 0, 1.0, -1.0)
    moves = sign * beta * z[:, 0] + rng.normal(size=len(z)) * MINUTE_STD_BPS
    labels = forward_sums(moves, horizon, minutes)
    return z[:minutes], labels, sign[:minutes] * beta * c * z[:minutes, 0]


def run(cfg: Config, seed: int, days: float, corr: float, persistence: float, bots: int = 1,
        flip_days: float = 0.0) -> Result:
    horizon = cfg.time.horizon_minutes
    minutes = int(days * 1440)
    z, labels, truth = synthetic(seed, minutes, horizon, corr, persistence, flip_days)
    window = MaturedWindow(cfg.model.replay_window_minutes, N_FEATURES)
    scale = LabelScale(cfg.features.standardizer_half_life_hours * 60, cfg.features.min_samples)
    population = [Bot.newborn(cfg, i + 1, i, 0) for i in range(bots)]
    linear = LinearBaseline(N_FEATURES, cfg.features.standardizer_half_life_hours * 60)
    preds = np.full((minutes, bots), np.nan)
    lin = np.full(minutes, np.nan)
    for t in range(minutes):
        now = t * MS_PER_MINUTE
        matured = None
        if t >= horizon:
            matured = Matured((t - horizon) * MS_PER_MINUTE, z[t - horizon], float(labels[t - horizon]))
            window.add(matured)
            scale.update(matured.label_bps)
            linear.update(matured.z, matured.label_bps)
        ctx = MinuteContext(now, None, z[t], matured, window, scale)
        for i, bot in enumerate(population):
            if matured is not None and scale.ready:
                bot._learn(ctx)
            if scale.ready:
                lin[t] = linear.predict(z[t])
                preds[t, i] = bot.model.predict(z[t]) * scale.scale
                bot.pending_preds[now] = preds[t, i]
    start = min(minutes // 10, 1440)  # skip the first day, or the first tenth of a short run
    p, y, ideal, q = np.nanmean(preds[start:], axis=1), labels[start:], truth[start:], lin[start:]
    ok = np.isfinite(p)
    p, y, ideal, q = p[ok], y[ok], ideal[ok], q[ok]
    corr_p = float(np.corrcoef(p, y)[0, 1]) if p.std() > 0 else 0.0
    corr_i = float(np.corrcoef(ideal, y)[0, 1]) if ideal.std() > 0 else 0.0
    slope = float(np.cov(p, y)[0, 1] / p.var()) if p.var() > 0 else 0.0
    spread = float(np.nanstd(preds[start:, 0]))
    corr_q = float(np.corrcoef(q, y)[0, 1]) if q.std() > 0 else 0.0
    return Result(corr_p, corr_i, spread, float(np.mean((y - p) ** 2) / np.mean(y**2)), slope, corr_q)


def with_model(cfg: Config, **changes) -> Config:
    model = {k: v for k, v in changes.items() if hasattr(cfg.model, k)}
    genes = {k: v for k, v in changes.items() if hasattr(cfg.genes, k)}
    return dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, **model),
                               genes=dataclasses.replace(cfg.genes, **genes))


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic learner capability check. Never reads market data.")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--days", type=float, default=28.0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--bots", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None, help="override the learning rate gene for a comparison")
    parser.add_argument("--flip-days", type=float, default=0.0, help="flip the planted signal's sign this often")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.lr is not None:
        cfg = with_model(cfg, lr_init=args.lr, lr_min=min(args.lr, cfg.genes.lr_min), lr_max=max(args.lr, cfg.genes.lr_max))
    print(f"window {cfg.model.replay_window_minutes} min, batch {cfg.model.batch_size}, lr {cfg.genes.lr_init}, "
          f"horizon {cfg.time.horizon_minutes} min, {args.days:g} days, {args.seeds} seeds, {args.bots} bots, "
          f"flip every {args.flip_days:g} days")
    for corr in (0.0, 0.05, 0.1, 0.2):
        for persistence in (0.9, 0.99):
            rs = [run(cfg, s, args.days, corr, persistence, args.bots, args.flip_days) for s in range(args.seeds)]
            mean = {f: float(np.mean([getattr(r, f) for r in rs])) for f in Result.__dataclass_fields__}
            print(f"signal {corr:.2f} persistence {persistence:.2f}  learner {mean['corr']:+.3f} of ideal "
                  f"{mean['ideal']:+.3f}  spread {mean['spread_bps']:.1f} bps  mse vs zero {mean['mse_vs_zero']:.3f}  "
                  f"slope {mean['slope']:+.2f}  linear {mean['linear']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
