"""Synthetic learning rate check of PLAN_V2 section 6.4. It never reads market data, a recording or a database.

It builds pooled sessions of the eight symbols on the real session calendar from the launch, each symbol with ten
AR(1) raw features, a common market factor in the features and in the one minute moves, and a signal planted in the
symbol's first feature whose conditional mean has a correlation of 0, 0.05 or 0.1 with its 30 minute label. It then
runs the stock process's own learning code minute by minute, the per symbol Standardizer and LabelScale, the session
bounded label queue, the PooledWindow and StockBot's own training step and predict_rows, with session 1 as history
only, labels that never cross the close, training once every label scale is ready, and batches of the samples that
matured this minute plus draws from the window. Learners of every rate run side by side on the same minutes, born
alike. Over the minutes in which they are Active and a label can mature it reports, in label standard deviations,

* spread, the root mean square of one learner's predictions, which is what the margin gate of section 10.5 sees,
* squared error vs zero, one learner's squared error over that of predicting zero,
* learner corr, the correlation of the learners' mean prediction with the label, beside the ideal correlation of the
  planted conditional mean.

The rule takes the largest of 0.0003, 0.0001, 0.00003 and 0.00001 whose spread with no signal is at most 0.10 and
whose squared error with no signal is within 2 percent of predicting zero, or 0.00001 if none qualifies. Every
generator value is a planning value set by reasoning before launch, the 30 minute volatilities of section 12 and the
market shares of section 11.2, never fitted to recordings. Every figure printed is synthetic, which section 13.3 allows
in default output, so the rows always print.

    python scripts/v2_learner_check.py [--config configs/v2/s1_prices.toml] [--sessions 40] [--seeds 3] [--learners 3]
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from btc_survivor.bot import ACTIVE
from btc_survivor.v2 import output, sessions
from btc_survivor.v2.config import StockConfig, load_stock_config
from btc_survivor.v2.sessions import Session
from btc_survivor.v2.stocks.bot import StockBot, StockMinute
from btc_survivor.v2.stocks.features import N_PRICE_FEATURES, make_standardizer
from btc_survivor.v2.stocks.labels import SessionLabelQueue, make_label_scale, push_k
from btc_survivor.v2.stocks.learning import PooledWindow, network_inputs

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "v2" / "s1_prices.toml"
LAUNCH = date(2026, 10, 19)
RATES = (0.0003, 0.0001, 0.00003, 0.00001)  # the candidates of section 6.4
SIGNALS = (0.0, 0.05, 0.1)
MAX_SPREAD = 0.10  # label standard deviations, with no signal
MAX_ERROR_RATIO = 1.02  # squared error over that of predicting zero, with no signal
FALLBACK_RATE = 0.00001
# Planning 30 minute standard deviations in bps, section 12.
SD_BPS = {"QQQ": 30.0, "MSFT": 33.0, "AAPL": 38.0, "GOOGL": 45.0, "AMZN": 45.0, "META": 53.0, "NVDA": 68.0,
          "TSLA": 91.0}
START_PRICE = 100.0
MIN_SESSIONS = 3  # history, then training, then Active minutes


@dataclass(frozen=True)
class Generator:
    """Planning values of the synthetic sessions, set by reasoning before launch."""

    stock_market_share: float = 0.45  # each stock's variance from the market factor, section 11.2
    index_market_share: float = 0.90  # the index symbol's
    signal_persistence: float = 0.97  # feature 0 per session minute, near a 30 minute return sampled every minute
    persistence_low: float = 0.0  # the other nine features spread evenly between these two, as in version 1
    persistence_high: float = 0.99


@dataclass(frozen=True)
class Settings:
    """What one check covers, the defaults being the full check run before launch."""

    sessions: int = 40  # from the launch, the length of the evaluation window
    seeds: int = 3
    learners: int = 3  # per rate on each seeded run
    rates: tuple[float, ...] = RATES
    signals: tuple[float, ...] = SIGNALS
    seed: int = 12345
    launch: date = LAUNCH
    generator: Generator = field(default_factory=Generator)


@dataclass(frozen=True)
class Synthetic:
    """One seeded run's sessions and its arrays indexed by k minus the first k, raw features shaped (minute, symbol,
    feature), reference prices, labels in bps, NaN where the horizon would cross the close, and the planted
    conditional mean of each label in bps."""

    days: tuple[Session, ...]
    raw: np.ndarray
    prices: np.ndarray
    labels: np.ndarray
    truth: np.ndarray

    @property
    def first_k(self) -> int:
        return self.days[0].first_k


@dataclass(frozen=True)
class Outcome:
    """One learning rate's figures on one seeded run, over Active minutes with a label, in label standard deviations."""

    spread: float
    error_ratio: float
    corr: float
    ideal: float
    rows: int


@dataclass(frozen=True)
class Scenario:
    """One signal and learning rate, its figures averaged over the seeded runs."""

    signal: float
    rate: float
    spread: float
    error_ratio: float
    corr: float
    ideal: float
    rows: int  # symbol minutes judged, summed over the runs
    samples: int  # training samples that matured, summed over the runs


def run_sessions(launch: date, count: int) -> tuple[Session, ...]:
    """The first count sessions of the calendar on or after the launch date."""
    days = tuple(s for s in sessions.all_sessions() if s.day >= launch)[:count]
    if len(days) < count:
        raise ValueError(f"the session table holds only {len(days)} sessions from {launch.isoformat()}")
    return days


def ar1(rng: np.random.Generator, steps: int, shape: tuple[int, ...], phis: np.ndarray) -> np.ndarray:
    """Unit variance AR(1) series along the first axis, one coefficient per entry of the last axis, started
    stationary."""
    x = np.empty((steps, *shape))
    x[0] = rng.normal(size=shape)
    shocks = rng.normal(size=(steps, *shape)) * np.sqrt(1.0 - phis ** 2)
    for t in range(1, steps):
        x[t] = phis * x[t - 1] + shocks[t]
    return x


def carried(persistence: float, horizon: int) -> float:
    """How much of feature 0 at t the moves of t plus 1 to t plus the horizon carry, the sum of its autocorrelations."""
    return sum(persistence ** j for j in range(1, horizon + 1))


def signal_beta(corr: float, persistence: float, horizon: int, minute_sd: float) -> float:
    """The weight of feature 0 in each one minute move that gives the planted conditional mean the correlation corr
    with its label over noise of minute_sd bps a minute, as in the version 1 check."""
    c = carried(persistence, horizon)
    v = horizon + 2 * sum((horizon - d) * persistence ** d for d in range(1, horizon))
    if corr * corr * v >= c * c:
        raise ValueError(f"a correlation of {corr} is not reachable with persistence {persistence} at a {horizon} "
                         f"minute horizon, the most is {c / math.sqrt(v):.3f}")
    return corr * minute_sd * math.sqrt(horizon) / math.sqrt(c * c - corr * corr * v)


def market_shares(symbols: Sequence[str], index_symbol: str, gen: Generator) -> np.ndarray:
    """Each symbol's share of variance from the market factor, the index symbol's larger."""
    return np.array([gen.index_market_share if s == index_symbol else gen.stock_market_share for s in symbols])


def synthesize(seed: np.random.SeedSequence, symbols: Sequence[str], index_symbol: str, days: tuple[Session, ...],
               signal: float, horizon: int, gen: Generator) -> Synthetic:
    """One seeded run. Every draw comes before the signal is applied, so one seed gives the same noise at every
    signal. Labels are the moves of the next horizon minutes, set only where the horizon ends by the close."""
    rng = np.random.default_rng(seed)
    n, count = days[-1].last_k - days[0].first_k + 1, len(symbols)
    minute_sd = np.array([SD_BPS[s] for s in symbols]) / math.sqrt(horizon)
    share = market_shares(symbols, index_symbol, gen)
    phis = np.r_[gen.signal_persistence, np.linspace(gen.persistence_low, gen.persistence_high, N_PRICE_FEATURES - 1)]
    own = ar1(rng, n, (count, N_PRICE_FEATURES), phis)
    market = ar1(rng, n, (N_PRICE_FEATURES,), phis)
    w = np.sqrt(share)[None, :, None]
    raw = w * market[:, None, :] + np.sqrt(1.0 - w * w) * own
    raw[:, :, 0] = own[:, :, 0]
    noise = minute_sd * (np.sqrt(share) * rng.normal(size=(n, 1)) + np.sqrt(1.0 - share) * rng.normal(size=(n, count)))
    beta = np.array([signal_beta(signal, gen.signal_persistence, horizon, sd) for sd in minute_sd])
    level = np.cumsum(beta * own[:, :, 0] + noise, axis=0)
    labels = np.full((n, count), np.nan)
    for day in days:
        a, b = day.first_k - days[0].first_k, day.last_k - horizon - days[0].first_k + 1
        labels[a:b] = level[a + horizon:b + horizon] - level[a:b]
    truth = beta * carried(gen.signal_persistence, horizon) * own[:, :, 0]
    return Synthetic(days, raw, START_PRICE * np.exp(level / 1e4), labels, truth)


def with_rate(cfg: StockConfig, rate: float) -> StockConfig:
    """The run config with only the learning rate changed."""
    return dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, learning_rate=rate))


def learn_and_predict(bot: StockBot, ctx: StockMinute) -> None:
    """The learning half of StockBot.step, a training step on this minute's samples and then predictions for every
    symbol, without the trading upkeep and decisions."""
    bot.preds = np.full(len(bot.preds), np.nan)
    if ctx.news_ok and ctx.matured and ctx.window is not None:
        bot._learn(ctx)
    bot._predict(ctx)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """The Pearson correlation, or 0 when either side never varies."""
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and a.std() > 0 and b.std() > 0 else 0.0


def judge(out: np.ndarray, data: Synthetic, label_sd: np.ndarray) -> Outcome:
    """Figures over the symbol minutes where every learner of the rate was Active and a label matures, with
    predictions and labels in the label standard deviation of their minute. out is shaped (learner, minute, symbol)."""
    y = data.labels / label_sd
    ok = np.isfinite(out).all(axis=0) & np.isfinite(y)
    p, y, truth = out[:, ok], y[ok], (data.truth / label_sd)[ok]
    if not len(y):
        return Outcome(math.nan, math.nan, 0.0, 0.0, 0)
    spread = float(np.mean(np.sqrt(np.mean(p ** 2, axis=1))))
    ratio = float(np.mean(np.sum((y - p) ** 2, axis=1) / np.sum(y ** 2)))
    return Outcome(spread, ratio, correlation(p.mean(axis=0), y), correlation(truth, y), int(ok.sum()))


def market_minutes(cfg: StockConfig, data: Synthetic) -> Iterator[tuple[int, StockMinute]]:
    """The shared market layer at every minute from session 2 on, with its index into data's arrays. Each symbol
    matures its sample, standardizes its raw vector and pushes a record that fits before the close, every symbol and
    news health always healthy, the news inputs zero, and the window handed over once every label scale is ready."""
    symbols, horizon = cfg.stock_feed.symbols, cfg.time.horizon_minutes
    count = len(symbols)
    queues = [SessionLabelQueue(horizon) for _ in symbols]
    scales = [make_label_scale(cfg.features) for _ in symbols]
    standardizers = [make_standardizer(cfg.features) for _ in symbols]
    window = PooledWindow.from_config(cfg)
    news = np.zeros(cfg.model.inputs - N_PRICE_FEATURES)
    every = np.ones(count, dtype=bool)
    for epoch, day in enumerate(data.days[1:], 1):  # session 1 only records history, and each close bumps the epoch
        for k in range(day.first_k, day.last_k + 1):
            i = k - data.first_k
            matured = []
            for s, symbol in enumerate(symbols):
                sample = queues[s].mature(k, epoch, float(data.prices[i, s]))
                if sample is not None:
                    window.add(k, symbol, sample.z, sample.label_bps)
                    scales[s].update(sample.label_bps)
                    matured.append((s, push_k(sample), sample.label_bps))
            rows = np.array([network_inputs(standardizers[s].transform_then_update(data.raw[i, s]), news)
                             for s in range(count)])
            for s in range(count):
                queues[s].push(k, epoch, rows[s], float(data.prices[i, s]))
            label_sd = np.array([scale.scale if scale.ready else np.nan for scale in scales])
            ready = all(scale.ready for scale in scales)
            yield i, StockMinute(day.boundary(k - day.first_k + 1), k, day, rows, every, every, True, data.prices[i],
                                 label_sd, every, tuple(matured), window if ready else None)


def newborns(cfg: StockConfig, rates: Sequence[float], learners: int, first_bot_id: int, born_ms: int
             ) -> dict[float, list[StockBot]]:
    """Each rate's learners, the same bot ids and slots at every rate, so learners of every rate are born alike and
    draw the same batches until their weights part."""
    return {rate: [StockBot.newborn(with_rate(cfg, rate), first_bot_id + j, j, born_ms) for j in range(learners)]
            for rate in rates}


def run(cfg: StockConfig, data: Synthetic, rates: Sequence[float], learners: int, first_bot_id: int
        ) -> tuple[dict[float, Outcome], int]:
    """Every rate's learners on one seeded run, minute by minute as the shared market layer and the bots do, and the
    number of training samples that matured."""
    bots = newborns(cfg, rates, learners, first_bot_id, data.days[0].open_ms)
    shape = (len(data.prices), len(cfg.stock_feed.symbols))
    out = {rate: np.full((learners, *shape), np.nan) for rate in rates}
    sd_at = np.full(shape, np.nan)
    samples = 0
    for i, ctx in market_minutes(cfg, data):
        sd_at[i] = ctx.label_sd
        samples += len(ctx.matured)
        for rate, group in bots.items():
            for j, bot in enumerate(group):
                learn_and_predict(bot, ctx)
                if bot.state == ACTIVE:
                    out[rate][j, i] = bot.preds / ctx.label_sd
    return {rate: judge(out[rate], data, sd_at) for rate in rates}, samples


def simulate(cfg: StockConfig, settings: Settings) -> list[Scenario]:
    """Every signal and rate over the same seeded runs, each seed's learners distinct and alike across rates."""
    days = run_sessions(settings.launch, settings.sessions)
    symbols, index = cfg.stock_feed.symbols, cfg.stock_feed.index_symbol
    found: dict[tuple[float, float], list[tuple[Outcome, int]]] = {
        (signal, rate): [] for signal in settings.signals for rate in settings.rates}
    for seed in range(settings.seeds):
        for signal in settings.signals:
            data = synthesize(np.random.SeedSequence([settings.seed, seed]), symbols, index, days, signal,
                              cfg.time.horizon_minutes, settings.generator)
            outcomes, samples = run(cfg, data, settings.rates, settings.learners, 1 + seed * settings.learners)
            for rate, outcome in outcomes.items():
                found[signal, rate].append((outcome, samples))
    scenarios = []
    for (signal, rate), runs in found.items():
        mean = {f: float(np.mean([getattr(o, f) for o, _ in runs])) for f in ("spread", "error_ratio", "corr", "ideal")}
        scenarios.append(Scenario(signal, rate, **mean, rows=sum(o.rows for o, _ in runs),
                                  samples=sum(s for _, s in runs)))
    return scenarios


def qualifies(scenario: Scenario) -> bool:
    """Whether a run with no signal meets both limits of the rule."""
    return scenario.signal == 0 and scenario.spread <= MAX_SPREAD and scenario.error_ratio <= MAX_ERROR_RATIO


def choose_rate(scenarios: Sequence[Scenario]) -> float:
    """The largest rate whose run with no signal meets both limits, or 0.00001 when none does."""
    return max((s.rate for s in scenarios if qualifies(s)), default=FALLBACK_RATE)


def rate_text(rate: float) -> str:
    """A rate in plain decimals, such as 0.00003."""
    return f"{rate:.8f}".rstrip("0")


def report(cfg: StockConfig, settings: Settings, scenarios: Sequence[Scenario], seconds: float) -> output.Report:
    """Counts, the rule's verdict and the run time, then one row per signal and rate, all synthetic."""
    chosen = choose_rate(scenarios)
    rows = tuple({"signal": f"{s.signal:.2f}", "learning rate": rate_text(s.rate), "spread": f"{s.spread:.4f}",
                  "squared error vs zero": f"{s.error_ratio:.4f}", "learner corr": f"{s.corr:+.4f}",
                  "ideal corr": f"{s.ideal:+.4f}", "rows": s.rows,
                  "rule": ("pass" if qualifies(s) else "fail") if s.signal == 0 else "",
                  "chosen": "yes" if s.signal == 0 and s.rate == chosen else ""} for s in scenarios)
    return output.Report("synthetic learner check", (
        ("sessions", output.Count(settings.sessions)),
        ("seeds", output.Count(settings.seeds)),
        ("learners per rate", output.Count(settings.learners)),
        ("symbols", output.Count(len(cfg.stock_feed.symbols))),
        ("scenarios", output.Count(len(scenarios))),
        ("matured samples per scenario", output.Count(scenarios[0].samples if scenarios else 0)),
        ("a rate meets the rule", any(qualifies(s) for s in scenarios)),
        ("run time", output.Seconds(seconds)),
    ), rows)


def chosen_report(scenarios: Sequence[Scenario]) -> output.Report:
    """The rate the rule chose, a setting of this system rather than a market figure, so it always prints."""
    return output.Report("chosen learning rate", rows=({"learning rate": rate_text(choose_rate(scenarios))},))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic learning rate check of PLAN_V2 section 6.4. Never reads "
                                                 "market data. Every figure printed is synthetic.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="a stock run config, whose settings other than the learning rate the learners use")
    parser.add_argument("--sessions", type=int, default=Settings.sessions,
                        help=f"sessions from the launch, at least {MIN_SESSIONS}, with session 1 as history only")
    parser.add_argument("--seeds", type=int, default=Settings.seeds)
    parser.add_argument("--learners", type=int, default=Settings.learners, help="learners per rate on each run")
    parser.add_argument("--seed", type=int, default=Settings.seed)
    args = parser.parse_args(argv)
    if args.sessions < MIN_SESSIONS or args.seeds < 1 or args.learners < 1:
        parser.error(f"sessions must be at least {MIN_SESSIONS} and seeds and learners at least 1")
    cfg = load_stock_config(args.config)
    settings = Settings(sessions=args.sessions, seeds=args.seeds, learners=args.learners, seed=args.seed)
    started = time.monotonic()
    scenarios = simulate(cfg, settings)
    output.emit(report(cfg, settings, scenarios, time.monotonic() - started), detail=True)
    output.emit(chosen_report(scenarios), detail=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
