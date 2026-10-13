"""Planted news check of PLAN_V2 section 6.4, a descriptive power curve for S2 against S1 published before launch. It
never reads market data, a recording or a database, and every figure it prints is synthetic.

It adds sparse synthetic news to the learner check of scripts/v2_learner_check.py. Items tagged with each of the eight
symbols and with MACRO arrive at the planning rates of scripts/v2_power.py, 4 per tag per session day, 40 percent inside
the session and the rest overnight, half neutral and the rest with prompt_v2's strength as scripts/v2_power.py draws
it, direction times confidence over 5 times the horizon's weight times (2 + surprise) over 5. They
pass through the stock process's own NewsInputs as Haiku scores received when scored, so each symbol's company and
macro inputs decay with the 120 session minute half life, keep full weight to the open, drop after 600 session minutes
and carry one neutral prior item, and QQQ's company input takes every company's items.

The price features are the learner check's AR(1) noise with no price signal. Each one minute move drifts by beta times
a planted predictor at the boundary before it, company plus macro from the same items through NewsInputs at a reaction
half life, 120 session minutes when the price absorbs news at the pace the input assumes and 15 when it absorbs news
fast. Beta is set per symbol from separate calibration draws, so that the planted predictor's correlation with the 30
minute label over the evaluation minutes is 0, 0.05 or 0.1 in the population.

S2 learners see the news inputs and S1 learners, born alike with the same bot ids, see zeros, on one shared market layer
with a PooledWindow per run, through StockBot's own training step and predict_rows. Each population forecast is the
mean of a run's learners, judged over the evaluation minutes of sessions 2 to N by the pooled statistics evaluate
uses. Each scenario reports, averaged over seeds,

* ideal, the planted predictor's pooled correlation with the label,
* news, the pooled correlation and t of the news predictor S2 writes, with the share of seeds above 2.33,
* extraction by window third, the correlation of S2's population forecast minus S1's with the planted predictor,
* the pooled t of each population forecast and of S2 minus S1,
* S2 against S1, the pooled_dm t on paired rows and the share of seeds above 2.33, which is the power curve,
* the mean absolute news weight of S2's learners at the end of each third, beside their price weights.

By default it prints counts, a verdict and the run time, and the rows only with --detail. Every generator value is a
planning value set by reasoning before launch and never fitted to recordings. The result changes no setting.

    python scripts/v2_news_check.py [--config configs/v2/s2_news.toml] [--sessions 40] [--seeds 24] [--learners 24]
        [--calibration-seeds 8] [--strength-scale 1.0] [--seed 12345] [--workers 1] [--detail]
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

import v2_learner_check as lc
from v2_power import draw_strengths
from btc_survivor.v2 import output, sessions, stats
from btc_survivor.v2.config import StockConfig, load_stock_config
from btc_survivor.v2.news.prefilter import LLM, MACRO
from btc_survivor.v2.sessions import Session
from btc_survivor.v2.stocks.bot import StockBot, StockMinute
from btc_survivor.v2.stocks.features import N_PRICE_FEATURES, make_standardizer
from btc_survivor.v2.stocks.labels import SessionLabelQueue, make_label_scale, push_k
from btc_survivor.v2.stocks.learning import N_NEWS_INPUTS, PooledWindow, network_inputs
from btc_survivor.v2.stocks.news_inputs import HAIKU_SCORER, NEWS, Impact, NewsInputs, ScoreLine

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "v2" / "s2_news.toml"
SIGNALS = (0.0, 0.05, 0.1)
# How the price absorbs an item, at the pace the input decays or fast, and the scenario with no planted news.
MATCHED, FAST, NONE = "matched", "fast", "none"
ABSORPTIONS = (MATCHED, FAST)
S1, S2 = 0, 1
MIN_SESSIONS = 3  # history, then training, then a third session with forecasts
# Seed streams, so prices, news and the calibration draws never share a generator.
PRICE_STREAM, NEWS_STREAM, CALIBRATION_STREAM = 0, 1, 2
LINE_SOURCE, LINE_KIND = "synthetic", "news"


@dataclass(frozen=True)
class Generator:
    """Planning values of the synthetic news, set by reasoning before launch, the arrival values those of
    scripts/v2_power.py."""

    impacts_per_session: float = 4.0  # new items per tag per session day
    session_share: float = 0.4  # arriving inside the session, the rest overnight and counting from the open
    neutral_share: float = 0.5  # direction 0, strength 0 with full weight in the denominator
    strength_scale: float = 1.0  # every strength times this, a check that input size barely moves the curve
    fast_half_life: float = 15.0  # session minutes, how fast the price absorbs an item under fast absorption
    prices: lc.Generator = field(default_factory=lc.Generator)


@dataclass(frozen=True)
class Settings:
    """What one check covers, the defaults being the full check run before launch."""

    sessions: int = 40  # from the launch, the length of the evaluation window
    seeds: int = 24
    learners: int = 24  # per run, the population of section 10.1
    calibration_seeds: int = 8  # news draws that only set beta
    signals: tuple[float, ...] = SIGNALS
    absorptions: tuple[str, ...] = ABSORPTIONS
    seed: int = 12345
    launch: date = lc.LAUNCH
    generator: Generator = field(default_factory=Generator)


@dataclass(frozen=True)
class Draw:
    """Everything one seed draws before any signal, on the minute axis of k minus the first k. Raw features are
    shaped (minute, symbol, feature), the one minute moves without drift and the planted predictors (minute, symbol),
    and the inputs S2's network sees (minute, symbol, company or macro)."""

    days: tuple[Session, ...]
    raw: np.ndarray
    noise: np.ndarray
    inputs: np.ndarray
    planted: dict[str, np.ndarray]
    items: int


@dataclass(frozen=True)
class Planted:
    """One seed under one scenario, its reference prices, labels in bps, NaN where the horizon would cross the close,
    and the planted predictor the drift follows, zero with no planted news."""

    days: tuple[Session, ...]
    raw: np.ndarray
    inputs: np.ndarray
    prices: np.ndarray
    labels: np.ndarray
    planted: np.ndarray

    @property
    def first_k(self) -> int:
        return self.days[0].first_k


@dataclass(frozen=True)
class Learned:
    """What the paired learners left on one seed and scenario, population forecasts in bps shaped (minute, symbol),
    the label scale at each minute, and S2's mean absolute news weight at the end of each window third."""

    s1: np.ndarray
    s2: np.ndarray
    label_sd: np.ndarray
    news_weight: tuple[float, ...]
    price_weight: float
    s1_news_zero: bool


@dataclass(frozen=True)
class Figures:
    """One seed's figures under one scenario, NaN where a statistic has no value."""

    ideal: float
    news_corr: float
    news_t: float
    s2_t: float
    s1_t: float
    part_t: float
    dm_t: float
    extraction: tuple[float, ...]
    news_weight: tuple[float, ...]
    price_weight: float
    rows: int
    s1_news_zero: bool


@dataclass(frozen=True)
class Scenario:
    """One absorption and planted correlation, its figures averaged over seeds and the shares of seeds whose news
    test and S2 against S1 test passed 2.33."""

    absorption: str
    signal: float
    ideal: float
    news_corr: float
    news_t: float
    news_passed: float
    s2_t: float
    s1_t: float
    part_t: float
    dm_t: float
    s2_beat_s1: float
    extraction: tuple[float, ...]
    news_weight: tuple[float, ...]
    price_weight: float
    rows: int  # evaluation rows judged, summed over seeds
    seeds: int
    s1_news_zero: bool


def scenarios(settings: Settings) -> list[tuple[str, float]]:
    """Every absorption and planted correlation the settings cover, with one scenario for no planted news, where
    absorption makes no difference."""
    keys = [(NONE, 0.0)] if 0.0 in settings.signals else []
    return keys + [(a, r) for a in settings.absorptions for r in settings.signals if r > 0]


def previous_close(day: Session) -> int:
    """The close of the last session before day, from which overnight items for its open may arrive."""
    earlier = [s for s in sessions.all_sessions() if s.day < day.day]
    return earlier[-1].close_ms if earlier else day.open_ms - 17 * 60 * 60_000


def news_lines(rng: np.random.Generator, days: Sequence[Session], tags: Sequence[str], gen: Generator
               ) -> list[ScoreLine]:
    """Synthetic Haiku score lines in receipt order, each one item with one impact marked new, received when scored.
    Each tag gets Poisson items inside each session and overnight before its open."""
    found = []
    before = previous_close(days[0])
    for day in days:
        for tag in tags:
            inside = rng.poisson(gen.impacts_per_session * gen.session_share)
            outside = rng.poisson(gen.impacts_per_session * (1 - gen.session_share))
            at = np.r_[rng.integers(day.open_ms, day.close_ms, inside), rng.integers(before, day.open_ms, outside)]
            direction = np.where(rng.random(len(at)) < gen.neutral_share, 0, rng.choice([-1, 1], len(at)))
            strength = draw_strengths(rng, direction) * gen.strength_scale
            found += [(int(ms), tag, float(s)) for ms, s in zip(at, strength)]
        before = day.close_ms
    found.sort()
    return [ScoreLine(seq, seq, LINE_SOURCE, LINE_KIND, LLM, HAIKU_SCORER, "scored", ms, (tag,),
                      (Impact(tag, strength, True),), ms) for seq, (ms, tag, strength) in enumerate(found, 1)]


def decayed(lines: Sequence[ScoreLine], days: Sequence[Session], symbols: Sequence[str], index_symbol: str,
            half_life: float, drop_after_half_lives: float) -> np.ndarray:
    """Each symbol's company and macro inputs of the primary stream at every boundary of days, shaped (minute,
    symbol, 2), from the stock process's own NewsInputs fed every line received before the boundary."""
    news = NewsInputs(symbols, index_symbol, half_life, drop_after_half_lives)
    first = days[0].first_k
    out = np.zeros((days[-1].last_k - first + 1, len(symbols), N_NEWS_INPUTS))
    j = 0
    for day in days:
        for minute in range(1, day.minutes + 1):
            boundary = day.boundary(minute)
            while j < len(lines) and lines[j].recv_ms < boundary:
                news.add(lines[j])
                j += 1
            primary = news.step(boundary)[NEWS]
            out[day.first_k - first + minute - 1] = [(primary[s].company, primary[s].macro) for s in symbols]
    return out


def news_series(seed: np.random.SeedSequence, cfg: StockConfig, days: tuple[Session, ...], gen: Generator,
                absorptions: Sequence[str]) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """One seed's news, the inputs S2's network sees, the planted predictor of each absorption and the item count."""
    symbols, index = cfg.stock_feed.symbols, cfg.stock_feed.index_symbol
    lines = news_lines(np.random.default_rng(seed), days, (*symbols, MACRO), gen)
    half_life, drops = cfg.news_input.half_life_session_minutes, cfg.news_input.drop_after_half_lives
    inputs = decayed(lines, days, symbols, index, half_life, drops)
    planted = {MATCHED: inputs.sum(axis=-1)} if MATCHED in absorptions else {}
    if FAST in absorptions:  # the same items absorbed on a shorter half life, dropped at the same age
        fast = decayed(lines, days, symbols, index, gen.fast_half_life, half_life * drops / gen.fast_half_life)
        planted[FAST] = fast.sum(axis=-1)
    return inputs, planted, len(lines)


def price_series(seed: np.random.SeedSequence, symbols: Sequence[str], index_symbol: str, minutes: int, horizon: int,
                 gen: lc.Generator) -> tuple[np.ndarray, np.ndarray]:
    """The learner check's raw features with no price signal, every feature spread over the persistences, and the one
    minute moves in bps with a common market factor and no drift."""
    rng = np.random.default_rng(seed)
    count = len(symbols)
    minute_sd = np.array([lc.SD_BPS[s] for s in symbols]) / math.sqrt(horizon)
    share = lc.market_shares(symbols, index_symbol, gen)
    phis = np.linspace(gen.persistence_low, gen.persistence_high, N_PRICE_FEATURES)
    own = lc.ar1(rng, minutes, (count, N_PRICE_FEATURES), phis)
    market = lc.ar1(rng, minutes, (N_PRICE_FEATURES,), phis)
    w = np.sqrt(share)[None, :, None]
    raw = w * market[:, None, :] + np.sqrt(1.0 - w * w) * own
    noise = minute_sd * (np.sqrt(share) * rng.normal(size=(minutes, 1))
                         + np.sqrt(1.0 - share) * rng.normal(size=(minutes, count)))
    return raw, noise


def make_draw(cfg: StockConfig, settings: Settings, days: tuple[Session, ...], seed: int) -> Draw:
    """One seed's draw, prices and news from their own seed streams so one seed gives the same draws in every
    scenario."""
    symbols, gen = cfg.stock_feed.symbols, settings.generator
    minutes = days[-1].last_k - days[0].first_k + 1
    raw, noise = price_series(np.random.SeedSequence([settings.seed, PRICE_STREAM, seed]), symbols,
                              cfg.stock_feed.index_symbol, minutes, cfg.time.horizon_minutes, gen.prices)
    inputs, planted, items = news_series(np.random.SeedSequence([settings.seed, NEWS_STREAM, seed]), cfg, days, gen,
                                         settings.absorptions)
    return Draw(days, raw, noise, inputs, planted, items)


def horizon_sums(series: np.ndarray, days: Sequence[Session], horizon: int) -> np.ndarray:
    """At each minute whose horizon ends by the close, the sum of series over that minute and the next horizon minus
    one, which is the drift a label carries per unit of beta, and NaN elsewhere."""
    first = days[0].first_k
    total = np.cumsum(np.concatenate([np.zeros((1, *series.shape[1:])), series]), axis=0)
    out = np.full(series.shape, np.nan)
    for day in days:
        a, b = day.first_k - first, day.last_k - horizon - first + 1
        out[a:b] = total[a + horizon:b + horizon] - total[a:b]
    return out


def evaluation_rows(days: Sequence[Session], horizon: int) -> np.ndarray:
    """Indices on the minute axis of the evaluation minutes of sessions 2 on, 12 per full session."""
    first = days[0].first_k
    return np.array([day.first_k - first + m - 1 for day in days[1:] for m in day.evaluation_minutes(horizon)])


def window_thirds(days: Sequence[Session]) -> list[np.ndarray]:
    """The minutes of sessions 2 on split into three contiguous thirds."""
    first = days[0].first_k
    return np.array_split(np.arange(days[1].first_k - first, days[-1].last_k - first + 1), 3)


def signal_beta(r: float, planted: np.ndarray, sums: np.ndarray, sd_bps: float) -> float:
    """The drift weight that gives the planted predictor a correlation of r with labels whose noise has a standard
    deviation of sd_bps, from the predictor and its horizon sums on matching rows."""
    if r == 0:
        return 0.0
    var_p, var_d = planted.var(), sums.var()
    cov = float(np.mean((planted - planted.mean()) * (sums - sums.mean())))
    reach = cov * cov - r * r * var_p * var_d
    if reach <= 0:
        raise ValueError(f"a planted correlation of {r} is out of reach, the most is "
                         f"{abs(cov) / math.sqrt(var_p * var_d):.3f}")
    return r * sd_bps * math.sqrt(var_p) / math.sqrt(reach)


def calibration_part(cfg: StockConfig, settings: Settings, days: tuple[Session, ...], j: int
                     ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """One calibration draw's planted predictor and horizon sums at the evaluation minutes, per absorption."""
    _, planted, _ = news_series(np.random.SeedSequence([settings.seed, CALIBRATION_STREAM, j]), cfg, days,
                                settings.generator, settings.absorptions)
    rows, horizon = evaluation_rows(days, cfg.time.horizon_minutes), cfg.time.horizon_minutes
    return {a: (p[rows], horizon_sums(p, days, horizon)[rows]) for a, p in planted.items()}


def betas_from(cfg: StockConfig, settings: Settings, parts: Sequence[dict[str, tuple[np.ndarray, np.ndarray]]]
               ) -> dict[tuple[str, float], np.ndarray]:
    """Each scenario's per symbol beta from the calibration draws pooled, zero with no planted news."""
    symbols = cfg.stock_feed.symbols
    out = {key: np.zeros(len(symbols)) for key in scenarios(settings) if key[0] == NONE}
    for absorption in settings.absorptions:
        p = np.concatenate([part[absorption][0] for part in parts])
        d = np.concatenate([part[absorption][1] for part in parts])
        for r in (r for r in settings.signals if r > 0):
            out[absorption, r] = np.array([signal_beta(r, p[:, s], d[:, s], lc.SD_BPS[name])
                                           for s, name in enumerate(symbols)])
    return out


def plant(draw: Draw, absorption: str, beta: np.ndarray, horizon: int) -> Planted:
    """The draw with each minute's move drifting by beta times the planted predictor at the boundary before it.
    Labels are the moves of the next horizon minutes, set only where the horizon ends by the close."""
    planted = draw.planted[absorption] if absorption != NONE else np.zeros(draw.noise.shape)
    move = draw.noise.copy()
    move[1:] += beta * planted[:-1]
    level = np.cumsum(move, axis=0)
    labels = np.full(level.shape, np.nan)
    first = draw.days[0].first_k
    for day in draw.days:
        a, b = day.first_k - first, day.last_k - horizon - first + 1
        labels[a:b] = level[a + horizon:b + horizon] - level[a:b]
    return Planted(draw.days, draw.raw, draw.inputs, lc.START_PRICE * np.exp(level / 1e4), labels, planted)


def price_only(rows: np.ndarray) -> np.ndarray:
    """The rows with their news inputs at zero, as S1 holds them."""
    out = np.array(rows, dtype=float)
    out[..., N_PRICE_FEATURES:] = 0.0
    return out


def paired_minutes(cfg: StockConfig, data: Planted) -> Iterator[tuple[int, StockMinute, StockMinute]]:
    """The shared market layer of the learner check at every minute from session 2 on, handing S1 and S2 their own
    rows and PooledWindow as one stock process feeds its two engines. S2's rows carry the news inputs of the minute and
    each record keeps the inputs of its push minute, while S1's carry zeros."""
    symbols, horizon = cfg.stock_feed.symbols, cfg.time.horizon_minutes
    count = len(symbols)
    queues = [SessionLabelQueue(horizon) for _ in symbols]
    scales = [make_label_scale(cfg.features) for _ in symbols]
    standardizers = [make_standardizer(cfg.features) for _ in symbols]
    windows = (PooledWindow.from_config(cfg), PooledWindow.from_config(cfg))
    every = np.ones(count, dtype=bool)
    for epoch, day in enumerate(data.days[1:], 1):  # session 1 only records history, and each close bumps the epoch
        for k in range(day.first_k, day.last_k + 1):
            i = k - data.first_k
            matured = []
            for s, symbol in enumerate(symbols):
                sample = queues[s].mature(k, epoch, float(data.prices[i, s]))
                if sample is not None:
                    windows[S1].add(k, symbol, price_only(sample.z), sample.label_bps)
                    windows[S2].add(k, symbol, sample.z, sample.label_bps)
                    scales[s].update(sample.label_bps)
                    matured.append((s, push_k(sample), sample.label_bps))
            rows = np.array([network_inputs(standardizers[s].transform_then_update(data.raw[i, s]), data.inputs[i, s])
                             for s in range(count)])
            for s in range(count):
                queues[s].push(k, epoch, rows[s], float(data.prices[i, s]))
            label_sd = np.array([scale.scale if scale.ready else np.nan for scale in scales])
            ready = all(scale.ready for scale in scales)
            boundary = day.boundary(k - day.first_k + 1)
            yield i, *(StockMinute(boundary, k, day, x, every, every, True, data.prices[i], label_sd, every,
                                   tuple(matured), window if ready else None)
                       for x, window in ((price_only(rows), windows[S1]), (rows, windows[S2])))


def newborn_pairs(cfg: StockConfig, learners: int, first_bot_id: int, born_ms: int) -> tuple[list[StockBot], ...]:
    """S1's and S2's learners, the same bot ids and slots in both, so each pair is born alike and draws the same
    batches until news parts them."""
    return tuple([StockBot.newborn(cfg, first_bot_id + j, j, born_ms) for j in range(learners)] for _ in (S1, S2))


def mean_abs(bots: Sequence[StockBot], news: bool) -> float:
    """The learners' mean absolute first layer weight on the news inputs, or on the price inputs."""
    part = slice(N_PRICE_FEATURES, None) if news else slice(0, N_PRICE_FEATURES)
    return float(np.mean([np.abs(b.model.W1[:, part]).mean() for b in bots]))


def run_pair(cfg: StockConfig, data: Planted, learners: int, first_bot_id: int) -> Learned:
    """S1's and S2's learners on one planted run, minute by minute as the shared market layer and the bots do, each
    run's population forecast being the mean of its learners' predictions."""
    runs = newborn_pairs(cfg, learners, first_bot_id, data.days[0].open_ms)
    shape = data.prices.shape
    pops = (np.full(shape, np.nan), np.full(shape, np.nan))
    label_sd = np.full(shape, np.nan)
    ends = {int(third[-1]) for third in window_thirds(data.days)}
    weights = []
    for i, *contexts in paired_minutes(cfg, data):
        label_sd[i] = contexts[S2].label_sd
        for run, ctx in enumerate(contexts):
            for bot in runs[run]:
                lc.learn_and_predict(bot, ctx)
            pops[run][i] = np.mean([bot.preds for bot in runs[run]], axis=0)
        if i in ends:
            weights.append(mean_abs(runs[S2], news=True))
    return Learned(pops[S1], pops[S2], label_sd, tuple(weights), mean_abs(runs[S2], news=False),
                   not any(b.model.W1[:, N_PRICE_FEATURES:].any() for b in runs[S1]))


def finite_rows(rows: np.ndarray, *matrices: np.ndarray) -> tuple[np.ndarray, ...]:
    """The minute, symbol and values of every row of matrices at rows where every one is finite, as evaluate holds
    forecast rows."""
    picked = [m[rows] for m in matrices]
    ok = np.logical_and.reduce([np.isfinite(m) for m in picked])
    minute, symbol = np.nonzero(ok)
    return (rows[minute], symbol.astype(str), *(m[ok] for m in picked))


def value(x: float | None) -> float:
    """A statistic as a float, NaN where it has no value."""
    return math.nan if x is None else float(x)


def skill(rows: np.ndarray, pred: np.ndarray, label: np.ndarray) -> stats.PooledSkill:
    """The pooled skill of section 11.2 over the finite rows."""
    return stats.pooled_skill(*finite_rows(rows, pred, label))


def extraction(learned: Learned, planted: np.ndarray, minutes: np.ndarray) -> float:
    """The mean over symbols of the correlation of S2's population forecast minus S1's, on the network's scale, with
    the planted predictor over minutes, NaN with no planted news."""
    if not planted[minutes].any():
        return math.nan
    part = (learned.s2[minutes] - learned.s1[minutes]) / learned.label_sd[minutes]
    found = []
    for s in range(planted.shape[1]):
        ok = np.isfinite(part[:, s])
        found.append(lc.correlation(part[ok, s], planted[minutes][ok, s]))
    return float(np.mean(found))


def judge(data: Planted, learned: Learned, horizon: int) -> Figures:
    """One seed's figures over the evaluation minutes of sessions 2 on, population forecasts judged where they
    exist."""
    rows, y = evaluation_rows(data.days, horizon), data.labels
    news = skill(rows, data.inputs.sum(axis=-1), y)
    dm = stats.pooled_dm(*finite_rows(rows, learned.s2, learned.s1, y))
    return Figures(
        ideal=value(skill(rows, data.planted, y).corr), news_corr=value(news.corr), news_t=value(news.t),
        s2_t=value(skill(rows, learned.s2, y).t), s1_t=value(skill(rows, learned.s1, y).t),
        part_t=value(skill(rows, learned.s2 - learned.s1, y).t), dm_t=value(dm.t),
        extraction=tuple(extraction(learned, data.planted, third) for third in window_thirds(data.days)),
        news_weight=learned.news_weight, price_weight=learned.price_weight, rows=dm.rows,
        s1_news_zero=learned.s1_news_zero)


def seed_figures(cfg: StockConfig, settings: Settings, days: tuple[Session, ...],
                 betas: dict[tuple[str, float], np.ndarray], seed: int) -> dict[tuple[str, float], Figures]:
    """Every scenario on one seed's draw, its learners distinct from every other seed's and alike across scenarios."""
    draw = make_draw(cfg, settings, days, seed)
    horizon = cfg.time.horizon_minutes
    out = {}
    for absorption, r in scenarios(settings):
        data = plant(draw, absorption, betas[absorption, r], horizon)
        out[absorption, r] = judge(data, run_pair(cfg, data, settings.learners, 1 + seed * settings.learners),
                                   horizon)
    return out


def mean_of(values: Sequence[float]) -> float:
    """The mean of the finite values, NaN when there are none."""
    finite = [v for v in values if math.isfinite(v)]
    return float(np.mean(finite)) if finite else math.nan


def passed(values: Sequence[float]) -> float:
    """The share of values above the one sided 1 percent line, NaN never passing."""
    return float(np.mean([math.isfinite(v) and v > stats.ONE_SIDED_Z for v in values]))


def aggregate(key: tuple[str, float], found: Sequence[Figures]) -> Scenario:
    """One scenario's figures over its seeds."""
    def avg(name: str) -> float:
        return mean_of([getattr(f, name) for f in found])

    def each(name: str) -> tuple[float, ...]:
        return tuple(mean_of(column) for column in zip(*(getattr(f, name) for f in found)))

    return Scenario(key[0], key[1], avg("ideal"), avg("news_corr"), avg("news_t"), passed([f.news_t for f in found]),
                    avg("s2_t"), avg("s1_t"), avg("part_t"), avg("dm_t"), passed([f.dm_t for f in found]),
                    each("extraction"), each("news_weight"), avg("price_weight"), sum(f.rows for f in found),
                    len(found), all(f.s1_news_zero for f in found))


def pool_map(workers: int, fn, args: Sequence[tuple]) -> list:
    """fn over args in order, in worker processes when workers is above 1, so the result never depends on it."""
    if workers <= 1:
        return [fn(*a) for a in args]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(fn, *zip(*args)))


def simulate(cfg: StockConfig, settings: Settings, workers: int = 1) -> list[Scenario]:
    """Every scenario over the same seeded draws, beta set first from the calibration draws."""
    days = lc.run_sessions(settings.launch, settings.sessions)
    parts = pool_map(workers, calibration_part, [(cfg, settings, days, j) for j in range(settings.calibration_seeds)])
    betas = betas_from(cfg, settings, parts)
    per_seed = pool_map(workers, seed_figures, [(cfg, settings, days, betas, seed) for seed in range(settings.seeds)])
    return [aggregate(key, [found[key] for found in per_seed]) for key in scenarios(settings)]


def figure(x: float, spec: str = ".3f") -> str:
    """A figure for the rows, blank when it has no value."""
    return format(x, spec) if math.isfinite(x) else ""


def scenario_row(s: Scenario) -> dict:
    """One row of the curve, all synthetic."""
    thirds = {f"extraction {j}": figure(x) for j, x in enumerate(s.extraction, 1)}
    weights = {f"news weight {j}": figure(x) for j, x in enumerate(s.news_weight, 1)}
    return {"absorption": s.absorption, "planted": f"{s.signal:.2f}", "ideal": figure(s.ideal, "+.3f"),
            "news corr": figure(s.news_corr, "+.3f"), "news t": figure(s.news_t, "+.2f"),
            "news passed": figure(s.news_passed, ".2f"), **thirds, "S2 t": figure(s.s2_t, "+.2f"),
            "S1 t": figure(s.s1_t, "+.2f"), "S2 minus S1 t": figure(s.part_t, "+.2f"),
            "S2 vs S1 t": figure(s.dm_t, "+.2f"), "S2 beat S1": figure(s.s2_beat_s1, ".2f"), **weights,
            "price weight": figure(s.price_weight), "rows": s.rows}


def report(cfg: StockConfig, settings: Settings, found: Sequence[Scenario], seconds: float) -> output.Report:
    """Counts, the verdict that S1's news weights stayed zero and the run time, then one row per scenario."""
    days = lc.run_sessions(settings.launch, settings.sessions)
    rows = len(evaluation_rows(days, cfg.time.horizon_minutes)) * len(cfg.stock_feed.symbols)
    return output.Report("synthetic planted news check", (
        ("sessions", output.Count(settings.sessions)),
        ("seeds", output.Count(settings.seeds)),
        ("calibration seeds", output.Count(settings.calibration_seeds)),
        ("learners per run", output.Count(settings.learners)),
        ("symbols", output.Count(len(cfg.stock_feed.symbols))),
        ("scenarios", output.Count(len(found))),
        ("evaluation rows per seed", output.Count(rows)),
        ("price only learners keep zero news weights", all(s.s1_news_zero for s in found)),
        ("run time", output.Seconds(seconds)),
    ), tuple(scenario_row(s) for s in found))


def build_parser() -> argparse.ArgumentParser:
    """The command line, its defaults the full check run before launch."""
    parser = argparse.ArgumentParser(description="Synthetic planted news check, the descriptive power curve of S2 "
                                                 "against S1. Never reads market data. Every figure is synthetic.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="the S2 run config, whose model, features and news input settings the learners use")
    parser.add_argument("--sessions", type=int, default=Settings.sessions,
                        help=f"sessions from the launch, at least {MIN_SESSIONS}, with session 1 as history only")
    parser.add_argument("--seeds", type=int, default=Settings.seeds)
    parser.add_argument("--learners", type=int, default=Settings.learners, help="learners per run on each seed")
    parser.add_argument("--calibration-seeds", type=int, default=Settings.calibration_seeds,
                        help="news draws that set beta, apart from the seeds judged")
    parser.add_argument("--strength-scale", type=float, default=Generator.strength_scale,
                        help="every item strength times this, above 0 and at most 1, a check that the size of "
                             "the inputs barely moves the curve")
    parser.add_argument("--seed", type=int, default=Settings.seed)
    parser.add_argument("--workers", type=int, default=1, help="processes, which never change a figure")
    output.add_detail_flag(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Check the settings, run every scenario and print the report."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sessions < MIN_SESSIONS or min(args.seeds, args.learners, args.calibration_seeds, args.workers) < 1:
        parser.error(f"sessions must be at least {MIN_SESSIONS} and seeds, learners, calibration seeds and workers "
                     f"at least 1")
    if not 0 < args.strength_scale <= 1:
        parser.error("the strength scale must be above 0 and at most 1")
    cfg = load_stock_config(args.config)
    if cfg.model.inputs != N_PRICE_FEATURES + N_NEWS_INPUTS:
        parser.error(f"the config's network has {cfg.model.inputs} inputs, not the price and news inputs")
    settings = Settings(sessions=args.sessions, seeds=args.seeds, learners=args.learners,
                        calibration_seeds=args.calibration_seeds, seed=args.seed,
                        generator=Generator(strength_scale=args.strength_scale))
    started = time.monotonic()
    found = simulate(cfg, settings, args.workers)
    output.emit(report(cfg, settings, found, time.monotonic() - started), detail=args.detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
