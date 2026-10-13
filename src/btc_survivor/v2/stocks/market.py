"""The shared market layer of PLAN_V2 section 4.8 step 1, run once at each session boundary for both runs.

It owns the candles, price health, the StockSeries of every symbol, the standardizers, label scales and label queues,
the evaluation labels, linear_prices, the short sale restriction and split guard, the Roll accumulators and news
health, all of which S1 and S2 share. At a boundary it judges news health for the minute, then per symbol in config
order closes the candle, takes the reference price, judges price health and appends to the series. It then computes
and standardizes the features, matures the label pushed 30 minutes before within the same epoch, updates the label
scale and linear_prices, and pushes a new label record when the minute is usable and the vector valid. The frozen
MarketMinute it returns is handed to both engines in order, so their candles, labels and linear_prices rows match.

Prices, epochs and the maturity of pending labels run on price health alone, while a new label record needs news
health too. A symbol whose healthy minutes are more than max_resume_gap_minutes session minutes apart is flagged for
a gap flatten, or a restart flatten at its first healthy minute after a restart in session. The gap counts session
minutes, so a night, weekend or holiday never flattens anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from btc_survivor.baselines import LinearBaseline
from btc_survivor.clock import MS_PER_MINUTE, MS_PER_SECOND
from btc_survivor.types import FeedNotice
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import StockConfig
from btc_survivor.v2.sessions import Session
from btc_survivor.v2.stocks.broker import SymbolGuards, fill_prints, rejection
from btc_survivor.v2.stocks.candles import StockCandle, StockCandles
from btc_survivor.v2.stocks.costs import RollAccumulator, SpreadCheck, spread_check
from btc_survivor.v2.stocks.features import N_PRICE_FEATURES, FeatureBuilder, make_standardizer
from btc_survivor.v2.stocks.finnhub import Print, PrintBatch
from btc_survivor.v2.stocks.health import CONNECTED, HEALTHY, HealthEvent, PriceHealth, news_minute
from btc_survivor.v2.stocks.labels import (EvaluationLabels, LabelRow, SessionLabelQueue, is_evaluation_minute,
                                           make_label_scale, push_k)
from btc_survivor.v2.stocks.news_inputs import HeartbeatLine
from btc_survivor.v2.stocks.prints import PriceRules
from btc_survivor.v2.stocks.series import StockSeries

# Exit reasons of the flattens after a long gap in a symbol's healthy minutes.
GAP_FLATTEN, RESTART_FLATTEN = "gap_flatten", "restart_flatten"
# Feed event kinds the layer adds to those of price health and the split guard.
NEWS_UNHEALTHY, NEWS_HEALTHY, CALENDAR_WARNING = "news_unhealthy", "news_healthy", "calendar_warning"


def pooled_half_life(cfg: StockConfig) -> float:
    """The half life in samples of a pooled ridge, the standardizer's healthy session hours times every symbol's
    sample per minute, so linear_prices and linear_news forget at the pace of each symbol's own standardizer."""
    return cfg.features.standardizer_half_life_hours * 60 * len(cfg.stock_feed.symbols)


def _frozen(values: np.ndarray) -> np.ndarray:
    """An array that neither engine can change by accident."""
    values.setflags(write=False)
    return values


class Sample(NamedTuple):
    """A training sample that matured at this boundary, its symbol index, the minute k its record was pushed, the
    price z scores stored at the push and its label in bps."""

    symbol: int
    push_k: int
    z: np.ndarray
    label_bps: float


@dataclass(frozen=True, eq=False)
class MarketMinute:
    """Everything the shared layer decided at one session boundary, over the symbols in config order.

    healthy is price health of the minute, ref the reference price where healthy and NaN elsewhere, last_ref the last
    healthy reference price ever seen, z the 10 standardized features, zero where valid is false, and label_sd the
    label scale in bps, NaN until that symbol's scale is ready. pushed names the symbols whose label record was pushed
    at k, linear holds the linear_prices forecast in bps at an evaluation minute where it exists, and flatten the gap
    or restart flatten due at a healthy minute.
    """

    boundary_ms: int
    k: int
    session: Session
    candles: tuple[StockCandle, ...]
    healthy: np.ndarray
    ref: np.ndarray
    last_ref: np.ndarray
    z: np.ndarray
    valid: np.ndarray
    label_sd: np.ndarray
    shorts_ok: np.ndarray
    news_ok: bool
    news_reason: str | None
    channels_down: tuple[str, ...]
    matured: tuple[Sample, ...]
    pushed: tuple[int, ...]
    labels: tuple[LabelRow, ...]
    linear: np.ndarray
    evaluation: bool
    flatten: tuple[str | None, ...]
    gaps: tuple[int, ...]
    spread_checks: tuple[SpreadCheck, ...]
    healthy_minutes: int
    run_minutes: int
    run_minutes_total: int

    @property
    def scales_ready(self) -> bool:
        """Whether every symbol's label scale is ready, before which no learner trains."""
        return not np.isnan(self.label_sd).any()

    @property
    def last_minute(self) -> bool:
        return self.boundary_ms == self.session.close_ms


class MarketLayer:
    """The state S1 and S2 share, fed every decoded feed event and heartbeat line in stamp order, and asked to close
    each session boundary before anything stamped at or after it."""

    def __init__(self, cfg: StockConfig, launch_hashes: tuple[str, str, str]) -> None:
        feed, features = cfg.stock_feed, cfg.features
        self.cfg = cfg
        self.symbols = feed.symbols
        self.launch_hashes = tuple(launch_hashes)
        self.rules = PriceRules.from_config(feed)
        self.features = FeatureBuilder.from_config(cfg)
        self.series = {s: StockSeries(features.series_capacity_minutes) for s in self.symbols}
        self.standardizers = {s: make_standardizer(features) for s in self.symbols}
        self.label_scales = {s: make_label_scale(features) for s in self.symbols}
        self.queues = {s: SessionLabelQueue(cfg.time.horizon_minutes) for s in self.symbols}
        self.evaluation = {s: EvaluationLabels(s, cfg.time.horizon_minutes) for s in self.symbols}
        self.linear = LinearBaseline(N_PRICE_FEATURES, pooled_half_life(cfg))
        self.guards = SymbolGuards.from_config(cfg)
        self.max_gap = cfg.persistence.max_resume_gap_minutes
        self.heartbeat_max_age_ms = round(cfg.news_input.heartbeat_max_age_seconds * MS_PER_SECOND)
        self.last_healthy_k: dict[str, int | None] = {s: None for s in self.symbols}
        self.last_ref = np.full(len(self.symbols), np.nan)
        self.session_day: str | None = None
        self.healthy_minutes = self.run_minutes = self.run_minutes_total = 0
        self.news_ok = True
        self.events: list[HealthEvent] = []
        self._fresh(restarted=False)

    def _fresh(self, restarted: bool) -> None:
        """The state a restart rebuilds instead of restoring, the candles, price health, the heartbeat lines and the
        Roll accumulators, with every symbol marked restarted after a restart in session."""
        feed = self.cfg.stock_feed
        self.candles = StockCandles(self.symbols, self.rules)
        self.health = PriceHealth.from_config(feed)
        self.rolls = {s: RollAccumulator() for s in self.symbols}
        self.restarted = {s: restarted for s in self.symbols}
        self._beats: list[HeartbeatLine] = []
        self._beat: HeartbeatLine | None = None  # the newest line before the minute being judged, in force at its start
        self._channels: tuple[str, ...] = ()

    # Lines between boundaries

    def on_event(self, ev: PrintBatch | FeedNotice) -> tuple[Print, ...]:
        """Apply one decoded feed event to the candles and price health, returning the prints of a batch that may fill
        an order, which also feed the Roll accumulators."""
        if isinstance(ev, PrintBatch):
            self.candles.add(ev)
            self.health.on_event(ev)
            prints = fill_prints(ev, self.candles.session, self.rules, self.health.connected_ms)
            for p in prints:
                self.rolls[p.symbol].add(p.price)
        else:
            if ev.kind == CONNECTED:
                self.candles.on_connected(ev.recv_ms)
            self.health.on_event(ev)
            prints = ()
        self.events.extend(self.health.drain_events())
        return prints

    def on_heartbeat(self, line: HeartbeatLine) -> None:
        """Keep a news heartbeat line for the news health of the minute it was received in. A line more than a minute
        older than this one can only be the newest before a later minute, so only the newest such line is kept."""
        while self._beats and self._beats[0].recv_ms < line.recv_ms - MS_PER_MINUTE:
            old = self._beats.pop(0)
            self._beat, self._channels = old, old.channels_down
        self._beats.append(line)

    def advance(self, now_ms: int) -> None:
        """Apply every price health silence, close and open due by now_ms."""
        self.health.advance(now_ms)
        self.events.extend(self.health.drain_events())

    def rejection(self, symbol: str) -> str | None:
        """Why a pending order on symbol must be rejected now, the session end, a dropped connection or a symbol that
        is no longer healthy, or None while it may still fill."""
        healthy = self.health.symbols[symbol].state == HEALTHY
        return rejection(healthy, self.health.connected, self.health.session is not None)

    def drain_events(self) -> list[HealthEvent]:
        """The feed events since the last call, for both run databases."""
        events, self.events = self.events, []
        return events

    def resume(self, now_ms: int) -> None:
        """After a restart from a checkpoint, rebuild what is not restored and bump every epoch, so no label pending
        before the restart can mature. A restart in session is a gap, so a symbol whose next healthy minute comes more
        than max_resume_gap_minutes after its last one gets a restart flatten, while a restart outside the session
        flattens nothing, since every position still open then is a carry closed by session_flatten."""
        epochs = {s: h.epoch + 1 for s, h in self.health.symbols.items()}
        inside = sessions.session_at(now_ms) is not None
        self._fresh(restarted=inside)
        for symbol, epoch in epochs.items():
            self.health.symbols[symbol].epoch = epoch
            if not inside:
                self.last_healthy_k[symbol] = None

    # The boundary

    def close_minute(self, boundary_ms: int) -> MarketMinute | None:
        """Step 1 of the stock minute step at a session boundary, or None at a boundary outside every session."""
        session = sessions.session_of_boundary(boundary_ms)
        if session is None:
            return None
        verdict = self.health.judge_minute(boundary_ms)
        self.events.extend(self.health.drain_events())
        raw_candles = self.candles.close(boundary_ms)
        k = session.k(boundary_ms)
        news_reason = self._news_minute(boundary_ms)
        news_ok = news_reason is None
        self._session_start(session, boundary_ms)
        n = len(self.symbols)
        healthy = np.zeros(n, dtype=bool)
        ref = np.full(n, np.nan)
        candles, flatten, gaps = [], [None] * n, [0] * n
        for i, symbol in enumerate(self.symbols):
            candle = raw_candles[i].judged(verdict[symbol])
            candles.append(candle)
            self.series[symbol].append_candle(candle)
            if candle.valid and candle.ref_price is not None:
                healthy[i], ref[i] = True, candle.ref_price
                self.last_ref[i] = candle.ref_price
                for kind, detail in self.guards.on_price(symbol, session, k, candle.ref_price):
                    self.events.append(HealthEvent(boundary_ms, kind, symbol, detail))
                flatten[i], gaps[i] = self._gap(symbol, k)
        raw = self.features.compute(self.series, k)
        evaluation = is_evaluation_minute(k, self.cfg.time.horizon_minutes)
        z = np.zeros((n, N_PRICE_FEATURES))
        valid = np.zeros(n, dtype=bool)
        label_sd = np.full(n, np.nan)
        linear = np.full(n, np.nan)
        matured, pushed, labels = [], [], []
        for i, symbol in enumerate(self.symbols):
            vector = raw[symbol]
            if vector is not None and self.guards.features_allowed(symbol, k):
                z[i] = self.standardizers[symbol].transform_then_update(vector)
                valid[i] = True
            sample = self._labels(i, symbol, k, healthy[i], ref[i], news_ok and valid[i], z[i], pushed)
            if sample is not None:
                matured.append(sample)
            scale = self.label_scales[symbol]
            if scale.ready:
                label_sd[i] = scale.scale
                if evaluation and valid[i]:
                    linear[i] = self.linear.predict(z[i]) * scale.scale
            labels.extend(self.evaluation[symbol].step(k, candles[i].ref_price, bool(healthy[i]),
                                                       self.health.symbols[symbol].epoch))
        if healthy.any():
            self.healthy_minutes += 1
            if news_ok:
                self.run_minutes += 1
                self.run_minutes_total += 1
        checks = self._spread_checks() if boundary_ms == session.close_ms else ()
        shorts_ok = np.array([self.guards.shorts_allowed(s, session.day) for s in self.symbols])
        return MarketMinute(
            boundary_ms, k, session, tuple(candles), _frozen(healthy), _frozen(ref), _frozen(self.last_ref.copy()),
            _frozen(z), _frozen(valid), _frozen(label_sd), _frozen(shorts_ok), news_ok, news_reason, self._channels,
            tuple(matured), tuple(pushed), tuple(labels), _frozen(linear), evaluation, tuple(flatten), tuple(gaps),
            checks, self.healthy_minutes, self.run_minutes, self.run_minutes_total)

    def _labels(self, i: int, symbol: str, k: int, healthy: bool, ref: float, pushable: bool, z: np.ndarray,
                pushed: list[int]) -> Sample | None:
        """Mature the record pushed at k minus the horizon within the same epoch, update the label scale and
        linear_prices with it, and push a record at k when the minute is usable with a valid vector."""
        queue, epoch = self.queues[symbol], self.health.symbols[symbol].epoch
        queue.drop_other_epochs(epoch)
        done = queue.mature(k, epoch, ref if healthy else None)
        sample = None
        if done is not None:
            scale = self.label_scales[symbol]
            scale.update(done.label_bps)
            if scale.ready:
                self.linear.update(done.z, done.label_bps / scale.scale)
            sample = Sample(i, push_k(done), done.z, done.label_bps)
        if healthy and pushable and queue.push(k, epoch, z.copy(), ref):
            pushed.append(i)
        return sample

    def _gap(self, symbol: str, k: int) -> tuple[str | None, int]:
        """The flatten due at a healthy minute k, after more than max_resume_gap_minutes session minutes since the
        symbol's last healthy minute, and that gap."""
        last, self.last_healthy_k[symbol] = self.last_healthy_k[symbol], k
        restarted, self.restarted[symbol] = self.restarted[symbol], False
        gap = 0 if last is None else k - last
        if gap <= self.max_gap:
            return None, gap
        return (RESTART_FLATTEN if restarted else GAP_FLATTEN), gap

    def _news_minute(self, boundary_ms: int) -> str | None:
        """News health of the minute ending at boundary_ms from the heartbeat lines received in it and the newest one
        before it, which is judged whole since it was in force at the minute's start, writing a feed event whenever it
        changes."""
        start = boundary_ms - MS_PER_MINUTE
        before = [line for line in self._beats if line.recv_ms < start]
        inside = [line for line in self._beats if line.recv_ms >= start]
        if before:
            self._beat, self._channels = before[-1], before[-1].channels_down
        previous = None if self._beat is None else self._beat.heartbeat(self.launch_hashes)
        reason = news_minute(boundary_ms, previous, [line.heartbeat(self.launch_hashes) for line in inside],
                             self.heartbeat_max_age_ms)
        if inside:
            self._beat, self._channels = inside[-1], inside[-1].channels_down
        self._beats = []
        if (reason is None) != self.news_ok:
            self.news_ok = reason is None
            self.events.append(HealthEvent(boundary_ms, NEWS_HEALTHY if self.news_ok else NEWS_UNHEALTHY, None,
                                           reason or ""))
        return reason

    def _session_start(self, session: Session, boundary_ms: int) -> None:
        """At the first boundary a session reaches, reset its minute counts and write any calendar warning."""
        day = session.day.isoformat()
        if day == self.session_day:
            return
        self.session_day = day
        self.healthy_minutes = self.run_minutes = 0
        warning = sessions.calendar_warning(session.day, self.cfg.session.calendar_warn_days)
        if warning is not None:
            self.events.append(HealthEvent(boundary_ms, CALENDAR_WARNING, None, warning))

    def _spread_checks(self) -> tuple[SpreadCheck, ...]:
        """The session's Roll diagnostic per symbol at its last boundary, then fresh accumulators for the next."""
        checks = tuple(spread_check(self.cfg.stock_costs, s, self.rolls[s]) for s in self.symbols)
        self.rolls = {s: RollAccumulator() for s in self.symbols}
        return checks
