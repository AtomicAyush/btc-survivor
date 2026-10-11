"""The deterministic core shared by live and replay.

The engine never reads the wall clock. It learns that time has passed through advance_to and on_line, runs each
minute boundary and each persistence point in time order, settles paper orders against the local book, and
queues every row. The same messages therefore give the same candles and the same bot and ledger rows live and in
replay. Only feed event stamps, and live the population's birth time, follow the wall clock.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import deque

import numpy as np

from btc_survivor import checkpoint, records
from btc_survivor.baselines import LinearBaseline
from btc_survivor.benchmark import Anchor, buy_hold_equity, cash_equity, matched_equity
from btc_survivor.bot import DEAD, Bot, MinuteContext, StepResult
from btc_survivor.broker import Broker, Execution
from btc_survivor.clock import MS_PER_DAY, MS_PER_MINUTE, iso, minute_index, next_minute, parse_iso
from btc_survivor.config import Config
from btc_survivor.controls import TwinBot
from btc_survivor.features import N_FEATURES, FeatureBuilder
from btc_survivor.feed.base import FeedAdapter, decode
from btc_survivor.feed.candles import MinuteSeries
from btc_survivor.feed.market import Market
from btc_survivor.labels import LabelQueue, Matured, MaturedWindow
from btc_survivor.population import Population
from btc_survivor.standardizer import LabelScale, Standardizer
from btc_survivor.store import SCHEMA_VERSION, Store
from btc_survivor.types import BookDelta, BookSnapshot, FeedEvent, FeedNotice, Leg, Order

log = logging.getLogger(__name__)

PERSIST_OFFSET_MS = 30_000  # persistence points sit 30 s after a boundary, away from boundaries and fills
SLOW_STEP_SECONDS = 10.0


class EngineError(RuntimeError):
    """Raised when existing history cannot be resumed safely."""


class Engine:
    def __init__(self, cfg: Config, store: Store, adapter: FeedAdapter, population: Population,
                 start_ms: int) -> None:
        self.cfg, self.store, self.adapter, self.population = cfg, store, adapter, population
        feed = cfg.feed
        self.market = Market(feed.book_depth_levels, int(feed.stale_after_seconds * 1000),
                             int(feed.max_clock_skew_seconds * 1000))
        history = cfg.time.history_minutes_required
        self.series = MinuteSeries(history + 1)
        self.features = FeatureBuilder(cfg.features, history)
        half_life = cfg.features.standardizer_half_life_hours * 60
        self.standardizer = Standardizer(N_FEATURES, half_life, cfg.features.min_samples, cfg.features.clip_z)
        self.label_scale = LabelScale(half_life, cfg.features.min_samples)
        self.labels = LabelQueue(cfg.time.horizon_minutes)
        self.window = MaturedWindow(cfg.model.replay_window_minutes, N_FEATURES)
        self.linear = LinearBaseline(N_FEATURES, half_life)
        self.broker = Broker(cfg.costs, cfg.trading.notional)
        self.pending: dict[int, tuple[int, Bot]] = {}  # bot id to (send order, bot) for every unsettled order
        self._sent = 0
        self.dying: list[Bot] = []
        self.now_ms = start_ms
        self.next_boundary = next_minute(start_ms)
        self.next_persist = self.next_boundary + PERSIST_OFFSET_MS
        self.last_healthy_ms: int | None = None
        self.healthy_minutes = 0
        self.deaths = 0
        self.anchor: Anchor | None = None
        self.restarted = False
        self.flatten_owed: str | None = None  # a flatten close was rejected, so it is sent again
        self.last_checkpoint_ts: str | None = None
        self.feed_rows: list[tuple] = []
        self._decided: deque[tuple[int, list[int]]] = deque()  # decision minutes still waiting for their label
        self._stored_state: dict[int, str] = {}

    # Construction and resume

    @classmethod
    def fresh(cls, cfg: Config, store: Store, adapter: FeedAdapter, start_ms: int) -> "Engine":
        """A new population from the master seed, born at start_ms."""
        engine = cls(cfg, store, adapter, Population.fresh(cfg, start_ms), start_ms)
        store.queue_meta("master_seed", str(cfg.run.master_seed))
        store.queue_meta("config_toml", cfg.text)
        for bot in engine.slot_bots():
            engine._insert_bot(bot)
        return engine

    @classmethod
    def start(cls, cfg: Config, store: Store, adapter: FeedAdapter, start_ms: int) -> "Engine":
        """Resume from the newest checkpoint that reads back, or start fresh on an empty database."""
        rows = store.query("SELECT id, ts, meta, arrays FROM checkpoints ORDER BY id DESC")
        for i, row in enumerate(rows):
            try:
                meta, arrays = checkpoint.decode(row["meta"], row["arrays"])
            except ValueError as exc:
                log.error("checkpoint %d from %s cannot be read (%s), trying an older one", row["id"], row["ts"], exc)
                continue
            note = ""
            if i > 0:
                horizon_ms = cfg.time.horizon_minutes * MS_PER_MINUTE
                store.rollback_after(row["ts"], iso(parse_iso(row["ts"]) - horizon_ms))
                note = f", after {i} newer checkpoints could not be read and later rows were rolled back"
            engine = cls._restore(cfg, store, adapter, start_ms, meta, arrays, row["ts"], note)
            if i > 0:
                engine.persist(engine.now_ms)  # commit the repaired bot states and the reason at once
            return engine
        if store.query("SELECT 1 FROM bots LIMIT 1"):
            raise EngineError("the database holds bots but no readable checkpoint, refusing to seed a new "
                              "population over old history")
        return cls.fresh(cfg, store, adapter, start_ms)

    @classmethod
    def _restore(cls, cfg: Config, store: Store, adapter: FeedAdapter, start_ms: int, meta: dict,
                 arrays: dict[str, np.ndarray], ckpt_ts: str, note: str) -> "Engine":
        _check_same_run(cfg, store, meta)
        bots = {d["bot_id"]: checkpoint.load_bot(cfg, d, arrays) for d in meta["bots"]}
        population = Population(cfg, [bots[i] for i in meta["learners"]],
                                [None if i is None else bots[i] for i in meta["twins"]],
                                meta["next_bot_id"], checkpoint.rng_from_state(meta["population_rng"]))
        engine = cls(cfg, store, adapter, population, max(start_ms, meta["now_ms"]))
        engine.dying = [bots[i] for i in meta["dying"]]
        engine.broker.next_trade_id = meta["next_trade_id"]
        engine.last_healthy_ms = meta["last_healthy_ms"]
        engine.healthy_minutes, engine.deaths = meta["healthy_minutes"], meta["deaths"]
        engine.anchor = None if meta["anchor"] is None else Anchor(*meta["anchor"])
        engine.market.health.epoch = meta["epoch"] + 1  # a restart is a gap, so no pending label survives it
        engine.restarted = True
        engine.flatten_owed = meta["flatten_owed"]
        engine.last_checkpoint_ts = ckpt_ts
        _load_stats(engine.standardizer.stats, meta["standardizer_n"], arrays, "standardizer")
        _load_stats(engine.label_scale.stats, meta["label_scale_n"], arrays, "label_scale")
        engine.window.z, engine.window.label = arrays["window.z"], arrays["window.label"]
        engine.window.minute = arrays["window.minute"].astype(np.int64)
        engine.window.count, engine.window.head = meta["window_count"], meta["window_head"]
        engine.linear.n, engine.linear.weight = meta["linear_n"], meta["linear_weight"]
        engine.linear.xx, engine.linear.xy = arrays["linear.xx"], arrays["linear.xy"]
        engine.linear.coef = arrays["linear.coef"]
        engine._reload_series(meta["valid_total"])
        engine._stored_state = {r["bot_id"]: r["state"] for r in store.query("SELECT bot_id, state FROM bots")}
        for bot in engine.all_bots():
            if bot.pending_order is not None:
                engine._register(bot)
        engine._reject_all("restart", engine.now_ms)
        engine._event(engine.now_ms, "resume", f"restored the checkpoint from {ckpt_ts}{note}")
        return engine

    def _reload_series(self, valid_total: int) -> None:
        """Rebuild the minute series from the candles table, which is committed with every checkpoint."""
        rows = self.store.query("SELECT ts, mid, volume, valid FROM candles ORDER BY ts DESC LIMIT ?",
                                (self.series.capacity,))[::-1]
        reloaded = sum(1 for r in rows if r["valid"] and r["mid"] is not None)
        self.series.valid_total = max(valid_total - reloaded, 0)
        for r in rows:
            self.series.append(parse_iso(r["ts"]), r["mid"], r["volume"], bool(r["valid"]))

    # Bot collections

    def slot_bots(self) -> list[Bot]:
        """Every bot in a slot, learners first, then twins."""
        return self.population.learners + [t for t in self.population.twins if t is not None]

    def all_bots(self) -> list[Bot]:
        return self.slot_bots() + self.dying

    # Time

    def on_line(self, kind: str, raw: str, recv_ms: int, conn_id: int) -> None:
        """One recorded or live line. An unreadable message counts as a feed error, which forces a resync."""
        self.advance_to(recv_ms)
        try:
            events = decode(self.adapter, kind, raw, recv_ms, conn_id)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            events = [FeedNotice(recv_ms, conn_id, "error", detail=f"unreadable message, {exc}"[:200])]
        for event in events:
            self.on_event(event)

    def on_event(self, ev: FeedEvent) -> None:
        """Apply one event, then settle due orders if it was a book update, or reject them if health was lost."""
        self.advance_to(ev.recv_ms)
        self.market.on_event(ev)
        if self.pending:
            if not self.market.health.is_healthy:
                self._reject_all("feed_not_healthy", ev.recv_ms)
            elif isinstance(ev, (BookSnapshot, BookDelta)) and self.market.book.valid:
                self._settle(ev.recv_ms)
        self._drain_health()

    def advance_to(self, t: int) -> None:
        """Run every minute boundary and persistence point at or before t, in time order."""
        t = max(t, self.now_ms)
        while True:
            if self.next_boundary <= self.next_persist:
                if self.next_boundary > t:
                    break
                self._minute(self.next_boundary)
                self.next_boundary += MS_PER_MINUTE
            else:
                if self.next_persist > t:
                    break
                self._persistence_point(self.next_persist)
                self.next_persist += MS_PER_MINUTE
        self.now_ms = t
        self.market.advance(t)
        if self.pending and not self.market.health.is_healthy:
            self._reject_all("feed_not_healthy", t)
        self._drain_health()

    # The minute step

    def _minute(self, boundary: int) -> None:
        started = time.monotonic()
        candle, healthy = self.market.close_minute(boundary)
        self.series.append(boundary, candle.mid, candle.volume, healthy)
        self.store.queue(records.INSERT_CANDLE, records.candle_row(candle))
        if self.pending and not self.market.health.is_healthy:
            self._reject_all("feed_not_healthy", boundary)
        self._drain_health()
        if healthy:
            self._healthy_minute(boundary, candle.mid)
        elapsed = time.monotonic() - started
        if elapsed > SLOW_STEP_SECONDS:
            log.warning("slow minute step at %s took %.1f s", iso(boundary), elapsed)

    def _healthy_minute(self, boundary: int, mid: float) -> None:
        cfg, book, pop = self.cfg, self.market.book, self.population
        z, matured = self._shared_work(boundary, mid)
        self._gap_flatten(boundary)
        self.last_healthy_ms = boundary
        self.healthy_minutes += 1
        if self.anchor is None:
            self.anchor = Anchor(boundary, book.best_ask, mid)
            self.store.queue_meta("launch_ts", iso(boundary))
            self.store.queue_meta("launch_mid", repr(mid))
            self.store.queue_meta("launch_ask", repr(book.best_ask))
        for bot in self.dying:
            if bot.account.is_open and bot.pending_order is None:
                self._send(bot, [Leg("close", bot.account.side)], boundary, "death")
        ctx = MinuteContext(boundary, book, z, matured, self.window, self.label_scale)
        died: list[Bot] = []
        for slot in range(cfg.population.size):
            learner, twin = pop.learners[slot], pop.twins[slot]
            res = learner.step(ctx, self.broker)
            self._take(learner, res, died)
            if res.decision is not None:
                self._decision(res.decision)
            if twin is None:
                continue
            self._take(twin, twin.step(ctx, self.broker), died)
            if res.decision is not None:
                row = twin.mirror(res.decision, learner.bot_id, book, self.broker)
                if row is not None:
                    if row.pop("order", None) is not None:
                        self._register(twin)
                    self._decision(row)
            # A twin never holds a position its learner does not, so a rejected copy is retried here too.
            learner_out = not learner.account.is_open and learner.pending_order is None
            if res.stopped or res.died or learner_out:
                if twin.copy_exit(boundary, book, self.broker) is not None:
                    self._register(twin)
        self._forecasts(boundary, z)
        for bot in died:
            self._on_death(bot, boundary)
        if minute_index(boundary) % (cfg.persistence.equity_snapshot_every_seconds // 60) == 0:
            self._snapshots(boundary)
        self._benchmark(boundary, mid)

    def _shared_work(self, boundary: int, mid: float) -> tuple[np.ndarray | None, Matured | None]:
        """Features, z scores and the matured label, computed once for every bot."""
        epoch = self.market.health.epoch
        self.labels.drop_other_epochs(epoch)
        raw = self.features.compute(self.series, boundary, self.market.book)
        z = self.standardizer.transform_then_update(raw) if raw is not None else None
        matured = self.labels.mature(boundary, epoch, mid)
        if matured is not None:
            self.window.add(matured)
            self.label_scale.update(matured.label_bps)
            self.linear.update(matured.z, matured.label_bps)
            self._write_labels(matured)
        if z is not None:
            self.labels.push(boundary, epoch, z, mid)
        return z, matured

    def _forecasts(self, boundary: int, z: np.ndarray | None) -> None:
        """One row per predictor for the success criteria: the living learners' mean prediction and the linear baseline."""
        if z is None or not self.label_scale.ready:
            return
        preds = [b.pending_preds[boundary] for b in self.population.learners if boundary in b.pending_preds]
        if not preds:
            return
        ts = iso(boundary)
        self.store.queue(records.INSERT_FORECAST, (ts, "population", float(np.mean(preds))))
        self.store.queue(records.INSERT_FORECAST, (ts, "linear", self.linear.predict(z)))

    def _gap_flatten(self, boundary: int) -> None:
        """After a restart or outage longer than the limit, close every open position, as the market risk is the same.

        A flatten that was rejected before it filled is sent again at the next healthy minute, whatever that gap.
        """
        reason = "restart_flatten" if self.restarted else "gap_flatten"
        owed, self.restarted, self.flatten_owed = self.flatten_owed, False, None
        gap = None if self.last_healthy_ms is None else boundary - self.last_healthy_ms
        long_gap = gap is not None and gap > self.cfg.persistence.max_resume_gap_minutes * MS_PER_MINUTE
        if not long_gap:
            if owed is None:
                return
            reason = owed
        closed = 0
        for bot in self.slot_bots():
            if bot.account.is_open and bot.pending_order is None:
                self._send(bot, [Leg("close", bot.account.side)], boundary, reason)
                closed += 1
        why = f"{gap / MS_PER_MINUTE:.1f} minutes since the last healthy minute" if long_gap else \
            "the previous flatten was rejected"
        self._event(boundary, "resume", f"{why}, {reason} sent {closed} closing orders")

    def _take(self, bot: Bot, res: StepResult, died: list[Bot]) -> None:
        if res.orders:
            self._register(bot)
        if res.died:
            died.append(bot)

    def _decision(self, row: dict) -> None:
        self.store.queue(records.INSERT_DECISION, records.decision_row(row))
        if self._decided and self._decided[-1][0] == row["ts"]:
            self._decided[-1][1].append(row["bot_id"])
        else:
            self._decided.append((row["ts"], [row["bot_id"]]))

    def _write_labels(self, matured: Matured) -> None:
        """Copy the matured label onto the forecast and decision rows of its minute, for the success criteria."""
        self.store.queue(records.SET_FORECAST_LABEL, (matured.label_bps, iso(matured.minute_ms)))
        while self._decided and self._decided[0][0] < matured.minute_ms:
            self._decided.popleft()
        if self._decided and self._decided[0][0] == matured.minute_ms:
            _, bot_ids = self._decided.popleft()
            for bot_id in bot_ids:
                self.store.queue(records.SET_LABEL, (matured.label_bps, bot_id, iso(matured.minute_ms)))

    # Orders

    def _send(self, bot: Bot, legs: list[Leg], now: int, reason: str) -> None:
        bot.pending_order = self.broker.make_order(bot.bot_id, legs, now, self.market.book.mid, reason, bot.rng)
        self._register(bot)

    def _register(self, bot: Bot) -> None:
        self._sent += 1
        self.pending[bot.bot_id] = (self._sent, bot)

    def _settle(self, now: int) -> None:
        """Fill every order due by now, earliest due first, against the book as it stands."""
        due = sorted((bot.pending_order.due_ms, sent, bot) for sent, bot in self.pending.values()
                     if bot.pending_order.due_ms <= now)
        for _, _, bot in due:
            order = bot.pending_order
            self._settled(bot, order, self.broker.execute(order, self.market.book, now, bot.account), now)

    def _reject_all(self, reason: str, now: int) -> None:
        for _, bot in sorted(self.pending.values(), key=lambda item: item[0]):
            order = bot.pending_order
            self._settled(bot, order, self.broker.reject(order, reason, bot.account), now)

    def _settled(self, bot: Bot, order: Order, ex: Execution, now: int) -> None:
        del self.pending[bot.bot_id]
        bot.on_execution(ex)
        for fill in ex.fills:
            self.store.queue(records.INSERT_FILL, records.fill_row(fill))
        if ex.closed is not None:
            self.store.queue(records.CLOSE_TRADE, records.close_trade_row(ex.closed))
        if ex.opened_trade_id is not None:
            self.store.queue(records.OPEN_TRADE, records.open_trade_row(bot.bot_id, bot.account))
        if ex.rejected is not None and order.reason in ("gap_flatten", "restart_flatten") and bot.account.is_open:
            self.flatten_owed = order.reason
        if bot in self.dying and not bot.account.is_open and bot.pending_order is None:
            self._finalize(bot, now)

    # Deaths and births

    def _on_death(self, bot: Bot, boundary: int) -> None:
        """A dead bot leaves its slot at once. With a position it waits on the dying list for its close to fill."""
        self.deaths += 1
        log.info("bot %d died at %s with equity %.2f", bot.bot_id, iso(boundary), bot.equity)
        if bot.account.is_open or bot.pending_order is not None:
            self.dying.append(bot)  # an order still in flight decides whether a close is needed
        else:
            self._finalize(bot, boundary)
        if isinstance(bot, TwinBot):
            self._insert_bot(self.population.replace_twin(bot, boundary))
        else:
            child, lineage = self.population.replace(bot, boundary)
            self._insert_bot(child)
            if lineage is not None:
                self.store.queue(records.INSERT_LINEAGE, records.lineage_row(lineage))

    def _finalize(self, bot: Bot, now: int) -> None:
        bot.account.clamp_cash()
        self.store.queue(records.INSERT_DEATH, records.death_row(bot, now))
        self.store.queue(records.SET_BOT_DEAD, (iso(now), bot.bot_id))
        self._stored_state[bot.bot_id] = DEAD
        if bot in self.dying:
            self.dying.remove(bot)

    def _insert_bot(self, bot: Bot) -> None:
        self.store.queue(records.INSERT_BOT, records.bot_row(bot, self.cfg.run.master_seed))
        self._stored_state[bot.bot_id] = bot.state

    # Rows recorded each minute

    def _snapshots(self, boundary: int) -> None:
        """Equity snapshots for every living bot, and the balance checks for every bot that still has books open."""
        book = self.market.book
        for bot in self.all_bots():
            mark = bot.mark_price(book)
            errors = bot.account.invariant_errors(mark)
            if errors:
                log.error("invariant violation for bot %d: %s", bot.bot_id, " | ".join(errors))
                self._event(boundary, "invariant_violation", f"bot {bot.bot_id}: {' | '.join(errors)}")
            if bot.state != DEAD:
                self.store.queue(records.INSERT_SNAPSHOT, records.snapshot_row(bot, boundary, mark))

    def _benchmark(self, boundary: int, mid: float) -> None:
        cfg, book = self.cfg, self.market.book
        start, fee = cfg.trading.start_equity, cfg.costs.taker_fee_bps
        living = [b.equity for b in self.population.learners if b.state != DEAD]
        twins = [t.equity for t in self.population.twins if t is not None and t.state != DEAD]
        self.store.queue(records.INSERT_BENCHMARK, (
            iso(boundary), mid, buy_hold_equity(self.anchor, book.best_bid, start, fee),
            matched_equity(self.anchor, book.best_bid, start, cfg.trading.notional, fee),
            statistics.fmean(living), statistics.median(living), len(living),
            statistics.fmean(twins) if twins else None,
            cash_equity(start, cfg.costs.daily_running_cost, self.healthy_minutes)))

    def _event(self, ms: int, kind: str, detail: str) -> None:
        log.info("feed event %s %s %s", iso(ms), kind, detail)
        self.feed_rows.append((iso(ms), kind, detail))

    def _drain_health(self) -> None:
        for ms, kind, detail in self.market.health.drain_events():
            log.info("feed event %s %s %s", iso(ms), kind, detail)
            self.feed_rows.append((iso(ms), kind, detail or None))

    # Persistence

    def _persistence_point(self, at: int) -> None:
        """Feed events are committed every minute. Every checkpoint interval the rows and a checkpoint go together."""
        every = self.cfg.persistence.checkpoint_every_seconds // 60
        if minute_index(at - PERSIST_OFFSET_MS) % every == 0:
            self.persist(at)
        elif self.feed_rows:
            self.store.commit_now([(records.INSERT_FEED_EVENT, row) for row in self.feed_rows])
            self.feed_rows.clear()

    def persist(self, at: int) -> None:
        """Commit every buffered row with a checkpoint in one transaction, after proving the checkpoint reads back."""
        for bot in self.all_bots():
            state = "dying" if bot in self.dying else bot.state
            if self._stored_state.get(bot.bot_id) != state:
                self.store.queue(records.SET_BOT_STATE, (state, bot.bot_id))
                self._stored_state[bot.bot_id] = state
        for row in self.feed_rows:
            self.store.queue(records.INSERT_FEED_EVENT, row)
        self.feed_rows.clear()
        meta_text, blob = checkpoint.encode(*self._checkpoint_state(at))
        checkpoint.decode(meta_text, blob)
        ts = iso(at)
        keep, days = self.cfg.persistence.keep_checkpoints, self.cfg.persistence.keep_candles_days

        def extra(conn):
            conn.execute(records.INSERT_CHECKPOINT, (ts, SCHEMA_VERSION, meta_text, blob))
            conn.execute(records.PRUNE_CHECKPOINTS, (keep,))
            conn.execute(records.PRUNE_CANDLES, (iso(at - days * MS_PER_DAY),))

        self.store.flush(extra)
        self.last_checkpoint_ts = ts

    def finish(self, t: int | None = None) -> None:
        """Process everything up to t and write a final checkpoint, at shutdown or the end of a replay."""
        self.advance_to(self.now_ms if t is None else t)
        self.persist(self.now_ms)

    def _checkpoint_state(self, at: int) -> tuple[dict, checkpoint.Arrays]:
        arrays = checkpoint.Arrays()
        pop = self.population
        bots = [checkpoint.dump_bot(bot, arrays) for bot in self.all_bots()]
        for name, stats in (("standardizer", self.standardizer.stats), ("label_scale", self.label_scale.stats)):
            arrays.put(f"{name}.mean", stats.mean)
            arrays.put(f"{name}.var", stats.var)
        arrays.put("window.z", self.window.z)
        arrays.put("window.label", self.window.label)
        arrays.put("window.minute", self.window.minute)
        arrays.put("linear.xx", self.linear.xx)
        arrays.put("linear.xy", self.linear.xy)
        arrays.put("linear.coef", self.linear.coef)
        meta = {
            "now_ms": at, "bots": bots, "learners": [b.bot_id for b in pop.learners],
            "twins": [None if t is None else t.bot_id for t in pop.twins], "dying": [b.bot_id for b in self.dying],
            "next_bot_id": pop.next_bot_id, "population_rng": pop.rng.bit_generator.state,
            "next_trade_id": self.broker.next_trade_id, "last_healthy_ms": self.last_healthy_ms,
            "healthy_minutes": self.healthy_minutes, "deaths": self.deaths, "epoch": self.market.health.epoch,
            "anchor": None if self.anchor is None else [self.anchor.ms, self.anchor.ask, self.anchor.mid],
            "standardizer_n": self.standardizer.stats.n, "label_scale_n": self.label_scale.stats.n,
            "window_count": self.window.count, "window_head": self.window.head, "linear_n": self.linear.n,
            "linear_weight": self.linear.weight,
            "valid_total": self.series.valid_total, "flatten_owed": self.flatten_owed,
        }
        return meta, arrays


def _check_same_run(cfg: Config, store: Store, meta: dict) -> None:
    """Refuse to resume a run whose seed, population size or twin setting has changed."""
    problems = []
    if store.get_meta("master_seed") not in (None, str(cfg.run.master_seed)):
        problems.append(f"master_seed was {store.get_meta('master_seed')}, the config says {cfg.run.master_seed}")
    if len(meta["learners"]) != cfg.population.size:
        problems.append(f"population size was {len(meta['learners'])}, the config says {cfg.population.size}")
    if any(t is not None for t in meta["twins"]) != cfg.controls.coin_flip_twins:
        problems.append("coin_flip_twins differs from the run being resumed")
    if problems:
        raise EngineError("the config does not match the run in this database, " + ", ".join(problems))


def _load_stats(stats, n: int, arrays: dict[str, np.ndarray], name: str) -> None:
    stats.n, stats.mean, stats.var = n, arrays[f"{name}.mean"], arrays[f"{name}.var"]
