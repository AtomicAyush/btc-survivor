"""One stock run's minute step, PLAN_V2 section 4.8 steps 2 to 6, on the frozen MarketMinute of the shared layer.

The group drives both engines in stamp order. Before anything stamped at or after a boundary it runs every boundary
and persistence point up to that stamp, handing each session boundary's MarketMinute to on_minute after the shared
layer's feed events to on_events, and calling persistence_point 30 seconds after every boundary, inside or outside
the session. After each feed event the shared layer applied, and after every boundary and advance, it passes the new
feed events on and calls check_orders, then hands a print batch's fill prints to on_prints, and it gives every news
line to on_news. A restart calls the shared layer's resume and then each engine's resume.

At each session boundary an engine computes its two news inputs per symbol, zero in S1, stores them under the minute
of every label record the shared layer pushed, and joins each matured sample to the inputs stored at its push. It then
sends gap, restart and session flattens and the death closes of dying bots, steps each slot's learner and twin in slot
order, sends the session close orders from five minutes before the close, and buffers the rows of its own database,
candles, labels, forecasts and the price z scores of each evaluation minute, decisions, the benchmark, due snapshots and
in S2 the news minutes and every news line it read. Every decision of every learner and twin is written with its symbol
and later its label, so a reader can list and filter them all.

Orders settle on the first eligible print traded and received at or after their due time and are rejected when their
symbol leaves healthy, the connection drops or the session ends. Each fill row also records, for reports only, how long
after the due time its print traded and, in S2, whether the first line of an item bearing on its symbol, tagged with
it, with MACRO or, for the index symbol, with any company, arrived in the 30 minutes before its decision. A death
leaves its slot empty, and a bot that dies holding a position waits on the dying list until its death close fills. A position that survived a close, because its symbol was halted into it or the
process was down, closes at the symbol's first healthy minute of the next session with session_flatten and an
overnight_carry event. The engine never reads the wall clock.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Callable, Iterable
from types import SimpleNamespace

import numpy as np

from btc_survivor.baselines import LinearBaseline
from btc_survivor.benchmark import Anchor, buy_hold_equity, matched_equity
from btc_survivor.bot import DEAD
from btc_survivor.clock import MS_PER_DAY, MS_PER_MINUTE, iso, minute_index
from btc_survivor.types import LONG
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import StockConfig
from btc_survivor.v2.stocks import costs, records
from btc_survivor.v2.stocks.bot import (SESSION_CLOSE, SESSION_FLATTEN, BotStep, StockBot, StockMinute,
                                        session_close_due)
from btc_survivor.v2.stocks.broker import StockBroker, StockExecution, StockOrder, first_fill
from btc_survivor.v2.stocks.finnhub import Print
from btc_survivor.v2.stocks.health import HealthEvent
from btc_survivor.v2.stocks.learning import PooledWindow, network_inputs
from btc_survivor.v2.stocks.market import GAP_FLATTEN, RESTART_FLATTEN, MarketLayer, MarketMinute, pooled_half_life
from btc_survivor.v2.stocks.news_inputs import NEWS, STREAMS, EngineInputs, ScoreLine
from btc_survivor.v2.stocks.population import LEARNER, TWIN, StockPopulation
from btc_survivor.v2.stocks.store import SCHEMA_VERSION, StockStore
from btc_survivor.v2.stocks.twins import StockTwin, TwinStep

log = logging.getLogger(__name__)

PERSIST_OFFSET_MS = 30_000  # persistence points sit 30 s after a boundary, away from boundaries and fills
NEWS_WINDOW_MS = 30 * MS_PER_MINUTE  # the report only news window of a fill, the 30 minutes before its decision
DEATH, RESTART = "death", "restart"
# Feed event kinds the engine writes itself.
OVERNIGHT_CARRY, RESUME, INVARIANT_VIOLATION = "overnight_carry", "resume", "invariant_violation"
# Forecast predictors beside the score streams, whose names are the stream names.
POPULATION, LINEAR_PRICES, LINEAR_NEWS = "population", "linear_prices", "linear_news"
NEWS_COMPANY, NEWS_MACRO = "news_company", "news_macro"
# A line read earlier that a later line made the item's primary score, as a fallback or not.
SET_NEWS_PRIMARY = "UPDATE news_seen SET primary_use = 1, fallback = ? WHERE score_seq = ?"

Checkpointer = Callable[["StockEngine", int], tuple[str, bytes]]


def death_values(bot: StockBot, ms: int) -> tuple:
    """The version 1 deaths row of a stock bot, whose generation is always 0 because nothing evolves."""
    return records.death_row(SimpleNamespace(**{**vars(bot), "generation": 0}), ms)


class StockEngine:
    def __init__(self, cfg: StockConfig, store: StockStore, population: StockPopulation) -> None:
        self.cfg, self.store, self.population = cfg, store, population
        self.symbols = cfg.stock_feed.symbols
        self.index = self.symbols.index(cfg.stock_feed.index_symbol)
        self.broker = StockBroker.from_config(cfg)
        self.inputs = EngineInputs.from_config(cfg)
        self.window = PooledWindow.from_config(cfg)
        self.linear_news = LinearBaseline(cfg.model.inputs, pooled_half_life(cfg)) if self.inputs.use_news else None
        self.allowances = np.array([costs.allowance_bps(cfg.stock_costs, s) for s in self.symbols])
        self.rent_per_minute = costs.rent_per_minute(cfg.stock_costs)
        self.pending: dict[int, tuple[int, StockBot]] = {}  # bot id to (send order, bot) for every unsettled order
        self._sent = 0
        self.dying: list[StockBot] = []
        self.deaths = 0
        self.news_seq = 0
        self.owed: dict[str, str] = {}  # a gap or restart flatten was rejected, so it is sent again
        self.anchor: tuple[Anchor, ...] | None = None
        self.last_checkpoint_ts: str | None = None
        self.checkpointer: Checkpointer | None = None
        self.feed_rows: list[tuple] = []
        self._decided: dict[tuple[int, str], list[int]] = {}  # (k, symbol) to the bots that decided there
        self._carried: set[int] = set()  # trades whose overnight carry was written
        self._stored_state: dict[int, str] = {}

    @classmethod
    def fresh(cls, cfg: StockConfig, store: StockStore, born_ms: int) -> StockEngine:
        """A new population from the master seed, born at born_ms, with a bots row for every slot."""
        engine = cls(cfg, store, StockPopulation.fresh(cfg, born_ms))
        for bot in engine.population.slot_bots():
            engine._insert_bot(bot)
        return engine

    def all_bots(self) -> list[StockBot]:
        """Every bot in a slot, learners first, then twins, then the dying."""
        return self.population.slot_bots() + self.dying

    # Lines between boundaries

    def on_events(self, events: Iterable[HealthEvent]) -> None:
        """Buffer the shared layer's feed events for this run's database."""
        self.feed_rows.extend(records.feed_event_row(*ev) for ev in events)

    def on_news(self, line: ScoreLine) -> None:
        """Take one news line in receipt order, skipping a sequence number already read. S2 keeps every line it read
        in news_seen, marking the line each item's primary score uses, which may be one read earlier."""
        if line.seq <= self.news_seq:
            return
        self.news_seq = line.seq
        scores = self.inputs.add(line)
        if not self.inputs.use_news:
            return
        primary = [s for s in scores if s.stream == NEWS]
        own = [s for s in primary if s.score_seq == line.seq]
        self.store.queue(records.INSERT_NEWS_SEEN, records.news_seen_row(
            score_seq=line.seq, item_seq=line.item_seq, recv_ms=line.recv_ms, scored_at=iso(line.scored_ms),
            source=line.source, route=line.route, scorer=line.scorer, status=line.status, tickers=line.tickers,
            primary_use=bool(own), fallback=any(s.fallback for s in own)))
        for score in primary:
            if score.score_seq != line.seq:
                self.store.queue(SET_NEWS_PRIMARY, (int(score.fallback), score.score_seq))

    def on_prints(self, recv_ms: int, prints: tuple[Print, ...]) -> None:
        """Fill every order due by recv_ms whose symbol has an eligible print in this batch, earliest due first."""
        if not self.pending or not prints:
            return
        due = sorted((bot.pending_order.due_ms, sent, bot.bot_id) for sent, bot in self.pending.values()
                     if bot.pending_order.due_ms <= recv_ms)
        for _, _, bot_id in due:
            bot = self.pending[bot_id][1]
            order = bot.pending_order
            fill = first_fill(order, recv_ms, prints)
            if fill is not None:
                self._settled(bot, order, self.broker.execute(order, fill, recv_ms, bot.account), recv_ms)

    def news_window(self, symbol: str, decided_ms: int) -> bool | None:
        """Whether S2 received the first line of an item bearing on symbol in the 30 minutes before decided_ms, an item
        tagged with it, with MACRO or, for the index symbol, with any company, as its news inputs count them, from the
        lines those inputs keep and a checkpoint restores, or None in S1, which reads no news."""
        news = self.inputs.news
        if news is None:
            return None
        first: dict[int, int] = {}
        for line in news.kept_lines():
            if news.bears_on(symbol, line.tickers):
                first[line.item_seq] = min(line.recv_ms, first.get(line.item_seq, line.recv_ms))
        return any(decided_ms - NEWS_WINDOW_MS <= at < decided_ms for at in first.values())

    def check_orders(self, now_ms: int, market: MarketLayer) -> None:
        """Reject every pending order whose symbol left healthy, whose connection dropped or whose session ended."""
        for _, bot in sorted(self.pending.values(), key=lambda item: item[0]):
            order = bot.pending_order
            reason = market.rejection(order.symbol)
            if reason is not None:
                self._settled(bot, order, self.broker.reject(order, reason, bot.account), now_ms)

    def resume(self, now_ms: int) -> None:
        """After a restart from a checkpoint, reject every order that was in flight, since a restart is a gap,
        including the restored orders of bots the checkpoint loader did not register."""
        for bot in self.all_bots():
            if bot.pending_order is not None and bot.bot_id not in self.pending:
                self._register(bot)
        rejected = len(self.pending)
        for _, bot in sorted(self.pending.values(), key=lambda item: item[0]):
            self._settled(bot, bot.pending_order, self.broker.reject(bot.pending_order, RESTART, bot.account),
                          now_ms)
        self._event(now_ms, RESUME, None, f"resumed, {rejected} orders in flight rejected")

    # The minute step

    def on_minute(self, m: MarketMinute) -> None:
        """Steps 2 to 6 of the stock minute step at one session boundary."""
        news = self.inputs.step(m.boundary_ms)
        for i in m.pushed:
            self.inputs.push(m.k, self.symbols[i])
        self.window.evict(m.k)
        for sample in m.matured:
            self._matured(m, sample.symbol, sample.push_k, sample.z, sample.label_bps)
        rows = np.hstack([m.z, np.array([news[s] for s in self.symbols])])
        rows[~m.valid] = 0.0
        ctx = StockMinute(m.boundary_ms, m.k, m.session, rows, m.valid, m.healthy, m.news_ok, m.ref, m.label_sd,
                          m.shorts_ok, tuple((s.symbol, s.push_k, s.label_bps) for s in m.matured),
                          self.window if m.scales_ready else None)
        self._flattens(ctx, m)
        died = self._slots(ctx)
        self._session_close(ctx)
        for bot in died:
            self._on_death(bot, m.boundary_ms)
        self._rows(m, ctx)

    def _matured(self, m: MarketMinute, i: int, pushed_k: int, z: np.ndarray, label_bps: float) -> None:
        """Join a matured sample to the news inputs stored at its push, add it to the pooled window, update
        linear_news in S2 and copy the label onto the decisions made on its symbol at its push minute."""
        symbol = self.symbols[i]
        x = network_inputs(z, self.inputs.take(pushed_k, symbol))
        self.window.add(m.k, symbol, x, label_bps)
        if self.linear_news is not None and not np.isnan(m.label_sd[i]):
            self.linear_news.update(x, label_bps / m.label_sd[i])
        ts = iso(sessions.boundary_of_k(pushed_k))
        for bot_id in self._decided.pop((pushed_k, symbol), ()):
            self.store.queue(records.SET_LABEL, (label_bps, bot_id, ts))

    def _flattens(self, ctx: StockMinute, m: MarketMinute) -> None:
        """At each healthy symbol minute, close every open position that survived a close with session_flatten, every
        other open position after a long gap with the shared layer's flatten or one owed after a rejection, and every
        dying bot's position with its death close."""
        owed = {s: self.owed.pop(s) for s in list(self.owed) if ctx.healthy[self.symbols.index(s)]}
        reasons = [m.flatten[i] or owed.get(s) for i, s in enumerate(self.symbols)]
        sent = [0] * len(self.symbols)
        for bot in self.all_bots():
            i, acct = bot.home, bot.account
            if not ctx.healthy[i] or not acct.is_open or bot.pending_order is not None:
                continue
            if bot in self.dying:
                reason = DEATH
            elif acct.entry_ms < m.session.open_ms:
                reason = SESSION_FLATTEN
                if acct.trade_id not in self._carried:
                    self._carried.add(acct.trade_id)
                    self._event(m.boundary_ms, OVERNIGHT_CARRY, bot.symbol,
                                f"bot {bot.bot_id} carried trade {acct.trade_id} from {iso(acct.entry_ms)}")
            elif reasons[i] is not None:
                reason = reasons[i]
                sent[i] += 1
            else:
                continue
            self._close(bot, ctx, reason)
        for i, reason in enumerate(m.flatten):
            if reason is not None:
                self._event(m.boundary_ms, RESUME, self.symbols[i], f"{m.gaps[i]} session minutes since the last "
                                                                    f"healthy minute, {reason} sent {sent[i]} closes")

    def _slots(self, ctx: StockMinute) -> list[StockBot]:
        """Step each slot's learner, then its twin after it, and return the bots that died."""
        died: list[StockBot] = []
        pop = self.population
        for slot in range(pop.size):
            learner, twin = pop.learners[slot], pop.twins[slot]
            res = None
            if learner is not None:
                res = learner.step(ctx, self.broker)
                self._took(learner, res.orders, res.died, died)
                if res.decision is not None:
                    self._decision(learner, ctx.k, res.decision)
            if twin is None:
                continue
            out = twin.follow(learner, res, ctx, self.broker) if learner is not None else self._orphan(twin, ctx)
            self._took(twin, out.orders, out.died, died)
            if out.decision is not None:
                self._decision(twin, ctx.k, out.decision)
        return died

    def _orphan(self, twin: StockTwin, ctx: StockMinute) -> TwinStep:
        """A twin whose learner died keeps its upkeep and closes any position, since its learner never trades again."""
        own: BotStep = twin.step(ctx, self.broker)
        exit_order = None if own.died else twin.copy_exit(ctx, self.broker)
        return TwinStep(None, own.orders + ([exit_order] if exit_order is not None else []), own.died)

    def _session_close(self, ctx: StockMinute) -> None:
        """From five minutes before the close, send every open position with no order pending its session close at
        each healthy minute of its symbol, while a print can still fill it."""
        session = ctx.session
        if ctx.boundary_ms >= session.close_ms or \
                not session_close_due(session, ctx.session_minute, self.cfg.session):
            return
        for bot in self.all_bots():
            if ctx.healthy[bot.home] and bot.account.is_open and bot.pending_order is None:
                self._close(bot, ctx, DEATH if bot in self.dying else SESSION_CLOSE)

    def _close(self, bot: StockBot, ctx: StockMinute, reason: str) -> None:
        if bot.send_close(ctx.boundary_ms, float(ctx.ref[bot.home]), self.broker, reason) is not None:
            self._register(bot)

    def _took(self, bot: StockBot, orders: list[StockOrder], died: bool, out: list[StockBot]) -> None:
        if orders:
            self._register(bot)
        if died:
            out.append(bot)

    def _decision(self, bot: StockBot, k: int, row: dict) -> None:
        self.store.queue(records.INSERT_DECISION, records.decision_row(row))
        self._decided.setdefault((k, row["symbol"]), []).append(bot.bot_id)

    # Orders

    def _register(self, bot: StockBot) -> None:
        self._sent += 1
        self.pending[bot.bot_id] = (self._sent, bot)

    def _settled(self, bot: StockBot, order: StockOrder, ex: StockExecution, now_ms: int) -> None:
        del self.pending[bot.bot_id]
        bot.on_execution(ex)
        window = self.news_window(order.symbol, order.decided_ms)
        for fill in ex.fills:
            self.store.queue(records.INSERT_FILL_WITH_REPORTS, records.fill_row_with_reports(fill, window))
        if ex.closed is not None:
            self.store.queue(records.CLOSE_TRADE, records.close_trade_row(ex.closed))
            self._carried.discard(ex.closed.trade_id)
        if ex.opened_trade_id is not None:
            self.store.queue(records.OPEN_TRADE, records.open_trade_row(bot.bot_id, bot.symbol, bot.account))
        if ex.rejected is not None and order.reason in (GAP_FLATTEN, RESTART_FLATTEN) and bot.account.is_open:
            self.owed[order.symbol] = order.reason
        if bot in self.dying and not bot.account.is_open and bot.pending_order is None:
            self._finalize(bot, now_ms)

    # Deaths

    def _on_death(self, bot: StockBot, now_ms: int) -> None:
        """A dead bot leaves its slot empty at once. With a position it waits on the dying list for its close."""
        self.deaths += 1
        log.info("bot %d died at %s", bot.bot_id, iso(now_ms))
        self.population.remove(bot)
        if bot.account.is_open or bot.pending_order is not None:
            self.dying.append(bot)
        else:
            self._finalize(bot, now_ms)

    def _finalize(self, bot: StockBot, now_ms: int) -> None:
        bot.account.clamp_cash()
        self.store.queue(records.INSERT_DEATH, death_values(bot, now_ms))
        self.store.queue(records.SET_BOT_DEAD, (iso(now_ms), bot.bot_id))
        self._stored_state[bot.bot_id] = DEAD
        if bot in self.dying:
            self.dying.remove(bot)

    def _insert_bot(self, bot: StockBot) -> None:
        cfg = self.cfg
        self.store.queue(records.INSERT_BOT, records.bot_row(
            bot_id=bot.bot_id, slot=bot.slot, home_symbol=bot.symbol, born_ms=bot.born_ms, state=bot.state,
            margin_sigma=cfg.trading.margin_sigma, lr=cfg.model.learning_rate, seed=cfg.run.master_seed,
            origin=TWIN if isinstance(bot, StockTwin) else LEARNER))
        self._stored_state[bot.bot_id] = bot.state

    # Rows

    def _rows(self, m: MarketMinute, ctx: StockMinute) -> None:
        """Buffer this minute's rows for the run database."""
        store = self.store
        for candle in m.candles:
            store.queue(records.INSERT_CANDLE, records.candle_row(candle))
        for row in m.labels:
            store.queue(*records.label_statement(row))
        if m.evaluation:
            self._forecasts(m, ctx)
            self._price_z(m, ctx)
        self._benchmark(m)
        if m.k % (self.cfg.persistence.equity_snapshot_every_seconds // 60) == 0:
            self._snapshots(m)
        if self.inputs.use_news:
            store.queue(records.INSERT_NEWS_MINUTE, records.news_minute_row(m.boundary_ms, m.news_reason,
                                                                             m.channels_down))
        store.queue(records.PUT_SESSION, records.session_row(m.session, m.healthy_minutes, m.run_minutes))
        for check in m.spread_checks:
            store.queue(records.PUT_SPREAD_CHECK, records.spread_check_row(
                m.session.day, check.symbol, check.prints, check.estimate_bps, check.allowance_bps))
        self._decided = {key: ids for key, ids in self._decided.items()
                         if key[0] >= m.k - self.cfg.time.horizon_minutes}

    def _forecasts(self, m: MarketMinute, ctx: StockMinute) -> None:
        """One row per predictor for each usable symbol at an evaluation minute, the bps predictors where a valid
        vector and a label scale exist, and in S2 every score stream with its items and newest receive time."""
        ms, learners = m.boundary_ms, self.population.living_learners()
        streams = self.inputs.streams if self.inputs.use_news else {}
        for i, symbol in enumerate(self.symbols):
            if not ctx.usable(i):
                continue
            newest = streams[NEWS][symbol].newest_ms if streams else None
            rows = []
            preds = [float(b.preds[i]) for b in learners if not np.isnan(b.preds[i])]
            if preds:
                rows.append((POPULATION, float(np.mean(preds)), None, newest))
            if not np.isnan(m.linear[i]):
                rows.append((LINEAR_PRICES, float(m.linear[i]), None, None))
                if self.linear_news is not None:
                    rows.append((LINEAR_NEWS, self.linear_news.predict(ctx.rows[i]) * float(m.label_sd[i]), None,
                                 newest))
            if streams:
                main = streams[NEWS][symbol]
                rows.append((NEWS_COMPANY, main.company, main.company_items, main.newest_ms))
                rows.append((NEWS_MACRO, main.macro, main.macro_items, main.newest_ms))
                for stream in STREAMS:
                    value = streams[stream][symbol]
                    rows.append((stream, value.score, value.company_items + value.macro_items, value.newest_ms))
            for predictor, prediction, items, recv_ms in rows:
                self.store.queue(records.INSERT_FORECAST,
                                 records.forecast_row(ms, symbol, predictor, prediction, items, recv_ms))

    def _price_z(self, m: MarketMinute, ctx: StockMinute) -> None:
        """The 10 price z scores of each usable symbol at an evaluation minute, null where its vector was invalid, which
        the partial news test adjusts for."""
        for i, symbol in enumerate(self.symbols):
            if ctx.usable(i):
                self.store.queue(records.INSERT_PRICE_Z,
                                 records.price_z_row(m.k, m.boundary_ms, symbol, m.z[i] if m.valid[i] else None))

    def _benchmark(self, m: MarketMinute) -> None:
        """The benchmark row once every symbol has had a healthy price, at each minute with any symbol healthy. Buy
        and hold buys at launch through the allowance and marks at the last healthy reference price."""
        trading = self.cfg.trading
        if self.anchor is None:
            if not m.healthy.all():
                return
            self.anchor = tuple(Anchor(m.boundary_ms, costs.fill_price(float(ref), costs.BUY, allow), float(ref))
                                for ref, allow in zip(m.ref, self.allowances))
        living = [b.equity for b in self.population.living_learners()]
        if not m.healthy.any() or not living:
            return
        marks = [costs.mark_price(float(ref), LONG, allow) for ref, allow in zip(m.last_ref, self.allowances)]
        share = trading.start_equity / len(self.symbols)
        q, start = self.index, trading.start_equity
        twins = [t.equity for t in self.population.living_twins()]
        self.store.queue(records.INSERT_BENCHMARK, records.benchmark_row(
            m.boundary_ms,
            ew_buy_hold=sum(buy_hold_equity(a, mark, share, 0.0) for a, mark in zip(self.anchor, marks)),
            qqq_buy_hold=buy_hold_equity(self.anchor[q], marks[q], start, 0.0),
            matched_equity=matched_equity(self.anchor[q], marks[q], start, trading.notional, 0.0),
            living_mean_equity=statistics.fmean(living), living_median_equity=statistics.median(living),
            living=len(living), twin_mean_equity=statistics.fmean(twins) if twins else None,
            cash_equity=start - self.rent_per_minute * m.run_minutes_total))

    def _snapshots(self, m: MarketMinute) -> None:
        """Equity snapshots for every living bot and the balance checks for every bot whose books are still open,
        marking a position at its symbol's last healthy reference price."""
        for bot in self.all_bots():
            mark = bot.mark(float(m.last_ref[bot.home]))
            errors = bot.account.invariant_errors(mark)
            if errors:
                log.error("invariant violation for bot %d, its figures only in feed_events", bot.bot_id)
                self._event(m.boundary_ms, INVARIANT_VIOLATION, bot.symbol, f"bot {bot.bot_id}: {' | '.join(errors)}")
            if bot.state != DEAD:
                self.store.queue(records.INSERT_SNAPSHOT, records.snapshot_row(bot, m.boundary_ms, mark))

    def _event(self, ms: int, kind: str, symbol: str | None, detail: str) -> None:
        """Queue one feed event, logging only its time, kind and symbol, because a detail can hold a price move."""
        log.info("feed event %s %s %s", iso(ms), kind, symbol or "")
        self.feed_rows.append(records.feed_event_row(ms, kind, symbol, detail))

    # Persistence

    def persistence_point(self, at: int) -> None:
        """Feed events are committed at every point, and every checkpoint interval the rows go with a checkpoint."""
        every = self.cfg.persistence.checkpoint_every_seconds // 60
        if minute_index(at - PERSIST_OFFSET_MS) % every == 0:
            self.persist(at)
        elif self.feed_rows:
            self.store.commit_now([(records.INSERT_FEED_EVENT, row) for row in self.feed_rows])
            self.feed_rows.clear()

    def persist(self, at: int) -> None:
        """Commit every buffered row, bot state changes and feed events in one transaction, with the checkpoint the
        group's checkpointer encodes when it has one."""
        for bot in self.all_bots():
            state = "dying" if bot in self.dying else bot.state
            if self._stored_state.get(bot.bot_id) != state:
                self.store.queue(records.SET_BOT_STATE, (state, bot.bot_id))
                self._stored_state[bot.bot_id] = state
        for row in self.feed_rows:
            self.store.queue(records.INSERT_FEED_EVENT, row)
        self.feed_rows.clear()
        ts = iso(at)
        encoded = None if self.checkpointer is None else self.checkpointer(self, at)
        keep, days = self.cfg.persistence.keep_checkpoints, self.cfg.persistence.keep_candles_days

        def extra(conn):
            if encoded is not None:
                conn.execute(records.INSERT_CHECKPOINT, (ts, SCHEMA_VERSION, *encoded))
                conn.execute(records.PRUNE_CHECKPOINTS, (keep,))
            if days:
                conn.execute(records.PRUNE_CANDLES, (iso(at - days * MS_PER_DAY),))

        self.store.flush(extra)
        if encoded is not None:
            self.last_checkpoint_ts = ts

