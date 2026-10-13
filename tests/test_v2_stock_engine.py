"""The stock engine on seeded synthetic sessions, driven through a stand in for the group with S1 and S2 on one shared
market layer: a synthetic day with 390 candle rows per symbol, no flatten after a night, a weekend or a holiday, a
process killed at 15:54 with a position closing it next morning with session_flatten and overnight_carry, a 15 minute
feed outage flattening, a 20 minute halt flattening only that symbol, a 15 minute news outage pausing rent, learning and
decisions in both runs while bumping no epoch, flattening nothing and leaving the next day's feature 3 valid, a death
leaving its slot empty, rent only in usable minutes, the DST Monday and the invariants at every snapshot. Also the
session close at 15:55 with a symbol halted into the close, a restart in session flattening, every label against its
recomputation from the candles and every decision against its label, the split guard and the short sale restriction
in the shared layer, a changed news hash stopping news health, orders rejected at a restart, the news_seen bookkeeping
and S1 and S2 writing identical rows while S2 reads no news. Also the price z scores of every usable symbol at each
evaluation minute, and the report only fill columns, how long after due each print traded and S2's news window. Under
the holding rule, kept positions send no order and still end by stops, closes and flattens, and a learner keeps a long
across several decisions to the session close while its twin keeps its copy."""

import copy
import re
import sqlite3
from datetime import date, datetime, time
from types import SimpleNamespace

import numpy as np
import pytest

from btc_survivor.bot import ACTIVE, DEAD
from btc_survivor.clock import MS_PER_MINUTE, MS_PER_SECOND, iso, parse_iso
from btc_survivor.types import LONG, FeedNotice, Leg
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import parse_stock_config
from btc_survivor.v2.news.health import FINBERT_SCORER
from btc_survivor.v2.stocks import costs
from btc_survivor.v2.stocks.candles import StockCandle
from btc_survivor.v2.stocks.bot import SESSION_CLOSE, BotStep
from btc_survivor.v2.stocks.engine import NEWS_WINDOW_MS, PERSIST_OFFSET_MS, StockEngine
from btc_survivor.v2.stocks.finnhub import Print, PrintBatch
from btc_survivor.v2.stocks.labels import recompute_labels
from btc_survivor.v2.stocks.market import MarketLayer
from btc_survivor.v2.stocks.news_inputs import HAIKU_SCORER, HeartbeatLine, Impact, ScoreLine
from btc_survivor.v2.stocks.population import LEARNER, TWIN
from btc_survivor.v2.stocks.store import StockStore
from helpers import ROOT

HASHES = ("c" * 64, "p" * 64, "s" * 64)  # the fake news hashes recorded at launch
PRICES = (250.0, 500.0, 180.0, 220.0, 250.0, 700.0, 400.0, 600.0)
STEP_SIGMA = 0.5e-4  # one second log price step, about 21 bps over 30 minutes
THU, FRI, MON, TUE, WED = (sessions.session_on(day) for day in (
    date(2026, 10, 29), date(2026, 10, 30), date(2026, 11, 2), date(2026, 11, 3), date(2026, 11, 4)))
HOLIDAY_EVE, EARLY_CLOSE = sessions.session_on(date(2026, 11, 25)), sessions.session_on(date(2026, 11, 27))
DRAINED_SLOT, HOLDING_SLOT = 5, 10
CLOSING_SLOT, CARRY_SLOT = 13, 12  # META holds into the session close, GOOGL holds and is halted into the close


def at(session, clock: str) -> int:
    """The UTC milliseconds of a New York clock time on a session's date."""
    parts = [int(p) for p in clock.split(":")]
    moment = datetime.combine(session.day, time(*parts), tzinfo=sessions.new_york())
    return int(moment.timestamp()) * MS_PER_SECOND


def config(folder, name, **values):
    """A committed run config with every path moved into folder and some settings replaced."""
    text = (ROOT / "configs" / "v2" / f"{name}.toml").read_text(encoding="utf-8")
    settings = {"data_dir": f'"data/{name}"', "secrets_file": '"secrets.env"', "raw_dir": '"data/raw_stocks"',
                "lock_file": '"lock/finnhub_ws.lock"', "news_db": '"data/news.db"', **values}
    for key, value in settings.items():
        text, found = re.subn(rf"(?m)^{key} = [^#\n]*", f"{key} = {value} ", text)
        assert found == 1, key
    return parse_stock_config(text, folder)


def pair(folder, **values):
    return config(folder, "s1_prices", **values), config(folder, "s2_news", **values)


class Group:
    """A stand in for group.py, one market layer feeding both engines every boundary, persistence point, feed event
    and heartbeat line in stamp order, and keeping each MarketMinute with what the engines held after it."""

    def __init__(self, market, engines, now_ms):
        self.market, self.engines = market, engines
        self.now = now_ms
        self.next_boundary = now_ms - now_ms % MS_PER_MINUTE + MS_PER_MINUTE
        self.next_persist = self.next_boundary + PERSIST_OFFSET_MS
        self.minutes = []

    def advance_to(self, t):
        t = max(t, self.now)
        while True:
            if self.next_boundary <= self.next_persist:
                if self.next_boundary > t:
                    break
                self._boundary(self.next_boundary)
                self.next_boundary += MS_PER_MINUTE
            else:
                if self.next_persist > t:
                    break
                for engine in self.engines:
                    engine.persistence_point(self.next_persist)
                self.next_persist += MS_PER_MINUTE
        self.now = t
        self.market.advance(t)
        self._settle(t)

    def _boundary(self, boundary):
        m = self.market.close_minute(boundary)
        if m is not None:
            events = self.market.drain_events()
            for engine in self.engines:
                engine.on_events(events)
                engine.on_minute(m)
            self.minutes.append(SimpleNamespace(
                m=m, rent=tuple(sum(b.account.rent for b in e.all_bots()) for e in self.engines),
                trained=tuple(sum(b.trained_minutes for b in e.population.learners if b) for e in self.engines),
                epochs=tuple(h.epoch for h in self.market.health.symbols.values())))
        self._settle(boundary)

    def _settle(self, t):
        events = self.market.drain_events()
        for engine in self.engines:
            engine.on_events(events)
            engine.check_orders(t, self.market)

    def feed(self, ev):
        self.advance_to(ev.recv_ms)
        prints = self.market.on_event(ev)
        self._settle(ev.recv_ms)
        if prints:
            for engine in self.engines:
                engine.on_prints(ev.recv_ms, prints)

    def heartbeat(self, line):
        self.advance_to(line.recv_ms)
        self.market.on_heartbeat(line)

    def news(self, line):
        self.advance_to(line.recv_ms)
        for engine in self.engines:
            engine.on_news(line)


class Tape:
    """A seeded print every second for every symbol in session that is not halted while the feed is up, received
    150 ms after its vendor time, and a healthy heartbeat with the launch hashes every 30 seconds unless news is
    down."""

    def __init__(self, group, seed, start_ms):
        self.group = group
        self.rng = np.random.default_rng(seed)
        self.prices = np.array(PRICES)
        self.t = start_ms
        self.conn = 1
        self.beats = 0
        self.halted: set[str] = set()
        self.offline = False
        self.news_down = False
        self.hashes = HASHES
        group.feed(FeedNotice(start_ms, self.conn, "connected"))

    def run_to(self, end_ms):
        session = sessions.next_session(self.t)
        symbols = self.group.market.symbols
        while self.t < end_ms:
            t = self.t
            if t % 30_000 == 0 and not self.news_down:
                self.beats += 1
                self.group.heartbeat(HeartbeatLine(self.beats, t, True, None, *self.hashes, (), recv_ms=t))
            if session.open_ms <= t < session.close_ms:
                self.prices *= np.exp(self.rng.normal(0.0, STEP_SIGMA, len(symbols)))
                if not self.offline:
                    prints = tuple(Print(s, round(float(p), 2), 100.0, t - 100, ())
                                   for s, p in zip(symbols, self.prices) if s not in self.halted)
                    self.group.feed(PrintBatch(t + 50, self.conn, prints))
            self.t += MS_PER_SECOND
        self.group.advance_to(end_ms)

    def skip_to(self, t):
        """Jump over a night or weekend, when nothing arrives."""
        self.group.advance_to(t)
        self.t = t

    def disconnect(self):
        self.group.feed(FeedNotice(self.t, self.conn, "disconnected", detail="test outage"))
        self.offline = True

    def reconnect(self):
        self.conn += 1
        self.group.feed(FeedNotice(self.t, self.conn, "connected"))
        self.offline = False


def start(folder, launch_ms, **values):
    """Both runs born at launch_ms on one shared market layer, with a tape feeding them."""
    c1, c2 = pair(folder, **values)
    engines = [StockEngine.fresh(c, StockStore.open(c.db_path), launch_ms) for c in (c1, c2)]
    group = Group(MarketLayer(c1, HASHES), engines, launch_ms)
    return c1, c2, group, Tape(group, 7, launch_ms)


def checkpoint(group):
    """A deep copy of the shared layer and both engines without their stores, standing in for a checkpoint that
    round trips bit for bit."""
    stores = [e.store for e in group.engines]
    for engine in group.engines:
        engine.store = None
    try:
        return copy.deepcopy((group.market, group.engines))
    finally:
        for engine, store in zip(group.engines, stores):
            engine.store = store


def hold(engines, slot):
    """Make one learner predict five label standard deviations up in both runs, so it opens a long at its next phase
    minute and holds it at every later one."""
    for engine in engines:
        model = engine.population.learners[slot].model
        model.W2[:] = 0.0
        model.b2[0] = 5.0


def open_positions(engine):
    return {b.bot_id: (b.symbol, b.account.side, b.account.trade_id) for b in engine.all_bots() if b.account.is_open}


def query(path, sql, params=()):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def restart(group, saved, restart_ms, tape, dbs):
    """Kill the process, losing every row it had not committed, and start a new one at restart_ms from the saved
    checkpoint, which reconnects to the feed."""
    for engine in group.engines:
        engine.store.close()
    market, engines = copy.deepcopy(saved)
    for engine, path in zip(engines, dbs):
        engine.store = StockStore.open(path)
    group = Group(market, engines, restart_ms)
    market.resume(restart_ms)
    for engine in engines:
        engine.resume(restart_ms)
    tape.group, tape.t = group, restart_ms
    tape.reconnect()
    return group


@pytest.fixture(scope="module")
def week(tmp_path_factory):
    """Thursday 29 October 2026 as the first session, a Friday with a death, a news outage, a halt and a feed outage,
    the DST Monday after a weekend with one symbol halted into the close, a Tuesday killed at 15:54 holding positions,
    a restart before Wednesday's open, and a second kill at 10:20 with a restart in session at 10:30."""
    launch = at(THU, "09:00")
    c1, c2, group, tape = start(tmp_path_factory.mktemp("week"), launch, warmup_hours="0.5",
                                force_trade_after_hours="0.5")
    w = SimpleNamespace(c1=c1, c2=c2, dbs=(c1.db_path, c2.db_path), symbols=c1.stock_feed.symbols)
    tape.run_to(at(THU, "16:05"))
    tape.skip_to(at(FRI, "09:28"))
    tape.run_to(at(FRI, "11:45:30"))
    w.active_before_death = all(e.population.learners[DRAINED_SLOT].state == ACTIVE for e in group.engines)
    for engine in group.engines:
        bot = engine.population.learners[DRAINED_SLOT]
        mark = bot.mark(float(group.market.last_ref[bot.home]))
        bot.account.charge_rent(bot.account.equity(mark) - 0.5)
    w.drained = group.engines[0].population.learners[DRAINED_SLOT].bot_id
    tape.run_to(at(FRI, "12:00"))
    tape.news_down = True
    tape.run_to(at(FRI, "12:15"))
    tape.news_down = False
    tape.run_to(at(FRI, "12:59:59"))
    held = open_positions(group.engines[0])
    w.halted = next(s for s in w.symbols if any(v[0] == s for v in held.values()))
    w.open_at_halt = {i: v for i, v in held.items() if v[0] == w.halted}
    w.open_elsewhere_at_halt = {i: v for i, v in held.items() if v[0] != w.halted}
    tape.halted = {w.halted}
    tape.run_to(at(FRI, "13:20"))
    tape.halted = set()
    tape.run_to(at(FRI, "13:59:59"))
    w.open_at_outage = open_positions(group.engines[0])
    tape.disconnect()
    tape.run_to(at(FRI, "14:15"))
    tape.reconnect()
    tape.run_to(at(FRI, "16:05"))
    tape.skip_to(at(MON, "09:28"))
    tape.run_to(at(MON, "15:00:30"))
    hold(group.engines, CLOSING_SLOT)
    hold(group.engines, CARRY_SLOT)
    tape.run_to(at(MON, "15:54:59"))
    w.open_before_close = open_positions(group.engines[0])
    tape.run_to(at(MON, "15:55"))
    w.carry_symbol = group.engines[0].population.learners[CARRY_SLOT].symbol
    tape.halted = {w.carry_symbol}
    tape.run_to(at(MON, "16:05"))
    tape.halted = set()
    w.open_after_monday = open_positions(group.engines[0])
    tape.skip_to(at(TUE, "09:28"))
    tape.run_to(at(TUE, "15:00:30"))
    hold(group.engines, HOLDING_SLOT)
    tape.run_to(at(TUE, "15:50:30"))
    saved = checkpoint(group)
    w.carried = open_positions(saved[1][0])
    tape.run_to(at(TUE, "15:54:30"))
    w.open_at_kill = open_positions(group.engines[0])
    w.rent = [(b.bot_id, b.symbol, b.account.rent,
               sum(1 for o in group.minutes if o.m.healthy[b.home] and o.m.news_ok))
              for b in group.engines[0].population.slot_bots()]
    w.minutes = list(group.minutes)
    w.killed = [e.population for e in group.engines]
    group = restart(group, saved, at(WED, "09:15"), tape, w.dbs)
    tape.run_to(at(WED, "10:15:30"))
    w.wednesday = list(group.minutes)
    saved = checkpoint(group)
    w.open_at_wednesday_checkpoint = open_positions(saved[1][0])
    tape.run_to(at(WED, "10:20"))
    group = restart(group, saved, at(WED, "10:30"), tape, w.dbs)
    tape.run_to(at(WED, "10:45"))
    for engine in group.engines:
        engine.persist(group.now)
        engine.store.close()
    w.after_restart_in_session = list(group.minutes)
    w.engines = group.engines
    return w


def minute_of(minutes, session, clock):
    boundary = at(session, clock)
    return next(o for o in minutes if o.m.boundary_ms == boundary)


def between(session, start, end):
    return iso(at(session, start)), iso(at(session, end))


def test_a_synthetic_day_has_390_candle_rows_per_symbol_identical_in_both_runs(week):
    first, last = between(THU, "09:31", "16:00")
    per_run = []
    for db in week.dbs:
        counts = dict(query(db, "SELECT symbol, count(*) FROM stock_candles WHERE ts BETWEEN ? AND ? GROUP BY symbol",
                            (first, last)))
        assert counts == {s: 390 for s in week.symbols}
        assert query(db, "SELECT min(ts), max(ts), min(valid) FROM stock_candles WHERE ts BETWEEN ? AND ?",
                     (first, last)) == [("2026-10-29T13:31:00.000Z", "2026-10-29T20:00:00.000Z", 1)]
        assert query(db, "SELECT healthy_minutes, run_minutes, early_close FROM sessions WHERE session_date = ?",
                     ("2026-10-29",)) == [(390, 390, 0)]
        labels = query(db, "SELECT symbol, status, count(*) FROM stock_labels WHERE ts BETWEEN ? AND ? "
                           "GROUP BY symbol, status", (first, last))
        assert labels == [(s, "matured", 12) for s in sorted(week.symbols)]
        per_run.append(query(db, "SELECT * FROM stock_candles ORDER BY symbol, ts"))
    assert per_run[0] == per_run[1]


def test_no_flatten_after_a_night_or_a_weekend_and_the_dst_monday_opens_at_14_30_utc(week):
    for session in (FRI, MON, TUE):
        first = minute_of(week.minutes, session, "09:31").m
        gaps = [6 if session is TUE and s == week.carry_symbol else 1 for s in week.symbols]
        assert first.flatten == (None,) * 8 and first.gaps == tuple(gaps)
        assert first.k == sessions.session_of_k(first.k - 1).last_k + 1
    for db in week.dbs:
        mondays = query(db, "SELECT symbol, count(*), min(ts), min(session_k) FROM stock_candles "
                            "WHERE session_date = '2026-11-02' GROUP BY symbol ORDER BY symbol")
        assert mondays == [(s, 390, "2026-11-02T14:31:00.000Z", FRI.last_k + 1) for s in sorted(week.symbols)]
        assert query(db, "SELECT min(ts) FROM stock_candles WHERE session_date = '2026-10-30'") == \
            [("2026-10-30T13:31:00.000Z",)]
        lo, hi = between(FRI, "13:00", "14:30")
        flattens = query(db, "SELECT decided_at FROM fills WHERE reason IN ('gap_flatten', 'restart_flatten') "
                             "AND decided_at < ?", (iso(at(WED, "10:30")),))
        assert flattens and all(lo <= ts <= hi for ts, in flattens)
        opens = query(db, "SELECT min(ts) FROM decisions WHERE action IN ('open_long', 'open_short', 'flip') "
                          "AND ts >= '2026-11-02'")
        assert opens[0][0] >= iso(at(MON, "10:00"))


@pytest.fixture(scope="module")
def holiday(tmp_path_factory):
    """Two bots per run from Wednesday 25 November 2026 over Thanksgiving to the 13:00 early close on Friday."""
    launch = at(HOLIDAY_EVE, "09:00")
    c1, c2, group, tape = start(tmp_path_factory.mktemp("holiday"), launch, size="2")
    tape.run_to(at(HOLIDAY_EVE, "16:05"))
    tape.skip_to(at(EARLY_CLOSE, "09:28"))
    tape.run_to(at(EARLY_CLOSE, "13:05"))
    for engine in group.engines:
        engine.persist(group.now)
        engine.store.close()
    return SimpleNamespace(minutes=group.minutes, dbs=(c1.db_path, c2.db_path), symbols=c1.stock_feed.symbols)


def test_no_flatten_after_a_holiday_and_the_early_close_has_210_minutes(holiday):
    first = minute_of(holiday.minutes, EARLY_CLOSE, "09:31").m
    assert first.flatten == (None,) * 8 and first.gaps == (1,) * 8
    assert first.k == HOLIDAY_EVE.last_k + 1
    last = holiday.minutes[-1].m
    assert last.boundary_ms == EARLY_CLOSE.close_ms and last.last_minute and len(last.spread_checks) == 8
    for db in holiday.dbs:
        assert dict(query(db, "SELECT symbol, count(*) FROM stock_candles WHERE session_date = '2026-11-27' "
                              "GROUP BY symbol")) == {s: 210 for s in holiday.symbols}
        assert query(db, "SELECT healthy_minutes, run_minutes, early_close FROM sessions "
                         "WHERE session_date = '2026-11-27'") == [(210, 210, 1)]
        assert dict(query(db, "SELECT symbol, count(*) FROM stock_labels WHERE ts >= '2026-11-27' GROUP BY symbol")) \
            == {s: 6 for s in holiday.symbols}
        assert query(db, "SELECT count(*) FROM spread_checks WHERE session_date = '2026-11-27'") == [(8,)]
        assert query(db, "SELECT count(*) FROM fills WHERE reason LIKE '%flatten'") == [(0,)]


def test_a_process_killed_at_15_54_closes_its_positions_next_morning_with_session_flatten(week):
    holder = week.killed[0].learners[HOLDING_SLOT]
    assert holder.bot_id in week.open_at_kill and week.open_at_kill[holder.bot_id][1] == LONG
    assert holder.bot_id in week.carried
    flatten_at = iso(at(WED, "09:31"))
    for db in week.dbs:
        fills = query(db, "SELECT bot_id, trade_id, status FROM fills WHERE reason = 'session_flatten' "
                          "AND decided_at = ?", (flatten_at,))
        assert sorted(fills) == sorted((i, v[2], "filled") for i, v in week.carried.items())
        closed = query(db, "SELECT id, exit_reason FROM trades WHERE closed_at >= ?", (flatten_at,))
        assert {i for i, reason in closed if reason == "session_flatten"} == {v[2] for v in week.carried.values()}
        carries = query(db, "SELECT symbol, detail FROM feed_events WHERE kind = 'overnight_carry' AND ts >= ?",
                        (flatten_at,))
        assert sorted(int(detail.split()[1]) for _, detail in carries) == sorted(week.carried)
        morning = (iso(at(WED, "09:15")), iso(at(WED, "10:00")))
        assert query(db, "SELECT count(*) FROM fills WHERE reason IN ('gap_flatten', 'restart_flatten') "
                         "AND decided_at BETWEEN ? AND ?", morning) == [(0,)]
        assert query(db, "SELECT symbol FROM feed_events WHERE kind = 'resume' AND ts BETWEEN ? AND ?",
                     morning) == [(None,)]
        # The rows the dead process had not committed are gone, and the label waiting on 16:00 ended as a gap.
        assert query(db, "SELECT max(ts) FROM stock_candles WHERE session_date = '2026-11-03'") == \
            [(iso(at(TUE, "15:50")),)]
        assert {status for status, in query(db, "SELECT status FROM stock_labels WHERE ts = ?",
                                            (iso(at(TUE, "15:30")),))} == {"gap"}
    first = week.wednesday[0].m
    assert first.boundary_ms == at(WED, "09:31") and first.flatten == (None,) * 8 and first.gaps == (0,) * 8


def test_a_restart_in_session_after_more_than_10_minutes_flattens_with_restart_flatten(week):
    assert week.open_at_wednesday_checkpoint
    back = week.after_restart_in_session[0].m
    assert back.boundary_ms == at(WED, "10:31") and back.healthy.all()
    assert set(back.flatten) == {"restart_flatten"} and back.gaps == (16,) * 8
    for db in week.dbs:
        fills = query(db, "SELECT bot_id, trade_id, status FROM fills WHERE reason = 'restart_flatten' "
                          "AND decided_at = ?", (iso(back.boundary_ms),))
        assert sorted(fills) == sorted((i, v[2], "filled") for i, v in week.open_at_wednesday_checkpoint.items())
        assert query(db, "SELECT count(*) FROM fills WHERE decided_at > ? AND decided_at < ?",
                     between(WED, "10:15:30", "10:31")) == [(0,)]
        assert query(db, "SELECT status FROM stock_labels WHERE ts = ? GROUP BY status",
                     (iso(at(WED, "10:00")),)) == [("gap",)]
        assert query(db, "SELECT count(*) FROM feed_events WHERE kind = 'resume' AND ts BETWEEN ? AND ?",
                     between(WED, "10:30", "10:31")) == [(1 + 8,)]
    for engine in week.engines:
        assert engine.deaths == 1 and all(b.state != DEAD for b in engine.population.slot_bots())


def test_the_session_close_at_15_55_and_a_symbol_halted_into_the_close_carried_to_the_next_morning(week):
    carried = {i: v for i, v in week.open_before_close.items() if v[0] == week.carry_symbol}
    closed = {i: v for i, v in week.open_before_close.items() if v[0] != week.carry_symbol}
    assert closed and carried and week.open_after_monday == carried
    closing = week.killed[0].learners[CLOSING_SLOT].bot_id
    assert closing in closed and week.killed[0].learners[CARRY_SLOT].bot_id in carried
    close_at, flatten_at = iso(at(MON, "15:55")), iso(at(TUE, "09:31"))
    for db in week.dbs:
        fills = query(db, "SELECT bot_id, trade_id, status, reason FROM fills WHERE decided_at = ? AND leg = 'close'",
                      (close_at,))
        assert sorted(fills) == sorted([(i, v[2], "filled", "session_close") for i, v in closed.items()] +
                                       [(i, v[2], "rejected", "symbol_not_healthy") for i, v in carried.items()])
        assert query(db, "SELECT count(*) FROM fills WHERE decided_at > ? AND decided_at < ?",
                     (close_at, iso(at(TUE, "09:30")))) == [(0,)]
        flattened = query(db, "SELECT bot_id, trade_id, status FROM fills WHERE reason = 'session_flatten' "
                              "AND decided_at = ?", (flatten_at,))
        assert sorted(flattened) == sorted((i, v[2], "filled") for i, v in carried.items())
        carries = query(db, "SELECT symbol, detail FROM feed_events WHERE kind = 'overnight_carry' AND ts = ?",
                        (flatten_at,))
        assert sorted(int(detail.split()[1]) for _, detail in carries) == sorted(carried)
        assert {symbol for symbol, _ in carries} == {week.carry_symbol}
        trades = query(db, "SELECT exit_reason, count(*) FROM trades WHERE closed_at BETWEEN ? AND ? "
                           "GROUP BY exit_reason ORDER BY exit_reason", (close_at, iso(at(TUE, "09:32"))))
        assert trades == [("session_close", len(closed)), ("session_flatten", len(carried))]


@pytest.fixture(scope="module")
def resend(tmp_path_factory):
    """Two bots per run from Thursday, the first learner holding a long into Friday's session close while its symbol
    halts from 15:55 to 15:57:30 only. Its stale thresholds are pinned at 5 and 15 seconds, so the halt marks the
    symbol stale whatever the measured thresholds in the committed configs are."""
    c1, c2, group, tape = start(tmp_path_factory.mktemp("resend"), at(THU, "09:00"), size="2", warmup_hours="0.5",
                                force_trade_after_hours="0.5", feed_stale_after_seconds="5",
                                symbol_stale_after_seconds="15")
    tape.run_to(at(THU, "16:05"))
    tape.skip_to(at(FRI, "09:28"))
    tape.run_to(at(FRI, "14:20:30"))
    hold(group.engines, 0)   # its phase minutes are on the half hour, so it opens at 14:30 inside the open window
    tape.run_to(at(FRI, "15:54:59"))
    bot = group.engines[0].population.learners[0]
    held = (bot.bot_id, bot.symbol, bot.account.is_open)
    tape.run_to(at(FRI, "15:55"))
    tape.halted = {bot.symbol}
    tape.run_to(at(FRI, "15:57:30"))
    tape.halted = set()
    tape.run_to(at(FRI, "16:05"))
    for engine in group.engines:
        engine.persist(group.now)
        engine.store.close()
    return SimpleNamespace(held=held, dbs=(c1.db_path, c2.db_path))


def test_a_session_close_rejected_while_its_symbol_halts_is_sent_again_at_the_next_healthy_minute(resend):
    bot_id, symbol, was_open = resend.held
    assert was_open
    for db in resend.dbs:
        fills = query(db, "SELECT decided_at, status, reason FROM fills WHERE bot_id = ? AND leg = 'close' "
                          "AND decided_at >= ? ORDER BY id", (bot_id, iso(at(FRI, "15:55"))))
        assert fills[0] == (iso(at(FRI, "15:55")), "rejected", "symbol_not_healthy")
        assert len(fills) == 2 and fills[1][1:] == ("filled", "session_close")
        assert iso(at(FRI, "15:57:30")) < fills[1][0] < iso(at(FRI, "16:00"))
        assert query(db, "SELECT count(*) FROM feed_events WHERE kind = 'overnight_carry'") == [(0,)]
        assert query(db, "SELECT exit_reason FROM trades WHERE bot_id = ? AND closed_at > ?",
                     (bot_id, iso(at(FRI, "15:55")))) == [("session_close",)]


@pytest.fixture(scope="module")
def keeper(tmp_path_factory):
    """Two bots per run from Thursday. On Friday the first learner predicts five label standard deviations up from
    13:20, so it holds a long from its 13:30 decision, and from 13:30:30 a twentieth of one, about 1 bp, below the edge
    an open needs but still up, so the holding rule keeps the long at 14:00, 14:30, 15:00 and 15:30, the last after the
    final open at 15:25, until the session close at 15:55. Its twin keeps its own copy beside it."""
    c1, c2, group, tape = start(tmp_path_factory.mktemp("keeper"), at(THU, "09:00"), size="2", warmup_hours="0.5",
                                force_trade_after_hours="0.5")
    tape.run_to(at(THU, "16:05"))
    tape.skip_to(at(FRI, "09:28"))
    tape.run_to(at(FRI, "13:20:30"))
    hold(group.engines, 0)
    tape.run_to(at(FRI, "13:30:30"))
    for engine in group.engines:
        engine.population.learners[0].model.b2[0] = 0.05
    held = []
    for clock in ("13:30:30", "13:59:59", "14:00:30", "14:30:30", "15:00:30", "15:30:30", "15:54:59"):
        tape.run_to(at(FRI, clock))
        held.append(open_positions(group.engines[0]))
    tape.run_to(at(FRI, "16:05"))
    for engine in group.engines:
        engine.persist(group.now)
        engine.store.close()
    learner, twin = group.engines[0].population.learners[0], group.engines[0].population.twins[0]
    return SimpleNamespace(held=held, learner=learner.bot_id, twin=twin.bot_id, dbs=(c1.db_path, c2.db_path))


def test_a_learner_keeps_its_long_across_decisions_to_the_session_close_and_its_twin_keeps_its_copy(keeper):
    learner, twin = keeper.learner, keeper.twin
    assert all(held == keeper.held[0] for held in keeper.held)    # every position unchanged from 13:30:30 to 15:54:59
    (_, side, trade), (_, _, copied) = keeper.held[0][learner], keeper.held[0][twin]
    assert side == LONG
    kept = [iso(at(FRI, clock)) for clock in ("14:00", "14:30", "15:00", "15:30")]
    after, close = iso(at(FRI, "13:30")), iso(at(FRI, "15:55"))
    for db in keeper.dbs:
        rows = query(db, "SELECT ts, position_before, action, gate_passed, forced, required_edge_bps, reason "
                         "FROM decisions WHERE bot_id = ? AND ts > ? AND ts < ? ORDER BY ts", (learner, after, close))
        assert [row[0] for row in rows] == kept
        for _, before, action, gate, forced, required, reason in rows:
            assert (before, action, gate, forced) == ("long", "hold", 0, 0) and required == pytest.approx(4.6)
            assert re.fullmatch(r"the prediction of 0\.\d{3} bps still points up, above the 0\.000 bps holding cost, "
                                r"so the long is kept", reason), reason
        # Neither the learner nor its twin sent an order while the long was kept, and the twin copied nothing.
        assert query(db, "SELECT count(*) FROM fills WHERE bot_id IN (?, ?) AND decided_at > ? AND decided_at < ?",
                     (learner, twin, after, close)) == [(0,)]
        assert query(db, "SELECT count(*) FROM decisions WHERE bot_id = ? AND ts > ? AND ts < ?",
                     (twin, after, close)) == [(0,)]
        # The learner's long and the twin's copy opened at the 13:30 decision, so the twin test pairs them, and both
        # ended at the session close.
        opened = query(db, "SELECT f.bot_id, f.decided_at, d.action FROM fills f JOIN decisions d ON d.bot_id = "
                           "f.bot_id AND d.ts = f.decided_at WHERE f.leg = 'open' AND f.trade_id IN (?, ?) "
                           "ORDER BY f.bot_id", (trade, copied))
        assert [(bot, ts) for bot, ts, _ in opened] == [(learner, after), (twin, after)]
        assert opened[0][2] in ("open_long", "flip") and opened[1][2] == opened[0][2]
        assert query(db, "SELECT bot_id, decided_at, reason FROM fills WHERE leg = 'close' AND trade_id IN (?, ?) "
                         "ORDER BY bot_id", (trade, copied)) == [(learner, close, SESSION_CLOSE),
                                                                 (twin, close, SESSION_CLOSE)]
        assert query(db, "SELECT exit_reason FROM trades WHERE id IN (?, ?)", (trade, copied)) == \
            [(SESSION_CLOSE,), (SESSION_CLOSE,)]
    for sql in ("SELECT * FROM decisions ORDER BY id", f"SELECT {FILL_COLUMNS} FROM fills ORDER BY id",
                "SELECT * FROM trades ORDER BY id"):
        first, second = (query(db, sql) for db in keeper.dbs)
        assert first and first == second, sql


def test_s2_marks_the_line_each_primary_score_uses_even_when_it_was_read_first(tmp_path):
    c1, c2 = pair(tmp_path, size="1")
    engines = [StockEngine.fresh(c, StockStore.open(c.db_path), MON.open_ms) for c in (c1, c2)]
    t = MON.open_ms

    def line(seq, item, scorer, status, route="llm"):
        impacts = (Impact("AAPL", 0.4, None if scorer == FINBERT_SCORER else True),) if status == "scored" else ()
        return ScoreLine(seq, item, "finnhub", "media", route, scorer, status, t + seq * 1000, ("AAPL",), impacts,
                         recv_ms=t + seq * 1000 + 500)

    lines = [line(1, 1, FINBERT_SCORER, "scored"), line(2, 1, "haiku_v1", "failed"), line(3, 2, "haiku_v1", "scored"),
             line(4, 2, FINBERT_SCORER, "scored"), line(3, 2, "haiku_v1", "scored"),
             line(5, 3, FINBERT_SCORER, "scored", route="finbert"), line(6, 2, "sonnet_v1", "scored"),
             line(7, 2, "haiku_v1", "scored")]
    for item in lines:
        for engine in engines:
            engine.on_news(item)
    for engine in engines:
        engine.persist(t + 60_000)
        engine.store.close()
    assert query(c1.db_path, "SELECT count(*) FROM news_seen") == [(0,)]
    assert query(c2.db_path, "SELECT score_seq, item_seq, scorer, status, primary_use, fallback FROM news_seen "
                             "ORDER BY score_seq") == [
        (1, 1, FINBERT_SCORER, "scored", 1, 1), (2, 1, "haiku_v1", "failed", 0, 0), (3, 2, "haiku_v1", "scored", 1, 0),
        (4, 2, FINBERT_SCORER, "scored", 0, 0), (5, 3, FINBERT_SCORER, "scored", 1, 0),
        (6, 2, "sonnet_v1", "scored", 0, 0), (7, 2, "haiku_v1", "scored", 0, 0)]
    assert [e.news_seq for e in engines] == [7, 7] and engines[1].inputs.news.streams.ignored == 1


def test_every_label_matches_its_recomputation_from_the_candles(week):
    for db in week.dbs:
        for symbol in week.symbols:
            candles = [StockCandle(symbol, parse_iso(ts), k, None, None, None, None, ref, 0.0, 0.0, 0.0, 0, 0, 0,
                                   bool(valid))
                       for ts, k, ref, valid in query(db, "SELECT ts, session_k, ref_price, valid FROM stock_candles "
                                                          "WHERE symbol = ? ORDER BY session_k", (symbol,))]
            rebuilt = {iso(r.ts): (r.status, r.label_bps) for r in recompute_labels(candles, 30)}
            stored = {ts: (status, value) for ts, status, value in
                      query(db, "SELECT ts, status, label_bps FROM stock_labels WHERE symbol = ?", (symbol,))}
            settled = {ts: row for ts, row in stored.items() if row[0] != "pending"}
            assert set(stored) == set(rebuilt) and len(settled) > len(stored) - 2
            for ts, (status, value) in settled.items():
                assert rebuilt[ts][0] == status and rebuilt[ts][1] == pytest.approx(value, rel=1e-12)
            assert {status for status, _ in settled.values()} == {"matured", "gap"}


def test_a_15_minute_feed_outage_flattens_every_open_position(week):
    assert week.open_at_outage
    after = [o.m for o in week.minutes if at(FRI, "14:15") < o.m.boundary_ms <= at(FRI, "14:30")]
    back = next(m for m in after if m.healthy.all())
    assert set(back.flatten) == {"gap_flatten"} and min(back.gaps) > 15
    for db in week.dbs:
        fills = query(db, "SELECT bot_id, trade_id FROM fills WHERE reason = 'gap_flatten' AND decided_at = ?",
                      (iso(back.boundary_ms),))
        assert sorted(fills) == sorted((i, v[2]) for i, v in week.open_at_outage.items())
        rejected = query(db, "SELECT DISTINCT reason FROM fills WHERE status = 'rejected' AND decided_at BETWEEN ? "
                             "AND ?", between(FRI, "13:59", "14:15"))
        assert set(rejected) <= {("disconnected",)}
        assert query(db, "SELECT count(*) FROM feed_events WHERE kind = 'resume' AND ts = ?",
                     (iso(back.boundary_ms),)) == [(8,)]


def test_a_20_minute_halt_flattens_only_that_symbol(week):
    assert week.open_at_halt and week.open_elsewhere_at_halt
    index = week.symbols.index(week.halted)
    during = [o.m for o in week.minutes if at(FRI, "13:00") < o.m.boundary_ms <= at(FRI, "13:59")]
    back = next(m for m in during if m.healthy[index] and m.boundary_ms > at(FRI, "13:20"))
    assert back.flatten[index] == "gap_flatten" and back.gaps[index] > 20
    assert all(m.flatten[i] is None for m in during for i in range(8) if i != index)
    for db in week.dbs:
        fills = query(db, "SELECT bot_id, symbol FROM fills WHERE reason = 'gap_flatten' AND decided_at BETWEEN ? "
                          "AND ?", between(FRI, "13:00", "13:59"))
        assert sorted(fills) == sorted((i, week.halted) for i in week.open_at_halt)
        events = query(db, "SELECT symbol FROM feed_events WHERE kind = 'resume' AND ts BETWEEN ? AND ?",
                       between(FRI, "13:00", "13:59"))
        assert events == [(week.halted,)]


def test_a_15_minute_news_outage_pauses_rent_learning_and_decisions_without_touching_prices(week):
    friday = [o for o in week.minutes if FRI.open_ms < o.m.boundary_ms <= FRI.close_ms]
    out = [i for i, o in enumerate(friday) if not o.m.news_ok]
    assert len(out) == 15 and out == list(range(out[0], out[0] + 15))
    assert friday[out[0]].m.boundary_ms == at(FRI, "12:02")
    for i in out:
        o, before = friday[i], friday[i - 1]
        assert o.rent == before.rent and o.trained == before.trained
        assert o.m.pushed == () and o.m.healthy.all() and o.m.valid.all()
        assert o.m.flatten == (None,) * 8
    assert friday[out[0] - 1].epochs == friday[out[-1] + 1].epochs == friday[out[-1] + 10].epochs
    assert friday[out[-1] + 1].rent != friday[out[-1]].rent
    first, last = iso(friday[out[0]].m.boundary_ms), iso(friday[out[-1]].m.boundary_ms)
    for db in week.dbs:
        for table in ("decisions", "stock_forecasts"):
            assert query(db, f"SELECT count(*) FROM {table} WHERE ts BETWEEN ? AND ?", (first, last)) == [(0,)]
        assert query(db, "SELECT count(*) FROM stock_candles WHERE valid = 1 AND ts BETWEEN ? AND ?",
                     (first, last)) == [(15 * 8,)]
        assert query(db, "SELECT count(*) FROM feed_events WHERE kind IN ('stale', 'closed') AND ts BETWEEN ? AND ?",
                     (first, last)) == [(0,)]
        assert query(db, "SELECT count(*) FROM fills WHERE reason LIKE '%flatten' AND decided_at BETWEEN ? AND ?",
                     between(FRI, "12:00", "12:59")) == [(0,)]
        assert query(db, "SELECT kind, detail FROM feed_events WHERE kind LIKE 'news_%' AND ts BETWEEN ? AND ? "
                         "ORDER BY id", between(FRI, "12:00", "12:20")) == \
            [("news_unhealthy", "heartbeat_old"), ("news_healthy", None)]
    assert query(week.dbs[0], "SELECT count(*) FROM news_minutes") == [(0,)]
    assert query(week.dbs[1], "SELECT count(*), min(reason), max(reason) FROM news_minutes WHERE healthy = 0") == \
        [(15, "heartbeat_old", "heartbeat_old")]
    # The next session's one session return reaches back to the outage minutes, whose prices stayed healthy.
    monday = {o.m.k: o.m for o in week.minutes if MON.open_ms < o.m.boundary_ms <= MON.close_ms}
    assert all(monday[friday[i].m.k + 390].valid.all() for i in out)
    outage = [o.m.k for o in friday if at(FRI, "14:01") <= o.m.boundary_ms <= at(FRI, "14:15")]
    assert not any(monday[k + 390].valid.any() for k in outage)


def test_a_death_leaves_its_slot_empty_and_nothing_replaces_it(week):
    assert week.active_before_death
    for engine, population in zip(week.engines, week.killed):
        assert population.learners[DRAINED_SLOT] is None and engine.population.learners[DRAINED_SLOT] is None
        assert engine.deaths == 1 and not engine.dying
        twin = engine.population.twins[DRAINED_SLOT]
        assert twin is not None and twin.state != DEAD
    for db in week.dbs:
        assert query(db, "SELECT count(*), max(bot_id) FROM bots") == [(48, 48)]
        assert query(db, "SELECT bot_id, state, died_at IS NOT NULL FROM bots WHERE state = 'dead'") == \
            [(week.drained, "dead", 1)]
        (died_at, cause, threshold), = query(db, "SELECT died_at, cause, threshold FROM deaths")
        assert between(FRI, "11:46", "11:47")[0] <= died_at < between(FRI, "11:46", "11:47")[1]
        assert cause == "rent" and 1.5 < threshold < 3
        twin_id = week.drained + 24
        assert query(db, "SELECT count(*) FROM trades WHERE bot_id = ? AND opened_at > ?",
                     (twin_id, died_at)) == [(0,)]
        assert query(db, "SELECT count(*) FROM decisions WHERE bot_id = ? AND ts > ?",
                     (week.drained, died_at)) == [(0,)]


def test_rent_only_in_usable_minutes(week):
    per_minute = costs.rent_per_minute(week.c1.stock_costs)
    for _, _, rent, usable in week.rent:
        assert rent == pytest.approx(per_minute * usable, rel=1e-9)
    usable = {symbol: count for _, symbol, _, count in week.rent}
    minutes = len(week.minutes)
    assert len({count for _, symbol, _, count in week.rent if symbol == week.halted}) == 1
    others = [count for symbol, count in usable.items() if symbol != week.halted]
    assert max(usable.values()) < minutes and usable[week.halted] < min(others)
    assert minutes == 4 * 390 - 6


def test_the_invariants_hold_at_every_snapshot(week):
    for db in week.dbs:
        assert query(db, "SELECT count(*) FROM feed_events WHERE kind = 'invariant_violation'") == [(0,)]
        (count, worst_cash, worst_equity), = query(db, """
            SELECT count(*), max(abs(cash - (10000 + realized_gross - fees_total - funding_total - rent_total))),
                   max(abs(equity - cash - unrealized)) FROM equity_snapshots""")
        assert count > 47 * 78 * 3 and worst_cash < 1e-6 and worst_equity < 1e-9
        assert query(db, "SELECT count(DISTINCT ts) FROM equity_snapshots WHERE ts BETWEEN ? AND ?",
                     between(MON, "09:31", "16:00")) == [(78,)]


# Trades of learners that kept their position at one decision or more after they opened.
KEPT_TRADES = """SELECT t.exit_reason, count(*) FROM trades t WHERE EXISTS (SELECT 1 FROM decisions d WHERE d.bot_id =
    t.bot_id AND d.action = 'hold' AND d.ts > t.opened_at AND (t.closed_at IS NULL OR d.ts < t.closed_at))
    GROUP BY t.exit_reason"""


def test_kept_positions_send_no_order_and_still_end_by_stops_closes_and_flattens(week):
    for db in week.dbs:
        holds = dict(query(db, "SELECT position_before, count(*) FROM decisions WHERE action = 'hold' "
                               "GROUP BY position_before"))
        assert set(holds) == {"long", "short"} and min(holds.values()) > 50
        assert query(db, "SELECT DISTINCT b.origin FROM decisions d JOIN bots b ON b.bot_id = d.bot_id WHERE "
                         "d.action = 'hold'") == [(LEARNER,)]
        assert query(db, "SELECT count(*) FROM decisions d JOIN bots b ON b.bot_id = d.bot_id WHERE b.origin = ?",
                     (TWIN,))[0][0] > 0
        assert query(db, "SELECT count(*) FROM decisions d JOIN fills f ON f.bot_id = d.bot_id AND f.decided_at = "
                         "d.ts WHERE d.action = 'hold'") == [(0,)]
        for before, reason in query(db, "SELECT position_before, reason FROM decisions WHERE action = 'hold'"):
            assert reason.startswith("the prediction of ") and reason.endswith(f"so the {before} is kept"), reason
        ended = dict(query(db, KEPT_TRADES))
        assert {"stop", "session_close", "signal", "gap_flatten", "session_flatten"} <= set(ended)


FILL_COLUMNS = ("id, bot_id, trade_id, decided_at, filled_at, leg, side, qty, ref_mid, book_avg_price, price, "
                "levels_consumed, slippage_bps, fee, latency_ms, status, reason, symbol, print_price, allowance_bps, "
                "notional, sec_fee, taf_fee, print_after_due_ms")


def test_s1_and_s2_write_identical_rows_while_s2_reads_no_news(week):
    for sql in ("SELECT * FROM decisions ORDER BY id", f"SELECT {FILL_COLUMNS} FROM fills ORDER BY id",
                "SELECT * FROM trades ORDER BY id", "SELECT * FROM equity_snapshots ORDER BY bot_id, ts",
                "SELECT * FROM stock_labels ORDER BY ts, symbol", "SELECT * FROM deaths",
                "SELECT * FROM benchmark ORDER BY ts", "SELECT * FROM price_z ORDER BY k, symbol",
                "SELECT * FROM stock_forecasts WHERE predictor IN ('population', 'linear_prices') ORDER BY ts, symbol, "
                "predictor"):
        first, second = (query(db, sql) for db in week.dbs)
        assert first and first == second, sql
    assert query(week.dbs[0], "SELECT count(*) FROM decisions WHERE action = 'hold'")[0][0] > 100   # holds included
    # The news window is S2's alone, null in S1, which reads no news, and 0 in S2, which read none this week.
    assert query(week.dbs[0], "SELECT DISTINCT news_window FROM fills") == [(None,)]
    assert query(week.dbs[1], "SELECT DISTINCT news_window FROM fills") == [(0,)]
    s2 = dict(query(week.dbs[1], "SELECT predictor, count(*) FROM stock_forecasts GROUP BY predictor"))
    assert s2["linear_news"] == s2["linear_prices"] and s2["news"] > s2["population"]
    assert query(week.dbs[1], "SELECT max(abs(prediction)), max(items), max(inputs_max_recv_at) FROM stock_forecasts "
                              "WHERE predictor = 'news'") == [(0.0, 0, None)]
    assert set(query(week.dbs[0], "SELECT DISTINCT predictor FROM stock_forecasts")) == \
        {("population",), ("linear_prices",)}


def test_every_fill_print_traded_at_or_after_due_and_records_how_long_after(week):
    for db in week.dbs:
        (filled, lowest, highest), = query(db, "SELECT count(*), min(print_after_due_ms), max(print_after_due_ms) "
                                               "FROM fills WHERE status = 'filled'")
        # The tape prints every second with vendor time 100 ms before the second, so with a due 1,000 to 1,300 ms after
        # the decision every order fills on the print traded 1,900 ms after it, never the one traded at 900 ms.
        assert filled > 100 and 600 <= lowest and highest <= 900
        assert query(db, "SELECT DISTINCT latency_ms + print_after_due_ms FROM fills WHERE status = 'filled'") == \
            [(1900,)]
        assert query(db, "SELECT count(*) FROM fills WHERE status = 'filled' AND print_after_due_ms IS NULL") == [(0,)]
        assert query(db, "SELECT count(*) FROM fills WHERE status = 'rejected' AND print_after_due_ms IS NOT NULL") \
            == [(0,)]
        assert query(db, "SELECT count(*) FROM fills WHERE status = 'rejected'")[0][0] > 0


def test_price_z_holds_the_z_scores_of_every_usable_symbol_at_each_evaluation_minute(week):
    minutes = [o.m for o in week.minutes + week.wednesday + week.after_restart_in_session]
    seen = invalid = 0
    for db in week.dbs:
        stored = {(k, symbol): (ts, z) for k, symbol, ts, *z in query(db, "SELECT * FROM price_z")}
        written = {ts for ts, in query(db, "SELECT DISTINCT ts FROM stock_candles")}
        for m in minutes:
            if iso(m.boundary_ms) not in written:
                continue                        # a minute whose rows a kill lost
            for i, symbol in enumerate(week.symbols):
                row = stored.pop((m.k, symbol), None)
                if not (m.evaluation and m.healthy[i] and m.news_ok):
                    assert row is None, (iso(m.boundary_ms), symbol)
                    continue
                ts, z = row
                assert ts == iso(m.boundary_ms)
                if m.valid[i]:
                    assert z == list(m.z[i]) and not any(v is None for v in z)
                else:
                    assert z == [None] * 10
                    invalid += 1
                seen += 1
        assert not stored                       # no row for a minute or symbol outside the rule
    assert seen > 2 * 100 and invalid > 0
    # Every row sits at an evaluation minute of one of its symbol's candles, and matches the news forecasts in S2.
    for db in week.dbs:
        assert query(db, "SELECT count(*) FROM price_z p LEFT JOIN stock_candles c ON c.symbol = p.symbol "
                         "AND c.ts = p.ts AND c.session_k = p.k WHERE c.ts IS NULL") == [(0,)]
    assert query(week.dbs[1], "SELECT ts, symbol FROM price_z ORDER BY ts, symbol") == \
        query(week.dbs[1], "SELECT ts, symbol FROM stock_forecasts WHERE predictor = 'news' ORDER BY ts, symbol")


def test_s2_flags_a_fill_whose_symbol_had_news_in_the_30_minutes_before_its_decision(tmp_path):
    c1, c2 = pair(tmp_path, size="1")
    engines = [StockEngine.fresh(c, StockStore.open(c.db_path), MON.open_ms) for c in (c1, c2)]
    s1, s2 = engines
    t = at(MON, "10:30")

    def line(seq, item, scorer, recv, tickers=("AAPL",), route="llm"):
        impact = Impact(tickers[0], 0.4, None if scorer == FINBERT_SCORER else True)
        return ScoreLine(seq, item, "finnhub", "media", route, scorer, "scored", recv - 300, tickers, (impact,),
                         recv_ms=recv)

    # Item 1 tagged AAPL first arrives 40 minutes before t and again 10 minutes before, item 2 tagged AAPL arrives 25
    # minutes before t, item 3 tagged MSFT 5 minutes before and item 4 tagged NVDA exactly at t.
    minute = MS_PER_MINUTE
    lines = [line(1, 1, FINBERT_SCORER, t - 40 * minute), line(2, 2, FINBERT_SCORER, t - 25 * minute, route="finbert"),
             line(3, 1, HAIKU_SCORER, t - 10 * minute),
             line(4, 3, FINBERT_SCORER, t - 5 * minute, ("MSFT",), route="finbert"),
             line(5, 4, FINBERT_SCORER, t, ("NVDA",), route="finbert")]
    for item in lines:
        for engine in engines:
            engine.on_news(item)
    assert s1.news_window("AAPL", t) is None
    assert s2.news_window("AAPL", t - 40 * minute) is False      # a line received at the decision is not before it
    assert s2.news_window("AAPL", t - 40 * minute + 1) is True
    assert s2.news_window("AAPL", t - 25 * minute + NEWS_WINDOW_MS) is True
    assert s2.news_window("AAPL", t - 25 * minute + NEWS_WINDOW_MS + 1) is False   # item 1 counts from its first line
    assert s2.news_window("MSFT", t) is True and s2.news_window("MSFT", t - 5 * minute) is False
    assert s2.news_window("NVDA", t) is False and s2.news_window("NVDA", t + 1) is True
    # A company item reaches QQQ's inputs, so it counts for QQQ too, and nothing reached TSLA.
    assert s2.news_window("QQQ", t) is True and s2.news_window("QQQ", t - 40 * minute) is False
    assert s2.news_window("TSLA", t + minute) is False
    # A MACRO item reaches every symbol's inputs, so it counts for each of them.
    for engine in engines:
        engine.on_news(line(6, 5, FINBERT_SCORER, t + 60 * minute, ("MACRO",), route="finbert"))
    assert all(s2.news_window(s, t + 61 * minute) is True for s in c2.stock_feed.symbols)
    assert s2.news_window("TSLA", t + 60 * minute) is False
    for engine in engines:
        bot = engine.population.learners[0]
        bot._send(engine.broker, [Leg("open", LONG, 2.0)], t, 250.0, "decision", BotStep())
        engine._register(bot)
        due = bot.pending_order.due_ms
        engine.on_prints(due + 50, (Print("AAPL", 250.1, 10.0, due - 1, ()), Print("AAPL", 250.2, 10.0, due + 30, ())))
        engine.persist(t + 60_000)
        engine.store.close()
    assert query(c1.db_path, "SELECT print_price, print_after_due_ms, news_window FROM fills") == [(250.2, 30, None)]
    assert query(c2.db_path, "SELECT print_price, print_after_due_ms, news_window FROM fills") == [(250.2, 30, 1)]


def test_a_split_keeps_features_invalid_for_391_minutes_restricts_shorts_and_a_hash_change_stops_news(tmp_path):
    first, second, third = (sessions.session_on(date(2026, 10, d)) for d in (28, 29, 30))
    c1, c2, group, tape = start(tmp_path, at(first, "09:00"), size="1")
    tape.run_to(at(first, "16:05"))
    tape.skip_to(at(second, "09:28"))
    tape.prices[0] *= 0.5  # AAPL opens at half its last price, a move the split guard and the restriction both see
    tape.run_to(at(second, "16:05"))
    tape.skip_to(at(third, "09:28"))
    tape.run_to(at(third, "09:33"))
    tape.hashes = (HASHES[0], "q" * 64, HASHES[2])
    tape.run_to(at(third, "09:40"))
    for engine in group.engines:
        engine.persist(group.now)
        engine.store.close()
    by_day = {}
    for o in group.minutes:
        by_day.setdefault(o.m.session.day, []).append(o.m)
    assert all(m.valid[1:].all() and not m.valid[0] for m in by_day[second.day])
    assert all(not m.shorts_ok[0] and m.shorts_ok[1:].all() for m in by_day[second.day] + by_day[third.day])
    assert by_day[third.day][0].k == by_day[second.day][0].k + 390 and by_day[third.day][0].valid.all()
    reasons = [m.news_reason for m in by_day[third.day]]
    assert reasons[:3] == [None] * 3 and set(reasons[3:]) == {"hash_mismatch"}
    for db in (c1.db_path, c2.db_path):
        guard = query(db, "SELECT ts, symbol FROM feed_events WHERE kind = 'split_guard'")
        assert guard == [(iso(at(second, "09:31")), "AAPL")]
        assert query(db, "SELECT kind, detail FROM feed_events WHERE kind LIKE 'news_%'") == \
            [("news_unhealthy", "hash_mismatch")]
        assert query(db, "SELECT count(*) FROM decisions WHERE ts > ?", (iso(at(third, "09:33")),)) == [(0,)]


def test_orders_in_flight_at_a_restart_are_rejected_and_a_closed_session_rejects_every_order(tmp_path):
    c1, _ = pair(tmp_path, size="1")
    engine = StockEngine.fresh(c1, StockStore.open(c1.db_path), MON.open_ms)
    bot = engine.population.learners[0]
    bot.pending_order = engine.broker.make_order(bot.bot_id, bot.symbol, [Leg("open", LONG, 2.0)],
                                                 MON.open_ms + 600_000, 250.0, "decision", bot.rng)
    engine.resume(MON.open_ms + 700_000)
    engine.persist(MON.open_ms + 730_000)
    assert bot.pending_order is None and not engine.pending
    assert query(c1.db_path, "SELECT bot_id, status, reason FROM fills") == [(bot.bot_id, "rejected", "restart")]
    assert query(c1.db_path, "SELECT kind, detail FROM feed_events") == \
        [("resume", "resumed, 1 orders in flight rejected")]
    engine.store.close()
    market = MarketLayer(c1, HASHES)
    market.advance(MON.open_ms - 60_000)
    assert market.rejection("AAPL") == "session_closed"


def test_every_decision_gets_the_label_of_its_symbol_from_its_minute(week):
    for db in week.dbs:
        rows = query(db, """
            SELECT d.bot_id, d.label_bps, a.ref_price, b.ref_price, a.session_k - s.first_k FROM decisions d
            JOIN stock_candles a ON a.symbol = d.symbol AND a.ts = d.ts
            LEFT JOIN stock_candles b ON b.symbol = d.symbol AND b.session_k = a.session_k + 30
            JOIN (SELECT session_date, min(session_k) AS first_k FROM stock_candles GROUP BY session_date) s
              ON s.session_date = a.session_date""")
        labelled = [r for r in rows if r[1] is not None]
        assert len(labelled) > 0.8 * len(rows)
        assert all(r[1] == pytest.approx(1e4 * np.log(r[3] / r[2]), rel=1e-9, abs=1e-9) for r in labelled)
        assert all(minute < 360 for *_, minute in labelled)
        twins = {bot_id for bot_id, *_ in labelled if bot_id > 24}
        assert twins and query(db, "SELECT count(*) FROM decisions WHERE symbol IS NULL OR symbol = ''") == [(0,)]
