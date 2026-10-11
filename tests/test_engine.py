"""The engine end to end on synthetic recordings: trading, deaths, labels, health gates, outages and restarts."""

import math
import sqlite3

import numpy as np
import pytest

from btc_survivor.account import CAUSES
from btc_survivor.clock import MS_PER_MINUTE, iso, minute_index, parse_iso
from btc_survivor.engine import Engine, EngineError
from btc_survivor.feed.base import load_adapter
from btc_survivor.clock import LiveClock
from btc_survivor.replay import replay_config, replay_lines
from btc_survivor.store import Store
from btc_survivor.synthetic import Fault
from btc_survivor.types import ASK, BID, LONG, BookDelta, FeedNotice, Leg
from enginekit import TABLES, START, deadly_config, digest, fast_config, feed, new_engine, synthetic


@pytest.fixture(scope="module")
def deadly(tmp_path_factory):
    """Two hours of trading at ruinous fees, with a 400 second disconnect after 25 minutes."""
    cfg = deadly_config(tmp_path_factory.mktemp("deadly"))
    engine = new_engine(cfg)
    feed(engine, synthetic(5, 7200, (Fault(1500, "disconnect", 400),)))
    engine.finish()
    db = sqlite3.connect(cfg.db_path)
    db.row_factory = sqlite3.Row
    yield cfg, engine, db
    db.close()
    engine.store.close()


def healthy_engine(tmp_path, seconds=200):
    cfg = fast_config(tmp_path)
    engine = new_engine(cfg)
    lines = synthetic(3, seconds)
    feed(engine, lines)
    assert engine.market.health.is_healthy
    return engine, lines[-1][3]


def test_dead_bots_are_replaced_in_the_same_minute_and_lineage_grows(deadly):
    cfg, engine, db = deadly
    size = cfg.population.size
    assert engine.deaths >= 10
    assert all(r["living"] == size for r in db.execute("SELECT living FROM benchmark"))
    assert db.execute("SELECT count(*) FROM bots").fetchone()[0] == 2 * size + engine.deaths
    assert db.execute("SELECT count(*) FROM deaths").fetchone()[0] == engine.deaths - len(engine.dying)
    living = db.execute("SELECT origin = 'control' AS twin, count(*) AS n FROM bots "
                        "WHERE state IN ('warmup', 'active') GROUP BY twin").fetchall()
    assert sorted((r["twin"], r["n"]) for r in living) == [(0, size), (1, size)]
    for slot in range(size):
        rows = db.execute("SELECT born_at, died_at FROM bots WHERE slot = ? AND origin != 'control' ORDER BY bot_id",
                          (slot,)).fetchall()
        for before, after in zip(rows, rows[1:]):
            assert after["born_at"] <= before["died_at"]  # the slot never waits for a dying bot's close
    lineage = db.execute("SELECT l.generation, p.generation AS parent_generation, p.origin AS parent_origin "
                         "FROM lineage l JOIN bots p ON p.bot_id = l.parent_id").fetchall()
    assert lineage and max(r["generation"] for r in lineage) >= 2
    assert all(r["generation"] == r["parent_generation"] + 1 and r["parent_origin"] != "control" for r in lineage)


def test_books_balance_for_every_bot_and_every_death_adds_up(deadly):
    cfg, engine, db = deadly
    assert not db.execute("SELECT 1 FROM feed_events WHERE kind = 'invariant_violation'").fetchall()
    for bot in engine.all_bots():
        assert bot.account.invariant_errors(bot.mark_price(engine.market.book)) == []
    deaths = db.execute("SELECT * FROM deaths").fetchall()
    assert deaths
    for d in deaths:
        parts = {"fees": d["fees_total"], "funding": d["funding_total"], "rent": d["rent_total"],
                 "stop_losses": d["stop_loss_total"], "trading_losses": d["other_losses_total"]}
        assert cfg.trading.start_equity - d["final_equity"] == pytest.approx(sum(parts.values()) - d["writeoff"])
        assert d["cause"] == max(CAUSES, key=lambda c: parts[c]) and d["final_equity"] >= 0


def test_death_closes_finish_the_death_when_they_fill(deadly):
    _, _, db = deadly
    closes = db.execute("SELECT bot_id, filled_at FROM fills WHERE reason = 'death' AND status = 'filled'").fetchall()
    assert closes
    for row in closes:
        died = db.execute("SELECT d.died_at, b.state, b.died_at AS bot_died FROM deaths d "
                          "JOIN bots b USING (bot_id) WHERE bot_id = ?", (row["bot_id"],)).fetchone()
        assert died["died_at"] == row["filled_at"] == died["bot_died"] and died["state"] == "dead"


def test_trades_and_fills_are_consistent(deadly):
    cfg, _, db = deadly
    costs = cfg.costs
    closed = db.execute("SELECT * FROM trades WHERE closed_at IS NOT NULL").fetchall()
    assert len(closed) > 50
    for t in closed:
        assert t["pnl_net"] == pytest.approx(t["pnl_gross"] - t["fee_open"] - t["fee_close"] - t["funding"])
    for f in db.execute("SELECT * FROM fills WHERE status = 'filled'"):
        assert costs.latency_ms <= f["latency_ms"] <= costs.latency_ms + costs.latency_jitter_ms
        assert parse_iso(f["filled_at"]) >= parse_iso(f["decided_at"]) + f["latency_ms"]
    opens = db.execute("SELECT f.price, t.entry_price FROM fills f JOIN trades t ON t.id = f.trade_id "
                       "WHERE f.leg = 'open' AND f.status = 'filled'").fetchall()
    assert opens and all(r["price"] == r["entry_price"] for r in opens)


def test_labels_written_back_match_the_candles_and_never_span_a_gap(deadly):
    cfg, _, db = deadly
    h = cfg.time.horizon_minutes * MS_PER_MINUTE
    candles = {r["ts"]: (r["mid"], r["valid"]) for r in db.execute("SELECT ts, mid, valid FROM candles")}
    labelled = db.execute("SELECT ts, label_bps FROM decisions WHERE label_bps IS NOT NULL").fetchall()
    assert len(labelled) > 100
    for r in labelled:
        start = parse_iso(r["ts"])
        span = [candles[iso(start + k * MS_PER_MINUTE)] for k in range(h // MS_PER_MINUTE + 1)]
        assert all(valid for _, valid in span)
        assert r["label_bps"] == pytest.approx(1e4 * math.log(span[-1][0] / span[0][0]))


def test_decisions_only_at_healthy_phase_minutes_and_twins_copy_learner_timing(deadly):
    cfg, _, db = deadly
    valid = {r["ts"] for r in db.execute("SELECT ts FROM candles WHERE valid = 1")}
    assert len(valid) < db.execute("SELECT count(*) FROM candles").fetchone()[0]  # the disconnect cost minutes
    rows = db.execute("SELECT d.bot_id, d.ts, d.action, d.reason, b.slot, b.origin FROM decisions d "
                      "JOIN bots b USING (bot_id)").fetchall()
    moves = {(r["bot_id"], r["ts"]) for r in rows
             if r["origin"] != "control" and r["action"] in ("open_long", "open_short", "close", "flip")}
    twin_rows = [r for r in rows if r["origin"] == "control"]
    assert twin_rows and any(r["action"].startswith("open") for r in twin_rows)
    for r in rows:
        assert r["ts"] in valid
        assert minute_index(parse_iso(r["ts"])) % cfg.time.horizon_minutes == r["slot"] % cfg.time.horizon_minutes
    for r in twin_rows:
        learner = int(r["reason"].split("twin of bot ")[1].split()[0].rstrip(","))
        assert (learner, r["ts"]) in moves


def test_checkpoints_sit_thirty_seconds_after_their_boundary(deadly):
    _, _, db = deadly
    stamps = [r[0] for r in db.execute("SELECT ts FROM checkpoints ORDER BY id")]
    assert len(stamps) == 3
    for ts in stamps[:-1]:  # the last one was written by finish
        assert ts.endswith(":30.000Z") and minute_index(parse_iso(ts)) % 2 == 0


def test_stale_minutes_charge_no_rent_and_move_no_bot(tmp_path):
    cfg = fast_config(tmp_path)
    engine = new_engine(cfg)
    feed(engine, synthetic(7, 2400, (Fault(1200, "silence", 30), Fault(1700, "disconnect", 200))))
    engine.finish()
    healthy = engine.store.query("SELECT count(*) FROM candles WHERE valid = 1")[0][0]
    total = engine.store.query("SELECT count(*) FROM candles")[0][0]
    assert healthy == engine.healthy_minutes and healthy < total - 4
    assert engine.store.query("SELECT count(*) FROM benchmark")[0][0] == healthy
    first = engine.population.learners[0]
    assert first.bot_id == 1 and first.alive_minutes == healthy
    assert first.account.rent == pytest.approx(healthy * cfg.costs.daily_running_cost / 1440)
    engine.store.close()


def test_order_fills_at_the_first_book_update_at_or_after_its_due_time(tmp_path):
    engine, conn = healthy_engine(tmp_path)
    bot, book = engine.population.learners[0], engine.market.book
    engine._send(bot, [Leg("open", LONG, engine.broker.open_qty(book.mid))], engine.now_ms, "decision")
    due = bot.pending_order.due_ms
    engine.on_event(BookDelta(due - 1, conn, [(BID, book.top(BID, 5)[-1][0], 3.0)]))
    assert bot.pending_order is not None and not bot.account.is_open
    best_ask = book.best_ask
    engine.on_event(BookDelta(due, conn, [(ASK, best_ask, 7.0)]))
    assert bot.pending_order is None and not engine.pending
    assert bot.account.entry_price == pytest.approx(best_ask * (1 + engine.cfg.costs.extra_slippage_bps / 1e4))
    engine.store.flush()
    fill = engine.store.query("SELECT * FROM fills")[0]
    assert fill["status"] == "filled" and fill["filled_at"] == iso(due)
    trade = engine.store.query("SELECT * FROM trades")[0]
    assert trade["side"] == "long" and trade["entry_price"] == bot.account.entry_price
    engine.store.close()


def test_leaving_healthy_rejects_every_pending_order(tmp_path):
    engine, conn = healthy_engine(tmp_path)
    bot = engine.population.learners[1]
    engine._send(bot, [Leg("open", LONG, 0.01)], engine.now_ms, "decision")
    engine.on_event(FeedNotice(engine.now_ms + 10, conn, "disconnected", detail="test"))
    assert bot.pending_order is None and not bot.account.is_open and not engine.pending
    engine.store.flush()
    fill = engine.store.query("SELECT status, reason, filled_at FROM fills")[0]
    assert tuple(fill) == ("rejected", "feed_not_healthy", None)
    engine.store.close()


@pytest.mark.parametrize("outage_s, flattened", [(20, False), (600, True)])
def test_outage_longer_than_the_limit_flattens_every_position(tmp_path, outage_s, flattened):
    cfg = fast_config(tmp_path)
    engine = new_engine(cfg)
    lines = synthetic(4, 1450, (Fault(630, "disconnect", outage_s),))
    used = feed(engine, lines, until_ms=START + 625_000)
    book = engine.market.book
    bots = engine.slot_bots()
    for i, bot in enumerate(bots):
        bot.account.open(LONG, engine.broker.open_qty(book.mid), book.best_ask, 0.0, engine.now_ms, trade_id=900 + i)
    feed(engine, lines[used:])
    engine.finish()
    reasons = [r[0] for r in engine.store.query("SELECT reason FROM fills WHERE status = 'filled'")]
    resumes = engine.store.query("SELECT detail FROM feed_events WHERE kind = 'resume'")
    if flattened:
        assert reasons == ["gap_flatten"] * len(bots) and not any(b.account.is_open for b in bots)
        assert len(resumes) == 1 and "gap_flatten" in resumes[0][0]
    else:
        assert reasons == [] and not resumes and all(b.account.is_open for b in bots)
    engine.store.close()


def checkpoint_state(engine, at):
    meta, arrays = engine._checkpoint_state(at)
    return meta, {k: v.copy() for k, v in arrays.items.items()}


def test_restart_restores_every_bot_bit_for_bit_and_charges_no_rent_for_downtime(tmp_path):
    cfg = fast_config(tmp_path)
    a = new_engine(cfg)
    feed(a, synthetic(6, 1800))
    waiting = a.population.learners[2]
    a._send(waiting, [Leg("open", LONG, 0.01)], a.now_ms, "decision")
    a.finish()
    meta_a, arrays_a = checkpoint_state(a, a.now_ms)
    a.store.close()

    later = a.now_ms + 2 * MS_PER_MINUTE
    b = Engine.start(cfg, Store.open(cfg.db_path), load_adapter(cfg.feed), later)
    meta_b, arrays_b = checkpoint_state(b, a.now_ms)
    assert sorted(arrays_a) == sorted(arrays_b)
    assert all(arrays_a[k].tobytes() == arrays_b[k].tobytes() for k in arrays_a)
    for d in meta_a["bots"]:
        d["pending_order"] = None  # a restored order is rejected with reason restart
    assert meta_b == {**meta_a, "epoch": meta_a["epoch"] + 1}
    assert b.restarted and b.now_ms == later and not b.pending
    assert b.series.valid_total == a.series.valid_total
    end = minute_index(a.last_healthy_ms)
    for got, want in zip(b.series.window(end, 16), a.series.window(end, 16)):
        assert np.array_equal(got, want, equal_nan=True)
    assert any(kind == "resume" for _, kind, _ in b.feed_rows)

    rent = {bot.bot_id: bot.account.rent for bot in b.slot_bots()}
    healthy = b.healthy_minutes
    feed(b, synthetic(8, 400, start=later))
    b.finish()
    gained = b.healthy_minutes - healthy
    assert gained >= 4
    for bot in b.slot_bots():
        assert bot.account.rent - rent[bot.bot_id] == pytest.approx(gained * cfg.costs.daily_running_cost / 1440)
    reasons = {tuple(r) for r in b.store.query("SELECT reason, status FROM fills")}
    assert ("restart", "rejected") in reasons and ("restart_flatten", "filled") not in reasons
    b.store.close()


def test_restart_after_a_long_gap_flattens_with_the_restart_reason(tmp_path):
    cfg = fast_config(tmp_path)
    a = new_engine(cfg)
    feed(a, synthetic(6, 900))
    book = a.market.book
    bots = a.slot_bots()
    for i, bot in enumerate(bots):
        bot.account.open(LONG, a.broker.open_qty(book.mid), book.best_ask, 0.0, a.now_ms, trade_id=900 + i)
    a.finish()
    a.store.close()
    later = a.now_ms + 30 * MS_PER_MINUTE
    b = Engine.start(cfg, Store.open(cfg.db_path), load_adapter(cfg.feed), later)
    feed(b, synthetic(8, 300, start=later))
    b.finish()
    reasons = [r[0] for r in b.store.query("SELECT reason FROM fills WHERE status = 'filled'")]
    assert reasons == ["restart_flatten"] * len(bots)
    details = [r[0] for r in b.store.query("SELECT detail FROM feed_events WHERE kind = 'resume'")]
    assert len(details) == 2 and "restored" in details[0] and "restart_flatten" in details[1]
    b.store.close()


def test_unreadable_newest_checkpoint_falls_back_and_rolls_back_newer_rows(tmp_path):
    cfg = fast_config(tmp_path)
    a = new_engine(cfg)
    feed(a, synthetic(6, 1500))
    a.store.close()
    db = sqlite3.connect(cfg.db_path)
    (newest, _), (older, older_ts) = db.execute("SELECT id, ts FROM checkpoints ORDER BY id DESC LIMIT 2").fetchall()
    db.execute("UPDATE checkpoints SET arrays = x'0001' WHERE id = ?", (newest,))
    db.commit()
    db.close()
    b = Engine.start(cfg, Store.open(cfg.db_path), load_adapter(cfg.feed), a.now_ms + MS_PER_MINUTE)
    for table, column in (("decisions", "ts"), ("candles", "ts"), ("fills", "decided_at")):
        assert not b.store.query(f"SELECT 1 FROM {table} WHERE {column} > ?", (older_ts,))
    # the fallback is committed at once with a fresh checkpoint, the repaired bot states and the reason
    assert [r[0] for r in b.store.query("SELECT ts FROM checkpoints WHERE ts > ?", (older_ts,))] == [iso(b.now_ms)]
    detail = b.store.query("SELECT detail FROM feed_events WHERE kind = 'resume'")[0][0]
    assert f"restored the checkpoint from {older_ts}" in detail and "rolled back" in detail
    states = {r["bot_id"]: r["state"] for r in b.store.query("SELECT bot_id, state FROM bots")}
    assert all(states[bot.bot_id] == bot.state for bot in b.slot_bots())
    b.store.close()


def test_history_without_a_readable_checkpoint_refuses_to_start(tmp_path):
    cfg = fast_config(tmp_path)
    a = new_engine(cfg)
    feed(a, synthetic(6, 400))
    a.store.close()
    db = sqlite3.connect(cfg.db_path)
    db.execute("UPDATE checkpoints SET meta = 'garbage'")
    db.commit()
    db.close()
    store = Store.open(cfg.db_path)
    with pytest.raises(EngineError, match="no readable checkpoint"):
        Engine.start(cfg, store, load_adapter(cfg.feed), a.now_ms + MS_PER_MINUTE)
    store.close()


def test_empty_database_starts_a_fresh_population(tmp_path):
    cfg = fast_config(tmp_path)
    store = Store.open(cfg.db_path)
    engine = Engine.start(cfg, store, load_adapter(cfg.feed), START)
    assert not engine.restarted and [b.bot_id for b in engine.slot_bots()] == list(range(1, 9))
    store.close()


def test_unreadable_message_counts_as_a_feed_error_and_asks_for_a_resync(tmp_path):
    engine, conn = healthy_engine(tmp_path)
    engine.on_line("message", "{not json", engine.now_ms + 10, conn)
    assert engine.market.take_resync() == "error"
    assert any(kind == "error" for _, kind, _ in engine.feed_rows)
    engine.store.close()


def test_benchmark_rows_carry_the_cash_baseline_and_the_population_averages(deadly):
    cfg, _, db = deadly
    rate, start = cfg.costs.daily_running_cost / 1440, cfg.trading.start_equity
    rows = db.execute("SELECT * FROM benchmark ORDER BY ts").fetchall()
    assert len(rows) > 60
    for n, row in enumerate(rows, 1):
        assert row["cash_equity"] == pytest.approx(start - rate * n)
        equity = {origin: [e for (e,) in db.execute(
            "SELECT s.equity FROM equity_snapshots s JOIN bots b USING (bot_id) WHERE s.ts = ? AND "
            f"b.origin {'=' if origin == 'control' else '!='} 'control'", (row["ts"],))] for origin in ("control", "learner")}
        assert len(equity["control"]) == len(equity["learner"]) == cfg.population.size
        assert row["twin_mean_equity"] == pytest.approx(np.mean(equity["control"]))
        assert row["living_mean_equity"] == pytest.approx(np.mean(equity["learner"]))


def die_with_a_position(engine, bot):
    """Give a bot an open long and push its cash under the death line, so it dies at the next healthy minute."""
    book = engine.market.book
    bot.account.open(LONG, engine.broker.open_qty(book.mid), book.best_ask, 0.0, engine.now_ms, trade_id=900 + bot.bot_id)
    bot.account.charge_rent(bot.account.cash - 5.0)


def close_fills(store, bot_id):
    return [tuple(r) for r in store.query("SELECT status, reason FROM fills WHERE bot_id = ? AND leg = 'close' "
                                           "ORDER BY id", (bot_id,))]


@pytest.mark.parametrize("restart", [False, True])
def test_rejected_death_close_is_resent_until_it_fills_even_across_a_restart(tmp_path, restart):
    cfg = fast_config(tmp_path)
    lines = synthetic(3, 600, (Fault(240.3, "disconnect", 20),))
    engine = new_engine(cfg)
    used = feed(engine, lines, until_ms=START + 230_000)
    die_with_a_position(engine, engine.population.learners[0])
    used += feed(engine, lines[used:], until_ms=START + 240_200)
    assert [b.bot_id for b in engine.dying] == [1] and engine.dying[0].pending_order.reason == "death"
    if restart:
        engine.finish()
        engine.store.close()
        engine = Engine.start(cfg, Store.open(cfg.db_path), load_adapter(cfg.feed), engine.now_ms)
        assert [b.bot_id for b in engine.dying] == [1] and engine.dying[0].pending_order is None
    feed(engine, lines[used:])
    engine.finish()
    first = "restart" if restart else "feed_not_healthy"
    assert close_fills(engine.store, 1) == [("rejected", first), ("filled", "death")]
    death = engine.store.query("SELECT d.died_at, b.state FROM deaths d JOIN bots b USING (bot_id) WHERE bot_id = 1")
    filled = engine.store.query("SELECT filled_at FROM fills WHERE bot_id = 1 AND status = 'filled'")[0][0]
    assert len(death) == 1 and tuple(death[0]) == (filled, "dead") and engine.dying == []
    engine.store.close()


def test_twin_close_copied_from_a_dying_learner_is_retried_after_a_rejection(tmp_path):
    cfg = fast_config(tmp_path)
    lines = synthetic(3, 600, (Fault(240.3, "disconnect", 20),))
    engine = new_engine(cfg)
    used = feed(engine, lines, until_ms=START + 230_000)
    learner, twin = engine.population.learners[0], engine.population.twins[0]
    book = engine.market.book
    twin.account.open(LONG, engine.broker.open_qty(book.mid), book.best_ask, 0.0, engine.now_ms, trade_id=950)
    die_with_a_position(engine, learner)
    feed(engine, lines[used:])
    engine.finish()
    assert close_fills(engine.store, twin.bot_id) == [("rejected", "feed_not_healthy"), ("filled", "copy_exit")]
    assert not twin.account.is_open and twin.state != "dead"
    engine.store.close()


def test_bot_that_dies_with_an_open_order_in_flight_closes_it_before_its_death_row(tmp_path):
    cfg = fast_config(tmp_path)
    lines = synthetic(3, 600)
    engine = new_engine(cfg)
    used = feed(engine, lines, until_ms=START + 299_600)
    bot = engine.population.learners[1]
    engine._send(bot, [Leg("open", LONG, engine.broker.open_qty(engine.market.book.mid))], engine.now_ms, "decision")
    bot.account.charge_rent(bot.account.cash - 5.0)
    feed(engine, lines[used:])
    engine.finish()
    fills = [tuple(r) for r in engine.store.query("SELECT leg, status, reason, filled_at FROM fills WHERE bot_id = 2")]
    assert [f[:3] for f in fills] == [("open", "filled", "decision"), ("close", "filled", "death")]
    assert engine.store.query("SELECT died_at FROM deaths WHERE bot_id = 2")[0][0] == fills[1][3]
    assert not bot.account.is_open and engine.dying == []
    engine.store.close()


def test_rejected_flatten_is_sent_again_at_the_next_healthy_minute(tmp_path):
    cfg = fast_config(tmp_path)
    lines = synthetic(4, 1500, (Fault(630, "disconnect", 600), Fault(1320.5, "disconnect", 20)))
    engine = new_engine(cfg)
    used = feed(engine, lines, until_ms=START + 625_000)
    book = engine.market.book
    bots = engine.slot_bots()
    for i, bot in enumerate(bots):
        bot.account.open(LONG, engine.broker.open_qty(book.mid), book.best_ask, 0.0, engine.now_ms, trade_id=900 + i)
    feed(engine, lines[used:])
    engine.finish()
    rows = [tuple(r) for r in engine.store.query("SELECT status, reason FROM fills ORDER BY id")]
    assert rows == [("rejected", "feed_not_healthy")] * len(bots) + [("filled", "gap_flatten")] * len(bots)
    details = [r[0] for r in engine.store.query("SELECT detail FROM feed_events WHERE kind = 'resume'")]
    assert len(details) == 2 and "previous flatten was rejected" in details[1]
    assert not any(b.account.is_open for b in bots)
    engine.store.close()


def test_resuming_with_a_different_population_size_or_seed_is_refused(tmp_path):
    cfg = fast_config(tmp_path)
    a = new_engine(cfg)
    feed(a, synthetic(6, 300))
    a.finish()
    a.store.close()
    for changed in (fast_config(tmp_path, population={"size": 5}), fast_config(tmp_path, run={"master_seed": 7})):
        store = Store.open(cfg.db_path)
        with pytest.raises(EngineError, match="does not match"):
            Engine.start(changed, store, load_adapter(cfg.feed), a.now_ms + MS_PER_MINUTE)
        store.close()


class ScriptedClock(LiveClock):
    def __init__(self) -> None:
        super().__init__()
        self.wall = 0

    def now_ms(self) -> int:
        return self.wall


def test_one_second_ticks_and_clamped_stamps_give_the_same_rows_as_replay(tmp_path):
    lines = synthetic(11, 5400, (Fault(900, "gap"), Fault(1700, "silence", 30), Fault(2100, "disconnect", 300)))
    cfg = replay_config(deadly_config(tmp_path / "unused"), tmp_path / "live")
    engine = Engine.fresh(cfg, Store.open(cfg.db_path), load_adapter(cfg.feed), lines[0][2])
    clock, stamped, tick = ScriptedClock(), [], lines[0][2] + 1000
    for kind, raw, t, conn in lines:
        while tick < t:
            clock.wall = tick + (1 if tick % MS_PER_MINUTE == 0 else 0)  # the ticker wakes just after a boundary
            engine.advance_to(clock.stamp(engine.now_ms))
            tick += 1000
        clock.wall = t - 200 if t % MS_PER_MINUTE < 100 else t  # the wall clock steps back below the boundary
        recv = clock.stamp(engine.now_ms)
        assert recv >= engine.now_ms and (not stamped or recv >= stamped[-1][2])
        stamped.append((kind, raw, recv, conn))
        engine.on_line(kind, raw, recv, conn)
    engine.finish()
    engine.store.close()
    assert any(s[2] != line[2] for s, line in zip(stamped, lines))  # some stamps really were clamped
    live = digest(cfg.db_path)
    replayed = digest(replay_lines(deadly_config(tmp_path / "unused"), stamped, tmp_path / "replay"))
    assert all(live[table] == replayed[table] for table in TABLES if table != "feed_events")
    assert sqlite3.connect(cfg.db_path).execute("SELECT count(*) FROM trades").fetchone()[0] > 0
