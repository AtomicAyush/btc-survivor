"""Skipped turns in the decisions explorer: the phase minutes a learner lived through with no decision row, the cause
the database shows for each, and how they page, filter, count and open beside the decisions."""

import contextlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from http.server import ThreadingHTTPServer

import pytest

from btc_survivor import dashboard, records
from btc_survivor.bot import DEAD
from btc_survivor.cli import data
from btc_survivor.clock import MS_PER_MINUTE, iso, minute_index, parse_iso
from btc_survivor.dashboard import SKIPPED, make_handler, parse_filters, skip_id
from btc_survivor.engine import Engine
from btc_survivor.features import VOLUME_WINDOW
from btc_survivor.feed.base import load_adapter
from btc_survivor.feed.candles import Candle
from btc_survivor.store import SCHEMA_VERSION, Store
from btc_survivor.synthetic import Fault
import dashkit
from enginekit import START, R, deadly_config, fast_config, feed, new_engine, synthetic

DAY_D = parse_iso("2026-10-13T00:00:00Z")  # the day of the run D case that asked for this view


class Run:
    """A run database written through the project's own Store and records, minute by minute as the engine writes it.

    Minutes count from base, which sits on a multiple of every horizon used here, so minute n is the phase minute of
    slot n modulo the horizon. A fraction of a minute places an event or a fill between boundaries.
    """

    def __init__(self, folder, base: int = START, **sections) -> None:
        self.cfg = fast_config(folder, **sections)
        self.base = base
        self.store = Store.open(self.cfg.db_path)

    def ms(self, minute: float) -> int:
        return self.base + round(minute * MS_PER_MINUTE)

    def bot(self, bot_id, slot, born, origin="random", died=None, state="active") -> None:
        self.store.queue(records.INSERT_BOT, (bot_id, slot, None, 0, iso(self.ms(born)), state, 8.0, 1e-4, 12345,
                                              origin))
        if died is not None:
            self.store.queue(records.SET_BOT_DEAD, (iso(self.ms(died)), bot_id))

    def minutes(self, first, last, unhealthy=(), missing=(), unpredicted=(), learners=(), undecided=()) -> None:
        """A candle for every minute the run closed, a benchmark row for every healthy one, forecast rows where a
        prediction was made, and then a decision of none for each living learner at its phase minute. learners holds
        (bot id, slot, birth minute, death minute or None)."""
        horizon = self.cfg.time.horizon_minutes
        for n in range(first, last + 1):
            if n in missing:
                continue
            healthy, ms, mid = n not in unhealthy, self.ms(n), 80_000.0 + n
            self.store.queue(records.INSERT_CANDLE, records.candle_row(
                Candle(ms, mid, mid, mid, mid, 0.5, 3, mid if healthy else None, healthy)))
            if not healthy:
                continue
            self.store.queue(records.INSERT_BENCHMARK, (iso(ms), mid, 1e4, 1e4, 1e4, 1e4, 2, 1e4, 1e4))
            if n in unpredicted:
                continue
            for name in ("population", "linear"):
                self.store.queue(records.INSERT_FORECAST, (iso(ms), name, 0.5))
            for bot_id, slot, born, died in learners:
                if (minute_index(ms) % horizon == slot % horizon and born < n and (died is None or n < died)
                        and (bot_id, n) not in undecided):
                    self.decision(bot_id, n)

    def decision(self, bot_id, minute, action="none", position="flat", prediction=0.5, reason=None, gate=0) -> None:
        self.store.queue(records.INSERT_DECISION, records.decision_row({
            "bot_id": bot_id, "ts": self.ms(minute), "position_before": position, "ref_mid": 80_000.0,
            "prediction_bps": prediction, "required_edge_bps": 20.0, "cost_estimate_bps": 12.0, "margin_bps": 8.0,
            "gate_passed": gate, "forced": 0, "action": action,
            "reason": reason or f"prediction {prediction:.2f} bps is below required edge 20.00 bps"}))

    def trade(self, trade_id, bot_id, side, opened, closed=None, exit_reason="decision") -> None:
        self.store.queue(records.OPEN_TRADE, (trade_id, bot_id, side, iso(self.ms(opened)), 0.00625, 80_000.0, 0.5,
                                              0.0, 0.4, 0))
        if closed is not None:
            self.store.queue(records.CLOSE_TRADE, (iso(self.ms(closed)), 80_050.0, 0.5, 0.0, 0.4, 0.3, -0.7,
                                                   exit_reason, trade_id))

    def event(self, minute, kind, detail=None) -> None:
        self.store.queue(records.INSERT_FEED_EVENT, (iso(self.ms(minute)), kind, detail))

    def checkpoint(self, minute) -> None:
        self.store.queue(records.INSERT_CHECKPOINT, (iso(self.ms(minute)), SCHEMA_VERSION, "{}", b""))

    def done(self):
        self.store.flush()
        self.store.close()
        return self.cfg


def living(*ids) -> list[tuple[int, int, int, None]]:
    """(bot id, slot) pairs as learners for Run.minutes, all born at the start of the run and still alive."""
    return [(bot_id, slot, 0, None) for bot_id, slot in ids]


def found(cfg) -> dict[tuple[int, int], dashboard.Skip]:
    """The skipped turns of a run, keyed by bot and minute index."""
    conn = data.connect(cfg.db_path)
    try:
        return {(s.bot_id, s.minute): s for s in dashboard.skipped_turns(conn, cfg).values()}
    finally:
        conn.close()


def keyed(run: Run, *turns) -> set[tuple[int, int]]:
    """(bot, minute index) for (bot, minute of the run) pairs."""
    return {(bot_id, minute_index(run.ms(n))) for bot_id, n in turns}


@contextlib.contextmanager
def serving(*configs):
    """A dashboard for these runs and a getter that returns (status, JSON body)."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(list(configs)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path: str) -> tuple[int, dict]:
        try:
            reply = urllib.request.urlopen(base + path)
        except urllib.error.HTTPError as err:
            body = err.read().decode()
            assert "Traceback" not in body and "/" not in json.loads(body)["error"]
            return err.code, json.loads(body)
        assert reply.headers["Content-Type"] == "application/json"
        return reply.status, json.loads(reply.read())

    try:
        yield get
    finally:
        server.shutdown()
        server.server_close()


# Synthetic runs, one per cause


def run_d_case(folder):
    """Run D's case. Bot 1 decides at :00 of each 4 hour block and opened a short at 16:00. A 2 second clock skew trip
    at 19:44:20 spoiled the minute ending 19:45, its 15 minute return needs a healthy minute exactly then, so at 20:00
    there was no prediction and no decision, and it kept the short until midnight."""
    run = Run(folder, base=DAY_D, time={"horizon_minutes": 240, "history_minutes_required": 360},
              features={"return_windows_minutes": (5, 15, 30, 60), "vol_windows_minutes": (30, 120),
                        "volume_baseline_minutes": 360}, population={"size": 2})

    def hm(h, m=0, s=0):
        return h * 60 + m + s / 60 if s else h * 60 + m

    for bot_id, slot, origin in ((1, 0, "random"), (2, 1, "random"), (3, 0, "control"), (4, 1, "control")):
        run.bot(bot_id, slot, hm(6, 0, 5), origin)
    run.minutes(hm(10), hm(24, 30), unhealthy={hm(19, 45)}, unpredicted={hm(19, 50), hm(20), hm(20, 15), hm(20, 45)})
    run.decision(1, hm(16), "open_short", prediction=-31.0, gate=1,
                 reason="prediction -31.00 bps clears required edge 20.00 bps")
    run.decision(3, hm(16), "open_long", prediction=-31.0, gate=1,
                 reason="twin of bot 1 copying open_short, the coin chose long")
    for minute in (hm(16, 1), hm(20, 1), hm(24, 1)):
        run.decision(2, minute)
    run.decision(1, hm(24), "close", position="short", prediction=4.0)
    run.decision(3, hm(24), "close", position="long", prediction=4.0,
                 reason="twin of bot 1 copying close, the coin chose long")
    run.trade(1, 1, "short", hm(16, 0, 1.25), hm(24, 0, 1.25))
    run.trade(2, 3, "long", hm(16, 0, 1.3), hm(24, 0, 1.3))
    run.event(hm(10), "healthy", "a full clean minute")
    run.event(hm(19, 44, 20), "clock_skew", "median 2000 ms")
    run.event(hm(19, 44, 20), "recovering", "clock skew")
    run.event(hm(19, 44, 52), "clock_skew", "cleared, median 40 ms")
    run.event(hm(19, 46), "healthy", "a full clean minute")
    run.checkpoint(hm(24, 30, 30))
    return run.done()


def stall_case(folder) -> tuple[Run, object]:
    """A 4 minute horizon and two learners with their twins. The feed goes stale at 04:30:30 and is healthy again at
    04:33, so minutes 31 and 32 count for nothing and the next four have a return window that starts in one of them.
    At minute 20 bot 1 opens a long and its twin copies it, everywhere else a learner decides none and its twin writes
    nothing."""
    run = Run(folder, time={"horizon_minutes": 4}, population={"size": 2})
    for bot_id, slot, origin in ((1, 0, "random"), (2, 1, "random"), (3, 0, "control"), (4, 1, "control")):
        run.bot(bot_id, slot, 0, origin)
    run.minutes(1, 60, unhealthy={31, 32}, unpredicted={33, 34, 35, 36}, learners=living((1, 0), (2, 1)),
                undecided={(1, 20)})
    run.decision(1, 20, "open_long", prediction=25.0, gate=1, reason="prediction 25.00 bps clears required edge "
                                                                     "20.00 bps")
    run.decision(3, 20, "open_short", prediction=25.0, gate=1, reason="twin of bot 1 copying open_long, the coin "
                                                                      "chose short")
    run.trade(1, 1, "long", 20 + 1.25 / 60)
    run.trade(2, 3, "short", 20 + 1.3 / 60)
    run.event(1, "healthy", "a full clean minute")
    run.event(30.5, "stale", "no messages")
    run.event(31 + 36 / 60, "recovering", "fresh data")
    run.event(33, "healthy", "a full clean minute")
    run.checkpoint(60.5)
    return run, run.done()


def restart_case(folder) -> tuple[Run, object]:
    """The run stops after minute 40, so no minute from 41 to 50 is recorded, and resumes at 04:50:40 from its
    checkpoint of 04:40:30. Minute 51 is the feed's first, not yet healthy, and from 52 the feature windows still reach
    back into the stop until 62."""
    run = Run(folder, time={"horizon_minutes": 4}, population={"size": 2})
    for bot_id, slot, origin in ((1, 0, "random"), (2, 1, "random"), (3, 0, "control"), (4, 1, "control")):
        run.bot(bot_id, slot, 0, origin)
    run.minutes(1, 80, missing=set(range(41, 51)), unhealthy={51}, unpredicted=set(range(52, 63)),
                learners=living((1, 0), (2, 1)))
    run.event(1, "healthy", "a full clean minute")
    run.event(40 + 10 / 60, "disconnected", "server closed")
    run.event(50 + 40 / 60, "resume", "restored the checkpoint from 2026-10-03T04:40:30.000Z")
    run.event(50 + 40 / 60, "connected")
    run.event(50 + 41 / 60, "recovering", "fresh data")
    run.event(52, "healthy", "a full clean minute")
    run.event(52, "resume", "12.0 minutes since the last healthy minute, restart_flatten sent 2 closing orders")
    run.checkpoint(80.5)
    return run, run.done()


def test_run_d_case_is_one_skipped_turn_with_its_cause_and_what_it_kept(tmp_path):
    cfg = run_d_case(tmp_path / "run_d")
    skips = list(found(cfg).values())
    assert [(s.bot_id, s.ts) for s in skips] == [(1, "2026-10-13T20:00:00.000Z")]
    skip = skips[0]
    assert skip.id == -(minute_index(parse_iso(skip.ts)) * 1_000_000 + 1) == skip_id(skip.minute, 1)
    assert skip.cause == "inputs" and not skip.pruned
    assert skip.reason == ("the feed was healthy but the features were incomplete, the 15 minute return starts at "
                           "2026-10-13T19:45:00.000Z, which was not healthy, as the feed went recovering at "
                           "2026-10-13T19:44:20.000Z on clock_skew median 2000 ms")
    assert [w["window"] for w in skip.windows] == ["15 minute return"]  # every other window was complete
    assert skip.position == {"side": "short", "trade_id": 1, "opened_at": "2026-10-13T16:00:01.250Z",
                             "entry_price": 80_000.0, "closed_at": "2026-10-14T00:00:01.250Z",
                             "exit_reason": "decision"}
    assert skip.twin == {"bot_id": 3, "position": {
        "side": "long", "trade_id": 2, "opened_at": "2026-10-13T16:00:01.300Z", "entry_price": 80_000.0,
        "closed_at": "2026-10-14T00:00:01.300Z", "exit_reason": "decision"}}
    assert (skip.next["ts"], skip.next["action"]) == ("2026-10-14T00:00:00.000Z", "close")
    with serving(cfg) as get:
        code, page = get("/api/decisions?run=run_d&action=skipped")
        assert code == 200 and page["total"] == 1 and page["action_counts"]["skipped"] == 1
        row = page["rows"][0]
        keys = ("id", "ts", "bot_id", "kind", "action", "position_before", "cause", "reason", "prediction_bps",
                "gate_passed", "forced", "label_bps", "label_state", "right")
        assert {k: row[k] for k in keys} == {
            "id": skip.id, "ts": skip.ts, "bot_id": 1, "kind": "learner", "action": SKIPPED, "position_before": "short",
            "cause": "inputs", "reason": skip.reason, "prediction_bps": None, "gate_passed": None, "forced": None,
            "label_bps": None, "label_state": None, "right": None}
        code, d = get(f"/api/decision?run=run_d&id={skip.id}")
    assert code == 200 and d["run"] == "run_d" and d["skipped"]["reason"] == skip.reason
    assert d["bot"]["bot_id"] == 1 and d["bot"]["kind"] == "learner" and d["position"]["side"] == "short"
    assert d["twin"]["bot_id"] == 3 and d["next"]["action"] == "close"
    named = [(e["ts"], e["kind"]) for e in d["events"] if e["named"]]
    assert named == [("2026-10-13T19:44:20.000Z", "clock_skew"), ("2026-10-13T19:44:20.000Z", "recovering")]
    assert ("2026-10-13T19:46:00.000Z", "healthy") in [(e["ts"], e["kind"]) for e in d["events"]]  # nearby, not named


def test_a_minute_that_was_not_healthy_names_what_took_the_feed_out(tmp_path):
    run, cfg = stall_case(tmp_path / "stall")
    skips = found(cfg)
    assert set(skips) == keyed(run, (1, 32), (1, 36), (2, 33))
    stale = skips[(1, minute_index(run.ms(32)))]
    assert stale.cause == "not_healthy" and stale.reason == (
        "the feed was not healthy, it went stale at 2026-10-03T04:30:30.000Z, no messages, and was healthy again at "
        "2026-10-03T04:33:00.000Z, 2.5 minutes later")
    assert [e["kind"] for e in stale.named] == ["stale", "healthy"]
    late = skips[(1, minute_index(run.ms(36)))]
    assert late.cause == "inputs" and late.reason == (
        "the feed was healthy but the features were incomplete, the 4 minute return starts at "
        "2026-10-03T04:32:00.000Z, which was not healthy, as the feed went stale at 2026-10-03T04:30:30.000Z, "
        "no messages")
    first = skips[(2, minute_index(run.ms(33)))].reason
    assert first.startswith("the feed was healthy but the features were incomplete, the 1 minute return starts at "
                            "2026-10-03T04:32:00.000Z, which was not healthy, as the feed went stale at ")
    assert ", and the 2 minute return starts at 2026-10-03T04:31:00.000Z, which was not healthy, and " in first
    assert first.count("as the feed went stale") == 1  # one outage is named once
    assert "the 2 minute volatility had 0 of the 2 pairs of consecutive healthy minutes it needs" in first


def test_twins_never_get_a_skipped_row_and_a_decision_of_none_is_no_skip(tmp_path):
    """A twin writes a row only when its learner opens, closes or flips, so its silence is normal. A learner that
    decided none made its turn, and its twin's missing row is no skip either."""
    run, cfg = stall_case(tmp_path / "stall")
    skips = found(cfg)
    assert {bot for bot, _ in skips} == {1, 2}
    conn = data.connect(cfg.db_path)
    twin_rows = {r[0] for r in conn.execute("SELECT ts FROM decisions WHERE bot_id = 3")}
    nones = {r[0] for r in conn.execute("SELECT ts FROM decisions WHERE bot_id = 1 AND action = 'none'")}
    conn.close()
    assert twin_rows == {iso(run.ms(20))} and iso(run.ms(28)) in nones  # minute 28 bot 1 stayed out, twin 3 silent
    assert (1, minute_index(run.ms(28))) not in skips
    stale = skips[(1, minute_index(run.ms(32)))]
    assert stale.position["side"] == "long" and stale.twin == {"bot_id": 3, "position": {
        "side": "short", "trade_id": 2, "opened_at": iso(run.ms(20 + 1.3 / 60)), "entry_price": 80_000.0,
        "closed_at": None, "exit_reason": None}}
    with serving(cfg) as get:
        assert get("/api/decisions?run=stall&kind=twin&action=skipped")[1]["total"] == 0
        assert get("/api/decisions?run=stall&kind=twin")[1]["action_counts"][SKIPPED] == 0
        assert get("/api/decisions?run=stall&kind=learner")[1]["action_counts"][SKIPPED] == 3


def test_a_stretch_the_run_was_not_running_names_its_restart(tmp_path):
    run, cfg = restart_case(tmp_path / "restart")
    skips = found(cfg)
    down = [(1, 44), (1, 48), (2, 41), (2, 45), (2, 49)]
    after = [(1, 52), (1, 56), (1, 60), (2, 53), (2, 57), (2, 61)]
    assert set(skips) == keyed(run, *down, *after)
    assert {skips[k].cause for k in keyed(run, *down)} == {"not_running"}
    assert {skips[k].cause for k in keyed(run, *after)} == {"inputs"}
    assert skips[(1, minute_index(run.ms(44)))].reason == (
        "the run was not running, no minute was recorded between 2026-10-03T04:40:00.000Z and "
        "2026-10-03T04:51:00.000Z, after the feed logged disconnected server closed at 2026-10-03T04:40:10.000Z, it "
        "resumed at 2026-10-03T04:50:40.000Z from the checkpoint of 2026-10-03T04:40:30.000Z, and at its first healthy "
        "minute it logged resume 12.0 minutes since the last healthy minute, restart_flatten sent 2 closing orders")
    back = skips[(1, minute_index(run.ms(52)))].reason
    assert back.startswith("the feed was healthy but the features were incomplete, the 1 minute return starts at "
                           "2026-10-03T04:51:00.000Z, which was not healthy, as the run had resumed at "
                           "2026-10-03T04:50:40.000Z, and the 2 minute return starts at 2026-10-03T04:50:00.000Z, "
                           "which was not recorded")
    assert "the 15 minute recent volume had 4 of the 12 healthy minutes it needs" in back
    last = skips[(2, minute_index(run.ms(61)))]
    assert last.reason == ("the feed was healthy but the features were incomplete, the 15 minute recent volume had 10 "
                           "of the 12 healthy minutes it needs, and the 15 minute volume baseline had 10 of the 12 "
                           "healthy minutes it needs")


def test_dead_bots_and_bots_born_mid_run(tmp_path):
    """Bot 1 dies at its own phase minute 24 with a position, so its close fills later, and bot 5 is born in its slot
    after that minute's step. Bot 2 dies at minute 30 and waits as dying for a close that never fills, while bot 6
    takes its slot. None of them had a turn there, and only bot 6's missing decision at 37 is a skipped turn."""
    run = Run(tmp_path / "deaths", time={"horizon_minutes": 4}, population={"size": 2})
    run.bot(1, 0, 0, died=24 + 1.5 / 60)
    run.bot(2, 1, 0, state="dying")
    run.bot(3, 0, 0, "control")
    run.bot(4, 1, 0, "control")
    run.bot(5, 0, 24)
    run.bot(6, 1, 30)
    run.minutes(1, 60, learners=[(1, 0, 0, 24), (2, 1, 0, 30), (5, 0, 24, None), (6, 1, 30, None)],
                undecided={(6, 37)})
    run.checkpoint(60.5)
    cfg = run.done()
    skips = found(cfg)
    assert set(skips) == keyed(run, (6, 37))
    assert skips[(6, minute_index(run.ms(37)))].reason == (
        "no decision recorded, cause not in the database, the minute was healthy and a prediction was made")
    assert dashkit.skipped_by_rule(cfg) == {(6, iso(run.ms(37)))}
    conn = data.connect(cfg.db_path)
    rows = {(r[0], r[1]) for r in conn.execute("SELECT bot_id, ts FROM decisions")}
    conn.close()
    assert (5, iso(run.ms(28))) in rows and (1, iso(run.ms(24))) not in rows  # the child's first turn is its next one


def test_minutes_after_the_newest_checkpoint_are_not_judged(tmp_path):
    """Rows are committed only with checkpoints, so a missing row after the newest one may still be in the run's
    buffer."""
    run = Run(tmp_path / "late", time={"horizon_minutes": 4}, population={"size": 2})
    run.bot(1, 0, 0)
    run.bot(2, 1, 0)
    run.minutes(1, 60, learners=living((1, 0), (2, 1)), undecided={(1, 36), (1, 44)})
    run.checkpoint(40.5)
    cfg = run.done()
    assert set(found(cfg)) == keyed(run, (1, 36))


def test_pruned_minutes_fall_back_to_benchmark_rows_and_feed_events(tmp_path):
    """Candles are pruned after keep_candles_days. Benchmark rows, written in every healthy minute and never pruned,
    still name every feature window that failed, the feed events still tell the causes apart, and each reason for a
    pruned minute says its records were pruned."""
    run = Run(tmp_path / "pruned", time={"horizon_minutes": 4}, population={"size": 2})
    run.bot(1, 0, 0)
    run.bot(2, 1, 0)
    run.minutes(1, 60, missing={5, 6, 7, 8}, unhealthy={12, 29},
                unpredicted={9, 10, 11, 13, 14, 15, 16, 30, 31, 32, 33}, learners=living((1, 0), (2, 1)),
                undecided={(1, 20), (1, 44)})
    run.event(1, "healthy", "a full clean minute")
    run.event(8.5, "resume", "restored the checkpoint from 2026-10-03T04:04:30.000Z")
    run.event(9, "healthy", "a full clean minute")
    run.event(11.5, "stale", "no messages")
    run.event(13, "healthy", "a full clean minute")
    run.event(28.5, "stale", "no messages")
    run.event(30, "healthy", "a full clean minute")
    run.store.queue(records.PRUNE_CANDLES, (iso(run.ms(30)),))
    run.checkpoint(60.5)
    cfg = run.done()
    skips = found(cfg)
    assert set(skips) == keyed(run, (1, 8), (2, 5), (1, 12), (2, 9), (2, 13), (1, 16), (1, 20), (2, 29), (1, 32),
                               (2, 33), (1, 44))
    tail = ", the minute records of that time were pruned"
    say = {n: skips[(b, minute_index(run.ms(n)))]
           for b, n in ((1, 8), (2, 9), (1, 12), (1, 16), (1, 20), (2, 29), (1, 32), (1, 44))}
    assert (say[8].cause, say[8].reason) == ("not_running", "the run was not running, it resumed at "
                                             "2026-10-03T04:08:30.000Z from the checkpoint of "
                                             "2026-10-03T04:04:30.000Z" + tail)
    assert (say[12].cause, say[12].reason) == ("not_healthy", "the feed was not healthy, it went stale at "
                                               "2026-10-03T04:11:30.000Z, no messages, and was healthy again at "
                                               "2026-10-03T04:13:00.000Z, 1.5 minutes later" + tail)
    assert (say[16].cause, say[16].reason) == ("inputs", "the feed was healthy but the features were incomplete, the 4 "
                                               "minute return starts at 2026-10-03T04:12:00.000Z, which was not "
                                               "healthy, as the feed went stale at 2026-10-03T04:11:30.000Z, no "
                                               "messages, and the 15 minute recent volume had 10 of the 12 healthy "
                                               "minutes it needs, and the 15 minute volume baseline had 10 of the 12 "
                                               "healthy minutes it needs" + tail)
    assert [w["window"] for w in say[16].windows] == ["4 minute return", "15 minute recent volume",
                                                      "15 minute volume baseline"]
    assert [e["kind"] for e in say[16].named] == ["stale"]
    assert say[9].cause == "inputs" and say[9].reason.endswith(tail)  # a minute of the stop is not recorded
    assert say[9].reason.startswith("the feed was healthy but the features were incomplete, the 1 minute return "
                                    "starts at 2026-10-03T04:08:00.000Z, which was not recorded, and the 2 minute "
                                    "return starts at ")
    assert (say[20].cause, say[20].reason) == ("unknown", "no decision recorded, cause not in the database, the minute "
                                               "was healthy and a prediction was made" + tail)
    assert (say[29].cause, say[29].reason) == ("not_healthy", "the feed was not healthy, it went stale at "
                                               "2026-10-03T04:28:30.000Z, no messages, and was healthy again at "
                                               "2026-10-03T04:30:00.000Z, 1.5 minutes later" + tail)
    assert all(say[n].pruned for n in (8, 9, 12, 16, 20, 29)) and not say[32].pruned and not say[44].pruned
    # Minute 32 is kept, and its windows reach back into the pruned minutes before 30.
    assert (say[32].cause, say[32].reason) == ("inputs", "the feed was healthy but the features were incomplete, the 3 "
                                               "minute return starts at 2026-10-03T04:29:00.000Z, which was not "
                                               "healthy, as the feed went stale at 2026-10-03T04:28:30.000Z, no "
                                               "messages, and the 3 minute volatility had 2 of the 3 pairs of "
                                               "consecutive healthy minutes it needs")
    assert say[44].cause == "unknown"
    assert not any("cannot be told" in s.reason for s in skips.values())


def test_the_panel_shows_the_feed_events_nearest_the_skipped_minute(tmp_path):
    """A storm of reconnects early in the window must not push out the events next to the minute, and only the
    nearest events after it are kept."""
    run = Run(tmp_path / "storm", time={"horizon_minutes": 4}, population={"size": 2})
    run.bot(1, 0, 0)
    run.bot(2, 1, 0)
    run.minutes(1, 60, unhealthy={32}, learners=living((1, 0), (2, 1)))
    run.event(1, "healthy", "a full clean minute")
    for i in range(50):
        run.event(18 + i / 5, "snapshot", f"storm {i}")
    run.event(31.5, "stale", "no messages")
    run.event(31.8, "connected")
    for i in range(15):
        run.event(32.1 + i / 10, "snapshot", f"after {i}")
    run.event(33, "healthy", "a full clean minute")
    run.checkpoint(60.5)
    cfg = run.done()
    sid = skip_id(minute_index(run.ms(32)), 1)
    with serving(cfg) as get:
        code, d = get(f"/api/decision?run=storm&id={sid}")
    assert code == 200 and d["skipped"]["cause"] == "not_healthy"
    seen = [(e["kind"], e["detail"]) for e in d["events"]]
    assert ("connected", None) in seen and ("stale", "no messages") in seen
    before = [e for e in d["events"] if e["ts"] <= iso(run.ms(32))]
    after = [e for e in d["events"] if e["ts"] > iso(run.ms(32))]
    assert len(before) == dashboard.NEAR_BEFORE_EVENTS and ("snapshot", "storm 49") in seen
    assert ("snapshot", "storm 0") not in seen  # the oldest of the storm gives way
    assert [e["detail"] for e in after if e["kind"] == "snapshot"] == [f"after {i}" for i in range(10)]
    assert [e["kind"] for e in after if e["named"]] == ["healthy"]  # named, so shown although past the nearest ten


def test_a_failure_working_out_skipped_turns_still_lists_the_decisions(tmp_path, monkeypatch, caplog):
    """The skipped turns are new code reading live databases. If they fail, the decisions still list, the reply says
    the skipped turns could not be worked out, and the log gets the traceback. Running out of time still answers
    503, with a reason that names the skipped turns, since narrowing the filters would not help."""
    _, cfg = stall_case(tmp_path / "stall")
    with serving(cfg) as get:
        code, fine = get("/api/decisions?run=stall")
    assert code == 200 and fine["skipped_error"] is False and fine["summary"]["skipped"] == 3

    def broken(conn, cfg):
        raise ValueError("a surprise in a live database")

    monkeypatch.setattr(dashboard, "_skip_cache", {})
    monkeypatch.setattr(dashboard, "_find_skips", broken)
    with serving(cfg) as get:
        code, page = get("/api/decisions?run=stall")
        assert code == 200 and page["skipped_error"] is True
        assert page["total"] == fine["total"] - 3 and page["summary"]["skipped"] == 0
        assert page["action_counts"][SKIPPED] == 0 and all(r["action"] != SKIPPED for r in page["rows"])
        assert get(f"/api/decision?run=stall&id={page['rows'][0]['id']}")[0] == 200
    assert "could not work out the skipped turns" in caplog.text and "a surprise in a live database" in caplog.text

    def slow(conn, cfg):
        raise sqlite3.OperationalError("interrupted")

    monkeypatch.setattr(dashboard, "_find_skips", slow)
    with serving(cfg) as get:
        code, reply = get("/api/decisions?run=stall")
    assert code == 503 and reply == {"error": "working out the skipped turns took too long, see the dashboard log"}


def test_a_run_with_no_decisions_has_no_skipped_turns(tmp_path):
    """Like run D2 before its first decision, or any run before its first checkpoint."""
    run = Run(tmp_path / "young", time={"horizon_minutes": 4}, population={"size": 2})
    run.bot(1, 0, 0)
    run.bot(2, 1, 0)
    run.minutes(1, 30, unhealthy={10, 11}, unpredicted=set(range(1, 31)))
    run.checkpoint(30.5)
    cfg = run.done()
    assert found(cfg) == {}
    with serving(cfg) as get:
        code, page = get("/api/decisions?run=young")
    assert code == 200 and page["total"] == 0 and page["summary"]["skipped"] == 0
    assert page["action_counts"][SKIPPED] == 0


def test_the_answer_is_kept_until_the_run_writes_again(tmp_path):
    run = Run(tmp_path / "cached", time={"horizon_minutes": 4}, population={"size": 2})
    run.bot(1, 0, 0)
    run.bot(2, 1, 0)
    run.minutes(1, 40, learners=living((1, 0), (2, 1)), undecided={(1, 36)})
    run.checkpoint(40.5)
    cfg = run.done()
    conn = data.connect(cfg.db_path)
    first = dashboard.skipped_turns(conn, cfg)
    assert dashboard.skipped_turns(conn, cfg) is first and len(first) == 1
    conn.close()
    store = Store.open(cfg.db_path)  # the run writes on, a missing decision at 44 and a new checkpoint
    run.store = store
    run.minutes(41, 50, learners=living((1, 0), (2, 1)), undecided={(1, 44)})
    run.checkpoint(50.5)
    run.done()
    assert set(found(cfg)) == keyed(run, (1, 36), (1, 44))


# A replayed run with deaths, a clock skew trip, a stall and a restart, through the HTTP views


@pytest.fixture(scope="module")
def faulty(tmp_path_factory):
    """About seventy minutes at ruinous fees, so learners die and are replaced, with a clock skew trip, a stall and a
    stop of nine minutes before the run resumes from its checkpoint."""
    cfg = deadly_config(tmp_path_factory.mktemp("faulty") / "faulty")
    engine = new_engine(cfg)
    feed(engine, synthetic(5, 2700, (Fault(1800, "skew", 30, 3000), Fault(2280, "silence", 30))))
    engine.finish()
    engine.store.close()
    again = engine.now_ms + 9 * MS_PER_MINUTE
    engine = Engine.start(cfg, Store.open(cfg.db_path), load_adapter(cfg.feed), again)
    feed(engine, synthetic(6, 1500, (), again))
    engine.finish()
    engine.store.close()
    return cfg


@pytest.fixture(scope="module")
def explorer(faulty):
    with serving(faulty) as get:
        yield get


SKIP_FILTERS = [
    {"action": SKIPPED}, {"action": "skipped,close"}, {"action": "none,skipped,hold"},
    {"kind": "learner", "action": SKIPPED}, {"kind": "twin", "action": SKIPPED}, {"bot": "1"},
    {"bot": "2", "action": SKIPPED}, {"q": "not healthy"},
    {"q": "RETURN STARTS"}, {"q": "not running"}, {"q": "clock_skew"}, {"q": "resumed at", "action": "skipped,none"},
    {"forced": "0", "action": SKIPPED}, {"gate": "0", "action": SKIPPED}, {"outcome": "missed", "action": SKIPPED},
    {"outcome": "waiting", "kind": "learner"}, {"from": "2026-10-03 04:30", "to": "2026-10-03 04:50"},
    {"from": "2026-10-03 04:55", "action": SKIPPED}, {"hours": "1"}, {"hours": "1", "action": SKIPPED},
]


@pytest.mark.parametrize("params", dashkit.FILTERS + SKIP_FILTERS,
                         ids=lambda p: "&".join(f"{k}={v}" for k, v in p.items()) or "none")
def test_skipped_turns_page_filter_and_count_with_the_decisions(explorer, faulty, params):
    rows, reply = dashkit.fetch_all(explorer, "faulty", params)
    ids, counts, summary = dashkit.expected(faulty, params)  # also checks the dashboard finds what the rule finds
    assert [r["id"] for r in rows] == ids  # newest first, a decision before a skipped turn of the same minute
    assert reply["total"] == len(ids) and reply["summary"] == summary and reply["action_counts"] == counts


def test_the_faulty_run_has_every_cause_and_the_filters_split_it(faulty):
    skips = found(faulty)
    assert {s.cause for s in skips.values()} == {"not_running", "not_healthy", "inputs"} < set(dashboard.CAUSES)
    conn = data.connect(faulty.db_path)
    assert conn.execute("SELECT count(*) FROM deaths").fetchone()[0] > 0  # replacements were born mid run
    conn.close()
    total = len(dashkit.expected(faulty, {})[0])
    for params in SKIP_FILTERS:
        n = len(dashkit.expected(faulty, params)[0])
        if params in ({"kind": "twin", "action": SKIPPED}, {"forced": "0", "action": SKIPPED},
                      {"gate": "0", "action": SKIPPED}, {"outcome": "missed", "action": SKIPPED}):
            assert n == 0, params  # a skipped turn has no twin, forced flag, gate or label
        elif "hours" in params:
            assert 0 < n <= total, params  # the run's decisions span less than an hour
        else:
            assert 0 < n < total, params


def test_paging_merges_decisions_and_skipped_turns(explorer, faulty):
    ids = dashkit.expected(faulty, {})[0]
    kinds = {r["id"]: r["action"] for r in dashkit.fetch_all(explorer, "faulty", {})[0]}
    pages = -(-len(ids) // 7)
    mixed = 0
    for page in range(1, pages + 1):
        _, reply = explorer(f"/api/decisions?run=faulty&per=7&page={page}")
        assert [r["id"] for r in reply["rows"]] == ids[7 * (page - 1):7 * page] and reply["pages"] == pages
        mixed += len({kinds[i] == SKIPPED for i in ids[7 * (page - 1):7 * page]}) == 2
    assert mixed >= 2  # some pages hold both, so the order across the two kinds is really tested
    _, beyond = explorer("/api/decisions?run=faulty&per=7&page=99999")
    assert beyond["page"] == pages


def test_the_summary_counts_skipped_turns_apart(explorer, faulty):
    skips = found(faulty)
    _, everything = explorer("/api/decisions?run=faulty")
    s = everything["summary"]
    assert s["skipped"] == len(skips) == everything["action_counts"][SKIPPED] > 0
    assert s["total"] == s["skipped"] + s["matured"] + s["waiting"] + s["missed"]
    _, only = explorer("/api/decisions?run=faulty&action=skipped")
    assert only["summary"] == {"total": len(skips), "skipped": len(skips), "forced": 0, "gate_passed": 0,
                               "matured": 0, "waiting": 0, "missed": 0, "right": 0, "wrong": 0}
    for outcome in ("matured", "right", "wrong", "waiting", "missed"):
        _, reply = explorer(f"/api/decisions?run=faulty&outcome={outcome}")
        assert reply["summary"]["skipped"] == reply["action_counts"][SKIPPED] == 0, outcome
    _, rows = explorer("/api/decisions?run=faulty&action=skipped&per=200")
    assert all(r["label_state"] is None and r["right"] is None and r["kind"] == "learner" for r in rows["rows"])


def test_every_skipped_turn_opens_in_full(explorer, faulty):
    rows = dashkit.fetch_all(explorer, "faulty", {"action": SKIPPED})[0]
    named = Counter()
    assert rows
    for r in rows:
        code, d = explorer(f"/api/decision?run=faulty&id={r['id']}")
        named[r["cause"]] += any(e["named"] for e in d["events"])
        assert code == 200 and d["skipped"]["id"] == r["id"] and d["skipped"]["reason"] == r["reason"]
        assert d["skipped"]["cause"] == r["cause"] and d["bot"]["bot_id"] == r["bot_id"]
        assert d["position"]["side"] == r["position_before"]
        assert d["twin"] is None or d["twin"]["bot_id"] != r["bot_id"]
        stamps = [(e["ts"], e["id"]) for e in d["events"]]
        assert stamps == sorted(stamps) and len(stamps) == len(set(stamps))
        if d["next"] is not None:
            assert d["next"]["ts"] > r["ts"]
            code, nxt = explorer(f"/api/decision?run=faulty&id={d['next']['id']}")
            assert code == 200 and nxt["decision"]["bot_id"] == r["bot_id"]
    causes = Counter(r["cause"] for r in rows)
    assert named["not_running"] == causes["not_running"] and named["not_healthy"] == causes["not_healthy"]
    assert named["inputs"] > 0  # a return window that starts in an unhealthy minute names what spoiled it


@pytest.mark.parametrize("value", ["-", "-abc", "-0", "--1", "-1", "-999999", "-1000000", "-1.5", "-1e6",
                                   "-4222846080000000", "-9007199254740991", "-9007199254740992",
                                   "-99999999999999999", "-12345678901234567", "- 1000001"])
def test_a_malformed_skipped_id_is_refused_with_400(explorer, value):
    code, reply = explorer("/api/decision?run=faulty&id=" + urllib.parse.quote(value))
    assert code == 400 and set(reply) == {"error"} and reply["error"].count(".") <= 1 and len(reply["error"]) < 200


def test_a_skipped_id_that_names_no_skipped_turn_is_404(explorer, faulty):
    conn = data.connect(faulty.db_path)
    bot_id, ts = conn.execute("SELECT bot_id, ts FROM decisions ORDER BY id LIMIT 1").fetchone()
    conn.close()
    for sid in (skip_id(minute_index(parse_iso(ts)), bot_id), -dashboard.SKIP_ID_MAX, -1_000_001):
        code, reply = explorer(f"/api/decision?run=faulty&id={sid}")
        assert code == 404 and "has no skipped turn at" in reply["error"], sid


def test_skipped_ids_stay_inside_the_integers_javascript_holds():
    assert skip_id(29_850_000, 7) == -29_850_000_000_007
    assert dashboard.split_skip_id(-29_850_000_000_007) == (29_850_000, 7)
    year_9998 = minute_index(parse_iso("9998-12-31T23:59:00Z"))
    assert year_9998 == dashboard.SKIP_LAST_MINUTE and -skip_id(year_9998, 999_999) == dashboard.SKIP_ID_MAX
    assert dashboard.SKIP_ID_MAX <= 2 ** 53 - 1  # Number.isSafeInteger holds for every id the server makes
    assert skip_id(year_9998 + 1, 1) is None and skip_id(year_9998, 1_000_000) is None
    assert skip_id(0, 1) is None and skip_id(5, 0) is None


def test_skipped_turn_sql_stays_parameterised(faulty, monkeypatch):
    """The temporary table, the cause lookups and the detail run fixed SQL, with every value a parameter."""
    seen: list[tuple[str, tuple]] = []
    real = dashboard.data.connect

    class Recorder:
        def __init__(self, conn):
            self.conn = conn

        def execute(self, sql, params=()):
            seen.append((sql, tuple(params)))
            return self.conn.execute(sql, params)

        def executemany(self, sql, rows):
            seen.append((sql, ()))
            return self.conn.executemany(sql, rows)

        def __getattr__(self, name):
            return getattr(self.conn, name)

    sid = next(iter(found(faulty).values())).id
    monkeypatch.setattr(dashboard.data, "connect", lambda path: Recorder(real(path)))
    monkeypatch.setattr(dashboard, "_skip_cache", {})  # work the causes out again through the recorder
    with serving(faulty) as get:
        assert get("/api/decisions?run=faulty&action=skipped&q=" + urllib.parse.quote("x' OR 1=1 --"))[1]["total"] == 0
        assert get(f"/api/decision?run=faulty&id={sid}")[0] == 200
    texts = [sql for sql, _ in seen]
    assert any("temp.skipped_turns" in t for t in texts) and any("FROM feed_events" in t for t in texts)
    for sql in texts:
        text = re.sub(r" LIMIT 1$", "", sql.replace("ESCAPE '\\'", ""))
        assert "'" not in text and "--" not in text and not re.search(r"[1-9]", text), sql
    values = [v for _, params in seen for v in params]
    assert SKIPPED in values and "%x' OR 1=1 --%" in values


# The real engine as the judge


class Watch:
    """What the engine itself saw in every minute it closed, recorded around its own minute step.

    A learner has a turn at a boundary when it holds its slot at its phase minute and is still alive in that slot
    after the step, since the death check runs before any decision and a newborn first steps one minute later.
    Minutes the run never closed are turns of the learners it restored. For a healthy minute with no features, the
    windows that failed are worked out on the engine's own minute series, by the rules of FeatureBuilder.compute.
    """

    def __init__(self, monkeypatch) -> None:
        self.turns, self.closed, self.windows, self.died_on_turn, self.down = set(), {}, {}, set(), []
        real_minute, real_shared = Engine._minute, Engine._shared_work
        watch = self

        def minute(engine, boundary):
            due = [b for b in engine.population.learners if b.state != DEAD and b.is_phase_minute(boundary)]
            real_minute(engine, boundary)
            for b in due:
                kept = b.state != DEAD and engine.population.learners[b.slot] is b
                (watch.turns if kept else watch.died_on_turn).add((b.bot_id, boundary))
            watch.closed[boundary] = engine.last_healthy_ms == boundary

        def shared_work(engine, boundary, mid):
            z, matured = real_shared(engine, boundary, mid)
            if z is None:
                watch.windows[boundary] = failing_windows(engine, boundary)
            return z, matured

        monkeypatch.setattr(Engine, "_minute", minute)
        monkeypatch.setattr(Engine, "_shared_work", shared_work)

    def stopped(self, learners, first: int, last: int) -> list[int]:
        """The minutes from first to last the run never closed, each a turn for the phase learners it restored."""
        self.down = [m for m in range(first, last, MS_PER_MINUTE) if m not in self.closed]
        self.turns.update((b.bot_id, m) for m in self.down for b in learners if b.is_phase_minute(m))
        return self.down


def failing_windows(engine, boundary) -> list[str]:
    """Every check of FeatureBuilder.compute that fails on the engine's minute series, named as the dashboard names
    its windows, or the history it still lacks when none does."""
    fc, end, cover = engine.cfg.features, minute_index(boundary), engine.cfg.features.min_window_coverage
    longest = max(max(fc.return_windows_minutes), max(fc.vol_windows_minutes)) + 1
    _, _, valid = engine.series.window(end, max(longest, VOLUME_WINDOW))
    out = [f"{k} minute return" for k in fc.return_windows_minutes if not valid[-1 - k]]
    for w in fc.vol_windows_minutes:
        ok = valid[-(w + 1):]
        pairs = int((ok[1:] & ok[:-1]).sum())
        if pairs < cover * w or pairs < 2:
            out.append(f"{w} minute volatility")
    _, _, base = engine.series.window(end, fc.volume_baseline_minutes)
    if base[-VOLUME_WINDOW:].sum() < cover * VOLUME_WINDOW:
        out.append(f"{VOLUME_WINDOW} minute recent volume")
    if base.sum() < cover * fc.volume_baseline_minutes:
        out.append(f"{fc.volume_baseline_minutes} minute volume baseline")
    if not out and engine.series.valid_total_at(end) < engine.cfg.time.history_minutes_required:
        out.append("history")
    return out


def run_with_restart(cfg, seeds, seconds, faults, gap_minutes, monkeypatch) -> tuple[Watch, Engine, Engine]:
    """A run fed by the first seed with the given faults, stopped for gap_minutes, then resumed from its checkpoint
    and fed by the second seed."""
    watch = Watch(monkeypatch)
    first = new_engine(cfg)
    feed(first, synthetic(seeds[0], seconds[0], faults))
    first.finish()
    first.store.close()
    again = first.now_ms + gap_minutes * MS_PER_MINUTE
    second = Engine.start(cfg, Store.open(cfg.db_path), load_adapter(cfg.feed), again)
    restored = list(second.population.learners)
    feed(second, synthetic(seeds[1], seconds[1], (), again))
    second.finish()
    second.store.close()
    assert len(watch.stopped(restored, min(watch.closed), max(watch.closed))) >= gap_minutes
    return watch, first, second


def listed_against(cfg, watch: Watch) -> dict:
    """The dashboard's skipped turns, checked to be exactly the turns the engine had with no decision row, each with
    the cause the engine saw and, for incomplete features, the very windows that failed on its minute series."""
    conn = data.connect(cfg.db_path)
    newest = conn.execute("SELECT ts FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()[0]
    begin = max(conn.execute("SELECT min(ts) FROM decisions").fetchone()[0],
                conn.execute("SELECT min(ts) FROM forecasts").fetchone()[0])
    rows = {(r[0], parse_iso(r[1])) for r in conn.execute("SELECT bot_id, ts FROM decisions")}
    listed = {(s.bot_id, s.minute * MS_PER_MINUTE): s for s in dashboard.skipped_turns(conn, cfg).values()}
    conn.close()
    judged = {t for t in watch.turns if parse_iso(begin) <= t[1] < parse_iso(newest)}
    assert set(listed) == {t for t in judged if t not in rows}
    assert not set(listed) & rows and len(judged & rows) > 30
    for (_, m), skip in listed.items():
        want = "not_running" if m not in watch.closed else "inputs" if watch.closed[m] else "not_healthy"
        assert skip.cause == want, (iso(m), skip.reason)
        if want == "inputs":
            assert [w["window"] for w in skip.windows] == watch.windows[m], (iso(m), skip.reason)
    return listed


def test_the_engine_skips_exactly_the_turns_the_explorer_lists(tmp_path, monkeypatch):
    """Drive the engine through a clock skew trip, a stall and a restart, watching every minute it closes. Every phase
    minute a living learner had without a decision row is listed as skipped, no phase minute with a row is, and each
    cause is what the engine saw in that minute."""
    cfg = fast_config(tmp_path / "run")
    watch, first, second = run_with_restart(cfg, (8, 9), (3000, 900),
                                            (Fault(1800, "skew", 30, 3000), Fault(2400, "silence", 30)), 8,
                                            monkeypatch)
    assert first.deaths == second.deaths == 0  # the same learners held every slot throughout
    listed = listed_against(cfg, watch)
    causes = Counter(s.cause for s in listed.values())
    assert set(causes) == {"not_running", "not_healthy", "inputs"}, causes
    reasons = " ".join(s.reason for s in listed.values())
    assert "on clock_skew median" in reasons and "went stale at" in reasons and "it resumed at" in reasons
    assert "had not been healthy since the run resumed at" in reasons and "cause not in the database" not in reasons


def test_the_engine_agrees_through_deaths_births_and_pruned_candles(tmp_path, monkeypatch):
    """At ruinous fees learners die and newborns take their slots. With a 1 minute horizon every minute is a phase
    minute, so every death lands on one. A learner that dies at its own phase minute has no turn there, since the
    death check comes first, and its newborn first steps a minute later, so neither is listed, even while died_at
    waits for the dying learner's close. Pruning the candles afterwards, as the engine does after keep_candles_days,
    changes no turn and no cause, the windows are still named from the benchmark rows, and only the reasons of pruned
    minutes change, by saying so."""
    base = deadly_config(tmp_path / "deadly")
    cfg = R(base, time=R(base.time, horizon_minutes=1))
    watch, _, second = run_with_restart(cfg, (5, 6), (2700, 1500),
                                        (Fault(1800, "skew", 30, 3000), Fault(2280, "silence", 30)), 9, monkeypatch)
    assert second.deaths >= 4 and len(watch.died_on_turn) >= 2  # the count is carried over by the checkpoint
    conn = data.connect(cfg.db_path)
    late = {r[0] for r in conn.execute(
        "SELECT bot_id FROM bots b WHERE origin != 'control' AND died_at > (SELECT min(born_at) FROM bots o "
        "WHERE o.slot = b.slot AND o.origin != 'control' AND o.born_at > b.born_at)")}
    conn.close()
    assert late, "a learner waited as dying while its newborn already held the slot"
    before = listed_against(cfg, watch)
    assert set(Counter(s.cause for s in before.values())) == {"not_running", "not_healthy", "inputs"}
    assert not {(b, m) for b, m in before} & watch.died_on_turn

    cut = max(watch.down) + 6 * MS_PER_MINUTE  # the stop, the skew trip and the stall all lose their candles
    store = Store.open(cfg.db_path)
    store.queue(records.PRUNE_CANDLES, (iso(cut),))
    store.flush()
    store.close()
    after = listed_against(cfg, watch)
    tail = ", the minute records of that time were pruned"
    pruned = [k for k in after if k[1] < cut]
    assert {after[k].cause for k in pruned} == {"not_running", "not_healthy", "inputs"}
    assert any(k[1] >= cut and after[k].cause == "inputs" for k in after)
    for key, skip in after.items():
        assert skip.pruned == (key[1] < cut) and skip.windows == before[key].windows, key
        if not skip.pruned:
            assert skip.reason == before[key].reason, key
        elif skip.cause == "not_running":
            assert skip.reason.startswith("the run was not running, it resumed at ") and skip.reason.endswith(tail)
        else:
            assert skip.reason == before[key].reason + tail, key


# Speed and the page


def test_a_long_run_finds_its_skipped_turns_fast(tmp_path):
    """About four months of a run with 20 learners and a 30 minute horizon, candles kept for the last 14 days, and a
    clock skew trip every 97 minutes that spoils that minute and the four return windows that start in it."""
    from btc_survivor.dashboard import QUERY_BUDGET_S, decision_detail, decision_page

    horizon, days, size = 30, 120, 20
    cfg = fast_config(tmp_path / "long", time={"horizon_minutes": horizon, "history_minutes_required": 360},
                      features={"return_windows_minutes": (5, 15, 30, 60), "vol_windows_minutes": (30, 120),
                                "volume_baseline_minutes": 360}, population={"size": size})
    store = Store.open(cfg.db_path)
    conn = store.conn
    total = days * 1440
    bad = set(range(400, total, 97))
    spoiled = {n + k for n in bad for k in (0, 5, 15, 30, 60)}
    kept = total - 14 * 1440

    def ts(n):
        return iso(START + n * MS_PER_MINUTE)

    conn.execute("BEGIN")
    conn.executemany(records.INSERT_BOT, [(b, (b - 1) % size, None, 0, ts(0), "active", 8.0, 1e-4, 1,
                                           "random" if b <= size else "control") for b in range(1, 2 * size + 1)])
    conn.executemany(records.INSERT_CANDLE, [(ts(n), 1, 1, 1, 1, 1.0, 1, None if n in bad else 1.0, int(n not in bad))
                                             for n in range(kept, total + 1)])
    conn.executemany(records.INSERT_BENCHMARK, [(ts(n), 1, 1, 1, 1, 1, size, 1, 1) for n in range(1, total + 1)
                                                if n not in bad])
    conn.executemany(records.INSERT_FORECAST, [(ts(n), p, 0.1) for n in range(400, total + 1) if n not in spoiled
                                               for p in ("population", "linear")])
    conn.executemany(records.INSERT_DECISION, [
        (b, ts(n), "flat", 1.0, 0.1, 20, 12, 8, 0, 0, "none", "prediction 0.10 bps is below required edge 20.00 bps")
        for b in range(1, size + 1) for n in range(400 + (b - 1 - 400) % horizon, total + 1, horizon)
        if n not in spoiled])
    conn.executemany(records.INSERT_FEED_EVENT, [e for n in sorted(bad) for e in (
        (iso(START + n * MS_PER_MINUTE - 40_000), "clock_skew", "median 2100 ms"),
        (iso(START + n * MS_PER_MINUTE - 40_000), "recovering", "clock skew"),
        (ts(n + 1), "healthy", "a full clean minute"))])
    conn.execute(records.INSERT_CHECKPOINT, (iso(START + total * MS_PER_MINUTE + 30_000), SCHEMA_VERSION, "{}", b""))
    conn.execute("COMMIT")
    store.close()
    ro = data.connect(cfg.db_path)
    started = time.monotonic()
    skips = dashboard.skipped_turns(ro, cfg)
    worst = time.monotonic() - started
    causes = Counter(s.cause for s in skips.values())
    assert len(skips) > 4000 and set(causes) == {"not_healthy", "inputs"}, causes  # every one explained
    for params in ({}, {"page": "1500"}, {"action": "skipped"}, {"q": "return starts", "kind": "learner"},
                   {"bot": "7", "hours": "24"}, {"action": "skipped", "per": "200", "page": "20"}):
        started = time.monotonic()
        decision_page(ro, cfg, parse_filters(params))
        worst = max(worst, time.monotonic() - started)
    started = time.monotonic()
    decision_detail(ro, cfg, next(iter(skips)))
    worst = max(worst, time.monotonic() - started)
    ro.close()
    assert worst < QUERY_BUDGET_S / 2, worst


def test_the_page_offers_skipped_turns():
    from btc_survivor.dashboard import PAGE

    page = PAGE.read_text()
    assert '<option value="skipped">skipped turns</option>' in page and '"hold", "skipped"]' in page
    assert "<b>Skipped turns.</b>" in page and 'class="chip skipped"' in page and "dxSkippedHtml" in page
    assert 'a !== "skipped" || a in res.action_counts' in page  # an older server sends no count, so no chip
    assert "Number.isSafeInteger(v) && v !== 0" in page  # a saved open skipped turn has a negative id
    assert "never counts toward the matured, right or waiting figures" in page


NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node runs the page's own drawing code")
def test_the_page_draws_skipped_rows_and_their_panel(explorer, faulty):
    """The page's own code, run on real answers, draws a skipped row with its chip and kept position, the summary
    with the skipped count apart, and the panel with the cause, the position, the twin, the next decision and the
    named events. An answer from an older server, with no skipped count, still draws without the chip, and an
    answer whose skipped turns failed says so under the decisions."""
    from btc_survivor.dashboard import PAGE

    page = PAGE.read_text()
    script = page[page.index("<script>") + len("<script>"):page.index("</script>")]
    script = script[:script.index('document.getElementById("zone").addEventListener')]  # no start up, no DOM
    _, listed = explorer("/api/decisions?run=faulty&per=200")
    skip = next(r for r in listed["rows"] if r["action"] == SKIPPED)
    _, detail = explorer(f"/api/decision?run=faulty&id={skip['id']}")
    old = {**listed, "summary": {k: v for k, v in listed["summary"].items() if k != "skipped"},
           "action_counts": {k: v for k, v in listed["action_counts"].items() if k != SKIPPED},
           "rows": [r for r in listed["rows"] if r["action"] != SKIPPED]}
    check = f"""
    const res = {json.dumps(listed)}, old = {json.dumps(old)};
    console.log(JSON.stringify({{table: dxTable(res, null), summary: dxSummary(res), counts: dxCounts(res),
      oldCounts: dxCounts(old), oldSummary: dxSummary(old), detail: dxDetailHtml({json.dumps(detail)}),
      failed: dxSummary({{...old, skipped_error: true}})}}));
    """
    run = subprocess.run([NODE, "-e", script + check], capture_output=True, text=True, timeout=120,
                         env={**os.environ, "TZ": "UTC"})
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    assert f'<tr data-id="{skip["id"]}"' in out["table"] and '<span class="chip skipped"' in out["table"]
    assert out["table"].count('class="chip skipped"') == sum(r["action"] == SKIPPED for r in listed["rows"])
    assert 'data-action="skipped"' in out["counts"] and 'data-action="skipped"' not in out["oldCounts"]
    n = listed["summary"]["skipped"]
    assert f"beside them that {'is' if n == 1 else 'are'} never scored" in out["summary"]
    assert "skipped" not in out["oldSummary"]
    assert "could not be worked out" in out["failed"] and "could not be worked out" not in out["summary"]
    assert "<b>Skipped turn</b>" in out["detail"] and "Why it made no decision" in out["detail"]
    if detail["next"] is not None:
        assert f'data-open="{detail["next"]["id"]}"' in out["detail"]
    if detail["twin"] is not None:
        assert f"bot {detail['twin']['bot_id']}, " in out["detail"]
    assert "What it kept" in out["detail"] and "Feed events around it" in out["detail"]
