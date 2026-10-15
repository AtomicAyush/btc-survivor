"""The group and the stock process of PLAN_V2 sections 4.8, 7.1, 13.2 and 14.2 on seeded synthetic recordings and a
news.db in a test folder: one decode feeding both engines, news kinds dispatched before the version 1 decode, recorded
order equal to stamp order, a puller opened before the writer exists seeing later rows, a reopen after 60 seconds
without a heartbeat, each engine resuming from its own news sequence, and a local_only event written to both databases
before each open and at a start inside a session. Also no tick once the feed is told to stop, even when the ticker's
wait times out as the stop lands, and the whole process run end to end in a subprocess against a fake Finnhub on
127.0.0.1 with a shifted clock, and its dry run and lock refusal. No test opens data/v2, the real secrets file or the
network beyond 127.0.0.1.

The helpers here, configs in a test folder, seeded tapes of feed lines with heartbeat and score lines and a launch
through open_runs, are shared with test_v2_stock_replay and test_v2_stock_checkpoint."""

from __future__ import annotations

import asyncio
import functools
import gzip
import heapq
import json
import re
import signal
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import Executor, Future
from contextlib import suppress
from datetime import date, datetime
from datetime import time as clock_time
from pathlib import Path


from btc_survivor.clock import MS_PER_MINUTE, MS_PER_SECOND, LiveClock, iso, parse_iso
from btc_survivor.feed.base import CLOSE, MESSAGE, OPEN, decode
from btc_survivor.recorder import read_recording
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import parse_stock_config
from btc_survivor.v2.local_secrets import FINNHUB
from btc_survivor.v2.localcheck import LocalCheck, LocalCheckError, Listener
from btc_survivor.v2.news import db as news_db
from btc_survivor.v2.news.health import FINBERT_SCORER
from btc_survivor.v2.stocks import checkpoint, process
from btc_survivor.v2.stocks.finnhub import FinnhubAdapter
from btc_survivor.v2.stocks.group import StockGroup
from btc_survivor.v2.stocks.news_inputs import (HAIKU_SCORER, HEARTBEAT_LINE, NEWS_LINE, HeartbeatLine, Impact,
                                                ScoreLine)
from btc_survivor.v2.stocks.replay import replay
from btc_survivor.v2.stocks.runner import StockRecorder
from btc_survivor.v2.stocks.synthetic import SyntheticTape, recorded, trade_message
from helpers import ROOT
from v2_feedkit import FakeFinnhub, receive, script_env

HASHES = ("c" * 64, "p" * 64, "s" * 64)  # the fake news hashes news.db holds and every heartbeat line carries
NEWS_META = {"config_hash": HASHES[0], "prompt_sha256": HASHES[1], "schema_sha256": HASHES[2],
             "haiku_model": "claude-haiku-5-5", "sonnet_model": "claude-sonnet-5-5"}
THU, FRI = sessions.session_on(date(2026, 10, 29)), sessions.session_on(date(2026, 10, 30))
WED = sessions.session_on(date(2026, 11, 4))
SYMBOLS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "QQQ")
# Four bots per run, label scales after 10 labels, 15 minutes of warmup and forced trades after 15 minutes.
FAST = {"size": "4", "min_samples": "10", "warmup_hours": "0.25", "force_trade_after_hours": "0.25"}
CANARY = "CANARYstocksTOKEN0077"
PATHS = {"data_dir": None, "secrets_file": '"secrets.env"', "raw_dir": '"data/raw_stocks"',
         "lock_file": '"lock/finnhub_ws.lock"', "news_db": '"data/news.db"'}


# Shared helpers


def at(session, clock: str) -> int:
    """The UTC milliseconds of a New York clock time on a session's date."""
    parts = [int(p) for p in clock.split(":")]
    return int(datetime.combine(session.day, clock_time(*parts), tzinfo=sessions.new_york()).timestamp()) * 1000


def config_text(name: str, **values: str) -> str:
    """A committed run config with every path moved into its folder and some settings replaced."""
    text = (ROOT / "configs" / "v2" / f"{name}.toml").read_text(encoding="utf-8")
    for key, value in {**PATHS, "data_dir": f'"data/{name}"', **values}.items():
        text, found = re.subn(rf"(?m)^{key} = [^#\n]*", f"{key} = {value} ", text)
        assert found == 1, key
    return text


def pair(folder: Path, **values: str):
    """S1 and S2 with every path in folder."""
    return tuple(parse_stock_config(config_text(name, **values), folder) for name in ("s1_prices", "s2_news"))


def write_pair(folder: Path, **values: str) -> tuple[Path, Path]:
    """S1 and S2 written into folder with a fixture secrets file holding only a fake token."""
    secrets = folder / "secrets.env"
    secrets.write_text(f"{FINNHUB}={CANARY}\n", encoding="utf-8")
    secrets.chmod(0o600)
    paths = []
    for name in ("s1_prices", "s2_news"):
        path = folder / f"{name}.toml"
        path.write_text(config_text(name, **values), encoding="utf-8")
        paths.append(path)
    return paths[0], paths[1]


def adapter() -> FinnhubAdapter:
    return FinnhubAdapter(SYMBOLS, "test")


def heartbeat_lines(start: int, end: int, every: int = 30 * MS_PER_SECOND) -> list[tuple]:
    """A healthy heartbeat line with the launch hashes every 30 seconds, as the puller records them."""
    return [(HEARTBEAT_LINE, HeartbeatLine(seq, t, True, None, *HASHES, ()).encode(), t, process.NEWS_CONN)
            for seq, t in enumerate(range(start - start % every + every, end, every), start=1)]


def score_lines(start: int, end: int, every: int = 7 * MS_PER_MINUTE) -> list[tuple]:
    """A FinBERT line and a Haiku line for one item every 7 minutes, on a company, the index or MACRO in turn, and
    every fifth item a failed Haiku row, which makes its FinBERT score the primary one."""
    out, seq = [], 0
    for item, t in enumerate(range(start + every, end, every), start=1):
        ticker = (*SYMBOLS, "MACRO")[item % 9]
        strength = round(0.9 * ((item * 37) % 11 - 5) / 5, 3)
        for scorer, offset in ((FINBERT_SCORER, -4000), (HAIKU_SCORER, 0)):
            seq += 1
            failed = scorer == HAIKU_SCORER and item % 5 == 0
            impacts = () if failed else (Impact(ticker, strength, None if scorer == FINBERT_SCORER else True),)
            line = ScoreLine(seq, item, "finnhub", "media", "llm", scorer, "failed" if failed else "scored",
                             t + offset - 1500, (ticker,), impacts)
            out.append((NEWS_LINE, line.encode(), t + offset, process.NEWS_CONN))
    return out


@functools.lru_cache(maxsize=4)
def feed_lines(start: int, end: int, seed: int = 7, faults: tuple = ()) -> tuple:
    """The lines of a seeded synthetic market and recorder from start to end through faults, with the control lines
    that mark the recorder's start for write_recording."""
    return tuple(SyntheticTape(seed, SYMBOLS, regular_rate=1.0).lines(start, end, faults))


def raw_tape(start: int, end: int, seed: int = 7, faults: tuple = ()) -> list[tuple]:
    """Feed lines with heartbeat and score lines merged in stamp order, with the control lines, for write_recording."""
    return list(heapq.merge(feed_lines(start, end, seed, faults), heartbeat_lines(start, end),
                            score_lines(start, end), key=lambda line: line[2]))


def tape(start: int, end: int, seed: int = 7, faults: tuple = ()) -> list[tuple]:
    """The lines one recording of the feed and the news lines holds, in stamp order."""
    return recorded(raw_tape(start, end, seed, faults))


def launch(folder: Path, start: int, **values: str):
    """Both runs launched fresh at start through open_runs, with news.db's meta faked, and their group."""
    cfgs = pair(folder, **{**FAST, **values})
    group, start_seq = process.open_runs(cfgs, adapter(), start, "test-commit", news_meta=lambda _: (NEWS_META, 0))
    assert start_seq == 0
    return cfgs, group


def run_lines(group: StockGroup, lines, until: int | None = None) -> int:
    """Hand lines to the group up to until, returning how many it took."""
    used = 0
    for line in lines:
        if until is not None and line[2] >= until:
            break
        group.on_line(*line)
        used += 1
    if until is not None:
        group.advance_to(until)
    return used


def query(path, sql: str, params=()) -> list[tuple]:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def close(group: StockGroup) -> None:
    for engine in group.engines:
        engine.store.close()


# news.db rows


def news_database(path: Path) -> sqlite3.Connection:
    """news.db created in a test folder with the meta the news process writes on start."""
    conn = news_db.open_writer(path)
    news_db.set_values(conn, "meta", NEWS_META)
    return conn


def add_score(conn: sqlite3.Connection, ticker: str, scored_at: str, title: str = "Fictional headline") -> int:
    """One routed item with a Haiku score and its impact, returning the score's sequence number."""
    item = news_db.insert(conn, "news_items", {
        "source": "finnhub", "channel": "finnhub_company", "vendor_id": f"{title} {scored_at}", "kind": "media",
        "received_at": scored_at, "title": title, "tickers": ticker, "relevance": "title", "status": "new",
        "title_hash": "h", "route": "llm"})
    with news_db.transaction(conn):
        seq = news_db.insert(conn, "item_scores", {"item_seq": item, "scorer": HAIKU_SCORER, "route": "llm",
                                                   "status": "scored", "attempts": 1, "scored_at": scored_at})
        news_db.insert(conn, "score_impacts", {"item_seq": item, "scorer": HAIKU_SCORER, "ticker": ticker,
                                               "strength": 0.4, "direction": 1, "confidence": 2, "is_new": 1})
    return seq


def add_heartbeat(conn: sqlite3.Connection, ts: str, hashes=HASHES) -> int:
    return news_db.insert(conn, "news_heartbeat", {
        "ts": ts, "healthy": 1, "reason": None, "config_hash": hashes[0], "prompt_sha256": hashes[1],
        "schema_sha256": hashes[2], "haiku_state": "on", "sonnet_state": "on", "finbert_state": "on", "waiting": 0,
        "oldest_wait_s": None, "channels_down": "", "spent_day_usd": 0.0, "spent_month_usd": 0.0, "spend_alert": 0})


class ScriptedClock(LiveClock):
    """A live clock whose wall time the test sets, even backwards, keeping the real stamp rule."""

    def __init__(self, t: int) -> None:
        super().__init__()
        self.t = t

    def now_ms(self) -> int:
        return self.t


class Inline(Executor):
    """Runs each submitted call at once, so the next tick sees its result."""

    def submit(self, fn, /, *args, **kwargs):
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future


PASSED = LocalCheck((), ())


# One decode for both engines


class CountingAdapter(FinnhubAdapter):
    """The Finnhub adapter keeping every text it was asked to parse."""

    def __init__(self) -> None:
        super().__init__(SYMBOLS, "test")
        self.parsed: list[str] = []

    def parse(self, raw, recv_ms, conn_id):
        self.parsed.append(raw)
        return super().parse(raw, recv_ms, conn_id)


def spy(engine, name: str, calls: list) -> None:
    """Record the run, method and identity of every argument of each call to one engine method."""
    original = getattr(engine, name)

    def wrapped(*args):
        calls.append((engine.cfg.run.name, name, tuple(id(a) for a in args)))
        return original(*args)

    setattr(engine, name, wrapped)


def test_one_decode_feeds_both_engines_the_same_frozen_results(tmp_path):
    cfgs, group = launch(tmp_path, at(THU, "09:20"))
    group.adapter = counting = CountingAdapter()
    calls: list = []
    for engine in group.engines:
        for name in ("on_prints", "on_minute", "on_events", "on_news", "check_orders"):
            spy(engine, name, calls)
    lines = tape(at(THU, "09:20"), at(THU, "10:05"))
    run_lines(group, lines, at(THU, "10:05"))
    messages = [raw for kind, raw, _, _ in lines if kind == MESSAGE]
    assert counting.parsed == messages and len(messages) > 2000
    assert group.counts["messages"] == len(messages) and group.counts["news"] > 0
    # Both engines get the same calls with the very same objects, each reaching S1 before S2.
    assert {name for _, name, _ in calls} >= {"on_prints", "on_minute", "on_news"}
    order = {run: [i for i, call in enumerate(calls) if call[0] == run] for run in ("s1_prices", "s2_news")}
    assert [calls[i][1:] for i in order["s1_prices"]] == [calls[i][1:] for i in order["s2_news"]]
    assert all(a < b for a, b in zip(order["s1_prices"], order["s2_news"]))
    assert sum(name == "on_minute" for _, name, _ in calls) == 2 * 35
    group.persist()
    close(group)
    candles = [query(c.db_path, "SELECT * FROM stock_candles ORDER BY symbol, ts") for c in cfgs]
    assert len(candles[0]) == 35 * 8 and candles[0] == candles[1]


def test_news_kinds_are_dispatched_before_the_version_1_decode(tmp_path):
    cfgs, group = launch(tmp_path, at(THU, "09:20"))
    group.adapter = counting = CountingAdapter()
    t = at(THU, "09:40")
    beat = HeartbeatLine(1, t, True, None, *HASHES, ()).encode()
    score = ScoreLine(1, 1, "finnhub", "media", "llm", HAIKU_SCORER, "scored", t - 500, ("AAPL",),
                      (Impact("AAPL", 0.6, True),)).encode()
    group.on_line(HEARTBEAT_LINE, beat, t, process.NEWS_CONN)
    group.on_line(NEWS_LINE, score, t + 1, process.NEWS_CONN)
    group.on_line(NEWS_LINE, "not a news line", t + 2, process.NEWS_CONN)
    group.on_line(HEARTBEAT_LINE, '{"seq": 2}', t + 3, process.NEWS_CONN)
    assert counting.parsed == [] and dict(group.counts) == {"heartbeats": 1, "news": 1, "unreadable": 2}
    assert [e.news_seq for e in group.engines] == [1, 1]
    assert [line.seq for line in group.engines[1].inputs.news.kept_lines()] == [1]
    # The version 1 decode hands any kind it does not know to the adapter, which would count a news line as an
    # unknown Finnhub message.
    assert decode(counting, NEWS_LINE, score, t, process.NEWS_CONN) == []
    assert counting.parsed == [score] and sum(counting.unknown.values()) == 1
    group.persist()
    close(group)
    assert query(cfgs[1].db_path, "SELECT score_seq, scorer, primary_use FROM news_seen") == [(1, HAIKU_SCORER, 1)]
    assert query(cfgs[0].db_path, "SELECT count(*) FROM news_seen") == [(0,)]


# Recorded order, the puller and the news sequences


def trade(t: int, price: float) -> str:
    return trade_message([{"p": price, "s": s, "t": t - 120, "v": 100} for s in SYMBOLS])


def test_recorded_order_equals_stamp_order_and_a_replay_sees_what_the_engines_saw(tmp_path):
    cfgs, group = launch(tmp_path, at(THU, "09:25"))
    conn = news_database(cfgs[0].news_db_path)
    clock = ScriptedClock(at(THU, "09:25"))
    recorder = StockRecorder(cfgs[0].raw_dir)
    puller = process.NewsPuller(cfgs[0].news_db_path, 0, 60 * MS_PER_SECOND)
    proc = process.StockProcess(group, puller, clock, recorder, check=lambda: PASSED, executor=Inline())
    seen: list[tuple[str, int]] = []
    original = group.on_line
    group.on_line = lambda kind, raw, recv_ms, conn_id: (seen.append((kind, recv_ms)),
                                                          original(kind, raw, recv_ms, conn_id))[1]
    proc.sink(OPEN, "", clock.stamp(group.now_ms), 1)
    for i in range(1500):
        # The wall clock steps back a little now and then, as an NTP slew can, and stamps never do.
        clock.t += 1300 if i % 7 else -900
        proc.sink(MESSAGE, trade(clock.t, 100.0 + (i % 13) / 10), clock.stamp(group.now_ms), 1)
        if i % 20 == 0:
            add_heartbeat(conn, iso(clock.t))
        if i % 150 == 75:
            add_score(conn, SYMBOLS[i % 8], iso(clock.t - 200))
        if i % 3 == 0:
            proc.tick()
    proc.sink(CLOSE, "shutdown", clock.stamp(group.now_ms), 1)
    proc.finish()
    proc.close()
    close(group)
    conn.close()
    lines = list(read_recording([cfgs[0].raw_dir]))
    stamps = [t for _, _, t, _ in lines]
    assert stamps == sorted(stamps)
    assert [(kind, t) for kind, _, t, _ in lines] == seen
    kinds = {kind for kind, *_ in lines}
    assert kinds == {OPEN, MESSAGE, CLOSE, NEWS_LINE, HEARTBEAT_LINE}
    assert sum(kind == NEWS_LINE for kind in (k for k, *_ in lines)) == 10
    assert all("Fictional headline" not in raw for _, raw, _, _ in lines)
    replayed = replay(cfgs[1], [cfgs[0].raw_dir], tmp_path / "replayed")
    for sql in ("SELECT * FROM stock_candles ORDER BY symbol, ts",
                "SELECT score_seq, item_seq, recv_at, scorer, primary_use FROM news_seen ORDER BY score_seq"):
        live = query(cfgs[1].db_path, sql)
        assert live and live == query(replayed.db_path, sql), sql


def test_a_puller_opened_before_the_writer_exists_sees_later_rows(tmp_path):
    path = tmp_path / "news" / "news.db"
    puller = process.NewsPuller(path, 0, 60 * MS_PER_SECOND)
    t = at(THU, "10:00")
    assert puller.pull(t) == [] and puller.failures == 1 and not path.exists()
    conn = news_database(path)
    assert puller.pull(t + 1000) == [] and puller.opens == 1 and puller.heartbeat_seq == 0
    add_heartbeat(conn, iso(t + 1500))
    seq = add_score(conn, "NVDA", iso(t + 1600), title="Nothing here may reach a line")
    lines = puller.pull(t + 2000)
    assert [kind for kind, _ in lines] == [NEWS_LINE, HEARTBEAT_LINE]
    assert "Nothing here" not in lines[0][1]
    line = ScoreLine.decode(lines[0][1], t + 2000)
    assert (line.seq, line.scorer, line.tickers, line.impacts) == (seq, HAIKU_SCORER, ("NVDA",),
                                                                    (Impact("NVDA", 0.4, True),))
    assert HeartbeatLine.decode(lines[1][1], t + 2000).hashes == HASHES
    # The writer stops cleanly and starts again, and the open read only connection still sees its new rows.
    conn.close()
    conn = news_db.open_writer(path)
    add_score(conn, "MACRO", iso(t + 3000))
    assert [kind for kind, _ in puller.pull(t + 4000)] == [NEWS_LINE] and puller.opens == 1
    assert puller.pull(t + 5000) == []
    conn.close()
    puller.close()


def test_the_puller_reopens_news_db_after_60_seconds_without_a_heartbeat_and_after_an_error(tmp_path, caplog):
    path = tmp_path / "news.db"
    conn = news_database(path)
    add_heartbeat(conn, "2026-10-29T14:00:00.000Z")
    t = at(THU, "10:00")
    puller = process.NewsPuller(path, 0, 60 * MS_PER_SECOND)
    assert puller.pull(t) == [] and puller.heartbeat_seq == 1
    for beat in range(1, 5):
        add_heartbeat(conn, iso(t + beat * 30_000))
        assert [kind for kind, _ in puller.pull(t + beat * 30_000 + 100)] == [HEARTBEAT_LINE]
    assert puller.opens == 1
    quiet = t + 4 * 30_000 + 100
    assert puller.pull(quiet + 59_000) == [] and puller.opens == 1
    with caplog.at_level("WARNING"):
        assert puller.pull(quiet + 60_000) == []
    assert puller.opens == 2 and "reopening news.db" in caplog.text
    add_heartbeat(conn, iso(quiet + 61_000))
    assert [kind for kind, _ in puller.pull(quiet + 62_000)] == [HEARTBEAT_LINE]
    assert puller.pull(quiet + 121_000) == [] and puller.opens == 2
    assert puller.pull(quiet + 122_000) == [] and puller.opens == 3
    # A connection that breaks is closed and opened again at the next pull, and the cursors survive the error.
    puller.conn.close()
    assert puller.pull(quiet + 123_000) == [] and puller.failures == 1 and puller.conn is None
    add_heartbeat(conn, iso(quiet + 124_000))
    assert [kind for kind, _ in puller.pull(quiet + 125_000)] == [HEARTBEAT_LINE] and puller.opens == 4
    assert puller.heartbeat_seq == 7
    conn.close()
    puller.close()


def rewrite_news_seq(path: Path, news_seq: int) -> None:
    """Put another news sequence into a run's newest checkpoint, as if its engine had read less before it."""
    row = query(path, "SELECT id, meta, arrays FROM checkpoints ORDER BY id DESC LIMIT 1")[0]
    meta, arrays = checkpoint.decode(row[1], row[2])
    meta["engine"]["news_seq"] = news_seq
    blob = checkpoint.v1.Arrays()
    for key, value in arrays.items():
        blob.put(key, value)
    text, data = checkpoint.v1.encode(meta, blob)
    conn = sqlite3.connect(path)
    with conn:
        conn.execute("UPDATE checkpoints SET meta = ?, arrays = ? WHERE id = ?", (text, data, row[0]))
    conn.close()


def test_a_first_launch_starts_at_the_newest_score_and_each_engine_resumes_from_its_own_news_sequence(tmp_path):
    cfgs = pair(tmp_path, **FAST)
    conn = news_database(cfgs[0].news_db_path)
    for minute in range(3):
        add_score(conn, "AAPL", iso(at(THU, "09:00") + minute * 60_000))
    clock = ScriptedClock(at(THU, "10:00"))
    group, start = process.open_runs(cfgs, adapter(), clock.stamp(), "test-commit")
    assert start == 3 and [e.news_seq for e in group.engines] == [3, 3]
    for c in cfgs:
        meta = dict(query(c.db_path, "SELECT key, value FROM meta"))
        assert meta["news_start_seq"] == "3" and meta["prompt_sha256"] == HASHES[1] and meta["kind"] == "stocks"
        assert meta["code_commit"] == "test-commit" and meta["launch_ts"] == iso(at(THU, "10:00"))
    puller = process.NewsPuller(cfgs[0].news_db_path, start, 60 * MS_PER_SECOND)
    proc = process.StockProcess(group, puller, clock, check=lambda: PASSED, executor=Inline())
    for minute in range(3, 6):
        add_score(conn, "MSFT", iso(at(THU, "10:00") + minute * 1000))
    clock.t += 10_000
    proc.tick()
    assert [e.news_seq for e in group.engines] == [6, 6]
    group.persist()
    close(group)
    rewrite_news_seq(cfgs[0].db_path, 4)
    add_score(conn, "META", iso(at(THU, "10:01")))
    add_score(conn, "MACRO", iso(at(THU, "10:01:30")))
    clock.t = at(THU, "10:05")
    group, start = process.open_runs(cfgs, adapter(), clock.stamp(), "test-commit")
    assert start == 4 and [e.news_seq for e in group.engines] == [4, 6]
    proc = process.StockProcess(group, process.NewsPuller(cfgs[0].news_db_path, start, 60_000), clock,
                                check=lambda: PASSED, executor=Inline())
    proc.tick()
    assert [e.news_seq for e in group.engines] == [8, 8]
    assert [line.seq for line in group.engines[1].inputs.news.kept_lines()] == [4, 5, 6, 7, 8]
    group.persist()
    close(group)
    conn.close()
    assert query(cfgs[1].db_path, "SELECT score_seq FROM news_seen ORDER BY score_seq") == [(s,) for s in range(4, 9)]
    for c in cfgs:
        commits = query(c.db_path, "SELECT detail FROM feed_events WHERE kind = 'code_commit'")
        assert commits == [("test-commit",), ("test-commit",)]
        assert query(c.db_path, "SELECT count(*) FROM feed_events WHERE kind = 'lock'") == [(2,)]


# The local only check


def local_events(path) -> list[tuple[int, dict]]:
    return [(parse_iso(ts), json.loads(detail)) for ts, detail in
            query(path, "SELECT ts, detail FROM feed_events WHERE kind = 'local_only' ORDER BY id")]


def test_a_local_only_event_goes_to_both_databases_before_each_open_and_at_a_start_inside_a_session(tmp_path):
    cfgs = pair(tmp_path, **FAST)
    news_database(cfgs[0].news_db_path).close()
    clock = ScriptedClock(at(THU, "09:00"))
    group, start = process.open_runs(cfgs, adapter(), clock.stamp(), "test-commit")
    checks = iter([PASSED, LocalCheck((Listener(7, "nc", "0.0.0.0", 8060),), ())])
    proc = process.StockProcess(group, process.NewsPuller(cfgs[0].news_db_path, start, 60_000), clock,
                                check=lambda: next(checks), executor=Inline())
    while clock.t < at(FRI, "09:40"):
        proc.tick()
        upcoming = sessions.next_session(clock.t)
        near = upcoming.open_ms - 15 * MS_PER_MINUTE <= clock.t <= upcoming.open_ms + 10 * MS_PER_MINUTE
        clock.t += 30_000 if near else 300_000
    group.persist()
    close(group)
    for c in cfgs:
        events = local_events(c.db_path)
        assert [detail for _, detail in events] == [
            {"offending_sockets": 0, "passed": True, "tunnel_processes": 0},
            {"offending_sockets": 1, "passed": False, "tunnel_processes": 0}]
        for (ms, _), session in zip(events, (THU, FRI)):
            assert session.open_ms - 10 * MS_PER_MINUTE <= ms < session.open_ms
    # A start inside a session checks at once, and a check that cannot run is written as failed.

    def broken():
        raise LocalCheckError("lsof could not run (FileNotFoundError)")

    clock.t = at(FRI, "11:00")
    group, start = process.open_runs(cfgs, adapter(), clock.stamp(), "test-commit")
    proc = process.StockProcess(group, process.NewsPuller(cfgs[0].news_db_path, start, 60_000), clock,
                                check=broken, executor=Inline())
    for _ in range(3):
        proc.tick()
        clock.t += 1000
    group.persist()
    close(group)
    for c in cfgs:
        ms, detail = local_events(c.db_path)[-1]
        assert at(FRI, "11:00") <= ms < at(FRI, "11:00:05") and detail["passed"] is False


# Raw recordings


def test_the_process_prunes_recordings_older_than_keep_raw_days_once_a_utc_day(tmp_path):
    cfgs, group = launch(tmp_path, at(THU, "09:00"))
    folder = cfgs[0].raw_dir
    folder.mkdir(parents=True)
    assert cfgs[0].stock_feed.keep_raw_days == 30
    kept = ["2026-09-29_000000.jsonl.gz", "2026-09-30_131500.jsonl.gz", "2026-10-29_120000.jsonl.gz",
            "notes.jsonl.gz", "2026-09-01_000000.txt"]
    for name in ["2026-09-28_235959.jsonl.gz", *kept]:
        (folder / name).write_bytes(b"")
    # Thursday 29 October at 13:00 UTC keeps the 30 days from 29 September on, and only names the recorder writes go.
    clock = ScriptedClock(at(THU, "09:00"))
    recorder = StockRecorder(folder)
    unpruned = process.StockProcess(group, None, clock, recorder, check=lambda: PASSED, executor=Inline())
    unpruned.tick()
    assert len(list(folder.iterdir())) == len(kept) + 1  # a soak or a test passes no keep_raw_days and keeps all
    proc = process.StockProcess(group, None, clock, recorder, check=lambda: PASSED, executor=Inline(),
                                keep_raw_days=cfgs[0].stock_feed.keep_raw_days)
    proc.tick()
    assert sorted(p.name for p in folder.iterdir()) == sorted(kept)
    # Once a UTC day, so a file left behind waits for the next day, which also moves the cutoff on by one day.
    (folder / "2026-09-27_080000.jsonl.gz").write_bytes(b"")
    clock.t += 5 * MS_PER_MINUTE
    proc.tick()
    assert (folder / "2026-09-27_080000.jsonl.gz").exists()
    clock.t = at(FRI, "09:00")
    proc.tick()
    assert sorted(p.name for p in folder.iterdir()) == sorted(kept[1:])
    assert process.prune_recordings(folder, at(FRI, "09:00"), 30) == 0
    proc.close()
    close(group)


# The whole process


class ClosingFeed:
    """A feed client that opens at once and records its close the moment the feed is told to stop, as the real one
    does when its close is already under way, after which the news process commits one more heartbeat."""

    def __init__(self, proc: process.StockProcess, news: sqlite3.Connection) -> None:
        self.proc, self.news = proc, news
        self.adapter, self.conn_id, self.connected_ms = adapter(), 1, None

    def stamp(self) -> int:
        return self.proc.clock.stamp(self.proc.group.now_ms)

    async def run(self, stop: asyncio.Event) -> None:
        self.proc.sink(OPEN, "", self.stamp(), self.conn_id)
        await stop.wait()
        self.proc.sink(CLOSE, "shutdown", self.stamp(), self.conn_id)
        add_heartbeat(self.news, iso(self.stamp()))

    async def send(self, text: str) -> bool:
        return False


def test_no_tick_runs_once_the_feed_is_told_to_stop_even_when_the_wait_times_out(tmp_path, monkeypatch):
    # A wait of zero always times out, even after the stop, as the one second wait can when its timeout and the stop
    # land within a few loop steps of each other, up to three on Python 3.11 and fewer on 3.13.
    monkeypatch.setattr(process, "TICK_SECONDS", 0)
    t = at(THU, "10:00")
    cfgs, group = launch(tmp_path, t)
    news = news_database(cfgs[0].news_db_path)
    add_heartbeat(news, iso(t))
    puller = process.NewsPuller(cfgs[0].news_db_path, 0, 60 * MS_PER_SECOND, heartbeat_seq=0)
    proc = process.StockProcess(group, puller, ScriptedClock(t), StockRecorder(cfgs[0].raw_dir), check=lambda: PASSED,
                                executor=Inline())
    proc.client = ClosingFeed(proc, news)

    async def main() -> bool:
        stop = asyncio.Event()
        served = asyncio.create_task(process.serve(proc, stop))
        deadline = time.monotonic() + 5
        while puller.last_heartbeat_ms is None and not served.done():
            assert time.monotonic() < deadline
            await asyncio.sleep(0.01)
        stop.set()
        done, _ = await asyncio.wait({served}, timeout=5)
        served.cancel()
        with suppress(asyncio.CancelledError):
            await served
        return bool(done)

    finished = asyncio.run(main())
    proc.close()
    news.close()
    assert [kind for kind, *_ in read_recording([cfgs[0].raw_dir])] == [OPEN, HEARTBEAT_LINE, CLOSE]
    assert finished and query(cfgs[0].db_path, "SELECT feed_state FROM runtime_status") == [("stopped",)]
    close(group)


SCRIPT = """
import os, sys, time
from btc_survivor import clock
from btc_survivor.v2 import localcheck
from btc_survivor.v2.stocks import process
OFFSET = int(os.environ["STOCKS_CLOCK_OFFSET_MS"])
clock.LiveClock.now_ms = lambda self: time.time_ns() // 1_000_000 + OFFSET
localcheck.run_check = lambda: localcheck.LocalCheck((), ())
process.code_commit = lambda: "test-commit"
sys.exit(process.main(sys.argv[1:]))
"""


def trades_play(offset: int):
    """A fake Finnhub that takes the eight subscribe frames, then sends a print of every symbol each 50 ms with a
    vendor time 120 ms before the shifted clock."""
    async def play(ws, frames):
        await receive(ws, frames, 8)
        price = 100.0
        while True:
            now = time.time_ns() // 1_000_000 + offset
            price += 0.01
            await ws.send(trade(now, round(price, 2)))
            await asyncio.sleep(0.05)
    return play


def recording_text(folder: Path) -> str:
    return "".join(gzip.open(path, "rt", encoding="utf-8").read() for path in sorted(folder.glob("*.jsonl.gz")))


def test_the_stock_process_runs_end_to_end_against_a_fake_finnhub(tmp_path):
    s1, s2 = write_pair(tmp_path, **FAST)
    cfgs = pair(tmp_path, **FAST)
    conn = news_database(cfgs[0].news_db_path)
    add_score(conn, "AAPL", "2026-11-04T16:00:00.000Z")
    offset = at(WED, "11:59:55") - time.time_ns() // 1_000_000
    with FakeFinnhub(trades_play(offset)) as server:
        child = subprocess.Popen([sys.executable, "-c", SCRIPT, str(s1), str(s2), "--test-url", server.url,
                                  "--log-dir", str(tmp_path / "logs")], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, cwd=tmp_path,
                                 env=script_env(tmp_path, STOCKS_CLOCK_OFFSET_MS=str(offset)))
        try:
            deadline = time.monotonic() + 150
            while not (cfgs[0].db_path.exists() and query(cfgs[0].db_path, "SELECT count(*) FROM sqlite_master")
                       [0][0] and query(cfgs[0].db_path, "SELECT count(*) FROM runtime_status")[0][0]):
                assert child.poll() is None and time.monotonic() < deadline
                time.sleep(0.2)
            launched = query(cfgs[0].db_path, "SELECT value FROM meta WHERE key = 'launch_ts'")[0][0]
            boundary = (parse_iso(launched) // MS_PER_MINUTE + 1) * MS_PER_MINUTE
            step = 0
            while step < 8 or time.time_ns() // 1_000_000 + offset < boundary + 3000:
                assert child.poll() is None and time.monotonic() < deadline
                add_heartbeat(conn, iso(time.time_ns() // 1_000_000 + offset))
                if step == 4:
                    add_score(conn, "MACRO", iso(time.time_ns() // 1_000_000 + offset - 300))
                step += 1
                time.sleep(0.5)
            child.send_signal(signal.SIGTERM)
            out, err = child.communicate(timeout=60)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(10)
    conn.close()
    assert child.returncode == 0, err[-2000:]
    assert out == "" and "Traceback" not in err and CANARY not in err
    assert sorted(server.frames[0][:8]) == sorted(json.dumps({"type": "subscribe", "symbol": s},
                                                             separators=(",", ":")) for s in SYMBOLS)
    assert CANARY in server.paths[0]
    text = recording_text(cfgs[0].raw_dir)
    assert CANARY not in text
    lines = list(read_recording([cfgs[0].raw_dir]))
    kinds = [kind for kind, *_ in lines]
    assert kinds[0] == OPEN and kinds[-1] == CLOSE and kinds.count(NEWS_LINE) == 1
    assert kinds.count(HEARTBEAT_LINE) >= 4 and kinds.count(MESSAGE) > 20
    stamps = [t for _, _, t, _ in lines]
    assert stamps == sorted(stamps)
    for c in cfgs:
        assert query(c.db_path, "SELECT count(*) FROM stock_candles WHERE ts = ?", (iso(boundary),)) == [(8,)]
        assert query(c.db_path, "SELECT feed_state FROM runtime_status") == [("stopped",)]
        assert query(c.db_path, "SELECT count(*) FROM checkpoints")[0][0] >= 2
        kinds = {kind for kind, in query(c.db_path, "SELECT kind FROM feed_events")}
        assert {"lock", "code_commit", "local_only", "connected", "session_open"} <= kinds
        dump = "\n".join(map(str, query(c.db_path, "SELECT * FROM feed_events")))
        assert CANARY not in dump
    assert query(cfgs[1].db_path, "SELECT count(*) FROM news_seen") == [(1,)]
    assert query(cfgs[0].db_path, "SELECT count(*) FROM news_seen") == [(0,)]
    logs = "".join(path.read_text(encoding="utf-8") for path in (tmp_path / "logs").glob("*.log"))
    assert "heartbeat feed=" in logs and CANARY not in logs
    # The recording alone replays to the candles the live engines wrote.
    replayed = replay(cfgs[1], [cfgs[0].raw_dir], tmp_path / "replayed")
    sql = "SELECT * FROM stock_candles ORDER BY symbol, ts"
    assert query(replayed.db_path, sql) == query(cfgs[1].db_path, sql)


def test_the_dry_run_takes_and_releases_the_lock_and_a_held_lock_is_refused(tmp_path):
    s1, s2 = write_pair(tmp_path, **FAST)
    command = [sys.executable, "-m", "btc_survivor.v2.stocks.process", str(s1), str(s2), "--dry-run",
               "--test-url", "ws://127.0.0.1:9"]
    done = subprocess.run(command, capture_output=True, text=True, timeout=60, cwd=tmp_path, env=script_env(tmp_path))
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == ["stocks dry run", "verdict: pass", "runs: 2", "symbols: 8",
                                        "lock: taken and released"]
    assert not (tmp_path / "data").exists() and CANARY not in done.stdout + done.stderr
    lock = tmp_path / "lock" / "finnhub_ws.lock"
    holder = subprocess.Popen([sys.executable, "-c", "import sys, time\nfrom btc_survivor.v2.stocks.runner import "
                               "websocket_lock\nwith websocket_lock(sys.argv[1]):\n    print('held', flush=True)\n"
                               "    time.sleep(60)\n", str(lock)], stdout=subprocess.PIPE, text=True,
                              env=script_env(tmp_path))
    try:
        assert holder.stdout.readline() == "held\n"
        refused = subprocess.run(command[:-3], capture_output=True, text=True, timeout=60, cwd=tmp_path,
                                 env=script_env(tmp_path))
        assert refused.returncode == 2 and refused.stdout == ""
        assert refused.stderr.splitlines()[-1].startswith(f"error: another process, id {holder.pid}, holds")
        assert not (tmp_path / "data").exists()
    finally:
        holder.kill()
        holder.wait(10)
    bad = subprocess.run([*command[:4], str(s1), "--dry-run"], capture_output=True, text=True, timeout=60,
                         cwd=tmp_path, env=script_env(tmp_path))
    assert bad.returncode == 2 and "error:" in bad.stderr
