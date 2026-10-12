"""Replay of stock recordings by PLAN_V2 sections 3 and 14.2 on one seeded synthetic recording of prints and news lines
written by the recorder's own StockRecorder, through a disconnect with a refused handshake, a halt and a silence: two
replays give identical table digests, a replay from files equals one from the lines in memory, S1 replays to the same
prices and labels as S2, and a replay never sends a frame, never opens a socket and never opens news.db. Also its
refusals and its command line output, which by default holds only counts, digests and a timing."""

import logging
import socket
import sqlite3
import stat
from types import SimpleNamespace

import pytest

from btc_survivor.clock import MS_PER_MINUTE
from btc_survivor.replay import NEVER_FOR_SELECTION, ReplayError
from btc_survivor.v2 import output
from btc_survivor.v2.news import db as news_db
from btc_survivor.v2.stocks import replay as stock_replay
from btc_survivor.v2.stocks.replay import recorded_hashes, replay, replay_lines, table_digests
from btc_survivor.v2.stocks.runner import StockFeedClient
from btc_survivor.v2.stocks.store import TABLES
from btc_survivor.v2.stocks.synthetic import DISCONNECT, HALT, SILENCE, Fault, write_recording
from test_v2_group import FAST, FRI, HASHES, THU, at, pair, query, raw_tape, tape, write_pair

START, END = at(THU, "09:00"), at(FRI, "11:10")
FAULTS = (Fault(DISCONNECT, at(THU, "11:00"), refused=1), Fault(HALT, at(THU, "13:00"), 20 * MS_PER_MINUTE,
                                                                 symbol="TSLA"),
          Fault(SILENCE, at(THU, "15:00"), 8000))


@pytest.fixture(autouse=True)
def no_leftover_log_handlers(monkeypatch):
    """A version 1 command line test leaves a root stream handler on a stream pytest has since closed, which a
    replay's log lines would reach as logging errors on stderr, so each test here starts without root handlers."""
    monkeypatch.setattr(logging.root, "handlers", [])


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    """The recording files in their own folder, the configs beside them and the lines they hold."""
    folder = tmp_path_factory.mktemp("recording")
    files = write_recording(folder / "data" / "raw_stocks", raw_tape(START, END, faults=FAULTS))
    return SimpleNamespace(folder=folder, files=files, cfgs=pair(folder, **FAST), lines=tape(START, END, faults=FAULTS))


def test_one_recording_replays_twice_with_identical_table_digests(recording, tmp_path):
    s1, s2 = recording.cfgs
    first = replay(s2, recording.files, tmp_path / "a")
    second = replay_lines(s2, recording.lines, tmp_path / "b" / "deeper", recorded_hashes(recording.lines))
    assert first.lines == second.lines == len(recording.lines)
    digests = table_digests(first.db_path)
    assert list(digests) == list(TABLES) and digests == table_digests(second.db_path)
    rows = {table: n for table, (n, _) in digests.items()}
    for table in ("decisions", "fills", "trades", "stock_candles", "stock_labels", "stock_forecasts", "news_seen",
                  "news_minutes", "equity_snapshots", "benchmark", "checkpoints", "sessions", "spread_checks"):
        assert rows[table] > 0, table
    assert rows["runtime_status"] == rows["symbol_status"] == 0
    assert query(first.db_path, "SELECT count(DISTINCT decided_at) FROM fills WHERE status = 'filled'")[0][0] > 1
    kinds = {kind for kind, in query(first.db_path, "SELECT kind FROM feed_events")}
    assert {"disconnected", "connected", "stale", "healthy", "session_close"} <= kinds
    meta = dict(query(first.db_path, "SELECT key, value FROM meta"))
    assert (meta["news_config_hash"], meta["prompt_sha256"], meta["schema_sha256"]) == HASHES
    assert meta["launch_ts"] == "2026-10-29T13:00:00.000Z" and meta["kind"] == "stocks"
    # S1 replays to the same candles, labels, sessions and linear_prices rows while reading no news.
    prices = replay(s1, [recording.folder / "data" / "raw_stocks"], tmp_path / "s1")
    other = table_digests(prices.db_path)
    for table in ("stock_candles", "stock_labels", "sessions", "spread_checks"):
        assert other[table] == digests[table], table
    sql = "SELECT * FROM stock_forecasts WHERE predictor = 'linear_prices' ORDER BY ts, symbol"
    assert query(prices.db_path, sql) == query(first.db_path, sql)
    assert other["news_seen"][0] == other["news_minutes"][0] == 0


def test_a_replay_never_sends_a_frame_never_opens_a_socket_and_never_opens_news_db(recording, tmp_path,
                                                                                  monkeypatch, capsys):
    s1, s2 = write_pair(tmp_path, **FAST)
    news_path = tmp_path / "data" / "news.db"
    opened: list[str] = []
    real_connect = sqlite3.connect

    def spy(target, *args, **kwargs):
        opened.append(str(target))
        return real_connect(target, *args, **kwargs)

    def refuse(*args, **kwargs):
        raise AssertionError("replay reached for the network or news.db")

    monkeypatch.setattr(sqlite3, "connect", spy)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    for name in ("run", "send", "request_resync"):
        monkeypatch.setattr(StockFeedClient, name, refuse)
    monkeypatch.setattr(news_db, "open_reader", refuse)
    monkeypatch.setattr(news_db, "open_writer", refuse)
    out = tmp_path / "replayed"
    assert stock_replay.main([str(s2), *map(str, recording.files), "--out", str(out)]) == 0
    printed = capsys.readouterr()
    lines = printed.out.splitlines()
    assert lines[:5] == ["stock replay", "verdict: pass", f"lines: {len(recording.lines)}", f"tables: {len(TABLES)}",
                         lines[4]]
    assert lines[4].startswith("rows: ") and lines[5].startswith("database digest: ") and len(lines[5]) == 81
    assert lines[6].startswith("run time: ") and lines[6].endswith(" s")
    assert lines[7:] == [f"{len(TABLES)} rows hidden, pass --detail in your own terminal to see them",
                         NEVER_FOR_SELECTION]
    assert printed.err == ""
    assert opened and all(path.startswith(str(out)) or path == ":memory:" or "mode=ro" in path for path in opened)
    assert not any("news.db" in path for path in opened) and not news_path.exists()
    assert not (tmp_path / "data").exists()
    assert [stat.S_IMODE(path.stat().st_mode) for path in (out, out / "stocks.db")] == [0o700, 0o600]
    # With --detail every table's row count and digest follow the same summary.
    report = stock_replay.report(stock_replay.Replayed(out / "stocks.db", len(recording.lines)), 1.0)
    detail = output.render(report, detail=True).splitlines()
    assert detail[:6] == lines[:6] and detail[7].split() == ["table", "rows", "digest"]
    assert [row.split()[0] for row in detail[8:]] == list(TABLES)


def test_a_replay_refuses_a_busy_folder_the_data_folder_and_an_empty_recording(recording, tmp_path, capsys):
    s1, s2 = recording.cfgs
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "file").write_text("x", encoding="utf-8")
    with pytest.raises(ReplayError, match="not empty"):
        replay(s2, recording.files, busy)
    with pytest.raises(ReplayError, match="data folder"):
        replay(s2, recording.files, s2.data_dir)
    with pytest.raises(ReplayError, match="no lines"):
        replay_lines(s2, [], tmp_path / "empty")
    assert not s2.data_dir.exists()
    config, _ = write_pair(tmp_path, **FAST)
    assert stock_replay.main([str(config), str(recording.files[0]), "--out", str(busy)]) == 2
    printed = capsys.readouterr()
    assert printed.out == "" and printed.err.startswith("error: ") and "not empty" in printed.err
    help_text = " ".join(stock_replay.build_parser().format_help().split())
    assert NEVER_FOR_SELECTION in help_text
