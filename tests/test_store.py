import sqlite3

import pytest

from btc_survivor.__main__ import main
from btc_survivor.store import SCHEMA, SCHEMA_VERSION, Store, StoreError
from helpers import example_dict, write_config


def test_new_database_gets_schema_and_pragmas(tmp_path):
    store = Store.open(tmp_path / "sub" / "a.db")
    assert store.query("PRAGMA journal_mode")[0][0] == "wal"
    assert store.query("PRAGMA synchronous")[0][0] == 1  # NORMAL
    assert store.get_meta("schema_version") == str(SCHEMA_VERSION)
    assert store.check_schema() == []
    tables = {r["name"] for r in store.query("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"bots", "decisions", "trades", "fills", "equity_snapshots", "deaths", "feed_events",
            "checkpoints", "candles", "benchmark", "runtime_status", "lineage", "meta"} <= tables
    store.close()


def test_reopening_an_existing_database_checks_and_keeps_data(tmp_path):
    path = tmp_path / "a.db"
    store = Store.open(path)
    store.queue("INSERT INTO feed_events (ts, kind, detail) VALUES (?, ?, ?)", ("t", "connected", None))
    store.flush()
    store.close()
    store = Store.open(path)
    assert len(store.query("SELECT * FROM feed_events")) == 1
    store.close()


def test_schema_mismatch_is_reported(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA.replace("  label_bps REAL\n);\nCREATE INDEX IF NOT EXISTS decisions_bot_ts",
                                      "  note TEXT\n);\nCREATE INDEX IF NOT EXISTS decisions_bot_ts"))
    conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    conn.commit()
    conn.close()
    with pytest.raises(StoreError) as info:
        Store.open(path)
    assert "table decisions is missing column label_bps" in str(info.value)
    assert "table decisions has unexpected column note" in str(info.value)


def test_wrong_schema_version_is_reported(tmp_path):
    path = tmp_path / "v.db"
    Store.open(path).close()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(StoreError, match="schema_version is '99'"):
        Store.open(path)


def test_flush_applies_writes_in_queue_order(tmp_path):
    store = Store.open(tmp_path / "a.db")
    insert = "INSERT INTO trades (id, bot_id, side, opened_at, qty, entry_price, fee_open, slippage_bps_open) VALUES (?, 1, 'long', 't0', 0.01, 100.0, 0.5, 0.1)"
    update = "UPDATE trades SET closed_at = 't1', exit_reason = 'signal' WHERE id = ?"
    store.queue(insert, (1,))
    store.queue(update, (1,))
    store.queue(update, (2,))  # runs before row 2 exists, so it must change nothing
    store.queue(insert, (2,))
    assert store.pending == 4
    store.flush()
    assert store.pending == 0
    rows = store.query("SELECT id, exit_reason FROM trades ORDER BY id")
    assert [(r["id"], r["exit_reason"]) for r in rows] == [(1, "signal"), (2, None)]
    store.close()


def test_database_with_tables_but_no_meta_is_rejected_not_adopted(tmp_path):
    path = tmp_path / "foreign.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE bots (bot_id INTEGER PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()
    with pytest.raises(StoreError) as info:
        Store.open(path)
    assert "missing table meta" in str(info.value)
    conn = sqlite3.connect(path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    conn.close()
    assert tables == {"bots"}  # nothing was written into the foreign database


def test_empty_file_is_treated_as_new(tmp_path):
    path = tmp_path / "empty.db"
    path.write_bytes(b"")
    store = Store.open(path)
    assert store.created and store.check_schema() == []
    store.close()


def test_file_that_is_not_a_database_gives_a_clear_error(tmp_path):
    path = tmp_path / "text.db"
    path.write_text("hello, this is not sqlite " * 100)
    with pytest.raises(StoreError, match="cannot open database"):
        Store.open(path)


def test_failed_flush_reports_the_real_error_when_sqlite_already_rolled_back(tmp_path):
    store = Store.open(tmp_path / "full.db")
    pages = store.query("PRAGMA page_count")[0][0]
    store.conn.execute(f"PRAGMA max_page_count = {pages + 2}")
    store.conn.execute("PRAGMA cache_size = 10")
    for i in range(3000):
        store.queue("INSERT INTO feed_events (ts, kind, detail) VALUES (?, ?, ?)", ("t", "gap", "x" * 200))
    with pytest.raises(sqlite3.OperationalError, match="full"):
        store.flush()
    assert store.pending == 3000
    store.close()


def test_slow_writes_are_logged(tmp_path, caplog, monkeypatch):
    import btc_survivor.store as store_module

    monkeypatch.setattr(store_module, "SLOW_WRITE_SECONDS", -1.0)
    store = Store.open(tmp_path / "a.db")
    store.queue("INSERT INTO feed_events (ts, kind) VALUES (?, ?)", ("t", "connected"))
    with caplog.at_level("WARNING", logger="btc_survivor.store"):
        store.flush()
    assert "slow database write, 1 rows" in caplog.text
    store.close()


def test_flush_is_all_or_nothing_and_keeps_the_queue_on_failure(tmp_path):
    store = Store.open(tmp_path / "a.db")
    store.queue("INSERT INTO feed_events (ts, kind) VALUES (?, ?)", ("t", "connected"))

    def failing_checkpoint(conn):
        conn.execute("INSERT INTO checkpoints (ts, schema_version, meta, arrays) VALUES ('t', 1, '{}', x'00')")
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        store.flush(failing_checkpoint)
    assert store.query("SELECT count(*) FROM feed_events")[0][0] == 0
    assert store.query("SELECT count(*) FROM checkpoints")[0][0] == 0
    assert store.pending == 1
    store.flush(lambda conn: conn.execute("INSERT INTO checkpoints (ts, schema_version, meta, arrays) VALUES ('t', 1, '{}', x'00')"))
    assert store.query("SELECT count(*) FROM feed_events")[0][0] == 1
    assert store.query("SELECT count(*) FROM checkpoints")[0][0] == 1
    store.close()


def test_execute_now_and_meta_upsert(tmp_path):
    store = Store.open(tmp_path / "a.db")
    store.execute_now("INSERT INTO feed_events (ts, kind) VALUES (?, ?)", ("t", "stale"))
    assert store.query("SELECT kind FROM feed_events")[0]["kind"] == "stale"
    store.queue_meta("launch_mid", "100.0")
    store.queue_meta("launch_mid", "101.5")
    store.flush()
    assert store.get_meta("launch_mid") == "101.5"
    store.close()


def test_init_db_command_creates_then_checks(tmp_path, capsys):
    cfg = write_config(tmp_path)
    assert main(["--config", str(cfg), "init-db"]) == 0
    assert "created" in capsys.readouterr().out
    assert main(["--config", str(cfg), "init-db"]) == 0
    assert "matches schema version" in capsys.readouterr().out
    assert (tmp_path / "data" / "btc_survivor.db").exists()


def test_init_db_reports_config_that_is_not_utf8(tmp_path, capsys):
    cfg = write_config(tmp_path)
    cfg.write_bytes(cfg.read_bytes() + b"# caf\xe9\n")
    assert main(["--config", str(cfg), "init-db"]) == 2
    assert "not valid UTF-8" in capsys.readouterr().err


def test_init_db_reports_bad_config(tmp_path, capsys):
    data = example_dict()
    data["trading"]["notional"] = -1.0
    cfg = write_config(tmp_path, data)
    assert main(["--config", str(cfg), "init-db"]) == 2
    assert "notional" in capsys.readouterr().err


def test_rollback_after_removes_or_reopens_everything_newer_and_keeps_the_rest(tmp_path):
    from btc_survivor import records

    store = Store.open(tmp_path / "a.db")
    old, labels_from, mid, cut, new = (f"2026-01-01T00:0{m}:00.000Z" for m in (0, 3, 4, 5, 9))
    rows = [
        (records.INSERT_BOT, (1, 0, None, 0, old, "active", 8.0, 0.001, 1, "random")),
        (records.SET_BOT_DEAD, (new, 1)),
        (records.INSERT_BOT, (2, 0, 1, 1, new, "warmup", 8.0, 0.001, 1, "mutated")),
        (records.INSERT_BOT, (3, 1, None, 0, old, "active", 8.0, 0.001, 1, "random")),
        (records.SET_BOT_DEAD, (old, 3)),
        (records.INSERT_LINEAGE, (2, 1, 1, new, 0.05, 0.2, 1.0, 1.0, 8.0, 0.001, 0.0, None)),
        (records.INSERT_LINEAGE, (4, 3, 1, old, 0.05, 0.2, 1.0, 1.0, 8.0, 0.001, 0.0, None)),
    ]
    for ts, label in ((old, 1.0), (mid, 2.0), (new, 3.0)):
        rows.append((records.INSERT_DECISION, (1, ts, "flat", 1.0, 0.0, 1.0, 1.0, 8.0, 0, 0, "none", "r")))
        rows.append(("UPDATE decisions SET label_bps = ? WHERE ts = ?", (label, ts)))
        rows.append((records.INSERT_FILL, (1, None, ts, ts, "open", "buy", 1.0, 1.0, 1.0, 1.0, 1, 0.0, 0.0, 1000,
                                           "filled", "decision")))
        rows.append((records.INSERT_SNAPSHOT, (1, ts, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "flat", None, None, None, 0)))
        rows.append((records.INSERT_CANDLE, (ts, 1.0, 1.0, 1.0, 1.0, 0.0, 0, 1.0, 1)))
        rows.append((records.INSERT_BENCHMARK, (ts, 1.0, 1.0, 1.0, 1.0, 1.0, 1, None, 1.0)))
        rows.append((records.INSERT_CHECKPOINT, (ts, 1, "{}", b"")))
        rows.append((records.INSERT_FEED_EVENT, (ts, "stale", None)))
        rows.append((records.INSERT_FORECAST, (ts, "population", 1.0)))
        rows.append((records.SET_FORECAST_LABEL, (label, ts)))
    rows += [
        (records.OPEN_TRADE, (1, 1, "long", old, 1.0, 1.0, 0.1, 0.0, 0.0, 0)),
        (records.CLOSE_TRADE, (new, 2.0, 0.1, 0.0, 0.0, 1.0, 0.8, "signal", 1)),
        (records.OPEN_TRADE, (2, 1, "long", new, 1.0, 1.0, 0.1, 0.0, 0.0, 0)),
        (records.OPEN_TRADE, (3, 3, "long", old, 1.0, 1.0, 0.1, 0.0, 0.0, 0)),
        (records.CLOSE_TRADE, (old, 2.0, 0.1, 0.0, 0.0, 1.0, 0.8, "signal", 3)),
        (records.INSERT_DEATH, (1, new, 0, "fees", 1, 0, 0, 0, 0, 0, 1, 1, 1, 1.0)),
        (records.INSERT_DEATH, (3, old, 0, "fees", 1, 0, 0, 0, 0, 0, 1, 1, 1, 1.0)),
    ]
    store.commit_now(rows)
    store.rollback_after(cut, labels_from)

    def col(sql):
        return [tuple(r) for r in store.query(sql)]

    assert col("SELECT bot_id, died_at FROM bots ORDER BY bot_id") == [(1, None), (3, old)]
    assert col("SELECT bot_id FROM lineage") == [(4,)] and col("SELECT bot_id FROM deaths") == [(3,)]
    assert col("SELECT ts, label_bps FROM decisions ORDER BY ts") == [(old, 1.0), (mid, None)]
    assert col("SELECT ts, label_bps FROM forecasts ORDER BY ts") == [(old, 1.0), (mid, None)]
    assert col("SELECT id, closed_at, pnl_net, exit_reason FROM trades ORDER BY id") == [
        (1, None, None, None), (3, old, 0.8, "signal")]
    for table, column in (("fills", "decided_at"), ("equity_snapshots", "ts"), ("candles", "ts"),
                          ("benchmark", "ts"), ("checkpoints", "ts")):
        assert col(f"SELECT {column} FROM {table} ORDER BY {column}") == [(old,), (mid,)]
    assert len(col("SELECT ts FROM feed_events")) == 3  # feed events are never rolled back
    store.close()
