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
    conn.executescript(SCHEMA.replace("  reason TEXT NOT NULL\n);\nCREATE INDEX IF NOT EXISTS decisions_bot_ts",
                                      "  note TEXT\n);\nCREATE INDEX IF NOT EXISTS decisions_bot_ts"))
    conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    conn.commit()
    conn.close()
    with pytest.raises(StoreError) as info:
        Store.open(path)
    assert "table decisions is missing column reason" in str(info.value)
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
