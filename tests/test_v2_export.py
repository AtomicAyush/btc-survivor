"""The export command on a seeded run database, one CSV file per table in a folder only the owner can open, every
row read back exactly from one read snapshot, only counts and a hash printed, and a database of another kind refused
before any folder is made."""

import csv
import hashlib
import os
import sqlite3
import stat

import pytest

from btc_survivor.v2 import export
from btc_survivor.v2.config import load_stock_config
from btc_survivor.v2.stocks import store
from btc_survivor.v2.stocks.store import StockStore
from test_v2_store import contents, news_database, v1_database
import v2_runkit as kit

EXPECTED_TABLES = ("meta", "bots", "decisions", "trades", "fills", "equity_snapshots", "deaths", "feed_events",
                   "stock_candles", "sessions", "stock_labels", "stock_forecasts", "news_seen", "news_minutes",
                   "spread_checks", "benchmark", "runtime_status", "symbol_status")
CANARIES = (str(kit.CANARY_PRICE), str(kit.CANARY_FILL))


def mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def run_export(capsys, config):
    code = export.main([str(config)])
    out, err = capsys.readouterr()
    return code, out, err


def folder_of(config):
    return load_stock_config(config).data_dir.parent / "exports" / load_stock_config(config).run.name


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def as_text(value):
    """A database value as the csv module writes it, an empty cell for a null."""
    return "" if value is None else str(value)


def test_every_table_but_checkpoints_is_exported_and_only_counts_and_a_hash_are_printed(tmp_path, capsys):
    runs = kit.seeded(tmp_path)
    code, out, err = run_export(capsys, runs.s2)
    assert (code, err) == (0, "")
    assert export.TABLES == EXPECTED_TABLES
    conn = sqlite3.connect(runs.s2_db)
    counts = {table: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in EXPECTED_TABLES}
    conn.close()
    lines = out.splitlines()
    assert lines[:4] == ["export", "verdict: pass", f"tables: {len(EXPECTED_TABLES)}", f"rows: {sum(counts.values())}"]
    assert lines[4:-1] == [f"{table} rows: {n}" for table, n in counts.items()]
    assert lines[-1].startswith("export hash: ") and len(lines[-1].split()[-1]) == 64
    assert counts["decisions"] == 8 and counts["stock_forecasts"] == 1152
    assert not [canary for canary in CANARIES if canary in out]
    folder = folder_of(runs.s2)
    assert folder.resolve() == (tmp_path / "data" / "v2" / "exports" / "s2_news").resolve()
    assert sorted(p.name for p in folder.iterdir()) == sorted(f"{table}.csv" for table in EXPECTED_TABLES)
    assert mode(folder) == 0o700 and {mode(p) for p in folder.iterdir()} == {0o600}


def test_every_row_reads_back_exactly_in_insertion_order(tmp_path, capsys):
    runs = kit.seeded(tmp_path)
    assert run_export(capsys, runs.s1)[0] == 0
    folder = folder_of(runs.s1)
    conn = store.open_reader(runs.s1_db)
    try:
        for table in EXPECTED_TABLES:
            cursor = conn.execute(f"SELECT * FROM {table} ORDER BY rowid")
            header, *rows = read_csv(folder / f"{table}.csv")
            assert header == [column[0] for column in cursor.description], table
            assert rows == [[as_text(value) for value in row] for row in cursor], table
        prices = [float(row[8]) for row in read_csv(folder / "stock_candles.csv")[1:] if row[8]]
        assert kit.CANARY_PRICE in prices  # floats come back exactly
        meta = dict(read_csv(folder / "meta.csv")[1:])
        assert meta["kind"] == "stocks" and meta["experiment_hash"] == kit.EXPERIMENT
    finally:
        conn.close()


def test_a_second_export_gives_the_same_hash_and_replaces_the_files(tmp_path, capsys):
    runs = kit.seeded(tmp_path)
    first = run_export(capsys, runs.s1)[1].splitlines()[-1]
    folder = folder_of(runs.s1)
    (folder / "bots.csv").write_text("edited\n", encoding="utf-8")
    os.chmod(folder / "bots.csv", 0o644)
    os.chmod(folder, 0o755)
    second = run_export(capsys, runs.s1)[1].splitlines()[-1]
    assert first == second
    assert read_csv(folder / "bots.csv")[0][0] == "bot_id"
    assert mode(folder) == 0o700 and mode(folder / "bots.csv") == 0o600


def test_the_hash_covers_each_file_in_table_order(tmp_path, capsys):
    runs = kit.seeded(tmp_path)
    digest = run_export(capsys, runs.s1)[1].splitlines()[-1].split()[-1]
    folder = folder_of(runs.s1)
    expected = hashlib.sha256()
    for table in EXPECTED_TABLES:
        expected.update(f"{table}\n".encode())
        expected.update((folder / f"{table}.csv").read_bytes())
    assert digest == expected.hexdigest()


def test_the_export_reads_one_snapshot_while_the_stock_process_writes_on(tmp_path, capsys, monkeypatch):
    runs = kit.seeded(tmp_path)
    writer = StockStore.open(runs.s1_db)
    original = export.write_table

    def write_then_commit(conn, table, path):
        n = original(conn, table, path)
        if table == "meta":
            writer.commit_now([kit.decision(1, "2026-10-23T14:00:00.000Z", "AAPL", "none")])
        return n

    monkeypatch.setattr(export, "write_table", write_then_commit)
    try:
        assert run_export(capsys, runs.s1)[0] == 0
    finally:
        writer.close()
    assert len(read_csv(folder_of(runs.s1) / "decisions.csv")) == 1 + 7  # the decision committed meanwhile is absent
    assert sqlite3.connect(runs.s1_db).execute("SELECT count(*) FROM decisions").fetchone()[0] == 8


@pytest.mark.parametrize("make, problem", [(news_database, "meta.kind is 'news', not 'stocks'"),
                                           (v1_database, "it has no meta.kind")])
def test_a_news_or_version_1_database_is_refused_before_any_folder_is_made(tmp_path, capsys, make, problem):
    s1, _, _ = kit.configs(tmp_path)
    path = load_stock_config(s1).db_path
    path.parent.mkdir(parents=True)
    make(path)
    before = contents(path)
    code, out, err = run_export(capsys, s1)
    assert (code, out) == (2, "") and "is not a stock run database" in err and problem in err
    assert contents(path) == before
    assert not (tmp_path / "data" / "v2" / "exports").exists()


def test_a_missing_database_or_config_is_refused(tmp_path, capsys):
    s1, _, _ = kit.configs(tmp_path)
    code, _, err = run_export(capsys, s1)
    assert code == 2 and "stocks.db does not exist" in err and not (tmp_path / "data").exists()
    code, _, err = run_export(capsys, tmp_path / "absent.toml")
    assert code == 2 and "cannot read config file" in err
