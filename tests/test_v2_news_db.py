"""news.db of PLAN_V2 section 9.1, created by init-db from the news config in WAL mode, owner only, with the plan's
tables and meta.kind set to news, and refused by the news process and by readers when it is of another kind."""

import re
import sqlite3
import stat

import pytest

from btc_survivor.clock import iso
from btc_survivor.v2.config import parse_news_config
from btc_survivor.v2.news import db
from helpers import ROOT

NEWS = ROOT / "configs" / "v2" / "news.toml"
NOW = 1_792_000_000_000


def plan_schema() -> str:
    """The first SQL block of PLAN_V2 section 9.1, the news database as the plan writes it."""
    text = (ROOT / "PLAN_V2.md").read_text(encoding="utf-8")
    section = text[text.index("### 9.1 The news database"):text.index("### 9.2")]
    return re.search(r"```sql\n(.*?)```", section, re.S).group(1)


def layout(conn):
    """Each table's columns with type, not null, default and key position, and its indexes, unique constraints and
    key with their columns."""
    tables = {}
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"):
        columns = [tuple(row)[1:] for row in conn.execute(f"PRAGMA table_info({name})")]
        indexes = sorted((index if origin == "c" else origin, unique, [r[2] for r in conn.execute(
            f"PRAGMA index_info('{index}')")]) for _, index, unique, origin, _ in conn.execute(
            f"PRAGMA index_list({name})"))
        tables[name] = (columns, indexes)
    return tables


def news_config(folder):
    """The committed news config read as if it sat two folders below folder, so its data_dir lands in folder."""
    return parse_news_config(NEWS.read_text(encoding="utf-8"), folder / "configs" / "v2")


def mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def item(vendor_id, **values):
    row = {"source": "finnhub", "channel": "finnhub_general", "vendor_id": vendor_id, "kind": "media",
           "received_at": iso(NOW), "title": "Fictional headline", "tickers": "", "relevance": "macro",
           "status": "new", "title_hash": "0123456789abcdef", "route": "llm"}
    return {**row, **values}


def test_init_db_creates_news_db_in_wal_mode_owner_only_with_kind_news(tmp_path):
    config = news_config(tmp_path)
    path = config.db_path
    assert path == tmp_path / "configs" / "v2" / "../../data/v2/news" / "news.db"
    assert db.init_db(config) is True
    conn = db.open_writer(path)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1
        assert db.get_value(conn, "meta", "kind") == "news"
        assert db.get_value(conn, "meta", "schema_version") == str(db.SCHEMA_VERSION)
        db.insert(conn, "collector_events", {"ts": iso(NOW), "kind": "started", "detail": None})
        files = sorted(p.name for p in path.parent.iterdir())
        assert files == ["news.db", "news.db-shm", "news.db-wal"]
        assert all(mode(path.parent / name) == 0o600 for name in files)
        assert mode(path.parent) == 0o700
    finally:
        conn.close()
    assert db.init_db(config) is False


def test_reopening_a_database_that_lost_its_modes_makes_it_owner_only_again(tmp_path):
    folder = tmp_path / "copied"
    path = folder / "news.db"
    db.open_writer(path).close()
    path.chmod(0o644)
    folder.chmod(0o755)
    conn = db.open_writer(path)
    try:
        assert mode(path) == 0o600 and mode(folder) == 0o700
        db.insert(conn, "collector_events", {"ts": iso(NOW), "kind": "started", "detail": None})
        assert mode(folder / "news.db-wal") == 0o600 and mode(folder / "news.db-shm") == 0o600
    finally:
        conn.close()


def test_the_tables_and_indexes_are_those_of_plan_section_9_1(tmp_path):
    plan = sqlite3.connect(":memory:")
    plan.executescript(plan_schema())
    conn = db.open_writer(tmp_path / "news.db")
    try:
        assert layout(conn) == layout(plan)
        assert set(layout(conn)) == set(db.TABLES)
        assert ("u", 1, ["source", "vendor_id"]) in layout(conn)["news_items"][1]
        triggers = {row[0] for row in conn.execute("SELECT tbl_name FROM sqlite_master WHERE type = 'trigger'")}
        assert triggers == set(db.INSERT_ONLY)
    finally:
        conn.close()
        plan.close()


def other_kind(path, kind):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    if kind is not None:
        conn.execute("INSERT INTO meta VALUES ('kind', ?)", (kind,))
    conn.execute("INSERT INTO meta VALUES ('schema_version', '1')")
    conn.commit()
    conn.close()


@pytest.mark.parametrize("kind, reason", [
    ("stocks", "its meta.kind is 'stocks', not 'news'"),
    (None, "it has no meta.kind, so it is not a version 2 news database"),
])
def test_the_news_process_and_readers_refuse_a_database_of_another_kind(tmp_path, kind, reason):
    path = tmp_path / "other.db"
    other_kind(path, kind)
    before = path.read_bytes()
    for opener in (db.open_writer, db.open_reader):
        with pytest.raises(db.NewsDbError, match=re.escape(reason)):
            opener(path)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["other.db"]


def test_a_file_that_is_not_a_database_is_refused_unchanged(tmp_path):
    path = tmp_path / "news.db"
    path.write_bytes(b"not a database at all, just text that is long enough to fill a header page" * 10)
    before = path.read_bytes()
    for opener in (db.open_writer, db.open_reader):
        with pytest.raises(db.NewsDbError, match="cannot be read as a database"):
            opener(path)
    assert path.read_bytes() == before


def test_a_news_database_with_another_schema_is_refused(tmp_path):
    path = tmp_path / "news.db"
    db.open_writer(path).close()
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE news_items ADD COLUMN extra TEXT")
    conn.execute("DROP INDEX news_polls_channel")
    conn.execute("DROP TRIGGER news_items_insert_only")
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', '2')")
    conn.commit()
    conn.close()
    with pytest.raises(db.NewsDbError) as info:
        db.open_writer(path)
    message = str(info.value)
    for problem in ("table news_items differs from the schema", "missing index news_polls_channel",
                    "missing trigger news_items_insert_only", "schema_version is '2', expected 1"):
        assert problem in message


def test_the_reader_is_read_only_and_sees_later_commits(tmp_path):
    path = tmp_path / "news.db"
    writer = db.open_writer(path)
    reader = db.open_reader(path)
    try:
        assert reader.execute("SELECT count(*) FROM news_items").fetchone()[0] == 0
        db.insert(writer, "news_items", item("1"))
        assert reader.execute("SELECT count(*) FROM news_items").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.execute("INSERT INTO collector_events (ts, kind) VALUES ('x', 'y')")
    finally:
        reader.close()
        writer.close()
    with pytest.raises(db.NewsDbError, match="does not exist"):
        db.open_reader(tmp_path / "absent.db")


def test_a_reader_opens_after_a_clean_stop_of_the_writer(tmp_path):
    path = tmp_path / "news.db"
    writer = db.open_writer(path)
    db.insert(writer, "news_items", item("1"))
    writer.close()
    reader = db.open_reader(path)
    try:
        assert reader.execute("SELECT count(*) FROM news_items").fetchone()[0] == 1
    finally:
        reader.close()


def test_insert_returns_seq_in_commit_order_and_never_replaces(tmp_path):
    conn = db.open_writer(tmp_path / "news.db")
    try:
        seqs = [db.insert(conn, "news_items", item(str(n))) for n in range(3)]
        assert seqs == [1, 2, 3]
        with pytest.raises(sqlite3.IntegrityError):
            db.insert(conn, "news_items", item("1", title="A changed fictional headline"))
        assert conn.execute("SELECT title FROM news_items WHERE vendor_id = '1'").fetchone()[0] == "Fictional headline"
        for table, row in (("meta", {"key": "a", "value": "b"}), ("news_things", {"a": 1}),
                           ("news_items", {}), ("news_items", {"title; DROP TABLE meta": "x"})):
            with pytest.raises(ValueError):
                db.insert(conn, table, row)
    finally:
        conn.close()


def test_a_transaction_commits_together_or_not_at_all(tmp_path):
    conn = db.open_writer(tmp_path / "news.db")
    try:
        with db.transaction(conn):
            seq = db.insert(conn, "news_items", item("8k-1", source="edgar", kind="8k"))
            db.insert(conn, "news_bodies", {"item_seq": seq, "text": "Fictional filing text", "chars_total": 21,
                                            "truncated": 0, "doc_urls": ""})
        with pytest.raises(sqlite3.IntegrityError):
            with db.transaction(conn):
                seq = db.insert(conn, "news_items", item("8k-2", source="edgar", kind="8k"))
                db.insert(conn, "news_bodies", {"item_seq": seq, "text": None, "chars_total": 0, "truncated": 0,
                                                "doc_urls": ""})
        assert [row[0] for row in conn.execute("SELECT vendor_id FROM news_items")] == ["8k-1"]
        assert not conn.in_transaction
    finally:
        conn.close()


def test_meta_and_scorer_state_values_change_but_kind_never_does(tmp_path):
    conn = db.open_writer(tmp_path / "news.db")
    try:
        db.set_values(conn, "meta", {"config_hash": "a" * 64, "cursor.finnhub_general": "41"})
        db.set_values(conn, "meta", {"cursor.finnhub_general": "42"})
        db.set_values(conn, "scorer_state", {"haiku_state": "on"})
        assert db.get_value(conn, "meta", "cursor.finnhub_general") == "42"
        assert db.get_value(conn, "meta", "config_hash") == "a" * 64
        assert db.get_value(conn, "scorer_state", "haiku_state") == "on"
        assert db.get_value(conn, "meta", "absent") is None
        for values in ({"kind": "stocks"}, {"schema_version": "9"}):
            with pytest.raises(ValueError):
                db.set_values(conn, "meta", values)
        with pytest.raises(ValueError):
            db.set_values(conn, "news_items", {"a": "b"})
        with pytest.raises(ValueError):
            db.get_value(conn, "news_items", "a")
        assert db.get_value(conn, "meta", "kind") == "news"
    finally:
        conn.close()
