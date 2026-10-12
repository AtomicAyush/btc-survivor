"""Rows of the news tables are never updated after insert, only pruned, PLAN_V2 sections 7.3 and 9.1. No source file
updates, replaces or upserts a row of a news table, a trigger aborts any UPDATE that slips through, and pruning
deletes exactly the rows past their retention."""

import ast
import re
import sqlite3

import pytest

from btc_survivor.clock import MS_PER_DAY, iso
from btc_survivor.v2.config import load_news_config
from btc_survivor.v2.news import db
from helpers import ROOT

NEWS_PACKAGE = ROOT / "src" / "btc_survivor" / "v2" / "news"
SCANNED = sorted([*(ROOT / "src" / "btc_survivor" / "v2").rglob("*.py"), *(ROOT / "scripts").glob("*.py")])
DYNAMIC = "{}"
NOW = 1_792_000_000_000

_TABLE = r"[\"`\[]?(\w+|\{\})"
PATTERNS = (
    re.compile(rf"\bUPDATE\s+(?:OR\s+\w+\s+)?(?!ON\b){_TABLE}", re.I),
    re.compile(rf"\bREPLACE\s+INTO\s+{_TABLE}", re.I),
    re.compile(rf"\bINTO\s+{_TABLE}(?=.*\bDO\s+UPDATE\b)", re.I | re.S),
)


def literals(source):
    """Every string literal, adjacent literals joined as Python joins them, and each f-string with {} for a field."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value
        elif isinstance(node, ast.JoinedStr):
            yield "".join(part.value if isinstance(part, ast.Constant) else DYNAMIC for part in node.values)


def tables_changed_in_place(source):
    """The tables whose rows SQL in source updates, replaces or upserts, with {} for a table named at run time. A
    trigger's BEFORE UPDATE ON names no table here."""
    return {match.group(1) for text in literals(source) for pattern in PATTERNS for match in pattern.finditer(text)}


def test_the_scan_finds_every_way_to_change_a_row_in_place():
    caught = {
        "sql = \"UPDATE news_items SET status = 'x' WHERE seq = 1\"": "news_items",
        "sql = 'update item_scores set status = ?'": "item_scores",
        "sql = 'UPDATE OR IGNORE news_polls SET detail = 1'": "news_polls",
        "sql = 'UPDATE \"news_heartbeat\" SET healthy = 1'": "news_heartbeat",
        "sql = ('UPDATE '\n       'llm_calls SET status = 1')": "llm_calls",
        "sql = 'INSERT OR REPLACE INTO llm_call_items (call_id) VALUES (1)'": "llm_call_items",
        "sql = 'REPLACE INTO score_impacts VALUES (1)'": "score_impacts",
        "sql = '''INSERT INTO news_dups (id) VALUES (1)\n  ON CONFLICT (id) DO UPDATE SET channel = 2'''": "news_dups",
        "sql = f'UPDATE {table} SET x = 1'": DYNAMIC,
    }
    for source, table in caught.items():
        assert table in tables_changed_in_place(source), source
    innocent = (
        "sql = 'CREATE TRIGGER t BEFORE UPDATE ON news_items BEGIN SELECT 1; END;'",
        "sql = 'INSERT INTO news_items (seq) VALUES (1)'",
        "# UPDATE news_items in a comment is not SQL\nx = 1",
        "doc = 'the collector never updates news rows in place'",
    )
    for source in innocent:
        assert not set(db.INSERT_ONLY) & tables_changed_in_place(source), source
    assert tables_changed_in_place("sql = 'INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)'") == {"meta"}
    assert tables_changed_in_place("sql = \"UPDATE reservations SET status = 'lost'\"") == {"reservations"}


def test_no_source_changes_a_news_row_in_place():
    assert NEWS_PACKAGE / "db.py" in SCANNED
    hits = []
    for path in SCANNED:
        found = tables_changed_in_place(path.read_text(encoding="utf-8"))
        refused = set(db.INSERT_ONLY) | ({DYNAMIC} if NEWS_PACKAGE in path.parents else set())
        hits += [f"{path.relative_to(ROOT)}: {table}" for table in sorted(found & refused)]
    assert not hits, "news rows changed in place\n" + "\n".join(hits)


def minimal_row(conn, table):
    """A row holding a value in every column that needs one, chosen by the column's type."""
    values = {"TEXT": "x", "INTEGER": 1, "REAL": 1.0, "BLOB": b"x"}
    return {row[1]: values[row[2]] for row in conn.execute(f"PRAGMA table_info({table})") if row[3]}


@pytest.mark.parametrize("table", db.INSERT_ONLY)
def test_a_trigger_aborts_any_update_of_a_news_row(tmp_path, table):
    conn = db.open_writer(tmp_path / "news.db")
    try:
        row = minimal_row(conn, table)
        db.insert(conn, table, row)
        column = next(iter(row))
        with pytest.raises(sqlite3.IntegrityError, match=f"{table} rows are never updated"):
            conn.execute(f"UPDATE {table} SET {column} = {column}")
        assert [tuple(r) for r in conn.execute(f"SELECT {', '.join(row)} FROM {table}")] == [tuple(row.values())]
    finally:
        conn.close()


def ago(days, extra_ms=0):
    return iso(NOW - days * MS_PER_DAY - extra_ms)


def seed(conn):
    """Rows on both sides of every retention cutoff, plus old rows of every kind that is kept forever."""
    def item(vendor_id, status, received_at):
        return db.insert(conn, "news_items", {
            "source": "finnhub", "channel": "finnhub_general", "vendor_id": vendor_id, "kind": "media",
            "received_at": received_at, "title": "Fictional headline", "tickers": "", "relevance": "none",
            "status": status, "title_hash": "0123456789abcdef", "route": "none"})

    def body(seq):
        db.insert(conn, "news_bodies", {"item_seq": seq, "text": "Fictional text", "chars_total": 14, "truncated": 0,
                                        "doc_urls": ""})

    def dup(seq, seen_at):
        db.insert(conn, "news_dups", {"item_seq": seq, "channel": "finnhub_company_AAPL", "vendor_id": "d",
                                      "seen_at": seen_at, "method": "hash"})

    old_filtered = item("old_filtered", "filtered", ago(30, 1))
    body(old_filtered)
    dup(old_filtered, ago(1))
    edge_filtered = item("edge_filtered", "filtered", ago(30))
    body(edge_filtered)
    for status in ("new", "seed", "backlog"):
        kept = item(f"old_{status}", status, ago(400))
        body(kept)
        db.insert(conn, "item_scores", {"item_seq": kept, "scorer": "finbert_v1", "route": "none",
                                        "status": "scored", "attempts": 1, "scored_at": ago(400)})
        db.insert(conn, "score_impacts", {"item_seq": kept, "scorer": "finbert_v1", "ticker": "MACRO",
                                          "strength": 0.1})
    dup(edge_filtered, ago(14, 1))
    dup(edge_filtered, ago(14))
    for days, extra in ((14, 1), (14, 0)):
        db.insert(conn, "news_polls", {"channel": "fed_speeches", "started_at": ago(days, extra + 5),
                                       "finished_at": ago(days, extra), "cadence_s": 600, "status": "ok"})
        db.insert(conn, "news_heartbeat", {
            "ts": ago(days, extra), "healthy": 1, "config_hash": "c", "prompt_sha256": "p", "schema_sha256": "s",
            "haiku_state": "on", "sonnet_state": "on", "finbert_state": "on", "waiting": 0, "channels_down": "",
            "spent_day_usd": 0.0, "spent_month_usd": 0.0, "spend_alert": 0})
    db.insert(conn, "collector_events", {"ts": ago(400), "kind": "started"})
    db.insert(conn, "llm_probes", {"run_at": ago(400), "probe_id": "p1", "call_id": 1, "ticker": "AAPL"})
    db.set_values(conn, "meta", {"first_poll.fed_speeches": ago(400)})


def count(conn, table, where="1"):
    return conn.execute(f"SELECT count(*) FROM {table} WHERE {where}").fetchone()[0]


def test_pruning_deletes_exactly_the_rows_past_their_retention(tmp_path):
    retention = db.Retention.from_config(load_news_config(ROOT / "configs" / "v2" / "news.toml"))
    assert retention == db.Retention(filtered_days=30, polls_days=14, dups_days=14, heartbeat_days=14)
    conn = db.open_writer(tmp_path / "news.db")
    try:
        seed(conn)
        before = {table: count(conn, table) for table in db.TABLES}
        deleted = db.prune(conn, NOW, retention)
        assert deleted == {"news_bodies": 1, "news_dups": 2, "news_items": 1, "news_polls": 1, "news_heartbeat": 1}
        after = {table: count(conn, table) for table in db.TABLES}
        assert {table: before[table] - after[table] for table in db.TABLES if before[table] != after[table]} == deleted
        assert count(conn, "news_items", "vendor_id = 'old_filtered'") == 0
        assert count(conn, "news_items", "vendor_id = 'edge_filtered'") == 1
        assert count(conn, "news_items", "status IN ('new', 'seed', 'backlog')") == 3
        assert count(conn, "news_bodies") == 4 and count(conn, "item_scores") == 3
        assert count(conn, "news_dups", f"seen_at = '{ago(14)}'") == 1
        assert count(conn, "news_polls", f"finished_at = '{ago(14)}'") == 1
        assert count(conn, "news_heartbeat", f"ts = '{ago(14)}'") == 1
        assert db.prune(conn, NOW, retention) == dict.fromkeys(deleted, 0)
        assert not conn.in_transaction
    finally:
        conn.close()
