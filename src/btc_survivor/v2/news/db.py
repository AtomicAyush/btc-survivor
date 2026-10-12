"""news.db of PLAN_V2 section 9.1, created owner only in WAL mode with meta.kind set to news.

The news process holds the one writer connection. Rows of the news tables are inserted once and never changed, a
trigger on each of them aborts any UPDATE, and pruning by retention is the only delete. Every other process opens the
file read only with mode=ro, never immutable, so it always sees later commits. Times are ISO 8601 UTC text with
milliseconds and a trailing Z, as btc_survivor.clock.iso writes them.
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from btc_survivor.clock import MS_PER_DAY, iso
from btc_survivor.v2.config import NewsConfig

KIND = "news"
SCHEMA_VERSION = 1

# The tables whose rows are inserted once and never changed, only pruned, and the two that hold changing values.
INSERT_ONLY = ("news_items", "news_bodies", "news_dups", "news_polls", "collector_events", "item_scores",
               "score_impacts", "llm_calls", "llm_call_items", "llm_probes", "news_heartbeat")
KEY_VALUE = ("meta", "scorer_state")
TABLES = KEY_VALUE + INSERT_ONLY

TABLES_SQL = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE news_items (
  seq INTEGER PRIMARY KEY,
  source TEXT NOT NULL, channel TEXT NOT NULL,
  vendor_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  received_at TEXT NOT NULL, detected_at TEXT, vendor_ts TEXT,
  title TEXT NOT NULL, summary TEXT, url TEXT, publisher TEXT,
  tickers TEXT NOT NULL,
  relevance TEXT NOT NULL,
  status TEXT NOT NULL, filter_reason TEXT,
  title_hash TEXT NOT NULL, items_8k TEXT,
  route TEXT NOT NULL,
  UNIQUE (source, vendor_id));
CREATE INDEX news_items_received ON news_items (received_at);
CREATE INDEX news_items_hash ON news_items (source, title_hash);
CREATE TABLE news_bodies (item_seq INTEGER PRIMARY KEY, text TEXT NOT NULL, chars_total INTEGER NOT NULL,
  truncated INTEGER NOT NULL, doc_urls TEXT NOT NULL);
CREATE TABLE news_dups (id INTEGER PRIMARY KEY, item_seq INTEGER NOT NULL, channel TEXT NOT NULL,
  vendor_id TEXT NOT NULL, seen_at TEXT NOT NULL, method TEXT NOT NULL, jaccard REAL, new_tickers TEXT);
CREATE TABLE news_polls (
  seq INTEGER PRIMARY KEY, channel TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT NOT NULL,
  cadence_s INTEGER NOT NULL,
  status TEXT NOT NULL,
  http_status INTEGER, bytes INTEGER, duration_ms INTEGER, budget_wait_ms INTEGER,
  n_raw INTEGER, n_new INTEGER, n_backlog INTEGER, n_dup INTEGER, n_filtered INTEGER,
  detail TEXT);
CREATE INDEX news_polls_channel ON news_polls (channel, finished_at);
CREATE TABLE collector_events (id INTEGER PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT NULL, detail TEXT);
CREATE TABLE item_scores (
  seq INTEGER PRIMARY KEY, item_seq INTEGER NOT NULL,
  scorer TEXT NOT NULL, route TEXT NOT NULL,
  status TEXT NOT NULL,
  call_id INTEGER, attempts INTEGER NOT NULL,
  scored_at TEXT NOT NULL,
  p_pos REAL, p_neg REAL, p_neu REAL,
  UNIQUE (item_seq, scorer));
CREATE TABLE score_impacts (item_seq INTEGER NOT NULL, scorer TEXT NOT NULL, ticker TEXT NOT NULL,
  strength REAL NOT NULL,
  direction INTEGER, surprise INTEGER, event_type TEXT, confidence INTEGER, horizon TEXT, is_new INTEGER,
  PRIMARY KEY (item_seq, scorer, ticker));
CREATE TABLE llm_calls (
  call_id INTEGER PRIMARY KEY, root_call_id INTEGER, ledger_id INTEGER NOT NULL,
  kind TEXT NOT NULL,
  scorer TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
  status TEXT NOT NULL,
  n_items INTEGER NOT NULL, model TEXT NOT NULL, response_model TEXT, request_id TEXT, sdk_version TEXT NOT NULL,
  prompt_sha256 TEXT NOT NULL, schema_sha256 TEXT NOT NULL, request_sha256 TEXT NOT NULL,
  effort TEXT NOT NULL, thinking TEXT NOT NULL, max_tokens INTEGER NOT NULL,
  stop_reason TEXT, refusal_category TEXT, http_status INTEGER, error_class TEXT,
  thinking_blocks INTEGER NOT NULL DEFAULT 0, counted_tokens INTEGER, count_basis TEXT,
  input_tokens INTEGER, cache_creation_input_tokens INTEGER, cache_read_input_tokens INTEGER, output_tokens INTEGER,
  reserved_usd REAL NOT NULL, cost_usd REAL, cost_basis TEXT,
  price_table TEXT NOT NULL, latency_ms INTEGER, response_z BLOB);
CREATE TABLE llm_call_items (call_id INTEGER NOT NULL, item_seq INTEGER NOT NULL, position TEXT NOT NULL,
  PRIMARY KEY (call_id, item_seq));
CREATE TABLE llm_probes (run_at TEXT NOT NULL, probe_id TEXT NOT NULL, call_id INTEGER NOT NULL,
  ticker TEXT NOT NULL, event_type TEXT, is_new INTEGER, direction INTEGER, surprise INTEGER, horizon TEXT,
  confidence INTEGER, PRIMARY KEY (run_at, probe_id, ticker));
CREATE TABLE scorer_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE news_heartbeat (
  seq INTEGER PRIMARY KEY, ts TEXT NOT NULL, healthy INTEGER NOT NULL, reason TEXT,
  config_hash TEXT NOT NULL, prompt_sha256 TEXT NOT NULL, schema_sha256 TEXT NOT NULL,
  haiku_state TEXT NOT NULL, sonnet_state TEXT NOT NULL, finbert_state TEXT NOT NULL,
  waiting INTEGER NOT NULL, oldest_wait_s REAL,
  channels_down TEXT NOT NULL,
  spent_day_usd REAL NOT NULL, spent_month_usd REAL NOT NULL, spend_alert INTEGER NOT NULL);
"""

SCHEMA = TABLES_SQL + "".join(
    f"CREATE TRIGGER {table}_insert_only BEFORE UPDATE ON {table} "
    f"BEGIN SELECT RAISE(ABORT, '{table} rows are never updated'); END;\n" for table in INSERT_ONLY)

# Written out in full, so the immutability scan can read every table a statement replaces rows in.
_PUT = {"meta": "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        "scorer_state": "INSERT OR REPLACE INTO scorer_state (key, value) VALUES (?, ?)"}
# Items with status filtered are pruned after keep_filtered_days, with their bodies and duplicate rows.
_OLD_FILTERED = "SELECT seq FROM news_items WHERE status = 'filtered' AND received_at < ?"
_COLUMN = re.compile(r"[a-z][a-z0-9_]*")


class NewsDbError(RuntimeError):
    """Raised when a file is not a news database of this schema, such as a stock run or a version 1 database."""


@dataclass(frozen=True)
class Retention:
    """Days kept for the pruned rows of section 9.1. Every other row is kept forever."""

    filtered_days: int
    polls_days: int
    dups_days: int
    heartbeat_days: int

    @classmethod
    def from_config(cls, config: NewsConfig) -> Retention:
        """The keep_*_days settings of the news config."""
        news = config.news
        return cls(news.keep_filtered_days, news.keep_polls_days, news.keep_dups_days, news.keep_heartbeat_days)


def _connect(target: str, uri: bool = False) -> sqlite3.Connection:
    """A connection in autocommit mode, so only transaction() opens a transaction, with rows readable by name."""
    conn = sqlite3.connect(target, uri=uri, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def _objects(conn: sqlite3.Connection) -> dict[tuple[str, str], str]:
    """Every table, index and trigger by type and name, with its SQL in single spaces."""
    rows = conn.execute("SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'")
    return {(row[0], row[1]): " ".join(row[2].split()) for row in rows}


def _expected() -> dict[tuple[str, str], str]:
    """The objects SCHEMA creates, built in memory."""
    conn = _connect(":memory:")
    try:
        conn.executescript(SCHEMA)
        return _objects(conn)
    finally:
        conn.close()


def check(conn: sqlite3.Connection) -> list[str]:
    """The reasons conn is not a news database of this schema, none when it is. The kind is checked first, so a
    stock run or version 1 database is named as such."""
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'kind'").fetchone()
    except sqlite3.OperationalError:
        row = None
    if row is None:
        return ["it has no meta.kind, so it is not a version 2 news database"]
    if row[0] != KIND:
        return [f"its meta.kind is {row[0]!r}, not {KIND!r}"]
    want, have = _expected(), _objects(conn)
    problems = [f"missing {kind} {name}" for kind, name in sorted(want.keys() - have.keys())]
    problems += [f"{kind} {name} differs from the schema" for kind, name in sorted(want.keys() & have.keys())
                 if want[kind, name] != have[kind, name]]
    problems += [f"unexpected {kind} {name}" for kind, name in sorted(have.keys() - want.keys())]
    version = get_value(conn, "meta", "schema_version")
    if version != str(SCHEMA_VERSION):
        problems.append(f"schema_version is {version!r}, expected {SCHEMA_VERSION}")
    return problems


def _require(conn: sqlite3.Connection, path: Path) -> None:
    """Raise NewsDbError naming every problem unless conn is a news database of this schema."""
    try:
        problems = check(conn)
    except sqlite3.DatabaseError as exc:
        problems = [f"it cannot be read as a database, {exc}"]
    if problems:
        raise NewsDbError(f"{path} is not a news database of schema version {SCHEMA_VERSION}\n  "
                          + "\n  ".join(problems))


def _create(conn: sqlite3.Connection, path: Path) -> None:
    """Make an empty file a news database, owner only and in WAL mode, with the schema and meta in one transaction."""
    os.chmod(path, 0o600)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        conn.executescript(f"BEGIN IMMEDIATE;\n{SCHEMA}INSERT INTO meta (key, value) VALUES ('kind', '{KIND}'), "
                           f"('schema_version', '{SCHEMA_VERSION}');\nCOMMIT;\n")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def open_writer(path: Path) -> sqlite3.Connection:
    """The news process's one writer connection, creating news.db with mode 600 in a mode 700 folder when it is
    absent or empty, and refusing a database of another kind or schema before changing anything in it. Every open
    sets those modes again, on the WAL and shared memory files too, so a copy that lost them, as a move between
    machines can, is for the owner alone before anything is written."""
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    try:
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    except FileExistsError:
        pass
    for file in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if file.exists():
            os.chmod(file, 0o600)
    conn = _connect(str(path))
    try:
        try:
            empty = conn.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] == 0
        except sqlite3.DatabaseError as exc:
            raise NewsDbError(f"{path} is not a news database, it cannot be read as a database, {exc}") from None
        if empty:
            _create(conn, path)
        _require(conn, path)
        if conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
            raise NewsDbError(f"{path} could not be put in WAL mode")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA journal_size_limit=67108864")
    except BaseException:
        conn.close()
        raise
    return conn


def open_reader(path: Path) -> sqlite3.Connection:
    """A read only connection for any other process, opened with mode=ro and never immutable so it sees every later
    commit, refusing a database of another kind or schema."""
    path = Path(path)
    if not path.is_file():
        raise NewsDbError(f"{path} does not exist")
    conn = _connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        _require(conn, path)
    except BaseException:
        conn.close()
        raise
    return conn


def init_db(config: NewsConfig) -> bool:
    """What init-db does with the news config, creating news.db or checking an existing one, and whether it created
    the database."""
    path = config.db_path
    existed = path.is_file() and path.stat().st_size > 0
    open_writer(path).close()
    return not existed


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """One BEGIN IMMEDIATE to COMMIT, rolled back on any error, so an item with its body or a score with its impacts
    is committed together. Outside one, each statement commits on its own."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def insert(conn: sqlite3.Connection, table: str, row: Mapping[str, Any]) -> int:
    """Insert one row into a news table and return its rowid, the seq of an item, poll, score or heartbeat. A plain
    INSERT fails on a duplicate key, so no row is ever replaced."""
    if table not in INSERT_ONLY:
        raise ValueError(f"{table} is not an insert only news table")
    columns = list(row)
    if not columns or not all(_COLUMN.fullmatch(column) for column in columns):
        raise ValueError(f"bad column names for {table}")
    sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})"
    return conn.execute(sql, [row[column] for column in columns]).lastrowid


def set_values(conn: sqlite3.Connection, table: str, values: Mapping[str, str]) -> None:
    """Write key value pairs to meta or scorer_state, the two tables whose values change, such as the hashes and model
    ids written on every start. The kind and schema version are written only when the database is created."""
    if table not in _PUT:
        raise ValueError(f"{table} is not meta or scorer_state")
    if table == "meta" and {"kind", "schema_version"} & set(values):
        raise ValueError("meta kind and schema_version are set only when news.db is created")
    conn.executemany(_PUT[table], list(values.items()))


def get_value(conn: sqlite3.Connection, table: str, key: str) -> str | None:
    """One value from meta or scorer_state, or None when the key is absent."""
    if table not in KEY_VALUE:
        raise ValueError(f"{table} is not meta or scorer_state")
    row = conn.execute(f"SELECT value FROM {table} WHERE key = ?", (key,)).fetchone()
    return None if row is None else row[0]


def prune(conn: sqlite3.Connection, now_ms: int, retention: Retention) -> dict[str, int]:
    """Delete filtered items older than their retention with their bodies and duplicate rows, and polls, duplicates
    and heartbeats older than theirs, in one transaction, returning the rows deleted per table. A row exactly at its
    cutoff is kept, and nothing else is ever deleted."""
    def cutoff(days: int) -> str:
        return iso(now_ms - days * MS_PER_DAY)

    filtered, dups = cutoff(retention.filtered_days), cutoff(retention.dups_days)
    with transaction(conn):
        return {
            "news_bodies": conn.execute(f"DELETE FROM news_bodies WHERE item_seq IN ({_OLD_FILTERED})",
                                        (filtered,)).rowcount,
            "news_dups": conn.execute(f"DELETE FROM news_dups WHERE seen_at < ? OR item_seq IN ({_OLD_FILTERED})",
                                      (dups, filtered)).rowcount,
            "news_items": conn.execute("DELETE FROM news_items WHERE status = 'filtered' AND received_at < ?",
                                       (filtered,)).rowcount,
            "news_polls": conn.execute("DELETE FROM news_polls WHERE finished_at < ?",
                                       (cutoff(retention.polls_days),)).rowcount,
            "news_heartbeat": conn.execute("DELETE FROM news_heartbeat WHERE ts < ?",
                                           (cutoff(retention.heartbeat_days),)).rowcount,
        }
