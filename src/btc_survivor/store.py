"""SQLite storage. One connection does all writes, buffered in memory and committed in batches."""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SLOW_WRITE_SECONDS = 1.0

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bots (
  bot_id INTEGER PRIMARY KEY,
  slot INTEGER NOT NULL,
  parent_id INTEGER,
  generation INTEGER NOT NULL,
  born_at TEXT NOT NULL,
  died_at TEXT,
  state TEXT NOT NULL,
  margin_bps REAL NOT NULL,
  lr REAL NOT NULL,
  seed INTEGER NOT NULL,
  origin TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lineage (
  bot_id INTEGER PRIMARY KEY,
  parent_id INTEGER NOT NULL,
  generation INTEGER NOT NULL,
  born_at TEXT NOT NULL,
  sigma_weights REAL NOT NULL,
  sigma_genes REAL NOT NULL,
  margin_factor REAL NOT NULL,
  lr_factor REAL NOT NULL,
  parent_margin_bps REAL NOT NULL,
  parent_lr REAL NOT NULL,
  parent_fitness REAL,
  parent_skill_corr REAL
);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY,
  bot_id INTEGER NOT NULL,
  ts TEXT NOT NULL,
  position_before TEXT NOT NULL,
  ref_mid REAL NOT NULL,
  prediction_bps REAL NOT NULL,
  required_edge_bps REAL NOT NULL,
  cost_estimate_bps REAL NOT NULL,
  margin_bps REAL NOT NULL,
  gate_passed INTEGER NOT NULL,
  forced INTEGER NOT NULL,
  action TEXT NOT NULL,
  reason TEXT NOT NULL,
  label_bps REAL
);
CREATE INDEX IF NOT EXISTS decisions_bot_ts ON decisions (bot_id, ts);
CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY,
  bot_id INTEGER NOT NULL,
  side TEXT NOT NULL,
  opened_at TEXT NOT NULL,
  closed_at TEXT,
  qty REAL NOT NULL,
  entry_price REAL NOT NULL,
  exit_price REAL,
  fee_open REAL NOT NULL,
  fee_close REAL,
  funding REAL NOT NULL DEFAULT 0,
  slippage_bps_open REAL NOT NULL,
  slippage_bps_close REAL,
  pnl_gross REAL,
  pnl_net REAL,
  forced INTEGER NOT NULL DEFAULT 0,
  exit_reason TEXT
);
CREATE INDEX IF NOT EXISTS trades_bot ON trades (bot_id);
CREATE TABLE IF NOT EXISTS fills (
  id INTEGER PRIMARY KEY,
  bot_id INTEGER NOT NULL,
  trade_id INTEGER,
  decided_at TEXT NOT NULL,
  filled_at TEXT,
  leg TEXT NOT NULL,
  side TEXT NOT NULL,
  qty REAL NOT NULL,
  ref_mid REAL NOT NULL,
  book_avg_price REAL,
  price REAL,
  levels_consumed INTEGER,
  slippage_bps REAL,
  fee REAL,
  latency_ms INTEGER NOT NULL,
  status TEXT NOT NULL,
  reason TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS fills_bot ON fills (bot_id);
CREATE TABLE IF NOT EXISTS equity_snapshots (
  bot_id INTEGER NOT NULL,
  ts TEXT NOT NULL,
  cash REAL NOT NULL,
  equity REAL NOT NULL,
  unrealized REAL NOT NULL,
  realized_gross REAL NOT NULL,
  fees_total REAL NOT NULL,
  funding_total REAL NOT NULL,
  rent_total REAL NOT NULL,
  stop_loss_total REAL NOT NULL,
  position TEXT NOT NULL,
  skill_corr REAL,
  skill_dir_acc REAL,
  skill_mae REAL,
  skill_n INTEGER NOT NULL,
  PRIMARY KEY (bot_id, ts)
);
CREATE TABLE IF NOT EXISTS deaths (
  bot_id INTEGER PRIMARY KEY,
  died_at TEXT NOT NULL,
  generation INTEGER NOT NULL,
  cause TEXT NOT NULL,
  fees_total REAL NOT NULL,
  funding_total REAL NOT NULL,
  rent_total REAL NOT NULL,
  stop_loss_total REAL NOT NULL,
  other_losses_total REAL NOT NULL,
  writeoff REAL NOT NULL,
  threshold REAL NOT NULL,
  final_equity REAL NOT NULL,
  trades INTEGER NOT NULL,
  lifespan_hours REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS feed_events (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  kind TEXT NOT NULL,
  detail TEXT
);
CREATE TABLE IF NOT EXISTS checkpoints (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  schema_version INTEGER NOT NULL,
  meta TEXT NOT NULL,
  arrays BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS candles (
  ts TEXT PRIMARY KEY,
  open REAL,
  high REAL,
  low REAL,
  close REAL,
  volume REAL NOT NULL,
  trade_count INTEGER NOT NULL,
  mid REAL,
  valid INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS benchmark (
  ts TEXT PRIMARY KEY,
  btc_mid REAL NOT NULL,
  buy_hold_equity REAL NOT NULL,
  matched_equity REAL NOT NULL,
  living_mean_equity REAL NOT NULL,
  living_median_equity REAL NOT NULL,
  living INTEGER NOT NULL,
  twin_mean_equity REAL,
  cash_equity REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runtime_status (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  ts TEXT NOT NULL,
  feed_state TEXT NOT NULL,
  best_bid REAL,
  best_ask REAL,
  mid REAL,
  living INTEGER NOT NULL,
  last_checkpoint_ts TEXT,
  rss_mb REAL,
  rss_peak_mb REAL
);
"""


class StoreError(RuntimeError):
    """Raised when the database cannot be opened or does not match the expected schema."""


def _connect(path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


class Store:
    """Owns the single SQLite connection. Writes are queued and committed together by flush."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = _connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA journal_size_limit=67108864")
        self._pending: list[tuple[str, tuple]] = []
        self.created = False

    @classmethod
    def open(cls, path: Path) -> "Store":
        """Open a database, creating the schema only in an empty file. Any other database must match the schema."""
        try:
            store = cls(path)
            store.created = store.conn.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] == 0
            if store.created:
                store.create_schema()
                return store
            problems = store.check_schema()
        except sqlite3.DatabaseError as exc:
            raise StoreError(f"cannot open database {path}: {exc}") from exc
        if problems:
            store.close()
            raise StoreError("database does not match the expected schema\n  " + "\n  ".join(problems))
        return store

    def create_schema(self) -> None:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in _statements(SCHEMA):
                self.conn.execute(statement)
            self.conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
            )
            self.conn.execute("COMMIT")
        except BaseException:
            self._rollback()
            raise

    def check_schema(self) -> list[str]:
        """Compare tables, columns, indexes and the schema version with the expected schema."""
        expected = _connect(":memory:")
        try:
            for statement in _statements(SCHEMA):
                expected.execute(statement)
            problems = _diff_schema(expected, self.conn)
        finally:
            expected.close()
        if not problems:
            version = self.get_meta("schema_version")
            if version != str(SCHEMA_VERSION):
                problems.append(f"schema_version is {version!r}, expected {SCHEMA_VERSION}")
        return problems

    def queue(self, sql: str, params: Sequence[Any] = ()) -> None:
        """Buffer one write. Nothing reaches the database until flush."""
        self._pending.append((sql, tuple(params)))

    @property
    def pending(self) -> int:
        return len(self._pending)

    def flush(self, extra: Callable[[sqlite3.Connection], None] | None = None) -> None:
        """Commit every queued write, plus whatever extra does, in one transaction.

        On failure nothing is committed and the queue is kept, so the same writes can be retried.
        """
        start = time.monotonic()
        rows = len(self._pending)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for sql, params in self._pending:
                self.conn.execute(sql, params)
            if extra is not None:
                extra(self.conn)
            self.conn.execute("COMMIT")
        except BaseException:
            self._rollback()
            raise
        self._pending.clear()
        _log_if_slow(start, rows)

    def execute_now(self, sql: str, params: Sequence[Any] = ()) -> None:
        """Commit a single write immediately, for rows that never feed back into bot state."""
        self.commit_now([(sql, params)])

    def commit_now(self, statements: Sequence[tuple[str, Sequence[Any]]]) -> None:
        """Commit a few writes in their own transaction, leaving the queue untouched."""
        start = time.monotonic()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for sql, params in statements:
                self.conn.execute(sql, tuple(params))
            self.conn.execute("COMMIT")
        except BaseException:
            self._rollback()
            raise
        _log_if_slow(start, len(statements))

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, tuple(params)).fetchall()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row["value"]

    def queue_meta(self, key: str, value: str) -> None:
        self.queue("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))

    def rollback_after(self, ts: str, labels_from: str) -> None:
        """Remove every bot and ledger row written after a checkpoint, in one transaction, keeping feed events.

        Labels matured after ts belong to decisions from labels_from on, so those are cleared too. Used only when the
        newest checkpoint cannot be read and an older one is restored instead.
        """
        statements = [
            ("DELETE FROM decisions WHERE ts > ?", (ts,)),
            ("UPDATE decisions SET label_bps = NULL WHERE ts > ?", (labels_from,)),
            ("DELETE FROM fills WHERE decided_at > ?", (ts,)),
            ("DELETE FROM equity_snapshots WHERE ts > ?", (ts,)),
            ("DELETE FROM candles WHERE ts > ?", (ts,)),
            ("DELETE FROM benchmark WHERE ts > ?", (ts,)),
            ("DELETE FROM deaths WHERE died_at > ?", (ts,)),
            ("DELETE FROM lineage WHERE born_at > ?", (ts,)),
            ("DELETE FROM bots WHERE born_at > ?", (ts,)),
            ("UPDATE bots SET died_at = NULL WHERE died_at > ?", (ts,)),
            ("DELETE FROM trades WHERE opened_at > ?", (ts,)),
            ("UPDATE trades SET closed_at = NULL, exit_price = NULL, fee_close = NULL, slippage_bps_close = NULL, "
             "pnl_gross = NULL, pnl_net = NULL, exit_reason = NULL WHERE closed_at > ?", (ts,)),
            ("DELETE FROM checkpoints WHERE ts > ?", (ts,)),
        ]
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for sql, params in statements:
                self.conn.execute(sql, params)
            self.conn.execute("COMMIT")
        except BaseException:
            self._rollback()
            raise

    def _rollback(self) -> None:
        """Roll back only if SQLite has not already done so itself, so the original error is the one raised."""
        if self.conn.in_transaction:
            self.conn.execute("ROLLBACK")

    def close(self) -> None:
        self.conn.close()


def _log_if_slow(start: float, rows: int) -> None:
    elapsed = time.monotonic() - start
    if elapsed > SLOW_WRITE_SECONDS:
        log.warning("slow database write, %d rows took %.2f s", rows, elapsed)


def _statements(script: str) -> Iterable[str]:
    for statement in script.split(";"):
        if statement.strip():
            yield statement.strip()


def _describe(conn: sqlite3.Connection) -> tuple[dict[str, list[tuple]], set[str]]:
    tables: dict[str, list[tuple]] = {}
    for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"):
        cols = conn.execute(f"PRAGMA table_info({row['name']})").fetchall()
        tables[row["name"]] = [(c["name"], c["type"].upper(), c["notnull"], c["pk"]) for c in cols]
    indexes = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL")}
    return tables, indexes


def _diff_schema(expected: sqlite3.Connection, actual: sqlite3.Connection) -> list[str]:
    want_tables, want_indexes = _describe(expected)
    have_tables, have_indexes = _describe(actual)
    problems: list[str] = []
    for name, cols in want_tables.items():
        if name not in have_tables:
            problems.append(f"missing table {name}")
            continue
        have = {c[0]: c for c in have_tables[name]}
        for col in cols:
            if col[0] not in have:
                problems.append(f"table {name} is missing column {col[0]}")
            elif have[col[0]] != col:
                problems.append(f"table {name} column {col[0]} differs, expected {col[1:]}, found {have[col[0]][1:]}")
        extra = set(have) - {c[0] for c in cols}
        for col_name in sorted(extra):
            problems.append(f"table {name} has unexpected column {col_name}")
    for name in sorted(set(have_tables) - set(want_tables)):
        problems.append(f"unexpected table {name}")
    for name in sorted(want_indexes - have_indexes):
        problems.append(f"missing index {name}")
    return problems
