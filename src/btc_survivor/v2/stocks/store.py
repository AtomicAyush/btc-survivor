"""The stock run database of PLAN_V2 section 9.2, one per run with meta.kind set to stocks, and the read only open
and kind check that every reader of a run database calls.

StockStore keeps the version 1 Store's one writer connection, its queue and its flush, and replaces the schema.
equity_snapshots, deaths and checkpoints keep their version 1 shape, and lineage, candles and forecasts are not
created. bots gains home_symbol, decisions and trades gain symbol, fills gains symbol, print_price, allowance_bps,
notional, sec_fee and taf_fee, and feed_events gains symbol, null for a whole feed event, each as the last columns.
Every table after checkpoints is the SQL block of section 9.2 as written. A version 1 database has no meta.kind and
a news database has kind news, so the stock process, evaluate, status, export and the dashboard all refuse both.
Times are ISO 8601 UTC text with milliseconds and a trailing Z, as btc_survivor.clock.iso writes them.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from btc_survivor.clock import parse_iso
from btc_survivor.store import Store, StoreError, _diff_schema
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import StockConfig

KIND = "stocks"
SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE bots (
  bot_id INTEGER PRIMARY KEY, slot INTEGER NOT NULL, parent_id INTEGER, generation INTEGER NOT NULL,
  born_at TEXT NOT NULL, died_at TEXT, state TEXT NOT NULL, margin_bps REAL NOT NULL, lr REAL NOT NULL,
  seed INTEGER NOT NULL, origin TEXT NOT NULL,
  home_symbol TEXT NOT NULL);
CREATE TABLE decisions (
  id INTEGER PRIMARY KEY, bot_id INTEGER NOT NULL, ts TEXT NOT NULL, position_before TEXT NOT NULL,
  ref_mid REAL NOT NULL, prediction_bps REAL NOT NULL, required_edge_bps REAL NOT NULL,
  cost_estimate_bps REAL NOT NULL, margin_bps REAL NOT NULL, gate_passed INTEGER NOT NULL, forced INTEGER NOT NULL,
  action TEXT NOT NULL, reason TEXT NOT NULL, label_bps REAL,
  symbol TEXT NOT NULL);
CREATE INDEX decisions_bot_ts ON decisions (bot_id, ts);
CREATE TABLE trades (
  id INTEGER PRIMARY KEY, bot_id INTEGER NOT NULL, side TEXT NOT NULL, opened_at TEXT NOT NULL, closed_at TEXT,
  qty REAL NOT NULL, entry_price REAL NOT NULL, exit_price REAL, fee_open REAL NOT NULL, fee_close REAL,
  funding REAL NOT NULL DEFAULT 0, slippage_bps_open REAL NOT NULL, slippage_bps_close REAL, pnl_gross REAL,
  pnl_net REAL, forced INTEGER NOT NULL DEFAULT 0, exit_reason TEXT,
  symbol TEXT NOT NULL);
CREATE INDEX trades_bot ON trades (bot_id);
CREATE TABLE fills (
  id INTEGER PRIMARY KEY, bot_id INTEGER NOT NULL, trade_id INTEGER, decided_at TEXT NOT NULL, filled_at TEXT,
  leg TEXT NOT NULL, side TEXT NOT NULL, qty REAL NOT NULL, ref_mid REAL NOT NULL, book_avg_price REAL, price REAL,
  levels_consumed INTEGER, slippage_bps REAL, fee REAL, latency_ms INTEGER NOT NULL, status TEXT NOT NULL,
  reason TEXT NOT NULL,
  symbol TEXT NOT NULL, print_price REAL, allowance_bps REAL NOT NULL, notional REAL, sec_fee REAL, taf_fee REAL);
CREATE INDEX fills_bot ON fills (bot_id);
CREATE TABLE equity_snapshots (
  bot_id INTEGER NOT NULL, ts TEXT NOT NULL, cash REAL NOT NULL, equity REAL NOT NULL, unrealized REAL NOT NULL,
  realized_gross REAL NOT NULL, fees_total REAL NOT NULL, funding_total REAL NOT NULL, rent_total REAL NOT NULL,
  stop_loss_total REAL NOT NULL, position TEXT NOT NULL, skill_corr REAL, skill_dir_acc REAL, skill_mae REAL,
  skill_n INTEGER NOT NULL,
  PRIMARY KEY (bot_id, ts));
CREATE TABLE deaths (
  bot_id INTEGER PRIMARY KEY, died_at TEXT NOT NULL, generation INTEGER NOT NULL, cause TEXT NOT NULL,
  fees_total REAL NOT NULL, funding_total REAL NOT NULL, rent_total REAL NOT NULL, stop_loss_total REAL NOT NULL,
  other_losses_total REAL NOT NULL, writeoff REAL NOT NULL, threshold REAL NOT NULL, final_equity REAL NOT NULL,
  trades INTEGER NOT NULL, lifespan_hours REAL NOT NULL);
CREATE TABLE feed_events (
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT NULL, detail TEXT,
  symbol TEXT);
CREATE TABLE checkpoints (
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL, schema_version INTEGER NOT NULL, meta TEXT NOT NULL,
  arrays BLOB NOT NULL);
CREATE TABLE stock_candles (symbol TEXT NOT NULL, ts TEXT NOT NULL, session_date TEXT NOT NULL,
  session_k INTEGER NOT NULL, open REAL, high REAL, low REAL, close REAL, ref_price REAL,
  volume REAL NOT NULL, buy_volume REAL NOT NULL, sell_volume REAL NOT NULL,
  prints INTEGER NOT NULL, price_prints INTEGER NOT NULL, late_prints INTEGER NOT NULL, valid INTEGER NOT NULL,
  PRIMARY KEY (symbol, ts));
CREATE TABLE sessions (session_date TEXT PRIMARY KEY, open_ts TEXT NOT NULL, close_ts TEXT NOT NULL,
  early_close INTEGER NOT NULL, healthy_minutes INTEGER NOT NULL, run_minutes INTEGER NOT NULL);
CREATE TABLE stock_labels (ts TEXT NOT NULL, symbol TEXT NOT NULL, entry_price REAL NOT NULL,
  exit_price REAL, label_bps REAL, status TEXT NOT NULL,
  PRIMARY KEY (ts, symbol));
CREATE TABLE stock_forecasts (ts TEXT NOT NULL, symbol TEXT NOT NULL, predictor TEXT NOT NULL,
  prediction REAL NOT NULL,
  items INTEGER, inputs_max_recv_at TEXT,
  PRIMARY KEY (ts, symbol, predictor));
CREATE TABLE news_seen (
  score_seq INTEGER PRIMARY KEY, item_seq INTEGER NOT NULL, recv_at TEXT NOT NULL, scored_at TEXT NOT NULL,
  source TEXT NOT NULL, route TEXT NOT NULL, scorer TEXT NOT NULL, status TEXT NOT NULL,
  tickers TEXT NOT NULL, primary_use INTEGER NOT NULL, fallback INTEGER NOT NULL);
CREATE TABLE news_minutes (ts TEXT PRIMARY KEY, healthy INTEGER NOT NULL, reason TEXT,
  channels_down TEXT NOT NULL);
CREATE TABLE spread_checks (session_date TEXT NOT NULL, symbol TEXT NOT NULL, prints INTEGER NOT NULL,
  roll_half_spread_bps REAL, allowance_bps REAL NOT NULL, flagged INTEGER NOT NULL,
  PRIMARY KEY (session_date, symbol));
CREATE TABLE benchmark (ts TEXT PRIMARY KEY,
  ew_buy_hold REAL NOT NULL,
  qqq_buy_hold REAL NOT NULL, matched_equity REAL NOT NULL,
  living_mean_equity REAL NOT NULL, living_median_equity REAL NOT NULL, living INTEGER NOT NULL,
  twin_mean_equity REAL, cash_equity REAL NOT NULL);
CREATE TABLE runtime_status (id INTEGER PRIMARY KEY CHECK (id = 1), ts TEXT NOT NULL, feed_state TEXT NOT NULL,
  session_state TEXT NOT NULL, prints_per_s REAL, msgs_per_s REAL, news_age_s REAL, living INTEGER NOT NULL,
  last_checkpoint_ts TEXT, rss_mb REAL, rss_peak_mb REAL);
CREATE TABLE symbol_status (symbol TEXT PRIMARY KEY, ts TEXT NOT NULL, state TEXT NOT NULL,
  ref_price REAL, last_print_age_s REAL);
"""

# Every table in schema order, the version 1 shapes first and then the tables of the section 9.2 block.
TABLES = ("meta", "bots", "decisions", "trades", "fills", "equity_snapshots", "deaths", "feed_events", "checkpoints",
          "stock_candles", "sessions", "stock_labels", "stock_forecasts", "news_seen", "news_minutes",
          "spread_checks", "benchmark", "runtime_status", "symbol_status")


def _connect(target: str, uri: bool = False) -> sqlite3.Connection:
    """A connection in autocommit mode with rows readable by name, as the version 1 schema diff needs."""
    conn = sqlite3.connect(target, uri=uri, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def kind_problems(conn: sqlite3.Connection, kind: str) -> list[str]:
    """Why conn is not a version 2 database of the given kind, none when its meta.kind matches. A version 1 database
    has a meta table without a kind, so it is named here before any table is compared."""
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'kind'").fetchone()
    except sqlite3.OperationalError:
        return ["it has no meta table, so it is not a version 2 database"]
    if row is None:
        return ["it has no meta.kind, so it is a version 1 database or not a version 2 database at all"]
    if row[0] != kind:
        return [f"its meta.kind is {row[0]!r}, not {kind!r}"]
    return []


def schema_problems(conn: sqlite3.Connection) -> list[str]:
    """Why conn is not a stock run database of this schema, none when it is, checking the kind first and then every
    table, column and index through the version 1 schema diff and the schema version."""
    problems = kind_problems(conn, KIND)
    if problems:
        return problems
    expected = _connect(":memory:")
    try:
        expected.executescript(SCHEMA)
        problems = _diff_schema(expected, conn)
    finally:
        expected.close()
    version = read_meta(conn).get("schema_version")
    if version != str(SCHEMA_VERSION):
        problems.append(f"schema_version is {version!r}, expected {SCHEMA_VERSION}")
    return problems


def read_meta(conn: sqlite3.Connection) -> dict[str, str]:
    """Every meta value of a version 2 database, where evaluate reads the settings a run recorded at launch."""
    return {row[0]: row[1] for row in conn.execute("SELECT key, value FROM meta")}


class StockStore(Store):
    """The version 1 Store on the stock run tables, a new file created owner only with meta.kind set to stocks."""

    def __init__(self, path: Path) -> None:
        """Create a missing file mode 600 in a mode 700 folder before the version 1 Store opens it, so no other user
        can read stock data, and leave an existing file as it is."""
        path = Path(path)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
        except FileExistsError:
            pass
        super().__init__(path)

    @classmethod
    def open(cls, path: Path) -> StockStore:
        """Store.open, naming the file when it is refused, since one stock process opens two run databases."""
        try:
            return super().open(path)
        except StoreError as exc:
            raise StoreError(f"{path} is not a stock run database of schema version {SCHEMA_VERSION}, {exc}") from exc

    def create_schema(self) -> None:
        """Create every table with meta kind and schema_version in one transaction, only ever in an empty file."""
        try:
            self.conn.executescript(f"BEGIN IMMEDIATE;\n{SCHEMA}INSERT INTO meta (key, value) VALUES "
                                    f"('kind', '{KIND}'), ('schema_version', '{SCHEMA_VERSION}');\nCOMMIT;\n")
        except BaseException:
            self._rollback()
            raise

    def check_schema(self) -> list[str]:
        """The problems schema_problems finds, so Store.open refuses a news, version 1 or changed database."""
        return schema_problems(self.conn)

    def rollback_after(self, ts: str, labels_from: str) -> None:
        """Remove every row written after the checkpoint at ts in one transaction, keeping feed events, meta and the
        two status tables, which the next minute overwrites.

        labels_from is the boundary 30 session minutes before ts. A label is written pending at its minute and settled
        30 session minutes later, so every label and decision label after labels_from settled after ts and goes back
        to pending. The session rows and spread checks of every session still open at ts go too, and are written
        again with their final values once that session closes.
        """
        first_open = sessions.next_session(parse_iso(ts)).day.isoformat()
        self.commit_now([
            ("DELETE FROM decisions WHERE ts > ?", (ts,)),
            ("UPDATE decisions SET label_bps = NULL WHERE ts > ?", (labels_from,)),
            ("DELETE FROM stock_forecasts WHERE ts > ?", (ts,)),
            ("DELETE FROM stock_labels WHERE ts > ?", (ts,)),
            ("UPDATE stock_labels SET exit_price = NULL, label_bps = NULL, status = 'pending' WHERE ts > ?",
             (labels_from,)),
            ("DELETE FROM fills WHERE decided_at > ?", (ts,)),
            ("DELETE FROM equity_snapshots WHERE ts > ?", (ts,)),
            ("DELETE FROM stock_candles WHERE ts > ?", (ts,)),
            ("DELETE FROM benchmark WHERE ts > ?", (ts,)),
            ("DELETE FROM news_seen WHERE recv_at > ?", (ts,)),
            ("DELETE FROM news_minutes WHERE ts > ?", (ts,)),
            ("DELETE FROM spread_checks WHERE session_date >= ?", (first_open,)),
            ("DELETE FROM sessions WHERE session_date >= ?", (first_open,)),
            ("DELETE FROM deaths WHERE died_at > ?", (ts,)),
            ("DELETE FROM bots WHERE born_at > ?", (ts,)),
            ("UPDATE bots SET died_at = NULL WHERE died_at > ?", (ts,)),
            ("DELETE FROM trades WHERE opened_at > ?", (ts,)),
            ("UPDATE trades SET closed_at = NULL, exit_price = NULL, fee_close = NULL, slippage_bps_close = NULL, "
             "pnl_gross = NULL, pnl_net = NULL, exit_reason = NULL WHERE closed_at > ?", (ts,)),
            ("DELETE FROM checkpoints WHERE ts > ?", (ts,)),
        ])


def open_reader(path: Path | str) -> sqlite3.Connection:
    """A read only connection to a stock run database for evaluate, status, export and the dashboard, opened with
    mode=ro and never immutable so it sees every later commit, refusing a database of another kind or schema."""
    path = Path(path)
    if not path.is_file():
        raise StoreError(f"{path} does not exist")
    conn = _connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        problems = schema_problems(conn)
    except sqlite3.DatabaseError as exc:
        problems = [f"it cannot be read as a database, {exc}"]
    if problems:
        conn.close()
        raise StoreError(f"{path} is not a stock run database of schema version {SCHEMA_VERSION}\n  "
                         + "\n  ".join(problems))
    return conn


def init_db(config: StockConfig) -> bool:
    """What init-db does with a stock run config, creating its database or checking an existing one, and whether it
    created the database."""
    store = StockStore.open(config.db_path)
    created = store.created
    store.close()
    return created
