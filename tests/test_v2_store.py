"""The stock run database of PLAN_V2 section 9.2. init-db creates a run database with meta.kind stocks and a news
database with meta.kind news, StockStore has the plan's tables, checks them and rolls back after a checkpoint, the
row builders write every table, and every reader refuses a database of the wrong kind or a version 1 database.

Every database lives in a temporary folder, with fictional prices and the committed configs read as if they sat
in that folder, so no test opens a real run database or the real news.db."""

import json
import re
import sqlite3
import stat
from dataclasses import dataclass
from datetime import date
from typing import NamedTuple

import pytest

from btc_survivor.account import Account
from btc_survivor.clock import iso, parse_iso
from btc_survivor.store import SCHEMA as V1_SCHEMA
from btc_survivor.store import Store, StoreError
from btc_survivor.types import SHORT, Fill
from btc_survivor.v2 import redact, sessions
from btc_survivor.v2.config import parse_news_config, parse_stock_config
from btc_survivor.v2.news import db as news_db
from btc_survivor.v2.news.db import NewsDbError
from btc_survivor.v2.stocks import records, store
from btc_survivor.v2.stocks.candles import StockCandle
from btc_survivor.v2.stocks.health import HealthEvent
from btc_survivor.v2.stocks.store import StockStore
from helpers import ROOT

CONFIGS = ROOT / "configs" / "v2"
# The version 1 tables a run database keeps, each with the columns section 9.2 adds at its end.
GAINED = {"bots": ["home_symbol"], "decisions": ["symbol"], "trades": ["symbol"],
          "fills": ["symbol", "print_price", "allowance_bps", "notional", "sec_fee", "taf_fee"],
          "equity_snapshots": [], "deaths": [], "feed_events": ["symbol"], "checkpoints": []}
NEWS_META = {"config_hash": "a" * 64, "prompt_sha256": "b" * 64, "schema_sha256": "c" * 64,
             "haiku_model": "claude-haiku-5-5", "sonnet_model": "claude-sonnet-5-5"}
CANARY = "fake-canary-for-store-tests-7d1c"


@dataclass(frozen=True, slots=True)
class StockFill(Fill):
    """The fields of the broker's StockFill, the version 1 Fill plus the stock columns of fills."""

    symbol: str
    print_price: float | None
    allowance_bps: float
    notional: float | None
    sec_fee: float | None
    taf_fee: float | None


class LabelRow(NamedTuple):
    """The fields of the evaluation label rows labels.py emits, named after the stock_labels columns."""

    ts: int
    symbol: str
    entry_price: float
    exit_price: float | None
    label_bps: float | None
    status: str


def plan_block() -> str:
    """The SQL block of PLAN_V2 section 9.2, the run tables as the plan writes them."""
    text = (ROOT / "PLAN_V2.md").read_text(encoding="utf-8")
    section = text[text.index("### 9.2 A stock run database"):text.index("## 10. Runs and predictors")]
    return re.search(r"```sql\n(.*?)```", section, re.S).group(1)


def plan_meta_keys() -> list[str]:
    """The meta keys in the comment heading that block, kind = stocks counted as kind."""
    text = " ".join(line[2:].strip() for line in plan_block().splitlines() if line.startswith("--"))
    return [part.split("=")[0].strip() for part in text.removeprefix("meta keys:").split(",")]


def columns(conn, table):
    """Name, type, not null, default and key position of each column, in order."""
    return [tuple(row)[1:] for row in conn.execute(f"PRAGMA table_info({table})")]


def layout(conn):
    """Each table's columns and its indexes, unique constraints and key with their columns."""
    tables = {}
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"):
        indexes = sorted((index if origin == "c" else origin, unique, [r[2] for r in conn.execute(
            f"PRAGMA index_info('{index}')")]) for _, index, unique, origin, _ in conn.execute(
            f"PRAGMA index_list({name})"))
        tables[name] = (columns(conn, name), indexes)
    return tables


def stock_config(name, folder):
    """A committed stock run config read as if it sat two folders below folder, so its data lands in folder."""
    return parse_stock_config((CONFIGS / name).read_text(encoding="utf-8"), folder / "configs" / "v2")


def news_config(folder):
    return parse_news_config((CONFIGS / "news.toml").read_text(encoding="utf-8"), folder / "configs" / "v2")


def mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def fresh(tmp_path):
    return StockStore.open(tmp_path / "run" / "stocks.db")


def rows(st, sql):
    return [dict(row) for row in st.query(sql)]


def contents(path):
    """Every schema object and meta row of a database, to show a refused file was left as it was."""
    conn = sqlite3.connect(path)
    try:
        objects = sorted(conn.execute("SELECT type, name, sql FROM sqlite_master"))
        meta = sorted(conn.execute("SELECT key, value FROM meta")) if any(o[1] == "meta" for o in objects) else []
        return objects, meta
    finally:
        conn.close()


def v1_database(path):
    Store.open(path).close()
    return path


def news_database(path):
    news_db.open_writer(path).close()
    return path


def run_database(path):
    StockStore.open(path).close()
    return path


# init-db


@pytest.mark.parametrize("name", ["s1_prices.toml", "s2_news.toml"])
def test_init_db_creates_a_run_database_owner_only_with_kind_stocks_then_checks_it(tmp_path, name):
    config = stock_config(name, tmp_path)
    path = config.db_path
    assert path == tmp_path / "configs" / "v2" / f"../../data/v2/{name.removesuffix('.toml')}" / "stocks.db"
    assert store.init_db(config) is True
    assert mode(path) == 0o600 and mode(path.parent) == 0o700
    conn = store.open_reader(path)
    try:
        assert store.read_meta(conn) == {"kind": "stocks", "schema_version": str(store.SCHEMA_VERSION)}
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()
    assert store.init_db(config) is False


def test_init_db_creates_a_news_database_with_kind_news(tmp_path):
    config = news_config(tmp_path)
    assert news_db.init_db(config) is True
    conn = news_db.open_reader(config.db_path)
    try:
        assert news_db.get_value(conn, "meta", "kind") == "news"
        assert store.kind_problems(conn, "news") == []
        assert store.kind_problems(conn, "stocks") == ["its meta.kind is 'news', not 'stocks'"]
    finally:
        conn.close()
    assert news_db.init_db(config) is False


def test_init_db_refuses_a_run_path_holding_a_news_or_version_1_database(tmp_path):
    config = stock_config("s1_prices.toml", tmp_path)
    config.db_path.parent.mkdir(parents=True)
    news_database(config.db_path)
    before = contents(config.db_path)
    with pytest.raises(StoreError, match="its meta.kind is 'news', not 'stocks'"):
        store.init_db(config)
    assert contents(config.db_path) == before
    other = stock_config("s2_news.toml", tmp_path)
    other.db_path.parent.mkdir(parents=True)
    v1_database(other.db_path)
    with pytest.raises(StoreError, match="no meta.kind, so it is a version 1 database"):
        store.init_db(other)


# The schema


def test_the_new_tables_are_the_sql_block_of_plan_section_9_2(tmp_path):
    plan = sqlite3.connect(":memory:")
    plan.executescript(plan_block())
    st = fresh(tmp_path)
    try:
        want = layout(plan)
        have = layout(st.conn)
        assert len(want) == 10
        assert {name: have[name] for name in want} == want
        assert list(want) == list(store.TABLES[-10:])
    finally:
        st.close()
        plan.close()


def test_the_version_1_tables_keep_their_shape_with_only_the_columns_section_9_2_adds(tmp_path):
    v1 = sqlite3.connect(":memory:")
    v1.executescript(V1_SCHEMA)
    st = fresh(tmp_path)
    try:
        for table, gained in GAINED.items():
            old, new = columns(v1, table), columns(st.conn, table)
            assert new[:len(old)] == old, table
            assert [column[0] for column in new[len(old):]] == gained, table
        have = layout(st.conn)
        old = layout(v1)
        for table in ("decisions", "trades", "fills"):
            assert have[table][1] == old[table][1]
        assert set(have) == set(store.TABLES) == {"meta", *GAINED, *layout_names(plan_block())}
        assert not {"lineage", "candles", "forecasts"} & set(have)
        assert columns(st.conn, "meta") == columns(v1, "meta")
    finally:
        st.close()
        v1.close()


def layout_names(sql):
    conn = sqlite3.connect(":memory:")
    conn.executescript(sql)
    try:
        return set(layout(conn))
    finally:
        conn.close()


def test_a_new_run_database_passes_its_check_and_reopens_with_its_rows(tmp_path):
    st = fresh(tmp_path)
    assert st.created and st.check_schema() == []
    assert st.query("PRAGMA synchronous")[0][0] == 1
    st.queue(records.INSERT_FEED_EVENT, records.feed_event_row(parse_iso("2026-10-19T13:30:00.000Z"),
                                                               "session_open"))
    st.flush()
    st.close()
    st = fresh(tmp_path)
    assert not st.created and len(st.query("SELECT * FROM feed_events")) == 1
    st.close()


def changed_schema(path, old, new, extra=""):
    """A file with the run schema edited, meta.kind stocks and the current schema version."""
    sql = store.SCHEMA.replace(old, new)
    assert sql != store.SCHEMA
    conn = sqlite3.connect(path)
    conn.executescript(sql + extra)
    conn.execute("INSERT INTO meta VALUES ('kind', 'stocks'), ('schema_version', ?)", (str(store.SCHEMA_VERSION),))
    conn.commit()
    conn.close()
    return path


def test_the_check_names_a_missing_or_extra_column_and_an_extra_table(tmp_path):
    path = changed_schema(tmp_path / "changed.db", "notional REAL, sec_fee REAL, taf_fee REAL);",
                          "notional REAL, sec_fee REAL, note TEXT);",
                          "CREATE TABLE lineage (bot_id INTEGER PRIMARY KEY);")
    for opener in (StockStore.open, store.open_reader):
        with pytest.raises(StoreError) as info:
            opener(path)
        message = str(info.value)
        assert "table fills is missing column taf_fee" in message
        assert "table fills has unexpected column note" in message
        assert "unexpected table lineage" in message


def test_the_check_names_a_missing_table_index_or_key_change(tmp_path):
    path = changed_schema(tmp_path / "changed.db", "PRIMARY KEY (ts, symbol, predictor));",
                          "PRIMARY KEY (ts, symbol));")
    with pytest.raises(StoreError, match="table stock_forecasts column predictor differs"):
        StockStore.open(path)
    path = changed_schema(tmp_path / "dropped.db", "CREATE INDEX fills_bot ON fills (bot_id);", "")
    with pytest.raises(StoreError, match="missing index fills_bot"):
        StockStore.open(path)


def test_the_check_names_a_wrong_schema_version(tmp_path):
    path = run_database(tmp_path / "v.db")
    conn = sqlite3.connect(path)
    conn.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(StoreError, match="schema_version is '99', expected 1"):
        StockStore.open(path)
    with pytest.raises(StoreError, match="schema_version is '99', expected 1"):
        store.open_reader(path)


# Every reader refuses a database of the wrong kind


RUN_READERS = {
    "the stock process": lambda path: StockStore.open(path).close(),
    "evaluate": lambda path: store.open_reader(path).close(),
    "status": lambda path: store.open_reader(path).close(),
    "export": lambda path: store.open_reader(path).close(),
    "the dashboard": lambda path: store.open_reader(path).close(),
}
NEWS_READERS = {
    "the puller": lambda path: news_db.open_reader(path).close(),
    "evaluate": lambda path: news_db.open_reader(path).close(),
    "the dashboard": lambda path: news_db.open_reader(path).close(),
}
WRONG_FOR_A_RUN = {
    "news": (news_database, "its meta.kind is 'news', not 'stocks'"),
    "version 1": (v1_database, "it has no meta.kind, so it is a version 1 database or not a version 2 database"),
}
WRONG_FOR_NEWS = {
    "stocks": (run_database, "its meta.kind is 'stocks', not 'news'"),
    "version 1": (v1_database, "it has no meta.kind, so it is not a version 2 news database"),
}


@pytest.mark.parametrize("kind", WRONG_FOR_A_RUN)
@pytest.mark.parametrize("reader", RUN_READERS)
def test_every_reader_of_a_run_database_refuses_a_news_or_version_1_database(tmp_path, reader, kind):
    make, reason = WRONG_FOR_A_RUN[kind]
    path = make(tmp_path / "other.db")
    before = contents(path)
    with pytest.raises(StoreError) as info:
        RUN_READERS[reader](path)
    assert reason in str(info.value)
    assert contents(path) == before


@pytest.mark.parametrize("kind", WRONG_FOR_NEWS)
@pytest.mark.parametrize("reader", NEWS_READERS)
def test_every_reader_of_news_db_refuses_a_run_or_version_1_database(tmp_path, reader, kind):
    make, reason = WRONG_FOR_NEWS[kind]
    path = make(tmp_path / "other.db")
    before = contents(path)
    with pytest.raises(NewsDbError) as info:
        NEWS_READERS[reader](path)
    assert reason in str(info.value)
    assert contents(path) == before


def test_version_1_refuses_a_run_database_too(tmp_path):
    path = run_database(tmp_path / "stocks.db")
    with pytest.raises(StoreError, match="database does not match the expected schema"):
        Store.open(path)


def test_a_file_with_no_meta_or_no_database_or_nothing_at_all_is_refused(tmp_path):
    bare = tmp_path / "bare.db"
    conn = sqlite3.connect(bare)
    conn.execute("CREATE TABLE bots (bot_id INTEGER PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()
    for opener in (StockStore.open, store.open_reader):
        with pytest.raises(StoreError, match="it has no meta table, so it is not a version 2 database"):
            opener(bare)
    junk = tmp_path / "junk.db"
    junk.write_bytes(b"not a database at all, just some bytes " * 200)
    for opener in (StockStore.open, store.open_reader):
        with pytest.raises(StoreError):
            opener(junk)
    missing = tmp_path / "missing.db"
    with pytest.raises(StoreError, match="does not exist"):
        store.open_reader(missing)
    assert not missing.exists()


def test_the_reader_is_read_only_and_sees_later_commits(tmp_path):
    st = fresh(tmp_path)
    conn = store.open_reader(st.path)
    try:
        assert conn.execute("SELECT count(*) FROM feed_events").fetchone()[0] == 0
        st.execute_now(records.INSERT_FEED_EVENT, records.feed_event_row(parse_iso("2026-10-19T13:30:00.000Z"),
                                                                         "session_open"))
        assert conn.execute("SELECT count(*) FROM feed_events").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM feed_events")
    finally:
        conn.close()
        st.close()


# Launch meta


@pytest.mark.parametrize("name", ["s1_prices.toml", "s2_news.toml"])
def test_launch_meta_holds_the_keys_section_9_2_lists(tmp_path, name):
    config = stock_config(name, tmp_path)
    launch = parse_iso("2026-10-19T13:25:00.000Z")
    values = records.launch_meta(config, launch, NEWS_META, 4321, "0" * 40)
    keys = set(plan_meta_keys()) - {"kind", "schema_version"}
    s2 = config.evaluation is not None
    assert set(values) == (keys if s2 else keys - {"evaluation_json", "run_minute_rule"})
    assert values["run_name"] == config.run.name and values["master_seed"] == "12345"
    assert values["launch_ts"] == "2026-10-19T13:25:00.000Z" and values["config_toml"] == config.text
    assert values["experiment_hash"] == config.experiment_hash
    assert values["news_config_hash"] == NEWS_META["config_hash"]
    assert all(values[key] == NEWS_META[key] for key in ("prompt_sha256", "schema_sha256", "haiku_model",
                                                          "sonnet_model"))
    assert values["news_start_seq"] == "4321" and values["code_commit"] == "0" * 40
    assert json.loads(values["symbols_json"]) == list(config.stock_feed.symbols)
    if s2:
        assert json.loads(values["evaluation_json"])["final_healthy_session_minutes"] == 15600
        assert values["run_minute_rule"] == "news_healthy_and_any_symbol_healthy"
    st = fresh(tmp_path)
    for key, value in values.items():
        st.queue_meta(key, value)
    st.flush()
    assert st.check_schema() == [] and st.get_meta("kind") == "stocks"
    st.close()


def test_launch_meta_refuses_a_news_db_the_news_process_never_started_on(tmp_path):
    config = stock_config("s2_news.toml", tmp_path)
    with pytest.raises(ValueError, match="news.db meta has no haiku_model"):
        records.launch_meta(config, 0, {**NEWS_META, "haiku_model": ""}, 0, "0" * 40)


# Row builders


def ms(text):
    return parse_iso(text)


def candle(symbol, ts, valid=True):
    return StockCandle(symbol, ts, sessions.k_at(ts), 100.0, 101.0, 99.5, 100.5, 100.25, 1200.0, 700.0, 400.0, 30,
                       28, 1, valid)


def test_every_row_builder_writes_its_table(tmp_path, monkeypatch):
    monkeypatch.setattr(redact, "_values", set())
    redact.register(CANARY)
    t = ms("2026-10-19T14:00:00.000Z")
    later = t + 30 * 60_000
    acct = Account(10000.0)
    acct.open(SHORT, 2.0, 250.0, 0.03, t, trade_id=7, slippage_bps=3.0)
    fill = StockFill(bot_id=1, trade_id=7, leg="open", taker_side="sell", qty=2.0, ref_mid=250.1, decided_ms=t,
                     filled_ms=t + 1200, book_avg_price=None, price=249.92, levels=None, slippage_bps=3.0, fee=0.03,
                     latency_ms=1200, status="filled", reason="decision", symbol="TSLA", print_price=250.0,
                     allowance_bps=3.0, notional=499.84, sec_fee=0.02, taf_fee=0.01)
    rejected = StockFill(bot_id=1, trade_id=None, leg="open", taker_side="sell", qty=2.0, ref_mid=250.1,
                         decided_ms=t, filled_ms=None, book_avg_price=None, price=None, levels=None, slippage_bps=None,
                         fee=None, latency_ms=1200, status="rejected", reason="symbol_stale", symbol="TSLA",
                         print_price=None, allowance_bps=3.0, notional=None, sec_fee=None, taf_fee=None)
    decision = {"bot_id": 1, "ts": t, "position_before": "flat", "ref_mid": 250.1, "prediction_bps": -4.0,
                "required_edge_bps": 9.0, "cost_estimate_bps": 6.6, "margin_bps": 2.4, "gate_passed": False,
                "forced": True, "action": "open_short", "reason": "forced", "symbol": "TSLA"}
    session = sessions.session_on(date(2026, 11, 27))
    st = fresh(tmp_path)
    st.commit_now([
        (records.INSERT_BOT, records.bot_row(bot_id=1, slot=6, home_symbol="TSLA", born_ms=t, state="warmup",
                                             margin_sigma=0.25, lr=0.0001, seed=12345, origin="random")),
        (records.INSERT_DECISION, records.decision_row(decision)),
        (records.SET_LABEL, (12.5, 1, iso(t))),
        (records.INSERT_FILL, records.fill_row(fill)),
        (records.INSERT_FILL, records.fill_row(rejected)),
        (records.OPEN_TRADE, records.open_trade_row(1, "TSLA", acct)),
        (records.INSERT_FEED_EVENT, records.feed_event_row(*HealthEvent(t, "stale", "AAPL", "no eligible print"))),
        (records.INSERT_FEED_EVENT, records.feed_event_row(t, "disconnected", None, f"close reason {CANARY}")),
        (records.INSERT_FEED_EVENT, records.local_only_row(t, True, 0, 0)),
        (records.INSERT_CANDLE, records.candle_row(candle("AAPL", t))),
        (records.PUT_SESSION, records.session_row(session, 200, 195)),
        (records.PUT_SESSION, records.session_row(session, 210, 204)),
        records.label_statement(LabelRow(t, "AAPL", 100.25, None, None, "pending")),
        records.label_statement(LabelRow(t, "MSFT", 400.0, None, None, "pending")),
        records.label_statement(LabelRow(t, "AAPL", 100.25, 100.5, 24.9, "matured")),
        records.label_statement(LabelRow(t, "MSFT", 400.0, None, None, "gap")),
        records.label_statement(LabelRow(t, "MSFT", 400.0, 401.0, 25.0, "matured")),
        (records.INSERT_FORECAST, records.forecast_row(t, "AAPL", "news", 0.12, 3, t - 5000)),
        (records.INSERT_FORECAST, records.forecast_row(t, "AAPL", "population", 1.5)),
        (records.INSERT_NEWS_SEEN, records.news_seen_row(
            score_seq=11, item_seq=5, recv_ms=t - 5000, scored_at="2026-10-19T13:59:54.100Z", source="edgar",
            route="llm", scorer="haiku_v1", status="scored", tickers=["AAPL", "MACRO"], primary_use=True,
            fallback=False)),
        (records.INSERT_NEWS_MINUTE, records.news_minute_row(t, None, ["fed_speech", "bls"])),
        (records.INSERT_NEWS_MINUTE, records.news_minute_row(later, "heartbeat_old", "")),
        (records.PUT_SPREAD_CHECK, records.spread_check_row(date(2026, 10, 19), "META", 5000, 2.5, 3.0)),
        (records.PUT_SPREAD_CHECK, records.spread_check_row(date(2026, 10, 19), "META", 5000, 3.5, 3.0)),
        (records.PUT_SPREAD_CHECK, records.spread_check_row(date(2026, 10, 19), "QQQ", 9000, None, 1.5)),
        (records.INSERT_BENCHMARK, records.benchmark_row(
            t, ew_buy_hold=10001.0, qqq_buy_hold=10002.0, matched_equity=10000.5, living_mean_equity=9999.0,
            living_median_equity=9998.5, living=24, twin_mean_equity=None, cash_equity=10000.0)),
        (records.PUT_RUNTIME_STATUS, records.runtime_status_row(
            t, feed_state="healthy", session_state="open", prints_per_s=40.0, msgs_per_s=12.0, news_age_s=3.0,
            living=24, last_checkpoint_ms=None, rss_mb=120.0, rss_peak_mb=130.0)),
        (records.PUT_RUNTIME_STATUS, records.runtime_status_row(
            later, feed_state="stale", session_state="open", prints_per_s=None, msgs_per_s=None, news_age_s=None,
            living=23, last_checkpoint_ms=t, rss_mb=121.0, rss_peak_mb=131.0)),
        (records.PUT_SYMBOL_STATUS, records.symbol_status_row("AAPL", t, "healthy", 100.25, 0.4)),
        (records.PUT_SYMBOL_STATUS, records.symbol_status_row("AAPL", later, "stale", None, 16.0)),
    ])
    stamp = iso(t)
    assert rows(st, "SELECT * FROM bots") == [{
        "bot_id": 1, "slot": 6, "parent_id": None, "generation": 0, "born_at": stamp, "died_at": None,
        "state": "warmup", "margin_bps": 0.25, "lr": 0.0001, "seed": 12345, "origin": "random", "home_symbol": "TSLA"}]
    assert rows(st, "SELECT symbol, ts, gate_passed, forced, label_bps FROM decisions") == [
        {"symbol": "TSLA", "ts": stamp, "gate_passed": 0, "forced": 1, "label_bps": 12.5}]
    assert rows(st, "SELECT decided_at, filled_at, book_avg_price, levels_consumed, price, status, symbol, "
                    "print_price, allowance_bps, notional, sec_fee, taf_fee FROM fills ORDER BY id") == [
        {"decided_at": stamp, "filled_at": iso(t + 1200), "book_avg_price": None, "levels_consumed": None,
         "price": 249.92, "status": "filled", "symbol": "TSLA", "print_price": 250.0, "allowance_bps": 3.0,
         "notional": 499.84, "sec_fee": 0.02, "taf_fee": 0.01},
        {"decided_at": stamp, "filled_at": None, "book_avg_price": None, "levels_consumed": None, "price": None,
         "status": "rejected", "symbol": "TSLA", "print_price": None, "allowance_bps": 3.0, "notional": None,
         "sec_fee": None, "taf_fee": None}]
    with pytest.raises(ValueError, match="not 'open'"):
        records.label_statement(LabelRow(t, "AAPL", 100.25, None, None, "open"))
    assert rows(st, "SELECT id, side, qty, symbol, exit_reason FROM trades") == [
        {"id": 7, "side": "short", "qty": 2.0, "symbol": "TSLA", "exit_reason": None}]
    events = rows(st, "SELECT kind, symbol, detail FROM feed_events ORDER BY id")
    assert events[:2] == [{"kind": "stale", "symbol": "AAPL", "detail": "no eligible print"},
                          {"kind": "disconnected", "symbol": None, "detail": f"close reason {redact.MASK}"}]
    assert events[2]["kind"] == "local_only" and events[2]["symbol"] is None
    assert json.loads(events[2]["detail"]) == {"passed": True, "offending_sockets": 0, "tunnel_processes": 0}
    assert rows(st, "SELECT symbol, ts, session_date, session_k, ref_price, valid FROM stock_candles") == [
        {"symbol": "AAPL", "ts": stamp, "session_date": "2026-10-19", "session_k": sessions.k_at(t),
         "ref_price": 100.25, "valid": 1}]
    assert rows(st, "SELECT * FROM sessions") == [
        {"session_date": "2026-11-27", "open_ts": "2026-11-27T14:30:00.000Z", "close_ts": "2026-11-27T18:00:00.000Z",
         "early_close": 1, "healthy_minutes": 210, "run_minutes": 204}]
    assert rows(st, "SELECT * FROM stock_labels ORDER BY symbol") == [
        {"ts": stamp, "symbol": "AAPL", "entry_price": 100.25, "exit_price": 100.5, "label_bps": 24.9,
         "status": "matured"},
        {"ts": stamp, "symbol": "MSFT", "entry_price": 400.0, "exit_price": None, "label_bps": None, "status": "gap"}]
    assert rows(st, "SELECT * FROM stock_forecasts ORDER BY predictor") == [
        {"ts": stamp, "symbol": "AAPL", "predictor": "news", "prediction": 0.12, "items": 3,
         "inputs_max_recv_at": iso(t - 5000)},
        {"ts": stamp, "symbol": "AAPL", "predictor": "population", "prediction": 1.5, "items": None,
         "inputs_max_recv_at": None}]
    assert rows(st, "SELECT * FROM news_seen") == [
        {"score_seq": 11, "item_seq": 5, "recv_at": iso(t - 5000), "scored_at": "2026-10-19T13:59:54.100Z",
         "source": "edgar", "route": "llm", "scorer": "haiku_v1", "status": "scored", "tickers": "AAPL,MACRO",
         "primary_use": 1, "fallback": 0}]
    assert rows(st, "SELECT * FROM news_minutes ORDER BY ts") == [
        {"ts": stamp, "healthy": 1, "reason": None, "channels_down": "fed_speech,bls"},
        {"ts": iso(later), "healthy": 0, "reason": "heartbeat_old", "channels_down": ""}]
    assert rows(st, "SELECT symbol, roll_half_spread_bps, flagged FROM spread_checks ORDER BY symbol") == [
        {"symbol": "META", "roll_half_spread_bps": 3.5, "flagged": 1},
        {"symbol": "QQQ", "roll_half_spread_bps": None, "flagged": 0}]
    assert rows(st, "SELECT ts, ew_buy_hold, living, twin_mean_equity FROM benchmark") == [
        {"ts": stamp, "ew_buy_hold": 10001.0, "living": 24, "twin_mean_equity": None}]
    assert rows(st, "SELECT id, ts, feed_state, living, last_checkpoint_ts FROM runtime_status") == [
        {"id": 1, "ts": iso(later), "feed_state": "stale", "living": 23, "last_checkpoint_ts": stamp}]
    assert rows(st, "SELECT * FROM symbol_status") == [
        {"symbol": "AAPL", "ts": iso(later), "state": "stale", "ref_price": None, "last_print_age_s": 16.0}]
    st.close()


# rollback_after


OLD_DAY = "2026-10-16T19:00:00.000Z"
OLD, LABELS_FROM, MID, CUT, NEW = (f"2026-10-19T{hm}.000Z" for hm in ("14:00:00", "14:30:00", "14:45:00",
                                                                        "15:00:30", "15:30:00"))
NEXT_DAY = "2026-10-20T14:00:00.000Z"


def history(st):
    """Rows on both sides of the checkpoint at CUT in every table, written through the row builders."""
    acct_old, acct_new = Account(10000.0), Account(10000.0)
    acct_old.open(SHORT, 2.0, 250.0, 0.03, ms(OLD), trade_id=1)
    acct_new.open(SHORT, 2.0, 250.0, 0.03, ms(NEW), trade_id=2)
    acct_kept = Account(10000.0)
    acct_kept.open(SHORT, 2.0, 250.0, 0.03, ms(OLD), trade_id=3)

    def bot(bot_id, born):
        return records.INSERT_BOT, records.bot_row(bot_id=bot_id, slot=bot_id, home_symbol="TSLA", born_ms=ms(born),
                                                   state="active", margin_sigma=0.25, lr=0.0001, seed=12345,
                                                   origin="random")

    out = [bot(1, OLD), (records.SET_BOT_DEAD, (NEW, 1)), bot(2, NEW), bot(3, OLD), (records.SET_BOT_DEAD, (OLD, 3)),
           (records.OPEN_TRADE, records.open_trade_row(1, "TSLA", acct_old)),
           (records.CLOSE_TRADE, (NEW, 249.0, 0.0, 0.0, 0.0, 2.0, 1.97, "session_close", 1)),
           (records.OPEN_TRADE, records.open_trade_row(1, "TSLA", acct_new)),
           (records.OPEN_TRADE, records.open_trade_row(3, "TSLA", acct_kept)),
           (records.CLOSE_TRADE, (OLD, 249.0, 0.0, 0.0, 0.0, 2.0, 1.97, "session_flatten", 3)),
           (records.INSERT_DEATH, (1, NEW, 0, "rent", 1, 0, 0, 0, 0, 0, 1, 1, 1, 1.0)),
           (records.INSERT_DEATH, (3, OLD, 0, "rent", 1, 0, 0, 0, 0, 0, 1, 1, 1, 1.0))]
    for seq, (ts, label) in enumerate(((OLD, 1.0), (MID, 2.0), (NEW, 3.0)), start=1):
        t = ms(ts)
        fill = StockFill(1, None, "open", "sell", 2.0, 250.0, t, t + 1000, None, 250.0, None, 3.0, 0.03, 1000,
                         "filled", "decision", "TSLA", 250.0, 3.0, 500.0, 0.02, 0.01)
        decision = {"bot_id": 1, "ts": t, "position_before": "flat", "ref_mid": 250.0, "prediction_bps": 0.0,
                    "required_edge_bps": 9.0, "cost_estimate_bps": 6.6, "margin_bps": 2.4, "gate_passed": 0,
                    "forced": 0, "action": "none", "reason": "gate", "symbol": "TSLA"}
        out += [(records.INSERT_DECISION, records.decision_row(decision)), (records.SET_LABEL, (label, 1, ts)),
                (records.INSERT_FILL, records.fill_row(fill)),
                (records.INSERT_SNAPSHOT, (1, ts, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "flat", None, None, None, 0)),
                (records.INSERT_CANDLE, records.candle_row(candle("AAPL", t))),
                (records.INSERT_BENCHMARK, records.benchmark_row(
                    t, ew_buy_hold=1.0, qqq_buy_hold=1.0, matched_equity=1.0, living_mean_equity=1.0,
                    living_median_equity=1.0, living=24, twin_mean_equity=None, cash_equity=1.0)),
                (records.INSERT_CHECKPOINT, (ts, 1, "{}", b"")),
                (records.INSERT_FEED_EVENT, records.feed_event_row(t, "stale", "AAPL")),
                (records.INSERT_FORECAST, records.forecast_row(t, "AAPL", "population", 1.0)),
                (records.INSERT_NEWS_SEEN, records.news_seen_row(
                    score_seq=seq, item_seq=seq, recv_ms=t, scored_at=ts, source="finnhub", route="llm",
                    scorer="haiku_v1", status="scored", tickers="AAPL", primary_use=True, fallback=False)),
                (records.INSERT_NEWS_MINUTE, records.news_minute_row(t, None, ""))]
    for hm, settled in (("14:00", True), ("14:30", True), ("15:00", True), ("15:30", False)):
        t = ms(f"2026-10-19T{hm}:00.000Z")
        out.append(records.label_statement(LabelRow(t, "AAPL", 100.0, None, None, "pending")))
        if settled:
            out.append(records.label_statement(LabelRow(t, "AAPL", 100.0, 100.5, 49.9, "matured")))
    for day in (date(2026, 10, 16), date(2026, 10, 19), date(2026, 10, 20)):
        out += [(records.PUT_SESSION, records.session_row(sessions.session_on(day), 390, 390)),
                (records.PUT_SPREAD_CHECK, records.spread_check_row(day, "META", 5000, 2.5, 3.0))]
    out += [(records.PUT_RUNTIME_STATUS, records.runtime_status_row(
                ms(NEXT_DAY), feed_state="healthy", session_state="open", prints_per_s=1.0, msgs_per_s=1.0,
                news_age_s=1.0, living=24, last_checkpoint_ms=ms(NEW), rss_mb=1.0, rss_peak_mb=1.0)),
            (records.PUT_SYMBOL_STATUS, records.symbol_status_row("AAPL", ms(NEXT_DAY), "healthy", 100.0, 0.5)),
            (records.INSERT_CANDLE, records.candle_row(candle("AAPL", ms(OLD_DAY))))]
    st.commit_now(out)
    st.queue_meta("run_name", "s2_news")
    st.flush()


def test_rollback_after_removes_or_reopens_everything_newer_and_keeps_the_rest(tmp_path):
    st = fresh(tmp_path)
    history(st)
    st.rollback_after(CUT, LABELS_FROM)

    def col(sql):
        return [tuple(r) for r in st.query(sql)]

    assert col("SELECT bot_id, died_at FROM bots ORDER BY bot_id") == [(1, None), (3, OLD)]
    assert col("SELECT bot_id FROM deaths") == [(3,)]
    assert col("SELECT ts, label_bps FROM decisions ORDER BY ts") == [(OLD, 1.0), (MID, None)]
    assert col("SELECT id, closed_at, pnl_net, exit_reason FROM trades ORDER BY id") == [
        (1, None, None, None), (3, OLD, 1.97, "session_flatten")]
    for table, column in (("fills", "decided_at"), ("equity_snapshots", "ts"), ("benchmark", "ts"),
                          ("checkpoints", "ts"), ("stock_forecasts", "ts"), ("news_seen", "recv_at"),
                          ("news_minutes", "ts")):
        assert col(f"SELECT {column} FROM {table} ORDER BY {column}") == [(OLD,), (MID,)], table
    assert col("SELECT ts FROM stock_candles ORDER BY ts") == [(OLD_DAY,), (OLD,), (MID,)]
    assert col("SELECT ts, exit_price, label_bps, status FROM stock_labels ORDER BY ts") == [
        ("2026-10-19T14:00:00.000Z", 100.5, 49.9, "matured"), ("2026-10-19T14:30:00.000Z", 100.5, 49.9, "matured"),
        ("2026-10-19T15:00:00.000Z", None, None, "pending")]
    assert col("SELECT session_date FROM sessions") == [("2026-10-16",)]
    assert col("SELECT session_date FROM spread_checks") == [("2026-10-16",)]
    assert len(col("SELECT ts FROM feed_events")) == 3
    assert col("SELECT ts FROM runtime_status") == [(NEXT_DAY,)] and len(col("SELECT * FROM symbol_status")) == 1
    assert st.get_meta("run_name") == "s2_news" and st.check_schema() == []
    st.close()


def test_rollback_to_a_checkpoint_after_the_close_keeps_that_session(tmp_path):
    st = fresh(tmp_path)
    history(st)
    st.rollback_after("2026-10-19T22:00:30.000Z", "2026-10-19T19:30:00.000Z")
    assert [tuple(r) for r in st.query("SELECT session_date FROM sessions ORDER BY session_date")] == [
        ("2026-10-16",), ("2026-10-19",)]
    assert [tuple(r) for r in st.query("SELECT session_date FROM spread_checks ORDER BY session_date")] == [
        ("2026-10-16",), ("2026-10-19",)]
    assert len(st.query("SELECT * FROM stock_labels WHERE status = 'pending'")) == 1
    st.close()


def test_a_failed_rollback_changes_nothing(tmp_path):
    st = fresh(tmp_path)
    history(st)
    before = [tuple(r) for r in st.query("SELECT * FROM decisions ORDER BY id")]
    st.conn.execute("CREATE TRIGGER refuse BEFORE DELETE ON checkpoints BEGIN SELECT RAISE(ABORT, 'refused'); END")
    with pytest.raises(sqlite3.IntegrityError, match="refused"):
        st.rollback_after(CUT, LABELS_FROM)
    assert [tuple(r) for r in st.query("SELECT * FROM decisions ORDER BY id")] == before
    assert not st.conn.in_transaction
    st.close()
