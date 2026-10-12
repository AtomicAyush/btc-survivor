"""The evaluate command of PLAN_V2 sections 11 and 12 on seeded synthetic runs.

Each test builds or copies a prices run and a news run with news.db, a key ledger and tape check records in a temporary
folder, from fictional prices and the committed configs read as if they sat in that folder, so no test opens a real run
database, the real news.db or the real key ledger. The runs launch on Monday 19 October 2026, record 7 sessions and
reach their cutoff after 6 sessions of run minutes, so the window holds sessions 2 to 6. Every row of the news
predictor in the window is built with a correlation of exactly 0.2 with its label within each symbol, or exactly 0."""

import dataclasses
import json
import math
import re
import shutil
import sqlite3
from datetime import date
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from btc_survivor.clock import MS_PER_MINUTE, MS_PER_SECOND, iso, parse_iso
from btc_survivor.v2 import evaluate, sessions, stats
from btc_survivor.v2.config import HAIKU_MODEL, SONNET_MODEL, EvaluationConfig, load_run_pair
from btc_survivor.v2.evaluate import EvaluateError, Run, Window
from btc_survivor.v2.news import db as news_db
from btc_survivor.v2.news import ledger
from btc_survivor.v2.stocks import costs, records, store
from btc_survivor.v2.stocks.candles import StockCandle
from btc_survivor.v2.stocks.labels import PENDING, recompute_labels
from btc_survivor.v2.stocks.store import StockStore
from helpers import ROOT
from test_v2_store import contents, v1_database

CONFIGS = ROOT / "configs" / "v2"
SYMBOLS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "QQQ")
LAUNCH = "2026-10-19T13:25:00.000Z"
LAUNCH_MS = parse_iso(LAUNCH)
RECORDED = 7
FINAL = 6 * 390
SESSIONS = tuple(evaluate.nth_session(LAUNCH_MS, n) for n in range(1, RECORDED + 1))
MINUTES = tuple(s.boundary(m) for s in SESSIONS for m in range(1, s.minutes + 1))
START, END = iso(SESSIONS[1].open_ms), iso(SESSIONS[5].close_ms)
EVAL_MS = tuple(s.boundary(m) for s in SESSIONS for m in s.evaluation_minutes())
NEWS_META = {"config_hash": "a" * 64, "prompt_sha256": "b" * 64, "schema_sha256": "c" * 64,
             "haiku_model": HAIKU_MODEL, "sonnet_model": SONNET_MODEL}
COMMIT = "0123456789abcdef0123456789abcdef01234567"
TWIN_BASE = 100
NOTIONAL = 500.0
SCORES = ("news_company", "news_macro", "finbert", "haiku", "finbert_matched", "sonnet", "haiku_official")


class Built(NamedTuple):
    folder: Path

    @property
    def s1(self) -> Path:
        return self.folder / "configs" / "v2" / "s1_prices.toml"

    @property
    def s2(self) -> Path:
        return self.folder / "configs" / "v2" / "s2_news.toml"

    @property
    def s1_db(self) -> Path:
        return self.folder / "data" / "v2" / "s1_prices" / "stocks.db"

    @property
    def s2_db(self) -> Path:
        return self.folder / "data" / "v2" / "s2_news" / "stocks.db"

    @property
    def news_db(self) -> Path:
        return self.folder / "data" / "v2" / "news" / "news.db"

    @property
    def ledger(self) -> Path:
        return self.folder / "ledger.db"

    @property
    def checks(self) -> Path:
        return self.folder / "data" / "v2" / "checks"


# The synthetic runs


def in_window(ms):
    return START < iso(ms) <= END


def present(predictor, i, j):
    """Which rows a predictor has at evaluation minute i of symbol j, so each model pair overlaps only in part."""
    return {"haiku": (i + j) % 5 != 0, "finbert_matched": (i + j) % 7 != 0, "sonnet": (i + j) % 3 == 0,
            "haiku_official": (i + j) % 2 == 0}.get(predictor, True)


def make_candles(rng):
    """Every symbol minute of the recorded sessions, healthy, on a seeded random walk."""
    out = {}
    for j, symbol in enumerate(SYMBOLS):
        prices = 100.0 * (1 + j) * np.exp(np.cumsum(rng.normal(0, 5e-4, len(MINUTES))))
        out[symbol] = [StockCandle(symbol, ms, sessions.k_at(ms), float(p), float(p), float(p), float(p), float(p),
                                   10.0, 6.0, 4.0, 3, 3, 0, True) for ms, p in zip(MINUTES, prices)]
    return out


def planted(rng, labels, corr):
    """Scores whose correlation with labels is exactly corr, by Gram Schmidt against the standardized labels."""
    z = (labels - labels.mean()) / labels.std()
    e = rng.normal(size=len(z))
    e -= e.mean()
    e -= (e @ z) / (z @ z) * z
    e /= e.std()
    return 0.1 * (corr * z + math.sqrt(1 - corr * corr) * e)


def make_forecasts(rng, labels, news_corr):
    """The forecast rows of both runs, (ts, symbol, predictor, prediction, items, inputs_max_recv_ms)."""
    s1, s2 = [], []
    news = {}
    for symbol in SYMBOLS:
        rows = [row for row in labels[symbol] if in_window(row.ts)]
        values = planted(rng, np.array([row.label_bps for row in rows]), news_corr)
        news.update({(row.ts, symbol): float(v) for row, v in zip(rows, values)})
    for j, symbol in enumerate(SYMBOLS):
        for row in labels[symbol]:
            i, ts = EVAL_MS.index(row.ts), row.ts
            population, linear = float(rng.normal(0, 2)), float(rng.normal(0, 2))
            later = population + (float(rng.normal(0, 0.5)) if ts > SESSIONS[0].close_ms else 0.0)
            s1 += [(ts, symbol, "population", population, None, None),
                   (ts, symbol, "linear_prices", linear, None, None)]
            s2 += [(ts, symbol, "population", later, None, None), (ts, symbol, "linear_prices", linear, None, None),
                   (ts, symbol, "linear_news", float(rng.normal(0, 2)), None, None),
                   (ts, symbol, "news", news.get((ts, symbol), float(rng.normal(0, 0.1))), 2, ts - MS_PER_MINUTE)]
            s2 += [(ts, symbol, p, float(rng.normal(0, 0.1)), 1, ts - MS_PER_MINUTE) for p in SCORES
                   if present(p, i, j)]
    return s1, s2


def news_lines(i, ts):
    """The score lines S2 read before evaluation minute i: Haiku on AAPL at even minutes, the primary line, FinBERT
    on MSFT, Sonnet on NVDA, and an AAPL line too old to make an event at odd minutes."""
    lines = [(ts - 9 * MS_PER_MINUTE, "finbert_v1", "MSFT", 0), (ts - 5 * MS_PER_MINUTE, "sonnet_v1", "NVDA", 0)]
    lines.append((ts - (10 if i % 2 == 0 else 40) * MS_PER_MINUTE, "haiku_v1", "AAPL", 1))
    return lines


def trading(rng, cfg):
    """Eight learners and their twins. Each learner opens a forced long at session minutes 90 and 270 plus its slot
    in sessions 3 to 7, closed 30 minutes later, and its twin copies it, with an equity snapshot at every close."""
    statements = []
    rpm = costs.rent_per_minute(cfg.stock_costs)
    trade_id = 0
    for slot, symbol in enumerate(SYMBOLS):
        learner, twin = slot + 1, TWIN_BASE + slot + 1
        statements += [(records.INSERT_BOT, records.bot_row(
            bot_id=bot_id, slot=slot, home_symbol=symbol, born_ms=LAUNCH_MS, state="active", margin_sigma=0.25,
            lr=1e-4, seed=12345, origin=origin)) for bot_id, origin in ((learner, "random"), (twin, "control"))]
        allowance = costs.allowance_bps(cfg.stock_costs, symbol)
        closed = []
        for session in SESSIONS[2:]:
            for minute in (m for m in (90 + slot, 270 + slot) if m + 30 < session.minutes):
                decided, close = session.boundary(minute), session.boundary(minute + 30)
                statements.append((records.INSERT_DECISION, (
                    learner, iso(decided), "flat", 100.0, 0.5, 15.0, 4.6, 10.0, 0, 1, "open_long", "forced", symbol)))
                for bot_id, side in ((learner, "long"), (twin, "long" if rng.random() < 0.5 else "short")):
                    trade_id += 1
                    move = float(rng.normal(0, 0.003))
                    sign = 1 if side == "long" else -1
                    qty = NOTIONAL / 100.0
                    entry, out_print = 100.0 * (1 + sign * allowance / 1e4), 100.0 * (1 + move)
                    exit_price = out_print * (1 - sign * allowance / 1e4)
                    gross = sign * (exit_price - entry) * qty
                    statements += [
                        (records.INSERT_FILL, (bot_id, trade_id, iso(decided), iso(decided + 1200), "open",
                                               "buy" if sign > 0 else "sell", qty, 100.0, None, entry, None, allowance,
                                               0.0, 1200, "filled", "decision", symbol, 100.0, allowance,
                                               qty * entry, 0.0, 0.0)),
                        (records.INSERT_FILL, (bot_id, trade_id, iso(close), iso(close + 1200), "close",
                                               "sell" if sign > 0 else "buy", qty, 100.0, None, exit_price, None,
                                               allowance, 0.03, 1200, "filled", "decision", symbol, out_print,
                                               allowance, qty * exit_price, 0.02, 0.01)),
                        ("INSERT INTO trades (id, bot_id, side, opened_at, closed_at, qty, entry_price, exit_price, "
                         "fee_open, fee_close, funding, slippage_bps_open, slippage_bps_close, pnl_gross, pnl_net, "
                         "forced, exit_reason, symbol) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         (trade_id, bot_id, side, iso(decided + 1200), iso(close + 1200), qty, entry, exit_price,
                          0.0, 0.03, 0.0, allowance, allowance, gross, gross - 0.03, 1, "signal", symbol))]
                    if bot_id == learner:
                        closed.append((close + 1200, gross))
        for session in SESSIONS:
            done = [gross for ms, gross in closed if ms <= session.close_ms]
            minutes = sum(1 for ms in MINUTES if ms <= session.close_ms)
            rent, fees = rpm * minutes, 0.03 * len(done)
            equity = cfg.trading.start_equity + sum(done) - fees - rent
            statements.append((records.INSERT_SNAPSHOT, (learner, iso(session.close_ms), equity, equity, 0.0,
                                                         sum(done), fees, 0.0, rent, 0.0, "flat", None, None, None, 0)))
    return statements


def write_configs(folder):
    target = folder / "configs" / "v2"
    target.mkdir(parents=True)
    (target / "s1_prices.toml").write_text((CONFIGS / "s1_prices.toml").read_text(encoding="utf-8"), encoding="utf-8")
    s2 = (CONFIGS / "s2_news.toml").read_text(encoding="utf-8")
    (target / "s2_news.toml").write_text(s2.replace("final_healthy_session_minutes = 15600",
                                                    f"final_healthy_session_minutes = {FINAL}"), encoding="utf-8")


def write_news(built):
    """news.db with one Haiku and one Sonnet call, healthy heartbeats and a few scored items, all with the launch
    hashes and models, the key ledger with October spend and the consolidated tape check before launch."""
    conn = news_db.open_writer(built.news_db)
    call = {"ledger_id": 1, "kind": "single", "status": "ok", "n_items": 1, "sdk_version": "0",
            "request_sha256": "d" * 64, "effort": "low", "thinking": "disabled", "max_tokens": 1024,
            "reserved_usd": 0.01, "price_table": "{}",
            "prompt_sha256": NEWS_META["prompt_sha256"], "schema_sha256": NEWS_META["schema_sha256"]}
    news_db.insert(conn, "llm_calls", {**call, "scorer": "haiku_v1", "model": HAIKU_MODEL,
                                       "response_model": HAIKU_MODEL, "started_at": iso(SESSIONS[1].boundary(5))})
    news_db.insert(conn, "llm_calls", {**call, "scorer": "sonnet_v1", "model": SONNET_MODEL,
                                       "response_model": SONNET_MODEL, "started_at": iso(SESSIONS[2].boundary(5))})
    for session in SESSIONS:
        news_db.insert(conn, "news_heartbeat", {
            "ts": iso(session.boundary(1)), "healthy": 1, "config_hash": NEWS_META["config_hash"],
            "prompt_sha256": NEWS_META["prompt_sha256"], "schema_sha256": NEWS_META["schema_sha256"],
            "haiku_state": "on", "sonnet_state": "on", "finbert_state": "on", "waiting": 0, "channels_down": "",
            "spent_day_usd": 0.1, "spent_month_usd": 1.0, "spend_alert": 0})
        received = session.boundary(20)
        seq = news_db.insert(conn, "news_items", {
            "source": "finnhub", "channel": "finnhub_general", "vendor_id": str(received), "kind": "media",
            "received_at": iso(received), "title": "a fictional headline", "tickers": "AAPL", "relevance": "title",
            "status": "new", "title_hash": f"{received:016x}", "route": "llm"})
        for scorer, delay in (("haiku_v1", 4), ("finbert_v1", 2)):
            news_db.insert(conn, "item_scores", {"item_seq": seq, "scorer": scorer, "route": "llm", "status": "scored",
                                                 "attempts": 1, "scored_at": iso(received + delay * MS_PER_SECOND)})
    conn.close()
    book = sqlite3.connect(built.ledger)
    book.executescript(ledger.SCHEMA)
    book.executemany("INSERT INTO reservations (ts, process, model, ref, n_items, reserved_usd, cost_usd, cost_basis, "
                     "status) VALUES (?, 'p', ?, 'r', 1, ?, ?, ?, ?)",
                     [(iso(SESSIONS[1].boundary(5)), HAIKU_MODEL, 0.5, 0.25, "usage", "done"),
                      (iso(SESSIONS[2].boundary(5)), SONNET_MODEL, 0.5, None, None, "started")])
    book.commit()
    book.close()
    built.checks.mkdir(parents=True)
    (built.checks / "tape_2026_10_12.json").write_text(json.dumps(
        {"session": "2026-10-12", "clean": True, "ratio_verdict": "consolidated", "verdict": "consolidated"}))


def build(folder, news_corr=0.2, seed=11):
    """Both runs with identical candles, labels, linear_prices rows and trades, S2 with every news predictor."""
    built = Built(folder)
    write_configs(folder)
    configs = load_run_pair(built.s1, built.s2)
    rng = np.random.default_rng(seed)
    candles = make_candles(rng)
    labels = {symbol: recompute_labels(candles[symbol], 30) for symbol in SYMBOLS}
    forecasts = make_forecasts(rng, labels, news_corr)
    shared = [(records.INSERT_CANDLE, records.candle_row(c)) for symbol in SYMBOLS for c in candles[symbol]]
    for symbol in SYMBOLS:
        for row in labels[symbol]:
            shared += [records.label_statement(dataclasses.replace(row, exit_price=None, label_bps=None,
                                                                   status=PENDING)), records.label_statement(row)]
    shared += [(records.PUT_SESSION, records.session_row(s, 390, 390)) for s in SESSIONS]
    shared += [(records.INSERT_FEED_EVENT, records.local_only_row(s.open_ms - 10 * MS_PER_MINUTE, True, 0, 0))
               for s in SESSIONS]
    shared += trading(rng, configs[0])
    for config, rows in zip(configs, forecasts):
        st = StockStore.open(config.db_path)
        for key, value in records.launch_meta(config, LAUNCH_MS, NEWS_META, 1, COMMIT).items():
            st.queue_meta(key, value)
        for sql, params in shared:
            st.queue(sql, params)
        for ts, symbol, predictor, prediction, items, recv in rows:
            st.queue(records.INSERT_FORECAST, records.forecast_row(ts, symbol, predictor, prediction, items, recv))
        if config.evaluation is not None:
            outage = {SESSIONS[3].boundary(m) for m in SESSIONS[3].evaluation_minutes()}
            for ms in MINUTES:
                st.queue(records.INSERT_NEWS_MINUTE, records.news_minute_row(
                    ms, None, "fed_speeches" if ms in outage else ""))
            seq = 0
            for i, ts in enumerate(EVAL_MS):
                for recv, scorer, ticker, primary in news_lines(i, ts):
                    seq += 1
                    st.queue(records.INSERT_NEWS_SEEN, records.news_seen_row(
                        score_seq=seq, item_seq=seq, recv_ms=recv, scored_at=iso(recv - 2000), source="finnhub",
                        route="llm", scorer=scorer, status="scored", tickers=ticker, primary_use=bool(primary),
                        fallback=False))
        st.flush()
        st.close()
    write_news(built)
    return built


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return build(tmp_path_factory.mktemp("base"))


@pytest.fixture
def copy(base, tmp_path):
    shutil.copytree(base.folder, tmp_path / "copy")
    return Built(tmp_path / "copy")


@pytest.fixture(autouse=True)
def no_real_ledger(monkeypatch, tmp_path):
    """Point the default ledger into the test folder, so no test can ever open the real key ledger."""
    monkeypatch.setattr(evaluate, "DEFAULT_LEDGER", tmp_path / "absent" / "llm_ledger.db")


def run_eval(built):
    return evaluate.evaluate_runs(built.s1, built.s2, ledger=built.ledger)


def execute(path, sql, params=()):
    conn = sqlite3.connect(path)
    with conn:
        conn.execute(sql, params)
    conn.close()


def failing(ev):
    return {check.name for check in ev.checks if not check.passed}


# The base runs


def test_the_synthetic_runs_are_valid_and_every_outcome_matches(base):
    ev = run_eval(base)
    assert ev.window == Window(LAUNCH, START, END, True, FINAL)
    assert failing(ev) == set() and ev.valid and ev.configs_match
    assert ev.labels_compared == 2 * 8 * 12 * 6
    assert {(o.run, o.name): o.matched for o in ev.outcomes} == {
        **{(run, name): True for run in evaluate.RUNS for name in (
            "first trades", "rhythm", "burn", "founder equity", "no deaths", "flat at every close", "gate")},
        ("both runs", "pairing"): True}
    assert ev.twin_pairs == 2 * 8 * 2 * 4 and ev.deaths == ()
    assert ev.fallback_share == 0.0 and ev.spend_alert_days == 0


# Confirmatory test


def test_a_planted_correlation_of_0_2_passes_and_zero_fails(base, tmp_path):
    ev = run_eval(base)
    assert ev.confirmatory.rows == 5 * 12 * 8 and ev.confirmatory.minutes == 60
    assert ev.confirmatory.corr == pytest.approx(0.2, abs=1e-12)
    assert ev.confirmatory.t > ev.settings.one_sided_z and ev.confirmatory.passed
    zero = run_eval(build(tmp_path / "zero", news_corr=0.0))
    assert zero.confirmatory.corr == pytest.approx(0.0, abs=1e-12)
    assert abs(zero.confirmatory.t) < 1e-6 and not zero.confirmatory.passed
    assert zero.valid


# The cutoff and the window


def test_the_cutoff_counts_run_minutes_from_launch_through_partial_halts_and_news_outages(copy):
    third, fourth, fifth = SESSIONS[2], SESSIONS[3], SESSIONS[4]
    # A healthy minute before launch never counts, seven symbols halted with QQQ healthy still make run minutes, a
    # halt of every symbol for 15 minutes and a news outage of 25 minutes, 5 of them inside that halt, lose 35.
    before = iso(LAUNCH_MS - 30 * MS_PER_MINUTE)
    execute(copy.s2_db, "INSERT INTO news_minutes VALUES (?, 1, NULL, '')", (before,))
    execute(copy.s2_db, "INSERT INTO stock_candles VALUES ('QQQ', ?, '2026-10-16', 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, "
            "1)", (before,))
    execute(copy.s2_db, "UPDATE stock_candles SET valid = 0 WHERE symbol != 'QQQ' AND ts > ? AND ts <= ?",
            (iso(third.boundary(100)), iso(third.boundary(120))))
    execute(copy.s2_db, "UPDATE stock_candles SET valid = 0 WHERE ts > ? AND ts <= ?",
            (iso(fourth.boundary(200)), iso(fourth.boundary(215))))
    execute(copy.s2_db, "UPDATE news_minutes SET healthy = 0, reason = 'heartbeat_old' WHERE ts > ? AND ts <= ?",
            (iso(fourth.boundary(210)), iso(fourth.boundary(235))))
    conn = store.open_reader(copy.s2_db)
    minutes = evaluate.run_minutes(conn, LAUNCH)
    conn.close()
    assert len(minutes) == RECORDED * 390 - 35 and minutes[0] == iso(SESSIONS[0].boundary(1))
    ev = run_eval(copy)
    assert ev.window == Window(LAUNCH, START, iso(SESSIONS[6].boundary(35)), True, FINAL)
    # Sessions 2 to 6 give 60 minutes, and minute 30 of session 7 is left out because its label matures after the end.
    assert ev.confirmatory.rows == 60 * 8
    quality = {row["measure"]: row["value"] for row in ev.tables["data quality"]}
    assert quality["run minutes lost to heartbeat_old"] == 20 and quality["run minutes"] == FINAL
    assert fifth.close_ms < parse_iso(ev.window.end)


def settings(final, burn_in=1):
    return EvaluationConfig("news", final, evaluate.RUN_MINUTE_RULE, burn_in, 12, 2.33, 1.645, (10, 20), (10, 20, 40))


def every_minute(launch_ms, count):
    days = [evaluate.nth_session(launch_ms, n) for n in range(1, count + 1)]
    return [iso(s.boundary(m)) for s in days for m in range(1, s.minutes + 1)]


def test_the_window_runs_from_session_2_to_40():
    launch = parse_iso("2027-01-04T14:25:00.000Z")  # a Monday whose next 45 sessions are all full
    window = evaluate.find_window(every_minute(launch, 45), iso(launch), settings(15600))
    assert window.final and window.run_minutes == 15600
    assert window.start == iso(evaluate.nth_session(launch, 2).open_ms)
    assert window.end == iso(evaluate.nth_session(launch, 40).close_ms)
    days = evaluate.window_sessions(window)
    assert [evaluate.session_number(s.open_ms + 1, launch) for s in (days[0], days[-1])] == [2, 40]
    assert len(days) == 39
    # From the planned launch the early close of 27 November moves the cutoff into the 41st session, 15 December.
    window = evaluate.find_window(every_minute(LAUNCH_MS, 45), LAUNCH, settings(15600))
    assert window.end == "2026-12-15T17:30:00.000Z"
    assert evaluate.session_number(parse_iso(window.end), LAUNCH_MS) == 41
    # Short of the final count the window ends at the newest run minute and stays descriptive.
    window = evaluate.find_window(every_minute(LAUNCH_MS, 3), LAUNCH, settings(15600))
    assert not window.final and window.end == iso(SESSIONS[2].close_ms) and window.run_minutes == 3 * 390


def test_labels_are_recomputed_from_candles_older_than_30_days(tmp_path):
    first, later = SESSIONS[0], sessions.session_on(date(2026, 11, 20))
    assert later.open_ms - first.close_ms > 30 * 24 * 60 * MS_PER_MINUTE
    candles = [StockCandle("AAPL", s.boundary(m), s.k(s.boundary(m)), 1.0, 1.0, 1.0, 1.0, 100.0 + m / 10 + i, 1.0,
                           1.0, 0.0, 1, 1, 0, not (i == 1 and m == 75))
               for i, s in enumerate((first, later)) for m in range(1, s.minutes + 1)]
    st = StockStore.open(tmp_path / "run" / "stocks.db")
    for c in candles:
        st.queue(records.INSERT_CANDLE, records.candle_row(c))
    rows = recompute_labels(candles, 30)
    for row in rows:
        st.queue(*records.label_statement(dataclasses.replace(row, exit_price=None, label_bps=None, status=PENDING)))
        st.queue(*records.label_statement(row))
    st.flush()
    st.close()
    window = Window(LAUNCH, START, iso(later.close_ms), True, 0)
    conn = store.open_reader(tmp_path / "run" / "stocks.db")
    run = Run("prices run", conn, {}, None)
    check, compared = evaluate.check_labels([run], window, 30)
    assert check.passed and compared == 24 and sum(r.status == "gap" for r in rows) == 1
    conn.close()
    execute(tmp_path / "run" / "stocks.db", "UPDATE stock_labels SET label_bps = label_bps + 1e-9 WHERE ts = ?",
            (iso(first.boundary(30)),))
    conn = store.open_reader(tmp_path / "run" / "stocks.db")
    check, _ = evaluate.check_labels([Run("prices run", conn, {}, None)], window, 30)
    conn.close()
    assert not check.passed and len(check.problems) == 1 and iso(first.boundary(30)) in check.problems[0]


def test_deaths_are_reported(copy):
    died = SESSIONS[3].boundary(100)
    execute(copy.s1_db, records.INSERT_DEATH,
            (4, iso(died), 0, "equity_below_threshold", 1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.9, 1.8, 3, 70.0))
    execute(copy.s1_db, "UPDATE bots SET state = 'dead', died_at = ? WHERE bot_id = 4", (iso(died),))
    ev = run_eval(copy)
    assert ev.deaths == ({"run": "prices run", "bot_id": 4, "kind": "learner", "symbol": "AMZN", "died_at": iso(died),
                          "session": 4, "cause": "equity_below_threshold"},)
    assert {(o.run, o.name): o.matched for o in ev.outcomes}[("prices run", "no deaths")] is False
    assert ev.valid
    summary, _ = evaluate.report(ev)
    assert dict(summary.summary)["deaths"] == evaluate.output.Count(1)


def test_settings_come_from_meta_and_never_from_the_config_files(copy):
    text = copy.s2.read_text(encoding="utf-8")
    copy.s2.write_text(text.replace(f"final_healthy_session_minutes = {FINAL}", "final_healthy_session_minutes = 780")
                       .replace("newey_west_lags = 12", "newey_west_lags = 3"), encoding="utf-8")
    ev = run_eval(copy)
    assert ev.window.end == END and ev.settings.newey_west_lags == 12 and not ev.configs_match
    conn = store.open_reader(copy.s2_db)
    meta = json.loads(store.read_meta(conn)["evaluation_json"])
    conn.close()
    execute(copy.s2_db, "UPDATE meta SET value = ? WHERE key = 'evaluation_json'",
            (json.dumps({**meta, "newey_west_lags": 3, "one_sided_z": 9.0}),))
    again = run_eval(copy)
    conn = store.open_reader(copy.s2_db)
    rows = evaluate.scored_rows(conn, again.window, 30)["news"]
    conn.close()
    assert again.confirmatory == stats.pooled_skill(*zip(*rows), lags=3, one_sided_z=9.0)
    assert again.confirmatory.se != ev.confirmatory.se and not again.confirmatory.passed
    execute(copy.s2_db, "UPDATE meta SET value = 'any_symbol_healthy' WHERE key = 'run_minute_rule'")
    with pytest.raises(EvaluateError, match="run minute rule"):
        run_eval(copy)


# Each condition of criterion 1


def window_row(db, predictor="news"):
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT ts, symbol FROM stock_forecasts WHERE predictor = ? AND ts > ? ORDER BY ts LIMIT 1",
                       (predictor, iso(SESSIONS[2].open_ms))).fetchone()
    conn.close()
    return row


def plant_lookahead(b):
    ts, symbol = window_row(b.s2_db)
    execute(b.s2_db, "UPDATE stock_forecasts SET inputs_max_recv_at = ? WHERE ts = ? AND symbol = ? AND predictor = ?",
            (iso(parse_iso(ts) + 1), ts, symbol, "news"))


def plant_late_line(b):
    execute(b.s2_db, "UPDATE news_seen SET scored_at = ? WHERE score_seq = 7",
            (iso(EVAL_MS[2] + MS_PER_MINUTE),))


def plant_label_mismatch(b):
    for db in (b.s1_db, b.s2_db):
        execute(db, "UPDATE stock_labels SET label_bps = label_bps + 0.5 WHERE ts = ? AND symbol = 'META'",
                (iso(EVAL_MS[30]),))


def plant_differing_labels(b):
    """A different reference price at the close of session 2 in S2 alone, with its label recomputed, so S2 is
    consistent with its own candles but differs from S1."""
    close, ts = SESSIONS[1].close_ms, SESSIONS[1].boundary(360)
    conn = sqlite3.connect(b.s2_db)
    with conn:
        conn.execute("UPDATE stock_candles SET ref_price = ref_price * 1.001 WHERE symbol = 'TSLA' AND ts = ?",
                     (iso(close),))
        entry, exit_price = [conn.execute("SELECT ref_price FROM stock_candles WHERE symbol = 'TSLA' AND ts = ?",
                                          (iso(ms),)).fetchone()[0] for ms in (ts, close)]
        conn.execute("UPDATE stock_labels SET exit_price = ?, label_bps = ? WHERE symbol = 'TSLA' AND ts = ?",
                     (exit_price, 1e4 * math.log(exit_price / entry), iso(ts)))
    conn.close()


def plant_invariant(b):
    execute(b.s1_db, records.INSERT_FEED_EVENT, records.feed_event_row(SESSIONS[3].boundary(10), evaluate.INVARIANT))


def news_call(b, **values):
    conn = news_db.open_writer(b.news_db)
    news_db.insert(conn, "llm_calls", {
        "ledger_id": 2, "kind": "batch", "scorer": "haiku_v1", "started_at": iso(SESSIONS[4].boundary(9)),
        "status": "ok", "n_items": 3, "model": HAIKU_MODEL, "sdk_version": "0",
        "prompt_sha256": NEWS_META["prompt_sha256"], "schema_sha256": NEWS_META["schema_sha256"],
        "request_sha256": "e" * 64, "effort": "low", "thinking": "disabled", "max_tokens": 4096, "reserved_usd": 0.003,
        "price_table": "{}", **values})
    conn.close()


def plant_heartbeat(b):
    conn = news_db.open_writer(b.news_db)
    news_db.insert(conn, "news_heartbeat", {
        "ts": iso(SESSIONS[4].boundary(9)), "healthy": 1, "config_hash": "f" * 64,
        "prompt_sha256": NEWS_META["prompt_sha256"], "schema_sha256": NEWS_META["schema_sha256"], "haiku_state": "on",
        "sonnet_state": "on", "finbert_state": "on", "waiting": 0, "channels_down": "", "spent_day_usd": 0.0,
        "spent_month_usd": 0.0, "spend_alert": 0})
    conn.close()


def plant_spend(b, usd=19.5):
    conn = sqlite3.connect(b.ledger)
    with conn:
        conn.execute("INSERT INTO reservations (ts, process, model, ref, n_items, reserved_usd, cost_usd, cost_basis, "
                     "status) VALUES (?, 'p', ?, 'r', 1, ?, ?, 'usage', 'done')",
                     (iso(SESSIONS[5].boundary(9)), SONNET_MODEL, usd, usd))
    conn.close()


PLANTED = {
    "a lookahead forecast row": (plant_lookahead, {"lookahead audit"}),
    "a score line counted before it was scored": (plant_late_line, {"lookahead audit"}),
    "a label mismatch": (plant_label_mismatch, {"labels recomputed"}),
    "differing labels": (plant_differing_labels, {"runs identical"}),
    "an invariant violation": (plant_invariant, {"invariants"}),
    "a second prompt hash": (lambda b: news_call(b, prompt_sha256="9" * 64), {"one config prompt and model"}),
    "a second model id for one scorer": (lambda b: news_call(b, response_model=SONNET_MODEL),
                                         {"one config prompt and model"}),
    "a heartbeat with another config hash": (plant_heartbeat, {"one config prompt and model"}),
    "a hash mismatch minute": (lambda b: execute(b.s2_db, "UPDATE news_minutes SET healthy = 0, reason = "
                                                 "'hash_mismatch' WHERE ts = ?", (iso(SESSIONS[0].boundary(3)),)),
                               {"one config prompt and model"}),
    "a month over 20 dollars": (plant_spend, {"spend within cap"}),
    "different run minutes in one session": (lambda b: execute(
        b.s2_db, "UPDATE sessions SET run_minutes = 389 WHERE session_date = ?", (SESSIONS[2].day.isoformat(),)),
        {"runs identical"}),
    "a different linear_prices row": (lambda b: execute(b.s2_db, "UPDATE stock_forecasts SET prediction = prediction "
                                                        "+ 1 WHERE predictor = 'linear_prices' AND ts = ? AND symbol = "
                                                        "'AAPL'", (iso(EVAL_MS[20]),)), {"runs identical"}),
    "a failed local only check beside a passed one": (lambda b: execute(b.s1_db, records.INSERT_FEED_EVENT,
                                                                        records.local_only_row(
        SESSIONS[2].open_ms - 5 * MS_PER_MINUTE, False, 1, 0)), {"local only checks"}),
    "a local only check only after the open": (lambda b: [execute(db, "UPDATE feed_events SET ts = ? WHERE ts = ?", (
        iso(SESSIONS[3].boundary(330)), iso(SESSIONS[3].open_ms - 10 * MS_PER_MINUTE))) for db in (b.s1_db, b.s2_db)],
                                               {"local only checks"}),
    "a missing ledger": (lambda b: b.ledger.unlink(), {"spend within cap"}),
    "a failed local only check": (lambda b: execute(b.s2_db, "UPDATE feed_events SET detail = ? WHERE ts = ?", (
        json.dumps({"passed": False, "offending_sockets": 1, "tunnel_processes": 0}),
        iso(SESSIONS[2].open_ms - 10 * MS_PER_MINUTE))), {"local only checks"}),
    "a missing local only check": (lambda b: execute(b.s1_db, "DELETE FROM feed_events WHERE ts = ?", (
        iso(SESSIONS[3].open_ms - 10 * MS_PER_MINUTE),)), {"local only checks"}),
    "no tape check before launch": (lambda b: (b.checks / "tape_2026_10_12.json").unlink(), {"tape checks"}),
    "a start without its commit": (lambda b: execute(b.s1_db, records.INSERT_FEED_EVENT, records.feed_event_row(
        SESSIONS[2].boundary(9), evaluate.CODE_COMMIT)), {"code commits listed"}),
}


@pytest.mark.parametrize("name", list(PLANTED))
def test_the_run_is_invalid_with(copy, name):
    plant, failed = PLANTED[name]
    plant(copy)
    ev = run_eval(copy)
    assert failing(ev) == failed and not ev.valid
    assert all(check.problems for check in ev.checks if check.name in failed)
    assert ev.confirmatory.passed and ev.window.final


def test_spend_exactly_at_the_cap_passes_and_commits_are_listed(copy):
    plant_spend(copy, 19.25)  # with 0.75 already spent, exactly the cap
    execute(copy.s2_db, records.INSERT_FEED_EVENT, records.feed_event_row(SESSIONS[3].boundary(2), evaluate.CODE_COMMIT,
                                                                          None, "f" * 40))
    ev = run_eval(copy)
    assert failing(ev) == set()
    assert [row["commit"] for row in ev.tables["commits"]] == [COMMIT, "f" * 40]
    spent = {(row["month"], row["model"]): float(row["dollars"]) for row in ev.tables["spend"]}
    assert spent == {("2026-10", HAIKU_MODEL): 0.25, ("2026-10", SONNET_MODEL): 19.75}


def test_spend_after_the_cutoff_never_changes_the_result(copy):
    plant_spend(copy, 19.5)
    execute(copy.ledger, "UPDATE reservations SET ts = ? WHERE reserved_usd = 19.5", (iso(SESSIONS[6].boundary(200)),))
    ev = run_eval(copy)
    assert ev.window.end < iso(SESSIONS[6].boundary(200)) and failing(ev) == set()
    assert sum(float(row["dollars"]) for row in ev.tables["spend"]) == 0.75


def test_a_local_only_check_at_a_start_inside_the_session_counts_only_before_its_first_candle(copy):
    for db in (copy.s1_db, copy.s2_db):
        execute(db, "UPDATE feed_events SET ts = ? WHERE ts = ?", (iso(SESSIONS[3].open_ms + 30 * MS_PER_SECOND),
                                                                    iso(SESSIONS[3].open_ms - 10 * MS_PER_MINUTE)))
    assert failing(run_eval(copy)) == set()


def test_a_paired_comparison_judges_both_sides_on_the_same_symbols(copy):
    execute(copy.s2_db, "UPDATE stock_forecasts SET prediction = 0.5 WHERE predictor = 'sonnet' AND symbol = 'MSFT'")
    ev = run_eval(copy)
    both = next(c for c in ev.comparisons if (c.first, c.subset) == ("sonnet", "all"))
    assert both.left_out == ("MSFT",) and both.first_skill.left_out == both.second_skill.left_out == ()
    assert both.first_skill.rows == both.second_skill.rows == both.rows
    assert both.rows == len({k for k in keys(copy.s2_db, "sonnet") & keys(copy.s2_db, "haiku_official")
                             if k[1] != "MSFT"})


def test_pairing_fails_when_s2_never_diverges_or_a_row_is_only_in_one_run(copy):
    pairing = {(o.run, o.name): o for o in run_eval(copy).outcomes}[("both runs", "pairing")]
    assert pairing.matched is True
    execute(copy.s1_db, "DELETE FROM stock_forecasts WHERE predictor = 'population' AND ts = ? AND symbol = 'AAPL'",
            (iso(SESSIONS[0].boundary(30)),))
    pairing = {(o.run, o.name): o for o in run_eval(copy).outcomes}[("both runs", "pairing")]
    assert (pairing.matched, pairing.actual) == (False, "session 1")
    conn = sqlite3.connect(copy.s2_db)
    conn.execute("ATTACH DATABASE ? AS s1", (str(copy.s1_db),))
    with conn:
        conn.execute("DELETE FROM stock_forecasts WHERE predictor = 'population'")
        conn.execute("INSERT INTO stock_forecasts SELECT * FROM s1.stock_forecasts WHERE predictor = 'population'")
    conn.close()
    pairing = {(o.run, o.name): o for o in run_eval(copy).outcomes}[("both runs", "pairing")]
    assert (pairing.matched, pairing.actual) == (False, "none yet")


def test_an_interim_look_says_descriptive_and_never_pass_or_fail(base):
    ev = run_eval(base)
    early = dataclasses.replace(ev, window=dataclasses.replace(ev.window, final=False))
    summary, detail = evaluate.report(early)
    assert dict(summary.summary)["confirmatory test"] == evaluate.output.Word("descriptive")
    assert dict(summary.summary)["cutoff reached"] is False


def test_tape_checks_at_sessions_10_20_and_40_need_a_decided_consolidated_verdict():
    launch, due = LAUNCH_MS, (10, 20, 40)
    window = Window(LAUNCH, START, iso(evaluate.nth_session(launch, 25).close_ms), False, 0)
    before = (date(2026, 10, 12), True, "consolidated")

    def day(n):
        return evaluate.nth_session(launch, n).day

    passed = [before, (day(10), True, "inconclusive"), (day(11), True, "consolidated"), (day(21), True, "consolidated")]
    assert evaluate.check_tape(passed, window, due) == evaluate.Check("tape checks", True)
    unclean = [before, (day(10), False, "consolidated"), (day(12), True, "not consolidated"),
               (day(20), True, "consolidated")]
    assert evaluate.check_tape(unclean, window, due).problems == ("the tape check at or after session 10 is "
                                                                  "not consolidated",)
    # Before the cutoff a record not written yet is not yet due, and at the cutoff a missing record fails.
    assert evaluate.check_tape([before], window, due) == evaluate.Check("tape checks", True)
    final = dataclasses.replace(window, final=True)
    assert evaluate.check_tape([before], final, due).problems == (
        "the tape check at or after session 10 is missing", "the tape check at or after session 20 is missing")


# Descriptive results


def keys(db, predictor):
    conn = sqlite3.connect(db)
    found = {(ts, symbol) for ts, symbol in conn.execute(
        "SELECT f.ts, f.symbol FROM stock_forecasts f JOIN stock_labels l USING (ts, symbol) WHERE f.predictor = ? "
        "AND l.status = 'matured' AND f.ts > ? AND f.ts <= ?", (predictor, START, END))}
    conn.close()
    return found


def rows_on(db, predictor, chosen):
    conn = sqlite3.connect(db)
    out = [(ts, s, p, y) for ts, s, p, y in conn.execute(
        "SELECT f.ts, f.symbol, f.prediction, l.label_bps FROM stock_forecasts f "
        "JOIN stock_labels l USING (ts, symbol) WHERE f.predictor = ? ORDER BY f.ts, f.symbol", (predictor,))
        if (ts, s) in chosen]
    conn.close()
    return out


def test_the_model_comparisons_use_exactly_their_paired_rows_and_paired_event_rows(base):
    ev = run_eval(base)
    event_symbol = {"haiku": lambda ts, s: s == "AAPL" and EVAL_MS.index(parse_iso(ts)) % 2 == 0,
                    "sonnet": lambda ts, s: s == "NVDA"}
    assert [(c.first, c.second, c.subset) for c in ev.comparisons] == [
        ("haiku", "finbert_matched", "all"), ("haiku", "finbert_matched", "events"),
        ("sonnet", "haiku_official", "all"), ("sonnet", "haiku_official", "events")]
    for c in ev.comparisons:
        first, second = keys(base.s2_db, c.first), keys(base.s2_db, c.second)
        chosen = first & second
        assert chosen != first and chosen != second
        if c.subset == "events":
            chosen = {key for key in chosen if event_symbol[c.first](*key)}
        assert c.rows == len(chosen) > 0
        assert c.first_skill == stats.pooled_skill(*zip(*rows_on(base.s2_db, c.first, chosen)))
        assert c.second_skill == stats.pooled_skill(*zip(*rows_on(base.s2_db, c.second, chosen)))
    summary, detail = evaluate.report(ev)
    sections = {report.title: report for report in detail}
    rows = sections["paired comparisons"].rows
    assert len(rows) == 4 and all("verdict" not in row and "passed" not in row for row in rows)
    assert dict(summary.summary)["paired comparisons"] == evaluate.output.Count(4)
    assert not any("haiku" in label or "sonnet" in label for label, _ in summary.summary)
    verdicts = [row for row in sections["skill"].rows if "verdict" in row]
    assert verdicts == [row for row in sections["skill"].rows if row["predictor"] == "news" and row["rows"] == "all"
                        and row["run"] == "news run"] and verdicts[0]["verdict"] == "pass"
    events = {row["predictor"]: row["n"] for row in sections["skill"].rows if row["rows"] == "events"}
    assert events["finbert"] == len({k for k in keys(base.s2_db, "finbert") if k[1] == "MSFT"})
    assert events["news"] == len({k for k in keys(base.s2_db, "news") if event_symbol["haiku"](*k)})


def test_twin_pairs_and_long_shares(base):
    ev = run_eval(base)
    twins = {row["run"]: row for row in ev.tables["twins"]}
    assert twins["prices run"]["pairs"] == 64 and twins["prices run"]["blocks"] == 4
    assert "passed" not in twins["news run"]
    shares = [row for row in ev.tables["long opens"] if row["run"] == "news run"]
    assert [(row["slot"], row["symbol"], row["opens"], row["long share"]) for row in shares] == [
        (slot, symbol, 8, "1.0000") for slot, symbol in enumerate(SYMBOLS)]


def test_a_position_held_over_a_close_while_healthy_is_not_flat(copy):
    close = SESSIONS[3].close_ms
    execute(copy.s1_db, "INSERT INTO trades (id, bot_id, side, opened_at, qty, entry_price, fee_open, funding, "
            "slippage_bps_open, forced, symbol) VALUES (9999, 2, 'long', ?, 5, 100, 0, 0, 2, 1, 'MSFT')",
            (iso(close - 20 * MS_PER_MINUTE),))
    outcome = {(o.run, o.name): o for o in run_eval(copy).outcomes}[("prices run", "flat at every close")]
    assert outcome.matched is False
    assert outcome.actual == ", ".join(f"bot 2 MSFT {s.day}" for s in SESSIONS[3:6])
    # A symbol stale at the close excuses nothing by itself, only a session close rejected as stale does.
    execute(copy.s1_db, "UPDATE stock_candles SET valid = 0 WHERE symbol = 'MSFT' AND ts IN (?, ?, ?)",
            tuple(iso(s.close_ms) for s in SESSIONS[3:6]))
    outcome = {(o.run, o.name): o for o in run_eval(copy).outcomes}[("prices run", "flat at every close")]
    assert outcome.matched is False
    for session in SESSIONS[3:6]:
        execute(copy.s1_db, records.INSERT_FILL, (
            2, 9999, iso(session.close_ms - 5 * MS_PER_MINUTE), None, "close", "sell", 5.0, 100.0, None, None, None,
            None, None, 1200, "rejected", "symbol_not_healthy", "MSFT", None, 2.0, None, None, None))
    outcome = {(o.run, o.name): o for o in run_eval(copy).outcomes}[("prices run", "flat at every close")]
    assert outcome.matched is True
    assert outcome.actual == "none, excused after a close rejected as stale " + ", ".join(
        f"bot 2 MSFT {s.day}" for s in SESSIONS[3:6])


def test_data_quality_and_spend(base):
    ev = run_eval(base)
    quality = {row["measure"]: row["value"] for row in ev.tables["data quality"]}
    assert quality["session minutes"] == FINAL and quality["healthy fraction"] == "1.0000"
    assert quality["outage minutes fed_speeches"] == 12 and quality["items from finnhub"] == 6
    assert quality["haiku_v1 scoring lag median s"] == "4.0" and quality["finbert_v1 scoring lag p90 s"] == "2.0"
    assert quality["fallback share in session"] == "0.0000"
    assert {(row["month"], row["model"]): row["dollars"] for row in ev.tables["spend"]} == {
        ("2026-10", HAIKU_MODEL): "0.2500", ("2026-10", SONNET_MODEL): "0.5000"}


# The command


LINE = re.compile(r"[a-z][a-z_ ]*: (pass|fail|\d+|[a-z][a-z ]*)")


def test_by_default_it_prints_only_verdicts_and_counts(base, capsys):
    assert evaluate.main([str(base.s1), str(base.s2), "--ledger", str(base.ledger)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "evaluate" and re.fullmatch(r"\d+ rows hidden, pass --detail in your own terminal to see them",
                                                    lines[-1])
    assert all(LINE.fullmatch(line) for line in lines[1:-1]) and "." not in "".join(lines[1:-1])
    assert {"run valid: pass", "confirmatory test: pass", "cutoff reached: pass", "run minutes: 2340",
            "window sessions: 5", "confirmatory rows: 480", "founder equity: pass"} <= set(lines)
    assert evaluate.main([str(base.s1), str(base.s2), "--ledger", str(base.ledger), "--detail"]) == 0
    out = capsys.readouterr().out
    assert "\nskill\n" in out and "\npaired comparisons\n" in out and "rows hidden" not in out


def test_an_invalid_run_exits_1_and_an_unlaunched_run_exits_2(copy, capsys):
    plant_invariant(copy)
    assert evaluate.main([str(copy.s1), str(copy.s2), "--ledger", str(copy.ledger)]) == 1
    assert "run valid: fail" in capsys.readouterr().out
    execute(copy.s1_db, "DELETE FROM meta WHERE key = 'launch_ts'")
    assert evaluate.main([str(copy.s1), str(copy.s2), "--ledger", str(copy.ledger)]) == 2
    out, err = capsys.readouterr()
    assert out == "" and err.startswith("error: the prices run has not launched")


def test_a_news_database_at_a_run_path_is_refused(copy, capsys):
    copy.s2_db.unlink()
    shutil.copy(copy.news_db, copy.s2_db)
    assert evaluate.main([str(copy.s1), str(copy.s2), "--ledger", str(copy.ledger)]) == 2
    assert "is not a stock run database" in capsys.readouterr().err


def test_a_version_1_database_at_a_run_path_is_refused_and_left_unchanged(copy, capsys):
    for suffix in ("", "-wal", "-shm"):
        copy.s1_db.with_name(copy.s1_db.name + suffix).unlink(missing_ok=True)
    v1_database(copy.s1_db)
    before = contents(copy.s1_db)
    assert evaluate.main([str(copy.s1), str(copy.s2), "--ledger", str(copy.ledger)]) == 2
    assert "it has no meta.kind" in capsys.readouterr().err and contents(copy.s1_db) == before
