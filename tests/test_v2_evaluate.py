"""The evaluate command of PLAN_V2 sections 11 and 12 on seeded synthetic runs.

Each test builds or copies a prices run and a news run with news.db, a key ledger and tape check records in a temporary
folder, from fictional prices and the committed configs read as if they sat in that folder, so no test opens a real run
database, the real news.db or the real key ledger. The runs launch on Monday 19 October 2026, record 7 sessions and
reach their cutoff after 6 sessions of run minutes, so the window holds sessions 2 to 6. Every row of the news
predictor in the window is built with a correlation of exactly 0.2 with its label within each symbol, or exactly 0.
Both runs store 10 price z scores for every evaluation minute and symbol, noise by default, or with confound built so
that news carries nothing beyond them. The gate session's coverage record is Monday 12 October 2026."""

import dataclasses
import json
import math
import re
import shutil
import sqlite3
import statistics
import sys
from datetime import date
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

import btc_survivor.v2
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
GATE, BASELINE = date(2026, 10, 12), 0.04  # the gate session and its median ratio, about the free feed's share


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


def price_z(zrng, rows, values, confound):
    """The 10 price z scores of a symbol's window rows, noise, or with confound the first one the news score itself,
    with the news score returned plus a part orthogonal to the label, the intercept and every score, so the partial
    correlation of news and label beyond the scores is exactly 0 while the raw one is 0.2 over sqrt(1.09)."""
    z = zrng.normal(size=(len(rows), 10))
    if not confound:
        return z, values
    y = np.array([row.label_bps for row in rows])
    z[:, 0] = values / values.std()
    design = np.column_stack([np.ones(len(rows)), z, y])
    f = zrng.normal(size=len(rows))
    f -= design @ np.linalg.lstsq(design, f, rcond=None)[0]
    return z, values + 0.3 * values.std() * f / f.std()


def make_forecasts(rng, labels, news_corr, zrng, confound=False):
    """The forecast rows of both runs, (ts, symbol, predictor, prediction, items, inputs_max_recv_ms), and the price z
    scores of every evaluation minute and symbol, drawn from their own generator so the forecasts never change."""
    s1, s2 = [], []
    news, scores = {}, {}
    for symbol in SYMBOLS:
        rows = [row for row in labels[symbol] if in_window(row.ts)]
        values = planted(rng, np.array([row.label_bps for row in rows]), news_corr)
        z, values = price_z(zrng, rows, values, confound)
        news.update({(row.ts, symbol): float(v) for row, v in zip(rows, values)})
        scores.update({(row.ts, symbol): z[i] for i, row in enumerate(rows)})
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
            if (ts, symbol) not in scores:
                scores[ts, symbol] = zrng.normal(size=10)
    return s1, s2, scores


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
    hashes and models, the key ledger with October spend and the gate session's coverage record before launch."""
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
    write_coverage(built.checks, GATE, BASELINE)


def coverage_record(day, median, clean=True, early_close=False):
    """A coverage record in the format tape-check writes, every symbol at the median ratio."""
    ratios = {symbol: median for symbol in SYMBOLS}
    return {"kind": "coverage", "session": day.isoformat(), "clean": clean and not early_close,
            "reason": "a full session connected throughout", "early_close": early_close, "median_ratio": median,
            "median_ratio_covered": median if clean else None, "ratios": ratios,
            "ratios_covered": ratios if clean else None, "disconnected_seconds": 0.0 if clean else 30.0,
            "outage_seconds": 0.0, "covered_share": {"pre": 1.0, "regular": 1.0, "post": 1.0}}


def write_coverage(folder, day, median, clean=True, early_close=False):
    (folder / f"coverage_{day:%Y_%m_%d}.json").write_text(json.dumps(coverage_record(day, median, clean, early_close)))


def build(folder, news_corr=0.2, seed=11, confound=False):
    """Both runs with identical candles, labels, linear_prices rows, price z scores and trades, S2 with every news
    predictor."""
    built = Built(folder)
    write_configs(folder)
    configs = load_run_pair(built.s1, built.s2)
    rng = np.random.default_rng(seed)
    candles = make_candles(rng)
    labels = {symbol: recompute_labels(candles[symbol], 30) for symbol in SYMBOLS}
    *forecasts, scores = make_forecasts(rng, labels, news_corr, np.random.default_rng([seed, 1]), confound)
    shared = [(records.INSERT_CANDLE, records.candle_row(c)) for symbol in SYMBOLS for c in candles[symbol]]
    shared += [(records.INSERT_PRICE_Z, records.price_z_row(sessions.k_at(ms), ms, symbol, z))
               for (ms, symbol), z in sorted(scores.items())]
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
            "first trades", "burn", "founder equity", "no deaths", "flat at every close")},
        ("both runs", "pairing"): True}
    # The round trips per session are reported beside the outcomes and never judged, every one forced here.
    trips = ev.tables["round trips"]
    assert [row["run"] for row in trips] == list(evaluate.RUNS) and all(row["learners"] > 0 for row in trips)
    assert all(row["own"] == "0.00" and row["all"] == row["forced"] and float(row["forced"]) > 0 for row in trips)
    assert all(row["minutes held"] == "30" and row["timer alone"] == "at most about 2" for row in trips)
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
    # H2 is run only after H1 passes, so the familywise error stays at one sided 1 percent.
    assert zero.incremental is None and zero.without_scores == 0
    summary, detail = evaluate.report(zero)
    assert dict(summary.summary)["incremental test"] == evaluate.output.Word("not run")
    assert dict(summary.summary)["incremental rows"] == evaluate.output.Count(0)
    assert {r.title: r for r in detail}["incremental test"].rows[0]["verdict"] == "not run"


def window_scores(db):
    conn = store.open_reader(db)
    try:
        return evaluate.price_scores(conn, Window(LAUNCH, START, END, True, FINAL))
    finally:
        conn.close()


def test_h2_judges_the_rows_of_h1_beyond_the_10_price_z_scores(base):
    ev = run_eval(base)
    rows = rows_on(base.s2_db, "news", keys(base.s2_db, "news"))
    scores = window_scores(base.s2_db)
    assert len(rows) == 480 and scores == window_scores(base.s1_db)
    assert ev.incremental == stats.partial_skill(*zip(*[(ts, s, p, y, scores[ts, s]) for ts, s, p, y in rows]))
    assert ev.incremental.rows == 480 and ev.without_scores == 0 and ev.incremental.left_out == ()
    assert ev.incremental.passed and ev.incremental.corr == pytest.approx(0.2, abs=0.05)
    summary, detail = evaluate.report(ev)
    assert dict(summary.summary)["incremental test"] is True
    row = {r.title: r for r in detail}["incremental test"].rows[0]
    assert (row["predictor"], row["n"], row["rows without price scores"], row["verdict"]) == (
        "news beyond price scores", 480, "0", "pass")


def test_news_that_only_repeats_a_price_score_passes_h1_and_fails_h2(tmp_path):
    ev = run_eval(build(tmp_path / "confound", confound=True))
    assert ev.confirmatory.passed and ev.confirmatory.corr == pytest.approx(0.2 / math.sqrt(1.09), abs=1e-12)
    assert ev.incremental.rows == 480 and ev.incremental.corr == pytest.approx(0.0, abs=1e-9)
    assert not ev.incremental.passed and ev.valid
    summary, _ = evaluate.report(ev)
    assert dict(summary.summary)["incremental test"] is False


def test_rows_without_price_scores_are_left_out_of_h2_and_counted(copy):
    execute(copy.s2_db, "DELETE FROM price_z WHERE symbol = 'AAPL' AND ts > ? AND ts <= ?",
            (iso(SESSIONS[2].open_ms), iso(SESSIONS[2].close_ms)))
    execute(copy.s2_db, "UPDATE price_z SET z3 = NULL WHERE symbol = 'MSFT' AND ts = ?", (iso(EVAL_MS[30]),))
    ev = run_eval(copy)
    assert ev.confirmatory.rows == 480 and ev.without_scores == 13 and ev.incremental.rows == 467
    summary, _ = evaluate.report(ev)
    assert dict(summary.summary)["rows without price scores"] == evaluate.output.Count(13)


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
    # The survivors' mean leaves the dead learner out, and the mean over all learners counts it at its final equity.
    conn = sqlite3.connect(copy.s1_db)
    survivors = [conn.execute("SELECT equity FROM equity_snapshots WHERE bot_id = ? AND ts <= ? ORDER BY ts DESC "
                              "LIMIT 1", (bot_id, END)).fetchone()[0] for bot_id in range(1, 9) if bot_id != 4]
    conn.close()
    equity = {row["run"]: row for row in ev.tables["learner equity"]}
    assert equity["prices run"] == {"run": "prices run", "learners": 8, "survivors": 7,
                                    "survivors mean equity": f"{statistics.fmean(survivors):.2f}",
                                    "all learners mean equity": f"{statistics.fmean([*survivors, 1.8]):.2f}"}
    assert equity["news run"]["survivors"] == 8
    assert equity["news run"]["survivors mean equity"] == equity["news run"]["all learners mean equity"]


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


def plant_coverage(b):
    """Coverage due at session 2 as recorded at launch, with session 3 the first clean one at 2.01 times the gate."""
    conn = store.open_reader(b.s2_db)
    meta = json.loads(store.read_meta(conn)["evaluation_json"])
    conn.close()
    execute(b.s2_db, "UPDATE meta SET value = ? WHERE key = 'evaluation_json'",
            (json.dumps({**meta, "tape_check_sessions": [2]}),))
    write_coverage(b.checks, SESSIONS[1].day, BASELINE, clean=False)
    write_coverage(b.checks, SESSIONS[2].day, 2.01 * BASELINE)


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
    "no coverage record before launch": (lambda b: (b.checks / "coverage_2026_10_12.json").unlink(),
                                         {"coverage stability"}),
    "a coverage record outside a factor of 2 of the gate session": (plant_coverage, {"coverage stability"}),
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
    assert dict(summary.summary)["incremental test"] == evaluate.output.Word("descriptive")
    assert dict(summary.summary)["cutoff reached"] is False


def test_coverage_at_sessions_10_20_and_40_stays_within_a_factor_of_2_of_the_gate_session():
    launch, due = LAUNCH_MS, (10, 20, 40)
    window = Window(LAUNCH, START, iso(evaluate.nth_session(launch, 25).close_ms), False, 0)
    gate = (GATE, True, BASELINE)

    def day(n):
        return evaluate.nth_session(launch, n).day

    # The edges count as within, and session 21 stands for session 20 when 20 has no record.
    passed = [gate, (day(10), True, BASELINE / 2), (day(21), True, 2 * BASELINE)]
    assert evaluate.check_coverage(passed, window, due) == evaluate.Check("coverage stability", True)
    # An unclean session is passed over, so the first clean one after session 10 is judged, at 2.5 times the gate.
    unclean = [gate, (day(10), False, BASELINE), (day(12), True, 2.5 * BASELINE), (day(20), True, BASELINE)]
    assert evaluate.check_coverage(unclean, window, due).problems == (
        f"the median ratio of {day(12)}, the first clean session at or after session 10, is 2.50 times the gate "
        f"session's",)
    low = [gate, (day(10), True, 0.49 * BASELINE)]
    assert not evaluate.check_coverage(low, window, due).passed
    # Before the cutoff a record not written yet is not yet due, and at the cutoff a missing record fails.
    assert evaluate.check_coverage([gate], window, due) == evaluate.Check("coverage stability", True)
    final = dataclasses.replace(window, final=True)
    assert evaluate.check_coverage([gate], final, due).problems == (
        "the coverage record at or after session 10 is missing",
        "the coverage record at or after session 20 is missing")


def test_the_baseline_is_the_gate_session_the_earliest_full_session_before_launch_clean_or_not():
    launch = LAUNCH_MS
    window = Window(LAUNCH, START, iso(evaluate.nth_session(launch, 12).close_ms), True, 0)
    later = (evaluate.nth_session(launch, 10).day, True, 0.05)
    # 12 October is the gate even when a silence close left it unclean, and the later 0.2 is never the baseline.
    records = [(GATE, False, 0.04), (date(2026, 10, 14), True, 0.2), later]
    assert evaluate.check_coverage(records, window, (10,)) == evaluate.Check("coverage stability", True)
    assert not evaluate.check_coverage(records[1:], window, (10,)).passed  # 0.05 against a baseline of 0.2
    # A later unclean session is never judged against the gate, only the first clean one at or after session 10.
    assert evaluate.check_coverage([*records, (evaluate.nth_session(launch, 11).day, False, 0.5)], window,
                                   (10,)).passed
    # An early close never counts, so the Friday after Thanksgiving, session 29, is passed over for the Monday after.
    end = Window(LAUNCH, START, iso(evaluate.nth_session(launch, 30).close_ms), True, 0)
    thanksgiving = [(GATE, True, 0.04), (evaluate.nth_session(launch, 29).day, True, 0.4),
                    (evaluate.nth_session(launch, 30).day, True, 0.05)]
    assert evaluate.nth_session(launch, 29).early_close and evaluate.check_coverage(thanksgiving, end, (29,)).passed
    assert evaluate.check_coverage([later], window, (10,)).problems == (
        "no full session coverage record before launch, so no gate session",)
    early = sessions.session_on(date(2026, 11, 27))   # an early close before a later launch is never the gate
    assert early.early_close and not evaluate.full_session(early.day)
    assert evaluate.check_coverage([(GATE, True, 0.0), later], window, (10,)).problems == (
        "the gate session's median ratio is not above zero",)


def test_coverage_records_are_read_from_the_checks_folder_and_old_tape_verdicts_are_ignored(copy):
    (copy.checks / "coverage_2026_10_12.json").unlink()
    (copy.checks / "tape_2026_10_12.json").write_text(json.dumps(
        {"session": "2026-10-12", "clean": True, "ratio_verdict": "consolidated", "verdict": "consolidated"}))
    write_coverage(copy.checks, date(2026, 10, 13), BASELINE, early_close=False, clean=False)
    write_coverage(copy.checks, date(2026, 10, 14), BASELINE)
    assert evaluate.coverage_records(copy.checks) == [(date(2026, 10, 13), False, BASELINE),
                                                      (date(2026, 10, 14), True, BASELINE)]
    assert failing(run_eval(copy)) == set()
    (copy.checks / "coverage_2026_10_14.json").write_text(json.dumps({"session": "2026-10-14", "clean": True}))
    with pytest.raises(EvaluateError, match="coverage_2026_10_14.json is not a coverage record"):
        run_eval(copy)


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


def test_own_trades_are_reported_and_never_judged(copy):
    ev = run_eval(copy)
    assert "gate" not in {o.name for o in ev.outcomes}
    own = {row["run"]: row for row in ev.tables["own trades"]}
    # Every decision predicts 0.5 bps, 8 a session on each home symbol, so the spread comes from the label scales.
    conn = sqlite3.connect(copy.s1_db)
    sd = {symbol: np.std([y for (y,) in conn.execute(
        "SELECT label_bps FROM stock_labels WHERE symbol = ? AND status = 'matured' AND ts > ? AND ts <= ?",
        (symbol, START, END))]) for symbol in SYMBOLS}
    conn.close()
    margin = load_run_pair(copy.s1, copy.s2)[0].trading.margin_sigma
    assert own["prices run"] == {"run": "prices run", "margin sigma": f"{margin:g}", "decisions": 64, "opens": 64,
                                 "forced opens": 64, "own opens": 0, "own share": "0.0000",
                                 "gate passed share": "0.0000",
                                 "prediction spread sd": f"{np.std([0.5 / sd[s] for s in SYMBOLS]):.4f}"}
    # With no margin a learner opens on its own once its prediction clears the cost, and a twin never counts.
    for bot_id in (3, TWIN_BASE + 3):
        execute(copy.s2_db, records.INSERT_DECISION, (bot_id, iso(SESSIONS[4].boundary(200)), "flat", 100.0, 30.0,
                                                      4.6, 4.6, 0.0, 1, 0, "open_long", "gate", "NVDA"))
    ev = run_eval(copy)
    own = {row["run"]: row for row in ev.tables["own trades"]}["news run"]
    assert (own["decisions"], own["opens"], own["forced opens"], own["own opens"]) == (65, 65, 64, 1)
    assert own["own share"] == own["gate passed share"] == f"{1 / 65:.4f}" and float(own["prediction spread sd"]) > 0
    summary = dict(evaluate.report(ev)[0].summary)
    assert summary["news run learner opens"] == evaluate.output.Count(65)
    assert summary["news run own opens"] == evaluate.output.Count(1)
    assert summary["prices run own opens"] == evaluate.output.Count(0) and ev.valid


def test_the_founder_equity_bound_grows_with_the_horizons_positions_were_held(copy):
    def founder():
        ev = run_eval(copy)
        outcome = {(o.run, o.name): o for o in ev.outcomes}[("prices run", "founder equity")]
        trips = {row["run"]: row for row in ev.tables["round trips"]}["prices run"]
        return outcome, float(outcome.expected.split(" within ")[1]), trips["minutes held"]

    before, bound, minutes = founder()
    assert before.matched and minutes == "30"
    # Every learner trade opened 90 minutes earlier, so each was held four horizons, as the holding rule allows.
    execute(copy.s1_db, "UPDATE trades SET opened_at = strftime('%Y-%m-%dT%H:%M:%fZ', opened_at, '-90 minutes') "
                        "WHERE bot_id < ?", (TWIN_BASE,))
    after, wider, minutes = founder()
    assert after.matched and minutes == "120" and wider == pytest.approx(2 * bound, abs=0.01)


def test_a_position_carried_overnight_or_over_a_weekend_counts_its_session_time_only():
    fri, mon, thu = (sessions.session_on(date(2026, 10, day)) for day in (16, 19, 15))
    held = evaluate.session_ms_held
    assert held(iso(fri.boundary(90)), iso(fri.boundary(240))) == 150 * MS_PER_MINUTE
    # Opened 15:50 Friday, carried over the weekend and flattened 2 seconds after Monday's open.
    assert held(iso(fri.boundary(380)), iso(mon.open_ms + 2 * MS_PER_SECOND)) == 10 * MS_PER_MINUTE + 2 * MS_PER_SECOND
    # Two full sessions and the 15:30 to 16:00 and 09:30 to 10:00 ends around them, never the nights between.
    assert held(iso(thu.boundary(360)), iso(mon.boundary(30))) == (30 + 390 + 30) * MS_PER_MINUTE
    assert held(iso(fri.close_ms + MS_PER_MINUTE), iso(mon.open_ms)) == 0


def test_the_largest_notional_over_equity(copy):
    rows = {row["run"]: row for row in run_eval(copy).tables["notional over equity"]}
    assert rows["prices run"]["open fills"] == 128 and float(rows["prices run"]["largest"]) == pytest.approx(0.05,
                                                                                                            abs=5e-4)
    # A 2,000 dollar open of learner 3 in session 4, against its equity at the close of session 3.
    decided = SESSIONS[3].boundary(100)
    execute(copy.s1_db, records.INSERT_FILL, (3, None, iso(decided), iso(decided + 1200), "open", "buy", 20.0, 100.0,
                                              None, 100.0, None, 1.0, 0.0, 1200, "filled", "decision", "NVDA", 100.0,
                                              1.0, 2000.0, 0.0, 0.0))
    conn = sqlite3.connect(copy.s1_db)
    equity = conn.execute("SELECT equity FROM equity_snapshots WHERE bot_id = 3 AND ts = ?",
                          (iso(SESSIONS[2].close_ms),)).fetchone()[0]
    conn.close()
    row = {row["run"]: row for row in run_eval(copy).tables["notional over equity"]}["prices run"]
    assert (row["open fills"], row["bot_id"], row["kind"], row["decided_at"]) == (129, 3, "learner", iso(decided))
    assert row["largest"] == f"{2000.0 / equity:.4f}"


def test_data_quality_and_spend(base):
    ev = run_eval(base)
    quality = {row["measure"]: row["value"] for row in ev.tables["data quality"]}
    assert quality["session minutes"] == FINAL and quality["healthy fraction"] == "1.0000"
    assert quality["outage minutes fed_speeches"] == 12 and quality["items from finnhub"] == 6
    assert quality["haiku_v1 scoring lag median s"] == "4.0" and quality["finbert_v1 scoring lag p90 s"] == "2.0"
    assert quality["fallback share in session"] == "0.0000"
    assert {(row["month"], row["model"]): row["dollars"] for row in ev.tables["spend"]} == {
        ("2026-10", HAIKU_MODEL): "0.2500", ("2026-10", SONNET_MODEL): "0.5000"}


# The scorecards


def test_the_scorecards_are_descriptive_sections_with_counts_by_default_and_figures_in_detail(base):
    from btc_survivor.v2 import scorecards
    ev = run_eval(base)
    summary, detail = evaluate.report(ev)
    lines, sections = dict(summary.summary), {r.title: r for r in detail}
    conns = [store.open_reader(path) for path in (base.s1_db, base.s2_db)]
    try:
        configs = load_run_pair(base.s1, base.s2)
        sources = [scorecards.Source(name, conn, config) for name, conn, config in zip(evaluate.RUNS, conns, configs)]
        cards = [card for source in sources for card in scorecards.chooser_cards(source, START, END, 12, 1.645)]
        day = scorecards.day_card(sources, LAUNCH, END, 1.645)
    finally:
        for conn in conns:
            conn.close()
    assert [(c.run, c.predictor) for c in cards] == [("prices run", "population"), ("prices run", "linear_prices"),
                                                     ("news run", "population"), ("news run", "linear_prices"),
                                                     ("news run", "linear_news")]
    for card in cards:
        assert lines[f"{card.run} {card.predictor} chooser picks"] == evaluate.output.Count(card.picks)
        assert lines[f"{card.run} {card.predictor} chooser flat minutes"] == evaluate.output.Count(card.flat)
        assert card.minutes == 60 and card.unscaled_minutes == 48
    # The window holds sessions 2 to 6, so the day scorecard covers sessions 3 to 6 of 8 symbols.
    assert (lines["day scorecard sessions"], lines["day scorecard counted symbol days"],
            lines["day scorecard failed symbol days"]) == tuple(evaluate.output.Count(n) for n in (4, 32, 0))
    assert day.sessions == 4 and len(sections["day scorecard"].rows) == 2 * 11
    assert len(sections["chooser"].rows) == 5 * 6 and len(sections["chooser against comparisons"].rows) == 5 * 5
    assert len(sections["chooser picks per symbol"].rows) == 5 and len(sections["day labels"].rows) == 2
    first = sections["chooser"].rows[0]
    assert first["strategy"] == "chooser" and first["no scale minutes"] == 48 and isinstance(first["total bps"], str)
    # No pass label anywhere in the scorecards, and the figures print only with --detail.
    titles = evaluate.SCORECARD_TITLES
    assert not any({"verdict", "passed", "threshold"} & set(row) for t in titles if t in sections
                   for row in sections[t].rows)
    assert not any("chooser" in label and not isinstance(value, evaluate.output.Count)
                   for label, value in summary.summary)


def test_scorecards_that_cannot_be_built_change_no_verdict_and_no_exit_code(base, monkeypatch, capsys):
    from btc_survivor.v2 import scorecards
    good = run_eval(base)
    monkeypatch.setattr(scorecards, "day_card", lambda *args, **kwargs: 1 / 0)
    broken = run_eval(base)
    assert (broken.checks, broken.outcomes, broken.confirmatory, broken.incremental, broken.valid) == (
        good.checks, good.outcomes, good.confirmatory, good.incremental, good.valid)
    assert broken.scorecards == (("scorecards", evaluate.output.Word("not computed")),)
    keep = [line for line in evaluate.report(good)[0].summary if line[0] not in dict(good.scorecards)]
    assert [line for line in evaluate.report(broken)[0].summary if line[0] != "scorecards"] == keep
    assert broken.tables["scorecards"][0]["problem"].startswith("the scorecards could not be built, ZeroDivisionError")
    assert evaluate.main([str(base.s1), str(base.s2), "--ledger", str(base.ledger)]) == 0
    out = capsys.readouterr().out
    assert "scorecards: not computed" in out and "run valid: pass" in out


def test_a_scorecards_module_that_cannot_be_imported_changes_no_verdict_and_no_exit_code(base, monkeypatch, capsys):
    good = run_eval(base)
    # None in sys.modules makes the import fail, as a module broken by a later edit would.
    monkeypatch.delattr(btc_survivor.v2, "scorecards", raising=False)
    monkeypatch.setitem(sys.modules, "btc_survivor.v2.scorecards", None)
    broken = run_eval(base)
    assert (broken.checks, broken.outcomes, broken.confirmatory, broken.valid) == (
        good.checks, good.outcomes, good.confirmatory, good.valid)
    assert broken.scorecards == (("scorecards", evaluate.output.Word("not computed")),)
    assert "ModuleNotFoundError" in broken.tables["scorecards"][0]["problem"]
    assert evaluate.main([str(base.s1), str(base.s2), "--ledger", str(base.ledger)]) == 0
    out = capsys.readouterr().out
    assert "scorecards: not computed" in out and "run valid: pass" in out and "Traceback" not in out


def test_the_forecasts_the_scorecards_leave_out_for_late_inputs_are_counted(base, copy):
    lines = dict(evaluate.report(run_eval(base))[0].summary)
    assert lines["prices run scorecard forecasts left out for late inputs"] == evaluate.output.Count(0)
    assert lines["news run scorecard forecasts left out for late inputs"] == evaluate.output.Count(0)
    plant_lookahead(copy)
    ev = run_eval(copy)
    lines = dict(evaluate.report(ev)[0].summary)
    assert lines["news run scorecard forecasts left out for late inputs"] == evaluate.output.Count(1)
    assert lines["prices run scorecard forecasts left out for late inputs"] == evaluate.output.Count(0)
    # The run is invalid through its lookahead audit, never through the scorecards.
    assert not ev.valid and {c.name for c in ev.checks if not c.passed} == {"lookahead audit"}


# The command


LINE = re.compile(r"[a-z][a-z_ ]*: (pass|fail|\d+|[a-z][a-z ]*)")


def test_by_default_it_prints_only_verdicts_and_counts(base, capsys):
    assert evaluate.main([str(base.s1), str(base.s2), "--ledger", str(base.ledger)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "evaluate" and re.fullmatch(r"\d+ rows hidden, pass --detail in your own terminal to see them",
                                                    lines[-1])
    assert all(LINE.fullmatch(line) for line in lines[1:-1]) and "." not in "".join(lines[1:-1])
    assert {"run valid: pass", "confirmatory test: pass", "cutoff reached: pass", "run minutes: 2340",
            "window sessions: 5", "confirmatory rows: 480", "founder equity: pass", "coverage stability: pass",
            "incremental test: pass", "incremental rows: 480", "rows without price scores: 0",
            "prices run learner opens: 64", "prices run own opens: 0", "day scorecard sessions: 4",
            "day scorecard counted symbol days: 32", "day scorecard failed symbol days: 0"} <= set(lines)
    assert sum(1 for line in lines if " chooser picks: " in line) == 5
    assert not any(line.startswith("gate") for line in lines)
    assert evaluate.main([str(base.s1), str(base.s2), "--ledger", str(base.ledger), "--detail"]) == 0
    out = capsys.readouterr().out
    assert "\nskill\n" in out and "\npaired comparisons\n" in out and "rows hidden" not in out
    for title in ("chooser", "chooser against comparisons", "day labels", "day scorecard", "day scorecard per symbol"):
        assert f"\n{title}\n" in out


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
