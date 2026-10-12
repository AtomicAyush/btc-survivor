"""Seeded stock run databases and news.db for the status, export, dashboard and decisions tests, written through the
row builders into a test folder, so no test opens data/v2, the real secrets file or the network.

The committed S1 and S2 configs are copied two folders below the test folder, so their data lands in its data/v2, and
their secrets file is a fixture with fake values. Every number is synthetic. CANARY_PRICE is a reference price and
CANARY_FILL a fill price that must never reach default command output.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from btc_survivor.account import Account
from btc_survivor.clock import iso, parse_iso
from btc_survivor.types import LONG, SHORT
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import StockConfig, load_run_pair
from btc_survivor.v2.news import db as news_db
from btc_survivor.v2.stocks import records
from btc_survivor.v2.stocks.broker import StockFill
from btc_survivor.v2.stocks.candles import StockCandle
from btc_survivor.v2.stocks.labels import LabelRow
from btc_survivor.v2.stocks.store import StockStore
from helpers import ROOT

CONFIGS = ROOT / "configs" / "v2"
REAL_SECRETS = 'secrets_file = "~/.config/btc_survivor/secrets.env"'
FAKE = {"FINNHUB_API_KEY": "fake-finnhub-dashboard-canary", "ANTHROPIC_API_KEY": "fake-anthropic-dashboard-canary",
        "CONTACT_EMAIL": "fake.dashboard.canary@example.invalid"}
CANARY_PRICE = 187.4321
CANARY_FILL = 187.4399
EXPERIMENT = "e" * 64
SYMBOLS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "QQQ")
LAUNCH = "2026-10-19T13:30:00.000Z"
WINDOW_DAYS = (date(2026, 10, 20), date(2026, 10, 21), date(2026, 10, 22))
NOW = "2026-10-22T20:10:00.000Z"


def ms(text: str) -> int:
    return parse_iso(text)


@dataclass(frozen=True)
class Runs:
    """The two seeded run configs, their databases and news.db."""

    s1: Path
    s2: Path
    s1_db: Path
    s2_db: Path
    news_db: Path
    secrets: Path


def secrets_file(folder: Path) -> Path:
    """A fixture secrets file with fake values, mode 600."""
    path = folder / "secrets.env"
    path.write_text("".join(f"{name}={value}\n" for name, value in FAKE.items()), encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def configs(folder: Path) -> tuple[Path, Path, Path]:
    """S1 and S2 copied to folder/configs/v2 with the fixture secrets file, and that file."""
    target = folder / "configs" / "v2"
    target.mkdir(parents=True, exist_ok=True)
    secrets = secrets_file(folder)
    paths = []
    for name in ("s1_prices.toml", "s2_news.toml"):
        text = (CONFIGS / name).read_text(encoding="utf-8")
        assert REAL_SECRETS in text
        (target / name).write_text(text.replace(REAL_SECRETS, f'secrets_file = "{secrets}"'), encoding="utf-8")
        paths.append(target / name)
    return paths[0], paths[1], secrets


def bot(bot_id: int, slot: int, home: str, origin: str, state: str = "active") -> tuple[str, tuple]:
    return records.INSERT_BOT, records.bot_row(bot_id=bot_id, slot=slot, home_symbol=home, born_ms=ms(LAUNCH),
                                               state=state, margin_sigma=0.25, lr=0.0001, seed=12345, origin=origin)


def decision(bot_id: int, ts: str, symbol: str, action: str, *, before: str = "flat", pred: float = 1.0,
             required: float = 9.0, forced: int = 0, gate: int = 0, reason: str = "") -> tuple[str, tuple]:
    return records.INSERT_DECISION, records.decision_row({
        "bot_id": bot_id, "ts": ms(ts), "symbol": symbol, "position_before": before, "ref_mid": CANARY_PRICE,
        "prediction_bps": pred, "required_edge_bps": required, "cost_estimate_bps": required - 2.5,
        "margin_bps": 2.5, "gate_passed": gate, "forced": forced, "action": action,
        "reason": reason or f"prediction {pred:.2f} bps is below required edge {required:.2f} bps"})


def label(bot_id: int, ts: str, value: float) -> tuple[str, tuple]:
    return records.SET_LABEL, (value, bot_id, ts)


def fill(bot_id: int, ts: str, symbol: str, leg: str, side: str, *, trade_id: int | None, qty: float = 2.0,
         status: str = "filled", reason: str = "decision") -> tuple[str, tuple]:
    filled = status == "filled"
    return records.INSERT_FILL, records.fill_row(StockFill(
        bot_id=bot_id, trade_id=trade_id, leg=leg, taker_side=side, qty=qty, ref_mid=CANARY_PRICE,
        decided_ms=ms(ts), filled_ms=ms(ts) + 1200 if filled else None, book_avg_price=None,
        price=CANARY_FILL if filled else None, levels=None, slippage_bps=0.42 if filled else None,
        fee=0.03 if filled else None, latency_ms=1200, status=status, reason=reason, symbol=symbol,
        print_price=CANARY_PRICE if filled else None, allowance_bps=2.0, notional=qty * CANARY_FILL if filled else None,
        sec_fee=0.02 if filled else None, taf_fee=0.01 if filled else None))


def open_trade(bot_id: int, trade_id: int, ts: str, symbol: str, side: int, forced: bool = False) -> tuple[str, tuple]:
    acct = Account(10000.0)
    acct.open(side, 2.0, CANARY_FILL, 0.03, ms(ts) + 1200, trade_id=trade_id, slippage_bps=0.42, forced=forced)
    return records.OPEN_TRADE, records.open_trade_row(bot_id, symbol, acct)


def close_trade(trade_id: int, ts: str, reason: str) -> tuple[str, tuple]:
    return records.CLOSE_TRADE, (iso(ms(ts) + 1200), CANARY_FILL + 0.5, 0.05, 0.0, 0.4, 1.0, 0.92, reason, trade_id)


# The decisions both runs share, in the order the engine writes them, as (statement, values).
SHARED = [
    bot(1, 0, "AAPL", "random"), bot(2, 6, "TSLA", "random"), bot(3, 0, "AAPL", "control"),
    bot(4, 6, "TSLA", "control", state="dead"),
    (records.SET_BOT_DEAD, ("2026-10-22T15:30:00.000Z", 4)),
    decision(1, "2026-10-21T14:00:00.000Z", "AAPL", "none", reason="warmup, 300 of 360 training minutes"),
    label(1, "2026-10-21T14:00:00.000Z", 3.0),
    decision(1, "2026-10-21T14:30:00.000Z", "AAPL", "open_long", pred=1.2, forced=1,
             reason="forced by the exploration timer, prediction 1.20 bps"),
    label(1, "2026-10-21T14:30:00.000Z", 5.0),
    fill(1, "2026-10-21T14:30:00.000Z", "AAPL", "open", "buy", trade_id=1),
    open_trade(1, 1, "2026-10-21T14:30:00.000Z", "AAPL", LONG, forced=True),
    decision(3, "2026-10-21T14:30:00.000Z", "AAPL", "open_short", pred=1.2,
             reason="twin of bot 1, the coin chose short"),
    label(3, "2026-10-21T14:30:00.000Z", 5.0),
    fill(3, "2026-10-21T14:30:00.000Z", "AAPL", "open", "sell", trade_id=2),
    open_trade(3, 2, "2026-10-21T14:30:00.000Z", "AAPL", SHORT),
    decision(1, "2026-10-21T15:00:00.000Z", "AAPL", "close", before="long", pred=-0.5),
    fill(1, "2026-10-21T15:00:00.000Z", "AAPL", "close", "sell", trade_id=1),
    close_trade(1, "2026-10-21T15:00:00.000Z", "signal"),
    decision(2, "2026-10-21T15:06:00.000Z", "TSLA", "open_short", pred=-12.0, gate=1,
             reason="prediction -12.00 bps clears required edge 9.00 bps"),
    label(2, "2026-10-21T15:06:00.000Z", -8.0),
    fill(2, "2026-10-21T15:06:00.000Z", "TSLA", "open", "sell", trade_id=None, status="rejected",
         reason="symbol_not_healthy"),
    decision(2, "2026-10-22T14:36:00.000Z", "TSLA", "open_short", pred=-3.0, forced=1,
             reason="forced by the exploration timer, prediction -3.00 bps"),
    label(2, "2026-10-22T14:36:00.000Z", 2.0),
    fill(2, "2026-10-22T14:36:00.000Z", "TSLA", "open", "sell", trade_id=3),
    open_trade(2, 3, "2026-10-22T14:36:00.000Z", "TSLA", SHORT, forced=True),
    decision(2, "2026-10-22T15:06:00.000Z", "TSLA", "flip", before="short", pred=10.5, gate=1,
             reason="prediction 10.50 bps clears required edge 9.00 bps"),
    fill(2, "2026-10-22T15:06:00.000Z", "TSLA", "close", "buy", trade_id=3),
    close_trade(3, "2026-10-22T15:06:00.000Z", "flip"),
    fill(2, "2026-10-22T15:06:00.000Z", "TSLA", "open", "buy", trade_id=4),
    open_trade(2, 4, "2026-10-22T15:06:00.000Z", "TSLA", LONG),
]
# S2 alone, an open whose order has not settled.
S2_ONLY = [decision(1, "2026-10-22T19:30:00.000Z", "AAPL", "open_long", pred=0.8, forced=1,
                    reason="forced by the exploration timer, prediction 0.80 bps")]


def candle(symbol: str, ts: str, ref: float) -> tuple[str, tuple]:
    t = ms(ts)
    return records.INSERT_CANDLE, records.candle_row(StockCandle(
        symbol, t, sessions.k_at(t), ref, ref + 0.1, ref - 0.1, ref, ref, 1200.0, 700.0, 400.0, 30, 28, 1, True))


def grid(news: bool) -> list[tuple[str, tuple]]:
    """Matured labels at every evaluation minute of session 1 and the three window sessions, the last minute pending,
    with population and linear_prices forecasts, and news in S2, from one seeded generator so both runs share the
    labels and the price forecasts."""
    rng = np.random.default_rng(12345)
    out = []
    days = (date(2026, 10, 19), *WINDOW_DAYS)
    for day in days:
        session = sessions.session_on(day)
        for minute in session.evaluation_minutes():
            t = session.boundary(minute)
            last = day == days[-1] and minute == session.evaluation_minutes()[-1]
            for symbol in SYMBOLS:
                value = float(rng.normal(0.0, 20.0))
                population = 0.3 * value + float(rng.normal(0.0, 5.0))
                linear = float(rng.normal(0.0, 5.0))
                noise = float(rng.normal(0.0, 1.0))
                out.append(records.label_statement(LabelRow(t, symbol, 100.0, None, None, records.PENDING)))
                if not last:
                    out.append(records.label_statement(LabelRow(t, symbol, 100.0, 100.0, value, records.MATURED)))
                out.append((records.INSERT_FORECAST, records.forecast_row(t, symbol, "population",
                                                                          population + (0.5 * noise if news else 0))))
                out.append((records.INSERT_FORECAST, records.forecast_row(t, symbol, "linear_prices", linear)))
                if news:
                    out.append((records.INSERT_FORECAST, records.forecast_row(t, symbol, "news", noise, 1, t - 5000)))
    return out


def status_rows() -> list[tuple[str, tuple]]:
    """Sessions, candles, benchmark, snapshots, events and the two status tables."""
    out = []
    for day in (date(2026, 10, 19), *WINDOW_DAYS):
        out.append((records.PUT_SESSION, records.session_row(sessions.session_on(day), 390, 388)))
    out.append((records.PUT_SPREAD_CHECK, records.spread_check_row(date(2026, 10, 22), "META", 5000, 2.5, 3.0)))
    for minute, ref in (("13:31", 186.0), ("15:00", 186.9), ("19:59", CANARY_PRICE)):
        out.append(candle("AAPL", f"2026-10-22T{minute}:00.000Z", ref))
    out.append(candle("TSLA", "2026-10-22T13:31:00.000Z", 250.0))
    out.append(candle("TSLA", "2026-10-21T19:59:00.000Z", 249.0))
    for i, ts in enumerate(("2026-10-21T14:00:00.000Z", "2026-10-21T19:00:00.000Z", "2026-10-22T19:00:00.000Z")):
        out.append((records.INSERT_BENCHMARK, records.benchmark_row(
            ms(ts), ew_buy_hold=10000.0 + i, qqq_buy_hold=10000.0 - i, matched_equity=10000.5,
            living_mean_equity=9999.0, living_median_equity=9998.5, living=3, twin_mean_equity=9998.0,
            cash_equity=10000.0)))
        for bot_id in (1, 2, 3, 4):
            out.append((records.INSERT_SNAPSHOT, (bot_id, ts, 9990.0 + bot_id, 9990.0 + bot_id + i, 0.0, 0.0, 0.0,
                                                  0.0, 0.0, 0.0, "flat", None, None, None, 0)))
    out.append((records.INSERT_DEATH, (4, "2026-10-22T15:30:00.000Z", 0, "rent", 1, 0, 0, 0, 0, 0, 1, 1, 1, 26.0)))
    out += [
        (records.INSERT_FEED_EVENT, records.local_only_row(ms("2026-10-21T13:20:00.000Z"), True, 0, 0)),
        (records.INSERT_FEED_EVENT, records.local_only_row(ms("2026-10-22T13:20:00.000Z"), False, 1, 0)),
        (records.INSERT_FEED_EVENT, records.feed_event_row(ms("2026-10-22T14:00:00.000Z"), "stale", "TSLA",
                                                           "no eligible print")),
        (records.INSERT_FEED_EVENT, records.feed_event_row(ms("2026-10-22T14:05:00.000Z"), "print_delay", None,
                                                           "median delay 1.2 s")),
        (records.INSERT_CHECKPOINT, ("2026-10-22T20:05:00.000Z", 1, "{}", b"")),
        (records.PUT_RUNTIME_STATUS, records.runtime_status_row(
            ms("2026-10-22T20:09:00.000Z"), feed_state="healthy", session_state="closed", prints_per_s=0.0,
            msgs_per_s=0.5, news_age_s=12.0, living=3, last_checkpoint_ms=ms("2026-10-22T20:05:00.000Z"),
            rss_mb=80.0, rss_peak_mb=95.0)),
    ]
    for symbol in SYMBOLS:
        state = "stale" if symbol == "TSLA" else "healthy"
        out.append((records.PUT_SYMBOL_STATUS, records.symbol_status_row(
            symbol, ms("2026-10-22T20:09:00.000Z"), state, CANARY_PRICE if symbol == "AAPL" else 100.0, 4.0)))
    return out


UNHEALTHY_NEWS = "2026-10-22T14:00:00.000Z"


def run_minute_rows() -> list[tuple[str, tuple]]:
    """S2's run minutes at every evaluation minute of the seeded sessions, a healthy news minute with a valid QQQ
    candle, except the one news minute seeded unhealthy, so evaluate's window ends at the last of them."""
    out = []
    for day in (date(2026, 10, 19), *WINDOW_DAYS):
        session = sessions.session_on(day)
        for minute in session.evaluation_minutes():
            t = session.boundary(minute)
            if iso(t) != UNHEALTHY_NEWS:
                out.append((records.INSERT_NEWS_MINUTE, records.news_minute_row(t, None, "")))
                out.append(candle("QQQ", iso(t), 480.0))
    return out


def seed_run(config: StockConfig, news: bool) -> None:
    """One run database with launch meta, the shared decisions, S2's extra rows and recorded evaluation settings when
    news is set, and the grid."""
    st = StockStore.open(config.db_path)
    out = SHARED + (S2_ONLY if news else []) + grid(news) + status_rows() + (run_minute_rows() if news else [])
    if news:
        out.append((records.INSERT_NEWS_SEEN, records.news_seen_row(
            score_seq=1, item_seq=1, recv_ms=ms("2026-10-22T14:00:02.000Z"), scored_at="2026-10-22T14:00:01.500Z",
            source="finnhub", route="llm", scorer="haiku_v1", status="scored", tickers=["AAPL"], primary_use=True,
            fallback=False)))
        out.append((records.INSERT_NEWS_MINUTE, records.news_minute_row(ms(UNHEALTHY_NEWS), "heartbeat too old", "")))
    st.commit_now(out)
    meta = {"launch_ts": LAUNCH, "experiment_hash": EXPERIMENT, "run_name": config.db_path.parent.name}
    if news:  # as launch_meta records S2's [evaluation]
        meta.update(evaluation_json=json.dumps(dataclasses.asdict(config.evaluation), sort_keys=True),
                    run_minute_rule=config.evaluation.run_minute_rule)
    for key, value in meta.items():
        st.queue_meta(key, value)
    st.flush()
    st.close()


def seed_news(path: Path, spend_alert: bool, heartbeat_ts: str = "2026-10-22T20:09:30.000Z",
              title: str = "Apple supplier lifts its outlook") -> None:
    """news.db with one routed item scored by Haiku and FinBERT, one filtered item and one heartbeat."""
    conn = news_db.open_writer(path)
    try:
        item = {"source": "finnhub", "channel": "finnhub_company", "vendor_id": "1", "kind": "media",
                "received_at": "2026-10-22T14:00:00.000Z", "title": title, "tickers": "AAPL", "relevance": "title",
                "status": "new", "title_hash": "h1", "route": "llm"}
        news_db.insert(conn, "news_items", item)
        news_db.insert(conn, "news_items", {**item, "vendor_id": "2", "title": "Celebrity gossip", "tickers": "",
                                            "relevance": "none", "status": "filtered", "title_hash": "h2",
                                            "route": "none"})
        news_db.insert(conn, "item_scores", {"item_seq": 1, "scorer": "haiku_v1", "route": "llm", "status": "scored",
                                             "attempts": 1, "scored_at": "2026-10-22T14:00:01.500Z"})
        news_db.insert(conn, "item_scores", {"item_seq": 1, "scorer": "finbert_v1", "route": "llm", "status": "scored",
                                             "attempts": 1, "scored_at": "2026-10-22T14:00:00.800Z", "p_pos": 0.7,
                                             "p_neg": 0.1, "p_neu": 0.2})
        news_db.insert(conn, "score_impacts", {"item_seq": 1, "scorer": "haiku_v1", "ticker": "AAPL", "strength": 0.6,
                                               "direction": 1, "surprise": 1, "event_type": "guidance",
                                               "confidence": 3, "horizon": "days", "is_new": 1})
        news_db.insert(conn, "news_heartbeat", {
            "ts": heartbeat_ts, "healthy": 1, "reason": None, "config_hash": "a" * 64, "prompt_sha256": "b" * 64,
            "schema_sha256": "c" * 64, "haiku_state": "on", "sonnet_state": "on", "finbert_state": "on", "waiting": 0,
            "oldest_wait_s": None, "channels_down": "fed_speech", "spent_day_usd": 5.5 if spend_alert else 0.04,
            "spent_month_usd": 7.25, "spend_alert": int(spend_alert)})
    finally:
        conn.close()


def seeded(folder: Path, spend_alert: bool = False) -> Runs:
    """Both run databases and news.db seeded under folder, with the configs that find them."""
    s1, s2, secrets = configs(folder)
    c1, c2 = load_run_pair(s1, s2)
    seed_run(c1, news=False)
    seed_run(c2, news=True)
    seed_news(c2.news_db_path, spend_alert)
    return Runs(s1, s2, c1.db_path, c2.db_path, c2.news_db_path, secrets)
