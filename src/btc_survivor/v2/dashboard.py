"""The private stock dashboard of PLAN_V2 section 13.4 on 127.0.0.1:8060, modelled on the version 1 dashboard.

It serves one self contained page, a JSON view of both runs and news.db, and the decisions explorer, every decision of
both runs paged on the server with filters and the full detail of one. Every database opens read only, the runs
through the stock store's reader and news.db through its own, so a database of another kind is refused, and each
response body is shared by every request for five seconds. The host cannot be configured, port 8050 of the version 1
dashboard is refused, any Host header other than 127.0.0.1 or localhost on its port gets 403, which blocks DNS
rebinding, every method that could write gets 405, and every response carries the version 1 security headers. The
values of the secrets file are registered with the redactor at start and every string of every JSON body passes it,
so no secret reaches the page even if a row held one. Errors on the page name a run or a panel, never a path.

The server keeps UTC alone. The page shows every time in UTC or in the viewer's own zone, as the version 1 page does,
and turns a from or to typed in either zone into the UTC minute the explorer's filters take.

The skill panel uses evaluate's own functions, so it shows what evaluate would print. The window and its cutoff come
from S2's run minutes, the lags and z values from S2's meta, and a row counts only when its label matured by the end
of the window. Every figure is descriptive until the cutoff, and after it S2's news predictor alone carries the pass or
fail word of H1, the first test of the confirmatory sequence, while H2, the test beyond price, is evaluate's alone.

The scorecards panel shows the chooser scorecard of section 11.8 for each run and the day scorecard of section 11.9
through the functions evaluate prints them with, over the same window, with each run's settings from its meta at
launch, descriptive with no pass label. They are imported only when the view is built, as evaluate does, because the
console script that starts the stock and news processes imports this module too. They change only when a label
matures or a session reaches its flatten, so the server rebuilds them at most once a minute.

    btc_survivor_v2 dashboard configs/v2/s1_prices.toml configs/v2/s2_news.toml [--port 8060]
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import statistics
import sys
import threading
import time
from collections.abc import Callable, Hashable
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from btc_survivor.clock import MS_PER_DAY, MS_PER_SECOND, iso, parse_iso
from btc_survivor.dashboard import SECURITY_HEADERS, chart_step, clean
from btc_survivor.logsetup import setup_logging
from btc_survivor.store import StoreError
from btc_survivor.v2 import decisions, evaluate, local_secrets, redact, sessions, status
from btc_survivor.v2.config import (MAX_MONTHLY_USD, PREDICTORS, ConfigError, EvaluationConfig, StockConfig,
                                    load_run_pair, parse_stock_config)
from btc_survivor.v2.local_secrets import SecretsError
from btc_survivor.v2.news import db as news_db
from btc_survivor.v2.news.db import NewsDbError
from btc_survivor.v2.news.health import FINBERT_SCORER
from btc_survivor.v2.sessions import CalendarError
from btc_survivor.v2.stocks import store
from btc_survivor.v2.stocks.news_inputs import HAIKU_SCORER, SCORED_STATUSES, SONNET_SCORER

log = logging.getLogger(__name__)

HOST = "127.0.0.1"   # fixed, never a setting, so nothing outside this machine can reach the page
PORT = 8060
V1_PORT = 8050       # the version 1 dashboard's port, always refused
PAGE = Path(__file__).with_name("dashboard.html")
STATE_TTL_S = 5.0    # each response body is shared by every request for this long
SCORECARDS_TTL_S = 60.0  # the scorecards panel, which reads every label and forecast of the window, is kept this long
CACHE_ENTRIES = 64
MAX_QUERY_FIELDS = 32
RECENT = 25
RECENT_NEWS = 20
BPS_PREDICTORS = ("population", "linear_prices", "linear_news")
SCORERS = {HAIKU_SCORER: "haiku", SONNET_SCORER: "sonnet", FINBERT_SCORER: "finbert"}
SPEND_ALERT = ("The news process raised the daily spend alert, today's model spend passed a quarter of the monthly "
               "cap. Nothing was stopped, so look for a loop, a flood of items or a price change.")
OLD_HEARTBEAT = "The newest news heartbeat is too old, so news health fails in both runs."
NOT_READABLE = "the decisions cannot be read, see the dashboard log"
NO_VIEW = "the dashboard cannot build its view, see the dashboard log"


class DashboardError(ValueError):
    """Raised for a port the private dashboard may not use."""


EXPECTED = (ConfigError, SecretsError, DashboardError, CalendarError, OSError)


def check_port(port: int) -> int:
    """Refuse port 8050 of the version 1 dashboard and anything outside the unprivileged range."""
    if port == V1_PORT:
        raise DashboardError(f"port {V1_PORT} belongs to the version 1 dashboard, use {PORT}")
    if not 1024 <= port <= 65535:
        raise DashboardError(f"port {port} is outside 1024 to 65535")
    return port


def allowed_hosts(port: int) -> frozenset[str]:
    """The only Host headers served, this machine's loopback names on the port the server listens on."""
    return frozenset({f"127.0.0.1:{port}", f"localhost:{port}"})


class TtlCache:
    """Response bodies by key, each built at most once per ttl_s however many requests arrive, keeping the newest
    few keys, so many viewers and repeated filters cost the same as one."""

    def __init__(self, ttl_s: float = STATE_TTL_S, size: int = CACHE_ENTRIES,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl_s, self.size, self.clock = ttl_s, size, clock
        self.entries: dict[Hashable, tuple[float, Any]] = {}
        self.lock = threading.Lock()

    def get(self, key: Hashable, build: Callable[[], Any]) -> Any:
        with self.lock:
            now = self.clock()
            hit = self.entries.get(key)
            if hit is None or now - hit[0] >= self.ttl_s:
                hit = (now, build())
                self.entries.pop(key, None)
                self.entries[key] = hit
                while len(self.entries) > self.size:
                    self.entries.pop(next(iter(self.entries)))
            return hit[1]


def guarded(what: str, build: Callable[[], Any]) -> Any:
    """One part of the view, or an error entry naming the part, so one problem never blanks the page."""
    try:
        return build()
    except Exception:  # noqa: BLE001, the page must show the problem rather than drop every panel
        log.exception("dashboard could not build %s", what)
        return {"error": f"{what} cannot be read, see the dashboard log"}


def scrub(value: Any) -> Any:
    """Every string of a view, keys included, through the redactor, after which a registered secret is only the mask,
    with tuples as lists."""
    if isinstance(value, str):
        return redact.redact(value)
    if isinstance(value, dict):
        return {scrub(key): scrub(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(item) for item in value]
    return value


def body(value: Any) -> bytes:
    """A view as JSON with secrets masked and infinities and NaN as null."""
    return json.dumps(clean(scrub(value)), allow_nan=False, default=str).encode()


def age_s(now_ms: int, ts: str | None) -> float | None:
    return None if ts is None else (now_ms - parse_iso(ts)) / MS_PER_SECOND


# Market session


def times(ms: int) -> dict:
    """One instant in UTC and in New York time with its zone name."""
    return {"utc": iso(ms), "et": datetime.fromtimestamp(ms / MS_PER_SECOND, sessions.new_york()).strftime(
        "%Y-%m-%d %H:%M %Z")}


def market_state(now_ms: int) -> dict:
    """Whether the regular session is open, the next open and close, and the holiday and early close flags."""
    upcoming = sessions.next_session(now_ms)
    is_open = upcoming.open_ms <= now_ms
    following = sessions.next_session(upcoming.close_ms) if is_open else upcoming
    today = sessions.ny_date(now_ms)
    return {"state": "open" if is_open else "closed", "next_open": times(following.open_ms),
            "next_close": times(upcoming.close_ms), "session_day": upcoming.day.isoformat(),
            "early_close": upcoming.early_close,
            "holiday_today": today.weekday() < 5 and sessions.session_on(today) is None}


# Runs


def open_run(config: StockConfig) -> sqlite3.Connection | str:
    """A read only connection to a run database, or the error the page shows in its place."""
    name = config.run.name
    try:
        return store.open_reader(config.db_path)
    except (StoreError, sqlite3.Error, OSError) as exc:
        log.warning("dashboard could not open run %s: %s", name, exc)
        if not config.db_path.exists():
            return f"run {name} has no database yet, run init-db and start the stock process"
        if isinstance(exc, StoreError):
            return f"the database of run {name} is not a stock run database of this schema, see the dashboard log"
        return f"the database of run {name} cannot be opened, see the dashboard log"


def feed(conn: sqlite3.Connection, symbols: tuple[str, ...], now_ms: int) -> dict:
    """The feed panel, the runtime row, each symbol's state and last print age, the newest print delay event and the
    recent feed events."""
    runtime = conn.execute("SELECT * FROM runtime_status WHERE id = 1").fetchone()
    by_symbol = {row["symbol"]: dict(row) for row in conn.execute("SELECT * FROM symbol_status")}
    delay = conn.execute("SELECT ts, symbol, detail FROM feed_events WHERE kind = 'print_delay' ORDER BY id DESC "
                         "LIMIT 1").fetchone()
    rows = [by_symbol.get(s, {"symbol": s, "ts": None, "state": None, "ref_price": None, "last_print_age_s": None})
            for s in symbols]
    return {
        "runtime": None if runtime is None else {**dict(runtime), "age_s": age_s(now_ms, runtime["ts"])},
        "symbols": [{**row, "age_s": age_s(now_ms, row["ts"])} for row in rows],
        "print_delay": None if delay is None else dict(delay),
        "events": [dict(row) for row in conn.execute(
            "SELECT ts, kind, symbol, detail FROM feed_events ORDER BY id DESC LIMIT ?", (RECENT,))],
    }


def prices(conn: sqlite3.Connection, symbols: tuple[str, ...]) -> dict:
    """The newest session's one minute reference price line per symbol, with its last price and session return."""
    day = conn.execute("SELECT max(session_date) FROM stock_candles").fetchone()[0]
    lines: dict[str, list] = {symbol: [] for symbol in symbols}
    for symbol, ts, ref in conn.execute("SELECT symbol, ts, ref_price FROM stock_candles WHERE session_date = ? "
                                        "AND ref_price IS NOT NULL ORDER BY ts", (day,)):
        lines.setdefault(symbol, []).append([ts, ref])
    return {"session_day": day, "symbols": [
        {"symbol": symbol, "last": line[-1][1] if line else None,
         "session_return": line[-1][1] / line[0][1] - 1 if line else None, "line": line}
        for symbol, line in lines.items()]}


def evaluation_window(conn: sqlite3.Connection) -> tuple[EvaluationConfig, evaluate.Window] | None:
    """S2's recorded settings and its evaluation window, found as evaluate finds them, or None before launch."""
    meta = store.read_meta(conn)
    launch = meta.get("launch_ts")
    if launch is None:
        return None
    settings = evaluate.settings_from(meta)
    return settings, evaluate.find_window(evaluate.run_minutes(conn, launch), launch, settings)


def window_view(window: evaluate.Window) -> dict:
    return {"start": window.start, "end": window.end, "final": window.final, "run_minutes": window.run_minutes}


def order(predictor: str) -> tuple[int, str]:
    return (PREDICTORS.index(predictor) if predictor in PREDICTORS else len(PREDICTORS), predictor)


def skill(rows: dict[str, list[tuple]], settings: EvaluationConfig, final: bool, confirmatory: bool) -> list[dict]:
    """Per predictor pooled correlation, t, detectable correlation and upper bound, and for a bps predictor the
    squared error ratio against zero, each through evaluate, with the verdict word descriptive except for S2's news
    predictor once the window has reached its cutoff."""
    out = []
    for predictor in sorted(rows, key=order):
        found = rows[predictor]
        s = evaluate.skill(found, settings)
        mse = (evaluate.squared_error([(t, n, p, 0.0, y) for t, n, p, y in found], settings).mse_ratio
               if predictor in BPS_PREDICTORS else None)
        tested = final and confirmatory and predictor == "news"
        out.append({"predictor": predictor, "rows": s.rows, "minutes": s.minutes, "corr": s.corr, "t": s.t,
                    "detectable": s.detectable, "upper_bound": s.upper_bound, "mse_vs_zero": mse,
                    "left_out": list(s.left_out),
                    "verdict": ("pass" if s.passed else "fail") if tested else "descriptive"})
    return out


def s2_against_s1(s1_rows: dict[str, list[tuple]], s2_rows: dict[str, list[tuple]],
                  settings: EvaluationConfig) -> dict | None:
    """S2's population against S1's on their paired rows through evaluate, a squared error ratio below 1 and a
    positive t favouring S2, or None before both have rows."""
    pairs = evaluate.paired(s2_rows.get("population", []), s1_rows.get("population", []))
    if not pairs:
        return None
    dm = evaluate.squared_error(pairs, settings)
    return {"rows": dm.rows, "minutes": dm.minutes, "mse_ratio": dm.mse_ratio, "t": dm.t}


def equity_curve(conn: sqlite3.Connection) -> list[dict]:
    """The benchmark table thinned to a few hundred points, oldest first, always ending at the newest minute."""
    return [dict(row) for row in conn.execute(
        "SELECT ts, ew_buy_hold, qqq_buy_hold, matched_equity, living_mean_equity, twin_mean_equity, cash_equity "
        "FROM benchmark WHERE (rowid - 1) % ? = 0 OR rowid = (SELECT max(rowid) FROM benchmark) ORDER BY ts",
        (chart_step(conn),))]


def run_state(config: StockConfig, conn: sqlite3.Connection | str, rows: dict,
              judged: tuple[EvaluationConfig, evaluate.Window] | None) -> dict:
    """One run's population panel and skill, or its error entry, so one broken run never blanks the other. rows are
    its forecast rows in the window, or the error entry that reading them gave, and judged S2's settings and window."""
    name = config.run.name
    if isinstance(conn, str):
        return {"name": name, "error": conn}
    meta = guarded(f"the meta of run {name}", lambda: store.read_meta(conn))
    if "error" in rows or judged is None:
        found: Any = rows if "error" in rows else []
    else:
        settings, window = judged
        found = guarded(f"the skill of run {name}",
                        lambda: skill(rows, settings, window.final, config.news_input.use_news))
    return {"name": name, "news": config.news_input.use_news,
            "launch_ts": None if "error" in meta else meta.get("launch_ts"),
            "skill": found,
            "bots": guarded(f"the bots of run {name}", lambda: status.bot_rows(conn)),
            "equity_curve": guarded(f"the equity of run {name}", lambda: equity_curve(conn)),
            "chart_minutes_per_point": guarded(f"the chart of run {name}", lambda: chart_step(conn))}


def launched_config(config: StockConfig, conn: sqlite3.Connection) -> StockConfig:
    """The config a run recorded in its meta at launch, as evaluate reads it, or the config file before launch."""
    meta = store.read_meta(conn)
    return parse_stock_config(meta["config_toml"], config.base_dir) if "config_toml" in meta else config


def scorecards_view(configs: tuple[StockConfig, StockConfig], conns: list[sqlite3.Connection | str],
                    judged: tuple[EvaluationConfig, evaluate.Window]) -> dict:
    """The chooser of each run that opened and the day scorecard, as evaluate builds them over its window, empty
    while a run has no rows yet."""
    # Imported here and never at the top, as in evaluate, so the stock and news processes never load the scorecards.
    from btc_survivor.v2 import scorecards
    settings, window = judged
    sources = [scorecards.Source(config.run.name, conn, launched_config(config, conn))
               for config, conn in zip(configs, conns) if not isinstance(conn, str)]
    chooser = [{"run": card.run, "predictor": card.predictor, "minutes": card.minutes,
                "no_scale_minutes": card.unscaled_minutes, "skipped": card.skipped,
                "strategies": list(card.strategies), "differences": list(card.differences),
                "picks_per_symbol": card.picks_per_symbol}
               for source in sources for card in scorecards.chooser_cards(
                   source, window.start, window.end, settings.newey_west_lags, settings.upper_bound_z)]
    day = scorecards.day_card(sources, window.launch, window.end, settings.upper_bound_z)
    return {"chooser": chooser, "day": {"sessions": day.sessions, "labels": list(day.labels),
                                        "calls": list(day.calls), "symbols": list(day.symbols)},
            "late": {source.name: scorecards.late_forecasts(source, window.start, window.end) for source in sources}}


def cached_scorecards(cache: TtlCache | None, configs: tuple[StockConfig, StockConfig],
                      conns: list[sqlite3.Connection | str],
                      judged: tuple[EvaluationConfig, evaluate.Window]) -> dict:
    """The scorecards view, from the cache while it is younger than a minute, so the page's polling rebuilds it at most
    once a minute. A view that fails to build raises before the cache stores it, so a failure is never kept."""
    if cache is None:
        return scorecards_view(configs, conns, judged)
    window = judged[1]
    return cache.get(("scorecards", window.launch, window.start),
                     lambda: scorecards_view(configs, conns, judged))


# News


def news_heartbeat(conn: sqlite3.Connection, now_ms: int) -> dict | None:
    """The newest heartbeat with its age and its channels in outage as a list."""
    row = conn.execute("SELECT * FROM news_heartbeat ORDER BY seq DESC LIMIT 1").fetchone()
    if row is None:
        return None
    out = dict(row)
    out.update(age_s=age_s(now_ms, row["ts"]), healthy=bool(row["healthy"]), spend_alert=bool(row["spend_alert"]),
               channels_down=[c for c in row["channels_down"].split(",") if c])
    return out


def latencies(conn: sqlite3.Connection, since: str) -> dict[str, float | None]:
    """The median seconds from receipt to a final score, per scorer, for the items received since a time."""
    waits: dict[str, list[float]] = {name: [] for name in SCORERS.values()}
    for scorer, received, scored in conn.execute(
            "SELECT s.scorer, i.received_at, s.scored_at FROM item_scores s JOIN news_items i ON i.seq = s.item_seq "
            "WHERE i.received_at >= ? AND s.status IN (?, ?)", (since, *SCORED_STATUSES)):
        if scorer in SCORERS:
            waits[SCORERS[scorer]].append((parse_iso(scored) - parse_iso(received)) / MS_PER_SECOND)
    return {name: statistics.median(found) if found else None for name, found in waits.items()}


def recent_news(conn: sqlite3.Connection, s2: sqlite3.Connection | None) -> list[dict]:
    """The newest routed items with each scorer's final row and impacts and the score S2 used, if it read one."""
    items = [dict(row) for row in conn.execute(
        "SELECT seq, source, kind, tickers, received_at, title FROM news_items WHERE route != 'none' "
        "ORDER BY seq DESC LIMIT ?", (RECENT_NEWS,))]
    seqs = [item["seq"] for item in items]
    marks = ",".join("?" * len(seqs))
    scores: dict[tuple[int, str], dict] = {}
    for row in conn.execute(f"SELECT item_seq, scorer, status, scored_at, p_pos, p_neg, p_neu FROM item_scores "
                            f"WHERE item_seq IN ({marks})", seqs):
        if row["scorer"] in SCORERS:
            scores[row["item_seq"], SCORERS[row["scorer"]]] = {**dict(row), "impacts": []}
    for row in conn.execute(f"SELECT item_seq, scorer, ticker, strength, direction, surprise, event_type, confidence, "
                            f"horizon, is_new FROM score_impacts WHERE item_seq IN ({marks}) ORDER BY ticker", seqs):
        found = scores.get((row["item_seq"], SCORERS.get(row["scorer"], "")))
        if found is not None:
            found["impacts"].append(dict(row))
    used = {} if s2 is None else {row["item_seq"]: dict(row) for row in s2.execute(
        f"SELECT item_seq, scorer, fallback, recv_at FROM news_seen WHERE primary_use = 1 AND item_seq IN ({marks})",
        seqs)}
    for item in items:
        item["tickers"] = [t for t in item["tickers"].split(",") if t]
        item.update({name: scores.get((item["seq"], name)) for name in SCORERS.values()})
        item["s2_used"] = used.get(item["seq"])
    return items


def news_panel(conn: sqlite3.Connection, s2: sqlite3.Connection | None, now_ms: int, max_age_s: float) -> dict:
    """The news panel, the heartbeat with spend and model states, items per source and the fallback share today, the
    median receipt to scoring time, the alerts the page shows as banners, and the recent items. A heartbeat older than
    max_age_s fails news health on the stock side."""
    since = iso(now_ms - now_ms % MS_PER_DAY)
    heartbeat = news_heartbeat(conn, now_ms)
    alerts = []
    if heartbeat is not None and heartbeat["spend_alert"]:
        alerts.append(SPEND_ALERT)
    if heartbeat is None or heartbeat["age_s"] > max_age_s:
        alerts.append(OLD_HEARTBEAT)
    routed, fallback = conn.execute(
        "SELECT count(*), coalesce(sum(s.status NOT IN (?, ?)), 0) FROM news_items i "
        "JOIN item_scores s ON s.item_seq = i.seq AND s.scorer = ? WHERE i.route = 'llm' AND i.received_at >= ?",
        (*SCORED_STATUSES, HAIKU_SCORER, since)).fetchone()
    return {
        "heartbeat": heartbeat, "alerts": alerts, "monthly_cap_usd": MAX_MONTHLY_USD,
        "items_today": dict(conn.execute("SELECT source, count(*) FROM news_items WHERE received_at >= ? "
                                         "AND status != 'filtered' GROUP BY source ORDER BY source", (since,))),
        "fallback_share": fallback / routed if routed else None,
        "median_scoring_s": latencies(conn, since),
        "recent": recent_news(conn, s2),
    }


def news_state(config: StockConfig, s2: sqlite3.Connection | None, now_ms: int) -> dict:
    """The news panel from news.db opened read only, or its error entry."""
    try:
        conn = news_db.open_reader(config.news_db_path)
    except (NewsDbError, sqlite3.Error, OSError) as exc:
        log.warning("dashboard could not open news.db: %s", exc)
        if not config.news_db_path.exists():
            return {"error": "news.db does not exist yet, run init-db and start the news process"}
        if isinstance(exc, NewsDbError):
            return {"error": "news.db is not a news database of this schema, see the dashboard log"}
        return {"error": "news.db cannot be opened, see the dashboard log"}
    try:
        return guarded("news.db", lambda: news_panel(conn, s2, now_ms, config.news_input.heartbeat_max_age_seconds))
    finally:
        conn.close()


# The whole view


def page_version() -> str:
    """Changes whenever the page file changes, so an open tab reloads itself onto a new version."""
    return str(PAGE.stat().st_mtime_ns)


def build_state(s1: StockConfig, s2: StockConfig, now_ms: int, scorecards_cache: TtlCache | None = None) -> dict:
    """Everything the page shows, every database opened read only for this one view and closed after it, the
    scorecards from scorecards_cache when one is given."""
    conns = [open_run(config) for config in (s1, s2)]
    try:
        live = [conn for conn in conns if not isinstance(conn, str)]
        symbols = s1.stock_feed.symbols
        judged = None if isinstance(conns[1], str) else guarded("the evaluation window",
                                                                 lambda: evaluation_window(conns[1]))
        if isinstance(judged, dict):
            rows, judged = [judged, judged], None
        else:
            rows = [{} if isinstance(conn, str) or judged is None else guarded(
                f"the forecasts of run {c.run.name}",
                lambda conn=conn, c=c: evaluate.scored_rows(conn, judged[1], c.time.horizon_minutes))
                for c, conn in zip((s1, s2), conns)]
        unpaired = len(live) < 2 or judged is None or any("error" in r for r in rows)
        return {
            "now_ms": now_ms, "page_version": page_version(),
            "explorer": {"runs": [s1.run.name, s2.run.name], "symbols": list(symbols),
                         "actions": list(decisions.ACTIONS), "kinds": list(decisions.KINDS),
                         "page_rows": decisions.PAGE_ROWS},
            "market": guarded("the market session", lambda: market_state(now_ms)),
            "feed": guarded("the feed", lambda: feed(live[0], symbols, now_ms)) if live else None,
            "prices": guarded("the prices", lambda: prices(live[0], symbols)) if live else None,
            "news": news_state(s2, conns[1] if not isinstance(conns[1], str) else None, now_ms),
            "window": None if judged is None else window_view(judged[1]),
            "runs": [run_state(c, conn, r, judged) for c, conn, r in zip((s1, s2), conns, rows)],
            "s2_against_s1": None if unpaired else guarded("S2 against S1",
                                                           lambda: s2_against_s1(*rows, judged[0])),
            "scorecards": None if judged is None else guarded(
                "the scorecards", lambda: cached_scorecards(scorecards_cache, (s1, s2), conns, judged)),
        }
    finally:
        for conn in conns:
            if not isinstance(conn, str):
                conn.close()


# The decisions explorer


def with_runs(s1: StockConfig, s2: StockConfig, read: Callable[[list[sqlite3.Connection | None]], dict]) -> dict:
    """Open both run databases read only, hand the open ones to read, close them, and list the runs that failed."""
    opened = [open_run(config) for config in (s1, s2)]
    try:
        out = read([None if isinstance(conn, str) else conn for conn in opened])
    finally:
        for conn in opened:
            if not isinstance(conn, str):
                conn.close()
    return {**out, "errors": [conn for conn in opened if isinstance(conn, str)]}


def decisions_reply(s1: StockConfig, s2: StockConfig, query: str, cache: TtlCache) -> tuple[int, bytes]:
    """One page of the decisions explorer for a query string, or 400 naming a filter it cannot use."""
    names = (s1.run.name, s2.run.name)
    try:
        filters = decisions.parse_filters(parse_qs(query, max_num_fields=MAX_QUERY_FIELDS), names,
                                          s1.stock_feed.symbols)
    except ValueError as exc:
        return 400, body({"error": str(exc) if isinstance(exc, decisions.FilterError) else "the query is too long"})

    def build() -> tuple[int, bytes]:
        try:
            return 200, body(with_runs(s1, s2, lambda conns: decisions.page(conns, names, filters)))
        except Exception:  # noqa: BLE001, one bad query must answer with an error rather than drop the connection
            log.exception("dashboard could not read the decisions")
            return 500, body({"error": NOT_READABLE})

    return cache.get(("decisions", filters), build)


def decision_reply(s1: StockConfig, s2: StockConfig, query: str, cache: TtlCache) -> tuple[int, bytes]:
    """The full detail of one decision named by run and id, 400 for a bad query and 404 when there is none."""
    names = (s1.run.name, s2.run.name)
    try:
        values = parse_qs(query, max_num_fields=MAX_QUERY_FIELDS)
        if set(values) != {"run", "id"} or len(values["run"]) != 1 or len(values["id"]) != 1:
            raise decisions.FilterError("a decision is named by run and id, each given once")
        run = decisions.choice("run", values["run"][0], names)
        decision_id = decisions.number("id", values["id"][0], decisions.MAX_ID)
    except ValueError as exc:
        return 400, body({"error": str(exc) if isinstance(exc, decisions.FilterError) else "the query is too long"})

    def build() -> tuple[int, bytes]:
        index = names.index(run)
        try:
            found = with_runs(s1, s2, lambda conns: {"decision": None if conns[index] is None
                                                      else decisions.detail(conns[index], run, decision_id)})
        except Exception:  # noqa: BLE001, as above
            log.exception("dashboard could not read decision %s of run %s", decision_id, run)
            return 500, body({"error": NOT_READABLE})
        if found["decision"] is None:
            return 404, body({"error": f"run {run} has no decision {decision_id}", "errors": found["errors"]})
        return 200, body(found)

    return cache.get(("decision", run, decision_id), build)


# The server


def wall_ms() -> int:
    return time.time_ns() // 1_000_000


def make_handler(s1: StockConfig, s2: StockConfig,
                 now_ms: Callable[[], int] = wall_ms) -> type[BaseHTTPRequestHandler]:
    """The request handler for both runs, reading the time from now_ms so a test can fix it."""
    cache, cards = TtlCache(), TtlCache(SCORECARDS_TTL_S, size=2)

    def state() -> tuple[int, bytes]:
        try:
            return 200, body(build_state(s1, s2, now_ms(), cards))
        except Exception:  # noqa: BLE001, the page must get an answer it can show
            log.exception("dashboard could not build its view")
            return 500, body({"error": NO_VIEW})

    class Handler(BaseHTTPRequestHandler):
        def host_allowed(self) -> bool:
            """Whether the Host header names this machine on the server's own port, never another name."""
            host = (self.headers.get("Host") or "").strip().lower()
            return host in allowed_hosts(self.server.server_address[1])

        def do_GET(self) -> None:
            url = urlsplit(self.path)
            if not self.host_allowed():
                self.reply(403, b"forbidden", "text/plain")
            elif url.path in ("/", "/index.html") and not url.query:
                self.reply(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/state" and not url.query:
                self.reply(*cache.get("state", state), "application/json")
            elif url.path == "/api/decisions":
                self.reply(*decisions_reply(s1, s2, url.query, cache), "application/json")
            elif url.path == "/api/decision":
                self.reply(*decision_reply(s1, s2, url.query, cache), "application/json")
            else:
                self.reply(404, b"not found", "text/plain")

        def refuse(self) -> None:
            """Fail every method that could write, with 403 first for a foreign Host."""
            if not self.host_allowed():
                self.reply(403, b"forbidden", "text/plain")
            else:
                self.reply(405, b"read only", "text/plain", {"Allow": "GET"})

        do_POST = do_PUT = do_PATCH = do_DELETE = refuse

        def reply(self, code: int, payload: bytes, kind: str, extra: dict[str, str] | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            for name, value in (extra or {}).items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def end_headers(self) -> None:
            """The version 1 security headers and no caching on every response, error pages included."""
            for name, value in SECURITY_HEADERS.items():
                self.send_header(name, value)
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, fmt: str, *args: Any) -> None:
            log.debug("dashboard %s", fmt % args)

    return Handler


def make_server(s1: StockConfig, s2: StockConfig, port: int = PORT) -> ThreadingHTTPServer:
    """A server bound to 127.0.0.1 alone on an allowed port."""
    return ThreadingHTTPServer((HOST, check_port(port)), make_handler(s1, s2))


def serve(server: ThreadingHTTPServer) -> None:
    """Serve until Ctrl+C on 127.0.0.1 only, so nothing outside this machine can reach it."""
    print(f"private dashboard at http://{HOST}:{server.server_address[1]}, local only, Ctrl+C stops it", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


DESCRIPTION = (f"Serve the private stock dashboard on {HOST} only, never tunnelled, shared or published. Every "
               f"database opens read only.")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The dashboard arguments, for this module's own parser or the btc_survivor_v2 subcommand. There is no host."""
    parser.add_argument("s1", metavar="S1", type=Path, help="the prices run config, such as configs/v2/s1_prices.toml")
    parser.add_argument("s2", metavar="S2", type=Path, help="the news run config, such as configs/v2/s2_news.toml")
    parser.add_argument("--port", type=int, default=PORT, help=f"the port on {HOST} (default {PORT}, never {V1_PORT})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="btc_survivor_v2 dashboard", description=DESCRIPTION)
    add_arguments(parser)
    return parser


def run(args: argparse.Namespace) -> int:
    """Check the configs and the port, register the secrets for redaction, bind and serve until Ctrl+C, or print an
    expected failure as one error line and exit 2."""
    try:
        s1, s2 = load_run_pair(args.s1, args.s2)
        check_port(args.port)
        local_secrets.load_secrets(s1.path(s1.run.secrets_file))
        server = make_server(s1, s2, args.port)
    except EXPECTED as exc:
        print(f"error: {redact.redact(str(exc))}", file=sys.stderr)
        return 2
    setup_logging(s1.run.log_level)
    redact.install()
    serve(server)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the dashboard from the command line."""
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
