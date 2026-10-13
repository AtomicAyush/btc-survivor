"""A read only local dashboard for one or more runs, served on this machine only.

It serves one self contained page and a JSON view built from the same read only queries as the status command, so
it can never change a run. Two more JSON views back the decisions explorer, one page of filtered decisions and one
decision with the order, fills, trade and label it led to, built from fixed SQL with every filter as a parameter. The
explorer also lists each turn a learner skipped, a phase minute it lived through without a decision row, with the
cause the database shows for it, paged together with the decisions through a temporary table on the read only
connection. It binds to 127.0.0.1, refuses any request whose Host header names something other than 127.0.0.1 or
localhost on its own port, and needs nothing beyond the project's own dependencies.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import threading
import time
import urllib.parse
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from btc_survivor.cli import data, self_financing
from btc_survivor.cli.evaluate import SKILL_LAGS, forecasts
from btc_survivor.cli.stats import diebold_mariano, skill_test
from btc_survivor.clock import MS_PER_DAY, MS_PER_HOUR, MS_PER_MINUTE, iso, parse_iso
from btc_survivor.config import Config
from btc_survivor.features import VOLUME_WINDOW

log = logging.getLogger(__name__)

PAGE = Path(__file__).with_name("dashboard.html")
STATE_TTL_S = 5.0  # one computed view is shared by every request for this long, so many viewers cost the same as one
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                               "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; "
                               "frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
LOCAL_NAMES = ("127.0.0.1", "localhost")  # the only names a request may address, with the serving port
CHART_POINTS = 400
RECENT = 25

# The decisions explorer. ACTIONS is every action bot.decide and the twins' mirror write, ORDER_ACTIONS those that
# send an order. The limits keep one request cheap however long a run grows.
ACTIONS = ("none", "open_long", "open_short", "close", "flip", "hold")
ORDER_ACTIONS = ("open_long", "open_short", "close", "flip")
OUTCOMES = ("matured", "right", "wrong", "waiting", "missed")
TWIN_ORIGIN = "control"
PAGE_SIZE = 50
MAX_PAGE_SIZE = 200
MAX_PAGE = 100_000
MAX_SEARCH = 100
MAX_HOURS = 24 * 366
MAX_ID = 2 ** 62
QUERY_BUDGET_S = 3.0  # a query still running after this is stopped and answered with 503
PROGRESS_STEPS = 10_000  # SQLite virtual machine steps between checks of that budget
CACHE_ENTRIES = 128
LIST_PARAMS = frozenset({"run", "bot", "kind", "action", "forced", "gate", "outcome", "q", "from", "to", "hours",
                         "page", "per"})
DETAIL_PARAMS = frozenset({"run", "id"})
_INT = re.compile(r"[0-9]{1,19}")
_WHEN = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2})(?:[T ]([0-9]{2}:[0-9]{2})(?::([0-9]{2})(?:\.([0-9]{1,6}))?)?)?Z?")

# A decision is scored by its own call. A learner's call is its prediction. A twin's row copies its learner's
# prediction, so its call is the side the coin chose, read from the action of an open and from the reason that
# TwinBot.mirror writes for a flip, which names the coin's side even when the order is later rejected. A twin's close
# or wait makes no call. CALL_SQL is positive for long and negative for short, and CALL_ARGS carry every value.
COIN_SAYS = "the coin chose "
CALL_SQL = ("CASE WHEN b.origin != ? THEN d.prediction_bps "
            "WHEN d.action = ? OR (d.action = ? AND d.reason LIKE ?) THEN ? "
            "WHEN d.action = ? OR (d.action = ? AND d.reason LIKE ?) THEN ? END")
CALL_ARGS = (TWIN_ORIGIN, "open_long", "flip", "%" + COIN_SAYS + "long", 1, "open_short", "flip",
             "%" + COIN_SAYS + "short", -1)

# Skipped turns. A learner decides only at its phase minute, and only in a healthy minute where its prediction exists,
# so a phase minute it lived through with no decision row is a turn it skipped. Each one is listed beside the
# decisions with the action SKIPPED, read from a temporary table on the read only connection so the same filters,
# order and paging apply to both. Its id is minus (minute index times SKIP_ID_BOTS plus the bot id), which never
# collides with a decision id and stays inside the integers JavaScript holds exactly.
SKIPPED = "skipped"
LIST_ACTIONS = ACTIONS + (SKIPPED,)
SKIP_ID_BOTS = 1_000_000
SKIP_LAST_MINUTE = int(datetime(9999, 1, 1, tzinfo=timezone.utc).timestamp()) // 60 - 1  # 9998 is the last year
SKIP_ID_MAX = SKIP_LAST_MINUTE * SKIP_ID_BOTS + SKIP_ID_BOTS - 1  # below 2 ** 53, so JavaScript holds it exactly
SKIP_CACHE_ENTRIES = 16
CAUSES = ("not_running", "not_healthy", "inputs", "unknown")
STATE_KINDS = ("healthy", "recovering", "stale")
RESTORED = "restored the checkpoint from "  # how Engine._restore words its resume event
NEAR_BEFORE_MS, NEAR_AFTER_MS = 15 * MS_PER_MINUTE, 5 * MS_PER_MINUTE  # the feed events a skipped turn shows
NEAR_BEFORE_EVENTS, NEAR_AFTER_EVENTS = 30, 10  # at most, the nearest to its minute on each side
_SKIP_ID = re.compile(r"-[0-9]{1,16}")
ROW_COLUMNS = ("id, bot_id, ts, position_before, prediction_bps, required_edge_bps, gate_passed, forced, action, "
               "reason, label_bps")
ROWS_SQL = ("(SELECT " + ROW_COLUMNS + ", NULL AS cause FROM decisions UNION ALL SELECT " + ROW_COLUMNS +
            ", cause FROM temp.skipped_turns) d JOIN bots b USING (bot_id)")
SKIP_TABLE_SQL = ("CREATE TEMP TABLE IF NOT EXISTS skipped_turns (id INTEGER PRIMARY KEY, bot_id INTEGER NOT NULL, "
                  "ts TEXT NOT NULL, position_before TEXT, prediction_bps REAL, required_edge_bps REAL, "
                  "gate_passed INTEGER, forced INTEGER, action TEXT NOT NULL, reason TEXT NOT NULL, label_bps REAL, "
                  "cause TEXT NOT NULL)")
SKIP_INSERT_SQL = ("INSERT INTO temp.skipped_turns (id, bot_id, ts, position_before, action, reason, cause) "
                   "VALUES (?, ?, ?, ?, ?, ?, ?)")

LIST_SQL = ("SELECT d.id, d.ts, d.bot_id, b.origin, b.generation, d.position_before, d.prediction_bps, "
            "d.required_edge_bps, d.gate_passed, d.forced, d.action, d.reason, d.label_bps, d.cause, " + CALL_SQL +
            " AS direction FROM " + ROWS_SQL)
COUNT_SQL = ("SELECT action, count(*) AS n, sum(forced) AS forced, sum(gate_passed) AS gate, "
             "count(label_bps) AS matured, sum(label_bps IS NULL AND ts > ?) AS waiting, "
             "sum(direction * label_bps > 0) AS n_right, sum(direction * label_bps < 0) AS n_wrong "
             "FROM (SELECT d.action, d.forced, d.gate_passed, d.label_bps, d.ts, " + CALL_SQL +
             " AS direction FROM " + ROWS_SQL)
FILLS_SQL = ("SELECT id, trade_id, leg, side, qty, ref_mid, book_avg_price, price, levels_consumed, slippage_bps, fee, "
             "latency_ms, decided_at, filled_at, status, reason FROM fills "
             "WHERE bot_id = ? AND decided_at = ? AND (reason = ? OR status = ?) ORDER BY id")
PAIR_SQL = ("SELECT d.id, d.bot_id, d.action FROM decisions d WHERE d.ts = ? AND d.id != ? AND d.bot_id IN "
            "(SELECT bot_id FROM bots WHERE slot = ? AND (origin = ?) = ?) ORDER BY d.id LIMIT 1")


def run_state(cfg: Config, now_ms: int, name: str | None = None) -> dict:
    """Everything the page shows for one run, or an error entry, so one broken run never blanks the others."""
    name = name or cfg.data_dir.name
    try:
        conn = data.connect(cfg.db_path)
    except FileNotFoundError:
        return {"name": name, "error": f"run {name} has no database yet, has it been started"}
    except sqlite3.Error as exc:
        log.warning("dashboard could not open %s: %s", cfg.db_path, exc)
        return {"name": name, "error": f"the database of run {name} cannot be opened, see the dashboard log"}
    try:
        return _state(conn, cfg, name, now_ms)
    except Exception:  # noqa: BLE001, a report must show the problem rather than drop every run
        log.exception("dashboard could not read %s", cfg.db_path)
        return {"name": name, "error": f"the database of run {name} cannot be read, see the dashboard log"}
    finally:
        conn.close()


def _state(conn, cfg: Config, name: str, now_ms: int) -> dict:
    meta = data.checkpoint_meta(conn)
    live = data.runtime(conn, now_ms)
    rows = data.bots(conn, cfg, meta, live, now_ms)
    return {
        "name": name, "summary": data.summary(conn, cfg, meta, rows, live, now_ms), "bots": rows,
        "self_financing": self_financing.value_at(conn, cfg),
        "config": {"fee_bps": cfg.costs.taker_fee_bps, "start_equity": cfg.trading.start_equity,
                   "rent_per_day": cfg.costs.daily_running_cost, "candle_minutes": 1,
                   "force_hours": cfg.trading.force_trade_after_hours,
                   "notional": cfg.trading.notional, "horizon_minutes": cfg.time.horizon_minutes,
                   "history_minutes": cfg.time.history_minutes_required},
        "history": {"valid": meta["valid_total"] if meta else 0, "labels": meta["label_scale_n"] if meta else 0,
                    "labels_needed": cfg.features.min_samples},
        "equity_curve": equity_curve(conn), "chart_minutes_per_point": chart_step(conn), "skill": skill_so_far(conn, cfg.time.horizon_minutes),
        "decisions": [dict(r) for r in conn.execute(
            "SELECT d.ts, d.bot_id, b.origin, d.action, d.forced, d.prediction_bps, d.required_edge_bps, d.reason "
            "FROM decisions d JOIN bots b USING (bot_id) WHERE d.action != 'none' ORDER BY d.id DESC LIMIT ?",
            (RECENT,))],
        "fills": [dict(r) for r in conn.execute(
            "SELECT filled_at, decided_at, bot_id, leg, side, price, fee, status, reason FROM fills "
            "ORDER BY id DESC LIMIT ?", (RECENT,))],
        "events": [dict(r) for r in conn.execute(
            "SELECT ts, kind, detail FROM feed_events ORDER BY id DESC LIMIT ?", (RECENT,))],
    }


def chart_step(conn) -> int:
    """How many healthy minutes each chart point stands for, one until the run outgrows CHART_POINTS."""
    return max(1, math.ceil(conn.execute("SELECT count(*) FROM benchmark").fetchone()[0] / CHART_POINTS))


def equity_curve(conn) -> list[dict]:
    """The benchmark table thinned to a few hundred points, oldest first, always ending at the newest minute."""
    step = chart_step(conn)
    rows = conn.execute("SELECT ts, btc_mid, buy_hold_equity, matched_equity, living_mean_equity, twin_mean_equity, "
                        "cash_equity FROM benchmark WHERE (rowid - 1) % ? = 0 OR rowid = (SELECT max(rowid) "
                        "FROM benchmark) ORDER BY ts", (step,)).fetchall()
    return [dict(r) for r in rows]


def skill_so_far(conn, horizon: int) -> dict:
    """Descriptive skill on the same rows and with the same statistics as criterion 1. Only the final day counts.

    Beside the correlation come its Newey West t statistic, which passes above 2.33, the one sided 95 percent upper
    bound and the correlation detectable with 90 percent power from the realised Newey West error, as evaluate prints.
    """
    out = {}
    for name, rows in forecasts(conn, horizon).items():
        pred = np.array([p for p, _ in rows.values()])
        label = np.array([y for _, y in rows.values()])
        test = skill_test(pred, label, SKILL_LAGS)
        out[name] = {"windows": len(pred), "corr": test["corr"], "t": test["t"], "passed": test["passed"],
                     "upper_95": test["upper_95"], "detectable_nw": test["detectable_nw"],
                     "mse_vs_zero": diebold_mariano(label - pred, label, SKILL_LAGS)["mse_ratio"]}
    return out


def clean(value):
    """Replace infinities and NaN with None so the page always receives valid JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def page_version() -> str:
    """Changes whenever the page file changes, so an open tab can reload itself onto a new version."""
    return str(PAGE.stat().st_mtime_ns)


def run_names(configs: list[Config]) -> list[str]:
    """Tab names from the data folders, numbered when two runs share a folder name so every run can be opened."""
    names, seen = [], Counter(c.data_dir.name for c in configs)
    used: Counter = Counter()
    for c in configs:
        base = c.data_dir.name
        used[base] += 1
        names.append(base if seen[base] == 1 else f"{base} {used[base]}")
    return names


class BadRequest(ValueError):
    """A request the explorer refuses, answered with 400 and this plain message."""


class NotFound(LookupError):
    """A run or decision that does not exist, answered with 404."""


class TooSlow(RuntimeError):
    """A query that ran past QUERY_BUDGET_S, answered with 503."""


@dataclass(frozen=True)
class Filters:
    """One validated explorer request. Every value reaches SQL only as a parameter."""

    bot: int | None = None
    kind: str | None = None
    actions: tuple[str, ...] | None = None
    forced: int | None = None
    gate: int | None = None
    outcome: str | None = None
    q: str | None = None
    since: str | None = None
    until: str | None = None
    hours: int | None = None
    page: int = 1
    per: int = PAGE_SIZE


def read_params(query: str, allowed: frozenset[str]) -> dict[str, str]:
    """The query string as single values, refusing unknown or repeated names. Blank values count as absent."""
    try:
        pairs = urllib.parse.parse_qsl(query, keep_blank_values=True, strict_parsing=True,
                                       max_num_fields=len(allowed), errors="strict")
    except ValueError:
        raise BadRequest("the query string cannot be read, or it holds too many parameters") from None
    out: dict[str, str] = {}
    for key, value in pairs:
        if key not in allowed:
            raise BadRequest(f"unknown parameter {key[:40]!r}, the known ones are {', '.join(sorted(allowed))}")
        if key in out:
            raise BadRequest(f"parameter {key} is given twice")
        if value.strip():
            out[key] = value.strip()
    return out


def _int(params: dict[str, str], key: str, low: int, high: int, default: int | None = None) -> int | None:
    text = params.get(key)
    if text is None:
        return default
    if not _INT.fullmatch(text) or not low <= int(text) <= high:
        raise BadRequest(f"{key} must be a whole number from {low} to {high:,}")
    return int(text)


def detail_id(params: dict[str, str]) -> int:
    """A decision's id, or a skipped turn's negative id, which must name a minute from 1970 to 9998 and a bot."""
    text = params.get("id")
    if text is None:
        raise BadRequest("id is required, the number of one decision")
    if not text.startswith("-"):
        return _int(params, "id", 1, MAX_ID)
    value = -int(text[1:]) if _SKIP_ID.fullmatch(text) else 0
    minute, bot_id = split_skip_id(value)
    if value == 0 or -value > SKIP_ID_MAX or minute < 1 or bot_id < 1:
        raise BadRequest(f"a skipped turn's id is minus its minute index times {SKIP_ID_BOTS:,} plus its bot id, "
                         f"from -{SKIP_ID_BOTS + 1:,} to -{SKIP_ID_MAX:,}")
    return value


def _choice(params: dict[str, str], key: str, options: tuple[str, ...]) -> str | None:
    text = params.get(key)
    if text is not None and text not in options:
        raise BadRequest(f"{key} must be one of {', '.join(options)}")
    return text


def parse_when(text: str, key: str, end: bool = False) -> str:
    """A UTC time as YYYY-MM-DD with an optional HH:MM, seconds and fraction, in the stored ISO form.

    A bare date given as an end means the end of that day, so to=2026-10-12 includes the whole day.
    """
    match = _WHEN.fullmatch(text)
    if not match:
        raise BadRequest(f"{key} must be a UTC time like 2026-10-12 07:30")
    day, hm, sec, frac = match.groups()
    try:
        moment = datetime.strptime(f"{day}T{hm or '00:00'}:{sec or '00'}", "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        raise BadRequest(f"{key} must be a UTC time like 2026-10-12 07:30") from None
    if not 1970 <= moment.year <= 9998:
        raise BadRequest(f"{key} must be a UTC time between the years 1970 and 9998")
    ms = int(moment.replace(tzinfo=timezone.utc).timestamp()) * 1000 + int((frac or "0")[:3].ljust(3, "0"))
    if end and hm is None:
        ms += MS_PER_DAY - 1
    return iso(ms)


def parse_filters(params: dict[str, str]) -> Filters:
    actions = None
    if "action" in params:
        actions = tuple(dict.fromkeys(a.strip() for a in params["action"].split(",")))
        if not all(a in LIST_ACTIONS for a in actions):
            raise BadRequest(f"action must be one or more of {', '.join(LIST_ACTIONS)}, separated by commas")
    flags = {}
    for key in ("forced", "gate"):
        flags[key] = _int(params, key, 0, 1)
    q = params.get("q")
    if q is not None and (len(q) > MAX_SEARCH or any(ord(c) < 32 for c in q)):
        raise BadRequest(f"q must be plain text of at most {MAX_SEARCH} characters")
    since = parse_when(params["from"], "from") if "from" in params else None
    until = parse_when(params["to"], "to", end=True) if "to" in params else None
    if since is not None and until is not None and since > until:
        raise BadRequest("from must not be later than to")
    return Filters(
        bot=_int(params, "bot", 1, MAX_ID), kind=_choice(params, "kind", ("learner", "twin")), actions=actions,
        forced=flags["forced"], gate=flags["gate"], outcome=_choice(params, "outcome", OUTCOMES), q=q,
        since=since, until=until, hours=_int(params, "hours", 1, MAX_HOURS),
        page=_int(params, "page", 1, MAX_PAGE, 1), per=_int(params, "per", 1, MAX_PAGE_SIZE, PAGE_SIZE))


def run_clock(conn) -> str | None:
    """The run's newest healthy minute. Labels and benchmark rows are committed together, so it dates them too."""
    clock = conn.execute("SELECT max(ts) FROM benchmark").fetchone()[0]
    return clock or conn.execute("SELECT max(ts) FROM decisions").fetchone()[0]


def label_cutoff(clock: str | None, horizon_ms: int) -> str:
    """Decisions after this can still get a label. An earlier one without a label never will."""
    return "" if clock is None else iso(parse_iso(clock) - horizon_ms)


def _like(text: str) -> str:
    return "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def where(f: Filters, clock: str | None, horizon_ms: int, with_actions: bool = True) -> tuple[str, list]:
    """The WHERE clause from fixed fragments, and the parameters that carry every value."""
    parts: list[str] = []
    args: list = []

    def add(fragment: str, *values) -> None:
        parts.append(fragment)
        args.extend(values)

    if f.bot is not None:
        add("d.bot_id = ?", f.bot)
    if f.kind is not None:
        add("b.origin = ?" if f.kind == "twin" else "b.origin != ?", TWIN_ORIGIN)
    if with_actions and f.actions:
        add("d.action IN (" + ", ".join("?" * len(f.actions)) + ")", *f.actions)
    if f.forced is not None:
        add("d.forced = ?", f.forced)
    if f.gate is not None:
        add("d.gate_passed = ?", f.gate)  # a skipped turn has no gate or forced flag, so these leave it out
    cutoff = label_cutoff(clock, horizon_ms)
    if f.outcome is not None:
        add("d.action != ?", SKIPPED)  # nor a label, so no label filter keeps it
    if f.outcome == "matured":
        add("d.label_bps IS NOT NULL")
    elif f.outcome == "right":
        add("(" + CALL_SQL + ") * d.label_bps > 0", *CALL_ARGS)
    elif f.outcome == "wrong":
        add("(" + CALL_SQL + ") * d.label_bps < 0", *CALL_ARGS)
    elif f.outcome == "waiting":
        add("d.label_bps IS NULL AND d.ts > ?", cutoff)
    elif f.outcome == "missed":
        add("d.label_bps IS NULL AND d.ts <= ?", cutoff)
    if f.q is not None:
        add("d.reason LIKE ? ESCAPE '\\'", _like(f.q))
    if f.since is not None:
        add("d.ts >= ?", f.since)
    if f.until is not None:
        add("d.ts <= ?", f.until)
    if f.hours is not None and clock is not None:
        add("d.ts > ?", iso(parse_iso(clock) - f.hours * MS_PER_HOUR))
    return (" WHERE " + " AND ".join(parts)) if parts else "", args


def _right(direction: float | None, label: float | None) -> bool | None:
    """Whether a prediction or position pointed the way Bitcoin went, unknown while either side is zero or missing."""
    if direction is None or label is None or direction * label == 0:
        return None
    return direction * label > 0


def coin_side(action: str, reason: str | None) -> str | None:
    """The side a twin's coin chose for this decision, the same reading as CALL_SQL, or None for a close or a wait."""
    if action in ("open_long", "open_short"):
        return action[len("open_"):]
    if action == "flip":
        return next((s for s in ("long", "short") if (reason or "").lower().endswith(COIN_SAYS + s)), None)
    return None


def decision_page(conn, cfg: Config, f: Filters) -> dict:
    """One page of decisions and skipped turns, newest first, with the total, counts per action under the other
    filters and a summary.

    Learners are scored by their prediction and twins by the side their coin chose, so right and wrong judge each
    bot's own call. Waiting and missed together are the decisions whose label has not matured. A skipped turn counts
    toward the total and its own skipped figure only, never toward forced, gate, matured, waiting, missed, right or
    wrong.
    """
    horizon_ms = cfg.time.horizon_minutes * MS_PER_MINUTE
    skips, skip_error = listed_skips(conn, cfg)
    load_skips(conn, skips)
    clock = run_clock(conn)
    cutoff = label_cutoff(clock, horizon_ms)
    clause, args = where(f, clock, horizon_ms, with_actions=False)
    counts = {a: 0 for a in LIST_ACTIONS}
    summary = {"total": 0, "skipped": 0, "forced": 0, "gate_passed": 0, "matured": 0, "waiting": 0, "missed": 0,
               "right": 0, "wrong": 0}
    for r in conn.execute(COUNT_SQL + clause + ") GROUP BY action", [cutoff, *CALL_ARGS, *args]):
        counts[r["action"]] = r["n"]
        if f.actions is not None and r["action"] not in f.actions:
            continue
        summary["total"] += r["n"]
        if r["action"] == SKIPPED:
            summary["skipped"] += r["n"]
            continue
        for key, col in (("forced", "forced"), ("gate_passed", "gate"), ("matured", "matured"),
                         ("waiting", "waiting"), ("right", "n_right"), ("wrong", "n_wrong")):
            summary[key] += r[col] or 0
    summary["missed"] = summary["total"] - summary["skipped"] - summary["matured"] - summary["waiting"]
    pages = max(1, math.ceil(summary["total"] / f.per))
    page = min(f.page, pages)
    clause, args = where(f, clock, horizon_ms)
    rows = []
    for r in conn.execute(LIST_SQL + clause + " ORDER BY d.ts DESC, d.id DESC LIMIT ? OFFSET ?",
                          [*CALL_ARGS, *args, f.per, (page - 1) * f.per]):
        row = dict(r)
        direction = row.pop("direction")
        row["kind"] = "twin" if row.pop("origin") == TWIN_ORIGIN else "learner"
        if row["action"] == SKIPPED:
            row["label_state"] = row["coin_side"] = row["right"] = None
        else:
            row["label_state"] = "matured" if row["label_bps"] is not None else \
                "waiting" if row["ts"] > cutoff else "missed"
            row["coin_side"] = coin_side(row["action"], row["reason"]) if row["kind"] == "twin" else None
            row["right"] = _right(direction, row["label_bps"])
        rows.append(row)
    filters = {k: v for k, v in asdict(f).items() if v is not None and k not in ("page", "per")}
    return {"clock": clock, "horizon_minutes": cfg.time.horizon_minutes, "page": page, "per": f.per, "pages": pages,
            "total": summary["total"], "action_counts": counts, "summary": summary, "filters": filters,
            "skipped_error": skip_error, "rows": rows}


def listed_skips(conn, cfg: Config) -> tuple[list, bool]:
    """The skipped turns to list, and whether working them out failed.

    A failure is logged and the decisions still list without the skipped turns, so a surprise in a live database
    never hides the explorer that worked before them. Running out of time stops the whole answer, as any slow query
    does, since every later query of it would be stopped too.
    """
    try:
        return list(skipped_turns(conn, cfg).values()), False
    except sqlite3.OperationalError as exc:
        if "interrupt" in str(exc):
            log.warning("working out the skipped turns of %s took longer than %.0f s", cfg.db_path, QUERY_BUDGET_S)
            raise TooSlow("working out the skipped turns took too long, see the dashboard log") from None
        log.exception("dashboard could not work out the skipped turns of %s", cfg.db_path)
    except Exception:  # noqa: BLE001, the decisions still list, and the log gets the traceback
        log.exception("dashboard could not work out the skipped turns of %s", cfg.db_path)
    return [], True


def decision_detail(conn, cfg: Config, decision_id: int) -> dict:
    """One decision with every column, its bot, the fills of the order it sent, the trades they touched and its label.

    A decision's order carries the bot and the decision minute, and its fills keep the reason decision, or a rejection
    reason when refused, so a stop or death close in the same minute is never mistaken for it. A negative id names a
    skipped turn, answered by skipped_detail.
    """
    if decision_id < 0:
        return skipped_detail(conn, cfg, decision_id)
    row = conn.execute("SELECT * FROM decisions WHERE id = ?", (decision_id,)).fetchone()
    if row is None:
        raise NotFound(f"there is no decision {decision_id} in this run")
    dec = dict(row)
    horizon_ms = cfg.time.horizon_minutes * MS_PER_MINUTE
    bot = conn.execute("SELECT bot_id, slot, parent_id, generation, origin, state, born_at, died_at, margin_bps, lr "
                       "FROM bots WHERE bot_id = ?", (dec["bot_id"],)).fetchone()
    bot = dict(bot) if bot is not None else None
    if bot is not None:
        bot["kind"] = "twin" if bot["origin"] == TWIN_ORIGIN else "learner"
    fills = []
    if dec["action"] in ORDER_ACTIONS:
        fills = [dict(r) for r in conn.execute(FILLS_SQL, (dec["bot_id"], dec["ts"], "decision", "rejected"))]
    trades = []
    ids = sorted({f["trade_id"] for f in fills if f["trade_id"] is not None})
    if ids:
        for r in conn.execute("SELECT * FROM trades WHERE id IN (" + ", ".join("?" * len(ids)) + ") ORDER BY id", ids):
            t = dict(r)
            legs = {(f["leg"], f["status"]) for f in fills if f["trade_id"] == t["id"]}
            t["role"] = "closed" if ("close", "filled") in legs else "opened" if ("open", "filled") in legs \
                else "close rejected"
            t["state"] = "closed" if t["closed_at"] else "open"
            t["hold_minutes"] = (parse_iso(t["closed_at"]) - parse_iso(t["opened_at"])) / MS_PER_MINUTE \
                if t["closed_at"] else None
            t["fees"] = t["fee_open"] + (t["fee_close"] or 0.0)
            trades.append(t)
    clock = run_clock(conn)
    label = dec["label_bps"]
    opened = next((t for t in trades if t["role"] == "opened"), None)
    sign = {"long": 1, "short": -1}
    coin = coin_side(dec["action"], dec["reason"]) if bot is not None and bot["kind"] == "twin" else None
    own_call = sign.get(coin) if bot is not None and bot["kind"] == "twin" else dec["prediction_bps"]
    outcome = {
        "state": "matured" if label is not None else "waiting" if dec["ts"] > label_cutoff(clock, horizon_ms)
        else "missed",
        "label_bps": label, "matures_at": iso(parse_iso(dec["ts"]) + horizon_ms),
        "right": _right(own_call, label), "prediction_right": _right(dec["prediction_bps"], label),
        "coin_side": coin,
        "position_side": opened["side"] if opened else None,
        "position_right": _right(sign.get(opened["side"]) if opened else None, label),
    }
    pair = None
    if bot is not None:
        found = conn.execute(PAIR_SQL, (dec["ts"], dec["id"], bot["slot"], TWIN_ORIGIN,
                                        0 if bot["kind"] == "twin" else 1)).fetchone()
        pair = dict(found) if found is not None else None
    return {"decision": dec, "bot": bot, "fills": fills, "trades": trades, "outcome": outcome, "pair": pair,
            "horizon_minutes": cfg.time.horizon_minutes, "clock": clock}


# Skipped turns, found from the database alone.


@dataclass(frozen=True)
class Skip:
    """One turn a learner skipped, with the cause the database shows and the learner's state at that minute."""

    id: int
    bot_id: int
    minute: int  # the minute index of the phase minute, whose boundary is ts
    ts: str
    cause: str  # one of CAUSES
    reason: str  # one sentence, also what the reason filter searches
    named: tuple  # the feed events the reason names
    windows: tuple  # for the inputs cause, each feature window that failed
    pruned: bool  # the candles of that time were pruned, so the cause rests on benchmark rows and feed events
    position: dict  # what the learner held, flat or the open trade it kept
    twin: dict | None  # its twin at that minute and what the twin held
    next: dict | None  # its next decision after the skipped turn


def skip_id(minute: int, bot_id: int) -> int | None:
    """The id of a skipped turn, or None for one beyond the integers a browser holds exactly."""
    value = minute * SKIP_ID_BOTS + bot_id
    return -value if 0 < bot_id < SKIP_ID_BOTS and 0 < minute <= SKIP_LAST_MINUTE else None


def split_skip_id(value: int) -> tuple[int, int]:
    """The minute index and bot id inside a skipped turn's id."""
    return divmod(-value, SKIP_ID_BOTS)


def _minute_index(ts: str, days: dict[str, int]) -> int | None:
    """The minute index of a stored boundary time such as 2026-10-13T20:00:00.000Z, or None for any other time.

    Each day is parsed once, so a long run's decisions and candles are read quickly.
    """
    if len(ts) != 24 or ts[10] != "T" or not ts.endswith(":00.000Z"):
        return None
    base = days.get(ts[:10])
    if base is None:
        base = days[ts[:10]] = parse_iso(ts[:10] + "T00:00:00Z") // MS_PER_MINUTE
    return base + int(ts[11:13]) * 60 + int(ts[14:16])


def _lives(bots: list[dict]) -> dict[int, tuple[int, int, int | None]]:
    """Each bot's slot and the minute indexes it stepped in, from the first to one past the last.

    A bot is born after the step of its birth minute, at launch or when it replaces a bot that died in that step, so
    it first steps one minute later. It steps until its death, which is when its successor in the slot is born, even
    while it waits as dying for a close that sets died_at later.
    """
    groups: dict[tuple[int, bool], list[dict]] = {}
    for b in bots:
        groups.setdefault((b["slot"], b["origin"] == TWIN_ORIGIN), []).append(b)
    out = {}
    for group in groups.values():
        group.sort(key=lambda b: (b["born_at"], b["bot_id"]))
        for i, b in enumerate(group):
            ends = [parse_iso(b["died_at"])] if b["died_at"] else []
            if i + 1 < len(group):
                ends.append(parse_iso(group[i + 1]["born_at"]))
            stop = -(-min(ends) // MS_PER_MINUTE) if ends else None
            out[b["bot_id"]] = (b["slot"], parse_iso(b["born_at"]) // MS_PER_MINUTE + 1, stop)
    return out


def _say(event: dict | None) -> str:
    return "" if event is None else f"{event['kind']} {event['detail']}" if event["detail"] else event["kind"]


def _ago(start: str, end: str) -> str:
    return f"{(parse_iso(end) - parse_iso(start)) / MS_PER_MINUTE:.1f} minutes later"


class _Judge:
    """What the database says about each skipped minute, read once for all the skipped turns of one run.

    Benchmark rows are written in exactly the healthy minutes and never pruned, so they say which minutes counted,
    back to the run's first, and every feature window is worked out from them. The candles, kept for keep_candles_days,
    tell a minute that was not healthy from one the run never recorded. The feed events say what took the feed out of
    health and when the run resumed, and where candles were pruned they also tell a stop from a feed outage. Forecast
    rows say whether a prediction was made.
    """

    def __init__(self, conn, cfg: Config, bots: list[dict], lives: dict, decided: dict) -> None:
        self.conn, self.cfg, self.lives, self.decided = conn, cfg, lives, decided
        self.twins: dict[int, list[int]] = {}
        for b in bots:
            if b["origin"] == TWIN_ORIGIN:
                self.twins.setdefault(b["slot"], []).append(b["bot_id"])
        days: dict[str, int] = {}
        self.valid: dict[int, bool] = {}
        for r in conn.execute("SELECT ts, valid FROM candles"):
            minute = _minute_index(r["ts"], days)
            if minute is not None:
                self.valid[minute] = bool(r["valid"])
        self.recorded = sorted(self.valid)
        good = [m for m in (_minute_index(r[0], days) for r in conn.execute("SELECT ts FROM benchmark"))
                if m is not None]
        self.good = set(good)
        # Candles were pruned when a healthy minute is older than the oldest candle kept.
        self.cut = bool(good and self.recorded and min(good) < self.recorded[0])
        # Running counts of healthy minutes and of healthy pairs of consecutive minutes, so every feature window is
        # counted in constant time however long the run.
        ends = self.recorded[:1] + self.recorded[-1:] + ([min(good), max(good)] if good else [])
        self.base, self.top = (min(ends), max(ends)) if ends else (None, None)
        healthy = np.zeros(self.top - self.base + 1 if ends else 0, dtype=np.int64)
        if good:
            healthy[np.array(good, dtype=np.int64) - self.base] = 1
        self.healthy_sum = np.concatenate(([0], np.cumsum(healthy)))
        self.pair_sum = np.concatenate(([0], np.cumsum(healthy[1:] & healthy[:-1])))
        history = conn.execute("SELECT ts FROM benchmark ORDER BY ts LIMIT ? OFFSET ?",
                               (1, max(cfg.time.history_minutes_required - 1, 0))).fetchone()
        self.history_from = history[0] if history is not None else None  # the first minute with enough history
        self.events = [dict(r) for r in conn.execute("SELECT id, ts, kind, detail FROM feed_events ORDER BY ts, id")]
        self.stamps = [e["ts"] for e in self.events]
        tests = {"healthy": lambda e: e["kind"] == "healthy", "exit": lambda e: e["kind"] in STATE_KINDS[1:],
                 "restore": lambda e: e["kind"] == "resume" and (e["detail"] or "").startswith(RESTORED),
                 "resume": lambda e: e["kind"] == "resume" and not (e["detail"] or "").startswith(RESTORED),
                 "disconnected": lambda e: e["kind"] == "disconnected"}
        self.groups = {}
        for name, test in tests.items():
            picked = [e for e in self.events if test(e)]
            self.groups[name] = ([e["ts"] for e in picked], picked)
        self.trades: dict[int, list[dict]] = {}
        for r in conn.execute("SELECT id, bot_id, side, opened_at, closed_at, entry_price, exit_reason FROM trades "
                              "ORDER BY opened_at, id"):
            self.trades.setdefault(r["bot_id"], []).append(dict(r))
        self.opened = {k: [t["opened_at"] for t in v] for k, v in self.trades.items()}
        self.seen: dict[tuple[str, str], bool] = {}

    # Reading the tables

    def state(self, minute: int) -> bool | None:
        """Whether a minute's candle says it counted, False when it was not healthy or has no record, None when the
        candles of that time were pruned."""
        if not self.recorded or minute < self.recorded[0]:
            return None
        return self.valid.get(minute, False)

    def missing(self, minute: int) -> str:
        """Why a minute that did not count is missing, from its candle while candles are kept, and before them from
        the restarts, a stop lasting from the checkpoint a restart restored to the restart itself."""
        if self.recorded and minute >= self.recorded[0]:
            return "not healthy" if minute in self.valid else "not recorded"
        if not self.cut or self.down(iso(minute * MS_PER_MINUTE)) is not None:
            return "not recorded"  # before the run's first minute, or while it was stopped
        return "not healthy"

    def down(self, ts: str) -> dict | None:
        """The restart that ended a stop holding this minute, the first restart after it whose checkpoint is older."""
        restore = self.first("restore", ts)
        return restore if restore is not None and restore["detail"][len(RESTORED):][:24] <= ts else None

    def has_row(self, table: str, ts: str) -> bool:
        key = (table, ts)
        if key not in self.seen:
            sql = {"forecasts": "SELECT count(*) FROM forecasts WHERE ts = ?"}[table]
            self.seen[key] = self.conn.execute(sql, (ts,)).fetchone()[0] > 0
        return self.seen[key]

    def last(self, group: str, below: str, inclusive: bool = False) -> dict | None:
        stamps, events = self.groups[group]
        i = (bisect_right if inclusive else bisect_left)(stamps, below)
        return events[i - 1] if i else None

    def first(self, group: str, above: str, inclusive: bool = False) -> dict | None:
        stamps, events = self.groups[group]
        i = (bisect_left if inclusive else bisect_right)(stamps, above)
        return events[i] if i < len(events) else None

    def trigger(self, exit_event: dict) -> dict | None:
        """The feed event logged in the same millisecond just before a change of state, such as the clock skew or
        the disconnect that caused it."""
        found = None
        for e in self.events[bisect_left(self.stamps, exit_event["ts"]):bisect_right(self.stamps, exit_event["ts"])]:
            if e is exit_event:
                break
            if e["kind"] not in STATE_KINDS and e["kind"] != "resume":
                found = e
        return found

    def position(self, bot_id: int, ts: str) -> dict:
        """The position a bot held at a boundary, from its trades, flat when none was open."""
        trades = self.trades.get(bot_id, [])
        i = bisect_right(self.opened.get(bot_id, []), ts) - 1
        if i >= 0 and (trades[i]["closed_at"] is None or trades[i]["closed_at"] > ts):
            t = trades[i]
            return {"side": t["side"], "trade_id": t["id"], "opened_at": t["opened_at"],
                    "entry_price": t["entry_price"], "closed_at": t["closed_at"], "exit_reason": t["exit_reason"]}
        return {"side": "flat"}

    # The causes, most specific first

    def judge(self, minute: int, bot_id: int, slot: int, sid: int) -> Skip:
        ts = iso(minute * MS_PER_MINUTE)
        state, windows, pruned = self.state(minute), [], False
        if state is None:
            pruned = True
            cause, reason, named, windows = self.pruned(minute, ts)
        elif minute not in self.valid:
            cause, reason, named = self.not_running(minute, ts)
        elif not state:
            cause, reason, named = self.not_healthy(ts)
        else:
            cause, reason, windows = self.healthy(minute, ts)
            named = ()
        named = tuple(named) + tuple(e for w in windows for e in w.get("events", ()))
        stamps, rows = self.decided.get(bot_id, ([], []))
        i = bisect_right(stamps, ts)
        nxt = dict(zip(("ts", "id", "action"), rows[i])) if i < len(rows) else None
        twin = next((t for t in self.twins.get(slot, []) if self.lives[t][1] <= minute
                     and (self.lives[t][2] is None or minute < self.lives[t][2])), None)
        return Skip(sid, bot_id, minute, ts, cause, reason, tuple(e for e in named if e), tuple(windows), pruned,
                    self.position(bot_id, ts),
                    None if twin is None else {"bot_id": twin, "position": self.position(twin, ts)}, nxt)

    def not_running(self, minute: int, ts: str) -> tuple[str, str, tuple]:
        """No candle at a minute inside the kept candles means the engine never closed it."""
        i = bisect_left(self.recorded, minute)
        before = self.recorded[i - 1] if i else None
        after = self.recorded[i] if i < len(self.recorded) else None
        text = "the run was not running"
        if before is not None and after is not None:
            text += (f", no minute was recorded between {iso(before * MS_PER_MINUTE)} and "
                     f"{iso(after * MS_PER_MINUTE)}")
        elif after is not None:
            text += f", no minute was recorded before {iso(after * MS_PER_MINUTE)}"
        else:
            text += f", no minute was recorded after {iso(before * MS_PER_MINUTE)}"
        lost = self.last("disconnected", ts, inclusive=True)
        if lost is not None and (before is None or lost["ts"] < iso(before * MS_PER_MINUTE)):
            lost = None
        if lost is not None:
            text += f", after the feed logged {_say(lost)} at {lost['ts']}"
        restore = self.first("restore", ts)
        if restore is not None and after is not None and restore["ts"] > iso(after * MS_PER_MINUTE):
            restore = None
        text, flatten = self.resumed(text, restore)
        return "not_running", text, (lost, restore, flatten)

    def resumed(self, text: str, restore: dict | None) -> tuple[str, dict | None]:
        """Name the restart that ended a stretch, and the resume event its first healthy minute logged, if any."""
        if restore is None:
            return text, None
        text += f", it resumed at {restore['ts']} from the checkpoint of {restore['detail'][len(RESTORED):][:24]}"
        back = self.first("healthy", restore["ts"], inclusive=True)
        flatten = self.first("resume", restore["ts"]) if back is not None else None
        if flatten is not None and flatten["ts"] == back["ts"]:
            return text + f", and at its first healthy minute it logged {_say(flatten)}", flatten
        return text, None

    def left_health(self, ts: str) -> tuple[dict | None, dict | None, dict | None]:
        """For a minute that was not healthy, the restart or the change of state that ended the last healthy stretch
        before it, as (restore, change, its trigger)."""
        prev = self.last("healthy", ts)
        since = prev["ts"] if prev is not None else ""
        restore = self.last("restore", ts, inclusive=True)
        if restore is not None and restore["ts"] <= since:
            restore = None
        out = self.first("exit", since) if restore is None else None
        if out is not None and out["ts"] > ts:
            out = None
        return restore, out, self.trigger(out) if out is not None else None

    def not_healthy(self, ts: str) -> tuple[str, str, tuple]:
        """The feed event that took the feed out of health before this minute, and when it was healthy again."""
        restore, out, cause = self.left_health(ts)
        back = self.first("healthy", ts)
        if restore is not None:
            text = f"the feed was not healthy, it had not been healthy since the run resumed at {restore['ts']}"
        elif out is not None:
            text = f"the feed was not healthy, it went {out['kind']} at {out['ts']}"
            text += f" on {_say(cause)}" if cause else f", {out['detail']}" if out["detail"] else ""
        else:
            text = "the feed was not healthy, and no feed event says what took it out of health"
        start = restore or out
        if back is not None:
            text += f", and was healthy again at {back['ts']}" + (f", {_ago(start['ts'], back['ts'])}" if start else "")
        else:
            text += ", and no healthy minute has been logged since"
        return "not_healthy", text, (cause, out, restore, back)

    def healthy(self, minute: int, ts: str) -> tuple[str, str, list[dict]]:
        """A healthy minute, explained by its feature windows when no prediction was made."""
        if self.has_row("forecasts", ts):
            return "unknown", ("no decision recorded, cause not in the database, the minute was healthy and a "
                               "prediction was made"), []
        return self.inputs(minute, ts)

    def inputs(self, minute: int, ts: str) -> tuple[str, str, list[dict]]:
        """A healthy minute with no prediction, explained by the feature windows of features.py that failed. A return
        window that starts in a minute that was not healthy also names what took the feed out of health."""
        failed = self.windows(minute)
        head = "the feed was healthy but the features were incomplete, "
        if failed:
            told = set()
            for w in failed:
                if w.get("state") != "not healthy":
                    continue
                restore, out, cause = self.left_health(w["minute"])
                said = restore or out
                if said is None or said["id"] in told:
                    continue
                told.add(said["id"])
                w["events"] = [e for e in (cause, out, restore) if e]
                w["text"] += (f", as the run had resumed at {restore['ts']}" if restore else
                              f", as the feed went {out['kind']} at {out['ts']}" +
                              (f" on {_say(cause)}" if cause else f", {out['detail']}" if out["detail"] else ""))
            return "inputs", head + ", and ".join(w["text"] for w in failed), failed
        history = self.cfg.time.history_minutes_required
        if self.history_from is None or self.history_from > ts:  # benchmark rows are the healthy minutes, all kept
            have = self.conn.execute("SELECT count(*) FROM benchmark WHERE ts <= ?", (ts,)).fetchone()[0]
            short = {"window": "history", "text": f"the run had {have} of the {history} healthy minutes of history "
                                                  f"it needs"}
            return "inputs", head + short["text"], [short]
        return "unknown", ("no decision recorded, cause not in the database, the minute was healthy and every "
                           "feature window looks complete, yet no prediction was written"), []

    def healthy_in(self, first: int, last: int) -> int:
        """Healthy minutes from first to last inclusive. No minute outside the run's records was healthy."""
        if self.base is None:
            return 0
        lo, hi = max(first, self.base) - self.base, min(last, self.top) - self.base
        return int(self.healthy_sum[hi + 1] - self.healthy_sum[lo]) if lo <= hi else 0

    def pairs_in(self, first: int, last: int) -> int:
        """Minutes i from first to last inclusive where i and i minus 1 were both healthy."""
        if self.base is None:
            return 0
        lo, hi = max(first, self.base + 1) - self.base - 1, min(last, self.top) - self.base - 1
        return int(self.pair_sum[hi + 1] - self.pair_sum[lo]) if lo <= hi else 0

    def windows(self, minute: int) -> list[dict]:
        """Every feature window that failed at this minute by the rules of FeatureBuilder.compute, counted on the
        healthy minutes the benchmark rows record."""
        fc = self.cfg.features
        cover, failed = fc.min_window_coverage, []
        for k in fc.return_windows_minutes:
            start = minute - k
            if start not in self.good:
                at, why = iso(start * MS_PER_MINUTE), self.missing(start)
                failed.append({"window": f"{k} minute return", "minute": at, "state": why,
                               "text": f"the {k} minute return starts at {at}, which was {why}"})
        counted = [(f"{w} minute volatility", "pairs of consecutive healthy minutes", max(2, math.ceil(cover * w)),
                    self.pairs_in(minute - w + 1, minute)) for w in fc.vol_windows_minutes]
        counted += [(f"{n} minute {name}", "healthy minutes", math.ceil(cover * n),
                     self.healthy_in(minute - n + 1, minute))
                    for name, n in (("recent volume", VOLUME_WINDOW), ("volume baseline", fc.volume_baseline_minutes))]
        for window, unit, need, have in counted:
            if have < need:
                failed.append({"window": window, "have": have, "need": need,
                               "text": f"the {window} had {have} of the {need} {unit} it needs"})
        return failed

    def pruned(self, minute: int, ts: str) -> tuple[str, str, tuple, list[dict]]:
        """Before the oldest kept candle, a benchmark row marks a healthy minute, whose feature windows are worked out
        as for any other, and the feed events tell a stretch the run was down from one the feed was not healthy."""
        tail = ", the minute records of that time were pruned"
        if minute in self.good:
            cause, text, windows = self.healthy(minute, ts)
            return cause, text + tail, (), windows
        restore = self.down(ts)
        if restore is not None:
            text, flatten = self.resumed("the run was not running", restore)
            return "not_running", text + tail, (restore, flatten), []
        cause, text, named = self.not_healthy(ts)
        return cause, text + tail, named, []


def _find_skips(conn, cfg: Config) -> dict[int, Skip]:
    """Every skipped turn of a run, newest first, keyed by id.

    A learner, a bot whose origin is not control, skips a turn at a phase minute it lived through, from the run's
    first decision to the newest checkpoint, that has no decision row of its own. The run's first decision is taken
    no earlier than its first forecast row, which in a real run is the same minute, since both need a prediction.
    Rows after the newest checkpoint are still buffered in the run, so those minutes are not judged.
    """
    newest = conn.execute("SELECT ts FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()
    first = conn.execute("SELECT min(ts) FROM decisions").fetchone()[0]
    if newest is None or first is None:
        return {}
    predicted = conn.execute("SELECT min(ts) FROM forecasts").fetchone()[0]
    start = -(-parse_iso(max(first, predicted or first)) // MS_PER_MINUTE)
    stop = -(-parse_iso(newest["ts"]) // MS_PER_MINUTE)
    horizon = cfg.time.horizon_minutes
    bots = [dict(r) for r in conn.execute("SELECT bot_id, slot, origin, born_at, died_at FROM bots")]
    lives = _lives(bots)
    days: dict[str, int] = {}
    minutes: dict[int, set[int]] = {}
    decided: dict[int, tuple[list[str], list[tuple]]] = {}
    for r in conn.execute("SELECT d.id, d.bot_id, d.ts, d.action FROM decisions d JOIN bots b USING (bot_id) "
                          "WHERE b.origin != ? ORDER BY d.bot_id, d.ts, d.id", (TWIN_ORIGIN,)):
        stamps, rows = decided.setdefault(r["bot_id"], ([], []))
        stamps.append(r["ts"])
        rows.append((r["ts"], r["id"], r["action"]))
        minute = _minute_index(r["ts"], days)
        if minute is not None:
            minutes.setdefault(r["bot_id"], set()).add(minute)
    found = []
    for b in bots:
        if b["origin"] == TWIN_ORIGIN:
            continue
        slot, lo, hi = lives[b["bot_id"]]
        lo, hi = max(lo, start), stop if hi is None else min(hi, stop)
        have = minutes.get(b["bot_id"], set())
        found += [(m, b["bot_id"], slot) for m in range(lo + (slot - lo) % horizon, hi, horizon) if m not in have]
    if not found:
        return {}
    judge = _Judge(conn, cfg, bots, lives, decided)
    skips = []
    for minute, bot_id, slot in found:
        sid = skip_id(minute, bot_id)
        if sid is None:
            log.warning("bot %d at minute %d has an id too large to list as a skipped turn", bot_id, minute)
            continue
        skips.append(judge.judge(minute, bot_id, slot, sid))
    skips.sort(key=lambda s: (s.ts, s.id), reverse=True)
    return {s.id: s for s in skips}


_skip_cache: dict = {}
_skip_lock = threading.Lock()


def skipped_turns(conn, cfg: Config) -> dict[int, Skip]:
    """The run's skipped turns, worked out on request and kept until the run commits new rows or feed events.

    Every table they read grows only by rowid, so the newest rowid of each names the database's state cheaply. Rows
    reach it with checkpoints, but feed events are committed every minute, so in practice the turns are worked out
    again once per checkpoint or new feed event. Pruning drops the oldest candles, so their oldest rowid counts too.
    """
    marks = tuple(conn.execute(
        "SELECT (SELECT max(id) FROM checkpoints), (SELECT max(ts) FROM checkpoints), (SELECT max(id) FROM decisions), "
        "(SELECT max(id) FROM feed_events), (SELECT min(rowid) FROM candles), (SELECT max(rowid) FROM candles), "
        "(SELECT max(rowid) FROM bots), (SELECT count(died_at) FROM bots), (SELECT max(id) FROM trades), "
        "(SELECT count(closed_at) FROM trades), (SELECT max(rowid) FROM forecasts), "
        "(SELECT max(rowid) FROM benchmark)").fetchone())
    key = (str(Path(cfg.db_path).resolve()), marks, cfg.time, cfg.features)
    with _skip_lock:
        hit = _skip_cache.get(key)
    if hit is not None:
        return hit
    found = _find_skips(conn, cfg)
    with _skip_lock:
        if len(_skip_cache) >= SKIP_CACHE_ENTRIES:
            _skip_cache.clear()
        _skip_cache[key] = found
    return found


def load_skips(conn, skips) -> None:
    """Put the skipped turns in a temporary table on this connection, so the explorer's SQL reads them beside the
    decisions. Only the connection's own temporary database is written, never the run's."""
    conn.execute(SKIP_TABLE_SQL)
    conn.execute("DELETE FROM temp.skipped_turns")
    conn.executemany(SKIP_INSERT_SQL, [(s.id, s.bot_id, s.ts, s.position["side"], SKIPPED, s.reason, s.cause)
                                       for s in skips])


def skipped_detail(conn, cfg: Config, sid: int) -> dict:
    """One skipped turn in full, with its cause, the feed events around it, the position kept and the next decision."""
    skip = skipped_turns(conn, cfg).get(sid)
    if skip is None:
        minute, bot_id = split_skip_id(sid)
        raise NotFound(f"bot {bot_id} has no skipped turn at {iso(minute * MS_PER_MINUTE)} in this run")
    bot = conn.execute("SELECT bot_id, slot, parent_id, generation, origin, state, born_at, died_at, margin_bps, lr "
                       "FROM bots WHERE bot_id = ?", (skip.bot_id,)).fetchone()
    bot = {**dict(bot), "kind": "learner"} if bot is not None else None
    at = skip.minute * MS_PER_MINUTE
    # The events nearest the minute, so a storm of reconnects earlier in the window never pushes them out.
    near = {r["id"]: dict(r) for r in conn.execute(
        "SELECT id, ts, kind, detail FROM feed_events WHERE ts >= ? AND ts <= ? ORDER BY ts DESC, id DESC LIMIT ?",
        (iso(at - NEAR_BEFORE_MS), iso(at), NEAR_BEFORE_EVENTS))}
    near.update({r["id"]: dict(r) for r in conn.execute(
        "SELECT id, ts, kind, detail FROM feed_events WHERE ts > ? AND ts <= ? ORDER BY ts, id LIMIT ?",
        (iso(at), iso(at + NEAR_AFTER_MS), NEAR_AFTER_EVENTS))})
    named = {e["id"] for e in skip.named}
    near.update({e["id"]: e for e in skip.named})
    events = [{**e, "named": e["id"] in named} for e in sorted(near.values(), key=lambda e: (e["ts"], e["id"]))]
    return {"skipped": {"id": skip.id, "bot_id": skip.bot_id, "ts": skip.ts, "action": SKIPPED, "cause": skip.cause,
                        "reason": skip.reason, "pruned": skip.pruned, "windows": list(skip.windows)},
            "bot": bot, "position": skip.position, "twin": skip.twin, "next": skip.next, "events": events,
            "horizon_minutes": cfg.time.horizon_minutes, "clock": run_clock(conn)}


def read_run(cfg: Config, name: str, work):
    """Run work on a read only connection, stopped once it passes QUERY_BUDGET_S.

    Every query of one answer runs in one read transaction, so a checkpoint the live run commits halfway through
    cannot leave the total, the counts and the rows from different moments. Closing the connection ends it.
    """
    try:
        conn = data.connect(cfg.db_path)
    except FileNotFoundError:
        raise NotFound(f"run {name} has no database yet") from None
    deadline = time.monotonic() + QUERY_BUDGET_S
    try:
        conn.set_progress_handler(lambda: time.monotonic() > deadline, PROGRESS_STEPS)
        conn.execute("PRAGMA temp_store = MEMORY")  # the skipped turns table stays in memory, it cannot be set in BEGIN
        conn.execute("BEGIN")
        return work(conn)
    except sqlite3.OperationalError as exc:
        if "interrupt" in str(exc):
            raise TooSlow("the query took too long, narrow the filters") from None
        raise
    finally:
        conn.close()


class Recent:
    """Answers kept for STATE_TTL_S, so viewers with the same request cost one query between them."""

    def __init__(self, size: int = CACHE_ENTRIES) -> None:
        self.size = size
        self.items: dict = {}
        self.lock = threading.Lock()

    def get(self, key, make):
        with self.lock:
            hit = self.items.get(key)
            if hit is not None and time.monotonic() - hit[0] < STATE_TTL_S:
                return hit[1]
        value = make()
        with self.lock:
            if len(self.items) >= self.size:
                self.items.clear()
            self.items[key] = (time.monotonic(), value)
        return value


def _json(value) -> bytes:
    return json.dumps(clean(value), allow_nan=False, default=str).encode()


def host_allowed(hosts: list[str], port: int) -> bool:
    """True when the request carries one Host header naming 127.0.0.1 or localhost with the serving port.

    A page on another site can point its own name at 127.0.0.1, which is DNS rebinding, and its requests then reach
    this server carrying that other name, so they are refused. Browsers leave out port 80, so only there the bare
    name is accepted too.
    """
    if len(hosts) != 1:
        return False
    host = hosts[0].strip().lower()
    return any(host == f"{name}:{port}" or (port == 80 and host == name) for name in LOCAL_NAMES)


def make_handler(configs: list[Config]):
    names = run_names(configs)
    by_name = dict(zip(names, configs))
    cache = {"at": -STATE_TTL_S, "body": b""}
    lock = threading.Lock()
    recent = Recent()

    def state_body() -> bytes:
        """The JSON view, recomputed at most once per STATE_TTL_S however many requests arrive."""
        with lock:
            if time.monotonic() - cache["at"] >= STATE_TTL_S:
                now = time.time_ns() // 1_000_000
                state = clean({"now_ms": now, "page_version": page_version(),
                               "runs": [run_state(c, now, n) for c, n in zip(configs, names)]})
                cache["body"] = json.dumps(state, allow_nan=False, default=str).encode()
                cache["at"] = time.monotonic()
            return cache["body"]

    def explore(route: str, query: str) -> tuple[int, bytes]:
        """The decisions list or one decision as JSON. Errors are one plain sentence, never a traceback or a path."""
        name = None
        try:
            params = read_params(query, LIST_PARAMS if route == "/api/decisions" else DETAIL_PARAMS)
            name = params.get("run")
            if name is None:
                raise BadRequest("run is required, the name of one tab")
            if name not in by_name:
                raise NotFound(f"there is no run named {name[:40]!r}")
            cfg = by_name[name]
            if route == "/api/decisions":
                f = parse_filters(params)
                body = recent.get((route, name, f), lambda: _json(
                    {"run": name, **read_run(cfg, name, lambda conn: decision_page(conn, cfg, f))}))
            else:
                decision_id = detail_id(params)
                body = recent.get((route, name, decision_id), lambda: _json(
                    {"run": name, **read_run(cfg, name, lambda conn: decision_detail(conn, cfg, decision_id))}))
            return 200, body
        except BadRequest as exc:
            return 400, _json({"error": str(exc)})
        except NotFound as exc:
            return 404, _json({"error": str(exc)})
        except TooSlow as exc:
            return 503, _json({"error": str(exc)})
        except Exception:  # noqa: BLE001, the page gets one sentence and the log gets the traceback
            log.exception("dashboard could not answer %s for run %s", route, name)
            return 500, _json({"error": f"the decisions of run {name} cannot be read, see the dashboard log"})

    class Handler(BaseHTTPRequestHandler):
        def parse_request(self) -> bool:
            """Read the request line and headers, then refuse a request addressed to any other name, whatever its
            method, before its path is looked at."""
            if not super().parse_request():
                return False  # the standard library has answered already, with the headers end_headers adds
            if host_allowed(self.headers.get_all("Host") or [], self.server.server_address[1]):
                return True
            self.close_connection = True
            self._send(403, "text/plain", b"this dashboard only answers requests addressed to 127.0.0.1 or "
                                          b"localhost on its own port")
            return False

        def do_GET(self) -> None:
            try:
                url = urllib.parse.urlsplit(self.path)
            except ValueError:  # a request line such as http://[x/ that cannot be split, answered without a traceback
                self._send(400, "text/plain", b"the request path cannot be read")
                return
            if url.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())
            elif url.path == "/api/state":
                self._send(200, "application/json", state_body())
            elif url.path in ("/api/decisions", "/api/decision"):
                code, body = explore(url.path, url.query)
                self._send(code, "application/json", body)
            else:
                self._send(404, "text/plain", b"not found")

        def _send(self, code: int, kind: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":  # an answer to HEAD gives the length but never the body
                self.wfile.write(body)

        def end_headers(self) -> None:
            """Every answer is never stored and carries the security headers, including the standard library's own
            error pages for a method this server does not offer or a request it cannot read."""
            self.send_header("Cache-Control", "no-store")
            for name, value in SECURITY_HEADERS.items():
                self.send_header(name, value)
            super().end_headers()

        def log_message(self, fmt: str, *args) -> None:
            log.debug("dashboard %s", fmt % args)

    return Handler


def serve(configs: list[Config], port: int) -> None:
    """Serve until Ctrl+C on 127.0.0.1 only, so nothing outside this machine can reach it, and the Host check stops
    a page on another site from reaching it through a name of its own."""
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(configs))
    print(f"dashboard for {', '.join(c.data_dir.name for c in configs)} at http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
