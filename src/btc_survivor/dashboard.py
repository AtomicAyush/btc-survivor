"""A read only local dashboard for one or more runs, served on this machine only.

It serves one self contained page and a JSON view built from the same read only queries as the status command, so
it can never change a run. Two more JSON views back the decisions explorer, one page of filtered decisions and one
decision with the order, fills, trade and label it led to, built from fixed SQL with every filter as a parameter. It
binds to 127.0.0.1, refuses any request whose Host header names something other than 127.0.0.1 or localhost on its
own port, and needs nothing beyond the project's own dependencies.
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
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from btc_survivor.cli import data
from btc_survivor.cli.evaluate import SKILL_LAGS, forecasts
from btc_survivor.cli.stats import diebold_mariano, skill_test
from btc_survivor.clock import MS_PER_DAY, MS_PER_HOUR, MS_PER_MINUTE, iso, parse_iso
from btc_survivor.config import Config

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
LIST_SQL = ("SELECT d.id, d.ts, d.bot_id, b.origin, b.generation, d.position_before, d.prediction_bps, "
            "d.required_edge_bps, d.gate_passed, d.forced, d.action, d.reason, d.label_bps, " + CALL_SQL +
            " AS direction FROM decisions d JOIN bots b USING (bot_id)")
COUNT_SQL = ("SELECT action, count(*) AS n, sum(forced) AS forced, sum(gate_passed) AS gate, "
             "count(label_bps) AS matured, sum(label_bps IS NULL AND ts > ?) AS waiting, "
             "sum(direction * label_bps > 0) AS n_right, sum(direction * label_bps < 0) AS n_wrong "
             "FROM (SELECT d.action, d.forced, d.gate_passed, d.label_bps, d.ts, " + CALL_SQL +
             " AS direction FROM decisions d JOIN bots b USING (bot_id)")
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
    """Descriptive skill on the same rows and with the same statistics as criterion 1. Only the final day counts."""
    out = {}
    for name, rows in forecasts(conn, horizon).items():
        pred = np.array([p for p, _ in rows.values()])
        label = np.array([y for _, y in rows.values()])
        out[name] = {"windows": len(pred), "corr": skill_test(pred, label, SKILL_LAGS)["corr"],
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
        if not all(a in ACTIONS for a in actions):
            raise BadRequest(f"action must be one or more of {', '.join(ACTIONS)}, separated by commas")
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
        add("d.gate_passed = ?", f.gate)
    cutoff = label_cutoff(clock, horizon_ms)
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
    """One page of decisions, newest first, with the total, counts per action under the other filters and a summary.

    Learners are scored by their prediction and twins by the side their coin chose, so right and wrong judge each
    bot's own call. Waiting and missed together are the decisions whose label has not matured.
    """
    horizon_ms = cfg.time.horizon_minutes * MS_PER_MINUTE
    clock = run_clock(conn)
    cutoff = label_cutoff(clock, horizon_ms)
    clause, args = where(f, clock, horizon_ms, with_actions=False)
    counts = {a: 0 for a in ACTIONS}
    summary = {"total": 0, "forced": 0, "gate_passed": 0, "matured": 0, "waiting": 0, "missed": 0, "right": 0,
               "wrong": 0}
    for r in conn.execute(COUNT_SQL + clause + ") GROUP BY action", [cutoff, *CALL_ARGS, *args]):
        counts[r["action"]] = r["n"]
        if f.actions is None or r["action"] in f.actions:
            for key, col in (("total", "n"), ("forced", "forced"), ("gate_passed", "gate"), ("matured", "matured"),
                             ("waiting", "waiting"), ("right", "n_right"), ("wrong", "n_wrong")):
                summary[key] += r[col] or 0
    summary["missed"] = summary["total"] - summary["matured"] - summary["waiting"]
    pages = max(1, math.ceil(summary["total"] / f.per))
    page = min(f.page, pages)
    clause, args = where(f, clock, horizon_ms)
    rows = []
    for r in conn.execute(LIST_SQL + clause + " ORDER BY d.ts DESC, d.id DESC LIMIT ? OFFSET ?",
                          [*CALL_ARGS, *args, f.per, (page - 1) * f.per]):
        row = dict(r)
        direction = row.pop("direction")
        row["kind"] = "twin" if row.pop("origin") == TWIN_ORIGIN else "learner"
        row["label_state"] = "matured" if row["label_bps"] is not None else \
            "waiting" if row["ts"] > cutoff else "missed"
        row["coin_side"] = coin_side(row["action"], row["reason"]) if row["kind"] == "twin" else None
        row["right"] = _right(direction, row["label_bps"])
        rows.append(row)
    filters = {k: v for k, v in asdict(f).items() if v is not None and k not in ("page", "per")}
    return {"clock": clock, "horizon_minutes": cfg.time.horizon_minutes, "page": page, "per": f.per, "pages": pages,
            "total": summary["total"], "action_counts": counts, "summary": summary, "filters": filters,
            "rows": rows}


def decision_detail(conn, cfg: Config, decision_id: int) -> dict:
    """One decision with every column, its bot, the fills of the order it sent, the trades they touched and its label.

    A decision's order carries the bot and the decision minute, and its fills keep the reason decision, or a rejection
    reason when refused, so a stop or death close in the same minute is never mistaken for it.
    """
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
                decision_id = _int(params, "id", 1, MAX_ID)
                if decision_id is None:
                    raise BadRequest("id is required, the number of one decision")
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
