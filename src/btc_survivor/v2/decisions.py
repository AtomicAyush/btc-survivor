"""The decisions explorer of the private dashboard, every decision of both stock runs paged on the server, newest
first, with filters, and the full detail of one decision.

A page is read from each chosen run database, opened read only by the caller, merged newest first by time, run and
row id, and continued with a cursor holding the last row's key, so decisions arriving at the top never shift an older
page. The filters are a run, a home symbol, a slot or a bot, learners or twins, an action, forced or not, gate passed
or not, text in the reason and a time range in UTC. A bound is a date or a minute, and the end bound includes that
whole day or minute. Every page carries the counts of all matching decisions, how many were forced or passed the gate,
and how often the prediction's sign matched the label once it matured.

The detail of one decision adds its bot, the fills of the orders its bot sent at that minute with their costs, the
trades those fills opened or closed, the position after them and its label, which stays empty until it matures.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields

from btc_survivor.clock import MS_PER_DAY, MS_PER_MINUTE, iso, parse_iso
from btc_survivor.v2.status import TWIN_ORIGIN
from btc_survivor.v2.stocks.costs import BUY

ACTIONS = ("none", "hold", "open_long", "open_short", "close", "flip")
KINDS = ("learner", "twin")
CHOICES = ("yes", "no")
PAGE_ROWS = 50
MAX_PAGE_ROWS = 200
MAX_TEXT = 100
MAX_ID = 10**12
FILTERS = ("run", "symbol", "slot", "bot", "kind", "action", "forced", "gate", "reason", "from", "to", "before",
           "limit")
DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
MINUTE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")
NUMBER = re.compile(r"\d{1,13}")

ROW = f"""
SELECT d.id, d.ts, d.bot_id, b.slot, CASE WHEN b.origin = '{TWIN_ORIGIN}' THEN 'twin' ELSE 'learner' END AS kind,
  d.symbol, d.position_before, d.action, d.forced, d.gate_passed, d.prediction_bps, d.required_edge_bps,
  d.label_bps, d.reason
FROM decisions d JOIN bots b ON b.bot_id = d.bot_id
"""
COUNTS = """
SELECT count(*) AS decisions, coalesce(sum(d.forced), 0) AS forced, coalesce(sum(d.gate_passed), 0) AS gate_passed,
  count(d.label_bps) AS labelled,
  coalesce(sum(d.label_bps IS NOT NULL AND d.prediction_bps != 0), 0) AS directional,
  coalesce(sum(d.label_bps IS NOT NULL AND d.prediction_bps * d.label_bps > 0), 0) AS right,
  coalesce(sum(CASE WHEN d.label_bps IS NOT NULL AND d.prediction_bps != 0
                    THEN d.label_bps * (CASE WHEN d.prediction_bps > 0 THEN 1.0 ELSE -1.0 END) END), 0.0)
    AS directional_label_bps
FROM decisions d JOIN bots b ON b.bot_id = d.bot_id
"""
SUMS = ("decisions", "forced", "gate_passed", "labelled", "directional", "right", "directional_label_bps")
DETAIL = """
SELECT d.*, b.slot, b.origin, b.home_symbol, b.state AS bot_state, b.born_at, b.died_at
FROM decisions d JOIN bots b ON b.bot_id = d.bot_id WHERE d.id = ?
"""
FILLS = """
SELECT id, trade_id, leg, side, qty, status, reason, decided_at, filled_at, latency_ms, ref_mid, print_price, price,
  allowance_bps, slippage_bps, notional, fee, sec_fee, taf_fee
FROM fills WHERE bot_id = ? AND decided_at = ? ORDER BY id
"""
TRADES = """
SELECT id, symbol, side, opened_at, closed_at, qty, entry_price, exit_price, fee_open, fee_close, funding,
  slippage_bps_open, slippage_bps_close, pnl_gross, pnl_net, forced, exit_reason
FROM trades WHERE id IN ({}) ORDER BY id
"""


class FilterError(ValueError):
    """Raised for a filter the explorer cannot use, with a message the page may show."""


@dataclass(frozen=True)
class Cursor:
    """The key of the last row of a page, so the next page holds only older rows."""

    ts: str
    run: int
    id: int

    def text(self) -> str:
        return f"{self.ts},{self.run},{self.id}"


@dataclass(frozen=True)
class Filters:
    """One explorer query, runs as indexes into the run list, the prices run first, and times as ISO text."""

    runs: tuple[int, ...]
    symbol: str | None = None
    slot: int | None = None
    bot_id: int | None = None
    kind: str | None = None
    action: str | None = None
    forced: bool | None = None
    gate: bool | None = None
    reason: str | None = None
    since: str | None = None   # inclusive
    until: str | None = None   # exclusive
    before: Cursor | None = None
    limit: int = PAGE_ROWS


def number(name: str, text: str, high: int) -> int:
    """A whole number from 0 to high, or an error naming the filter."""
    if not NUMBER.fullmatch(text) or int(text) > high:
        raise FilterError(f"{name} must be a whole number from 0 to {high}")
    return int(text)


def choice(name: str, text: str, allowed: Sequence[str]) -> str:
    if text not in allowed:
        raise FilterError(f"{name} must be one of {', '.join(allowed)}")
    return text


def bound(name: str, text: str, end: bool) -> str:
    """A UTC date or minute as ISO text, moved one day or one minute on for an end bound so it is included."""
    if DAY.fullmatch(text):
        start, step = f"{text}T00:00:00Z", MS_PER_DAY
    elif MINUTE.fullmatch(text):
        start, step = f"{text}:00Z", MS_PER_MINUTE
    else:
        raise FilterError(f"{name} must be a UTC date such as 2026-10-19 or a minute such as 2026-10-19T14:30")
    try:
        ms = parse_iso(start)
    except ValueError:
        raise FilterError(f"{name} is not a real date or time") from None
    return iso(ms + step if end else ms)


def cursor(text: str, runs: int) -> Cursor:
    """A cursor from its text, refusing anything a page did not hand out."""
    parts = text.split(",")
    try:
        ts, run, row_id = parts[0], int(parts[1]), int(parts[2])
        valid = len(parts) == 3 and iso(parse_iso(ts)) == ts and 0 <= run < runs and 0 <= row_id <= MAX_ID
    except (IndexError, ValueError):
        valid = False
    if not valid:
        raise FilterError("before must be a cursor handed out by an earlier page")
    return Cursor(ts, run, row_id)


def like(text: str) -> str:
    """A LIKE pattern matching text anywhere, with the wildcards in it taken literally."""
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def parse_filters(query: Mapping[str, Sequence[str]], runs: Sequence[str], symbols: Sequence[str]) -> Filters:
    """Filters from a parsed query string, each given at most once, where an empty value means no filter."""
    unknown = sorted(set(query) - set(FILTERS))
    if unknown:
        raise FilterError(f"unknown filter {unknown[0]}, the filters are {', '.join(FILTERS)}")
    values = {}
    for name, given in query.items():
        if len(given) > 1:
            raise FilterError(f"{name} may be given only once")
        if given and given[0].strip():
            values[name] = given[0].strip()
    if len(values.get("reason", "")) > MAX_TEXT:
        raise FilterError(f"reason text is at most {MAX_TEXT} characters")
    run = choice("run", values["run"], tuple(runs)) if "run" in values else None
    since = bound("from", values["from"], end=False) if "from" in values else None
    until = bound("to", values["to"], end=True) if "to" in values else None
    if since is not None and until is not None and since >= until:
        raise FilterError("from must come before to")
    return Filters(
        runs=tuple(range(len(runs))) if run is None else (list(runs).index(run),),
        symbol=choice("symbol", values["symbol"], tuple(symbols)) if "symbol" in values else None,
        slot=number("slot", values["slot"], 9999) if "slot" in values else None,
        bot_id=number("bot", values["bot"], MAX_ID) if "bot" in values else None,
        kind=choice("kind", values["kind"], KINDS) if "kind" in values else None,
        action=choice("action", values["action"], ACTIONS) if "action" in values else None,
        forced=choice("forced", values["forced"], CHOICES) == "yes" if "forced" in values else None,
        gate=choice("gate", values["gate"], CHOICES) == "yes" if "gate" in values else None,
        reason=values.get("reason"),
        since=since, until=until,
        before=cursor(values["before"], len(runs)) if "before" in values else None,
        limit=max(1, number("limit", values["limit"], MAX_PAGE_ROWS)) if "limit" in values else PAGE_ROWS,
    )


def where(f: Filters) -> tuple[str, list]:
    """The WHERE clause and its values for every filter but the cursor."""
    clauses: list[tuple[str, object]] = []
    if f.symbol is not None:
        clauses.append(("d.symbol = ?", f.symbol))
    if f.slot is not None:
        clauses.append(("b.slot = ?", f.slot))
    if f.bot_id is not None:
        clauses.append(("d.bot_id = ?", f.bot_id))
    if f.kind is not None:
        clauses.append(("b.origin = ?" if f.kind == "twin" else "b.origin != ?", TWIN_ORIGIN))
    if f.action is not None:
        clauses.append(("d.action = ?", f.action))
    if f.forced is not None:
        clauses.append(("d.forced = ?", int(f.forced)))
    if f.gate is not None:
        clauses.append(("d.gate_passed = ?", int(f.gate)))
    if f.reason is not None:
        clauses.append(("d.reason LIKE ? ESCAPE '\\'", like(f.reason)))
    if f.since is not None:
        clauses.append(("d.ts >= ?", f.since))
    if f.until is not None:
        clauses.append(("d.ts < ?", f.until))
    return " AND ".join(sql for sql, _ in clauses) or "1", [value for _, value in clauses]


def older(before: Cursor | None, run: int) -> tuple[str, list]:
    """Rows of one run whose key (time, run, id) sorts below the cursor, every row without one."""
    if before is None:
        return "1", []
    if run < before.run:
        return "d.ts <= ?", [before.ts]
    if run > before.run:
        return "d.ts < ?", [before.ts]
    return "(d.ts < ? OR (d.ts = ? AND d.id < ?))", [before.ts, before.ts, before.id]


def fetch(conn: sqlite3.Connection, run: int, name: str, f: Filters) -> list[dict]:
    """Up to one more row than a page from one run, newest first, so the caller knows whether another page follows."""
    sql, params = where(f)
    page_sql, page_params = older(f.before, run)
    rows = conn.execute(f"{ROW} WHERE {sql} AND {page_sql} ORDER BY d.ts DESC, d.id DESC LIMIT ?",
                        [*params, *page_params, f.limit + 1])
    return [{"run": name, "run_index": run, **dict(row)} for row in rows]


def counts(conn: sqlite3.Connection, f: Filters) -> dict:
    """The counts of every decision of one run matching the filters, whatever the page, with the actions taken."""
    sql, params = where(f)
    out = dict(conn.execute(f"{COUNTS} WHERE {sql}", params).fetchone())
    out["actions"] = dict(conn.execute(f"SELECT d.action, count(*) FROM decisions d JOIN bots b "
                                       f"ON b.bot_id = d.bot_id WHERE {sql} GROUP BY d.action", params).fetchall())
    return out


def rates(total: dict) -> dict:
    """The counts with the share of directional matured decisions whose sign was right and the mean label in the
    predicted direction."""
    directional = total["directional"]
    return {**total, "hit_rate": total["right"] / directional if directional else None,
            "mean_directional_label_bps": total["directional_label_bps"] / directional if directional else None}


def combine(by_run: Mapping[str, dict]) -> dict:
    """The counts of several runs added together."""
    total = {name: sum(run[name] for run in by_run.values()) for name in SUMS}
    actions: dict[str, int] = {}
    for run in by_run.values():
        for action, n in run["actions"].items():
            actions[action] = actions.get(action, 0) + n
    return rates({**total, "actions": actions})


def page(conns: Sequence[sqlite3.Connection | None], names: Sequence[str], f: Filters) -> dict:
    """One page of decisions newest first from every chosen run whose database is open, the cursor of the next page,
    None on the last, and the counts of every matching decision, in total and per run."""
    rows: list[dict] = []
    by_run: dict[str, dict] = {}
    for run in f.runs:
        if conns[run] is None:
            continue
        rows += fetch(conns[run], run, names[run], f)
        by_run[names[run]] = counts(conns[run], f)
    rows.sort(key=lambda r: (r["ts"], r["run_index"], r["id"]), reverse=True)
    more, rows = len(rows) > f.limit, rows[:f.limit]
    last = rows[-1] if rows else None
    return {"rows": rows, "next": Cursor(last["ts"], last["run_index"], last["id"]).text() if more else None,
            "summary": combine(by_run), "by_run": {name: rates(run) for name, run in by_run.items()},
            "filters": {field.name: getattr(f, field.name) for field in fields(f) if field.name != "before"}}


def position_after(before: str, action: str, fills: Sequence[Mapping]) -> str | None:
    """The position once the orders of the decision's minute settled, the position before when no order went out, and
    None while one is still pending."""
    if not fills:
        return before if action in ("none", "hold") else None
    position = before
    for fill in fills:
        if fill["status"] != "filled":
            break
        position = "flat" if fill["leg"] == "close" else "long" if fill["side"] == BUY else "short"
    return position


def detail(conn: sqlite3.Connection, name: str, decision_id: int) -> dict | None:
    """One decision with its bot, the fills and trades of its minute, the position after them and its label, or None
    when the run has no such decision."""
    row = conn.execute(DETAIL, (decision_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    fills = [dict(f) for f in conn.execute(FILLS, (d["bot_id"], d["ts"]))]
    trade_ids = sorted({f["trade_id"] for f in fills if f["trade_id"] is not None})
    trades = [dict(t) for t in conn.execute(TRADES.format(",".join("?" * len(trade_ids))), trade_ids)]
    evaluation = conn.execute("SELECT status, entry_price, exit_price, label_bps FROM stock_labels "
                              "WHERE ts = ? AND symbol = ?", (d["ts"], d["symbol"])).fetchone()
    label, pred = d["label_bps"], d["prediction_bps"]
    return {
        "run": name, "id": d["id"], "ts": d["ts"], "symbol": d["symbol"], "action": d["action"],
        "reason": d["reason"], "forced": bool(d["forced"]), "gate_passed": bool(d["gate_passed"]),
        "prediction_bps": pred, "required_edge_bps": d["required_edge_bps"],
        "cost_estimate_bps": d["cost_estimate_bps"], "margin_bps": d["margin_bps"], "ref_price": d["ref_mid"],
        "edge_gap_bps": abs(pred) - d["required_edge_bps"],
        "bot": {"bot_id": d["bot_id"], "slot": d["slot"], "kind": "twin" if d["origin"] == TWIN_ORIGIN else "learner",
                "home_symbol": d["home_symbol"], "state": d["bot_state"], "born_at": d["born_at"],
                "died_at": d["died_at"]},
        "position_before": d["position_before"], "position_after": position_after(d["position_before"], d["action"],
                                                                                  fills),
        "fills": fills, "trades": trades,
        "label": {"label_bps": label, "matured": label is not None,
                  "direction_right": None if label is None or pred == 0 else pred * label > 0,
                  "evaluation": None if evaluation is None else dict(evaluation)},
    }
