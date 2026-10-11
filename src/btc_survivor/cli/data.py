"""Read only views of a run's database, shared by the reporting commands and the dashboard.

Bot details come from the newest checkpoint, so they can be up to checkpoint_every_seconds old. The live price comes
from runtime_status, which the runner writes once a minute, and positions are marked at that price.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from pathlib import Path

from btc_survivor.clock import MS_PER_DAY, MS_PER_HOUR, MS_PER_MINUTE, iso, parse_iso
from btc_survivor.config import Config
from btc_survivor.types import LONG, SHORT

STALE_STATUS_MS = 2 * MS_PER_MINUTE
SIDES = {LONG: "long", SHORT: "short", 0: "flat"}


def connect(path: Path) -> sqlite3.Connection:
    """Open a run database read only, so a report can never change a run."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no database at {path}, has this run been started")
    uri = path.resolve().as_uri() + "?mode=ro"
    if not path.with_name(path.name + "-wal").exists():
        uri += "&immutable=1"  # no writer has it open, and some SQLite builds cannot open it read only otherwise
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def checkpoint_meta(conn: sqlite3.Connection) -> dict | None:
    """The scalar part of the newest checkpoint, with its timestamp, or None before the first one."""
    row = conn.execute("SELECT ts, meta FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        return None
    return {**json.loads(row["meta"]), "checkpoint_ts": row["ts"]}


def runtime(conn: sqlite3.Connection, now_ms: int) -> dict | None:
    row = conn.execute("SELECT * FROM runtime_status WHERE id = 1").fetchone()
    if row is None:
        return None
    out = dict(row)
    out["age_s"] = (now_ms - parse_iso(row["ts"])) / 1000
    return out


def meta_value(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else row["value"]


def threshold_estimate(cfg: Config) -> float:
    """The death line for a flat bot, with spread and impact taken as zero, as section 3 of the plan does."""
    costs = cfg.costs
    return (2 * costs.taker_fee_bps + 2 * costs.extra_slippage_bps) / 1e4 * cfg.trading.notional \
        + costs.daily_running_cost


def bots(conn: sqlite3.Connection, cfg: Config, meta: dict | None, live: dict | None, now_ms: int) -> list[dict]:
    """One dict per living or dying bot from the newest checkpoint, marked at the live price when there is one.

    Ages use the run's own clock from the checkpoint, so a replayed database reads the same as a live one.
    """
    if meta is None:
        return []
    dying = set(meta["dying"])
    skill = {r["bot_id"]: r for r in conn.execute(
        "SELECT bot_id, skill_corr, skill_n FROM equity_snapshots WHERE ts = (SELECT max(ts) FROM equity_snapshots)")}
    burn = burn_rates(conn, meta["now_ms"])
    force_minutes = cfg.trading.force_trade_after_hours * 60
    out = []
    for d in meta["bots"]:
        acct = d["account"]
        side, qty, entry = acct["side"], acct["qty"], acct["entry_price"]
        equity = d["equity"]
        mark = None
        if side and live and live.get("best_bid") and live.get("best_ask"):
            mark = live["best_bid"] if side == LONG else live["best_ask"]
            equity = acct["cash"] + side * qty * (mark - entry)
        threshold = d["death_threshold"] or threshold_estimate(cfg)
        rate = burn.get(d["bot_id"])
        state = "dying" if d["bot_id"] in dying else d["state"]
        last = conn.execute("SELECT ts, action, reason FROM decisions WHERE bot_id = ? ORDER BY ts DESC LIMIT 1",
                            (d["bot_id"],)).fetchone()
        snap = skill.get(d["bot_id"])
        out.append({
            "bot_id": d["bot_id"], "kind": "twin" if d["twin"] else "learner", "slot": d["slot"],
            "generation": d["generation"], "origin": d["origin"], "parent_id": d["parent_id"], "state": state,
            "age_hours": (meta["now_ms"] - d["born_ms"]) / MS_PER_HOUR, "trained_minutes": d["trained_minutes"],
            "warmup_minutes": d["warmup_minutes"], "equity": equity, "cash": acct["cash"],
            "pnl": equity - cfg.trading.start_equity, "trades": acct["trades"],
            "win_rate": acct["wins"] / acct["trades"] if acct["trades"] else None,
            "fees": acct["fees"], "rent": acct["rent"], "funding": acct["funding"],
            "position": SIDES[side], "qty": qty, "entry_price": entry if side else None, "mark": mark,
            "skill_corr": snap["skill_corr"] if snap else None, "skill_n": snap["skill_n"] if snap else 0,
            "burn_per_day": rate, "days_left": (equity - threshold) / rate if rate and rate > 0 else None,
            "forced_in_hours": None if side or d["twin"] else max(
                force_minutes - d["minutes_since_open"], d["warmup_minutes"] - d["trained_minutes"], 0) / 60,
            "margin_bps": d["genes"][0], "lr": d["genes"][1],
            "last_action": last["action"] if last else None, "last_reason": last["reason"] if last else None,
            "last_decision_ts": last["ts"] if last else None,
        })
    return out


def burn_rates(conn: sqlite3.Connection, now_ms: int) -> dict[int, float]:
    """Each bot's equity loss per day over its equity snapshots from the last 24 hours, when they span an hour."""
    cutoff = now_ms - MS_PER_DAY
    first: dict[int, tuple[int, float]] = {}
    last: dict[int, tuple[int, float]] = {}
    for r in conn.execute("SELECT bot_id, ts, equity FROM equity_snapshots WHERE ts >= ? ORDER BY ts",
                          (iso(cutoff),)):
        point = (parse_iso(r["ts"]), r["equity"])
        first.setdefault(r["bot_id"], point)
        last[r["bot_id"]] = point
    rates = {}
    for bot_id, (t0, e0) in first.items():
        t1, e1 = last[bot_id]
        if t1 - t0 >= MS_PER_HOUR:
            rates[bot_id] = (e0 - e1) / ((t1 - t0) / MS_PER_DAY)
    return rates


def summary(conn: sqlite3.Connection, cfg: Config, meta: dict | None, rows: list[dict], live: dict | None,
            now_ms: int) -> dict:
    """Population totals, benchmarks, the launch pipeline stage and any warnings."""
    start = cfg.trading.start_equity
    learners = [b for b in rows if b["kind"] == "learner" and b["state"] != "dying"]
    equities = [b["equity"] for b in learners]
    dead = conn.execute("SELECT count(*) AS n, coalesce(sum(d.final_equity), 0) AS final FROM deaths d "
                        "JOIN bots b USING (bot_id) WHERE b.origin != 'control'").fetchone()
    born = conn.execute("SELECT count(*) FROM bots WHERE origin != 'control'").fetchone()[0]
    bench = conn.execute("SELECT * FROM benchmark ORDER BY ts DESC LIMIT 1").fetchone()
    healthy = meta["healthy_minutes"] if meta else 0
    warnings = []
    if live is None or live["age_s"] * 1000 > STALE_STATUS_MS:
        warnings.append("the system is not running, runtime_status is missing or more than two minutes old")
    elif live["feed_state"] != "healthy":
        warnings.append(f"the feed is {live['feed_state']}, so bots are paused")
    checkpoint_age = (now_ms - parse_iso(meta["checkpoint_ts"])) / 1000 if meta else None
    return {
        "launch_ts": meta_value(conn, "launch_ts"), "healthy_days": healthy / 1440,
        "stage": stage(cfg, meta, learners), "living": len(learners),
        "mean_equity": statistics.fmean(equities) if equities else None,
        "median_equity": statistics.median(equities) if equities else None,
        "deaths": dead["n"], "born": born,
        "all_bots_equity": sum(equities) + dead["final"], "all_bots_capital": born * start,
        "buy_hold": bench["buy_hold_equity"] if bench else None, "matched": bench["matched_equity"] if bench else None,
        "cash": bench["cash_equity"] if bench else None, "twin_mean": bench["twin_mean_equity"] if bench else None,
        "price": live["mid"] if live else None, "price_age_s": live["age_s"] if live else None,
        "feed_state": live["feed_state"] if live else "unknown", "checkpoint_age_s": checkpoint_age,
        "warnings": warnings,
    }


def stage(cfg: Config, meta: dict | None, learners: list[dict]) -> str:
    """Where a launch stands, from collecting history through training to trading."""
    if meta is None:
        return "starting, the first checkpoint is written within a few minutes of launch"
    history = cfg.time.history_minutes_required
    if meta["valid_total"] < history:
        return f"collecting history, {meta['valid_total']} of {history} healthy minutes"
    if meta["label_scale_n"] < cfg.features.min_samples:
        return f"waiting for outcomes to mature, {meta['label_scale_n']} of {cfg.features.min_samples} labels"
    active = sum(1 for b in learners if b["state"] == "active")
    if active < len(learners):
        return f"training, {active} of {len(learners)} learners have finished warmup"
    return "trading, every learner is active"

