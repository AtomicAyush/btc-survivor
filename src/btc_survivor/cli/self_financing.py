"""The self financing value of a run's population, read only, shared by scripts/population_value.py and the dashboard.

The benchmark table's living mean is the mean over the current slots, and a dead learner's slot is refilled in the
same minute by a newborn holding full start equity, so after deaths that mean is topped up with new money. This
module values the population the way an investor who funded the founders once would see it. Every learner ever born
counts, a living one at its newest equity snapshot and a dead one at its final equity less any writeoff, and every
replacement birth costs that investor start equity.

    value = N * S0 + sum over every learner born by then of (equity - S0)

where N is the number of founders and S0 the start equity. A dead learner's residual returns to a pool that pays for
the births, and the outside money is the most that pool ever fell below zero, the money that had to come from
elsewhere. Events are netted by minute with deaths first, because a replacement is born in the minute its slot
empties while a death row waits a second or so for the last close to fill. Before the first death the value per
slot is the mean living equity. Twins are left out, as in the living mean.
"""

from __future__ import annotations

import sqlite3

from btc_survivor.config import Config

TWIN_ORIGIN = "control"


def _learners(conn: sqlite3.Connection, at: str) -> list[sqlite3.Row]:
    return conn.execute("SELECT b.bot_id, b.born_at, d.died_at, d.final_equity, d.writeoff FROM bots b "
                        "LEFT JOIN deaths d ON d.bot_id = b.bot_id AND d.died_at <= ? "
                        "WHERE b.origin != ? AND b.born_at <= ? ORDER BY b.born_at, b.bot_id",
                        (at, TWIN_ORIGIN, at)).fetchall()


def newest_snapshot(conn: sqlite3.Connection) -> str | None:
    """The newest equity snapshot of any learner without a death row, found through the primary key one bot at a
    time, so the dashboard never scans the whole snapshot table for it."""
    best = None
    for (bot_id,) in conn.execute("SELECT bot_id FROM bots b WHERE origin != ? AND NOT EXISTS "
                                  "(SELECT 1 FROM deaths d WHERE d.bot_id = b.bot_id)", (TWIN_ORIGIN,)):
        ts = conn.execute("SELECT max(ts) FROM equity_snapshots WHERE bot_id = ?", (bot_id,)).fetchone()[0]
        if ts is not None and (best is None or ts > best):
            best = ts
    return best


def value_at(conn: sqlite3.Connection, cfg: Config, at: str | None = None) -> dict | None:
    """The self financing value at time at, by default the newest equity snapshot, or None before any snapshot.

    A learner born by then with no death row by then is valued at its newest snapshot at or before then, which is
    its equity before death for one still closing its last position, and at start equity when it was born after the
    newest snapshot. The benchmark row at or before then gives buy and hold, cash and the living mean beside it.
    """
    start = cfg.trading.start_equity
    at = at or newest_snapshot(conn)
    if at is None:
        return None
    rows = _learners(conn, at)
    if not rows:
        return None
    first_born = rows[0]["born_at"]
    founders = sum(1 for r in rows if r["born_at"] == first_born)
    living_equity, dead_residual, writeoffs, living, dead = 0.0, 0.0, 0.0, [], 0
    events: list[tuple[str, int, float]] = []  # minute, 0 for a death and 1 for a birth, amount
    for r in rows:
        if r["died_at"] is not None:
            residual = r["final_equity"] - r["writeoff"]
            dead_residual += residual
            writeoffs += r["writeoff"]
            dead += 1
            events.append((r["died_at"][:16], 0, residual))
        else:
            snap = conn.execute("SELECT equity FROM equity_snapshots WHERE bot_id = ? AND ts <= ? ORDER BY ts DESC "
                                "LIMIT 1", (r["bot_id"], at)).fetchone()
            equity = snap["equity"] if snap is not None else start
            living_equity += equity
            living.append(equity)
        if r["born_at"] != first_born:
            events.append((r["born_at"][:16], 1, -start))
    births = len(rows) - founders
    pool = lowest = 0.0
    for _, _, amount in sorted(events):
        pool += amount
        lowest = min(lowest, pool)
    value = living_equity + dead_residual - births * start
    bench = conn.execute("SELECT ts, buy_hold_equity, cash_equity, living_mean_equity FROM benchmark WHERE ts <= ? "
                         "ORDER BY ts DESC LIMIT 1", (at,)).fetchone()
    healthy = conn.execute("SELECT count(*) FROM benchmark WHERE ts <= ?", (at,)).fetchone()[0]
    return {
        "at": at, "healthy_days": healthy / 1440, "start_equity": start, "founders": founders, "born": len(rows),
        "births": births, "deaths": dead, "living": len(living), "writeoffs": writeoffs,
        "living_equity": living_equity, "dead_residual": dead_residual, "births_cost": births * start,
        "value": value, "value_per_slot": value / founders, "return": value / (founders * start) - 1,
        "outside_money": -lowest if lowest < 0 else 0.0, "pool": pool,
        "living_mean": living_equity / len(living) if living else None,
        "benchmark_ts": bench["ts"] if bench else None,
        "buy_hold": bench["buy_hold_equity"] if bench else None,
        "buy_hold_return": bench["buy_hold_equity"] / start - 1 if bench else None,
        "cash": bench["cash_equity"] if bench else None,
        "cash_return": bench["cash_equity"] / start - 1 if bench else None,
        "benchmark_living_mean": bench["living_mean_equity"] if bench else None,
    }


def daily(conn: sqlite3.Connection, cfg: Config) -> list[dict]:
    """The value at the end of every whole healthy day and at the newest snapshot, oldest first."""
    total = conn.execute("SELECT count(*) FROM benchmark").fetchone()[0]
    out = []
    for day in range(1, total // 1440 + 1):
        ts = conn.execute("SELECT ts FROM benchmark ORDER BY ts LIMIT 1 OFFSET ?", (day * 1440 - 1,)).fetchone()[0]
        row = value_at(conn, cfg, ts)
        if row is not None:
            out.append(row)
    newest = value_at(conn, cfg)
    if newest is not None and (not out or newest["at"] > out[-1]["at"]):
        out.append(newest)
    return out
