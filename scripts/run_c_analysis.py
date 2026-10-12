"""Run C analyses from the ledger, read only. Any run can be read, run C is the one where they matter.

Five things, all descriptive and none part of the success criteria.

1. Notional over equity at every open. Opening a position moves only the fee, the broker never checks buying power,
   and the notional is fixed, so as equity falls the exposure grows. Equity before an open is the bot's newest equity
   snapshot at or before the decision minute, less the fees of any fill it made since, such as the close of a flip.
   A close since the snapshot is counted at the snapshot's mark, so the figure is good to the price move over at
   most one snapshot interval, five minutes in the live runs.
2. The largest of those leverages.
3. Death days with and without a margin cap, 5x by default. With the cap a bot whose open would exceed it stops
   trading at that open and lives on rent until it crosses its death threshold. Recorded deaths come from the deaths
   table, and the arithmetic path of a bot that trades only when forced is printed beside them.
4. The spread of death days against the effect of real skill, from a seeded Monte Carlo of quiet bots under the run's
   death rule, the one the external review's verifier ran, next to the spread of the founders' recorded death days.
5. Parents' skill against the pool they were drawn from. Each mutated birth's parent skill correlation, written in
   the lineage table at birth, beside the skill of the other learners that could have been parents then, and beside
   every other learner alive then, from their newest snapshots.

The arithmetic and the Monte Carlo space a quiet bot's forced opens the way the engine does. An open resets the forced
timer in its decision minute, the timer fires once force_trade_after_hours of healthy minutes have passed, and the bot
then waits for its next phase minute. So forced opens come the timer rounded up to a whole number of horizons apart,
12 hours in all four runs, because 720 minutes is a whole number of 30 and 240 minute horizons. Only a gap in the feed
adds a wait, and every figure here is in healthy days. The review's verifier spaced them 12.25 hours apart, which
--period-hours 12.25 reproduces.

    python scripts/run_c_analysis.py configs/run_c.toml [--copies DIR] [--cap 5] [--every-open] [--period-hours H]

With --copies the database is read from DIR/<run folder>/btc_survivor.db, a copy, instead of the live run. Every
database is opened read only.
"""

from __future__ import annotations

import argparse
import bisect
import math
import sqlite3
import statistics
import sys
from pathlib import Path

import numpy as np

from btc_survivor.cli import data
from btc_survivor.config import Config, load_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from population_value import on_copies  # noqa: E402

TWIN_ORIGIN = "control"
CAP = 5.0
# The verifier's Monte Carlo of quiet bots in run C. A founder's first forced open comes a 12 hour timer after a random
# delay of up to 12 hours and never before it turns active at about 14.6 hours, about 18.3 hours on average, and the
# verifier used 19. Then one round trip of one horizon comes every forced period, see forced_period_minutes. The 30
# minute return spread is 32 bps, and a stop at stop_loss_pct caps a trade's loss.
MC_SEED = 7
MC_BOTS = 20_000
MC_ROUNDS = 200
MC_SIGMA = 32e-4
FIRST_OPEN_HOURS = 19.0
MC_RHOS = (0.0, 0.1, 0.3)


def forced_period_minutes(cfg: Config) -> int:
    """Healthy minutes from one forced open of a quiet bot to the next, as the engine spaces them.

    The bot resets its timer when the open fills, in the decision minute, and counts one per healthy minute. The timer
    is due at the first count of at least force_trade_after_hours times 60, and the bot decides only on its phase
    minutes, one per horizon. A position is held for one horizon and forced exploration only opens from flat, so the
    spacing is never under two horizons.
    """
    horizon = cfg.time.horizon_minutes
    due = math.ceil(cfg.trading.force_trade_after_hours * 60)
    return max(-(-due // horizon) * horizon, 2 * horizon)


def _period_hours(cfg: Config, period_hours: float | None) -> float:
    return forced_period_minutes(cfg) / 60 if period_hours is None else period_hours


class HealthyDays:
    """Days of healthy data at a time, from the run's one benchmark row per healthy minute, read once."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.minutes = [r[0] for r in conn.execute("SELECT ts FROM benchmark ORDER BY ts")]

    def at(self, ts: str) -> float:
        return bisect.bisect_right(self.minutes, ts) / 1440

    @property
    def total(self) -> float:
        return len(self.minutes) / 1440


def opens(conn: sqlite3.Connection, cfg: Config, days: HealthyDays) -> list[dict]:
    """Every filled open with the bot's equity just before it and the notional over that equity, oldest first."""
    start = cfg.trading.start_equity
    out = []
    for f in conn.execute("SELECT f.id, f.bot_id, f.decided_at, f.qty, f.price, f.fee, b.origin FROM fills f "
                          "JOIN bots b USING (bot_id) WHERE f.leg = 'open' AND f.status = 'filled' "
                          "ORDER BY f.decided_at, f.id").fetchall():
        snap = conn.execute("SELECT ts, equity FROM equity_snapshots WHERE bot_id = ? AND ts <= ? ORDER BY ts DESC "
                            "LIMIT 1", (f["bot_id"], f["decided_at"])).fetchone()
        equity, since = (snap["equity"], snap["ts"]) if snap is not None else (start, "")
        equity -= conn.execute("SELECT coalesce(sum(fee), 0) FROM fills WHERE bot_id = ? AND status = 'filled' "
                               "AND decided_at >= ? AND id < ?", (f["bot_id"], since, f["id"])).fetchone()[0]
        notional = f["qty"] * f["price"]
        after = equity - f["fee"]
        out.append({"bot_id": f["bot_id"], "kind": "twin" if f["origin"] == TWIN_ORIGIN else "learner",
                    "ts": f["decided_at"], "day": days.at(f["decided_at"]), "equity": equity,
                    "notional": notional, "fee": f["fee"],
                    "before": notional / equity if equity > 0 else math.inf,
                    "after": notional / after if after > 0 else math.inf})
    return out


def capped_deaths(conn: sqlite3.Connection, cfg: Config, rows: list[dict], cap: float,
                  days: HealthyDays) -> list[dict]:
    """Each recorded death with the day it would have come under the cap, the same day when no open exceeded it."""
    rent = cfg.costs.daily_running_cost
    by_bot: dict[int, list[dict]] = {}
    for o in rows:
        by_bot.setdefault(o["bot_id"], []).append(o)
    out = []
    for d in conn.execute("SELECT d.bot_id, d.died_at, d.threshold, d.final_equity, b.origin, b.born_at, "
                          "b.born_at = (SELECT min(born_at) FROM bots) AS founder FROM deaths d "
                          "JOIN bots b USING (bot_id) ORDER BY d.died_at").fetchall():
        mine = by_bot.get(d["bot_id"], [])
        day = days.at(d["died_at"])
        stop = next((o for o in mine if o["before"] > cap), None)
        capped = day if stop is None else stop["day"] + max(stop["equity"] - d["threshold"], 0.0) / rent
        out.append({"bot_id": d["bot_id"], "kind": "twin" if d["origin"] == TWIN_ORIGIN else "learner",
                    "founder": bool(d["founder"]), "day": day, "capped_day": capped, "cap_binds": stop is not None,
                    "last_open_before": mine[-1]["before"] if mine else None})
    return out


def arithmetic_path(cfg: Config, cap: float, first_open_hours: float = FIRST_OPEN_HOURS,
                    period_hours: float | None = None) -> dict:
    """A bot that trades only when forced and never gains or loses on the market, from the config alone.

    Rent runs from launch, each leg costs the taker fee plus the extra slippage on the notional, and a round trip of
    one horizon comes every forced period, the engine's unless period_hours is given. The path stops at the first open
    the cap refuses, then the bot lives on rent, which gives the capped death day.
    """
    costs, trading = cfg.costs, cfg.trading
    rent_h = costs.daily_running_cost / 24
    leg = trading.notional * (costs.taker_fee_bps + costs.extra_slippage_bps) / 1e4
    hold = cfg.time.horizon_minutes / 60
    period = _period_hours(cfg, period_hours)
    threshold = data.threshold_estimate(cfg)
    t, equity, path, capped = first_open_hours, trading.start_equity - first_open_hours * rent_h, [], None
    for _ in range(1_000_000):
        if equity < threshold:
            break
        if capped is None and trading.notional / equity > cap:
            capped = t / 24 + (equity - threshold) / costs.daily_running_cost
        path.append((t / 24, equity, trading.notional / equity, trading.notional / (equity - leg)))
        equity -= 2 * leg + hold * rent_h
        if equity < threshold:
            break
        t += period
        equity -= (period - hold) * rent_h
    death = (t + hold) / 24
    reach = {}
    for multiple in (2, 5, 10, 25):
        hit = next((p for p in path if p[3] >= multiple), None)
        reach[multiple] = None if hit is None else (hit[0], hit[1] - leg)
    return {"opens": len(path), "death_day": death, "capped_death_day": capped if capped is not None else death,
            "reach": reach, "last": path[-1] if path else None, "threshold": threshold,
            "max_alive_leverage": trading.notional / threshold, "period_hours": period, "leg": leg}


def monte_carlo(cfg: Config, rhos=MC_RHOS, n: int = MC_BOTS, seed: int = MC_SEED,
                first_open_hours: float = FIRST_OPEN_HOURS,
                period_hours: float | None = None) -> dict[float, np.ndarray]:
    """Death days of quiet bots whose direction has correlation rho with the next move, one generator for all rhos.

    This is the verifier's Monte Carlo with the run's own numbers and the engine's forced period unless period_hours is
    given. Each round trip pays two legs and rent, gains or loses the move on the notional with a stop, and a bot dies
    when it could not survive the next open.
    """
    rng = np.random.default_rng(seed)
    costs, trading = cfg.costs, cfg.trading
    rent_h = costs.daily_running_cost / 24
    leg = trading.notional * (costs.taker_fee_bps + costs.extra_slippage_bps) / 1e4
    hold = cfg.time.horizon_minutes / 60
    period = _period_hours(cfg, period_hours)
    threshold = data.threshold_estimate(cfg)
    stop = trading.stop_loss_pct / 100 * trading.notional
    out = {}
    for rho in rhos:
        equity = np.full(n, trading.start_equity - first_open_hours * rent_h)
        day = np.full(n, np.nan)
        t = first_open_hours
        for _ in range(MC_ROUNDS):
            alive = np.isnan(day)
            if not alive.any():
                break
            r = rng.normal(0, MC_SIGMA, n)
            signal = rho * r / MC_SIGMA + math.sqrt(1 - rho**2) * rng.normal(0, 1, n)
            gross = np.maximum(np.sign(signal) * r * trading.notional, -stop)
            equity = np.where(alive, equity - 2 * leg + gross - hold * rent_h - (period - hold) * rent_h, equity)
            dead_now = alive & (equity < threshold + 2 * leg)
            day = np.where(dead_now, (t + period) / 24, day)
            t += period
        out[rho] = day
    return out


def pools_at(conn: sqlite3.Connection, cfg: Config, days: HealthyDays, born_at: str,
             parent_id: int | None = None) -> tuple[list[int], list[int]]:
    """The learners alive at a birth minute and those of them the parent tournament could draw from, oldest first.

    Alive means born before the birth minute and not yet replaced in its slot by the end of it. A dead learner's slot
    is refilled in the minute it dies, while its death row waits for its close to fill, a second later or, after a
    rejected close, minutes later, so the slot is the surer sign, and a death row in or before the birth minute also
    counts. Eligible means at least
    parent_min_age_hours of healthy minutes stepped by the end of the birth minute, counted from the first healthy
    minute after the bot's birth, and at least parent_min_trades closed trades by then. The parent, when given, is
    left out of both lists.
    """
    pop = cfg.population
    minute = born_at[:16]
    end = bisect.bisect_right(days.minutes, born_at)
    living, eligible = [], []
    rows = conn.execute("SELECT b.bot_id, b.born_at FROM bots b LEFT JOIN deaths d USING (bot_id) "
                        "WHERE b.origin != ? AND b.born_at < ? AND (d.died_at IS NULL OR substr(d.died_at, 1, 16) > ?) "
                        "AND NOT EXISTS (SELECT 1 FROM bots n WHERE n.slot = b.slot AND n.origin != ? "
                        "AND n.born_at > b.born_at AND n.born_at <= ?) ORDER BY b.born_at, b.bot_id",
                        (TWIN_ORIGIN, born_at, minute, TWIN_ORIGIN, born_at)).fetchall()
    for b in rows:
        if b["bot_id"] == parent_id:
            continue
        living.append(b["bot_id"])
        first = bisect.bisect_right(days.minutes, b["born_at"])
        closed = conn.execute("SELECT count(*) FROM trades WHERE bot_id = ? AND closed_at IS NOT NULL "
                              "AND closed_at <= ?", (b["bot_id"], born_at)).fetchone()[0]
        if end - first >= pop.parent_min_age_hours * 60 and closed >= pop.parent_min_trades:
            eligible.append(b["bot_id"])
    return living, eligible


def _skills(conn: sqlite3.Connection, bot_ids: list[int], at: str) -> list[float]:
    """Each bot's skill correlation from its newest snapshot at or before at, where that snapshot has one."""
    out = []
    for bot_id in bot_ids:
        snap = conn.execute("SELECT skill_corr FROM equity_snapshots WHERE bot_id = ? AND ts <= ? ORDER BY ts DESC "
                            "LIMIT 1", (bot_id, at)).fetchone()
        if snap is not None and snap["skill_corr"] is not None:
            out.append(snap["skill_corr"])
    return out


def parents_against_population(conn: sqlite3.Connection, cfg: Config, days: HealthyDays) -> list[dict]:
    """Each mutated birth with its parent's skill, the other eligible learners' skill and every other living one's.

    Parents are always old and trained, so the fair comparison is the eligible pool the tournament drew from, which
    leaves out the young, the untraded, the dying and the parent itself. The mean over every other living learner is
    kept beside it. Skills come from the newest snapshot at or before the birth minute. The snapshot taken in that
    minute is written after the births and holds the same skill the tournament read.
    """
    out = []
    for lin in conn.execute("SELECT bot_id, parent_id, born_at, parent_skill_corr FROM lineage ORDER BY born_at, bot_id"
                            ).fetchall():
        living, eligible = pools_at(conn, cfg, days, lin["born_at"], lin["parent_id"])
        pool, others = _skills(conn, eligible, lin["born_at"]), _skills(conn, living, lin["born_at"])
        parent = lin["parent_skill_corr"]
        out.append({"bot_id": lin["bot_id"], "parent_id": lin["parent_id"], "born_at": lin["born_at"],
                    "day": days.at(lin["born_at"]), "parent_skill": parent,
                    "eligible_ids": eligible, "eligible_n": len(pool),
                    "eligible_mean": statistics.fmean(pool) if pool else None,
                    "eligible_median": statistics.median(pool) if pool else None,
                    "parent_share_below": (sum(x < parent for x in pool) / len(pool))
                    if pool and parent is not None else None,
                    "living_n": len(others), "living_mean": statistics.fmean(others) if others else None})
    return out


def _x(value: float | None) -> str:
    return "-" if value is None else "inf" if math.isinf(value) else f"{value:.2f}x"


def _mean_sd(values: list[float]) -> str:
    if not values:
        return "none"
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"mean {statistics.fmean(values):.2f}, sd {sd:.2f}, n {len(values)}"


def render(conn: sqlite3.Connection, cfg: Config, cap: float = CAP, every_open: bool = False,
           first_open_hours: float = FIRST_OPEN_HOURS, period_hours: float | None = None) -> str:
    name = cfg.data_dir.name
    days = HealthyDays(conn)
    rows = opens(conn, cfg, days)
    lines = [f"{name}, {days.total:.2f} healthy days, notional {cfg.trading.notional:,.0f} on start equity "
             f"{cfg.trading.start_equity:,.0f}", "",
             "1. Notional over equity at every open, before and after the open's fee"]
    if not rows:
        lines.append("   no opens yet")
    else:
        lines.append("   day  learner opens  median  largest after fee  twin opens  largest after fee")
        for day in sorted({int(o["day"]) for o in rows}):
            ls = [o for o in rows if int(o["day"]) == day and o["kind"] == "learner"]
            ts = [o for o in rows if int(o["day"]) == day and o["kind"] == "twin"]
            median = statistics.median(o["before"] for o in ls) if ls else None
            lines.append(f"   {day:3d}  {len(ls):13d}  {_x(median):>6}"
                         f"  {_x(max((o['after'] for o in ls), default=None)):>17}  {len(ts):10d}  "
                         f"{_x(max((o['after'] for o in ts), default=None)):>17}")
        if every_open:
            lines.append("   every open, time, bot, kind, healthy day, equity before, notional, before fee, after fee")
            lines += [f"   {o['ts']}  bot {o['bot_id']:4d}  {o['kind']:7s}  day {o['day']:6.2f}  equity "
                      f"{o['equity']:10,.2f}  notional {o['notional']:9,.2f}  {_x(o['before']):>7}  {_x(o['after']):>7}"
                      for o in rows]
    lines.append("2. The largest leverage")
    for kind in ("learner", "twin"):
        mine = [o for o in rows if o["kind"] == kind]
        if mine:
            top = max(mine, key=lambda o: o["after"])
            lines.append(f"   {kind}s, {_x(top['before'])} before the fee and {_x(top['after'])} after, bot "
                         f"{top['bot_id']} on day {top['day']:.2f} with equity {top['equity']:,.2f}, over "
                         f"{len(mine)} opens with {sum(o['before'] > cap for o in mine)} above {cap:g}x")
        else:
            lines.append(f"   {kind}s, no opens yet")
    path = arithmetic_path(cfg, cap, first_open_hours, period_hours)
    deaths = capped_deaths(conn, cfg, rows, cap, days)
    lines.append(f"3. Death days, recorded and with a {cap:g}x margin cap that refuses any open above it")
    for kind in ("learner", "twin"):
        mine = [d for d in deaths if d["kind"] == kind]
        if not mine:
            lines.append(f"   {kind}s, no deaths yet")
            continue
        last_over = sum(1 for d in mine if d["last_open_before"] is not None and d["last_open_before"] > cap)
        lines.append(f"   {kind}s, {len(mine)} deaths, recorded day {_mean_sd([d['day'] for d in mine])}, with the "
                     f"cap {_mean_sd([d['capped_day'] for d in mine])}, the cap binds for "
                     f"{sum(d['cap_binds'] for d in mine)}, last open above {cap:g}x for {last_over}")
    dead = {d["bot_id"] for d in deaths}
    for kind in ("learner", "twin"):
        past = {o["bot_id"] for o in rows if o["kind"] == kind and o["before"] > cap} - dead
        if past:
            lines.append(f"   {len(past)} living {kind}s have had an open above {cap:g}x, so under the cap they would "
                         f"now be living on rent")
    reach = ", ".join(f"{m}x on day {r[0]:.1f} at equity {r[1]:,.2f}" if r else f"{m}x never"
                      for m, r in path["reach"].items())
    lines.append(f"   arithmetic for a bot that trades only when forced, no market moves, first open "
                 f"{first_open_hours:g} hours after launch and one every {path['period_hours']:g} hours"
                 + (" as the engine spaces them" if period_hours is None else ", a spacing given on the command line"))
    lines.append(f"   {path['opens']} opens, after the open fee it reaches {reach}")
    if path["last"]:
        d, e, before, after = path["last"]
        lines.append(f"   last open on day {d:.1f} at equity {e:,.2f}, {before:.1f}x before the fee and "
                     f"{after:.1f}x after, death threshold {path['threshold']:.2f} so a living bot carries at most "
                     f"{path['max_alive_leverage']:.1f}x")
    lines.append(f"   death on day {path['death_day']:.1f} without a cap and on day {path['capped_death_day']:.1f} "
                 f"with the {cap:g}x cap")
    lines.append("4. The spread of death days against the effect of real skill")
    if path["opens"] > MC_ROUNDS / 2:
        lines.append(f"   skipped, a quiet bot is expected to die on day {path['death_day']:.0f} after {path['opens']} "
                     f"opens, beyond the {MC_ROUNDS} round trips the Monte Carlo follows")
    else:
        simulated = monte_carlo(cfg, first_open_hours=first_open_hours, period_hours=period_hours)
        base = simulated[MC_RHOS[0]]
        for rho, d in simulated.items():
            outlive = float(np.mean(d > base) + 0.5 * np.mean(d == base))
            lines.append(f"   correlation {rho:.1f}, death day mean {np.nanmean(d):.2f}, sd {np.nanstd(d):.2f}, "
                         f"mean shift {np.nanmean(d) - np.nanmean(base):+.2f} days, outlives a no skill bot "
                         f"{100 * outlive:.0f} percent of the time")
        founders = [d["day"] for d in deaths if d["kind"] == "learner" and d["founder"]]
        lines.append(f"   recorded founder learner death days, {_mean_sd(founders)}")
    lines.append("5. Parents' skill against the eligible pool at each mutated birth")
    births = parents_against_population(conn, cfg, days)
    known = [b for b in births if b["parent_skill"] is not None and b["eligible_mean"] is not None]
    if not births:
        lines.append("   no mutated births yet")
    else:
        lines.append(f"   {len(births)} mutated births, {len(known)} with a parent skill and other eligible learners "
                     f"with a skill to compare, eligible meaning at least {cfg.population.parent_min_age_hours:g} "
                     f"healthy hours old with {cfg.population.parent_min_trades} closed trades")
        if known:
            diff = [b["parent_skill"] - b["eligible_mean"] for b in known]
            sd = statistics.stdev(diff) if len(diff) > 1 else 0.0
            t = statistics.fmean(diff) / (sd / math.sqrt(len(diff))) if sd > 0 else None
            above = sum(b["parent_share_below"] > 0.5 for b in known)
            living = [b["living_mean"] for b in known if b["living_mean"] is not None]
            lines.append(f"   parent skill mean {statistics.fmean(b['parent_skill'] for b in known):+.4f}, eligible "
                         f"pool mean {statistics.fmean(b['eligible_mean'] for b in known):+.4f}, difference "
                         f"{statistics.fmean(diff):+.4f}, t {'-' if t is None else f'{t:+.2f}'} treating births as "
                         f"independent, which they are not, parent above the eligible median at {above} of "
                         f"{len(known)}")
            if living:
                lines.append(f"   every other living learner, young and untraded ones too, mean "
                             f"{statistics.fmean(living):+.4f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run C analyses from the ledger. Read only.")
    parser.add_argument("configs", nargs="+", type=Path, help="run configs, such as configs/run_c.toml")
    parser.add_argument("--copies", type=Path, help="read DIR/<run folder>/btc_survivor.db instead of the live run")
    parser.add_argument("--cap", type=float, default=CAP, help="the margin cap on notional over equity, default 5")
    parser.add_argument("--every-open", action="store_true", help="also print one line per open")
    parser.add_argument("--first-open-hours", type=float, default=FIRST_OPEN_HOURS,
                        help="hours from launch to the first forced open in the arithmetic and the Monte Carlo")
    parser.add_argument("--period-hours", type=float,
                        help="hours between forced opens in the arithmetic and the Monte Carlo instead of the engine's "
                             "spacing, 12.25 reproduces the review's verifier")
    args = parser.parse_args(argv)
    for i, path in enumerate(args.configs):
        cfg = on_copies(load_config(path), args.copies)
        conn = data.connect(cfg.db_path)
        try:
            if i:
                print()
            print(render(conn, cfg, args.cap, args.every_open, args.first_open_hours, args.period_hours))
        finally:
            conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
