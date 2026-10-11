"""The evaluate command, the success criteria of PLAN section 21 and the outcomes fixed by arithmetic for one run.

Every number is computed from rows up to the moment the run reached its final day of healthy data, so running the
command later never changes the result. Before that day every number is descriptive, and the report says so.
"""

from __future__ import annotations

import sqlite3
from collections import Counter

import numpy as np

from btc_survivor.cli import data
from btc_survivor.cli.stats import block_test, diebold_mariano, skill_test
from btc_survivor.clock import MS_PER_DAY, MS_PER_HOUR, MS_PER_MINUTE, iso, parse_iso
from btc_survivor.config import Config

SKILL_LAGS = 12  # windows in the Newey West standard error, about six hours at a 30 minute horizon
BLOCK_HOURS = 12
BURN_IN_MS = MS_PER_DAY  # every predictor is scored from one day after the first forecast row
OPEN_END = "9999-12-31T23:59:59.999Z"


def final_cutoff(conn: sqlite3.Connection, final_day: float) -> str | None:
    """The time of the last healthy minute of the final day, or None before the run reaches it."""
    row = conn.execute("SELECT ts FROM benchmark ORDER BY ts LIMIT 1 OFFSET ?",
                       (max(int(final_day * 1440) - 1, 0),)).fetchone()
    return row["ts"] if row else None


def healthy_day(conn: sqlite3.Connection, ts: str) -> float:
    """How many days of healthy data the run had at ts, from its one benchmark row per healthy minute."""
    return conn.execute("SELECT count(*) FROM benchmark WHERE ts <= ?", (ts,)).fetchone()[0] / 1440


def forecasts(conn: sqlite3.Connection, horizon: int, burn_in_ms: int = BURN_IN_MS,
              cut: str = OPEN_END) -> dict[str, dict[str, tuple[float, float]]]:
    """Labelled forecast rows at one minute per horizon window, by predictor and minute, so labels never overlap.

    Rows from the first day after the first forecast are left out for every predictor, so startup noise of a fresh
    model is never scored, and so are windows whose label matured after the cutoff.
    """
    first = conn.execute("SELECT min(ts) FROM forecasts").fetchone()[0]
    if first is None:
        return {}
    start = iso(parse_iso(first) + burn_in_ms)
    end = cut if cut == OPEN_END else iso(parse_iso(cut) - horizon * MS_PER_MINUTE)
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for r in conn.execute(
            "SELECT ts, predictor, prediction_bps, label_bps FROM forecasts WHERE label_bps IS NOT NULL "
            "AND ts >= ? AND ts <= ? AND (CAST(strftime('%s', ts) AS INTEGER) / 60) % ? = 0 ORDER BY ts",
            (start, end, horizon)):
        out.setdefault(r["predictor"], {})[r["ts"]] = (r["prediction_bps"], r["label_bps"])
    return out


def criterion_skill(conn: sqlite3.Connection, horizon: int, burn_in_ms: int = BURN_IN_MS,
                    cut: str = OPEN_END) -> dict:
    rows = forecasts(conn, horizon, burn_in_ms, cut)
    population = rows.get("population", {})
    times = sorted(population)
    pred = np.array([population[t][0] for t in times])
    label = np.array([population[t][1] for t in times])
    result = {"skill": skill_test(pred, label, SKILL_LAGS),
              "vs_zero": diebold_mariano(label - pred, label, SKILL_LAGS)}
    linear = rows.get("linear", {})
    shared = [t for t in times if t in linear]
    lin = np.array([linear[t][0] for t in shared])
    lab = np.array([linear[t][1] for t in shared])
    pop = np.array([population[t][0] for t in shared])
    result["vs_linear"] = diebold_mariano(lab - pop, lab - lin, SKILL_LAGS)
    result["linear_skill"] = skill_test(lin, lab, SKILL_LAGS)
    return result


def criterion_twins(conn: sqlite3.Connection, cut: str = OPEN_END) -> dict:
    """Learner minus twin gross profit for trades opened by the same decision, grouped into 12 hour blocks."""
    opens: dict[tuple[int, str], dict[str, int]] = {}
    for r in conn.execute("SELECT f.decided_at, f.trade_id, b.slot, b.origin FROM fills f JOIN bots b USING (bot_id) "
                          "WHERE f.leg = 'open' AND f.status = 'filled' AND f.decided_at <= ?", (cut,)):
        kind = "twin" if r["origin"] == "control" else "learner"
        opens.setdefault((r["slot"], r["decided_at"]), {})[kind] = r["trade_id"]
    gross = {r["id"]: r["pnl_gross"] for r in conn.execute(
        "SELECT id, pnl_gross FROM trades WHERE closed_at IS NOT NULL AND closed_at <= ?", (cut,))}
    blocks: dict[int, list[float]] = {}
    for (_, decided_at), pair in opens.items():
        a, b = pair.get("learner"), pair.get("twin")
        if a in gross and b in gross:
            block = parse_iso(decided_at) // (BLOCK_HOURS * MS_PER_HOUR)
            blocks.setdefault(block, []).append(gross[a] - gross[b])
    return block_test(blocks)


def criterion_survival(conn: sqlite3.Connection, cut: str = OPEN_END) -> dict:
    rows = conn.execute("SELECT b.origin = 'control' AS twin, d.cause, d.lifespan_hours, d.died_at FROM deaths d "
                        "JOIN bots b USING (bot_id) WHERE d.died_at <= ?", (cut,)).fetchall()
    out = {}
    for kind, flag in (("learners", 0), ("twins", 1)):
        mine = [r for r in rows if r["twin"] == flag]
        days = sorted(healthy_day(conn, r["died_at"]) for r in mine)
        out[kind] = {"deaths": len(mine), "causes": dict(Counter(r["cause"] for r in mine)),
                     "mean_lifespan_days": float(np.mean([r["lifespan_hours"] for r in mine])) / 24 if mine else None,
                     "death_days": days}
    return out


def criterion_machinery(conn: sqlite3.Connection, cut: str = OPEN_END) -> dict:
    births = conn.execute("SELECT count(*) FROM lineage WHERE born_at <= ?", (cut,)).fetchone()[0]
    top = conn.execute("SELECT coalesce(max(generation), 0) FROM bots WHERE origin != 'control' AND born_at <= ?",
                       (cut,)).fetchone()[0]
    first_born = {}
    for generation in (1, 2):
        row = conn.execute("SELECT min(born_at) FROM bots WHERE origin != 'control' AND generation = ? "
                           "AND born_at <= ?", (generation, cut)).fetchone()[0]
        first_born[generation] = healthy_day(conn, row) if row else None
    return {"mutated_births": births, "highest_generation": top, "passed": births >= 1 and top >= 2,
            "generation_1_day": first_born[1], "generation_2_day": first_born[2]}


def arithmetic(conn: sqlite3.Connection, cfg: Config, healthy_days: float, cut: str = OPEN_END) -> dict:
    """What fees, notional, start equity and the forced timer imply, next to what happened."""
    costs, trading = cfg.costs, cfg.trading
    round_trip = (2 * costs.taker_fee_bps + 2 * costs.extra_slippage_bps) / 1e4 * trading.notional
    burn = 24 / trading.force_trade_after_hours * round_trip + costs.daily_running_cost
    threshold = data.threshold_estimate(cfg)
    opens = conn.execute("SELECT count(*) AS n, coalesce(sum(forced), 0) AS forced, coalesce(sum(gate_passed), 0) "
                         "AS gate FROM decisions d JOIN bots b USING (bot_id) WHERE b.origin != 'control' AND "
                         "d.action IN ('open_long', 'open_short', 'flip') AND d.ts <= ?", (cut,)).fetchone()
    first_death = conn.execute("SELECT min(d.died_at) FROM deaths d JOIN bots b USING (bot_id) "
                               "WHERE b.origin != 'control' AND d.died_at <= ?", (cut,)).fetchone()[0]
    founders = conn.execute(
        "SELECT s.equity, (SELECT count(*) FROM trades t WHERE t.bot_id = b.bot_id AND t.closed_at <= ?) AS trips "
        "FROM equity_snapshots s JOIN bots b USING (bot_id) WHERE b.origin = 'random' AND b.generation = 0 "
        "AND b.born_at = (SELECT min(born_at) FROM bots) "
        "AND s.ts = (SELECT max(ts) FROM equity_snapshots WHERE ts <= ?)", (cut, cut)).fetchall()
    trips = float(np.mean([r["trips"] for r in founders])) if founders else 0.0
    return {
        "round_trip_dollars": round_trip, "quiet_burn_per_day": burn, "founders": len(founders),
        "founder_round_trips": trips,
        "expected_founder_equity": trading.start_equity - costs.daily_running_cost * healthy_days - round_trip * trips,
        "founder_equity_mean": float(np.mean([r["equity"] for r in founders])) if founders else None,
        "expected_first_death_day": (trading.start_equity - threshold) / burn,
        "first_death_day": healthy_day(conn, first_death) if first_death else None,
        "learner_opens": opens["n"], "forced_opens": opens["forced"], "gate_passes_on_opens": opens["gate"],
    }


def evaluate(cfg: Config, final_day: float, burn_in_ms: int = BURN_IN_MS) -> dict:
    conn = data.connect(cfg.db_path)
    try:
        cut = final_cutoff(conn, final_day)
        meta = data.checkpoint_meta(conn)
        healthy_days = final_day if cut else (meta["healthy_minutes"] if meta else 0) / 1440
        end = cut or OPEN_END
        return {"run": cfg.data_dir.name, "healthy_days": healthy_days, "final_day": final_day,
                "final": cut is not None, "cutoff": cut,
                "skill": criterion_skill(conn, cfg.time.horizon_minutes, burn_in_ms, end),
                "twins": criterion_twins(conn, end), "survival": criterion_survival(conn, end),
                "machinery": criterion_machinery(conn, end), "arithmetic": arithmetic(conn, cfg, healthy_days, end)}
    finally:
        conn.close()


def _f(value, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def _days(days: list[float]) -> str:
    if not days:
        return "none"
    return f"first {days[0]:.1f}, median {days[len(days) // 2]:.1f}, last {days[-1]:.1f}"


def render(r: dict) -> str:
    sk, tw, ar, mach = r["skill"], r["twins"], r["arithmetic"], r["machinery"]
    pop = sk["skill"]
    label = f"final, rows up to {r['cutoff']}" if r["final"] else \
        f"descriptive, the final day is day {r['final_day']:g}"
    lines = [f"{r['run']}, {r['healthy_days']:.2f} healthy days ({label})", "",
             "1. Skill, population forecast against the matured label, one row per window after the first day",
             f"   correlation {_f(pop['corr'], '+.4f')} over {pop['n']} windows, Newey West t {_f(pop['t'], '+.2f')}, "
             f"needs t above 2.33, {'PASS' if pop['passed'] else 'not passed'}",
             f"   detectable with 90 percent power at this length, a correlation of about {_f(pop['detectable'], '.3f')}",
             f"   squared error against predicting zero {_f(sk['vs_zero']['mse_ratio'], '.4f')}, "
             f"Diebold Mariano t {_f(sk['vs_zero']['t'], '+.2f')} (positive favours the population)",
             f"   squared error against the linear baseline {_f(sk['vs_linear']['mse_ratio'], '.4f')}, "
             f"Diebold Mariano t {_f(sk['vs_linear']['t'], '+.2f')}, linear correlation "
             f"{_f(sk['linear_skill']['corr'], '+.4f')}",
             "2. Beats coin flips, learner minus twin gross profit per pair, 12 hour blocks",
             f"   {tw['pairs']} pairs in {tw['blocks']} blocks, mean of block means {_f(tw['mean'], '+.3f')} dollars, "
             f"t {_f(tw['t'], '+.2f')} against {_f(tw['threshold'], '.2f')}, "
             f"{'PASS' if tw['passed'] else 'not passed'}, supporting evidence with about a third of criterion 1's power",
             "3. Survival, days counted in healthy data"]
    for kind, s in r["survival"].items():
        lines.append(f"   {kind}: {s['deaths']} deaths ({_days(s['death_days'])}), mean lifespan "
                     f"{_f(s['mean_lifespan_days'], '.1f')} days, causes {s['causes'] or 'none'}")
    lines += [f"4. Machinery, {mach['mutated_births']} mutated births, highest generation {mach['highest_generation']}, "
              f"generation 1 from day {_f(mach['generation_1_day'], '.1f')}, generation 2 from day "
              f"{_f(mach['generation_2_day'], '.1f')}, {'PASS' if mach['passed'] else 'not yet'}",
              "Outcomes fixed by arithmetic",
              f"   a quiet bot burns {ar['quiet_burn_per_day']:.2f} dollars a day, round trip {ar['round_trip_dollars']:.2f}",
              f"   {ar['founders']} founders with {ar['founder_round_trips']:.1f} round trips each on average, mean equity "
              f"{_f(ar['founder_equity_mean'], ',.2f')} against {ar['expected_founder_equity']:,.2f} from rent and fees, "
              f"which should agree within about 20 dollars",
              f"   first learner death expected near day {ar['expected_first_death_day']:.1f}, happened on day "
              f"{_f(ar['first_death_day'], '.1f')}",
              f"   {ar['forced_opens']} of {ar['learner_opens']} learner opens were forced, "
              f"{ar['gate_passes_on_opens']} passed the gate"]
    return "\n".join(lines)
