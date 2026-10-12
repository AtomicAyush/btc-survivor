"""The self financing value of each run's population, read only, beside buy and hold and cash.

The dashboard's average learner covers the living learners only, and every dead learner's slot is refilled with a
newborn holding full start equity, so after deaths that average is topped up with new money. This script values every
learner ever born instead, a living one at its newest equity snapshot and a dead one at its final equity less any
writeoff, charges each replacement birth its start equity, and reports the outside money those births needed beyond
what the dead returned. The numbers come from btc_survivor.cli.self_financing, the same function the dashboard shows.

    python scripts/population_value.py configs/run_c.toml [configs/run_a.toml ...] [--copies DIR]

With --copies the database is read from DIR/<run folder>/btc_survivor.db, a copy, instead of the live run. Every
database is opened read only.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from btc_survivor.cli import data
from btc_survivor.cli.self_financing import daily
from btc_survivor.config import Config, load_config


def on_copies(cfg: Config, copies: Path | None) -> Config:
    """The config pointed at DIR/<run folder> when copies is given, so nothing reads the live database."""
    if copies is None:
        return cfg
    return dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, data_dir=str(copies / cfg.data_dir.name)))


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:+.2f} percent"


def _usd(value: float | None) -> str:
    return "-" if value is None else f"{value:,.2f}"


def render(name: str, rows: list[dict]) -> str:
    if not rows:
        return f"{name}, no equity snapshots yet"
    v = rows[-1]
    lines = [
        f"{name}, {v['healthy_days']:.2f} healthy days, as of the snapshot at {v['at']}",
        f"   learners ever born {v['born']}, {v['founders']} founders and {v['births']} births, {v['deaths']} deaths, "
        f"{v['living']} living, writeoffs {_usd(v['writeoffs'])}",
        f"   value over every learner ever born {_usd(v['value'])} on {_usd(v['founders'] * v['start_equity'])} "
        f"invested, {_usd(v['value_per_slot'])} per slot, self financing return {_pct(v['return'])}",
        f"   outside money injected by births {_usd(v['outside_money'])}, births cost {_usd(v['births_cost'])} and "
        f"the dead returned {_usd(v['dead_residual'])}",
        f"   beside it per slot, buy and hold {_usd(v['buy_hold'])} ({_pct(v['buy_hold_return'])}), cash "
        f"{_usd(v['cash'])} ({_pct(v['cash_return'])}), living mean {_usd(v['living_mean'])} "
        f"(living learners only, refilled at deaths)",
    ]
    if len(rows) > 1:
        lines.append("   day   value per slot   return   outside money   living mean   buy and hold   cash   deaths")
        for r in rows:
            lines.append(f"   {r['healthy_days']:5.2f}  {r['value_per_slot']:14,.2f}  {100 * r['return']:+6.2f}%  "
                         f"{r['outside_money']:14,.2f}  {_usd(r['living_mean']):>12}  {_usd(r['buy_hold']):>13}  "
                         f"{_usd(r['cash']):>6}  {r['deaths']:6d}")
    return "\n".join(lines)


def report(cfg: Config) -> str:
    conn = data.connect(cfg.db_path)
    try:
        return render(cfg.data_dir.name, daily(conn, cfg))
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Self financing value of each run's population. Read only.")
    parser.add_argument("configs", nargs="+", type=Path, help="run configs, such as configs/run_c.toml")
    parser.add_argument("--copies", type=Path, help="read DIR/<run folder>/btc_survivor.db instead of the live run")
    args = parser.parse_args(argv)
    for i, path in enumerate(args.configs):
        if i:
            print()
        print(report(on_copies(load_config(path), args.copies)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
