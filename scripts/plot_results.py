"""Charts from an export folder, for a laptop. Needs matplotlib from requirements-plot.txt, never installed on the Pi.

    python -m btc_survivor --config configs/run_a.toml export --out exports/run_a
    python scripts/plot_results.py exports/run_a
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path


def read(folder: Path, table: str) -> list[dict]:
    with (folder / f"{table}.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def when(ts: str) -> datetime:
    return datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot an exported run. Reads CSV files only.")
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    bench = read(args.folder, "benchmark")
    snaps = read(args.folder, "equity_snapshots")
    twins = {r["bot_id"] for r in read(args.folder, "bots") if r["origin"] == "control"}
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    by_bot: dict[str, list[tuple[datetime, float]]] = {}
    for r in snaps:
        if r["bot_id"] not in twins:
            by_bot.setdefault(r["bot_id"], []).append((when(r["ts"]), float(r["equity"])))
    for points in by_bot.values():
        top.plot(*zip(*points), color="0.75", linewidth=0.7)
    times = [when(r["ts"]) for r in bench]
    for key, label in (("living_mean_equity", "mean living learner"), ("twin_mean_equity", "twin mean"),
                       ("buy_hold_equity", "buy and hold"), ("cash_equity", "cash")):
        values = [float(r[key]) if r[key] else None for r in bench]
        top.plot(times, values, label=label, linewidth=1.6)
    top.set_ylabel("equity in dollars")
    top.legend()
    top.set_title(f"{args.folder.name}, each grey line is one learner")
    forecasts = [r for r in read(args.folder, "forecasts") if r["label_bps"] and r["predictor"] == "population"]
    if forecasts:
        bottom.scatter([when(r["ts"]) for r in forecasts], [float(r["label_bps"]) for r in forecasts], s=2,
                       color="0.7", label="realized move")
        bottom.plot([when(r["ts"]) for r in forecasts], [float(r["prediction_bps"]) for r in forecasts],
                    linewidth=1.0, label="population forecast")
        bottom.set_ylabel("bps over the horizon")
        bottom.legend()
    out = args.folder / "results.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
