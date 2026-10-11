"""The export command, one CSV file per table, for charts and analysis on another machine."""

from __future__ import annotations

import csv
from pathlib import Path

from btc_survivor.cli import data
from btc_survivor.config import Config

TABLES = ("bots", "trades", "fills", "decisions", "equity_snapshots", "deaths", "lineage", "benchmark", "forecasts")


def export(cfg: Config, out_dir: Path) -> dict[str, int]:
    """Write each table to out_dir in insertion order and return the row counts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = data.connect(cfg.db_path)
    counts = {}
    try:
        for table in TABLES:
            cursor = conn.execute(f"SELECT * FROM {table} ORDER BY rowid")
            with (out_dir / f"{table}.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow([c[0] for c in cursor.description])
                n = 0
                for row in cursor:
                    writer.writerow(row)
                    n += 1
            counts[table] = n
    finally:
        conn.close()
    return counts
