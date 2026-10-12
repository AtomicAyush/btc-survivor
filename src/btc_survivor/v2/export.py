"""The export command of PLAN_V2 section 3, every table of one stock run database as CSV under data/v2/exports, for
analysis on this machine.

It opens the run database read only through the stock store's reader, so a database of another kind or a version 1
database is refused, and reads every table inside one read transaction, so the files agree with each other while the
stock process writes on. It writes one file per table with a header row, in insertion order, into
data/v2/exports/<run name> beside the run's data folder, a folder only the owner can open, replacing the files of an
earlier export of the same run. The checkpoints table is left out, since its arrays are binary model state. By default
it prints only the row count of each table and one SHA 256 over the files written, never a row.

    btc_survivor_v2 export configs/v2/s1_prices.toml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sqlite3
import sys
from pathlib import Path

from btc_survivor.store import StoreError
from btc_survivor.v2 import output, redact
from btc_survivor.v2.config import ConfigError, StockConfig, load_stock_config
from btc_survivor.v2.stocks import store

TABLES = tuple(table for table in store.TABLES if table != "checkpoints")
EXPECTED = (ConfigError, StoreError, OSError)


def export_dir(config: StockConfig) -> Path:
    """data/v2/exports/<run name>, beside the run's own data folder."""
    return config.data_dir.parent / "exports" / config.run.name


def owner_folder(path: Path) -> None:
    """Make the folder and any missing parent under umask 077, then set the folder to mode 700 even if it existed."""
    umask = os.umask(0o077)
    try:
        path.mkdir(parents=True, exist_ok=True)
    finally:
        os.umask(umask)
    os.chmod(path, 0o700)


def write_table(conn: sqlite3.Connection, table: str, path: Path) -> int:
    """Write one table to a mode 600 file in insertion order with its column names first, returning the rows written.
    An empty cell is a null, and a float is written with every digit, so it reads back exactly."""
    cursor = conn.execute(f"SELECT * FROM {table} ORDER BY rowid")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with open(fd, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([column[0] for column in cursor.description])
        count = 0
        for row in cursor:
            writer.writerow(row)
            count += 1
    return count


def export(conn: sqlite3.Connection, out_dir: Path) -> tuple[dict[str, int], str]:
    """Write every exported table into out_dir within one read transaction, returning the rows per table and the
    SHA 256 over each table's name and file in table order."""
    owner_folder(out_dir)
    counts, digest = {}, hashlib.sha256()
    conn.execute("BEGIN")
    try:
        for table in TABLES:
            path = out_dir / f"{table}.csv"
            counts[table] = write_table(conn, table, path)
            digest.update(f"{table}\n".encode())
            digest.update(path.read_bytes())
    finally:
        conn.execute("ROLLBACK")
    return counts, digest.hexdigest()


def report(counts: dict[str, int], digest: str) -> output.Report:
    """The default output, the row count of every table, the total and the hash, never a row."""
    return output.Report("export", (
        ("verdict", True),
        ("tables", output.Count(len(counts))),
        ("rows", output.Count(sum(counts.values()))),
        *((f"{table} rows", output.Count(n)) for table, n in counts.items()),
        ("export hash", output.Hash(digest)),
    ))


DESCRIPTION = ("Write every table of one stock run database as CSV to data/v2/exports/<run name>, a folder only the "
               "owner can open. It prints only row counts and a hash.")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The export arguments, for this module's own parser or the btc_survivor_v2 subcommand."""
    parser.add_argument("config", metavar="CONFIG", type=Path,
                        help="a stock run config, such as configs/v2/s1_prices.toml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="btc_survivor_v2 export", description=DESCRIPTION)
    add_arguments(parser)
    return parser


def run(args: argparse.Namespace) -> int:
    """Export one run and print the counts, or print an expected failure as one error line and exit 2, refusing a
    database of another kind before any folder is made."""
    try:
        config = load_stock_config(args.config)
        conn = store.open_reader(config.db_path)
        try:
            counts, digest = export(conn, export_dir(config))
        finally:
            conn.close()
    except EXPECTED as exc:
        print(f"error: {redact.redact(str(exc))}", file=sys.stderr)
        return 2
    output.emit(report(counts, digest))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run export from the command line."""
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
