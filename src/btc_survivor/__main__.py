"""Command line entry point. Every command takes --config, defaulting to config.toml."""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
import time
from pathlib import Path

from btc_survivor import __version__
from btc_survivor.cli import evaluate, explain, export, lineage, status
from btc_survivor.config import ConfigError, load_config
from btc_survivor.dashboard import serve
from btc_survivor.engine import EngineError
from btc_survivor.logsetup import setup_logging
from btc_survivor.replay import NEVER_FOR_SELECTION, ReplayError, replay
from btc_survivor.runner import run_live
from btc_survivor.store import SCHEMA_VERSION, Store, StoreError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="btc_survivor",
        description="Paper trading population of tiny neural nets on live keyless BTC market data. No real orders are ever placed.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", default="config.toml", help="path to the TOML config (default config.toml)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-db", help="create an empty database, or check an existing one against the schema")
    sub.add_parser("run", help="run the live system until SIGINT or SIGTERM")
    st = sub.add_parser("status", help="print the living bots, the live price, the benchmarks and a summary")
    st.add_argument("--watch", action="store_true", help="redraw every minute until Ctrl+C")
    st.add_argument("--twins", action="store_true", help="also list the coin flip twins")
    ex = sub.add_parser("explain", help="print one bot's life story and why it died")
    ex.add_argument("bot_id", type=int)
    ln = sub.add_parser("lineage", help="print a bot's chain of parents back to its random ancestor")
    ln.add_argument("bot_id", type=int)
    out = sub.add_parser("export", help="write every table to CSV files")
    out.add_argument("--out", required=True, type=Path)
    ev = sub.add_parser("evaluate", help="print the success criteria from PLAN section 21 for this run")
    ev.add_argument("--final-day", type=float, default=28.0, help="the day that counts, 28 for runs A and B, 56 for C")
    dash = sub.add_parser("dashboard", help="serve a read only dashboard on 127.0.0.1 until Ctrl+C")
    dash.add_argument("configs", nargs="*", type=Path, help="run configs to show, default the --config file")
    dash.add_argument("--port", type=int, default=8050)
    rep = sub.add_parser("replay", help="run recordings through the pipeline into a new folder. " + NEVER_FOR_SELECTION,
                         description=NEVER_FOR_SELECTION)
    rep.add_argument("--out", required=True, type=Path, help="a new or empty folder for the replay database")
    rep.add_argument("files", nargs="+", type=Path, help="recording files or folders of them")
    return parser


def cmd_init_db(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    path = config.db_path
    store = Store.open(path)
    created = store.created
    store.close()
    if created:
        print(f"created {path} with schema version {SCHEMA_VERSION}")
    else:
        print(f"{path} matches schema version {SCHEMA_VERSION}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    setup_logging(config.run.log_level, config.data_dir / "logs")
    return asyncio.run(run_live(config))


def cmd_status(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.watch:
        status.watch(config, args.twins)
    else:
        print(status.render(config, time.time_ns() // 1_000_000, args.twins))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    print(explain.render(explain.life(load_config(args.config), args.bot_id, time.time_ns() // 1_000_000)))
    return 0


def cmd_lineage(args: argparse.Namespace) -> int:
    print(lineage.render(lineage.chain(load_config(args.config), args.bot_id, time.time_ns() // 1_000_000)))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    counts = export.export(load_config(args.config), args.out)
    print(", ".join(f"{table} {n}" for table, n in counts.items()) + f" rows written to {args.out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    print(evaluate.render(evaluate.evaluate(load_config(args.config), args.final_day)))
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    serve([load_config(path) for path in (args.configs or [args.config])], args.port)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    setup_logging(config.run.log_level)
    path = replay(config, args.files, args.out)
    print(f"replayed into {path}. {NEVER_FOR_SELECTION}")
    return 0


COMMANDS = {"init-db": cmd_init_db, "run": cmd_run, "status": cmd_status, "explain": cmd_explain,
            "lineage": cmd_lineage, "export": cmd_export, "evaluate": cmd_evaluate, "dashboard": cmd_dashboard,
            "replay": cmd_replay}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (ConfigError, StoreError, EngineError, ReplayError, OSError, LookupError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
