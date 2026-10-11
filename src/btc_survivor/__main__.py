"""Command line entry point. Every command takes --config, defaulting to config.toml."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from btc_survivor import __version__
from btc_survivor.config import ConfigError, load_config
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


def cmd_replay(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    setup_logging(config.run.log_level)
    path = replay(config, args.files, args.out)
    print(f"replayed into {path}. {NEVER_FOR_SELECTION}")
    return 0


COMMANDS = {"init-db": cmd_init_db, "run": cmd_run, "replay": cmd_replay}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (ConfigError, StoreError, EngineError, ReplayError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
