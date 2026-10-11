"""Command line entry point. Every command takes --config, defaulting to config.toml."""

from __future__ import annotations

import argparse
import sys

from btc_survivor import __version__
from btc_survivor.config import ConfigError, load_config
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


COMMANDS = {"init-db": cmd_init_db}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (ConfigError, StoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
