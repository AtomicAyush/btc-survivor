"""The btc_survivor_v2 command line of PLAN_V2 section 3. By default every subcommand prints only counts, pass or
fail verdicts, hashes and timings, and --detail adds rows for the owner's own terminal.

    btc_survivor_v2 init-db configs/v2/news.toml
    btc_survivor_v2 init-db configs/v2/s1_prices.toml
    btc_survivor_v2 stocks configs/v2/s1_prices.toml configs/v2/s2_news.toml [--dry-run]
    btc_survivor_v2 news configs/v2/news.toml [--dry-run]
    btc_survivor_v2 news-status [--days 7] [--detail [--samples 50]]
    btc_survivor_v2 tape-check data/v2/raw_stocks data/v2/checks/official_YYYY_MM_DD.toml [--detail]
    btc_survivor_v2 evaluate configs/v2/s1_prices.toml configs/v2/s2_news.toml [--detail]
    btc_survivor_v2 status configs/v2/s1_prices.toml configs/v2/s2_news.toml [--detail]
    btc_survivor_v2 export configs/v2/s1_prices.toml
    btc_survivor_v2 replay configs/v2/s2_news.toml data/v2/raw_stocks --out data/v2/replays/one [--detail]
    btc_survivor_v2 dashboard configs/v2/s1_prices.toml configs/v2/s2_news.toml [--port 8060]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

from btc_survivor import __version__
from btc_survivor.replay import NEVER_FOR_SELECTION
from btc_survivor.store import StoreError
from btc_survivor.v2 import dashboard, evaluate, export, local_secrets, localcheck, news_status, output, redact, status
from btc_survivor.v2.config import ConfigError, config_kind, load_news_config, load_stock_config
from btc_survivor.v2.evaluate import EvaluateError
from btc_survivor.v2.local_secrets import SecretsError
from btc_survivor.v2.localcheck import LocalCheckError
from btc_survivor.v2.news import db as news_db
from btc_survivor.v2.news import runner as news_runner
from btc_survivor.v2.news.db import NewsDbError
from btc_survivor.v2.sessions import CalendarError
from btc_survivor.v2.stocks import process as stock_process
from btc_survivor.v2.stocks import replay as stock_replay
from btc_survivor.v2.stocks import store as stock_store
from btc_survivor.v2.stocks import tapecheck
from btc_survivor.v2.stocks.tapecheck import TapeCheckError

EXPECTED = (SecretsError, LocalCheckError, ConfigError, NewsDbError, StoreError, EvaluateError, TapeCheckError,
            CalendarError, OSError)


def cmd_init_db(args: argparse.Namespace) -> int:
    """Create news.db from a news config or a run database from a stock run config, or check an existing one, under
    umask 077 so every folder it makes is the owner's alone, printing only the verdict."""
    kind = config_kind(args.config)
    umask = os.umask(0o077)
    try:
        if kind == "news":
            created = news_db.init_db(load_news_config(args.config))
        else:
            created = stock_store.init_db(load_stock_config(args.config))
    finally:
        os.umask(umask)
    output.emit(output.Report("init db", (("verdict", True), ("kind", output.Word(kind)),
                                          ("database", output.Word("created" if created else "checked")))))
    return 0


def cmd_check_secrets(args: argparse.Namespace) -> int:
    """Name the keys present and missing in the secrets file, never a value."""
    ok, lines = local_secrets.check(args.secrets_file)
    print(redact.redact("\n".join(lines)))
    return 0 if ok else 1


def local_report(result: localcheck.LocalCheck) -> output.Report:
    """The local only verdict and counts, with one row per offending socket or tunnel for --detail."""
    rows = [{"pid": s.pid, "program": s.program, "problem": f"listens on {s.address}:{s.port}"} for s in result.exposed]
    rows += [{"pid": "", "program": "any user", "problem": f"listens on {s.address}:{s.port}"}
             for s in result.exposed_sockets]
    rows += [{"pid": t.pid, "program": t.kind, "problem": "tunnel or remote forward"} for t in result.tunnels]
    return output.Report("local only check", (
        ("verdict", result.passed),
        ("offending sockets", output.Count(result.offending_sockets)),
        ("tunnel processes", output.Count(len(result.tunnels))),
        ("loopback listeners", output.Count(result.loopback_listeners)),
    ), tuple(rows))


def cmd_check_local(args: argparse.Namespace) -> int:
    """Run the check of PLAN_V2 section 13.2 and print its report, exiting 1 when it fails."""
    result = localcheck.run_check()
    output.emit(local_report(result), args.detail)
    return 0 if result.passed else 1


def _command(sub: argparse._SubParsersAction, name: str, help_text: str, handler: Callable[[argparse.Namespace], int],
             positionals: tuple[str, ...] = (), detail: bool = False, **kwargs: str) -> argparse.ArgumentParser:
    """Register one subcommand with its positional arguments, an optional --detail and its handler."""
    parser = sub.add_parser(name, help=help_text, **kwargs)
    for positional in positionals:
        parser.add_argument(positional.lower(), metavar=positional, type=Path)
    if detail:
        output.add_detail_flag(parser)
    parser.set_defaults(handler=handler)
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Every subcommand of PLAN_V2 section 3, with the arguments the plan gives it."""
    parser = argparse.ArgumentParser(
        prog="btc_survivor_v2",
        description="Paper trading big tech stocks with scored news. No real orders are ever placed, and by default "
                    "every command prints only counts, verdicts, hashes and timings.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    _command(sub, "init-db", "create a run or news database with its meta.kind", handler=cmd_init_db,
             positionals=("CONFIG",))
    stock_process.add_arguments(_command(sub, "stocks", "run the stock process for both runs",
                                         handler=stock_process.run, description=stock_process.DESCRIPTION))
    news_runner.add_arguments(_command(sub, "news", "run the news process", handler=news_runner.run,
                                       description=news_runner.DESCRIPTION))
    news_status.add_arguments(_command(sub, "news-status", "print the week one and data quality report",
                                       handler=news_status.run, description=news_status.DESCRIPTION))
    tapecheck.add_arguments(_command(sub, "tape-check", "print the tape verdict from a raw recording alone",
                                     handler=tapecheck.run, description=tapecheck.DESCRIPTION))
    evaluate.add_arguments(_command(sub, "evaluate", "print the criteria of PLAN_V2 section 11",
                                    handler=evaluate.run, description=evaluate.DESCRIPTION))
    status.add_arguments(_command(sub, "status", "print the state of both runs", handler=status.run,
                                  description=status.DESCRIPTION))
    export.add_arguments(_command(sub, "export", "write CSV files under data/v2/exports", handler=export.run,
                                  description=export.DESCRIPTION))
    stock_replay.add_arguments(_command(sub, "replay", "run recordings into a new folder. " + NEVER_FOR_SELECTION,
                                        handler=stock_replay.run, description=stock_replay.DESCRIPTION))
    dashboard.add_arguments(_command(sub, "dashboard", "serve the private page on 127.0.0.1:8060",
                                     handler=dashboard.run, description=dashboard.DESCRIPTION))
    secrets = _command(sub, "check-secrets", "name the keys present in the secrets file, never a value",
                       handler=cmd_check_secrets)
    secrets.add_argument("--secrets-file", type=Path, default=local_secrets.DEFAULT_PATH,
                         help=f"the secrets file (default {local_secrets.DEFAULT_PATH})")
    _command(sub, "check-local", "check that ports 8050 and 8060 listen only on 127.0.0.1 and no tunnel runs",
             handler=cmd_check_local, detail=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand, printing an expected failure as one error line with exit code 2."""
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except EXPECTED as exc:
        print(f"error: {redact.redact(str(exc))}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
