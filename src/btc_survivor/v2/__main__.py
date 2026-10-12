"""The btc_survivor_v2 command line of PLAN_V2 section 3. By default every subcommand prints only counts, pass or
fail verdicts, hashes and timings, and --detail adds rows for the owner's own terminal. Subcommands of later phases
are registered now and fail with exit code 2 until their phase builds them."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from btc_survivor import __version__
from btc_survivor.replay import NEVER_FOR_SELECTION
from btc_survivor.v2 import local_secrets, localcheck, output, redact
from btc_survivor.v2.local_secrets import SecretsError
from btc_survivor.v2.localcheck import LocalCheckError


def cmd_not_built(args: argparse.Namespace) -> int:
    """Refuse a subcommand whose phase has not been built yet."""
    print(f"error: {args.command} is not built yet", file=sys.stderr)
    return 2


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


def _command(sub: argparse._SubParsersAction, name: str, help_text: str,
             handler: Callable[[argparse.Namespace], int] = cmd_not_built, positionals: tuple[str, ...] = (),
             detail: bool = False, **kwargs: str) -> argparse.ArgumentParser:
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
    _command(sub, "init-db", "create a run or news database with its meta.kind", positionals=("CONFIG",))
    _command(sub, "stocks", "run the stock process for both runs", positionals=("S1", "S2"))
    _command(sub, "news", "run the news process", positionals=("CONFIG",))
    news_status = _command(sub, "news-status", "print the week one and data quality report", detail=True)
    news_status.add_argument("--samples", type=int, default=0, help="sampled headlines, only with --detail")
    _command(sub, "tape-check", "print the tape verdict from a raw recording alone",
             positionals=("RECORDING", "OFFICIAL"), detail=True)
    _command(sub, "evaluate", "print the criteria of PLAN_V2 section 11", positionals=("S1", "S2"), detail=True)
    _command(sub, "status", "print the state of both runs", positionals=("S1", "S2"), detail=True)
    _command(sub, "export", "write CSV files under data/v2/exports", positionals=("CONFIG",))
    replay = _command(sub, "replay", "run recordings into a new folder. " + NEVER_FOR_SELECTION,
                      positionals=("CONFIG",), description=NEVER_FOR_SELECTION)
    replay.add_argument("files", metavar="FILES", nargs="+", type=Path, help="recording files or folders of them")
    replay.add_argument("--out", required=True, type=Path, help="a new or empty folder for the replay database")
    _command(sub, "dashboard", "serve the private page on 127.0.0.1:8060", positionals=("S1", "S2"))
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
    except (SecretsError, LocalCheckError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
