"""The status command of PLAN_V2 section 3, the state of both stock runs read from their databases, which it opens
read only through the stock store's reader, so a database of another kind or a version 1 database is refused.

By default it prints, for the prices run and the news run, only state words, counts, timings of this system and the
experiment hash, so no price, volume or fill can reach it. --detail adds per symbol rows with the last reference
price, per bot rows with equity and position, and the runtime figures, for the owner's own terminal. The bot rows are
the ones the private dashboard shows too.

    btc_survivor_v2 status configs/v2/s1_prices.toml configs/v2/s2_news.toml [--detail]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from btc_survivor.clock import MS_PER_SECOND, LiveClock, parse_iso
from btc_survivor.store import StoreError
from btc_survivor.v2 import output, redact
from btc_survivor.v2.config import ConfigError, load_run_pair
from btc_survivor.v2.stocks import store
# The origin version 1 gives a coin flip twin, which the stock population keeps, so every other origin is a learner.
from btc_survivor.v2.stocks.population import TWIN as TWIN_ORIGIN
from btc_survivor.v2.stocks.records import GAP, LOCAL_ONLY, MATURED, PENDING

RUNS = ("prices run", "news run")
EXPECTED = (ConfigError, StoreError, OSError)

BOTS = """
SELECT b.bot_id, b.slot, b.origin, b.home_symbol, b.state,
  (SELECT e.equity FROM equity_snapshots e WHERE e.bot_id = b.bot_id ORDER BY e.ts DESC LIMIT 1) AS equity,
  (SELECT count(*) FROM trades t WHERE t.bot_id = b.bot_id AND t.closed_at IS NOT NULL) AS trades,
  o.side, o.qty, o.entry_price
FROM bots b LEFT JOIN trades o ON o.bot_id = b.bot_id AND o.closed_at IS NULL
ORDER BY b.origin = ?, b.slot, b.bot_id
"""


def word(text: str | None) -> output.Word:
    """A state word as stored, or unknown when it is missing or not a plain lower case word."""
    try:
        return output.Word(text or "unknown")
    except ValueError:
        return output.Word("unknown")


def age(now_ms: int, ts: str | None) -> output.Seconds | output.Word:
    """How long ago a stored time was, or never when there is none."""
    return output.Word("never") if ts is None else output.Seconds((now_ms - parse_iso(ts)) / MS_PER_SECOND)


def scalar(conn: sqlite3.Connection, sql: str) -> Any:
    return conn.execute(sql).fetchone()[0]


def bot_rows(conn: sqlite3.Connection) -> list[dict]:
    """Every learner and twin, learners first by slot, with its latest equity snapshot, closed trades and open
    position, the rows of status --detail and of the dashboard's population panel."""
    return [{"bot_id": row["bot_id"], "slot": row["slot"],
             "kind": "twin" if row["origin"] == TWIN_ORIGIN else "learner", "home_symbol": row["home_symbol"],
             "state": row["state"], "equity": row["equity"], "trades": row["trades"], "position": row["side"] or "flat",
             "qty": row["qty"], "entry_price": row["entry_price"]}
            for row in conn.execute(BOTS, (TWIN_ORIGIN,))]


def local_only_checks(conn: sqlite3.Connection) -> tuple[int, int, bool | None]:
    """How many local_only checks passed and failed, and the verdict of the newest, None before the first. A detail
    that cannot be read counts as failed."""
    verdicts = []
    for (detail,) in conn.execute("SELECT detail FROM feed_events WHERE kind = ? ORDER BY id", (LOCAL_ONLY,)):
        try:
            verdicts.append(json.loads(detail or "{}").get("passed") is True)
        except (ValueError, AttributeError):
            verdicts.append(False)
    return sum(verdicts), len(verdicts) - sum(verdicts), verdicts[-1] if verdicts else None


def experiment(meta: dict[str, str]) -> output.Hash | output.Word:
    """The experiment hash recorded at launch, or a word saying why there is none."""
    if "experiment_hash" not in meta:
        return output.Word("not launched")
    try:
        return output.Hash(meta["experiment_hash"])
    except ValueError:
        return output.Word("invalid")


def summary(conn: sqlite3.Connection, prefix: str, now_ms: int) -> tuple[tuple[str, Any], ...]:
    """One run's counts, state words and timings, each label starting with the run's name."""
    meta = store.read_meta(conn)
    status = conn.execute("SELECT ts, feed_state, session_state FROM runtime_status WHERE id = 1").fetchone()
    symbols = dict(conn.execute("SELECT state, count(*) FROM symbol_status GROUP BY state").fetchall())
    labels = dict(conn.execute("SELECT status, count(*) FROM stock_labels GROUP BY status").fetchall())
    passed, failed, last = local_only_checks(conn)
    living = [b for b in bot_rows(conn) if b["state"] != "dead"]
    values = (
        ("launch", output.Word("launched" if "launch_ts" in meta else "not launched")),
        ("experiment hash", experiment(meta)),
        ("feed state", word(status["feed_state"] if status else None)),
        ("session state", word(status["session_state"] if status else None)),
        ("heartbeat age", age(now_ms, status["ts"] if status else None)),
        ("checkpoint age", age(now_ms, scalar(conn, "SELECT max(ts) FROM checkpoints"))),
        ("living learners", output.Count(sum(b["kind"] == "learner" for b in living))),
        ("living twins", output.Count(sum(b["kind"] == "twin" for b in living))),
        ("deaths", output.Count(scalar(conn, "SELECT count(*) FROM deaths"))),
        ("open positions", output.Count(scalar(conn, "SELECT count(*) FROM trades WHERE closed_at IS NULL"))),
        ("closed trades", output.Count(scalar(conn, "SELECT count(*) FROM trades WHERE closed_at IS NOT NULL"))),
        ("decisions", output.Count(scalar(conn, "SELECT count(*) FROM decisions"))),
        ("forced decisions", output.Count(scalar(conn, "SELECT count(*) FROM decisions WHERE forced = 1"))),
        ("gate passed decisions", output.Count(scalar(conn, "SELECT count(*) FROM decisions WHERE gate_passed = 1"))),
        ("fills", output.Count(scalar(conn, "SELECT count(*) FROM fills WHERE status = 'filled'"))),
        ("rejected fills", output.Count(scalar(conn, "SELECT count(*) FROM fills WHERE status != 'filled'"))),
        ("healthy symbols", output.Count(symbols.get("healthy", 0))),
        ("other symbols", output.Count(sum(symbols.values()) - symbols.get("healthy", 0))),
        ("sessions", output.Count(scalar(conn, "SELECT count(*) FROM sessions"))),
        ("run minutes", output.Count(scalar(conn, "SELECT coalesce(sum(run_minutes), 0) FROM sessions"))),
        ("matured labels", output.Count(labels.get(MATURED, 0))),
        ("pending labels", output.Count(labels.get(PENDING, 0))),
        ("gap labels", output.Count(labels.get(GAP, 0))),
        ("forecast rows", output.Count(scalar(conn, "SELECT count(*) FROM stock_forecasts"))),
        ("unhealthy news minutes", output.Count(scalar(conn, "SELECT count(*) FROM news_minutes WHERE healthy = 0"))),
        ("local only checks passed", output.Count(passed)),
        ("local only checks failed", output.Count(failed)),
        ("last local only check", output.Word("none") if last is None else last),
    )
    return tuple((f"{prefix} {label}", value) for label, value in values)


def sections(conn: sqlite3.Connection, prefix: str) -> list[output.Report]:
    """The rows --detail adds for one run, per symbol, per bot and the runtime figures."""
    symbols = tuple(dict(row) for row in conn.execute(
        "SELECT symbol, state, ref_price, last_print_age_s, ts FROM symbol_status ORDER BY symbol"))
    runtime = tuple(dict(row) for row in conn.execute(
        "SELECT ts, prints_per_s, msgs_per_s, news_age_s, living, last_checkpoint_ts, rss_mb, rss_peak_mb "
        "FROM runtime_status"))
    return [output.Report(f"{prefix} symbols", (), symbols), output.Report(f"{prefix} bots", (), tuple(bot_rows(conn))),
            output.Report(f"{prefix} runtime", (), runtime)]


def build(conns: Sequence[sqlite3.Connection], now_ms: int) -> tuple[output.Report, list[output.Report]]:
    """The summary of both runs and the detail sections, the prices run first."""
    lines: list[tuple[str, Any]] = []
    detail: list[output.Report] = []
    for prefix, conn in zip(RUNS, conns):
        lines += summary(conn, prefix, now_ms)
        detail += sections(conn, prefix)
    return output.Report("status", tuple(lines)), detail


DESCRIPTION = ("Print the state of the prices run and the news run from their databases, opened read only. By default "
               "it prints only state words, counts, timings and the experiment hash.")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The status arguments, for this module's own parser or the btc_survivor_v2 subcommand."""
    parser.add_argument("s1", metavar="S1", type=Path, help="the prices run config, such as configs/v2/s1_prices.toml")
    parser.add_argument("s2", metavar="S2", type=Path, help="the news run config, such as configs/v2/s2_news.toml")
    output.add_detail_flag(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="btc_survivor_v2 status", description=DESCRIPTION)
    add_arguments(parser)
    return parser


def report(args: argparse.Namespace, now_ms: int) -> tuple[output.Report, list[output.Report]]:
    """Open both run databases read only and build the status, raising an expected error such as a missing one."""
    s1, s2 = load_run_pair(args.s1, args.s2)
    conns: list[sqlite3.Connection] = []
    try:
        for config in (s1, s2):
            conns.append(store.open_reader(config.db_path))
        return build(conns, now_ms)
    finally:
        for conn in conns:
            conn.close()


def run(args: argparse.Namespace) -> int:
    """Print the status of both runs and exit 0, or print an expected failure as one error line and exit 2."""
    try:
        summary_report, detail = report(args, LiveClock().now_ms())
    except EXPECTED as exc:
        print(f"error: {redact.redact(str(exc))}", file=sys.stderr)
        return 2
    if args.detail:
        output.emit(summary_report)
        for section in detail:
            output.emit(section, detail=True)
    else:
        output.emit(output.Report(summary_report.title, summary_report.summary,
                                  tuple(row for section in detail for row in section.rows)))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run status from the command line."""
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
