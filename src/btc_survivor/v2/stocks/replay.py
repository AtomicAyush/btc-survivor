"""Replay of raw stock recordings by PLAN_V2 sections 3 and 14.2, through the same group, decode and engine code as
the live stock process.

Replay results are never used to choose, tune, or rank strategies. Replay exists to test the plumbing. It runs one run
config at full speed over the recorded feed lines and the news lines the live puller recorded beside them, so it never
opens news.db and never sends a frame. It always starts a fresh population at the first recorded stamp, never loads a
checkpoint, never writes runtime_status, and writes only into a new or empty folder that is not the configured data
folder. The hashes of the first news heartbeat line stand in for the news hashes a run records at launch. By default it
prints only counts, a digest of every table and its run time, and --detail adds each table's rows and digest.

    btc_survivor_v2 replay configs/v2/s2_news.toml data/v2/raw_stocks --out data/v2/replays/check [--detail]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

from btc_survivor.clock import iso
from btc_survivor.recorder import read_recording
from btc_survivor import replay as v1
from btc_survivor.replay import NEVER_FOR_SELECTION, ReplayError
from btc_survivor.store import StoreError
from btc_survivor.v2 import output, redact
from btc_survivor.v2.config import ConfigError, StockConfig, load_stock_config
from btc_survivor.v2.stocks import store as stock_store
from btc_survivor.v2.stocks.engine import StockEngine
from btc_survivor.v2.stocks.finnhub import FinnhubAdapter
from btc_survivor.v2.stocks.group import StockGroup
from btc_survivor.v2.stocks.market import MarketLayer
from btc_survivor.v2.stocks.news_inputs import HEARTBEAT_LINE, HeartbeatLine

Line = tuple[str, str, int, int]
# The adapter needs a token shape, but replay never connects, so no secret is ever read.
REPLAY_TOKEN = "replay"
NO_HASHES = ("", "", "")


class Replayed(NamedTuple):
    db_path: Path
    lines: int


def recorded_hashes(lines: Iterable[Line]) -> tuple[str, str, str]:
    """The news hashes of the first heartbeat line, which stand in for those a run records at launch, or empty hashes
    for a recording without news lines, where news health never holds."""
    for kind, raw, recv_ms, _ in lines:
        if kind == HEARTBEAT_LINE:
            return HeartbeatLine.decode(raw, recv_ms).hashes
    return NO_HASHES


def replay_config(cfg: StockConfig, out_dir: Path) -> StockConfig:
    """The config writing into out_dir, after the version 1 checks that the folder is new or empty and not the
    configured data folder. Its data folder is out_dir itself, named relative to it, so every replay of one recording
    writes the same rows and checkpoints whatever folder it writes into."""
    out = v1.replay_config(cfg, out_dir).data_dir
    return dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, data_dir="."), base_dir=out)


def replay_meta(cfg: StockConfig, start_ms: int, hashes: tuple[str, str, str]) -> dict[str, str]:
    """The meta a replay writes, its config, its start and the hashes it judged news health against."""
    return {"run_name": cfg.run.name, "master_seed": str(cfg.run.master_seed), "launch_ts": iso(start_ms),
            "config_toml": cfg.text, "experiment_hash": cfg.experiment_hash, "news_config_hash": hashes[0],
            "prompt_sha256": hashes[1], "schema_sha256": hashes[2]}


def replay_lines(cfg: StockConfig, lines: Iterable[Line], out_dir: Path,
                 hashes: tuple[str, str, str] = NO_HASHES) -> Replayed:
    """Run recorded lines through a fresh group with one engine into out_dir, ending with a checkpoint."""
    run_cfg = replay_config(cfg, out_dir)
    stream: Iterator[Line] = iter(lines)
    first = next(stream, None)
    if first is None:
        raise ReplayError("the recordings hold no lines")
    start = first[2]
    store = stock_store.StockStore.open(run_cfg.db_path)
    try:
        for key, value in replay_meta(run_cfg, start, hashes).items():
            store.queue_meta(key, value)
        engine = StockEngine.fresh(run_cfg, store, start)
        group = StockGroup(MarketLayer(run_cfg, hashes), [engine], FinnhubAdapter(run_cfg.stock_feed.symbols,
                                                                                  REPLAY_TOKEN), start)
        group.on_line(*first)
        count = 1
        for line in stream:
            group.on_line(*line)
            count += 1
        group.finish()
    finally:
        store.close()
    return Replayed(run_cfg.db_path, count)


def replay(cfg: StockConfig, files: Iterable[Path], out_dir: Path) -> Replayed:
    """Replay recording files or folders of them in name order, which is time order."""
    files = list(files)
    return replay_lines(cfg, read_recording(files), out_dir, recorded_hashes(read_recording(files)))


def table_digests(path: Path) -> dict[str, tuple[int, str]]:
    """Each table's row count and the SHA 256 of its rows sorted by every column, so two replays compare exactly."""
    conn = stock_store.open_reader(path)
    try:
        out = {}
        for table in stock_store.TABLES:
            width = len(conn.execute(f"PRAGMA table_info({table})").fetchall())
            order = ", ".join(str(i) for i in range(1, width + 1))
            rows = [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY {order}")]
            out[table] = (len(rows), hashlib.sha256(repr(rows).encode("utf-8")).hexdigest())
        return out
    finally:
        conn.close()


def report(replayed: Replayed, seconds: float) -> output.Report:
    """The verdict, counts and digests of a replay, with one row per table for --detail."""
    digests = table_digests(replayed.db_path)
    whole = hashlib.sha256("".join(f"{t}:{d}\n" for t, (_, d) in digests.items()).encode("utf-8")).hexdigest()
    return output.Report("stock replay", (
        ("verdict", True),
        ("lines", output.Count(replayed.lines)),
        ("tables", output.Count(len(digests))),
        ("rows", output.Count(sum(n for n, _ in digests.values()))),
        ("database digest", output.Hash(whole)),
        ("run time", output.Seconds(seconds)),
    ), tuple({"table": t, "rows": n, "digest": d} for t, (n, d) in digests.items()))


# The version 1 sentence comes first, so help at any usual width keeps it on one line.
DESCRIPTION = f"{NEVER_FOR_SELECTION} Replay runs stock recordings into a new folder."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The replay arguments, for this module's own parser or the btc_survivor_v2 subcommand."""
    parser.add_argument("config", metavar="CONFIG", type=Path, help="a stock run config, such as "
                                                                     "configs/v2/s2_news.toml")
    parser.add_argument("files", metavar="FILES", nargs="+", type=Path, help="recording files or folders of them")
    parser.add_argument("--out", required=True, type=Path, help="a new or empty folder for the replay database")
    output.add_detail_flag(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="btc_survivor_v2 replay", description=DESCRIPTION)
    add_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Replay from the command line."""
    return run(build_parser().parse_args(argv))


def run(args: argparse.Namespace) -> int:
    """Replay the recordings under umask 077, so the replay database and its folder are the owner's alone, and print
    the report, or one error line with exit code 2."""
    started = time.monotonic()
    umask = os.umask(0o077)
    try:
        replayed = replay(load_stock_config(args.config), args.files, args.out)
    except (ReplayError, ConfigError, StoreError, OSError) as exc:
        print(f"error: {redact.redact(str(exc))}", file=sys.stderr)
        return 2
    finally:
        os.umask(umask)
    output.emit(report(replayed, time.monotonic() - started), getattr(args, "detail", False))
    print(NEVER_FOR_SELECTION)
    return 0


if __name__ == "__main__":
    sys.exit(main())
