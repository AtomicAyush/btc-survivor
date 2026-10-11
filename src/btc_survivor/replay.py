"""Drive the engine from recordings at full speed, through the same parsing code as live.

Replay results are never used to choose, tune, or rank strategies. Replay exists to test the plumbing. It always
starts a fresh population, never loads a checkpoint, never writes runtime_status, and only writes into a new or
empty folder that is not the configured data folder.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator
from pathlib import Path

from btc_survivor.config import Config
from btc_survivor.engine import Engine
from btc_survivor.feed.base import load_adapter
from btc_survivor.recorder import read_recording
from btc_survivor.store import Store

NEVER_FOR_SELECTION = "Replay results are never used to choose, tune, or rank strategies."


class ReplayError(RuntimeError):
    """Raised when the output folder or the recordings cannot be used."""


def replay_config(cfg: Config, out_dir: Path) -> Config:
    """The same config pointed at the output folder, after checking that the folder is safe to write."""
    out = Path(out_dir).resolve()
    if out == cfg.data_dir.resolve():
        raise ReplayError(f"{out} is the configured data folder, replay never writes there")
    if out.exists() and any(out.iterdir()):
        raise ReplayError(f"{out} is not empty, replay only writes into a new or empty folder")
    out.mkdir(parents=True, exist_ok=True)
    return dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, data_dir=str(out)))


def replay_lines(cfg: Config, lines: Iterable[tuple[str, str, int, int]], out_dir: Path) -> Path:
    """Run recorded lines through a fresh engine into out_dir and return the database path."""
    run_cfg = replay_config(cfg, out_dir)
    stream: Iterator = iter(lines)
    first = next(stream, None)
    if first is None:
        raise ReplayError("the recordings hold no lines")
    store = Store.open(run_cfg.db_path)
    try:
        engine = Engine.fresh(run_cfg, store, load_adapter(run_cfg.feed), first[2])
        engine.on_line(*first)
        for line in stream:
            engine.on_line(*line)
        engine.finish()
    finally:
        store.close()
    return run_cfg.db_path


def replay(cfg: Config, files: Iterable[Path], out_dir: Path) -> Path:
    return replay_lines(cfg, read_recording(files), out_dir)
