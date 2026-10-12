"""Small fast configurations and drivers for the engine, replay and runner tests."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import sqlite3

from btc_survivor.config import load_config
from btc_survivor.engine import Engine
from btc_survivor.feed.base import load_adapter
from btc_survivor.store import Store
from btc_survivor.synthetic import Fault, SyntheticFeed
from helpers import EXAMPLE_CONFIG

START = 1_791_000_000_000  # a UTC minute boundary
R = dataclasses.replace
TABLES = ("meta", "bots", "lineage", "decisions", "trades", "fills", "equity_snapshots", "deaths", "feed_events",
          "checkpoints", "candles", "benchmark", "forecasts", "runtime_status")


def fast_config(data_dir, **sections):
    """Features after 16 minutes, a 2 minute horizon, 3 minutes of warmup and forced trades every 3 minutes."""
    cfg = load_config(EXAMPLE_CONFIG)
    base = {
        "run": R(cfg.run, data_dir=str(data_dir)),
        "time": R(cfg.time, horizon_minutes=2, history_minutes_required=16, warmup_hours_random=0.05,
                  warmup_hours_inherited=0.02),
        "features": R(cfg.features, return_windows_minutes=(1, 2, 3, 4), vol_windows_minutes=(2, 3),
                      volume_baseline_minutes=15, min_samples=5, imbalance_levels=5),
        "model": R(cfg.model, skill_window_samples=30, replay_window_minutes=60),
        "trading": R(cfg.trading, force_trade_after_hours=0.05),
        "population": R(cfg.population, size=4, parent_min_age_hours=0.05, parent_min_trades=1),
        "persistence": R(cfg.persistence, checkpoint_every_seconds=120, equity_snapshot_every_seconds=60,
                         keep_candles_days=1, max_resume_gap_minutes=5),
    }
    for name, changes in sections.items():
        base[name] = R(base.get(name, getattr(cfg, name)), **changes)
    return R(cfg, **base)


def deadly_config(data_dir):
    """Every round trip costs about 18 percent of start equity, so deaths and births come within the hour."""
    return fast_config(data_dir, trading={"start_equity": 100.0, "notional": 100.0},
                       costs={"fee_profile": "test", "taker_fees_bps": {"test": 900.0}})


def selecting_config(data_dir):
    """Every round trip costs about 8 percent of start equity, so deaths, births and parents with a skill
    correlation all come within three hours."""
    return fast_config(data_dir, trading={"start_equity": 100.0, "notional": 100.0},
                       costs={"fee_profile": "test", "taker_fees_bps": {"test": 400.0}})


@functools.lru_cache(maxsize=8)
def synthetic(seed: int, seconds: int, faults: tuple[Fault, ...] = (), start: int = START) -> tuple:
    return tuple(SyntheticFeed(seed, start).lines(seconds, faults))


def new_engine(cfg, start_ms: int = START) -> Engine:
    return Engine.fresh(cfg, Store.open(cfg.db_path), load_adapter(cfg.feed), start_ms)


def feed(engine: Engine, lines, until_ms: int | None = None) -> int:
    """Feed lines received before until_ms and return how many were used."""
    used = 0
    for line in lines:
        if until_ms is not None and line[2] >= until_ms:
            break
        engine.on_line(*line)
        used += 1
    return used


def digest(db_path) -> dict[str, str]:
    """A hash of every row of every table, in insertion order."""
    conn = sqlite3.connect(db_path)
    out = {}
    for table in TABLES:
        rows = conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
        out[table] = hashlib.sha256(repr(rows).encode()).hexdigest()
    conn.close()
    return out
