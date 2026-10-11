"""Load the TOML configuration and validate it with clear error messages."""

from __future__ import annotations

import importlib.util
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when the configuration file is missing, malformed or out of range."""


@dataclass(frozen=True)
class RunConfig:
    master_seed: int
    data_dir: str
    log_level: str


@dataclass(frozen=True)
class FeedConfig:
    exchange: str
    product: str
    book_depth_levels: int
    stale_after_seconds: float
    reconnect_backoff_start_seconds: float
    reconnect_backoff_max_seconds: float
    record_raw_messages: bool
    max_clock_skew_seconds: float


@dataclass(frozen=True)
class TimeConfig:
    horizon_minutes: int
    history_minutes_required: int
    warmup_hours_random: float
    warmup_hours_inherited: float


@dataclass(frozen=True)
class FeaturesConfig:
    return_windows_minutes: tuple[int, ...]
    vol_windows_minutes: tuple[int, ...]
    volume_baseline_minutes: int
    imbalance_levels: int
    standardizer_half_life_hours: float
    clip_z: float
    min_samples: int
    min_window_coverage: float


@dataclass(frozen=True)
class ModelConfig:
    hidden_units: int
    init_output_scale: float
    batch_size: int
    replay_window_minutes: int
    huber_delta: float
    grad_clip: float
    adam_beta1: float
    adam_beta2: float
    gate_on_skill: bool
    skill_window_samples: int
    skill_gate_min_corr: float


@dataclass(frozen=True)
class TradingConfig:
    start_equity: float
    notional: float
    stop_loss_pct: float
    allow_short: bool
    force_trade_after_hours: float


@dataclass(frozen=True)
class CostsConfig:
    fee_profile: str
    taker_fees_bps: dict[str, float]
    latency_ms: int
    latency_jitter_ms: int
    extra_slippage_bps: float
    short_funding_bps_per_hour: float
    short_open_fee_bps: float
    daily_running_cost: float

    @property
    def taker_fee_bps(self) -> float:
        """The taker fee of the active profile."""
        return self.taker_fees_bps[self.fee_profile]


@dataclass(frozen=True)
class GenesConfig:
    margin_bps_init: float
    margin_bps_min: float
    margin_bps_max: float
    lr_init: float
    lr_min: float
    lr_max: float
    weight_decay: float


@dataclass(frozen=True)
class PopulationConfig:
    size: int
    newcomer_fraction: float
    parent_min_age_hours: float
    parent_min_trades: int
    mutation_sigma_weights: float
    mutation_sigma_genes: float
    fitness_window_hours: int
    fitness_std_floor_usd: float
    tournament_size: int


@dataclass(frozen=True)
class PersistenceConfig:
    checkpoint_every_seconds: int
    equity_snapshot_every_seconds: int
    max_resume_gap_minutes: float
    keep_candles_days: int
    keep_checkpoints: int


@dataclass(frozen=True)
class Config:
    run: RunConfig
    feed: FeedConfig
    time: TimeConfig
    features: FeaturesConfig
    model: ModelConfig
    trading: TradingConfig
    costs: CostsConfig
    genes: GenesConfig
    population: PopulationConfig
    persistence: PersistenceConfig
    base_dir: Path
    text: str

    @property
    def data_dir(self) -> Path:
        """The data folder, resolved against the folder that holds the config file."""
        path = Path(self.run.data_dir)
        return path if path.is_absolute() else self.base_dir / path

    @property
    def db_path(self) -> Path:
        return self.data_dir / "btc_survivor.db"


INF = float("inf")

# Each key maps to (kind, low, high, low_inclusive, high_inclusive). Kinds are
# int, float, bool, str and ints (a list of ints).
_SPEC: dict[str, dict[str, tuple]] = {
    "run": {
        "master_seed": ("int", 0, 2**63 - 1, True, True),
        "data_dir": ("str", None, None, True, True),
        "log_level": ("str", None, None, True, True),
    },
    "feed": {
        "exchange": ("str", None, None, True, True),
        "product": ("str", None, None, True, True),
        "book_depth_levels": ("int", 1, 1000, True, True),
        "stale_after_seconds": ("float", 0, 300, False, True),
        "reconnect_backoff_start_seconds": ("float", 0, 600, False, True),
        "reconnect_backoff_max_seconds": ("float", 0, 3600, False, True),
        "record_raw_messages": ("bool", None, None, True, True),
        "max_clock_skew_seconds": ("float", 0, 600, False, True),
    },
    "time": {
        "horizon_minutes": ("int", 1, 240, True, True),
        "history_minutes_required": ("int", 1, 10080, True, True),
        "warmup_hours_random": ("float", 0, 168, True, True),
        "warmup_hours_inherited": ("float", 0, 168, True, True),
    },
    "features": {
        "return_windows_minutes": ("ints", 1, 10080, True, True),
        "vol_windows_minutes": ("ints", 2, 10080, True, True),
        "volume_baseline_minutes": ("int", 15, 10080, True, True),
        "imbalance_levels": ("int", 1, 1000, True, True),
        "standardizer_half_life_hours": ("float", 0, 10000, False, True),
        "clip_z": ("float", 0, 100, False, True),
        "min_samples": ("int", 1, 100000, True, True),
        "min_window_coverage": ("float", 0, 1, False, True),
    },
    "model": {
        "hidden_units": ("int", 1, 1024, True, True),
        "init_output_scale": ("float", 0, 10, False, True),
        "batch_size": ("int", 1, 1024, True, True),
        "replay_window_minutes": ("int", 1, 10080, True, True),
        "huber_delta": ("float", 0, 1000, False, True),
        "grad_clip": ("float", 0, 1e6, False, True),
        "adam_beta1": ("float", 0, 1, True, False),
        "adam_beta2": ("float", 0, 1, True, False),
        "gate_on_skill": ("bool", None, None, True, True),
        "skill_window_samples": ("int", 2, 100000, True, True),
        "skill_gate_min_corr": ("float", -1, 1, True, True),
    },
    "trading": {
        "start_equity": ("float", 0, 1e12, False, True),
        "notional": ("float", 0, 1e12, False, True),
        "stop_loss_pct": ("float", 0, 100, False, False),
        "allow_short": ("bool", None, None, True, True),
        "force_trade_after_hours": ("float", 0, 10000, False, True),
    },
    "costs": {
        "fee_profile": ("str", None, None, True, True),
        "latency_ms": ("int", 0, 30000, True, True),
        "latency_jitter_ms": ("int", 0, 30000, True, True),
        "extra_slippage_bps": ("float", 0, 1000, True, True),
        "short_funding_bps_per_hour": ("float", 0, 1000, True, True),
        "short_open_fee_bps": ("float", 0, 1000, True, True),
        "daily_running_cost": ("float", 0, 1e9, True, True),
    },
    "genes": {
        "margin_bps_init": ("float", 0, 10000, True, True),
        "margin_bps_min": ("float", 0, 10000, True, True),
        "margin_bps_max": ("float", 0, 10000, True, True),
        "lr_init": ("float", 0, 1, False, True),
        "lr_min": ("float", 0, 1, False, True),
        "lr_max": ("float", 0, 1, False, True),
        "weight_decay": ("float", 0, 1, True, False),
    },
    "population": {
        "size": ("int", 1, 10000, True, True),
        "newcomer_fraction": ("float", 0, 1, True, True),
        "parent_min_age_hours": ("float", 0, 1e6, True, True),
        "parent_min_trades": ("int", 0, 10**9, True, True),
        "mutation_sigma_weights": ("float", 0, 100, True, True),
        "mutation_sigma_genes": ("float", 0, 100, True, True),
        "fitness_window_hours": ("int", 1, 10**6, True, True),
        "fitness_std_floor_usd": ("float", 0, 1e9, False, True),
        "tournament_size": ("int", 1, 10000, True, True),
    },
    "persistence": {
        "checkpoint_every_seconds": ("int", 60, 86400, True, True),
        "equity_snapshot_every_seconds": ("int", 60, 86400, True, True),
        "max_resume_gap_minutes": ("float", 0, 1e6, True, True),
        "keep_candles_days": ("int", 1, 3650, True, True),
        "keep_checkpoints": ("int", 1, 1000, True, True),
    },
}

_FEE_KEY = re.compile(r"^taker_fee_bps_([a-z][a-z0-9_]*)$")
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def _coerce(section: str, key: str, value: Any, spec: tuple, errors: list[str]) -> Any:
    """Check one value against its spec and return it in canonical form, or None after recording an error."""
    kind, low, high, low_inc, high_inc = spec
    where = f"[{section}] {key}"
    if kind == "bool":
        if not isinstance(value, bool):
            errors.append(f"{where}: expected true or false, got {value!r}")
            return None
        return value
    if kind == "str":
        if not isinstance(value, str) or not value:
            errors.append(f"{where}: expected a non empty string, got {value!r}")
            return None
        return value
    if kind == "ints":
        if not isinstance(value, list) or not value or not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
            errors.append(f"{where}: expected a non empty list of integers, got {value!r}")
            return None
        for v in value:
            if not _in_range(v, low, high, low_inc, high_inc):
                errors.append(f"{where}: element {v} is out of range, allowed {_range_text(low, high, low_inc, high_inc)}")
                return None
        return tuple(value)
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"{where}: expected an integer, got {value!r}")
            return None
    elif kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{where}: expected a number, got {value!r}")
            return None
        value = float(value)
        if value != value:
            errors.append(f"{where}: expected a number, got nan")
            return None
    if not _in_range(value, low, high, low_inc, high_inc):
        errors.append(f"{where} = {value!r} is out of range, allowed {_range_text(low, high, low_inc, high_inc)}")
        return None
    return value


def _in_range(v: float, low: float, high: float, low_inc: bool, high_inc: bool) -> bool:
    above = v >= low if low_inc else v > low
    below = v <= high if high_inc else v < high
    return above and below


def _range_text(low: float, high: float, low_inc: bool, high_inc: bool) -> str:
    return f"{'[' if low_inc else '('}{low}, {high}{']' if high_inc else ')'}"


def parse_config(text: str, base_dir: Path) -> Config:
    """Parse and validate configuration text. All problems are reported together in one ConfigError."""
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"config is not valid TOML: {exc}") from exc

    errors: list[str] = []
    values: dict[str, dict[str, Any]] = {}
    for section in raw:
        if section not in _SPEC:
            errors.append(f"unknown section [{section}]")
    for section, spec in _SPEC.items():
        table = raw.get(section)
        if not isinstance(table, dict):
            errors.append(f"missing section [{section}]")
            continue
        out: dict[str, Any] = {}
        fees: dict[str, float] = {}
        for key, value in table.items():
            fee = _FEE_KEY.match(key) if section == "costs" else None
            if fee:
                v = _coerce(section, key, value, ("float", 0, 1000, True, True), errors)
                if v is not None:
                    fees[fee.group(1)] = v
            elif key not in spec:
                errors.append(f"[{section}] {key}: unknown key")
        for key, key_spec in spec.items():
            if key not in table:
                errors.append(f"[{section}] {key}: missing")
                continue
            out[key] = _coerce(section, key, table[key], key_spec, errors)
        if section == "costs":
            out["taker_fees_bps"] = fees
        values[section] = out

    if errors:
        raise ConfigError(_format(errors))
    _cross_checks(values, errors)
    if errors:
        raise ConfigError(_format(errors))

    return Config(
        run=RunConfig(**values["run"]),
        feed=FeedConfig(**values["feed"]),
        time=TimeConfig(**values["time"]),
        features=FeaturesConfig(**values["features"]),
        model=ModelConfig(**values["model"]),
        trading=TradingConfig(**values["trading"]),
        costs=CostsConfig(**values["costs"]),
        genes=GenesConfig(**values["genes"]),
        population=PopulationConfig(**values["population"]),
        persistence=PersistenceConfig(**values["persistence"]),
        base_dir=base_dir,
        text=text,
    )


def _cross_checks(v: dict[str, dict[str, Any]], errors: list[str]) -> None:
    """Checks that relate several keys to each other."""
    run, feed, time_, feat = v["run"], v["feed"], v["time"], v["features"]
    trading, costs, genes, persist = v["trading"], v["costs"], v["genes"], v["persistence"]

    if run["log_level"] not in _LOG_LEVELS:
        errors.append(f"[run] log_level = {run['log_level']!r} must be one of {', '.join(_LOG_LEVELS)}")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", feed["exchange"]):
        errors.append(f"[feed] exchange = {feed['exchange']!r} must be a lower case module name")
    elif feed["exchange"] in ("base", "book", "candles", "health", "market") or \
            importlib.util.find_spec(f"btc_survivor.feed.{feed['exchange']}") is None:
        errors.append(f"[feed] exchange = {feed['exchange']!r} has no adapter module in btc_survivor.feed")
    if feed["reconnect_backoff_max_seconds"] < feed["reconnect_backoff_start_seconds"]:
        errors.append("[feed] reconnect_backoff_max_seconds must be at least reconnect_backoff_start_seconds")

    if len(feat["return_windows_minutes"]) != 4:
        errors.append("[features] return_windows_minutes must list exactly 4 windows, the model has 11 inputs")
    if len(feat["vol_windows_minutes"]) != 2:
        errors.append("[features] vol_windows_minutes must list exactly 2 windows, the model has 11 inputs")
    if feat["imbalance_levels"] > feed["book_depth_levels"]:
        errors.append("[features] imbalance_levels must not exceed [feed] book_depth_levels")
    needed = max(max(feat["return_windows_minutes"]) + 1, max(feat["vol_windows_minutes"]) + 1, feat["volume_baseline_minutes"])
    if time_["history_minutes_required"] < needed:
        errors.append(f"[time] history_minutes_required must be at least {needed} to cover the longest feature window")

    if trading["notional"] > trading["start_equity"]:
        errors.append("[trading] notional must not exceed start_equity")

    if costs["fee_profile"] not in costs["taker_fees_bps"]:
        known = ", ".join(sorted(costs["taker_fees_bps"])) or "none"
        errors.append(f"[costs] fee_profile = {costs['fee_profile']!r} has no taker_fee_bps_{costs['fee_profile']} key, known profiles are {known}")
    if costs["latency_ms"] + costs["latency_jitter_ms"] > 30000:
        errors.append("[costs] latency_ms plus latency_jitter_ms must stay at or under 30000, well inside one minute")

    for gene in ("margin_bps", "lr"):
        lo, init, hi = genes[f"{gene}_min"], genes[f"{gene}_init"], genes[f"{gene}_max"]
        if not lo <= hi:
            errors.append(f"[genes] {gene}_min must not exceed {gene}_max")
        elif not lo <= init <= hi:
            errors.append(f"[genes] {gene}_init must lie between {gene}_min and {gene}_max")

    for key in ("checkpoint_every_seconds", "equity_snapshot_every_seconds"):
        if persist[key] % 60:
            errors.append(f"[persistence] {key} must be a whole number of minutes")
    if persist["keep_candles_days"] * 1440 < time_["history_minutes_required"]:
        errors.append("[persistence] keep_candles_days must keep at least history_minutes_required minutes of candles")


def _format(errors: list[str]) -> str:
    return "invalid configuration\n  " + "\n  ".join(errors)


def load_config(path: str | Path) -> Config:
    """Read and validate a configuration file."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc.strerror}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"config file {path} is not valid UTF-8: {exc}") from exc
    return parse_config(text, path.resolve().parent)
