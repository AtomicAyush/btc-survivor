"""Load and validate the version 2 stock run and news configs into frozen dataclasses, with their hashes.

A stock run config carries [run] kind = "stocks" and a news config [news] kind = "news", so neither loads as the other
or as a version 1 config. Every rule here comes from PLAN_V2 section 15.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a version 2 config is missing, malformed, out of range or inconsistent."""


HAIKU_MODEL = "claude-haiku-5-5"
SONNET_MODEL = "claude-sonnet-5-5"
NEWS_KINDS = ("media", "8k", "8k_a", "fed_press", "fed_monetary", "fed_speech", "bls_release")
OFFICIAL_KINDS = ("8k", "fed_press", "fed_monetary", "fed_speech", "bls_release")
PREDICTORS = ("population", "linear_prices", "linear_news", "news", "news_company", "news_macro", "haiku",
              "finbert", "finbert_matched", "sonnet", "haiku_official")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
MAX_SYMBOLS = 50                 # the Finnhub websocket limit per key
EARLY_CLOSE_MINUTES = 210
MIN_KEEP_CANDLES_DAYS = 120      # twice the calendar length of 40 sessions
MAX_MONTHLY_USD = 20.0           # the Console workspace limit
MIN_SERIES_CAPACITY = 420        # the least StockSeries holds, the 391 minutes of history plus the 30 minute horizon
_SYMBOL = re.compile(r"[A-Z][A-Z0-9.]{0,9}")


@dataclass(frozen=True)
class RunConfig:
    kind: str
    name: str
    master_seed: int
    data_dir: str
    log_level: str
    secrets_file: str


@dataclass(frozen=True)
class StockFeedConfig:
    provider: str
    symbols: tuple[str, ...]
    index_symbol: str
    feed_stale_after_seconds: float
    symbol_stale_after_seconds: float
    max_print_delay_seconds: float
    late_print_seconds: float
    price_excluded_conditions: tuple[str, ...]
    reconnect_backoff_start_seconds: float
    reconnect_backoff_max_seconds: float
    mark_healthy_outside_session_seconds: float
    record_raw_messages: bool
    raw_dir: str
    keep_raw_days: int
    lock_file: str


@dataclass(frozen=True)
class SessionConfig:
    first_open_session_minute: int
    last_open_minutes_before_close: int
    flatten_minutes_before_close: int
    calendar_warn_days: int
    max_overnight_gap_pct: float


@dataclass(frozen=True)
class NewsInputConfig:
    use_news: bool
    news_db: str
    poll_seconds: float
    reopen_after_silent_seconds: float
    heartbeat_max_age_seconds: float
    half_life_session_minutes: float
    drop_after_half_lives: float


@dataclass(frozen=True)
class TimeConfig:
    horizon_minutes: int
    history_session_minutes: int
    warmup_hours: float


@dataclass(frozen=True)
class FeaturesConfig:
    return_windows_minutes: tuple[int, ...]
    day_return_minutes: int
    vol_window_minutes: int
    volume_window_minutes: int
    volume_baseline_minutes: int
    volume_ratio_clip: float
    tick_window_minutes: int
    standardizer_half_life_hours: float
    clip_z: float
    min_samples: int
    min_window_coverage: float
    series_capacity_minutes: int


@dataclass(frozen=True)
class ModelConfig:
    inputs: int
    hidden_units: int
    init_output_scale: float
    batch_size: int
    replay_window_minutes: int
    huber_delta: float
    grad_clip: float
    adam_beta1: float
    adam_beta2: float
    learning_rate: float
    weight_decay: float
    skill_window_samples: int


@dataclass(frozen=True)
class TradingConfig:
    start_equity: float
    notional: float
    qty_decimals: int
    short_whole_shares: bool
    margin_sigma: float
    stop_loss_sigma: float
    allow_short: bool
    force_trade_after_hours: float


@dataclass(frozen=True)
class StockCostsConfig:
    commission_per_order: float
    half_spread_bps: dict[str, float]
    extra_slippage_bps: float
    sec_fee_per_million: float
    finra_taf_per_share: float
    finra_taf_max_per_trade: float
    fee_round_up_to_cent: bool
    short_borrow_pct_per_year: float
    short_sale_restriction_drop_pct: float
    latency_ms: int
    latency_jitter_ms: int
    daily_running_cost: float
    trading_days_per_year: int


@dataclass(frozen=True)
class PopulationConfig:
    size: int


@dataclass(frozen=True)
class ControlsConfig:
    coin_flip_twins: bool


@dataclass(frozen=True)
class PersistenceConfig:
    checkpoint_every_seconds: int
    equity_snapshot_every_seconds: int
    max_resume_gap_minutes: float
    keep_candles_days: int
    keep_checkpoints: int


@dataclass(frozen=True)
class EvaluationConfig:
    confirmatory_predictor: str
    final_healthy_session_minutes: int
    run_minute_rule: str
    burn_in_sessions: int
    newey_west_lags: int
    one_sided_z: float
    upper_bound_z: float
    interim_sessions: tuple[int, ...]
    tape_check_sessions: tuple[int, ...]


def _resolve(base_dir: Path, value: str) -> Path:
    """A config path with ~ expanded, and resolved from the config file's folder when relative."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def _digest(sections: dict[str, Any]) -> str:
    """SHA 256 of the validated values as canonical JSON, so comments, layout and 10 against 10.0 never matter."""
    text = json.dumps(sections, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StockConfig:
    run: RunConfig
    stock_feed: StockFeedConfig
    session: SessionConfig
    news_input: NewsInputConfig
    time: TimeConfig
    features: FeaturesConfig
    model: ModelConfig
    trading: TradingConfig
    stock_costs: StockCostsConfig
    population: PopulationConfig
    controls: ControlsConfig
    persistence: PersistenceConfig
    evaluation: EvaluationConfig | None    # S2 only
    base_dir: Path
    text: str

    def path(self, value: str) -> Path:
        """Any path from this config, resolved from the config file's folder."""
        return _resolve(self.base_dir, value)

    @property
    def data_dir(self) -> Path:
        return self.path(self.run.data_dir)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "stocks.db"

    @property
    def raw_dir(self) -> Path:
        return self.path(self.stock_feed.raw_dir)

    @property
    def news_db_path(self) -> Path:
        return self.path(self.news_input.news_db)

    def sections(self) -> dict[str, Any]:
        """Every section as plain values, with evaluation None in S1."""
        return {name: None if getattr(self, name) is None else dataclasses.asdict(getattr(self, name))
                for name in _STOCK_SPEC}

    @property
    def experiment_hash(self) -> str:
        """The hash of every setting except [run] log_level, which resume and evaluate compare."""
        sections = self.sections()
        sections["run"].pop("log_level")
        return _digest(sections)


@dataclass(frozen=True)
class NewsProcessConfig:
    kind: str
    data_dir: str
    secrets_file: str
    master_seed: int
    universe: tuple[str, ...]
    schedule_tz: str
    http_timeout_seconds: float
    max_response_bytes: int
    backlog_gap_seconds: float
    backlog_exempt_kinds: tuple[str, ...]
    max_vendor_age_hours: float
    clock_skew_max_seconds: float
    clock_skew_samples: int
    failure_backoff_cap_seconds: float
    heartbeat_seconds: float
    outage_cadence_multiple: float
    outage_floor_seconds: float
    score_wait_seconds: float
    burst_items: int
    keep_filtered_days: int
    keep_polls_days: int
    keep_dups_days: int
    keep_heartbeat_days: int
    rest_lock_file: str


@dataclass(frozen=True)
class FinnhubNewsConfig:
    enabled: bool
    company_symbols: tuple[str, ...]
    general_category: str
    max_calls_per_minute: int
    min_call_spacing_seconds: float
    regular_window_et: tuple[str, str]
    extended_window_et: tuple[str, str]
    regular_poll_seconds: int
    extended_poll_seconds: int
    overnight_poll_seconds: int
    lookback_days: int
    headline_max_chars: int
    summary_max_chars: int
    rate_limit_pause_seconds: int
    send_to_llm: bool


@dataclass(frozen=True)
class EdgarConfig:
    enabled: bool
    ciks: dict[str, int]
    forms: tuple[str, ...]
    filing_window_et: tuple[str, str]
    atom_poll_seconds: int
    submissions_poll_seconds: int
    off_window_poll_seconds: int
    max_requests_per_second: int
    max_requests_per_minute: int
    block_pause_seconds: int
    main_doc_max_chars: int
    exhibit_max_chars: int


@dataclass(frozen=True)
class FedConfig:
    enabled: bool
    feeds: tuple[str, ...]
    monetary_link_pattern: str
    press_window_et: tuple[str, str]
    press_poll_seconds: int
    speeches_window_et: tuple[str, str]
    speeches_poll_seconds: int
    speeches_off_hours_poll_seconds: int
    off_hours_poll_seconds: int
    fomc_fast_window_et: tuple[str, str]
    fomc_fast_poll_seconds: int
    summary_max_chars: int
    monetary_body_max_chars: int
    max_requests_per_minute: int
    block_pause_seconds: int


@dataclass(frozen=True)
class BlsConfig:
    enabled: bool
    feeds: tuple[str, ...]
    window_et: tuple[str, str]
    poll_seconds: int
    off_hours_poll_seconds: int
    release_fast_poll_seconds: int
    release_fast_before_minutes: int
    release_fast_after_minutes: int
    summary_max_chars: int
    max_requests_per_minute: int
    block_pause_seconds: int


@dataclass(frozen=True)
class DedupeConfig:
    hash_window_hours: int
    near_dup_window_hours: int
    near_dup_jaccard: float
    near_dup_min_tokens: int


@dataclass(frozen=True)
class PrefilterConfig:
    macro_strip_pattern: str
    alias_excludes_macro: bool


@dataclass(frozen=True)
class CalendarConfig:
    """Scheduled release times in America/New_York as YYYY-MM-DDTHH:MM, used only to pick polling cadences."""
    fomc_statement: tuple[str, ...]
    cpi: tuple[str, ...]
    empsit: tuple[str, ...]
    ppi: tuple[str, ...]
    jolts: tuple[str, ...]


@dataclass(frozen=True)
class SonnetConfig:
    enabled: bool
    model: str
    effort: str
    thinking: str
    scorer_name: str
    kinds: tuple[str, ...]
    max_tokens: int
    thinking_tokens: int
    cache_control: bool


@dataclass(frozen=True)
class LlmCapsConfig:
    ledger_file: str
    ledger_lock_file: str
    monthly_usd: float
    daily_alert_share: float
    reserve_overhead_tokens: int


@dataclass(frozen=True)
class LlmPricesConfig:
    haiku_input_per_mtok: float
    haiku_output_per_mtok: float
    haiku_cache_write_multiplier: float
    haiku_cache_read_multiplier: float
    sonnet_input_per_mtok: float
    sonnet_output_per_mtok: float
    long_prompt_threshold_tokens: int


@dataclass(frozen=True)
class LlmConfig:
    """The Haiku scorer, with [llm.sonnet], [llm.caps] and [llm.prices] nested as in the file.

    prompt_file, schema_file and probe_file are plain file names inside the btc_survivor.v2.news package.
    """
    enabled: bool
    model: str
    effort: str
    thinking: str
    scorer_name: str
    prompt_file: str
    prompt_sha256: str
    schema_file: str
    schema_sha256: str
    request_timeout_seconds: float
    max_tokens: int
    flush_max_items: int
    flush_after_seconds: float
    max_items_per_call: int
    long_form_kinds: tuple[str, ...]
    alone_kinds: tuple[str, ...]
    max_queue_age_seconds: float
    headline_max_chars: int
    text_max_chars: int
    long_form_max_chars: int
    max_input_tokens: int
    max_calls_per_batch: int
    max_requeues: int
    rate_limit_pause_minutes: float
    refusal_pause_share: float
    refusal_window_calls: int
    probe_file: str
    probe_time_utc: str
    probe_alert_direction_agreement: float
    sonnet: SonnetConfig
    caps: LlmCapsConfig
    prices: LlmPricesConfig


@dataclass(frozen=True)
class FinbertConfig:
    model_file: str
    model_sha256: str
    tokenizer_file: str
    tokenizer_sha256: str
    manifest_file: str
    source_revision: str
    window_tokens: int
    max_windows: int
    threads: int


@dataclass(frozen=True)
class NewsConfig:
    news: NewsProcessConfig
    finnhub: FinnhubNewsConfig
    edgar: EdgarConfig
    fed: FedConfig
    bls: BlsConfig
    dedupe: DedupeConfig
    prefilter: PrefilterConfig
    calendar: CalendarConfig
    llm: LlmConfig
    finbert: FinbertConfig
    base_dir: Path
    text: str

    def path(self, value: str) -> Path:
        """Any path from this config, resolved from the config file's folder."""
        return _resolve(self.base_dir, value)

    @property
    def data_dir(self) -> Path:
        return self.path(self.news.data_dir)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "news.db"

    def sections(self) -> dict[str, Any]:
        """Every section as plain values, with the [llm] subtables nested inside llm."""
        return {name: dataclasses.asdict(getattr(self, name))
                for name in ("news", "finnhub", "edgar", "fed", "bls", "dedupe", "prefilter", "calendar", "llm",
                             "finbert")}

    @property
    def config_hash(self) -> str:
        """The hash of every setting, written to news.db meta and recorded by a stock run at launch."""
        return _digest(self.sections())


class _Bad(Exception):
    """One value that failed its spec, carrying the reason."""


def _range(v: float, lo: float, hi: float, lo_inc: bool = True, hi_inc: bool = True) -> float:
    """Return v when it lies in the range, else raise with the allowed range."""
    if not ((v >= lo if lo_inc else v > lo) and (v <= hi if hi_inc else v < hi)):
        raise _Bad(f"out of range, allowed {'[' if lo_inc else '('}{lo}, {hi}{']' if hi_inc else ')'}")
    return v


def _int(v: Any, lo: int, hi: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise _Bad("expected an integer")
    return _range(v, lo, hi)


def _float(v: Any, lo: float, hi: float, lo_inc: bool = True, hi_inc: bool = True) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        raise _Bad("expected a number")
    return _range(float(v), lo, hi, lo_inc, hi_inc)


def _bool(v: Any) -> bool:
    if not isinstance(v, bool):
        raise _Bad("expected true or false")
    return v


def _str(v: Any) -> str:
    if not isinstance(v, str) or not v:
        raise _Bad("expected a non empty string")
    return v


def _text(v: Any) -> str:
    if not isinstance(v, str):
        raise _Bad("expected a string, which may be empty")
    return v


def _choice(v: Any, allowed: tuple[str, ...]) -> str:
    if not isinstance(v, str) or v not in allowed:
        if len(allowed) == 1:
            raise _Bad(f"must be {allowed[0]!r}, the only accepted value")
        raise _Bad(f"must be one of {', '.join(allowed)}")
    return v


def _name(v: Any) -> str:
    if not isinstance(v, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", v):
        raise _Bad("expected lower case letters, digits and underscores, starting with a letter")
    return v


def _file(v: Any) -> str:
    if not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.]*", v):
        raise _Bad("expected a plain file name with no folder")
    return v


def _sha(v: Any) -> str:
    if not isinstance(v, str) or not re.fullmatch(r"(|[0-9a-f]{64})", v):
        raise _Bad("expected an empty string, until pinned, or 64 lower case hex digits")
    return v


def _hhmm(v: Any) -> str:
    if not isinstance(v, str) or not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", v):
        raise _Bad("expected a time as HH:MM")
    return v


def _window(v: Any) -> tuple[str, str]:
    if not isinstance(v, list) or len(v) != 2:
        raise _Bad("expected a start and an end as two HH:MM times")
    start, end = _hhmm(v[0]), _hhmm(v[1])
    if start >= end:
        raise _Bad("the start must come before the end")
    return start, end


def _ints(v: Any, lo: int, hi: int) -> tuple[int, ...]:
    if not isinstance(v, list) or not v:
        raise _Bad("expected a non empty list of integers")
    return tuple(_int(x, lo, hi) for x in v)


def _strs(v: Any, allowed: tuple[str, ...] | None = None) -> tuple[str, ...]:
    if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
        raise _Bad("expected a list of non empty strings")
    if len(set(v)) != len(v):
        raise _Bad("lists a value twice")
    bad = [x for x in v if allowed is not None and x not in allowed]
    if bad:
        raise _Bad(f"{bad[0]!r} is not allowed, allowed values are {', '.join(allowed)}")
    return tuple(v)


def _times(v: Any) -> tuple[str, ...]:
    if not isinstance(v, list):
        raise _Bad("expected a list of YYYY-MM-DDTHH:MM times")
    for x in v:
        try:
            if not isinstance(x, str) or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d", x):
                raise ValueError
            datetime.strptime(x, "%Y-%m-%dT%H:%M")
        except ValueError:
            raise _Bad(f"{x!r} is not a YYYY-MM-DDTHH:MM time") from None
    if list(v) != sorted(set(v)):
        raise _Bad("times must be in ascending order without repeats")
    return tuple(v)


def _by_symbol(v: Any, check: Any, lo: float, hi: float) -> dict[str, Any]:
    if not isinstance(v, dict) or not v:
        raise _Bad("expected a non empty table of symbol = value")
    return {k: check(x, lo, hi) for k, x in v.items()}


_SECONDS = (_float, 0, 86400, False, True)
_DAYS = (_int, 1, 3650)
_CHARS = (_int, 1, 100000)
_POLL = (_int, 1, 86400)
_PER_MINUTE = (_int, 1, 1000)

# Each key maps to (checker, *arguments). A dotted section name such as llm.sonnet is a subtable of its parent.
_STOCK_SPEC: dict[str, dict[str, tuple]] = {
    "run": {
        "kind": (_choice, ("stocks",)),
        "name": (_name,),
        "master_seed": (_int, 0, 2**63 - 1),
        "data_dir": (_str,),
        "log_level": (_choice, LOG_LEVELS),
        "secrets_file": (_str,),
    },
    "stock_feed": {
        "provider": (_choice, ("finnhub",)),
        "symbols": (_strs,),
        "index_symbol": (_str,),
        "feed_stale_after_seconds": (_float, 0, 300, False, True),
        "symbol_stale_after_seconds": (_float, 0, 3600, False, True),
        "max_print_delay_seconds": (_float, 2, 5),
        "late_print_seconds": (_float, 0, 10, False, True),
        "price_excluded_conditions": (_strs,),
        "reconnect_backoff_start_seconds": (_float, 0, 600, False, True),
        "reconnect_backoff_max_seconds": (_float, 0, 3600, False, True),
        "mark_healthy_outside_session_seconds": (_float, 0, 3600, False, True),
        "record_raw_messages": (_bool,),
        "raw_dir": (_str,),
        "keep_raw_days": _DAYS,
        "lock_file": (_str,),
    },
    "session": {
        "first_open_session_minute": (_int, 0, 390),
        "last_open_minutes_before_close": (_int, 0, 390),
        "flatten_minutes_before_close": (_int, 0, 390),
        "calendar_warn_days": _DAYS,
        "max_overnight_gap_pct": (_float, 0, 100, False, True),
    },
    "news_input": {
        "use_news": (_bool,),
        "news_db": (_str,),
        "poll_seconds": (_float, 0, 60, False, True),
        "reopen_after_silent_seconds": (_float, 0, 3600, False, True),
        "heartbeat_max_age_seconds": (_float, 0, 3600, False, True),
        "half_life_session_minutes": (_float, 0, 100000, False, True),
        "drop_after_half_lives": (_float, 0, 1000, False, True),
    },
    "time": {
        "horizon_minutes": (_int, 1, 390),
        "history_session_minutes": (_int, 1, 100000),
        "warmup_hours": (_float, 0, 1000),
    },
    "features": {
        "return_windows_minutes": (_ints, 1, 10000),
        "day_return_minutes": (_int, 1, 10000),
        "vol_window_minutes": (_int, 2, 10000),
        "volume_window_minutes": (_int, 1, 10000),
        "volume_baseline_minutes": (_int, 1, 10000),
        "volume_ratio_clip": (_float, 0, 100, False, True),
        "tick_window_minutes": (_int, 1, 10000),
        "standardizer_half_life_hours": (_float, 0, 10000, False, True),
        "clip_z": (_float, 0, 100, False, True),
        "min_samples": (_int, 1, 100000),
        "min_window_coverage": (_float, 0, 1, False, True),
        "series_capacity_minutes": (_int, 1, 100000),
    },
    "model": {
        "inputs": (_int, 12, 12),
        "hidden_units": (_int, 1, 1024),
        "init_output_scale": (_float, 0, 10, False, True),
        "batch_size": (_int, 1, 1024),
        "replay_window_minutes": (_int, 1, 100000),
        "huber_delta": (_float, 0, 1000, False, True),
        "grad_clip": (_float, 0, 1e6, False, True),
        "adam_beta1": (_float, 0, 1, True, False),
        "adam_beta2": (_float, 0, 1, True, False),
        "learning_rate": (_float, 0, 1, False, True),
        "weight_decay": (_float, 0, 1, True, False),
        "skill_window_samples": (_int, 2, 100000),
    },
    "trading": {
        "start_equity": (_float, 0, 1e12, False, True),
        "notional": (_float, 0, 1e12, False, True),
        "qty_decimals": (_int, 0, 9),
        "short_whole_shares": (_bool,),
        "margin_sigma": (_float, 0, 100),
        "stop_loss_sigma": (_float, 0, 100, False, True),
        "allow_short": (_bool,),
        "force_trade_after_hours": (_float, 0, 10000, False, True),
    },
    "stock_costs": {
        "commission_per_order": (_float, 0, 1000),
        "half_spread_bps": (_by_symbol, _float, 0, 1000),
        "extra_slippage_bps": (_float, 0, 1000),
        "sec_fee_per_million": (_float, 0, 1e6),
        "finra_taf_per_share": (_float, 0, 1),
        "finra_taf_max_per_trade": (_float, 0, 1e6),
        "fee_round_up_to_cent": (_bool,),
        "short_borrow_pct_per_year": (_float, 0, 100),
        "short_sale_restriction_drop_pct": (_float, 0, 100, False, True),
        "latency_ms": (_int, 0, 30000),
        "latency_jitter_ms": (_int, 0, 30000),
        "daily_running_cost": (_float, 0, 1e9),
        "trading_days_per_year": (_int, 1, 366),
    },
    "population": {
        "size": (_int, 1, 10000),
    },
    "controls": {
        "coin_flip_twins": (_bool,),
    },
    "persistence": {
        "checkpoint_every_seconds": (_int, 60, 86400),
        "equity_snapshot_every_seconds": (_int, 60, 86400),
        "max_resume_gap_minutes": (_float, 1, 100000),
        "keep_candles_days": (_int, 0, 3650),
        "keep_checkpoints": (_int, 1, 1000),
    },
    "evaluation": {
        "confirmatory_predictor": (_choice, PREDICTORS),
        "final_healthy_session_minutes": (_int, 1, 10**7),
        "run_minute_rule": (_choice, ("news_healthy_and_any_symbol_healthy",)),
        "burn_in_sessions": (_int, 0, 1000),
        "newey_west_lags": (_int, 0, 10000),
        "one_sided_z": (_float, 0, 10, False, True),
        "upper_bound_z": (_float, 0, 10, False, True),
        "interim_sessions": (_ints, 1, 10000),
        "tape_check_sessions": (_ints, 1, 10000),
    },
}

_NEWS_SPEC: dict[str, dict[str, tuple]] = {
    "news": {
        "kind": (_choice, ("news",)),
        "data_dir": (_str,),
        "secrets_file": (_str,),
        "master_seed": (_int, 0, 2**63 - 1),
        "universe": (_strs,),
        "schedule_tz": (_choice, ("America/New_York",)),
        "http_timeout_seconds": (_float, 0, 300, False, True),
        "max_response_bytes": (_int, 1, 10**9),
        "backlog_gap_seconds": (_float, 0, 86400),
        "backlog_exempt_kinds": (_strs, NEWS_KINDS),
        "max_vendor_age_hours": (_float, 0, 10000, False, True),
        "clock_skew_max_seconds": (_float, 0, 3600, False, True),
        "clock_skew_samples": (_int, 1, 10000),
        "failure_backoff_cap_seconds": _SECONDS,
        "heartbeat_seconds": (_float, 0, 3600, False, True),
        "outage_cadence_multiple": (_float, 0, 1000, False, True),
        "outage_floor_seconds": (_float, 0, 86400),
        "score_wait_seconds": _SECONDS,
        "burst_items": (_int, 1, 10**6),
        "keep_filtered_days": _DAYS,
        "keep_polls_days": _DAYS,
        "keep_dups_days": _DAYS,
        "keep_heartbeat_days": _DAYS,
        "rest_lock_file": (_str,),
    },
    "finnhub": {
        "enabled": (_bool,),
        "company_symbols": (_strs,),
        "general_category": (_str,),
        "max_calls_per_minute": (_int, 1, 60),
        "min_call_spacing_seconds": (_float, 0, 3600),
        "regular_window_et": (_window,),
        "extended_window_et": (_window,),
        "regular_poll_seconds": _POLL,
        "extended_poll_seconds": _POLL,
        "overnight_poll_seconds": _POLL,
        "lookback_days": (_int, 1, 30),
        "headline_max_chars": _CHARS,
        "summary_max_chars": _CHARS,
        "rate_limit_pause_seconds": _POLL,
        "send_to_llm": (_bool,),
    },
    "edgar": {
        "enabled": (_bool,),
        "ciks": (_by_symbol, _int, 1, 9999999999),
        "forms": (_strs, ("8-K", "8-K/A")),
        "filing_window_et": (_window,),
        "atom_poll_seconds": _POLL,
        "submissions_poll_seconds": _POLL,
        "off_window_poll_seconds": _POLL,
        "max_requests_per_second": (_int, 1, 10),
        "max_requests_per_minute": _PER_MINUTE,
        "block_pause_seconds": _POLL,
        "main_doc_max_chars": _CHARS,
        "exhibit_max_chars": _CHARS,
    },
    "fed": {
        "enabled": (_bool,),
        "feeds": (_strs, ("press_all", "press_monetary", "speeches")),
        "monetary_link_pattern": (_str,),
        "press_window_et": (_window,),
        "press_poll_seconds": _POLL,
        "speeches_window_et": (_window,),
        "speeches_poll_seconds": _POLL,
        "speeches_off_hours_poll_seconds": _POLL,
        "off_hours_poll_seconds": _POLL,
        "fomc_fast_window_et": (_window,),
        "fomc_fast_poll_seconds": _POLL,
        "summary_max_chars": _CHARS,
        "monetary_body_max_chars": _CHARS,
        "max_requests_per_minute": _PER_MINUTE,
        "block_pause_seconds": _POLL,
    },
    "bls": {
        "enabled": (_bool,),
        "feeds": (_strs, ("cpi", "empsit", "ppi", "jolts")),
        "window_et": (_window,),
        "poll_seconds": _POLL,
        "off_hours_poll_seconds": _POLL,
        "release_fast_poll_seconds": _POLL,
        "release_fast_before_minutes": (_int, 0, 1440),
        "release_fast_after_minutes": (_int, 0, 1440),
        "summary_max_chars": _CHARS,
        "max_requests_per_minute": _PER_MINUTE,
        "block_pause_seconds": _POLL,
    },
    "dedupe": {
        "hash_window_hours": (_int, 1, 10000),
        "near_dup_window_hours": (_int, 1, 10000),
        "near_dup_jaccard": (_float, 0, 1, False, True),
        "near_dup_min_tokens": (_int, 1, 1000),
    },
    "prefilter": {
        "macro_strip_pattern": (_str,),
        "alias_excludes_macro": (_bool,),
    },
    "calendar": {
        "fomc_statement": (_times,),
        "cpi": (_times,),
        "empsit": (_times,),
        "ppi": (_times,),
        "jolts": (_times,),
    },
    "llm": {
        "enabled": (_bool,),
        "model": (_choice, (HAIKU_MODEL,)),
        "effort": (_choice, ("low",)),
        "thinking": (_choice, ("disabled",)),
        "scorer_name": (_name,),
        "prompt_file": (_file,),
        "prompt_sha256": (_sha,),
        "schema_file": (_file,),
        "schema_sha256": (_sha,),
        "request_timeout_seconds": (_float, 0, 600, False, True),
        "max_tokens": (_int, 1, 64000),
        "flush_max_items": (_int, 1, 25),
        "flush_after_seconds": (_float, 0, 3600, False, True),
        "max_items_per_call": (_int, 1, 25),
        "long_form_kinds": (_strs, NEWS_KINDS),
        "alone_kinds": (_strs, NEWS_KINDS),
        "max_queue_age_seconds": _SECONDS,
        "headline_max_chars": _CHARS,
        "text_max_chars": _CHARS,
        "long_form_max_chars": _CHARS,
        "max_input_tokens": (_int, 1, 99999),
        "max_calls_per_batch": (_int, 1, 1000),
        "max_requeues": (_int, 0, 10),
        "rate_limit_pause_minutes": (_float, 0, 1440, False, True),
        "refusal_pause_share": (_float, 0, 1, False, True),
        "refusal_window_calls": (_int, 1, 10000),
        "probe_file": (_file,),
        "probe_time_utc": (_hhmm,),
        "probe_alert_direction_agreement": (_float, 0, 1),
    },
    "llm.sonnet": {
        "enabled": (_bool,),
        "model": (_choice, (SONNET_MODEL,)),
        "effort": (_choice, ("low",)),
        "thinking": (_choice, ("disabled", "between_tools", "adaptive")),
        "scorer_name": (_name,),
        "kinds": (_strs, OFFICIAL_KINDS),
        "max_tokens": (_int, 1, 64000),
        "thinking_tokens": (_int, 0, 64000),
        "cache_control": (_bool,),
    },
    "llm.caps": {
        "ledger_file": (_str,),
        "ledger_lock_file": (_str,),
        "monthly_usd": (_float, 0, MAX_MONTHLY_USD, False, True),
        "daily_alert_share": (_float, 0, 1, False, False),
        "reserve_overhead_tokens": (_int, 0, 10**6),
    },
    "llm.prices": {
        "haiku_input_per_mtok": (_float, 0, 1000, False, True),
        "haiku_output_per_mtok": (_float, 0, 1000, False, True),
        "haiku_cache_write_multiplier": (_float, 1, 10),
        "haiku_cache_read_multiplier": (_float, 0, 1),
        "sonnet_input_per_mtok": (_float, 0, 1000, False, True),
        "sonnet_output_per_mtok": (_float, 0, 1000, False, True),
        "long_prompt_threshold_tokens": (_int, 1, 10**7),
    },
    "finbert": {
        "model_file": (_str,),
        "model_sha256": (_sha,),
        "tokenizer_file": (_str,),
        "tokenizer_sha256": (_sha,),
        "manifest_file": (_str,),
        "source_revision": (_text,),
        "window_tokens": (_int, 2, 512),
        "max_windows": (_int, 1, 1000),
        "threads": (_int, 1, 64),
    },
}


def _values(raw: dict[str, Any], spec: dict[str, dict[str, tuple]], optional: tuple[str, ...],
            errors: list[str]) -> dict[str, dict[str, Any] | None]:
    """Check every table against its spec, recording each unknown, missing or bad key, and return the clean values."""
    for name in raw:
        if name not in {s.partition(".")[0] for s in spec}:
            errors.append(f"unknown section [{name}]")
    out: dict[str, dict[str, Any] | None] = {}
    for name, keys in spec.items():
        parent, _, child = name.partition(".")
        table = raw.get(parent)
        if child:
            table = table.get(child) if isinstance(table, dict) else None
        if table is None and name in optional:
            out[name] = None
            continue
        if not isinstance(table, dict):
            errors.append(f"missing section [{name}]")
            continue
        for key in table:
            if key not in keys and f"{name}.{key}" not in spec:
                errors.append(f"[{name}] {key}: unknown key")
        clean: dict[str, Any] = {}
        for key, (check, *args) in keys.items():
            if key not in table:
                errors.append(f"[{name}] {key}: missing")
                continue
            try:
                clean[key] = check(table[key], *args)
            except _Bad as exc:
                errors.append(f"[{name}] {key} = {table[key]!r}: {exc}")
        out[name] = clean
    return out


def _symbol_problems(where: str, symbols: tuple[str, ...]) -> list[str]:
    """Symbols must be upper case tickers, at least one and at most the 50 one websocket key allows."""
    problems = [f"{where} = {s!r} must be an upper case ticker" for s in symbols if not _SYMBOL.fullmatch(s)]
    if not symbols:
        problems.append(f"{where} must list at least one symbol")
    if len(symbols) > MAX_SYMBOLS:
        problems.append(f"{where} lists {len(symbols)} symbols, at most {MAX_SYMBOLS}, the websocket limit per key")
    return problems


def _stock_checks(v: dict[str, Any], errors: list[str]) -> None:
    """The cross checks of PLAN_V2 section 15 for one stock run config, plus window and capacity sanity."""
    feed, session, time_, feat = v["stock_feed"], v["session"], v["time"], v["features"]
    costs, persist, news_input = v["stock_costs"], v["persistence"], v["news_input"]
    symbols = feed["symbols"]

    errors += _symbol_problems("[stock_feed] symbols", symbols)
    if feed["index_symbol"] not in symbols:
        errors.append(f"[stock_feed] index_symbol = {feed['index_symbol']!r} must be one of [stock_feed] symbols")
    for s in sorted(set(costs["half_spread_bps"]) - set(symbols)):
        errors.append(f"[stock_costs] half_spread_bps names {s}, which is not in [stock_feed] symbols")
    for s in symbols:
        if s not in costs["half_spread_bps"]:
            errors.append(f"[stock_costs] half_spread_bps has no allowance for {s}")

    end = session["first_open_session_minute"] + time_["horizon_minutes"] + session["flatten_minutes_before_close"]
    if end > EARLY_CLOSE_MINUTES:
        errors.append(f"[session] first_open_session_minute plus [time] horizon_minutes plus [session] "
                      f"flatten_minutes_before_close is {end}, which must fit in an early close of "
                      f"{EARLY_CLOSE_MINUTES} minutes")
    if feed["symbol_stale_after_seconds"] < feed["feed_stale_after_seconds"]:
        errors.append("[stock_feed] symbol_stale_after_seconds must be at least feed_stale_after_seconds")
    if feed["late_print_seconds"] != 2 * feed["max_print_delay_seconds"]:
        errors.append(f"[stock_feed] late_print_seconds = {feed['late_print_seconds']!r} must be twice "
                      f"max_print_delay_seconds, {2 * feed['max_print_delay_seconds']!r}")
    if feed["reconnect_backoff_max_seconds"] < feed["reconnect_backoff_start_seconds"]:
        errors.append("[stock_feed] reconnect_backoff_max_seconds must be at least reconnect_backoff_start_seconds")

    if news_input["use_news"] and v["evaluation"] is None:
        errors.append("[news_input] use_news = true needs the [evaluation] section, only S2 sets use_news")
    if not news_input["use_news"] and v["evaluation"] is not None:
        errors.append("[evaluation] belongs to S2 only, so a run with [news_input] use_news = false may not have it")

    keep = persist["keep_candles_days"]
    if 0 < keep < MIN_KEEP_CANDLES_DAYS:
        errors.append(f"[persistence] keep_candles_days = {keep} must be 0, kept for the life of the run, or at "
                      f"least {MIN_KEEP_CANDLES_DAYS}, twice the calendar length of 40 sessions")
    for key in ("checkpoint_every_seconds", "equity_snapshot_every_seconds"):
        if persist[key] % 60:
            errors.append(f"[persistence] {key} must be a whole number of minutes")

    if len(feat["return_windows_minutes"]) != 2:
        errors.append("[features] return_windows_minutes must list exactly 2 windows, features 1 and 2")
    else:
        needed = max(max(feat["return_windows_minutes"]) + 1, feat["day_return_minutes"] + 1,
                     feat["vol_window_minutes"] + 1, feat["volume_baseline_minutes"], feat["tick_window_minutes"])
        if time_["history_session_minutes"] < needed:
            errors.append(f"[time] history_session_minutes must be at least {needed} to cover the longest feature "
                          f"window")
        if feat["series_capacity_minutes"] < time_["history_session_minutes"]:
            errors.append("[features] series_capacity_minutes must be at least [time] history_session_minutes")
    if feat["series_capacity_minutes"] < MIN_SERIES_CAPACITY:
        errors.append(f"[features] series_capacity_minutes = {feat['series_capacity_minutes']} must be at least "
                      f"{MIN_SERIES_CAPACITY}, the least a stock series holds")
    if v["trading"]["notional"] > v["trading"]["start_equity"]:
        errors.append("[trading] notional must not exceed start_equity")


def _news_checks(v: dict[str, Any], errors: list[str]) -> None:
    """The cross checks of PLAN_V2 section 15 for the news config that a single key cannot express."""
    universe = v["news"]["universe"]
    errors += _symbol_problems("[news] universe", universe)
    for s in v["finnhub"]["company_symbols"]:
        if s not in universe:
            errors.append(f"[finnhub] company_symbols names {s}, which is not in [news] universe")
    for s in v["edgar"]["ciks"]:
        if s not in universe:
            errors.append(f"[edgar] ciks names {s}, which is not in [news] universe")
    if v["llm"]["scorer_name"] == v["llm.sonnet"]["scorer_name"]:
        errors.append("[llm] scorer_name and [llm.sonnet] scorer_name must differ, each scorer has its own rows")
    sonnet = v["llm.sonnet"]
    if (sonnet["thinking"] == "disabled") != (sonnet["thinking_tokens"] == 0):
        errors.append("[llm.sonnet] thinking_tokens must be 0 while thinking is disabled and at least 1 otherwise")
    if sonnet["max_tokens"] + sonnet["thinking_tokens"] > 64000:
        errors.append("[llm.sonnet] max_tokens and thinking_tokens together must not exceed 64000")
    try:
        re.compile(v["prefilter"]["macro_strip_pattern"])
    except re.error as exc:
        errors.append(f"[prefilter] macro_strip_pattern does not compile: {exc}")


def _format(errors: list[str]) -> str:
    return "invalid configuration\n  " + "\n  ".join(errors)


def _toml(text: str) -> dict[str, Any]:
    """Parse TOML text, turning a syntax error into a ConfigError."""
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"config is not valid TOML: {exc}") from exc


def _kind(raw: dict[str, Any]) -> str | None:
    """Which version 2 config the parsed TOML claims to be, stocks or news, or None for anything else."""
    run, news = raw.get("run"), raw.get("news")
    if isinstance(run, dict) and run.get("kind") == "stocks":
        return "stocks"
    if isinstance(news, dict) and news.get("kind") == "news":
        return "news"
    return None


def parse_stock_config(text: str, base_dir: Path) -> StockConfig:
    """Parse and validate a stock run config. All problems are reported together in one ConfigError."""
    raw = _toml(text)
    if _kind(raw) != "stocks":
        found = "a news config" if _kind(raw) == "news" else "no [run] kind"
        raise ConfigError(f'not a version 2 stock run config, which needs [run] kind = "stocks", found {found}')
    errors: list[str] = []
    v = _values(raw, _STOCK_SPEC, ("evaluation",), errors)
    if not errors:
        _stock_checks(v, errors)
    if errors:
        raise ConfigError(_format(errors))
    classes = {"run": RunConfig, "stock_feed": StockFeedConfig, "session": SessionConfig,
               "news_input": NewsInputConfig, "time": TimeConfig, "features": FeaturesConfig, "model": ModelConfig,
               "trading": TradingConfig, "stock_costs": StockCostsConfig, "population": PopulationConfig,
               "controls": ControlsConfig, "persistence": PersistenceConfig, "evaluation": EvaluationConfig}
    sections = {name: None if v[name] is None else cls(**v[name]) for name, cls in classes.items()}
    return StockConfig(**sections, base_dir=base_dir, text=text)


def parse_news_config(text: str, base_dir: Path) -> NewsConfig:
    """Parse and validate the news config. All problems are reported together in one ConfigError."""
    raw = _toml(text)
    if _kind(raw) != "news":
        found = "a stock run config" if _kind(raw) == "stocks" else "no [news] kind"
        raise ConfigError(f'not a version 2 news config, which needs [news] kind = "news", found {found}')
    errors: list[str] = []
    v = _values(raw, _NEWS_SPEC, (), errors)
    if not errors:
        _news_checks(v, errors)
    if errors:
        raise ConfigError(_format(errors))
    llm = LlmConfig(**v["llm"], sonnet=SonnetConfig(**v["llm.sonnet"]), caps=LlmCapsConfig(**v["llm.caps"]),
                    prices=LlmPricesConfig(**v["llm.prices"]))
    return NewsConfig(
        news=NewsProcessConfig(**v["news"]),
        finnhub=FinnhubNewsConfig(**v["finnhub"]),
        edgar=EdgarConfig(**v["edgar"]),
        fed=FedConfig(**v["fed"]),
        bls=BlsConfig(**v["bls"]),
        dedupe=DedupeConfig(**v["dedupe"]),
        prefilter=PrefilterConfig(**v["prefilter"]),
        calendar=CalendarConfig(**v["calendar"]),
        llm=llm,
        finbert=FinbertConfig(**v["finbert"]),
        base_dir=base_dir,
        text=text,
    )


_PAIR_FREE = {("run", "name"), ("run", "data_dir"), ("news_input", "use_news")}


def check_run_pair(s1: StockConfig, s2: StockConfig) -> None:
    """S1 and S2 must be identical apart from [run] name and data_dir, [news_input] use_news and [evaluation]."""
    errors: list[str] = []
    if s1.news_input.use_news:
        errors.append("S1 may not set [news_input] use_news, the first config must be the prices run")
    if not s2.news_input.use_news:
        errors.append("S2 must set [news_input] use_news, the second config must be the news run")
    if s1.run.name == s2.run.name:
        errors.append("S1 and S2 must have different [run] name values")
    if s1.data_dir.resolve() == s2.data_dir.resolve():
        errors.append("S1 and S2 must have different [run] data_dir folders")
    if s1.base_dir.resolve() != s2.base_dir.resolve():
        errors.append("S1 and S2 must sit in one folder, so their relative paths, such as news_db and lock_file, "
                      "point at the same files")
    a, b = s1.sections(), s2.sections()
    for section, values in a.items():
        if section == "evaluation":
            continue
        for key, value in values.items():
            if (section, key) not in _PAIR_FREE and value != b[section][key]:
                errors.append(f"[{section}] {key} differs between S1 and S2, which may differ only in [run] name "
                              f"and data_dir, [news_input] use_news and [evaluation]")
    if errors:
        raise ConfigError(_format(errors))


def _read(path: str | Path) -> tuple[str, Path]:
    """The text of a config file and the folder its relative paths resolve from."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc.strerror}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"config file {path} is not valid UTF-8: {exc}") from exc
    return text, path.resolve().parent


def load_stock_config(path: str | Path) -> StockConfig:
    """Read and validate one stock run config."""
    return parse_stock_config(*_read(path))


def load_news_config(path: str | Path) -> NewsConfig:
    """Read and validate the news config."""
    return parse_news_config(*_read(path))


def load_run_pair(s1_path: str | Path, s2_path: str | Path) -> tuple[StockConfig, StockConfig]:
    """Read S1 and S2 for the one stock process and refuse them unless they differ only where allowed."""
    s1, s2 = load_stock_config(s1_path), load_stock_config(s2_path)
    check_run_pair(s1, s2)
    return s1, s2


def config_kind(path: str | Path) -> str:
    """Whether a file is a stocks or a news config, refusing anything else, such as a version 1 config."""
    kind = _kind(_toml(_read(path)[0]))
    if kind is None:
        raise ConfigError(f'{path} is not a version 2 config, which needs [run] kind = "stocks" or '
                          f'[news] kind = "news"')
    return kind


def load_v2_config(path: str | Path) -> StockConfig | NewsConfig:
    """Load a stocks or news config by its kind, as init-db does."""
    return load_stock_config(path) if config_kind(path) == "stocks" else load_news_config(path)
