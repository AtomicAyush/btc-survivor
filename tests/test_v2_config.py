"""The version 2 config loaders enforce every cross check of PLAN_V2 section 15 and hash the settings."""

import copy
import dataclasses
import json
import re
import tomllib
from pathlib import Path

import pytest

from btc_survivor.v2.config import (HAIKU_MODEL, OFFICIAL_KINDS, SONNET_MODEL, ConfigError, NewsConfig,
                                    StockConfig, check_run_pair, config_kind, load_news_config, load_run_pair,
                                    load_stock_config, load_v2_config, parse_news_config, parse_stock_config)
from helpers import ROOT

CONFIG_DIR = ROOT / "configs" / "v2"
S1, S2, NEWS = CONFIG_DIR / "s1_prices.toml", CONFIG_DIR / "s2_news.toml", CONFIG_DIR / "news.toml"
DATA_V2 = (ROOT / "data" / "v2").resolve()
OPUS = "claude-opus-5-5"


def raw(path: Path) -> dict:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def toml_value(v) -> str:
    """One TOML value, with every table written inline, which parses back to the same nested dict."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    return "{ " + ", ".join(f"{json.dumps(k)} = {toml_value(x)}" for k, x in v.items()) + " }"


def dump(data: dict) -> str:
    """Serialize a parsed config back to TOML, so tests can change one key and reload."""
    lines = []
    for section, table in data.items():
        lines.append(f"[{section}]")
        lines += [f"{json.dumps(k)} = {toml_value(v)}" for k, v in table.items()]
        lines.append("")
    return "\n".join(lines)


def stock(data: dict) -> StockConfig:
    return parse_stock_config(dump(data), CONFIG_DIR)


def news(data: dict) -> NewsConfig:
    return parse_news_config(dump(data), CONFIG_DIR)


def changed(path: Path, change) -> dict:
    data = copy.deepcopy(raw(path))
    change(data)
    return data


def _set(section, key, value):
    def change(data):
        table = data
        for part in section.split("."):
            table = table[part]
        table[key] = value
    return change


def _many(*changes):
    def change(data):
        for c in changes:
            c(data)
    return change


def _drop(section, key=None):
    def change(data):
        if key is None:
            del data[section]
        else:
            del data[section][key]
    return change


def _fifty_one_symbols(data):
    symbols = [f"Z{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(50)] + ["QQQ"]
    data["stock_feed"]["symbols"] = symbols
    data["stock_costs"]["half_spread_bps"] = {s: 1.0 for s in symbols}


def _fifty_symbols(data):
    _fifty_one_symbols(data)
    data["stock_feed"]["symbols"] = data["stock_feed"]["symbols"][1:]
    del data["stock_costs"]["half_spread_bps"]["ZAA"]


# --- the committed configs and the owner's decisions ---

def test_committed_configs_load_as_a_pair_and_a_news_config():
    s1, s2 = load_run_pair(S1, S2)
    cfg = load_news_config(NEWS)
    assert (s1.run.kind, s2.run.kind, cfg.news.kind) == ("stocks", "stocks", "news")
    assert (s1.run.name, s2.run.name) == ("s1_prices", "s2_news")
    assert s1.news_input.use_news is False and s1.evaluation is None
    assert s2.news_input.use_news is True and s2.evaluation.confirmatory_predictor == "news"
    assert s1.stock_feed.symbols == ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "QQQ")
    assert s1.stock_costs.half_spread_bps["QQQ"] == 0.5 and s1.stock_costs.half_spread_bps["TSLA"] == 2.0
    assert s1.model.inputs == 12 and s1.population.size == 24
    assert s1.persistence.keep_candles_days == 0
    assert "[run]" in s1.text and "[news]" in cfg.text


def test_owner_decisions_are_in_the_news_config():
    cfg = load_news_config(NEWS)
    assert cfg.llm.caps.monthly_usd == 20.0
    assert cfg.llm.caps.daily_alert_share == 0.25
    assert not [f.name for f in dataclasses.fields(cfg.llm.caps) if "daily" in f.name and f.name != "daily_alert_share"]
    assert cfg.finnhub.send_to_llm is True
    assert (cfg.llm.model, cfg.llm.effort, cfg.llm.thinking) == (HAIKU_MODEL, "low", "disabled")
    assert (cfg.llm.sonnet.model, cfg.llm.sonnet.effort, cfg.llm.sonnet.thinking) == (SONNET_MODEL, "low", "disabled")
    assert cfg.llm.sonnet.kinds == OFFICIAL_KINDS and cfg.llm.sonnet.cache_control is False
    assert set(re.findall(r"claude-[a-z0-9-]+", NEWS.read_text(encoding="utf-8"))) == {HAIKU_MODEL, SONNET_MODEL}


def test_universe_matches_across_the_configs():
    s1, cfg = load_stock_config(S1), load_news_config(NEWS)
    assert cfg.news.universe == s1.stock_feed.symbols
    assert set(cfg.finnhub.company_symbols) == set(s1.stock_feed.symbols) - {s1.stock_feed.index_symbol}
    assert set(cfg.edgar.ciks) == set(cfg.finnhub.company_symbols)


def test_data_folders_live_under_data_v2():
    s1, s2 = load_run_pair(S1, S2)
    cfg = load_news_config(NEWS)
    assert s1.data_dir.resolve() == DATA_V2 / "s1_prices"
    assert s2.data_dir.resolve() == DATA_V2 / "s2_news"
    assert s1.raw_dir.resolve() == s2.raw_dir.resolve() == DATA_V2 / "raw_stocks"
    assert cfg.data_dir.resolve() == DATA_V2 / "news"
    assert s1.news_db_path.resolve() == s2.news_db_path.resolve() == cfg.db_path.resolve()
    assert s1.db_path.parent == s1.data_dir


def test_secrets_and_locks_live_outside_the_repository():
    s1, cfg = load_stock_config(S1), load_news_config(NEWS)
    home = Path.home() / ".config" / "btc_survivor"
    for value in (s1.run.secrets_file, cfg.news.secrets_file):
        assert s1.path(value) == home / "secrets.env"
    for value in (s1.stock_feed.lock_file, cfg.news.rest_lock_file, cfg.llm.caps.ledger_file,
                  cfg.llm.caps.ledger_lock_file):
        assert s1.path(value).parent == home


def test_no_config_can_bind_a_host_or_port():
    for path in (S1, S2, NEWS):
        keys = set()

        def walk(table):
            for k, v in table.items():
                keys.add(k)
                if isinstance(v, dict):
                    walk(v)
        walk(raw(path))
        assert not keys & {"host", "port", "bind", "listen", "tunnel"}, path.name


def test_configs_are_frozen():
    s1, cfg = load_stock_config(S1), load_news_config(NEWS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s1.trading.notional = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.llm.caps.monthly_usd = 1000.0


def test_integers_are_accepted_for_float_keys():
    cfg = stock(changed(S1, _set("trading", "start_equity", 10000)))
    assert cfg.trading.start_equity == 10000.0 and isinstance(cfg.trading.start_equity, float)


def test_kind_dispatch():
    assert config_kind(S1) == config_kind(S2) == "stocks"
    assert config_kind(NEWS) == "news"
    assert isinstance(load_v2_config(S2), StockConfig)
    assert isinstance(load_v2_config(NEWS), NewsConfig)
    with pytest.raises(ConfigError, match="not a version 2 config"):
        config_kind(ROOT / "configs" / "run_a.toml")


def test_each_loader_refuses_the_other_kind():
    with pytest.raises(ConfigError, match="found a news config"):
        load_stock_config(NEWS)
    with pytest.raises(ConfigError, match="found a stock run config"):
        load_news_config(S1)


def test_unreadable_and_malformed_files(tmp_path):
    with pytest.raises(ConfigError, match="cannot read config file"):
        load_stock_config(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("[run\nkind = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_news_config(bad)


# --- stock run cross checks ---

STOCK_CASES = [
    ("kind news", _set("run", "kind", "news"), 'needs [run] kind = "stocks"'),
    ("kind missing", _drop("run", "kind"), 'needs [run] kind = "stocks"'),
    ("unknown section", lambda d: d.update({"dashboard": {"host": "0.0.0.0"}}), "unknown section [dashboard]"),
    ("unknown key", _set("run", "colour", "red"), "[run] colour: unknown key"),
    ("missing key", _drop("stock_feed", "index_symbol"), "[stock_feed] index_symbol: missing"),
    ("missing section", _drop("session"), "missing section [session]"),
    ("float for int", _set("time", "horizon_minutes", 30.5), "[time] horizon_minutes = 30.5: expected an integer"),
    ("string for float", _set("trading", "notional", "500"), "[trading] notional = '500': expected a number"),
    ("int for bool", _set("trading", "allow_short", 1), "[trading] allow_short = 1: expected true or false"),
    ("out of range names the range", _set("model", "adam_beta1", 1.0),
     "[model] adam_beta1 = 1.0: out of range, allowed [0, 1)"),
    ("log level", _set("run", "log_level", "LOUD"), "[run] log_level = 'LOUD': must be one of"),
    ("run name", _set("run", "name", "S1 Prices"), "[run] name = 'S1 Prices': expected lower case"),
    ("provider", _set("stock_feed", "provider", "coinbase"),
     "[stock_feed] provider = 'coinbase': must be 'finnhub', the only accepted value"),
    ("inputs", _set("model", "inputs", 11), "[model] inputs = 11: out of range, allowed [12, 12]"),
    ("symbols unique", _set("stock_feed", "symbols", ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
                                                      "QQQ", "AAPL"]), "lists a value twice"),
    ("symbols upper case", _many(_set("stock_feed", "symbols", ["aapl", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
                                                                "TSLA", "QQQ"]),
                                 _set("stock_costs", "half_spread_bps", {"aapl": 1.0, "MSFT": 1.0, "NVDA": 1.0,
                                                                         "AMZN": 1.0, "GOOGL": 1.0, "META": 2.0,
                                                                         "TSLA": 2.0, "QQQ": 0.5})),
     "[stock_feed] symbols = 'aapl' must be an upper case ticker"),
    ("at most 50 symbols", _fifty_one_symbols, "lists 51 symbols, at most 50"),
    ("index in symbols", _set("stock_feed", "index_symbol", "SPY"),
     "[stock_feed] index_symbol = 'SPY' must be one of [stock_feed] symbols"),
    ("early close first open", _set("session", "first_open_session_minute", 176),
     "is 211, which must fit in an early close of 210 minutes"),
    ("early close horizon", _set("time", "horizon_minutes", 180),
     "is 215, which must fit in an early close of 210 minutes"),
    ("early close flatten", _set("session", "flatten_minutes_before_close", 151),
     "is 211, which must fit in an early close of 210 minutes"),
    ("symbol stale below feed stale", _set("stock_feed", "symbol_stale_after_seconds", 4),
     "symbol_stale_after_seconds must be at least feed_stale_after_seconds"),
    ("print delay below 2", _many(_set("stock_feed", "max_print_delay_seconds", 1.9),
                                  _set("stock_feed", "late_print_seconds", 3.8)),
     "[stock_feed] max_print_delay_seconds = 1.9: out of range, allowed [2, 5]"),
    ("print delay above 5", _many(_set("stock_feed", "max_print_delay_seconds", 5.5),
                                  _set("stock_feed", "late_print_seconds", 11.0)),
     "[stock_feed] max_print_delay_seconds = 5.5: out of range, allowed [2, 5]"),
    ("late print not twice", _set("stock_feed", "late_print_seconds", 5),
     "late_print_seconds = 5.0 must be twice max_print_delay_seconds, 4.0"),
    ("S1 sets use_news", _set("news_input", "use_news", True), "only S2 sets use_news"),
    ("S1 with evaluation", lambda d: d.update({"evaluation": raw(S2)["evaluation"]}), "belongs to S2 only"),
    ("spread for a symbol outside the universe", _set("stock_costs", "half_spread_bps",
                                                      {**raw(S1)["stock_costs"]["half_spread_bps"], "IBM": 1.0}),
     "half_spread_bps names IBM, which is not in [stock_feed] symbols"),
    ("spread missing for a symbol", _set("stock_costs", "half_spread_bps",
                                         {k: v for k, v in raw(S1)["stock_costs"]["half_spread_bps"].items()
                                          if k != "TSLA"}),
     "half_spread_bps has no allowance for TSLA"),
    ("keep candles 119 days", _set("persistence", "keep_candles_days", 119),
     "keep_candles_days = 119 must be 0, kept for the life of the run, or at least 120"),
    ("keep candles negative", _set("persistence", "keep_candles_days", -1), "out of range, allowed [0, 3650]"),
    ("checkpoint not whole minutes", _set("persistence", "checkpoint_every_seconds", 90), "whole number of minutes"),
    ("backoff max below start", _set("stock_feed", "reconnect_backoff_max_seconds", 4),
     "reconnect_backoff_max_seconds must be at least reconnect_backoff_start_seconds"),
    ("three return windows", _set("features", "return_windows_minutes", [30, 60, 120]), "exactly 2 windows"),
    ("history too short", _set("time", "history_session_minutes", 390),
     "history_session_minutes must be at least 391"),
    ("capacity below history", _set("features", "series_capacity_minutes", 300),
     "series_capacity_minutes must be at least [time] history_session_minutes"),
    ("notional above equity", _set("trading", "notional", 20000.0), "notional must not exceed start_equity"),
]


@pytest.mark.parametrize("name, change, message", STOCK_CASES, ids=[c[0] for c in STOCK_CASES])
def test_stock_config_rejects(name, change, message):
    with pytest.raises(ConfigError) as info:
        stock(changed(S1, change))
    assert message in str(info.value)


S2_CASES = [
    ("S2 without evaluation", _drop("evaluation"), "needs the [evaluation] section"),
    ("unknown confirmatory predictor", _set("evaluation", "confirmatory_predictor", "vibes"),
     "[evaluation] confirmatory_predictor = 'vibes': must be one of"),
    ("unknown run minute rule", _set("evaluation", "run_minute_rule", "any_minute"), "must be"),
    ("empty interim sessions", _set("evaluation", "interim_sessions", []), "expected a non empty list"),
]


@pytest.mark.parametrize("name, change, message", S2_CASES, ids=[c[0] for c in S2_CASES])
def test_s2_config_rejects(name, change, message):
    with pytest.raises(ConfigError) as info:
        stock(changed(S2, change))
    assert message in str(info.value)


@pytest.mark.parametrize("change", [
    _set("session", "first_open_session_minute", 175),
    _many(_set("time", "horizon_minutes", 175), _set("session", "first_open_session_minute", 30)),
    _set("stock_feed", "symbol_stale_after_seconds", 5),
    _many(_set("stock_feed", "max_print_delay_seconds", 5.0), _set("stock_feed", "late_print_seconds", 10.0)),
    _many(_set("stock_feed", "max_print_delay_seconds", 3), _set("stock_feed", "late_print_seconds", 6)),
    _set("persistence", "keep_candles_days", 120),
    _set("persistence", "keep_candles_days", 0),
    _fifty_symbols,
    _set("time", "warmup_hours", 0.5),
    _set("trading", "force_trade_after_hours", 0.5),
], ids=["first open at the edge", "horizon at the edge", "symbol stale equal", "print delay 5", "print delay 3",
        "keep candles 120", "keep candles 0", "50 symbols", "rehearsal warmup", "rehearsal forced timer"])
def test_stock_config_accepts_edges(change):
    stock(changed(S1, change))


def test_every_problem_is_reported_together():
    data = changed(S1, _many(_set("run", "colour", "red"), _drop("stock_feed", "provider")))
    with pytest.raises(ConfigError) as info:
        stock(data)
    assert "colour: unknown key" in str(info.value) and "provider: missing" in str(info.value)


# --- the S1 and S2 pair ---

def test_pair_refuses_swapped_runs():
    with pytest.raises(ConfigError) as info:
        load_run_pair(S2, S1)
    assert "S1 may not set [news_input] use_news" in str(info.value)
    assert "S2 must set [news_input] use_news" in str(info.value)


def test_pair_refuses_any_other_difference():
    s1 = load_stock_config(S1)
    for section, key, value in [("model", "hidden_units", 32), ("stock_feed", "symbol_stale_after_seconds", 20),
                                ("run", "log_level", "DEBUG"), ("run", "master_seed", 1),
                                ("trading", "margin_sigma", 0.5), ("news_input", "half_life_session_minutes", 60)]:
        s2 = stock(changed(S2, _set(section, key, value)))
        with pytest.raises(ConfigError) as info:
            check_run_pair(s1, s2)
        assert f"[{section}] {key} differs between S1 and S2" in str(info.value)


def test_pair_allows_name_data_dir_use_news_and_evaluation_only():
    s1 = load_stock_config(S1)
    s2 = stock(changed(S2, _many(_set("evaluation", "newey_west_lags", 6), _set("run", "data_dir", "elsewhere"))))
    check_run_pair(s1, s2)


def test_pair_needs_distinct_names_and_folders():
    s1 = load_stock_config(S1)
    s2 = stock(changed(S2, _many(_set("run", "name", "s1_prices"), _set("run", "data_dir", s1.run.data_dir))))
    with pytest.raises(ConfigError) as info:
        check_run_pair(s1, s2)
    assert "different [run] name" in str(info.value) and "different [run] data_dir" in str(info.value)


def test_pair_refuses_configs_in_different_folders(tmp_path):
    s1 = load_stock_config(S1)
    copy_of_s2 = tmp_path / "s2_news.toml"
    copy_of_s2.write_text(S2.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        check_run_pair(s1, load_stock_config(copy_of_s2))
    assert "must sit in one folder" in str(info.value)
    assert "differs between S1 and S2" not in str(info.value)


# --- the news config cross checks ---

NEWS_CASES = [
    ("kind stocks", _set("news", "kind", "stocks"), 'needs [news] kind = "news"'),
    ("unknown section", lambda d: d.update({"dashboard": {"host": "0.0.0.0", "port": 8060}}),
     "unknown section [dashboard]"),
    ("unknown key", _set("finnhub", "token", "x"), "[finnhub] token: unknown key"),
    ("unknown subtable", _set("llm", "opus", {"model": OPUS}), "[llm] opus: unknown key"),
    ("missing subtable", lambda d: d["llm"].pop("caps"), "missing section [llm.caps]"),
    ("no daily cap", _set("llm.caps", "daily_usd", 5.0), "[llm.caps] daily_usd: unknown key"),
    ("monthly cap zero", _set("llm.caps", "monthly_usd", 0.0),
     "[llm.caps] monthly_usd = 0.0: out of range, allowed (0, 20.0]"),
    ("monthly cap above the workspace limit", _set("llm.caps", "monthly_usd", 20.01),
     "[llm.caps] monthly_usd = 20.01: out of range, allowed (0, 20.0]"),
    ("daily alert share zero", _set("llm.caps", "daily_alert_share", 0.0), "out of range, allowed (0, 1)"),
    ("daily alert share one", _set("llm.caps", "daily_alert_share", 1.0), "out of range, allowed (0, 1)"),
    ("sonnet on media", _set("llm.sonnet", "kinds", ["8k", "media"]),
     "[llm.sonnet] kinds = ['8k', 'media']: 'media' is not allowed"),
    ("sonnet on amendments", _set("llm.sonnet", "kinds", ["8k_a"]), "'8k_a' is not allowed"),
    ("haiku as opus", _set("llm", "model", OPUS),
     f"[llm] model = '{OPUS}': must be '{HAIKU_MODEL}', the only accepted value"),
    ("haiku as sonnet", _set("llm", "model", SONNET_MODEL), f"must be '{HAIKU_MODEL}'"),
    ("sonnet as opus", _set("llm.sonnet", "model", OPUS),
     f"[llm.sonnet] model = '{OPUS}': must be '{SONNET_MODEL}', the only accepted value"),
    ("sonnet as haiku", _set("llm.sonnet", "model", HAIKU_MODEL), f"must be '{SONNET_MODEL}'"),
    ("haiku effort", _set("llm", "effort", "high"), "[llm] effort = 'high': must be 'low'"),
    ("haiku thinking", _set("llm", "thinking", "enabled"), "[llm] thinking = 'enabled': must be 'disabled'"),
    ("sonnet effort", _set("llm.sonnet", "effort", "medium"), "[llm.sonnet] effort = 'medium': must be 'low'"),
    ("sonnet thinking", _set("llm.sonnet", "thinking", "adaptive"), "must be 'disabled'"),
    ("same scorer names", _set("llm.sonnet", "scorer_name", "haiku_v1"), "scorer_name must differ"),
    ("send_to_llm type", _set("finnhub", "send_to_llm", "yes"), "[finnhub] send_to_llm = 'yes': expected true"),
    ("universe duplicate", _set("news", "universe", ["AAPL", "AAPL"]), "lists a value twice"),
    ("universe lower case", _set("news", "universe", ["AAPL", "msft", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
                                                      "QQQ"]), "[news] universe = 'msft' must be an upper case"),
    ("company outside universe", _set("finnhub", "company_symbols", ["AAPL", "IBM"]),
     "[finnhub] company_symbols names IBM, which is not in [news] universe"),
    ("cik outside universe", _set("edgar", "ciks", {"AAPL": 320193, "IBM": 51143}),
     "[edgar] ciks names IBM, which is not in [news] universe"),
    ("unknown news kind", _set("news", "backlog_exempt_kinds", ["10q"]), "'10q' is not allowed"),
    ("unknown edgar form", _set("edgar", "forms", ["8-K", "10-Q"]), "'10-Q' is not allowed"),
    ("unknown fed feed", _set("fed", "feeds", ["press_all", "blog"]), "'blog' is not allowed"),
    ("unknown bls feed", _set("bls", "feeds", ["cpi", "gdp"]), "'gdp' is not allowed"),
    ("window reversed", _set("finnhub", "regular_window_et", ["16:00", "09:30"]),
     "the start must come before the end"),
    ("window format", _set("edgar", "filing_window_et", ["6:00", "22:00"]), "expected a time as HH:MM"),
    ("probe time", _set("llm", "probe_time_utc", "24:00"), "expected a time as HH:MM"),
    ("calendar bad date", _set("calendar", "cpi", ["2026-13-01T08:30"]), "is not a YYYY-MM-DDTHH:MM time"),
    ("calendar out of order", _set("calendar", "fomc_statement", ["2026-12-09T14:00", "2026-10-28T14:00"]),
     "ascending order without repeats"),
    ("calendar repeat", _set("calendar", "ppi", ["2026-11-13T08:30", "2026-11-13T08:30"]),
     "ascending order without repeats"),
    ("prompt hash", _set("llm", "prompt_sha256", "abc"), "64 lower case hex digits"),
    ("prompt file with a folder", _set("llm", "prompt_file", "../prompt_v1.txt"), "plain file name"),
    ("macro pattern", _set("prefilter", "macro_strip_pattern", "(NASDAQ"), "macro_strip_pattern does not compile"),
    ("ids above i25", _set("llm", "max_items_per_call", 26), "out of range, allowed [1, 25]"),
    ("input guard at the long prompt line", _set("llm", "max_input_tokens", 100000), "out of range"),
    ("time zone", _set("news", "schedule_tz", "UTC"), "must be 'America/New_York'"),
]


@pytest.mark.parametrize("name, change, message", NEWS_CASES, ids=[c[0] for c in NEWS_CASES])
def test_news_config_rejects(name, change, message):
    with pytest.raises(ConfigError) as info:
        news(changed(NEWS, change))
    assert message in str(info.value)


@pytest.mark.parametrize("change", [
    _set("llm.caps", "monthly_usd", 20),
    _set("llm.caps", "monthly_usd", 0.01),
    _set("llm.caps", "daily_alert_share", 0.5),
    _set("llm.sonnet", "kinds", ["8k"]),
    _set("finnhub", "send_to_llm", False),
    _set("llm", "prompt_sha256", "0" * 64),
    _set("calendar", "cpi", ["2026-11-12T08:30", "2026-12-10T08:30"]),
    _many(*[_set(s, "enabled", False) for s in ("finnhub", "edgar", "fed", "bls", "llm.sonnet")],
          _set("news", "data_dir", "../../data/v2/rehearsal/cap_drill"), _set("llm.caps", "monthly_usd", 0.01)),
], ids=["cap 20", "cap one cent", "alert share half", "sonnet one kind", "send_to_llm false", "pinned prompt hash",
        "cpi dates", "cap drill"])
def test_news_config_accepts(change):
    news(changed(NEWS, change))


def test_llm_subtables_load_nested():
    cfg = news(raw(NEWS))
    assert cfg.llm.sonnet.max_tokens == 1024
    assert cfg.llm.caps.reserve_overhead_tokens == 1000
    assert cfg.llm.prices.sonnet_output_per_mtok == 10.0
    assert cfg.calendar.fomc_statement == ("2026-10-28T14:00", "2026-12-09T14:00", "2027-01-27T14:00",
                                           "2027-03-17T14:00")
    assert cfg.finnhub.regular_window_et == ("09:30", "16:00")


# --- hashes ---

def test_experiment_hash_ignores_comments_layout_and_log_level():
    committed = load_stock_config(S1)
    assert re.fullmatch(r"[0-9a-f]{64}", committed.experiment_hash)
    assert stock(raw(S1)).experiment_hash == committed.experiment_hash
    assert stock(changed(S1, _set("trading", "start_equity", 10000))).experiment_hash == committed.experiment_hash
    assert stock(changed(S1, _set("run", "log_level", "DEBUG"))).experiment_hash == committed.experiment_hash


@pytest.mark.parametrize("section, key, value", [
    ("run", "name", "s1_other"), ("run", "data_dir", "../../data/v2/other"), ("run", "master_seed", 1),
    ("run", "secrets_file", "/elsewhere/secrets.env"),
    ("stock_feed", "price_excluded_conditions", ["12"]), ("session", "calendar_warn_days", 30),
    ("news_input", "poll_seconds", 2), ("time", "warmup_hours", 5), ("features", "clip_z", 4.0),
    ("model", "learning_rate", 0.0003), ("trading", "margin_sigma", 0.3), ("stock_costs", "latency_ms", 900),
    ("population", "size", 16), ("controls", "coin_flip_twins", False), ("persistence", "keep_checkpoints", 4),
])
def test_experiment_hash_covers_every_other_setting(section, key, value):
    before = load_stock_config(S1).experiment_hash
    assert stock(changed(S1, _set(section, key, value))).experiment_hash != before


def test_experiment_hash_covers_evaluation_and_differs_between_runs():
    s1, s2 = load_run_pair(S1, S2)
    assert s1.experiment_hash != s2.experiment_hash
    assert stock(changed(S2, _set("evaluation", "newey_west_lags", 6))).experiment_hash != s2.experiment_hash


def test_news_config_hash_is_stable_and_covers_every_section():
    committed = load_news_config(NEWS)
    assert re.fullmatch(r"[0-9a-f]{64}", committed.config_hash)
    assert news(raw(NEWS)).config_hash == committed.config_hash
    for change in [_set("calendar", "cpi", ["2026-11-12T08:30"]), _set("llm.caps", "monthly_usd", 19.0),
                   _set("llm.prices", "haiku_input_per_mtok", 0.2), _set("llm.sonnet", "max_tokens", 2048),
                   _set("finnhub", "send_to_llm", False), _set("finbert", "threads", 2),
                   _set("news", "keep_polls_days", 7), _set("dedupe", "near_dup_jaccard", 0.9),
                   _set("prefilter", "alias_excludes_macro", False), _set("edgar", "exhibit_max_chars", 1000),
                   _set("fed", "press_poll_seconds", 30), _set("bls", "poll_seconds", 900),
                   _set("llm", "prompt_sha256", "0" * 64)]:
        assert news(changed(NEWS, change)).config_hash != committed.config_hash
