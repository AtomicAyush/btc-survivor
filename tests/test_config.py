import dataclasses
from pathlib import Path

import pytest

from btc_survivor.config import ConfigError, load_config, parse_config
from helpers import EXAMPLE_CONFIG, ROOT, dump_toml, example_dict, write_config


def parse(data, base=Path("/tmp/cfg")):
    return parse_config(dump_toml(data), base)


def test_example_config_loads_with_expected_values():
    cfg = load_config(EXAMPLE_CONFIG)
    assert cfg.costs.fee_profile == "retail_default"
    assert cfg.costs.taker_fee_bps == 90.0
    assert cfg.costs.taker_fees_bps == {"retail_default": 90.0, "active_retail": 30.0, "low_fee_tier": 5.0}
    assert cfg.time.horizon_minutes == 30
    assert cfg.costs.daily_running_cost == 1.16
    assert cfg.features.return_windows_minutes == (5, 15, 30, 60)
    assert cfg.features.vol_windows_minutes == (30, 120)
    assert cfg.population.size == 20
    assert cfg.data_dir == ROOT / "data"
    assert cfg.db_path == ROOT / "data" / "btc_survivor.db"
    assert "[run]" in cfg.text


def test_integers_are_accepted_for_float_keys():
    data = example_dict()
    data["trading"]["start_equity"] = 10000
    cfg = parse(data)
    assert cfg.trading.start_equity == 10000.0
    assert isinstance(cfg.trading.start_equity, float)


def test_fee_profile_selects_rate():
    data = example_dict()
    data["costs"]["fee_profile"] = "low_fee_tier"
    assert parse(data).costs.taker_fee_bps == 5.0


def test_config_is_frozen():
    cfg = load_config(EXAMPLE_CONFIG)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.trading.notional = 1.0


def _set(section, key, value):
    def change(data):
        data[section][key] = value
    return change


def _delete(section, key=None):
    def change(data):
        if key is None:
            del data[section]
        else:
            del data[section][key]
    return change


def _many(*changes):
    def change(data):
        for c in changes:
            c(data)
    return change


CASES = [
    ("unknown section", lambda d: d.update({"extra": {"a": 1}}), "unknown section [extra]"),
    ("unknown key", _set("run", "colour", "red"), "[run] colour: unknown key"),
    ("missing key", _delete("feed", "product"), "[feed] product: missing"),
    ("missing section", _delete("genes"), "missing section [genes]"),
    ("bool for int", _set("feed", "book_depth_levels", True), "[feed] book_depth_levels: expected an integer"),
    ("float for int", _set("time", "horizon_minutes", 30.5), "[time] horizon_minutes: expected an integer"),
    ("string for float", _set("trading", "notional", "500"), "[trading] notional: expected a number"),
    ("int for bool", _set("trading", "allow_short", 1), "[trading] allow_short: expected true or false"),
    ("empty string", _set("feed", "product", ""), "[feed] product: expected a non empty string"),
    ("out of range", _set("feed", "stale_after_seconds", 0), "[feed] stale_after_seconds = 0.0 is out of range"),
    ("negative fee", _set("costs", "taker_fee_bps_retail_default", -1.0), "[costs] taker_fee_bps_retail_default = -1.0 is out of range"),
    ("list of non ints", _set("features", "return_windows_minutes", [5, "x", 30, 60]), "expected a non empty list of integers"),
    ("empty window list", _set("features", "vol_windows_minutes", []), "expected a non empty list of integers"),
    ("three return windows", _set("features", "return_windows_minutes", [5, 15, 30]), "exactly 4 windows"),
    ("one vol window", _set("features", "vol_windows_minutes", [30]), "exactly 2 windows"),
    ("notional above equity", _set("trading", "notional", 20000.0), "notional must not exceed start_equity"),
    ("imbalance deeper than book", _set("features", "imbalance_levels", 25), "imbalance_levels must not exceed"),
    ("history too short", _set("time", "history_minutes_required", 200), "history_minutes_required must be at least 360"),
    ("unknown fee profile", _set("costs", "fee_profile", "vip"), "fee_profile = 'vip' has no taker_fee_bps_vip key"),
    ("latency too long", _many(_set("costs", "latency_ms", 29900), _set("costs", "latency_jitter_ms", 200)), "must stay at or under 30000"),
    ("margin init outside bounds", _set("genes", "margin_bps_init", 100.0), "margin_bps_init must lie between"),
    ("lr min above max", _set("genes", "lr_min", 0.5), "lr_min must not exceed lr_max"),
    ("checkpoint not whole minutes", _set("persistence", "checkpoint_every_seconds", 90), "checkpoint_every_seconds must be a whole number of minutes"),
    ("candles kept too briefly", _many(_set("time", "history_minutes_required", 2000), _set("persistence", "keep_candles_days", 1)), "keep_candles_days must keep at least"),
    ("bad log level", _set("run", "log_level", "LOUD"), "log_level = 'LOUD' must be one of"),
    ("bad exchange name", _set("feed", "exchange", "Coin-Base"), "must be a lower case module name"),
    ("unknown exchange", _set("feed", "exchange", "kraken"), "has no adapter module"),
    ("feed helper as exchange", _set("feed", "exchange", "book"), "has no adapter module"),
    ("backoff max below start", _set("feed", "reconnect_backoff_max_seconds", 0.5), "reconnect_backoff_max_seconds must be at least"),
    ("beta of one", _set("model", "adam_beta2", 1.0), "[model] adam_beta2 = 1.0 is out of range"),
    ("resume gap under a minute", _set("persistence", "max_resume_gap_minutes", 0.5),
     "[persistence] max_resume_gap_minutes = 0.5 is out of range"),
]


@pytest.mark.parametrize("name,change,message", CASES, ids=[c[0] for c in CASES])
def test_invalid_config_is_rejected_with_clear_message(name, change, message):
    data = example_dict()
    change(data)
    with pytest.raises(ConfigError) as info:
        parse(data)
    assert message in str(info.value)


def test_all_problems_are_reported_together():
    data = example_dict()
    data["run"]["colour"] = "red"
    data["feed"]["book_depth_levels"] = "deep"
    with pytest.raises(ConfigError) as info:
        parse(data)
    text = str(info.value)
    assert "[run] colour: unknown key" in text
    assert "[feed] book_depth_levels: expected an integer" in text


def test_invalid_toml_is_rejected():
    with pytest.raises(ConfigError, match="not valid TOML"):
        parse_config("[run\nmaster_seed = ", Path("."))


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="cannot read config file"):
        load_config(tmp_path / "nope.toml")


def test_relative_data_dir_resolves_against_config_folder(tmp_path):
    path = write_config(tmp_path)
    assert load_config(path).data_dir == tmp_path.resolve() / "data"


def test_absolute_data_dir_is_kept(tmp_path):
    data = example_dict()
    data["run"]["data_dir"] = str(tmp_path / "elsewhere")
    path = write_config(tmp_path, data)
    assert load_config(path).data_dir == tmp_path / "elsewhere"
