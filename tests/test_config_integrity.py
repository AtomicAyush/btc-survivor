"""The config integrity check against the text a run stores at launch, on a replayed run and on the real configs."""

import sys

import pytest

from btc_survivor.cli import data
from btc_survivor.config import load_config
from btc_survivor.replay import replay_lines
from enginekit import synthetic
from helpers import ROOT, dump_toml, example_dict

sys.path.insert(0, str(ROOT / "scripts"))
import check_config_integrity as integrity  # noqa: E402


@pytest.fixture
def launched(tmp_path):
    """A config file and a short replay of it in copies/<run folder>, standing in for a launched run."""
    settings = example_dict()
    settings["run"]["data_dir"] = str(tmp_path / "live" / "run_x")
    config = tmp_path / "run_x.toml"
    config.write_text(dump_toml(settings))
    copies = tmp_path / "copies"
    replay_lines(load_config(config), synthetic(1, 120), copies / "run_x")
    return config, copies


def test_the_engine_stores_the_config_text_in_meta_at_launch(launched):
    config, copies = launched
    conn = data.connect(copies / "run_x" / "btc_survivor.db")
    try:
        assert data.meta_value(conn, "config_toml") == config.read_text()
        assert data.meta_value(conn, "master_seed") == str(load_config(config).run.master_seed)
        assert "config_toml" not in conn.execute("SELECT meta FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()[0]
    finally:
        conn.close()


def test_an_unchanged_config_matches(launched, capsys):
    config, copies = launched
    assert integrity.check(config, copies)["verdict"] == "match"
    assert integrity.main([str(config), "--copies", str(copies)]) == 0
    assert "match, the file is the text the run stored at launch" in capsys.readouterr().out


def test_comments_and_logging_do_not_count_as_a_changed_experiment(launched):
    config, copies = launched
    config.write_text("# a note added after launch\n" + config.read_text())
    assert integrity.check(config, copies)["verdict"] == "same settings"
    config.write_text(config.read_text().replace('log_level = "INFO"', 'log_level = "DEBUG"'))
    result = integrity.check(config, copies)
    assert result["verdict"] == "logging only" and [d[0] for d in result["differences"]] == ["run.log_level"]
    assert integrity.main([str(config), "--copies", str(copies)]) == 0


def test_a_changed_setting_is_named_with_both_values(launched, capsys):
    config, copies = launched
    settings = example_dict()
    settings["run"]["data_dir"] = str(config.parent / "live" / "run_x")
    settings["costs"]["taker_fee_bps_retail_default"] = 5.0
    settings["time"]["horizon_minutes"] = 15
    del settings["population"]["tournament_size"]
    config.write_text(dump_toml(settings))
    result = integrity.check(config, copies)
    assert result["verdict"] == "differs"
    keys = {key: (was, now) for key, was, now in result["differences"]}
    assert set(keys) == {"costs.taker_fee_bps_retail_default", "time.horizon_minutes", "population.tournament_size"}
    assert keys["costs.taker_fee_bps_retail_default"][1] == 5.0 and keys["time.horizon_minutes"] == (30, 15)
    assert keys["population.tournament_size"][1] is integrity.MISSING
    assert integrity.main([str(config), "--copies", str(copies)]) == 1
    out = capsys.readouterr().out
    assert "DIFFERS, 3 settings changed since launch" in out
    assert "time.horizon_minutes was 30 at launch, the file says 15" in out and "the file says absent" in out


def test_a_missing_database_is_reported(tmp_path, capsys):
    settings = example_dict()
    settings["run"]["data_dir"] = str(tmp_path / "never")
    config = tmp_path / "c.toml"
    config.write_text(dump_toml(settings))
    assert integrity.main([str(config)]) == 2
    assert "cannot check, no database" in capsys.readouterr().out


def test_the_check_only_reads(launched):
    """The database and its write ahead log are untouched. A read only reader of a WAL database may still update
    the shared memory index, as every reader does."""
    config, copies = launched
    folder = copies / "run_x"

    def files():
        return {p.name: (p.stat().st_size, p.stat().st_mtime_ns, p.read_bytes()) for p in folder.iterdir()
                if not p.name.endswith("-shm")}

    before = files()
    integrity.check(config, copies)
    assert files() == before


def test_a_relative_data_folder_resolves_from_the_config_folder(tmp_path):
    config = tmp_path / "configs" / "run_y.toml"
    found = integrity.database(config, '[run]\ndata_dir = "../data/run_y"\n', None)
    assert found.resolve() == (tmp_path / "data" / "run_y" / "btc_survivor.db").resolve()
    assert integrity.database(config, '[run]\ndata_dir = "../data/run_y"\n', tmp_path / "copies") == \
        tmp_path / "copies" / "run_y" / "btc_survivor.db"


@pytest.mark.parametrize("name", ["run_a", "run_b", "run_c", "run_d", "run_e"])
def test_the_run_configs_flatten_to_every_setting(name):
    """Only the files are read here, never a run's database."""
    import tomllib

    text = (ROOT / "configs" / f"{name}.toml").read_text()
    flat = integrity.flatten(tomllib.loads(text))
    assert flat["run.master_seed"] == 12345 and "costs.taker_fee_bps_retail_default" in flat
    assert integrity.differences(text, text) == []
