import pytest

from btc_survivor.__main__ import build_parser, main
from btc_survivor.recorder import Recorder
from btc_survivor.replay import NEVER_FOR_SELECTION, ReplayError, replay, replay_lines
from btc_survivor.synthetic import Fault
from enginekit import deadly_config, digest, synthetic
from helpers import dump_toml, example_dict

FAULTS = (Fault(900, "gap"), Fault(1300, "crossed"), Fault(1700, "silence", 30), Fault(2100, "disconnect", 300))


def test_same_seed_and_recording_give_identical_databases(tmp_path):
    lines = synthetic(11, 5400, FAULTS)
    first = digest(replay_lines(deadly_config(tmp_path / "unused"), lines, tmp_path / "a"))
    second = digest(replay_lines(deadly_config(tmp_path / "unused"), lines, tmp_path / "b"))
    assert first == second
    other = digest(replay_lines(deadly_config(tmp_path / "unused"), synthetic(12, 5400, FAULTS), tmp_path / "c"))
    assert other["decisions"] != first["decisions"]


def test_replay_never_writes_runtime_status_and_trades(tmp_path):
    import sqlite3

    db = replay_lines(deadly_config(tmp_path / "unused"), synthetic(11, 5400, FAULTS), tmp_path / "out")
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT count(*) FROM runtime_status").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM trades").fetchone()[0] > 0
    assert conn.execute("SELECT count(*) FROM deaths").fetchone()[0] > 0
    conn.close()


def test_replay_from_recorded_files_matches_replay_from_lines(tmp_path):
    lines = synthetic(11, 5400, FAULTS)
    recorder = Recorder(tmp_path / "raw")
    for line in lines:
        recorder.write(*line)
    recorder.close()
    from_files = replay(deadly_config(tmp_path / "unused"), [tmp_path / "raw"], tmp_path / "a")
    from_lines = replay_lines(deadly_config(tmp_path / "unused"), lines, tmp_path / "b")
    assert digest(from_files) == digest(from_lines)


def test_replay_refuses_a_folder_that_is_not_empty_and_the_data_folder(tmp_path):
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "file").write_text("x")
    with pytest.raises(ReplayError, match="not empty"):
        replay_lines(deadly_config(tmp_path / "unused"), synthetic(11, 60), busy)
    cfg = deadly_config(tmp_path / "data")
    with pytest.raises(ReplayError, match="data folder"):
        replay_lines(cfg, synthetic(11, 60), tmp_path / "data")
    with pytest.raises(ReplayError, match="no lines"):
        replay_lines(cfg, [], tmp_path / "empty")


def test_command_line_replay_states_it_never_selects_strategies(tmp_path, capsys):
    rep = build_parser()._subparsers._group_actions[0].choices["replay"]
    assert NEVER_FOR_SELECTION in rep.format_help().replace("\n", " ").replace("  ", " ")
    recorder = Recorder(tmp_path / "raw")
    for line in synthetic(11, 300):
        recorder.write(*line)
    recorder.close()
    config = tmp_path / "config.toml"
    data = example_dict()
    data["run"]["data_dir"] = "data"
    config.write_text(dump_toml(data))
    assert main(["--config", str(config), "replay", "--out", str(tmp_path / "out"), str(tmp_path / "raw")]) == 0
    assert NEVER_FOR_SELECTION in capsys.readouterr().out
    assert (tmp_path / "out" / "btc_survivor.db").exists() and not (tmp_path / "data").exists()
    assert main(["--config", str(config), "replay", "--out", str(tmp_path / "out"), str(tmp_path / "raw")]) == 2
