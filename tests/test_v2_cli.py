"""The btc_survivor_v2 command line, with fixed listings for check-local, a fake secrets file for check-secrets and the
news dry run, test folders for init-db, news-status and the stock commands and synthetic recordings for tape-check and
replay, so no test reads this machine's sockets, the real secrets file, data/v2 or the network."""

import argparse
import os
import re
import sqlite3
import stat
import subprocess
import sys
from datetime import date

import pytest

from btc_survivor import __version__
from btc_survivor.replay import NEVER_FOR_SELECTION
from btc_survivor.v2 import dashboard, evaluate, export, localcheck, sessions, status
from btc_survivor.v2.__main__ import build_parser, main
from btc_survivor.v2.config import load_news_config
from btc_survivor.v2.local_secrets import CONTACT, DEFAULT_PATH, FINNHUB, REFUSED
from btc_survivor.v2.news.schedule import Schedule
from btc_survivor.v2.stocks import process as stock_process
from btc_survivor.v2.stocks import replay as stock_replay
from btc_survivor.v2.stocks import synthetic, tapecheck
from helpers import ROOT
from test_v2_group import CANARY, FAST, THU, at, raw_tape, write_pair
from v2_feedkit import script_env

SECTION_3 = ["init-db", "stocks", "news", "news-status", "tape-check", "evaluate", "status", "export", "replay",
             "dashboard", "check-secrets", "check-local"]

# Each subcommand of Phases 16 and 17 with arguments its own module parses, and the module whose run it reaches.
LATER = [
    (["stocks", "s1.toml", "s2.toml", "--dry-run"], stock_process),
    (["evaluate", "s1.toml", "s2.toml", "--detail"], evaluate),
    (["status", "s1.toml", "s2.toml", "--detail"], status),
    (["export", "s1.toml"], export),
    (["replay", "s1.toml", "one.jsonl", "two.jsonl", "--out", "replayed", "--detail"], stock_replay),
    (["dashboard", "s1.toml", "s2.toml", "--port", "8061"], dashboard),
]
MODEL_RUNTIMES = ("anthropic", "onnxruntime", "tokenizers", "pydantic", "torch")

LOOPBACK_LSOF = "p501\ncPython\nf7\nn127.0.0.1:8050\np502\ncPython\nf9\nn127.0.0.1:8060\n"
EXPOSED_LSOF = LOOPBACK_LSOF + "p777\ncnc\nf3\nn0.0.0.0:8060\n"
QUIET_PS = "    1 /sbin/launchd\n  501 python -m btc_survivor dashboard\n"
TOKEN = "fake-tunnel-token-canary"
TUNNEL_PS = QUIET_PS + f"  9123 cloudflared tunnel run --token {TOKEN}\n"
LOOPBACK_NETSTAT = ("tcp4       0      0  127.0.0.1.8050         *.*                    LISTEN\n"
                    "tcp4       0      0  127.0.0.1.8060         *.*                    LISTEN\n")
OTHER_USER_NETSTAT = LOOPBACK_NETSTAT + "tcp46      0      0  *.8050                 *.*                    LISTEN\n"

FAKE = {"FINNHUB_API_KEY": "fake-finnhub-canary", "ANTHROPIC_API_KEY": "fake-anthropic-canary",
        "CONTACT_EMAIL": "fake-contact-canary@example.invalid"}
NEWS = ROOT / "configs" / "v2" / "news.toml"
S1 = ROOT / "configs" / "v2" / "s1_prices.toml"


def subcommands():
    parser = build_parser()
    return next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices


def listings(monkeypatch, lsof, ps, sockets=LOOPBACK_NETSTAT):
    monkeypatch.setattr(localcheck, "read_listings", lambda run, find_tailscale: (lsof, ps, sockets, ()))


def write_secrets(path, names, mode=0o600):
    path.write_text("".join(f"{name}={FAKE[name]}\n" for name in names), encoding="utf-8")
    os.chmod(path, mode)
    return path


def test_every_subcommand_of_section_3_is_registered():
    assert list(subcommands()) == SECTION_3


def test_the_console_script_points_at_main():
    assert 'btc_survivor_v2 = "btc_survivor.v2.__main__:main"' in (ROOT / "pyproject.toml").read_text()


def test_the_package_data_ships_every_version_2_data_file():
    import tomllib
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"]["package-data"]
    shipped = {f"src/{package.replace('.', '/')}/{name}" for package, names in data.items()
               if package.startswith("btc_survivor.v2") for name in names}
    found = {p.relative_to(ROOT).as_posix() for p in (ROOT / "src" / "btc_survivor" / "v2").rglob("*")
             if p.is_file() and p.suffix not in (".py", ".pyc") and "__pycache__" not in p.parts}
    assert shipped == found


@pytest.mark.parametrize("argv, module", LATER, ids=[a[0] for a, _ in LATER])
def test_a_later_subcommand_reaches_its_module_with_the_arguments_that_module_parses(argv, module, monkeypatch):
    seen = []
    monkeypatch.setattr(module, "run", lambda args: seen.append(args) or 0)
    assert main(argv) == 0
    expected = vars(module.build_parser().parse_args(argv[1:]))
    assert len(seen) == 1 and {k: v for k, v in vars(seen[0]).items() if k in expected} == expected


def test_the_command_line_loads_no_model_runtime_so_the_stock_process_never_does():
    script = ("import sys\nimport btc_survivor.v2.__main__\n"
              f"print(','.join(name for name in {MODEL_RUNTIMES!r} if name in sys.modules))")
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert (done.returncode, done.stdout, done.stderr) == (0, "\n", "")


def test_replay_help_says_it_never_chooses_tunes_or_ranks(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["replay", "--help"])
    assert exit_info.value.code == 0
    assert NEVER_FOR_SELECTION in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert capsys.readouterr().out.strip() == f"btc_survivor_v2 {__version__}"


def test_check_local_passes_on_loopback_listeners(monkeypatch, capsys):
    listings(monkeypatch, LOOPBACK_LSOF, QUIET_PS)
    assert main(["check-local"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines() == ["local only check", "verdict: pass", "offending sockets: 0", "tunnel processes: 0",
                                "loopback listeners: 2"]


def test_check_local_fails_with_counts_only_by_default(monkeypatch, capsys):
    listings(monkeypatch, EXPOSED_LSOF, TUNNEL_PS)
    assert main(["check-local"]) == 1
    out = capsys.readouterr().out
    assert "verdict: fail" in out and "offending sockets: 1" in out and "tunnel processes: 1" in out
    assert "2 rows hidden" in out
    assert "777" not in out and "9123" not in out and "0.0.0.0" not in out and TOKEN not in out


def test_check_local_detail_names_each_problem_by_pid_and_never_the_command_line(monkeypatch, capsys):
    listings(monkeypatch, EXPOSED_LSOF, TUNNEL_PS)
    assert main(["check-local", "--detail"]) == 1
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("777"))
    assert "nc" in row and "0.0.0.0:8060" in row
    row = next(line for line in out.splitlines() if line.startswith("9123"))
    assert "cloudflared" in row and "tunnel" in row
    assert TOKEN not in out


def test_check_local_fails_on_another_users_socket_that_lsof_cannot_see(monkeypatch, capsys):
    listings(monkeypatch, LOOPBACK_LSOF, QUIET_PS, OTHER_USER_NETSTAT)
    assert main(["check-local"]) == 1
    out = capsys.readouterr().out
    assert "verdict: fail" in out and "offending sockets: 1" in out and "1 rows hidden" in out
    assert main(["check-local", "--detail"]) == 1
    assert any("any user" in line and "*:8050" in line for line in capsys.readouterr().out.splitlines())


def test_check_local_reports_an_unreadable_listing_as_an_error(monkeypatch, capsys):
    def broken(run, find_tailscale):
        raise localcheck.LocalCheckError("lsof failed with exit code 2")
    monkeypatch.setattr(localcheck, "read_listings", broken)
    assert main(["check-local"]) == 2
    assert capsys.readouterr().err == "error: lsof failed with exit code 2\n"


def test_check_secrets_defaults_to_the_owner_file():
    assert build_parser().parse_args(["check-secrets"]).secrets_file == DEFAULT_PATH


def test_check_secrets_names_the_three_keys_and_no_value(tmp_path, capsys):
    path = write_secrets(tmp_path / "secrets.env", FAKE)
    assert main(["check-secrets", "--secrets-file", str(path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("pass")
    for name in FAKE:
        assert f"present: {name}" in out
    assert not [name for name in FAKE if FAKE[name] in out]


def test_check_secrets_names_a_missing_key(tmp_path, capsys):
    path = write_secrets(tmp_path / "secrets.env", ["FINNHUB_API_KEY", "ANTHROPIC_API_KEY"])
    assert main(["check-secrets", "--secrets-file", str(path)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("fail") and "missing: CONTACT_EMAIL" in out
    assert not [name for name in FAKE if FAKE[name] in out]


def test_check_secrets_refuses_a_group_readable_file(tmp_path, capsys):
    path = write_secrets(tmp_path / "secrets.env", FAKE, mode=0o640)
    assert main(["check-secrets", "--secrets-file", str(path)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("fail") and "chmod 600" in out
    assert not [name for name in FAKE if FAKE[name] in out]


# The subcommands of Phases 13 and 14.


def mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def umask_now():
    current = os.umask(0o022)
    os.umask(current)
    return current


def news_copy(tmp_path, **paths):
    """The committed news config copied two folders below tmp_path, so its data_dir lands in tmp_path/data, with any
    [news] path keys set to the given relative paths."""
    text = NEWS.read_text(encoding="utf-8")
    for key, value in paths.items():
        text, found = re.subn(rf'(?m)^{key} = "[^"]*"', f'{key} = "{value}"', text, count=1)
        assert found == 1, key
    path = tmp_path / "configs" / "v2" / "news.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path, tmp_path / "data" / "v2" / "news" / "news.db"


def test_init_db_creates_news_db_owner_only_then_checks_it(tmp_path, capsys):
    config, path = news_copy(tmp_path)
    umask = umask_now()
    assert main(["init-db", str(config)]) == 0
    assert capsys.readouterr().out.splitlines() == ["init db", "verdict: pass", "kind: news", "database: created"]
    assert umask_now() == umask
    assert mode(path) == 0o600
    assert [mode(folder) for folder in (path.parent, path.parent.parent, tmp_path / "data")] == [0o700] * 3
    assert main(["init-db", str(config)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "database: checked"


def test_init_db_refuses_a_database_of_another_kind(tmp_path, capsys):
    config, path = news_copy(tmp_path)
    path.parent.mkdir(parents=True)
    other = sqlite3.connect(path)
    other.executescript("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
                        "INSERT INTO meta VALUES ('kind', 'stocks');")
    other.close()
    assert main(["init-db", str(config)]) == 2
    out, err = capsys.readouterr()
    assert out == "" and err.startswith("error: ") and "meta.kind is 'stocks'" in err


def stock_copy(tmp_path):
    """The committed S1 config copied two folders below tmp_path, so its data_dir lands in tmp_path/data."""
    path = tmp_path / "configs" / "v2" / "s1_prices.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(S1.read_text(encoding="utf-8"), encoding="utf-8")
    return path, tmp_path / "data" / "v2" / "s1_prices" / "stocks.db"


def test_init_db_creates_a_run_database_owner_only_then_checks_it(tmp_path, capsys):
    config, path = stock_copy(tmp_path)
    umask = umask_now()
    assert main(["init-db", str(config)]) == 0
    assert capsys.readouterr().out.splitlines() == ["init db", "verdict: pass", "kind: stocks", "database: created"]
    assert umask_now() == umask
    assert mode(path) == 0o600
    assert [mode(folder) for folder in (path.parent, path.parent.parent, tmp_path / "data")] == [0o700] * 3
    assert main(["init-db", str(config)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "database: checked"


def test_init_db_refuses_a_news_database_at_a_run_path_and_a_version_1_config(tmp_path, capsys):
    config, path = stock_copy(tmp_path)
    path.parent.mkdir(parents=True)
    other = sqlite3.connect(path)
    other.executescript("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
                        "INSERT INTO meta VALUES ('kind', 'news');")
    other.close()
    assert main(["init-db", str(config)]) == 2
    out, err = capsys.readouterr()
    assert out == "" and err.startswith("error: ") and "meta.kind is 'news', not 'stocks'" in err
    other = sqlite3.connect(path)
    assert other.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall() == [("meta",)]
    assert other.execute("SELECT key, value FROM meta").fetchall() == [("kind", "news")]
    other.close()
    assert main(["init-db", str(ROOT / "configs" / "run_a.toml")]) == 2
    assert "is not a version 2 config" in capsys.readouterr().err


def test_news_dry_run_runs_through_the_command_line_with_fake_secrets(tmp_path):
    config, path = news_copy(tmp_path, secrets_file="secrets.env", rest_lock_file="lock/finnhub_rest.lock")
    secrets = config.parent / "secrets.env"
    secrets.write_text(f"{FINNHUB}={FAKE[FINNHUB]}\n{CONTACT}={FAKE[CONTACT]}\n", encoding="utf-8")
    secrets.chmod(0o600)
    env = {name: value for name, value in os.environ.items() if name.upper() not in REFUSED}
    done = subprocess.run([sys.executable, "-m", "btc_survivor.v2", "news", str(config), "--dry-run"],
                          capture_output=True, text=True, cwd=tmp_path, env={**env, "HOME": str(tmp_path)},
                          timeout=60)
    channels = len(Schedule(load_news_config(config)).channels)
    assert (done.returncode, done.stderr) == (0, "")
    assert done.stdout.splitlines() == ["news dry run", "verdict: pass", f"channels: {channels}",
                                        "sources enabled: 4", "lock: taken and released",
                                        "scoring: waits for the finbert pins"]
    assert not path.parent.exists() and FAKE[FINNHUB] not in done.stdout and FAKE[CONTACT] not in done.stdout


def test_news_status_runs_through_the_command_line(tmp_path, capsys):
    config, _ = news_copy(tmp_path)
    assert main(["news-status", "--config", str(config)]) == 2
    assert "does not exist" in capsys.readouterr().err
    main(["init-db", str(config)])
    capsys.readouterr()
    assert main(["news-status", "--config", str(config), "--days", "3"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == ["news status", "days: 3"]
    assert "order verdict: pass" in lines and "backlog verdict: pass" in lines


def test_tape_check_runs_through_the_command_line_on_a_synthetic_recording(tmp_path, capsys):
    day = sessions.session_on(date(2026, 10, 13))
    start, end = tapecheck.window(day)
    tape = synthetic.SyntheticTape(31, regular_rate=0.05, extended_rate=0.005)
    synthetic.write_recording(tmp_path / "raw", tape.lines(start - 30 * 60_000, end + 30 * 60_000))
    symbols = synthetic.SYMBOLS
    official = tmp_path / "official_2026_10_13.toml"
    official.write_text("consolidated_label = true\n" + "".join(
        f"\n[symbols.{s}]\nvolume = {tape.truth[(day.day, s)] * 25}\nclose = 123.25\n" for s in symbols),
        encoding="utf-8")
    assert main(["tape-check", str(tmp_path / "raw"), str(official), "--config", str(S1)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == ["tape coverage", "clean session: pass", "reason: a full session connected throughout"]
    assert "delay verdict: pass" in lines and "outage time: 0.000 s" in lines
    assert lines[-1] == "9 rows hidden, pass --detail in your own terminal to see them"
    assert [r.median_ratio for r in tapecheck.read_records(tmp_path)] == [pytest.approx(0.04)]
    assert main(["tape-check", str(tmp_path / "missing"), str(official), "--config", str(S1)]) == 2
    assert capsys.readouterr().err.startswith("error: no recording files")


# The subcommands of Phases 16 and 17.


def test_stocks_dry_run_through_the_command_line(tmp_path):
    s1, s2 = write_pair(tmp_path, **FAST)
    done = subprocess.run([sys.executable, "-m", "btc_survivor.v2", "stocks", str(s1), str(s2), "--dry-run",
                           "--test-url", "ws://127.0.0.1:9"], capture_output=True, text=True, timeout=60,
                          cwd=tmp_path, env=script_env(tmp_path))
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == ["stocks dry run", "verdict: pass", "runs: 2", "symbols: 8",
                                        "lock: taken and released"]
    assert CANARY not in done.stdout + done.stderr and not (tmp_path / "data").exists()


def test_replay_through_the_command_line(tmp_path, capsys):
    _, s2 = write_pair(tmp_path, **FAST)
    files = synthetic.write_recording(tmp_path / "rec", raw_tape(at(THU, "09:20"), at(THU, "09:50")))
    assert main(["replay", str(s2), *map(str, files), "--out", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[:2] == ["stock replay", "verdict: pass"] and out[-1] == NEVER_FOR_SELECTION
    assert main(["replay", str(s2), *map(str, files), "--out", str(tmp_path / "out")]) == 2
    assert "not empty" in capsys.readouterr().err


def test_status_export_and_dashboard_refuse_missing_runs_and_the_version_1_port(tmp_path, capsys):
    import v2_runkit as kit
    s1, s2, _ = kit.configs(tmp_path)
    for argv in (["status", s1, s2], ["export", s1]):
        assert main([str(a) for a in argv]) == 2
        assert "stocks.db does not exist" in capsys.readouterr().err
    assert main(["dashboard", str(s1), str(s2), "--port", "8050"]) == 2
    assert "port 8050 belongs to the version 1 dashboard" in capsys.readouterr().err


def test_evaluate_refuses_a_version_1_config(capsys):
    assert main(["evaluate", str(ROOT / "configs" / "run_a.toml"), str(S1)]) == 2
    assert capsys.readouterr().err.startswith("error: ")
