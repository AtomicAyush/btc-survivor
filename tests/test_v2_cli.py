"""The btc_survivor_v2 command line, with fixed listings for check-local and a fake secrets file for check-secrets,
so no test reads this machine's sockets or the real secrets file."""

import argparse
import os

import pytest

from btc_survivor import __version__
from btc_survivor.replay import NEVER_FOR_SELECTION
from btc_survivor.v2 import localcheck
from btc_survivor.v2.__main__ import build_parser, main
from btc_survivor.v2.local_secrets import DEFAULT_PATH
from helpers import ROOT

SECTION_3 = ["init-db", "stocks", "news", "news-status", "tape-check", "evaluate", "status", "export", "replay",
             "dashboard", "check-secrets", "check-local"]

NOT_BUILT = [
    ["init-db", "configs/v2/news.toml"],
    ["stocks", "s1.toml", "s2.toml"],
    ["news", "configs/v2/news.toml"],
    ["news-status", "--detail", "--samples", "50"],
    ["tape-check", "recording.jsonl", "official.csv"],
    ["evaluate", "s1.toml", "s2.toml"],
    ["status", "s1.toml", "s2.toml", "--detail"],
    ["export", "s1.toml"],
    ["replay", "s1.toml", "one.jsonl", "two.jsonl", "--out", "replayed"],
    ["dashboard", "s1.toml", "s2.toml"],
]

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


@pytest.mark.parametrize("argv", NOT_BUILT, ids=[a[0] for a in NOT_BUILT])
def test_a_later_subcommand_fails_clearly_with_exit_code_2(argv, capsys):
    assert main(argv) == 2
    out, err = capsys.readouterr()
    assert out == "" and err == f"error: {argv[0]} is not built yet\n"


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
