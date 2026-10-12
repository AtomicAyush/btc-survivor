"""The launchd agents in deploy/macos and the systemd units in deploy/systemd run the committed configs with the
settings of PLAN_V2 sections 14.4 and 14.5, name no secret and no real user, and are installed only by hand."""

import plistlib
import re
import shlex
import subprocess
from pathlib import Path

import pytest

from btc_survivor.v2 import __main__ as cli
from btc_survivor.v2.config import load_news_config, load_run_pair
from btc_survivor.v2.local_secrets import REFUSED
from btc_survivor.v2.news import runner as news_runner
from btc_survivor.v2.stocks import process as stock_process
from helpers import ROOT

MACOS = ROOT / "deploy" / "macos"
SYSTEMD = ROOT / "deploy" / "systemd"
NEWS, STOCKS = "btc_survivor_v2_news", "btc_survivor_v2_stocks"
COMMANDS = {NEWS: ["news", "configs/v2/news.toml"],
            STOCKS: ["stocks", "configs/v2/s1_prices.toml", "configs/v2/s2_news.toml"]}
MAC_HOME = "/Users/CHANGE_ME/code/btc_survivor_v2"
PI_HOME = "/home/CHANGE_ME/btc_survivor_v2"
PLIST_KEYS = {"Label", "ProgramArguments", "WorkingDirectory", "EnvironmentVariables", "Umask", "RunAtLoad",
              "KeepAlive", "ThrottleInterval", "ExitTimeOut", "ProcessType", "StandardOutPath", "StandardErrorPath"}
SYSTEM_PATH = ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]  # git and caffeinate, ps, lsof and netstat
SINGLE = ("Description", "Type", "User", "WorkingDirectory", "UMask", "ExecStart", "Restart", "RestartSec",
          "KillSignal", "TimeoutStopSec", "WantedBy")
OPERATIONS = ROOT / "docs" / "v2_operations.md"
# The handler each agent's subcommand must reach.
BUILT = {"news": news_runner.run, "stocks": stock_process.run}


def load_plist(name: str) -> dict:
    return plistlib.loads((MACOS / f"{name}.plist").read_bytes())


def parse_unit(text: str) -> dict[str, list[tuple[str, str]]]:
    """Sections of a unit file as lists of (key, value) pairs, keeping repeated keys in order."""
    sections: dict[str, list[tuple[str, str]]] = {}
    current = None
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1], [])
            continue
        assert current is not None and "=" in line, f"line {number} is not a key=value pair in a section: {raw}"
        key, value = line.split("=", 1)
        current.append((key.strip(), value.strip()))
    return sections


def load_unit(name: str) -> dict[str, list[tuple[str, str]]]:
    return parse_unit((SYSTEMD / f"{name}.service").read_text(encoding="utf-8"))


def values(unit, section: str, key: str) -> list[str]:
    return [v for k, v in unit.get(section, []) if k == key]


def one(unit, section: str, key: str) -> str:
    found = values(unit, section, key)
    assert len(found) == 1, f"[{section}] {key} appears {len(found)} times"
    return found[0]


def words(unit, section: str, key: str) -> list[str]:
    return " ".join(values(unit, section, key)).split()


def agent_argv(plist: dict) -> list[str]:
    """The command the agent's shell execs, after it starts caffeinate on the shell's own pid."""
    shell, flag, script = plist["ProgramArguments"]
    assert (shell, flag) == ("/bin/sh", "-c")
    keep_awake, command = script.split(" & exec ")
    assert shlex.split(keep_awake) == ["/usr/bin/caffeinate", "-s", "-w", "$$"]
    assert not set(";|<>`(){}\n") & set(script) and script.count("&") == 1 and script.count("$") == 2
    return shlex.split(command)


def test_deploy_holds_two_agents_and_two_units():
    assert sorted(p.name for p in MACOS.iterdir()) == [f"{NEWS}.plist", f"{STOCKS}.plist"]
    assert sorted(p.name for p in SYSTEMD.iterdir()) == [f"{NEWS}.service", f"{STOCKS}.service"]


@pytest.mark.parametrize("name", [NEWS, STOCKS])
def test_each_agent_keeps_its_process_alive_with_the_plan_settings(name):
    plist = load_plist(name)
    assert set(plist) == PLIST_KEYS
    assert plist["Label"] == name
    assert plist["RunAtLoad"] is True and plist["KeepAlive"] is True
    assert plist["ThrottleInterval"] == 10
    assert plist["ExitTimeOut"] == 60 and plist["ExitTimeOut"] >= 2 * news_runner.STOP_GRACE_SECONDS
    assert plist["Umask"] == 0o077
    assert plist["ProcessType"] == "Interactive"
    assert plist["WorkingDirectory"] == MAC_HOME


@pytest.mark.parametrize("name", [NEWS, STOCKS])
def test_each_agent_execs_its_command_under_caffeinate(name):
    argv = agent_argv(load_plist(name))
    assert argv[0] == f"{MAC_HOME}/.venv-v2/bin/btc_survivor_v2"
    assert argv[1:] == COMMANDS[name]


@pytest.mark.parametrize("name", [NEWS, STOCKS])
def test_each_command_parses_and_loads_the_committed_configs_from_the_working_directory(name):
    args = cli.build_parser().parse_args(agent_argv(load_plist(name))[1:])
    if args.command in BUILT:
        assert args.handler is BUILT[args.command]
    if args.command == "news":
        assert (ROOT / args.config).resolve() == (ROOT / "configs" / "v2" / "news.toml").resolve()
        assert load_news_config(ROOT / args.config).db_path.name == "news.db"
    else:
        s1, s2 = load_run_pair(ROOT / args.s1, ROOT / args.s2)
        assert (s1.run.name, s2.run.name) == ("s1_prices", "s2_news")
        assert (s1.news_input.use_news, s2.news_input.use_news) == (False, True)


@pytest.mark.parametrize("name", [NEWS, STOCKS])
def test_each_agent_environment_names_no_secret_proxy_or_sdk_variable(name):
    env = load_plist(name)["EnvironmentVariables"]
    assert set(env) == {"PATH", "PYTHONUNBUFFERED", "OPENBLAS_NUM_THREADS"}
    assert env["PYTHONUNBUFFERED"] == "1" and env["OPENBLAS_NUM_THREADS"] == "1"
    path = env["PATH"].split(":")
    assert path[:4] == SYSTEM_PATH and {"/opt/homebrew/bin", "/usr/local/bin"} <= set(path)
    assert not {n.upper() for n in env} & set(REFUSED)
    assert not any(n.upper().startswith(("ANTHROPIC", "FINNHUB", "CONTACT")) for n in env)


@pytest.mark.parametrize("name", [NEWS, STOCKS])
def test_each_agent_logs_into_the_folder_the_secret_scan_reads_and_git_ignores(name):
    plist = load_plist(name)
    log = f"{MAC_HOME}/data/logs/v2/launchd/{name}.log"
    assert plist["StandardOutPath"] == plist["StandardErrorPath"] == log
    relative = Path(log).relative_to(MAC_HOME)
    assert relative.parts[:3] == ("data", "logs", "v2")
    ignored = subprocess.run(["git", "check-ignore", "-q", "--no-index", str(relative)], cwd=ROOT)
    assert ignored.returncode == 0


def test_the_two_agents_differ_only_in_label_command_and_log():
    news, stocks = load_plist(NEWS), load_plist(STOCKS)
    for key in ("Label", "ProgramArguments", "StandardOutPath", "StandardErrorPath"):
        assert news.pop(key) != stocks.pop(key)
    assert news == stocks


@pytest.mark.parametrize("path", [*sorted(MACOS.glob("*.plist")), *sorted(SYSTEMD.glob("*.service"))],
                         ids=lambda p: p.name)
def test_every_path_is_absolute_and_names_only_the_placeholder_user(path):
    text = path.read_text(encoding="utf-8")
    assert "~" not in re.sub(r"<!--.*?-->|#[^\n]*", "", text, flags=re.S)  # neither launchd nor systemd expands it
    users = re.findall(r"/(?:Users|home)/([^/\s<]+)", text)
    assert users and set(users) == {"CHANGE_ME"}
    assert not re.search(r"--(detail|dry-run|test-url|port|log-dir)\b", text)


@pytest.mark.parametrize("name", [NEWS, STOCKS])
def test_each_unit_restarts_and_stops_like_the_plan(name):
    unit = load_unit(name)
    assert set(unit) == {"Unit", "Service", "Install"}
    for section, pairs in unit.items():
        for key in SINGLE:
            assert len([k for k, _ in pairs if k == key]) <= 1, f"[{section}] {key} is set more than once"
    assert one(unit, "Service", "Type") == "simple"
    assert one(unit, "Service", "Restart") == "always"
    assert one(unit, "Service", "RestartSec") == "10"
    assert one(unit, "Service", "KillSignal") == "SIGTERM"
    assert one(unit, "Service", "TimeoutStopSec") == "60"
    assert one(unit, "Service", "UMask") == "0077"
    assert one(unit, "Install", "WantedBy") == "multi-user.target"
    assert one(unit, "Service", "User") == "CHANGE_ME"
    assert one(unit, "Service", "WorkingDirectory") == PI_HOME
    env = dict(item.split("=", 1) for line in values(unit, "Service", "Environment") for item in shlex.split(line))
    assert env == {"PYTHONUNBUFFERED": "1", "OPENBLAS_NUM_THREADS": "1"}


@pytest.mark.parametrize("name", [NEWS, STOCKS])
def test_each_unit_waits_for_the_network_and_a_synchronized_clock(name):
    unit = load_unit(name)
    assert {"network-online.target", "time-sync.target"} <= set(words(unit, "Unit", "After"))
    assert "network-online.target" in words(unit, "Unit", "Wants")
    assert "systemctl enable systemd-time-wait-sync.service" in OPERATIONS.read_text(encoding="utf-8")
    assert "systemd-time-wait-sync.service" in (SYSTEMD / f"{name}.service").read_text(encoding="utf-8")


def test_the_stock_unit_starts_after_the_news_unit_and_never_the_reverse():
    stocks, news = load_unit(STOCKS), load_unit(NEWS)
    assert f"{NEWS}.service" in words(stocks, "Unit", "Wants")
    assert f"{NEWS}.service" in words(stocks, "Unit", "After")
    assert not any(STOCKS in v for pairs in news.values() for _, v in pairs)


@pytest.mark.parametrize("name", [NEWS, STOCKS])
def test_each_unit_runs_the_same_command_as_its_agent(name):
    argv = shlex.split(one(load_unit(name), "Service", "ExecStart"))
    assert argv[0] == f"{PI_HOME}/.venv-v2/bin/btc_survivor_v2"
    assert argv[1:] == agent_argv(load_plist(name))[1:] == COMMANDS[name]


def test_the_operations_guide_installs_each_file_by_hand_and_scans_its_logs():
    text = OPERATIONS.read_text(encoding="utf-8")
    for name in (NEWS, STOCKS):
        assert f"deploy/macos/{name}.plist" in text and f"deploy/systemd/{name}.service" in text
        assert f"launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/{name}.plist" in text
        assert f"launchctl bootout gui/$(id -u)/{name}" in text
    assert 's|CHANGE_ME|' in text and "mkdir -p data/logs/v2/launchd" in text
    assert "plutil -lint" in text
    assert "scripts/v2_secret_scan.py --logs data/logs/v2" in text
    assert "only with the owner's approval" in text


CHECKPOINT_ALLOWANCE_S = 30.0  # the final checkpoint of both runs, far above what the soak measured


def test_the_stock_agent_and_unit_wait_long_enough_for_fills_to_settle_and_both_checkpoints():
    s1, _ = load_run_pair(ROOT / "configs" / "v2" / "s1_prices.toml", ROOT / "configs" / "v2" / "s2_news.toml")
    settle = (s1.stock_costs.latency_ms + s1.stock_costs.latency_jitter_ms) / 1000
    assert load_plist(STOCKS)["ExitTimeOut"] >= settle + CHECKPOINT_ALLOWANCE_S
    assert int(one(load_unit(STOCKS), "Service", "TimeoutStopSec")) >= settle + CHECKPOINT_ALLOWANCE_S


def test_no_module_or_script_installs_or_controls_a_service():
    for path in [*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").rglob("*.py")]:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"launchctl|systemctl|LaunchAgents", text), path.relative_to(ROOT)


def test_the_data_guide_states_the_short_margin_and_day_trading_limits():
    text = (ROOT / "docs" / "v2_data_and_licensing.md").read_text(encoding="utf-8")
    section = text.split("## Shorts, margin and day trading", 1)[1].split("\n## ", 1)[0]
    for phrase in ("**The short sale restriction.** SEC Rule 201", "**Margin.** Shorting needs a margin account",
                   "**Day trading.**", "pattern day trader", "Regulatory Notice `26-10`",
                   "**Why the test does not depend on any of this.**"):
        assert phrase in section, phrase
