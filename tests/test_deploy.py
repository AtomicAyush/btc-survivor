"""The systemd template unit in deploy/ has every setting the plan requires and runs the instance's own config."""

import re
import shlex
from pathlib import Path

import pytest

from btc_survivor.__main__ import build_parser
from helpers import ROOT

UNIT = ROOT / "deploy" / "btc_survivor@.service"
PI_GUIDE = ROOT / "docs" / "raspberry_pi.md"
SINGLE = ("Description", "Type", "User", "WorkingDirectory", "ExecStart", "Restart", "RestartSec", "KillSignal",
          "TimeoutStopSec", "WantedBy")


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


@pytest.fixture(scope="module")
def unit():
    return parse_unit(UNIT.read_text(encoding="utf-8"))


def values(unit, section: str, key: str) -> list[str]:
    return [v for k, v in unit.get(section, []) if k == key]


def one(unit, section: str, key: str) -> str:
    found = values(unit, section, key)
    assert len(found) == 1, f"[{section}] {key} appears {len(found)} times"
    return found[0]


def test_the_unit_is_a_template_with_unit_service_and_install_sections(unit):
    assert UNIT.name.endswith("@.service")
    assert set(unit) == {"Unit", "Service", "Install"}
    for section, pairs in unit.items():
        for key in SINGLE:
            assert len([k for k, _ in pairs if k == key]) <= 1, f"[{section}] {key} is set more than once"
    assert one(unit, "Install", "WantedBy") == "multi-user.target"
    assert "%i" in one(unit, "Unit", "Description")


def test_restart_and_stop_settings(unit):
    assert one(unit, "Service", "Restart") == "always"
    assert one(unit, "Service", "RestartSec") == "5"
    assert one(unit, "Service", "KillSignal") == "SIGTERM"
    assert one(unit, "Service", "TimeoutStopSec") == "60"


def test_waits_for_the_network_and_a_synchronized_clock(unit):
    after = " ".join(values(unit, "Unit", "After")).split()
    wants = " ".join(values(unit, "Unit", "Wants")).split()
    assert {"network-online.target", "time-sync.target"} <= set(after)
    assert "network-online.target" in wants


def test_environment_is_unbuffered_with_one_blas_thread(unit):
    env = dict(item.split("=", 1) for line in values(unit, "Service", "Environment") for item in shlex.split(line))
    assert env.get("PYTHONUNBUFFERED") == "1"
    assert env.get("OPENBLAS_NUM_THREADS") == "1"


def test_user_and_working_directory_are_placeholders(unit):
    user = one(unit, "Service", "User")
    workdir = one(unit, "Service", "WorkingDirectory")
    assert user == "CHANGE_ME"
    assert workdir.startswith("/") and "CHANGE_ME" in workdir
    assert one(unit, "Service", "ExecStart").startswith(workdir + "/.venv/bin/btc_survivor ")


def test_exec_start_runs_the_instance_config(unit):
    exec_start = one(unit, "Service", "ExecStart")
    assert exec_start.count("%i") == 1 and "configs/%i.toml" in exec_start
    for config in sorted((ROOT / "configs").glob("run_*.toml")):
        argv = shlex.split(exec_start.replace("%i", config.stem))[1:]
        args = build_parser().parse_args(argv)
        assert args.command == "run"
        assert (ROOT / args.config).resolve() == config.resolve()


def test_every_instance_named_in_the_pi_guide_has_a_config():
    names = set(re.findall(r"btc_survivor@(run_\w+)", PI_GUIDE.read_text(encoding="utf-8")))
    assert {"run_a", "run_b", "run_c", "run_d"} <= names
    for name in names:
        assert (ROOT / "configs" / f"{name}.toml").is_file(), f"btc_survivor@{name} has no configs/{name}.toml"


def test_the_pi_guide_enables_waiting_for_time_sync():
    text = PI_GUIDE.read_text(encoding="utf-8")
    assert "systemctl enable systemd-time-wait-sync.service" in text
    assert "deploy/btc_survivor@.service" in text
