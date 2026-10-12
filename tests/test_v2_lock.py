"""The Finnhub websocket lock of PLAN_V2 section 4.1, on lock files in a test folder only."""

import os
import stat
import subprocess
import sys

import pytest

from btc_survivor.v2.stocks.runner import LockError, websocket_lock
from v2_feedkit import CANARY, CLOSED_URL, recorder_config, run_script, script_env

HOLD = """
import sys, time
from btc_survivor.v2.stocks.runner import websocket_lock
with websocket_lock(sys.argv[1]):
    print("held", flush=True)
    time.sleep(120)
"""


def refusal(lock, pid):
    return (f"another process, id {pid}, holds the Finnhub websocket lock {lock}, and the key allows one connection, "
            f"so this process exits")


def test_a_second_holder_is_refused_with_the_first_holder_named(tmp_path):
    lock = tmp_path / "config" / "finnhub_ws.lock"
    with websocket_lock(lock) as held:
        assert held == lock
        assert lock.read_text(encoding="ascii") == f"{os.getpid()}\n"
        assert stat.S_IMODE(lock.stat().st_mode) == 0o600
        assert stat.S_IMODE(lock.parent.stat().st_mode) == 0o700
        with pytest.raises(LockError) as info:
            with websocket_lock(lock):
                pass
        assert str(info.value) == refusal(lock, os.getpid())
        assert lock.read_text(encoding="ascii") == f"{os.getpid()}\n"
    assert lock.read_text(encoding="ascii") == ""
    with websocket_lock(lock):
        pass


def test_a_second_recorder_process_exits_with_a_clear_error(tmp_path):
    config = recorder_config(tmp_path)
    lock = tmp_path.resolve() / "lock" / "finnhub_ws.lock"
    holder = subprocess.Popen([sys.executable, "-c", HOLD, str(lock)], stdout=subprocess.PIPE, text=True,
                              env=script_env(tmp_path))
    try:
        assert holder.stdout.readline() == "held\n"
        for args in (["--dry-run"], ["--log-dir", str(tmp_path / "logs")]):
            result = run_script(config, *args, "--test-url", CLOSED_URL)
            assert result.returncode == 2
            assert result.stdout == ""
            assert result.stderr == f"error: {refusal(lock, holder.pid)}\n"
            assert CANARY not in result.stderr
        assert not (tmp_path / "data").exists()
    finally:
        holder.kill()
        holder.wait(10)
    result = run_script(config, "--dry-run", "--test-url", CLOSED_URL)
    assert result.returncode == 0, result.stderr
    with websocket_lock(lock):
        pass
