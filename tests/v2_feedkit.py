"""Shared pieces for the version 2 feed tests, a fake canary token, a throwaway config and secrets file whose paths
all point into a test folder, and a fake Finnhub server on 127.0.0.1 running in its own thread."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from btc_survivor.v2.local_secrets import FINNHUB, REFUSED
from helpers import ROOT

CANARY = "CANARYfinnhubTOKEN0042"
S1 = ROOT / "configs" / "v2" / "s1_prices.toml"
SCRIPT = ROOT / "scripts" / "v2_record_stocks.py"
PING = '{"type":"ping"}'

# A closed port on 127.0.0.1, passed to every recorder run that should never connect, so a regression in the dry
# run or the lock check would still reach no outside host.
CLOSED_URL = "ws://127.0.0.1:9"

# Every path the recorder could touch, moved into the test folder, and a fast backoff.
TEST_SETTINGS = {
    "data_dir": '"data/s1_prices"',
    "secrets_file": '"secrets.env"',
    "raw_dir": '"data/raw_stocks"',
    "lock_file": '"lock/finnhub_ws.lock"',
    "news_db": '"data/news.db"',
    "reconnect_backoff_start_seconds": "0.05",
    "reconnect_backoff_max_seconds": "0.2",
}


def recorder_config(folder: Path, secrets: str | None = None, **values: str) -> Path:
    """Write a secrets file, by default holding only the canary, and a copy of the S1 config whose every path points
    into folder, refusing to write it if any setting was not found exactly once."""
    secrets_path = folder / "secrets.env"
    secrets_path.write_text(secrets if secrets is not None else f"{FINNHUB}={CANARY}\n", encoding="utf-8")
    secrets_path.chmod(0o600)
    text = S1.read_text(encoding="utf-8")
    for key, value in {**TEST_SETTINGS, **values}.items():
        text, found = re.subn(rf"(?m)^{key} = [^#\n]*", f"{key} = {value} ", text)
        assert found == 1, key
    assert "~" not in text and "../" not in text
    path = folder / "s1_prices.toml"
    path.write_text(text, encoding="utf-8")
    return path


def script_env(folder: Path, **extra: str) -> dict[str, str]:
    """The environment for a recorder subprocess, with HOME moved into the test folder as a second guard and no proxy
    or certificate variable, which the recorder refuses."""
    env = {name: value for name, value in os.environ.items() if name.upper() not in REFUSED}
    return {**env, "HOME": str(folder), **extra}


def run_script(config: Path, *args: str, timeout: float = 60, **extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), str(config), *args], capture_output=True, text=True,
                          timeout=timeout, cwd=config.parent, env=script_env(config.parent, **extra))


def start_script(config: Path, *args: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, str(SCRIPT), str(config), *args], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, cwd=config.parent, env=script_env(config.parent))


async def receive(ws, frames: list[str], count: int) -> None:
    """Receive count frames into frames, such as the subscribe frames after a connect."""
    for _ in range(count):
        frames.append(await asyncio.wait_for(ws.recv(), 10))


class FakeFinnhub:
    """A websocket server on 127.0.0.1 that plays one script per connection, the last one repeating, and keeps the
    request paths and the frames it received. The paths hold the token, so tests compare them and never print them."""

    def __init__(self, *plays, process_request=None) -> None:
        self.plays = plays
        self.process_request = process_request
        self.paths: list[str] = []
        self.frames: list[list[str]] = []
        self.port = 0
        self._ready = threading.Event()
        self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True)

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    def __enter__(self) -> FakeFinnhub:
        self._thread.start()
        assert self._ready.wait(10)
        return self

    def __exit__(self, *exc) -> None:
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(10)

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        async with serve(self._handler, "127.0.0.1", 0, process_request=self.process_request) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self._stop.wait()

    async def _handler(self, ws) -> None:
        index = len(self.paths)
        self.paths.append(ws.request.path)
        frames: list[str] = []
        self.frames.append(frames)
        try:
            await self.plays[min(index, len(self.plays) - 1)](ws, frames)
        except ConnectionClosed:
            pass
