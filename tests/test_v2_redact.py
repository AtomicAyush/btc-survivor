"""A fake canary token registered with redact must never come out of a log line, a traceback, an uncaught
exception, the asyncio loop, a websockets error during connect, a version 1 close reason or a recorded line."""

import asyncio
import dataclasses
import gzip
import io
import json
import logging
import subprocess
import sys

import numpy as np
import pytest
from websockets.asyncio.server import serve

from btc_survivor.config import load_config
from btc_survivor.feed.base import CLOSE, MESSAGE, OPEN, FeedAdapter, FeedClient
from btc_survivor.recorder import Recorder, read_recording
from btc_survivor.v2 import redact
from helpers import EXAMPLE_CONFIG

CANARY = "CANARYtok3nFAKE0001"
MASK = redact.MASK
FEED = dataclasses.replace(load_config(EXAMPLE_CONFIG).feed, reconnect_backoff_start_seconds=0.01,
                           reconnect_backoff_max_seconds=0.05)


@pytest.fixture(autouse=True)
def canary(monkeypatch):
    monkeypatch.setattr(redact, "_values", set())
    redact.register(CANARY)


@pytest.fixture
def logs():
    """Capture formatted log output through install_formatter, restoring every handler's formatter afterwards."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    saved = [(h, h.formatter) for h in redact._all_handlers()]
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    redact.install_formatter()
    yield stream
    root.removeHandler(handler)
    root.setLevel(level)
    for h, formatter in saved:
        h.setFormatter(formatter)


class LocalAdapter(FeedAdapter):
    max_message_bytes = 1 << 20

    def __init__(self, url):
        self.url = url

    def subscribe_messages(self):
        return ["subscribe"]

    def on_open(self, conn_id):
        pass

    def parse(self, raw, recv_ms, conn_id):
        return []


def test_redact_masks_every_value_and_overlaps_whole():
    redact.register("")
    redact.register(CANARY + "XYZ")
    assert redact.redact("plain text") == "plain text"
    assert redact.redact(f"a {CANARY} b {CANARY}XYZ c") == f"a {MASK} b {MASK} c"


def test_log_record_is_masked(logs):
    log = logging.getLogger("btc_survivor.v2.test")
    log.info("token is %s", CANARY)
    log.warning(f"url wss://ws.example.invalid/?token={CANARY}")
    text = logs.getvalue()
    assert CANARY not in text
    assert text.count(MASK) == 2 and "token is" in text


def test_formatted_traceback_is_masked(logs):
    log = logging.getLogger("btc_survivor.v2.test")
    try:
        raise RuntimeError(f"connect to wss://ws.example.invalid/?token={CANARY} failed")
    except RuntimeError:
        log.exception("connect failed")
        direct = redact.RedactingFormatter().formatException(sys.exc_info())
    text = logs.getvalue()
    assert "Traceback" in text and "RuntimeError" in text and MASK in text
    assert CANARY not in text and CANARY not in direct and MASK in direct


def test_install_formatter_wraps_named_loggers_once(logs):
    own = logging.StreamHandler(io.StringIO())
    inner = logging.Formatter("%(message)s")
    own.setFormatter(inner)
    logger = logging.getLogger("btc_survivor.v2.test.own")
    logger.addHandler(own)
    try:
        redact.install_formatter()
        redact.install_formatter()
        assert isinstance(own.formatter, redact.RedactingFormatter) and own.formatter.inner is inner
        logger.error("own %s", CANARY)
        assert own.stream.getvalue() == f"own {MASK}\n"
    finally:
        logger.removeHandler(own)


CHILD = """
import logging, sys, threading, warnings
from btc_survivor.v2 import redact
redact.register(sys.argv[1])
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
redact.install()
warnings.warn("warning with token " + sys.argv[1])
def boom():
    raise ValueError("thread with token " + sys.argv[1])
thread = threading.Thread(target=boom, name="worker")
thread.start()
thread.join()
raise RuntimeError("main with token " + sys.argv[1])
"""


def test_excepthooks_and_warnings_reach_stderr_only_masked():
    result = subprocess.run([sys.executable, "-c", CHILD, CANARY], capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert CANARY not in result.stderr and CANARY not in result.stdout
    for expected in ("uncaught exception in thread worker", "ValueError: thread with token " + MASK,
                     "CRITICAL btc_survivor.v2 uncaught exception", "RuntimeError: main with token " + MASK,
                     "warning with token " + MASK):
        assert expected in result.stderr


EDGES = """
import logging, sys
from btc_survivor.v2 import redact
token = sys.argv[1]
redact.register(token)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
redact.install()
logging.getLogger("bad").error("format fails %s %s", "wss://ws.example.invalid/?token=" + token)
lonely = logging.getLogger("lonely")
lonely.propagate = False
lonely.warning("lonely with token " + token)
class Leaky:
    def __del__(self):
        raise ValueError("del with token " + token)
leaky = Leaky()
del leaky
class Pause:
    def __await__(self):
        yield
async def suspended():
    try:
        await Pause()
    finally:
        raise RuntimeError("cleanup with token " + token)
coroutine = suspended()
coroutine.send(None)
del coroutine
print("finished", file=sys.stderr)
"""


def test_ignored_errors_a_bad_log_call_and_the_last_resort_reach_stderr_only_masked():
    result = subprocess.run([sys.executable, "-c", EDGES, CANARY], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0
    assert CANARY not in result.stderr and CANARY not in result.stdout
    assert "format fails" not in result.stderr and "Arguments" not in result.stderr
    for expected in ("lonely with token " + MASK, "Exception ignored in: <function Leaky.__del__",
                     "ValueError: del with token " + MASK, "Exception ignored in: <coroutine object suspended",
                     "RuntimeError: cleanup with token " + MASK, "finished"):
        assert expected in result.stderr


def test_asyncio_handler_masks_message_context_and_traceback(logs):
    def boom(token):
        raise RuntimeError(f"callback with token {token}")

    async def main():
        loop = asyncio.get_running_loop()
        redact.install_asyncio_handler(loop)
        loop.call_soon(boom, CANARY)
        await asyncio.sleep(0.01)
        loop.call_exception_handler({"message": f"plain context {CANARY}"})

    asyncio.run(main())
    text = logs.getvalue()
    assert CANARY not in text
    assert "asyncio: Exception in callback" in text and "handle:" in text
    assert f"RuntimeError: callback with token {MASK}" in text and f"plain context {MASK}" in text


def test_invalid_uri_raised_during_connect_is_masked(logs):
    lines = []

    async def main():
        stop = asyncio.Event()
        client = FeedClient(LocalAdapter(f"https://ws.example.invalid/?token={CANARY}"), FEED,
                            redact.redacting_sink(lambda *row: lines.append(row)), np.random.default_rng(1))
        task = asyncio.create_task(client.run(stop))
        while "feed connection 1 ended" not in logs.getvalue():
            await asyncio.sleep(0.005)
        stop.set()
        await asyncio.wait_for(task, 10)

    asyncio.run(asyncio.wait_for(main(), 10))
    text = logs.getvalue()
    assert "InvalidURI" in text and MASK in text and CANARY not in text
    assert lines == []


def test_close_reason_and_recorded_lines_are_masked(logs, tmp_path):
    recorder = Recorder(tmp_path)
    lines = []

    async def handler(ws):
        await ws.recv()
        await ws.send(json.dumps({"type": "error", "msg": f"Invalid token {CANARY}"}))
        await ws.close(code=1008, reason=f"bad token {CANARY}")

    async def main():
        stop = asyncio.Event()

        def sink(kind, raw, recv_ms, conn_id):
            recorder.write(kind, raw, recv_ms, conn_id)
            lines.append((kind, raw))

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            client = FeedClient(LocalAdapter(f"ws://127.0.0.1:{port}"), FEED, redact.redacting_sink(sink),
                                np.random.default_rng(1))
            task = asyncio.create_task(client.run(stop))
            while "feed connection 1 ended" not in logs.getvalue():
                await asyncio.sleep(0.005)
            stop.set()
            await asyncio.wait_for(task, 10)

    asyncio.run(asyncio.wait_for(main(), 10))
    recorder.close()
    assert [kind for kind, _ in lines[:3]] == [OPEN, MESSAGE, CLOSE]
    assert json.loads(lines[1][1])["msg"] == f"Invalid token {MASK}"
    assert lines[2][1].startswith("ConnectionClosedError") and f"bad token {MASK}" in lines[2][1]
    assert not any(CANARY in text for _, text in lines)
    raw = b"".join(gzip.open(path).read() for path in tmp_path.glob("*.jsonl.gz"))
    assert CANARY.encode() not in raw and raw.count(MASK.encode()) >= 2
    assert [(kind, text) for kind, text, _, _ in read_recording([tmp_path])] == lines
    text = logs.getvalue()
    assert CANARY not in text and f"feed connection 1 ended (ConnectionClosedError" in text and MASK in text
