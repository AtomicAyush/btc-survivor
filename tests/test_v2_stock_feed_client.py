"""StockFeedClient and BackoffSettings of PLAN_V2 section 4.6, against local websocket servers on 127.0.0.1 and with a
fake canary token in the URL, which must never reach a log record, a close reason or an exception."""

import asyncio
import base64
import hashlib
import logging
import re
from http import HTTPStatus

import numpy as np
import pytest
from websockets.asyncio.server import serve

from btc_survivor import config as v1_config
from btc_survivor.clock import MS_PER_HOUR, LiveClock
from btc_survivor.feed.base import CLOSE, MESSAGE, OPEN, FeedClient
from btc_survivor.v2 import redact
from btc_survivor.v2.config import load_stock_config
from btc_survivor.v2.stocks import runner
from btc_survivor.v2.stocks.finnhub import FinnhubAdapter
from btc_survivor.v2.stocks.runner import BackoffSettings, StockFeedClient
from v2_feedkit import CANARY, PING, S1, receive

SYMBOLS = load_stock_config(S1).stock_feed.symbols
FAST = BackoffSettings(0.01, 0.05)
WS_GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


@pytest.fixture(autouse=True)
def canary(monkeypatch):
    monkeypatch.setattr(redact, "_values", set())
    redact.register(CANARY)


@pytest.fixture
def logs(caplog):
    """Capture every log record at DEBUG and above, from every logger, read through caplog.records."""
    caplog.set_level(logging.DEBUG)
    return caplog


class StepClock(LiveClock):
    """A clock that stands still until the test moves it, to cross the hour between 401 ERROR lines exactly."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def now_ms(self):
        return self.value


def drive(handler, condition, client_setup=None, clock=None, process_request=None, base_url=None):
    """Run a StockFeedClient against handler on 127.0.0.1 until condition(lines, client) holds, then stop it, and return
    the lines, the client and the contexts the loop's exception handler saw."""
    lines, problems = [], []

    async def main():
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: problems.append(context))
        stop = asyncio.Event()
        async with serve(handler, "127.0.0.1", 0, process_request=process_request) as server:
            port = server.sockets[0].getsockname()[1]
            adapter = FinnhubAdapter(SYMBOLS, CANARY, base_url or f"ws://127.0.0.1:{port}")
            client = StockFeedClient(adapter, FAST, redact.redacting_sink(lambda *row: lines.append(row)),
                                     np.random.default_rng(1), clock)
            if client_setup:
                client_setup(client)
            task = asyncio.create_task(client.run(stop))
            for _ in range(1000):
                if await condition(lines, client):
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("the condition never held")
            stop.set()
            await asyncio.wait_for(task, 10)
            return client

    client = asyncio.run(main())
    return lines, client, problems


def kinds(lines):
    return [line[0] for line in lines]


def test_backoff_settings_give_the_reused_client_5_to_60_seconds_with_seeded_jitter(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a version 1 FeedConfig was built")

    monkeypatch.setattr(v1_config.FeedConfig, "__init__", refuse)
    settings = BackoffSettings.from_config(load_stock_config(S1).stock_feed)
    assert settings == BackoffSettings(5.0, 60.0)
    adapter = FinnhubAdapter(SYMBOLS, CANARY)

    def delays(seed, n=8):
        client = StockFeedClient(adapter, settings, lambda *row: None, np.random.default_rng(seed))
        return client, [client.backoff_seconds() for _ in range(n)]

    client, first = delays(7)
    assert isinstance(client, FeedClient) and client.feed is settings
    for delay, base in zip(first, [5, 10, 20, 40, 60, 60, 60, 60]):
        assert base / 2 <= delay <= base
    assert delays(7)[1] == first and delays(8)[1] != first
    client.mark_healthy()
    assert 2.5 <= client.backoff_seconds() <= 5


def test_subscribe_frames_go_out_after_every_connect():
    frames = []

    async def handler(ws):
        mine = []
        frames.append(mine)
        await receive(ws, mine, len(SYMBOLS))
        await ws.send(PING)
        if len(frames) == 1:
            await ws.close()
        else:
            await ws.wait_closed()

    async def two_full_connections(lines, client):
        return len(frames) == 2 and len(frames[1]) == len(SYMBOLS) and MESSAGE in kinds(lines)[3:]

    lines, client, problems = drive(handler, two_full_connections)
    assert kinds(lines) == [OPEN, MESSAGE, CLOSE, OPEN, MESSAGE, CLOSE]
    assert [line[3] for line in lines] == [1, 1, 1, 2, 2, 2]
    assert lines[1][1] == PING and lines[2][1] == "closed by server" and lines[5][1] == "shutdown"
    assert frames == [FinnhubAdapter(SYMBOLS, CANARY).subscribe_messages()] * 2
    assert [line[2] for line in lines] == sorted(line[2] for line in lines)
    assert client.connected_ms is None and not problems


def test_send_reaches_the_server_and_the_connection_stamp_marks_health():
    extra = []

    async def handler(ws):
        await receive(ws, [], len(SYMBOLS))
        await receive(ws, extra, 1)
        await ws.wait_closed()

    sent = []

    async def resubscribed(lines, client):
        if client.connected_ms is not None and not sent:
            assert client.connected_ms == lines[0][2]
            client.attempt = 3
            assert not client.mark_healthy_if_open(client.connected_ms + 59_999, 60_000) and client.attempt == 3
            assert client.mark_healthy_if_open(client.connected_ms + 60_000, 60_000) and client.attempt == 0
            assert not client.mark_healthy_if_open(client.connected_ms + 60_000, 60_000)
            sent.append(await client.send('{"type":"subscribe","symbol":"AAPL"}'))
        return bool(extra)

    lines, client, problems = drive(handler, resubscribed)
    assert sent == [True] and extra == ['{"type":"subscribe","symbol":"AAPL"}']
    assert asyncio.run(client.send("late")) is False
    assert not client.mark_healthy_if_open(10**15, 60_000)
    assert not problems


def test_a_resync_acts_once_per_connection_and_keeps_its_first_reason():
    async def handler(ws):
        await receive(ws, [], len(SYMBOLS))
        await ws.send(PING)
        await ws.wait_closed()

    asked = []

    async def resynced_twice(lines, client):
        if MESSAGE in kinds(lines) and not asked:
            client.request_resync("first reason")
            asked.append(client.connected_ms)
            client.request_resync("second reason")
        return kinds(lines).count(OPEN) >= 2

    lines, client, problems = drive(handler, resynced_twice)
    assert asked == [None]
    assert [line[1] for line in lines if line[0] == CLOSE][0] == "first reason"
    assert kinds(lines)[:4] == [OPEN, MESSAGE, CLOSE, OPEN] and not problems


def test_the_connection_never_goes_through_a_proxy(monkeypatch):
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")

    async def handler(ws):
        await receive(ws, [], len(SYMBOLS))
        await ws.send(PING)
        await ws.wait_closed()

    async def got_a_message(lines, client):
        return MESSAGE in kinds(lines)

    lines, client, problems = drive(handler, got_a_message)
    assert kinds(lines)[:2] == [OPEN, MESSAGE] and not problems


def test_pings_every_20_seconds_with_a_20_second_timeout_close_a_silent_socket():
    assert (StockFeedClient.ping_interval, StockFeedClient.ping_timeout) == (20.0, 20.0)

    async def silent(reader, writer):
        """Accept the websocket handshake by hand and then never answer anything, not even a ping."""
        request = await reader.readuntil(b"\r\n\r\n")
        key = re.search(rb"(?im)^sec-websocket-key: *(\S+)", request).group(1)
        accept = base64.b64encode(hashlib.sha1(key + WS_GUID).digest())
        writer.write(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                     b"Sec-WebSocket-Accept: " + accept + b"\r\n\r\n")
        await writer.drain()
        await asyncio.sleep(30)

    lines = []

    async def main():
        stop = asyncio.Event()
        server = await asyncio.start_server(silent, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = StockFeedClient(FinnhubAdapter(SYMBOLS, CANARY, f"ws://127.0.0.1:{port}"), FAST,
                                 lambda *row: lines.append(row), np.random.default_rng(1))
        client.ping_interval = client.ping_timeout = client.close_timeout = 0.1
        task = asyncio.create_task(client.run(stop))
        for _ in range(500):
            if CLOSE in kinds(lines):
                break
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, 10)
        server.close()

    asyncio.run(main())
    close = next(line for line in lines if line[0] == CLOSE)
    assert "keepalive ping timeout" in close[1]


def test_a_handshake_refused_with_401_backs_off_with_one_error_line_an_hour(logs):
    start = 1_791_072_000_000
    clock = StepClock(start)
    refused = []

    def reject(connection, request):
        """Refuse every handshake, moving the clock to just under an hour after the first ERROR line before the
        fourth refusal and to exactly an hour before the fifth."""
        refused.append(request.path == f"/?token={CANARY}")
        clock.value = start + {4: MS_PER_HOUR - 1, 5: MS_PER_HOUR}.get(len(refused), clock.value - start)
        return connection.respond(HTTPStatus.UNAUTHORIZED, "Invalid API key\n")

    async def never(ws):
        raise AssertionError("the handshake should never succeed")

    async def six_refusals(lines, client):
        return len(refused) >= 6

    lines, client, problems = drive(never, six_refusals, clock=clock, process_request=reject)
    assert lines == [] and not problems
    assert all(refused) and len(refused) >= 6
    assert client.attempt >= 6
    records = logs.records
    errors = [r for r in records if r.levelno >= logging.ERROR]
    assert [r.name for r in errors] == [runner.__name__] * 2
    assert "HTTP 401" in errors[0].getMessage()
    ended = [r.getMessage() for r in records if "ended (InvalidStatus" in r.getMessage()]
    assert len(ended) >= 5 and all("HTTP 401" in m for m in ended)
    assert not any(CANARY in r.getMessage() for r in records)


def test_the_canary_reaches_no_log_record_close_reason_or_exception(logs):
    logging.getLogger("websockets").setLevel(logging.NOTSET)

    async def handler(ws):
        await receive(ws, [], len(SYMBOLS))
        await ws.send(f'{{"type":"error","msg":"Invalid token {CANARY}"}}')
        await ws.close(code=1008, reason=f"bad token {CANARY}")

    async def two_opens(lines, client):
        return kinds(lines).count(OPEN) >= 2

    lines, client, problems = drive(handler, two_opens)
    assert logging.getLogger("websockets").level == logging.WARNING
    assert kinds(lines)[:4] == [OPEN, MESSAGE, CLOSE, OPEN]
    assert redact.MASK in lines[1][1] and redact.MASK in lines[2][1]
    assert lines[2][1].startswith("ConnectionClosedError")

    invalid = drive(handler, lambda lines, client: asyncio.sleep(0.2, result=True), base_url="https://127.0.0.1:1")
    assert invalid[0] == [] and not invalid[2]
    records = logs.records
    texts = [r.getMessage() for r in records] + [r.exc_text or "" for r in records]
    assert any("InvalidURI: https://127.0.0.1:1?token=" + redact.MASK in text for text in texts)
    assert any("feed connection 1 open" in text for text in texts)
    assert not any(CANARY in text for text in texts)
    assert not any(CANARY in text for _, text, _, _ in lines)
    assert not problems
    assert CANARY not in repr(client) and CANARY not in repr(client.adapter)
