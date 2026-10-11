"""The live client against a local websocket server, so no network is needed."""

import asyncio
import dataclasses

import numpy as np
from websockets.asyncio.server import serve

from btc_survivor.config import load_config
from btc_survivor.feed.base import CLOSE, MESSAGE, OPEN, FeedAdapter, FeedClient
from btc_survivor.feed.coinbase import CoinbaseAdapter
from helpers import EXAMPLE_CONFIG

FEED = dataclasses.replace(load_config(EXAMPLE_CONFIG).feed, reconnect_backoff_start_seconds=0.01,
                           reconnect_backoff_max_seconds=0.05)


class LocalAdapter(FeedAdapter):
    max_message_bytes = CoinbaseAdapter.max_message_bytes

    def __init__(self, url):
        self.url = url

    def subscribe_messages(self):
        return ["subscribe"]

    def on_open(self, conn_id):
        pass

    def parse(self, raw, recv_ms, conn_id):
        return []


def run_client(handler, until):
    """Serve handler locally, run the client until until(lines) is true, and return the recorded lines."""
    lines = []

    async def main():
        stop = asyncio.Event()
        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            client = FeedClient(LocalAdapter(f"ws://127.0.0.1:{port}"), FEED,
                                lambda *row: lines.append(row) or (until(lines, client) and stop.set()),
                                np.random.default_rng(1))
            task = asyncio.create_task(client.run(stop))
            await asyncio.wait_for(stop.wait(), 10)
            await asyncio.wait_for(task, 10)
            return client

    client = asyncio.run(main())
    return lines, client


def test_messages_are_stamped_in_order_and_reconnects_get_new_ids():
    async def handler(ws):
        assert await ws.recv() == "subscribe"
        for i in range(3):
            await ws.send(f"m{i}")
        await ws.close()

    lines, client = run_client(handler, lambda lines, c: sum(1 for l in lines if l[0] == OPEN) >= 3)
    kinds = [l[0] for l in lines]
    assert kinds[:5] == [OPEN, MESSAGE, MESSAGE, MESSAGE, CLOSE]
    assert [l[1] for l in lines[1:4]] == ["m0", "m1", "m2"]
    assert [l[3] for l in lines if l[0] == OPEN][:3] == [1, 2, 3]
    stamps = [l[2] for l in lines]
    assert stamps == sorted(stamps)


def test_resync_request_closes_the_connection():
    async def handler(ws):
        await ws.recv()
        while True:
            await ws.send("tick")
            await asyncio.sleep(0.01)

    def until(lines, client):
        if len(lines) == 3:
            client.request_resync("gap")
        return any(l[0] == CLOSE for l in lines) and lines[-1][0] == OPEN

    lines, _ = run_client(handler, until)
    close = next(l for l in lines if l[0] == CLOSE)
    assert close[1] == "gap"


def test_large_snapshot_fits_the_message_limit():
    big = "x" * (5 * 1024 * 1024)

    async def handler(ws):
        await ws.recv()
        await ws.send(big)
        await ws.wait_closed()

    lines, _ = run_client(handler, lambda lines, c: any(l[0] == MESSAGE for l in lines))
    assert len(next(l for l in lines if l[0] == MESSAGE)[1]) == len(big)


def test_backoff_doubles_with_jitter_caps_and_resets_when_healthy():
    feed = dataclasses.replace(FEED, reconnect_backoff_start_seconds=1.0, reconnect_backoff_max_seconds=60.0)
    client = FeedClient(LocalAdapter("ws://unused"), feed, lambda *a: None, np.random.default_rng(3))
    delays = [client.backoff_seconds() for _ in range(9)]
    for attempt, delay in enumerate(delays):
        base = min(60.0, 2.0 ** attempt)
        assert base / 2 <= delay <= base
    client.mark_healthy()
    assert client.backoff_seconds() <= 1.0


def test_coinbase_limit_fits_a_full_depth_snapshot():
    assert CoinbaseAdapter.max_message_bytes >= 8 * 1024 * 1024


def test_resync_closes_a_silent_socket_from_another_task():
    async def handler(ws):
        await ws.recv()
        await ws.send("only message")
        await ws.wait_closed()  # silent but still open, like a stalled feed

    lines = []

    async def main():
        stop = asyncio.Event()
        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            client = FeedClient(LocalAdapter(f"ws://127.0.0.1:{port}"), FEED, lambda *row: lines.append(row),
                                np.random.default_rng(1))
            task = asyncio.create_task(client.run(stop))
            while not any(l[0] == MESSAGE for l in lines):
                await asyncio.sleep(0.01)
            client.request_resync("stale")  # what the ticker does when health goes stale
            for _ in range(500):
                if [l[0] for l in lines].count(OPEN) >= 2:
                    break
                await asyncio.sleep(0.01)
            stop.set()
            await asyncio.wait_for(task, 10)

    asyncio.run(main())
    closes = [l for l in lines if l[0] == CLOSE]
    assert closes[0][1] == "stale"
    assert [l[0] for l in lines].count(OPEN) >= 2


def test_resync_with_no_open_socket_does_not_kill_the_next_connection():
    client = FeedClient(LocalAdapter("ws://unused"), FEED, lambda *a: None, np.random.default_rng(1))
    client.request_resync("stale")
    assert client._resync is None


def test_backoff_keeps_growing_across_connections_that_fail_right_away():
    async def handler(ws):
        await ws.close()

    lines, client = run_client(handler, lambda lines, c: [l[0] for l in lines].count(OPEN) >= 4)
    assert client.attempt >= 3  # never reset by a successful handshake, only by mark_healthy
