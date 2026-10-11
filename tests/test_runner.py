"""The live runner against a local websocket server, ended by a real SIGTERM, so no network is needed."""

import asyncio
import os
import signal
import sqlite3
import time

from websockets.asyncio.server import serve

from btc_survivor.feed.coinbase import CoinbaseAdapter
from btc_survivor.runner import memory_mb, run_live
from btc_survivor.synthetic import SyntheticFeed
from enginekit import fast_config


def run_once(cfg):
    """Serve three seconds of synthetic messages stamped now, send SIGTERM once the book is valid, return the code."""

    async def handler(ws):
        for _ in range(3):
            await ws.recv()
        for kind, raw, _, _ in SyntheticFeed(1, time.time_ns() // 1_000_000).lines(3):
            if kind == "message":
                await ws.send(raw)
        await ws.wait_closed()

    async def main():
        async with serve(handler, "127.0.0.1", 0, max_size=None) as server:
            adapter = CoinbaseAdapter(cfg.feed.product)
            adapter.url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            ready = asyncio.get_running_loop().create_future()
            task = asyncio.create_task(run_live(cfg, adapter=adapter, ready=ready))
            runner = await asyncio.wait_for(ready, 10)
            for _ in range(500):
                if runner.engine.market.book.valid:
                    break
                await asyncio.sleep(0.01)
            assert runner.engine.market.book.valid
            os.kill(os.getpid(), signal.SIGTERM)
            return await asyncio.wait_for(task, 20)

    return asyncio.run(main())


def test_sigterm_ends_with_a_checkpoint_and_exit_code_zero_and_the_next_start_resumes(tmp_path):
    cfg = fast_config(tmp_path / "data")
    assert run_once(cfg) == 0
    db = sqlite3.connect(cfg.db_path)
    assert db.execute("SELECT count(*) FROM checkpoints").fetchone()[0] == 1
    status = db.execute("SELECT feed_state, living, last_checkpoint_ts FROM runtime_status").fetchone()
    assert status[0] == "stopped" and status[1] == cfg.population.size and status[2] is not None
    events = db.execute("SELECT kind, detail FROM feed_events").fetchall()
    assert ("disconnected", "shutdown") in events and any(kind == "snapshot" for kind, _ in events)
    assert db.execute("SELECT count(*) FROM bots").fetchone()[0] == 2 * cfg.population.size
    db.close()

    assert run_once(cfg) == 0
    db = sqlite3.connect(cfg.db_path)
    assert db.execute("SELECT count(*) FROM feed_events WHERE kind = 'resume'").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM bots").fetchone()[0] == 2 * cfg.population.size
    assert db.execute("SELECT count(*) FROM checkpoints").fetchone()[0] == 2
    db.close()


def test_memory_reading_reports_a_peak():
    current, peak = memory_mb()
    assert peak > 10 and (current is None or current > 10)
