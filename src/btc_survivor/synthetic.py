"""Synthetic recordings in the Coinbase Advanced Trade message format, for tests and soak runs. Never used live.

The market is a random walk with a small book and Poisson trades. Faults are injected on request, and after a
gap, a crossed book or a long silence the generator also emits the reconnect that the live client would make,
so a synthetic recording replays like a real one.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from btc_survivor.clock import iso
from btc_survivor.feed.base import CLOSE, MESSAGE, OPEN

TICK = 0.01


@dataclass(frozen=True)
class Fault:
    at_s: float
    kind: str  # gap, crossed, silence, disconnect, skew
    duration_s: float = 0.0
    skew_ms: int = 0


class SyntheticFeed:
    def __init__(self, seed: int, start_ms: int, product: str = "BTC-USD", mid: float = 80_000.0,
                 vol_bps_per_min: float = 6.0, levels: int = 25, level_step: float = 0.5,
                 ticks_per_s: int = 4, trades_per_s: float = 1.5, latency_ms: int = 20,
                 reconnect_s: float = 1.5, stale_s: float = 5.0) -> None:
        self.rng = np.random.default_rng(seed)
        self.start_ms = start_ms
        self.product = product
        self.mid = mid
        self.sigma = vol_bps_per_min / 1e4 / math.sqrt(60 * ticks_per_s)
        self.levels = levels
        self.level_step = level_step
        self.ticks_per_s = ticks_per_s
        self.trades_per_s = trades_per_s
        self.latency_ms = latency_ms
        self.reconnect_s = reconnect_s
        self.stale_s = stale_s
        self.trade_id = 1_000_000
        self.book: dict[tuple[str, float], float] = {}
        self.conn_id = 0
        self.seq = 0
        self.skew_ms = 0
        self.last_ms = start_ms

    def _envelope(self, channel: str, recv_ms: int, events: list) -> str:
        msg = {"channel": channel, "timestamp": iso(recv_ms - self.latency_ms + self.skew_ms),
               "sequence_num": self.seq, "events": events}
        self.seq += 1
        self.last_ms = recv_ms
        return json.dumps(msg, separators=(",", ":"))

    def _target_book(self) -> dict[tuple[str, float], float]:
        best_bid = math.floor((self.mid - TICK / 2) / TICK) * TICK
        book = {}
        for i in range(self.levels):
            bid = round(best_bid - i * self.level_step, 2)
            ask = round(best_bid + TICK + i * self.level_step, 2)
            book[("bid", bid)] = self.book.get(("bid", bid)) or round(float(self.rng.uniform(0.01, 2.0)), 8)
            book[("offer", ask)] = self.book.get(("offer", ask)) or round(float(self.rng.uniform(0.01, 2.0)), 8)
        for key in list(book)[: max(1, self.levels // 5)]:
            if self.rng.random() < 0.5:
                book[key] = round(float(self.rng.uniform(0.01, 2.0)), 8)
        return book

    @staticmethod
    def _updates(items, event_time: str) -> list[dict]:
        return [{"side": side, "event_time": event_time, "price_level": f"{price:.2f}", "new_quantity": f"{size:.8f}"}
                for (side, price), size in items]

    def _connect(self, recv_ms: int) -> Iterator[tuple[str, str, int, int]]:
        self.conn_id += 1
        self.seq = 0
        yield OPEN, "", recv_ms, self.conn_id
        self.book = self._target_book()
        stamp = iso(recv_ms)
        snapshot = [{"type": "snapshot", "product_id": self.product,
                     "updates": self._updates(sorted(self.book.items()), stamp)}]
        yield MESSAGE, self._envelope("l2_data", recv_ms + 1, snapshot), recv_ms + 1, self.conn_id
        trades = [{"type": "snapshot", "trades": [self._trade_dict(recv_ms, "SELL") for _ in range(3)][::-1]}]
        yield MESSAGE, self._envelope("market_trades", recv_ms + 2, trades), recv_ms + 2, self.conn_id
        ack = [{"subscriptions": {"heartbeats": ["heartbeats"], "level2": [self.product], "market_trades": [self.product]}}]
        yield MESSAGE, self._envelope("subscriptions", recv_ms + 3, ack), recv_ms + 3, self.conn_id

    def _trade_dict(self, ms: int, side: str) -> dict:
        self.trade_id += 1
        best = min(p for (s, p) in self.book if s == "offer") if side == "SELL" else max(p for (s, p) in self.book if s == "bid")
        return {"product_id": self.product, "trade_id": str(self.trade_id), "price": f"{best:.2f}",
                "size": f"{float(self.rng.uniform(0.0001, 0.05)):.8f}", "time": iso(ms), "side": side}

    def lines(self, seconds: float, faults: tuple[Fault, ...] = ()) -> Iterator[tuple[str, str, int, int]]:
        """Yield (kind, raw, recv_ms, conn_id) for seconds of market time, like a recording."""
        tick_ms = 1000 // self.ticks_per_s
        pending = sorted(faults, key=lambda f: f.at_s)
        silent_until: int | None = None
        skew_until: int | None = None
        reconnect_at: int | None = None
        drop_seq = False
        yield from self._connect(self.start_ms)
        connected = True
        end = self.start_ms + int(seconds * 1000)
        t = self.start_ms + tick_ms
        tick = 0
        while t < end:
            tick += 1
            self.mid *= math.exp(float(self.rng.normal(0.0, self.sigma)))
            if skew_until is not None and t >= skew_until:
                self.skew_ms, skew_until = 0, None
            while pending and self.start_ms + int(pending[0].at_s * 1000) <= t:
                fault = pending.pop(0)
                if not connected and fault.kind in ("gap", "crossed", "disconnect"):
                    continue
                if fault.kind == "gap":
                    drop_seq = True
                elif fault.kind == "silence":
                    silent_until = t + int(min(fault.duration_s, self.stale_s + 1) * 1000)
                elif fault.kind == "skew":
                    self.skew_ms, skew_until = fault.skew_ms, t + int(fault.duration_s * 1000)
                elif fault.kind == "disconnect":
                    yield CLOSE, "server closed", t, self.conn_id
                    connected = False
                    reconnect_at = t + int(max(fault.duration_s, self.reconnect_s) * 1000)
                elif fault.kind == "crossed":
                    best_ask = min(p for (s, p) in self.book if s == "offer")
                    cross = [{"type": "update", "product_id": self.product,
                              "updates": self._updates([(("bid", round(best_ask + TICK, 2)), 1.0)], iso(t))}]
                    yield MESSAGE, self._envelope("l2_data", t, cross), t, self.conn_id
                    yield CLOSE, "resync after crossed", t + 50, self.conn_id
                    connected = False
                    reconnect_at = t + int(self.reconnect_s * 1000)
            if not connected:
                if reconnect_at is not None and t >= reconnect_at:
                    yield from self._connect(t)
                    connected, reconnect_at = True, None
                t += tick_ms
                continue
            if silent_until is not None:
                if t < silent_until:
                    t += tick_ms
                    continue
                silent_until = None
                if t - self.last_ms > self.stale_s * 1000:
                    yield CLOSE, "resync after stale", t, self.conn_id
                    connected = False
                    reconnect_at = t + int(self.reconnect_s * 1000)
                    t += tick_ms
                    continue
            yield from self._tick(t, drop_seq)
            if drop_seq:
                drop_seq = False
                yield CLOSE, "resync after gap", t + 60, self.conn_id
                connected = False
                reconnect_at = t + int(self.reconnect_s * 1000)
            elif tick % self.ticks_per_s == 0:
                beat = [{"current_time": iso(t), "heartbeat_counter": tick // self.ticks_per_s}]
                yield MESSAGE, self._envelope("heartbeats", t + 5, beat), t + 5, self.conn_id
            t += tick_ms

    def _tick(self, t: int, drop_seq: bool) -> Iterator[tuple[str, str, int, int]]:
        if drop_seq:
            self.seq += 1
        target = self._target_book()
        changes = [(key, 0.0) for key in self.book if key not in target]
        changes += [(key, size) for key, size in target.items() if self.book.get(key) != size]
        self.book = target
        if changes:
            update = [{"type": "update", "product_id": self.product, "updates": self._updates(changes, iso(t))}]
            yield MESSAGE, self._envelope("l2_data", t, update), t, self.conn_id
        count = int(self.rng.poisson(self.trades_per_s / self.ticks_per_s))
        if count:
            trades = [self._trade_dict(t, "BUY" if self.rng.random() < 0.5 else "SELL") for _ in range(count)][::-1]
            yield MESSAGE, self._envelope("market_trades", t + 2, [{"type": "update", "trades": trades}]), t + 2, self.conn_id
