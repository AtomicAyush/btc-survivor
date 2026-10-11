"""Coinbase Advanced Trade public market data. Keyless channels only: level2, market_trades and heartbeats.

Every message on a connection carries sequence_num, one counter shared by all channels. Any jump means a
message was lost, so the book must be rebuilt. The market_trades snapshot holds trades from before the
connection, so it only sets the trade id cursor and never reaches a candle.
"""

from __future__ import annotations

import json

from btc_survivor.clock import parse_iso
from btc_survivor.config import FeedConfig
from btc_survivor.feed.base import FeedAdapter
from btc_survivor.types import ASK, BID, BookDelta, BookSnapshot, FeedEvent, FeedNotice, Trade

URL = "wss://advanced-trade-ws.coinbase.com"
MAX_MESSAGE_BYTES = 16 * 1024 * 1024  # the full depth level2 snapshot is about 4.5 MB
SIDES = {"bid": BID, "offer": ASK}


class CoinbaseAdapter(FeedAdapter):
    url = URL
    max_message_bytes = MAX_MESSAGE_BYTES

    def __init__(self, product: str) -> None:
        self.product = product
        self._expected_seq: int | None = None
        self._trade_cursor: int | None = None

    def subscribe_messages(self) -> list[str]:
        return [json.dumps({"type": "subscribe", "product_ids": [self.product], "channel": channel})
                for channel in ("level2", "market_trades", "heartbeats")]

    def on_open(self, conn_id: int) -> None:
        self._expected_seq = None
        self._trade_cursor = None

    def parse(self, raw: str, recv_ms: int, conn_id: int) -> list[FeedEvent]:
        msg = json.loads(raw)
        if msg.get("type") == "error":
            return [FeedNotice(recv_ms, conn_id, "error", None, str(msg.get("message", raw))[:200])]
        stamp = msg.get("timestamp")
        exch_ms = parse_iso(stamp) if isinstance(stamp, str) else None
        out: list[FeedEvent] = []
        seq = msg.get("sequence_num")
        if isinstance(seq, int):
            if self._expected_seq is not None and seq != self._expected_seq:
                out.append(FeedNotice(recv_ms, conn_id, "gap", exch_ms, f"expected {self._expected_seq}, got {seq}"))
            self._expected_seq = seq + 1
        channel = msg.get("channel")
        events = msg.get("events") or []
        if channel == "l2_data":
            for event in events:
                if event.get("product_id") == self.product:
                    out.append(self._book_event(event, recv_ms, conn_id, exch_ms))
        elif channel == "market_trades":
            for event in events:
                out.extend(self._trades(event, recv_ms, conn_id, exch_ms))
        elif channel == "heartbeats":
            out.append(FeedNotice(recv_ms, conn_id, "heartbeat", exch_ms))
        elif channel == "subscriptions":
            out.append(FeedNotice(recv_ms, conn_id, "subscribed", exch_ms, json.dumps(events)[:200]))
        return out

    def _book_event(self, event: dict, recv_ms: int, conn_id: int, exch_ms: int | None) -> FeedEvent:
        updates = event.get("updates") or []
        if event.get("type") == "snapshot":
            bids, asks = [], []
            for u in updates:
                level = (float(u["price_level"]), float(u["new_quantity"]))
                (bids if u["side"] == "bid" else asks).append(level)
            return BookSnapshot(recv_ms, conn_id, bids, asks, exch_ms)
        changes = [(SIDES[u["side"]], float(u["price_level"]), float(u["new_quantity"])) for u in updates]
        return BookDelta(recv_ms, conn_id, changes, exch_ms)

    def _trades(self, event: dict, recv_ms: int, conn_id: int, exch_ms: int | None) -> list[FeedEvent]:
        trades = sorted((t for t in event.get("trades") or [] if t.get("product_id") == self.product),
                        key=lambda t: int(t["trade_id"]))
        if event.get("type") == "snapshot":
            if trades:
                self._trade_cursor = int(trades[-1]["trade_id"])
            return []
        out: list[FeedEvent] = []
        for t in trades:
            trade_id = int(t["trade_id"])
            if self._trade_cursor is not None:
                if trade_id <= self._trade_cursor:
                    continue
                if trade_id != self._trade_cursor + 1:
                    out.append(FeedNotice(recv_ms, conn_id, "trade_gap", exch_ms,
                                          f"missing trade ids {self._trade_cursor + 1} to {trade_id - 1}"))
            out.append(Trade(recv_ms, conn_id, trade_id, float(t["price"]), float(t["size"]), exch_ms))
            self._trade_cursor = trade_id
        return out


def create_adapter(feed: FeedConfig) -> CoinbaseAdapter:
    return CoinbaseAdapter(feed.product)
