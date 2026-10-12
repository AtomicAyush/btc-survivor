"""The Finnhub trades websocket of PLAN_V2 section 4.2, with a pure parse into print batches and notices.

The token travels in the URL, so the URL is built in memory from the token, the token may hold only characters a URL
carries unchanged, so redact always finds it whole, and the repr never shows it. The field names s, p, v, t and c, the
ping type and the absence of a trade id come from Finnhub's docs and are unverified until the first recording.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from btc_survivor.feed.base import FeedAdapter
from btc_survivor.types import FeedNotice
from btc_survivor.v2.redact import redact

log = logging.getLogger(__name__)

URL = "wss://ws.finnhub.io"
MAX_MESSAGE_BYTES = 4 * 1024 * 1024
ERROR_DETAIL_CHARS = 200
UNREADABLE = "unreadable"

# Drop reasons, counted on the adapter.
UNSUBSCRIBED, BAD_PRICE, BAD_SIZE, MALFORMED = "unsubscribed", "price", "size", "malformed"

_TOKEN = re.compile(r"[A-Za-z0-9_-]+")
_TYPE_NAME = re.compile(r"[a-z][a-z_]{0,31}")


class Print(NamedTuple):
    symbol: str
    price: float
    size: float
    vendor_ms: int
    conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PrintBatch:
    """The prints of one trade message in vendor time order, then message order, with the message's receive stamp."""

    recv_ms: int
    conn_id: int
    prints: tuple[Print, ...]


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _conditions(value: Any) -> tuple[str, ...] | None:
    """The condition codes as text, no codes when c is missing or null, or None when c is not a list of codes."""
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(c, (str, int)) and not isinstance(c, bool) for c in value):
        return None
    return tuple(str(c) for c in value)


class FinnhubAdapter(FeedAdapter):
    max_message_bytes = MAX_MESSAGE_BYTES

    def __init__(self, symbols: Sequence[str], token: str, base_url: str = URL) -> None:
        if not symbols:
            raise ValueError("the Finnhub adapter needs at least one symbol")
        if not isinstance(token, str) or not _TOKEN.fullmatch(token):
            raise ValueError("the Finnhub token may hold only letters, digits, underscores and hyphens")
        self.symbols = tuple(symbols)
        self._subscribed = frozenset(self.symbols)
        self._token = token
        self._base_url = base_url
        self.dropped: Counter[str] = Counter()
        self.unknown: Counter[str] = Counter()
        self._unknown_seen: set[str] = set()

    def __repr__(self) -> str:
        return f"FinnhubAdapter(symbols={len(self.symbols)}, base_url={self._base_url!r})"

    @property
    def url(self) -> str:
        """The websocket URL with the token, built in memory each time and never logged."""
        return f"{self._base_url}?token={self._token}"

    def subscribe_frame(self, symbol: str) -> str:
        return json.dumps({"type": "subscribe", "symbol": symbol}, separators=(",", ":"))

    def subscribe_messages(self) -> list[str]:
        """One subscribe frame per symbol in config order, sent after each connect without waiting for a reply."""
        return [self.subscribe_frame(symbol) for symbol in self.symbols]

    def on_open(self, conn_id: int) -> None:
        self._unknown_seen = set()

    def parse(self, raw: str, recv_ms: int, conn_id: int) -> list[PrintBatch | FeedNotice]:
        """Turn one raw message into at most one batch or notice. A ping needs no reply and an error asks the caller
        for a reconnect, while drops and unknown types are only counted."""
        try:
            msg = json.loads(raw)
        except (ValueError, RecursionError):
            msg = None
        kind = msg.get("type") if isinstance(msg, dict) else None
        if kind == "trade":
            return self._trades(msg.get("data"), recv_ms, conn_id)
        if kind == "ping":
            return [FeedNotice(recv_ms, conn_id, "ping")]
        if kind == "error":
            return [FeedNotice(recv_ms, conn_id, "error", detail=redact(str(msg.get("msg", "")))[:ERROR_DETAIL_CHARS])]
        self._count_unknown(kind, conn_id)
        return []

    def _trades(self, data: Any, recv_ms: int, conn_id: int) -> list[PrintBatch | FeedNotice]:
        """One batch of the valid prints, sorted stably by vendor time, or nothing when every print was dropped."""
        if not isinstance(data, list):
            self.dropped[MALFORMED] += 1
            return []
        prints = []
        for item in data:
            result = self._print(item)
            if isinstance(result, str):
                self.dropped[result] += 1
            else:
                prints.append(result)
        if not prints:
            return []
        return [PrintBatch(recv_ms, conn_id, tuple(sorted(prints, key=lambda p: p.vendor_ms)))]

    def _print(self, item: Any) -> Print | str:
        """One print from a data entry, or the reason it is dropped. An integer too large for a float is malformed."""
        if not isinstance(item, dict):
            return MALFORMED
        symbol, price, size, vendor_ms = item.get("s"), item.get("p"), item.get("v"), item.get("t")
        conditions = _conditions(item.get("c"))
        try:
            if (not isinstance(symbol, str) or not _number(price) or not _number(size) or not _number(vendor_ms)
                    or not math.isfinite(vendor_ms) or conditions is None):
                return MALFORMED
            if symbol not in self._subscribed:
                return UNSUBSCRIBED
            if not math.isfinite(price) or price <= 0:
                return BAD_PRICE
            if not math.isfinite(size) or size <= 0:
                return BAD_SIZE
            return Print(symbol, float(price), float(size), int(vendor_ms), conditions)
        except OverflowError:
            return MALFORMED

    def _count_unknown(self, kind: Any, conn_id: int) -> None:
        """Count a message of any other type, logging each type name once per connection, and only a plain name."""
        name = kind if isinstance(kind, str) and _TYPE_NAME.fullmatch(kind) else UNREADABLE
        self.unknown[name] += 1
        if name not in self._unknown_seen:
            self._unknown_seen.add(name)
            log.warning("Finnhub message of type %s on connection %d, counted from now on", name, conn_id)
