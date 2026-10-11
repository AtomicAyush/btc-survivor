"""UTC millisecond time helpers and the live clock.

All event times are integer milliseconds since the UTC epoch. The monotonic clock is only
used to measure durations such as how long a minute step took.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone

MS_PER_SECOND = 1000
MS_PER_MINUTE = 60 * MS_PER_SECOND
MS_PER_HOUR = 60 * MS_PER_MINUTE
MS_PER_DAY = 24 * MS_PER_HOUR
MINUTES_PER_DAY = 1440


def minute_floor(ms: int) -> int:
    return ms - ms % MS_PER_MINUTE


def next_minute(ms: int) -> int:
    """The first minute boundary strictly after ms."""
    return minute_floor(ms) + MS_PER_MINUTE


def minute_index(ms: int) -> int:
    return ms // MS_PER_MINUTE


def minute_of_day(ms: int) -> int:
    return (ms // MS_PER_MINUTE) % MINUTES_PER_DAY


def iso(ms: int) -> str:
    """Format as ISO 8601 UTC with milliseconds, so text order matches time order."""
    seconds, millis = divmod(ms, MS_PER_SECOND)
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return f"{dt:%Y-%m-%dT%H:%M:%S}.{millis:03d}Z"


_ISO = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]00:00)$")


def parse_iso(text: str) -> int:
    """Parse an RFC 3339 UTC timestamp with any number of fractional digits, truncated to milliseconds."""
    match = _ISO.match(text)
    if not match:
        raise ValueError(f"not a UTC timestamp: {text!r}")
    whole, frac, _ = match.groups()
    dt = datetime.strptime(whole, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    millis = int((frac or "0")[:3].ljust(3, "0"))
    return int(dt.timestamp()) * MS_PER_SECOND + millis


class LiveClock:
    """Wall clock time for live receive stamps, made non decreasing, plus a monotonic clock for durations."""

    def __init__(self) -> None:
        self._last = 0

    def now_ms(self) -> int:
        return time.time_ns() // 1_000_000

    def stamp(self, floor_ms: int = 0) -> int:
        """A receive stamp that never goes backwards and is never earlier than floor_ms, usually the engine's time."""
        ms = max(self.now_ms(), self._last, floor_ms)
        self._last = ms
        return ms

    @staticmethod
    def monotonic() -> float:
        return time.monotonic()
