"""Labels mature only after the horizon, and only if no feed gap happened in between.

Features and labels are the same for every bot, so the pending queue and the matured window are shared. A label
for minute t is 1e4 times the log return of the minute mid from t to t plus the horizon.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from btc_survivor.clock import MS_PER_MINUTE


@dataclass(frozen=True, slots=True)
class Pending:
    minute_ms: int
    epoch: int
    z: np.ndarray
    entry_mid: float


@dataclass(frozen=True, slots=True)
class Matured:
    minute_ms: int
    z: np.ndarray
    label_bps: float


class LabelQueue:
    def __init__(self, horizon_minutes: int) -> None:
        self.horizon_ms = horizon_minutes * MS_PER_MINUTE
        self.pending: deque[Pending] = deque()
        self.discarded = 0

    def push(self, minute_ms: int, epoch: int, z: np.ndarray, entry_mid: float) -> None:
        self.pending.append(Pending(minute_ms, epoch, z, entry_mid))

    def mature(self, now_ms: int, epoch: int, mid_now: float) -> Matured | None:
        """Return the sample whose horizon ends exactly now, if it is from the current feed epoch."""
        result = None
        while self.pending and self.pending[0].minute_ms + self.horizon_ms <= now_ms:
            record = self.pending.popleft()
            if record.minute_ms + self.horizon_ms == now_ms and record.epoch == epoch:
                result = Matured(record.minute_ms, record.z, 1e4 * math.log(mid_now / record.entry_mid))
            else:
                self.discarded += 1
        return result

    def drop_other_epochs(self, epoch: int) -> None:
        """After a gap nothing pending can mature, so forget it at once."""
        while self.pending and self.pending[0].epoch != epoch:
            self.pending.popleft()
            self.discarded += 1


class MaturedWindow:
    """The last capacity matured samples, which every bot draws its replay batches from."""

    def __init__(self, capacity: int, n_features: int) -> None:
        self.capacity = capacity
        self.z = np.zeros((capacity, n_features))
        self.label = np.zeros(capacity)
        self.minute = np.zeros(capacity, dtype=np.int64)
        self.count = 0
        self.head = 0

    def add(self, sample: Matured) -> None:
        self.z[self.head] = sample.z
        self.label[self.head] = sample.label_bps
        self.minute[self.head] = sample.minute_ms
        self.head = (self.head + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def newest(self) -> int:
        return (self.head - 1) % self.capacity

    def batch(self, rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
        """The newest sample plus size minus one draws with replacement from the window."""
        rows = [self.newest()]
        if size > 1:
            rows.extend(int(i) for i in rng.integers(0, self.count, size - 1))
        return self.z[rows], self.label[rows]
