"""Session bounded labels of PLAN_V2 sections 4.7, 6.2 and 10.4.

A training record of one symbol rides the version 1 LabelQueue, stamped with the session minute index k times 60,000,
so a night, weekend or holiday adds no minutes. It is pushed at k only when k plus the horizon is at or before the
session's last minute, so the minutes ending 09:31 to 15:30 push records and a label never crosses the close, and it
matures only if the symbol's epoch is unchanged, otherwise it is discarded. The label is 1e4 times the log return of the
reference price over the horizon.

The evaluation labels of the stock_labels table come one per evaluation minute and symbol, session minutes 30, 60 and
so on while the minute plus the horizon stays within the session, 12 in a full session and 6 on an early close. Each
matures only if both symbol minutes were healthy in one epoch and otherwise ends as gap. Both rules read price health
alone, never news health.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from btc_survivor.clock import MS_PER_MINUTE
from btc_survivor.labels import LabelQueue, Matured
from btc_survivor.standardizer import LabelScale
from btc_survivor.v2 import sessions
from btc_survivor.v2.config import FeaturesConfig
from btc_survivor.v2.stocks.candles import StockCandle

PENDING, MATURED, GAP = "pending", "matured", "gap"


def label_bps(entry: float, exit_price: float) -> float:
    """The label in bps, the same expression as the version 1 LabelQueue."""
    return 1e4 * math.log(exit_price / entry)


def stamp(k: int) -> int:
    """The minute stamp the version 1 LabelQueue keeps for session minute index k."""
    return k * MS_PER_MINUTE


def push_k(sample: Matured) -> int:
    """The session minute index at which a matured sample's record was pushed."""
    return sample.minute_ms // MS_PER_MINUTE


def fits(k: int, horizon_minutes: int) -> bool:
    """Whether a record pushed at k ends at or before the close of its session."""
    return k + horizon_minutes <= sessions.session_of_k(k).last_k


def is_evaluation_minute(k: int, horizon_minutes: int) -> bool:
    """Whether k is an evaluation minute, a multiple of the horizon whose horizon ends within the session."""
    session = sessions.session_of_k(k)
    return k - session.first_k + 1 in session.evaluation_minutes(horizon_minutes)


class SessionLabelQueue:
    """One symbol's pending training records on the version 1 LabelQueue."""

    def __init__(self, horizon_minutes: int) -> None:
        self.horizon = horizon_minutes
        self.queue = LabelQueue(horizon_minutes)

    @property
    def discarded(self) -> int:
        return self.queue.discarded

    def __len__(self) -> int:
        return len(self.queue.pending)

    def push(self, k: int, epoch: int, z: np.ndarray, entry_price: float) -> bool:
        """Push a record at k unless its horizon would cross the close, and say whether it was pushed."""
        if not fits(k, self.horizon):
            return False
        self.queue.push(stamp(k), epoch, z, entry_price)
        return True

    def mature(self, k: int, epoch: int, price: float | None) -> Matured | None:
        """The sample pushed at k minus the horizon when the epoch is unchanged, discarding every record that is due
        but cannot mature, including all of them when there is no reference price at k."""
        if price is None:
            while self.queue.pending and self.queue.pending[0].minute_ms + self.queue.horizon_ms <= stamp(k):
                self.queue.pending.popleft()
                self.queue.discarded += 1
            return None
        return self.queue.mature(stamp(k), epoch, price)

    def drop_other_epochs(self, epoch: int) -> None:
        """Forget at once every record from another epoch, as after a gap."""
        self.queue.drop_other_epochs(epoch)


@dataclass(frozen=True)
class LabelRow:
    """One stock_labels row, ts the boundary of its evaluation minute."""

    ts: int
    symbol: str
    entry_price: float
    exit_price: float | None
    label_bps: float | None
    status: str


@dataclass(frozen=True)
class _Open:
    row: LabelRow
    exit_k: int
    healthy: bool
    epoch: int


class EvaluationLabels:
    """One symbol's stock_labels rows, a pending row at each evaluation minute with a reference price, and its final
    row at the minute its horizon ends, or at the first minute after it this symbol is stepped."""

    def __init__(self, symbol: str, horizon_minutes: int) -> None:
        self.symbol = symbol
        self.horizon = horizon_minutes
        self._open: list[_Open] = []

    @property
    def pending(self) -> int:
        return len(self._open)

    def step(self, k: int, ref_price: float | None, healthy: bool, epoch: int) -> list[LabelRow]:
        """The rows that end or start at session minute k, final rows first."""
        out = []
        while self._open and self._open[0].exit_k <= k:
            done = self._open.pop(0)
            ok = done.exit_k == k and done.healthy and healthy and ref_price is not None and epoch == done.epoch
            out.append(_final(done.row, ref_price if ok else None))
        if ref_price is not None and is_evaluation_minute(k, self.horizon):
            row = LabelRow(sessions.boundary_of_k(k), self.symbol, ref_price, None, None, PENDING)
            self._open.append(_Open(row, k + self.horizon, healthy, epoch))
            out.append(row)
        return out


def _final(row: LabelRow, exit_price: float | None) -> LabelRow:
    """The matured row at exit_price, or the gap row when there is none."""
    if exit_price is None:
        return LabelRow(row.ts, row.symbol, row.entry_price, None, None, GAP)
    return LabelRow(row.ts, row.symbol, row.entry_price, exit_price, label_bps(row.entry_price, exit_price), MATURED)


def recompute_labels(candles: Iterable[StockCandle], horizon_minutes: int) -> list[LabelRow]:
    """The final stock_labels rows of one symbol from its stored candles alone, as evaluate checks them. A row matures
    when every minute from its evaluation minute to the end of its horizon is valid, which is how a healthy symbol's
    epoch stays unchanged, since every exit from healthy leaves at least one minute not healthy."""
    by_k = {c.k: c for c in candles}
    rows = []
    for k in sorted(by_k):
        entry = by_k[k]
        if entry.ref_price is None or not is_evaluation_minute(k, horizon_minutes):
            continue
        row = LabelRow(entry.ts, entry.symbol, entry.ref_price, None, None, PENDING)
        span = [by_k.get(j) for j in range(k, k + horizon_minutes + 1)]
        ok = all(c is not None and c.valid and c.ref_price is not None for c in span)
        rows.append(_final(row, span[-1].ref_price if ok else None))
    return rows


def make_label_scale(cfg: FeaturesConfig) -> LabelScale:
    """One symbol's version 1 LabelScale, with the standardizer's half life in healthy session minutes."""
    return LabelScale(cfg.standardizer_half_life_hours * 60, cfg.min_samples)
