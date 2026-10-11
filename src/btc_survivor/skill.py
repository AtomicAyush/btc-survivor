"""Honest out of sample skill. Each prediction is scored against its label before that label trains anything."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_PAIRS = 30


@dataclass(frozen=True, slots=True)
class SkillStats:
    corr: float | None
    dir_acc: float | None
    mae: float | None
    n: int


class RollingSkill:
    """Ring buffers of the last window prediction and label pairs, in bps."""

    def __init__(self, window: int) -> None:
        self.window = window
        self.pred = np.zeros(window)
        self.label = np.zeros(window)
        self.n = 0
        self.head = 0

    def add(self, pred_bps: float, label_bps: float) -> None:
        self.pred[self.head] = pred_bps
        self.label[self.head] = label_bps
        self.head = (self.head + 1) % self.window
        self.n = min(self.n + 1, self.window)

    @property
    def full(self) -> bool:
        return self.n == self.window

    def stats(self) -> SkillStats:
        if self.n == 0:
            return SkillStats(None, None, None, 0)
        p, y = self.pred[:self.n], self.label[:self.n]
        mae = float(np.mean(np.abs(p - y)))
        both = (p != 0) & (y != 0)
        dir_acc = float(np.mean(np.sign(p[both]) == np.sign(y[both]))) if both.any() else None
        corr = None
        if self.n >= MIN_PAIRS and np.std(p) > 0 and np.std(y) > 0:
            corr = float(np.corrcoef(p, y)[0, 1])
        return SkillStats(corr, dir_acc, mae, self.n)
