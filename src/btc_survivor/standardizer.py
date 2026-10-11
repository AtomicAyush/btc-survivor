"""Running statistics shared by all bots: z scores for features and the scale of the labels.

Each new sample gets weight max(alpha, 1/n). Early on that is a plain running mean and variance, and after about
1/alpha samples it becomes exponentially weighted with the configured half life, so the first weeks are not biased.
"""

from __future__ import annotations

import numpy as np

STD_FLOOR = 1e-9


def alpha_for_half_life(half_life_samples: float) -> float:
    return 1.0 - 0.5 ** (1.0 / half_life_samples)


class RunningStats:
    def __init__(self, size: int, half_life_samples: float) -> None:
        self.alpha = alpha_for_half_life(half_life_samples)
        self.n = 0
        self.mean = np.zeros(size)
        self.var = np.zeros(size)

    def update(self, x: np.ndarray) -> None:
        """Weighted incremental mean and variance in the non negative West form."""
        self.n += 1
        weight = max(self.alpha, 1.0 / self.n)
        delta = x - self.mean
        self.mean = self.mean + weight * delta
        self.var = (1.0 - weight) * (self.var + weight * delta * delta)

    @property
    def std(self) -> np.ndarray:
        return np.maximum(np.sqrt(np.maximum(self.var, 0.0)), STD_FLOOR)


class Standardizer:
    """Turns raw features into z scores using only statistics from before the current sample."""

    def __init__(self, size: int, half_life_minutes: float, min_samples: int, clip: float) -> None:
        self.stats = RunningStats(size, half_life_minutes)
        self.min_samples = min_samples
        self.clip = clip

    def transform_then_update(self, x: np.ndarray) -> np.ndarray:
        """Zero until min_samples samples have been seen, then clipped z scores. The sample is added afterwards."""
        if self.stats.n < self.min_samples:
            z = np.zeros_like(x)
        else:
            z = np.clip((x - self.stats.mean) / self.stats.std, -self.clip, self.clip)
        self.stats.update(x)
        return z


class LabelScale:
    """The spread of matured labels in bps. The network learns label divided by this scale."""

    def __init__(self, half_life_minutes: float, min_samples: int) -> None:
        self.stats = RunningStats(1, half_life_minutes)
        self.min_samples = min_samples

    def update(self, label_bps: float) -> None:
        self.stats.update(np.array([label_bps]))

    @property
    def ready(self) -> bool:
        return self.stats.n >= self.min_samples

    @property
    def scale(self) -> float:
        return float(self.stats.std[0])
