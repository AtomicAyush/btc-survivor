"""The linear baseline, an online ridge regression on the shared z scores that is scored next to the population.

It learns from the same matured samples as the bots, with the standardizer's warmup weighting and half life, so it
shows whether the small networks add anything over a one line rule. It only ever sees labels that have matured.
"""

from __future__ import annotations

import numpy as np

from btc_survivor.standardizer import alpha_for_half_life

PRIOR_SAMPLES = 1000.0  # the ridge counts as this many samples of zero, so early fits shrink toward zero


class LinearBaseline:
    def __init__(self, n_features: int, half_life_samples: float) -> None:
        self.alpha = alpha_for_half_life(half_life_samples)
        self.n = 0
        size = n_features + 1
        self.xx = np.zeros((size, size))
        self.xy = np.zeros(size)
        self.coef = np.zeros(size)
        self.weight = 1.0

    def update(self, z: np.ndarray, label_bps: float) -> None:
        """Add one matured sample with weight max(alpha, 1/n) and refit with a prior worth PRIOR_SAMPLES samples."""
        self.n += 1
        self.weight = max(self.alpha, 1.0 / self.n)
        x = np.r_[1.0, z]
        self.xx = (1.0 - self.weight) * self.xx + self.weight * np.outer(x, x)
        self.xy = (1.0 - self.weight) * self.xy + self.weight * label_bps * x
        self.coef = np.linalg.solve(self.xx + PRIOR_SAMPLES * self.weight * np.eye(len(x)), self.xy)

    def predict(self, z: np.ndarray) -> float:
        """The predicted label in bps for the current z scores."""
        return float(self.coef @ np.r_[1.0, z])
