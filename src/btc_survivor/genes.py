"""The two evolvable genes: the extra edge a bot demands beyond costs, and its learning rate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from btc_survivor.config import GenesConfig


@dataclass(frozen=True)
class Genes:
    margin_bps: float
    lr: float

    def clamped(self, cfg: GenesConfig) -> "Genes":
        return Genes(
            margin_bps=min(max(self.margin_bps, cfg.margin_bps_min), cfg.margin_bps_max),
            lr=min(max(self.lr, cfg.lr_min), cfg.lr_max),
        )


def initial_genes(cfg: GenesConfig) -> Genes:
    return Genes(cfg.margin_bps_init, cfg.lr_init)


def mutate(parent: Genes, cfg: GenesConfig, sigma: float, rng: np.random.Generator) -> tuple[Genes, float, float]:
    """Multiply each gene by a lognormal factor and clamp. Returns the child genes and both factors."""
    margin_factor = float(np.exp(rng.normal(0.0, sigma)))
    lr_factor = float(np.exp(rng.normal(0.0, sigma)))
    child = Genes(parent.margin_bps * margin_factor, parent.lr * lr_factor).clamped(cfg)
    return child, margin_factor, lr_factor
