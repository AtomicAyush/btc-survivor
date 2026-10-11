"""The two evolvable genes: the extra edge a bot demands beyond costs, and its learning rate."""

from __future__ import annotations

from dataclasses import dataclass

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
