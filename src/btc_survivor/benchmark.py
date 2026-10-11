"""Context curves recorded next to the population: buy and hold, an exposure matched version, and cash."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Anchor:
    """The launch moment that every curve starts from."""

    ms: int
    ask: float
    mid: float


def buy_hold_equity(anchor: Anchor, bid: float, start_equity: float, fee_bps: float) -> float:
    """All of start equity bought at the launch ask paying one taker fee, marked at today's bid."""
    qty = start_equity * (1.0 - fee_bps / 1e4) / anchor.ask
    return qty * bid


def matched_equity(anchor: Anchor, bid: float, start_equity: float, notional: float, fee_bps: float) -> float:
    """Only the trading notional bought at launch, the rest kept as cash."""
    qty = notional * (1.0 - fee_bps / 1e4) / anchor.ask
    return start_equity - notional + qty * bid


def cash_equity(start_equity: float, daily_rent: float, healthy_minutes: int) -> float:
    """The bot that never trades: start equity minus the rent of every healthy minute since launch."""
    return start_equity - daily_rent / 1440.0 * healthy_minutes
