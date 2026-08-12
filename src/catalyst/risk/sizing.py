"""Position sizing: fixed-fractional only.

Kelly (hard-capped fractional) is deliberately NOT implemented — it is a
phase-2 item gated on the backtest producing reliable per-engine win/payoff
stats, per the governing spec.

Sizing never sees performance targets: its only inputs are current equity,
the engine's configured risk fraction, and the structure's max loss.
"""

from __future__ import annotations

import math


def fixed_fractional_units(
    equity: float,
    per_trade_risk_fraction: float,
    unit_max_loss: float,
) -> int:
    """Whole units such that units × unit_max_loss×100 ≤ equity × fraction.

    Returns 0 when even one unit exceeds the risk budget — the trade is
    skipped, never up-sized.
    """
    if equity <= 0 or unit_max_loss <= 0:
        return 0
    budget = equity * per_trade_risk_fraction
    return max(math.floor(budget / (unit_max_loss * 100.0)), 0)
