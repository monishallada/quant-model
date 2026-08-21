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
    multiplier: float = 100.0,
) -> int:
    """Whole units such that units × unit_max_loss×multiplier ≤ equity × fraction.

    ``multiplier`` is the instrument's dollars-per-point scale (100 for listed
    options — the historical default, so existing callers are unchanged — and
    1 for shares). Returns 0 when even one unit exceeds the risk budget — the
    trade is skipped, never up-sized.
    """
    # NaN compares False against everything, so 'unit_max_loss <= 0' let NaN
    # flow into floor() and crash mid-session; a zero/negative multiplier is a
    # caller bug that must not size anything (audit D-226).
    if (equity != equity or unit_max_loss != unit_max_loss
            or multiplier != multiplier):
        return 0
    if equity <= 0 or unit_max_loss <= 0 or multiplier <= 0:
        return 0
    budget = equity * per_trade_risk_fraction
    return max(math.floor(budget / (unit_max_loss * multiplier)), 0)
