"""Payoff transform: underlying quantile grid → option P&L distribution.

The master ranker needs, per candidate structure, the distribution of P&L at
the expected exit horizon — not a point estimate. Procedure:

1. The conditional-distribution engine supplies horizon return quantiles
   (7 points). We interpolate them to a 15-node scenario grid with tail
   extension (the 5%/95% points extended linearly to 1%/99% equivalents).
2. Each scenario reprices every leg with Black-Scholes at the horizon:
   S → S·e^r, t → t − h, and IV mapped by the chosen surface dynamic:
     - sticky_strike: leg keeps ITS OWN current IV (per-strike)
     - sticky_delta:  leg IV follows the CURRENT smile evaluated at the NEW
       log-moneyness (smile rides with spot)
   Short intraday horizons empirically sit between the two; which one the
   strategy uses is a fitted/validated choice, and the gap between them is a
   model-risk band the EV gate must clear.
3. Exit pricing is at the modeled EXECUTION side (the caller passes the cost
   model), so EV is net of exit friction; entry friction is charged by the
   caller at entry evaluation.

Everything is pure functions of explicit inputs; ~50 candidates × 15 nodes ×
2 legs ≈ 1.5k BS evaluations per decision minute — microseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from catalyst.core.types import OptionRight
from catalyst.data.black_scholes import bs_price

GRID_PROBS = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                       0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
#: probability mass of each node (trapezoid over GRID_PROBS)
_EDGES = np.concatenate([[0.0], (GRID_PROBS[:-1] + GRID_PROBS[1:]) / 2, [1.0]])
NODE_WEIGHTS = np.diff(_EDGES)


@dataclass(frozen=True)
class LegSpec:
    right: OptionRight
    strike: float
    t_years: float          # time to expiry NOW
    iv: float               # this leg's CURRENT mid IV
    qty: int                # signed: +long, -short
    entry_price: float      # per-share premium PAID (+) / received (−) per unit


@dataclass(frozen=True)
class PayoffStats:
    ev: float               # expected P&L per unit structure, net of exit cost
    p_profit: float
    p05: float              # 5th percentile P&L (downside)
    p95: float
    worst: float            # worst grid node (≈99% adverse)
    node_pnl: tuple         # full grid for downstream use


def scenario_returns(quantiles7: np.ndarray, horizon_vol_scale: float) -> np.ndarray:
    """7 fitted scaled-return quantiles → 13-node return grid (with linear
    tail extension to the 1%/99% nodes)."""
    q = np.asarray(quantiles7, dtype=float) * horizon_vol_scale
    inner_probs = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    # linear tail extension using the outermost slope
    lo_slope = (q[1] - q[0]) / (inner_probs[1] - inner_probs[0])
    hi_slope = (q[6] - q[5]) / (inner_probs[6] - inner_probs[5])
    grid = np.interp(GRID_PROBS, inner_probs, q)
    grid[0] = q[0] - lo_slope * (inner_probs[0] - GRID_PROBS[0])
    grid[-1] = q[6] + hi_slope * (GRID_PROBS[-1] - inner_probs[6])
    return grid


def _leg_value(leg: LegSpec, spot: float, ret: float, horizon_years: float,
               smile_iv_at, dynamics: str) -> float:
    s_h = spot * float(np.exp(ret))
    t_h = max(leg.t_years - horizon_years, 0.0)
    if dynamics == "sticky_delta" and smile_iv_at is not None:
        k_new = float(np.log(leg.strike / s_h))
        iv_h = max(float(smile_iv_at(k_new)), 0.01)
    else:                                   # sticky_strike
        iv_h = leg.iv
    if t_h <= 0:
        intrinsic = (max(s_h - leg.strike, 0.0) if leg.right is OptionRight.CALL
                     else max(leg.strike - s_h, 0.0))
        return intrinsic
    return bs_price(s_h, leg.strike, t_h, iv_h, leg.right, r=0.0)


def evaluate(
    legs: list[LegSpec],
    spot: float,
    quantiles7: np.ndarray,
    horizon_vol_scale: float,
    horizon_minutes: float,
    exit_cost_per_unit: float,
    smile_iv_at=None,
    dynamics: str = "sticky_strike",
) -> PayoffStats | None:
    """P&L distribution of the structure over the horizon.

    ``exit_cost_per_unit``: modeled friction to CLOSE one unit (per share),
    charged in every scenario. Returns None on non-finite inputs — the ranker
    treats that as "cannot price", never as zero risk.
    """
    if not (spot > 0 and horizon_vol_scale == horizon_vol_scale
            and np.all(np.isfinite(quantiles7))):
        return None
    horizon_years = horizon_minutes / (252.0 * 390.0)
    rets = scenario_returns(quantiles7, horizon_vol_scale)
    pnl = np.zeros(len(rets))
    for leg in legs:
        if not (leg.iv == leg.iv and leg.iv > 0 and leg.strike > 0):
            return None
        vals = np.array([_leg_value(leg, spot, r_, horizon_years,
                                    smile_iv_at, dynamics) for r_ in rets])
        pnl += leg.qty * (vals - leg.entry_price)
    pnl -= exit_cost_per_unit
    ev = float(np.dot(pnl, NODE_WEIGHTS))
    p_profit = float(np.dot((pnl > 0).astype(float), NODE_WEIGHTS))
    return PayoffStats(
        ev=ev, p_profit=p_profit,
        p05=float(np.interp(0.05, GRID_PROBS, np.sort(pnl))),
        p95=float(np.interp(0.95, GRID_PROBS, np.sort(pnl))),
        worst=float(pnl.min()),
        node_pnl=tuple(float(x) for x in pnl),
    )
