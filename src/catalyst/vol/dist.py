"""Conditional short-horizon return distributions.

Empirical conditional quantiles of forward log returns, conditioned on a
small, pre-registered state space:

    time-of-day bucket (default 6)  ×  vol state (3)  ×  trend state (3)

54 cells; ~1,900 sessions × ~370 usable minutes ≈ 700k observations in a
full training window → thousands per cell. Few enough cells to be honest,
many enough observations to be stable. Fitting is strictly point-in-time:
the table is built from history the strategy layer hands in (visible-only).

Two hard rules baked in here:
- a cell below ``min_obs`` falls back to its time-of-day marginal, and a
  time-of-day marginal below ``min_obs`` returns None (NO fabricated
  distributions);
- quantiles are of VOL-SCALED returns (return / expected horizon vol), so
  the table stores the SHAPE and the current RV forecast supplies the scale.
  This is what makes a table fitted on calm months usable in a storm without
  pretending the storm was in-sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


@dataclass(frozen=True)
class StateSpec:
    n_tod_buckets: int = 6
    vol_edges: tuple[float, float] = (0.75, 1.5)    # ratio to trailing median vol
    trend_edges: tuple[float, float] = (-0.5, 0.5)  # 30-min return / horizon vol
    min_obs: int = 300


@dataclass
class ConditionalQuantiles:
    """The fitted table: (tod, vol, trend) -> scaled-return quantiles."""

    spec: StateSpec
    table: dict[tuple[int, int, int], np.ndarray] = field(default_factory=dict)
    tod_marginal: dict[int, np.ndarray] = field(default_factory=dict)
    n_obs: int = 0

    def lookup(self, tod: int, vol: int, trend: int) -> np.ndarray | None:
        q = self.table.get((tod, vol, trend))
        if q is not None:
            return q
        return self.tod_marginal.get(tod)


def classify_vol(current_vol: float, trailing_median_vol: float,
                 spec: StateSpec) -> int:
    if not (current_vol == current_vol and trailing_median_vol == trailing_median_vol
            and trailing_median_vol > 0):
        return 1
    ratio = current_vol / trailing_median_vol
    lo, hi = spec.vol_edges
    return 0 if ratio < lo else (2 if ratio > hi else 1)


def classify_trend(recent_return: float, horizon_vol_return_scale: float,
                   spec: StateSpec) -> int:
    if not (recent_return == recent_return and horizon_vol_return_scale > 0):
        return 1
    z = recent_return / horizon_vol_return_scale
    lo, hi = spec.trend_edges
    return 0 if z < lo else (2 if z > hi else 1)


def tod_bucket(minute_of_session: int, spec: StateSpec,
               minutes_per_session: int = 390) -> int:
    b = int(minute_of_session / minutes_per_session * spec.n_tod_buckets)
    return min(max(b, 0), spec.n_tod_buckets - 1)


def fit(
    scaled_returns: np.ndarray,
    tod: np.ndarray,
    vol_state: np.ndarray,
    trend_state: np.ndarray,
    spec: StateSpec = StateSpec(),
) -> ConditionalQuantiles:
    """Fit the table from aligned observation arrays.

    ``scaled_returns``: forward horizon log-return / expected horizon vol at
    observation time (the strategy layer computes both point-in-time).
    """
    mask = np.isfinite(scaled_returns)
    scaled, tod, vol_state, trend_state = (scaled_returns[mask], tod[mask],
                                           vol_state[mask], trend_state[mask])
    out = ConditionalQuantiles(spec=spec, n_obs=int(len(scaled)))
    for t in range(spec.n_tod_buckets):
        sel_t = tod == t
        if sel_t.sum() >= spec.min_obs:
            out.tod_marginal[t] = np.quantile(scaled[sel_t], QUANTILES)
        for v in range(3):
            for tr in range(3):
                sel = sel_t & (vol_state == v) & (trend_state == tr)
                if sel.sum() >= spec.min_obs:
                    out.table[(t, v, tr)] = np.quantile(scaled[sel], QUANTILES)
    return out


def coverage_test(
    fitted: ConditionalQuantiles,
    scaled_returns: np.ndarray,
    tod: np.ndarray,
    vol_state: np.ndarray,
    trend_state: np.ndarray,
) -> dict[str, float]:
    """Out-of-sample calibration: empirical frequency below each fitted
    quantile. Well-calibrated ⇒ frequency ≈ nominal. Returns per-quantile
    coverage plus the max absolute miscoverage (the headline number)."""
    hits = {q: [] for q in QUANTILES}
    for r, t, v, tr in zip(scaled_returns, tod, vol_state, trend_state):
        if not r == r:
            continue
        q = fitted.lookup(int(t), int(v), int(tr))
        if q is None:
            continue
        for nominal, cut in zip(QUANTILES, q):
            hits[nominal].append(1.0 if r <= cut else 0.0)
    out: dict[str, float] = {}
    worst = 0.0
    for nominal in QUANTILES:
        if hits[nominal]:
            cov = float(np.mean(hits[nominal]))
            out[f"cov_{nominal}"] = cov
            worst = max(worst, abs(cov - nominal))
    out["max_miscoverage"] = worst
    out["n_scored"] = float(len(hits[QUANTILES[0]]))
    return out
