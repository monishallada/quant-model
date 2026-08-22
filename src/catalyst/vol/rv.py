"""Realized-volatility engine: estimation, diurnal normalization, forecast.

All estimators consume 1-minute CLOSE-to-close log returns of the session so
far (plus prior sessions for fitting). Annualization uses minutes-per-year =
252 x 390. Bipower variation is the jump-robust workhorse; realized variance
is kept as the cross-check (a large RV/BV gap IS the jump signal).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MINUTES_PER_SESSION = 390
ANNUALIZER = 252 * MINUTES_PER_SESSION
_MU1 = math.sqrt(2.0 / math.pi)          # E|Z| for bipower variation


def minute_log_returns(closes: pd.Series) -> pd.Series:
    """Log returns from a minute-close series (NaN rows dropped, not filled)."""
    r = np.log(closes.astype(float)).diff()
    return r.dropna()


def realized_variance(returns: pd.Series) -> float:
    """Sum of squared returns (per-period variance, NOT annualized)."""
    if len(returns) == 0:
        return float("nan")
    return float(np.sum(np.square(returns.to_numpy())))


def bipower_variation(returns: pd.Series) -> float:
    """Barndorff-Nielsen & Shephard bipower variation — jump-robust
    integrated-variance estimate (per-period, NOT annualized)."""
    r = np.abs(returns.to_numpy(dtype=float))
    if len(r) < 2:
        return float("nan")
    return float((1.0 / (_MU1 * _MU1)) * np.sum(r[1:] * r[:-1]))


def jump_ratio(returns: pd.Series) -> float:
    """RV/BV - 1: ≈0 diffusive, >>0 when jumps dominate. NaN-safe."""
    rv, bv = realized_variance(returns), bipower_variation(returns)
    if not (rv == rv and bv == bv) or bv <= 0:
        return float("nan")
    return rv / bv - 1.0


def annualized_vol(per_period_var: float, n_minutes: int) -> float:
    """Annualized vol from a per-period variance over n_minutes of returns."""
    if not per_period_var == per_period_var or n_minutes <= 0 or per_period_var < 0:
        return float("nan")
    per_minute = per_period_var / n_minutes
    return math.sqrt(per_minute * ANNUALIZER)


# ---------------------------------------------------------------------------
# Diurnal (U-curve) seasonality
# ---------------------------------------------------------------------------

def fit_diurnal_curve(minute_returns_by_day: list[pd.Series],
                      n_buckets: int = 26) -> np.ndarray:
    """Median-of-days per-bucket variance share, normalized to mean 1.0.

    Buckets are 15-minute slots (26 x 15 = 390). Median across days (not
    mean) so single crash days don't own the curve. Returns the multiplier
    m[b]: expected variance in bucket b relative to the session average.
    """
    per_day = []
    for r in minute_returns_by_day:
        if len(r) < MINUTES_PER_SESSION // 2:
            continue
        sq = np.square(r.to_numpy(dtype=float))
        # position within session by row order (bars are session-ordered)
        idx = np.linspace(0, n_buckets, num=len(sq), endpoint=False).astype(int)
        tot = sq.sum()
        if tot <= 0:
            continue
        shares = np.bincount(idx, weights=sq, minlength=n_buckets) / tot
        per_day.append(shares)
    if not per_day:
        return np.ones(n_buckets)
    med = np.median(np.vstack(per_day), axis=0)
    med = med / med.mean() if med.mean() > 0 else np.ones(n_buckets)
    return med


def remaining_variance_share(curve: np.ndarray, minute_of_session: int) -> float:
    """Fraction of a typical session's variance still ahead after this minute."""
    n_buckets = len(curve)
    b = min(int(minute_of_session / MINUTES_PER_SESSION * n_buckets), n_buckets - 1)
    total = curve.sum()
    if total <= 0:
        return max(0.0, 1.0 - minute_of_session / MINUTES_PER_SESSION)
    frac_within = (minute_of_session / MINUTES_PER_SESSION * n_buckets) - b
    ahead = curve[b] * (1.0 - frac_within) + curve[b + 1:].sum()
    return float(max(ahead / total, 0.0))


def forecast_horizon_vol(
    session_returns_so_far: pd.Series,
    prior_session_var: float,
    curve: np.ndarray,
    minute_of_session: int,
    horizon_minutes: int,
    blend: float = 0.5,
) -> float:
    """Annualized vol forecast for the NEXT ``horizon_minutes``.

    Blend of (a) today's seasonally-adjusted bipower run-rate and (b) the
    prior-session baseline variance, both projected through the diurnal curve
    over the horizon window. ``blend`` weights today's information; 0.5 is
    the pre-registered default, swept in robustness testing.
    """
    n = len(session_returns_so_far)
    today_bv = bipower_variation(session_returns_so_far)
    # seasonally de-trend today's run rate to a full-session-equivalent var
    elapsed_share = 1.0 - remaining_variance_share(curve, minute_of_session)
    today_session_var = (today_bv / max(elapsed_share, 0.05)
                         if today_bv == today_bv and n >= 15 else float("nan"))
    base = prior_session_var
    if today_session_var == today_session_var and base == base and base > 0:
        session_var = blend * today_session_var + (1 - blend) * base
    elif today_session_var == today_session_var:
        session_var = today_session_var
    elif base == base:
        session_var = base
    else:
        return float("nan")
    ahead_now = remaining_variance_share(curve, minute_of_session)
    ahead_after = remaining_variance_share(
        curve, min(minute_of_session + horizon_minutes, MINUTES_PER_SESSION))
    horizon_var = session_var * max(ahead_now - ahead_after, 0.0)
    if horizon_var <= 0:
        return float("nan")
    # annualize the horizon variance over horizon_minutes
    return annualized_vol(horizon_var, max(horizon_minutes, 1))
