"""Per-minute smile fitting on a sparse strike window + dislocation scores.

With 10-12 NBBO mids per expiry per minute, a full SVI fit (5 params) is
under-determined and unstable minute-to-minute. The workhorse here is a
weighted QUADRATIC in log-moneyness — 3 parameters with direct readings:

    iv(k) = a + b·k + c·k²      k = ln(K/S)

    a = ATM IV      b = skew      c = curvature (smile)

Weights are inverse relative spread (tight quotes speak louder). The fit is
refused (returns None) when fewer than ``min_points`` usable quotes exist or
the fit is degenerate — no fabricated smiles.

Dislocation: residual of each contract's mid-IV from the fitted smile,
normalized by the cross-sectional residual scale (robust MAD), giving a
z-like score whose distribution the strategy layer tracks point-in-time.
A rich/cheap CONTRACT is a candidate; a rich/cheap SMILE vs the RV forecast
is a different (vega) candidate. Both feed the master EV ranker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SmileFit:
    atm_iv: float          # a: fitted IV at k=0
    skew: float            # b: d(iv)/dk at ATM
    curvature: float       # c
    rmse: float            # weighted fit RMSE (quality gate)
    n_points: int

    def iv_at(self, log_moneyness: float) -> float:
        return (self.atm_iv + self.skew * log_moneyness
                + self.curvature * log_moneyness * log_moneyness)


def fit_smile(
    log_moneyness: np.ndarray,
    mid_iv: np.ndarray,
    rel_spread: np.ndarray,
    min_points: int = 5,
    max_rel_spread: float = 0.5,
) -> SmileFit | None:
    """Weighted least squares quadratic. None when the data can't support it."""
    m = (np.isfinite(log_moneyness) & np.isfinite(mid_iv) & (mid_iv > 0)
         & np.isfinite(rel_spread) & (rel_spread < max_rel_spread))
    k, iv, rs = log_moneyness[m], mid_iv[m], rel_spread[m]
    if len(k) < min_points or np.ptp(k) < 1e-4:
        return None
    w = 1.0 / np.maximum(rs, 0.01)
    X = np.column_stack([np.ones_like(k), k, k * k])

    def _wls(kk, vv, ww):
        Xl = np.column_stack([np.ones_like(kk), kk, kk * kk])
        try:
            beta, *_ = np.linalg.lstsq(Xl * ww[:, None], vv * ww, rcond=None)
        except np.linalg.LinAlgError:
            return None, None
        return beta, vv - Xl @ beta

    beta, resid0 = _wls(k, iv, w)
    if beta is None:
        return None
    # One robust pass (research spec): drop at most 2 gross outliers and
    # refit, so a genuinely dislocated contract cannot drag the benchmark
    # toward itself and poison every NEIGHBOR's score.
    finite = resid0[np.isfinite(resid0)]
    if len(finite) >= min_points + 1:
        mad = np.median(np.abs(finite - np.median(finite)))
        if mad > 0:
            bad = np.abs(resid0) > 3.0 * 1.4826 * mad
            if 0 < bad.sum() <= 2 and (~bad).sum() >= min_points:
                beta2, _ = _wls(k[~bad], iv[~bad], w[~bad])
                if beta2 is not None:
                    beta = beta2
    a, b, c = (float(x) for x in beta)
    if not (0.0 < a < 5.0):
        return None                      # degenerate: refuse, don't fabricate
    resid = iv - X @ beta
    # rmse over the INLIER set: a single dislocation is signal, not fit error
    finite_r = resid[np.isfinite(resid)]
    if len(finite_r) > min_points:
        keep = np.abs(finite_r) <= np.partition(np.abs(finite_r),
                                                len(finite_r) - 2)[-2] + 1e-12
        rmse = float(math.sqrt(np.average(np.square(finite_r[keep])[:],
                                          weights=w[np.isfinite(resid)][keep])))
    else:
        rmse = float(math.sqrt(np.average(np.square(finite_r),
                                          weights=w[np.isfinite(resid)])))
    return SmileFit(atm_iv=a, skew=b, curvature=c, rmse=rmse, n_points=int(len(k)))


def dislocation_scores(
    fit: SmileFit,
    log_moneyness: np.ndarray,
    mid_iv: np.ndarray,
) -> np.ndarray:
    """Residual / robust scale for each contract. Positive = rich vs smile.

    Scale = 1.4826·MAD of the residuals this minute (cross-sectional), floored
    to avoid divide-by-tiny on perfectly smooth smiles. NaN in ⇒ NaN out.
    """
    fitted = np.array([fit.iv_at(k) if k == k else np.nan for k in log_moneyness])
    resid = mid_iv - fitted
    finite = resid[np.isfinite(resid)]
    if len(finite) < 3:
        return np.full_like(resid, np.nan)
    mad = float(np.median(np.abs(finite - np.median(finite))))
    # Scale floor is ECONOMIC, not statistical (research spec): 0.1 vol point
    # or the fit's own inlier RMSE, whichever is larger. Without it, a
    # near-perfect fit (exact synthetic smiles, ultra-tight books) divides
    # noise by ~1e-6 and every contract screams dislocation.
    scale = max(1.4826 * mad, fit.rmse, 1e-3)
    return resid / scale


def forward_variance(
    near_atm_iv: float, near_t_years: float,
    far_atm_iv: float, far_t_years: float,
) -> float:
    """Forward variance between two expiries; negative ⇒ calendar arbitrage
    signal (or bad data — the strategy layer decides which)."""
    if not all(x == x for x in (near_atm_iv, far_atm_iv)):
        return float("nan")
    if far_t_years <= near_t_years or near_t_years < 0:
        return float("nan")
    tot_near = near_atm_iv ** 2 * near_t_years
    tot_far = far_atm_iv ** 2 * far_t_years
    return (tot_far - tot_near) / (far_t_years - near_t_years)
