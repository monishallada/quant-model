"""Black-Scholes pricing, greeks, and implied volatility.

Fallback path only: ThetaData v3 serves greeks + IV directly (verified at M0);
this module covers contracts/timestamps where served greeks are missing, and
provides the pricing kernel for tests and the fill-model sanity checks.

Conventions: annualized vol/rates, ``t`` in years, continuous compounding.
Theta is returned per calendar day, vega and rho per 1.0 change (100 vol pts /
100 rate pts) divided to per-point to match vendor conventions.
"""

from __future__ import annotations

import math

from catalyst.core.models import Greeks, OptionRight

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _d1_d2(s: float, k: float, t: float, r: float, sigma: float, q: float) -> tuple[float, float]:
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    return d1, d1 - sigma * math.sqrt(t)


def bs_price(
    s: float, k: float, t: float, sigma: float, right: OptionRight, r: float = 0.0, q: float = 0.0
) -> float:
    """Black-Scholes(-Merton) price.

    Degenerate limits: at t<=0, plain intrinsic; at sigma<=0 with t>0, the
    *discounted forward* intrinsic (the true European sigma->0 limit — using
    raw intrinsic there would misprice deep-ITM low-IV puts and break the IV
    solver's no-arbitrage lower bound).
    """
    if t <= 0:
        intrinsic = s - k if right is OptionRight.CALL else k - s
        return max(intrinsic, 0.0)
    if sigma <= 0:
        fwd = s * math.exp(-q * t) - k * math.exp(-r * t)
        return max(fwd if right is OptionRight.CALL else -fwd, 0.0)
    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    if right is OptionRight.CALL:
        return s * math.exp(-q * t) * _norm_cdf(d1) - k * math.exp(-r * t) * _norm_cdf(d2)
    return k * math.exp(-r * t) * _norm_cdf(-d2) - s * math.exp(-q * t) * _norm_cdf(-d1)


def bs_greeks(
    s: float, k: float, t: float, sigma: float, right: OptionRight, r: float = 0.0, q: float = 0.0
) -> Greeks:
    """Full first-order greeks (+gamma) at the given vol."""
    if t <= 0 or sigma <= 0:
        # Expired/degenerate: delta is a step function, everything else 0.
        if right is OptionRight.CALL:
            delta = 1.0 if s > k else 0.0
        else:
            delta = -1.0 if s < k else 0.0
        return Greeks(delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0, iv=max(sigma, 0.0))
    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    pdf_d1 = _norm_pdf(d1)
    sqrt_t = math.sqrt(t)
    disc_q = math.exp(-q * t)
    disc_r = math.exp(-r * t)

    if right is OptionRight.CALL:
        delta = disc_q * _norm_cdf(d1)
        theta_annual = (
            -s * disc_q * pdf_d1 * sigma / (2 * sqrt_t)
            - r * k * disc_r * _norm_cdf(d2)
            + q * s * disc_q * _norm_cdf(d1)
        )
        rho = k * t * disc_r * _norm_cdf(d2) / 100.0
    else:
        delta = -disc_q * _norm_cdf(-d1)
        theta_annual = (
            -s * disc_q * pdf_d1 * sigma / (2 * sqrt_t)
            + r * k * disc_r * _norm_cdf(-d2)
            - q * s * disc_q * _norm_cdf(-d1)
        )
        rho = -k * t * disc_r * _norm_cdf(-d2) / 100.0

    return Greeks(
        delta=delta,
        gamma=disc_q * pdf_d1 / (s * sigma * sqrt_t),
        theta=theta_annual / 365.0,
        vega=s * disc_q * pdf_d1 * sqrt_t / 100.0,
        rho=rho,
        iv=sigma,
    )


def implied_vol(
    price: float,
    s: float,
    k: float,
    t: float,
    right: OptionRight,
    r: float = 0.0,
    q: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float | None:
    """Implied volatility via bisection (robust; monotone in sigma).

    Returns None when the price is outside no-arbitrage bounds or below
    intrinsic — callers treat that as "no reliable IV".
    """
    if t <= 0 or price <= 0:
        return None
    intrinsic = bs_price(s, k, t, 0.0, right, r, q)
    if price < intrinsic - tol:
        return None
    lo, hi = 1e-4, 5.0
    if price > bs_price(s, k, t, hi, right, r, q):
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        diff = bs_price(s, k, t, mid, right, r, q) - price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
