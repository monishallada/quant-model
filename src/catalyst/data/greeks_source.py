"""Single Greeks/IV source.

HONESTY NOTE (audit D-049/D-213): the py_vollib_vectorized path this module
was written around is DEAD in the deployed environment — every call raises a
numba TypingError — so the in-house bisection solver in ``black_scholes.py``
is what actually produced every measured number in this repository. The old
docstring claimed a Brent search and machine-precision vollib inversion;
neither was true at runtime.

The contract now: probe vollib ONCE at import with a real computation. If the
probe succeeds it serves as primary (with in-house fallback for quotes outside
no-arbitrage bounds); if it fails, log the fact once and use the in-house
solver directly — which handles dividend yield q properly on every path,
unlike py_vollib's q-less black_scholes greeks (audit D-130).
"""

from __future__ import annotations

import logging
import warnings

from catalyst.core.types import Greeks, OptionRight

logger = logging.getLogger(__name__)

from catalyst.data import black_scholes as _bs


def _probe_vollib():
    """Import AND execute vollib once; only a working solver is a solver."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import py_vollib_vectorized  # noqa: F401  (patches py_vollib)
            from py_vollib.black_scholes.greeks import analytical as g
            from py_vollib.black_scholes.implied_volatility import (
                implied_volatility as iv)
            probe = float(iv(2.50, 100.0, 100.0, 0.25, 0.045, "c"))
            if not (0 < probe < 5):
                raise ValueError(f"probe IV {probe} implausible")
        return g, iv
    except Exception as e:                               # noqa: BLE001
        logger.warning("py_vollib unavailable/broken in this environment (%s) "
                       "— using the in-house solver for ALL greeks/IV "
                       "(audit D-049)", str(e).splitlines()[0][:120])
        return None, None


_greeks, _iv = _probe_vollib()


def implied_vol(
    price: float, spot: float, strike: float, t_years: float, r: float,
    is_call: bool, q: float = 0.0,
) -> float | None:
    """IV from a mid price. None when the quote admits no solution."""
    if price <= 0 or spot <= 0 or strike <= 0 or t_years <= 0:
        return None
    if _iv is not None and q == 0.0:      # vollib black_scholes has no q
        flag = "c" if is_call else "p"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                iv = float(_iv(price, spot, strike, t_years, r, flag))
            if iv > 0 and iv == iv and iv != float("inf"):
                return iv
        except Exception:                                # noqa: BLE001
            pass
    # Quotes outside no-arbitrage bounds reach here; the Brent solver handles
    # some of them, and a contract with no Greek would vanish from selection.
    right = OptionRight.CALL if is_call else OptionRight.PUT
    return _bs.implied_vol(price, spot, strike, t_years, right, r, q)


def greeks(
    price: float, spot: float, strike: float, t_years: float, r: float,
    is_call: bool, q: float = 0.0,
) -> Greeks | None:
    iv = implied_vol(price, spot, strike, t_years, r, is_call, q)
    if iv is None:
        return None
    right = OptionRight.CALL if is_call else OptionRight.PUT
    # A non-zero dividend yield must NEVER be silently dropped (audit D-130):
    # py_vollib's black_scholes greeks take no q, so any q!=0 routes in-house.
    if _greeks is not None and q == 0.0:
        flag = "c" if is_call else "p"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return Greeks(
                    delta=float(_greeks.delta(flag, spot, strike, t_years, r, iv)),
                    gamma=float(_greeks.gamma(flag, spot, strike, t_years, r, iv)),
                    theta=float(_greeks.theta(flag, spot, strike, t_years, r, iv)),
                    vega=float(_greeks.vega(flag, spot, strike, t_years, r, iv)),
                    rho=float(_greeks.rho(flag, spot, strike, t_years, r, iv)),
                    iv=iv)
        except Exception:                                # noqa: BLE001
            pass
    return _bs.bs_greeks(spot, strike, t_years, iv, right, r, q)
