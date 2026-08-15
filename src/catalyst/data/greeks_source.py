"""Single Greeks/IV source, backed by py_vollib_vectorized.

py_vollib_vectorized wraps "Let's Be Rational" (Jaeckel), which inverts
Black-Scholes to near machine precision in a fixed number of steps. The
in-house solver in ``black_scholes.py`` is a Brent search — correct, but slower
and with its own convergence edges, one of which this project already hit:
deep-ITM low-IV puts returned ``None`` until the sigma-to-zero limit was fixed
to use the discounted forward intrinsic.

Switching the Greeks source would change every measured number if the two
disagreed, so this module does not simply replace the old one. It exposes the
vollib path and a cross-check, and ``tests/data/test_greeks_agreement.py``
asserts the two agree across a wide grid. Agreement is what makes the swap
behaviour-preserving; without the test it would be a silent methodology change.

Fallback is deliberate: if vollib cannot solve a quote (an option priced
outside no-arbitrage bounds, which real NBBO data does contain), we fall back
to the in-house solver rather than dropping the contract, because a missing
Greek silently removes a candidate from a strategy's universe.
"""

from __future__ import annotations

import logging
import warnings

from catalyst.core.types import Greeks, OptionRight

logger = logging.getLogger(__name__)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import py_vollib_vectorized  # noqa: F401  (import patches py_vollib in place)
    from py_vollib.black_scholes.greeks import analytical as _greeks
    from py_vollib.black_scholes.implied_volatility import implied_volatility as _iv

from catalyst.data import black_scholes as _bs


def implied_vol(
    price: float, spot: float, strike: float, t_years: float, r: float,
    is_call: bool, q: float = 0.0,
) -> float | None:
    """IV from a mid price. None when the quote admits no solution."""
    if price <= 0 or spot <= 0 or strike <= 0 or t_years <= 0:
        return None
    flag = "c" if is_call else "p"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iv = float(_iv(price, spot, strike, t_years, r, flag))
        if iv > 0 and iv == iv and iv != float("inf"):
            return iv
    except Exception:                                    # noqa: BLE001
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
    except Exception:                                    # noqa: BLE001
        right = OptionRight.CALL if is_call else OptionRight.PUT
        return _bs.bs_greeks(spot, strike, t_years, iv, right, r, q)
