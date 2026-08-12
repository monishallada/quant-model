"""Black-Scholes kernel: known values, put-call parity, IV inversion."""

import math

import pytest

from catalyst.core.models import OptionRight
from catalyst.data.black_scholes import bs_greeks, bs_price, implied_vol


def test_known_price_call() -> None:
    # Classic textbook case: S=100, K=100, t=1y, sigma=20%, r=5% -> C ~= 10.4506
    price = bs_price(100, 100, 1.0, 0.20, OptionRight.CALL, r=0.05)
    assert price == pytest.approx(10.4506, abs=1e-3)


def test_put_call_parity() -> None:
    s, k, t, sigma, r = 525.94, 533.0, 18 / 365, 0.11, 0.045
    c = bs_price(s, k, t, sigma, OptionRight.CALL, r=r)
    p = bs_price(s, k, t, sigma, OptionRight.PUT, r=r)
    assert c - p == pytest.approx(s - k * math.exp(-r * t), abs=1e-9)


def test_iv_inversion_round_trip() -> None:
    s, k, t, r = 450.0, 460.0, 30 / 365, 0.045
    for sigma in (0.08, 0.15, 0.35, 0.80):
        price = bs_price(s, k, t, sigma, OptionRight.PUT, r=r)
        solved = implied_vol(price, s, k, t, OptionRight.PUT, r=r)
        assert solved == pytest.approx(sigma, abs=1e-4)


def test_iv_none_below_intrinsic() -> None:
    # Deep ITM call priced below intrinsic has no valid IV.
    assert implied_vol(price=1.0, s=200.0, k=100.0, t=0.1, right=OptionRight.CALL) is None


def test_greeks_sanity_atm() -> None:
    g = bs_greeks(100, 100, 30 / 365, 0.25, OptionRight.CALL, r=0.0)
    assert 0.45 < g.delta < 0.60
    assert g.gamma is not None and g.gamma > 0
    assert g.theta < 0
    assert g.vega > 0


def test_delta_matches_thetadata_served_value() -> None:
    # Real fixture from the M0 verification pull (SPY 2024-06-03 15:45 ET):
    # 533C 2024-06-21, IV 0.1106, spot 525.94, served delta 0.3361.
    g = bs_greeks(525.94, 533.0, 18 / 365, 0.1106, OptionRight.CALL, r=0.045)
    assert g.delta == pytest.approx(0.3361, abs=0.03)
