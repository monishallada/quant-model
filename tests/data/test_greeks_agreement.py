"""py_vollib_vectorized must agree with the in-house solver.

Agreement is what makes adopting vollib a behaviour-preserving change rather
than a silent methodology change: if the two disagreed, every archived result
computed with the old solver would stop being comparable with new ones.
"""

import pytest

from catalyst.core.types import OptionRight
from catalyst.data import black_scholes as bs
from catalyst.data import greeks_source as gs


def _right(is_call):
    return OptionRight.CALL if is_call else OptionRight.PUT

GRID = [
    (spot, strike, t, r, is_call, sigma)
    for spot in (50.0, 100.0, 400.0)
    for strike in (0.85, 1.0, 1.15)
    for t in (7 / 365, 35 / 365, 180 / 365)
    for r in (0.0, 0.04)
    for is_call in (True, False)
    for sigma in (0.15, 0.35, 0.80)
]


@pytest.mark.parametrize("spot,k_mult,t,r,is_call,sigma", GRID)
def test_implied_vol_round_trips_within_tolerance(spot, k_mult, t, r, is_call, sigma):
    """Price at a known sigma, then recover it. Both solvers must find it."""
    strike = spot * k_mult
    price = bs.bs_price(spot, strike, t, sigma, _right(is_call), r)
    if price < 0.01:
        pytest.skip("premium below a quotable tick")

    # IV is only identifiable where price actually responds to vol. On a
    # deep-ITM 7-day contract vega is ~0: every sigma prices the same, so
    # "recovering sigma" is not a property either solver can have. This is a
    # fact about options, not a solver defect — so the property is scoped to
    # where it is meaningful rather than the tolerance being loosened.
    vega = bs.bs_greeks(spot, strike, t, sigma, _right(is_call), r).vega
    if vega < 0.02:
        pytest.skip(f"vega {vega:.4f} too small for IV to be identifiable")

    vollib = gs.implied_vol(price, spot, strike, t, r, is_call)
    inhouse = bs.implied_vol(price, spot, strike, t, _right(is_call), r)
    assert vollib is not None and inhouse is not None
    assert vollib == pytest.approx(sigma, abs=1e-4), "vollib did not recover sigma"
    assert vollib == pytest.approx(inhouse, abs=1e-3), (
        f"solvers disagree: vollib {vollib:.6f} vs in-house {inhouse:.6f}")


@pytest.mark.parametrize("is_call", [True, False])
def test_greeks_agree_on_a_representative_contract(is_call):
    spot, strike, t, r, sigma = 100.0, 105.0, 35 / 365, 0.04, 0.35
    price = bs.bs_price(spot, strike, t, sigma, _right(is_call), r)
    a = gs.greeks(price, spot, strike, t, r, is_call)
    b = bs.bs_greeks(spot, strike, t, sigma, _right(is_call), r)
    assert a is not None and b is not None
    assert a.delta == pytest.approx(b.delta, abs=1e-3)
    assert a.vega == pytest.approx(b.vega, abs=1e-2)
    assert a.iv == pytest.approx(b.iv, abs=1e-3)


def test_unsolvable_quote_returns_none_rather_than_a_guess():
    """A price below intrinsic has no IV. Inventing one would be worse than
    dropping the contract."""
    assert gs.implied_vol(0.0, 100.0, 100.0, 0.1, 0.04, True) is None
    assert gs.implied_vol(5.0, 100.0, 100.0, 0.0, 0.04, True) is None
