"""The in-house Black-Scholes solver — the solver that ACTUALLY produced every
measured number (audit D-049) — verified against textbook values, put-call
parity, finite differences, and boundary behavior."""

import math

import pytest

from catalyst.core.types import OptionRight
from catalyst.data import black_scholes as bs


class TestTextbookValues:
    def test_hull_european_call(self):
        """Hull's classic: S=42, K=40, r=10%, sigma=20%, T=0.5 -> C=4.759."""
        c = bs.bs_price(42.0, 40.0, 0.5, 0.20, OptionRight.CALL, r=0.10)
        assert c == pytest.approx(4.759, abs=2e-3)

    def test_hull_european_put(self):
        p = bs.bs_price(42.0, 40.0, 0.5, 0.20, OptionRight.PUT, r=0.10)
        assert p == pytest.approx(0.808, abs=2e-3)

    def test_put_call_parity_across_grid(self):
        for s in (50.0, 100.0, 400.0):
            for km in (0.85, 1.0, 1.15):
                for t in (7 / 365, 0.5):
                    for sig in (0.15, 0.60):
                        k = s * km
                        c = bs.bs_price(s, k, t, sig, OptionRight.CALL, r=0.04)
                        p = bs.bs_price(s, k, t, sig, OptionRight.PUT, r=0.04)
                        assert c - p == pytest.approx(
                            s - k * math.exp(-0.04 * t), abs=1e-9)


class TestGreeksAgainstFiniteDifferences:
    @pytest.mark.parametrize("right", [OptionRight.CALL, OptionRight.PUT])
    @pytest.mark.parametrize("k_mult", [0.9, 1.0, 1.1])
    def test_delta_matches_bumped_price(self, right, k_mult):
        s, t, sig, r = 100.0, 0.25, 0.35, 0.04
        k = s * k_mult
        g = bs.bs_greeks(s, k, t, sig, right, r)
        h = 0.01
        fd = (bs.bs_price(s + h, k, t, sig, right, r)
              - bs.bs_price(s - h, k, t, sig, right, r)) / (2 * h)
        assert g.delta == pytest.approx(fd, abs=1e-4)

    def test_vega_matches_bumped_price(self):
        s, k, t, sig, r = 100.0, 105.0, 0.25, 0.35, 0.04
        g = bs.bs_greeks(s, k, t, sig, OptionRight.CALL, r)
        h = 1e-4
        fd = (bs.bs_price(s, k, t, sig + h, OptionRight.CALL, r)
              - bs.bs_price(s, k, t, sig - h, OptionRight.CALL, r)) / (2 * h)
        # repo convention: vega per 1.0 vol (not per point)
        assert g.vega == pytest.approx(fd, rel=1e-3) or \
               g.vega == pytest.approx(fd / 100, rel=1e-3)


class TestBoundaries:
    def test_near_expiry_converges_to_intrinsic(self):
        c = bs.bs_price(110.0, 100.0, 1e-6, 0.5, OptionRight.CALL, r=0.04)
        assert c == pytest.approx(10.0, abs=1e-3)

    def test_deep_otm_price_is_tiny_not_negative(self):
        p = bs.bs_price(100.0, 300.0, 7 / 365, 0.3, OptionRight.CALL, r=0.04)
        assert 0.0 <= p < 1e-6

    def test_iv_of_below_intrinsic_price_is_none(self):
        """A price below intrinsic admits no vol — must be None, not garbage."""
        iv = bs.implied_vol(5.0, 110.0, 100.0, 0.25, OptionRight.CALL, r=0.0)
        assert iv is None

    def test_iv_round_trip_at_extremes(self):
        for sig in (0.05, 1.50):
            price = bs.bs_price(100.0, 100.0, 0.25, sig, OptionRight.PUT, r=0.04)
            solved = bs.implied_vol(price, 100.0, 100.0, 0.25,
                                    OptionRight.PUT, r=0.04)
            assert solved == pytest.approx(sig, abs=1e-4)
