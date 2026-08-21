"""Regression tests for audit D-121 (adjusted OCC roots) and D-206 (malformed
OSI emitted silently)."""

from datetime import date

import pytest

from catalyst.core.symbology import parse_osi, to_osi, to_schwab_symbol
from catalyst.core.types import OptionKey, OptionRight


def _key(underlying="AAPL", strike=150.0):
    return OptionKey(underlying=underlying, expiry=date(2024, 4, 19),
                     right=OptionRight.CALL, strike=strike)


class TestRoundTrip:
    def test_compact_and_padded_round_trip(self):
        k = _key()
        for pad in (True, False):
            assert parse_osi(to_osi(k, pad_root=pad)) == k

    def test_adjusted_root_with_digit_parses(self):
        """Corporate-action roots like AAPL1 are legal OCC and must not be
        silently dropped by reconciliation (D-121)."""
        k = _key(underlying="AAPL1")
        assert parse_osi(to_osi(k, pad_root=False)) == k
        assert parse_osi(to_schwab_symbol(k)) == k

    def test_leading_digit_root_still_rejected(self):
        with pytest.raises(ValueError):
            parse_osi("1AAPL240419C00150000")


class TestMalformedInputsFailLoudly:
    def test_seven_char_root_raises(self):
        with pytest.raises(ValueError):
            to_osi(_key(underlying="ABCDEFG"))

    def test_six_figure_strike_raises(self):
        with pytest.raises(ValueError):
            to_osi(_key(strike=100_000.0))

    def test_zero_strike_raises(self):
        with pytest.raises(ValueError):
            to_osi(_key(strike=0.0000001))
