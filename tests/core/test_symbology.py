"""Symbology round-trips: canonical ⇄ Alpaca compact ⇄ Schwab 21-char OSI."""

from datetime import date

import pytest

from catalyst.core.models import OptionKey, OptionRight
from catalyst.core.symbology import parse_osi, to_alpaca_symbol, to_osi, to_schwab_symbol

AAPL_CALL = OptionKey(
    underlying="AAPL", expiry=date(2024, 4, 19), right=OptionRight.CALL, strike=150.0
)


def test_schwab_osi_is_21_chars_padded() -> None:
    sym = to_schwab_symbol(AAPL_CALL)
    assert sym == "AAPL  240419C00150000"
    assert len(sym) == 21


def test_alpaca_symbol_is_compact() -> None:
    assert to_alpaca_symbol(AAPL_CALL) == "AAPL240419C00150000"


@pytest.mark.parametrize("strike", [0.5, 7.5, 150.0, 152.5, 1234.125, 5870.0])
def test_round_trip_both_formats(strike: float) -> None:
    key = OptionKey(underlying="SPY", expiry=date(2026, 1, 16), right=OptionRight.PUT, strike=strike)
    assert parse_osi(to_osi(key, pad_root=True)) == key
    assert parse_osi(to_osi(key, pad_root=False)) == key


def test_parse_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_osi("NOT_AN_OPTION")
