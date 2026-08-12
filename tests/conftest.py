"""Shared fixtures: synthetic option chains with Black-Scholes-consistent
greeks so engine selection logic is exercised against realistic deltas."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from catalyst.core.models import OptionChain, OptionContract, OptionKey, OptionRight
from catalyst.data.black_scholes import bs_greeks, bs_price


def build_chain(
    symbol: str = "SPY",
    spot: float = 100.0,
    as_of: datetime = datetime(2024, 6, 3, 15, 45),
    expiry_ivs: dict[date, float] | None = None,
    strikes: list[float] | None = None,
    spread_frac: float = 0.02,
    volume: int = 2000,
    open_interest: int = 5000,
) -> OptionChain:
    expiry_ivs = expiry_ivs or {date(2024, 6, 21): 0.25, date(2024, 7, 19): 0.24}
    # $1 grid near the money (like liquid real chains) so delta-range selection
    # always has candidates; coarser wings.
    strikes = strikes or [80, 85, 90, 93, 94, 95, 96, 97, 98, 99, 100,
                          101, 102, 103, 104, 105, 106, 107, 110, 115, 120]
    contracts: list[OptionContract] = []
    for expiry, iv in expiry_ivs.items():
        t = max((expiry - as_of.date()).days, 1) / 365.0
        for strike in strikes:
            for right in (OptionRight.CALL, OptionRight.PUT):
                mid = bs_price(spot, strike, t, iv, right, r=0.045)
                if mid < 0.01:
                    mid = 0.01
                half = max(mid * spread_frac / 2, 0.005)
                contracts.append(
                    OptionContract(
                        key=OptionKey(
                            underlying=symbol, expiry=expiry, right=right, strike=float(strike)
                        ),
                        bid=round(mid - half, 4),
                        ask=round(mid + half, 4),
                        volume=volume,
                        open_interest=open_interest,
                        greeks=bs_greeks(spot, strike, t, iv, right, r=0.045),
                    )
                )
    return OptionChain(
        underlying=symbol, underlying_price=spot, timestamp=as_of, contracts=contracts
    )


class FakeIVRank:
    """Stub IVRankProvider: returns a fixed rank."""

    def __init__(self, rank: float | None) -> None:
        self.rank = rank

    def iv_rank(self, symbol: str, day: date) -> float | None:
        return self.rank

    def prepare(self, symbol: str, start: date, end: date) -> None:  # pragma: no cover
        pass


@pytest.fixture
def chain() -> OptionChain:
    return build_chain()
