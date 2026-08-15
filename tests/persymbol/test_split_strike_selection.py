"""Regression: strike selection must use the option chain's OWN spot.

Alpaca daily bars are split-ADJUSTED; ThetaData strikes are raw as-of-date.
Selecting strikes from the adjusted price against a raw chain targeted a $165
AMZN strike in a chain running $860-$5300 (the 20-for-1 split) and bought
deep-ITM contracts for years. It hid on MSFT, which never split — half the
universe looked fine, which is exactly why it survived a full campaign.
"""

from datetime import date, timedelta

import pandas as pd

from catalyst.core.types import (Greeks, OptionChain, OptionContract, OptionKey,
                                  OptionRight)
from catalyst.strategies.archive.long_options_standalone.engine import LongOptionConfig, run_long_options

RAW_SPOT = 3000.0     # what the chain quotes (pre-split)
ADJ_SPOT = 150.0      # what the split-adjusted price series says (20-for-1)

IDX = pd.bdate_range("2022-01-03", periods=140)
FRIDAYS = [d.date() for d in pd.date_range("2022-01-07", "2022-09-30", freq="W-FRI")]


def _contracts(expiry: date) -> list[OptionContract]:
    out = []
    for k in range(2600, 3600, 50):
        for right in (OptionRight.CALL, OptionRight.PUT):
            out.append(OptionContract(
                key=OptionKey(underlying="AMZN", expiry=expiry, right=right, strike=float(k)),
                bid=20.0, ask=22.0, volume=500, open_interest=1000,
                greeks=Greeks(delta=0.35, theta=-0.5, vega=1.2, iv=0.4)))
    return out


class _RawChainData:
    """Chain quotes RAW prices while the caller holds a split-ADJUSTED series."""

    def list_expirations(self, symbol):
        return list(FRIDAYS)

    def get_chain(self, symbol, when, expiries=None, include_liquidity=False):
        contracts = []
        for e in (expiries or FRIDAYS):
            contracts += _contracts(e)
        return OptionChain(underlying="AMZN", timestamp=when,
                           underlying_price=RAW_SPOT, contracts=contracts)


def test_strike_comes_from_chain_spot_not_adjusted_series():
    px = pd.Series([ADJ_SPOT] * len(IDX), index=IDX)
    cfg = LongOptionConfig(moneyness=1.05, capital_fraction=0.10)

    res = run_long_options("AMZN", px, _RawChainData(), cfg, list(FRIDAYS), ruin_floor=0.0)

    assert res.trades, "expected at least one position"
    for t in res.trades:
        # 5% OTM on the RAW $3,000 spot is ~$3,150. The adjusted-price bug
        # targeted $157 and would have grabbed the $2,600 floor: deep ITM.
        assert abs(t.strike - 3150.0) <= 50.0, (
            f"strike {t.strike} is not ~5% OTM of the raw spot — "
            "selection used the adjusted series")


def test_missing_exit_contract_is_skipped_not_fabricated():
    """A contract absent at exit is unpriceable — never marked at intrinsic.

    The intrinsic fallback marked a vanished strike against a post-split spot
    and manufactured a 924x put.
    """
    class _VanishingExit(_RawChainData):
        def get_chain(self, symbol, when, expiries=None, include_liquidity=False):
            if when.date() > date(2022, 1, 25):     # contracts gone after entry
                return OptionChain(underlying="AMZN", timestamp=when,
                                   underlying_price=ADJ_SPOT, contracts=[])
            return super().get_chain(symbol, when, expiries, include_liquidity)

    px = pd.Series([ADJ_SPOT] * len(IDX), index=IDX)
    cfg = LongOptionConfig(moneyness=1.05, capital_fraction=0.10)

    res = run_long_options("AMZN", px, _VanishingExit(), cfg, list(FRIDAYS), ruin_floor=0.0)

    assert res.unpriceable > 0, "unpriceable cycles must be counted, not priced"
    assert all(t.multiple < 50 for t in res.trades), (
        f"fabricated multiple {max((t.multiple for t in res.trades), default=0)} — "
        "intrinsic fallback resurrected")
