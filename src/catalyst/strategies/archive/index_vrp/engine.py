"""v8 — Index volatility risk premium harvest on SPY/QQQ put credit spreads.

This is the one options idea that six campaigns of evidence actually point to,
and it exists because of a single measured fact: **index options cost 3.1x less
to cross than single-name options** (SPY 0.65% of mid vs 4.43% average across
single names). Transaction cost, not signal, is what killed every previous
options campaign — v4 lost 53% of collected credit to spreads alone. Moving the
same trade to the cheapest venue in the options market is the highest-leverage
change available.

Why put spreads specifically, rather than condors or long options:
- The index variance risk premium is concentrated on the PUT side, because it
  is created by structural hedging demand — institutions buying protection.
  That is also why index skew is persistently steep.
- Selling it is the only options structure this project has measured as
  gross-positive (v4: +$7/trade before costs on single names).
- A defined-risk spread caps the loss at (width - credit), so the RiskManager's
  sizing, cash floor and heat caps bind exactly as they do for debit premium,
  and a tail event cannot exceed a known number.
- Two legs, not four: half the spread crossings of an iron condor.

Entry is calendar-driven rather than event-driven, which is the point — this
harvests a structural premium that is always present, instead of predicting
anything. No directional forecast is involved, which matters because six
campaigns established that our directional forecasts do not work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time

import pandas as pd

from catalyst.core.config import IndexVRPConfig
from catalyst.core.types import OptionChain, OptionContract, OptionRight
from catalyst.data.thetadata_historical import DataUnavailableError, ThetaDataHistorical
from catalyst.core.chains import nearest_delta, quotable

logger = logging.getLogger(__name__)

SNAP = time(15, 45)


@dataclass
class VRPPosition:
    symbol: str
    expiry: date
    short_strike: float
    long_strike: float
    credit: float          # per unit, received at entry (positive)
    width: float
    max_loss: float        # width - credit, the risk basis
    contracts: int
    entry_date: date

    @property
    def risk_dollars(self) -> float:
        return self.max_loss * self.contracts * 100.0


def select_expiry(available: list[date], today: date, window: tuple[int, int]) -> list[date]:
    """Candidate expiries inside the DTE window, Fridays first.

    Friday expirations are tried first because ThetaData's coverage of the
    daily SPY expiries is sparse at a fixed intraday snapshot — a systematic
    monthly/weekly program would use the liquid Friday series anyway.
    """
    lo, hi = window
    eligible = [e for e in sorted(available) if lo <= (e - today).days <= hi]
    fridays = [e for e in eligible if e.weekday() == 4]
    return fridays + [e for e in eligible if e not in fridays]


def build_put_spread(
    chain: OptionChain, expiry: date, cfg: IndexVRPConfig, spot: float
) -> tuple[OptionContract, OptionContract] | None:
    """(short put at target delta, long protective put below it)."""
    short = nearest_delta(chain, expiry, OptionRight.PUT, cfg.short_delta)
    if short is None:
        return None
    target_long = short.key.strike - spot * cfg.wing_width_pct
    lower = [c for c in quotable(chain.slice(expiry=expiry, right=OptionRight.PUT))
             if c.key.strike < short.key.strike]
    if not lower:
        return None
    long_wing = min(lower, key=lambda c: abs(c.key.strike - target_long))
    return short, long_wing


class IndexVRPEngine:
    """Builds one defined-risk short-put-spread proposal per symbol-day."""

    def __init__(self, cfg: IndexVRPConfig, data: ThetaDataHistorical) -> None:
        self._cfg = cfg
        self._data = data
        self.skipped_no_chain = 0
        self.skipped_no_structure = 0
        self.skipped_thin_credit = 0
        self.built = 0

    def propose(self, symbol: str, today: date, expiries: list[date]
                ) -> tuple[VRPPosition, OptionChain] | None:
        cfg = self._cfg
        for expiry in select_expiry(expiries, today, cfg.dte_window)[:4]:
            try:
                chain = self._data.get_chain(symbol, datetime.combine(today, SNAP),
                                             expiries=[expiry], include_liquidity=False)
            except DataUnavailableError:
                continue
            spot = chain.underlying_price
            if spot <= 0:
                continue
            pair = build_put_spread(chain, expiry, cfg, spot)
            if pair is None:
                self.skipped_no_structure += 1
                continue
            short, long_wing = pair

            # Credit at WORST-CASE fills: sell the short at its bid, buy the
            # wing at its ask. Anything better is upside, never assumed.
            credit = short.bid - long_wing.ask
            width = short.key.strike - long_wing.key.strike
            if width <= 0 or credit <= 0:
                self.skipped_thin_credit += 1
                continue
            if credit / width < cfg.min_credit_fraction:
                self.skipped_thin_credit += 1
                continue
            max_loss = width - credit
            if max_loss <= 0:
                self.skipped_thin_credit += 1
                continue
            self.built += 1
            return VRPPosition(symbol=symbol, expiry=expiry,
                               short_strike=short.key.strike, long_strike=long_wing.key.strike,
                               credit=credit, width=width, max_loss=max_loss,
                               contracts=0, entry_date=today), chain
        self.skipped_no_chain += 1
        return None

    def mark(self, pos: VRPPosition, day: date) -> float | None:
        """Current cost to CLOSE the spread per unit (buy short back, sell wing).

        Returns the liability: 0 means the spread expired worthless and the full
        credit was kept; ``width`` means maximum loss.
        """
        try:
            chain = self._data.get_chain(pos.symbol, datetime.combine(day, SNAP),
                                         expiries=[pos.expiry], include_liquidity=False)
        except DataUnavailableError:
            return None
        short = next((c for c in chain.contracts
                      if c.key.strike == pos.short_strike and c.key.right is OptionRight.PUT), None)
        wing = next((c for c in chain.contracts
                     if c.key.strike == pos.long_strike and c.key.right is OptionRight.PUT), None)
        if short is None or wing is None:
            return None
        # Closing costs the short's ASK and receives the wing's BID.
        liability = short.ask - wing.bid
        return max(min(liability, pos.width), 0.0)
