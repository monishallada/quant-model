"""Drift Harvest — State 1: the measured post-earnings setup.

Two inputs, both *measured facts* rather than forecasts — the property that
distinguishes the one strategy in this project that worked (Engine C) from the
several that did not:

1. **Realized overnight gap** ``G = (open_reaction_day - prior_close) / prior_close``.
   This replaces the EPS surprise as the direction input. Consensus EPS is
   gamed and guidance usually matters more; the gap is the market's own
   aggregated verdict on the news, already printed and observable.

2. **IV richness** ``R = ATM implied vol / trailing realized vol``. The trade
   only makes sense when the premium being sold is actually rich relative to
   how much the stock has been moving. Both terms are measured — IV from the
   option chain at entry, realized vol from the underlying's own closes.

Lookahead discipline: the gap uses the reaction day's open and the prior
session's close, both strictly before any entry (entry is 2-5 sessions later).
Realized vol uses closes up to the session BEFORE entry.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date

import pandas as pd

from catalyst.core.config import DriftConfig
from catalyst.core.types import Catalyst, OptionChain, OptionRight
from catalyst.core.tradingcal import trading_days_between
from catalyst.data.catalysts import resolve_reaction_session
from catalyst.core.chains import atm_contract

logger = logging.getLogger(__name__)

_ANNUALIZATION = 252


@dataclass(frozen=True)
class DriftSetup:
    symbol: str
    reaction_session: date
    gap: float | None
    direction: str | None  # "up" | "down" — the drift the gap implies
    realized_vol: float | None
    reject_reason: str | None


def realized_vol(closes: pd.Series, upto: date, window: int) -> float | None:
    """Annualized close-to-close realized vol over ``window`` sessions ending
    strictly before ``upto``."""
    prior = closes[closes.index < upto]
    if len(prior) < window + 1:
        return None
    tail = prior.tail(window + 1).astype(float)
    rets = (tail / tail.shift(1)).dropna()
    if len(rets) < window // 2 or (rets <= 0).any():
        return None
    log_rets = rets.apply(math.log)
    sd = float(log_rets.std(ddof=1))
    if sd <= 0 or not math.isfinite(sd):
        return None
    return sd * math.sqrt(_ANNUALIZATION)


class DriftScreener:
    """Computes the measured setup per (symbol, earnings event), memoized."""

    def __init__(self, cfg: DriftConfig, daily: dict[str, pd.DataFrame]) -> None:
        self._cfg = cfg
        self._daily = daily  # symbol -> DataFrame with open/close, date-indexed
        self._memo: dict[tuple[str, date], DriftSetup] = {}

    def setup_for(self, catalyst: Catalyst) -> DriftSetup:
        reaction = resolve_reaction_session(catalyst)
        key = (catalyst.symbol, reaction)
        if key in self._memo:
            return self._memo[key]
        result = self._compute(catalyst.symbol, reaction)
        self._memo[key] = result
        return result

    def _compute(self, symbol: str, reaction: date) -> DriftSetup:
        cfg = self._cfg
        df = self._daily.get(symbol)
        if df is None or df.empty:
            return DriftSetup(symbol, reaction, None, None, None, "no_price_data")

        idx = df.index
        if reaction not in idx:
            return DriftSetup(symbol, reaction, None, None, None, "no_reaction_bar")
        prior = df[idx < reaction]
        if prior.empty:
            return DriftSetup(symbol, reaction, None, None, None, "no_prior_close")

        prior_close = float(prior["close"].iloc[-1])
        open_px = float(df.loc[reaction, "open"])
        if prior_close <= 0 or open_px <= 0:
            return DriftSetup(symbol, reaction, None, None, None, "bad_prices")
        gap = (open_px - prior_close) / prior_close

        rv = realized_vol(df["close"], reaction, cfg.realized_vol_window)
        if abs(gap) < cfg.min_abs_gap:
            return DriftSetup(symbol, reaction, gap, None, rv, "gap_too_small")
        direction = "up" if gap > 0 else "down"
        return DriftSetup(symbol, reaction, gap, direction, rv, None)

    def entry_window_open(self, reaction: date, session: date) -> bool:
        lo, hi = self._cfg.entry_window_trading_days
        if session < reaction:
            return False
        return lo <= trading_days_between(reaction, session) <= hi

    def iv_richness(self, chain: OptionChain, expiry: date, rv: float | None) -> float | None:
        """ATM implied vol divided by trailing realized vol at entry."""
        if not rv or rv <= 0:
            return None
        call = atm_contract(chain, expiry, OptionRight.CALL)
        put = atm_contract(chain, expiry, OptionRight.PUT)
        ivs = [c.greeks.iv for c in (call, put) if c is not None and c.greeks and c.greeks.iv > 0]
        if not ivs:
            return None
        return (sum(ivs) / len(ivs)) / rv
