"""Real-chain validation of the tournament architecture.

The fast simulator prices options with Black-Scholes off trailing realized
volatility. That is fine for exploring the configuration space, but every
number it produces is a model output, not a market outcome. This module
replaces the model with actual historical NBBO chains: entries fill at the
ask, exits at the bid, and the option is whatever the market said it was
worth on those two dates.

This is the same discipline that caught the Kinetic campaign's fabricated
fills — a strategy is only as real as the prices it is measured against.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time

import pandas as pd

from catalyst.core.models import OptionRight
from catalyst.data.thetadata_historical import DataUnavailableError, ThetaDataHistorical
from catalyst.engines.util import quotable
from catalyst.tournament.engine import TournamentConfig

logger = logging.getLogger(__name__)

_SNAP = time(15, 45)


@dataclass
class ChainTradeResult:
    symbol: str
    entry: date
    exit: date
    strike: float
    entry_price: float
    exit_price: float
    multiple: float
    synthetic: bool = False


class RealChainPricer:
    """Prices one long-option position from actual chains, with caching."""

    def __init__(self, data: ThetaDataHistorical, cfg: TournamentConfig) -> None:
        self._data = data
        self._cfg = cfg
        self.misses = 0
        self.hits = 0

    def _chain(self, symbol: str, day: date, expiry: date):
        try:
            return self._data.get_chain(symbol, datetime.combine(day, _SNAP),
                                        expiries=[expiry], include_liquidity=False)
        except DataUnavailableError:
            return None

    def price_position(
        self, symbol: str, entry_day: date, exit_day: date, spot: float,
        expiries: list[date],
    ) -> ChainTradeResult | None:
        """Buy an OTM call at the ask on ``entry_day``; sell at the bid on ``exit_day``."""
        cfg = self._cfg
        target_exp = [e for e in expiries
                      if cfg.dte_days - 12 <= (e - entry_day).days <= cfg.dte_days + 12]
        if not target_exp:
            self.misses += 1
            return None
        expiry = min(target_exp, key=lambda e: abs((e - entry_day).days - cfg.dte_days))

        chain = self._chain(symbol, entry_day, expiry)
        if chain is None:
            self.misses += 1
            return None
        target_strike = spot * cfg.moneyness
        calls = [c for c in quotable(chain.slice(expiry=expiry, right=OptionRight.CALL))
                 if c.ask > 0.02]
        if not calls:
            self.misses += 1
            return None
        pick = min(calls, key=lambda c: abs(c.key.strike - target_strike))
        entry_price = pick.ask + cfg.cost_per_contract

        # Exit: same contract, priced on the exit date at the bid.
        exit_chain = self._chain(symbol, exit_day, expiry)
        exit_price = 0.0
        if exit_chain is not None:
            match = exit_chain.find(pick.key)
            if match is not None and match.bid >= 0:
                exit_price = max(match.bid - cfg.cost_per_contract, 0.0)
            else:
                # Contract still exists but is unquoted: worth intrinsic at most.
                exit_price = max(exit_chain.underlying_price - pick.key.strike, 0.0)
        if entry_price <= 0:
            self.misses += 1
            return None
        self.hits += 1
        return ChainTradeResult(
            symbol=symbol, entry=entry_day, exit=exit_day, strike=pick.key.strike,
            entry_price=entry_price, exit_price=exit_price,
            multiple=exit_price / entry_price,
        )


def run_real_chain_path(
    prices: pd.DataFrame, signal: pd.DataFrame, data: ThetaDataHistorical,
    start: pd.Timestamp, end: pd.Timestamp, cfg: TournamentConfig,
    expiry_cache: dict[str, list[date]], starting_equity: float = 100_000.0,
    trade_log: list[ChainTradeResult] | None = None,
) -> pd.Series:
    """One competition entry priced entirely on real chains."""
    pricer = RealChainPricer(data, cfg)
    window = prices.loc[start:end]
    if len(window) < cfg.hold_days + 2:
        return pd.Series(dtype=float)

    equity = starting_equity
    curve: dict[pd.Timestamp, float] = {window.index[0]: equity}
    dates = window.index
    i = 0
    while i + cfg.hold_days < len(dates):
        entry_ts, exit_ts = dates[i], dates[i + cfg.hold_days]
        row = signal.loc[entry_ts].dropna() if entry_ts in signal.index else pd.Series(dtype=float)
        available = [s for s in row.index if pd.notna(prices.loc[entry_ts, s])]
        if not available:
            curve[exit_ts] = equity
            i += cfg.hold_days
            continue
        picks = list(row[available].sort_values().index[-cfg.n_positions:])

        deployed = equity * cfg.capital_fraction
        per_name = deployed / max(len(picks), 1)
        proceeds = 0.0
        for sym in picks:
            spot = float(prices.loc[entry_ts, sym])
            res = pricer.price_position(sym, entry_ts.date(), exit_ts.date(), spot,
                                        expiry_cache.get(sym, []))
            if res is None:
                proceeds += per_name  # untradeable: capital stays in cash
                continue
            if trade_log is not None:
                trade_log.append(res)
            proceeds += per_name * res.multiple
        equity = equity - deployed + proceeds
        curve[exit_ts] = equity
        if equity < starting_equity * 0.01:
            for d in dates[i + cfg.hold_days:]:
                curve[d] = equity
            break
        i += cfg.hold_days
    return pd.Series(curve).sort_index()
