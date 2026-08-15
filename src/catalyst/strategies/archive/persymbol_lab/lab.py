"""v9 — per-symbol strategy lab.

Runs each candidate strategy on each stock INDEPENDENTLY and reports average
monthly return per symbol, so the question "which names should we run this on"
is answered by realized P&L rather than by a cost ratio.

Three strategies per symbol, chosen because each answers a distinct question:

1. **Buy and hold** — the benchmark every strategy must beat. Any options
   overlay that underperforms simply owning the stock has destroyed value,
   however good its win rate looks.

2. **Covered calls** — own 100 shares, sell an OTM call each cycle. The classic
   options income strategy, never tested in this project. It converts some
   upside into premium, so it should beat buy-and-hold in flat markets and lose
   to it in strong ones. On names that ran 10x over the period this is a real
   risk, and the test will show it plainly.

3. **Put credit spreads** — the v8 structure applied per-name: defined risk,
   collects the volatility premium, no share ownership.

All three are priced on real NBBO chains: options sold at the bid, bought at
the ask, shares crossed at a modeled spread. No mid fills anywhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time

import pandas as pd

from catalyst.core.config import PerSymbolConfig
from catalyst.core.types import OptionRight
from catalyst.data.thetadata_historical import DataUnavailableError, ThetaDataHistorical
from catalyst.core.chains import nearest_delta, quotable
from catalyst.backtest import metrics as m

logger = logging.getLogger(__name__)

SNAP = time(15, 45)
SHARE_COST_BPS = 3.5 / 1e4  # half-spread + slippage per side, liquid large caps


@dataclass
class SymbolResult:
    symbol: str
    strategy: str
    equity: pd.Series
    n_trades: int = 0
    wins: int = 0
    notes: dict = field(default_factory=dict)

    def headline(self) -> dict:
        h = m.headline(self.equity)
        h["n_trades"] = self.n_trades
        h["win_rate"] = self.wins / self.n_trades if self.n_trades else 0.0
        return h


def _friday_expiries(available: list[date], today: date, window: tuple[int, int]) -> list[date]:
    lo, hi = window
    elig = [e for e in sorted(available) if lo <= (e - today).days <= hi]
    fri = [e for e in elig if e.weekday() == 4]
    return fri + [e for e in elig if e not in fri]


def buy_and_hold(prices: pd.Series, starting: float = 100_000.0) -> SymbolResult:
    """The benchmark: buy once, pay the spread once, hold."""
    px = prices.dropna()
    if px.empty:
        return SymbolResult("", "buy_and_hold", pd.Series(dtype=float))
    shares = (starting * (1 - SHARE_COST_BPS)) / px.iloc[0]
    return SymbolResult("", "buy_and_hold", px * shares, n_trades=1, wins=0)


def covered_calls(
    symbol: str, prices: pd.Series, data: ThetaDataHistorical, cfg: PerSymbolConfig,
    expiries: list[date], starting: float = 100_000.0,
) -> SymbolResult:
    """Own shares, sell an OTM call each cycle, keep the premium.

    Assignment is modeled honestly: if the stock closes above the short strike
    at expiry the shares are called away at the strike (capping that cycle's
    gain) and immediately repurchased at the market to re-establish the
    position, paying the share spread both ways.
    """
    px = prices.dropna()
    if px.empty:
        return SymbolResult(symbol, "covered_calls", pd.Series(dtype=float))
    dates = list(px.index)
    shares = (starting * (1 - SHARE_COST_BPS)) / float(px.iloc[0])
    cash = 0.0
    curve: dict[pd.Timestamp, float] = {}
    n_trades = wins = 0
    i = 0
    open_call: tuple[date, float, float] | None = None  # (expiry, strike, credit)

    while i < len(dates):
        ts = dates[i]
        today = ts.date()
        spot = float(px.loc[ts])

        # Settle an expiring call.
        if open_call and today >= open_call[0]:
            expiry, strike, credit = open_call
            n_trades += 1
            if spot > strike:  # assigned: shares sold at strike, repurchased at market
                proceeds = shares * strike * (1 - SHARE_COST_BPS)
                shares = (proceeds * (1 - SHARE_COST_BPS)) / spot
            else:
                wins += 1  # expired worthless, full premium kept
            open_call = None

        # Sell a new call when flat.
        if open_call is None:
            for expiry in _friday_expiries(expiries, today, cfg.dte_window)[:3]:
                try:
                    chain = data.get_chain(symbol, datetime.combine(today, SNAP),
                                           expiries=[expiry], include_liquidity=False)
                except DataUnavailableError:
                    continue
                leg = nearest_delta(chain, expiry, OptionRight.CALL, cfg.covered_call_delta)
                if leg and leg.bid > 0:
                    contracts = max(int(shares // 100), 0)
                    if contracts >= 1:
                        cash += leg.bid * contracts * 100.0   # sold at the BID
                        open_call = (expiry, leg.key.strike, leg.bid)
                    break

        # Mark: shares plus cash minus the liability of the short call.
        liability = 0.0
        if open_call:
            expiry, strike, _ = open_call
            liability = max(spot - strike, 0.0) * max(int(shares // 100), 0) * 100.0
        curve[ts] = shares * spot + cash - liability
        i += 1

    return SymbolResult(symbol, "covered_calls", pd.Series(curve), n_trades, wins)


def put_credit_spreads(
    symbol: str, prices: pd.Series, data: ThetaDataHistorical, cfg: PerSymbolConfig,
    expiries: list[date], starting: float = 100_000.0, risk_fraction: float = 0.05,
) -> SymbolResult:
    """v8's structure applied to one name: defined-risk short put spreads."""
    px = prices.dropna()
    if px.empty:
        return SymbolResult(symbol, "put_credit_spread", pd.Series(dtype=float))
    dates = list(px.index)
    cash = starting
    equity = starting
    curve: dict[pd.Timestamp, float] = {}
    n_trades = wins = 0
    open_pos = None  # (expiry, short_strike, long_strike, credit, width, contracts)

    for i, ts in enumerate(dates):
        today = ts.date()

        if open_pos:
            expiry, ks, kl, credit, width, contracts = open_pos
            liability = None
            try:
                chain = data.get_chain(symbol, datetime.combine(today, SNAP),
                                       expiries=[expiry], include_liquidity=False)
                short = next((c for c in chain.contracts if c.key.strike == ks
                              and c.key.right is OptionRight.PUT), None)
                wing = next((c for c in chain.contracts if c.key.strike == kl
                             and c.key.right is OptionRight.PUT), None)
                if short and wing:
                    liability = max(min(short.ask - wing.bid, width), 0.0)
            except DataUnavailableError:
                pass
            if liability is not None:
                captured = (credit - liability) / credit
                close = (captured >= cfg.tp_credit_fraction
                         or liability >= cfg.stop_credit_multiple * credit
                         or (expiry - today).days <= cfg.close_before_expiry_days)
                if close:
                    cash -= liability * contracts * 100.0
                    n_trades += 1
                    if liability < credit:
                        wins += 1
                    open_pos = None
                else:
                    equity = cash - liability * contracts * 100.0
                    curve[ts] = equity
                    continue

        if open_pos is None and i % cfg.entry_every_days == 0:
            spot = float(px.loc[ts])
            for expiry in _friday_expiries(expiries, today, cfg.dte_window)[:3]:
                try:
                    chain = data.get_chain(symbol, datetime.combine(today, SNAP),
                                           expiries=[expiry], include_liquidity=False)
                except DataUnavailableError:
                    continue
                short = nearest_delta(chain, expiry, OptionRight.PUT, cfg.put_spread_delta)
                if short is None:
                    continue
                lower = [c for c in quotable(chain.slice(expiry=expiry, right=OptionRight.PUT))
                         if c.key.strike < short.key.strike]
                if not lower:
                    continue
                target = short.key.strike - spot * cfg.wing_width_pct
                wing = min(lower, key=lambda c: abs(c.key.strike - target))
                credit = short.bid - wing.ask          # worst-case fills
                width = short.key.strike - wing.key.strike
                if credit <= 0 or width <= 0 or credit >= width:
                    continue
                max_loss = width - credit
                contracts = max(int((equity * risk_fraction) // (max_loss * 100.0)), 0)
                if contracts < 1:
                    break
                cash += credit * contracts * 100.0
                open_pos = (expiry, short.key.strike, wing.key.strike, credit, width, contracts)
                break

        liability = 0.0
        if open_pos:
            liability = open_pos[3] * open_pos[5] * 100.0  # hold at entry value if unmarked
        equity = cash - liability
        curve[ts] = equity

    return SymbolResult(symbol, "put_credit_spread", pd.Series(curve), n_trades, wins)
