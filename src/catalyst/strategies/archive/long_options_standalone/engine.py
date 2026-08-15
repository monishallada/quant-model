"""v9 standalone loop — the ARCHIVED reference implementation.

This is the code that produced the reported v9 campaign numbers (785 positions,
mean multiple 1.131x, TSLA P(10x)=7.3%). It is kept runnable and unchanged so
those results stay reproducible and can be compared against the pipeline-routed
version in ``strategies/active/long_options.py``.

It is deliberately NOT the active strategy, because it demonstrates exactly the
problem this repository was refactored to remove: it runs its own loop, prices
fills by reading ``.ask``/``.bid`` directly, and never constructs an ``Order``.
So it pays no commission and no slippage, and never meets the RiskManager — the
sizing here is a raw ``capital_fraction``, not a risk-gated quantity. Its
numbers are therefore optimistic relative to every campaign that paid costs,
which is precisely why the single cost model now exists.

The split fix IS present and must stay: strike selection uses the chain's own
contemporaneous spot, and a contract missing at exit is counted unpriceable
rather than marked at a fabricated intrinsic value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time

import numpy as np
import pandas as pd

from catalyst.core.chains import quotable
from catalyst.core.types import OptionRight
from catalyst.data.thetadata_historical import DataUnavailableError, ThetaDataHistorical

logger = logging.getLogger(__name__)

SNAP = time(15, 45)


@dataclass
class LongOptionConfig:
    hold_days: int = 15            # trading days held per cycle
    dte_days: int = 35             # tenor at entry
    moneyness: float = 1.05        # strike / spot for calls (mirrored for puts)
    capital_fraction: float = 0.50 # deployed per cycle; the rest survives a zero
    entry_every_days: int = 15
    use_signal: bool = True        # False = always long calls (directional control)
    allow_puts: bool = True        # False = calls only
    momentum_lookback: int = 60


@dataclass
class LongOptionTrade:
    symbol: str
    entry: date
    exit: date
    right: str
    strike: float
    entry_price: float
    exit_price: float
    multiple: float


@dataclass
class LongOptionResult:
    symbol: str
    equity: pd.Series
    trades: list[LongOptionTrade] = field(default_factory=list)
    ruined_on: date | None = None   # set when equity fell below the ruin floor
    unpriceable: int = 0            # cycles skipped: contract absent at exit (splits/gaps)

    @property
    def ruined(self) -> bool:
        return self.ruined_on is not None

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return float(np.mean([t.multiple > 1.0 for t in self.trades]))

    @property
    def best_multiple(self) -> float:
        return max((t.multiple for t in self.trades), default=0.0)

    @property
    def wipeout_rate(self) -> float:
        """Share of positions that expired essentially worthless — the cost of
        convexity, and the reason capital_fraction must stay below 1."""
        if not self.trades:
            return 0.0
        return float(np.mean([t.multiple < 0.02 for t in self.trades]))


def _friday_expiries(available: list[date], today: date, dte: int, tol: int = 12) -> list[date]:
    elig = [e for e in sorted(available) if dte - tol <= (e - today).days <= dte + tol]
    fri = [e for e in elig if e.weekday() == 4]
    ordered = fri + [e for e in elig if e not in fri]
    return sorted(ordered, key=lambda e: abs((e - today).days - dte))


def run_long_options(
    symbol: str, prices: pd.Series, data: ThetaDataHistorical, cfg: LongOptionConfig,
    expiries: list[date], starting: float = 100_000.0, ruin_floor: float = 0.01,
) -> LongOptionResult:
    """Buy calls when momentum is positive, puts when negative; hold; repeat."""
    px = prices.dropna()
    if px.empty:
        return LongOptionResult(symbol, pd.Series(dtype=float))
    dates = list(px.index)
    equity = starting
    curve: dict[pd.Timestamp, float] = {dates[0]: equity}
    trades: list[LongOptionTrade] = []
    ruined_on: date | None = None
    unpriceable = 0
    i = 0

    while i + cfg.hold_days < len(dates):
        entry_ts, exit_ts = dates[i], dates[i + cfg.hold_days]
        entry_day, exit_day = entry_ts.date(), exit_ts.date()

        # Direction from trailing momentum, using only prior closes.
        hist = px.loc[:entry_ts]
        if cfg.use_signal and len(hist) > cfg.momentum_lookback:
            mom = float(hist.iloc[-1] / hist.iloc[-cfg.momentum_lookback] - 1.0)
        else:
            mom = 1.0
        right = OptionRight.CALL if (mom >= 0 or not cfg.allow_puts) else OptionRight.PUT

        filled = False
        for expiry in _friday_expiries(expiries, entry_day, cfg.dte_days)[:3]:
            try:
                chain = data.get_chain(symbol, datetime.combine(entry_day, SNAP),
                                       expiries=[expiry], include_liquidity=False)
            except DataUnavailableError:
                continue
            # Strike selection MUST use the chain's own contemporaneous spot.
            # Alpaca daily bars are split-ADJUSTED; ThetaData strikes are raw
            # as-of-date. Mixing them targeted a $165 AMZN strike against a
            # chain running $860-$5300 (ratio 20.03x = the 20-for-1 split) and
            # silently bought deep-ITM contracts for years. Unsplit MSFT
            # matched 1.00x, which is why the error hid on half the universe.
            raw_spot = chain.underlying_price
            if raw_spot <= 0:
                continue
            strike_target = (raw_spot * cfg.moneyness if right is OptionRight.CALL
                             else raw_spot * (2.0 - cfg.moneyness))
            candidates = [c for c in quotable(chain.slice(expiry=expiry, right=right))
                          if c.ask > 0.05]
            if not candidates:
                continue
            pick = min(candidates, key=lambda c: abs(c.key.strike - strike_target))
            entry_price = pick.ask                      # pay the ASK
            # Exit: the SAME contract at the BID. If it is absent from the exit
            # chain the terms changed (split/adjustment) or the data is missing
            # — either way the position is unpriceable. Skip it. The previous
            # intrinsic fallback fabricated prices across split boundaries and
            # manufactured a 924x put.
            try:
                exit_chain = data.get_chain(symbol, datetime.combine(exit_day, SNAP),
                                            expiries=[expiry], include_liquidity=False)
            except DataUnavailableError:
                break
            match = exit_chain.find(pick.key)
            if match is None:
                unpriceable += 1
                break
            exit_price = match.bid if match.bid > 0 else 0.0

            multiple = exit_price / entry_price if entry_price > 0 else 0.0
            deployed = equity * cfg.capital_fraction
            equity = equity - deployed + deployed * multiple
            trades.append(LongOptionTrade(symbol, entry_day, exit_day, right.value,
                                          pick.key.strike, entry_price, exit_price, multiple))
            filled = True
            break

        curve[exit_ts] = equity
        # Ruin is a REPORTED OUTCOME, not a silent early exit. A run that stops
        # here has a trade count far below the schedule (TSLA: 13 of ~144),
        # which reads like missing data unless the wipeout is stated outright.
        if equity < starting * ruin_floor:
            ruined_on = exit_day
            break
        i += cfg.entry_every_days if filled else cfg.hold_days

    # Reindex onto every session and forward-fill. Both avg_monthly_return and
    # rolling_windows derive elapsed time from the number of rows, so a curve
    # marked only at exits silently compresses years into weeks.
    #
    # Forward-filling means equity steps at exits rather than moving daily, so
    # max drawdown is measured exit-to-exit and understates the intra-hold
    # swing. That is the decision-relevant series here — there is no intra-hold
    # stop, so a mark between entry and exit is not something we could act on —
    # but the reported drawdown is a floor, not the true worst mark.
    eq = pd.Series(curve).sort_index().reindex(pd.DatetimeIndex(dates)).ffill().dropna()
    return LongOptionResult(symbol, eq, trades, ruined_on, unpriceable)
