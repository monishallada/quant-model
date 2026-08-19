"""The intraday strategy contract — minute-resolution decisions, same walls.

An intraday strategy sees only what existed at decision time and returns only
intent. The walls from the daily contract hold unchanged: no broker, no
RiskManager, no cost model, no position book. The engine dedups entries
(one open position per (strategy, symbol)), interprets exits, prices fills and
enforces risk — a strategy that needed to see its own positions to behave is a
strategy whose exit logic belongs in ExitRules.

Timestamp convention (leak-critical, tested):

    A minute bar labeled T covers [T, T+1min) and becomes VISIBLE at T+1min.
    ``on_minute(ts)`` is a decision at time ts; ``ctx.bars`` contains only bars
    labeled <= ts - 1min. Orders decided at ts fill from the market at ts
    (equities: the open of the bar labeled ts; options: contract NBBO at ts).

So the loop is: see everything through ts-1min -> decide at ts -> fill at ts.
Nothing decided at ts can be priced with information formed after ts.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Mapping

import pandas as pd

from catalyst.core.interfaces.strategy import Cadence, Opportunity, Strategy, StrategyContext
from catalyst.core.types import OptionChain, ProposedTrade


@dataclass(frozen=True)
class IntradayContext:
    """Read-only view of the world at one decision minute."""

    session: date
    now: datetime                      # decision time ts (ET, tz-naive)
    bars: Mapping[str, pd.DataFrame]   # VISIBLE bars only: labels <= now - 1min
    data: Any = None                   # DataSource, for prior-session chain context

    def spot(self, symbol: str) -> float | None:
        """Last visible close — the freshest price a decision may use."""
        df = self.bars.get(symbol)
        if df is None or df.empty:
            return None
        return float(df["close"].iloc[-1])

    def chain_snapshot(self, symbol: str, expiries: list[date] | None = None
                       ) -> OptionChain | None:
        """PRIOR-session 15:45 chain, for strike-selection context only.

        Deliberately stale: the cached EOD snapshot is the point-in-time chain
        a live system would have had on file overnight. Entry pricing never
        comes from here — the engine prices fills from the contract's own
        minute NBBO at the decision minute.
        """
        if self.data is None:
            return None
        from datetime import time as dtime, timedelta

        prior = self.session - timedelta(days=1)
        while prior.weekday() >= 5:
            prior -= timedelta(days=1)
        try:
            return self.data.get_chain(symbol, datetime.combine(prior, dtime(15, 45)),
                                       expiries=expiries, include_liquidity=False)
        except Exception:                               # noqa: BLE001 — gap = no context
            return None


class IntradayStrategy(Strategy):
    """Minute-driven strategy. Emits unsized intent; everything else is engine."""

    cadence = Cadence.INTRADAY

    @abstractmethod
    def session_universe(self, session: date) -> list[str]:
        """Symbols whose minute bars this strategy needs today.

        Declared up front because minute data is fetched per symbol-day and
        billed in requests: the engine loads exactly this, nothing more.
        """

    @abstractmethod
    def on_minute(self, ctx: IntradayContext) -> list[ProposedTrade]:
        """Entry intent at this minute (may be empty; usually is).

        The engine enforces one open position per (strategy, symbol): emitting
        the same intent every minute while a position is open is expected and
        harmless, not a pyramiding instruction.
        """

    # -- daily-pipeline hooks: an intraday strategy never runs there --------
    def opportunities(self, session: date, ctx: StrategyContext) -> list[Opportunity]:
        return []

    def plan(self, opp: Opportunity, ctx: StrategyContext) -> ProposedTrade | None:
        return None
