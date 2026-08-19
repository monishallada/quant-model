"""The plug-in contract.

Every strategy — catalyst-driven, scheduled, or cross-sectional — implements
this one interface and flows through one fixed pipeline:

    screener -> strategy -> RiskManager -> cost/fill model -> exits
             -> metrics -> report

The direction of control is the invariant that makes the guarantees real: the
**pipeline calls the strategy, never the reverse**. A strategy receives an
``Opportunity`` and a read-only ``StrategyContext`` and returns trade *intent*.
It holds no Broker, no RiskManager, and no cost model, so there is no code path
by which a strategy can size its own position, price its own fill, or skip the
out-of-sample split. Those are not conventions to be remembered; they are
absent from the type signatures.

Three cadences cover every campaign this project has run, and the pipeline
drives each identically:

- ``CATALYST``  — opportunities come from screener catalyst hits (Engines A-D)
- ``SCHEDULED`` — opportunities recur every N sessions (v9 long options, v8 VRP)
- ``DAILY``     — one cross-sectional opportunity per session (v5 alpha, pairs)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping

import pandas as pd

from catalyst.core.types import Catalyst, OptionChain, ProposedTrade, SignalResult


class Cadence(str, Enum):
    """What makes an opportunity appear. Selects how the pipeline enumerates
    work for this strategy — it never changes how the strategy is judged."""

    CATALYST = "catalyst"
    SCHEDULED = "scheduled"
    DAILY = "daily"
    #: Minute-resolution decision points inside the session; driven by the
    #: IntradayBacktester, not the daily loop.
    INTRADAY = "intraday"


@dataclass(frozen=True)
class Opportunity:
    """One unit of work handed to a strategy by the pipeline.

    ``symbols`` is a list rather than a single ticker so cross-sectional
    strategies (pairs, alpha portfolios) are first-class rather than bolted on.
    ``catalyst`` is populated only for CATALYST cadence.
    """

    session: date
    symbols: tuple[str, ...]
    catalyst: Catalyst | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def symbol(self) -> str:
        """The single underlying, for the common one-name case."""
        return self.symbols[0]


@dataclass(frozen=True)
class StrategyContext:
    """Read-only view of the world a strategy is allowed to see.

    Deliberately excludes Broker, RiskManager and the cost model. A strategy
    cannot reach them, so it cannot bypass them.
    """

    as_of: datetime
    data: Any                      # DataSource (typed loosely to avoid a cycle)
    signal: SignalResult | None = None
    history: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    chains: Mapping[str, OptionChain] = field(default_factory=dict)

    def chain(self, symbol: str, expiries: list[date] | None = None) -> OptionChain | None:
        """The chain the pipeline already pulled for this session, if any.

        Serving the pre-pulled slice matters for more than speed: it is what
        makes routing an engine through the generalized contract give it the
        byte-identical inputs it saw before the refactor. Historical chain
        assembly is billed per expiration, so re-fetching would also cost real
        requests for no gain.
        """
        cached = self.chains.get(symbol)
        if cached is not None:
            return cached
        # Unavailable data is NORMAL — holidays, thin names, gaps in coverage.
        # Returning None lets plan() honour its documented contract ("None when
        # you cannot act") instead of forcing every strategy to wrap this in a
        # try/except, which is the kind of boilerplate that eventually gets
        # copied wrong and swallows a real error.
        try:
            return self.data.get_chain(symbol, self.as_of, expiries=expiries)
        except Exception:                                # noqa: BLE001
            return None


class Strategy(ABC):
    """Trade intent only. No data mutation, no sizing, no fills, no orders."""

    name: str = "unnamed"
    cadence: Cadence = Cadence.CATALYST

    #: Set by the archive tooling; the deploy gate reads these and refuses to
    #: run anything in live mode that has not passed backtest AND paper.
    validated: bool = False
    paper_tested: bool = False

    @abstractmethod
    def opportunities(self, session: date, ctx: StrategyContext) -> list[Opportunity]:
        """Everything this strategy would consider acting on today.

        Enumeration only — no trade decision is made here, so the pipeline can
        count and log candidate flow independently of conversion.
        """

    @abstractmethod
    def plan(self, opp: Opportunity, ctx: StrategyContext) -> ProposedTrade | None:
        """Unsized trade intent for one opportunity, or None if gates fail.

        The returned ``ProposedTrade`` carries legs, direction, exit rules and
        a worst-case ``unit_max_loss``. It carries no quantity: sizing belongs
        to the RiskManager and is not negotiable by the strategy.
        """

    def required_expiries(
        self, opp: Opportunity, available: list[date], as_of: date
    ) -> list[date] | None:
        """Expirations worth fetching for this opportunity.

        Historical chain assembly is billed per expiration, so narrowing this
        is a real cost saving. None means "no restriction".
        """
        return None


class CatalystStrategy(Strategy):
    """Adapter for the catalyst-driven engines.

    Preserves the original ``evaluate(catalyst, chain, signal, as_of)`` shape
    exactly, so Engines A-D, Drift and Kinetic keep their logic byte-for-byte
    while still being driven by the generalized pipeline.
    """

    cadence: Cadence = Cadence.CATALYST

    @abstractmethod
    def evaluate(
        self,
        catalyst: Catalyst,
        chain: OptionChain,
        signal: SignalResult,
        as_of: datetime,
    ) -> ProposedTrade | None: ...

    def opportunities(self, session: date, ctx: StrategyContext) -> list[Opportunity]:
        cats = ctx.data.get_catalyst_calendar(session, session)
        return [Opportunity(session=session, symbols=(c.symbol,), catalyst=c) for c in cats]

    def plan(self, opp: Opportunity, ctx: StrategyContext) -> ProposedTrade | None:
        if opp.catalyst is None:
            return None
        chain = ctx.chain(opp.symbol)
        if chain is None:
            return None
        return self.evaluate(opp.catalyst, chain, ctx.signal, ctx.as_of)

    def required_expiries(
        self, opp: Opportunity, available: list[date], as_of: date
    ) -> list[date] | None:
        if opp.catalyst is None:
            return None
        return self.catalyst_expiries(opp.catalyst, available, as_of)

    def catalyst_expiries(
        self, catalyst: Catalyst, available: list[date], as_of: date
    ) -> list[date] | None:
        """Catalyst-shaped expiry hint. Engines override this; the pipeline
        only ever calls ``required_expiries``."""
        return None
