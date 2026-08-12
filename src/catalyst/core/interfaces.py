"""Abstract interfaces that make identical strategy code run in backtest,
paper, and live: only the injected DataSource/Broker implementations differ.

Architectural invariant (enforced across the codebase, see risk/manager.py):
Strategy engines produce ``ProposedTrade`` objects and hold no Broker
reference. Every order reaches a Broker only through the execution layer,
which requires a RiskManager approval. Nothing overrides the risk layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any

import pandas as pd

from catalyst.core.models import (
    AccountState,
    Catalyst,
    Greeks,
    OptionChain,
    OptionKey,
    Order,
    OrderResult,
    Position,
    ProposedTrade,
    Quote,
    SignalResult,
)


class DataSource(ABC):
    """Market + catalyst data. Implementations: ThetaDataHistorical, LiveDataSource."""

    @abstractmethod
    def get_chain(
        self,
        symbol: str,
        at: datetime,
        expiries: list[date] | None = None,
        max_dte: int | None = None,
    ) -> OptionChain:
        """Option chain snapshot for ``symbol`` as of ``at`` (ET).

        ``expiries`` restricts the snapshot to specific expirations and
        ``max_dte`` caps days-to-expiry (default: config horizon). Callers
        should request only what they need — historical chain assembly is
        billed in terminal requests per expiration.
        """

    @abstractmethod
    def list_expirations(self, symbol: str) -> list[date]:
        """All known option expirations for ``symbol`` (past and future)."""

    @abstractmethod
    def get_quote(self, symbol: str, at: datetime | None = None) -> Quote:
        """Underlying quote (live: now; backtest: as of ``at``)."""

    @abstractmethod
    def get_greeks(self, key: OptionKey, at: datetime) -> Greeks:
        """Greeks for one contract as of ``at``."""

    @abstractmethod
    def get_history(
        self, symbol: str, start: date, end: date, interval: str = "1d"
    ) -> pd.DataFrame:
        """Underlying OHLCV history, indexed by timestamp, columns open/high/low/close/volume."""

    @abstractmethod
    def get_catalyst_calendar(self, start: date, end: date) -> list[Catalyst]:
        """Known catalysts (earnings/CPI/FOMC/...) in [start, end]."""


class Broker(ABC):
    """Order routing + account state. Implementations: SimulatedBroker,
    AlpacaBroker, SchwabBroker (post-validation drop-in)."""

    @abstractmethod
    def place_order(self, order: Order) -> OrderResult: ...

    @abstractmethod
    def modify_order(self, order_id: str, changes: dict[str, Any]) -> OrderResult: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> OrderResult: ...

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Broker-truth open positions. Live code reconciles against this
        before every action — internal memory is never trusted alone."""

    @abstractmethod
    def get_account(self) -> AccountState: ...


class DirectionalSignal(ABC):
    """Directional opinion provider. Baselines ship as UNVALIDATED test
    candidates; the backtester's job is to find whether any has real
    out-of-sample edge (assume most do not)."""

    name: str = "unnamed"

    @abstractmethod
    def evaluate(
        self, symbol: str, history: pd.DataFrame, chain: OptionChain | None = None
    ) -> SignalResult: ...


class Strategy(ABC):
    """A strategy engine: turns (catalyst, chain, signal) into an unsized
    ProposedTrade, or None when its entry gates are not met."""

    name: str = "unnamed"

    @abstractmethod
    def evaluate(
        self,
        catalyst: Catalyst,
        chain: OptionChain,
        signal: SignalResult,
        as_of: datetime,
    ) -> ProposedTrade | None: ...

    def required_expiries(
        self, catalyst: Catalyst, available: list[date], as_of: date
    ) -> list[date] | None:
        """Expirations this engine could use for this catalyst right now.

        Lets the backtester pull only the needed slices of a chain (historical
        chain assembly costs terminal requests per expiration). None means
        "no restriction" — the full config-horizon chain is fetched.
        """
        return None
