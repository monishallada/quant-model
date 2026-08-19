"""Typed data models shared across backtest, paper, and live environments.

These are the *only* shapes strategy, risk, execution, and exit code ever see.
Broker- and vendor-specific formats (option symbol strings, payloads) live in
their adapters and never leak past them.
"""

from __future__ import annotations

import enum
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _MutableModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ---------------------------------------------------------------------------
# Option identity
# ---------------------------------------------------------------------------


class OptionRight(str, enum.Enum):
    CALL = "C"
    PUT = "P"


class OptionKey(_Model):
    """Canonical, broker-agnostic identity of one option contract."""

    underlying: str
    expiry: date
    right: OptionRight
    strike: float = Field(gt=0)

    #: Dollars of exposure per 1.0 of quoted price per unit. Money math must
    #: multiply by THIS, never by a literal 100 — that literal is what made the
    #: whole stack options-only.
    @property
    def multiplier(self) -> float:
        return 100.0

    def __str__(self) -> str:
        return f"{self.underlying} {self.expiry.isoformat()} {self.right.value} {self.strike:g}"


class EquityKey(_Model):
    """Canonical identity of an equity instrument (shares).

    Fields are disjoint from OptionKey (no expiry/right/strike) and _Model
    forbids extras, so the InstrumentKey union discriminates unambiguously.
    """

    underlying: str

    @property
    def multiplier(self) -> float:
        return 1.0

    def __str__(self) -> str:
        return f"{self.underlying} SHARES"


#: One leg trades exactly one instrument. Every ledger line multiplies price
#: by key.multiplier: 100 for listed US options, 1 for shares.
InstrumentKey = OptionKey | EquityKey


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


class Greeks(_Model):
    delta: float
    gamma: float | None = None  # second-order; fetched only when needed
    theta: float
    vega: float
    rho: float | None = None
    iv: float


class Quote(_Model):
    symbol: str
    bid: float
    ask: float
    last: float | None = None
    timestamp: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


class OptionContract(_Model):
    """One contract inside an OptionChain snapshot: identity + market data."""

    key: OptionKey
    bid: float
    ask: float
    volume: int = 0
    open_interest: int = 0
    greeks: Greeks | None = None
    quote_timestamp: datetime | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_pct_of_mid(self) -> float:
        """Bid/ask spread as a fraction of mid (inf when mid is 0)."""
        mid = self.mid
        if mid <= 0:
            return float("inf")
        return (self.ask - self.bid) / mid


class OptionChain(_Model):
    """Snapshot of an underlying's option chain at one moment."""

    underlying: str
    underlying_price: float
    timestamp: datetime
    contracts: list[OptionContract]

    def expirations(self) -> list[date]:
        return sorted({c.key.expiry for c in self.contracts})

    def slice(
        self,
        expiry: date | None = None,
        right: OptionRight | None = None,
    ) -> list[OptionContract]:
        out = self.contracts
        if expiry is not None:
            out = [c for c in out if c.key.expiry == expiry]
        if right is not None:
            out = [c for c in out if c.key.right == right]
        return out

    def find(self, key: OptionKey) -> OptionContract | None:
        for c in self.contracts:
            if c.key == key:
                return c
        return None


# ---------------------------------------------------------------------------
# Catalysts
# ---------------------------------------------------------------------------


class CatalystType(str, enum.Enum):
    EARNINGS = "earnings"
    CPI = "cpi"
    FOMC = "fomc"
    OTHER = "other"


class Catalyst(_Model):
    symbol: str  # underlying it applies to (index catalysts apply to SPY/QQQ/...)
    type: CatalystType
    when: datetime  # event datetime (ET); earnings BMO/AMC resolved to session edges
    source: str
    expected_move: float | None = None  # ATM-straddle implied move (fraction), when known
    # Earnings-specific (Engine C consumes the surprise sign — measured, not predicted):
    eps_actual: float | None = None
    eps_estimate: float | None = None

    @property
    def surprise_pct(self) -> float | None:
        """(actual - estimate) / |estimate|, None when either side is missing."""
        if self.eps_actual is None or self.eps_estimate is None or self.eps_estimate == 0:
            return None
        return (self.eps_actual - self.eps_estimate) / abs(self.eps_estimate)

    @property
    def ref(self) -> str:
        """Stable reference string for tagging orders/positions."""
        return f"{self.type.value}:{self.symbol}:{self.when.date().isoformat()}"


# ---------------------------------------------------------------------------
# Orders / positions / account
# ---------------------------------------------------------------------------


class Side(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class Direction(str, enum.Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"  # = no-trade for directional engines


class ExitRules(_Model):
    """Mechanical exit plan attached to every position at entry.

    The ExitManager is the only interpreter of these fields; engines only
    declare them. All values originate from config.
    """

    tp1_gain: float | None = None  # scale out tp1_fraction at this gain (e.g. 1.0 = +100%)
    tp1_fraction: float | None = None
    tp_fraction_of_max: float | None = None  # spreads: close at this fraction of max value
    # CREDIT structures (net premium received; entry_price < 0). Both default to
    # None so debit-only strategies keep exactly their previous behavior.
    tp_credit_fraction: float | None = None  # close after capturing this share of the credit
    stop_credit_multiple: float | None = None  # close if liability reaches N x credit received
    trail_stop_pct: float | None = None  # trailing stop on remainder after tp1
    stop_loss_pct: float | None = None  # soft stop as fraction of premium (negative)
    use_stops: bool = True
    hard_exit_date: date | None = None  # mandatory time stop (e.g. day before catalyst,
    #   or catalyst + N days) — never held past this
    max_hold_trading_days: int | None = None
    close_before_expiry_days: int = 1


class OrderLeg(_Model):
    key: InstrumentKey
    side: Side
    qty: int = Field(gt=0)


class OrderIntent(str, enum.Enum):
    OPEN = "open"
    CLOSE = "close"


class Order(_Model):
    legs: list[OrderLeg]  # qty = TOTAL contracts per leg (sized order, not per-unit)
    limit_price: float  # net price per 1x unit of the structure (debit > 0, credit < 0)
    time_in_force: str = "day"
    tag: str = ""  # "<engine>:<catalyst_ref>" — traceability for every order
    intent: OrderIntent = OrderIntent.OPEN
    position_id: str | None = None  # required when intent == CLOSE
    exit_rules: ExitRules | None = None  # attached on OPEN; stored on the resulting position
    direction: Direction | None = None  # thesis direction; stored on the position
    max_loss: float | None = None  # per-unit max loss; required for credit structures,
    #   where the premium paid is negative and cannot serve as the risk basis

    @property
    def is_multi_leg(self) -> bool:
        return len(self.legs) > 1

    @property
    def multiplier(self) -> float:
        return self.legs[0].key.multiplier if self.legs else 100.0

    @model_validator(mode="after")
    def _no_mixed_multipliers(self) -> "Order":
        """Shares and options in one order would make unit_cost meaningless —
        a '1x unit' must have one dollars-per-point scale."""
        if len({leg.key.multiplier for leg in self.legs}) > 1:
            raise ValueError("mixed instrument multipliers in one order")
        return self


class OrderStatus(str, enum.Enum):
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CANCELED = "canceled"


class OrderResult(_Model):
    order_id: str
    status: OrderStatus
    filled_qty: int = 0
    avg_fill_price: float | None = None  # net per 1x unit, includes modeled slippage
    commission: float = 0.0
    message: str = ""


class PositionLeg(_Model):
    key: InstrumentKey
    side: Side
    qty: int


class Position(_MutableModel):
    """An open structure. Mutable: qty scales out, marks update, state evolves."""

    position_id: str
    legs: list[PositionLeg]
    qty: int  # number of 1x units of the structure currently open
    entry_price: float  # net per 1x unit actually paid (debit > 0)
    entry_time: datetime
    engine_tag: str
    direction: Direction | None = None  # thesis direction (hedge sizing input)
    catalyst_ref: str = ""
    exit_rules: ExitRules
    current_value: float = 0.0  # net mark per 1x unit
    high_water_value: float = 0.0  # best mark seen, for trailing stops
    tp1_done: bool = False
    realized_pnl: float = 0.0  # from partial exits, total dollars
    max_loss: float | None = None  # per-unit max loss when it is not the debit paid

    @property
    def multiplier(self) -> float:
        """Instrument multiplier. Mixed-multiplier structures are forbidden at
        order construction, so the first leg speaks for all of them."""
        return self.legs[0].key.multiplier if self.legs else 100.0

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_value - self.entry_price) * self.qty * self.multiplier

    @property
    def premium_at_risk(self) -> float:
        """Max remaining loss in dollars.

        Debit structures: the premium still paid. Credit structures set
        ``max_loss`` explicitly (width - credit), because their entry price is
        negative and would otherwise report zero risk to the portfolio caps.
        """
        basis = self.max_loss if self.max_loss is not None else max(self.entry_price, 0.0)
        return max(basis, 0.0) * self.qty * self.multiplier

    @property
    def underlying(self) -> str:
        return self.legs[0].key.underlying


class AccountState(_Model):
    equity: float
    cash: float
    buying_power: float
    timestamp: datetime


# ---------------------------------------------------------------------------
# Strategy plumbing
# ---------------------------------------------------------------------------


class SignalResult(_Model):
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)


class ProposedTrade(_Model):
    """What an engine proposes. Unsized: RiskManager alone decides quantity.

    ``unit_cost`` / ``unit_max_loss`` describe ONE 1x unit of the structure in
    net premium per share (x100 for dollars per unit).
    """

    engine: str
    catalyst_ref: str
    legs: list[OrderLeg]  # qty on each leg is the per-unit ratio (usually 1)
    unit_cost: float  # estimated net debit per 1x unit (per share)
    unit_max_loss: float  # max loss per 1x unit (per share); debit structures = unit_cost
    direction: Direction
    exit_rules: ExitRules
    per_trade_risk_fraction: float  # engine's configured equity fraction (sizing input)
    rationale: dict[str, Any] = Field(default_factory=dict)  # logged, never parsed

    @property
    def multiplier(self) -> float:
        return self.legs[0].key.multiplier if self.legs else 100.0


class TradeRecord(_Model):
    """Closed-trade record emitted by the backtester / live reconciler."""

    position_id: str
    engine: str
    catalyst_ref: str
    underlying: str
    direction: Direction
    entry_time: datetime
    exit_time: datetime
    entry_price: float  # net per unit
    exit_price: float  # net per unit (weighted across partial exits)
    qty: int
    pnl: float  # dollars, after commissions/slippage
    exit_reason: str
    max_qty: int


# ---------------------------------------------------------------------------
# Backtest output
# ---------------------------------------------------------------------------


class MonthlyReturnStats(_Model):
    mean: float
    median: float
    std: float
    p05: float
    p25: float
    p75: float
    p95: float
    min: float
    max: float
    histogram_bins: list[float]  # bin edges
    histogram_counts: list[int]
    monthly_returns: dict[str, float]  # "YYYY-MM" -> return fraction


class BacktestResult(_Model):
    """Full metrics suite for one backtest run."""

    label: str
    start: date
    end: date
    starting_capital: float
    ending_equity: float
    total_return: float
    n_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    max_drawdown: float
    sharpe: float
    sortino: float
    calmar: float
    probability_of_ruin: float  # Monte Carlo: P(equity < ruin threshold)
    mc_survival_rate: float  # 1 - probability_of_ruin
    monthly: MonthlyReturnStats
    per_engine_pnl: dict[str, float] = Field(default_factory=dict)
    per_engine_trades: dict[str, int] = Field(default_factory=dict)
    trades: list[TradeRecord] = Field(default_factory=list)
    equity_curve: dict[str, float] = Field(default_factory=dict)  # ISO date -> equity
    config_overrides: dict[str, Any] = Field(default_factory=dict)  # sweep provenance

__all__ = [
    "AccountState",
    "EquityKey",
    "InstrumentKey",
    "BacktestResult",
    "Catalyst",
    "CatalystType",
    "Direction",
    "ExitRules",
    "Greeks",
    "MonthlyReturnStats",
    "OptionChain",
    "OptionContract",
    "OptionKey",
    "OptionRight",
    "Order",
    "OrderIntent",
    "OrderLeg",
    "OrderResult",
    "OrderStatus",
    "Position",
    "PositionLeg",
    "ProposedTrade",
    "Quote",
    "Side",
    "SignalResult",
    "TradeRecord",
]
