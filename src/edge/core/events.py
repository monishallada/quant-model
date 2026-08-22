"""Event types and clocks for the event-driven core.

Every event carries a tz-aware ``ts`` normalized to America/New_York — a naive
timestamp is a construction error, never a silent assumption. Consumers read
time only through an :class:`EventClock`, so the SAME strategy/risk/execution
code runs in backtest (a :class:`BacktestClock` replaying historical ``ts``)
and live (a :class:`LiveClock` wrapping wall time). Both clocks are monotonic:
time never moves backwards.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Iterable, Iterator, Literal, Protocol, runtime_checkable
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

#: Exchange timezone every event timestamp is normalized to.
MARKET_TZ = ZoneInfo("America/New_York")


class _Event(BaseModel):
    """Base for all events: immutable, extras forbidden, tz-aware ``ts``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ts: datetime

    @field_validator("ts")
    @classmethod
    def _require_tz_aware(cls, value: datetime) -> datetime:
        """Reject naive timestamps; normalize aware ones to America/New_York."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("ts must be tz-aware (America/New_York); got a naive datetime")
        return value.astimezone(MARKET_TZ)


class Side(str, enum.Enum):
    """Direction of a signal, order, or fill."""

    BUY = "buy"
    SELL = "sell"


class QuoteEvent(_Event):
    """Top-of-book NBBO update for one symbol."""

    symbol: str
    bid: float = Field(ge=0.0)
    ask: float = Field(ge=0.0)
    bid_size: int = Field(default=0, ge=0)
    ask_size: int = Field(default=0, ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


class TradeEvent(_Event):
    """One printed trade (tape) for a symbol."""

    symbol: str
    price: float = Field(gt=0.0)
    size: int = Field(gt=0)


class BarEvent(_Event):
    """One completed bar; ``ts`` is the bar's close time."""

    symbol: str
    open: float = Field(gt=0.0)
    high: float = Field(gt=0.0)
    low: float = Field(gt=0.0)
    close: float = Field(gt=0.0)
    volume: int = Field(ge=0)


class SignalEvent(_Event):
    """A strategy's directional opinion, sized later by risk — never here."""

    symbol: str
    side: Side
    #: Strength in [0, 1]; risk scales exposure with it, 0 means no trade.
    conviction: float = Field(ge=0.0, le=1.0)
    horizon_minutes: int = Field(gt=0)
    signal_name: str


class OrderEvent(_Event):
    """An instruction sent toward the market (real or simulated)."""

    order_id: str
    symbol: str
    side: Side
    quantity: int = Field(gt=0)
    order_type: Literal["market", "limit"]
    limit_price: float | None = Field(default=None, gt=0.0)
    signal_name: str | None = None


class FillEvent(_Event):
    """Confirmation that (part of) an order executed at a price."""

    order_id: str
    symbol: str
    side: Side
    quantity: int = Field(gt=0)
    price: float = Field(gt=0.0)
    commission: float = Field(default=0.0, ge=0.0)


@runtime_checkable
class EventClock(Protocol):
    """Monotonic time source consumers query instead of the wall clock.

    Contract: ``now()`` returns a tz-aware America/New_York datetime and never
    moves backwards across calls. Strategy/risk/execution code depends only on
    this protocol, so backtest replay and live trading share one code path.
    """

    def now(self) -> datetime:
        """Current time in America/New_York; monotonically non-decreasing."""
        ...


class BacktestClock:
    """Clock driven by historical event timestamps during replay.

    ``advance()`` moves the clock to an event's ``ts`` and raises on any
    attempt to move backwards — an out-of-order feed is a data bug, not
    something to paper over. ``replay()`` advances through an event stream,
    yielding each event once the clock has reached it.
    """

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None or start.utcoffset() is None:
            raise ValueError("BacktestClock start must be tz-aware")
        self._now = start.astimezone(MARKET_TZ)

    def now(self) -> datetime:
        return self._now

    def advance(self, ts: datetime) -> None:
        """Move the clock forward to ``ts``; raise if ``ts`` is in the past."""
        if ts.tzinfo is None or ts.utcoffset() is None:
            raise ValueError("advance() requires a tz-aware timestamp")
        ts = ts.astimezone(MARKET_TZ)
        if ts < self._now:
            raise ValueError(f"clock is monotonic: cannot advance from {self._now} back to {ts}")
        self._now = ts

    def replay(self, events: Iterable[_Event]) -> Iterator[_Event]:
        """Yield events in stream order, advancing the clock to each ``ts``."""
        for event in events:
            self.advance(event.ts)
            yield event


class LiveClock:
    """Clock wrapping wall time, expressed in America/New_York."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc).astimezone(MARKET_TZ)
