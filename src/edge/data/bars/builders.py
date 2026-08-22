"""Bar builders: TradeEvent streams -> completed BarEvents labeled at close.

Every builder consumes one symbol's trades in non-decreasing ``ts`` order and
lazily yields :class:`~edge.core.events.BarEvent`. ``ts`` on each bar is the
bar's CLOSE time, and a bar is emitted only once its closing condition has
been OBSERVED inside the stream: the trade that crosses the threshold
(volume / dollar / imbalance bars) or the first trade proving the interval
elapsed (time bars). A consumer replaying events therefore sees every bar
strictly after its constituent trades — no lookahead. A trailing bar whose
closing condition never arrives is dropped, never flushed. Out-of-order
timestamps or a symbol change mid-stream raise ``ValueError``: a malformed
feed is a data bug, not something to paper over.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta, timezone

import pandas as pd

from edge.core.events import MARKET_TZ, BarEvent, TradeEvent

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class _Acc:
    """OHLCV state of one open bar, accumulated over constituent trades."""

    __slots__ = ("open", "high", "low", "close", "volume")

    def __init__(self, trade: TradeEvent) -> None:
        self.open = trade.price
        self.high = trade.price
        self.low = trade.price
        self.close = trade.price
        self.volume = trade.size

    def add(self, trade: TradeEvent) -> None:
        self.high = max(self.high, trade.price)
        self.low = min(self.low, trade.price)
        self.close = trade.price
        self.volume += trade.size

    def emit(self, symbol: str, ts: datetime) -> BarEvent:
        return BarEvent(
            ts=ts,
            symbol=symbol,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )


def _checked(trades: Iterable[TradeEvent]) -> Iterator[TradeEvent]:
    """Yield trades, enforcing single-symbol and non-decreasing ``ts``."""
    symbol: str | None = None
    prev: datetime | None = None
    for trade in trades:
        if symbol is None:
            symbol = trade.symbol
        elif trade.symbol != symbol:
            raise ValueError(f"mixed symbols in one bar stream: {symbol!r} then {trade.symbol!r}")
        if prev is not None and trade.ts < prev:
            raise ValueError(f"out-of-order trade stream: {trade.ts} after {prev}")
        prev = trade.ts
        yield trade


def time_bars(trades: Iterable[TradeEvent], interval: timedelta) -> Iterator[BarEvent]:
    """Fixed-interval bars aligned to epoch multiples of ``interval``.

    A bar covering ``[t0, t0 + interval)`` closes at ``t0 + interval`` and is
    emitted when the first trade at or past that boundary arrives — the
    in-stream proof the interval elapsed. Its ``ts`` is the boundary itself,
    which never precedes any constituent trade. Intervals without trades
    produce no bars (no zero-volume padding), and the trailing partial bar is
    dropped.
    """
    if interval <= timedelta(0):
        raise ValueError(f"interval must be positive, got {interval}")
    acc: _Acc | None = None
    bucket = 0
    for trade in _checked(trades):
        n = (trade.ts.astimezone(timezone.utc) - _EPOCH) // interval
        if acc is not None and n != bucket:
            close = (_EPOCH + (bucket + 1) * interval).astimezone(MARKET_TZ)
            yield acc.emit(trade.symbol, close)
            acc = None
        if acc is None:
            bucket = n
            acc = _Acc(trade)
        else:
            acc.add(trade)


def volume_bars(trades: Iterable[TradeEvent], volume_per_bar: int) -> Iterator[BarEvent]:
    """Bars closing on the trade that lifts cumulative volume to the threshold.

    The crossing trade is INCLUDED in the closing bar — trades are never
    split — so realized bar volume may overshoot ``volume_per_bar``. Bar
    ``ts`` is the crossing trade's ``ts``.
    """
    if volume_per_bar <= 0:
        raise ValueError(f"volume_per_bar must be positive, got {volume_per_bar}")
    acc: _Acc | None = None
    for trade in _checked(trades):
        if acc is None:
            acc = _Acc(trade)
        else:
            acc.add(trade)
        if acc.volume >= volume_per_bar:
            yield acc.emit(trade.symbol, trade.ts)
            acc = None


def dollar_bars(trades: Iterable[TradeEvent], dollar_per_bar: float) -> Iterator[BarEvent]:
    """Bars closing on the trade that lifts cumulative price*size notional to
    the threshold. The crossing trade is included (never split); bar ``ts`` is
    the crossing trade's ``ts``."""
    if dollar_per_bar <= 0:
        raise ValueError(f"dollar_per_bar must be positive, got {dollar_per_bar}")
    acc: _Acc | None = None
    dollars = 0.0
    for trade in _checked(trades):
        if acc is None:
            acc = _Acc(trade)
        else:
            acc.add(trade)
        dollars += trade.price * trade.size
        if dollars >= dollar_per_bar:
            yield acc.emit(trade.symbol, trade.ts)
            acc = None
            dollars = 0.0


def imbalance_bars(
    trades: Iterable[TradeEvent],
    *,
    expected_ticks_init: float,
    expected_imbalance_init: float,
    alpha_ticks: float,
    alpha_imbalance: float,
    min_expected_imbalance: float,
) -> Iterator[BarEvent]:
    """Tick imbalance bars (Lopez de Prado, AFML ch. 2, sec. 2.3.2.1).

    Tick rule: ``b_t = +1`` if the price ticked up, ``-1`` if down, else
    ``b_{t-1}`` (the first trade has no prior price and is seeded ``+1``, the
    book's arbitrary-seed convention). The bar accumulates the signed-tick
    imbalance ``theta_T = sum(b_t)`` and closes at the FIRST trade where::

        |theta_T| >= E[T] * max(|E[b]|, min_expected_imbalance)

    which is AFML's ``T* = argmin{T : |theta_T| >= E0[T] * |2P[b=1] - 1|}``
    with ``2P[b=1] - 1`` tracked as an EWMA of tick signs. The recursion:

    - per tick:      ``E[b] <- (1 - alpha_imbalance) * E[b] + alpha_imbalance * b_t``
      (seeded with ``expected_imbalance_init``, updated before the check);
    - per bar close: ``E[T] <- (1 - alpha_ticks) * E[T] + alpha_ticks * T_realized``
      (seeded with ``expected_ticks_init``), then ``theta`` resets to 0.

    ``min_expected_imbalance`` floors ``|E[b]|`` because the raw recursion
    degenerates in balanced flow: ``E[b] -> 0`` collapses the threshold and
    every tick closes a bar. With the floor, balanced flow keeps ``theta``
    oscillating below threshold (few bars) while one-sided flow drives
    ``theta`` up one tick per trade and closes bars early. Bar ``ts`` is the
    crossing trade's ``ts``.
    """
    if expected_ticks_init <= 0:
        raise ValueError(f"expected_ticks_init must be positive, got {expected_ticks_init}")
    if not 0.0 < alpha_ticks <= 1.0 or not 0.0 < alpha_imbalance <= 1.0:
        raise ValueError(
            f"EWMA alphas must be in (0, 1], got alpha_ticks={alpha_ticks}, "
            f"alpha_imbalance={alpha_imbalance}"
        )
    if min_expected_imbalance <= 0:
        raise ValueError(f"min_expected_imbalance must be positive, got {min_expected_imbalance}")

    exp_ticks = expected_ticks_init
    exp_b = expected_imbalance_init
    prev_price: float | None = None
    b = 1
    theta = 0
    count = 0
    acc: _Acc | None = None
    for trade in _checked(trades):
        if prev_price is not None:
            if trade.price > prev_price:
                b = 1
            elif trade.price < prev_price:
                b = -1
            # unchanged price: b inherits the previous sign (tick rule)
        prev_price = trade.price
        exp_b = (1.0 - alpha_imbalance) * exp_b + alpha_imbalance * b
        if acc is None:
            acc = _Acc(trade)
        else:
            acc.add(trade)
        theta += b
        count += 1
        threshold = exp_ticks * max(abs(exp_b), min_expected_imbalance)
        if abs(theta) >= threshold:
            yield acc.emit(trade.symbol, trade.ts)
            exp_ticks = (1.0 - alpha_ticks) * exp_ticks + alpha_ticks * count
            acc = None
            theta = 0
            count = 0


def bars_to_frame(bars: Iterable[BarEvent]) -> pd.DataFrame:
    """Completed bars as a ts-indexed OHLCV frame (research convenience);
    empty input yields an empty typed frame with a DatetimeIndex."""
    rows = [
        {"ts": b.ts, "open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume}
        for b in bars
    ]
    if not rows:
        frame = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        frame.index = pd.DatetimeIndex([], name="ts")
        return frame
    return pd.DataFrame(rows).set_index("ts")
