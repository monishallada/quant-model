"""Edge core: events construct (tz-aware, frozen), clocks replay in order,
config loads. All timestamps are synthetic and predate the lockbox window."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from edge.core.config import EdgeConfig, load_config
from edge.core.events import (
    BacktestClock,
    BarEvent,
    EventClock,
    FillEvent,
    LiveClock,
    MARKET_TZ,
    OrderEvent,
    QuoteEvent,
    Side,
    SignalEvent,
    TradeEvent,
)

ET = ZoneInfo("America/New_York")
# Synthetic session well BEFORE the 2026-02-22 lockbox start.
T0 = datetime(2026, 1, 5, 9, 30, tzinfo=ET)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def test_events_construct() -> None:
    quote = QuoteEvent(ts=T0, symbol="SPY", bid=99.98, ask=100.02, bid_size=10, ask_size=12)
    assert quote.mid == pytest.approx(100.0)

    trade = TradeEvent(ts=T0, symbol="SPY", price=100.01, size=200)
    assert trade.size == 200

    bar = BarEvent(
        ts=T0 + timedelta(minutes=1),
        symbol="SPY", open=100.0, high=100.5, low=99.9, close=100.3, volume=12_000,
    )
    assert bar.close == 100.3

    signal = SignalEvent(
        ts=T0, symbol="SPY", side=Side.BUY, conviction=0.7,
        horizon_minutes=60, signal_name="trend_v1",
    )
    assert signal.side is Side.BUY

    order = OrderEvent(
        ts=T0, order_id="o-1", symbol="SPY", side=Side.BUY, quantity=3,
        order_type="limit", limit_price=100.05, signal_name="trend_v1",
    )
    fill = FillEvent(
        ts=T0, order_id=order.order_id, symbol="SPY", side=Side.BUY,
        quantity=3, price=100.04, commission=0.65,
    )
    assert fill.order_id == order.order_id


def test_events_reject_naive_ts() -> None:
    with pytest.raises(ValidationError):
        TradeEvent(ts=datetime(2026, 1, 5, 9, 30), symbol="SPY", price=100.0, size=1)


def test_events_normalize_ts_to_new_york() -> None:
    utc_ts = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
    trade = TradeEvent(ts=utc_ts, symbol="SPY", price=100.0, size=1)
    assert trade.ts.tzinfo is MARKET_TZ
    assert trade.ts == utc_ts  # same instant, exchange-tz representation
    assert trade.ts.hour == 9 and trade.ts.minute == 30


def test_events_are_frozen() -> None:
    trade = TradeEvent(ts=T0, symbol="SPY", price=100.0, size=1)
    with pytest.raises(ValidationError):
        trade.price = 0.01  # type: ignore[misc]


def test_signal_conviction_bounded() -> None:
    with pytest.raises(ValidationError):
        SignalEvent(
            ts=T0, symbol="SPY", side=Side.SELL, conviction=1.5,
            horizon_minutes=60, signal_name="trend_v1",
        )


# ---------------------------------------------------------------------------
# Clocks
# ---------------------------------------------------------------------------


def test_backtest_clock_replays_in_order() -> None:
    events = [
        TradeEvent(ts=T0 + timedelta(minutes=i), symbol="SPY", price=100.0 + i, size=1)
        for i in range(5)
    ]
    clock = BacktestClock(start=T0)
    assert isinstance(clock, EventClock)

    seen: list[datetime] = []
    for event in clock.replay(events):
        assert clock.now() == event.ts  # clock tracks the event being consumed
        seen.append(event.ts)
    assert seen == sorted(seen)
    assert clock.now() == events[-1].ts


def test_backtest_clock_is_monotonic() -> None:
    clock = BacktestClock(start=T0)
    clock.advance(T0 + timedelta(minutes=10))
    with pytest.raises(ValueError):
        clock.advance(T0 + timedelta(minutes=5))


def test_backtest_clock_rejects_out_of_order_replay() -> None:
    events = [
        TradeEvent(ts=T0 + timedelta(minutes=2), symbol="SPY", price=100.0, size=1),
        TradeEvent(ts=T0 + timedelta(minutes=1), symbol="SPY", price=100.0, size=1),
    ]
    stream = BacktestClock(start=T0).replay(events)
    next(stream)
    with pytest.raises(ValueError):
        next(stream)


def test_live_clock_satisfies_protocol_and_is_tz_aware() -> None:
    clock: EventClock = LiveClock()
    first = clock.now()
    assert first.tzinfo is MARKET_TZ
    assert clock.now() >= first


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_loads() -> None:
    cfg = load_config()
    assert isinstance(cfg, EdgeConfig)
    assert cfg.data.lockbox_start == date(2026, 2, 22)
    assert cfg.validation.min_oos_trades == 300
    assert cfg.validation.pbo_max == 0.5
    assert cfg.validation.bootstrap_resamples == 10_000
    assert cfg.execution.spread_fill_fraction == 0.6
    assert cfg.execution.slippage_pct == 0.02
    assert cfg.execution.commission_per_contract == 0.65
    assert cfg.execution.latency_ms == [100, 500, 2000]
    assert cfg.risk.kelly_fraction == 0.25
    assert cfg.risk.per_trade_cap_pct == 2.0
    assert cfg.risk.daily_loss_halt_pct == 3.0
    assert cfg.risk.target_daily_vol_pct is None


def test_config_is_immutable() -> None:
    cfg = load_config()
    with pytest.raises(ValidationError):
        cfg.risk.kelly_fraction = 1.0  # type: ignore[misc]


def test_config_dotted_overrides_still_validated() -> None:
    cfg = load_config(overrides={"risk.kelly_fraction": 0.1})
    assert cfg.risk.kelly_fraction == 0.1
    with pytest.raises(ValidationError):
        load_config(overrides={"risk.kelly_fraction": 2.0})


# ---------------------------------------------------------------------------
# Runner stubs
# ---------------------------------------------------------------------------


def test_runner_stubs_import_and_gate() -> None:
    from edge.runners import backtest, live, paper

    assert backtest.main([]) == 0
    assert paper.main([]) == 1
    assert live.main([]) == 1
