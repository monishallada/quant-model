"""Kinetic Ignition Engine: State 1-4 logic on hand-built minute data."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pandas as pd
import pytest

from catalyst.core.config import CommissionsConfig, FillModelConfig, load_config
from catalyst.core.models import (
    OptionChain,
    OptionContract,
    OptionKey,
    OptionRight,
    Order,
    OrderLeg,
    Side,
)
from catalyst.brokers.simulated import SimulatedBroker
from catalyst.data.intraday import five_minute_candles
from catalyst.kinetic.engine import KineticIgnitionEngine
from catalyst.kinetic.exits import EmaTracker, KineticExitState
from catalyst.kinetic.screener import KineticScreener

CFG = load_config("backtest")
SESSION = date(2024, 2, 2)  # a Friday -> 0DTE weekly available
PRIOR_CLOSE = 100.0


class StubBars:
    """Minute bars we control exactly, so State 1/2 math is verifiable."""

    def __init__(self, pm_last: float, pm_volume: int, open_px: float, close_0934: float,
                 rth_path: list[float] | None = None) -> None:
        self.pm_last, self.pm_volume = pm_last, pm_volume
        self.open_px, self.close_0934 = open_px, close_0934
        self.rth_path = rth_path or []

    def get_day(self, symbol: str, day: date) -> pd.DataFrame:
        if day != SESSION:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        rows, idx = [], []
        # Pre-market 04:00-09:29 (30 bars), volume split evenly, last close = pm_last.
        for i in range(30):
            ts = datetime.combine(day, time(9, 0)) + timedelta(minutes=i)
            px = self.pm_last
            idx.append(ts)
            rows.append({"open": px, "high": px, "low": px, "close": px,
                         "volume": self.pm_volume // 30})
        # 09:30-09:34 opening candle.
        for i in range(5):
            ts = datetime.combine(day, time(9, 30)) + timedelta(minutes=i)
            px = self.open_px if i == 0 else self.close_0934
            idx.append(ts)
            rows.append({"open": self.open_px if i == 0 else px, "high": max(px, self.open_px),
                         "low": min(px, self.open_px), "close": px, "volume": 10_000})
        # Post-signal path (one bar per minute from 09:35).
        for i, px in enumerate(self.rth_path):
            ts = datetime.combine(day, time(9, 35)) + timedelta(minutes=i)
            idx.append(ts)
            rows.append({"open": px, "high": px, "low": px, "close": px, "volume": 5_000})
        return pd.DataFrame(rows, index=pd.to_datetime(idx))

    def prior_close(self, symbol: str, day: date, lookback_days: int = 7) -> float:
        return PRIOR_CLOSE


def screener(pm_last=110.0, pm_volume=2_000_000, open_px=110.0, close_0934=111.0):
    bars = StubBars(pm_last, pm_volume, open_px, close_0934)
    return KineticScreener(CFG.kinetic, bars)


# ---------------------------------------------------------------------------
# State 1 + 2
# ---------------------------------------------------------------------------


def test_gap_and_premarket_volume_computed_per_spec() -> None:
    sig = screener(pm_last=110.0, pm_volume=2_000_000).evaluate("TEST", SESSION)
    assert sig.gap == pytest.approx((110.0 - 100.0) / 100.0)  # +10%
    assert sig.premarket_volume == 1_999_980  # 30 bars x floor(2M/30)
    assert sig.fires and sig.direction == "long"


def test_gap_below_threshold_rejected() -> None:
    sig = screener(pm_last=105.0, open_px=105.0, close_0934=106.0).evaluate("TEST", SESSION)
    assert sig.gap == pytest.approx(0.05)
    assert not sig.fires and sig.reject_reason == "gap_too_small"


def test_premarket_volume_below_threshold_rejected() -> None:
    sig = screener(pm_volume=300_000).evaluate("TEST", SESSION)
    assert not sig.fires and sig.reject_reason == "premarket_volume_too_low"


def test_momentum_contradicting_gap_aborts() -> None:
    # Gapped up +10% but the first 5-minute candle is red -> ABORT.
    sig = screener(pm_last=110.0, open_px=110.0, close_0934=109.0).evaluate("TEST", SESSION)
    assert sig.momentum < 0
    assert not sig.fires and sig.reject_reason == "momentum_contradicts_gap"


def test_gap_down_with_red_candle_fires_short() -> None:
    sig = screener(pm_last=90.0, open_px=90.0, close_0934=89.0).evaluate("TEST", SESSION)
    assert sig.gap == pytest.approx(-0.10)
    assert sig.fires and sig.direction == "short"


def test_signal_uses_only_pre_0935_data() -> None:
    """Lookahead guard: a violent post-09:35 reversal cannot change the signal."""
    bars = StubBars(110.0, 2_000_000, 110.0, 111.0, rth_path=[80.0] * 60)
    sig = KineticScreener(CFG.kinetic, bars).evaluate("TEST", SESSION)
    assert sig.price_open == 110.0 and sig.price_0935 == 111.0
    assert sig.fires and sig.direction == "long"


# ---------------------------------------------------------------------------
# State 3 — contract selection
# ---------------------------------------------------------------------------


def build_chain(spot: float, expiries: list[date]) -> OptionChain:
    contracts = []
    for expiry in expiries:
        for strike in [spot - 5, spot - 2, spot, spot + 2, spot + 5]:
            for right in (OptionRight.CALL, OptionRight.PUT):
                contracts.append(OptionContract(
                    key=OptionKey(underlying="TEST", expiry=expiry, right=right,
                                  strike=float(round(strike))),
                    bid=1.90, ask=2.10))
    return OptionChain(underlying="TEST", underlying_price=spot,
                       timestamp=datetime.combine(SESSION, time(9, 35)), contracts=contracts)


def make_engine():
    return KineticIgnitionEngine(CFG.kinetic, CFG.risk, screener())


def test_selects_nearest_expiry_within_max_dte() -> None:
    engine = make_engine()
    available = [SESSION, SESSION + timedelta(days=7), SESSION + timedelta(days=14)]
    assert engine.select_expiry(available, SESSION) == SESSION  # 0 DTE preferred


def test_skips_when_no_expiry_within_max_dte() -> None:
    engine = make_engine()
    # Monday gap day: nearest Friday is 4 calendar days out -> outside 0-3 DTE.
    monday = date(2024, 1, 29)
    assert engine.select_expiry([date(2024, 2, 2)], monday) is None


def test_selects_atm_strike_and_right() -> None:
    engine = make_engine()
    chain = build_chain(111.0, [SESSION])
    call = engine.select_contract(chain, SESSION, OptionRight.CALL, 111.0)
    put = engine.select_contract(chain, SESSION, OptionRight.PUT, 111.0)
    assert call.key.strike == 111.0 and call.key.right is OptionRight.CALL
    assert put.key.strike == 111.0 and put.key.right is OptionRight.PUT


def test_proposal_requests_nav_fraction_and_prices_at_ask() -> None:
    from catalyst.core.models import Catalyst, CatalystType, Direction

    engine = make_engine()
    chain = build_chain(111.0, [SESSION])
    catalyst = Catalyst(symbol="TEST", type=CatalystType.EARNINGS,
                        when=datetime.combine(SESSION, time(9, 30)), source="t")
    trade = engine.evaluate(catalyst, chain, None, datetime.combine(SESSION, time(9, 35)))
    assert trade is not None
    assert trade.direction is Direction.LONG
    assert trade.per_trade_risk_fraction == CFG.kinetic.execution.nav_fraction
    assert trade.unit_cost == 2.10 == trade.unit_max_loss  # the ASK, per spec
    assert trade.legs[0].side is Side.BUY


# ---------------------------------------------------------------------------
# State 4 — exits
# ---------------------------------------------------------------------------


def make_exit_state(direction="long", entry=2.10):
    return KineticExitState(direction=direction, entry_price=entry,
                            hard_stop_loss=CFG.kinetic.exits.hard_stop_loss,
                            time_stop=time(15, 50),
                            ema=EmaTracker(period=CFG.kinetic.exits.ema_period))


def test_ema_formula_matches_spec() -> None:
    ema = EmaTracker(period=9)
    assert ema.update(100.0) == 100.0  # seed
    # k = 2/(9+1) = 0.2 -> 0.2*110 + 0.8*100 = 102
    assert ema.update(110.0) == pytest.approx(102.0)


def test_seed_candle_cannot_trigger_exit() -> None:
    state = make_exit_state()
    state.on_candle_close(111.0)  # seed only
    assert not state.pending_trailing_exit
    assert not state.evaluate_minute(time(9, 40), 2.10).should_exit


def test_long_exits_when_candle_closes_below_ema() -> None:
    state = make_exit_state()
    state.on_candle_close(111.0)  # seed
    state.on_candle_close(110.0)  # below seed -> below EMA
    decision = state.evaluate_minute(time(9, 45), 2.10)
    assert decision.should_exit and decision.reason == "trailing_ema_stop"


def test_long_holds_while_candles_close_above_ema() -> None:
    state = make_exit_state()
    for close in (111.0, 112.0, 113.0, 114.0):
        state.on_candle_close(close)
    assert not state.evaluate_minute(time(10, 0), 2.50).should_exit


def test_short_exits_when_candle_closes_above_ema() -> None:
    state = make_exit_state(direction="short")
    state.on_candle_close(89.0)
    state.on_candle_close(90.0)
    assert state.evaluate_minute(time(9, 45), 2.10).reason == "trailing_ema_stop"


def test_hard_stop_fires_at_fifty_percent_loss() -> None:
    state = make_exit_state(entry=2.00)
    state.on_candle_close(111.0)
    assert not state.evaluate_minute(time(10, 0), 1.05).should_exit  # -47.5%
    assert state.evaluate_minute(time(10, 0), 1.00).reason == "hard_stop"  # -50%


def test_hard_stop_takes_precedence_over_trailing() -> None:
    state = make_exit_state(entry=2.00)
    state.on_candle_close(111.0)
    state.on_candle_close(110.0)  # arms trailing exit
    assert state.evaluate_minute(time(10, 0), 0.50).reason == "hard_stop"


def test_time_stop_fires_at_configured_time() -> None:
    state = make_exit_state()
    state.on_candle_close(111.0)
    assert not state.evaluate_minute(time(15, 49), 2.20).should_exit
    assert state.evaluate_minute(time(15, 50), 2.20).reason == "time_stop"


# ---------------------------------------------------------------------------
# Fill enforcement: entries at ask, exits at bid, flat per-contract slippage
# ---------------------------------------------------------------------------


def test_entry_at_ask_exit_at_bid_with_flat_slippage() -> None:
    key = OptionKey(underlying="TEST", expiry=SESSION, right=OptionRight.CALL, strike=111.0)
    chain = OptionChain(underlying="TEST", underlying_price=111.0,
                        timestamp=datetime.combine(SESSION, time(9, 35)),
                        contracts=[OptionContract(key=key, bid=1.90, ask=2.10)])
    broker = SimulatedBroker(
        fill_model=FillModelConfig(spread_fill_fraction=1.0, slippage_pct_of_premium=0.0,
                                   slippage_per_contract=0.02),
        commissions=CommissionsConfig(alpaca_per_contract=0.0,
                                      schwab_per_contract_per_leg=0.65,
                                      active_profile="alpaca"),
        starting_cash=100_000.0)
    broker.update_market({"TEST": chain}, datetime.combine(SESSION, time(9, 35)))
    entry = broker.place_order(Order(legs=[OrderLeg(key=key, side=Side.BUY, qty=10)],
                                     limit_price=2.10, tag="kinetic:TEST"))
    assert entry.avg_fill_price == pytest.approx(2.12)  # ask 2.10 + $0.02 slippage

    from catalyst.core.models import OrderIntent

    exit_result = broker.place_order(Order(
        legs=[OrderLeg(key=key, side=Side.SELL, qty=10)], limit_price=1.90,
        intent=OrderIntent.CLOSE, position_id=f"pos-{entry.order_id}"))
    assert exit_result.avg_fill_price == pytest.approx(1.88)  # bid 1.90 - $0.02


def test_default_fill_model_unchanged_by_new_field() -> None:
    """The archived campaigns must behave identically: default flat slippage is 0."""
    assert FillModelConfig(spread_fill_fraction=0.6,
                           slippage_pct_of_premium=0.02).slippage_per_contract == 0.0


# ---------------------------------------------------------------------------
# Candle aggregation
# ---------------------------------------------------------------------------


def test_five_minute_candles_anchored_at_open() -> None:
    idx = [datetime.combine(SESSION, time(9, 30)) + timedelta(minutes=i) for i in range(15)]
    bars = pd.DataFrame({"open": range(100, 115), "high": range(100, 115),
                         "low": range(100, 115), "close": range(100, 115),
                         "volume": [1] * 15}, index=pd.to_datetime(idx))
    candles = five_minute_candles(bars, time(9, 30), 5)
    assert len(candles) == 3
    assert candles.index[0].time() == time(9, 30)
    assert candles.index[1].time() == time(9, 35)
    assert candles["close"].iloc[0] == 104  # 09:30-09:34 inclusive


# ---------------------------------------------------------------------------
# Data-layer regression: minute quotes must not be NaN-aligned away
# ---------------------------------------------------------------------------


def test_minute_quotes_preserve_values_not_nan(tmp_path) -> None:
    """Regression: building the frame from Series against a DatetimeIndex makes
    pandas align on labels and silently produce all-NaN quotes, which would
    route every fill through the synthetic haircut."""
    from catalyst.data.cache import ParquetCache
    from catalyst.data.intraday import ThetaMinuteQuotes

    cache = ParquetCache(tmp_path)
    key = OptionKey(underlying="DIS", expiry=date(2024, 2, 9),
                    right=OptionRight.CALL, strike=107.0)
    raw = pd.DataFrame({
        "timestamp": ["2024-02-08T09:35:00.000", "2024-02-08T09:36:00.000"],
        "bid": [1.60, 2.25], "ask": [1.75, 2.39],
    })
    cache.put("theta_minute_quote", "DIS_20240209_107.000_call_20240208_0935_1555", raw)
    quotes = ThetaMinuteQuotes(CFG.data.thetadata, cache, client=object())
    out = quotes.contract_day(key, date(2024, 2, 8), time(9, 35), time(15, 55))
    assert len(out) == 2
    assert out["bid"].notna().all() and out["ask"].notna().all()
    assert out["bid"].iloc[0] == pytest.approx(1.60)
    assert out["ask"].iloc[1] == pytest.approx(2.39)
