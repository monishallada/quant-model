"""IntradayBacktester behavioral battery — synthetic data, zero network.

Every test here pins an honesty property: the leak convention, cost-bearing
fills, the mandatory flatten, minute-resolution exits, flagged synthetic
fills, and the risk gate. These are the properties that make an intraday
result a measurement rather than a story.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

from catalyst.backtest.intraday import IntradayBacktester, IntradayParams
from catalyst.brokers.simulated import SimulatedBroker
from catalyst.core.config import load_config
from catalyst.core.interfaces.intraday import IntradayContext, IntradayStrategy
from catalyst.core.types import (
    Direction, EquityKey, ExitRules, OptionKey, OptionRight, OrderLeg,
    ProposedTrade, Side,
)

SESSION = date(2025, 6, 2)   # a Monday
CFG = load_config("backtest").model_copy(update={
    "risk": load_config("backtest").risk.model_copy(update={
        "cash_floor_fraction": 0.05, "max_deployed": 0.95})})


def _bars(session, open_px=100.0, drift_per_min=0.01, n=390):
    idx = pd.date_range(datetime.combine(session, time(9, 30)), periods=n, freq="1min")
    opens = open_px + drift_per_min * np.arange(n)
    return pd.DataFrame({"open": opens, "high": opens + 0.02, "low": opens - 0.02,
                         "close": opens + drift_per_min, "volume": 10_000}, index=idx)


class FakeBars:
    def __init__(self, frames): self.frames = frames
    def get_day(self, symbol, day): return self.frames.get((symbol, day), pd.DataFrame())


class FakeOptionQuotes:
    def __init__(self, frames=None): self.frames = frames or {}
    def contract_day(self, key, day, start, end):
        return self.frames.get((key, day), pd.DataFrame())


class BuyAt10(IntradayStrategy):
    """Buys 100-share-risk SPY at 10:00, holds to the flatten."""
    name = "buy_at_10"

    def __init__(self, stop=None, max_hold=None, close_by=None):
        self.rules = ExitRules(use_stops=stop is not None, stop_loss_pct=stop,
                               max_hold_minutes=max_hold, close_by_time=close_by)
        self.seen_bar_labels: list[pd.Timestamp] = []

    def session_universe(self, session): return ["SPY"]

    def on_minute(self, ctx: IntradayContext):
        df = ctx.bars.get("SPY")
        if df is not None and len(df):
            self.seen_bar_labels.append(df.index[-1])
        if ctx.now.time() != time(10, 0):
            return []
        spot = ctx.spot("SPY")
        return [ProposedTrade(
            engine=self.name, catalyst_ref=f"SPY:{ctx.session}",
            legs=[OrderLeg(key=EquityKey(underlying="SPY"), side=Side.BUY, qty=1)],
            unit_cost=spot, unit_max_loss=spot * 0.02,   # 2% defined stop distance
            direction=Direction.LONG, exit_rules=self.rules,
            per_trade_risk_fraction=0.01)]


def _run(strat, frames=None, oq=None, params=None):
    bt = IntradayBacktester(CFG, FakeBars(frames or {("SPY", SESSION): _bars(SESSION)}),
                            FakeOptionQuotes(oq), params=params)
    return bt.run([strat], SESSION, SESSION, lambda: SimulatedBroker(
        fill_model=CFG.execution.fill_model, commissions=CFG.execution.commissions,
        starting_cash=100_000.0))


class TestLeakConvention:
    def test_decisions_never_see_the_current_bar(self):
        strat = BuyAt10()
        _run(strat)
        # at decision time ts the newest visible label must be <= ts - 1min:
        # bar T covers [T, T+1m) and only exists as a fact at T+1m.
        for i, label in enumerate(strat.seen_bar_labels):
            ts = datetime.combine(SESSION, time(9, 31)) + timedelta(minutes=i)
            assert label <= pd.Timestamp(ts) - pd.Timedelta(minutes=1), \
                f"decision at {ts} saw bar labeled {label} — lookahead"

    def test_fill_is_next_bar_open_not_decision_close(self):
        strat = BuyAt10()
        res = _run(strat)
        t = res.trades[0]
        # decision at 10:00 uses bars <= 09:59; fill must be the 10:00 bar's
        # OPEN (± costs), which is strictly above the 09:59 close in this ramp.
        bar_1000_open = _bars(SESSION).loc[datetime.combine(SESSION, time(10, 0)), "open"]
        assert t.entry_price >= bar_1000_open, "filled better than the next tradeable print"


class TestSessionMechanics:
    def test_flattens_at_eod_and_ends_flat(self):
        res = _run(BuyAt10())
        assert len(res.trades) == 1
        assert res.trades[0].exit_reason == "eod_flatten"
        assert res.trades[0].exit_time.time() >= time(15, 58)
        assert res.diagnostics["forced_flattens"] == 1

    def test_ramp_day_long_is_profitable_after_costs(self):
        res = _run(BuyAt10())
        t = res.trades[0]
        assert t.pnl > 0, "steady +1c/min ramp held 6h must beat 2bps of costs"
        assert res.equity.iloc[-1] > 100_000.0

    def test_stop_loss_fires_at_minute_resolution(self):
        frames = {("SPY", SESSION): _bars(SESSION, drift_per_min=-0.05)}  # falling day
        res = _run(BuyAt10(stop=-0.02), frames)
        t = res.trades[0]
        assert t.exit_reason == "stop_loss"
        assert t.exit_time.time() < time(11, 0), "a -3%/hr slide must stop out fast"
        assert t.pnl < 0

    def test_max_hold_minutes_and_close_by_time(self):
        res = _run(BuyAt10(max_hold=30))
        assert res.trades[0].exit_reason == "max_hold_minutes"
        assert res.trades[0].exit_time - res.trades[0].entry_time <= timedelta(minutes=31)
        res2 = _run(BuyAt10(close_by=time(12, 0)))
        assert res2.trades[0].exit_reason == "close_by_time"
        assert res2.trades[0].exit_time.time() <= time(12, 1)

    def test_one_position_per_strategy_symbol(self):
        class SpamEntries(BuyAt10):
            name = "spam"
            def on_minute(self, ctx):
                spot = ctx.spot("SPY")
                if spot is None or ctx.now.time() < time(10, 0):
                    return []
                return [ProposedTrade(
                    engine=self.name, catalyst_ref="SPY:x",
                    legs=[OrderLeg(key=EquityKey(underlying="SPY"), side=Side.BUY, qty=1)],
                    unit_cost=spot, unit_max_loss=spot * 0.02,
                    direction=Direction.LONG, exit_rules=self.rules,
                    per_trade_risk_fraction=0.01)]
        res = _run(SpamEntries())
        assert len(res.trades) == 1, "dedup must hold against per-minute spam"


class TestOptionsPath:
    KEY = OptionKey(underlying="SPXW", expiry=SESSION, right=OptionRight.PUT, strike=5900)

    def _oq_frames(self, vanish_after=None):
        idx = pd.date_range(datetime.combine(SESSION, time(9, 30)), periods=390, freq="1min")
        bid = np.linspace(4.00, 2.00, 390); ask = bid + 0.10
        df = pd.DataFrame({"bid": bid, "ask": ask}, index=idx)
        if vanish_after is not None:
            df = df[df.index <= datetime.combine(SESSION, vanish_after)]
        return {(self.KEY, SESSION): df}

    def _strat(self):
        key = self.KEY
        class ShortPut(IntradayStrategy):
            name = "short_put_0dte"
            def session_universe(self, session): return ["SPY"]
            def on_minute(self, ctx):
                if ctx.now.time() != time(10, 0):
                    return []
                return [ProposedTrade(
                    engine=self.name, catalyst_ref="SPXW:x",
                    legs=[OrderLeg(key=key, side=Side.BUY, qty=1)],  # long put for test
                    unit_cost=3.5, unit_max_loss=3.5,
                    direction=Direction.SHORT, exit_rules=ExitRules(max_hold_minutes=60),
                    per_trade_risk_fraction=0.01)]
        return ShortPut()

    def test_option_entry_at_ask_exit_at_bid_via_minute_nbbo(self):
        res = _run(self._strat(), oq=self._oq_frames())
        t = res.trades[0]
        assert t.exit_reason == "max_hold_minutes"
        # decaying premium, bought at ask, sold at bid an hour later -> loss
        assert t.pnl < 0
        assert res.diagnostics["synthetic_fills"] == 0

    def test_vanished_exit_quote_is_flagged_synthetic_never_silent(self):
        res = _run(self._strat(), oq=self._oq_frames(vanish_after=time(10, 30)))
        assert res.diagnostics["synthetic_fills"] == 1, \
            "missing exit quote must be a FLAGGED haircut settlement"
        assert len(res.trades) == 1 and res.trades[0].exit_reason == "eod_flatten"


class TestRiskGate:
    def test_risk_manager_blocks_oversized_entries(self):
        class HugeBet(BuyAt10):
            name = "huge"
            def on_minute(self, ctx):
                props = super().on_minute(ctx)
                return [p.model_copy(update={"per_trade_risk_fraction": 0.000001})
                        for p in props]   # $0.10 budget < one $2.10 risk unit
        res = _run(HugeBet())
        assert res.diagnostics["risk_blocked"] >= 1
        assert len(res.trades) == 0


class TestCreditStop:
    """A short spread's stop must fire when liability reaches Nx the credit —
    the positive-entry ratio stop is structurally blind to credit positions."""

    KEY_S = OptionKey(underlying="SPXW", expiry=SESSION, right=OptionRight.PUT, strike=5900)
    KEY_W = OptionKey(underlying="SPXW", expiry=SESSION, right=OptionRight.PUT, strike=5875)

    def _frames(self):
        idx = pd.date_range(datetime.combine(SESSION, time(9, 30)), periods=390, freq="1min")
        # short leg premium EXPLODES mid-day (adverse move); wing follows less
        s_bid = np.where(np.arange(390) < 120, 0.80, 4.00)
        w_bid = np.where(np.arange(390) < 120, 0.30, 1.20)
        return {
            (self.KEY_S, SESSION): pd.DataFrame({"bid": s_bid, "ask": s_bid + 0.10}, index=idx),
            (self.KEY_W, SESSION): pd.DataFrame({"bid": w_bid, "ask": w_bid + 0.10}, index=idx),
        }

    def test_liability_multiple_stop_fires(self):
        ks, kw = self.KEY_S, self.KEY_W
        class ShortSpread(IntradayStrategy):
            name = "credit_test"
            def session_universe(self, session): return ["SPY"]
            def on_minute(self, ctx):
                if ctx.now.time() != time(10, 0):
                    return []
                return [ProposedTrade(
                    engine=self.name, catalyst_ref="SPXW:x",
                    legs=[OrderLeg(key=ks, side=Side.SELL, qty=1),
                          OrderLeg(key=kw, side=Side.BUY, qty=1)],
                    unit_cost=-0.45, unit_max_loss=25 - 0.45,
                    direction=Direction.NEUTRAL,
                    exit_rules=ExitRules(use_stops=True, stop_loss_pct=-3.0,
                                         close_by_time=time(15, 55)),
                    per_trade_risk_fraction=0.03)]
        res = _run(ShortSpread(), oq=self._frames())
        assert len(res.trades) == 1
        t = res.trades[0]
        assert t.exit_reason == "credit_stop", f"got {t.exit_reason}"
        assert t.exit_time.time() <= time(11, 35), "stop must fire when liability jumps"
        assert t.pnl < 0
