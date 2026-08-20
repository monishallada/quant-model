"""Regressions for the 2026-08-20 adversarial audit findings.

Each test pins one confirmed result-corrupting bug. If any of these fail,
backtest results cannot be trusted — fix the code, never the test.
"""

from __future__ import annotations

from datetime import date, datetime, time

import pandas as pd
import pytest


class TestCachePoisoning:
    """CRITICAL: transient failures must never be cached as 'no data'."""

    def test_contract_day_does_not_cache_failures(self, tmp_path):
        from catalyst.core.config import load_config
        from catalyst.core.types import OptionKey, OptionRight
        from catalyst.data.cache import ParquetCache
        from catalyst.data.intraday import ThetaMinuteQuotes
        from catalyst.data.thetadata_client import ThetaDataError

        cfg = load_config("backtest")
        cache = ParquetCache(str(tmp_path))
        key = OptionKey(underlying="SPXW", expiry=date(2024, 6, 21),
                        right=OptionRight.PUT, strike=5400)

        class FailingClient:
            def get_dataframe(self, *a, **k):
                raise ThetaDataError("simulated outage")

        class HealthyClient:
            def get_dataframe(self, *a, **k):
                return pd.DataFrame({
                    "timestamp": [datetime(2024, 6, 21, 10, 0)],
                    "bid": [1.0], "ask": [1.1]})

        broken = ThetaMinuteQuotes(cfg.data.thetadata, cache, client=FailingClient())
        out1 = broken.contract_day(key, date(2024, 6, 21), time(9, 30), time(16, 0))
        assert out1.empty

        healthy = ThetaMinuteQuotes(cfg.data.thetadata, cache, client=HealthyClient())
        out2 = healthy.contract_day(key, date(2024, 6, 21), time(9, 30), time(16, 0))
        assert not out2.empty, (
            "the outage was cached as permanent 'no data' — cache poisoning")


class TestDeadSession:
    """CRITICAL: HTTP 478 must abort loudly, not degrade per-request."""

    def test_478_raises_dead_session(self):
        import httpx

        from catalyst.core.config import load_config
        from catalyst.data.thetadata_client import (
            ThetaDataClient, ThetaDataDeadSession)

        client = ThetaDataClient(load_config("backtest").data.thetadata)

        def dead(request):
            return httpx.Response(478, text="Invalid session ID")
        client._http = httpx.Client(  # noqa: SLF001 — test seam
            transport=httpx.MockTransport(dead),
            base_url="http://127.0.0.1:25503")
        with pytest.raises(ThetaDataDeadSession):
            client.get_dataframe("/v3/option/history/quote", {})


class TestVerdictNeverInSample:
    """HIGH: a missing test segment must never yield an 'out-of-sample' verdict."""

    def test_full_only_report_is_no_result(self):
        from catalyst.reporting.report import SegmentReport, StrategyReport

        r = StrategyReport(strategy="x", start=date(2020, 1, 1), end=date(2024, 1, 1))
        r.segments.append(SegmentReport(
            segment="full", cost_profile="real", avg_monthly_return=0.02,
            cagr=0.26, max_drawdown=-0.1, n_trades=500, win_rate=0.6,
            profit_factor=1.5, expected_value=50.0, concentration_share=0.1,
            pct_months_positive=0.7))
        assert "NO RESULT" in r.verdict
        assert "out-of-sample" not in r.verdict or "missing" in r.verdict

    def test_promotion_refuses_missing_test_verdict(self, tmp_path, monkeypatch):
        from catalyst.strategies import promotion
        monkeypatch.setattr(promotion, "LEDGER_ROOT", tmp_path)
        rec = promotion.record_backtest(
            "x", "CANDIDATE — +26.8%/yr out-of-sample, beats baseline "
                 "(test segment missing)", 0.02)
        assert not rec.validated


class TestPointInTimeHistory:
    """HIGH: strategy context history must end strictly before the session."""

    def test_pit_history_excludes_session_and_future(self):
        from catalyst.backtest.backtester import Backtester

        idx = pd.date_range("2024-01-01", periods=100, freq="B")
        hist = {"SPY": pd.DataFrame({"close": range(100)}, index=idx)}
        sliced = Backtester._pit_history(hist, idx[50].date())
        assert sliced["SPY"].index.max() < idx[50]
        assert len(sliced["SPY"]) == 50


class TestHalfDaySessions:
    """HIGH: engines must clamp to the real exchange close on early-close days."""

    def test_session_close_time_knows_half_days(self):
        from catalyst.core.tradingcal import session_close_time
        assert session_close_time(date(2024, 11, 29)) == time(13, 0)   # post-Thanksgiving
        assert session_close_time(date(2024, 12, 24)) == time(13, 0)   # Christmas Eve
        assert session_close_time(date(2024, 6, 21)) == time(16, 0)    # normal Friday


class TestDateBasedTime:
    """MEDIUM: headline()/calmar() must derive years from dates, not rows."""

    def test_sparse_curve_cagr_is_sane(self):
        from catalyst.backtest import metrics as m

        idx = pd.to_datetime([f"2024-{mm:02d}-15" for mm in range(1, 13)])
        eq = pd.Series([100_000 * (1.10 ** (i / 11)) for i in range(12)], index=idx)
        h = m.headline(eq)
        assert 0.08 < h["cagr"] < 0.14, f"CAGR {h['cagr']:.1%} — row-count bug back?"


class TestVerifierCatches:
    """Second-round catches: fixes that were themselves subtly wrong."""

    def test_intraday_records_reconcile_with_ledger_including_commissions(self):
        """Sum of TradeRecord.pnl must equal the actual equity change to the
        cent — the entry-side commission was silently $0 because it was
        computed from pos.qty AFTER the broker zeroed it on close."""
        import numpy as np

        from catalyst.backtest.intraday import IntradayBacktester
        from catalyst.brokers.simulated import SimulatedBroker
        from catalyst.core.config import load_config
        from catalyst.core.interfaces.intraday import IntradayStrategy
        from catalyst.core.types import (
            Direction, ExitRules, OptionKey, OptionRight, OrderLeg,
            ProposedTrade, Side)

        cfg = load_config("backtest")
        assert cfg.execution.commissions.per_contract > 0, \
            "backtest profile must charge real commissions (audit MEDIUM)"
        SESSION = date(2025, 6, 2)
        key = OptionKey(underlying="SPXW", expiry=SESSION,
                        right=OptionRight.PUT, strike=5900)
        idx = pd.date_range(datetime.combine(SESSION, time(9, 30)),
                            periods=390, freq="1min")
        bid = np.linspace(4.0, 3.0, 390)
        oq_frames = {(key, SESSION): pd.DataFrame(
            {"bid": bid, "ask": bid + 0.10}, index=idx)}
        bars = pd.DataFrame({"open": [600.0]*390, "high": [600.1]*390,
                             "low": [599.9]*390, "close": [600.0]*390,
                             "volume": [1000]*390}, index=idx)

        class OneTrade(IntradayStrategy):
            name = "one"
            def session_universe(self, session): return ["SPY"]
            def on_minute(self, ctx):
                if ctx.now.time() != time(10, 0):
                    return []
                return [ProposedTrade(
                    engine=self.name, catalyst_ref="x",
                    legs=[OrderLeg(key=key, side=Side.BUY, qty=1)],
                    unit_cost=4.0, unit_max_loss=4.0, direction=Direction.LONG,
                    exit_rules=ExitRules(max_hold_minutes=60),
                    per_trade_risk_fraction=0.01)]

        class B:
            def __init__(self, frames): self.frames = frames
            def get_day(self, s, d): return self.frames.get((s, d), pd.DataFrame())
        class OQ:
            def __init__(self, frames): self.frames = frames
            def contract_day(self, k, d, s_, e_): return self.frames.get((k, d), pd.DataFrame())

        start_cash = 100_000.0
        broker_box = {}
        def factory():
            b = SimulatedBroker(fill_model=cfg.execution.fill_model,
                                commissions=cfg.execution.commissions,
                                starting_cash=start_cash)
            broker_box["b"] = b
            return b
        bt = IntradayBacktester(cfg, B({("SPY", SESSION): bars}), OQ(oq_frames))
        res = bt.run([OneTrade()], SESSION, SESSION, factory)
        equity_change = broker_box["b"].get_account().equity - start_cash
        record_sum = sum(t.pnl for t in res.trades)
        assert record_sum == pytest.approx(equity_change, abs=0.01), (
            f"records {record_sum:.2f} vs ledger {equity_change:.2f} — "
            "commission reconciliation broken again")

    def test_settlement_uses_expiry_day_spot_even_after_next_session_chain(self):
        """The expiry-day spot must survive the next session's update_market —
        a single last-spot slot was overwritten before settlement ran."""
        from catalyst.brokers.simulated import SimulatedBroker
        from catalyst.core.config import load_config
        from catalyst.core.types import (
            Direction, ExitRules, OptionChain, OptionContract, OptionKey,
            OptionRight, Order, OrderIntent, OrderLeg, Side)

        cfg = load_config("backtest")
        b = SimulatedBroker(fill_model=cfg.execution.fill_model,
                            commissions=cfg.execution.commissions,
                            starting_cash=100_000.0)
        expiry = date(2025, 6, 20)
        key = OptionKey(underlying="SPY", expiry=expiry,
                        right=OptionRight.CALL, strike=600)
        def chain(spot, b_, a_):
            return OptionChain(underlying="SPY",
                               timestamp=datetime.combine(expiry, time(15, 45)),
                               underlying_price=spot,
                               contracts=[OptionContract(key=key, bid=b_, ask=a_)])
        # expiry day: spot 600 (option worthless OTM at 600 strike -> ~0)
        b.update_market({"SPY": chain(600.0, 0.40, 0.60)},
                        datetime.combine(expiry, time(15, 45)))
        b.place_order(Order(legs=[OrderLeg(key=key, side=Side.BUY, qty=1)],
                            limit_price=0.60, intent=OrderIntent.OPEN,
                            direction=Direction.LONG, exit_rules=ExitRules(),
                            tag="t:x"))
        # next session: spot GAPS to 612 — settlement must still use 600
        nxt = date(2025, 6, 23)
        b.update_market({"SPY": chain(612.0, 11.90, 12.10)},
                        datetime.combine(nxt, time(15, 45)))
        assert not b.get_positions(), "position should have expiry-settled"
        pid, value, pnl = b.settlements[-1]
        assert value == pytest.approx(0.0, abs=0.01), (
            f"settled at {value} — the overnight gap was fabricated into "
            "settlement again")
