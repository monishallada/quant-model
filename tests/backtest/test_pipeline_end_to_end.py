"""Pipeline + report machinery under test for the first time (audit D-002/
D-014/D-026/D-063/D-100/D-146): the mandatory six-segment matrix, the verdict
decision table, missing-segment refusal, artifact persistence, and the
zeroed-twin construction."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from catalyst.backtest.pipeline import Pipeline
from catalyst.core.config import load_config
from catalyst.core.interfaces.engine import BacktestEngine, EngineResult
from catalyst.core.interfaces.strategy import Cadence
from catalyst.core.types import Direction, TradeRecord
from catalyst.reporting.report import (
    SegmentReport,
    StrategyReport,
    build_segment,
)


def _trade(pnl, when=datetime(2024, 6, 3, 15, 45)):
    return TradeRecord(position_id="p", engine="fake", catalyst_ref="r",
                       underlying="SPY", direction=Direction.LONG,
                       entry_time=when, exit_time=when, entry_price=1.0,
                       exit_price=1.0, qty=1, pnl=pnl, exit_reason="t",
                       max_qty=1)


class CannedEngine(BacktestEngine):
    name = "native"
    verifies = "test double"

    def __init__(self, monthly=0.02, fail_segments=()):
        self._monthly = monthly
        self._fail = set(fail_segments)
        self.calls = []

    def available(self):
        return True, "ok"

    def run(self, strategy, start, end, *, cfg, data, signal,
            catalysts=None, screener=None, zero_cost=False):
        self.calls.append((start, end, zero_cost))
        if (start, end) in self._fail:
            return EngineResult(engine=self.name, error="canned failure")
        idx = pd.date_range(start, end, freq="D")
        # smooth compounding curve at the requested monthly rate
        growth = (1 + self._monthly) ** (pd.RangeIndex(len(idx)) / 30.4)
        equity = pd.Series(100_000.0 * growth, index=idx)
        trades = [_trade(100.0, datetime.combine(start, datetime.min.time()))
                  for _ in range(120)]
        return EngineResult(engine=self.name, equity=equity, trades=trades)


class DummyStrategy:
    name = "pipetest"
    cadence = Cadence.CATALYST


def _pipeline(engine):
    cfg = load_config("backtest")
    return Pipeline(cfg=cfg, data=None, signal=None, catalysts=[],
                    engines=[engine])


class TestMandatoryMatrix:
    def test_six_segments_produced(self):
        eng = CannedEngine()
        report = _pipeline(eng).run(DummyStrategy(), date(2020, 1, 2), date(2024, 1, 2))
        cells = {(s.segment, s.cost_profile) for s in report.segments}
        assert cells == {(a, b) for a in ("full", "train", "test")
                         for b in ("real", "zero")}
        # zero-cost twin actually ran with zero_cost=True half the time
        assert sum(1 for *_, z in eng.calls if z) == 3
        assert "segments_missing" not in report.extras

    def test_candidate_verdict_on_positive_oos(self):
        report = _pipeline(CannedEngine(monthly=0.02)).run(
            DummyStrategy(), date(2020, 1, 2), date(2024, 1, 2))
        assert report.verdict.startswith("CANDIDATE")

    def test_reject_verdict_on_negative_oos(self):
        report = _pipeline(CannedEngine(monthly=-0.01)).run(
            DummyStrategy(), date(2020, 1, 2), date(2024, 1, 2))
        assert report.verdict.startswith("REJECT")

    def test_failed_segment_refuses_verdict(self):
        """Audit D-026: an engine error must make the report SAY so, not
        silently omit the diagnostic."""
        eng = CannedEngine()
        report = _pipeline(eng).run(DummyStrategy(), date(2020, 1, 2), date(2024, 1, 2))
        # simulate: drop the zero-cost full segment as if the engine had failed
        report.segments = [s for s in report.segments
                           if not (s.segment == "full" and s.cost_profile == "zero")]
        report.extras["segments_missing"] = ["full/zero"]
        assert report.verdict.startswith("NO RESULT")

    def test_save_persists_ledgers_equity_and_json(self, tmp_path):
        report = _pipeline(CannedEngine()).run(
            DummyStrategy(), date(2020, 1, 2), date(2024, 1, 2))
        report.save(tmp_path)
        assert (tmp_path / "report.json").exists()
        assert (tmp_path / "report.txt").exists()
        assert (tmp_path / "trades_full_real.csv").exists()
        assert (tmp_path / "equity_full_real.csv").exists()
        import json
        parsed = json.loads((tmp_path / "report.json").read_text())  # strict
        assert parsed["strategy"] == "pipetest"


class TestVerdictDecisionTable:
    def _report(self, **seg_kw):
        base = dict(segment="test", cost_profile="real",
                    avg_monthly_return=0.01, cagr=0.12, max_drawdown=-0.05,
                    n_trades=150, win_rate=0.6, profit_factor=1.5,
                    expected_value=10.0, concentration_share=0.2,
                    pct_months_positive=0.6)
        base.update(seg_kw)
        r = StrategyReport(strategy="t", start=date(2020, 1, 1),
                           end=date(2024, 1, 1))
        r.segments.append(SegmentReport(**base))
        return r

    def test_no_trades(self):
        assert self._report(n_trades=0).verdict == "NO TRADES"

    def test_small_sample_inconclusive(self):
        assert self._report(n_trades=40).verdict.startswith("INCONCLUSIVE")

    def test_lottery_flag_rejects(self):
        v = self._report(concentration_share=0.8).verdict
        assert v.startswith("REJECT") and "lottery" in v

    def test_below_baseline_rejects(self):
        v = self._report(avg_monthly_return=0.0005).verdict
        assert v.startswith("REJECT") and "baseline" in v

    def test_missing_test_segment_is_no_result(self):
        r = StrategyReport(strategy="t", start=date(2020, 1, 1),
                           end=date(2024, 1, 1))
        assert r.verdict.startswith("NO RESULT")


class TestJsonSanitation:
    def test_infinite_profit_factor_serializes_strict(self):
        """Audit D-146: PF=inf emitted invalid JSON under allow_nan."""
        seg = build_segment("test", "real",
                            pd.Series([100.0, 110.0],
                                      index=pd.date_range("2024-01-01", periods=2)),
                            [_trade(50.0)])
        assert seg.profit_factor == float("inf")
        r = StrategyReport(strategy="t", start=date(2024, 1, 1),
                           end=date(2024, 2, 1))
        r.segments.append(seg)
        import json
        parsed = json.loads(r.to_json())
        assert parsed["segments"][0]["profit_factor"] == "inf"
