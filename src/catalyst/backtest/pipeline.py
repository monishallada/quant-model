"""The one fixed path every strategy travels.

    screener -> strategy -> RiskManager -> cost/fill model -> exits
             -> metrics -> report

There is deliberately no argument, config key, or flag that removes a stage.
The out-of-sample split, the zero-cost diagnostic, the concentration check and
the N>100 flag are not options a caller may decline — they are the shape of
the function. A strategy author cannot skip them because the strategy never
drives the run; the pipeline does.

Why this exists: the audit found seven of nine campaigns carrying a bespoke
backtest loop, four bypassing the RiskManager entirely and four pricing their
own fills. Each loop was individually defensible and collectively made the
campaigns incomparable — which is the failure mode this module removes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from catalyst.backtest.engines import build_engines
from catalyst.core.interfaces.engine import BacktestEngine, EngineResult
from catalyst.backtest.walkforward import chronological_split_exclusive
from catalyst.core.config import Config
from catalyst.core.interfaces import DataSource, DirectionalSignal, Strategy
from catalyst.core.types import BacktestResult, Catalyst
from catalyst.reporting.comparison import EngineComparison, EngineView
from catalyst.reporting.report import StrategyReport, build_segment
from catalyst.risk.hedge import HedgeManager
from catalyst.strategies.promotion import record_backtest
from catalyst.risk.manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class PipelineRun:
    """Raw output of one (segment, cost-profile) execution."""

    segment: str
    cost_profile: str
    result: BacktestResult
    equity: pd.Series
    trades: list = field(default_factory=list)


class Pipeline:
    """Runs a strategy through every mandatory stage and returns the report."""

    def __init__(
        self,
        cfg: Config,
        data: DataSource,
        signal: DirectionalSignal,
        catalysts: list[Catalyst] | None = None,
        screener: object | None = None,
        hedge: bool = False,
        engines: list[BacktestEngine] | None = None,
    ) -> None:
        self._cfg = cfg
        self._data = data
        self._signal = signal
        self._catalysts = list(catalysts or [])
        self._screener = screener
        self._hedge = hedge
        # Every available engine runs. The native one is the reference;
        # the rest exist to disagree with it when it is wrong.
        self._engines = engines if engines is not None else build_engines()

    # -- the single execution primitive -----------------------------------
    def _zeroed(self, cfg):
        """Frictionless twin: identical routing, frictions zeroed. Built by
        copying the config rather than by branching inside the fill code, so
        the diagnostic exercises the SAME code path as the real run."""
        cfg = cfg.model_copy(deep=True)
        object.__setattr__(cfg.execution.fill_model, "spread_fill_fraction", 0.0)
        object.__setattr__(cfg.execution.fill_model, "slippage_pct_of_premium", 0.0)
        object.__setattr__(cfg.execution.fill_model, "slippage_per_contract", 0.0)
        object.__setattr__(cfg.execution.fill_model, "equity_slippage_bps", 0.0)
        object.__setattr__(cfg.execution.commissions, "alpaca_per_contract", 0.0)
        object.__setattr__(cfg.execution.commissions,
                           "schwab_per_contract_per_leg", 0.0)
        return cfg

    def _execute_all(
        self, strategy: Strategy, start: date, end: date, *, zero_cost: bool, label: str
    ) -> list[EngineResult]:
        """Run EVERY engine over the same window and cost profile.

        One engine failing must never abort the others — a missing runtime
        should degrade the comparison, not destroy the report.
        """
        cfg = self._zeroed(self._cfg) if zero_cost else self._cfg
        results: list[EngineResult] = []
        for engine in self._engines_for(strategy):
            ok, reason = engine.available()
            if not ok:
                results.append(EngineResult(engine=engine.name, error=reason))
                continue
            results.append(engine.run(
                strategy, start, end, cfg=cfg, data=self._data, signal=self._signal,
                catalysts=self._catalysts, screener=self._screener,
                zero_cost=zero_cost))
        return results

    def _engines_for(self, strategy) -> list:
        """Cadence picks the engine family; the honesty machinery is shared.

        An INTRADAY strategy on the daily engine would silently trade nothing
        (no opportunities at the daily cadence) and report a clean zero — the
        exact green-but-meaningless failure this project keeps refusing to
        ship. Route by cadence instead."""
        from catalyst.backtest.engines import IntradayNativeEngine
        from catalyst.core.interfaces.strategy import Cadence

        if getattr(strategy, "cadence", None) is Cadence.INTRADAY:
            return [IntradayNativeEngine()]
        return self._engines

    # -- the mandatory sequence -------------------------------------------
    def run(self, strategy: Strategy, start: date, end: date) -> StrategyReport:
        """Full + train + test, each at real and zero cost, on every engine."""
        train_end, test_start = chronological_split_exclusive(
            start, end, self._cfg.backtest.train_test_split)

        segments: list[tuple[str, date, date]] = [
            ("full", start, end),
            ("train", start, train_end),
            ("test", test_start, end),
        ]
        report = StrategyReport(strategy=strategy.name, start=start, end=end)
        for seg, s, e in segments:
            for zero in (False, True):
                profile = "zero" if zero else "real"
                results = self._execute_all(strategy, s, e, zero_cost=zero, label=seg)

                # The headline segment comes from the reference engine — the
                # first in the cadence-selected family (native for daily,
                # intraday_native for minute strategies). Matching the literal
                # name "native" silently dropped every intraday segment and
                # reported NO RESULT after a clean 8-hour run.
                native = next((r for r in results
                               if r.engine in ("native", "intraday_native") and r.ok),
                              None)
                if native is not None:
                    report.segments.append(
                        build_segment(seg, profile, native.equity, native.trades))

                report.comparisons.append(EngineComparison(
                    strategy=strategy.name, segment=seg, cost_profile=profile,
                    views=[EngineView.from_result(r) for r in results]))
                logger.info("%s %s/%s: %s", strategy.name, seg, profile,
                            {r.engine: (len(r.trades) if r.ok else r.error)
                             for r in results})

        report.extras["cadence"] = strategy.cadence.value
        report.extras["train_test_split"] = self._cfg.backtest.train_test_split
        report.extras["oos_boundary"] = f"train<= {train_end} | test>= {test_start}"
        report.extras["engines"] = ", ".join(
            f"{e.name}({'ok' if e.available()[0] else 'unavailable'})"
            for e in self._engines)
        return report

    def run_and_save(
        self, strategy: Strategy, start: date, end: date, root: Path
    ) -> StrategyReport:
        """Run, save, and record the verdict in the promotion ledger.

        `validated` is granted here or nowhere: it is written from the report
        this pipeline just produced, so it cannot be asserted by hand in a
        strategy's own source.
        """
        report = self.run(strategy, start, end)
        report.save(root)
        full = report.get("full", "real")
        record_backtest(
            strategy.name, report.verdict,
            full.avg_monthly_return if full else None,
            module_path=_module_path(strategy))
        return report


def _module_path(strategy: Strategy) -> Path | None:
    """Source file of the strategy, for the validation code fingerprint."""
    import inspect
    try:
        return Path(inspect.getfile(type(strategy)))
    except (TypeError, OSError):
        return None


def _equity_series(result: BacktestResult) -> pd.Series:
    curve = getattr(result, "equity_curve", None)
    if curve is None:
        return pd.Series(dtype=float)
    if isinstance(curve, pd.Series):
        s = curve
    else:
        s = pd.Series(curve)
    if not isinstance(s.index, pd.DatetimeIndex) and len(s):
        s.index = pd.to_datetime(s.index)
    return s.sort_index()
