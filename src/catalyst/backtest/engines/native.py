"""The native engine — this repository's own event-driven backtester.

It is the reference implementation: the RiskManager, the cost model, the exit
manager and the OOS methodology all live on this path, and every archived
campaign was measured with it. A second engine is a check ON this one, not a
replacement for it.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from catalyst.core.interfaces.engine import BacktestEngine, EngineResult
from catalyst.risk.hedge import HedgeManager
from catalyst.risk.manager import RiskManager

logger = logging.getLogger(__name__)


class NativeEngine(BacktestEngine):
    name = "native"
    verifies = ("full path: screener, RiskManager sizing, NBBO cost model, "
                "exit manager, cash ledger")

    def __init__(self, hedge: bool = False) -> None:
        self._hedge = hedge

    def available(self) -> tuple[bool, str]:
        return True, "in-repo"

    def run(self, strategy, start: date, end: date, *, cfg, data, signal,
            catalysts=None, screener=None, zero_cost: bool = False) -> EngineResult:
        from catalyst.backtest.backtester import Backtester

        try:
            bt = Backtester(
                cfg=cfg, data=data, strategies=[strategy], signal=signal,
                catalysts=list(catalysts or []),
                gate=RiskManager(cfg.risk),
                hedger=HedgeManager(cfg.risk) if self._hedge else None,
                screener=screener,
                label=f"{strategy.name}:native")
            result = bt.run(start, end)
        except Exception as e:                          # noqa: BLE001
            logger.exception("native engine failed")
            return EngineResult(engine=self.name, error=f"{type(e).__name__}: {e}")

        return EngineResult(
            engine=self.name,
            equity=_series(getattr(result, "equity_curve", {})),
            trades=list(result.trades),
            diagnostics={"starting_capital": cfg.account.starting_capital})


def _series(curve) -> pd.Series:
    if curve is None:
        return pd.Series(dtype=float)
    s = curve if isinstance(curve, pd.Series) else pd.Series(curve)
    if len(s) and not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    return s.sort_index()
