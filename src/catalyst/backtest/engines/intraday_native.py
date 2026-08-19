"""IntradayNativeEngine — the wrapper Kinetic never had.

Implements ``BacktestEngine`` over the IntradayBacktester so minute-resolution
strategies inherit the entire honesty apparatus for free: the pipeline's
mandatory full/train/test x real/zero-cost matrix, the exclusive OOS split, the
N>100 and concentration checks, the standard report, and the promotion ledger.
Kinetic sat outside that machinery; its successor cannot skip a stage even if
it wanted to.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from catalyst.backtest.intraday import IntradayBacktester, IntradayParams
from catalyst.core.interfaces.engine import BacktestEngine, EngineResult
from catalyst.core.interfaces.strategy import Cadence

logger = logging.getLogger(__name__)


class IntradayNativeEngine(BacktestEngine):
    name = "intraday_native"
    verifies = ("minute-resolution event loop: RiskManager-gated entries, "
                "cost-truth fills (equities bps + options NBBO), flagged "
                "synthetic fills, mandatory EOD flatten")

    def __init__(self, params: IntradayParams | None = None) -> None:
        self._params = params

    def available(self) -> tuple[bool, str]:
        return True, "in-repo"

    def run(self, strategy, start: date, end: date, *, cfg, data, signal,
            catalysts=None, screener=None, zero_cost: bool = False) -> EngineResult:
        if getattr(strategy, "cadence", None) is not Cadence.INTRADAY:
            return EngineResult(engine=self.name,
                                error=f"'{getattr(strategy, 'name', '?')}' is not an "
                                      "INTRADAY-cadence strategy")
        try:
            from catalyst.brokers.simulated import SimulatedBroker
            from catalyst.data.cache import ParquetCache
            from catalyst.data.intraday import AlpacaMinuteBars, ThetaMinuteQuotes

            cache = ParquetCache(cfg.data.cache_dir)
            bars = AlpacaMinuteBars(cfg.data.alpaca, cache, feed="sip")
            oq = ThetaMinuteQuotes(cfg.data.thetadata, cache)

            def broker_factory():
                # cfg arrives pre-zeroed from the pipeline's zero-cost twin, so
                # one factory serves both cost profiles through the same path.
                return SimulatedBroker(fill_model=cfg.execution.fill_model,
                                       commissions=cfg.execution.commissions,
                                       starting_cash=cfg.account.starting_capital)

            bt = IntradayBacktester(cfg, bars, oq, data=data, params=self._params)
            result = bt.run([strategy], start, end, broker_factory)
        except Exception as e:                          # noqa: BLE001
            logger.exception("intraday engine failed")
            return EngineResult(engine=self.name, error=f"{type(e).__name__}: {e}")

        return EngineResult(engine=self.name, equity=result.equity,
                            trades=result.trades, diagnostics=result.diagnostics)
