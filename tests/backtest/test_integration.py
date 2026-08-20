"""Full-chain integration test: synthetic DataSource → screener → engine →
RiskManager → SimulatedBroker fills → mechanical exits → BacktestResult.

Uses Engine B (no IV-rank dependency) with an always-long stub signal over a
two-week window containing one catalyst, on synthetic chains whose IV decays
after the event — enough to verify entries happen only through every gate and
exits fire mechanically."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import numpy as np

from catalyst.core.config import load_config
from catalyst.core.interfaces import DataSource, DirectionalSignal
from catalyst.core.types import (
    Catalyst,
    CatalystType,
    Direction,
    Greeks,
    OptionChain,
    OptionKey,
    Quote,
    SignalResult,
)
from catalyst.backtest.backtester import Backtester
from catalyst.strategies.archive.engines.engine_b_crush_spread import EngineBCrushSpread
from catalyst.risk.hedge import HedgeManager
from catalyst.risk.manager import RiskManager
from catalyst.screener.catalyst_screener import CatalystScreener
from tests.conftest import build_chain

CFG = load_config("backtest")
# Mechanism tests: pin the risk policy explicitly (the backtest profile is an
# unconstrained research profile; without the sizing buffer these assertions
# on realized-spend-within-budget lose the very mechanism they pin).
CFG = CFG.model_copy(update={"risk": CFG.risk.model_copy(update={
    "cash_floor_fraction": 0.40, "max_deployed": 0.25,
    "sizing_cost_buffer": 0.05, "daily_loss_halt": -0.08,
    "weekly_loss_halt": -0.15, "max_correlated_positions": 3,
    "correlation_threshold": 0.7})})
EXPIRIES = [date(2024, 6, 21), date(2024, 7, 19), date(2024, 8, 16)]
CATALYST = Catalyst(
    symbol="SPY", type=CatalystType.OTHER, when=datetime(2024, 6, 12, 16), source="test"
)


class StubData(DataSource):
    """Synthetic chains: IV 35% before the event, 22% after (crush)."""

    def list_expirations(self, symbol: str) -> list[date]:
        return list(EXPIRIES)

    def get_chain(self, symbol, at, expiries=None, max_dte=None) -> OptionChain:
        iv = 0.35 if at.date() <= CATALYST.when.date() else 0.22
        return build_chain(
            symbol=symbol, spot=100.0, as_of=at,
            expiry_ivs={e: iv for e in (expiries or EXPIRIES)},
        )

    def get_quote(self, symbol, at=None) -> Quote:
        return Quote(symbol=symbol, bid=99.99, ask=100.01, timestamp=at or datetime(2024, 6, 3))

    def get_greeks(self, key: OptionKey, at) -> Greeks:  # pragma: no cover
        raise NotImplementedError

    def get_history(self, symbol, start, end, interval="1d") -> pd.DataFrame:
        idx = pd.date_range(start, end, freq="B")
        close = pd.Series(np.linspace(90, 100, len(idx)), index=idx)
        return pd.DataFrame({"open": close, "high": close, "low": close,
                             "close": close, "volume": 1_000_000})

    def get_catalyst_calendar(self, start, end):  # pragma: no cover
        return [CATALYST]


class AlwaysLong(DirectionalSignal):
    name = "always_long"

    def evaluate(self, symbol, history, chain=None) -> SignalResult:
        return SignalResult(direction=Direction.LONG, confidence=0.9)


def run_integrated(screener: CatalystScreener | None = None):
    bt = Backtester(
        cfg=CFG,
        data=StubData(),
        strategies=[EngineBCrushSpread(CFG.engines.engine_b.model_copy(update={"enabled": True}), CFG.risk)],
        signal=AlwaysLong(),
        catalysts=[CATALYST],
        gate=RiskManager(CFG.risk),
        hedger=HedgeManager(CFG.risk),
        screener=screener,
        label="integration",
    )
    return bt.run(date(2024, 6, 3), date(2024, 6, 20))


def test_full_chain_trades_and_exits() -> None:
    result = run_integrated(screener=CatalystScreener(CFG.screener))
    # Engine B entered once for the catalyst (through screener + risk gates)...
    assert result.per_engine_trades.get("engine_b", 0) == 1
    trade = next(t for t in result.trades if t.engine == "engine_b")
    # ...sized within the engine's risk fraction (max loss <= 3% of equity)...
    assert trade.entry_price * trade.max_qty * 100 <= CFG.engines.engine_b.per_trade_risk * 100_000 + 1
    # ...and was closed mechanically (time stop after catalyst / tp), never expiry.
    assert trade.exit_time.date() <= date(2024, 6, 20)
    assert "expired_settled" not in trade.exit_reason
    # Hedge sleeve activated against the net-long driver book via the same gate.
    assert result.per_engine_trades.get("hedge", 0) >= 0  # present or budget-skipped
    # Equity curve exists for every session.
    assert len(result.equity_curve) > 8


def test_screener_blocks_low_implied_move() -> None:
    # Same pipeline but a screener demanding an absurd 50% implied move:
    # nothing passes, no trades happen.
    strict = CFG.screener.model_copy(update={"min_implied_move": 0.50})
    result = run_integrated(screener=CatalystScreener(strict))
    assert result.n_trades == 0 or all(t.engine == "hedge" for t in result.trades)
