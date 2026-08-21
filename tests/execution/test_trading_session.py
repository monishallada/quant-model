"""End-to-end TradingSession test (audit D-016/D-066/D-067): the paper loop
must actually trade — entry on a catalyst, exit by rule, round trip counted —
through the REAL ExecutionEngine, RiskManager and SimulatedBroker."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from catalyst.brokers.simulated import SimulatedBroker
from catalyst.core.config import CommissionsConfig, FillModelConfig, load_config
from catalyst.core.interfaces.strategy import Cadence
from catalyst.core.types import (
    Catalyst,
    CatalystType,
    Direction,
    ExitRules,
    OptionChain,
    OptionContract,
    OptionKey,
    OptionRight,
    OrderLeg,
    ProposedTrade,
    Side,
    SignalResult,
)
from catalyst.execution.engine import ExecutionEngine
from catalyst.execution.session import TradingSession
from catalyst.observability.killswitch import KillSwitch
from catalyst.risk.manager import RiskManager

DAY1 = datetime(2024, 6, 3, 15, 45)
EVENT = date(2024, 6, 4)
EXPIRY = date(2024, 6, 21)
K1 = OptionKey(underlying="SPY", expiry=EXPIRY, right=OptionRight.CALL, strike=533.0)
K2 = OptionKey(underlying="SPY", expiry=EXPIRY, right=OptionRight.CALL, strike=540.0)


def _chain(when):
    return OptionChain(underlying="SPY", underlying_price=525.94, timestamp=when,
                       contracts=[OptionContract(key=K1, bid=2.80, ask=3.00),
                                  OptionContract(key=K2, bid=1.00, ask=1.10)])


class FakeData:
    def list_expirations(self, symbol):
        return [EXPIRY]

    def get_chain(self, symbol, at, expiries=None, max_dte=None):
        return _chain(at)

    def get_history(self, symbol, start, end):
        idx = pd.bdate_range(end=pd.Timestamp(DAY1.date()), periods=60)
        return pd.DataFrame({"close": 525.0}, index=idx)


class OneShotStrategy:
    """Buys a call spread on the session before the event; exits by hard date."""
    name = "session_test"
    cadence = Cadence.CATALYST

    def __init__(self):
        self.evaluated = 0

    def catalyst_expiries(self, catalyst, available, as_of):
        return [EXPIRY]

    def evaluate(self, catalyst, chain, signal, as_of):
        self.evaluated += 1
        if as_of.date() != date(2024, 6, 3):
            return None
        return ProposedTrade(
            engine=self.name, catalyst_ref=catalyst.ref,
            legs=[OrderLeg(key=K1, side=Side.BUY, qty=1),
                  OrderLeg(key=K2, side=Side.SELL, qty=1)],
            unit_cost=2.00, unit_max_loss=2.00, direction=Direction.LONG,
            exit_rules=ExitRules(hard_exit_date=date(2024, 6, 4),
                                 close_before_expiry_days=1),
            per_trade_risk_fraction=0.02)


class NeutralSignal:
    name = "neutral"
    def evaluate(self, symbol, history, chain=None):
        return SignalResult(direction=Direction.NEUTRAL, confidence=0.5)


def test_session_completes_a_round_trip(tmp_path):
    broker = SimulatedBroker(
        fill_model=FillModelConfig(spread_fill_fraction=0.6,
                                   slippage_pct_of_premium=0.02),
        commissions=CommissionsConfig(alpaca_per_contract=0.0,
                                      schwab_per_contract_per_leg=0.65,
                                      active_profile="alpaca"),
        starting_cash=100_000.0)
    cfg = load_config("backtest")
    engine = ExecutionEngine(broker=broker, risk=RiskManager(cfg.risk),
                             kill=KillSwitch(path=tmp_path / "KILL"))
    clock = {"i": 0}
    days = [DAY1, DAY1 + timedelta(days=1), DAY1 + timedelta(days=2)]

    def now_fn():
        at = days[min(clock["i"], len(days) - 1)]
        clock["i"] += 1
        broker.update_market({"SPY": _chain(at)}, at)
        return at

    strategy = OneShotStrategy()
    session = TradingSession(
        strategy=strategy, signal=NeutralSignal(), execution=engine,
        data=FakeData(),
        catalysts=[Catalyst(symbol="SPY", type=CatalystType.EARNINGS,
                            when=datetime(2024, 6, 4, 8, 0), source="test")],
        interval_seconds=0, max_cycles=3, now_fn=now_fn)
    stats = session.run()

    assert stats.cycles == 3
    assert strategy.evaluated >= 1, "the strategy must actually be invoked (D-067)"
    assert stats.entries_filled == 1
    assert stats.exits_filled == 1, "hard exit on day 2 must close the position"
    assert stats.round_trips == 1
    assert broker.get_positions() == [], "book flat at session end"
    assert not stats.errors


def test_paper_tested_requires_the_round_trip(tmp_path, monkeypatch):
    """The promotion grant is bound to session evidence (D-016)."""
    from catalyst.strategies import promotion
    monkeypatch.setattr(promotion, "LEDGER_ROOT", tmp_path)
    promotion.record_backtest("session_test", "CANDIDATE — test", 0.01)
    with pytest.raises(PermissionError, match="round trip"):
        promotion.record_paper_session("session_test", "PA-1",
                                       orders_seen=0, round_trips=0)
    rec = promotion.record_paper_session("session_test", "PA-1",
                                         orders_seen=2, round_trips=1)
    assert rec.paper_tested
