"""End-to-end coverage of the paper/live order path (audit D-011/12/13/58/59/142).

Before v15 this path had never been executed: submit() and close() built
Orders with five fields that do not exist on the model and crashed on first
use. These tests drive the REAL ExecutionEngine against the REAL
SimulatedBroker and RiskManager — no mocks in the money path.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

from catalyst.brokers.simulated import SimulatedBroker
from catalyst.core.config import CommissionsConfig, FillModelConfig, load_config
from catalyst.core.types import (
    Direction,
    ExitRules,
    OptionChain,
    OptionContract,
    OptionKey,
    OptionRight,
    OrderLeg,
    OrderResult,
    OrderStatus,
    ProposedTrade,
    Side,
)
from catalyst.execution.engine import ExecutionEngine
from catalyst.observability.killswitch import KillSwitch
from catalyst.risk.manager import RiskManager

NOW = datetime(2024, 6, 3, 15, 45)
_COMM = CommissionsConfig(alpaca_per_contract=0.0,
                          schwab_per_contract_per_leg=0.65,
                          active_profile="alpaca")
K1 = OptionKey(underlying="SPY", expiry=date(2024, 6, 21),
               right=OptionRight.CALL, strike=533.0)
K2 = OptionKey(underlying="SPY", expiry=date(2024, 6, 21),
               right=OptionRight.CALL, strike=540.0)


def _chain() -> OptionChain:
    return OptionChain(
        underlying="SPY", underlying_price=525.94, timestamp=NOW,
        contracts=[OptionContract(key=K1, bid=2.80, ask=3.00),
                   OptionContract(key=K2, bid=1.00, ask=1.10)])


def _broker() -> SimulatedBroker:
    b = SimulatedBroker(
        fill_model=FillModelConfig(spread_fill_fraction=0.6,
                                   slippage_pct_of_premium=0.02),
        commissions=_COMM, starting_cash=100_000.0)
    b.update_market({"SPY": _chain()}, NOW)
    return b


def _proposal() -> ProposedTrade:
    # debit call spread: buy K1, sell K2 — worst case pay 3.00 - 1.00 = 2.00
    return ProposedTrade(
        engine="testeng", catalyst_ref="ref-1",
        legs=[OrderLeg(key=K1, side=Side.BUY, qty=1),
              OrderLeg(key=K2, side=Side.SELL, qty=1)],
        unit_cost=2.00, unit_max_loss=2.00,
        direction=Direction.LONG,
        exit_rules=ExitRules(), per_trade_risk_fraction=0.02)


def _engine(broker=None, kill=None, dry_run=False) -> ExecutionEngine:
    cfg = load_config("backtest")
    return ExecutionEngine(broker=broker or _broker(),
                           risk=RiskManager(cfg.risk),
                           kill=kill or KillSwitch(path=Path("/tmp/nonexistent-kill")),
                           dry_run=dry_run)


class TestSubmit:
    def test_end_to_end_fill_with_sized_legs(self):
        eng = _engine()
        result = eng.submit(_proposal(), NOW, quote_time=NOW)
        assert result is not None and result.status is OrderStatus.FILLED
        assert result.filled_qty >= 1
        pos = eng.broker.get_positions()
        assert len(pos) == 1
        # broker re-bases to per-unit ratios; units carry the sizing
        assert pos[0].qty == result.filled_qty
        assert {leg.key for leg in pos[0].legs} == {K1, K2}

    def test_kill_switch_refuses_entry(self, tmp_path):
        kill = KillSwitch(path=tmp_path / "KILL")
        kill.engage("test halt")
        eng = _engine(kill=kill)
        assert eng.submit(_proposal(), NOW, quote_time=NOW) is None
        assert eng.broker.get_positions() == []

    def test_stale_quotes_refused(self):
        eng = _engine()
        old = NOW - timedelta(minutes=5)
        assert eng.submit(_proposal(), NOW, quote_time=old) is None
        assert eng.broker.get_positions() == []

    def test_dry_run_transmits_nothing(self):
        eng = _engine(dry_run=True)
        result = eng.submit(_proposal(), NOW, quote_time=NOW)
        assert result is not None and result.status is OrderStatus.ACCEPTED
        assert eng.broker.get_positions() == []

    def test_in_flight_order_blocks_duplicate(self):
        class AcceptingBroker(SimulatedBroker):
            def place_order(self, order):
                return OrderResult(order_id="pending-1",
                                   status=OrderStatus.ACCEPTED,
                                   message="resting")
        b = AcceptingBroker(
            fill_model=FillModelConfig(spread_fill_fraction=0.6,
                                       slippage_pct_of_premium=0.02),
            commissions=_COMM, starting_cash=100_000.0)
        b.update_market({"SPY": _chain()}, NOW)
        eng = _engine(broker=b)
        first = eng.submit(_proposal(), NOW, quote_time=NOW)
        assert first is not None and first.status is OrderStatus.ACCEPTED
        assert eng.submit(_proposal(), NOW, quote_time=NOW) is None  # D-142


class TestClose:
    def test_close_flattens_position(self):
        eng = _engine()
        eng.submit(_proposal(), NOW, quote_time=NOW)
        pos = eng.broker.get_positions()[0]
        result = eng.close(pos, NOW, "test-exit")
        assert result is not None and result.status is OrderStatus.FILLED
        assert eng.broker.get_positions() == []

    def test_close_unknown_position_refused_before_broker(self):
        eng = _engine()
        eng.submit(_proposal(), NOW, quote_time=NOW)
        pos = eng.broker.get_positions()[0]
        eng.close(pos, NOW, "first")
        # second close of the SAME (now gone) position: refused, not crashed
        assert eng.close(pos, NOW, "double-close") is None

    def test_close_ignores_kill_switch(self, tmp_path):
        eng = _engine()
        eng.submit(_proposal(), NOW, quote_time=NOW)
        pos = eng.broker.get_positions()[0]
        kill = KillSwitch(path=tmp_path / "KILL")
        kill.engage("halt entries")
        eng.kill = kill
        result = eng.close(pos, NOW, "exit-under-halt")
        assert result is not None and result.status is OrderStatus.FILLED


class TestExitRulesOnEquityPositions:
    def test_equity_position_does_not_crash_exit_evaluation(self):
        """Audit D-060: min(leg.key.expiry) raised AttributeError on shares."""
        from datetime import date as _date
        from catalyst.core.types import EquityKey, ExitRules, Position, PositionLeg
        from catalyst.exits.manager import evaluate_exits
        pos = Position(
            position_id="eq-1",
            legs=[PositionLeg(key=EquityKey(underlying="SPY"), side=Side.BUY, qty=1)],
            qty=100, entry_price=500.0, entry_time=NOW, engine_tag="t",
            direction=Direction.LONG,
            exit_rules=ExitRules(max_hold_trading_days=99),
            current_value=505.0)
        assert evaluate_exits(pos, _date(2024, 6, 5)) == []
