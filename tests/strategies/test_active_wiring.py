"""Every active strategy must actually run through the pipeline.

This exists because the framework's 477 guarantee-tests all passed while the
active strategy could not execute at all: the signal was wired to the wrong
config section and ExitRules was given a field that does not exist. Those are
config-wiring faults, invisible to tests that check the framework's invariants.

Cheap insurance, run on synthetic data, no network.
"""

from __future__ import annotations

import inspect
from datetime import date

import pytest

from catalyst.core.config import load_config
from catalyst.core.interfaces import Cadence, Opportunity, Strategy, StrategyContext
from catalyst.core.types import ProposedTrade
from catalyst.strategies.registry import load_strategy, registry

ACTIVE = [n for n, m in registry().items() if m.status == "active"]


@pytest.fixture(scope="module")
def cfg():
    return load_config("backtest")


@pytest.mark.parametrize("name", ACTIVE)
class TestActiveStrategyIsWiredCorrectly:
    def test_builds_from_config(self, name, cfg):
        s = load_strategy(name, cfg)
        assert isinstance(s, Strategy)
        assert s.name == name
        assert isinstance(s.cadence, Cadence)

    def test_opportunities_needs_no_chain(self, name, cfg):
        """Enumeration must not require market data — the pipeline calls it to
        decide what data to buy."""
        s = load_strategy(name, cfg)
        ctx = StrategyContext(as_of=_dt(), data=_ExplodingData())
        opps = s.opportunities(date(2024, 3, 15), ctx)
        assert isinstance(opps, list)
        for o in opps:
            assert isinstance(o, Opportunity) and o.symbols

    def test_plan_returns_intent_or_none_never_raises(self, name, cfg):
        """A missing chain is normal (holiday, thin name). It must return None,
        not blow up the session."""
        s = load_strategy(name, cfg)
        ctx = StrategyContext(as_of=_dt(), data=_NoDataAvailable())
        opp = Opportunity(session=date(2024, 3, 15), symbols=("SPY",))
        out = s.plan(opp, ctx)
        assert out is None or isinstance(out, ProposedTrade)

    def test_exit_rules_and_proposal_fields_are_real(self, name, cfg):
        """Catches `ExitRules(time_stop_days=...)`-class typos.

        Uses the AST rather than a regex: `ExitRules\\(([^)]*)\\)` stops at the
        first close-paren, which lands inside a nested call like
        `timedelta(days=2)` and reports 'days' as a bogus ExitRules field.
        """
        import ast

        from catalyst.core.types import ExitRules
        valid = set(ExitRules.model_fields)
        src = inspect.getsource(inspect.getmodule(type(load_strategy(name, cfg))))
        for node in ast.walk(ast.parse(src)):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "ExitRules"):
                continue
            for kw in node.keywords:
                assert kw.arg in valid, (
                    f"{name}: ExitRules has no field '{kw.arg}'; "
                    f"valid: {sorted(valid)}")


def _dt():
    from datetime import datetime, time
    return datetime.combine(date(2024, 3, 15), time(15, 45))


class _ExplodingData:
    """Chain access fails loudly — proves enumeration is data-free."""

    def get_chain(self, *a, **k):
        raise AssertionError("opportunities() must not fetch chains")

    def list_expirations(self, symbol):
        return []

    def get_catalyst_calendar(self, start, end):
        return []


class _NoDataAvailable:
    """Coverage gap: the realistic failure a strategy must survive."""

    def get_chain(self, *a, **k):
        from catalyst.data.thetadata_historical import DataUnavailableError
        raise DataUnavailableError("no chain for this session")

    def list_expirations(self, symbol):
        return []

    def get_catalyst_calendar(self, start, end):
        return []
