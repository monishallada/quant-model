"""Multi-engine cross-check.

The point of a second engine is to DISAGREE when the first is wrong. These
tests plant the exact bug classes this project has actually shipped and assert
the comparison catches them.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from catalyst.backtest.engines import build_engines
from catalyst.core.interfaces.engine import EngineResult
from catalyst.reporting.comparison import EngineComparison, EngineView




class TestNautilusIsAGenuinelyIndependentEngine:
    """The Nautilus engine must run its OWN backtest, not replay native fills.

    An earlier version replayed the native engine's fill stream. That could only
    catch arithmetic faults in the ledger — it shared entries, exits and fill
    prices, so it could never disagree about whether a fill was achievable or
    what it cost. These tests pin the independence structurally.
    """

    def test_it_does_not_import_or_wrap_the_native_engine(self):
        import inspect

        from catalyst.backtest.engines import nautilus
        src = inspect.getsource(nautilus)
        assert "NativeEngine" not in src, (
            "nautilus engine references NativeEngine — it is replaying native "
            "fills rather than running its own backtest")

    def test_it_declares_independence_in_what_it_verifies(self):
        from catalyst.backtest.engines import NautilusEngine
        assert "independent" in NautilusEngine().verifies.lower()

    def test_it_builds_its_own_venue_fill_and_fee_models(self):
        """Own matching engine, own fill model, own fee model, own account."""
        import inspect

        from catalyst.backtest.engines import nautilus
        src = inspect.getsource(nautilus)
        for required in ("add_venue", "fill_model", "fee_model", "add_instrument",
                         "add_data", "add_strategy"):
            assert required in src, f"nautilus engine never calls {required}"

    def test_money_strings_are_parsed(self):
        """Nautilus reports Money as '123.45 USD'; float() on that returns 0.0
        and silently reads as a flat strategy."""
        from catalyst.backtest.engines.nautilus import _money
        assert _money("123.45 USD") == pytest.approx(123.45)
        assert _money("-3,773.17 USD") == pytest.approx(-3773.17)
        assert _money(None) == 0.0
        assert _money(42.0) == pytest.approx(42.0)


class TestComparisonSurfacesDivergence:
    def _view(self, engine, monthly, **kw):
        return EngineView(engine=engine, available=True, avg_monthly_return=monthly,
                          cagr=monthly * 12, max_drawdown=-0.1,
                          final_equity=100_000.0, n_trades=120, **kw)

    def test_agreement_reported_as_agreement(self):
        c = EngineComparison("s", "full", "real",
                             [self._view("native", 0.0100), self._view("nautilus", 0.0100)])
        assert c.agree and not c.divergences
        assert "ENGINES AGREE" in c.to_text()

    def test_material_gap_is_flagged(self):
        c = EngineComparison("s", "full", "real",
                             [self._view("native", 0.0400), self._view("nautilus", 0.0007)])
        assert not c.agree
        assert "ENGINES DIVERGE" in c.to_text()

    def test_per_trade_divergence_flags_even_when_totals_match(self):
        """Aggregates can cancel; per-trade mismatches cannot be netted away."""
        c = EngineComparison("s", "full", "real", [
            self._view("native", 0.01),
            self._view("nautilus", 0.01,
                       diagnostics={"per_trade_divergences": 4, "divergence_sample": []}),
        ])
        assert not c.agree

    def test_single_engine_is_not_agreement(self):
        """One engine running alone must never read as corroborated."""
        c = EngineComparison("s", "full", "real", [self._view("native", 0.01)])
        assert not c.agree
        assert "no cross-check performed" in c.to_text()

    def test_unavailable_engine_is_shown_not_hidden(self):
        c = EngineComparison("s", "full", "real", [
            self._view("native", 0.01),
            EngineView(engine="lean", available=False, error="docker daemon not running"),
        ])
        text = c.to_text()
        assert "UNAVAILABLE" in text and "docker" in text


class TestEngineRegistry:
    def test_default_flow_is_native_only(self):
        """The default path is the validated native pipeline alone."""
        from catalyst.backtest.engines import DEFAULT_ENGINES
        assert DEFAULT_ENGINES == ("native",)
        assert [e.name for e in build_engines()] == ["native"]

    def test_second_engines_remain_opt_in_and_available(self):
        """Nothing was deleted — the cross-check engines still build and run."""
        from catalyst.backtest.engines import ALL_ENGINES
        avail = {e.name: e.available()[0] for e in build_engines(ALL_ENGINES)}
        assert avail["native"] and avail["nautilus"]

    def test_unavailable_engine_reports_reason_and_never_raises(self):
        from catalyst.backtest.engines import ALL_ENGINES
        for e in build_engines(ALL_ENGINES):
            ok, why = e.available()
            assert isinstance(ok, bool) and isinstance(why, str) and why

    def test_every_engine_states_what_it_verifies(self):
        """Stops a reader over-reading agreement between engines that check
        different things."""
        from catalyst.backtest.engines import ALL_ENGINES
        for e in build_engines(ALL_ENGINES):
            assert e.verifies, f"{e.name} does not state what it verifies"
