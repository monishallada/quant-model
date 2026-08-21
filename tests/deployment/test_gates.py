"""Deployment gates. These stand between a backtest and real money.

The promotion flags are deliberately NOT readable from a strategy's own source:
these tests exercise the on-disk ledger, because that is what the live gate
actually consults.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from catalyst.runners.deploy_runner import (
    DeploymentRefused,
    Mode,
    Wiring,
    enforce_gates,
    enforce_preconditions,
)
from catalyst.strategies import promotion
from catalyst.strategies.promotion import (
    PromotionRecord,
    check_live_eligibility,
    record_backtest,
    record_paper_session,
)
from catalyst.strategies.registry import StrategyMeta

PASS = "CANDIDATE — +14.2%/yr out-of-sample, beats baseline"
FAIL = "REJECT — negative out-of-sample (-3.1%/yr)"


@pytest.fixture(autouse=True)
def ledger_in_tmp(tmp_path, monkeypatch):
    """Never touch the real results/ ledger from a test."""
    monkeypatch.setattr(promotion, "LEDGER_ROOT", tmp_path)
    return tmp_path


def _meta(name="demo") -> StrategyMeta:
    # a module that RESOLVES: since D-064 an unresolvable module is itself a
    # live refusal, so the happy-path fixture must point at real code
    return StrategyMeta(name=name, module="catalyst.strategies.archive.short_vrp.strategy")


class TestPromotionIsEarnedNotDeclared:
    def test_failing_backtest_does_not_validate(self):
        rec = record_backtest("demo", FAIL, -0.01)
        assert not rec.validated

    def test_passing_backtest_validates(self):
        rec = record_backtest("demo", PASS, 0.011)
        assert rec.validated and rec.validated_verdict == PASS

    def test_paper_requires_prior_validation(self):
        """Running paper on an unvalidated strategy must not grant anything."""
        record_backtest("demo", FAIL, -0.01)
        with pytest.raises(PermissionError):
            record_paper_session("demo", "PA-TEST", orders_seen=2, round_trips=1)
        assert not PromotionRecord.load("demo").paper_tested

    def test_full_path_backtest_then_paper_grants_live(self):
        record_backtest("demo", PASS, 0.011)
        record_paper_session("demo", "PA-TEST", orders_seen=3, round_trips=1)
        ok, reason = check_live_eligibility("demo")
        assert ok, reason

    def test_a_failing_rerun_withdraws_validation_and_paper(self):
        """A strategy cannot be validated once and then edited freely."""
        record_backtest("demo", PASS, 0.011)
        record_paper_session("demo", "PA-TEST", orders_seen=2, round_trips=1)
        record_backtest("demo", FAIL, -0.02)
        rec = PromotionRecord.load("demo")
        assert not rec.validated and not rec.paper_tested

    def test_editing_the_strategy_after_validation_revokes_eligibility(self, tmp_path):
        """Validation attaches to a specific body of code, not to a name."""
        module = tmp_path / "strat.py"
        module.write_text("# v1\n")
        record_backtest("demo", PASS, 0.011, module_path=module)
        record_paper_session("demo", "PA-TEST", orders_seen=2, round_trips=1)
        assert check_live_eligibility("demo", module)[0]

        module.write_text("# v2 — quietly different\n")
        ok, reason = check_live_eligibility("demo", module)
        assert not ok and "changed since validation" in reason


class TestLiveRequiresBacktestAndPaper:
    def test_unvalidated_strategy_is_refused(self):
        record_backtest("demo", FAIL, -0.01)
        with pytest.raises(DeploymentRefused, match="not validated"):
            enforce_preconditions(_meta(), Mode.LIVE)

    def test_validated_but_never_papered_is_refused(self):
        record_backtest("demo", PASS, 0.011)
        with pytest.raises(DeploymentRefused, match="never paper-tested"):
            enforce_preconditions(_meta(), Mode.LIVE)

    def test_both_passed_clears_preconditions(self):
        record_backtest("demo", PASS, 0.011)
        record_paper_session("demo", "PA-TEST", orders_seen=2, round_trips=1)
        enforce_preconditions(_meta(), Mode.LIVE)

    @pytest.mark.parametrize("mode", [Mode.BACKTEST, Mode.PAPER])
    def test_non_live_modes_skip_the_validation_requirement(self, mode):
        """Paper is how a strategy EARNS paper_tested; requiring it there would
        be a deadlock."""
        enforce_preconditions(_meta(), mode)

    def test_a_strategy_that_never_ran_is_refused(self):
        """No ledger entry at all must fail closed, not open."""
        with pytest.raises(DeploymentRefused, match="not validated"):
            enforce_preconditions(_meta("never_run"), Mode.LIVE)


class TestConfirmationCannotBeAutomatedAway:
    def test_yes_flag_is_refused_in_live_mode(self):
        wiring = Wiring(Mode.LIVE, data=None, broker=_FakeBroker(is_paper=False),
                        describe="fake live")
        with pytest.raises(DeploymentRefused, match="not honoured in live"):
            enforce_gates(_meta(), wiring, _cfg(), assume_yes=True)

    def test_live_mode_refuses_a_paper_broker(self):
        """A mode/broker mismatch means the wiring is wrong; refuse rather than
        quietly trade somewhere other than where the operator was told."""
        wiring = Wiring(Mode.LIVE, data=None, broker=_FakeBroker(is_paper=True),
                        describe="paper broker in live mode")
        with pytest.raises(DeploymentRefused, match="resolved to a paper broker"):
            enforce_gates(_meta(), wiring, _cfg(), assume_yes=False)

    def test_backtest_needs_no_confirmation(self):
        wiring = Wiring(Mode.BACKTEST, data=None, broker=_FakeBroker(), describe="sim")
        enforce_gates(_meta(), wiring, _cfg(), assume_yes=False)


class _FakeBroker:
    def __init__(self, is_paper: bool = True) -> None:
        self.is_paper = is_paper

    def preflight(self) -> dict:
        return {"endpoint": "fake://", "account_number": "TEST", "equity": 100_000.0}


def _cfg():
    from catalyst.core.config import load_config
    return load_config("backtest")
