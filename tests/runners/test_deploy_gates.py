"""deploy_runner gate coverage (audit D-015/D-150/D-152/D-154): the argument
gates that guard mode/config binding and research overrides."""

import pytest

from catalyst.runners.deploy_runner import DeploymentRefused, Mode, build_wiring, main
from catalyst.core.config import load_config


class TestModeConfigBinding:
    def test_paper_mode_with_backtest_config_refused(self):
        """Audit D-015: paper silently ran under backtest risk limits."""
        with pytest.raises(DeploymentRefused, match="requires --config 'paper'"):
            main(["--mode", "paper", "--strategy", "short_vrp",
                  "--config", "backtest"])

    def test_set_refused_outside_backtest(self):
        with pytest.raises(DeploymentRefused, match="research knob"):
            main(["--mode", "paper", "--strategy", "short_vrp",
                  "--set", "risk.cash_floor_fraction=0.0"])

    def test_malformed_set_refused(self):
        with pytest.raises(DeploymentRefused, match="malformed"):
            main(["--mode", "backtest", "--strategy", "short_vrp",
                  "--set", "not_a_kv"])


class TestBuildWiring:
    def test_backtest_wiring_is_simulated(self):
        cfg = load_config("backtest")
        w = build_wiring(Mode.BACKTEST, cfg)
        from catalyst.brokers.simulated import SimulatedBroker
        assert isinstance(w.broker, SimulatedBroker)
        assert "Simulated" in w.describe

    def test_live_describe_admits_historical_data(self):
        """Audit D-151: the strings claimed 'live data' while wiring the
        historical source."""
        import inspect
        from catalyst.runners import deploy_runner
        src = inspect.getsource(deploy_runner.build_wiring)
        assert "live data +" not in src
