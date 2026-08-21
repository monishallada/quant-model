"""KillSwitch behavior (audit D-062: engage/release/reason had zero coverage)
and the config-file unification (D-042/D-061)."""

from catalyst.core.config import load_config
from catalyst.observability.killswitch import KillSwitch


class TestKillSwitch:
    def test_engage_reason_release_cycle(self, tmp_path):
        k = KillSwitch(path=tmp_path / "KILL")
        assert not k.engaged()
        k.engage("drawdown breach")
        assert k.engaged()
        assert "drawdown breach" in k.reason()
        k.release()
        assert not k.engaged()

    def test_empty_reason_file_reports_placeholder(self, tmp_path):
        p = tmp_path / "KILL"
        p.write_text("")
        assert KillSwitch(path=p).reason() == "no reason recorded"

    def test_config_declares_the_watched_file(self):
        """deploy_runner constructs KillSwitch from THIS config value; the
        declared file and the watched file can no longer diverge (D-042)."""
        cfg = load_config("backtest")
        assert cfg.observability.kill_switch_file  # declared and non-empty
