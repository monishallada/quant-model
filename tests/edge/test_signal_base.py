"""Signal ABC + registry: registration/toggling, the hypothesis contract
(MissingHypothesis at registration AND at run_signal), the pre-run
record_trial ordering (the trial line exists even when the backtest raises),
duplicate-name rejection, and extra='forbid' configs. All data is synthetic;
explicit timestamps predate the 2026-02-22 lockbox start."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

import edge.signals.registry as sigreg
from edge.core.events import BarEvent, Side, SignalEvent
from edge.research.registry import TrialRegistry, canonical_config_hash
from edge.signals.base import (
    ALL_REGIMES,
    MIN_HYPOTHESIS_CHARS,
    MissingHypothesis,
    Signal,
    SignalConfig,
    SignalContext,
    regime_allowed,
    require_hypothesis,
)
from edge.signals.registry import (
    DuplicateSignal,
    UnknownSignal,
    register,
    run_signal,
    trial_config_for,
)

ET = ZoneInfo("America/New_York")
# Synthetic session well BEFORE the 2026-02-22 lockbox start.
T0 = datetime(2026, 1, 12, 10, 0, tzinfo=ET)

HYPOTHESIS = (
    "Opening-gap continuation: market-on-open retail flow chases overnight "
    "gaps while index-arb desks hedge mechanically into the same direction; "
    "the flow is price-inelastic, so it keeps paying momentum for the first "
    "thirty minutes of the session."
)
assert len(HYPOTHESIS.strip()) >= MIN_HYPOTHESIS_CHARS


class GapConfig(SignalConfig):
    """Per-signal config used throughout: two swept parameters."""

    lookback_bars: int = 5
    min_gap_pct: float = 0.5


def _make_cls(
    name: str = "gap_go",
    hypothesis: str = HYPOTHESIS,
    allowed: object = ALL_REGIMES,
    warmup: int = 3,
) -> type[Signal]:
    """A fresh concrete Signal class (unregistered) with the given declarations."""
    n, h, a, w = name, hypothesis, allowed, warmup

    class GapGo(Signal):
        name = n
        hypothesis = h
        allowed_regimes = a  # type: ignore[assignment]
        warmup_bars = w
        config_type = GapConfig

        def on_bar(self, ctx: SignalContext) -> list[SignalEvent]:
            bar = ctx.bar
            return [
                SignalEvent(
                    ts=bar.ts,
                    symbol=bar.symbol,
                    side=Side.BUY,
                    conviction=1.0,
                    horizon_minutes=30,
                    signal_name=self.name,
                )
            ]

    return GapGo


class FakeCtx:
    """Minimal engine ctx satisfying the SignalContext protocol."""

    def __init__(self, bar: BarEvent) -> None:
        self._bar = bar

    @property
    def decision_ts(self) -> datetime:
        return self._bar.ts

    @property
    def bar(self) -> BarEvent:
        return self._bar

    def history(self, symbol: str, n_bars: int):  # pragma: no cover - unused
        raise NotImplementedError

    def frame(self, kind: str, symbol: str):  # pragma: no cover - unused
        raise NotImplementedError


def _bar(minute: int = 0) -> BarEvent:
    ts = T0 + timedelta(minutes=minute)
    return BarEvent(ts=ts, symbol="AAA", open=100.0, high=101.0, low=99.5, close=100.5, volume=1000)


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Each test sees an empty signal catalog and leaves no trace."""
    saved = sigreg._snapshot()
    sigreg.clear_registry()
    yield
    sigreg._restore(saved)


# ---------------------------------------------------------------------------
# registration + toggling
# ---------------------------------------------------------------------------


def test_register_get_and_default_enabled() -> None:
    cls = register(_make_cls())
    assert sigreg.get("gap_go") is cls
    assert sigreg.list_signals() == ["gap_go"]
    assert sigreg.is_enabled("gap_go") is True


def test_decorator_syntax_returns_class_unchanged() -> None:
    @register
    class Momo(Signal):
        name = "momo"
        hypothesis = HYPOTHESIS
        config_type = GapConfig

        def on_bar(self, ctx: SignalContext) -> list[SignalEvent]:
            return []

    assert sigreg.get("momo") is Momo
    # Defaults from the ABC survive registration.
    assert Momo.allowed_regimes == ALL_REGIMES
    assert Momo.warmup_bars == 0


def test_disable_enable_toggling() -> None:
    register(_make_cls("a"))
    register(_make_cls("b"))
    sigreg.disable("a")
    assert sigreg.is_enabled("a") is False
    assert sigreg.list_signals() == ["a", "b"]
    assert sigreg.list_signals(enabled_only=True) == ["b"]
    sigreg.enable("a")
    assert sigreg.is_enabled("a") is True
    assert sigreg.list_signals(enabled_only=True) == ["a", "b"]


def test_unknown_names_raise() -> None:
    with pytest.raises(UnknownSignal):
        sigreg.get("ghost")
    with pytest.raises(UnknownSignal):
        sigreg.enable("ghost")
    with pytest.raises(UnknownSignal):
        sigreg.disable("ghost")
    with pytest.raises(UnknownSignal):
        sigreg.is_enabled("ghost")


def test_duplicate_name_rejected() -> None:
    register(_make_cls("dup"))
    with pytest.raises(DuplicateSignal):
        register(_make_cls("dup"))
    # Re-registering the very same class is also a duplicate: silent
    # re-admission would invite name collisions across reloads.
    cls = sigreg.get("dup")
    with pytest.raises(DuplicateSignal):
        register(cls)
    # The original entry is untouched.
    assert sigreg.get("dup") is cls


def test_register_rejects_abstract_and_non_signal() -> None:
    class NoOnBar(Signal):
        name = "no_on_bar"
        hypothesis = HYPOTHESIS

    with pytest.raises(TypeError):
        register(NoOnBar)  # abstract: on_bar not implemented
    with pytest.raises(TypeError):
        register(object)  # type: ignore[arg-type]
    assert sigreg.list_signals() == []


def test_register_validates_declarations() -> None:
    with pytest.raises(ValueError, match="name"):
        register(_make_cls(name=""))
    with pytest.raises(ValueError, match="warmup_bars"):
        register(_make_cls(warmup=-1))
    with pytest.raises(ValueError, match="allowed_regimes"):
        register(_make_cls(allowed=set()))  # nowhere to trade is a mistake
    with pytest.raises(ValueError, match="allowed_regimes"):
        register(_make_cls(allowed=["high_vol"]))  # a list is not a set


# ---------------------------------------------------------------------------
# the hypothesis contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   \n\t ", "Buy dips.", "x" * (MIN_HYPOTHESIS_CHARS - 1)])
def test_registration_refuses_missing_or_short_hypothesis(bad: str) -> None:
    with pytest.raises(MissingHypothesis):
        register(_make_cls(hypothesis=bad))
    assert sigreg.list_signals() == []


def test_whitespace_padding_does_not_rescue_a_short_hypothesis() -> None:
    padded = "Momentum works." + " " * 200  # long raw, short stripped
    with pytest.raises(MissingHypothesis):
        register(_make_cls(hypothesis=padded))


def test_hypothesis_boundary_exactly_min_chars_passes() -> None:
    exact = "m" * MIN_HYPOTHESIS_CHARS
    assert require_hypothesis(exact) == exact
    register(_make_cls(name="exact", hypothesis=exact))
    assert sigreg.list_signals() == ["exact"]


def test_run_signal_refuses_short_hypothesis_and_records_nothing(tmp_path: Path) -> None:
    # Unregistered class: run_signal enforces the contract independently.
    sig = _make_cls(hypothesis="Too short to count as a mechanism.")()
    trials = TrialRegistry(root=tmp_path)
    with pytest.raises(MissingHypothesis):
        run_signal(sig, lambda s: {"sharpe": 9.9}, trials, ts=T0)
    assert trials.cumulative_trial_count() == 0
    assert not (tmp_path / "TRIALS.jsonl").exists()


def test_run_signal_rejects_non_signal() -> None:
    with pytest.raises(TypeError):
        run_signal("not a signal", lambda s: None, TrialRegistry())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# run_signal: the pre-run record_trial ordering
# ---------------------------------------------------------------------------


def test_run_signal_records_trial_before_backtest_runs(tmp_path: Path) -> None:
    register(_make_cls())
    sig = sigreg.create("gap_go", GapConfig(lookback_bars=2))
    trials = TrialRegistry(root=tmp_path)
    seen_during_backtest: list[int] = []

    def backtest(s: Signal):
        # The trial line must already exist while the backtest is running.
        seen_during_backtest.append(trials.cumulative_trial_count())
        return {"sharpe": 1.2}

    record, result = run_signal(sig, backtest, trials, ts=T0)
    assert seen_during_backtest == [1]
    assert result == {"sharpe": 1.2}
    assert record.trial_id == 1
    assert record.hypothesis == HYPOTHESIS
    assert record.ts == T0
    assert record.result == {"status": "pending"}
    expected_config = trial_config_for(sig)
    assert record.config == expected_config
    assert record.config_hash == canonical_config_hash(expected_config)
    # The full signal config landed in the recorded dict.
    assert record.config["signal_config"] == {"lookback_bars": 2, "min_gap_pct": 0.5}
    assert record.config["signal"] == "gap_go"
    assert record.config["warmup_bars"] == 3
    assert record.config["allowed_regimes"] == ALL_REGIMES


def test_run_signal_trial_line_survives_a_raising_backtest(tmp_path: Path) -> None:
    register(_make_cls())
    sig = sigreg.create("gap_go")
    trials = TrialRegistry(root=tmp_path)

    def exploding_backtest(s: Signal):
        raise RuntimeError("engine blew up")

    with pytest.raises(RuntimeError, match="engine blew up"):
        run_signal(sig, exploding_backtest, trials, ts=T0)

    # The experiment still exists in the registry — that is the point.
    records = list(trials.query())
    assert len(records) == 1
    assert records[0].hypothesis == HYPOTHESIS
    assert records[0].result == {"status": "pending"}
    assert records[0].config_hash == canonical_config_hash(trial_config_for(sig))


def test_run_signal_run_config_hashes_and_duplicates_flag(tmp_path: Path) -> None:
    register(_make_cls())
    sig = sigreg.create("gap_go")
    trials = TrialRegistry(root=tmp_path)
    run_cfg = {"start": "2026-01-05", "end": "2026-01-30", "universe": ["AAA"]}

    r1, _ = run_signal(sig, lambda s: {}, trials, run_config=run_cfg, ts=T0)
    assert r1.config["run"] == run_cfg
    assert r1.duplicate_of is None

    # Identical experiment re-run: recorded again, flagged as a duplicate.
    r2, _ = run_signal(sig, lambda s: {}, trials, run_config=run_cfg, ts=T0 + timedelta(minutes=1))
    assert r2.trial_id == 2
    assert r2.duplicate_of == 1

    # A different run_config is a different experiment.
    r3, _ = run_signal(
        sig, lambda s: {}, trials, run_config={**run_cfg, "end": "2026-02-06"},
        ts=T0 + timedelta(minutes=2),
    )
    assert r3.duplicate_of is None
    assert trials.cumulative_trial_count() == 3


# ---------------------------------------------------------------------------
# instantiation, configs, on_bar, regimes
# ---------------------------------------------------------------------------


def test_signal_config_forbids_extras_and_is_frozen() -> None:
    with pytest.raises(ValidationError):
        GapConfig(lookback_bars=3, bogus_knob=1)  # type: ignore[call-arg]
    cfg = GapConfig(lookback_bars=3)
    with pytest.raises(ValidationError):
        cfg.lookback_bars = 4  # type: ignore[misc]


def test_signal_independently_instantiable_and_backtestable() -> None:
    cls = _make_cls()  # never registered: still fully usable
    sig = cls()  # None -> config_type() defaults
    assert isinstance(sig.config, GapConfig)
    assert sig.config.lookback_bars == 5

    ctx = FakeCtx(_bar())
    assert isinstance(ctx, SignalContext)  # protocol satisfied
    events = sig.on_bar(ctx)
    assert len(events) == 1
    evt = events[0]
    assert isinstance(evt, SignalEvent)
    assert evt.signal_name == "gap_go"
    assert evt.symbol == "AAA"
    assert evt.side is Side.BUY


def test_signal_rejects_wrong_config_type() -> None:
    class OtherConfig(SignalConfig):
        knob: int = 1

    with pytest.raises(TypeError):
        _make_cls()(OtherConfig())


def test_regime_gating() -> None:
    gated = _make_cls(name="gated", allowed={"high_vol", "trend_up"})
    register(gated)
    assert regime_allowed(gated, "high_vol") is True
    assert regime_allowed(gated, "chop") is False
    assert regime_allowed(gated(), "trend_up") is True  # instance too
    ungated = _make_cls(name="ungated")
    assert regime_allowed(ungated, "anything") is True
    # The gated set is recorded (sorted) in the trial config.
    assert trial_config_for(gated())["allowed_regimes"] == ["high_vol", "trend_up"]
