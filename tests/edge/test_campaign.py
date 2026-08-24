"""The campaign runner must APPLY the execution parameters it RECORDS."""

from __future__ import annotations

from datetime import date

from edge.core.config import load_config
from edge.runners.campaign import RunSpec, _run_config


def _spec(**kw) -> RunSpec:
    return RunSpec(
        signal_name="probe",
        build_signal=lambda: None,  # type: ignore[arg-type,return-value]
        span="OOS",
        symbols=("SPY",),
        start=date(2024, 1, 2),
        end=date(2024, 2, 1),
        **kw,
    )


def test_recorded_overrides_are_derived_from_the_applied_ones() -> None:
    """No parallel claim: the record is a projection of what is applied."""
    spec = _spec(execution_overrides={"slippage_pct": 0.0, "spread_fill_fraction": 1.0})
    recorded = _run_config(spec)["config_overrides"]
    assert recorded == {
        "execution.slippage_pct": 0.0,
        "execution.spread_fill_fraction": 1.0,
    }


def test_overrides_change_the_config_the_engine_would_see() -> None:
    """A recorded zero-slippage run must really be zero-slippage."""
    config = load_config()
    assert config.execution.slippage_pct > 0.0  # the option-oriented default
    spec = _spec()
    applied = config.model_copy(
        update={"execution": config.execution.model_copy(
            update=dict(spec.execution_overrides))}
    )
    assert applied.execution.slippage_pct == 0.0
    assert applied.execution.commission_per_contract == 0.0
    assert applied.execution.spread_fill_fraction == 1.0
    # the loaded config is untouched — overrides are per-run, never global
    assert config.execution.slippage_pct > 0.0
