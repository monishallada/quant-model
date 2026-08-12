"""Config system: load, env interpolation, immutability, sweep overrides."""

import pytest
from pydantic import ValidationError

from catalyst.core.config import load_config


def test_loads_backtest_environment() -> None:
    cfg = load_config("backtest")
    assert cfg.environment == "backtest"
    assert cfg.account.starting_capital == 100_000.0
    assert cfg.risk.cash_floor_fraction == 0.40


def test_risk_config_is_immutable() -> None:
    cfg = load_config("backtest")
    with pytest.raises(ValidationError):
        cfg.risk.cash_floor_fraction = 0.0  # type: ignore[misc]


def test_dotted_overrides_apply() -> None:
    cfg = load_config(
        "backtest",
        overrides={"engines.engine_a.per_trade_risk": 0.02, "risk.max_deployed": 0.15},
    )
    assert cfg.engines.engine_a.per_trade_risk == 0.02
    assert cfg.risk.max_deployed == 0.15


def test_overrides_still_validated() -> None:
    # A sweep can never construct an out-of-schema config (e.g. cash floor >= 1).
    with pytest.raises(ValidationError):
        load_config("backtest", overrides={"risk.cash_floor_fraction": 1.5})


def test_commission_profile_switch() -> None:
    alpaca = load_config("backtest")
    schwab = load_config("backtest", overrides={"execution.commissions.active_profile": "schwab"})
    assert alpaca.execution.commissions.per_contract == 0.0
    assert schwab.execution.commissions.per_contract == 0.65
