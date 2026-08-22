"""Typed, YAML-driven configuration for the edge platform.

All numeric parameters live in ``config/edge.yaml`` — code contains no
hardcoded numbers. The YAML is validated into the frozen pydantic models
below at load time; nothing can mutate a limit at runtime. Dotted-path
overrides exist for sweeps and are applied to the raw dict BEFORE
validation, so every swept config passes the same schema as a hand-written
one.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _FrozenModel(BaseModel):
    """Base for all config sections: immutable and intolerant of unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DataConfig(_FrozenModel):
    #: First day of the lockbox window; market data on/after this date is
    #: untouchable by research and fitting until the lockbox is opened.
    lockbox_start: date


class ValidationConfig(_FrozenModel):
    """Deployment gates: what a strategy must survive before paper/live."""

    min_oos_trades: int = Field(gt=0)
    pbo_max: float = Field(gt=0.0, le=1.0)
    bootstrap_resamples: int = Field(gt=0)


class ExecutionConfig(_FrozenModel):
    """Fill-model frictions applied identically in backtest and paper."""

    spread_fill_fraction: float = Field(ge=0.0, le=1.0)
    slippage_pct: float = Field(ge=0.0)
    commission_per_contract: float = Field(ge=0.0)
    #: Latency scenarios (milliseconds) every fill model must be run under.
    latency_ms: list[int]


class RiskConfig(_FrozenModel):
    """Authoritative limits; the risk module is their only interpreter."""

    kelly_fraction: float = Field(gt=0.0, le=1.0)
    per_trade_cap_pct: float = Field(gt=0.0)
    daily_loss_halt_pct: float = Field(gt=0.0)
    #: Optional vol targeting (percent); null disables it.
    target_daily_vol_pct: float | None = Field(default=None, gt=0.0)


class EdgeConfig(_FrozenModel):
    """Root configuration object injected throughout the edge platform."""

    data: DataConfig
    validation: ValidationConfig
    execution: ExecutionConfig
    risk: RiskConfig


def _apply_dotted_overrides(tree: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply {"risk.kelly_fraction": 0.1}-style overrides to the raw dict."""
    out = dict(tree)
    for dotted, value in overrides.items():
        node = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            node[part] = dict(node[part])
            node = node[part]
        node[parts[-1]] = value
    return out


def load_config(
    config_path: str | Path | None = None,
    repo_root: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> EdgeConfig:
    """Load ``config/edge.yaml`` into a validated, frozen :class:`EdgeConfig`.

    Args:
        config_path: explicit YAML path (default: ``<repo_root>/config/edge.yaml``).
        repo_root: repository root used to locate config/
            (default: three parents above this module's package directory).
        overrides: dotted-path parameter overrides (sweep use); applied to the
            raw dict before validation so overridden configs obey the schema.
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    path = Path(config_path) if config_path else root / "config" / "edge.yaml"

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    if overrides:
        raw = _apply_dotted_overrides(raw, overrides)
    return EdgeConfig.model_validate(raw)
