"""Typed, YAML-driven configuration.

All numeric parameters in the system live in ``config/*.yaml`` — code contains no
hardcoded numbers. Secrets are referenced in YAML as ``${ENV_VAR}`` and resolved
from the process environment (populated from ``.env``) at load time.

Loading semantics: ``base.yaml`` is loaded first, then the environment overlay
(``backtest.yaml`` / ``paper.yaml`` / ``live.yaml``) is deep-merged on top, then
``${VAR}`` references are interpolated, and the result is validated into the
frozen pydantic models below. Models are immutable: nothing can mutate risk
limits at runtime.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


class _FrozenModel(BaseModel):
    """Base for all config sections: immutable and intolerant of unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class AccountConfig(_FrozenModel):
    starting_capital: float


class ThetaDataConfig(_FrozenModel):
    base_url: str
    api_key: str
    request_timeout_seconds: float
    max_retries: int
    retry_backoff_seconds: float
    reconnect_wait_seconds: float
    max_concurrent_requests: int


class FMPConfig(_FrozenModel):
    base_url: str
    api_key: str
    request_timeout_seconds: float
    max_retries: int


class AlpacaConfig(_FrozenModel):
    endpoint: str
    api_key: str
    secret_key: str


class DataConfig(_FrozenModel):
    thetadata: ThetaDataConfig
    fmp: FMPConfig
    alpaca: AlpacaConfig
    cache_dir: str
    snapshot_time: str
    chain_max_dte: int
    chain_memory_cache_size: int
    risk_free_rate: float
    max_quote_age_seconds: float


class CatalystsConfig(_FrozenModel):
    calendars_dir: str
    economic_symbols: list[str]
    earnings_source_backtest: Literal["yfinance", "fmp"]
    earnings_source_live: Literal["yfinance", "fmp"]


class LiquidityConfig(_FrozenModel):
    max_spread_pct_of_mid: float
    min_option_volume: int
    min_open_interest: int


class ScoreWeights(_FrozenModel):
    liquidity: float
    implied_move: float
    days_to_catalyst_fit: float


class ScreenerConfig(_FrozenModel):
    min_implied_move: float
    liquidity: LiquidityConfig
    fit_ideal_days: int
    fit_tolerance_days: int
    move_score_saturation: float
    score_weights: ScoreWeights


class TrendSignalConfig(_FrozenModel):
    fast_lookback: int
    slow_lookback: int
    min_confidence: float


class MeanReversionSignalConfig(_FrozenModel):
    rsi_period: int
    rsi_oversold: float
    rsi_overbought: float
    zscore_window: int
    zscore_threshold: float


class SignalsConfig(_FrozenModel):
    trend: TrendSignalConfig
    mean_reversion: MeanReversionSignalConfig


class EngineAConfig(_FrozenModel):
    enabled: bool
    dte_window: tuple[int, int]
    iv_rank_max: float
    iv_rank_lookback_days: int
    target_delta: float
    delta_range: tuple[float, float]
    catalyst_life_fraction: tuple[float, float]
    per_trade_risk: float
    tp1_gain: float
    tp1_fraction: float
    trail_stop_pct: float
    hold_through_catalyst: bool
    use_stops: bool


class EngineBConfig(_FrozenModel):
    enabled: bool
    dte_window: tuple[int, int]
    long_delta: float
    short_delta: float
    per_trade_risk: float
    tp_fraction_of_max: float
    use_stops: bool


class EngineCConfig(_FrozenModel):
    enabled: bool
    entry_window_days: tuple[int, int]
    min_surprise_pct: float
    iv_rank_max: float
    structure: Literal["debit_spread", "long_option"]
    long_delta: float
    short_delta: float
    single_leg_delta: float
    expiry_weeks: tuple[int, int]
    per_trade_risk: float
    hold_days: int
    trail_stop_pct: float
    use_stops: bool


class EngineDConfig(_FrozenModel):
    enabled: bool
    dte_window: tuple[int, int]
    term_structure_ratio_min: float
    per_trade_risk: float
    max_loss_guard_pct: float
    use_stops: bool


class EnginesConfig(_FrozenModel):
    engine_a: EngineAConfig
    engine_b: EngineBConfig
    engine_c: EngineCConfig
    engine_d: EngineDConfig


class HedgeConfig(_FrozenModel):
    symbol: str
    put_delta: float
    dte_window_days: tuple[int, int]
    min_add_fraction: float
    check_weekday: int = Field(ge=0, le=4)


class RiskConfig(_FrozenModel):
    """Authoritative limits. Frozen; RiskManager is their only interpreter.

    Nothing in this schema accepts a return target, and no field here may be
    loosened at runtime — the models are immutable by construction.
    """

    cash_floor_fraction: float = Field(gt=0.0, lt=1.0)
    sizing_cost_buffer: float = Field(ge=0.0, le=0.5)
    hedge_fraction: float = Field(ge=0.0, lt=1.0)
    hedge: HedgeConfig
    max_deployed: float = Field(gt=0.0, lt=1.0)
    max_correlated_positions: int = Field(ge=1)
    correlation_lookback_days: int
    correlation_threshold: float
    daily_loss_halt: float = Field(lt=0.0)
    weekly_loss_halt: float = Field(lt=0.0)
    stop_loss_pct: float = Field(lt=0.0)
    close_before_expiry_days: int = Field(ge=0)
    time_stop_after_catalyst_days: int = Field(ge=0)


class FillModelConfig(_FrozenModel):
    spread_fill_fraction: float = Field(ge=0.0, le=1.0)
    slippage_pct_of_premium: float = Field(ge=0.0)


class CommissionsConfig(_FrozenModel):
    alpaca_per_contract: float
    schwab_per_contract_per_leg: float
    active_profile: Literal["alpaca", "schwab"]

    @property
    def per_contract(self) -> float:
        """Commission per contract per leg for the active profile."""
        if self.active_profile == "alpaca":
            return self.alpaca_per_contract
        return self.schwab_per_contract_per_leg


class ExecutionConfig(_FrozenModel):
    fill_model: FillModelConfig
    commissions: CommissionsConfig
    limit_price_offset: str
    limit_ticks_toward_fill: int
    max_order_retries: int
    retry_reprice_ticks: int
    order_poll_seconds: float
    unfilled_cancel_seconds: float


class WalkForwardConfig(_FrozenModel):
    train_months: int
    test_months: int
    step_months: int


class MonteCarloConfig(_FrozenModel):
    n_paths: int
    ruin_threshold_fraction: float
    resample_block_size: int


class SweepConfig(_FrozenModel):
    mode: Literal["grid", "random"]
    random_samples: int
    parameters: dict[str, list[Any]]


class BacktestConfig(_FrozenModel):
    start_date: str
    end_date: str
    train_test_split: float
    walk_forward: WalkForwardConfig
    monte_carlo: MonteCarloConfig
    sweep: SweepConfig


class ObservabilityConfig(_FrozenModel):
    log_dir: str
    log_level: str
    heartbeat_seconds: float
    stale_data_alert_seconds: float
    kill_switch_file: str
    kill_switch_flatten: bool


class Config(_FrozenModel):
    """Root configuration object injected throughout the system."""

    environment: Literal["backtest", "paper", "live"]
    account: AccountConfig
    data: DataConfig
    catalysts: CatalystsConfig
    watchlist: list[str]
    screener: ScreenerConfig
    signals: SignalsConfig
    engines: EnginesConfig
    risk: RiskConfig
    execution: ExecutionConfig
    backtest: BacktestConfig
    observability: ObservabilityConfig


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base`` (overlay wins; dicts merge)."""
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _interpolate_env(node: Any) -> Any:
    """Replace ``${VAR}`` string references with environment values.

    Raises if a referenced variable is unset — a missing secret must fail loudly
    at startup, never silently downstream.
    """
    if isinstance(node, dict):
        return {k: _interpolate_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_interpolate_env(v) for v in node]
    if isinstance(node, str):

        def _sub(match: re.Match[str]) -> str:
            var = match.group(1)
            value = os.environ.get(var)
            if value is None:
                raise KeyError(f"Config references ${{{var}}} but it is not set in the environment/.env")
            return value

        return _ENV_VAR_PATTERN.sub(_sub, node)
    return node


def _apply_dotted_overrides(tree: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply {"engines.engine_a.per_trade_risk": 0.02}-style overrides.

    This is how the parameter sweep varies configs: overrides are applied to
    the raw dict BEFORE validation, so every swept config passes the same
    schema (and the same risk-limit bounds) as a hand-written one.
    """
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
    environment: Literal["backtest", "paper", "live"],
    config_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load base.yaml + the environment overlay into a validated, frozen Config.

    Args:
        environment: which overlay to apply (backtest / paper / live).
        config_dir: directory holding the YAML files (default: <repo_root>/config).
        repo_root: repository root, used to locate config/ and .env
            (default: two parents above this package's src/ directory).
        overrides: dotted-path parameter overrides (sweep/walk-forward use).
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    cfg_dir = Path(config_dir) if config_dir else root / "config"

    load_dotenv(root / ".env")

    with open(cfg_dir / "base.yaml") as f:
        base = yaml.safe_load(f)
    with open(cfg_dir / f"{environment}.yaml") as f:
        overlay = yaml.safe_load(f) or {}

    merged = _deep_merge(base, overlay)
    if overrides:
        merged = _apply_dotted_overrides(merged, overrides)
    merged = _interpolate_env(merged)
    return Config.model_validate(merged)
