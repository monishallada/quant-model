"""Baseline signals: direction logic and neutral defaults."""

from __future__ import annotations

import numpy as np
import pandas as pd

from catalyst.core.config import load_config
from catalyst.core.models import Direction
from catalyst.signals.mean_reversion import MeanReversionSignal, rsi
from catalyst.signals.neutral import NeutralSignal
from catalyst.signals.trend import TrendSignal

CFG = load_config("backtest")


def make_history(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2023-01-02", periods=len(closes), freq="B")
    c = pd.Series(closes, index=idx)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                         "volume": 1_000_000})


def test_neutral_always_no_trade() -> None:
    result = NeutralSignal().evaluate("SPY", make_history([100.0] * 60))
    assert result.direction is Direction.NEUTRAL


def test_trend_long_in_uptrend() -> None:
    closes = list(np.linspace(100, 160, 80))
    result = TrendSignal(CFG.signals.trend).evaluate("SPY", make_history(closes))
    assert result.direction is Direction.LONG
    assert result.confidence > 0


def test_trend_short_in_downtrend() -> None:
    closes = list(np.linspace(160, 100, 80))
    result = TrendSignal(CFG.signals.trend).evaluate("SPY", make_history(closes))
    assert result.direction is Direction.SHORT


def test_trend_neutral_without_enough_history() -> None:
    result = TrendSignal(CFG.signals.trend).evaluate("SPY", make_history([100.0] * 10))
    assert result.direction is Direction.NEUTRAL


def test_rsi_bounds() -> None:
    up = pd.Series(np.linspace(100, 130, 40))
    down = pd.Series(np.linspace(130, 100, 40))
    assert rsi(up, 14) > 80
    assert rsi(down, 14) < 20


def test_mean_reversion_long_after_capitulation() -> None:
    # Fast crash (6 sessions): slow declines raise the rolling std enough to
    # self-dampen the z-score, which is correct no-trade behavior.
    closes = [100.0] * 40 + list(np.linspace(100, 80, 6))
    result = MeanReversionSignal(CFG.signals.mean_reversion).evaluate("SPY", make_history(closes))
    assert result.direction is Direction.LONG


def test_mean_reversion_neutral_in_calm_market() -> None:
    rng = np.random.default_rng(5)
    closes = list(100 + np.cumsum(rng.normal(0, 0.2, 80)))
    result = MeanReversionSignal(CFG.signals.mean_reversion).evaluate("SPY", make_history(closes))
    assert result.direction is Direction.NEUTRAL
