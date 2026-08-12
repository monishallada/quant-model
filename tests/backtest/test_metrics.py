"""Metrics on hand-computed cases."""

import numpy as np
import pandas as pd
import pytest

from catalyst.core.config import MonteCarloConfig
from catalyst.backtest.metrics import calmar, max_drawdown, monthly_stats, sharpe, trade_stats
from catalyst.backtest.montecarlo import probability_of_ruin


def test_max_drawdown_hand_case() -> None:
    equity = pd.Series(
        [100.0, 110.0, 99.0, 121.0, 90.75, 130.0],
        index=pd.date_range("2024-01-01", periods=6, freq="B"),
    )
    # Worst: 121 -> 90.75 = -25%
    assert max_drawdown(equity) == pytest.approx(-0.25)


def test_sharpe_zero_for_flat_series() -> None:
    flat = pd.Series([0.0] * 100)
    assert sharpe(flat) == 0.0


def test_monthly_stats_shape() -> None:
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    rng = np.random.default_rng(3)
    equity = pd.Series(100_000 * np.cumprod(1 + rng.normal(0.0005, 0.01, 300)), index=idx)
    stats = monthly_stats(equity)
    assert stats.min <= stats.p05 <= stats.p25 <= stats.median <= stats.p75 <= stats.p95 <= stats.max
    assert sum(stats.histogram_counts) == len(stats.monthly_returns)


def test_trade_stats() -> None:
    from datetime import date, datetime

    from catalyst.core.models import Direction, TradeRecord

    def tr(pnl: float) -> TradeRecord:
        return TradeRecord(
            position_id="p", engine="e", catalyst_ref="c", underlying="SPY",
            direction=Direction.LONG, entry_time=datetime(2024, 1, 2),
            exit_time=datetime(2024, 1, 9), entry_price=1.0, exit_price=1.0,
            qty=1, pnl=pnl, exit_reason="test", max_qty=1,
        )

    stats = trade_stats([tr(100), tr(300), tr(-100), tr(-100)])
    assert stats["win_rate"] == 0.5
    assert stats["avg_win"] == 200
    assert stats["avg_loss"] == -100
    assert stats["win_loss_ratio"] == 2.0


def test_probability_of_ruin_extremes() -> None:
    cfg = MonteCarloConfig(n_paths=200, ruin_threshold_fraction=0.5, resample_block_size=5)
    steady_up = pd.Series([0.001] * 500)
    assert probability_of_ruin(steady_up, cfg) == 0.0
    crash = pd.Series([-0.05] * 500)
    assert probability_of_ruin(crash, cfg) == 1.0


def test_calmar_positive_for_uptrend() -> None:
    idx = pd.date_range("2024-01-01", periods=504, freq="B")
    equity = pd.Series(np.linspace(100_000, 140_000, 504), index=idx)
    equity.iloc[100] = 95_000  # inject a drawdown
    assert calmar(equity) > 0
