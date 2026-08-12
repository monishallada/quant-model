"""Backtest metrics suite: distribution, risk-adjusted ratios, drawdown."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from catalyst.core.models import MonthlyReturnStats, TradeRecord

TRADING_DAYS_PER_YEAR = 252
_HISTOGRAM_BINS = 20


def monthly_stats(equity: pd.Series) -> MonthlyReturnStats:
    """Monthly return distribution from a daily equity curve."""
    monthly_equity = equity.resample("ME").last().dropna()
    returns = monthly_equity.pct_change().dropna()
    if returns.empty:
        returns = pd.Series([0.0])
        index_labels = ["n/a"]
    else:
        index_labels = [ts.strftime("%Y-%m") for ts in returns.index]
    values = returns.to_numpy()
    counts, edges = np.histogram(values, bins=_HISTOGRAM_BINS)
    return MonthlyReturnStats(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        p05=float(np.percentile(values, 5)),
        p25=float(np.percentile(values, 25)),
        p75=float(np.percentile(values, 75)),
        p95=float(np.percentile(values, 95)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        histogram_bins=[float(e) for e in edges],
        histogram_counts=[int(c) for c in counts],
        monthly_returns=dict(zip(index_labels, (float(v) for v in values))),
    )


def max_drawdown(equity: pd.Series) -> float:
    """Peak-to-trough drawdown as a negative fraction."""
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min()) if len(dd) else 0.0


def sharpe(daily_returns: pd.Series) -> float:
    if len(daily_returns) < 2 or daily_returns.std(ddof=1) == 0:
        return 0.0
    return float(daily_returns.mean() / daily_returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))


def sortino(daily_returns: pd.Series) -> float:
    downside = daily_returns[daily_returns < 0]
    if len(daily_returns) < 2 or len(downside) == 0:
        return 0.0
    downside_std = float(np.sqrt(np.mean(np.square(downside))))
    if downside_std == 0:
        return 0.0
    return float(daily_returns.mean() / downside_std * math.sqrt(TRADING_DAYS_PER_YEAR))


def calmar(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    years = len(equity) / TRADING_DAYS_PER_YEAR
    if years <= 0 or equity.iloc[0] <= 0:
        return 0.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return float(cagr / mdd)


def trade_stats(trades: list[TradeRecord]) -> dict[str, float]:
    if not trades:
        return {"win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "win_loss_ratio": 0.0}
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return {
        "win_rate": len(wins) / len(pnls),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": abs(avg_win / avg_loss) if avg_loss != 0 else 0.0,
    }
