"""Backtest metrics suite: distribution, risk-adjusted ratios, drawdown."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from catalyst.core.types import MonthlyReturnStats, TradeRecord

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


def _years_spanned(equity: pd.Series) -> float:
    """Elapsed years from DATES when available — the row-count bug class
    (audit MEDIUM: a 16-mark curve spanning one year read as 16 days)."""
    idx = equity.index
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 1:
        return (idx[-1] - idx[0]).days / 365.25
    return len(equity) / TRADING_DAYS_PER_YEAR


def calmar(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    years = _years_spanned(equity)
    if years <= 0 or equity.iloc[0] <= 0:
        return 0.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return float(cagr / mdd)


def avg_monthly_return(equity: pd.Series) -> float:
    """THE headline number for every strategy report.

    Geometric (compounding-consistent) average monthly return derived from the
    equity curve: the constant monthly rate that reproduces the realized total
    return over the period. Reported ahead of every other statistic because it
    is the figure decisions are actually made on.
    """
    if len(equity) < 2 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return 0.0
    # Elapsed time comes from the DATES when we have them, never from the row
    # count. A curve marked only at exits has ~16 rows across a year; reading
    # that as 16 trading days turns a +10% year into "+12.8% per month". Row
    # count is only a fallback for curves with no date index.
    idx = equity.index
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 1:
        months = (idx[-1] - idx[0]).days / (365.25 / 12.0)
    else:
        months = len(equity) / (TRADING_DAYS_PER_YEAR / 12.0)
    if months <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / months) - 1.0)


def monthly_return_series(equity: pd.Series) -> pd.Series:
    """Realized month-by-month returns, for distribution and consistency stats."""
    if equity.empty:
        return pd.Series(dtype=float)
    monthly = equity.resample("ME").last().dropna()
    return monthly.pct_change().dropna()


def headline(equity: pd.Series) -> dict[str, float]:
    """The standard header block every strategy report leads with."""
    monthly = monthly_return_series(equity)
    years = _years_spanned(equity)
    cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0) if years > 0 and len(equity) > 1 else 0.0
    return {
        "avg_monthly_return": avg_monthly_return(equity),
        "median_monthly_return": float(monthly.median()) if len(monthly) else 0.0,
        "best_month": float(monthly.max()) if len(monthly) else 0.0,
        "worst_month": float(monthly.min()) if len(monthly) else 0.0,
        "pct_months_positive": float((monthly > 0).mean()) if len(monthly) else 0.0,
        "cagr": cagr,
        "max_drawdown": max_drawdown(equity),
        "n_months": len(monthly),
    }


def profit_factor(trades: list[TradeRecord]) -> float:
    """Gross wins / gross losses. inf when there are no losing trades."""
    gross_win = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def expected_value(trades: list[TradeRecord]) -> float:
    """Mean P&L per trade in dollars."""
    return sum(t.pnl for t in trades) / len(trades) if trades else 0.0


def concentration(trades: list[TradeRecord], top_n: int = 3) -> dict[str, float | None]:
    """Top-N share of total P&L — the lottery-ticket artifact check.

    ``share`` is only defined when total P&L is positive: the check exists to
    ask whether a WINNING result rests on a handful of trades. Dividing a
    top-N sum by a negative total produces a meaningless negative percentage,
    so that case reports None instead of a number that reads like a finding.
    """
    if not trades:
        return {"top_n_pnl": 0.0, "total_pnl": 0.0, "share": None}
    pnls = sorted((t.pnl for t in trades), reverse=True)
    top, total = sum(pnls[:top_n]), sum(pnls)
    return {"top_n_pnl": top, "total_pnl": total,
            "share": (top / total) if total > 0 else None}


def drawdown_curve(equity: pd.Series) -> pd.Series:
    """Running drawdown as a negative fraction, for plotting."""
    if equity.empty:
        return equity
    return equity / equity.cummax() - 1.0


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
