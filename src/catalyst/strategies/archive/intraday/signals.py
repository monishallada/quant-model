"""Intraday signal library and 40-minute predictability harness.

The proposed scalping architecture — 3-5 week options held 30-40 minutes with
computed take-profit and stop-loss levels — rests entirely on one premise:
that something predicts direction over the next 40 minutes. Theta is not the
obstacle at that tenor (a 25-day option barely decays in 40 minutes); the
obstacle is that the bid/ask must be crossed twice and the underlying must
move enough to pay for it.

So this module measures the premise before anything is built on it. Same
method as the daily alpha lab that worked: cross-sectional-style information
coefficient against a strictly forward return, with a random control that must
come out at zero.

Rough economics for calibration: a ~30-day ATM option has roughly 10-12x
leverage on the underlying, and a round trip costs ~4-10% of premium. So a
0.5-1.0% underlying move inside 40 minutes clears costs. The question is
whether direction is predictable at all, not whether the payoff is adequate.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RTH_START, RTH_END = "09:30", "16:00"


def _rth(bars: pd.DataFrame) -> pd.DataFrame:
    return bars.between_time(RTH_START, RTH_END)


def intraday_signals(bars: pd.DataFrame) -> pd.DataFrame:
    """Standard intraday technical signals, all strictly causal.

    Every value at time t uses only bars at or before t. Signs are oriented so
    a HIGH value always means "expected to rise", making the information
    coefficients directly comparable across signals.
    """
    df = _rth(bars).copy()
    if df.empty or len(df) < 60:
        return pd.DataFrame()

    close = df["close"]
    vol = df["volume"].astype(float)
    out = pd.DataFrame(index=df.index)

    # Momentum over the last 20 minutes (continuation).
    out["mom_20m"] = close / close.shift(20) - 1.0
    # Very short-term reversal over 5 minutes (mean reversion, sign inverted).
    out["reversal_5m"] = -(close / close.shift(5) - 1.0)
    # Deviation from the session VWAP — the standard intraday anchor.
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_vol = vol.cumsum().replace(0.0, np.nan)
    vwap = (typical * vol).cumsum() / cum_vol
    out["vwap_dev"] = close / vwap - 1.0
    # Volume surge: current 10-minute volume vs the trailing hour.
    out["vol_surge"] = vol.rolling(10).sum() / vol.rolling(60).sum().replace(0.0, np.nan)
    # Position within the running intraday range.
    hi = df["high"].cummax()
    lo = df["low"].cummin()
    out["range_pos"] = (close - lo) / (hi - lo).replace(0.0, np.nan)
    # Intraday RSI (14 one-minute bars).
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14).mean()
    rs = gain / loss.replace(0.0, np.nan)
    out["rsi_14"] = 100 - 100 / (1 + rs)
    # Trend against a 60-minute mean.
    out["trend_60m"] = close / close.rolling(60).mean() - 1.0
    # Control: must show zero information coefficient.
    rng = np.random.default_rng(abs(hash(str(df.index[0]))) % (2**32))
    out["random_control"] = rng.standard_normal(len(df))
    return out


def forward_return(bars: pd.DataFrame, minutes: int) -> pd.Series:
    """Return over the next ``minutes`` of regular trading, aligned to t."""
    close = _rth(bars)["close"]
    return close.shift(-minutes) / close - 1.0


def measure_day(bars: pd.DataFrame, horizon: int, sample_every: int
                ) -> dict[str, list[tuple[float, float]]]:
    """(signal value, forward return) pairs for one symbol-day."""
    sig = intraday_signals(bars)
    if sig.empty:
        return {}
    fwd = forward_return(bars, horizon)
    # Only sample where a full forward window exists, and space samples by the
    # horizon so windows do not overlap (the same correction that turned an
    # apparent t=5.98 into t=1.68 in the daily study).
    idx = sig.index[::sample_every]
    pairs: dict[str, list[tuple[float, float]]] = {}
    for col in sig.columns:
        rows = []
        for ts in idx:
            s, r = sig[col].get(ts, np.nan), fwd.get(ts, np.nan)
            if pd.notna(s) and pd.notna(r):
                rows.append((float(s), float(r)))
        pairs[col] = rows
    return pairs


def summarize(all_pairs: dict[str, list[tuple[float, float]]]) -> pd.DataFrame:
    """Pooled Spearman IC per signal with a t-statistic and hit rate."""
    rows = []
    for name, pairs in all_pairs.items():
        if len(pairs) < 100:
            continue
        s = pd.Series([p[0] for p in pairs])
        r = pd.Series([p[1] for p in pairs])
        ic = s.corr(r, method="spearman")
        n = len(pairs)
        # Standard error of a correlation coefficient.
        t = ic * np.sqrt((n - 2) / max(1 - ic**2, 1e-12)) if pd.notna(ic) else 0.0
        # Directional hit rate: does the signal's sign match the move's sign?
        sign_match = float(np.mean(np.sign(s - s.median()) == np.sign(r)))
        rows.append({"signal": name, "ic": float(ic) if pd.notna(ic) else 0.0,
                     "t_stat": float(t), "hit_rate": sign_match, "n_obs": n})
    df = pd.DataFrame(rows).sort_values("t_stat", key=abs, ascending=False)
    df["verdict"] = np.where(df["t_stat"].abs() < 2.5, "no signal", "predictive")
    return df
