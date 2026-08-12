"""MeanReversionSignal — UNVALIDATED test candidate.

Fades extremes: RSI oversold + deep negative z-score → long; RSI overbought +
high z-score → short; otherwise neutral. Confidence scales with how extreme
both readings are. Shipped as a baseline for A/B testing; assume no edge
until the out-of-sample backtest says otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from catalyst.core.config import MeanReversionSignalConfig
from catalyst.core.interfaces import DirectionalSignal
from catalyst.core.models import Direction, OptionChain, SignalResult


def rsi(closes: pd.Series, period: int) -> float:
    delta = closes.diff().dropna()
    if len(delta) < period:
        return 50.0
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = float(gains.ewm(alpha=1.0 / period, min_periods=period).mean().iloc[-1])
    avg_loss = float(losses.ewm(alpha=1.0 / period, min_periods=period).mean().iloc[-1])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


class MeanReversionSignal(DirectionalSignal):
    name = "mean_reversion"

    def __init__(self, cfg: MeanReversionSignalConfig) -> None:
        self._cfg = cfg

    def evaluate(
        self, symbol: str, history: pd.DataFrame, chain: OptionChain | None = None
    ) -> SignalResult:
        cfg = self._cfg
        closes = history["close"] if "close" in history else pd.Series(dtype=float)
        if len(closes) < max(cfg.rsi_period + 1, cfg.zscore_window):
            return SignalResult(direction=Direction.NEUTRAL, confidence=0.0)

        window = closes.tail(cfg.zscore_window)
        std = float(window.std(ddof=1))
        if std == 0 or np.isnan(std):
            return SignalResult(direction=Direction.NEUTRAL, confidence=0.0)
        z = (float(closes.iloc[-1]) - float(window.mean())) / std
        r = rsi(closes, cfg.rsi_period)

        if r <= cfg.rsi_oversold and z <= -cfg.zscore_threshold:
            confidence = min(abs(z) / (2 * cfg.zscore_threshold), 1.0)
            return SignalResult(direction=Direction.LONG, confidence=confidence)
        if r >= cfg.rsi_overbought and z >= cfg.zscore_threshold:
            confidence = min(abs(z) / (2 * cfg.zscore_threshold), 1.0)
            return SignalResult(direction=Direction.SHORT, confidence=confidence)
        return SignalResult(direction=Direction.NEUTRAL, confidence=0.0)
