"""TrendSignal — UNVALIDATED test candidate.

Price vs. moving-average crossover: long when close > fast MA > slow MA,
short when close < fast MA < slow MA, else neutral. Confidence scales with
the fast/slow MA separation. Shipped as a baseline for A/B testing; assume
no edge until the out-of-sample backtest says otherwise.
"""

from __future__ import annotations

import pandas as pd

from catalyst.core.config import TrendSignalConfig
from catalyst.core.interfaces import DirectionalSignal
from catalyst.core.models import Direction, OptionChain, SignalResult


class TrendSignal(DirectionalSignal):
    name = "trend"

    def __init__(self, cfg: TrendSignalConfig) -> None:
        self._cfg = cfg

    def evaluate(
        self, symbol: str, history: pd.DataFrame, chain: OptionChain | None = None
    ) -> SignalResult:
        cfg = self._cfg
        closes = history["close"] if "close" in history else pd.Series(dtype=float)
        if len(closes) < cfg.slow_lookback:
            return SignalResult(direction=Direction.NEUTRAL, confidence=0.0)
        fast = float(closes.tail(cfg.fast_lookback).mean())
        slow = float(closes.tail(cfg.slow_lookback).mean())
        last = float(closes.iloc[-1])
        if slow <= 0:
            return SignalResult(direction=Direction.NEUTRAL, confidence=0.0)
        separation = abs(fast - slow) / slow
        confidence = min(separation / 0.05, 1.0)  # saturate at 5% separation
        if last > fast > slow:
            direction = Direction.LONG
        elif last < fast < slow:
            direction = Direction.SHORT
        else:
            return SignalResult(direction=Direction.NEUTRAL, confidence=0.0)
        if confidence < cfg.min_confidence:
            return SignalResult(direction=Direction.NEUTRAL, confidence=confidence)
        return SignalResult(direction=direction, confidence=confidence)
