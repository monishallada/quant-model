"""NeutralSignal: the control arm. Always says no-trade.

Any engine whose backtest edge disappears under a real signal but persists
under baselines is fooling itself; any engine that trades under Neutral has a
bug (directional engines must skip on neutral).
"""

from __future__ import annotations

import pandas as pd

from catalyst.core.interfaces import DirectionalSignal
from catalyst.core.types import Direction, OptionChain, SignalResult


class NeutralSignal(DirectionalSignal):
    name = "neutral"

    def evaluate(
        self, symbol: str, history: pd.DataFrame, chain: OptionChain | None = None
    ) -> SignalResult:
        return SignalResult(direction=Direction.NEUTRAL, confidence=0.0)
