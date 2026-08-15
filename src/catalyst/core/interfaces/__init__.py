"""Abstract boundaries of the system.

Swapping an implementation behind one of these is the ONLY thing that differs
between backtest, paper and live. Strategy, risk, cost and execution code is
identical in all three modes.
"""

from catalyst.core.interfaces.base import Broker, DataSource, DirectionalSignal
from catalyst.core.interfaces.strategy import (
    Cadence,
    CatalystStrategy,
    Opportunity,
    Strategy,
    StrategyContext,
)

__all__ = [
    "Broker",
    "Cadence",
    "CatalystStrategy",
    "DataSource",
    "DirectionalSignal",
    "Opportunity",
    "Strategy",
    "StrategyContext",
]
