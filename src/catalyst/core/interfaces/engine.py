"""Backtest engine abstraction — run the same strategy on more than one engine.

The reason to run two engines is not that one is "more accurate". It is that
**two independent implementations disagreeing is a bug signal**, and a bug
signal is the only thing that has ever actually changed a conclusion in this
project. Every wrong number here came from an accounting or data-handling
fault — a credit double-count, a split mismatch, a row-count time base — and
each of those would have shown up immediately as divergence against a second
engine computing the same P&L a different way.

Agreement is weaker evidence than disagreement. Two engines agreeing means
neither has an obvious ledger bug; it does not make a strategy correct, and it
certainly does not make it profitable.

An engine that cannot run must say so cleanly via ``available()`` rather than
failing mid-backtest, so a missing runtime degrades the report rather than
destroying it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass
class EngineResult:
    """One engine's view of one (strategy, window, cost-profile) run."""

    engine: str
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trades: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and len(self.equity) > 1

    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1]) if len(self.equity) else 0.0


class BacktestEngine(ABC):
    """A way of turning a strategy into an equity curve and a trade list."""

    name: str = "unnamed"
    #: Human-readable statement of what this engine independently verifies —
    #: printed in the comparison so nobody over-reads an agreement.
    verifies: str = ""

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """(usable, reason). Checked before every run; never raises."""

    @abstractmethod
    def run(
        self,
        strategy,
        start: date,
        end: date,
        *,
        cfg,
        data,
        signal,
        catalysts=None,
        screener=None,
        zero_cost: bool = False,
    ) -> EngineResult:
        """Execute. Must return an EngineResult with ``error`` set rather than
        raising — one engine failing must not abort the others."""
