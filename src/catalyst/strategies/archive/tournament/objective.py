"""Tournament objective: maximize P(reaching a target multiple), not E[return].

This is a genuinely different optimization from every prior campaign and the
distinction drives every design choice downstream.

Compounding capital maximizes expected log wealth — Kelly, diversification,
drawdown control. A winner-take-all tournament with a bounded entry stake
maximizes ``P(final >= target)``. Under that objective:

- Variance becomes an ASSET. Diversification, which reduces variance, is
  actively harmful once the median outcome is already far below the target.
- Negative expected value can be optimal. A bet with EV < 0 but a fat right
  tail beats a bet with EV > 0 and no tail, if only the tail reaches the target.
- Taking profits is harmful. Capping the upside removes exactly the outcomes
  that win.
- A high probability of ruin is not a flaw; it is the price of the tail, and
  it is the correct trade when the downside is a fixed, affordable stake.

This module reports the honest distribution — the probability of hitting each
threshold AND the probability of being wiped out — so the tradeoff is always
visible rather than buried in an average.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_MONTH = 21


@dataclass(frozen=True)
class TournamentResult:
    label: str
    n_windows: int
    window_months: int
    multiples: list[float]  # final equity multiple per window

    def p_at_least(self, x: float) -> float:
        if not self.multiples:
            return 0.0
        return float(np.mean([m >= x for m in self.multiples]))

    @property
    def p_ruin(self) -> float:
        """Fraction of windows ending below 10% of the stake."""
        return self.p_at_most(0.10)

    def p_at_most(self, x: float) -> float:
        if not self.multiples:
            return 0.0
        return float(np.mean([m <= x for m in self.multiples]))

    @property
    def median(self) -> float:
        return float(np.median(self.multiples)) if self.multiples else 0.0

    @property
    def mean(self) -> float:
        return float(np.mean(self.multiples)) if self.multiples else 0.0

    @property
    def best(self) -> float:
        return float(np.max(self.multiples)) if self.multiples else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "n_windows": self.n_windows,
            "median_multiple": self.median,
            "mean_multiple": self.mean,
            "best_multiple": self.best,
            "p_2x": self.p_at_least(2.0),
            "p_5x": self.p_at_least(5.0),
            "p_10x": self.p_at_least(10.0),
            "p_20x": self.p_at_least(20.0),
            "p_ruin": self.p_ruin,
        }

    def report(self) -> str:
        s = self.summary()
        return (
            f"{self.label}\n"
            f"  windows: {self.n_windows} x {self.window_months}mo | "
            f"median {s['median_multiple']:.2f}x | mean {s['mean_multiple']:.2f}x | "
            f"best {s['best_multiple']:.1f}x\n"
            f"  P(2x) {s['p_2x']:.1%} | P(5x) {s['p_5x']:.1%} | "
            f"P(10x) {s['p_10x']:.1%} | P(20x) {s['p_20x']:.1%} | "
            f"P(ruin) {s['p_ruin']:.1%}"
        )


def rolling_windows(
    index: pd.DatetimeIndex, window_months: int, step_days: int = 21
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Overlapping (start, end) pairs of ``window_months`` length.

    Overlapping windows inflate apparent sample size — 90 windows drawn from
    8 years are nowhere near 90 independent observations. They are still the
    right tool here because the question is "how often would a competition
    entrant starting at an arbitrary date have hit the target", and every
    start date is a real scenario. The independence caveat belongs in the
    report, not in a decision to discard the analysis.
    """
    length = window_months * TRADING_DAYS_PER_MONTH
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    i = 0
    while i + length < len(index):
        out.append((index[i], index[i + length]))
        i += step_days
    return out


def evaluate_paths(label: str, equity_paths: list[pd.Series], window_months: int
                   ) -> TournamentResult:
    """Convert per-window equity curves into a threshold distribution."""
    multiples = []
    for path in equity_paths:
        if path.empty or path.iloc[0] <= 0:
            continue
        multiples.append(float(path.iloc[-1] / path.iloc[0]))
    return TournamentResult(label, len(multiples), window_months, multiples)


def kelly_fraction(win_prob: float, payoff_ratio: float) -> float:
    """Kelly stake — included as the REFERENCE point the tournament deliberately
    overbets relative to. Kelly maximizes long-run growth; when the objective is
    reaching a target before a deadline with unfavorable odds, bold play (betting
    more than Kelly) raises P(target) even though it lowers expected log wealth."""
    if payoff_ratio <= 0:
        return 0.0
    edge = win_prob * (1 + payoff_ratio) - 1
    return max(edge / payoff_ratio, 0.0)
