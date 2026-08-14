"""v7 Combined Allocator: a stable base sleeve plus a convex sleeve.

The structure is the one design idea from the competition brief that survived
measurement, and it is genuinely sound: a majority of capital in the strategy
with a measured, out-of-sample, survivorship-robust edge, and a minority in a
high-variance convex sleeve that supplies the right tail a tournament needs.

    80% — alpha sleeve: the v5 market-neutral momentum book run as a portable
          alpha overlay on SPY (measured +18%/yr, alpha stable train->test,
          robust to removing the 25 best-performing names)
    20% — convex sleeve: the v6 tournament engine's multi-week OTM calls on
          high-volatility names, ranked by the same validated signal

Ramp rule (from the brief): while equity is below ``ramp_threshold`` the
convex sleeve takes the whole account, because a small stake needs the tail
more than it needs stability; above it, the sleeve is capped at
``convex_fraction``.

Sleeve rebalancing is a real decision, not a detail. Rebalancing back to the
target weights harvests convex-sleeve gains into the stable book — better for
compounding, worse for the extreme right tail. Letting sleeves run preserves
the tail. Both are supported and reported, because which is correct depends
entirely on whether the objective is growth or placement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from catalyst.backtest import metrics as m

logger = logging.getLogger(__name__)


@dataclass
class AllocatorConfig:
    convex_fraction: float = 0.20      # steady-state share in the convex sleeve
    ramp_threshold: float = 10_000.0   # below this, the convex sleeve takes everything
    rebalance_sleeves: bool = True     # True: harvest to target weights; False: let run
    rebalance_days: int = 21


@dataclass
class AllocatorResult:
    equity: pd.Series
    stable_sleeve: pd.Series
    convex_sleeve: pd.Series
    benchmark: pd.Series

    def report(self, label: str) -> str:
        h = m.headline(self.equity)
        bh = m.headline(self.benchmark)
        monthly = m.monthly_return_series(self.equity)
        return "\n".join([
            f"===== {label} =====",
            f"  AVG MONTHLY RETURN: {h['avg_monthly_return']:+.2%}"
            f"    (SPY benchmark: {bh['avg_monthly_return']:+.2%})",
            f"  median month {h['median_monthly_return']:+.2%} | "
            f"best {h['best_month']:+.2%} | worst {h['worst_month']:+.2%} | "
            f"months positive {h['pct_months_positive']:.0%} ({h['n_months']} months)",
            f"  CAGR {h['cagr']:+.2%} | max drawdown {h['max_drawdown']:+.2%} | "
            f"final {self.equity.iloc[-1]:,.0f} from {self.equity.iloc[0]:,.0f}",
            f"  sleeves: stable ends {self.stable_sleeve.iloc[-1]:,.0f} | "
            f"convex ends {self.convex_sleeve.iloc[-1]:,.0f}",
        ])

    def summary(self, label: str) -> dict[str, float]:
        h = m.headline(self.equity)
        return {"strategy": label, **h,
                "final_equity": float(self.equity.iloc[-1]),
                "benchmark_monthly": m.headline(self.benchmark)["avg_monthly_return"]}


def combine_sleeves(
    stable_returns: pd.Series, convex_returns: pd.Series, benchmark: pd.Series,
    cfg: AllocatorConfig, starting_equity: float = 100_000.0,
) -> AllocatorResult:
    """Compound two return streams under the allocation and ramp rules.

    Both inputs are daily fractional returns on their OWN sleeve capital, so a
    sleeve that loses everything takes only its own allocation with it — which
    is the entire reason the convex sleeve is survivable at all.
    """
    idx = stable_returns.index.intersection(convex_returns.index)
    stable_r = stable_returns.reindex(idx).fillna(0.0)
    convex_r = convex_returns.reindex(idx).fillna(0.0)

    equity = starting_equity
    # Ramp: a sub-threshold account puts everything in the convex sleeve.
    convex_share = 1.0 if equity < cfg.ramp_threshold else cfg.convex_fraction
    convex_val = equity * convex_share
    stable_val = equity - convex_val

    eq_curve, stable_curve, convex_curve = {}, {}, {}
    for i, date in enumerate(idx):
        stable_val *= (1.0 + stable_r.loc[date])
        convex_val *= (1.0 + convex_r.loc[date])
        equity = stable_val + convex_val

        if cfg.rebalance_sleeves and i > 0 and i % cfg.rebalance_days == 0 and equity > 0:
            target_convex = 1.0 if equity < cfg.ramp_threshold else cfg.convex_fraction
            convex_val = equity * target_convex
            stable_val = equity - convex_val

        eq_curve[date] = equity
        stable_curve[date] = stable_val
        convex_curve[date] = convex_val

    eq = pd.Series(eq_curve)
    bench = benchmark.reindex(eq.index).ffill()
    bench = bench / bench.iloc[0] * starting_equity
    return AllocatorResult(eq, pd.Series(stable_curve), pd.Series(convex_curve), bench)


def to_daily_returns(equity: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Reindex a sleeve equity curve onto the master calendar as daily returns.

    Sleeve curves are recorded at their own cadence (the convex sleeve only
    marks at cycle boundaries), so gaps are forward-filled before differencing
    rather than treated as zero-return days.
    """
    curve = equity.reindex(index.union(equity.index)).ffill().reindex(index).ffill()
    return curve.pct_change().fillna(0.0)
