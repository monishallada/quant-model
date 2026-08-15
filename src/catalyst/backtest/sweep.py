"""Parameter sweep: grid/random search over config overrides.

Every combination re-validates through the same frozen config schema, so a
sweep can explore sizing/deltas/DTE/stops but can never construct a config
that violates schema bounds. Output reports BOTH the return distribution and
the survival stats per combination — the aggressiveness dial is read off this
table, never optimized on return alone.
"""

from __future__ import annotations

import itertools
import random
from typing import Any, Callable, Iterator

import pandas as pd

from catalyst.core.config import SweepConfig
from catalyst.core.types import BacktestResult


def iter_combinations(cfg: SweepConfig, seed: int = 11) -> Iterator[dict[str, Any]]:
    names = list(cfg.parameters.keys())
    value_lists = [cfg.parameters[n] for n in names]
    if cfg.mode == "grid":
        for combo in itertools.product(*value_lists):
            yield dict(zip(names, combo))
    else:
        rng = random.Random(seed)
        for _ in range(cfg.random_samples):
            yield {n: rng.choice(vals) for n, vals in zip(names, value_lists)}


def run_sweep(
    cfg: SweepConfig,
    run_with_overrides: Callable[[dict[str, Any], str], BacktestResult],
) -> pd.DataFrame:
    """Run one backtest per combination; return the comparison table."""
    rows: list[dict[str, Any]] = []
    for i, overrides in enumerate(iter_combinations(cfg)):
        result = run_with_overrides(overrides, f"sweep-{i}")
        rows.append(
            {
                **{f"param:{k}": str(v) for k, v in overrides.items()},
                "total_return": result.total_return,
                "monthly_mean": result.monthly.mean,
                "monthly_median": result.monthly.median,
                "monthly_p05": result.monthly.p05,
                "monthly_p95": result.monthly.p95,
                "win_rate": result.win_rate,
                "max_drawdown": result.max_drawdown,
                "sharpe": result.sharpe,
                "sortino": result.sortino,
                "calmar": result.calmar,
                "prob_ruin": result.probability_of_ruin,
                "mc_survival": result.mc_survival_rate,
                "n_trades": result.n_trades,
            }
        )
    return pd.DataFrame(rows)
