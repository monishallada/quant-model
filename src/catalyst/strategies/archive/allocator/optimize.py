"""Walk-forward optimization with an explicit overfitting measurement.

Two numbers come out of every parameter search and only one of them is real:

- **In-sample best** — the configuration that looked best on the data used to
  choose it. This number is always flattering and is not a forecast.
- **Out-of-sample result of that same configuration** — what the choice
  actually delivered on data it never saw. This is the honest expectation.

The gap between them is the overfitting premium. Reporting it is the whole
point: a search that improves in-sample by 3%/month and out-of-sample by
0.1%/month has found noise, and the only way to know is to measure both.

A third number, the ORACLE (best configuration chosen with hindsight on the
test period), bounds how much a search could possibly have extracted. When the
oracle is far above the honest out-of-sample result, the parameter surface is
mostly noise and further tuning is wasted effort.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from catalyst.backtest import metrics as m

logger = logging.getLogger(__name__)


@dataclass
class SweepRow:
    params: dict[str, float]
    train_monthly: float
    test_monthly: float
    train_dd: float
    test_dd: float
    test_cagr: float


def summarize_sweep(rows: list[SweepRow], baseline_test_monthly: float) -> str:
    """The report that separates a real improvement from a fitted one."""
    if not rows:
        return "no configurations evaluated"
    by_train = sorted(rows, key=lambda r: -r.train_monthly)
    by_test = sorted(rows, key=lambda r: -r.test_monthly)
    picked = by_train[0]     # what an honest search would have selected
    oracle = by_test[0]      # what hindsight could have selected

    lines = [
        "OVERFITTING MEASUREMENT",
        f"  configurations evaluated: {len(rows)}",
        "",
        f"  Chosen on TRAIN (the honest procedure): {picked.params}",
        f"    train  avg monthly {picked.train_monthly:+.2%}  (this is the flattering number)",
        f"    test   avg monthly {picked.test_monthly:+.2%}  (this is what it actually delivered)",
        f"    overfitting premium: {picked.train_monthly - picked.test_monthly:+.2%}/month",
        "",
        f"  Baseline (no optimization) test avg monthly: {baseline_test_monthly:+.2%}",
        f"  Genuine improvement from optimizing: "
        f"{picked.test_monthly - baseline_test_monthly:+.2%}/month",
        "",
        f"  ORACLE — best on TEST with hindsight: {oracle.params}",
        f"    test avg monthly {oracle.test_monthly:+.2%}",
        f"    unreachable gap (hindsight - honest): "
        f"{oracle.test_monthly - picked.test_monthly:+.2%}/month",
    ]
    # Rank correlation between train and test performance: does doing well
    # in-sample predict doing well out-of-sample at all?
    tr = pd.Series([r.train_monthly for r in rows])
    te = pd.Series([r.test_monthly for r in rows])
    rho = tr.corr(te, method="spearman")
    lines += [
        "",
        f"  train-vs-test rank correlation across configs: {rho:+.2f}",
        "    (near zero means in-sample ranking carries no information about"
        " out-of-sample performance — i.e. the search is fitting noise)",
    ]
    return "\n".join(lines)


def grid(**axes) -> list[dict]:
    """Cartesian product of named parameter axes."""
    keys = list(axes)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(axes[k] for k in keys))]


def evaluate_grid(
    configs: list[dict],
    run_fn: Callable[[dict, str], pd.Series],
) -> list[SweepRow]:
    """Run every configuration on train and test, returning both results."""
    rows: list[SweepRow] = []
    for i, params in enumerate(configs, 1):
        try:
            train_eq = run_fn(params, "train")
            test_eq = run_fn(params, "test")
        except Exception as exc:  # noqa: BLE001 — one bad config must not stop a sweep
            logger.warning("config %s failed: %s", params, exc)
            continue
        if train_eq.empty or test_eq.empty:
            continue
        th, sh = m.headline(train_eq), m.headline(test_eq)
        rows.append(SweepRow(
            params=params,
            train_monthly=th["avg_monthly_return"], test_monthly=sh["avg_monthly_return"],
            train_dd=th["max_drawdown"], test_dd=sh["max_drawdown"],
            test_cagr=sh["cagr"],
        ))
        if i % 10 == 0:
            logger.info("evaluated %d/%d configurations", i, len(configs))
    return rows
