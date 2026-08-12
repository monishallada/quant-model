"""Walk-forward window generation + execution.

Overfitting control #2 (with the chronological train/test split): rolling
(train, test) windows over the full range. Parameters are chosen on train
segments only; every reported edge must survive on the untouched test
segments that follow them in time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

from dateutil.relativedelta import relativedelta

from catalyst.core.config import WalkForwardConfig
from catalyst.core.models import BacktestResult


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: date
    train_end: date
    test_start: date
    test_end: date


def walk_forward_windows(start: date, end: date, cfg: WalkForwardConfig) -> list[WalkForwardWindow]:
    windows: list[WalkForwardWindow] = []
    cursor = start
    while True:
        train_end = cursor + relativedelta(months=cfg.train_months)
        test_end = train_end + relativedelta(months=cfg.test_months)
        if test_end > end:
            break
        windows.append(
            WalkForwardWindow(
                train_start=cursor,
                train_end=train_end,
                test_start=train_end,
                test_end=test_end,
            )
        )
        cursor = cursor + relativedelta(months=cfg.step_months)
    return windows


def run_walk_forward(
    windows: list[WalkForwardWindow],
    run_segment: Callable[[date, date, str], BacktestResult],
) -> list[tuple[WalkForwardWindow, BacktestResult]]:
    """Run the test segment of each window (train segments are consumed by the
    sweep when parameter optimization is in play)."""
    results = []
    for i, w in enumerate(windows):
        results.append((w, run_segment(w.test_start, w.test_end, f"wf-test-{i}")))
    return results


def chronological_split(start: date, end: date, train_fraction: float) -> tuple[date, date]:
    """(train_end, test_start) boundary for the configured chronological split."""
    span_days = (end - start).days
    boundary = start + relativedelta(days=int(span_days * train_fraction))
    return boundary, boundary
