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

from datetime import timedelta

from dateutil.relativedelta import relativedelta

from catalyst.core.config import WalkForwardConfig
from catalyst.core.types import BacktestResult


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
                # STRICTLY after train_end: consumers slice sessions with
                # both ends inclusive, so test_start == train_end put the
                # boundary session in BOTH windows (audit D-027)
                test_start=train_end + timedelta(days=1),
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
    """(train_end, test_start) for the LEGACY split — both are the same date.

    Callers run train as [start, boundary] and test as [boundary, end], so the
    boundary session belongs to BOTH segments: a one-session overlap out of
    ~2160. Left exactly as-is so every archived campaign reproduces its
    original numbers. New work should use ``chronological_split_exclusive``.
    """
    span_days = (end - start).days
    boundary = start + relativedelta(days=int(span_days * train_fraction))
    return boundary, boundary


def chronological_split_exclusive(
    start: date, end: date, train_fraction: float
) -> tuple[date, date]:
    """Non-overlapping (train_end, test_start): test begins the day AFTER train ends.

    Same 70/30 proportion and the same cut point; the only difference is that
    the boundary session is no longer counted twice. This is what the shared
    pipeline uses, so no future strategy inherits the overlap.
    """
    boundary, _ = chronological_split(start, end, train_fraction)
    return boundary, boundary + relativedelta(days=1)
