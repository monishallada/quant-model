"""Leakage-safe cross-validation: purged/embargoed k-fold + walk-forward windows.

Overfitting control (Lopez de Prado, AFML ch. 7): samples carry label spans
[t0, t1]; any training sample whose label interval overlaps ANY test-sample
interval is purged, and an embargo of ``embargo_pct`` of samples after each
test block is dropped from training as well. Walk-forward windows are built
over an explicit session calendar with fit strictly before test. All window
and fold sizes are caller-supplied — nothing is hardcoded here.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class PurgedKFold:
    """K-fold CV over time-ordered samples with label spans, purge + embargo.

    Test folds are contiguous positional blocks covering all samples exactly
    once. For each fold, a candidate training sample ``i`` is PURGED when its
    closed label interval ``[t0_i, t1_i]`` overlaps the closed interval of any
    test sample, and the first ``ceil(embargo_pct * n_samples)`` samples
    positioned immediately after the test block are dropped from training
    (embargo). Samples are never dropped from test folds.

    Args:
        n_splits: number of folds; must be >= 2.
        embargo_pct: fraction of the sample count embargoed after each test
            block; must lie in [0, 1).
    """

    n_splits: int
    embargo_pct: float

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {self.n_splits}")
        if not 0.0 <= self.embargo_pct < 1.0:
            raise ValueError(f"embargo_pct must be in [0, 1), got {self.embargo_pct}")

    def split(
        self, times: pd.DataFrame
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Yield ``(train_idx, test_idx)`` positional index arrays per fold.

        Args:
            times: one row per sample with columns ``t0`` (label start) and
                ``t1`` (label end, inclusive), rows in chronological order of
                ``t0``. The yielded arrays are positions into these rows.

        Raises:
            ValueError: if rows are not sorted by ``t0``, any ``t1`` precedes
                its ``t0``, or there are fewer samples than folds.
        """
        t0 = times["t0"].to_numpy()
        t1 = times["t1"].to_numpy()
        n = len(times)
        if n < self.n_splits:
            raise ValueError(f"{n} samples cannot fill {self.n_splits} folds")
        if bool(np.any(t1 < t0)):
            raise ValueError("label interval end t1 precedes start t0")
        if bool(np.any(t0[1:] < t0[:-1])):
            raise ValueError("times must be sorted by t0")

        embargo = int(np.ceil(n * self.embargo_pct))
        indices = np.arange(n, dtype=np.intp)
        for test_idx in np.array_split(indices, self.n_splits):
            # Exact pairwise closed-interval overlap of every sample vs every
            # test sample — no envelope shortcut, so irregular span lengths
            # cannot cause over- or under-purging.
            overlaps = (t0[:, None] <= t1[test_idx][None, :]) & (
                t1[:, None] >= t0[test_idx][None, :]
            )
            banned = overlaps.any(axis=1)
            banned[test_idx] = True
            embargo_start = int(test_idx[-1]) + 1
            banned[embargo_start : embargo_start + embargo] = True
            yield indices[~banned], test_idx


@dataclass(frozen=True)
class DateRange:
    """Inclusive ``[start, end]`` range of calendar sessions."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"end {self.end} precedes start {self.start}")


def walkforward(
    fit_window_days: int,
    test_window_days: int,
    step_days: int,
    calendar: Sequence[date],
) -> list[tuple[DateRange, DateRange]]:
    """Ordered ``(fit_range, test_range)`` session-window pairs over ``calendar``.

    Windows are counted in calendar sessions: each pair fits on
    ``fit_window_days`` consecutive sessions and tests on the
    ``test_window_days`` sessions immediately after them (disjoint — the test
    range starts on the session following the fit range's last session);
    successive pairs advance by ``step_days`` sessions. Pairs are emitted in
    chronological order. Before returning, every pair is re-checked and an
    ``AssertionError`` is raised if any test range fails to start strictly
    after the last session of its fit data.

    Args:
        fit_window_days: fit window length in sessions; > 0.
        test_window_days: test window length in sessions; > 0.
        step_days: sessions to advance between consecutive pairs; > 0.
        calendar: strictly increasing session dates spanning the study period.

    Raises:
        ValueError: on non-positive window/step sizes, a non-increasing
            calendar, or a calendar too short for a single fit+test pair.
        AssertionError: if a constructed test range precedes its fit data.
    """
    if fit_window_days <= 0 or test_window_days <= 0 or step_days <= 0:
        raise ValueError("fit_window_days, test_window_days, and step_days must be positive")
    sessions = list(calendar)
    if any(later <= earlier for earlier, later in pairwise(sessions)):
        raise ValueError("calendar must be strictly increasing")
    if len(sessions) < fit_window_days + test_window_days:
        raise ValueError(
            f"calendar has {len(sessions)} sessions; "
            f"{fit_window_days + test_window_days} needed for one fit+test pair"
        )

    pairs: list[tuple[DateRange, DateRange]] = []
    start = 0
    while start + fit_window_days + test_window_days <= len(sessions):
        fit_end = start + fit_window_days - 1
        test_end = fit_end + test_window_days
        pairs.append(
            (
                DateRange(start=sessions[start], end=sessions[fit_end]),
                DateRange(start=sessions[fit_end + 1], end=sessions[test_end]),
            )
        )
        start += step_days

    for fit_range, test_range in pairs:
        if not fit_range.end < test_range.start:
            raise AssertionError(
                f"test range {test_range} does not start strictly after fit data {fit_range}"
            )
    return pairs
