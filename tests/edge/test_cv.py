"""PurgedKFold purges every overlapping-label leak a naive KFold admits,
honors the embargo width, and walkforward emits disjoint ordered windows.
All timestamps are synthetic and predate the 2026-02-22 lockbox window."""

from __future__ import annotations

from datetime import date, timedelta
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from edge.validation.walkforward import DateRange, PurgedKFold, walkforward

# Synthetic session grid well BEFORE the lockbox start.
BAR0 = pd.Timestamp("2025-03-03")
N_BARS = 60
SPAN_BARS = 5  # each label covers its own bar plus the next 4
N_SPLITS = 3


def _spanning_times(n: int = N_BARS, span_bars: int = SPAN_BARS) -> pd.DataFrame:
    """Label i spans bars i..i+span_bars-1 (inclusive) on a daily grid."""
    t0 = pd.date_range(BAR0, periods=n, freq="D")
    return pd.DataFrame({"t0": t0, "t1": t0 + pd.Timedelta(days=span_bars - 1)})


def _point_times(n: int) -> pd.DataFrame:
    """Degenerate labels (t1 == t0): no purge possible, isolates the embargo."""
    t0 = pd.date_range(BAR0, periods=n, freq="D")
    return pd.DataFrame({"t0": t0, "t1": t0})


def _naive_folds(n: int, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Plain contiguous KFold: no purge, no embargo."""
    indices = np.arange(n)
    folds = []
    for test_idx in np.array_split(indices, k):
        train_idx = np.setdiff1d(indices, test_idx)
        folds.append((train_idx, test_idx))
    return folds


def _overlap_pairs(
    times: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray
) -> list[tuple[int, int]]:
    """(train, test) sample pairs whose closed label intervals overlap."""
    t0 = times["t0"].to_numpy()
    t1 = times["t1"].to_numpy()
    return [
        (int(i), int(j))
        for i in train_idx
        for j in test_idx
        if t0[i] <= t1[j] and t1[i] >= t0[j]
    ]


# ---------------------------------------------------------------------------
# Purging
# ---------------------------------------------------------------------------


def test_naive_kfold_leaks_and_purged_kfold_removes_every_leak() -> None:
    times = _spanning_times()

    # Naive KFold DOES leak: count the overlapping (train, test) pairs.
    # With 5-bar labels and 20-bar blocks, each block boundary leaks
    # 4+3+2+1 = 10 pairs; the middle fold has two boundaries.
    naive_leaks: list[list[tuple[int, int]]] = [
        _overlap_pairs(times, train_idx, test_idx)
        for train_idx, test_idx in _naive_folds(N_BARS, N_SPLITS)
    ]
    assert [len(leaks) for leaks in naive_leaks] == [10, 20, 10]
    assert sum(len(leaks) for leaks in naive_leaks) == 40

    # PurgedKFold excludes every one of them: zero overlap pairs remain.
    cv = PurgedKFold(n_splits=N_SPLITS, embargo_pct=0.0)
    purged_folds = list(cv.split(times))
    assert len(purged_folds) == N_SPLITS
    for (train_idx, test_idx), leaks in zip(purged_folds, naive_leaks):
        assert _overlap_pairs(times, train_idx, test_idx) == []
        leaking_train = {i for i, _ in leaks}
        assert leaking_train.isdisjoint(train_idx)


def test_purge_is_minimal_and_test_folds_partition_samples() -> None:
    times = _spanning_times()
    cv = PurgedKFold(n_splits=N_SPLITS, embargo_pct=0.0)

    seen_test: list[int] = []
    for (train_idx, test_idx), (naive_train, naive_test), leaks in zip(
        cv.split(times),
        _naive_folds(N_BARS, N_SPLITS),
        [
            _overlap_pairs(times, tr, te)
            for tr, te in _naive_folds(N_BARS, N_SPLITS)
        ],
    ):
        # Test blocks are untouched by purging.
        assert np.array_equal(test_idx, naive_test)
        # Exactly the leaking samples are dropped — nothing more.
        expected_train = sorted(set(naive_train) - {i for i, _ in leaks})
        assert train_idx.tolist() == expected_train
        seen_test.extend(test_idx.tolist())

    assert sorted(seen_test) == list(range(N_BARS))  # partition, no repeats


# ---------------------------------------------------------------------------
# Embargo
# ---------------------------------------------------------------------------


def test_embargo_width_honored() -> None:
    n = 100
    embargo_pct = 0.05
    width = 5  # ceil(100 * 0.05)
    times = _point_times(n)  # point labels: purge removes nothing
    cv = PurgedKFold(n_splits=4, embargo_pct=embargo_pct)

    for train_idx, test_idx in cv.split(times):
        train = set(train_idx.tolist())
        after = int(test_idx[-1]) + 1
        embargoed = [i for i in range(after, min(after + width, n))]
        # The full embargo window after the test block is out of training...
        assert all(i not in train for i in embargoed)
        # ...and the very next sample is back in (embargo is not wider).
        if after + width < n:
            assert after + width in train
        # Samples immediately BEFORE the block are not embargoed.
        if int(test_idx[0]) - 1 >= 0:
            assert int(test_idx[0]) - 1 in train


def test_zero_embargo_with_point_labels_drops_nothing() -> None:
    n = 40
    times = _point_times(n)
    cv = PurgedKFold(n_splits=4, embargo_pct=0.0)
    for train_idx, test_idx in cv.split(times):
        assert len(train_idx) + len(test_idx) == n
        assert set(train_idx.tolist()).isdisjoint(test_idx.tolist())


# ---------------------------------------------------------------------------
# PurgedKFold input validation
# ---------------------------------------------------------------------------


def test_purged_kfold_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=1, embargo_pct=0.0)
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=3, embargo_pct=1.0)
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=3, embargo_pct=-0.1)


def test_purged_kfold_rejects_bad_times() -> None:
    cv = PurgedKFold(n_splits=2, embargo_pct=0.0)

    unsorted = _spanning_times(10).iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="sorted"):
        next(cv.split(unsorted))

    inverted = _spanning_times(10)
    inverted.loc[3, "t1"] = inverted.loc[3, "t0"] - pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="t1"):
        next(cv.split(inverted))

    with pytest.raises(ValueError, match="folds"):
        next(PurgedKFold(n_splits=5, embargo_pct=0.0).split(_spanning_times(3)))


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

CAL_START = date(2025, 3, 3)
CAL_LEN = 60
FIT_DAYS = 20
TEST_DAYS = 10
STEP_DAYS = 10


def _calendar(n: int = CAL_LEN) -> list[date]:
    return [CAL_START + timedelta(days=i) for i in range(n)]


def _sessions_in(r: DateRange, calendar: list[date]) -> list[date]:
    return [d for d in calendar if r.start <= d <= r.end]


def test_walkforward_pairs_disjoint_ordered_cover_span() -> None:
    calendar = _calendar()
    pairs = walkforward(FIT_DAYS, TEST_DAYS, STEP_DAYS, calendar)
    assert len(pairs) == 4

    for fit_range, test_range in pairs:
        # Fit strictly before test, never sharing a session.
        assert fit_range.end < test_range.start
        assert test_range.start == fit_range.end + timedelta(days=1)
        assert len(_sessions_in(fit_range, calendar)) == FIT_DAYS
        assert len(_sessions_in(test_range, calendar)) == TEST_DAYS

    # Ordered: fit and test starts strictly increase; consecutive test
    # ranges are disjoint (step == test window tiles them exactly).
    for (fit_a, test_a), (fit_b, test_b) in pairwise(pairs):
        assert fit_a.start < fit_b.start
        assert test_a.start < test_b.start
        assert test_a.end < test_b.start

    # Coverage: the test windows tile every session after the first fit
    # window, through the end of the calendar, with no gaps or repeats.
    tested: list[date] = []
    for _, test_range in pairs:
        tested.extend(_sessions_in(test_range, calendar))
    assert tested == calendar[FIT_DAYS:]


def test_walkforward_windows_slide_by_step() -> None:
    calendar = _calendar()
    pairs = walkforward(FIT_DAYS, TEST_DAYS, STEP_DAYS, calendar)
    for k, (fit_range, _) in enumerate(pairs):
        assert fit_range.start == calendar[k * STEP_DAYS]


def test_walkforward_rejects_bad_inputs() -> None:
    calendar = _calendar()
    for bad in [(0, TEST_DAYS, STEP_DAYS), (FIT_DAYS, 0, STEP_DAYS), (FIT_DAYS, TEST_DAYS, 0)]:
        with pytest.raises(ValueError, match="positive"):
            walkforward(*bad, calendar)
    with pytest.raises(ValueError, match="increasing"):
        walkforward(FIT_DAYS, TEST_DAYS, STEP_DAYS, list(reversed(calendar)))
    with pytest.raises(ValueError, match="too short|needed"):
        walkforward(FIT_DAYS, TEST_DAYS, STEP_DAYS, _calendar(FIT_DAYS + TEST_DAYS - 1))


def test_date_range_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError):
        DateRange(start=CAL_START, end=CAL_START - timedelta(days=1))
