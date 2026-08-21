"""Sweep combination generation + walk-forward windowing."""

from datetime import date

from catalyst.core.config import SweepConfig, WalkForwardConfig
from catalyst.backtest.sweep import iter_combinations
from catalyst.backtest.walkforward import chronological_split, walk_forward_windows


def test_grid_combinations_cartesian() -> None:
    cfg = SweepConfig(
        mode="grid",
        random_samples=0,
        parameters={"a.x": [1, 2, 3], "b.y": [True, False]},
    )
    combos = list(iter_combinations(cfg))
    assert len(combos) == 6
    assert {"a.x": 1, "b.y": True} in combos
    assert all(set(c) == {"a.x", "b.y"} for c in combos)


def test_random_combinations_count_and_membership() -> None:
    cfg = SweepConfig(
        mode="random",
        random_samples=25,
        parameters={"a.x": [1, 2, 3], "b.y": [0.1, 0.2]},
    )
    combos = list(iter_combinations(cfg))
    # dedup since audit D-196: the space holds 3x2=6 distinct combos, so 25
    # requested samples yield each exactly once
    assert len(combos) == 6
    keys = {tuple(sorted((k, str(v)) for k, v in c.items())) for c in combos}
    assert len(keys) == 6
    assert all(c["a.x"] in (1, 2, 3) and c["b.y"] in (0.1, 0.2) for c in combos)


def test_walk_forward_windows_roll() -> None:
    cfg = WalkForwardConfig(train_months=24, test_months=6, step_months=6)
    # Valid train starts: 2018-01 .. 2023-07 stepping 6 months = 12 windows
    # (each spans 30 months and must fit inside the 96-month range).
    windows = walk_forward_windows(date(2018, 1, 1), date(2026, 1, 1), cfg)
    assert len(windows) == 12
    first = windows[0]
    assert first.train_start == date(2018, 1, 1)
    assert first.train_end == date(2020, 1, 1)
    # boundary EXCLUSIVE since audit D-027: the shared session
    # no longer appears in both windows
    assert first.test_start == date(2020, 1, 2)
    assert first.test_end == date(2020, 7, 1)
    # No window may peek past the end.
    assert all(w.test_end <= date(2026, 1, 1) for w in windows)
    # Consecutive windows step by step_months.
    assert windows[1].train_start == date(2018, 7, 1)


def test_chronological_split_fraction() -> None:
    boundary, test_start = chronological_split(date(2020, 1, 1), date(2021, 1, 1), 0.70)
    assert boundary == test_start
    assert date(2020, 9, 12) <= boundary <= date(2020, 9, 13)
