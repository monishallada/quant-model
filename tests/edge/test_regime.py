"""Regime classifier tests: fixed economic definitions, point-in-time by session.

ALL data is synthetic and every timestamp is strictly before 2026-02-22 (the
lockbox wall), so a locked default loader passes everything through. The stub
backend serves crafted ``vol_indices`` + ``bars`` frames THROUGH a real
:class:`EdgeDataLoader` — the only sanctioned gateway — and trims to the
requested range exactly like a real backend, so prefix-invariance is tested
against genuinely truncated data, not a no-op.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from edge.data.loader import EdgeDataLoader
from edge.regime.classifier import (
    BUCKETS,
    DEFAULT_HISTORY_START,
    MARKET_WIDE_SYMBOL,
    REGIME_COLUMNS,
    TREND_STATES,
    TREND_WINDOW,
    VOL_STATES,
    classify,
    gate,
)

ET = ZoneInfo("America/New_York")

#: All synthetic sessions start here (a Monday); ~45 bdays end mid-Dec 2025,
#: comfortably before the 2026-02-22 lockbox wall.
BASE_DAY = "2025-10-06"


def bdays(n: int, start: str = BASE_DAY) -> list[date]:
    return [ts.date() for ts in pd.bdate_range(start, periods=n)]


def et_at(day: date, hour: int, minute: int = 0) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(day, time(hour, minute))).tz_localize(ET)


def evening_stamps(days: list[date]) -> pd.Series:
    naive = pd.to_datetime(pd.Series(days)) + pd.Timedelta(hours=18)
    return naive.dt.tz_localize(ET)


def make_vol(
    days: list[date],
    vix: list[float] | np.ndarray,
    vix9d: list[float] | np.ndarray | None = None,
    include_vix9d: bool = True,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "asof_date": pd.to_datetime(pd.Series(days)),
            "vix": np.asarray(vix, dtype=float),
            "available_at": evening_stamps(days),
        }
    )
    if include_vix9d:
        nine = np.asarray(vix9d, dtype=float) if vix9d is not None else frame["vix"] * 0.9
        frame["vix9d"] = nine
    return frame


def make_bars(
    days: list[date],
    close: list[float] | np.ndarray,
    high: list[float] | np.ndarray | None = None,
    low: list[float] | np.ndarray | None = None,
    available_at: pd.Series | list[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "asof_date": pd.to_datetime(pd.Series(days)),
            "close": np.asarray(close, dtype=float),
        }
    )
    if high is not None:
        frame["high"] = np.asarray(high, dtype=float)
    if low is not None:
        frame["low"] = np.asarray(low, dtype=float)
    if available_at is not None:
        frame["available_at"] = pd.Series(available_at)
    return frame


class StubBackend:
    """Serves canned frames per kind, trimmed to the requested date range."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self._frames = frames
        self.calls: list[tuple[str, date, date, str]] = []

    def fetch(self, symbol: str, start: date, end: date, kind: str) -> pd.DataFrame:
        self.calls.append((symbol, start, end, kind))
        frame = self._frames[kind]
        asof = pd.to_datetime(frame["asof_date"])
        mask = (asof >= pd.Timestamp(start)) & (asof <= pd.Timestamp(end))
        return frame[mask].reset_index(drop=True)


def make_loader(vol: pd.DataFrame, bars: pd.DataFrame) -> tuple[EdgeDataLoader, StubBackend]:
    backend = StubBackend({"vol_indices": vol, "bars": bars})
    return EdgeDataLoader(backend), backend


# ---------------------------------------------------------------------------
# vol_state: expanding-percentile terciles and stressed backwardation
# ---------------------------------------------------------------------------

#: 30 days of VIX 10..39, then 12 (low), 25 (mid), 38 (high). Hand-computed
#: midrank percentiles of the session-latest value within history to date.
TERCILE_VIX = list(range(10, 40)) + [12.0, 25.0, 38.0]
TERCILE_PCTILES = [100 * 6 / 62, 100 * 34 / 64, 100 * 62 / 66]


def tercile_fixture() -> tuple[EdgeDataLoader, StubBackend, list[date]]:
    days_b = bdays(36)
    vol = make_vol(days_b[:33], TERCILE_VIX)  # vix9d = 0.9*vix: never stressed
    bars = make_bars(days_b, [100.0] * 36)  # flat: every session chops
    loader, backend = make_loader(vol, bars)
    return loader, backend, days_b


def test_vol_terciles_hit_low_mid_high() -> None:
    loader, _, days_b = tercile_fixture()
    # Sessions the day AFTER each probe value publishes (evening convention).
    out = classify(loader, days_b[31], days_b[33])

    assert list(out["session"].dt.date) == days_b[31:34]
    assert list(out["vol_state"]) == ["low", "mid", "high"]
    assert list(out["vix_pctile"]) == pytest.approx(TERCILE_PCTILES)
    assert list(out["trend_state"]) == ["chop", "chop", "chop"]
    assert list(out["bucket"]) == [
        "low_vol_chopping",  # pctile ~9.7 < 50
        "high_vol_chopping",  # pctile ~53.1 >= 50
        "high_vol_chopping",  # pctile ~93.9 >= 50
    ]


def test_stressed_backwardation_and_no_same_day_leak() -> None:
    """VIX9D>VIX published the evening of day 30 flips day 31, NEVER day 30."""
    days = bdays(32)
    vix9d = [18.0] * 30 + [25.0]  # ratio 0.9 ... then 1.25 on the last vol day
    vol = make_vol(days[:31], [20.0] * 31, vix9d=vix9d)
    bars = make_bars(days, [100.0] * 32)
    loader, _ = make_loader(vol, bars)

    out = classify(loader, days[30], days[31])

    assert list(out["session"].dt.date) == [days[30], days[31]]
    # Day 30 opens BEFORE the inversion publishes: constant history -> mid.
    assert list(out["vol_state"]) == ["mid", "stressed_backwardation"]
    # The bucket still uses the percentile (constant history sits at 50).
    assert list(out["bucket"]) == ["high_vol_chopping", "high_vol_chopping"]


def test_missing_vix9d_column_never_stresses() -> None:
    days = bdays(32)
    vol = make_vol(days[:31], list(range(10, 41)), include_vix9d=False)
    bars = make_bars(days, [100.0] * 32)
    loader, _ = make_loader(vol, bars)

    out = classify(loader, days[31], days[31])

    assert list(out["vol_state"]) == ["high"]  # percentile path, not stressed


# ---------------------------------------------------------------------------
# trend_state: 21d change vs +/- 1 ATR(21)
# ---------------------------------------------------------------------------

N_TREND = 40


def trend_scenarios() -> dict[str, tuple[pd.DataFrame, str]]:
    days = bdays(N_TREND)
    climb = [100.0 + i for i in range(N_TREND)]
    return {
        "steady_climb_is_up": (make_bars(days, climb), "up"),
        "steady_decline_is_down": (
            make_bars(days, [200.0 - i for i in range(N_TREND)]),
            "down",
        ),
        "oscillation_is_chop": (
            make_bars(days, [100.0 + (i % 2) for i in range(N_TREND)]),
            "chop",
        ),
        # Climbing +1/day but with 30-point intraday ranges: ATR(21) ~ 30
        # dwarfs the 21-point move — trend must be judged vs ATR, not raw.
        "wide_range_climb_is_chop": (
            make_bars(
                days,
                climb,
                high=[c + 15.0 for c in climb],
                low=[c - 15.0 for c in climb],
            ),
            "chop",
        ),
    }


@pytest.mark.parametrize("name", sorted(trend_scenarios()))
def test_trend_states(name: str) -> None:
    bars, expected = trend_scenarios()[name]
    days = bdays(N_TREND)
    vol = make_vol(days, [20.0] * N_TREND)
    loader, _ = make_loader(vol, bars)

    out = classify(loader, days[-1], days[-1])

    assert list(out["trend_state"]) == [expected]


def test_trend_warmup_is_chop_and_first_session_dropped() -> None:
    days = bdays(10)  # < TREND_WINDOW + 1 usable bars everywhere
    assert len(days) < TREND_WINDOW + 1
    vol = make_vol(days, [20.0 + i for i in range(10)])
    bars = make_bars(days, [100.0 + i for i in range(10)])  # climbing, but warmup
    loader, _ = make_loader(vol, bars)

    out = classify(loader, days[0], days[9])

    # Session 0 has NO input published before its open -> omitted, not defaulted.
    assert list(out["session"].dt.date) == days[1:]
    assert set(out["trend_state"]) == {"chop"}
    assert set(out["bucket"]) <= {"high_vol_chopping", "low_vol_chopping"}


# ---------------------------------------------------------------------------
# The four promotion buckets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("vol_kind", "trend_kind", "expected"),
    [
        ("rising", "climb", "high_vol_trending"),
        ("rising", "flat", "high_vol_chopping"),
        ("falling", "climb", "low_vol_trending"),
        ("falling", "flat", "low_vol_chopping"),
    ],
)
def test_all_four_promotion_buckets(vol_kind: str, trend_kind: str, expected: str) -> None:
    days = bdays(N_TREND)
    vix = (
        [10.0 + i for i in range(N_TREND)]  # latest = max of history -> pctile ~99
        if vol_kind == "rising"
        else [60.0 - i for i in range(N_TREND)]  # latest = min -> pctile ~1
    )
    closes = (
        [100.0 + i for i in range(N_TREND)] if trend_kind == "climb" else [100.0] * N_TREND
    )
    loader, _ = make_loader(make_vol(days, vix), make_bars(days, closes))

    out = classify(loader, days[-1], days[-1])

    assert list(out["bucket"]) == [expected]
    assert expected in BUCKETS


# ---------------------------------------------------------------------------
# Point-in-time guarantees
# ---------------------------------------------------------------------------


def test_expanding_percentile_is_prefix_invariant() -> None:
    """Truncating the future must not change any past session's row."""
    loader_full, backend_full, days_b = tercile_fixture()
    loader_trunc, backend_trunc, _ = tercile_fixture()

    full = classify(loader_full, days_b[25], days_b[35])
    trunc = classify(loader_trunc, days_b[25], days_b[30])

    # The truncated run really saw less data (the stub trims to the range).
    full_ends = {call[2] for call in backend_full.calls}
    trunc_ends = {call[2] for call in backend_trunc.calls}
    assert full_ends == {days_b[35]} and trunc_ends == {days_b[30]}

    common = full[full["session"].dt.date <= days_b[30]].reset_index(drop=True)
    pd.testing.assert_frame_equal(trunc, common)


def test_available_at_is_prior_session_evening_and_precedes_open() -> None:
    loader, _, days_b = tercile_fixture()

    out = classify(loader, days_b[31], days_b[33])

    for row in out.itertuples():
        session_day = row.session.date()
        prior = days_b[days_b.index(session_day) - 1]
        assert row.available_at == et_at(prior, 18)
        assert row.available_at < et_at(session_day, 9, 30)


def test_late_published_bar_is_invisible_until_available() -> None:
    """A bar published after the next open must not feed that session's trend.

    Flat closes, then a spike on day 28. Published normally (18:00 the same
    evening) the spike makes session 29 'up'; published late (10:00 the NEXT
    day, after session 29's open) session 29 must still be 'chop'.
    """
    days = bdays(30)
    closes = [100.0] * 28 + [200.0, 200.0]
    vol = make_vol(days, [20.0] * 30)

    on_time = make_bars(days, closes)
    loader_on_time, _ = make_loader(vol, on_time)
    out_on_time = classify(loader_on_time, days[29], days[29])
    assert list(out_on_time["trend_state"]) == ["up"]

    stamps = list(evening_stamps(days))
    stamps[28] = et_at(days[29], 10)  # spike bar first visible AFTER open 29
    late = make_bars(days, closes, available_at=stamps)
    loader_late, _ = make_loader(vol, late)
    out_late = classify(loader_late, days[29], days[29])
    assert list(out_late["trend_state"]) == ["chop"]
    assert out_late["available_at"].iloc[0] < et_at(days[29], 9, 30)


# ---------------------------------------------------------------------------
# Output contract and gateway usage
# ---------------------------------------------------------------------------


def test_output_schema_and_loader_usage() -> None:
    loader, backend, days_b = tercile_fixture()

    out = classify(loader, days_b[31], days_b[33])

    assert tuple(out.columns) == REGIME_COLUMNS
    assert out["session"].dt.tz is None  # naive midnight session stamps
    assert str(out["available_at"].dt.tz) == "America/New_York"
    assert set(out["vol_state"]) <= set(VOL_STATES)
    assert set(out["trend_state"]) <= set(TREND_STATES)
    assert set(out["bucket"]) <= BUCKETS
    # Exactly two gated loads, full history to date, market-wide vol symbol.
    assert backend.calls == [
        (MARKET_WIDE_SYMBOL, DEFAULT_HISTORY_START, days_b[33], "vol_indices"),
        ("SPY", DEFAULT_HISTORY_START, days_b[33], "bars"),
    ]


def test_empty_range_yields_empty_frame_with_schema() -> None:
    loader, _, _ = tercile_fixture()
    out = classify(loader, date(2025, 1, 6), date(2025, 1, 7))  # before all data
    assert out.empty
    assert tuple(out.columns) == REGIME_COLUMNS


def test_start_after_end_raises() -> None:
    loader, _, days_b = tercile_fixture()
    with pytest.raises(ValueError, match="start"):
        classify(loader, days_b[5], days_b[4])


def test_naive_vol_available_at_rejected() -> None:
    days = bdays(30)
    vol = make_vol(days, [20.0] * 30)
    vol["available_at"] = vol["available_at"].dt.tz_localize(None)  # naive
    bars = make_bars(days, [100.0] * 30)
    loader, _ = make_loader(vol, bars)
    with pytest.raises(ValueError, match="tz-aware"):
        classify(loader, days[-1], days[-1])


# ---------------------------------------------------------------------------
# gate(): the engine hook
# ---------------------------------------------------------------------------


def test_gate_membership() -> None:
    assert gate({"high_vol_trending", "low_vol_trending"}, "high_vol_trending") is True
    assert gate({"high_vol_trending"}, "low_vol_chopping") is False
    assert gate("high_vol_chopping", "high_vol_chopping") is True  # str shorthand
    assert gate(["low_vol_chopping"], "low_vol_chopping") is True


def test_gate_none_allows_and_empty_blocks() -> None:
    for bucket in sorted(BUCKETS):
        assert gate(None, bucket) is True  # opted out of regime gating
        assert gate(set(), bucket) is False  # explicitly allows nothing


def test_gate_rejects_unknown_names_loudly() -> None:
    with pytest.raises(ValueError, match="unknown regime bucket"):
        gate({"high_vol_trend"}, "high_vol_trending")  # typo'd allowed set
    with pytest.raises(ValueError, match="unknown session bucket"):
        gate({"high_vol_trending"}, "nonsense")
